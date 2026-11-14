#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

#define BF16X8_WRITE(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define BF16X8_READ(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

constexpr int EMPTY_SLOT = -1;

// Multiplicative hash function (Knuth's)
__device__ __forceinline__ int hash_func(int key, int mask) {
    return ((unsigned int)key * 2654435761u) & mask;
}

template <int HASH_TABLE_SIZE>
__global__ void fifo_update_kernel_optimized(
    const int64_t* __restrict__ topk_indices,   
    int* __restrict__ lru_indices,          // [B, KH, allocated_size]
    int* __restrict__ lru_ptr,              
    const __nv_bfloat16* __restrict__ global_k,     
    const __nv_bfloat16* __restrict__ global_v,     
    __nv_bfloat16* __restrict__ cache_k,            
    __nv_bfloat16* __restrict__ cache_v,  
    float* __restrict__ hit_rates,
    const int B, 
    const int KH, 
    const int lru_logical_size,
    const int allocated_size,
    const int top_budget, 
    const int D, 
    const int MaxT
) {
    constexpr int HASH_MASK = HASH_TABLE_SIZE - 1;
    int b = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    int batch_head_idx = b * KH + head_idx;

    // Shared memory layout
    extern __shared__ char shared_mem[];
    int* hash_table = reinterpret_cast<int*>(shared_mem);
    int* s_diff_indices = hash_table + HASH_TABLE_SIZE;
    int* s_count_ptr = s_diff_indices + top_budget;

    // Initialize hash table to EMPTY_SLOT (-1)
    for (int i = tid; i < HASH_TABLE_SIZE; i += blockDim.x) {
        hash_table[i] = EMPTY_SLOT;
    }
    
    if (tid == 0) {
        *s_count_ptr = 0;
    }

    __syncthreads();

    int* global_lru_indices_ptr = lru_indices + (batch_head_idx * allocated_size);
    
    // Insert all current indices into hash table
    for (int i = tid; i < lru_logical_size; i += blockDim.x) {
        int idx = global_lru_indices_ptr[i];
        if (idx >= 0) {
            int hash_pos = hash_func(idx, HASH_MASK);
            while (true) {
                int old = atomicCAS(&hash_table[hash_pos], EMPTY_SLOT, idx);
                if (old == EMPTY_SLOT || old == idx) {
                    break;
                }
                hash_pos = (hash_pos + 1) & HASH_MASK;
            }
        }
    }
    
    __syncthreads();

    // Check each candidate against hash table
    const int64_t* my_topk_ptr = topk_indices + (batch_head_idx * top_budget);
    for (int i = tid; i < top_budget; i += blockDim.x) {
        int candidate_idx = static_cast<int>(my_topk_ptr[i]);
        if (candidate_idx >= 0) {
            bool exists = false;
            int hash_pos = hash_func(candidate_idx, HASH_MASK);
            
            while (hash_table[hash_pos] != EMPTY_SLOT) {
                if (hash_table[hash_pos] == candidate_idx) {
                    exists = true;
                    break;
                }
                hash_pos = (hash_pos + 1) & HASH_MASK;
            }
            
            if (!exists) {
                int pos = atomicAdd(s_count_ptr, 1);
                s_diff_indices[pos] = candidate_idx;
            }
        }
    }
    __syncthreads();

    int num_updates = *s_count_ptr;

    if (tid == 0) {
        float rate = 0.0f;
        if (top_budget > 0) {
            int hits = top_budget - num_updates;
            rate = static_cast<float>(hits) / static_cast<float>(top_budget);
        }
        hit_rates[batch_head_idx] = rate;
    }

    if (num_updates == 0) return;

    int current_ptr = lru_ptr[batch_head_idx];

    if (tid == 0) {
        for (int i = 0; i < num_updates; i++) {
            int write_pos = (current_ptr + i) % lru_logical_size;
            global_lru_indices_ptr[write_pos] = s_diff_indices[i]; 
        }
        lru_ptr[batch_head_idx] = (current_ptr + num_updates) % lru_logical_size;
    }
    
    __syncthreads(); 

    int D_vec = D / 8; 
    int total_tasks = num_updates * D_vec;

    for (int task_id = tid; task_id < total_tasks; task_id += blockDim.x) {
        int update_idx = task_id / D_vec;
        int d_vec_idx = task_id % D_vec;
        int d_offset = d_vec_idx * 8;

        int token_global_idx = s_diff_indices[update_idx];
        int write_pos_logical = (current_ptr + update_idx) % lru_logical_size;

        long src_offset = ((long)b * MaxT * KH * D) + 
                          ((long)token_global_idx * KH * D) + 
                          ((long)head_idx * D) + 
                          d_offset;

        long dst_offset = ((long)b * allocated_size * KH * D) + 
                          ((long)write_pos_logical * KH * D) + 
                          ((long)head_idx * D) + 
                          d_offset;

        float4 k_val = BF16X8_READ(global_k[src_offset]);
        BF16X8_WRITE(cache_k[dst_offset]) = k_val;

        float4 v_val = BF16X8_READ(global_v[src_offset]);
        BF16X8_WRITE(cache_v[dst_offset]) = v_val;
    }
}

torch::Tensor fifo_update_cuda(
    torch::Tensor topk_indices,
    torch::Tensor lru_indices,
    torch::Tensor lru_ptr,
    torch::Tensor global_k,
    torch::Tensor global_v,
    torch::Tensor cache_k,
    torch::Tensor cache_v,
    int lru_logical_size, 
    int top_budget
) {
    TORCH_CHECK(global_k.scalar_type() == torch::kBFloat16, "global_k must be bfloat16");
    TORCH_CHECK(global_v.scalar_type() == torch::kBFloat16, "global_v must be bfloat16");
    TORCH_CHECK(cache_k.scalar_type() == torch::kBFloat16, "cache_k must be bfloat16");
    TORCH_CHECK(cache_v.scalar_type() == torch::kBFloat16, "cache_v must be bfloat16");
    
    int hash_size = 1024;
    if (lru_logical_size * 2 > 8192) hash_size = 16384;
    else if (lru_logical_size * 2 > 4096) hash_size = 8192;
    else if (lru_logical_size * 2 > 2048) hash_size = 4096;
    else if (lru_logical_size * 2 > 1024) hash_size = 2048;
    else hash_size = 1024;

    TORCH_CHECK(lru_logical_size <= hash_size / 2, "lru_logical_size too large for compiled hash sizes");

    const int B = topk_indices.size(0);
    const int KH = topk_indices.size(1);
    const int MaxT = global_k.size(1);
    const int D = global_k.size(3);
    const int allocated_size = cache_k.size(1);

    TORCH_CHECK(lru_indices.size(2) == allocated_size, "lru_indices last dim mismatch");
    TORCH_CHECK(D % 8 == 0, "D must be divisible by 8");
    TORCH_CHECK(lru_logical_size <= allocated_size, "Logical size cannot exceed allocated size.");

    auto hit_rates = torch::empty({B, KH}, topk_indices.options().dtype(torch::kFloat32));

    dim3 grid(B, KH);
    int threads = 256; 
    size_t smem_size = hash_size * sizeof(int) + top_budget * sizeof(int) + sizeof(int);

    auto stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH_FIFO_KERNEL(SIZE) \
        fifo_update_kernel_optimized<SIZE><<<grid, threads, smem_size, stream>>>( \
            topk_indices.data_ptr<int64_t>(), lru_indices.data_ptr<int>(), lru_ptr.data_ptr<int>(), \
            reinterpret_cast<const __nv_bfloat16*>(global_k.data_ptr<at::BFloat16>()), \
            reinterpret_cast<const __nv_bfloat16*>(global_v.data_ptr<at::BFloat16>()), \
            reinterpret_cast<__nv_bfloat16*>(cache_k.data_ptr<at::BFloat16>()), \
            reinterpret_cast<__nv_bfloat16*>(cache_v.data_ptr<at::BFloat16>()), \
            hit_rates.data_ptr<float>(), B, KH, lru_logical_size, allocated_size, top_budget, D, MaxT \
        );

    if (smem_size > 48 * 1024) {
        int device; cudaGetDevice(&device);
        int max_smem; cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (smem_size <= (size_t)max_smem) {
            switch(hash_size) {
                case 8192: cudaFuncSetAttribute(fifo_update_kernel_optimized<8192>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); break;
                case 16384: cudaFuncSetAttribute(fifo_update_kernel_optimized<16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); break;
            }
        }
    }

    switch (hash_size) {
        case 1024:  LAUNCH_FIFO_KERNEL(1024); break;
        case 2048:  LAUNCH_FIFO_KERNEL(2048); break;
        case 4096:  LAUNCH_FIFO_KERNEL(4096); break;
        case 8192:  LAUNCH_FIFO_KERNEL(8192); break;
        case 16384: LAUNCH_FIFO_KERNEL(16384); break;
        default: TORCH_CHECK(false, "Unsupported hash size");
    }

    return hit_rates;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update", &fifo_update_cuda, "FIFO Cache Update (CUDA) - Returns Hit Rate [B, KH]");
}