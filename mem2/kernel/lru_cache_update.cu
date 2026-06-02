#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

#define BF16X8_WRITE(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define BF16X8_READ(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// Hash Table 大小必须是 2 的幂
// 注意：shared memory 限制通常是 48KB，8192*2*4=64KB 会超限
// 使用 4096 -> 32KB，安全范围内
constexpr int EMPTY_SLOT = -1;
constexpr int INVALID_TIME = 0x7FFFFFFF; 

// Knuth's Multiplicative Hash
__device__ __forceinline__ int hash_func(int key, int mask) {
    return ((unsigned int)key * 2654435761u) & mask;
}

// Warp-level reduction to find minimum timestamp AND its position
__device__ __forceinline__ void warp_reduce_min_pair(int& val, int& idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        int other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val < val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

template <int HASH_TABLE_SIZE>
__global__ void optimized_lru_update_kernel(
    const int64_t* __restrict__ topk_indices,   // [B, KH, top_budget]
    int* __restrict__ lru_indices,          // [B, KH, allocated_size]
    int* __restrict__ lru_timestamps,       // [B, KH, allocated_size]
    int* __restrict__ current_time,         // [B, KH]
    const __nv_bfloat16* __restrict__ global_k,     
    const __nv_bfloat16* __restrict__ global_v,     
    __nv_bfloat16* __restrict__ cache_k,            
    __nv_bfloat16* __restrict__ cache_v,  
    float* __restrict__ hit_rates,
    const int B, 
    const int KH, 
    const int cache_size,           
    const int allocated_size,       
    const int top_budget, 
    const int D, 
    const int MaxT
) {
    // -----------------------------------------------------------
    // 0. Setup & Shared Memory Layout
    // -----------------------------------------------------------
    constexpr int HASH_MASK = HASH_TABLE_SIZE - 1;
    int b = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    int batch_head_idx = b * KH + head_idx;

    // Shared Memory Layout:
    extern __shared__ char shared_mem[];
    int* hash_table = reinterpret_cast<int*>(shared_mem); 
    int* hash_values = hash_table + HASH_TABLE_SIZE;
    
    // Pointers for stable storage (placed after the hash table area)
    int* s_miss_indices = hash_values + HASH_TABLE_SIZE;
    int* s_victim_positions = s_miss_indices + top_budget;
    int* s_miss_count = s_victim_positions + top_budget;
    int* s_hit_count = s_miss_count + 1;
    int* s_global_base_time = s_hit_count + 1; // Used for coalesced atomic add
    int* s_warp_min_val = s_global_base_time + 1;
    int* s_warp_min_pos = s_warp_min_val + num_warps;

    // Initialize counters
    if (tid == 0) {
        *s_miss_count = 0;
        *s_hit_count = 0;
        *s_global_base_time = 0;
    }

    // Clear Hash Table
    for (int i = tid; i < HASH_TABLE_SIZE; i += blockDim.x) {
        hash_table[i] = EMPTY_SLOT;
        hash_values[i] = EMPTY_SLOT;
    }
    __syncthreads();

    // Pointers to Global Memory
    int* global_lru_indices_ptr = lru_indices + (batch_head_idx * allocated_size);
    int* global_lru_timestamps_ptr = lru_timestamps + (batch_head_idx * allocated_size);

    // -----------------------------------------------------------
    // Phase 1: Build Hash Table
    // -----------------------------------------------------------
    for (int i = tid; i < cache_size; i += blockDim.x) {
        int token_idx = global_lru_indices_ptr[i];
        if (token_idx >= 0) {
            int hash_pos = hash_func(token_idx, HASH_MASK);
            while (true) {
                int old = atomicCAS(&hash_values[hash_pos], EMPTY_SLOT, token_idx);
                if (old == EMPTY_SLOT || old == token_idx) {
                    if (old == EMPTY_SLOT) hash_table[hash_pos] = i;
                    break;
                }
                hash_pos = (hash_pos + 1) & HASH_MASK;
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------
    // Phase 2: Process TopK (Check Hit/Miss)
    // -----------------------------------------------------------
    const int64_t* my_topk_ptr = topk_indices + (batch_head_idx * top_budget);
    
    // Per-thread registers to store operation result
    int my_hit_cache_pos[32];
    int my_hit_ranks[32];
    int my_hit_count = 0;

    for (int i = tid; i < top_budget; i += blockDim.x) {
        int candidate_idx = static_cast<int>(my_topk_ptr[i]);
        if (candidate_idx >= 0) {
            int hash_pos = hash_func(candidate_idx, HASH_MASK);
            int cache_pos = -1;
            
            while (hash_values[hash_pos] != EMPTY_SLOT) {
                if (hash_values[hash_pos] == candidate_idx) {
                    cache_pos = hash_table[hash_pos];
                    break;
                }
                hash_pos = (hash_pos + 1) & HASH_MASK;
            }
            
            if (cache_pos >= 0) {
                // HIT
                if (my_hit_count < 32) {
                    my_hit_cache_pos[my_hit_count] = cache_pos;
                    my_hit_ranks[my_hit_count] = atomicAdd(s_hit_count, 1);
                    my_hit_count++;
                }
            } else {
                // MISS
                int miss_idx = atomicAdd(s_miss_count, 1);
                s_miss_indices[miss_idx] = candidate_idx;
            }
        }
    }
    __syncthreads();

    int num_misses = *s_miss_count;
    int num_hits = *s_hit_count;

    // Report Hit Rate
    if (tid == 0) {
        float rate = 0.0f;
        int total = num_hits + num_misses;
        if (total > 0) rate = (float)num_hits / total;
        hit_rates[batch_head_idx] = rate;

        if (num_hits > 0) {
            *s_global_base_time = atomicAdd(&current_time[batch_head_idx], num_hits);
        }
    }
    __syncthreads();

    // Apply Hit Updates to Global Memory (Parallel)
    int base = *s_global_base_time;
    for (int k = 0; k < my_hit_count; k++) {
        int new_ts = base + my_hit_ranks[k] + 1; 
        global_lru_timestamps_ptr[my_hit_cache_pos[k]] = new_ts;
    }
    __syncthreads();
    
    if (num_misses == 0) return;

    // -----------------------------------------------------------
    // Phase 3: Optimized Victim Finding (Using Shared Memory)
    // -----------------------------------------------------------
    int* s_ts_cache = hash_table; 

    __shared__ int s_insert_base_time;
    if (tid == 0) {
        s_insert_base_time = atomicAdd(&current_time[batch_head_idx], num_misses) + 1;
    }

    for (int i = tid; i < cache_size; i += blockDim.x) {
        int idx = global_lru_indices_ptr[i];
        int ts = global_lru_timestamps_ptr[i];
        if (idx < 0) ts = -1; 
        s_ts_cache[i] = ts;
    }
    __syncthreads();

    for (int m = 0; m < num_misses; m++) {
        int local_min_ts = INVALID_TIME;
        int local_min_pos = -1;

        for (int i = tid; i < cache_size; i += blockDim.x) {
            int ts = s_ts_cache[i];
            if (ts != INVALID_TIME) {
                if (ts < local_min_ts) {
                    local_min_ts = ts;
                    local_min_pos = i;
                }
            }
        }

        warp_reduce_min_pair(local_min_ts, local_min_pos);

        if (lane_id == 0) {
            s_warp_min_val[warp_id] = local_min_ts;
            s_warp_min_pos[warp_id] = local_min_pos;
        }
        __syncthreads();

        if (tid == 0) {
            int global_min_ts = INVALID_TIME;
            int global_min_pos = -1;
            
            for (int w = 0; w < num_warps; w++) {
                if (s_warp_min_val[w] < global_min_ts) {
                    global_min_ts = s_warp_min_val[w];
                    global_min_pos = s_warp_min_pos[w];
                }
            }
            s_victim_positions[m] = global_min_pos;
            if (global_min_pos != -1) {
                s_ts_cache[global_min_pos] = INVALID_TIME;
            }
        }
        __syncthreads();
    }

    // -----------------------------------------------------------
    // Phase 4: Update Global Metadata (Indices & timestamps for misses)
    // -----------------------------------------------------------
    if (tid == 0) {
        int base_time = s_insert_base_time;
        for (int i = 0; i < num_misses; i++) {
            int victim_pos = s_victim_positions[i];
            int new_token_idx = s_miss_indices[i];
            
            if (victim_pos >= 0) {
                global_lru_indices_ptr[victim_pos] = new_token_idx;
                global_lru_timestamps_ptr[victim_pos] = base_time + i;
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------
    // Phase 5: Vectorized Data Copy (K/V)
    // -----------------------------------------------------------
    int D_vec = D / 8; 
    int total_tasks = num_misses * D_vec;

    for (int task_id = tid; task_id < total_tasks; task_id += blockDim.x) {
        int miss_idx = task_id / D_vec;
        int d_vec_idx = task_id % D_vec;
        int d_offset = d_vec_idx * 8;

        int token_global_idx = s_miss_indices[miss_idx];
        int write_pos = s_victim_positions[miss_idx];

        if (write_pos >= 0) {
            long src_offset = ((long)b * MaxT * KH * D) + 
                              ((long)token_global_idx * KH * D) + 
                              ((long)head_idx * D) + 
                              d_offset;

            long dst_offset = ((long)b * allocated_size * KH * D) + 
                              ((long)write_pos * KH * D) + 
                              ((long)head_idx * D) + 
                              d_offset;

            float4 k_val = BF16X8_READ(global_k[src_offset]);
            BF16X8_WRITE(cache_k[dst_offset]) = k_val;

            float4 v_val = BF16X8_READ(global_v[src_offset]);
            BF16X8_WRITE(cache_v[dst_offset]) = v_val;
        }
    }
}

// -----------------------------------------------------------
// C++ Host Wrapper
// -----------------------------------------------------------
torch::Tensor lru_update_cuda(
    torch::Tensor topk_indices,
    torch::Tensor lru_indices,
    torch::Tensor lru_timestamps,
    torch::Tensor current_time,
    torch::Tensor global_k,
    torch::Tensor global_v,
    torch::Tensor cache_k,
    torch::Tensor cache_v,
    int cache_size,
    int top_budget
) {
    // Basic checks
    TORCH_CHECK(global_k.scalar_type() == torch::kBFloat16, "global_k must be bf16");
    TORCH_CHECK(cache_k.scalar_type() == torch::kBFloat16, "cache_k must be bf16");
    TORCH_CHECK(lru_timestamps.scalar_type() == torch::kInt32, "timestamps must be int32");
    

    const int B = topk_indices.size(0);
    const int KH = topk_indices.size(1);
    const int MaxT = global_k.size(1);
    const int D = global_k.size(3);
    const int allocated_size = cache_k.size(1);
    int hash_size = 1024;
    if (cache_size * 2 > 8192) hash_size = 16384;
    else if (cache_size * 2 > 4096) hash_size = 8192;
    else if (cache_size * 2 > 2048) hash_size = 4096;
    else if (cache_size * 2 > 1024) hash_size = 2048;
    else hash_size = 1024;

    TORCH_CHECK(cache_size <= hash_size / 2, "cache_size too large for compiled hash sizes");

    auto hit_rates = torch::empty({B, KH}, topk_indices.options().dtype(torch::kFloat32));

    dim3 grid(B, KH);
    int threads = 256;
    
    size_t smem_size = (hash_size * 2 * sizeof(int)) 
                     + (top_budget * 2 * sizeof(int))      
                     + (128 * sizeof(int));                

    auto stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH_LRU_KERNEL(SIZE) \
        optimized_lru_update_kernel<SIZE><<<grid, threads, smem_size, stream>>>( \
            topk_indices.data_ptr<int64_t>(), \
            lru_indices.data_ptr<int>(), \
            lru_timestamps.data_ptr<int>(), \
            current_time.data_ptr<int>(), \
            reinterpret_cast<const __nv_bfloat16*>(global_k.data_ptr<at::BFloat16>()), \
            reinterpret_cast<const __nv_bfloat16*>(global_v.data_ptr<at::BFloat16>()), \
            reinterpret_cast<__nv_bfloat16*>(cache_k.data_ptr<at::BFloat16>()), \
            reinterpret_cast<__nv_bfloat16*>(cache_v.data_ptr<at::BFloat16>()), \
            hit_rates.data_ptr<float>(), \
            B, KH, cache_size, allocated_size, top_budget, D, MaxT \
        );

    if (smem_size > 48 * 1024) {
        int device;
        cudaGetDevice(&device);
        int max_smem;
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (smem_size <= (size_t)max_smem) {
            switch(hash_size) {
                case 1024:  cudaFuncSetAttribute(optimized_lru_update_kernel<1024>,  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); break;
                case 2048:  cudaFuncSetAttribute(optimized_lru_update_kernel<2048>,  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); break;
                case 4096:  cudaFuncSetAttribute(optimized_lru_update_kernel<4096>,  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); break;
                case 8192:  cudaFuncSetAttribute(optimized_lru_update_kernel<8192>,  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); break;
                case 16384: cudaFuncSetAttribute(optimized_lru_update_kernel<16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); break;
            }
        }
    }

    switch (hash_size) {
        case 1024:  LAUNCH_LRU_KERNEL(1024); break;
        case 2048:  LAUNCH_LRU_KERNEL(2048); break;
        case 4096:  LAUNCH_LRU_KERNEL(4096); break;
        case 8192:  LAUNCH_LRU_KERNEL(8192); break;
        case 16384: LAUNCH_LRU_KERNEL(16384); break;
        default: TORCH_CHECK(false, "Unsupported hash size");
    }

    return hit_rates;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update", &lru_update_cuda, "Optimized True LRU Update");
}