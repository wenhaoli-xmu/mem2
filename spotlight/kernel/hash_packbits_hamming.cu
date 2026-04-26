/**
 * Fused Hash + Packbits + Hamming Distance Kernel
 * 
 * Combines:
 *   1. HashModule (2-layer MLP) for query
 *   2. Packbits (on-the-fly, no intermediate storage)
 *   3. Hamming distance computation with all cached key_bins
 * 
 * This eliminates the intermediate q_bin tensor allocation.
 * 
 * Input:
 *   query: [B, 1, QH, D] bf16 - single query token
 *   proj0: [QH, D, D] bf16 - first layer weights
 *   proj1: [QH, D, D] bf16 - second layer weights
 *   key_bins: [B, MaxT, KH, D/32] int32 - cached key bins
 *   kv_length: scalar int32 - actual number of cached keys
 * 
 * Output:
 *   hamming: [B, KH, 1, MaxT] int16 - hamming distances
 */

#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

namespace {

// Configuration - compile time constants (hash dimension is always 128)
constexpr int BLOCK_N = 64;    // Keys per block
constexpr int D_PACK = 4;      // D/32 = 128/32 = 4 int32s per head
constexpr int D = 128;         // Hash dimension
// KH, QH, G are now runtime parameters to support different GQA configs

// Helper: bf16 → float
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// SiLU activation
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * Kernel: hash_packbits_hamming_kernel
 * 
 * Each block computes hamming distances for one (batch, kv_head, key_block) tuple.
 * 
 * Grid: (B * KH * n_blocks, 1, 1)
 * Block: (BLOCK_N, 1, 1)
 * 
 * Shared memory layout:
 *   - q_packed[G][D_PACK]: packed query bins for G query heads in this KV group
 *   - k_packed[BLOCK_N][D_PACK]: packed key bins for BLOCK_N keys
 */
__global__ void hash_packbits_hamming_kernel(
    const __nv_bfloat16* __restrict__ query,     // [B, 1, QH, D]
    const __nv_bfloat16* __restrict__ proj0,     // [QH, D, D]
    const __nv_bfloat16* __restrict__ proj1,     // [QH, D, D]
    const int32_t* __restrict__ key_bins,        // [B, MaxT, KH, D_PACK]
    int16_t* __restrict__ output,                // [B, KH, 1, MaxT]
    const int32_t* __restrict__ kv_length_ptr,
    int B, int MaxT, int n_blocks,
    int KH, int QH, int G
) {
    // Decode block index
    int linear_idx = blockIdx.x;
    int n_block_idx = linear_idx % n_blocks;
    linear_idx /= n_blocks;
    int kv_head = linear_idx % KH;
    int b = linear_idx / KH;
    
    int n_start = n_block_idx * BLOCK_N;
    int n_local = threadIdx.x;
    int n = n_start + n_local;
    
    int KN_active = *kv_length_ptr;
    if (KN_active < 0) KN_active = 0;
    if (KN_active > MaxT) KN_active = MaxT;
    
    // Shared memory
    extern __shared__ char shared_raw[];
    
    // Layout: q_hash[G][D], y_temp[D], q_packed[G][D_PACK], k_packed[BLOCK_N][D_PACK]
    float* q_hash = reinterpret_cast<float*>(shared_raw);                           // [G][D]
    float* y_temp = q_hash + G * D;                                                   // [D]
    int32_t* q_packed = reinterpret_cast<int32_t*>(y_temp + D);                       // [G][D_PACK]
    int32_t* k_packed = q_packed + G * D_PACK;                                        // [BLOCK_N][D_PACK]
    
    // Step 1: Compute hash for G query heads that belong to this KV head
    // Query heads [kv_head * G, kv_head * G + G - 1] map to this kv_head
    
    // Each thread helps compute the hash for all G query heads
    // We'll do this cooperatively
    
    for (int g = 0; g < G; g++) {
        int q_head = kv_head * G + g;
        size_t q_offset = (size_t)b * QH * D + (size_t)q_head * D;
        size_t proj_offset = (size_t)q_head * D * D;
        
        // Load query input into y_temp (using it as scratch for input)
        __syncthreads();
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            y_temp[i] = to_float(query[q_offset + i]);
        }
        __syncthreads();
        
        // Layer 0: z = silu(y_temp @ proj0) + y_temp
        float* z_out = &q_hash[g * D];  // Store in q_hash for this head
        for (int out_idx = threadIdx.x; out_idx < D; out_idx += blockDim.x) {
            float acc = 0.0f;
            for (int k = 0; k < D; k++) {
                float proj_val = to_float(proj0[proj_offset + k * D + out_idx]);
                acc += y_temp[k] * proj_val;
            }
            z_out[out_idx] = silu(acc) + y_temp[out_idx];
        }
        __syncthreads();
        
        // Copy z_out to y_temp for layer 1 input
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            y_temp[i] = z_out[i];
        }
        __syncthreads();
        
        // Layer 1: z = (y_temp @ proj1) + y_temp
        for (int out_idx = threadIdx.x; out_idx < D; out_idx += blockDim.x) {
            float acc = 0.0f;
            for (int k = 0; k < D; k++) {
                float proj_val = to_float(proj1[proj_offset + k * D + out_idx]);
                acc += y_temp[k] * proj_val;
            }
            z_out[out_idx] = acc + y_temp[out_idx];
        }
        __syncthreads();
    }
    
    // Step 2: Packbits for all G query heads
    // Each warp packs 32 bits
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    
    for (int g = 0; g < G; g++) {
        float* z_head = &q_hash[g * D];
        for (int pack_idx = warp_id; pack_idx < D_PACK; pack_idx += num_warps) {
            int bit_idx = pack_idx * 32 + lane_id;
            bool bit = (z_head[bit_idx] > 0.0f);
            uint32_t packed = __ballot_sync(0xFFFFFFFF, bit);
            if (lane_id == 0) {
                q_packed[g * D_PACK + pack_idx] = static_cast<int32_t>(packed);
            }
        }
    }
    __syncthreads();
    
    // Step 3: Load key_bins for this block
    // key_bins: [B, MaxT, KH, D_PACK]
    size_t k_base = (size_t)b * MaxT * KH * D_PACK + (size_t)kv_head * D_PACK;
    
    for (int idx = threadIdx.x; idx < BLOCK_N * D_PACK; idx += blockDim.x) {
        int n_local_load = idx / D_PACK;
        int d_idx = idx % D_PACK;
        int n_global = n_start + n_local_load;
        
        if (n_global < KN_active) {
            k_packed[n_local_load * D_PACK + d_idx] = 
                key_bins[k_base + (size_t)n_global * KH * D_PACK + d_idx];
        } else {
            k_packed[n_local_load * D_PACK + d_idx] = 0;
        }
    }
    __syncthreads();
    
    // Step 4: Compute hamming distance
    // Each thread handles one key position
    if (n < MaxT) {
        int accum = 0;
        
        // Sum over all G query heads in this group
        for (int g = 0; g < G; g++) {
            int32_t* q_head = &q_packed[g * D_PACK];
            int32_t* k_head = &k_packed[n_local * D_PACK];
            
            #pragma unroll
            for (int d = 0; d < D_PACK; d++) {
                accum += __popc(~(q_head[d] ^ k_head[d]));
            }
        }
        
        // Output: [B, KH, 1, MaxT]
        size_t out_idx = (size_t)b * KH * MaxT + (size_t)kv_head * MaxT + (size_t)n;
        
        if (n < KN_active) {
            output[out_idx] = static_cast<int16_t>(accum);
        } else {
            output[out_idx] = static_cast<int16_t>(-32768);
        }
    }
}

} // namespace

/**
 * Python interface: hash_packbits_hamming
 * 
 * Fused hash + packbits + hamming for query path.
 * 
 * Args:
 *   query: [B, 1, QH, D] bf16 tensor (single token)
 *   proj0: [QH, D, D] bf16 tensor
 *   proj1: [QH, D, D] bf16 tensor
 *   key_bins: [B, MaxT, KH, D/32] int32 tensor
 *   kv_length: scalar int32 tensor
 * 
 * Returns:
 *   output: [B, KH, MaxT] int16 tensor (squeezed from [B, KH, 1, MaxT])
 */
torch::Tensor hash_packbits_hamming(
    torch::Tensor query,
    torch::Tensor proj0,
    torch::Tensor proj1,
    torch::Tensor key_bins,
    torch::Tensor kv_length
) {
    TORCH_CHECK(query.is_cuda(), "query must be on CUDA");
    TORCH_CHECK(proj0.is_cuda(), "proj0 must be on CUDA");
    TORCH_CHECK(proj1.is_cuda(), "proj1 must be on CUDA");
    TORCH_CHECK(key_bins.is_cuda(), "key_bins must be on CUDA");
    TORCH_CHECK(kv_length.is_cuda(), "kv_length must be on CUDA");
    
    TORCH_CHECK(query.scalar_type() == torch::kBFloat16, "query must be bfloat16");
    TORCH_CHECK(proj0.scalar_type() == torch::kBFloat16, "proj0 must be bfloat16");
    TORCH_CHECK(proj1.scalar_type() == torch::kBFloat16, "proj1 must be bfloat16");
    TORCH_CHECK(key_bins.scalar_type() == torch::kInt32, "key_bins must be int32");
    TORCH_CHECK(kv_length.scalar_type() == torch::kInt32, "kv_length must be int32");
    
    TORCH_CHECK(query.dim() == 4, "query must be 4D [B, 1, QH, D]");
    TORCH_CHECK(query.size(1) == 1, "query sequence length must be 1");
    TORCH_CHECK(key_bins.dim() == 4, "key_bins must be 4D [B, MaxT, KH, D/32]");
    
    query = query.contiguous();
    proj0 = proj0.contiguous();
    proj1 = proj1.contiguous();
    key_bins = key_bins.contiguous();
    
    int B = query.size(0);
    int Q_H = query.size(2);
    int D_dim = query.size(3);
    int MaxT = key_bins.size(1);
    int K_H = key_bins.size(2);
    
    TORCH_CHECK(key_bins.size(3) == D_PACK, "key_bins last dim must be D/32");
    TORCH_CHECK(Q_H % K_H == 0, "QH must be divisible by KH");
    
    int G = Q_H / K_H;
    
    torch::Device device = query.device();
    
    // Output: [B, K_H, MaxT]
    torch::Tensor output = torch::empty(
        {B, K_H, MaxT},
        torch::dtype(torch::kInt16).device(device)
    );
    
    // Launch kernel
    int n_blocks = (MaxT + BLOCK_N - 1) / BLOCK_N;
    int total_blocks = B * K_H * n_blocks;
    int threads_per_block = BLOCK_N;
    
    // Shared memory: q_hash[G][D] + y_temp[D] + q_packed[G][D_PACK] + k_packed[BLOCK_N][D_PACK]
    size_t shared_mem_size = (G * D + D) * sizeof(float) 
                           + (G * D_PACK + BLOCK_N * D_PACK) * sizeof(int32_t);
    
    hash_packbits_hamming_kernel<<<total_blocks, threads_per_block, shared_mem_size,
                                    at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(proj0.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(proj1.data_ptr<at::BFloat16>()),
        key_bins.data_ptr<int32_t>(),
        output.data_ptr<int16_t>(),
        kv_length.data_ptr<int32_t>(),
        B, MaxT, n_blocks,
        K_H, Q_H, G
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hash_packbits_hamming", &hash_packbits_hamming,
          "Fused hash + packbits + hamming kernel for query path",
          py::arg("query"), py::arg("proj0"), py::arg("proj1"),
          py::arg("key_bins"), py::arg("kv_length"));
}

