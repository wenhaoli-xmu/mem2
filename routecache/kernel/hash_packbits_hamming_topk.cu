/**
 * Fused Hash + Packbits + Hamming Distance + Top-K Kernel
 *
 * Fuses hash_packbits_hamming + torch.topk into a single kernel launch,
 * eliminating the [B, KH, MaxT] intermediate tensor.
 *
 * Two-phase approach:
 *   Phase 1: Each block processes BLOCK_N=64 keys, computes hamming distances,
 *            keeps block-local top LOCAL_K candidates.
 *            Also computes query hash + packbits (same as original kernel).
 *   Phase 2: Each block handles one (batch, kv_head), merges all block-local
 *            candidates into final top_budget indices.
 *
 * Input:
 *   query: [B, 1, QH, D] bf16
 *   proj0: [QH, D, D] bf16
 *   proj1: [QH, D, D] bf16
 *   key_bins: [B, MaxT, KH, D/32] int32
 *   kv_length: scalar int32
 *   top_budget: int
 *
 * Output:
 *   topk_indices: [B, KH, top_budget] int64
 */

#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int BLOCK_N = 64;
constexpr int D_PACK = 4;      // 128 / 32
constexpr int D = 128;
constexpr int LOCAL_K = 64;    // block-local top-k candidates

__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * Phase 1: hash + hamming + block-local top-k
 *
 * Grid: (B * KH * n_blocks)
 * Block: (BLOCK_N) = 64 threads
 *
 * Each block:
 *   1. Computes query hash for G query heads (same as original)
 *   2. Loads BLOCK_N key_bins from global memory
 *   3. Computes hamming distance for each of 64 keys
 *   4. Thread 0 collects scores, keeps LOCAL_K best via insertion sort
 *   5. Writes block-local top-k to intermediate buffer
 *
 * Output:
 *   block_vals: [B, KH, n_blocks, LOCAL_K] int16
 *   block_idxs: [B, KH, n_blocks, LOCAL_K] int32
 */
__global__ void phase1_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ proj0,
    const __nv_bfloat16* __restrict__ proj1,
    const int32_t* __restrict__ key_bins,
    int16_t* __restrict__ block_vals,
    int32_t* __restrict__ block_idxs,
    const int32_t* __restrict__ kv_length_ptr,
    int B, int MaxT, int n_blocks,
    int KH, int QH, int G
) {
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
    float* q_hash = reinterpret_cast<float*>(shared_raw);
    float* y_temp = q_hash + G * D;
    int32_t* q_packed = reinterpret_cast<int32_t*>(y_temp + D);
    int32_t* k_packed = q_packed + G * D_PACK;
    // block-local top-k storage after k_packed
    int16_t* local_vals = reinterpret_cast<int16_t*>(k_packed + BLOCK_N * D_PACK);
    int32_t* local_idxs = reinterpret_cast<int32_t*>(local_vals + LOCAL_K);

    // Step 1: Query hash (identical to original)
    for (int g = 0; g < G; g++) {
        int q_head = kv_head * G + g;
        size_t q_offset = (size_t)b * QH * D + (size_t)q_head * D;
        size_t proj_offset = (size_t)q_head * D * D;

        __syncthreads();
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            y_temp[i] = to_float(query[q_offset + i]);
        }
        __syncthreads();

        float* z_out = &q_hash[g * D];
        for (int out_idx = threadIdx.x; out_idx < D; out_idx += blockDim.x) {
            float acc = 0.0f;
            for (int k = 0; k < D; k++) {
                acc += y_temp[k] * to_float(proj0[proj_offset + k * D + out_idx]);
            }
            z_out[out_idx] = silu(acc) + y_temp[out_idx];
        }
        __syncthreads();

        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            y_temp[i] = z_out[i];
        }
        __syncthreads();

        for (int out_idx = threadIdx.x; out_idx < D; out_idx += blockDim.x) {
            float acc = 0.0f;
            for (int k = 0; k < D; k++) {
                acc += y_temp[k] * to_float(proj1[proj_offset + k * D + out_idx]);
            }
            z_out[out_idx] = acc + y_temp[out_idx];
        }
        __syncthreads();
    }

    // Step 2: Packbits
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

    // Step 3: Load key_bins
    size_t k_base = (size_t)b * MaxT * KH * D_PACK + (size_t)kv_head * D_PACK;
    for (int idx = threadIdx.x; idx < BLOCK_N * D_PACK; idx += blockDim.x) {
        int nl = idx / D_PACK;
        int di = idx % D_PACK;
        int ng = n_start + nl;
        if (ng < KN_active) {
            k_packed[nl * D_PACK + di] = key_bins[k_base + (size_t)ng * KH * D_PACK + di];
        } else {
            k_packed[nl * D_PACK + di] = 0;
        }
    }
    __syncthreads();

    // Step 4: Each thread computes its hamming distance
    int16_t my_score;
    if (n < KN_active) {
        int accum = 0;
        for (int g = 0; g < G; g++) {
            int32_t* qh = &q_packed[g * D_PACK];
            int32_t* kh = &k_packed[n_local * D_PACK];
            #pragma unroll
            for (int d = 0; d < D_PACK; d++) {
                accum += __popc(~(qh[d] ^ kh[d]));
            }
        }
        my_score = static_cast<int16_t>(accum);
    } else {
        my_score = static_cast<int16_t>(-32768);
    }

    // Step 5: Block-local top-k via warp-level parallel collection
    // Put score into shared memory for thread 0 to process
    // Reuse y_temp area for scores (D=128 floats = 512 bytes, we need 64 int16 = 128 bytes)
    int16_t* block_scores = reinterpret_cast<int16_t*>(y_temp);
    int32_t* block_positions = reinterpret_cast<int32_t*>(block_scores + BLOCK_N);

    block_scores[n_local] = my_score;
    block_positions[n_local] = n;  // global position
    __syncthreads();

    // Thread 0 does insertion into local top-k buffer
    // LOCAL_K=64 = BLOCK_N, so we keep all valid entries from this block
    // (Just copy: each block has at most BLOCK_N=64 entries, LOCAL_K=64)
    if (threadIdx.x == 0) {
        for (int i = 0; i < LOCAL_K; i++) {
            local_vals[i] = block_scores[i];
            local_idxs[i] = block_positions[i];
        }
    }
    __syncthreads();

    // Write block-local results to global
    // block_vals: [B, KH, n_blocks, LOCAL_K]
    size_t out_base = ((size_t)b * KH + kv_head) * n_blocks * LOCAL_K
                    + (size_t)n_block_idx * LOCAL_K;
    for (int i = threadIdx.x; i < LOCAL_K; i += blockDim.x) {
        block_vals[out_base + i] = local_vals[i];
        block_idxs[out_base + i] = local_idxs[i];
    }
}


/**
 * Phase 2: Merge block-local top-k into global top-k
 *
 * Grid: (B * KH)
 * Block: (256) threads
 *
 * Each block handles one (batch, kv_head):
 *   - Reads n_blocks * LOCAL_K candidates directly from global memory
 *   - Histogram-based O(n) selection to find threshold
 *   - Two-pass collect: above threshold, then equal to threshold
 *
 * Shared memory: histogram[1024] + 3 scalars = ~4KB (fixed)
 */
__global__ void phase2_kernel(
    const int16_t* __restrict__ block_vals,  // [B, KH, n_blocks, LOCAL_K]
    const int32_t* __restrict__ block_idxs,  // [B, KH, n_blocks, LOCAL_K]
    int64_t* __restrict__ topk_out,          // [B, KH, top_budget]
    int B, int KH, int n_blocks, int top_budget
) {
    int linear_idx = blockIdx.x;
    int kv_head = linear_idx % KH;
    int b = linear_idx / KH;

    int total_candidates = n_blocks * LOCAL_K;
    size_t in_base = ((size_t)b * KH + kv_head) * n_blocks * LOCAL_K;
    size_t out_base = ((size_t)b * KH + kv_head) * top_budget;

    // Shared memory: only histogram + counters (fixed ~4KB)
    constexpr int HIST_SIZE = 1024;
    __shared__ int histogram[HIST_SIZE];
    __shared__ int16_t threshold;
    __shared__ int out_pos;

    // Step 1: Build histogram from global memory
    for (int i = threadIdx.x; i < HIST_SIZE; i += blockDim.x) {
        histogram[i] = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < total_candidates; i += blockDim.x) {
        int score = static_cast<int>(block_vals[in_base + i]);
        int bin = score;
        if (bin < 0) bin = 0;
        if (bin >= HIST_SIZE) bin = HIST_SIZE - 1;
        atomicAdd(&histogram[bin], 1);
    }
    __syncthreads();

    // Thread 0: scan from top to find threshold
    if (threadIdx.x == 0) {
        int cumsum = 0;
        threshold = 0;
        for (int bin = HIST_SIZE - 1; bin >= 0; bin--) {
            cumsum += histogram[bin];
            if (cumsum >= top_budget) {
                threshold = static_cast<int16_t>(bin);
                break;
            }
        }
        out_pos = 0;
    }
    __syncthreads();

    // Step 2: Collect from global memory — above threshold first
    for (int i = threadIdx.x; i < total_candidates; i += blockDim.x) {
        if (block_vals[in_base + i] > threshold) {
            int pos = atomicAdd(&out_pos, 1);
            if (pos < top_budget) {
                topk_out[out_base + pos] = static_cast<int64_t>(block_idxs[in_base + i]);
            }
        }
    }
    __syncthreads();

    // Fill remaining with elements equal to threshold
    for (int i = threadIdx.x; i < total_candidates; i += blockDim.x) {
        if (block_vals[in_base + i] == threshold) {
            int pos = atomicAdd(&out_pos, 1);
            if (pos < top_budget) {
                topk_out[out_base + pos] = static_cast<int64_t>(block_idxs[in_base + i]);
            }
        }
    }
}

} // namespace


/**
 * Python interface: hash_packbits_hamming_topk
 */
torch::Tensor hash_packbits_hamming_topk(
    torch::Tensor query,
    torch::Tensor proj0,
    torch::Tensor proj1,
    torch::Tensor key_bins,
    torch::Tensor kv_length,
    int top_budget
) {
    TORCH_CHECK(query.is_cuda(), "query must be on CUDA");
    TORCH_CHECK(proj0.is_cuda(), "proj0 must be on CUDA");
    TORCH_CHECK(proj1.is_cuda(), "proj1 must be on CUDA");
    TORCH_CHECK(key_bins.is_cuda(), "key_bins must be on CUDA");
    TORCH_CHECK(kv_length.is_cuda(), "kv_length must be on CUDA");

    TORCH_CHECK(query.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(proj0.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(proj1.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(key_bins.scalar_type() == torch::kInt32);
    TORCH_CHECK(kv_length.scalar_type() == torch::kInt32);

    TORCH_CHECK(query.dim() == 4 && query.size(1) == 1);
    TORCH_CHECK(key_bins.dim() == 4);

    query = query.contiguous();
    proj0 = proj0.contiguous();
    proj1 = proj1.contiguous();
    key_bins = key_bins.contiguous();

    int B = query.size(0);
    int Q_H = query.size(2);
    int MaxT = key_bins.size(1);
    int K_H = key_bins.size(2);

    TORCH_CHECK(key_bins.size(3) == D_PACK);
    TORCH_CHECK(Q_H % K_H == 0);

    int G = Q_H / K_H;
    int n_blocks = (MaxT + BLOCK_N - 1) / BLOCK_N;

    torch::Device device = query.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Intermediate: block-local top-k
    torch::Tensor block_vals = torch::empty(
        {B, K_H, n_blocks, LOCAL_K}, torch::dtype(torch::kInt16).device(device));
    torch::Tensor block_idxs = torch::empty(
        {B, K_H, n_blocks, LOCAL_K}, torch::dtype(torch::kInt32).device(device));

    // Output
    torch::Tensor topk_out = torch::zeros(
        {B, K_H, top_budget}, torch::dtype(torch::kInt64).device(device));

    // Phase 1
    {
        int total_blocks_p1 = B * K_H * n_blocks;
        int threads_p1 = BLOCK_N;  // 64

        // Shared: q_hash[G*D] + y_temp[D] + q_packed[G*D_PACK] + k_packed[BLOCK_N*D_PACK]
        //       + local_vals[LOCAL_K] + local_idxs[LOCAL_K]
        size_t smem_p1 = (G * D + D) * sizeof(float)
                       + (G * D_PACK + BLOCK_N * D_PACK) * sizeof(int32_t)
                       + LOCAL_K * sizeof(int16_t)
                       + LOCAL_K * sizeof(int32_t);

        phase1_kernel<<<total_blocks_p1, threads_p1, smem_p1, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(proj0.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(proj1.data_ptr<at::BFloat16>()),
            key_bins.data_ptr<int32_t>(),
            block_vals.data_ptr<int16_t>(),
            block_idxs.data_ptr<int32_t>(),
            kv_length.data_ptr<int32_t>(),
            B, MaxT, n_blocks, K_H, Q_H, G
        );
    }

    // Phase 2
    {
        int total_blocks_p2 = B * K_H;
        int threads_p2 = 256;

        // Shared: histogram[1024] + threshold + out_pos ≈ 4KB (fixed)
        size_t smem_p2 = 1024 * sizeof(int) + sizeof(int16_t) + sizeof(int);

        phase2_kernel<<<total_blocks_p2, threads_p2, smem_p2, stream>>>(
            block_vals.data_ptr<int16_t>(),
            block_idxs.data_ptr<int32_t>(),
            topk_out.data_ptr<int64_t>(),
            B, K_H, n_blocks, top_budget
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    return topk_out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hash_packbits_hamming_topk", &hash_packbits_hamming_topk,
          "Fused hash + packbits + hamming + topk kernel",
          py::arg("query"), py::arg("proj0"), py::arg("proj1"),
          py::arg("key_bins"), py::arg("kv_length"), py::arg("top_budget"));
}
