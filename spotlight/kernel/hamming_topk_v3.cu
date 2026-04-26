#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>

// ============================================================
// hamming_topk_v3: Optimized over v2
//   1. key_bins layout: [B, KH, MaxT, D_PACK]  (coalesced access)
//   2. Phase1: G_SPEC template specialization with unrolled scoring
//   3. Phase1: int4 aligned 16-byte vectorized key reads
//   4. Phase2: kv_length-bounded active range scan
// ============================================================

namespace {

constexpr int D_PACK = 4;
constexpr uint32_t IDX_MASK  = (1u << 18) - 1u;
constexpr uint32_t SCORE_MASK = (1u << 10) - 1u;

__device__ __forceinline__ uint32_t pack_candidate_u32(int score, int idx, bool valid) {
    uint32_t v = valid ? 0x80000000u : 0u;
    uint32_t s = (static_cast<uint32_t>(score) & SCORE_MASK) << 18;
    uint32_t i = static_cast<uint32_t>(idx) & IDX_MASK;
    return v | s | i;
}

// -----------------------------------------------------------------
// Compile-time unrolled scoring for known G values
// -----------------------------------------------------------------
template<int G_SPEC>
__device__ __forceinline__ int score_unrolled(
    const int32_t* __restrict__ q_ptr,
    int32_t k0, int32_t k1, int32_t k2, int32_t k3
) {
    int accum = 0;
    #pragma unroll
    for (int g = 0; g < G_SPEC; ++g) {
        const int32_t* qh = q_ptr + g * D_PACK;
        accum += __popc(~(qh[0] ^ k0));
        accum += __popc(~(qh[1] ^ k1));
        accum += __popc(~(qh[2] ^ k2));
        accum += __popc(~(qh[3] ^ k3));
    }
    return accum;
}

// Runtime scoring fallback for non-specialized G
__device__ __forceinline__ int score_runtime(
    const int32_t* __restrict__ q_shared,
    int32_t k0, int32_t k1, int32_t k2, int32_t k3,
    int G
) {
    int accum = 0;
    for (int g = 0; g < G; ++g) {
        const int32_t* qh = q_shared + g * D_PACK;
        accum += __popc(~(qh[0] ^ k0));
        accum += __popc(~(qh[1] ^ k1));
        accum += __popc(~(qh[2] ^ k2));
        accum += __popc(~(qh[3] ^ k3));
    }
    return accum;
}

// -----------------------------------------------------------------
// Phase 1:  Hamming distance  +  block-local packing
//   key_bins is [B, KH, MaxT, D_PACK]  => consecutive tokens are
//   contiguous in memory => coalesced 16-byte loads.
//   G_SPEC: compile-time G for unrolled scoring (0 = runtime fallback)
// -----------------------------------------------------------------
template<int BLOCK_N, int THREADS, int G_SPEC>
__global__ void phase1_kernel(
    const int32_t* __restrict__ q_packed,       // [B, KH, G, D_PACK]
    const int32_t* __restrict__ key_bins,       // [B, KH, MaxT, D_PACK]
    uint32_t*      __restrict__ block_packed,   // [B, KH, n_blocks, BLOCK_N]
    const int32_t* __restrict__ kv_length_ptr,
    int B, int MaxT, int KH, int G, int n_blocks
) {
    int linear_idx  = blockIdx.x;
    int n_block_idx = linear_idx % n_blocks;
    linear_idx     /= n_blocks;
    int kv_head     = linear_idx % KH;
    int b           = linear_idx / KH;

    int KN_active = *kv_length_ptr;
    if (KN_active < 0) KN_active = 0;
    if (KN_active > MaxT) KN_active = MaxT;

    int n_start = n_block_idx * BLOCK_N;
    if (n_start >= KN_active) return;

    extern __shared__ int32_t q_shared[];

    for (int i = threadIdx.x; i < G * D_PACK; i += THREADS) {
        q_shared[i] = q_packed[((size_t)b * KH * G * D_PACK)
                              + (size_t)kv_head * G * D_PACK + i];
    }
    __syncthreads();

    // Copy query to registers for compile-time-specialized path
    int32_t q_local[(G_SPEC > 0 ? G_SPEC : 1) * D_PACK];
    const int32_t* q_ptr = q_shared;
    if (G_SPEC > 0) {
        #pragma unroll
        for (int i = 0; i < G_SPEC * D_PACK; ++i) q_local[i] = q_shared[i];
        q_ptr = q_local;
    }

    const int32_t* key_base = key_bins
        + ((size_t)b * KH + kv_head) * (size_t)MaxT * D_PACK;

    // Check 16-byte alignment for vectorized int4 loads
    const bool aligned16 = ((((uintptr_t)key_base) & 0xF) == 0);
    const int4* key_base4 = reinterpret_cast<const int4*>(key_base);

    size_t out_base = (((size_t)b * KH + kv_head) * n_blocks
                       + n_block_idx) * BLOCK_N;

    for (int t = threadIdx.x; t < BLOCK_N; t += THREADS) {
        int n = n_start + t;
        bool valid = (n < KN_active);
        uint32_t packed;

        if (valid) {
            int32_t k0, k1, k2, k3;
            if (aligned16) {
                int4 kv = key_base4[n];
                k0 = kv.x; k1 = kv.y; k2 = kv.z; k3 = kv.w;
            } else {
                const int32_t* k_ptr = key_base + (size_t)n * D_PACK;
                k0 = k_ptr[0]; k1 = k_ptr[1]; k2 = k_ptr[2]; k3 = k_ptr[3];
            }

            int accum;
            if (G_SPEC == 1) {
                accum = score_unrolled<1>(q_ptr, k0, k1, k2, k3);
            } else if (G_SPEC == 2) {
                accum = score_unrolled<2>(q_ptr, k0, k1, k2, k3);
            } else if (G_SPEC == 4) {
                accum = score_unrolled<4>(q_ptr, k0, k1, k2, k3);
            } else if (G_SPEC == 8) {
                accum = score_unrolled<8>(q_ptr, k0, k1, k2, k3);
            } else {
                accum = score_runtime(q_shared, k0, k1, k2, k3, G);
            }
            packed = pack_candidate_u32(accum, n, true);
        } else {
            packed = 0u;
        }

        block_packed[out_base + t] = packed;
    }
}

// -----------------------------------------------------------------
// Phase 2:  Histogram-based top-k with kv_length bounding
// -----------------------------------------------------------------
template<int THREADS>
__global__ void phase2_kernel(
    const uint32_t* __restrict__ block_packed,
    int64_t*        __restrict__ topk_out,
    const int32_t*  __restrict__ kv_length_ptr,
    int B, int KH, int n_blocks, int block_n, int top_budget, int MaxT
) {
    int linear_idx = blockIdx.x;
    int kv_head = linear_idx % KH;
    int b       = linear_idx / KH;

    int KN_active = *kv_length_ptr;
    if (KN_active < 0) KN_active = 0;
    if (KN_active > MaxT) KN_active = MaxT;

    int total_candidates_full = n_blocks * block_n;
    int total_candidates_active = ((KN_active + block_n - 1) / block_n) * block_n;
    if (total_candidates_active > total_candidates_full) {
        total_candidates_active = total_candidates_full;
    }

    size_t in_base  = ((size_t)b * KH + kv_head) * (size_t)total_candidates_full;
    size_t out_base = ((size_t)b * KH + kv_head) * top_budget;

    constexpr int HIST_SIZE = 1024;
    __shared__ int histogram[HIST_SIZE];
    __shared__ int threshold;
    __shared__ int out_pos;

    for (int i = threadIdx.x; i < HIST_SIZE; i += THREADS) histogram[i] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < total_candidates_active; i += THREADS) {
        uint32_t p = block_packed[in_base + i];
        if (p >> 31) {
            int bin = static_cast<int>((p >> 18) & SCORE_MASK);
            atomicAdd(&histogram[bin], 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int cumsum = 0;
        threshold = 0;
        for (int bin = HIST_SIZE - 1; bin >= 0; --bin) {
            cumsum += histogram[bin];
            if (cumsum >= top_budget) { threshold = bin; break; }
        }
        out_pos = 0;
    }
    __syncthreads();

    int thresh = threshold;

    for (int i = threadIdx.x; i < total_candidates_active; i += THREADS) {
        uint32_t p = block_packed[in_base + i];
        int valid = static_cast<int>(p >> 31);
        int s     = static_cast<int>((p >> 18) & SCORE_MASK);
        if (valid && s > thresh) {
            int pos = atomicAdd(&out_pos, 1);
            if (pos < top_budget)
                topk_out[out_base + pos] = static_cast<int64_t>(p & IDX_MASK);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < total_candidates_active; i += THREADS) {
        uint32_t p = block_packed[in_base + i];
        int valid = static_cast<int>(p >> 31);
        int s     = static_cast<int>((p >> 18) & SCORE_MASK);
        if (valid && s == thresh) {
            int pos = atomicAdd(&out_pos, 1);
            if (pos < top_budget)
                topk_out[out_base + pos] = static_cast<int64_t>(p & IDX_MASK);
        }
    }
    __syncthreads();

    int filled = out_pos;
    if (filled > top_budget) filled = top_budget;
    for (int i = filled + (int)threadIdx.x; i < top_budget; i += THREADS) {
        topk_out[out_base + i] = -1;
    }
}

// -----------------------------------------------------------------
// Launch helpers with G_SPEC specialization
// -----------------------------------------------------------------
template<int BLOCK_N, int THREADS, int G_SPEC>
void launch_impl(
    torch::Tensor q_packed, torch::Tensor key_bins, torch::Tensor kv_length,
    torch::Tensor block_packed, torch::Tensor topk_out, int top_budget
) {
    int B   = q_packed.size(0);
    int KH  = q_packed.size(1);
    int G   = q_packed.size(2);
    int MaxT = key_bins.size(2);
    int n_blocks = (MaxT + BLOCK_N - 1) / BLOCK_N;

    auto stream = at::cuda::getCurrentCUDAStream();
    int grid_p1 = B * KH * n_blocks;
    size_t smem_p1 = G * D_PACK * sizeof(int32_t);

    phase1_kernel<BLOCK_N, THREADS, G_SPEC><<<grid_p1, THREADS, smem_p1, stream>>>(
        q_packed.data_ptr<int32_t>(), key_bins.data_ptr<int32_t>(),
        block_packed.data_ptr<uint32_t>(), kv_length.data_ptr<int32_t>(),
        B, MaxT, KH, G, n_blocks);

    int grid_p2 = B * KH;
    phase2_kernel<256><<<grid_p2, 256, 0, stream>>>(
        block_packed.data_ptr<uint32_t>(), topk_out.data_ptr<int64_t>(),
        kv_length.data_ptr<int32_t>(),
        B, KH, n_blocks, BLOCK_N, top_budget, MaxT);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}

// -----------------------------------------------------------------
// Python-facing entry point  (same signature as before)
// -----------------------------------------------------------------
void hamming_topk_from_qpacked_cfg_out(
    torch::Tensor q_packed,       // [B, KH, G, D_PACK]  int32
    torch::Tensor key_bins,       // [B, KH, MaxT, D_PACK]  int32
    torch::Tensor kv_length,      // scalar int32
    torch::Tensor block_packed,   // [B, KH, n_blocks, BLOCK_N] uint32
    torch::Tensor topk_out,       // [B, KH, top_budget] int64
    int top_budget,
    int block_n,
    int threads
) {
    q_packed     = q_packed.contiguous();
    key_bins     = key_bins.contiguous();
    kv_length    = kv_length.contiguous();
    block_packed = block_packed.contiguous();
    topk_out     = topk_out.contiguous();

    int G = q_packed.size(2);

    // Dispatch with G_SPEC specialization for common group sizes
    #define DISPATCH_G(BN, TH, GS) \
        if (block_n == BN && threads == TH && G == GS) { \
            launch_impl<BN, TH, GS>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget); \
            return; \
        }

    // Generic fallback (G_SPEC=0 => runtime loop)
    #define DISPATCH_GENERIC(BN, TH) \
        if (block_n == BN && threads == TH) { \
            launch_impl<BN, TH, 0>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget); \
            return; \
        }

    // Specialized paths for G = 1, 2, 4, 8
    DISPATCH_G(64,  128, 1)
    DISPATCH_G(64,  128, 2)
    DISPATCH_G(64,  128, 4)
    DISPATCH_G(64,  128, 8)
    DISPATCH_G(64,  256, 1)
    DISPATCH_G(64,  256, 2)
    DISPATCH_G(64,  256, 4)
    DISPATCH_G(64,  256, 8)

    DISPATCH_G(128, 128, 1)
    DISPATCH_G(128, 128, 2)
    DISPATCH_G(128, 128, 4)
    DISPATCH_G(128, 128, 8)
    DISPATCH_G(128, 256, 1)
    DISPATCH_G(128, 256, 2)
    DISPATCH_G(128, 256, 4)
    DISPATCH_G(128, 256, 8)

    DISPATCH_G(256, 128, 1)
    DISPATCH_G(256, 128, 2)
    DISPATCH_G(256, 128, 4)
    DISPATCH_G(256, 128, 8)
    DISPATCH_G(256, 256, 1)
    DISPATCH_G(256, 256, 2)
    DISPATCH_G(256, 256, 4)
    DISPATCH_G(256, 256, 8)

    // Fallback for other G values
    DISPATCH_GENERIC(64,  128)
    DISPATCH_GENERIC(64,  256)
    DISPATCH_GENERIC(128, 128)
    DISPATCH_GENERIC(128, 256)
    DISPATCH_GENERIC(256, 128)
    DISPATCH_GENERIC(256, 256)

    #undef DISPATCH_G
    #undef DISPATCH_GENERIC
    TORCH_CHECK(false, "Unsupported (block_n, threads) config");
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hamming_topk_from_qpacked_cfg_out", &hamming_topk_from_qpacked_cfg_out,
          "Hamming+TopK v3 – coalesced key_bins [B,KH,MaxT,D_PACK] with G specialization",
          py::arg("q_packed"), py::arg("key_bins"), py::arg("kv_length"),
          py::arg("block_packed"), py::arg("topk_out"), py::arg("top_budget"),
          py::arg("block_n"), py::arg("threads"));
}
