#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>

namespace {

constexpr int D_PACK = 4;
constexpr int CHUNK_MAX = 128;
constexpr uint32_t IDX_MASK = (1u << 18) - 1u;
constexpr uint32_t SCORE_MASK = (1u << 10) - 1u;

__device__ __forceinline__ bool is_aligned_16(const void* ptr) {
    return ((reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ uint32_t smem_addr_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_cg_16B(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = smem_addr_u32(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

__device__ __forceinline__ uint32_t pack_candidate_u32(int score, int idx, bool valid) {
    uint32_t v = valid ? 0x80000000u : 0u;
    uint32_t s = (static_cast<uint32_t>(score) & SCORE_MASK) << 18;
    uint32_t i = static_cast<uint32_t>(idx) & IDX_MASK;
    return v | s | i;
}

template<int BLOCK_N, int THREADS, bool USE_CP_ASYNC>
__global__ void phase1_qpacked_kernel(
    const int32_t* __restrict__ q_packed,
    const int32_t* __restrict__ key_bins,
    uint32_t* __restrict__ block_packed,
    const int32_t* __restrict__ kv_length_ptr,
    int B, int MaxT, int KHv, int Gv, int n_blocks
) {
    constexpr int CHUNK = BLOCK_N / 2;

    int linear_idx = blockIdx.x;
    int n_block_idx = linear_idx % n_blocks;
    linear_idx /= n_blocks;
    int kv_head = linear_idx % KHv;
    int b = linear_idx / KHv;

    int KN_active = *kv_length_ptr;
    if (KN_active < 0) KN_active = 0;
    if (KN_active > MaxT) KN_active = MaxT;

    int n_start = n_block_idx * BLOCK_N;

    extern __shared__ char shared_raw[];
    int32_t* q_shared = reinterpret_cast<int32_t*>(shared_raw);
    int32_t* k_buf = q_shared + Gv * D_PACK;
    uint32_t* block_payload = reinterpret_cast<uint32_t*>(k_buf + 2 * CHUNK_MAX * D_PACK);

    for (int i = threadIdx.x; i < Gv * D_PACK; i += THREADS) {
        q_shared[i] = q_packed[(((size_t)b * KHv * Gv * D_PACK) + (size_t)kv_head * Gv * D_PACK) + i];
    }
    __syncthreads();

    auto load_chunk = [&](int stage, int chunk_base, int chunk_tokens) {
        size_t k_base = (size_t)b * MaxT * KHv * D_PACK + (size_t)kv_head * D_PACK;
        for (int t = threadIdx.x; t < chunk_tokens; t += THREADS) {
            int ng = n_start + chunk_base + t;
            int32_t* dst = &k_buf[(stage * CHUNK_MAX + t) * D_PACK];

            if (ng < KN_active) {
                const int32_t* src = key_bins + k_base + (size_t)ng * KHv * D_PACK;
                bool can_vec16 = (D_PACK == 4) && is_aligned_16(src) && is_aligned_16(dst);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                if constexpr (USE_CP_ASYNC) {
                    if (can_vec16) {
                        cp_async_cg_16B(dst, src);
                    } else {
                        #pragma unroll
                        for (int di = 0; di < D_PACK; di++) dst[di] = src[di];
                    }
                } else {
                    if (can_vec16) {
                        int4 v = *reinterpret_cast<const int4*>(src);
                        *reinterpret_cast<int4*>(dst) = v;
                    } else {
                        #pragma unroll
                        for (int di = 0; di < D_PACK; di++) dst[di] = src[di];
                    }
                }
#else
                if (can_vec16) {
                    int4 v = *reinterpret_cast<const int4*>(src);
                    *reinterpret_cast<int4*>(dst) = v;
                } else {
                    #pragma unroll
                    for (int di = 0; di < D_PACK; di++) dst[di] = src[di];
                }
#endif
            } else {
                #pragma unroll
                for (int di = 0; di < D_PACK; di++) dst[di] = 0;
            }
        }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        if constexpr (USE_CP_ASYNC) {
            cp_async_commit();
        }
#endif
    };

    int stage = 0;
    int first_tokens = (BLOCK_N < CHUNK) ? BLOCK_N : CHUNK;
    load_chunk(stage, 0, first_tokens);

    for (int chunk_base = 0; chunk_base < BLOCK_N; chunk_base += CHUNK) {
        int chunk_tokens = BLOCK_N - chunk_base;
        if (chunk_tokens > CHUNK) chunk_tokens = CHUNK;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        if constexpr (USE_CP_ASYNC) {
            cp_async_wait_all();
        }
#endif
        __syncthreads();

        int next_base = chunk_base + CHUNK;
        if (next_base < BLOCK_N) {
            int next_tokens = BLOCK_N - next_base;
            if (next_tokens > CHUNK) next_tokens = CHUNK;
            load_chunk(stage ^ 1, next_base, next_tokens);
        }

        for (int t = threadIdx.x; t < chunk_tokens; t += THREADS) {
            int nl = chunk_base + t;
            int n = n_start + nl;

            bool valid = (n < KN_active);
            int score = 0;

            if (valid) {
                int accum = 0;
                int32_t* kh = &k_buf[(stage * CHUNK_MAX + t) * D_PACK];

                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    if (g < Gv) {
                        int32_t* qh = &q_shared[g * D_PACK];
                        accum += __popc(~(qh[0] ^ kh[0]));
                        accum += __popc(~(qh[1] ^ kh[1]));
                        accum += __popc(~(qh[2] ^ kh[2]));
                        accum += __popc(~(qh[3] ^ kh[3]));
                    }
                }
                score = accum;
            }

            block_payload[nl] = pack_candidate_u32(score, n, valid);
        }

        __syncthreads();
        stage ^= 1;
    }

    size_t out_base = (((size_t)b * KHv + kv_head) * n_blocks + n_block_idx) * BLOCK_N;
    for (int i = threadIdx.x; i < BLOCK_N; i += THREADS) {
        block_packed[out_base + i] = block_payload[i];
    }
}

template<int THREADS>
__global__ void phase2_kernel(
    const uint32_t* __restrict__ block_packed,
    int64_t* __restrict__ topk_out,
    int B, int KHv, int n_blocks, int local_k, int top_budget
) {
    int linear_idx = blockIdx.x;
    int kv_head = linear_idx % KHv;
    int b = linear_idx / KHv;

    int total_candidates = n_blocks * local_k;
    size_t in_base = ((size_t)b * KHv + kv_head) * (size_t)total_candidates;
    size_t out_base = ((size_t)b * KHv + kv_head) * top_budget;

    constexpr int HIST_SIZE = 1024;
    __shared__ int histogram[HIST_SIZE];
    __shared__ int threshold;
    __shared__ int out_pos;

    for (int i = threadIdx.x; i < HIST_SIZE; i += THREADS) histogram[i] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < total_candidates; i += THREADS) {
        uint32_t p = block_packed[in_base + i];
        int bin = static_cast<int>((p >> 18) & SCORE_MASK);
        atomicAdd(&histogram[bin], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int cumsum = 0;
        threshold = 0;
        for (int bin = HIST_SIZE - 1; bin >= 0; --bin) {
            cumsum += histogram[bin];
            if (cumsum >= top_budget) {
                threshold = bin;
                break;
            }
        }
        out_pos = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < total_candidates; i += THREADS) {
        uint32_t p = block_packed[in_base + i];
        int valid = static_cast<int>(p >> 31);
        int s = static_cast<int>((p >> 18) & SCORE_MASK);
        if (valid && s > threshold) {
            int pos = atomicAdd(&out_pos, 1);
            if (pos < top_budget) {
                int idx = static_cast<int>(p & IDX_MASK);
                topk_out[out_base + pos] = static_cast<int64_t>(idx);
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < total_candidates; i += THREADS) {
        uint32_t p = block_packed[in_base + i];
        int valid = static_cast<int>(p >> 31);
        int s = static_cast<int>((p >> 18) & SCORE_MASK);
        if (valid && s == threshold) {
            int pos = atomicAdd(&out_pos, 1);
            if (pos < top_budget) {
                int idx = static_cast<int>(p & IDX_MASK);
                topk_out[out_base + pos] = static_cast<int64_t>(idx);
            }
        }
    }
}

template<int BLOCK_N, int THREADS, bool USE_CP_ASYNC>
void launch_qpacked_topk_out_impl(
    torch::Tensor q_packed,
    torch::Tensor key_bins,
    torch::Tensor kv_length,
    torch::Tensor block_packed,
    torch::Tensor topk_out,
    int top_budget
) {
    int B = q_packed.size(0);
    int KHv = q_packed.size(1);
    int Gv = q_packed.size(2);
    int MaxT = key_bins.size(1);
    int n_blocks = (MaxT + BLOCK_N - 1) / BLOCK_N;

    auto stream = at::cuda::getCurrentCUDAStream();

    topk_out.fill_(-1);

    int grid_p1 = B * KHv * n_blocks;
    size_t smem_p1 = (Gv * D_PACK) * sizeof(int32_t)
                   + (2 * CHUNK_MAX * D_PACK) * sizeof(int32_t)
                   + BLOCK_N * sizeof(uint32_t);

    phase1_qpacked_kernel<BLOCK_N, THREADS, USE_CP_ASYNC><<<grid_p1, THREADS, smem_p1, stream>>>(
        q_packed.data_ptr<int32_t>(),
        key_bins.data_ptr<int32_t>(),
        block_packed.data_ptr<uint32_t>(),
        kv_length.data_ptr<int32_t>(),
        B, MaxT, KHv, Gv, n_blocks
    );

    int grid_p2 = B * KHv;
    phase2_kernel<256><<<grid_p2, 256, 0, stream>>>(
        block_packed.data_ptr<uint32_t>(),
        topk_out.data_ptr<int64_t>(),
        B, KHv, n_blocks, BLOCK_N, top_budget
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}

template<int BLOCK_N, int THREADS, bool USE_CP_ASYNC>
torch::Tensor launch_qpacked_topk(
    torch::Tensor q_packed,
    torch::Tensor key_bins,
    torch::Tensor kv_length,
    int top_budget
) {
    int B = q_packed.size(0);
    int KHv = q_packed.size(1);
    int MaxT = key_bins.size(1);
    int n_blocks = (MaxT + BLOCK_N - 1) / BLOCK_N;
    auto device = q_packed.device();

    auto block_packed = torch::empty({B, KHv, n_blocks, BLOCK_N}, torch::dtype(torch::kUInt32).device(device));
    auto topk_out = torch::empty({B, KHv, top_budget}, torch::dtype(torch::kInt64).device(device));

    launch_qpacked_topk_out_impl<BLOCK_N, THREADS, USE_CP_ASYNC>(
        q_packed, key_bins, kv_length, block_packed, topk_out, top_budget
    );
    return topk_out;
}

void hamming_topk_from_qpacked_cfg_out(
    torch::Tensor q_packed,
    torch::Tensor key_bins,
    torch::Tensor kv_length,
    torch::Tensor block_packed,
    torch::Tensor topk_out,
    int top_budget,
    int block_n,
    int threads,
    bool use_cp_async
) {
    TORCH_CHECK(q_packed.is_cuda(), "q_packed must be CUDA");
    TORCH_CHECK(key_bins.is_cuda(), "key_bins must be CUDA");
    TORCH_CHECK(kv_length.is_cuda(), "kv_length must be CUDA");
    TORCH_CHECK(block_packed.is_cuda(), "block_packed must be CUDA");
    TORCH_CHECK(topk_out.is_cuda(), "topk_out must be CUDA");

    TORCH_CHECK(q_packed.scalar_type() == torch::kInt32, "q_packed must be int32");
    TORCH_CHECK(key_bins.scalar_type() == torch::kInt32, "key_bins must be int32");
    TORCH_CHECK(kv_length.scalar_type() == torch::kInt32, "kv_length must be int32");
    TORCH_CHECK(block_packed.scalar_type() == torch::kUInt32, "block_packed must be uint32");
    TORCH_CHECK(topk_out.scalar_type() == torch::kInt64, "topk_out must be int64");

    TORCH_CHECK(q_packed.dim() == 4, "q_packed must be [B,KH,G,D_PACK]");
    TORCH_CHECK(key_bins.dim() == 4, "key_bins must be [B,MaxT,KH,D_PACK]");
    TORCH_CHECK(block_packed.dim() == 4, "block_packed must be [B,KH,n_blocks,BLOCK_N]");
    TORCH_CHECK(topk_out.dim() == 3, "topk_out must be [B,KH,top_budget]");

    q_packed = q_packed.contiguous();
    key_bins = key_bins.contiguous();
    kv_length = kv_length.contiguous();
    block_packed = block_packed.contiguous();
    topk_out = topk_out.contiguous();

    int B = q_packed.size(0);
    int KHv = q_packed.size(1);
    int Gv = q_packed.size(2);
    int DP = q_packed.size(3);
    int MaxT = key_bins.size(1);

    TORCH_CHECK(DP == D_PACK, "q_packed D_PACK mismatch");
    TORCH_CHECK(key_bins.size(0) == B, "B mismatch");
    TORCH_CHECK(key_bins.size(2) == KHv, "KH mismatch");
    TORCH_CHECK(key_bins.size(3) == D_PACK, "key_bins D_PACK mismatch");
    TORCH_CHECK(Gv >= 1 && Gv <= 4, "G out of supported range");

    TORCH_CHECK(topk_out.size(0) == B && topk_out.size(1) == KHv && topk_out.size(2) == top_budget,
                "topk_out shape mismatch");

    int n_blocks = (MaxT + block_n - 1) / block_n;
    TORCH_CHECK(block_packed.size(0) == B && block_packed.size(1) == KHv &&
                block_packed.size(2) == n_blocks && block_packed.size(3) == block_n,
                "block_packed shape mismatch");

    if (block_n == 64 && threads == 128) {
        if (use_cp_async) launch_qpacked_topk_out_impl<64, 128, true>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        else launch_qpacked_topk_out_impl<64, 128, false>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        return;
    } else if (block_n == 64 && threads == 256) {
        if (use_cp_async) launch_qpacked_topk_out_impl<64, 256, true>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        else launch_qpacked_topk_out_impl<64, 256, false>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        return;
    } else if (block_n == 128 && threads == 128) {
        if (use_cp_async) launch_qpacked_topk_out_impl<128, 128, true>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        else launch_qpacked_topk_out_impl<128, 128, false>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        return;
    } else if (block_n == 128 && threads == 256) {
        if (use_cp_async) launch_qpacked_topk_out_impl<128, 256, true>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        else launch_qpacked_topk_out_impl<128, 256, false>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        return;
    } else if (block_n == 256 && threads == 128) {
        if (use_cp_async) launch_qpacked_topk_out_impl<256, 128, true>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        else launch_qpacked_topk_out_impl<256, 128, false>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        return;
    } else if (block_n == 256 && threads == 256) {
        if (use_cp_async) launch_qpacked_topk_out_impl<256, 256, true>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        else launch_qpacked_topk_out_impl<256, 256, false>(q_packed, key_bins, kv_length, block_packed, topk_out, top_budget);
        return;
    }

    TORCH_CHECK(false, "Unsupported (block_n, threads) config");
}

torch::Tensor hamming_topk_from_qpacked_cfg(
    torch::Tensor q_packed,
    torch::Tensor key_bins,
    torch::Tensor kv_length,
    int top_budget,
    int block_n,
    int threads,
    bool use_cp_async
) {
    TORCH_CHECK(q_packed.is_cuda(), "q_packed must be CUDA");
    TORCH_CHECK(key_bins.is_cuda(), "key_bins must be CUDA");
    TORCH_CHECK(kv_length.is_cuda(), "kv_length must be CUDA");

    TORCH_CHECK(q_packed.scalar_type() == torch::kInt32, "q_packed must be int32");
    TORCH_CHECK(key_bins.scalar_type() == torch::kInt32, "key_bins must be int32");
    TORCH_CHECK(kv_length.scalar_type() == torch::kInt32, "kv_length must be int32");

    TORCH_CHECK(q_packed.dim() == 4, "q_packed must be [B,KH,G,D_PACK]");
    TORCH_CHECK(key_bins.dim() == 4, "key_bins must be [B,MaxT,KH,D_PACK]");

    q_packed = q_packed.contiguous();
    key_bins = key_bins.contiguous();
    kv_length = kv_length.contiguous();

    int B = q_packed.size(0);
    int KHv = q_packed.size(1);
    int Gv = q_packed.size(2);
    int DP = q_packed.size(3);
    TORCH_CHECK(DP == D_PACK, "q_packed D_PACK mismatch");
    TORCH_CHECK(key_bins.size(0) == B, "B mismatch");
    TORCH_CHECK(key_bins.size(2) == KHv, "KH mismatch");
    TORCH_CHECK(key_bins.size(3) == D_PACK, "key_bins D_PACK mismatch");
    TORCH_CHECK(Gv >= 1 && Gv <= 4, "G out of supported range");

    if (block_n == 64 && threads == 128) {
        if (use_cp_async) return launch_qpacked_topk<64, 128, true>(q_packed, key_bins, kv_length, top_budget);
        else return launch_qpacked_topk<64, 128, false>(q_packed, key_bins, kv_length, top_budget);
    } else if (block_n == 64 && threads == 256) {
        if (use_cp_async) return launch_qpacked_topk<64, 256, true>(q_packed, key_bins, kv_length, top_budget);
        else return launch_qpacked_topk<64, 256, false>(q_packed, key_bins, kv_length, top_budget);
    } else if (block_n == 128 && threads == 128) {
        if (use_cp_async) return launch_qpacked_topk<128, 128, true>(q_packed, key_bins, kv_length, top_budget);
        else return launch_qpacked_topk<128, 128, false>(q_packed, key_bins, kv_length, top_budget);
    } else if (block_n == 128 && threads == 256) {
        if (use_cp_async) return launch_qpacked_topk<128, 256, true>(q_packed, key_bins, kv_length, top_budget);
        else return launch_qpacked_topk<128, 256, false>(q_packed, key_bins, kv_length, top_budget);
    } else if (block_n == 256 && threads == 128) {
        if (use_cp_async) return launch_qpacked_topk<256, 128, true>(q_packed, key_bins, kv_length, top_budget);
        else return launch_qpacked_topk<256, 128, false>(q_packed, key_bins, kv_length, top_budget);
    } else if (block_n == 256 && threads == 256) {
        if (use_cp_async) return launch_qpacked_topk<256, 256, true>(q_packed, key_bins, kv_length, top_budget);
        else return launch_qpacked_topk<256, 256, false>(q_packed, key_bins, kv_length, top_budget);
    }

    TORCH_CHECK(false, "Unsupported (block_n, threads) config");
}

torch::Tensor hamming_topk_from_qpacked(
    torch::Tensor q_packed,
    torch::Tensor key_bins,
    torch::Tensor kv_length,
    int top_budget
) {
    return hamming_topk_from_qpacked_cfg(
        q_packed, key_bins, kv_length, top_budget, 128, 256, true
    );
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hamming_topk_from_qpacked", &hamming_topk_from_qpacked,
          "Hamming+TopK from pre-packed query bits");
    m.def("hamming_topk_from_qpacked_cfg", &hamming_topk_from_qpacked_cfg,
          "Configurable Hamming+TopK from pre-packed query bits",
          py::arg("q_packed"), py::arg("key_bins"), py::arg("kv_length"),
          py::arg("top_budget"), py::arg("block_n"), py::arg("threads"), py::arg("use_cp_async"));
    m.def("hamming_topk_from_qpacked_cfg_out", &hamming_topk_from_qpacked_cfg_out,
          "Configurable Hamming+TopK from pre-packed query bits (preallocated outputs)",
          py::arg("q_packed"), py::arg("key_bins"), py::arg("kv_length"),
          py::arg("block_packed"), py::arg("topk_out"), py::arg("top_budget"),
          py::arg("block_n"), py::arg("threads"), py::arg("use_cp_async"));
}
