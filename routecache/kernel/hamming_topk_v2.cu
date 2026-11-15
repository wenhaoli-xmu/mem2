#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>

namespace {
constexpr int DP = 4, CM = 128;
constexpr uint32_t IM = (1u << 18) - 1u, SM = (1u << 10) - 1u;

__device__ __forceinline__ bool a16(const void* p) { return !(reinterpret_cast<uintptr_t>(p) & 0xF); }
__device__ __forceinline__ uint32_t pk(int s, int i, bool v) {
    return (v ? 0x80000000u : 0u) | ((uint32_t(s) & SM) << 18) | (uint32_t(i) & IM);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa(void* d, const void* s) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa(d)), "l"(s));
}
__device__ __forceinline__ void cpc() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cpw() { asm volatile("cp.async.wait_group 0;\n" ::); }
#endif

template <int BN, int NT, bool CA>
__global__ void p1k(const int32_t* __restrict__ qp, const int32_t* __restrict__ kb, uint32_t* __restrict__ bp_out,
                    const int32_t* __restrict__ kl, int B, int MT, int KH, int G, int nb) {
    constexpr int CH = BN / 2;
    int i = blockIdx.x, nbi = i % nb, kh = i / nb % KH, b = i / nb / KH;
    int ka = *kl;
    ka = ka < 0 ? 0 : (ka > MT ? MT : ka);
    int ns = nbi * BN;

    extern __shared__ char sm[];
    int32_t* qs = reinterpret_cast<int32_t*>(sm);
    int32_t* kbuf = qs + G * DP;
    uint32_t* bp = reinterpret_cast<uint32_t*>(kbuf + 2 * CM * DP);

    for (int x = threadIdx.x; x < G * DP; x += NT)
        qs[x] = qp[(size_t)b * KH * G * DP + (size_t)kh * G * DP + x];
    __syncthreads();

    auto load = [&](int st, int cb, int ct) {
        size_t base = (size_t)b * MT * KH * DP + (size_t)kh * DP;
        for (int t = threadIdx.x; t < ct; t += NT) {
            int ng = ns + cb + t;
            int32_t* d = &kbuf[(st * CM + t) * DP];
            if (ng >= ka) {
                #pragma unroll
                for (int j = 0; j < DP; j++) d[j] = 0;
                continue;
            }
            const int32_t* s = kb + base + (size_t)ng * KH * DP;
            if (!(DP == 4 && a16(s) && a16(d))) {
                #pragma unroll
                for (int j = 0; j < DP; j++) d[j] = s[j];
                continue;
            }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            if constexpr (CA) cpa(d, s);
            else *reinterpret_cast<int4*>(d) = *reinterpret_cast<const int4*>(s);
#else
            *reinterpret_cast<int4*>(d) = *reinterpret_cast<const int4*>(s);
#endif
        }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        if constexpr (CA) cpc();
#endif
    };

    int st = 0, ft = BN < CH ? BN : CH;
    load(st, 0, ft);

    for (int cb = 0; cb < BN; cb += CH) {
        int ct = BN - cb;
        if (ct > CH) ct = CH;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        if constexpr (CA) cpw();
#endif
        __syncthreads();

        int nxb = cb + CH;
        if (nxb < BN) {
            int nt2 = BN - nxb;
            if (nt2 > CH) nt2 = CH;
            load(st ^ 1, nxb, nt2);
        }

        for (int t = threadIdx.x; t < ct; t += NT) {
            int nl = cb + t, n = ns + nl;
            bool v = n < ka;
            int a = 0;
            if (v) {
                int32_t* c = &kbuf[(st * CM + t) * DP];
                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    if (g >= G) continue;
                    int32_t* h = &qs[g * DP];
                    a += __popc(~(h[0] ^ c[0])) + __popc(~(h[1] ^ c[1]))
                       + __popc(~(h[2] ^ c[2])) + __popc(~(h[3] ^ c[3]));
                }
            }
            bp[nl] = pk(a, n, v);
        }
        __syncthreads();
        st ^= 1;
    }

    size_t ob = ((size_t)b * KH + kh) * nb * BN;
    for (int x = threadIdx.x; x < BN; x += NT) bp_out[ob + x] = bp[x];
}

template <int NT>
__global__ void p2k(const uint32_t* __restrict__ bp, int64_t* __restrict__ tk,
                    int B, int KH, int nb, int lk, int tb) {
    int i = blockIdx.x, kh = i % KH, b = i / KH, tc = nb * lk;
    size_t ib = ((size_t)b * KH + kh) * tc, ob = ((size_t)b * KH + kh) * tb;
    constexpr int HS = 1024;
    __shared__ int h[HS];
    __shared__ int th, op;

    for (int x = threadIdx.x; x < HS; x += NT) h[x] = 0;
    __syncthreads();
    for (int x = threadIdx.x; x < tc; x += NT) atomicAdd(&h[(bp[ib + x] >> 18) & SM], 1);
    __syncthreads();
    if (!threadIdx.x) {
        int cs = 0;
        th = 0;
        for (int k = HS - 1; k >= 0; k--) {
            cs += h[k];
            if (cs >= tb) { th = k; break; }
        }
        op = 0;
    }
    __syncthreads();
    for (int x = threadIdx.x; x < tc; x += NT) {
        uint32_t p = bp[ib + x];
        if (!(p >> 31) || (int)((p >> 18) & SM) <= th) continue;
        int q = atomicAdd(&op, 1);
        if (q < tb) tk[ob + q] = (int64_t)(p & IM);
    }
    __syncthreads();
    for (int x = threadIdx.x; x < tc; x += NT) {
        uint32_t p = bp[ib + x];
        if (!(p >> 31) || (int)((p >> 18) & SM) != th) continue;
        int q = atomicAdd(&op, 1);
        if (q < tb) tk[ob + q] = (int64_t)(p & IM);
    }
}

template <int BN, int NT, bool CA>
void impl(torch::Tensor qp, torch::Tensor kb, torch::Tensor kl, torch::Tensor bp, torch::Tensor tk, int tb) {
    int B = qp.size(0), KH = qp.size(1), G = qp.size(2), MT = kb.size(1), nb = (MT + BN - 1) / BN;
    auto st = at::cuda::getCurrentCUDAStream();
    tk.fill_(-1);
    size_t sm = (G * DP) * 4 + (2 * CM * DP) * 4 + BN * 4;
    p1k<BN, NT, CA><<<B * KH * nb, NT, sm, st>>>(qp.data_ptr<int32_t>(), kb.data_ptr<int32_t>(),
        bp.data_ptr<uint32_t>(), kl.data_ptr<int32_t>(), B, MT, KH, G, nb);
    p2k<256><<<B * KH, 256, 0, st>>>(bp.data_ptr<uint32_t>(), tk.data_ptr<int64_t>(), B, KH, nb, BN, tb);
    if (auto e = cudaGetLastError(); e) throw std::runtime_error(cudaGetErrorString(e));
}

template <int BN, int NT, bool CA>
torch::Tensor alloc_launch(torch::Tensor qp, torch::Tensor kb, torch::Tensor kl, int tb) {
    int B = qp.size(0), KH = qp.size(1), MT = kb.size(1), nb = (MT + BN - 1) / BN;
    auto bp = torch::empty({B, KH, nb, BN}, qp.options().dtype(torch::kUInt32));
    auto tk = torch::empty({B, KH, tb}, qp.options().dtype(torch::kInt64));
    impl<BN, NT, CA>(qp, kb, kl, bp, tk, tb);
    return tk;
}

void hamming_topk_from_qpacked_cfg_out(torch::Tensor qp, torch::Tensor kb, torch::Tensor kl, torch::Tensor bp,
                                       torch::Tensor tk, int tb, int bn, int nt, bool ca) {
    TORCH_CHECK(qp.is_cuda() && kb.is_cuda() && kl.is_cuda() && bp.is_cuda() && tk.is_cuda());
    TORCH_CHECK(qp.scalar_type() == torch::kInt32 && kb.scalar_type() == torch::kInt32 &&
                kl.scalar_type() == torch::kInt32);
    TORCH_CHECK(bp.scalar_type() == torch::kUInt32 && tk.scalar_type() == torch::kInt64);
    TORCH_CHECK(qp.dim() == 4 && kb.dim() == 4 && bp.dim() == 4 && tk.dim() == 3);
    qp = qp.contiguous(); kb = kb.contiguous(); kl = kl.contiguous();
    bp = bp.contiguous(); tk = tk.contiguous();

    int B = qp.size(0), KH = qp.size(1), G = qp.size(2), MT = kb.size(1);
    TORCH_CHECK(qp.size(3) == DP && kb.size(0) == B && kb.size(2) == KH && kb.size(3) == DP);
    TORCH_CHECK(G >= 1 && G <= 4);
    TORCH_CHECK(tk.size(0) == B && tk.size(1) == KH && tk.size(2) == tb);
    int nb = (MT + bn - 1) / bn;
    TORCH_CHECK(bp.size(0) == B && bp.size(1) == KH && bp.size(2) == nb && bp.size(3) == bn);

    #define GO(BN, NT) do { if (ca) impl<BN, NT, true>(qp, kb, kl, bp, tk, tb); \
                             else impl<BN, NT, false>(qp, kb, kl, bp, tk, tb); return; } while (0)
    if (bn == 64 && nt == 128) GO(64, 128);
    if (bn == 64 && nt == 256) GO(64, 256);
    if (bn == 128 && nt == 128) GO(128, 128);
    if (bn == 128 && nt == 256) GO(128, 256);
    if (bn == 256 && nt == 128) GO(256, 128);
    if (bn == 256 && nt == 256) GO(256, 256);
    #undef GO
    TORCH_CHECK(false);
}

torch::Tensor hamming_topk_from_qpacked_cfg(torch::Tensor qp, torch::Tensor kb, torch::Tensor kl,
                                            int tb, int bn, int nt, bool ca) {
    TORCH_CHECK(qp.is_cuda() && kb.is_cuda() && kl.is_cuda());
    TORCH_CHECK(qp.scalar_type() == torch::kInt32 && kb.scalar_type() == torch::kInt32 &&
                kl.scalar_type() == torch::kInt32);
    TORCH_CHECK(qp.dim() == 4 && kb.dim() == 4);
    qp = qp.contiguous(); kb = kb.contiguous(); kl = kl.contiguous();

    int B = qp.size(0), KH = qp.size(1), G = qp.size(2);
    TORCH_CHECK(qp.size(3) == DP && kb.size(0) == B && kb.size(2) == KH && kb.size(3) == DP);
    TORCH_CHECK(G >= 1 && G <= 4);

    #define GO(BN, NT) return ca ? alloc_launch<BN, NT, true>(qp, kb, kl, tb) : alloc_launch<BN, NT, false>(qp, kb, kl, tb)
    if (bn == 64 && nt == 128) GO(64, 128);
    if (bn == 64 && nt == 256) GO(64, 256);
    if (bn == 128 && nt == 128) GO(128, 128);
    if (bn == 128 && nt == 256) GO(128, 256);
    if (bn == 256 && nt == 128) GO(256, 128);
    if (bn == 256 && nt == 256) GO(256, 256);
    #undef GO
    TORCH_CHECK(false);
}

torch::Tensor hamming_topk_from_qpacked(torch::Tensor qp, torch::Tensor kb, torch::Tensor kl, int tb) {
    return hamming_topk_from_qpacked_cfg(qp, kb, kl, tb, 128, 256, true);
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hamming_topk_from_qpacked", &hamming_topk_from_qpacked);
    m.def("hamming_topk_from_qpacked_cfg", &hamming_topk_from_qpacked_cfg);
    m.def("hamming_topk_from_qpacked_cfg_out", &hamming_topk_from_qpacked_cfg_out);
}
