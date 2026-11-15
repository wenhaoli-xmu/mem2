#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>

namespace {
constexpr int DP = 4;
constexpr uint32_t IM = (1u << 18) - 1u, SM = (1u << 10) - 1u;

__device__ __forceinline__ uint32_t pk(int s, int i, bool v) {
    return (v ? 0x80000000u : 0u) | ((uint32_t(s) & SM) << 18) | (uint32_t(i) & IM);
}

template <int GS>
__device__ __forceinline__ int sc(const int32_t* q, int k0, int k1, int k2, int k3) {
    int a = 0;
    #pragma unroll
    for (int g = 0; g < GS; g++) {
        const int32_t* h = q + g * DP;
        a += __popc(~(h[0] ^ k0)) + __popc(~(h[1] ^ k1))
           + __popc(~(h[2] ^ k2)) + __popc(~(h[3] ^ k3));
    }
    return a;
}

template <int BN, int NT, int GS>
__global__ void p1k(const int32_t* __restrict__ qp, const int32_t* __restrict__ kb, uint32_t* __restrict__ bp,
                    const int32_t* __restrict__ kl, int B, int MT, int KH, int G, int nb) {
    int i = blockIdx.x, nbi = i % nb, kh = i / nb % KH, b = i / nb / KH;
    int ka = *kl;
    ka = ka < 0 ? 0 : (ka > MT ? MT : ka);
    int ns = nbi * BN;
    if (ns >= ka) return;

    extern __shared__ int32_t qs[];
    for (int x = threadIdx.x; x < G * DP; x += NT)
        qs[x] = qp[(size_t)b * KH * G * DP + (size_t)kh * G * DP + x];
    __syncthreads();

    int32_t ql[(GS > 0 ? GS : 1) * DP];
    const int32_t* q = qs;
    if (GS > 0) {
        #pragma unroll
        for (int x = 0; x < GS * DP; x++) ql[x] = qs[x];
        q = ql;
    }

    const int32_t* kbase = kb + ((size_t)b * KH + kh) * (size_t)MT * DP;
    bool al = !((uintptr_t)kbase & 0xF);
    const int4* k4 = reinterpret_cast<const int4*>(kbase);
    size_t ob = (((size_t)b * KH + kh) * nb + nbi) * BN;

    for (int t = threadIdx.x; t < BN; t += NT) {
        int n = ns + t;
        if (n >= ka) { bp[ob + t] = 0u; continue; }
        int k0, k1, k2, k3;
        if (al) {
            int4 v = k4[n];
            k0 = v.x; k1 = v.y; k2 = v.z; k3 = v.w;
        } else {
            const int32_t* p = kbase + (size_t)n * DP;
            k0 = p[0]; k1 = p[1]; k2 = p[2]; k3 = p[3];
        }
        int a = 0;
        if (GS == 1) a = sc<1>(q, k0, k1, k2, k3);
        else if (GS == 2) a = sc<2>(q, k0, k1, k2, k3);
        else if (GS == 4) a = sc<4>(q, k0, k1, k2, k3);
        else if (GS == 8) a = sc<8>(q, k0, k1, k2, k3);
        else
            for (int g = 0; g < G; g++) {
                const int32_t* h = qs + g * DP;
                a += __popc(~(h[0] ^ k0)) + __popc(~(h[1] ^ k1))
                   + __popc(~(h[2] ^ k2)) + __popc(~(h[3] ^ k3));
            }
        bp[ob + t] = pk(a, n, true);
    }
}

template <int NT>
__global__ void p2k(const uint32_t* __restrict__ bp, int64_t* __restrict__ tk, const int32_t* __restrict__ kl,
                    int B, int KH, int nb, int bn, int tb, int MT) {
    int i = blockIdx.x, kh = i % KH, b = i / KH;
    int ka = *kl;
    ka = ka < 0 ? 0 : (ka > MT ? MT : ka);
    int full = nb * bn, act = ((ka + bn - 1) / bn) * bn;
    if (act > full) act = full;
    size_t ib = ((size_t)b * KH + kh) * full, ob = ((size_t)b * KH + kh) * tb;
    constexpr int HS = 1024;
    __shared__ int h[HS], th, op;

    for (int x = threadIdx.x; x < HS; x += NT) h[x] = 0;
    __syncthreads();
    for (int x = threadIdx.x; x < act; x += NT) {
        uint32_t p = bp[ib + x];
        if (p >> 31) atomicAdd(&h[(p >> 18) & SM], 1);
    }
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
    for (int x = threadIdx.x; x < act; x += NT) {
        uint32_t p = bp[ib + x];
        if (!(p >> 31) || (int)((p >> 18) & SM) <= th) continue;
        int q = atomicAdd(&op, 1);
        if (q < tb) tk[ob + q] = (int64_t)(p & IM);
    }
    __syncthreads();
    for (int x = threadIdx.x; x < act; x += NT) {
        uint32_t p = bp[ib + x];
        if (!(p >> 31) || (int)((p >> 18) & SM) != th) continue;
        int q = atomicAdd(&op, 1);
        if (q < tb) tk[ob + q] = (int64_t)(p & IM);
    }
    __syncthreads();
    int f = op < tb ? op : tb;
    for (int x = f + threadIdx.x; x < tb; x += NT) tk[ob + x] = -1;
}

template <int BN, int NT, int GS>
void launch_impl(torch::Tensor qp, torch::Tensor kb, torch::Tensor kl, torch::Tensor bp, torch::Tensor tk, int tb) {
    int B = qp.size(0), KH = qp.size(1), G = qp.size(2), MT = kb.size(2), nb = (MT + BN - 1) / BN;
    auto st = at::cuda::getCurrentCUDAStream();
    p1k<BN, NT, GS><<<B * KH * nb, NT, G * DP * 4, st>>>(qp.data_ptr<int32_t>(), kb.data_ptr<int32_t>(),
        bp.data_ptr<uint32_t>(), kl.data_ptr<int32_t>(), B, MT, KH, G, nb);
    p2k<256><<<B * KH, 256, 0, st>>>(bp.data_ptr<uint32_t>(), tk.data_ptr<int64_t>(),
                                     kl.data_ptr<int32_t>(), B, KH, nb, BN, tb, MT);
    if (auto e = cudaGetLastError(); e) throw std::runtime_error(cudaGetErrorString(e));
}

void hamming_topk_from_qpacked_cfg_out(torch::Tensor qp, torch::Tensor kb, torch::Tensor kl, torch::Tensor bp,
                                       torch::Tensor tk, int tb, int bn, int nt) {
    qp = qp.contiguous(); kb = kb.contiguous(); kl = kl.contiguous();
    bp = bp.contiguous(); tk = tk.contiguous();
    int G = qp.size(2);

    #define D(BN, TH, GS) if (bn == BN && nt == TH && G == GS) { launch_impl<BN, TH, GS>(qp, kb, kl, bp, tk, tb); return; }
    #define DG(BN, TH) if (bn == BN && nt == TH) { launch_impl<BN, TH, 0>(qp, kb, kl, bp, tk, tb); return; }
    #define ALL(BN, TH) D(BN, TH, 1) D(BN, TH, 2) D(BN, TH, 4) D(BN, TH, 8) DG(BN, TH)
    ALL(64, 128) ALL(64, 256) ALL(128, 128) ALL(128, 256) ALL(256, 128) ALL(256, 256)
    #undef ALL
    #undef DG
    #undef D
    TORCH_CHECK(false);
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hamming_topk_from_qpacked_cfg_out", &hamming_topk_from_qpacked_cfg_out);
}
