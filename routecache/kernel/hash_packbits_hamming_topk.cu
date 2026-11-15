#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

#define F __bfloat162float

namespace {
constexpr int BN = 64, DP = 4, D = 128, LK = 64;

__global__ void p1(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ p0,
                   const __nv_bfloat16* __restrict__ p1, const int32_t* __restrict__ kb,
                   int16_t* __restrict__ bv, int32_t* __restrict__ bi, const int32_t* __restrict__ kl,
                   int B, int MT, int nb, int KH, int QH, int G) {
    int i = blockIdx.x, nbi = i % nb, kh = i / nb % KH, b = i / nb / KH;
    int ns = nbi * BN, nl = threadIdx.x, n = ns + nl;
    int ka = *kl;
    ka = ka < 0 ? 0 : (ka > MT ? MT : ka);

    extern __shared__ char sm[];
    float* qh = reinterpret_cast<float*>(sm);
    float* yt = qh + G * D;
    int32_t* qp = reinterpret_cast<int32_t*>(yt + D);
    int32_t* kp = qp + G * DP;

    for (int g = 0; g < G; g++) {
        int h = kh * G + g;
        size_t qo = (size_t)b * QH * D + (size_t)h * D, po = (size_t)h * D * D;
        __syncthreads();
        for (int j = threadIdx.x; j < D; j += blockDim.x) yt[j] = F(q[qo + j]);
        __syncthreads();
        float* zo = qh + g * D;
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            float a = 0;
            for (int k = 0; k < D; k++) a += yt[k] * F(p0[po + k * D + j]);
            zo[j] = a / (1 + expf(-a)) + yt[j];
        }
        __syncthreads();
        for (int j = threadIdx.x; j < D; j += blockDim.x) yt[j] = zo[j];
        __syncthreads();
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            float a = 0;
            for (int k = 0; k < D; k++) a += yt[k] * F(p1[po + k * D + j]);
            zo[j] = a + yt[j];
        }
        __syncthreads();
    }

    int l = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int g = 0; g < G; g++) {
        float* zo = qh + g * D;
        for (int p = w; p < DP; p += nw) {
            unsigned pk = __ballot_sync(-1u, zo[p * 32 + l] > 0.f);
            if (!l) qp[g * DP + p] = pk;
        }
    }
    __syncthreads();

    size_t kbase = (size_t)b * MT * KH * DP + (size_t)kh * DP;
    for (int x = threadIdx.x; x < BN * DP; x += blockDim.x) {
        int nl2 = x / DP, d = x % DP, ng = ns + nl2;
        kp[nl2 * DP + d] = ng < ka ? kb[kbase + (size_t)ng * KH * DP + d] : 0;
    }
    __syncthreads();

    int sc = -32768;
    if (n < ka) {
        sc = 0;
        for (int g = 0; g < G; g++) {
            int32_t* a = qp + g * DP;
            int32_t* c = kp + nl * DP;
            #pragma unroll
            for (int d = 0; d < DP; d++) sc += __popc(~(a[d] ^ c[d]));
        }
    }
    size_t ob = (((size_t)b * KH + kh) * nb + nbi) * LK;
    bv[ob + nl] = (int16_t)sc;
    bi[ob + nl] = n;
}

__global__ void p2(const int16_t* __restrict__ bv, const int32_t* __restrict__ bi, int64_t* __restrict__ tk,
                   int B, int KH, int nb, int tb) {
    int i = blockIdx.x, kh = i % KH, b = i / KH, tc = nb * LK;
    size_t ib = ((size_t)b * KH + kh) * tc, ob = ((size_t)b * KH + kh) * tb;
    constexpr int HS = 1024;
    __shared__ int h[HS];
    __shared__ int16_t th;
    __shared__ int op;

    for (int x = threadIdx.x; x < HS; x += blockDim.x) h[x] = 0;
    __syncthreads();
    for (int x = threadIdx.x; x < tc; x += blockDim.x) {
        int s = bv[ib + x];
        s = s < 0 ? 0 : (s >= HS ? HS - 1 : s);
        atomicAdd(&h[s], 1);
    }
    __syncthreads();
    if (!threadIdx.x) {
        int cs = 0;
        th = 0;
        for (int k = HS - 1; k >= 0; k--) {
            cs += h[k];
            if (cs >= tb) { th = (int16_t)k; break; }
        }
        op = 0;
    }
    __syncthreads();
    for (int x = threadIdx.x; x < tc; x += blockDim.x) {
        if (bv[ib + x] <= th) continue;
        int p = atomicAdd(&op, 1);
        if (p < tb) tk[ob + p] = (int64_t)bi[ib + x];
    }
    __syncthreads();
    for (int x = threadIdx.x; x < tc; x += blockDim.x) {
        if (bv[ib + x] != th) continue;
        int p = atomicAdd(&op, 1);
        if (p < tb) tk[ob + p] = (int64_t)bi[ib + x];
    }
}
}

torch::Tensor hash_packbits_hamming_topk(torch::Tensor q, torch::Tensor p0, torch::Tensor p1,
                                         torch::Tensor kb, torch::Tensor kl, int tb) {
    TORCH_CHECK(q.is_cuda() && p0.is_cuda() && p1.is_cuda() && kb.is_cuda() && kl.is_cuda());
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && p0.scalar_type() == torch::kBFloat16 &&
                p1.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(kb.scalar_type() == torch::kInt32 && kl.scalar_type() == torch::kInt32);
    TORCH_CHECK(q.dim() == 4 && q.size(1) == 1 && kb.dim() == 4);

    q = q.contiguous(); p0 = p0.contiguous(); p1 = p1.contiguous(); kb = kb.contiguous();

    int B = q.size(0), QH = q.size(2), MT = kb.size(1), KH = kb.size(2);
    TORCH_CHECK(kb.size(3) == DP && QH % KH == 0);
    int G = QH / KH, nb = (MT + BN - 1) / BN;

    auto bv = torch::empty({B, KH, nb, LK}, q.options().dtype(torch::kInt16));
    auto bi = torch::empty({B, KH, nb, LK}, q.options().dtype(torch::kInt32));
    auto tk = torch::zeros({B, KH, tb}, q.options().dtype(torch::kInt64));
    size_t sm = (G * D + D) * sizeof(float) + (G * DP + BN * DP) * sizeof(int32_t);
    auto st = at::cuda::getCurrentCUDAStream();

    p1<<<B * KH * nb, BN, sm, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(p0.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(p1.data_ptr<at::BFloat16>()),
        kb.data_ptr<int32_t>(), bv.data_ptr<int16_t>(), bi.data_ptr<int32_t>(),
        kl.data_ptr<int32_t>(), B, MT, nb, KH, QH, G);
    p2<<<B * KH, 256, 0, st>>>(bv.data_ptr<int16_t>(), bi.data_ptr<int32_t>(),
                               tk.data_ptr<int64_t>(), B, KH, nb, tb);
    if (auto e = cudaGetLastError(); e) throw std::runtime_error(cudaGetErrorString(e));
    return tk;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("hash_packbits_hamming_topk", &hash_packbits_hamming_topk); }
