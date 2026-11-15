#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
constexpr int BN = 64, KH = 4, D = 4, QH = 32, G = QH / KH;

__global__ void ak(const int32_t* __restrict__ q, const int32_t* __restrict__ k, int16_t* __restrict__ o,
                   const int32_t* __restrict__ kl, int B, int QN, int KN, int nb) {
    int i = blockIdx.x, qn = i / nb % QN, b = i / nb / QN, ns = (i % nb) * BN;
    int nl = threadIdx.x, h = threadIdx.y, n = ns + nl;
    int ka = *kl;
    ka = ka < 0 ? 0 : (ka > KN ? KN : ka);

    extern __shared__ int32_t sm[];
    int32_t *qs = sm, *ks = qs + G * KH * D;
    int t = nl + h * BN, nt = BN * KH;

    size_t qb = (size_t)b * QN * QH * D + (size_t)qn * QH * D;
    for (int x = t; x < G * KH; x += nt) {
        int hl = x / G, qh = hl * G + x % G;
        int4 v = *reinterpret_cast<const int4*>(q + qb + qh * D);
        qs[qh * D] = v.x; qs[qh * D + 1] = v.y; qs[qh * D + 2] = v.z; qs[qh * D + 3] = v.w;
    }

    size_t kb = (size_t)b * KN * KH * D;
    for (int x = t; x < BN * KH; x += nt) {
        int l = x / KH, hl = x % KH, g = ns + l, so = l * KH * D + hl * D;
        if (g < ka) {
            int4 v = *reinterpret_cast<const int4*>(k + kb + (size_t)g * KH * D + hl * D);
            ks[so] = v.x; ks[so + 1] = v.y; ks[so + 2] = v.z; ks[so + 3] = v.w;
        } else {
            ks[so] = ks[so + 1] = ks[so + 2] = ks[so + 3] = 0;
        }
    }
    __syncthreads();

    if (n >= KN) return;
    size_t oi = ((size_t)b * KH + h) * QN * KN + (size_t)qn * KN + n;
    if (n >= ka) { o[oi] = -32768; return; }

    int acc = 0;
    #pragma unroll
    for (int g = 0; g < G; g++) {
        int a = (h * G + g) * D, c = (nl * KH + h) * D;
        acc += __popc(~(qs[a] ^ ks[c])) + __popc(~(qs[a + 1] ^ ks[c + 1]))
             + __popc(~(qs[a + 2] ^ ks[c + 2])) + __popc(~(qs[a + 3] ^ ks[c + 3]));
    }
    o[oi] = (int16_t)acc;
}
}

torch::Tensor attn_k4_q32(torch::Tensor q, torch::Tensor k, torch::Tensor kl) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && kl.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kInt32 && k.dtype() == torch::kInt32 && kl.dtype() == torch::kInt32);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && kl.numel() == 1);
    q = q.contiguous(); k = k.contiguous();

    int B = q.size(0), QN = q.size(1), KN = k.size(1);
    TORCH_CHECK(q.size(2) == QH && k.size(2) == KH && q.size(3) == D && k.size(3) == D);
    TORCH_CHECK(k.device() == q.device());
    const c10::cuda::CUDAGuard g(q.device());

    auto o = torch::empty({B, KH, QN, KN}, q.options().dtype(torch::kInt16));
    int nb = (KN + BN - 1) / BN;
    int64_t tb = (int64_t)B * QN * nb;
    TORCH_CHECK(tb > 0 && tb <= INT32_MAX);

    size_t sm = (G * KH * D + BN * KH * D) * sizeof(int32_t);
    ak<<<dim3((uint32_t)tb), dim3(BN, KH), sm, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<int32_t>(), k.data_ptr<int32_t>(), o.data_ptr<int16_t>(),
        kl.data_ptr<int32_t>(), B, QN, KN, nb);
    if (auto e = cudaGetLastError(); e) throw std::runtime_error(cudaGetErrorString(e));
    return QN == 1 ? o.squeeze(2) : o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_k4_q32", &attn_k4_q32); }
