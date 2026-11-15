#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <stdexcept>

#define F __bfloat162float

namespace {

__global__ void psab(const __nv_bfloat16* __restrict__ z2, const __nv_bfloat16* __restrict__ z1,
                     int32_t* __restrict__ o, int B, int QH, int KH, int G, int D, int DP) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, tot = B * KH * G * DP;
    if (i >= tot) return;
    int w = i % DP, t0 = i / DP, g = t0 % G, t1 = t0 / G, kh = t1 % KH, b = t1 / KH;
    int base = ((b * QH + kh * G + g) * D) + w * 32;
    unsigned bits = 0;
    #pragma unroll
    for (int k = 0; k < 32; k++) bits |= (unsigned)(F(z2[base + k]) + F(z1[base + k]) > 0.f) << k;
    o[i] = bits;
}

__global__ void gpb(const __nv_bfloat16* __restrict__ z1, const __nv_bfloat16* __restrict__ p1,
                    int32_t* __restrict__ o, int B, int QH, int KH, int G, int D, int DP) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, tot = B * KH * G * DP;
    if (i >= tot) return;
    int w = i % DP, t0 = i / DP, g = t0 % G, t1 = t0 / G, kh = t1 % KH, b = t1 / KH;
    int qh = kh * G + g, oc = w * 32, zb = (b * QH + qh) * D, pb = qh * D * D;
    unsigned bits = 0;
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        int c = oc + k;
        float a = 0;
        for (int j = 0; j < D; j++) a += F(z1[zb + j]) * F(p1[pb + j * D + c]);
        bits |= (unsigned)(a + F(z1[zb + c]) > 0.f) << k;
    }
    o[i] = bits;
}

inline torch::Tensor q3d(torch::Tensor q) {
    if (q.dim() == 4) { TORCH_CHECK(q.size(1) == 1); return q.select(1, 0); }
    TORCH_CHECK(q.dim() == 3);
    return q;
}

torch::Tensor psab_cu(torch::Tensor z2, torch::Tensor z1, int KH, int G) {
    TORCH_CHECK(z2.is_cuda() && z1.is_cuda());
    TORCH_CHECK(z2.scalar_type() == torch::kBFloat16 && z1.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(z2.dim() == 3 && z1.dim() == 3 && z2.sizes() == z1.sizes());
    z2 = z2.contiguous(); z1 = z1.contiguous();
    int B = z2.size(0), QH = z2.size(1), D = z2.size(2), DP = D / 32;
    TORCH_CHECK(QH == KH * G && D % 32 == 0);
    auto o = torch::empty({B, KH, G, DP}, z2.options().dtype(torch::kInt32));
    int tot = B * KH * G * DP;
    psab<<<(tot + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(z2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(z1.data_ptr<at::BFloat16>()),
        o.data_ptr<int32_t>(), B, QH, KH, G, D, DP);
    if (auto e = cudaGetLastError(); e) throw std::runtime_error(cudaGetErrorString(e));
    return o;
}

bool qhash_grouped_epilogue_supported(torch::Tensor q, torch::Tensor p0, torch::Tensor p1, int KH, int G) {
    if (!q.is_cuda() || !p0.is_cuda() || !p1.is_cuda()) return false;
    if (q.scalar_type() != torch::kBFloat16 || p0.scalar_type() != torch::kBFloat16 ||
        p1.scalar_type() != torch::kBFloat16) return false;
    q = q3d(q);
    if (q.dim() != 3 || p0.dim() != 3 || p1.dim() != 3) return false;
    int QH = q.size(1), D = q.size(2);
    if (QH != KH * G || D % 32) return false;
    if (p0.size(0) != QH || p1.size(0) != QH) return false;
    if (p0.size(1) != D || p0.size(2) != D || p1.size(1) != D || p1.size(2) != D) return false;
    cudaDeviceProp pr;
    if (cudaGetDeviceProperties(&pr, q.get_device()) != cudaSuccess || pr.major < 8) return false;
    return true;
}

torch::Tensor qhash_project_pack_bf16(torch::Tensor q, torch::Tensor p0, torch::Tensor p1, int KH, int G) {
    TORCH_CHECK(q.is_cuda() && p0.is_cuda() && p1.is_cuda());
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && p0.scalar_type() == torch::kBFloat16 &&
                p1.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(p0.dim() == 3 && p1.dim() == 3);
    q = q3d(q).contiguous(); p0 = p0.contiguous(); p1 = p1.contiguous();
    int QH = q.size(1), D = q.size(2);
    TORCH_CHECK(p0.size(0) == QH && p1.size(0) == QH && p0.size(1) == D && p0.size(2) == D &&
                p1.size(1) == D && p1.size(2) == D && QH == KH * G);
    auto z1 = at::matmul(q.unsqueeze(-2), p0.unsqueeze(0)).squeeze(-2);
    z1 = z1 * at::sigmoid(z1) + q;
    auto z2 = at::matmul(z1.unsqueeze(-2), p1.unsqueeze(0)).squeeze(-2);
    return psab_cu(z2, z1, KH, G);
}

torch::Tensor qhash_project_pack_bf16_grouped_epilogue(torch::Tensor q, torch::Tensor p0, torch::Tensor p1,
                                                       int KH, int G) {
    TORCH_CHECK(qhash_grouped_epilogue_supported(q, p0, p1, KH, G));
    q = q3d(q).contiguous(); p0 = p0.contiguous(); p1 = p1.contiguous();
    int B = q.size(0), QH = q.size(1), D = q.size(2), DP = D / 32;
    auto z1 = at::matmul(q.unsqueeze(-2), p0.unsqueeze(0)).squeeze(-2);
    z1 = (z1 * at::sigmoid(z1) + q).contiguous();
    auto o = torch::empty({B, KH, G, DP}, q.options().dtype(torch::kInt32));
    int tot = B * KH * G * DP;
    gpb<<<(tot + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(z1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(p1.data_ptr<at::BFloat16>()),
        o.data_ptr<int32_t>(), B, QH, KH, G, D, DP);
    if (auto e = cudaGetLastError(); e) throw std::runtime_error(cudaGetErrorString(e));
    return o;
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qhash_project_pack_bf16", &qhash_project_pack_bf16);
    m.def("qhash_grouped_epilogue_supported", &qhash_grouped_epilogue_supported);
    m.def("qhash_project_pack_bf16_grouped_epilogue", &qhash_project_pack_bf16_grouped_epilogue);
}
