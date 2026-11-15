#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <stdexcept>

namespace {

__global__ void psk(const float* __restrict__ z, int32_t* __restrict__ o,
                    int B, int QH, int KH, int G, int D, int DP) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, tot = B * KH * G * DP;
    if (i >= tot) return;
    int w = i % DP, t0 = i / DP, g = t0 % G, t1 = t0 / G, kh = t1 % KH, b = t1 / KH;
    int base = ((b * QH + kh * G + g) * D) + w * 32;
    unsigned bits = 0u;
    #pragma unroll
    for (int k = 0; k < 32; k++) bits |= (unsigned)(z[base + k] > 0.f) << k;
    o[i] = bits;
}

torch::Tensor pack_sign_cuda(torch::Tensor z, int KH, int G) {
    TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kFloat32 && z.dim() == 3 && z.is_contiguous());
    int B = z.size(0), QH = z.size(1), D = z.size(2);
    TORCH_CHECK(QH == KH * G && D % 32 == 0);
    int DP = D / 32, tot = B * KH * G * DP;
    auto o = torch::empty({B, KH, G, DP}, z.options().dtype(torch::kInt32));
    psk<<<(tot + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        z.data_ptr<float>(), o.data_ptr<int32_t>(), B, QH, KH, G, D, DP);
    if (auto e = cudaGetLastError(); e) throw std::runtime_error(cudaGetErrorString(e));
    return o;
}

inline torch::Tensor q3d(torch::Tensor q) {
    if (q.dim() == 4) { TORCH_CHECK(q.size(1) == 1); return q.select(1, 0); }
    TORCH_CHECK(q.dim() == 3);
    return q;
}

torch::Tensor qhash_project_pack_precast(torch::Tensor q, torch::Tensor p0, torch::Tensor p1, int KH, int G) {
    TORCH_CHECK(q.is_cuda() && p0.is_cuda() && p1.is_cuda());
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 && p0.scalar_type() == torch::kFloat32 &&
                p1.scalar_type() == torch::kFloat32);
    TORCH_CHECK(p0.dim() == 3 && p1.dim() == 3);
    q = q3d(q).contiguous(); p0 = p0.contiguous(); p1 = p1.contiguous();
    int B = q.size(0), QH = q.size(1), D = q.size(2);
    TORCH_CHECK(p0.size(0) == QH && p1.size(0) == QH && p0.size(1) == D && p0.size(2) == D &&
                p1.size(1) == D && p1.size(2) == D && QH == KH * G);
    auto z1 = at::matmul(q.unsqueeze(-2), p0.unsqueeze(0)).squeeze(-2);
    z1 = z1 * at::sigmoid(z1) + q;
    auto z2 = at::matmul(z1.unsqueeze(-2), p1.unsqueeze(0)).squeeze(-2) + z1;
    return pack_sign_cuda(z2.contiguous(), KH, G);
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
    auto z2 = at::matmul(z1.unsqueeze(-2), p1.unsqueeze(0)).squeeze(-2) + z1;
    return pack_sign_cuda(z2.to(torch::kFloat32).contiguous(), KH, G);
}

torch::Tensor qhash_project_pack(torch::Tensor q, torch::Tensor p0, torch::Tensor p1, int KH, int G) {
    TORCH_CHECK(q.is_cuda() && p0.is_cuda() && p1.is_cuda());
    return qhash_project_pack_precast(q.to(torch::kFloat32).contiguous(),
                                      p0.to(torch::kFloat32).contiguous(),
                                      p1.to(torch::kFloat32).contiguous(), KH, G);
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qhash_project_pack", &qhash_project_pack);
    m.def("qhash_project_pack_precast", &qhash_project_pack_precast);
    m.def("qhash_project_pack_bf16", &qhash_project_pack_bf16);
}
