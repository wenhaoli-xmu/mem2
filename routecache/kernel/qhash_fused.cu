#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <vector>
#include <stdexcept>

namespace {

__global__ void pack_sign_kernel(
    const float* __restrict__ z2,   // [B,QH,D]
    int32_t* __restrict__ out,      // [B,KH,G,D_PACK]
    int B, int QH, int KH, int G, int D, int D_PACK
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * KH * G * D_PACK;
    if (idx >= total) return;

    int w = idx % D_PACK;
    int t0 = idx / D_PACK;
    int g = t0 % G;
    int t1 = t0 / G;
    int kh = t1 % KH;
    int b = t1 / KH;

    int qh = kh * G + g;
    int base = ((b * QH + qh) * D) + w * 32;

    uint32_t bits = 0u;
    #pragma unroll
    for (int bit = 0; bit < 32; ++bit) {
        float v = z2[base + bit];
        bits |= (static_cast<uint32_t>(v > 0.0f) << bit);
    }

    out[idx] = static_cast<int32_t>(bits);
}

torch::Tensor pack_sign_cuda(torch::Tensor z2, int KH, int G) {
    TORCH_CHECK(z2.is_cuda(), "z2 must be CUDA");
    TORCH_CHECK(z2.scalar_type() == torch::kFloat32, "z2 must be float32");
    TORCH_CHECK(z2.dim() == 3, "z2 must be [B,QH,D]");
    TORCH_CHECK(z2.is_contiguous(), "z2 must be contiguous");

    int B = z2.size(0);
    int QH = z2.size(1);
    int D = z2.size(2);

    TORCH_CHECK(QH == KH * G, "QH mismatch with KH*G");
    TORCH_CHECK(D % 32 == 0, "D must be divisible by 32");

    int D_PACK = D / 32;
    auto out = torch::empty({B, KH, G, D_PACK}, torch::dtype(torch::kInt32).device(z2.device()));

    int total = B * KH * G * D_PACK;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    auto stream = at::cuda::getCurrentCUDAStream();
    pack_sign_kernel<<<blocks, threads, 0, stream>>>(
        z2.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        B, QH, KH, G, D, D_PACK
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("pack_sign_kernel CUDA error: ") + cudaGetErrorString(err));
    }

    return out;
}

inline torch::Tensor normalize_query_3d(torch::Tensor query_like) {
    if (query_like.dim() == 4) {
        TORCH_CHECK(query_like.size(1) == 1, "query second dim must be 1");
        return query_like.select(1, 0);
    }
    TORCH_CHECK(query_like.dim() == 3, "query must be [B,1,QH,D] or [B,QH,D]");
    return query_like;
}

torch::Tensor qhash_project_pack_precast(
    torch::Tensor query_fp32,  // [B,1,QH,D] or [B,QH,D], float32
    torch::Tensor proj0_fp32,  // [QH,D,D], float32
    torch::Tensor proj1_fp32,  // [QH,D,D], float32
    int KH,
    int G
) {
    TORCH_CHECK(query_fp32.is_cuda() && proj0_fp32.is_cuda() && proj1_fp32.is_cuda(), "all inputs must be CUDA");
    TORCH_CHECK(query_fp32.scalar_type() == torch::kFloat32, "query_fp32 must be float32");
    TORCH_CHECK(proj0_fp32.scalar_type() == torch::kFloat32, "proj0_fp32 must be float32");
    TORCH_CHECK(proj1_fp32.scalar_type() == torch::kFloat32, "proj1_fp32 must be float32");
    TORCH_CHECK(proj0_fp32.dim() == 3 && proj1_fp32.dim() == 3, "proj tensors must be [QH,D,D]");

    auto q = normalize_query_3d(query_fp32).contiguous();   // [B,QH,D]
    auto p0 = proj0_fp32.contiguous();
    auto p1 = proj1_fp32.contiguous();

    int B = q.size(0);
    int QH = q.size(1);
    int D = q.size(2);

    TORCH_CHECK(p0.size(0) == QH && p1.size(0) == QH, "QH mismatch");
    TORCH_CHECK(p0.size(1) == D && p0.size(2) == D, "proj0 shape mismatch");
    TORCH_CHECK(p1.size(1) == D && p1.size(2) == D, "proj1 shape mismatch");
    TORCH_CHECK(QH == KH * G, "QH must equal KH*G");

    auto z1 = at::matmul(q.unsqueeze(-2), p0.unsqueeze(0)).squeeze(-2); // [B,QH,D], fp32
    z1 = z1 * at::sigmoid(z1) + q;
    auto z2 = at::matmul(z1.unsqueeze(-2), p1.unsqueeze(0)).squeeze(-2); // [B,QH,D], fp32
    z2 = (z2 + z1).contiguous();

    return pack_sign_cuda(z2, KH, G);
}

torch::Tensor qhash_project_pack_bf16(
    torch::Tensor query_bf16,  // [B,1,QH,D] or [B,QH,D], bf16
    torch::Tensor proj0_bf16,  // [QH,D,D], bf16
    torch::Tensor proj1_bf16,  // [QH,D,D], bf16
    int KH,
    int G
) {
    TORCH_CHECK(query_bf16.is_cuda() && proj0_bf16.is_cuda() && proj1_bf16.is_cuda(), "all inputs must be CUDA");
    TORCH_CHECK(query_bf16.scalar_type() == torch::kBFloat16, "query_bf16 must be bf16");
    TORCH_CHECK(proj0_bf16.scalar_type() == torch::kBFloat16, "proj0_bf16 must be bf16");
    TORCH_CHECK(proj1_bf16.scalar_type() == torch::kBFloat16, "proj1_bf16 must be bf16");
    TORCH_CHECK(proj0_bf16.dim() == 3 && proj1_bf16.dim() == 3, "proj tensors must be [QH,D,D]");

    auto q = normalize_query_3d(query_bf16).contiguous();   // [B,QH,D], bf16
    auto p0 = proj0_bf16.contiguous();
    auto p1 = proj1_bf16.contiguous();

    int QH = q.size(1);
    int D = q.size(2);

    TORCH_CHECK(p0.size(0) == QH && p1.size(0) == QH, "QH mismatch");
    TORCH_CHECK(p0.size(1) == D && p0.size(2) == D, "proj0 shape mismatch");
    TORCH_CHECK(p1.size(1) == D && p1.size(2) == D, "proj1 shape mismatch");
    TORCH_CHECK(QH == KH * G, "QH must equal KH*G");

    auto z1 = at::matmul(q.unsqueeze(-2), p0.unsqueeze(0)).squeeze(-2); // bf16-native GEMM path
    z1 = z1 * at::sigmoid(z1) + q;
    auto z2 = at::matmul(z1.unsqueeze(-2), p1.unsqueeze(0)).squeeze(-2);
    z2 = z2 + z1;

    auto z2f = z2.to(torch::kFloat32).contiguous();
    return pack_sign_cuda(z2f, KH, G);
}

torch::Tensor qhash_project_pack(
    torch::Tensor query,  // [B,1,QH,D], bf16/fp16/fp32
    torch::Tensor proj0,  // [QH,D,D]
    torch::Tensor proj1,  // [QH,D,D]
    int KH,
    int G
) {
    TORCH_CHECK(query.is_cuda() && proj0.is_cuda() && proj1.is_cuda(), "all inputs must be CUDA");
    auto qf = query.to(torch::kFloat32).contiguous();
    auto p0f = proj0.to(torch::kFloat32).contiguous();
    auto p1f = proj1.to(torch::kFloat32).contiguous();
    return qhash_project_pack_precast(qf, p0f, p1f, KH, G);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qhash_project_pack", &qhash_project_pack,
          "Fused query hash projection + sign pack (legacy API)",
          py::arg("query"), py::arg("proj0"), py::arg("proj1"),
          py::arg("KH"), py::arg("G"));

    m.def("qhash_project_pack_precast", &qhash_project_pack_precast,
          "Fused query hash projection + sign pack (fp32 precast path)",
          py::arg("query_fp32"), py::arg("proj0_fp32"), py::arg("proj1_fp32"),
          py::arg("KH"), py::arg("G"));

    m.def("qhash_project_pack_bf16", &qhash_project_pack_bf16,
          "Fused query hash projection + sign pack (bf16-native path)",
          py::arg("query_bf16"), py::arg("proj0_bf16"), py::arg("proj1_bf16"),
          py::arg("KH"), py::arg("G"));
}
