#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <vector>
#include <stdexcept>

namespace {

__global__ void pack_sign_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ z2,
    const __nv_bfloat16* __restrict__ z1,
    int32_t* __restrict__ out,
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
        float v = __bfloat162float(z2[base + bit]) + __bfloat162float(z1[base + bit]);
        bits |= (static_cast<uint32_t>(v > 0.0f) << bit);
    }

    out[idx] = static_cast<int32_t>(bits);
}

__global__ void grouped_epilogue_pack_bf16_kernel(
    const __nv_bfloat16* __restrict__ z1,
    const __nv_bfloat16* __restrict__ p1,
    int32_t* __restrict__ out,
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

    int out_col_base = w * 32;
    int z1_base = ((b * QH + qh) * D);
    int p1_base = qh * D * D;

    uint32_t bits = 0u;
    #pragma unroll
    for (int bit = 0; bit < 32; ++bit) {
        int out_col = out_col_base + bit;
        float acc = 0.0f;

        for (int j = 0; j < D; ++j) {
            float a = __bfloat162float(z1[z1_base + j]);
            float bval = __bfloat162float(p1[p1_base + j * D + out_col]);
            acc += a * bval;
        }

        float v = acc + __bfloat162float(z1[z1_base + out_col]);
        bits |= (static_cast<uint32_t>(v > 0.0f) << bit);
    }

    out[idx] = static_cast<int32_t>(bits);
}

inline torch::Tensor normalize_query_3d(torch::Tensor query_like) {
    if (query_like.dim() == 4) {
        TORCH_CHECK(query_like.size(1) == 1, "query second dim must be 1");
        return query_like.select(1, 0);
    }
    TORCH_CHECK(query_like.dim() == 3, "query must be [B,1,QH,D] or [B,QH,D]");
    return query_like;
}

torch::Tensor pack_sign_add_bf16_cuda(torch::Tensor z2, torch::Tensor z1, int KH, int G) {
    TORCH_CHECK(z2.is_cuda() && z1.is_cuda(), "z1/z2 must be CUDA");
    TORCH_CHECK(z2.scalar_type() == torch::kBFloat16 && z1.scalar_type() == torch::kBFloat16, "z1/z2 must be bf16");
    TORCH_CHECK(z2.dim() == 3 && z1.dim() == 3, "z1/z2 must be [B,QH,D]");
    TORCH_CHECK(z2.sizes() == z1.sizes(), "z1/z2 shape mismatch");

    z2 = z2.contiguous();
    z1 = z1.contiguous();

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
    pack_sign_add_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(z2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(z1.data_ptr<at::BFloat16>()),
        out.data_ptr<int32_t>(),
        B, QH, KH, G, D, D_PACK
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("pack_sign_add_bf16_kernel CUDA error: ") + cudaGetErrorString(err));
    }
    return out;
}

bool qhash_grouped_epilogue_supported(
    torch::Tensor query_bf16,
    torch::Tensor proj0_bf16,
    torch::Tensor proj1_bf16,
    int KH,
    int G
) {
    if (!query_bf16.is_cuda() || !proj0_bf16.is_cuda() || !proj1_bf16.is_cuda()) return false;
    if (query_bf16.scalar_type() != torch::kBFloat16) return false;
    if (proj0_bf16.scalar_type() != torch::kBFloat16 || proj1_bf16.scalar_type() != torch::kBFloat16) return false;

    auto q = normalize_query_3d(query_bf16);
    if (q.dim() != 3 || proj0_bf16.dim() != 3 || proj1_bf16.dim() != 3) return false;

    int QH = q.size(1);
    int D = q.size(2);
    if (QH != KH * G) return false;
    if (D % 32 != 0) return false;
    if (proj0_bf16.size(0) != QH || proj1_bf16.size(0) != QH) return false;
    if (proj0_bf16.size(1) != D || proj0_bf16.size(2) != D) return false;
    if (proj1_bf16.size(1) != D || proj1_bf16.size(2) != D) return false;

    int device_idx = query_bf16.get_device();
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_idx) != cudaSuccess) return false;
    if (prop.major < 8) return false;

    return true;
}

torch::Tensor qhash_project_pack_bf16(
    torch::Tensor query_bf16,
    torch::Tensor proj0_bf16,
    torch::Tensor proj1_bf16,
    int KH,
    int G
) {
    TORCH_CHECK(query_bf16.is_cuda() && proj0_bf16.is_cuda() && proj1_bf16.is_cuda(), "all inputs must be CUDA");
    TORCH_CHECK(query_bf16.scalar_type() == torch::kBFloat16, "query_bf16 must be bf16");
    TORCH_CHECK(proj0_bf16.scalar_type() == torch::kBFloat16, "proj0_bf16 must be bf16");
    TORCH_CHECK(proj1_bf16.scalar_type() == torch::kBFloat16, "proj1_bf16 must be bf16");
    TORCH_CHECK(proj0_bf16.dim() == 3 && proj1_bf16.dim() == 3, "proj tensors must be [QH,D,D]");

    auto q = normalize_query_3d(query_bf16).contiguous();
    auto p0 = proj0_bf16.contiguous();
    auto p1 = proj1_bf16.contiguous();

    int QH = q.size(1);
    int D = q.size(2);

    TORCH_CHECK(p0.size(0) == QH && p1.size(0) == QH, "QH mismatch");
    TORCH_CHECK(p0.size(1) == D && p0.size(2) == D, "proj0 shape mismatch");
    TORCH_CHECK(p1.size(1) == D && p1.size(2) == D, "proj1 shape mismatch");
    TORCH_CHECK(QH == KH * G, "QH must equal KH*G");

    auto z1 = at::matmul(q.unsqueeze(-2), p0.unsqueeze(0)).squeeze(-2);
    z1 = z1 * at::sigmoid(z1) + q;
    auto z2 = at::matmul(z1.unsqueeze(-2), p1.unsqueeze(0)).squeeze(-2);

    return pack_sign_add_bf16_cuda(z2, z1, KH, G);
}

torch::Tensor qhash_project_pack_bf16_grouped_epilogue(
    torch::Tensor query_bf16,
    torch::Tensor proj0_bf16,
    torch::Tensor proj1_bf16,
    int KH,
    int G
) {
    TORCH_CHECK(qhash_grouped_epilogue_supported(query_bf16, proj0_bf16, proj1_bf16, KH, G),
                "grouped epilogue path unsupported for given inputs/device");

    auto q = normalize_query_3d(query_bf16).contiguous();
    auto p0 = proj0_bf16.contiguous();
    auto p1 = proj1_bf16.contiguous();

    int B = q.size(0);
    int QH = q.size(1);
    int D = q.size(2);
    int D_PACK = D / 32;

    auto z1 = at::matmul(q.unsqueeze(-2), p0.unsqueeze(0)).squeeze(-2);
    z1 = z1 * at::sigmoid(z1) + q;
    z1 = z1.contiguous();

    auto out = torch::empty({B, KH, G, D_PACK}, torch::dtype(torch::kInt32).device(q.device()));

    int total = B * KH * G * D_PACK;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    auto stream = at::cuda::getCurrentCUDAStream();
    grouped_epilogue_pack_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(z1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(p1.data_ptr<at::BFloat16>()),
        out.data_ptr<int32_t>(),
        B, QH, KH, G, D, D_PACK
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("grouped_epilogue_pack_bf16_kernel CUDA error: ")
                                 + cudaGetErrorString(err));
    }

    return out;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qhash_project_pack_bf16", &qhash_project_pack_bf16,
          "Fused query hash projection + residual-sign pack (bf16-native path)",
          py::arg("query_bf16"), py::arg("proj0_bf16"), py::arg("proj1_bf16"),
          py::arg("KH"), py::arg("G"));

    m.def("qhash_grouped_epilogue_supported", &qhash_grouped_epilogue_supported,
          "Check if grouped bf16 epilogue fused path is supported for current inputs/device",
          py::arg("query_bf16"), py::arg("proj0_bf16"), py::arg("proj1_bf16"),
          py::arg("KH"), py::arg("G"));

    m.def("qhash_project_pack_bf16_grouped_epilogue", &qhash_project_pack_bf16_grouped_epilogue,
          "Grouped bf16 qhash path with fused second projection + residual sign-pack epilogue",
          py::arg("query_bf16"), py::arg("proj0_bf16"), py::arg("proj1_bf16"),
          py::arg("KH"), py::arg("G"));
}
