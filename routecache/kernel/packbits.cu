#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void pk(const __nv_bfloat16* __restrict__ x, int32_t* __restrict__ o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned p = __ballot_sync(-1u, i < n && __bfloat162float(x[i]) > 0.f);
    if (!(threadIdx.x & 31) && i < n) o[i >> 5] = p;
}

torch::Tensor packbits(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.size(-1) % 32 == 0);
    auto c = x.contiguous(), n = c.numel();
    std::vector<int64_t> s(c.sizes().begin(), c.sizes().end());
    s.back() /= 32;
    auto y = torch::empty({n / 32}, c.options().dtype(torch::kInt32));
    pk<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(c.data_ptr<at::BFloat16>()), y.data_ptr<int32_t>(), n);
    if (auto e = cudaGetLastError(); e) throw std::runtime_error(cudaGetErrorString(e));
    return y.view(s);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("packbits", &packbits); }
