#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define F __bfloat162float

__global__ void hp(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ p0,
                   const __nv_bfloat16* __restrict__ p1, int32_t* __restrict__ o,
                   int B, int T, int H, int D) {
    int L = blockIdx.x, h = L % H, t = L / H % T, b = L / (H * T), i = threadIdx.x, nt = blockDim.x;
    extern __shared__ float s[];
    float *a = s, *c = s + D, *z = s + 2 * D;
    size_t xi = ((size_t)(b * T + t) * H + h) * D, pi = (size_t)h * D * D;
    size_t oi = ((size_t)(b * T + t) * H + h) * (D >> 5);

    for (int j = i; j < D; j += nt) a[j] = F(x[xi + j]);
    __syncthreads();
    for (int j = i; j < D; j += nt) {
        float v = 0;
        for (int k = 0; k < D; k++) v += a[k] * F(p0[pi + k * D + j]);
        c[j] = v / (1 + expf(-v)) + a[j];
    }
    __syncthreads();
    for (int j = i; j < D; j += nt) {
        float v = 0;
        for (int k = 0; k < D; k++) v += c[k] * F(p1[pi + k * D + j]);
        z[j] = v + c[j];
    }
    __syncthreads();

    int w = i >> 5, l = i & 31, nw = nt >> 5, np = D >> 5;
    for (int q = w; q < np; q += nw) {
        unsigned pk = __ballot_sync(-1u, z[q * 32 + l] > 0.f);
        if (!l) o[oi + q] = pk;
    }
}

torch::Tensor hash_packbits(torch::Tensor x, torch::Tensor p0, torch::Tensor p1) {
    TORCH_CHECK(x.is_cuda() && p0.is_cuda() && p1.is_cuda());
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && p0.scalar_type() == torch::kBFloat16 &&
                p1.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.dim() == 4 && p0.dim() == 3 && p1.dim() == 3);

    x = x.contiguous(); p0 = p0.contiguous(); p1 = p1.contiguous();
    int B = x.size(0), T = x.size(1), H = x.size(2), D = x.size(3);
    TORCH_CHECK(D % 32 == 0 && p0.size(0) == H && p0.size(1) == D && p0.size(2) == D &&
                p1.size(0) == H && p1.size(1) == D && p1.size(2) == D);

    auto o = torch::empty({B, T, H, D / 32}, x.options().dtype(torch::kInt32));
    hp<<<B * T * H, 128, 3 * D * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(p0.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(p1.data_ptr<at::BFloat16>()),
        o.data_ptr<int32_t>(), B, T, H, D);
    if (auto e = cudaGetLastError(); e) throw std::runtime_error(cudaGetErrorString(e));
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("hash_packbits", &hash_packbits); }
