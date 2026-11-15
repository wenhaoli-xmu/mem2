#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

#define WR(p) (reinterpret_cast<float4*>(&(p))[0])
#define RD(p) (reinterpret_cast<const float4*>(&(p))[0])
#define EMPTY -1

__device__ __forceinline__ int hf(int k, int m) { return ((unsigned)k * 2654435761u) & m; }

template <int TS>
__global__ void fk(const int64_t* __restrict__ tk, int* __restrict__ li, int* __restrict__ lp,
                   const __nv_bfloat16* __restrict__ gk, const __nv_bfloat16* __restrict__ gv,
                   __nv_bfloat16* __restrict__ ck, __nv_bfloat16* __restrict__ cv,
                   float* __restrict__ hr, int B, int KH, int LL, int AS, int TB, int D, int MT) {
    constexpr int M = TS - 1;
    int b = blockIdx.x, h = blockIdx.y, t = threadIdx.x, bh = b * KH + h;
    extern __shared__ char sm[];
    int *ht = reinterpret_cast<int*>(sm), *di = ht + TS, *cp = di + TB;

    for (int i = t; i < TS; i += blockDim.x) ht[i] = EMPTY;
    if (!t) *cp = 0;
    __syncthreads();

    int* g = li + bh * AS;
    for (int i = t; i < LL; i += blockDim.x) {
        int x = g[i];
        if (x < 0) continue;
        for (int p = hf(x, M);; p = (p + 1) & M) {
            int o = atomicCAS(&ht[p], EMPTY, x);
            if (o == EMPTY || o == x) break;
        }
    }
    __syncthreads();

    const int64_t* tp = tk + bh * TB;
    for (int i = t; i < TB; i += blockDim.x) {
        int x = (int)tp[i];
        if (x < 0) continue;
        int p = hf(x, M);
        bool e = false;
        while (ht[p] != EMPTY) {
            if (ht[p] == x) { e = true; break; }
            p = (p + 1) & M;
        }
        if (!e) di[atomicAdd(cp, 1)] = x;
    }
    __syncthreads();

    int nu = *cp;
    if (!t) hr[bh] = TB > 0 ? (float)(TB - nu) / TB : 0.f;
    if (!nu) return;

    int cur = lp[bh];
    if (!t) {
        for (int i = 0; i < nu; i++) g[(cur + i) % LL] = di[i];
        lp[bh] = (cur + nu) % LL;
    }
    __syncthreads();

    int dv = D / 8, tot = nu * dv;
    for (int x = t; x < tot; x += blockDim.x) {
        int u = x / dv, d = (x % dv) * 8, tok = di[u], wp = (cur + u) % LL;
        long s = ((long)b * MT * KH * D) + ((long)tok * KH * D) + ((long)h * D) + d;
        long o = ((long)b * AS * KH * D) + ((long)wp * KH * D) + ((long)h * D) + d;
        WR(ck[o]) = RD(gk[s]);
        WR(cv[o]) = RD(gv[s]);
    }
}

torch::Tensor fifo_update_cuda(torch::Tensor tk, torch::Tensor li, torch::Tensor lp,
                               torch::Tensor gk, torch::Tensor gv, torch::Tensor ck, torch::Tensor cv,
                               int LL, int TB) {
    TORCH_CHECK(gk.scalar_type() == torch::kBFloat16 && gv.scalar_type() == torch::kBFloat16 &&
                ck.scalar_type() == torch::kBFloat16 && cv.scalar_type() == torch::kBFloat16);

    int hs = 1024;
    for (int s : {1024, 2048, 4096, 8192}) if (LL * 2 > s) hs = s * 2;
    TORCH_CHECK(LL <= hs / 2);

    int B = tk.size(0), KH = tk.size(1), MT = gk.size(1), D = gk.size(3), AS = ck.size(1);
    TORCH_CHECK(li.size(2) == AS && D % 8 == 0 && LL <= AS);

    auto hr = torch::empty({B, KH}, tk.options().dtype(torch::kFloat32));
    size_t sm = hs * sizeof(int) + TB * sizeof(int) + sizeof(int);
    auto st = at::cuda::getCurrentCUDAStream();

    #define GO(S) fk<S><<<dim3(B, KH), 256, sm, st>>>(tk.data_ptr<int64_t>(), li.data_ptr<int>(), lp.data_ptr<int>(), \
        reinterpret_cast<const __nv_bfloat16*>(gk.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const __nv_bfloat16*>(gv.data_ptr<at::BFloat16>()), \
        reinterpret_cast<__nv_bfloat16*>(ck.data_ptr<at::BFloat16>()), \
        reinterpret_cast<__nv_bfloat16*>(cv.data_ptr<at::BFloat16>()), \
        hr.data_ptr<float>(), B, KH, LL, AS, TB, D, MT)

    if (sm > 48 * 1024) {
        int dev, mx;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (sm <= (size_t)mx) {
            if (hs == 8192) cudaFuncSetAttribute(fk<8192>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
            if (hs == 16384) cudaFuncSetAttribute(fk<16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        }
    }

    switch (hs) {
        case 1024: GO(1024); break;
        case 2048: GO(2048); break;
        case 4096: GO(4096); break;
        case 8192: GO(8192); break;
        case 16384: GO(16384); break;
        default: TORCH_CHECK(false);
    }
    return hr;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("update", &fifo_update_cuda); }
