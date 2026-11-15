#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

#define WR(p) (reinterpret_cast<float4*>(&(p))[0])
#define RD(p) (reinterpret_cast<const float4*>(&(p))[0])
#define EMPTY -1
#define INV 0x7FFFFFFF

__device__ __forceinline__ int hf(int k, int m) { return ((unsigned)k * 2654435761u) & m; }

__device__ __forceinline__ void wmin(int& v, int& i) {
    for (int o = 16; o > 0; o >>= 1) {
        int a = __shfl_down_sync(0xffffffff, v, o), b = __shfl_down_sync(0xffffffff, i, o);
        if (a < v) { v = a; i = b; }
    }
}

template <int TS>
__global__ void lk(const int64_t* __restrict__ tk, int* __restrict__ li, int* __restrict__ lt,
                   int* __restrict__ ct, const __nv_bfloat16* __restrict__ gk,
                   const __nv_bfloat16* __restrict__ gv, __nv_bfloat16* __restrict__ ck,
                   __nv_bfloat16* __restrict__ cv, float* __restrict__ hr,
                   int B, int KH, int CS, int AS, int TB, int D, int MT) {
    constexpr int M = TS - 1;
    int b = blockIdx.x, h = blockIdx.y, t = threadIdx.x, l = t & 31, w = t >> 5, nw = blockDim.x >> 5;
    int bh = b * KH + h;

    extern __shared__ char sm[];
    int* ht = reinterpret_cast<int*>(sm);
    int* hv = ht + TS;
    int* mi = hv + TS;
    int* vp = mi + TB;
    int* mc = vp + TB;
    int* hc = mc + 1;
    int* bt = hc + 1;
    int* wv = bt + 1;
    int* wp = wv + nw;

    if (!t) { *mc = 0; *hc = 0; *bt = 0; }
    for (int i = t; i < TS; i += blockDim.x) { ht[i] = EMPTY; hv[i] = EMPTY; }
    __syncthreads();

    int* gi = li + bh * AS;
    int* gt = lt + bh * AS;

    for (int i = t; i < CS; i += blockDim.x) {
        int x = gi[i];
        if (x < 0) continue;
        for (int p = hf(x, M);; p = (p + 1) & M) {
            int o = atomicCAS(&hv[p], EMPTY, x);
            if (o == EMPTY || o == x) {
                if (o == EMPTY) ht[p] = i;
                break;
            }
        }
    }
    __syncthreads();

    const int64_t* tp = tk + bh * TB;
    int hcp[32], hrn[32], nh = 0;
    for (int i = t; i < TB; i += blockDim.x) {
        int x = (int)tp[i];
        if (x < 0) continue;
        int p = hf(x, M), cp = -1;
        while (hv[p] != EMPTY) {
            if (hv[p] == x) { cp = ht[p]; break; }
            p = (p + 1) & M;
        }
        if (cp >= 0) {
            if (nh < 32) { hcp[nh] = cp; hrn[nh] = atomicAdd(hc, 1); nh++; }
        } else {
            mi[atomicAdd(mc, 1)] = x;
        }
    }
    __syncthreads();

    int nm = *mc, nhit = *hc;
    if (!t) {
        int tot = nhit + nm;
        hr[bh] = tot > 0 ? (float)nhit / tot : 0.f;
        if (nhit > 0) *bt = atomicAdd(&ct[bh], nhit);
    }
    __syncthreads();

    int base = *bt;
    for (int k = 0; k < nh; k++) gt[hcp[k]] = base + hrn[k] + 1;
    __syncthreads();
    if (nm == 0) return;

    int* ts = ht;
    __shared__ int ibt;
    if (!t) ibt = atomicAdd(&ct[bh], nm) + 1;

    for (int i = t; i < CS; i += blockDim.x) ts[i] = gi[i] < 0 ? -1 : gt[i];
    __syncthreads();

    for (int m = 0; m < nm; m++) {
        int lv = INV, lp = -1;
        for (int i = t; i < CS; i += blockDim.x) {
            int v = ts[i];
            if (v != INV && v < lv) { lv = v; lp = i; }
        }
        wmin(lv, lp);
        if (!l) { wv[w] = lv; wp[w] = lp; }
        __syncthreads();
        if (!t) {
            int gv = INV, gp = -1;
            for (int k = 0; k < nw; k++) if (wv[k] < gv) { gv = wv[k]; gp = wp[k]; }
            vp[m] = gp;
            if (gp != -1) ts[gp] = INV;
        }
        __syncthreads();
    }

    if (!t) {
        for (int i = 0; i < nm; i++) {
            int p = vp[i];
            if (p >= 0) { gi[p] = mi[i]; gt[p] = ibt + i; }
        }
    }
    __syncthreads();

    int dv = D / 8, tot = nm * dv;
    for (int x = t; x < tot; x += blockDim.x) {
        int u = x / dv, d = (x % dv) * 8, tok = mi[u], p = vp[u];
        if (p < 0) continue;
        long s = ((long)b * MT * KH * D) + ((long)tok * KH * D) + ((long)h * D) + d;
        long o = ((long)b * AS * KH * D) + ((long)p * KH * D) + ((long)h * D) + d;
        WR(ck[o]) = RD(gk[s]);
        WR(cv[o]) = RD(gv[s]);
    }
}

torch::Tensor lru_update_cuda(torch::Tensor tk, torch::Tensor li, torch::Tensor lt, torch::Tensor ct,
                              torch::Tensor gk, torch::Tensor gv, torch::Tensor ck, torch::Tensor cv,
                              int CS, int TB) {
    TORCH_CHECK(gk.scalar_type() == torch::kBFloat16 && ck.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(lt.scalar_type() == torch::kInt32);

    int B = tk.size(0), KH = tk.size(1), MT = gk.size(1), D = gk.size(3), AS = ck.size(1);
    int hs = 1024;
    for (int s : {1024, 2048, 4096, 8192}) if (CS * 2 > s) hs = s * 2;
    TORCH_CHECK(CS <= hs / 2);

    auto hr = torch::empty({B, KH}, tk.options().dtype(torch::kFloat32));
    size_t sm = hs * 2 * sizeof(int) + TB * 2 * sizeof(int) + 128 * sizeof(int);
    auto st = at::cuda::getCurrentCUDAStream();

    #define GO(S) lk<S><<<dim3(B, KH), 256, sm, st>>>(tk.data_ptr<int64_t>(), li.data_ptr<int>(), lt.data_ptr<int>(), \
        ct.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(gk.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const __nv_bfloat16*>(gv.data_ptr<at::BFloat16>()), \
        reinterpret_cast<__nv_bfloat16*>(ck.data_ptr<at::BFloat16>()), \
        reinterpret_cast<__nv_bfloat16*>(cv.data_ptr<at::BFloat16>()), \
        hr.data_ptr<float>(), B, KH, CS, AS, TB, D, MT)

    if (sm > 48 * 1024) {
        int dev, mx;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (sm <= (size_t)mx) {
            if (hs == 1024) cudaFuncSetAttribute(lk<1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
            if (hs == 2048) cudaFuncSetAttribute(lk<2048>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
            if (hs == 4096) cudaFuncSetAttribute(lk<4096>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
            if (hs == 8192) cudaFuncSetAttribute(lk<8192>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
            if (hs == 16384) cudaFuncSetAttribute(lk<16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("update", &lru_update_cuda); }
