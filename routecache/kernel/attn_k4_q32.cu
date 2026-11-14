#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int BLOCK_N = 64;
constexpr int KH = 4;
constexpr int D = 4;
constexpr int QH = 32;
constexpr int G = QH / KH;

__global__ void attn_k4_q32_kernel_unified(
    const int32_t* __restrict__ q_ptr,
    const int32_t* __restrict__ k_ptr,
    int16_t* __restrict__ o_ptr,
    const int32_t* __restrict__ kv_length_ptr,
    int B, 
    int QN, 
    int KN_stride,
    int n_blocks
) {
    // Flatten (B, QN, n_blocks) onto grid.x to avoid CUDA gridDim.{y,z} limits.
    // total_blocks = B * QN * n_blocks.
    int linear_idx = static_cast<int>(blockIdx.x);
    int n_block_idx = linear_idx % n_blocks;
    linear_idx /= n_blocks;
    int q_idx = linear_idx % QN;
    int b_idx = linear_idx / QN;
    
    int n_start = n_block_idx * BLOCK_N;
    int n_local = threadIdx.x;
    int h = threadIdx.y;
    int n = n_start + n_local;
    int KN_active = static_cast<int>(*kv_length_ptr);
    // Clamp on device to avoid OOB if caller passes a bad length.
    if (KN_active < 0) KN_active = 0;
    if (KN_active > KN_stride) KN_active = KN_stride;

    extern __shared__ int32_t shared_mem[];
    int32_t* q_shared = shared_mem;
    int32_t* k_shared = q_shared + G * KH * D;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int num_threads = BLOCK_N * KH;

    size_t q_offset_base = (size_t)b_idx * QN * QH * D + (size_t)q_idx * QH * D;
    
    // Load Q heads in standard GQA order: kv_head h corresponds to Q heads [h*G, h*G+1, ..., h*G+G-1]
    for (int gh_idx = tid; gh_idx < G * KH; gh_idx += num_threads) {
        int h_load = gh_idx / G;    // kv_head index
        int g = gh_idx % G;         // group index within kv_head
        int q_head = h_load * G + g; // Q head index (standard GQA order)
        
        int4 q_val = *reinterpret_cast<const int4*>(
            q_ptr + q_offset_base + q_head * D
        );
        int shared_offset = q_head * D;
        q_shared[shared_offset]     = q_val.x;
        q_shared[shared_offset + 1] = q_val.y;
        q_shared[shared_offset + 2] = q_val.z;
        q_shared[shared_offset + 3] = q_val.w;
    }

    size_t k_offset_base = (size_t)b_idx * KN_stride * KH * D;

    for (int nh_idx = tid; nh_idx < BLOCK_N * KH; nh_idx += num_threads) {
        int n_local_load = nh_idx / KH;
        int h_load = nh_idx % KH;
        int n_global = n_start + n_local_load;
        int k_shared_offset = n_local_load * KH * D + h_load * D;

        if (n_global < KN_active) {
            int4 k_val = *reinterpret_cast<const int4*>(
                k_ptr + k_offset_base + (size_t)n_global * KH * D + h_load * D
            );
            k_shared[k_shared_offset]     = k_val.x;
            k_shared[k_shared_offset + 1] = k_val.y;
            k_shared[k_shared_offset + 2] = k_val.z;
            k_shared[k_shared_offset + 3] = k_val.w;
        } else {
            k_shared[k_shared_offset]     = 0;
            k_shared[k_shared_offset + 1] = 0;
            k_shared[k_shared_offset + 2] = 0;
            k_shared[k_shared_offset + 3] = 0;
        }
    }

    __syncthreads();

    if (n < KN_stride) {
        size_t o_idx = (size_t)b_idx * (KH * QN * KN_stride)
                     + (size_t)h     * (QN * KN_stride)
                     + (size_t)q_idx * (KN_stride)
                     + (size_t)n;

        if (n < KN_active) {
            int accum = 0;
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                // Standard GQA: kv_head h corresponds to Q heads [h*G, h*G+1, ..., h*G+G-1]
                const int q_base = (h * G + g) * D;
                const int k_base = (n_local * KH + h) * D;

                int32_t q0 = q_shared[q_base];
                int32_t q1 = q_shared[q_base + 1];
                int32_t q2 = q_shared[q_base + 2];
                int32_t q3 = q_shared[q_base + 3];

                int32_t k0 = k_shared[k_base];
                int32_t k1 = k_shared[k_base + 1];
                int32_t k2 = k_shared[k_base + 2];
                int32_t k3 = k_shared[k_base + 3];

                accum += __popc(~(q0 ^ k0))
                       + __popc(~(q1 ^ k1))
                       + __popc(~(q2 ^ k2))
                       + __popc(~(q3 ^ k3));
            }
            o_ptr[o_idx] = static_cast<int16_t>(accum);
        } else {
            // Make inactive positions extremely small so top-k doesn't pick them.
            o_ptr[o_idx] = static_cast<int16_t>(-32768);
        }
    }
}

}

torch::Tensor attn_k4_q32(torch::Tensor q_hash, torch::Tensor k_hash, torch::Tensor kv_length) {
    TORCH_CHECK(q_hash.is_cuda() && k_hash.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(q_hash.dtype() == torch::kInt32 && k_hash.dtype() == torch::kInt32, "Inputs must be int32");
    TORCH_CHECK(q_hash.dim() == 4 && k_hash.dim() == 4, "Inputs must be 4D tensors");
    TORCH_CHECK(kv_length.is_cuda(), "kv_length must be a CUDA tensor");
    TORCH_CHECK(kv_length.dtype() == torch::kInt32, "kv_length must be int32");
    TORCH_CHECK(kv_length.numel() == 1, "kv_length must contain exactly one item");

    q_hash = q_hash.contiguous();
    k_hash = k_hash.contiguous();

    const int B = q_hash.size(0);
    const int QN = q_hash.size(1);
    const int KN_physical = k_hash.size(1); 
    // NOTE: For CUDA graph capture/replay, output shape must remain static.
    // We always compute an output of length KN_physical, and the kernel masks n >= KN_active.

    TORCH_CHECK(q_hash.size(2) == QH && k_hash.size(2) == KH, "Head dimension mismatch");
    TORCH_CHECK(q_hash.size(3) == D && k_hash.size(3) == D, "Feature dimension mismatch");

    torch::Device device = q_hash.device();
    TORCH_CHECK(k_hash.device() == device, "k_hash must be on the same device as q_hash");

    torch::Tensor output = torch::empty({B, KH, QN, KN_physical}, q_hash.options().dtype(torch::kInt16));

    torch::DeviceGuard device_guard(device);

    size_t shared_mem_size = (G * KH * D + BLOCK_N * KH * D) * sizeof(int32_t);

    dim3 blockDim(BLOCK_N, KH);
    const int n_blocks = (KN_physical + BLOCK_N - 1) / BLOCK_N;
    const int64_t total_blocks = static_cast<int64_t>(B) * static_cast<int64_t>(QN) * static_cast<int64_t>(n_blocks);
    TORCH_CHECK(total_blocks > 0, "Invalid launch: total_blocks must be > 0");
    TORCH_CHECK(total_blocks <= INT32_MAX, "Invalid launch: too many blocks (exceeds INT32_MAX)");
    dim3 gridDim(static_cast<uint32_t>(total_blocks), 1, 1);
    
    // IMPORTANT: respect PyTorch current stream.
    attn_k4_q32_kernel_unified<<<gridDim, blockDim, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        q_hash.data_ptr<int32_t>(),
        k_hash.data_ptr<int32_t>(),
        output.data_ptr<int16_t>(),
        kv_length.data_ptr<int32_t>(),
        B, QN, KN_physical, n_blocks
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    if (QN == 1) {
        return output.squeeze(2); 
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_k4_q32", &attn_k4_q32, "Attention kernel for k4 q32",
          py::arg("q_hash"), py::arg("k_hash"), py::arg("kv_length"));
}