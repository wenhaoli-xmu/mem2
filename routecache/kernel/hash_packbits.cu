/**
 * Fused Hash + Packbits Kernel
 * 
 * Combines HashModule (2-layer MLP) + packbits into a single kernel.
 * 
 * HashModule structure (dims=[128, 128, 128]):
 *   Layer 0: y = x @ proj0, y = silu(y), y = y + x  (with SiLU, with residual)
 *   Layer 1: z = y @ proj1, z = z + y              (no SiLU, with residual)
 * 
 * Packbits: convert z > 0 to packed int32 (32 bits per int32)
 * 
 * Input:  x      [B, T, H, D]   (bf16)
 *         proj0  [H, D, D]      (bf16)
 *         proj1  [H, D, D]      (bf16)
 * Output: bins   [B, T, H, D/32] (int32)
 */

#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

namespace {

// Helper: bf16 → float
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Helper: float → bf16
__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) {
    return __float2bfloat16(x);
}

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Configuration
constexpr int HASH_DIM = 128;      // Must match dims[0] = dims[1] = dims[2]
constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_HEAD = 128;  // Threads to process one head

/**
 * Kernel: hash_packbits_kernel
 * 
 * Each block processes one (batch, token, head) tuple.
 * Grid: (B * T * H, 1, 1)
 * Block: (THREADS_PER_HEAD, 1, 1)
 */
__global__ void hash_packbits_kernel(
    const __nv_bfloat16* __restrict__ input,   // [B, T, H, D]
    const __nv_bfloat16* __restrict__ proj0,   // [H, D, D]
    const __nv_bfloat16* __restrict__ proj1,   // [H, D, D]
    int32_t* __restrict__ output,              // [B, T, H, D/32]
    int B, int T, int H, int D
) {
    // D must be 128 and divisible by 32
    // Each block handles one (b, t, h) position
    int linear_idx = blockIdx.x;
    int h = linear_idx % H;
    int tmp = linear_idx / H;
    int t = tmp % T;
    int b = tmp / T;
    
    int tid = threadIdx.x;
    
    // Shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* x_shared = shared_mem;           // [D]
    float* y_shared = shared_mem + D;       // [D]
    float* z_shared = shared_mem + 2 * D;   // [D]
    
    // Pointers
    size_t input_offset = ((size_t)b * T * H * D) + ((size_t)t * H * D) + ((size_t)h * D);
    size_t proj_offset = (size_t)h * D * D;
    size_t output_offset = ((size_t)b * T * H * (D / 32)) + ((size_t)t * H * (D / 32)) + ((size_t)h * (D / 32));
    
    // Load input x into shared memory
    for (int i = tid; i < D; i += blockDim.x) {
        x_shared[i] = to_float(input[input_offset + i]);
    }
    __syncthreads();
    
    // === Layer 0: y = silu(x @ proj0) + x ===
    // Each thread computes one or more output elements
    for (int out_idx = tid; out_idx < D; out_idx += blockDim.x) {
        float acc = 0.0f;
        // Matrix-vector multiply: dot product of x with proj0[h, :, out_idx]
        for (int k = 0; k < D; k++) {
            float proj_val = to_float(proj0[proj_offset + k * D + out_idx]);
            acc += x_shared[k] * proj_val;
        }
        // SiLU activation + residual
        acc = silu(acc) + x_shared[out_idx];
        y_shared[out_idx] = acc;
    }
    __syncthreads();
    
    // === Layer 1: z = (y @ proj1) + y ===
    for (int out_idx = tid; out_idx < D; out_idx += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D; k++) {
            float proj_val = to_float(proj1[proj_offset + k * D + out_idx]);
            acc += y_shared[k] * proj_val;
        }
        // Residual (no SiLU for last layer)
        acc = acc + y_shared[out_idx];
        z_shared[out_idx] = acc;
    }
    __syncthreads();
    
    // === Packbits: z > 0 → packed int32 ===
    // Each warp packs 32 consecutive bits
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    int num_packs = D / WARP_SIZE;  // D/32 int32 outputs
    
    for (int pack_idx = warp_id; pack_idx < num_packs; pack_idx += num_warps) {
        int bit_idx = pack_idx * WARP_SIZE + lane_id;
        bool bit = (z_shared[bit_idx] > 0.0f);
        uint32_t packed = __ballot_sync(0xFFFFFFFF, bit);
        
        if (lane_id == 0) {
            output[output_offset + pack_idx] = static_cast<int32_t>(packed);
        }
    }
}

} // namespace

/**
 * Python interface: hash_packbits
 * 
 * Args:
 *   input: [B, T, H, D] bf16 tensor
 *   proj0: [H, D, D] bf16 tensor (first layer weights)
 *   proj1: [H, D, D] bf16 tensor (second layer weights)
 * 
 * Returns:
 *   output: [B, T, H, D/32] int32 tensor
 */
torch::Tensor hash_packbits(
    torch::Tensor input,
    torch::Tensor proj0,
    torch::Tensor proj1
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(proj0.is_cuda(), "proj0 must be on CUDA");
    TORCH_CHECK(proj1.is_cuda(), "proj1 must be on CUDA");
    
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(proj0.scalar_type() == torch::kBFloat16, "proj0 must be bfloat16");
    TORCH_CHECK(proj1.scalar_type() == torch::kBFloat16, "proj1 must be bfloat16");
    
    TORCH_CHECK(input.dim() == 4, "input must be 4D [B, T, H, D]");
    TORCH_CHECK(proj0.dim() == 3, "proj0 must be 3D [H, D, D]");
    TORCH_CHECK(proj1.dim() == 3, "proj1 must be 3D [H, D, D]");
    
    input = input.contiguous();
    proj0 = proj0.contiguous();
    proj1 = proj1.contiguous();
    
    int B = input.size(0);
    int T = input.size(1);
    int H = input.size(2);
    int D = input.size(3);
    
    TORCH_CHECK(D % 32 == 0, "D must be divisible by 32");
    TORCH_CHECK(proj0.size(0) == H && proj0.size(1) == D && proj0.size(2) == D,
                "proj0 shape mismatch");
    TORCH_CHECK(proj1.size(0) == H && proj1.size(1) == D && proj1.size(2) == D,
                "proj1 shape mismatch");
    
    torch::Device device = input.device();
    
    // Output: [B, T, H, D/32]
    torch::Tensor output = torch::empty(
        {B, T, H, D / 32},
        torch::dtype(torch::kInt32).device(device)
    );
    
    // Launch kernel
    int num_blocks = B * T * H;
    int threads_per_block = THREADS_PER_HEAD;
    size_t shared_mem_size = 3 * D * sizeof(float);  // x, y, z
    
    hash_packbits_kernel<<<num_blocks, threads_per_block, shared_mem_size, 
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(proj0.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(proj1.data_ptr<at::BFloat16>()),
        output.data_ptr<int32_t>(),
        B, T, H, D
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hash_packbits", &hash_packbits, 
          "Fused hash + packbits kernel",
          py::arg("input"), py::arg("proj0"), py::arg("proj1"));
}

