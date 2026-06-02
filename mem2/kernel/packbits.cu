#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>


__global__ void packbits_coalesced_kernel(
    const __nv_bfloat16* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    bool bit = false;
    if (idx < num_elements) {
        float val = __bfloat162float(input[idx]);
        bit = (val > 0.0f);
    }

    uint32_t packed = __ballot_sync(0xFFFFFFFF, bit);

    if ((threadIdx.x % 32 == 0) && (idx < num_elements)) {
        output[idx / 32] = static_cast<int32_t>(packed);
    }
}


torch::Tensor packbits(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input must be bfloat16");
    TORCH_CHECK(input.size(-1) % 32 == 0, "Last dimension must be divisible by 32");

    torch::Device device = input.device();
    
    auto input_contig = input.contiguous();
    auto in_sizes = input.sizes();
    
    std::vector<int64_t> out_shape(in_sizes.begin(), in_sizes.end());
    out_shape.back() /= 32;

    int64_t num_elements = input.numel();
    int64_t total_packs = num_elements / 32;
    
    torch::Tensor output = torch::empty({total_packs}, torch::dtype(torch::kInt32).device(device));

    int threads = 256; 
    int blocks = (num_elements + threads - 1) / threads;

    packbits_coalesced_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(input_contig.data_ptr<at::BFloat16>()),
        output.data_ptr<int32_t>(),
        num_elements
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA launch failed: ") + cudaGetErrorString(err));
    }

    return output.view(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packbits", &packbits);
}