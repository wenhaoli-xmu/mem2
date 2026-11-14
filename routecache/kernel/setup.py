import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

capability = torch.cuda.get_device_capability()
major, minor = capability
arch = f"sm_{major}{minor}"

cxx_args = ['-O3', '-std=c++17']
nvcc_args = [
    '-O3',
    '-std=c++17',
    f'-arch={arch}',
    '--use_fast_math',
    '--expt-relaxed-constexpr',
    '--threads=4',
    '-Xptxas', '-v',
]

setup(
    name='lsh_kernel_cuda',
    ext_modules=[
        CUDAExtension(
            'hash_packbits',
            ['hash_packbits.cu'],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            }
        ),
        CUDAExtension(
            'hamming_topk_v2',
            ['hamming_topk_v2.cu'],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            }
        ),
        CUDAExtension(
            'qhash_fused_v2',
            ['qhash_fused_v2.cu'],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            }
        ),
        CUDAExtension(
            'lru_cache_update',
            ['lru_cache_update.cu'],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            }
        ),
        CUDAExtension(
            'hamming_topk_v3',
            ['hamming_topk_v3.cu'],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            }
        ),
        CUDAExtension(
            'lru_cache_update_v2',
            ['lru_cache_update_v2.cu'],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
