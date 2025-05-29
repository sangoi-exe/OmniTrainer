# Arquivo: setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bf16_stochastic_cuda_custom',
    ext_modules=[
        CUDAExtension(
            name='bf16_stochastic_cuda_custom',
            sources=[
                'stochastic_copy.cpp',
                'stochastic_copy_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-g'],   # otimizações + símbolos de debug
                'nvcc': ['-std=c++17', '-O3']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)
