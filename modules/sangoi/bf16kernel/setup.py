from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bf16_stochastic_cuda_custom', # Nome do módulo para import no Python
    ext_modules=[
        CUDAExtension(
            name='bf16_stochastic_cuda_custom', # Deve ser o mesmo que o 'name' acima
            sources=[
                'stochastic_copy.cpp',      # Arquivo C++ com a interface Pybind11
                'stochastic_copy_cuda.cu',  # Arquivo CUDA com o kernel
            ],
            extra_compile_args={
                'cxx': [], # Remover flags -std=c++17 e -O3 para cl.exe, deixar PyTorch gerenciar
                'nvcc': ['-std=c++17', '-O3', # Manter para nvcc
                         # '--expt-relaxed-constexpr',
                         # '-gencode=arch=compute_86,code=sm_86' # O PyTorch deve detectar isso, mas pode ser explícito
                        ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)