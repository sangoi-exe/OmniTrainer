# Arquivo: setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Verifica se CUDA está disponível
if not torch.cuda.is_available():
    raise RuntimeError("CUDA não está disponível. Esta extensão requer CUDA.")

# Configurações de compilação otimizadas
def get_cuda_arch():
    """Detecta automaticamente a arquitetura CUDA do sistema"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            arch = result.stdout.strip().split('\n')[0].replace('.', '')
            return f'-gencode=arch=compute_{arch},code=sm_{arch}'
    except:
        pass
    return '-gencode=arch=compute_75,code=sm_75'  # Fallback para RTX 20xx/30xx

# Flags de compilação otimizadas
cuda_arch = get_cuda_arch()
extra_nvcc_flags = [
    '-O3',                          # Otimização máxima
    '-std=c++17',                   # C++17 para features modernas
    '--use_fast_math',              # Matemática rápida
    '--restrict',                   # Otimização de ponteiros
    '-Xptxas=-v',                   # Verbose PTX para debugging
    cuda_arch,                      # Arquitetura específica
		'-maxrregcount=64',
    '-DCUDA_HAS_FP16=1',           # Suporte FP16
    '--expt-relaxed-constexpr',     # Constexpr relaxado
    '--expt-extended-lambda',       # Lambda estendido
]

extra_cxx_flags = [
    '/O2' if os.name == 'nt' else '-O3',  # Windows vs Linux
    '/std:c++17' if os.name == 'nt' else '-std=c++17',
]

setup(
    name='prodigy_cuda_ext',
    version='1.1.0',
    description='CUDA Extensions otimizadas para One Trainer Plus',
    author='Seu Nome',
    python_requires='>=3.8',
    packages=find_packages(),
    ext_modules=[
        # Extensão existente bf16_stochastic
        CUDAExtension(
            name='bf16_stochastic_cuda_custom',
            sources=[
                'stochastic_copy.cpp',
                'stochastic_copy_cuda.cu',
            ],
            extra_compile_args={
                'cxx': extra_cxx_flags,
                'nvcc': extra_nvcc_flags
            }
        ),
        
        # Nova extensão Prodigy otimizada (arquivo único)
        CUDAExtension(
            name='prodigy_stats_cuda_custom',
            sources=[
                'prodigy_stats_binding.cu',  # Arquivo único otimizado
            ],
            extra_compile_args={
                'cxx': extra_cxx_flags,
                'nvcc': extra_nvcc_flags + [
                    '-DDEBUG_KERNEL_LAUNCH',    # Ativa logging (opcional)
                    # '-DDEBUG_KERNEL_SYNC',    # Ativa sync debug (descomente se necessário)
                ]
            },
            include_dirs=[
                # Adicione diretórios de include se necessário
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(
            use_ninja=True,     # Ninja para compilação mais rápida
            no_python_abi_suffix=True
        )
    },
    install_requires=[
        'torch>=1.12.0',
        'pybind11>=2.10.0',
    ],
    zip_safe=False,
)
