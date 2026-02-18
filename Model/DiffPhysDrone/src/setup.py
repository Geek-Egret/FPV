import os
import sys
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# 获取CUDA架构标志
def get_cuda_arch_flags():
    """获取当前GPU的CUDA架构"""
    try:
        # 尝试获取当前GPU的架构
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            arch = result.stdout.strip().replace('.', '')
            print(f"Detected GPU architecture: sm_{arch}")
            return [f'-gencode=arch=compute_{arch},code=sm_{arch}']
    except:
        pass
    
    # 默认支持常见的架构
    print("Using default CUDA architectures")
    arches = ['70', '75', '80', '86', '89', '90']
    flags = []
    for arch in arches:
        flags.append(f'-gencode=arch=compute_{arch},code=sm_{arch}')
    return flags

# 检查CUDA路径
def check_cuda():
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(cuda_home):
        cuda_home = '/usr/local/cuda-12'
    if not os.path.exists(cuda_home):
        cuda_home = '/usr/local/cuda-11'
    
    print(f"Using CUDA from: {cuda_home}")
    return cuda_home

cuda_home = check_cuda()
os.environ['CUDA_HOME'] = cuda_home

# 编译选项
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17', '-fopenmp'],
    'nvcc': [
        '-O3',
        '-std=c++17',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-use_fast_math',
    ]
}

# 添加CUDA架构标志
extra_compile_args['nvcc'].extend(get_cuda_arch_flags())

# 检查PyTorch版本
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")

# 包含路径
include_dirs = [
    torch.utils.cpp_extension.include_paths()[0],
    os.path.join(cuda_home, 'include')
]

# 库路径
library_dirs = [
    torch.utils.cpp_extension.library_paths()[0],
    os.path.join(cuda_home, 'lib64')
]

# 设置扩展模块
ext_modules = [
    CUDAExtension(
        name='quadsim_cuda',
        sources=[
            'quadsim.cpp',
            'quadsim_kernel.cu',
            'dynamics_kernel.cu',
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['cudart', 'c10', 'torch', 'torch_cpu', 'torch_cuda'],
        extra_compile_args=extra_compile_args,
        extra_link_args=['-O3']
    )
]

# 运行setup
setup(
    name='quadsim_cuda',
    version='0.1.0',
    description='CUDA extensions for quadrotor simulation',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    python_requires='>=3.11',
    install_requires=[
        'torch>=2.0.0',
    ],
    zip_safe=False,
)