#!/bin/bash

# 设置严格模式
set -e

echo "=== Building Quadsim CUDA Extension ==="

# 设置CUDA环境
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 检查CUDA
echo "CUDA version:"
nvcc --version

# 检查Python和PyTorch
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available in PyTorch:"
python -c "import torch; print(torch.cuda.is_available())"

# 清理之前的构建
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# 编译
echo "Building extension..."
python setup.py build_ext --inplace

# 安装
echo "Installing..."
pip install -e .

# 测试
echo "Running tests..."
python test.py

echo "=== Build completed successfully ==="