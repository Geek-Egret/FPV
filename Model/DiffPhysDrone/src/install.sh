#!/bin/bash

set -e

echo "=== Simple Installation Script ==="

# 设置CUDA路径
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 安装必要的包
echo "Installing build dependencies..."
pip install --upgrade pip setuptools wheel
pip install ninja

# 清理
echo "Cleaning..."
rm -rf build/ dist/ *.egg-info/

# 编译
echo "Building..."
python setup.py build

# 安装
echo "Installing..."
python setup.py install

# 测试
echo "Testing..."
python -c "import quadsim_cuda; print('Successfully imported quadsim_cuda')"

echo "Done!"