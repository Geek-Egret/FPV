#! /bin/bash

echo "============== Setup Options =============="
read -p "1.set python env?(true/false): " env
read -p "2.download&&set miniconda3?(true/false): " miniconda3
read -p "3.download&&install airsim?(true/false): " airsim
read -p "4.download diff-phys-drone-cuda12?(true/false): " diff_phys_drone_cuda12

if $env; then
    echo "============== Create python3 env [fpv] =============="
    conda create -n fpv python=3.11
    conda activate fpv
    echo "============== Install pytorch2.7.0 =============="
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    echo "============== Install matplotlib =============="
    pip install matplotlib
    echo "============== Install ffmpeg =============="
    sudo apt install ffmpeg -y
fi

if $miniconda3; then
    echo "============== Download miniconda3 =============="
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    echo "============== Set miniconda3 =============="
    source ~/miniconda3/bin/activate
    conda init --all
fi

if $airsim; then
    echo "============== Download airsim =============="
    git clone https://github.com/microsoft/AirSim.git
    cd AirSim/PythonClient
    echo "============== Install airsim =============="
    python setup.py install
fi

if $diff_phys_drone_cuda12; then
    echo "============== Download DiffPhysDrone =============="
    git clone https://github.com/0Leeeezy0/DiffPhysDrone.git
fi