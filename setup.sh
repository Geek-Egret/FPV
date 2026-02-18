#! /bin/bash

echo "============== Setup Options =============="
read -p "1.enable jetpack?(true/false): " is_jetpack_en
read -p "2.download all pack?(true/false): " download
read -p "3.unpack?(true/false): " unpack
read -p "4.compile&&install?(true/false): " compile_and_install
read -p "5.delete?(true/false): " delete


if $is_jetpack_en; then
    echo "============== Download JETPACK =============="
    sudo apt install nvidia-jetpack

    echo "============== Add NVCC Path =============="
    echo "export PATH=/usr/local/cuda-12.6/bin:$PATH" > ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH" > ~/.bashrc
    source ~/.bashrc
fi

if $download; then
    echo "============== Download All Pack =============="
    if [ -f "Pack" ]; then
        cd Pack
    else
        mkdir Pack
        cd Pack
    fi
    echo "============== Download opencv-4.10.0.tar.gz =============="
    if [ -f "opencv-4.10.0.tar.gz" ]; then
        echo "opencv-4.10.0.tar.gz has existed,skip"
    else
        wget -O opencv-4.10.0.tar.gz https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz
    fi
    echo "============== Download opencv_contrib-4.10.0.tar.gz =============="
    if [ -f "opencv_contrib-4.10.0.tar.gz" ]; then
        echo "opencv_contrib-4.10.0.tar.gz has existed"
    else
        wget -O opencv_contrib-4.10.0.tar.gz https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.tar.gz
    fi
    echo "============== Download eigen-3.3.7.tar =============="
    if [ -f "eigen-3.3.7.tar" ]; then
        echo "eigen-3.3.7.tar has existed"
    else
        wget -O eigen-3.3.7.tar https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    fi
    echo "============== Download Pangolin-0.6.tar.gz =============="
    if [ -f "Pangolin-0.6.tar.gz" ]; then
        echo "Pangolin-0.6.tar.gz has existed"
    else
        wget -O Pangolin-0.6.tar.gz https://github.com/stevenlovegrove/Pangolin/archive/refs/tags/v0.6.tar.gz
    fi
    echo "============== Download sigslot-1.0.0.tar.gz =============="
    if [ -f "sigslot-1.0.0.tar.gz" ]; then
        echo "sigslot-1.0.0.tar.gz has existed"
    else
        wget -O sigslot-1.0.0.tar.gz https://github.com/palacaze/sigslot/archive/refs/tags/v1.0.0.tar.gz
    fi
    echo "============== Download Sophus-1.22.10.tar.gz =============="
    if [ -f "Sophus-1.22.10.tar.gz" ]; then
        echo "Sophus-1.22.10.tar.gz has existed"
    else
        wget -O Sophus-1.22.10.tar.gz https://github.com/strasdat/Sophus/archive/refs/tags/1.22.10.tar.gz
    fi
    cd ..
fi

if $unpack; then
    echo "============== Unpack all pack start =============="
    cd Pack
    cp -r * ..
    cd ..

    echo "============== Unpack OpenCV =============="
    unzip opencv-4.10.0.tar.gz
    tar -xvf opencv_contrib-4.10.0.tar.gz

    echo "============== Unpack Eigen3 =============="
    tar -xvf eigen-3.3.7.tar

    echo "============== Unpack Pangolin =============="
    tar -xvf Pangolin-0.6.tar.gz

    echo "============== Unpack Sigslot =============="
    tar -xvf sigslot-1.0.0.tar.gz

    echo "============== Unpack Sophus =============="
    tar -xvf Sophus-1.22.10.tar.gz

    echo "============== Unpack all pack done =============="
    rm -r *.zip *.tar *.gz
fi

if $compile_and_install; then
    echo "============== Compile and Install all pack =============="
    cd opencv-4.10.0

fi

if $delete; then
    echo "============== Delete all pack =============="
    sudo rm -r eigen* opencv* Pangolin* sigslot* Sophus*
fi

