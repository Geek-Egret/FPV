#! /bin/bash

echo "============== Setup Options =============="
read -p "1.enable jetpack?(true/false): " is_jetpack_en
read -p "2.download all pack?(true/false): " download
read -p "3.unpack?(true/false): " unpack
read -p "4.compile&&install?(true/false): " compile_and_install
if $compile_and_install; then
    read -p "   4.1.compile jobs num? " jobs_num
fi
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
    echo "============== Download All Pack start =============="
    if [ -d "Pack" ]; then
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

    echo "============== Unpack opencv-4.10.0.tar.gz =============="
    tar -xvf opencv-4.10.0.tar.gz
    echo "============== Unpack opencv_contrib-4.10.0.tar.gz =============="
    tar -xvf opencv_contrib-4.10.0.tar.gz

    echo "============== Unpack eigen-3.3.7.tar =============="
    tar -xvf eigen-3.3.7.tar

    echo "============== Unpack Pangolin-0.6.tar.gz =============="
    tar -xvf Pangolin-0.6.tar.gz

    echo "============== Unpack sigslot-1.0.0.tar.gz =============="
    tar -xvf sigslot-1.0.0.tar.gz

    echo "============== Unpack Sophus-1.22.10.tar.gz =============="
    tar -xvf Sophus-1.22.10.tar.gz

    echo "============== Unpack all pack done =============="
    rm -r *.tar *.gz
fi

if $compile_and_install; then
    echo "============== Compile and Install all pack =============="
    echo "============== Compile opencv-4.10.0 =============="
    cd opencv-4.10.0
    mkdir build
    cd build
    read -p "1.platform?(x86_64/aarch64): " platform
    read -p "2.enable cuda?(true/false): " enable_cuda
    if [ "$platform" = "x86_64" ]; then
        if $enable_cuda; then
            echo "do not support right now"
        else
            cmake -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-4.10.0 \
            -D CMAKE_CXX_STANDARD=11 \
            -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.10.0/modules \
            -D ENABLE_POPCNT=ON \
            -D ENABLE_FAST_MATH=ON \
            -D WITH_EIGEN=ON \
            -D WITH_TBB=ON \
            -D WITH_OPENMP=ON \
            -D WITH_IPP=ON \
            -D WITH_FFMPEG=ON \
            -D WITH_GSTREAMER=ON \
            -D WITH_V4L=ON \
            -D WITH_LIBV4L=ON \
            -D WITH_GTK=ON \
            -D WITH_GTK=ON \
            -D WITH_GTK_2_X=ON \
            -D BUILD_opencv_highgui=ON \
            -D BUILD_opencv_imgcodecs=ON \
            -D BUILD_opencv_videoio=ON \
            -D WITH_OPENGL=ON \
            -D WITH_WEBP=ON \
            -D WITH_JPEG=ON \
            -D WITH_PNG=ON \
            -D WITH_TIFF=ON \
            -D WITH_QT=OFF \
            -D BUILD_EXAMPLES=ON \
            -D BUILD_TESTS=OFF \
            -D BUILD_PERF_TESTS=OFF \
            -D BUILD_opencv_python3=ON \
            ..
        fi
    else
        if $enable_cuda; then
            cmake -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-4.10.0 \
            -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.10.0/modules \
            -D WITH_CUDA=ON \
            -D WITH_CUDNN=ON \
            -D OPENCV_DNN_CUDA=ON \
            -D ENABLE_FAST_MATH=ON \
            -D CUDA_FAST_MATH=ON \
            -D WITH_CUBLAS=ON \
            -D CUDA_ARCH_BIN=8.7 \
            -D BUILD_opencv_python3=ON \
            -D BUILD_opencv_cudacodec=ON \
            -D BUILD_opencv_cudaarithm=ON \
            -D BUILD_opencv_cudaimgproc=ON \
            -D WITH_GTK=ON \
            -D WITH_V4L=ON \
            -D WITH_FFMPEG=ON \
            -D WITH_GSTREAMER=ON \
            -D WITH_TBB=ON \
            -D ENABLE_NEON=ON \
            -D WITH_IPP=OFF \
            ..
        else
            echo "do not support right now"
        fi
    fi
    make -j$jobs_num
    echo "============== Install opencv-4.10.0 =============="
    sudo make install 
    sudo sh -c 'echo "/usr/local/opencv-4.10.0/lib" > /etc/ld.so.conf.d/opencv-4.10.0.conf'
    cd ../../
    echo "============== Compile Pangolin-0.6 =============="
    echo "Please read README.md and add code into [pangolin/gl/colour.h]"
    read -p "1.finsh?(true/false): " finsh
    if $finsh; then
        cd Pangolin-0.6
        mkdir build
        cd build
        sudo apt update && sudo apt install -y \
            build-essential cmake git pkg-config \
            libjpeg-dev libpng-dev libtiff-dev \
            libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
            libgtk2.0-dev \
            libtbb-dev libeigen3-dev \
            libatlas-base-dev gfortran \
            python3-dev python3-numpy \
            libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$jobs_num
        echo "============== Install Pangolin-0.6 =============="
        sudo make install
        cd ../../
    fi
    echo "============== Compile Sophus-1.22.10 =============="
    cd Sophus-1.22.10
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$jobs_num
    echo "============== Install Sophus-1.22.10 =============="
    sudo make install
    cd ../../
    echo "============== Install sigslot-1.0.0 =============="
    cd sigslot-1.0.0
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    sudo make install
    cd ../../
    echo "============== Install eigen-3.3.7 =============="
    cd eigen-3.3.7
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    sudo make install
    cd ../../
fi

if $delete; then
    echo "============== Delete all pack =============="
    sudo rm -r eigen* opencv* Pangolin* sigslot* Sophus*
fi

