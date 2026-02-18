#! /bin/bash

echo "============== Setup Options =============="
read -p "1.download all pack?(true/false): " download
read -p "2.unpack all pack?(true/false): " unpack
read -p "3.compile orb-slam3?(true/false): " orb_slam3
read -p "4.compile orbbec-bridge?(true/false): " orbbec_bridge
if $orbbec_bridge; then
    read -p "4.1.compile jobs num? " jobs_num
fi

if $download; then
    echo "============== Download All Pack =============="
    if [ -d "Pack" ]; then
        cd Pack
    else
        mkdir Pack
        cd Pack
    fi
    echo "============== Download ORB_SLAM3.tar =============="
    if [ -f "ORB_SLAM3.tar" ]; then
        echo "ORB_SLAM3.tar has existed,skip"
    else
        wget -O ORB_SLAM3.tar https://github.com/UZ-SLAMLab/ORB_SLAM3/archive/refs/tags/v1.0-release.tar.gz
    fi
    echo "============== Download dataset-room4_512_16.tar =============="
    read -p "this pack is so large that it will take much time,do u want to continue?(true/false)" continue
    if $continue; then
        if [ -f "dataset-room4_512_16.tar" ]; then
            echo "dataset-room4_512_16.tar has existed,skip"
        else
            wget -O dataset-room4_512_16.tar http://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room4_512_16.tar
        fi
    fi
    echo "============== Download OrbbecSDK.zip =============="
    read -p "1.platform?(x86_64/aarch64): " platform
    if [ "$platform" = "x86_64" ]; then
        if [ -f "OrbbecSDK.zip" ]; then
            echo "OrbbecSDK.zip has existed,skip"
        else
            wget -O OrbbecSDK.zip https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_x64_release.zip
        fi
    else
        if [ -f "OrbbecSDK.zip" ]; then
            echo "OrbbecSDK.zip has existed,skip"
        else
            wget -O OrbbecSDK.zip https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_arm64_release.zip
        fi
    fi
    cd ..
fi

if $unpack; then
    echo "============== Unpack all pack start =============="
    cd Pack
    cp -r * ..
    cd ..

    echo "============== Unpack ORB_SLAM3.tar =============="
    tar -xvf ORB_SLAM3.tar

    echo "============== Unpack OrbbecSDK.zip =============="
    unzip OrbbecSDK.zip
    source 

    echo "============== Unpack all pack done =============="
    rm -r *.tar *.zip
fi

if $orb_slam3; then
    echo "============== Unpack ORB_SLAM3.tar =============="
    cd Pack
    cp -r * ..
    cd ..
    tar -xvf ORB_SLAM3.tar
    cd ORB_SLAM3
    echo "============== Compile orb-slam3 =============="
    ./build.sh
    sudo cp -r lib/libORB_SLAM3.so /usr/local/lib
    sudo cp -r Thirdparty/DBoW2/lib/libDBoW2.so /usr/local/lib
    sudo cp -r Thirdparty/g2o/lib/libg2o.so /usr/local/lib
    cd ..

if $orbbec_bridge; then
    echo "============== Compile orbbec-bridge =============="
    cd OrbbecBridge
    if [ -d "build" ]; then
        cd build
    else
        mkdir build
        cd build
    fi
    cmake ..
    make $jobs_num

