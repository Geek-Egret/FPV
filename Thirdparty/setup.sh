#! /bin/bash

echo "============== Setup Options =============="
echo    "1.download pack(y/n):"
read -p "   [0]all? " download_all
if [[ "$download_all" == "n" ]]; then
    read -p "   [1]orb-slam3? " download_orb_slam3
    read -p "   [2]TUM dataset? " download_tum_dataset
    read -p "   [3]orbbec SDK? " download_orbbec_sdk
    if [[ "$download_orbbec_sdk" == "y" ]]; then
        read -p "       [0]platform?(x86_64/aarch64): " orbbec_sdk_platform
    fi
    read -p "   [4]Micro-XRCE-DDS-Agent? " download_micro_xrce_dds_agent
fi

echo    "2.unpack(y/n):"
read -p "   [0]all? " unpack_all
if [[ "$unpack_all" == "n" ]]; then
    read -p "   [1]orb-slam3? " unpack_orb_slam3
    read -p "   [2]orbbec SDK? " unpack_orbbec_sdk
    read -p "   [2]Micro-XRCE-DDS-Agent? " unpack_micro_xrce_dds_agent
fi
echo    "4.compile&&install(y/n):"
read -p "   [0]all? " compile_install_all
if [[ "$compile_install_all" == "n" ]]; then
    read -p "   [1]orb-slam3? " compile_install_orb_slam3
    if [[ "$compile_install_orb_slam3" == "y" ]]; then
        read -p "       [0]compile jobs num? " orb_slam3_jobs_num
    fi
    read -p "   [2]orbbec-bridge? " compile_install_orbbec_bridge
    if [[ "$compile_install_orbbec_bridge" == "y" ]]; then
        read -p "       [0]compile jobs num? " orbbec_bridge_jobs_num
    fi
    read -p "   [3]Micro-XRCE-DDS-Agent? " compile_install_micro_xrce_dds_agent
    if [[ "$compile_install_micro_xrce_dds_agent" == "y" ]]; then
        read -p "       [0]compile jobs num? " micro_xrce_dds_agent_jobs_num
    fi
fi
if  [[ "$compile_install_all" == "y" ]]; then
    read -p "       [1]compile jobs num? " all_jobs_num
    orb_slam3_jobs_num=$all_jobs_num
    orbbec_bridge_jobs_num=$all_jobs_num
    micro_xrce_dds_agent_jobs_num=$all_jobs_num
fi

#=====================================================================#

if  [[ "$download_all" == "y" ]] || 
    [[ "$download_orb_slam3" == "y" ]] || 
    [[ "$download_tum_dataset" == "y" ]] ||
    [[ "$download_orbbec_sdk" == "y" ]] ||
    [[ "$download_micro_xrce_dds_agent" == "y" ]]; then
    sudo apt install wget2 -y
    if [ -d "Pack" ]; then
        cd Pack
    else
        mkdir Pack
        cd Pack
    fi
    if  [[ "$download_all" == "y" ]] || 
        [[ "$download_orb_slam3" == "y" ]]; then
        echo "============== Download ORB_SLAM3.tar =============="
        if [ -f "ORB_SLAM3.tar" ]; then
            echo "ORB_SLAM3.tar has existed,skip"
        else
            wget2 -O ORB_SLAM3.tar https://github.com/UZ-SLAMLab/ORB_SLAM3/archive/refs/tags/v1.0-release.tar.gz
        fi
    fi
    if  [[ "$download_all" == "y" ]] || 
        [[ "$download_tum_dataset" == "y" ]]; then
        echo "============== Download dataset-room4_512_16.tar =============="
        read -p "this pack is so large that it will take much time,do u want to continue?(true/false)" continue
        if $continue; then
            if [ -f "dataset-room4_512_16.tar" ]; then
                echo "dataset-room4_512_16.tar has existed,skip"
            else
                wget2 -O dataset-room4_512_16.tar http://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room4_512_16.tar
            fi
        fi
    fi
    if  [[ "$download_all" == "y" ]] || 
        [[ "$download_orbbec_sdk" == "y" ]]; then
        echo "============== Download OrbbecSDK.zip =============="
        if [ "$orbbec_sdk_platform" = "x86_64" ]; then
            if [ -f "OrbbecSDK.zip" ]; then
                echo "OrbbecSDK.zip has existed,skip"
            else
                wget2 -O OrbbecSDK.zip https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_x64_release.zip
            fi
        else
            if [ -f "OrbbecSDK.zip" ]; then
                echo "OrbbecSDK.zip has existed,skip"
            else
                wget2 -O OrbbecSDK.zip https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_arm64_release.zip
            fi
        fi
    fi
    if  [[ "$download_all" == "y" ]] || 
        [[ "$download_micro_xrce_dds_agent" == "y" ]]; then
        echo "============== Download Micro-XRCE-DDS-Agent.zip =============="
        if [ -f "Micro-XRCE-DDS-Agent.zip" ]; then
            echo "Micro-XRCE-DDS-Agent.zip has existed,skip"
        else
            wget2 -O Micro-XRCE-DDS-Agent.zip https://github.com/eProsima/Micro-XRCE-DDS-Agent/archive/refs/tags/v3.0.1.zip
        fi
    fi
    cd ..
fi

if  [[ "$unpack_all" == "y" ]] || 
    [[ "$unpack_orb_slam3" == "y" ]] || 
    [[ "$unpack_orbbec_sdk" == "y" ]] ||
    [[ "$unpack_micro_xrce_dds_agent" == "y" ]]; then
    cd Pack
    cp -r * ..
    cd ..

    if  [[ "$unpack_all" == "y" ]] || 
        [[ "$unpack_orb_slam3" == "y" ]]; then
        echo "============== Unpack ORB_SLAM3.tar =============="
        tar -xvf ORB_SLAM3.tar
        mv ORB_SLAM3-1.0-release ORB_SLAM3
        rm -r ORB_SLAM3.tar
    fi
    if  [[ "$unpack_all" == "y" ]] || 
        [[ "$unpack_orbbec_sdk" == "y" ]]; then
        echo "============== Unpack OrbbecSDK.zip =============="
        unzip OrbbecSDK.zip
        mv OrbbecSDK_C_C++* OrbbecSDK
        cp -r OrbbecSDK/OrbbecSDK_v1.10.27/SDK/* OrbbecSDK/
        cp -r OrbbecSDK/OrbbecSDK_v1.10.27/Script OrbbecSDK/
        rm -r OrbbecSDK/OrbbecSDK_v1.10.27
        rm -r OrbbecSDK.zip
    fi
    if  [[ "$unpack_all" == "y" ]] || 
        [[ "$unpack_micro_xrce_dds_agent" == "y" ]]; then
        echo "============== Unpack Micro-XRCE-DDS-Agent.zip =============="
        unzip Micro-XRCE-DDS-Agent.zip
        cd Micro-XRCE-DDS-Agent-master
        rm -r Micro-XRCE-DDS-Agent.zip
    fi
fi

if  [[ "$compile_install_all" == "y" ]] || 
    [[ "$compile_install_orb_slam3" == "y" ]] || 
    [[ "$compile_install_orbbec_bridge" == "y" ]]||
    [[ "$compile_install_micro_xrce_dds_agent" == "y" ]]; then

    if  [[ "$compile_install_all" == "y" ]] || 
        [[ "$compile_install_orb_slam3" == "y" ]]; then
        echo "============== Compile orb-slam3 =============="
        cd ORB_SLAM3
        cd Thirdparty/DBoW2
        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$orb_slam3_jobs_num
        cd ../../g2o
        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$orb_slam3_jobs_num
        cd ../../../
        cd Vocabulary
        tar -xf ORBvoc.txt.tar.gz
        cd ..
        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$orb_slam3_jobs_num
        cd ..
        sudo cp -r lib/libORB_SLAM3.so /usr/local/lib
        sudo cp -r Thirdparty/DBoW2/lib/libDBoW2.so /usr/local/lib
        sudo cp -r Thirdparty/g2o/lib/libg2o.so /usr/local/lib
        cd ..
    fi
    if  [[ "$compile_install_all" == "y" ]] || 
        [[ "$compile_install_orbbec_bridge" == "y" ]]; then
        echo "============== Compile orbbec-bridge =============="
        cd OrbbecBridge
        if [ -d "build" ]; then
            cd build
        else
            mkdir build
            cd build
        fi
        cmake ..
        make -j$orbbec_bridge_jobs_num
    fi
    if  [[ "$compile_install_all" == "y" ]] || 
        [[ "$compile_install_micro_xrce_dds_agent" == "y" ]]; then
        echo "============== Compile Micro-XRCE-DDS-Agent =============="
        cd Micro-XRCE-DDS-Agent-master
        if [ -d "build" ]; then
            cd build
        else
            mkdir build
            cd build
        fi
        cmake ..
        make -j$micro_xrce_dds_agent_jobs_num
        sudo make install
        sudo ldconfig /usr/local/lib/
    fi
fi

