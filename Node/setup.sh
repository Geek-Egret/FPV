#!/bin/bash

echo "============== Setup Options =============="
echo "1.compile&&install(y/n):"
read -p "   [0]all node? " compile_install_all
read -p "   [1]orbbec_camera node? " compile_install_orbbec_camera
read -p "   [2]orb_slam3 node? " compile_install_orb_slam3
read -p "   [3]cuvslam node? " compile_install_cuvslam
read -p "   [4]ego_planner node? " compile_install_ego_planner
read -p "   [5]px4_bridge node? " compile_install_px4_bridge

echo "2.run(y/n):"
read -p "   [0]orbbec_camera node? " run_orbbec_camera
if [[ "$run_orbbec_camera" == "y" ]]; then
    run_orb_slam3="n"
else
    read -p "   [1]orb_slam3 node? " run_orb_slam3
fi
if [[ "$run_orbbec_camera" == "y" ]] || [[ "$run_orb_slam3" == "y" ]]; then
    run_cuvslam="n"
else
    read -p "   [2]cuvslam node? " run_cuvslam
fi
if [[ "$run_orbbec_camera" == "y" ]] || [[ "$run_orb_slam3" == "y" ]] || [[ "$run_cuvslam" == "y" ]]; then
    run_ego_planner="n"
else
    read -p "   [3]ego_planner node? " run_ego_planner
fi
if [[ "$run_orbbec_camera" == "y" ]] || [[ "$run_orb_slam3" == "y" ]] || [[ "$run_cuvslam" == "y" ]] || [[ "$run_ego_planner" == "y" ]]; then
    run_px4_bridge="n"
else
    read -p "   [4]px4_bridge node? " run_px4_bridge
fi
if [[ "$run_orbbec_camera" == "y" ]] || [[ "$run_orb_slam3" == "y" ]] || [[ "$run_cuvslam" == "y" ]] || [[ "$run_ego_planner" == "y" ]] || [[ "$run_px4_bridge" == "y" ]]; then
    run_mavros="n"
else
    read -p "   [5]mavros node? " run_mavros
fi

if [[ "$run_mavros" == "y" ]]; then
    read -p "      [0]QGC ip: " QGC_ip
    read -p "      [1]QGC port: " QGC_port
fi

if [[ "$run_ego_planner" == "y" ]]; then
    read -p "       [0]has vins_estimator.py launched? " vins_estimator_state
    if [[ "$vins_estimator_state" == "y" ]]; then
        read -p "       [1]slam kind(orbslam/cuvslam)? " slam_kind
    fi
fi

# ============================================================
# Compile & Install
# ============================================================

# ============================================================
# Compile & Install
# ============================================================
if [[ "$compile_install_all" == "y" ]]; then
    compile_install_orbbec_camera="y"
    compile_install_orb_slam3="y"
    compile_install_cuvslam="y"
    compile_install_ego_planner="y"
    compile_install_px4_bridge="y"
fi

build_node() {
    echo "=== Building $1 ==="
    cd $1
    rm -rf build install log
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
    cd ../
}

if [[ "$compile_install_orbbec_camera" == "y" ]]; then
    echo "============== compile orbbec_camera =============="
    build_node OrbbecCamera
fi

if [[ "$compile_install_orb_slam3" == "y" ]]; then
    echo "============== compile orb_slam3 =============="
    build_node ORB_Slam3
fi

if [[ "$compile_install_cuvslam" == "y" ]]; then
    echo "============== compile cuvslam =============="
    build_node cuVSLAM
fi

if [[ "$compile_install_ego_planner" == "y" ]]; then
    echo "============== compile ego_planner =============="
    sudo apt install ros-humble-pcl-ros
    build_node EGO_Planner
    source EGO_Planner/install/setup.bash
    build_node EGO_Planner
fi

if [[ "$compile_install_px4_bridge" == "y" ]]; then
    echo "============== compile px4_bridge =============="
    source EGO_Planner/install/setup.bash
    build_node PX4_Bridge
fi

# ============================================================
# Run
# ============================================================
if [[ "$run_orbbec_camera" == "y" ]]; then
    echo "============== run orbbec_camera =============="
    source OrbbecCamera/install/setup.bash
    ros2 launch orbbec_camera orbbec_camera.launch.py
    sleep 2
fi

if [[ "$run_orb_slam3" == "y" ]]; then
    echo "============== run orb_slam3 =============="
    source ORB_Slam3/install/setup.bash
    ros2 launch orb_slam3 orb_slam3.launch.py
    sleep 2
fi

if [[ "$run_cuvslam" == "y" ]]; then
    echo "============== run cuvslam =============="
    source cuVSLAM/install/setup.bash
    ros2 launch cuvslam cuvslam.launch.py
    sleep 2
fi

if [[ "$run_ego_planner" == "y" ]]; then
    echo "============== run ego_planner =============="
    source EGO_Planner/install/setup.bash
    if [[ "$vins_estimator_state" == "y" ]]; then
        if [[ "$slam_kind" == "orbslam" ]]; then
            ros2 launch ego_planner advanced_param_orbslam3.launch.py
        fi
        if [[ "$slam_kind" == "cuvslam" ]]; then
            ros2 launch ego_planner advanced_param_cuvslam.launch.py
        fi
    fi
fi

if [[ "$run_px4_bridge" == "y" ]]; then
    echo "============== run px4_bridge =============="
    source PX4_Bridge/install/setup.bash
    source EGO_Planner/install/setup.bash
    ros2 launch px4_bridge px4_bridge.launch.py
fi

if [[ "$run_mavros" == "y" ]]; then
    echo "============== run mavros =============="
    ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:921600 gcs_url:="udp://@$QGC_ip:$QGC_port"
fi

wait
