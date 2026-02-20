#! /bin/bash

echo "============== Setup Options =============="
echo    "1.compile&&install(y/n):"
read -p "   [0]all node? " compile_install_all
if [[ "$compile_install_all" == "n" ]]; then
    read -p "   [1]orbbec_camera node? " compile_install_orbbec_camera
    read -p "   [2]orb_slam3 node? " compile_install_orb_slam3
fi
echo    "1.run(y/n):"
read -p "   [0]orbbec_camera node? " run_orbbec_camera
if [[ "$run_orbbec_camera" == "n" ]]; then
    read -p "   [1]orb_slam3 node? " run_install_orb_slam3
fi
if [[ "$run_orbbec_camera" == "y" ]]; then
    run_install_orb_slam3 = "n"
fi


if  [[ "$compile_install_all" == "y" ]] ||
    [[ "$compile_install_orbbec_camera" == "y" ]]; then
    echo "============== Compile orbbec_camera Node =============="
    cd OrbbecCamera
    colcon build
    cd ..
fi
if  [[ "$compile_install_all" == "y" ]] ||
    [[ "$compile_install_orb_slam3" == "y" ]]; then
    echo "============== Compile orb_slam3 Node =============="
    cd ORB_Slam3
    colcon build
    cd ..
fi

if  [[ "$run_orbbec_camera" == "y" ]]; then
    echo "============== Run orbbec_camera Node =============="
    cd OrbbecCamera
    source install/setup.bash
    ros2 run orbbec_camera orbbec_camera
fi
if  [[ "$run_install_orb_slam3" == "y" ]]; then
    echo "============== Run orb_slam3 Node =============="
    cd ORB_Slam3
    source install/setup.bash
    ros2 run orb_slam3 orb_slam3
fi