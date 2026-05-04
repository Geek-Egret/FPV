#! /bin/bash

echo "============== Setup Options =============="
echo    "1.compile&&install(y/n):"
read -p "   [0]all node? " compile_install_all
if [[ "$compile_install_all" == "n" ]]; then
    read -p "   [1]orbbec_camera node? " compile_install_orbbec_camera
    read -p "   [2]orb_slam3 node? " compile_install_orb_slam3
    read -p "   [3]ego_planner node? " compile_install_ego_planner
fi
echo    "2.run(y/n):"
read -p "   [0]orbbec_camera node? " run_orbbec_camera
if [[ "$run_orbbec_camera" == "n" ]]; then
    read -p "   [1]orb_slam3 node? " run_orb_slam3
    if [[ "$run_orb_slam3" == "n" ]]; then
    	read -p "   [2]ego_planner node? " run_ego_planner
    	if [[ "$run_ego_planner" == "n" ]]; then
            read -p "   [3]mavros node? " run_mavros
        fi
        if [[ "$run_ego_planner" == "y" ]]; then
            run_mavros = "n"
        fi
    fi
    if [[ "$run_orb_slam3" == "y" ]]; then
        run_ego_planner = "n"
        run_mavros = "n"
    fi
fi
if [[ "$run_orbbec_camera" == "y" ]]; then
    run_orb_slam3 = "n"
    run_ego_planner = "n"
    run_mavros = "n"
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
if  [[ "$compile_install_all" == "y" ]] ||
    [[ "$compile_install_ego_planner" == "y" ]]; then
    echo "============== Clone ego_planner Repo =============="
    cd EGO_Planner
    git clone -b ros2_version https://github.com/ZJU-FAST-Lab/ego-planner-swarm.git
    echo "============== Compile ego_planner Node =============="
    mv ego-planner-swarm Workspace
    colcon build
    cd ..

# node launch
if  [[ "$run_orbbec_camera" == "y" ]]; then
    echo "============== Run orbbec_camera Node =============="
    cd OrbbecCamera
    source install/setup.bash
    ros2 run orbbec_camera orbbec_camera
fi
if  [[ "$run_orb_slam3" == "y" ]]; then
    echo "============== Run orb_slam3 Node =============="
    cd ORB_Slam3
    source install/setup.bash
    ros2 run orb_slam3 orb_slam3
fi
if  [[ "$run_ego_planner" == "y" ]]; then
    echo "============== Run ego_planner Node =============="
    cd EGO_Planner
    source install/setup.bash
    cp -r self_launch/advanced_param.launch.py install/ego_planner/share/ego_planner/launch
    ros2 launch ego_planner advanced_param.launch.py
fi
if  [[ "$run_mavros" == "y" ]]; then
    echo "============== Run mavros Node =============="
    sudo chmod 777 /dev/ttyACM0
    ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:921600
fi
