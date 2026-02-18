# 节点介绍
# 多节点启动
- 在宿主机终端输入 `sudo docker exec -it [容器名或ID] bash` 启动分离的 DOCKER容器终端
- 在不同终端运行不同节点
# 安装依赖
- ROS节点依赖
    ```
    sudo apt install -y \
        ros-humble-ament-cmake* \
        ros-humble-rclcpp \
        ros-humble-std-msgs \
        ros-humble-builtin-interfaces \
        ros-humble-geometry-msgs \
        ros-humble-sensor-msgs \
        ros-humble-tf2-ros \
        ros-humble-tf2 \
        ros-humble-vision-opencv \
        ros-humble-cv-bridge \
        libopencv-dev \
        python3-colcon-common-extensions
    ```
- 多终端工具
    ```
    sudo apt install tmux
    ```
# 启动 orbbec_camera 节点
- 进入 `OrbbecCamera` 文件夹
- 使用 `colcon build` 编译
- 输入 `source install/setup.bash` 设置环境变量
- 使用 `ros2 run orbbec_camera orbbec_camera` 运行节点

# 启动 orb_slam3 节点
- 进入 `ORB_Slam3` 文件夹
- 使用 `colcon build` 编译
- 输入 `source install/setup.bash` 设置环境变量
- 使用 `ros2 run orb_slam3 orb_slam3` 运行节点