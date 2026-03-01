#! /bin/bash

git clone https://github.com/PX4/px4_msgs.git
git clone https://github.com/PX4/px4_ros_com.git

cd ../
source /opt/ros/humble/setup.bash
colcon build
