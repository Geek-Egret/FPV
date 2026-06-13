#! /bin/bash

# 构建 liborbbec_bridge.so
echo "### BUILD LIBRARY ###"
cd ORBBEC_BRIDGE/build
rm -r *
cmake ..
make -j2
cd ../../../

# 构建 orbbec_camera 节点
echo "### BUILD ORBBEC CAMERA NODE ###"
rm -rf build
colcon build
source install/setup.bash
ros2 run orbbec_camera orbbec_camera

