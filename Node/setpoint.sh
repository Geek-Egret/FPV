#!/bin/bash
# 悬停脚本：起飞前先发目标点让 PX4 收到 setpoint，再切 offboard
# 用法: bash takeoff.sh [x] [y] [z]
# 默认: (0, 0, 1)

X=${1:-0.0}
Y=${2:-0.0}
Z=${3:-1.0}

echo "Publishing goal at ($X, $Y, $Z)..."
ros2 topic pub /move_base_simple/goal geometry_msgs/msg/PoseStamped   "{pose: {position: {x: $X, y: $Y, z: $Z}}}" -1
echo "Done. Now switch to offboard mode."
