#!/bin/bash

read -p "[0]SLAM kind(orbslam/cuvslam): " SLAM_kind
read -p "[1]QGC ip: " QGC_ip
read -p "[2]QGC port: " QGC_port

SESSION="ros2_dev"

# 如果会话已存在则直接attach，否则新建
tmux has-session -t $SESSION 2>/dev/null
if [ $? -eq 0 ]; then
    tmux attach -t $SESSION
    exit 0
fi

# 创建主窗口
tmux new-session -d -s $SESSION -n "core"

# ---- 布局规划 ----
# 水平切分（上下）
tmux split-window -v
# 上窗口上下切分
tmux select-pane -t 0
tmux split-window -v
# 最上窗口左右切分
tmux select-pane -t 0
tmux split-window -h
# 第二上窗口左右切分
tmux select-pane -t 2
tmux split-window -h
# 最下窗口左右切分
tmux select-pane -t 4
tmux split-window -h
# 左下窗口上下切分
tmux select-pane -t 4
tmux split-window -v

# ---- 窗格分配 ----
# 左上 (pane 0): 相机节点
tmux send-keys -t 0 "source OrbbecCamera/install/setup.bash && ros2 launch orbbec_camera orbbec_camera.launch.py" C-m
# 右上 (pane 1): slam节点
if [[ "$SLAM_kind" == "orbslam" ]]; then
	tmux send-keys -t 1 "source ORB_Slam3/install/setup.bash && ros2 launch orb_slam3 orb_slam3.launch.py" C-m
fi
if [[ "$SLAM_kind" == "cuvslam" ]]; then
	tmux send-keys -t 1 "source cuVSLAM/install/setup.bash && ros2 launch cuvslam cuvslam.launch.py" C-m
fi
# 左中上 (pane 2): egoplanner节点
if [[ "$SLAM_kind" == "orbslam" ]]; then
	tmux send-keys -t 2 "source EGO_Planner/install/setup.bash && ros2 launch ego_planner advanced_param_orbslam3.launch.py" C-m	
fi
if [[ "$SLAM_kind" == "cuvslam" ]]; then
	tmux send-keys -t 2 "source EGO_Planner/install/setup.bash && ros2 launch ego_planner advanced_param_cuvslam.launch.py" C-m
fi
# 右中 (pane3): px4 bridge节点
tmux send-keys -t 3 "source PX4_Bridge/install/setup.bash && source EGO_Planner/install/setup.bash && ros2 launch px4_bridge px4_bridge.launch.py" C-m
# 左中下 (pane4): mavros节点
tmux send-keys -t 4 "ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:921600 gcs_url:=\"udp://@${QGC_ip}:${QGC_port}\"" C-m
# 右下 (pane6): JTOP
tmux send-keys -t 6 "jtop" C-m

# 回到主窗口
tmux select-window -t $SESSION:core
tmux select-pane -t 5

# 附加到会话
tmux attach -t $SESSION
