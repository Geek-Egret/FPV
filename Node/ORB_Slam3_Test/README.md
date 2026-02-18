- `colcon build`
- `source install/setup.bash`
- `ros2 run orb_slam3 orb_slam3 --ros-args- -p rgb_image_topic:="" -p rgb_info_topic:="" -p depth_image_topic:="" -p depth_info_topic:="" -p vocabulary_file_path:=""`

# 安装 udev-rules 文件
- 宿主机进入 `Thirdparty/OrbbecSDK`,以sudo运行 `Script` 下的 `install_udev_rules.sh`