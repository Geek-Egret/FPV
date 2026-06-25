# FPV Jetson 工程文档

## 目录结构

```
FPV/
├── Node/
│   ├── OrbbecCamera/    # Orbbec Gemini RGBD 相机 ROS2 驱动
│   ├── ORB_Slam3/       # ORB-SLAM3 RGBD 视觉 SLAM
│   ├── cuVSLAM/         # NVIDIA cuVSLAM RGBD (GPU加速)
│   └── EGO_Planner/     # EGO-Planner 路径规划
├── Thirdparty/
│   ├── cuvslam/         # cuVSLAM 库 (CUDA 12, aarch64)
│   ├── ORB_SLAM3/       # ORB-SLAM3 库 + 词袋
│   ├── OrbbecSDK/       # Orbbec 相机 SDK
│   └── OrbbecBridge/    # Orbbec 桥接库
└── DEBUG.md             # 调试记录
```

## 启动流程

### 1. 相机

```bash
source ~/Workspace/FPV/Node/OrbbecCamera/install/setup.bash
ros2 launch orbbec_camera orbbec_camera.launch.py
```

发布话题：`/rgb/image_raw`, `/depth/image_raw`, `/cloud_point`, `/rgb/camera_info`, `/depth/camera_info`

TF：`base_link -> camera_link -> camera_optical_frame`

### 2. SLAM（二选一）

**ORB_SLAM3：**
```bash
source ~/Workspace/FPV/Node/ORB_Slam3/install/setup.bash
ros2 launch orb_slam3 orb_slam3.launch.py
```

发布话题：`orb_slam3/odom`, `orb_slam3/pose`, `orb_slam3/trajectory`

TF：`odom -> base_link`, `world -> odom`

**cuVSLAM：**
```bash
source ~/Workspace/FPV/Node/cuVSLAM/install/setup.bash
ros2 launch cuvslam cuvslam.launch.py
```

发布话题：`cuvslam/odom`, `cuvslam/pose`, `cuvslam/trajectory`

### 3. EGO-Planner

```bash
source ~/Workspace/FPV/Node/EGO_Planner/install/setup.bash
# cuVSLAM 版本
ros2 launch /home/jetson/Workspace/FPV/Node/EGO_Planner/self_launch/advanced_param_cuvslam.launch.py
# ORB_SLAM3 版本
ros2 launch /home/jetson/Workspace/FPV/Node/EGO_Planner/self_launch/advanced_param_orbslam3.launch.py
```

### 4. 发送目标

```bash
ros2 topic pub /move_base_simple/goal geometry_msgs/msg/PoseStamped \
  "{pose: {position: {x: 5.0, y: 0.0, z: 1.0}}}" -1
```

注意：z 值被代码强制设为 1.0，填什么都无效。z < -0.1 的会被忽略。

---

## 坐标系 TF 链

```
world ──(静态恒等)──→ odom ──(SLAM动态)──→ base_link ──(可调)──→ camera_link ──(固定)──→ camera_optical_frame
```

| TF | 来源 | 说明 |
|----|------|------|
| `world -> odom` | ORB_SLAM3 | 静态恒等变换 |
| `odom -> base_link` | ORB_SLAM3 / cuVSLAM | 动态位姿 |
| `base_link -> camera_link` | OrbbecCamera | 可调参数 `tf_translation`, `tf_rotation_rpy` |
| `camera_link -> camera_optical_frame` | OrbbecCamera | 固定 RPY(-π/2, π/2, 0) |

---

## EGO-Planner 关键参数

### grid_map pose_type

| pose_type | 含义 |
|-----------|------|
| **0** | **不启动深度融合**。栅格地图仅由 `cloudCallback`（点云）填充，不使用深度图 |
| 1 (POSE_STAMPED) | 同步深度图 + PoseStamped，用 `depthPoseCallback` |
| 2 (ODOMETRY) | 同步深度图 + Odometry，用 `depthOdomCallback`，需要 `cam2body_`（`/vins_estimator/extrinsic`） |

**当前配置：pose_type=0**（仅点云，不需要深度图和外参）

### flight_type（目标模式）

| flight_type | 模式 | 触发方式 |
|-------------|------|----------|
| 1 (MANUAL_TARGET) | 手动目标 | 订阅 `/move_base_simple/goal` |
| 2 (PRESET_TARGET) | 预设航点 | launch 参数 + `/traj_start_trigger` |

### 核心栅格参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `grid_map/resolution` | 0.1 | 体素分辨率（米） |
| `grid_map/map_size_x` | 50.0 | X方向地图大小（米） |
| `grid_map/map_size_y` | 50.0 | Y方向地图大小（米） |
| `grid_map/map_size_z` | 5.0 | Z方向地图大小（米） |
| `grid_map/obstacles_inflation` | 0.099 | 障碍物膨胀半径（米） |
| `grid_map/ground_height` | -0.01 | 地面高度（米） |
| `grid_map/virtual_ceil_height` | 2.9 | 虚拟天花板高度（米） |

### 轨迹优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `manager/max_vel` | 2.0 | 最大速度（m/s） |
| `manager/max_acc` | 3.0 | 最大加速度（m/s²） |
| `manager/planning_horizon` | 7.5 | 规划视野（米） |
| `optimization/dist0` | 0.5 | B-spline 安全距离（米） |
| `optimization/lambda_collision` | 0.5 | 碰撞代价权重 |

---

## ORB_SLAM3 关键参数

### 传感器类型

| sensor_type | 含义 |
|-------------|------|
| MONOCULAR | 单目 |
| STEREO | 双目 |
| **RGBD** | **RGBD（当前使用）** |
| IMU_MONOCULAR | 单目+IMU |
| IMU_STEREO | 双目+IMU |
| IMU_RGBD | RGBD+IMU |

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocabulary_path` | Thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt | 词袋文件路径 |
| `settings_path` | ORB_Slam3/setting/orbbec_gemini.yaml | ORB参数文件路径 |
| `sensor_type` | RGBD | 传感器类型 |

---

## cuVSLAM 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `camera.width/height` | 640/480 | 图像分辨率 |
| `camera.fx/fy/cx/cy` | 455.483/... | 相机内参 |
| `camera.k1/k2/p1/p2/k3` | Brown畸变参数 | 全0=无畸变 |
| `depth.scale_factor` | 1000.0 | 深度值除数（mm转m） |
| `num_desired_tracks` | 400 | 期望特征点数 |

---

## OrbbecCamera 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `camera_name` | camera_link | 相机坐标系名 |
| `parent_frame` | base_link | 父坐标系名 |
| `camera.rgb_width/height` | 640/480 | RGB分辨率 |
| `camera.rgb_fps` | 60 | RGB帧率 |
| `camera.depth_width/height` | 640/400 | 深度分辨率 |
| `camera.depth_fps` | 60 | 深度帧率 |
| `tf_translation` | [0.00345, 0.0, 0.0038] | 相机安装偏移（米） |
| `tf_rotation_rpy` | [0.0, π/2, -π/2] | 相机安装旋转 |
| `timer_interval_ms` | 16 | 采集周期（~60Hz） |
| `pointcloud_min_distance` | 1e-6 | 点云最小距离过滤 |

---

## 源码修改记录

| 文件 | 改动 |
|------|------|
| `EGO_Planner/src/.../grid_map.cpp:274` | 新增 `camera_pos` 局部变量拷贝，修复多线程竞态 |
| `ORB_Slam3/src/orb_slam3.cpp` | 删除点云变换(PCL/cloud_point_tans)，SyncPolicy改为2消息；跟踪时间戳改为图像头 |
| `OrbbecCamera/src/orbbec_camera.cpp` | 发布 base_link→camera_optical_frame 两个静态TF；参数全部ROS2化 |
| `cuVSLAM/src/cuvslam.cpp` | 参数全部ROS2化 |

## vins_estimator/extrinsic

ego_planner 的 grid_map 需要相机外参 `camera_optical → base_link`，通过 `/vins_estimator/extrinsic` 话题（Odometry 格式）接收。

原版由 VINS-Fusion 发布。本工程中，由于使用 pose_type=0 不启用深度融合，`cam2body_` 不再需要，但 self_launch 中仍保留了 `extrinsic_publisher.py`（从TF树读外参并持续发布）以备将来使用 pose_type=2。

---

## 编译命令

```bash
# ORB_Slam3
cd ~/Workspace/FPV/Node/ORB_Slam3
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# cuVSLAM（需要 CUDA）
cd ~/Workspace/FPV/Node/cuVSLAM
source /opt/ros/humble/setup.bash
export PATH=/usr/local/cuda/bin:$PATH
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# OrbbecCamera
cd ~/Workspace/FPV/Node/OrbbecCamera
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# EGO_Planner
cd ~/Workspace/FPV/Node/EGO_Planner
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```


## PX4_Bridge 桥接节点

| 话题 | 类型 | 方向 |
|------|------|------|
| `/mavros/vision_pose/pose` | PoseStamped | SLAM里程计→PX4 EKF2 |
| `/mavros/setpoint_position/local` | PoseStamped | EGO-Planner指令→PX4 Offboard |

PX4参数: EKF2_AID_MASK=24
