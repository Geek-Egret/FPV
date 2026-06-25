# GE-FPV 四旋翼自主飞行平台

## 项目目录

```
FPV/
├── Node/               # ROS2 节点 (相机/SLAM/规划/桥接)
├── Thirdparty/         # 第三方库 (ORB_SLAM3/cuVSLAM/OrbbecSDK)
├── Model/              # 强化学习训练模型 (Isaac Gym)
├── Hardware/           # 硬件配置 (dtb/设备树)
├── PX4/                # PX4 飞控参数
├── Env/                # 编译环境依赖 (Eigen/Sophus/Pangolin)
├── Docker/             # Docker 构建环境
└── DEBUG.md            # 调试记录
```

---

## 硬件平台

- **飞控**: NVIDIA Jetson Orin NX + STM32G0 扩展板
- **相机**: Orbbec Gemini 335 RGBD (640×480@60Hz)
- **机体**: 250mm 轴距四旋翼
- **飞控固件**: PX4 (offboard 模式)

---

## 软件架构

```
OrbbecCamera ──RGB/Depth──→ ORB_SLAM3 / cuVSLAM ──里程计──→ EGO-Planner
     │                              │                          │
     └──点云──→ ────────────────────┘                          │
                                                              ↓
                                                         traj_server
                                                              │
                                                         /position_cmd
                                                              │
                                                         px4_bridge
                                                         ┌────┴────┐
                                                    MAVROS        MAVROS
                                                 vision_pose   setpoint
                                                         │         │
                                                         └────┬────┘
                                                              ↓
                                                            PX4
```

---

## Node 节点

### 启动顺序

```bash
cd ~/Workspace/FPV/Node
./setup.sh
```

| 顺序 | 节点 | 包名 | 功能 |
|------|------|------|------|
| 1 | OrbbecCamera | `orbbec_camera` | RGBD 相机驱动，发布 RGB/Depth/点云/CameraInfo/TF |
| 2 | ORB_Slam3 | `orb_slam3` | ORB-SLAM3 RGBD SLAM，发布 odom/pose/TF |
| 2 | cuVSLAM | `cuvslam` | NVIDIA cuVSLAM RGBD (GPU加速)，发布 odom/pose/TF |
| 3 | EGO_Planner | `ego_planner` | 路径规划，接收 /move_base_simple/goal |
| 4 | PX4_Bridge | `px4_bridge` | 转发里程计+控制指令到 MAVROS |
| 5 | MAVROS | `mavros` | PX4 通信 |

---

### 编译

```bash
# OrbbecCamera
cd Node/OrbbecCamera
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# ORB_Slam3
cd Node/ORB_Slam3
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# cuVSLAM (需要 CUDA)
cd Node/cuVSLAM
export PATH=/usr/local/cuda/bin:$PATH
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# EGO_Planner
cd Node/EGO_Planner
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# PX4_Bridge
cd Node/PX4_Bridge
source /opt/ros/humble/setup.bash
source Node/EGO_Planner/install/setup.bash
colcon build --symlink-install
```

---

## 坐标系 TF 树

```
world ──→ odom ──→ base_link ──→ camera_link ──→ camera_optical_frame
  ↑静态      ↑SLAM      ↑OrbbecCamera    ↑OrbbecCamera(固定)
 恒等       动态        可调参数          RPY(-π/2,π/2,0)
```

---

## EGO-Planner 参数参考

### flight_type

| 值 | 模式 | 触发 |
|----|------|------|
| 1 | MANUAL_TARGET | 订阅 `/move_base_simple/goal` |
| 2 | PRESET_TARGET | launch 预设航点 + `/traj_start_trigger` |

### pose_type (栅格地图)

| 值 | 含义 |
|----|------|
| 0 | 仅点云填充 (当前使用) |
| 1 | 深度图 + PoseStamped 同步 |
| 2 | 深度图 + Odometry 同步 (需要 cam2body_) |

### 核心参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `grid_map/resolution` | 0.1 | 体素分辨率 (m) |
| `grid_map/obstacles_inflation` | 0.099 | 障碍物膨胀 (m) |
| `manager/max_vel` | 2.0 | 最大速度 (m/s) |
| `manager/max_acc` | 3.0 | 最大加速度 (m/s²) |
| `manager/planning_horizon` | 7.5 | 规划视野 (m) |
| `optimization/dist0` | 0.5 | B 样条安全距离 (m) |

---

## 发送目标点

```bash
# 命令行
ros2 topic pub /move_base_simple/goal geometry_msgs/msg/PoseStamped \
  "{pose: {position: {x: 5.0, y: 0.0, z: 1.0}}}" -1

# 脚本
bash Node/setpoint.sh [x] [y] [z]
```

---

## PX4 设置

1. 飞控参数: 刷入 `PX4/GE_FPV.params`
2. 关键参数: `EKF2_AID_MASK = 24` (开启 vision position + yaw 融合)
3. 设备树: 使用 `Hardware/dtb/kernel_tegra234-p3768-0000+p3767-0005-nv-super.dtb`
4. 启动前先发悬停 setpoint，确认有数据后再切 offboard + 解锁

---

## Model 模型训练

基于 Isaac Gym 的强化学习四旋翼控制训练。训练脚本 `train.py`，推理脚本 `model.py`。最优模型保存在 `best/` 目录下。

---

## 源码修改记录

| 文件 | 改动 |
|------|------|
| `EGO_Planner/src/.../grid_map.cpp` | 新增 camera_pos 局部拷贝，修复多线程竞态 |
| `ORB_Slam3/src/orb_slam3.cpp` | 删除点云变换(PCL)，track时间戳改为图像头 |
| `OrbbecCamera/src/orbbec_camera.cpp` | 发布两个静态TF，参数全部ROS2化 |
| `cuVSLAM/src/cuvslam.cpp` | 参数全部ROS2化 |

## 环境依赖

| 组件 | 版本 | 用途 |
|------|------|------|
| ROS2 Humble | - | 通信框架 |
| CUDA | 12.6 | cuVSLAM GPU 加速 |
| OpenCV | 4.8 | 图像处理 |
| Eigen | 3.3.7 | 线性代数 |
| Sophus | 1.22.10 | 李群李代数 |
| Pangolin | 0.6 | ORB-SLAM3 可视化 |
| PCL | 1.14 | 点云处理 |
| OrbbecSDK | 1.10.27 | Orbbec 相机驱动 |
