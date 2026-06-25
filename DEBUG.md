# FPV Jetson 节点调试记录

## 项目概述

三个 ROS2 节点 + EGO-Planner 在 NVIDIA Jetson Orin 上的集成调试。

```
Node/
├── OrbbecCamera/    # Orbbec Gemini RGBD 相机驱动
├── ORB_Slam3/       # ORB-SLAM3 RGBD 视觉 SLAM
├── cuVSLAM/         # NVIDIA cuVSLAM RGBD (GPU加速)
└── EGO_Planner/     # EGO-Planner 路径规划 (ego-planner-swarm ros2_version)
```

---

## 1. 项目结构调整

**问题：** 三个节点工程都在 `Workspace/` 子目录下，多了一层。

**解决：** 删除 `Workspace/`，文件上移：

```
ORB_Slam3/Workspace/* → ORB_Slam3/*
cuVSLAM/Workspace/*   → cuVSLAM/*
OrbbecCamera/Workspace/* → OrbbecCamera/*
```

**连带修复：** CMakeLists.txt 中 Thirdparty 相对路径从 `../../../Thirdparty` 改为 `../../Thirdparty`。

---

## 2. 配置文件演进：JSON/YAML → Python Launch 参数

**ORB_SLAM3：** JSON → ROS2 declare_parameter + Python launch 脚本
- 添加 nlohmann-json3-dev 依赖
- 最终删除 JSON 支持，全部参数在 `launch/orb_slam3.launch.py` 中定义
- 保留 ORB_SLAM3 settings YAML（相机内参/ORB参数）

**cuVSLAM：** YAML → ROS2 declare_parameter + Python launch 脚本
- 删除 `config/cuvslam_rgbd.yaml`
- 所有参数在 `launch/cuvslam.launch.py` 中定义

**OrbbecCamera：** 硬编码 → ROS2 declare_parameter + Python launch 脚本
- 话题名、TF变换、定时器、相机分辨率/帧率全部参数化
- 相机内参从硬件读取后注册到参数服务器
- 发布两个静态TF：`base_link→camera_link`（可调）和 `camera_link→camera_optical_frame`（固定）

---

## 3. 坐标系 TF 链

```
world → odom → base_link → camera_link → camera_optical_frame
  ↑SLAM    ↑静态TF    ↑OrbbecCamera  ↑OrbbecCamera固定
```

**OrbbecCamera 发布：**
- `base_link → camera_link`：用户可调 (默认 RPY: 0, π/2, -π/2)
- `camera_link → camera_optical_frame`：固定 (RPY: -π/2, π/2, 0)

**SLAM 发布：**
- `odom → base_link`：动态位姿（里程计）
- `world → odom`：静态恒等或旋转变换

---

## 4. EGO-Planner 集成

### 4.1 克隆和编译

```bash
git clone -b ros2_version https://github.com/ZJU-FAST-Lab/ego-planner-swarm.git
```

**编译问题：**

| 错误 | 原因 | 解决 |
|------|------|------|
| `Bspline.idl does not exist` | colcon 并行构建 rosidl 生成冲突 | `--executor sequential` |
| `Could not find pcl_ros` | 未安装 | `sudo apt install ros-humble-pcl-ros` |

最终 20 个包全部编译成功。

### 4.2 目标点设置

EGO-Planner 支持两种模式（`fsm/flight_type`）：

| flight_type | 模式 | 触发方式 |
|-------------|------|----------|
| 1 (MANUAL_TARGET) | 手动目标 | 订阅 `/move_base_simple/goal` |
| 2 (PRESET_TARGET) | 预设航点 | launch 参数 `point0~4_x/y/z` + `/traj_start_trigger` |

**命令行发目标：**
```bash
ros2 topic pub /move_base_simple/goal geometry_msgs/msg/PoseStamped \
  "{pose: {position: {x: 5.0, y: 0.0, z: 1.0}}}" -1
```

注意：z 值被代码强制设为 1.0，填什么都无效。

### 4.3 grid_map 调试（核心问题）

#### 问题 1：栅格地图无法生成

**现象：** `Wrong target_type_ value! target_type_=0`

**原因：** launch 文件 `flight_type` 默认值为 0（无效值）

**解决：** 改为 `default=1`

---

#### 问题 2：栅格地图穿过障碍物

**现象：** 规划路径穿过墙面/障碍物

**原因：** `pose_type=0` → 深度融合未启动，栅格地图全空（未知区域 = 无障碍）

**解决：** 改为 `pose_type=2`（ODOMETRY 模式）

---

#### 问题 3：`pose_type=2` 下栅格地图时有时无

**原因：** `cam2body_` 矩阵未初始化（Eigen 垃圾值）。ego_planner 需要相机外参 `camera_optical → base_link`，发布在 `/vins_estimator/extrinsic` 话题（Odometry 格式）。原版由 VINS-Fusion 提供，我们没有。

**解决：** 编写 `extrinsic_publisher.py`，从 TF 树读取 `base_link → camera_optical_frame` 变换，转为 Odometry 发布到 `/vins_estimator/extrinsic`。

**踩坑过程：**
1. 最初在 ego_planner 的 remapping 中加了 `('/vins_estimator/extrinsic', odometry_topic)` → 每帧里程计都覆盖 `cam2body_`，位姿乱跳
2. 删掉重映射，extrinsic_publisher 发一次 → ego_planner 启动时订阅还没建好，消息丢失
3. 改为 10Hz 持续发布 → 深度帧投影仍有竞态
4. 最终：`pose_type=0`，完全不用深度图，只用点云

---

#### 问题 4：深度图投影"一半重合一半不重合"

**现象：** 栅格地图中部分区域始终贴合原始点云，部分区域时而贴合时而偏移，但偏移时点云形状保持不变（整体平移）。

**根因：** `projectDepthImage()` 中的多线程竞态。

```cpp
// grid_map.cpp 原代码：
Eigen::Matrix3d camera_r = md_.camera_r_m_;  // ✅ 拷贝到局部

for (遍历像素) {
    proj_pt = camera_r * proj_pt + md_.camera_pos_;  // ❌ 直接读共享变量
}
```

`depthOdomCallback`（消息回调线程）写 `md_.camera_pos_`，`projectDepthImage`（定时器线程）读。循环跑到一半时被写入，前一半像素用旧位姿、后一半用新位姿，导致整体偏移。

**修复（grid_map.cpp:274）：**
```cpp
Eigen::Vector3d camera_pos = md_.camera_pos_;  // 新增：拷贝到局部
proj_pt = camera_r * proj_pt + camera_pos;     // 使用局部变量
```

---

#### 问题 5：ORB_SLAM3 时间戳不一致

**现象：** 深度帧和里程计时间戳对不齐，导致 ego_planner 的消息同步不稳定。

**原因：** ORB_SLAM3 内部用 `get_clock()->now()` 做跟踪计时，但输出里程计用图像 `header.stamp`。

**修复：** 跟踪时间戳改用图像 `header.stamp`。

---

#### 问题 6：gridmap 随机出现点簇

**现象：** 栅格地图中偶尔出现孤立点簇，与原始点云形状一致但位置偏移。

**原因：** `cloudCallback` 和 `depthOdomCallback` 两套机制同时写入栅格。cloudCallback 在首帧深度到达前用 `odomCallback` 的 `camera_pos_`（无 cam2body 变换）做范围检查，导致部分点坐标偏移。

**最终方案：** `pose_type=0`，完全不使用深度融合，栅格地图仅由 `cloudCallback`（OrbbecCamera 原始点云）填充。深度图不再参与。

---

### 4.4 最终配置

**launch 文件关键参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `grid_map/pose_type` | 0 | 禁用深度融合，仅用点云 |
| `fsm/flight_type` | 1 | 手动目标模式 |
| `fsm/realworld_experiment` | False | 不等待遥控器触发 |
| `manager/drone_id` | -1 | 单机模式 |
| `odometry_topic` | cuvslam/odom 或 orb_slam3/odom | 里程计来源 |
| `grid_map/obstacles_inflation` | 0.099 | 障碍物膨胀半径（米） |
| `grid_map/resolution` | 0.1 | 栅格分辨率（米） |
| `grid_map/ground_height` | -0.01 | 地面高度 |

---

## 5. 启动流程

```bash
# 1. 相机
source ~/Workspace/FPV/Node/OrbbecCamera/install/setup.bash
ros2 launch orbbec_camera orbbec_camera.launch.py

# 2. SLAM
source ~/Workspace/FPV/Node/cuVSLAM/install/setup.bash
ros2 launch cuvslam cuvslam.launch.py

# 3. EGO-Planner
source ~/Workspace/FPV/Node/EGO_Planner/install/setup.bash
ros2 launch ego_planner advanced_param_cuvslam.launch.py

# 4. 发送目标
ros2 topic pub /move_base_simple/goal geometry_msgs/msg/PoseStamped \
  "{pose: {position: {x: 5.0, y: 0.0, z: 1.0}}}" -1
```

---

## 6. 源码修改汇总

| 文件 | 改动 |
|------|------|
| `EGO_Planner/src/planner/plan_env/src/grid_map.cpp:274` | 新增 `camera_pos` 局部变量拷贝，修复多线程竞态 |
| `EGO_Planner/src/planner/plan_manage/src/ego_replan_fsm.cpp:243` | z 坐标从硬编码 1.0 改为消息传入值 |
| `ORB_Slam3/src/orb_slam3.cpp:381` | 跟踪时间戳从 `now()` 改为图像 `header.stamp` |
| `OrbbecCamera/src/orbbec_camera.cpp` | 发布两个静态TF，参数全部 ROS2 化 |
| `cuVSLAM/src/cuvslam.cpp` | 参数全部 ROS2 化 |
| `EGO_Planner/self_launch/extrinsic_publisher.py` | 新增：TF→Odometry 外参发布器 |
| `EGO_Planner/self_launch/advanced_param_cuvslam.launch.py` | cuVSLAM 版 launch |
| `EGO_Planner/self_launch/advanced_param_orbslam3.launch.py` | ORB_SLAM3 版 launch |

---

## 7. EGO-Planner 源码修改详情

### 7.1 grid_map.cpp — 修复多线程竞态

**问题：** `projectDepthImage()` 中 `md_.camera_pos_` 被 `depthOdomCallback`（另一线程）中途改写，导致深度投影前一半像素用旧位姿、后一半用新位姿，栅格地图出现"一半贴合一半偏移"。

**文件：** `src/planner/plan_env/src/grid_map.cpp`

**修改（第 274 行新增）：**
```cpp
// 原代码：
Eigen::Matrix3d camera_r = md_.camera_r_m_;
// ...
proj_pt = camera_r * proj_pt + md_.camera_pos_;  // ❌ 直接读共享变量

// 修改后：
Eigen::Matrix3d camera_r = md_.camera_r_m_;
Eigen::Vector3d camera_pos = md_.camera_pos_;     // ✅ 拷贝到局部变量
// ...
proj_pt = camera_r * proj_pt + camera_pos;         // ✅ 使用局部变量
```

---

### 7.2 ego_replan_fsm.cpp — 修复目标点 z 值硬编码

**问题：** `waypointCallback()` 中目标点 z 坐标被强制设为 `1.0`，命令行传入的 z 值无效，无法飞到非 1m 高度。

**文件：** `src/planner/plan_manage/src/ego_replan_fsm.cpp`

**修改（第 243 行）：**
```cpp
// 原代码：
Eigen::Vector3d end_wp(msg->pose.position.x, msg->pose.position.y, 1.0);

// 修改后：
Eigen::Vector3d end_wp(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
```

---

### 7.3 planner_manager.cpp — A* 搜索池大小（未修改，仅供参考）

**位置：** `src/planner/plan_manage/src/planner_manager.cpp:43`

```cpp
// 当前值（未修改）：
bspline_optimizer_->a_star_->initGridMap(grid_map_, Eigen::Vector3i(100, 100, 100));
// 100×100×0.1m = 10m 范围

// 如需扩大搜索范围（改大 A* 池，目标可以超过 5m）：
// bspline_optimizer_->a_star_->initGridMap(grid_map_, Eigen::Vector3i(200, 200, 200));
// 200×200×0.1m = 20m 范围
```
