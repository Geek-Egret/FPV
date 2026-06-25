/**
 * @file cuvslam.cpp
 * @brief cuVSLAM RGBD ROS2 节点 - 后端参数全可配置
 *
 * 订阅 RGB 图像和深度图像，运行 NVIDIA cuVSLAM RGBD 模式（GPU加速），
 * 发布里程计(odometry)、位姿(pose)、轨迹(trajectory)和 TF 变换。
 *
 * 所有 cuVSLAM 后端参数（Odometry::Config、Slam::Config、TrackOptions、
 * Camera 边框裁剪等）均通过 ROS2 参数系统配置，由 launch.py 传入。
 * 支持运行时动态重配（dynamic_reconfigure 兼容）。
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <chrono>
#include <fstream>

/* ============================================================
 * ROS2 核心头文件
 * ============================================================ */
#include <rclcpp/rclcpp.hpp>

/* ============================================================
 * 消息类型
 * ============================================================ */
#include <sensor_msgs/msg/image.hpp>           // 图像消息
#include <std_msgs/msg/header.hpp>             // 标准头部
#include <nav_msgs/msg/odometry.hpp>           // 里程计消息
#include <geometry_msgs/msg/pose_stamped.hpp>  // 带戳位姿
#include <geometry_msgs/msg/transform_stamped.hpp>  // 变换消息
#include <visualization_msgs/msg/marker.hpp>   // 可视化标记（轨迹线）

/* ============================================================
 * 消息同步（RGB + Depth 时间对齐）
 * ============================================================ */
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

/* ============================================================
 * TF2 坐标变换
 * ============================================================ */
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

/* ============================================================
 * cuVSLAM 后端（NVIDIA GPU 加速 SLAM 库）
 * ============================================================ */
#include <cuvslam/cuvslam2.h>    // cuVSLAM 主 API（Odometry / Slam / Rig / Camera）
#include <cuvslam/cuvslam_gpu.h> // GPU 预热（WarmUpGPU）

/* ============================================================
 * 定义消息同步策略：RGB 图像 + 深度图像的近似时间同步
 * 当两路消息时间戳差小于阈值时触发回调
 * ============================================================ */
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image> SyncPolicy;

/**
 * @brief cuVSLAM RGBD ROS2 节点类
 *
 * 工作流程：
 * 1. 声明并加载所有 ROS2 参数（launch.py 传入）
 * 2. 发布 world -> odom 静态 TF（坐标系转换）
 * 3. 从 TF 树获取 base_link -> camera_link 的变换
 * 4. 订阅 RGB + Depth 话题，时间同步后调用回调
 * 5. 初始化 cuVSLAM Odometry（RGBD 模式）+ SLAM 后端
 * 6. 每帧执行：
 *    a. Odometry::Track() — 前端 VO，计算原始位姿
 *    b. Slam::Track() — 后端 SLAM，检测闭环 + 位姿图优化
 *    c. PGO 完成后计算 VO→SLAM 修正量
 *    d. 通过插值（平移 lerp + 旋转 slerp）将修正量在 N 帧内平滑生效
 *    e. 发布插值融合后的最终位姿（odom / TF / trajectory）
 */
class CuVSLAM_RGBD : public rclcpp::Node
{
public:
    CuVSLAM_RGBD() : Node("cuvslam")
    {
        /* ---- 步骤1：声明并加载参数 ---- */
        declare_all_parameters();
        load_parameters();

        /* ---- 步骤2：发布 world -> odom 静态 TF ---- */
        publish_static_world_to_odom();

        /* ---- 步骤3：查找 base_link -> camera_link TF ---- */
        lookup_base_to_camera_tf();

        /* ---- 步骤4：创建订阅器 + 同步器 ---- */
        rgb_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, topics_.rgb);
        depth_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, topics_.depth);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(sync_queue_size_),
            *rgb_img_sub_,
            *depth_img_sub_);
        sync_->setMaxIntervalDuration(std::chrono::milliseconds(sync_max_interval_ms_));
        sync_->registerCallback(std::bind(
            &CuVSLAM_RGBD::sync_callback, this,
            std::placeholders::_1, std::placeholders::_2));

        /* ---- 步骤5：创建发布者（QoS 可靠传输） ---- */
        auto sensor_qos = rclcpp::SensorDataQoS();
        sensor_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);

        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
            publishers_.odom, sensor_qos);
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            publishers_.pose, sensor_qos);
        trajectory_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
            publishers_.trajectory, sensor_qos);
        slam_pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            publishers_.slam_pose, sensor_qos);

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        /* ---- 步骤6：初始化 cuVSLAM 后端 ---- */
        init_cuvslam();

        /* ---- 打印启动信息 ---- */
        RCLCPP_INFO(this->get_logger(), "cuVSLAM RGBD 节点启动完成");
        RCLCPP_INFO(this->get_logger(), "  RGB 话题: %s", topics_.rgb.c_str());
        RCLCPP_INFO(this->get_logger(), "  深度话题: %s", topics_.depth.c_str());
        RCLCPP_INFO(this->get_logger(), "  相机参数: %dx%d fx=%.2f fy=%.2f",
                    cam_.width, cam_.height, cam_.fx, cam_.fy);
        RCLCPP_INFO(this->get_logger(), "  深度缩放: %.1f", cam_.depth_scale);
    }

private:
    /**
     * @brief 安全访问 vector 元素，越界则返回默认值
     * @param v   double 数组
     * @param i   索引
     * @param def 越界时的默认值
     */
    static double vec_at(const std::vector<double>& v, size_t i, double def)
    {
        return i < v.size() ? v[i] : def;
    }

    static double clamp_double(double v, double lo, double hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    /* ============================================================
     * 参数声明 —— 所有可调参数通过 ROS2 参数系统从 launch.py 传入
     *
     * 【重要】参数命名规则：
     *   - 顶级参数（如 use_motion_model）保持与旧版兼容
     *   - 带命名空间的参数（如 slam.sync_mode）用于分组
     * 所有参数都有合理的默认值，不传入时自适配
     * ============================================================ */
    void declare_all_parameters()
    {
        /* ---- 订阅话题 ---- */
        this->declare_parameter<std::string>("rgb_topic", "/rgb/image_raw");
        this->declare_parameter<std::string>("depth_topic", "/depth/image_raw");

        /* ---- 坐标系名称 ---- */
        this->declare_parameter<std::string>("world_frame", "world");
        this->declare_parameter<std::string>("camera_frame", "camera_link");
        this->declare_parameter<std::string>("robot_frame", "base_link");
        this->declare_parameter<std::string>("odom_frame", "odom");

        /* ---- 发布话题名称 ---- */
        this->declare_parameter<std::string>("odom_topic", "cuvslam/odom");
        this->declare_parameter<std::string>("pose_topic", "cuvslam/pose");
        this->declare_parameter<std::string>("trajectory_topic", "cuvslam/trajectory");
        this->declare_parameter<std::string>("slam_pose_topic", "cuvslam/slam_pose");
        /* SLAM 后端优化后的位姿话题（闭环校正后） */

        /* ---- 闭环修正插值帧数 ----
         * SLAM 检测到闭环并完成 PGO 后，修正量会在 N 帧内逐渐插值生效，
         * 避免位姿突然跳变。默认 150 帧（约 5 秒 @ 30fps）。
         * 设为 0 则立即应用修正（可能有跳变）。 */
        this->declare_parameter<int>("correction_ramp_frames", 150);

        /* ---- 相机内参 ----
         * 默认值为 Orbbec Gemini FPV 标定参数
         * 畸变模型：Brown（k1/k2/k3 径向畸变, p1/p2 切向畸变）
         * 全部设为 0.0 即为 Pinhole 无畸变模型 */
        this->declare_parameter<int>("camera.width", 640);
        this->declare_parameter<int>("camera.height", 480);
        this->declare_parameter<double>("camera.fx", 455.483);
        this->declare_parameter<double>("camera.fy", 455.483);
        this->declare_parameter<double>("camera.cx", 329.67);
        this->declare_parameter<double>("camera.cy", 243.265);
        this->declare_parameter<double>("camera.k1", 0.05382);
        this->declare_parameter<double>("camera.k2", -0.0703176);
        this->declare_parameter<double>("camera.p1", -0.0000575311);
        this->declare_parameter<double>("camera.p2", -0.000674939);
        this->declare_parameter<double>("camera.k3", 0.0);

        /* ---- RGBD 深度设置 ---- */
        this->declare_parameter<double>("depth.scale_factor", 1000.0);
        /* 深度值转换因子：毫米除以1000得米，浮点米的深度为1.0 */
        this->declare_parameter<int>("depth.camera_id", 0);
        /* 深度图对应的相机ID（多相机时使用） */

        /* ---- cuVSLAM Odometry::Config 算法参数 ----
         *
         * use_gpu        : 是否启用 GPU 加速（必须为 true，cuVSLAM 核心依赖 CUDA）
         * use_motion_model: 是否使用匀速运动模型预测初始位姿（建议 true）
         * use_denoising  : 是否对图像做去噪预处理（FPV 运动模糊场景建议 false）
         * rectified_stereo: 是否使用已校正的立体相机（RGBD 模式无效）
         * enable_observations_export: 是否导出 2D-3D 匹配观测数据
         * enable_landmarks_export   : 是否导出 3D 路标点
         * enable_final_landmarks_export: 是否在结束前最终导出路标
         * async_sba      : 是否启用异步集束调整（提升大场景性能）
         * max_frame_delta_s: 两帧之间的最大时间差（超过即重置，单位秒）
         * ransac_filter  : 是否用 RANSAC 过滤外点匹配（TrackOptions）
         * kf_survivor_from_last_pct: 关键帧筛选 - 与上一关键帧的共视比例
         * kf_max_timedelta_s: 关键帧最大时间间隔（秒，超过则强制新建关键帧）
         * num_desired_tracks: 每帧期望追踪的特征点数量
         * border_*      : 图像边缘裁剪像素数（忽略边界特征） */
        this->declare_parameter<bool>("use_gpu", true);
        this->declare_parameter<bool>("use_motion_model", true);
        this->declare_parameter<bool>("use_denoising", false);
        this->declare_parameter<bool>("rectified_stereo", false);
        this->declare_parameter<bool>("enable_observations_export", true);
        this->declare_parameter<bool>("enable_landmarks_export", true);
        this->declare_parameter<bool>("enable_final_landmarks_export", false);
        this->declare_parameter<bool>("async_sba", true);
        this->declare_parameter<double>("max_frame_delta_s", 1.0);
        this->declare_parameter<bool>("ransac_filter", false);
        this->declare_parameter<double>("kf_survivor_from_last_pct", 41.0);
        this->declare_parameter<int>("kf_max_timedelta_s", 60);
        this->declare_parameter<int>("num_desired_tracks", 400);
        this->declare_parameter<int>("border_top", 0);
        this->declare_parameter<int>("border_bottom", 0);
        this->declare_parameter<int>("border_left", 0);
        this->declare_parameter<int>("border_right", 0);

        /* ---- cuVSLAM Slam::Config 参数 ----
         *
         * slam.sync_mode        : 同步模式（false=异步, true=同步）
         * slam.planar_constraints: 是否启用平面约束（地面/墙面场景）
         * slam.enable_reading_internals: 是否允许读取内部数据（路标/位姿图）
         * slam.throttling_time_ms: SLAM 节流时间（毫秒，防止 CPU 过载）
         * slam.retention_time_ms: 地图点保留时间（毫秒，超时移除）
         * slam.max_map_size     : 地图最大关键帧/路标数量
         * slam.map_cache_path   : 地图缓存路径（留空不保存） */
        this->declare_parameter<bool>("slam.sync_mode", false);
        this->declare_parameter<bool>("slam.planar_constraints", false);
        this->declare_parameter<bool>("slam.enable_reading_internals", true);
        this->declare_parameter<int>("slam.throttling_time_ms", 1000);
        this->declare_parameter<int>("slam.retention_time_ms", 5000);
        this->declare_parameter<int>("slam.max_map_size", 300);
        this->declare_parameter<std::string>("slam.map_cache_path", "");

        /* ---- 消息同步器参数 ---- */
        this->declare_parameter<int>("sync_queue_size", 10);
        this->declare_parameter<int>("sync_max_interval_ms", 100);
        /* 同步队列大小和最大时间间隔（毫秒） */

        /* ---- TF 查找参数 ---- */
        this->declare_parameter<int>("tf_retry_count", 10);
        this->declare_parameter<int>("tf_retry_interval_ms", 500);
        this->declare_parameter<std::vector<double>>("default_tf_translation", {0.2, 0.0, 0.1});
        this->declare_parameter<std::vector<double>>("default_tf_rpy", {0.0, 0.0, 0.0});
        /* 当 TF 树中没有 base_link -> camera_link 时的后备外参 */

        /* ---- 里程计协方差矩阵（6x6 对角线元素） ---- */
        this->declare_parameter<std::vector<double>>("pose_covariance",
            {0.01, 0.01, 0.01, 0.01, 0.01, 0.01});
        this->declare_parameter<std::vector<double>>("twist_covariance",
            {0.1, 0.1, 0.1, 0.1, 0.1, 0.1});

        /* ---- 轨迹可视化参数 ---- */
        this->declare_parameter<double>("trajectory_line_width", 0.05);
        this->declare_parameter<int>("trajectory_max_points", 1000);
        this->declare_parameter<std::vector<double>>("trajectory_color", {0.0, 1.0, 0.0});

        /* ---- 日志输出频率 ---- */
        this->declare_parameter<int>("log_interval_frames", 300);
        /* 每 N 帧输出一次同步时间戳日志 */
    }

    /* ============================================================
     * 参数加载 —— 从 ROS2 参数服务器读取所有参数到成员变量
     * ============================================================ */
    void load_parameters()
    {
        /* 话题 */
        topics_.rgb = this->get_parameter("rgb_topic").as_string();
        topics_.depth = this->get_parameter("depth_topic").as_string();

        /* 坐标系 */
        world_frame_ = this->get_parameter("world_frame").as_string();
        camera_frame_ = this->get_parameter("camera_frame").as_string();
        robot_frame_ = this->get_parameter("robot_frame").as_string();
        odom_frame_ = this->get_parameter("odom_frame").as_string();

        /* 发布话题 */
        publishers_.odom = this->get_parameter("odom_topic").as_string();
        publishers_.pose = this->get_parameter("pose_topic").as_string();
        publishers_.trajectory = this->get_parameter("trajectory_topic").as_string();
        publishers_.slam_pose = this->get_parameter("slam_pose_topic").as_string();
        correction_ramp_frames_ = this->get_parameter("correction_ramp_frames").as_int();

        /* 相机内参 */
        cam_.width = this->get_parameter("camera.width").as_int();
        cam_.height = this->get_parameter("camera.height").as_int();
        cam_.fx = this->get_parameter("camera.fx").as_double();
        cam_.fy = this->get_parameter("camera.fy").as_double();
        cam_.cx = this->get_parameter("camera.cx").as_double();
        cam_.cy = this->get_parameter("camera.cy").as_double();
        cam_.k1 = this->get_parameter("camera.k1").as_double();
        cam_.k2 = this->get_parameter("camera.k2").as_double();
        cam_.p1 = this->get_parameter("camera.p1").as_double();
        cam_.p2 = this->get_parameter("camera.p2").as_double();
        cam_.k3 = this->get_parameter("camera.k3").as_double();
        cam_.depth_scale = this->get_parameter("depth.scale_factor").as_double();

        depth_camera_id_ = this->get_parameter("depth.camera_id").as_int();

        /* Odometry 参数 */
        use_gpu_ = this->get_parameter("use_gpu").as_bool();
        use_motion_model_ = this->get_parameter("use_motion_model").as_bool();
        use_denoising_ = this->get_parameter("use_denoising").as_bool();
        rectified_stereo_ = this->get_parameter("rectified_stereo").as_bool();
        enable_observations_ = this->get_parameter("enable_observations_export").as_bool();
        enable_landmarks_ = this->get_parameter("enable_landmarks_export").as_bool();
        enable_final_landmarks_ = this->get_parameter("enable_final_landmarks_export").as_bool();
        async_sba_ = this->get_parameter("async_sba").as_bool();
        max_frame_delta_s_ = this->get_parameter("max_frame_delta_s").as_double();
        ransac_filter_ = this->get_parameter("ransac_filter").as_bool();
        kf_survivor_pct_ = this->get_parameter("kf_survivor_from_last_pct").as_double();
        kf_max_timedelta_s_ = this->get_parameter("kf_max_timedelta_s").as_int();
        num_desired_tracks_ = this->get_parameter("num_desired_tracks").as_int();
        border_top_ = this->get_parameter("border_top").as_int();
        border_bottom_ = this->get_parameter("border_bottom").as_int();
        border_left_ = this->get_parameter("border_left").as_int();
        border_right_ = this->get_parameter("border_right").as_int();

        /* SLAM 参数 */
        slam_sync_mode_ = this->get_parameter("slam.sync_mode").as_bool();
        slam_planar_constraints_ = this->get_parameter("slam.planar_constraints").as_bool();
        slam_enable_reading_internals_ = this->get_parameter("slam.enable_reading_internals").as_bool();
        slam_throttling_ms_ = this->get_parameter("slam.throttling_time_ms").as_int();
        slam_retention_ms_ = this->get_parameter("slam.retention_time_ms").as_int();
        slam_max_map_size_ = this->get_parameter("slam.max_map_size").as_int();
        slam_map_cache_path_ = this->get_parameter("slam.map_cache_path").as_string();

        /* 同步器 */
        sync_queue_size_ = this->get_parameter("sync_queue_size").as_int();
        sync_max_interval_ms_ = this->get_parameter("sync_max_interval_ms").as_int();

        /* TF */
        tf_retry_count_ = this->get_parameter("tf_retry_count").as_int();
        tf_retry_interval_ms_ = this->get_parameter("tf_retry_interval_ms").as_int();

        auto def_tf_trans = this->get_parameter("default_tf_translation").as_double_array();
        default_tf_x_ = vec_at(def_tf_trans, 0, 0.2);
        default_tf_y_ = vec_at(def_tf_trans, 1, 0.0);
        default_tf_z_ = vec_at(def_tf_trans, 2, 0.1);

        auto def_tf_rpy = this->get_parameter("default_tf_rpy").as_double_array();
        default_tf_roll_ = vec_at(def_tf_rpy, 0, 0.0);
        default_tf_pitch_ = vec_at(def_tf_rpy, 1, 0.0);
        default_tf_yaw_ = vec_at(def_tf_rpy, 2, 0.0);

        /* 协方差 */
        auto pose_cov = this->get_parameter("pose_covariance").as_double_array();
        cov_pos_x_ = vec_at(pose_cov, 0, 0.01);
        cov_pos_y_ = vec_at(pose_cov, 1, 0.01);
        cov_pos_z_ = vec_at(pose_cov, 2, 0.01);
        cov_rot_x_ = vec_at(pose_cov, 3, 0.01);
        cov_rot_y_ = vec_at(pose_cov, 4, 0.01);
        cov_rot_z_ = vec_at(pose_cov, 5, 0.01);

        auto twist_cov = this->get_parameter("twist_covariance").as_double_array();
        cov_twist_linear_x_ = vec_at(twist_cov, 0, 0.1);
        cov_twist_linear_y_ = vec_at(twist_cov, 1, 0.1);
        cov_twist_linear_z_ = vec_at(twist_cov, 2, 0.1);
        cov_twist_angular_x_ = vec_at(twist_cov, 3, 0.1);
        cov_twist_angular_y_ = vec_at(twist_cov, 4, 0.1);
        cov_twist_angular_z_ = vec_at(twist_cov, 5, 0.1);

        /* 轨迹 */
        trajectory_line_width_ = this->get_parameter("trajectory_line_width").as_double();
        trajectory_max_points_ = this->get_parameter("trajectory_max_points").as_int();

        auto traj_color = this->get_parameter("trajectory_color").as_double_array();
        trajectory_color_r_ = vec_at(traj_color, 0, 0.0);
        trajectory_color_g_ = vec_at(traj_color, 1, 1.0);
        trajectory_color_b_ = vec_at(traj_color, 2, 0.0);

        /* 日志 */
        log_interval_ = this->get_parameter("log_interval_frames").as_int();
    }

    /* ============================================================
     * 发布 world -> odom 静态 TF
     *
     * cuVSLAM 输出的坐标系是 OpenCV 风格（Z向前，X向右，Y向下），
     * ROS 标准坐标系是（X向前，Y向左，Z向上）。
     * 这里通过两次旋转进行坐标系转换。
     * ============================================================ */
    void publish_static_world_to_odom()
    {
        /* 第一次旋转：绕 Y 轴转 90°，使 Z 指向上方 */
        tf2::Quaternion q1, q2, final_quat;
        q1.setRPY(0.0, 0.0, 0.0);
        /* 第二次旋转：绕 X 轴转 -90°，使 X 指向前方 */
        q2.setRPY(0.0, 0.0, 0.0);
        final_quat = q2 * q1;

        static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        geometry_msgs::msg::TransformStamped static_tf;
        static_tf.header.stamp = this->now();
        static_tf.header.frame_id = world_frame_;   /* 父坐标系：world */
        static_tf.child_frame_id = odom_frame_;      /* 子坐标系：odom */
        static_tf.transform.translation.x = 0.0;
        static_tf.transform.translation.y = 0.0;
        static_tf.transform.translation.z = 0.0;
        static_tf.transform.rotation.x = final_quat.x();
        static_tf.transform.rotation.y = final_quat.y();
        static_tf.transform.rotation.z = final_quat.z();
        static_tf.transform.rotation.w = final_quat.w();
        static_tf_broadcaster_->sendTransform(static_tf);
    }

    /* ============================================================
     * 查找 base_link -> camera_link 的 TF 变换
     *
     * 从 TF 树中获取机器人基座到相机的静态变换。
     * 如果 TF 树中不存在，则使用 launch.py 传入的默认值。
     * 同时计算并缓存逆变换（camera -> base），供位姿转换使用。
     * ============================================================ */
    void lookup_base_to_camera_tf()
    {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        /* 带重试的 TF 查找（相机驱动可能还没发布 TF） */
        bool got_tf = false;
        for (int retry = 0; retry < tf_retry_count_; ++retry) {
            try {
                if (tf_buffer_->canTransform(robot_frame_, camera_frame_,
                                             tf2::TimePointZero, tf2::durationFromSec(1.0))) {
                    base_to_camera_tf_ = tf_buffer_->lookupTransform(
                        robot_frame_, camera_frame_, tf2::TimePointZero);
                    RCLCPP_INFO(this->get_logger(), "成功获取 %s -> %s 静态 TF",
                                robot_frame_.c_str(), camera_frame_.c_str());
                    got_tf = true;
                    break;
                }
            } catch (const tf2::TransformException& ex) {
                RCLCPP_WARN(this->get_logger(), "TF 查找重试 %d/%d: %s",
                            retry + 1, tf_retry_count_, ex.what());
                rclcpp::sleep_for(std::chrono::milliseconds(tf_retry_interval_ms_));
            }
        }

        /* 若查找失败，使用 launch.py 传入的默认值 */
        if (!got_tf) {
            RCLCPP_WARN(this->get_logger(), "使用默认 %s -> %s TF",
                        robot_frame_.c_str(), camera_frame_.c_str());
            base_to_camera_tf_.header.frame_id = robot_frame_;
            base_to_camera_tf_.child_frame_id = camera_frame_;
            base_to_camera_tf_.transform.translation.x = default_tf_x_;
            base_to_camera_tf_.transform.translation.y = default_tf_y_;
            base_to_camera_tf_.transform.translation.z = default_tf_z_;

            tf2::Quaternion q;
            q.setRPY(default_tf_roll_, default_tf_pitch_, default_tf_yaw_);
            base_to_camera_tf_.transform.rotation = tf2::toMsg(q);
        }

        /* 计算 camera -> base 逆变换（后续坐标转换使用） */
        tf2::fromMsg(base_to_camera_tf_.transform, camera_to_base_tf_);
        camera_to_base_tf_ = camera_to_base_tf_.inverse();
    }

    /* ============================================================
     * 初始化 cuVSLAM 后端
     *
     * 按以下步骤初始化：
     * 1. GPU 预热（创建 CUDA 运行时上下文）
     * 2. 构建相机 Rig 配置（单目 RGBD 相机）
     * 3. 创建 Odometry 实例（RGBD 模式，参数全部来自 ROS2 params）
     * 4. 创建 SLAM 实例（闭环检测 + 位姿图优化）
     * ============================================================ */
    void init_cuvslam()
    {
        /* ---- 1. GPU 预热 ----
         * WarmUpGPU() 预分配 CUDA 资源，避免首次 Track 时卡顿 */
        try {
            cuvslam::WarmUpGPU();
            RCLCPP_INFO(this->get_logger(), "GPU 预热完成，cuVSLAM 版本: %s",
                        cuvslam::GetVersion(nullptr, nullptr, nullptr).data());
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "GPU 预热失败: %s", e.what());
        }

        /* ---- 2. 构建相机 Rig ----
         * cuVSLAM 使用 Rig 描述多相机+IMU 系统。RGBD 模式只需一个相机。 */
        cuvslam::Rig rig;
        cuvslam::Camera cam;

        /* 相机分辨率 */
        cam.size = {cam_.width, cam_.height};
        /* 主点坐标（光心） */
        cam.principal = {static_cast<float>(cam_.cx), static_cast<float>(cam_.cy)};
        /* 焦距 */
        cam.focal = {static_cast<float>(cam_.fx), static_cast<float>(cam_.fy)};
        /* 相机在 Rig 坐标系中的位姿（单相机场景为单位矩阵） */
        cam.rig_from_camera.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
        cam.rig_from_camera.translation = {0.0f, 0.0f, 0.0f};
        /* 图像边缘裁剪（忽略边缘特征点） */
        cam.border_top = border_top_;
        cam.border_bottom = border_bottom_;
        cam.border_left = border_left_;
        cam.border_right = border_right_;

        /* 选择畸变模型 */
        bool has_distortion = (cam_.k1 != 0.0 || cam_.k2 != 0.0 || cam_.k3 != 0.0 ||
                               cam_.p1 != 0.0 || cam_.p2 != 0.0);
        if (has_distortion) {
            /* Brown 模型（也叫 Plumb Bob）：k1/k2/k3 径向 + p1/p2 切向 */
            cam.distortion.model = cuvslam::Distortion::Model::Brown;
            cam.distortion.parameters = {
                static_cast<float>(cam_.k1), static_cast<float>(cam_.k2),
                static_cast<float>(cam_.k3),
                static_cast<float>(cam_.p1), static_cast<float>(cam_.p2)
            };
        } else {
            /* Pinhole 模型：无畸变 */
            cam.distortion.model = cuvslam::Distortion::Model::Pinhole;
        }

        rig.cameras.push_back(cam);

        /* ---- 3. 创建 Odometry（视觉里程计）----
         * 配置全部来自 ROS2 参数，无硬编码 */
        cuvslam::Odometry::Config cfg;
        cfg.odometry_mode = cuvslam::Odometry::OdometryMode::RGBD;
        /* RGBD 模式：使用 RGB 图像 + 深度图做 3D-2D 匹配 */

        cfg.use_gpu = use_gpu_;
        cfg.async_sba = async_sba_;
        /* 异步集束调整：后台线程优化，不阻塞前端 Track */
        cfg.use_motion_model = use_motion_model_;
        /* 匀速运动模型：预测当前帧的初始位姿，提升跟踪鲁棒性 */
        cfg.use_denoising = use_denoising_;
        /* 去噪：FPV 高速运动时建议关闭以降低延迟 */
        cfg.rectified_stereo_camera = rectified_stereo_;
        /* 是否使用已校正的立体图像对（RGBD 单相机时无效） */
        cfg.enable_observations_export = enable_observations_;
        /* 导出 2D-3D 观测匹配信息（用于外部调试） */
        cfg.enable_landmarks_export = enable_landmarks_;
        /* 导出 3D 路标点信息 */
        cfg.enable_final_landmarks_export = enable_final_landmarks_;
        /* 在节点结束前导出路标 */
        cfg.max_frame_delta_s = static_cast<float>(max_frame_delta_s_);
        /* 两帧最大时间差，超过此值视为跟踪丢失需重新初始化 */

        /* RGBD 特有设置 */
        cfg.rgbd_settings.depth_scale_factor = static_cast<float>(cam_.depth_scale);
        /* 深度值缩放因子：例如 16bit 毫米深度图 -> 除以 1000.0 得米 */
        cfg.rgbd_settings.depth_camera_id = depth_camera_id_;
        /* 深度图对应的相机 ID（多相机系统中指定与哪个相机对齐） */

        odometry_ = std::make_unique<cuvslam::Odometry>(rig, cfg);

        /* ---- 4. 创建 SLAM（同时定位与建图）----
         * 负责闭环检测、位姿图优化和地图管理 */
        cuvslam::Slam::Config slam_cfg;
        slam_cfg.use_gpu = use_gpu_;
        slam_cfg.sync_mode = slam_sync_mode_;
        /* 同步模式：false=异步，SLAM 在单独线程运行 */
        slam_cfg.enable_reading_internals = slam_enable_reading_internals_;
        /* 允许从 SLAM 读取路标和位姿图数据 */
        slam_cfg.planar_constraints = slam_planar_constraints_;
        /* 平面约束：针对地面/墙面场景，提升建图精度 */
        slam_cfg.throttling_time_ms = slam_throttling_ms_;
        /* SLAM 节流：每帧最多花费的毫秒数 */
        slam_cfg.retention_time_ms = slam_retention_ms_;
        /* 地图点保留时间：超过此时间的路标将被移除 */
        slam_cfg.max_map_size = slam_max_map_size_;
        /* 地图最大尺寸：限制关键帧和路标数量 */
        if (!slam_map_cache_path_.empty()) {
            slam_cfg.map_cache_path = slam_map_cache_path_;
            /* 地图缓存路径：设置后可跨会话保存/加载地图 */
        }

        const auto& primaries = odometry_->GetPrimaryCameras();
        /* 获取 Odometry 的主相机 ID 列表 */
        slam_ = std::make_unique<cuvslam::Slam>(rig, primaries, slam_cfg);

        /* 启用内部数据读取（供外部周期性获取路标/位姿图） */
        slam_->EnableReadingData(cuvslam::Slam::DataLayer::Landmarks, 1000);
        slam_->EnableReadingData(cuvslam::Slam::DataLayer::PoseGraph, 1000);

        RCLCPP_INFO(this->get_logger(), "cuVSLAM 初始化完成（RGBD 模式）");
    }

    /* ============================================================
     * 同步回调函数 —— RGB + Depth 时间对齐后触发
     *
     * 每收到一对同步的 RGB 和深度图像，执行：
     * 1. 将 ROS 图像消息转换为 cuVSLAM Image 格式（零拷贝）
     * 2. 调用 Odometry::Track() 计算前端 VO 位姿（高频，无延迟）
     * 3. 调用 Slam::Track() 做后端优化（闭环检测 + PGO）
     * 4. 若检测到 PGO 完成，计算 VO→SLAM 修正量
     * 5. 在 correction_ramp_frames 帧内平滑插值修正
     * 6. 发布融合后的最终位姿（odom / TF / trajectory）
     * ============================================================ */
    void sync_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
    {
        /* 定时打印同步信息 */
        frame_count_++;
        if (frame_count_ % log_interval_ == 0) {
            RCLCPP_INFO(this->get_logger(),
                        "同步 - RGB: %d.%09d, 深度: %d.%09d",
                        rgb_msg->header.stamp.sec, rgb_msg->header.stamp.nanosec,
                        depth_msg->header.stamp.sec, depth_msg->header.stamp.nanosec);
            frame_count_ = 0;
        }

        /* 提取时间戳（纳秒） */
        int64_t timestamp_ns = static_cast<int64_t>(rgb_msg->header.stamp.sec) * 1000000000LL +
                               static_cast<int64_t>(rgb_msg->header.stamp.nanosec);

        /* ---- 转换 RGB 图像 ----
         * cuvslam::Image 支持零拷贝：直接指向 ROS 消息的 data 指针 */
        cuvslam::Image rgb_image;
        rgb_image.timestamp_ns = timestamp_ns;
        rgb_image.camera_index = 0;
        rgb_image.pixels = rgb_msg->data.data();   /* 零拷贝：共享内存 */
        rgb_image.width = rgb_msg->width;
        rgb_image.height = rgb_msg->height;
        rgb_image.pitch = rgb_msg->step;
        rgb_image.data_type = cuvslam::ImageData::DataType::UINT8;
        rgb_image.is_gpu_mem = false;

        /* 编码格式映射 */
        if (rgb_msg->encoding == "bgr8" || rgb_msg->encoding == "rgb8") {
            rgb_image.encoding = cuvslam::ImageData::Encoding::RGB;
        } else if (rgb_msg->encoding == "mono8") {
            rgb_image.encoding = cuvslam::ImageData::Encoding::MONO;
        } else {
            rgb_image.encoding = cuvslam::ImageData::Encoding::RGB;
        }

        /* ---- 转换深度图像 ---- */
        cuvslam::Image depth_image;
        depth_image.timestamp_ns = timestamp_ns;
        depth_image.camera_index = 0;
        depth_image.pixels = depth_msg->data.data();
        depth_image.width = depth_msg->width;
        depth_image.height = depth_msg->height;
        depth_image.pitch = depth_msg->step;
        depth_image.encoding = cuvslam::ImageData::Encoding::MONO;
        depth_image.is_gpu_mem = false;

        /* 深度数据类型映射 */
        if (depth_msg->encoding == "mono16") {
            depth_image.data_type = cuvslam::ImageData::DataType::UINT16;
        } else if (depth_msg->encoding == "32FC1") {
            depth_image.data_type = cuvslam::ImageData::DataType::FLOAT32;
        } else {
            depth_image.data_type = cuvslam::ImageData::DataType::UINT16;
        }

        /* ---- 执行 cuVSLAM RGBD 跟踪 ---- */
        cuvslam::Odometry::ImageSet images = {rgb_image};
        cuvslam::Odometry::ImageSet depths = {depth_image};

        cuvslam::Odometry::TrackOptions track_opts;
        track_opts.num_desired_tracks = num_desired_tracks_;
        /* 这一帧期望追踪的特征点数量 */
        track_opts.ransac_filter = ransac_filter_;
        /* 是否使用 RANSAC 过滤错误匹配 */
        track_opts.kf_survivor_from_last = static_cast<float>(kf_survivor_pct_);
        /* 关键帧筛选：与上个关键帧共视比例低于此值则新建关键帧 */
        track_opts.kf_max_timedelta_between_kfs_s = static_cast<float>(kf_max_timedelta_s_);
        /* 两个关键帧之间的最大时间差 */

        cuvslam::PoseEstimate estimate;
        try {
            /* Track 函数：核心跟踪入口
             * 参数: images = RGB 图像集
             *       imu_data = IMU 数据（空 = 无 IMU）
             *       depths = 深度图像集
             *       track_opts = 跟踪选项 */
            estimate = odometry_->Track(images, {}, depths, track_opts);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "跟踪异常: %s", e.what());
            return;
        }

        /* 如果跟踪丢失，直接跳过该帧 */
        if (!estimate.world_from_rig.has_value()) {
            return;
        }

        /* ---- 提取位姿（四元数 + 平移向量）---- */
        const auto& wfc = estimate.world_from_rig.value();
        const auto& pose = wfc.pose;

        /* cuVSLAM 输出位姿：qx,qy,qz,qw + tx,ty,tz
         * 坐标系为 cuVSLAM 内部坐标系 */
        tf2::Quaternion q_odom_cam(
            pose.rotation[2], -pose.rotation[0],
            -pose.rotation[1], pose.rotation[3]);
        tf2::Vector3 t_odom_cam(
            pose.translation[2], -pose.translation[0], -pose.translation[1]);

        /* 从相机坐标系转换到机器人基座坐标系
         * odom_to_base = odom_to_camera * camera_to_base */

        /* ---- 前端(VO)位姿：立即计算，保证高频输出 ----
         * VO 位姿直接来自 Odometry::Track()，无延迟，每帧必出 */
        tf2::Transform odom_to_camera_tf(q_odom_cam, t_odom_cam);
        tf2::Transform vo_to_base_tf = odom_to_camera_tf * camera_to_base_tf_;

        /* ---- 后端(SLAM)处理 + 闭环修正插值 ----
         * SLAM 在后端运行，检测闭环并做位姿图优化(PGO)。
         * 当 PGO 完成时计算修正量(correction = slam_pose * inv(vo_pose))，
         * 然后通过插值逐渐应用到后续每帧输出，避免位姿跳变。 */
        tf2::Transform final_pose = vo_to_base_tf;

        try {
            cuvslam::Odometry::State state;
            odometry_->GetState(state);
            cuvslam::Pose slam_pose_raw = slam_->Track(state, nullptr);

            /* 将 SLAM 后端位姿转换为 tf2::Transform */
            tf2::Quaternion q_slam_cam(
                slam_pose_raw.rotation[2], -slam_pose_raw.rotation[0],
                -slam_pose_raw.rotation[1], slam_pose_raw.rotation[3]);
            tf2::Vector3 t_slam_cam(
                slam_pose_raw.translation[2], -slam_pose_raw.translation[0], -slam_pose_raw.translation[1]);
            tf2::Transform slam_to_camera_tf(q_slam_cam, t_slam_cam);
            tf2::Transform slam_to_base_tf = slam_to_camera_tf * camera_to_base_tf_;

            /* 保存最新 SLAM 位姿供外部使用 */
            latest_slam_pose_ = slam_to_base_tf;
            latest_slam_pose_stamp_ = rgb_msg->header.stamp;

            /* 发布 SLAM 后端位姿话题（始终发布，供调试/外部使用） */
            publish_slam_pose(slam_to_base_tf, rgb_msg->header.stamp);

            /* 检测 PGO（位姿图优化）事件 - 上升沿检测 */
            cuvslam::Slam::Metrics metrics;
            slam_->GetSlamMetrics(metrics);
            bool pgo_now = metrics.pgo_status && metrics.lc_status;
                        if (pgo_now && !last_pgo_active_) {
                /* PGO 刚刚完成：计算新修正目标，只修正增量部分
                 * new_target = slam_pose * inv(vo_pose)
                 * delta = new_target * inv(current_correction) —— 增量
                 * 启动 ramp：current_correction_ 从旧值渐变到新值 */
                tf2::Transform new_target = slam_to_base_tf * vo_to_base_tf.inverse();
                tf2::Transform delta = new_target * current_correction_.inverse();
                double t_len = delta.getOrigin().length();
                double r_angle = delta.getRotation().getAngle();
                if (t_len > 0.01 || r_angle > 0.0087) {  // 1cm or 0.5deg threshold
                    correction_start_value_ = current_correction_;
                    correction_target_ = new_target;
                    correction_start_frame_ = frame_count_;
                    correction_active_ = true;
                    RCLCPP_WARN(this->get_logger(),
                        "=== 增量回环! delta=(%.3f,%.3f,%.3f) t_len=%.3f r_deg=%.1f ===",
                        delta.getOrigin().x(), delta.getOrigin().y(), delta.getOrigin().z(),
                        t_len, r_angle * 180.0 / 3.14159265358979323846);
                }
            }
            last_pgo_active_ = pgo_now;

            /* ---- 增量式平滑插值修正 ----
             * 从 correction_start_value_（旧的累积修正量）
             * 渐变到 correction_target_（新的累积修正量），
             * 避免位姿回弹到原始 VO。 */
            if (correction_active_) {
                int frames_elapsed = frame_count_ - correction_start_frame_;
                double alpha = clamp_double(
                    (double)frames_elapsed / (double)correction_ramp_frames_, 0.0, 1.0);

                /* 平移线性插值（从起始值到目标值） */
                tf2::Vector3 interp_trans = correction_start_value_.getOrigin().lerp(
                    correction_target_.getOrigin(), alpha);

                /* 旋转 Slerp 插值（从起始值到目标值） */
                tf2::Quaternion interp_rot = correction_start_value_.getRotation().slerp(
                    correction_target_.getRotation(), alpha);

                current_correction_ = tf2::Transform(interp_rot, interp_trans);

                if (alpha >= 1.0) {
                    correction_active_ = false;
                }
            }

            /* 应用持久化修正量到当前 VO 位姿 */
            final_pose = current_correction_ * vo_to_base_tf;

            if (frame_count_ % 50 == 0) {
                                double progress = correction_active_ ?
                    clamp_double((double)(frame_count_ - correction_start_frame_) /
                        (double)correction_ramp_frames_, 0.0, 1.0) : 0.0;
                RCLCPP_INFO(this->get_logger(),
                    "SLAM - loop=%s PGO=%s lc_lmks=%d progress=%.2f %s",
                    metrics.lc_status ? "Y" : "N",
                    metrics.pgo_status ? "Y" : "N",
                    metrics.lc_good_landmarks_count,
                    progress,
                    correction_active_ ? "correcting" : "");            }
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "SLAM 跟踪异常: %s", e.what());
        }

        /* 发布最终位姿（前端 VO + 后端修正插值融合结果） */
        publish_odometry(final_pose, rgb_msg->header.stamp);
        publish_tf(final_pose, rgb_msg->header.stamp);
        update_trajectory(final_pose, rgb_msg->header.stamp);
    }

    /* ============================================================
     * 发布里程计消息（Odometry + PoseStamped）
     *
     * 填充 nav_msgs::Odometry 消息，设置协方差矩阵。
     * 同时发布 geometry_msgs::PoseStamped 供导航栈使用。
     * ============================================================ */
    void publish_odometry(const tf2::Transform& odom_to_base,
                          const builtin_interfaces::msg::Time& stamp)
    {
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = odom_frame_;
        odom_msg.child_frame_id = robot_frame_;

        odom_msg.pose.pose.position.x = odom_to_base.getOrigin().x();
        odom_msg.pose.pose.position.y = odom_to_base.getOrigin().y();
        odom_msg.pose.pose.position.z = odom_to_base.getOrigin().z();
        odom_msg.pose.pose.orientation = tf2::toMsg(odom_to_base.getRotation());

        /* 填充协方差矩阵对角线（6x6，仅设对角线非零） */
        std::fill(odom_msg.pose.covariance.begin(), odom_msg.pose.covariance.end(), 0.0);
        std::fill(odom_msg.twist.covariance.begin(), odom_msg.twist.covariance.end(), 0.0);
        odom_msg.pose.covariance[0] = cov_pos_x_;
        odom_msg.pose.covariance[7] = cov_pos_y_;
        odom_msg.pose.covariance[14] = cov_pos_z_;
        odom_msg.pose.covariance[21] = cov_rot_x_;
        odom_msg.pose.covariance[28] = cov_rot_y_;
        odom_msg.pose.covariance[35] = cov_rot_z_;
        odom_msg.twist.covariance[0] = cov_twist_linear_x_;
        odom_msg.twist.covariance[7] = cov_twist_linear_y_;
        odom_msg.twist.covariance[14] = cov_twist_linear_z_;
        odom_msg.twist.covariance[21] = cov_twist_angular_x_;
        odom_msg.twist.covariance[28] = cov_twist_angular_y_;
        odom_msg.twist.covariance[35] = cov_twist_angular_z_;

        odom_publisher_->publish(odom_msg);

        /* 同时发布简化版位姿消息 */
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = odom_msg.header;
        pose_msg.pose = odom_msg.pose.pose;
        pose_publisher_->publish(pose_msg);
    }

    /* ============================================================
     * 发布 odom -> base_link 的 TF 变换
     * ============================================================ */
    void publish_tf(const tf2::Transform& odom_to_base,
                    const builtin_interfaces::msg::Time& stamp)
    {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = odom_frame_;
        tf_msg.child_frame_id = robot_frame_;
        tf_msg.transform = tf2::toMsg(odom_to_base);
        tf_broadcaster_->sendTransform(tf_msg);
    }

    /* ============================================================
     * 发布 SLAM 后端位姿（闭环/PGO 校正后）
     *
     * 独立于 use_slam_pose 设置，始终发布到 slam_pose_topic，
     * 供外部节点/调试工具获取 SLAM 优化后的高精度位姿。
     * ============================================================ */
    void publish_slam_pose(const tf2::Transform& slam_to_base,
                           const builtin_interfaces::msg::Time& stamp)
    {
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = odom_frame_;
        msg.pose.position.x = slam_to_base.getOrigin().x();
        msg.pose.position.y = slam_to_base.getOrigin().y();
        msg.pose.position.z = slam_to_base.getOrigin().z();
        msg.pose.orientation = tf2::toMsg(slam_to_base.getRotation());
        slam_pose_publisher_->publish(msg);
    }

    /* ============================================================
     * 更新并发布轨迹可视化（Marker::LINE_STRIP）
     *
     * 将历史位姿点连接成线，在 RViz 中显示飞行轨迹。
     * 超过 max_points 时自动删除最早的点。
     * ============================================================ */
    void update_trajectory(const tf2::Transform& odom_to_base,
                           const builtin_interfaces::msg::Time& stamp)
    {
        /* 使用 static 变量保持轨迹线在多次回调间持续增长 */
        static visualization_msgs::msg::Marker trajectory;
        static bool initialized = false;

        if (!initialized) {
            trajectory.header.frame_id = odom_frame_;
            trajectory.ns = "cuvslam_trajectory";
            trajectory.id = 0;
            trajectory.type = visualization_msgs::msg::Marker::LINE_STRIP;
            trajectory.action = visualization_msgs::msg::Marker::ADD;
            trajectory.pose.orientation.w = 1.0;
            trajectory.scale.x = trajectory_line_width_;
            trajectory.color.r = trajectory_color_r_;
            trajectory.color.g = trajectory_color_g_;
            trajectory.color.b = trajectory_color_b_;
            trajectory.color.a = 1.0;
            initialized = true;
        }

        /* 添加当前位姿点到轨迹 */
        geometry_msgs::msg::Point point;
        point.x = odom_to_base.getOrigin().x();
        point.y = odom_to_base.getOrigin().y();
        point.z = odom_to_base.getOrigin().z();
        trajectory.points.push_back(point);

        /* 限制最大点数，防止内存膨胀 */
        if (trajectory.points.size() > static_cast<size_t>(trajectory_max_points_)) {
            trajectory.points.erase(trajectory.points.begin());
        }

        trajectory.header.stamp = stamp;
        trajectory_publisher_->publish(trajectory);
    }

    /* ============================================================
     * 数据结构
     * ============================================================ */

    /* 话题名称 */
    struct Topics { std::string rgb, depth; };

    /* 发布话题名称 */
    struct Publishers { std::string odom, pose, trajectory, slam_pose; };

    /* 相机内参结构体 */
    struct CameraParams {
        int width, height;
        double fx, fy, cx, cy;
        double k1, k2, p1, p2, k3;
        double depth_scale;
    };

    /* ============================================================
     * 成员变量
     * ============================================================ */

    /* 话题 */
    Topics topics_;
    Publishers publishers_;
    CameraParams cam_;

    /* 坐标系 */
    std::string world_frame_, odom_frame_, camera_frame_, robot_frame_;

        /* ---- 增量式回环修正系统 ----
     * current_correction_ 持续累积修正量，永不回弹。
     * PGO 触发时从当前修正值渐变到新目标值（只修正增量）。 */

    /* 当前持久化的累积修正量（持续生效，不回弹） */
    tf2::Transform current_correction_{tf2::Quaternion(0.0, 0.0, 0.0, 1.0), tf2::Vector3(0,0,0)};

    /* 渐进插值帧数，默认 150 帧 ≈ 5 秒 @ 30fps */
    int correction_ramp_frames_ = 150;

    /* 修正起始值（本次 ramp 开始时的 current_correction_） */
    tf2::Transform correction_start_value_{tf2::Quaternion(0.0, 0.0, 0.0, 1.0), tf2::Vector3(0,0,0)};

    /* 修正目标值（本次 ramp 结束时的 current_correction_） */
    tf2::Transform correction_target_{tf2::Quaternion(0.0, 0.0, 0.0, 1.0), tf2::Vector3(0,0,0)};

    /* 修正起始帧号 */
    int correction_start_frame_ = 0;

    /* 是否正在渐近修正中 */
    bool correction_active_ = false;
    /* 上一帧 PGO 状态（上升沿检测用） */
    bool last_pgo_active_ = false;

    /* 最新 SLAM 后端位姿缓存 */
    tf2::Transform latest_slam_pose_;
    builtin_interfaces::msg::Time latest_slam_pose_stamp_;

    /* ---- Odometry::Config 参数 ---- */
    bool use_gpu_ = true;
    bool use_motion_model_ = true;
    bool use_denoising_ = false;
    bool rectified_stereo_ = false;
    bool enable_observations_ = true;
    bool enable_landmarks_ = true;
    bool enable_final_landmarks_ = false;
    bool async_sba_ = true;
    double max_frame_delta_s_ = 1.0;
    bool ransac_filter_ = false;
    double kf_survivor_pct_ = 41.0;
    int kf_max_timedelta_s_ = 60;
    int num_desired_tracks_ = 400;
    int border_top_ = 0, border_bottom_ = 0, border_left_ = 0, border_right_ = 0;
    int depth_camera_id_ = 0;

    /* ---- Slam::Config 参数 ---- */
    bool slam_sync_mode_ = false;
    bool slam_planar_constraints_ = false;
    bool slam_enable_reading_internals_ = true;
    int slam_throttling_ms_ = 1000;
    int slam_retention_ms_ = 5000;
    int slam_max_map_size_ = 300;
    std::string slam_map_cache_path_;

    /* 同步 / TF / 日志 */
    int sync_queue_size_, sync_max_interval_ms_, tf_retry_count_, tf_retry_interval_ms_, log_interval_;

    /* 默认 TF 外参 */
    double default_tf_x_, default_tf_y_, default_tf_z_;
    double default_tf_roll_, default_tf_pitch_, default_tf_yaw_;

    /* 协方差 */
    double cov_pos_x_, cov_pos_y_, cov_pos_z_;
    double cov_rot_x_, cov_rot_y_, cov_rot_z_;
    double cov_twist_linear_x_, cov_twist_linear_y_, cov_twist_linear_z_;
    double cov_twist_angular_x_, cov_twist_angular_y_, cov_twist_angular_z_;

    /* 轨迹 */
    double trajectory_line_width_;
    int trajectory_max_points_;
    double trajectory_color_r_, trajectory_color_g_, trajectory_color_b_;

    /* TF */
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    geometry_msgs::msg::TransformStamped base_to_camera_tf_;
    tf2::Transform camera_to_base_tf_;

    /* cuVSLAM 实例 */
    std::unique_ptr<cuvslam::Odometry> odometry_;
    std::unique_ptr<cuvslam::Slam> slam_;

    /* 订阅器 + 同步器 */
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_img_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_img_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    /* 发布器 */
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr trajectory_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr slam_pose_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    /* 帧计数器 */
    int frame_count_ = 0;
};

/* ============================================================
 * 主函数
 * ============================================================ */
int main(int argc, char** argv)
{
    /* 初始化 ROS2 */
    rclcpp::init(argc, argv);
    /* 创建节点实例 */
    auto node = std::make_shared<CuVSLAM_RGBD>();
    /* 进入事件循环 */
    rclcpp::spin(node);
    /* 关闭 ROS2 */
    rclcpp::shutdown();
    return 0;
}
