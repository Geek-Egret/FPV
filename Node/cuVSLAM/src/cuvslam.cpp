/**
 * @file cuvslam.cpp
 * @brief cuVSLAM RGBD ROS2 节点
 *
 * 订阅RGB图像和深度图像，运行NVIDIA cuVSLAM RGBD模式（GPU加速），
 * 发布里程计(odom)、位姿(pose)、轨迹(trajectory)和TF变换。
 * 所有参数通过ROS2参数系统配置，由Python启动脚本传入。
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <chrono>
#include <fstream>

// ROS2 核心
#include <rclcpp/rclcpp.hpp>
// 消息类型
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
// 消息同步
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
// TF2
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// cuVSLAM
#include <cuvslam/cuvslam2.h>
#include <cuvslam/cuvslam_gpu.h>

// 定义消息同步策略：RGB图像 + 深度图像的近似时间同步
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image> SyncPolicy;

class CuVSLAM_RGBD : public rclcpp::Node
{
public:
    CuVSLAM_RGBD() : Node("cuvslam")
    {
        // 1. 声明并加载ROS参数
        declare_all_parameters();
        load_parameters();

        // 2. 发布 world -> odom 静态TF
        publish_static_world_to_odom();

        // 3. 获取 base_link -> camera_link 的TF变换
        lookup_base_to_camera_tf();

        // 4. 创建消息订阅器（RGB + Depth）
        rgb_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, topics_.rgb);
        depth_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, topics_.depth);

        // 5. 创建消息同步器（时间对齐）
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(sync_queue_size_),
            *rgb_img_sub_,
            *depth_img_sub_);
        sync_->setMaxIntervalDuration(std::chrono::milliseconds(sync_max_interval_ms_));
        sync_->registerCallback(std::bind(
            &CuVSLAM_RGBD::sync_callback,
            this,
            std::placeholders::_1,
            std::placeholders::_2));

        // 6. 创建发布者（QoS配置为可靠传输）
        auto sensor_qos = rclcpp::SensorDataQoS();
        sensor_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);

        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
            publishers_.odom, sensor_qos);
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            publishers_.pose, sensor_qos);
        trajectory_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
            publishers_.trajectory, sensor_qos);

        // 7. 创建TF广播器
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // 8. 初始化cuVSLAM（GPU预热 + 创建Odometry + SLAM）
        init_cuvslam();

        RCLCPP_INFO(this->get_logger(), "cuVSLAM RGBD node started");
        RCLCPP_INFO(this->get_logger(), "  RGB topic: %s", topics_.rgb.c_str());
        RCLCPP_INFO(this->get_logger(), "  Depth topic: %s", topics_.depth.c_str());
        RCLCPP_INFO(this->get_logger(), "  Camera: %dx%d fx=%.2f fy=%.2f",
                    cam_.width, cam_.height, cam_.fx, cam_.fy);
        RCLCPP_INFO(this->get_logger(), "  Depth scale: %.1f", cam_.depth_scale);
    }

private:
    // 安全访问vector元素，越界返回默认值
    static double vec_at(const std::vector<double>& v, size_t i, double def)
    {
        return i < v.size() ? v[i] : def;
    }

    // ================================================================
    // 参数声明 - 所有可调参数通过ROS2参数系统传入
    // ================================================================
    void declare_all_parameters()
    {
        // 订阅话题
        this->declare_parameter<std::string>("rgb_topic", "/rgb/image_raw");
        this->declare_parameter<std::string>("depth_topic", "/depth/image_raw");

        // 坐标系定义
        this->declare_parameter<std::string>("world_frame", "world");
        this->declare_parameter<std::string>("camera_frame", "camera_link");
        this->declare_parameter<std::string>("robot_frame", "base_link");
        this->declare_parameter<std::string>("odom_frame", "odom");

        // 发布话题
        this->declare_parameter<std::string>("odom_topic", "cuvslam/odom");
        this->declare_parameter<std::string>("pose_topic", "cuvslam/pose");
        this->declare_parameter<std::string>("trajectory_topic", "cuvslam/trajectory");

        // 相机内参（Brown畸变模型：k1/k2/k3径向, p1/p2切向）
        this->declare_parameter<int>("camera.width", 640);
        this->declare_parameter<int>("camera.height", 480);
        this->declare_parameter<double>("camera.fx", 455.483);
        this->declare_parameter<double>("camera.fy", 455.483);
        this->declare_parameter<double>("camera.cx", 329.67);
        this->declare_parameter<double>("camera.cy", 243.265);
        this->declare_parameter<double>("camera.k1", 0.0);
        this->declare_parameter<double>("camera.k2", 0.0);
        this->declare_parameter<double>("camera.p1", 0.0);
        this->declare_parameter<double>("camera.p2", 0.0);
        this->declare_parameter<double>("camera.k3", 0.0);
        this->declare_parameter<double>("depth.scale_factor", 1000.0);

        // cuVSLAM 算法参数
        this->declare_parameter<bool>("use_denoising", false);
        this->declare_parameter<bool>("rectified_stereo", false);
        this->declare_parameter<bool>("enable_observations_export", true);
        this->declare_parameter<bool>("enable_landmarks_export", true);
        this->declare_parameter<int>("num_desired_tracks", 400);

        // 同步器
        this->declare_parameter<int>("sync_queue_size", 10);
        this->declare_parameter<int>("sync_max_interval_ms", 100);

        // TF查找参数
        this->declare_parameter<int>("tf_retry_count", 10);
        this->declare_parameter<int>("tf_retry_interval_ms", 500);
        this->declare_parameter<std::vector<double>>("default_tf_translation", {0.2, 0.0, 0.1});
        this->declare_parameter<std::vector<double>>("default_tf_rpy", {0.0, 0.0, 0.0});

        // 里程计协方差矩阵
        this->declare_parameter<std::vector<double>>("pose_covariance",
            {0.01, 0.01, 0.01, 0.01, 0.01, 0.01});
        this->declare_parameter<std::vector<double>>("twist_covariance",
            {0.1, 0.1, 0.1, 0.1, 0.1, 0.1});

        // 轨迹可视化
        this->declare_parameter<double>("trajectory_line_width", 0.05);
        this->declare_parameter<int>("trajectory_max_points", 1000);
        this->declare_parameter<std::vector<double>>("trajectory_color", {0.0, 1.0, 0.0});

        // 日志
        this->declare_parameter<int>("log_interval_frames", 300);
    }

    // 从ROS参数服务器加载所有参数
    void load_parameters()
    {
        topics_.rgb = this->get_parameter("rgb_topic").as_string();
        topics_.depth = this->get_parameter("depth_topic").as_string();

        world_frame_ = this->get_parameter("world_frame").as_string();
        camera_frame_ = this->get_parameter("camera_frame").as_string();
        robot_frame_ = this->get_parameter("robot_frame").as_string();
        odom_frame_ = this->get_parameter("odom_frame").as_string();

        publishers_.odom = this->get_parameter("odom_topic").as_string();
        publishers_.pose = this->get_parameter("pose_topic").as_string();
        publishers_.trajectory = this->get_parameter("trajectory_topic").as_string();

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

        use_denoising_ = this->get_parameter("use_denoising").as_bool();
        rectified_stereo_ = this->get_parameter("rectified_stereo").as_bool();
        enable_observations_ = this->get_parameter("enable_observations_export").as_bool();
        enable_landmarks_ = this->get_parameter("enable_landmarks_export").as_bool();
        num_desired_tracks_ = this->get_parameter("num_desired_tracks").as_int();

        sync_queue_size_ = this->get_parameter("sync_queue_size").as_int();
        sync_max_interval_ms_ = this->get_parameter("sync_max_interval_ms").as_int();

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

        trajectory_line_width_ = this->get_parameter("trajectory_line_width").as_double();
        trajectory_max_points_ = this->get_parameter("trajectory_max_points").as_int();

        auto traj_color = this->get_parameter("trajectory_color").as_double_array();
        trajectory_color_r_ = vec_at(traj_color, 0, 0.0);
        trajectory_color_g_ = vec_at(traj_color, 1, 1.0);
        trajectory_color_b_ = vec_at(traj_color, 2, 0.0);

        log_interval_ = this->get_parameter("log_interval_frames").as_int();
    }

    // 发布 world -> odom 静态TF（OpenCV坐标系转ROS坐标系）
    void publish_static_world_to_odom()
    {
        tf2::Quaternion q1, q2, final_quat;
        q1.setRPY(0.0, 0.0, 0.0);
        q2.setRPY(0.0, 0.0, 0.0);
        final_quat = q2 * q1;

        static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        geometry_msgs::msg::TransformStamped static_tf;
        static_tf.header.stamp = this->now();
        static_tf.header.frame_id = world_frame_;
        static_tf.child_frame_id = odom_frame_;
        static_tf.transform.translation.x = 0.0;
        static_tf.transform.translation.y = 0.0;
        static_tf.transform.translation.z = 0.0;
        static_tf.transform.rotation.x = final_quat.x();
        static_tf.transform.rotation.y = final_quat.y();
        static_tf.transform.rotation.z = final_quat.z();
        static_tf.transform.rotation.w = final_quat.w();
        static_tf_broadcaster_->sendTransform(static_tf);
    }

    // 获取 base_link -> camera_link 的TF变换（带重试，失败则用默认值）
    void lookup_base_to_camera_tf()
    {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        bool got_tf = false;
        for (int retry = 0; retry < tf_retry_count_; ++retry) {
            try {
                if (tf_buffer_->canTransform(robot_frame_, camera_frame_,
                                             tf2::TimePointZero, tf2::durationFromSec(1.0))) {
                    base_to_camera_tf_ = tf_buffer_->lookupTransform(
                        robot_frame_, camera_frame_, tf2::TimePointZero);
                    RCLCPP_INFO(this->get_logger(), "Got %s -> %s static TF",
                                robot_frame_.c_str(), camera_frame_.c_str());
                    got_tf = true;
                    break;
                }
            } catch (const tf2::TransformException& ex) {
                RCLCPP_WARN(this->get_logger(), "TF retry %d/%d: %s",
                            retry + 1, tf_retry_count_, ex.what());
                rclcpp::sleep_for(std::chrono::milliseconds(tf_retry_interval_ms_));
            }
        }

        if (!got_tf) {
            RCLCPP_WARN(this->get_logger(), "Using default %s -> %s TF",
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

        // 计算 camera -> base 的逆变换（后续坐标转换使用）
        tf2::fromMsg(base_to_camera_tf_.transform, camera_to_base_tf_);
        camera_to_base_tf_ = camera_to_base_tf_.inverse();
    }

    // 初始化cuVSLAM：GPU预热 + 创建视觉里程计 + 创建SLAM
    void init_cuvslam()
    {
        // GPU预热，创建CUDA运行时上下文
        try {
            cuvslam::WarmUpGPU();
            RCLCPP_INFO(this->get_logger(), "GPU warmed up, cuVSLAM version: %s",
                        cuvslam::GetVersion(nullptr, nullptr, nullptr).data());
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "GPU warmup: %s", e.what());
        }

        // 构建相机Rig（单目RGBD相机）
        cuvslam::Rig rig;
        cuvslam::Camera cam;
        cam.size = {cam_.width, cam_.height};
        cam.principal = {static_cast<float>(cam_.cx), static_cast<float>(cam_.cy)};
        cam.focal = {static_cast<float>(cam_.fx), static_cast<float>(cam_.fy)};
        cam.rig_from_camera.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
        cam.rig_from_camera.translation = {0.0f, 0.0f, 0.0f};

        // 判断是否有畸变参数，选择畸变模型
        bool has_distortion = (cam_.k1 != 0.0 || cam_.k2 != 0.0 || cam_.k3 != 0.0 ||
                               cam_.p1 != 0.0 || cam_.p2 != 0.0);
        if (has_distortion) {
            cam.distortion.model = cuvslam::Distortion::Model::Brown;
            cam.distortion.parameters = {
                static_cast<float>(cam_.k1), static_cast<float>(cam_.k2),
                static_cast<float>(cam_.k3),
                static_cast<float>(cam_.p1), static_cast<float>(cam_.p2)
            };
        } else {
            cam.distortion.model = cuvslam::Distortion::Model::Pinhole;
        }

        rig.cameras.push_back(cam);

        // 配置视觉里程计（RGBD模式）
        cuvslam::Odometry::Config cfg;
        cfg.odometry_mode = cuvslam::Odometry::OdometryMode::RGBD;
        cfg.use_gpu = true;
        cfg.async_sba = true;
        cfg.use_motion_model = true;
        cfg.use_denoising = use_denoising_;
        cfg.rectified_stereo_camera = rectified_stereo_;
        cfg.enable_observations_export = enable_observations_;
        cfg.enable_landmarks_export = enable_landmarks_;
        cfg.rgbd_settings.depth_scale_factor = static_cast<float>(cam_.depth_scale);
        cfg.rgbd_settings.depth_camera_id = 0;  // 深度与相机0对齐

        odometry_ = std::make_unique<cuvslam::Odometry>(rig, cfg);

        // 创建SLAM实例
        cuvslam::Slam::Config slam_cfg;
        slam_cfg.use_gpu = true;
        slam_cfg.sync_mode = false;
        slam_cfg.planar_constraints = false;
        slam_cfg.enable_reading_internals = true;

        const auto& primaries = odometry_->GetPrimaryCameras();
        slam_ = std::make_unique<cuvslam::Slam>(rig, primaries, slam_cfg);
        slam_->EnableReadingData(cuvslam::Slam::DataLayer::Landmarks, 1000);
        slam_->EnableReadingData(cuvslam::Slam::DataLayer::PoseGraph, 1000);

        RCLCPP_INFO(this->get_logger(), "cuVSLAM initialized (RGBD mode)");
    }

    // ================================================================
    // 同步回调：RGB + Depth时间对齐后触发
    // ================================================================
    void sync_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
    {
        // 周期打印同步时间戳
        frame_count_++;
        if (frame_count_ % log_interval_ == 0) {
            RCLCPP_INFO(this->get_logger(),
                        "Sync - RGB: %d.%09d, Depth: %d.%09d",
                        rgb_msg->header.stamp.sec, rgb_msg->header.stamp.nanosec,
                        depth_msg->header.stamp.sec, depth_msg->header.stamp.nanosec);
            frame_count_ = 0;
        }

        // 提取时间戳（纳秒）
        int64_t timestamp_ns = static_cast<int64_t>(rgb_msg->header.stamp.sec) * 1000000000LL +
                               static_cast<int64_t>(rgb_msg->header.stamp.nanosec);

        // ROS图像消息转换为cuVSLAM Image格式（零拷贝）
        cuvslam::Image rgb_image;
        rgb_image.timestamp_ns = timestamp_ns;
        rgb_image.camera_index = 0;
        rgb_image.pixels = rgb_msg->data.data();
        rgb_image.width = rgb_msg->width;
        rgb_image.height = rgb_msg->height;
        rgb_image.pitch = rgb_msg->step;
        rgb_image.data_type = cuvslam::ImageData::DataType::UINT8;
        rgb_image.is_gpu_mem = false;

        if (rgb_msg->encoding == "bgr8" || rgb_msg->encoding == "rgb8") {
            rgb_image.encoding = cuvslam::ImageData::Encoding::RGB;
        } else if (rgb_msg->encoding == "mono8") {
            rgb_image.encoding = cuvslam::ImageData::Encoding::MONO;
        } else {
            rgb_image.encoding = cuvslam::ImageData::Encoding::RGB;
        }

        // 深度图像
        cuvslam::Image depth_image;
        depth_image.timestamp_ns = timestamp_ns;
        depth_image.camera_index = 0;
        depth_image.pixels = depth_msg->data.data();
        depth_image.width = depth_msg->width;
        depth_image.height = depth_msg->height;
        depth_image.pitch = depth_msg->step;
        depth_image.encoding = cuvslam::ImageData::Encoding::MONO;
        depth_image.is_gpu_mem = false;

        if (depth_msg->encoding == "mono16") {
            depth_image.data_type = cuvslam::ImageData::DataType::UINT16;
        } else if (depth_msg->encoding == "32FC1") {
            depth_image.data_type = cuvslam::ImageData::DataType::FLOAT32;
        } else {
            depth_image.data_type = cuvslam::ImageData::DataType::UINT16;
        }

        // 调用cuVSLAM RGBD追踪
        cuvslam::Odometry::ImageSet images = {rgb_image};
        cuvslam::Odometry::ImageSet depths = {depth_image};

        cuvslam::Odometry::TrackOptions track_opts;
        track_opts.num_desired_tracks = num_desired_tracks_;

        cuvslam::PoseEstimate estimate;
        try {
            estimate = odometry_->Track(images, {}, depths, track_opts);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Track error: %s", e.what());
            return;
        }

        // 追踪失败则跳过本帧
        if (!estimate.world_from_rig.has_value()) {
            return;
        }

        // 提取位姿（四元数 + 平移向量）
        const auto& wfc = estimate.world_from_rig.value();
        const auto& pose = wfc.pose;

        // 构建 odom -> camera 的TF变换（cuvslam X右Y下Z前 -> ROS X前Y左Z上）//
        tf2::Quaternion q_odom_cam(
            pose.rotation[2], -pose.rotation[0],
            -pose.rotation[1], pose.rotation[3]);
        tf2::Vector3 t_odom_cam(
            pose.translation[2], -pose.translation[0], -pose.translation[1]);

        // 构建 odom -> base 的TF变换
        tf2::Transform odom_to_camera_tf(q_odom_cam, t_odom_cam);
        tf2::Transform odom_to_base_tf = odom_to_camera_tf * camera_to_base_tf_;

        // 发布里程计、TF和轨迹
        publish_odometry(odom_to_base_tf, rgb_msg->header.stamp);
        publish_tf(odom_to_base_tf, rgb_msg->header.stamp);
        update_trajectory(odom_to_base_tf, rgb_msg->header.stamp);

        // SLAM后端处理
        try {
            cuvslam::Odometry::State state;
            odometry_->GetState(state);
            slam_->Track(state, nullptr);
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "SLAM track: %s", e.what());
        }
    }

    // 发布里程计消息（odom -> base_link）
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

        // 设置协方差
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

        // 同时发布 PoseStamped
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = odom_msg.header;
        pose_msg.pose = odom_msg.pose.pose;
        pose_publisher_->publish(pose_msg);
    }

    // 发布 odom -> base_link 的TF动态变换
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

    // 更新并发布SLAM轨迹（折线Marker）
    void update_trajectory(const tf2::Transform& odom_to_base,
                           const builtin_interfaces::msg::Time& stamp)
    {
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

        geometry_msgs::msg::Point point;
        point.x = odom_to_base.getOrigin().x();
        point.y = odom_to_base.getOrigin().y();
        point.z = odom_to_base.getOrigin().z();
        trajectory.points.push_back(point);

        if (trajectory.points.size() > static_cast<size_t>(trajectory_max_points_)) {
            trajectory.points.erase(trajectory.points.begin());
        }

        trajectory.header.stamp = stamp;
        trajectory_publisher_->publish(trajectory);
    }

    // ================================================================
    // 数据结构
    // ================================================================
    struct Topics {
        std::string rgb;
        std::string depth;
    };

    struct Publishers {
        std::string odom;
        std::string pose;
        std::string trajectory;
    };

    struct CameraParams {
        int width, height;
        double fx, fy, cx, cy;
        double k1, k2, p1, p2, k3;
        double depth_scale;
    };

    // ================================================================
    // 成员变量
    // ================================================================
    Topics topics_;
    Publishers publishers_;
    CameraParams cam_;

    // 坐标框架
    std::string world_frame_;
    std::string odom_frame_;
    std::string camera_frame_;
    std::string robot_frame_;

    // 算法参数
    bool use_denoising_;
    bool rectified_stereo_;
    bool enable_observations_;
    bool enable_landmarks_;
    int num_desired_tracks_;

    // 同步器
    int sync_queue_size_;
    int sync_max_interval_ms_;

    // TF
    int tf_retry_count_;
    int tf_retry_interval_ms_;
    int log_interval_;
    double default_tf_x_, default_tf_y_, default_tf_z_;
    double default_tf_roll_, default_tf_pitch_, default_tf_yaw_;

    // 协方差
    double cov_pos_x_, cov_pos_y_, cov_pos_z_;
    double cov_rot_x_, cov_rot_y_, cov_rot_z_;
    double cov_twist_linear_x_, cov_twist_linear_y_, cov_twist_linear_z_;
    double cov_twist_angular_x_, cov_twist_angular_y_, cov_twist_angular_z_;

    // 轨迹
    double trajectory_line_width_;
    int trajectory_max_points_;
    double trajectory_color_r_, trajectory_color_g_, trajectory_color_b_;

    // TF组件
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    geometry_msgs::msg::TransformStamped base_to_camera_tf_;
    tf2::Transform camera_to_base_tf_;

    // cuVSLAM实例
    std::unique_ptr<cuvslam::Odometry> odometry_;
    std::unique_ptr<cuvslam::Slam> slam_;

    // 订阅器
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_img_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_img_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // 发布器
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr trajectory_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    int frame_count_ = 0;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CuVSLAM_RGBD>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
