/**
 * @file orb_slam3.cpp
 * @brief ORB_SLAM3 RGBD ROS2 节点
 *
 * 订阅RGB图像和深度图像，运行ORB_SLAM3 RGBD模式，
 * 发布里程计(odom)、位姿(pose)、轨迹(trajectory)和TF变换。
 * 所有参数通过ROS2参数系统配置，由Python启动脚本传入。
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <chrono>

// ROS2 核心
#include <rclcpp/rclcpp.hpp>
// 消息类型
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
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
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.h>
// OpenCV
#include <opencv2/opencv.hpp>
// Eigen + Sophus (ORB-SLAM3依赖)
#include <sophus/se3.hpp>
#include <Eigen/Dense>
// ORB-SLAM3
#include "orb_slam3.h"

// 解决Eigen内存对齐问题
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SE3f)

// 定义消息同步策略：RGB图像 + 深度图像的近似时间同步
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image> SyncPolicy;

class ORB_SLAM3_ROS2 : public rclcpp::Node
{
public:
    ORB_SLAM3_ROS2() : Node("orb_slam3")
    {
        // 1. 声明并加载ROS参数
        declare_all_parameters();
        load_parameters();

        // 2. 发布 world -> odom 静态TF
        publish_static_world_to_odom();

        // 3. 获取 base_link -> camera_link 的TF变换（带重试）
        base_to_camera_tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        base_to_camera_tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*base_to_camera_tf_buffer_);

        bool got_tf = false;
        for (int retry = 0; retry < tf_retry_count_; ++retry) {
            try {
                if (base_to_camera_tf_buffer_->canTransform(
                        robot_frame_, camera_frame_,
                        tf2::TimePointZero, tf2::durationFromSec(1.0))) {
                    base_to_camera_tf_msgs = base_to_camera_tf_buffer_->lookupTransform(
                        robot_frame_, camera_frame_, tf2::TimePointZero);
                    RCLCPP_INFO(this->get_logger(),
                                "Got base_link -> %s static TF", camera_frame_.c_str());
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
            RCLCPP_WARN(this->get_logger(),
                        "Cannot get base_link -> %s TF, using defaults", camera_frame_.c_str());
            set_default_base_to_camera_transform();
        }

        // 4. 创建消息订阅器（RGB + Depth）
        rgb_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, topics_.rgb);
        depth_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, topics_.depth);

        // 5. 初始化ORB-SLAM3系统
        SLAM = std::make_shared<ORB_SLAM3::System>(
            orb_.vocabulary_path, orb_.settings_path,
            orb_.sensor_type, orb_.use_viewer);

        // 6. 创建消息同步器（时间对齐）
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(sync_queue_size_),
            *rgb_img_sub_,
            *depth_img_sub_);
        sync_->setMaxIntervalDuration(std::chrono::milliseconds(sync_max_interval_ms_));

        // 7. 创建发布者（QoS配置为可靠传输）
        auto sensor_qos = rclcpp::SensorDataQoS();
        sensor_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);

        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
            publishers_.odom, sensor_qos);
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            publishers_.pose, sensor_qos);
        trajectory_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
            publishers_.trajectory, sensor_qos);

        // 8. 创建TF广播器
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // 9. 注册同步回调
        sync_->registerCallback(std::bind(
            &ORB_SLAM3_ROS2::sync_callback,
            this,
            std::placeholders::_1,
            std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "ORB_SLAM3 node started");
        RCLCPP_INFO(this->get_logger(), "  RGB: %s / Depth: %s",
                    topics_.rgb.c_str(), topics_.depth.c_str());
        RCLCPP_INFO(this->get_logger(), "  Sensor: %s, Viewer: %s",
                    orb_.sensor_type_str.c_str(), orb_.use_viewer ? "ON" : "OFF");
    }

private:
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
        this->declare_parameter<std::string>("odom_topic", "orb_slam3/odom");
        this->declare_parameter<std::string>("pose_topic", "orb_slam3/pose");
        this->declare_parameter<std::string>("trajectory_topic", "orb_slam3/trajectory");

        // ORB-SLAM3核心配置
        this->declare_parameter<std::string>("vocabulary_path",
            "/home/jetson/Workspace/FPV/Thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt");
        this->declare_parameter<std::string>("settings_path",
            "/home/jetson/Workspace/FPV/Node/ORB_Slam3/setting/orbbec_gemini.yaml");
        this->declare_parameter<std::string>("sensor_type", "RGBD");
        this->declare_parameter<bool>("use_viewer", false);

        // 日志与同步器
        this->declare_parameter<int>("log_interval_frames", 300);
        this->declare_parameter<int>("sync_queue_size", 10);
        this->declare_parameter<int>("sync_max_interval_ms", 100);

        // TF查找参数
        this->declare_parameter<int>("tf_retry_count", 10);
        this->declare_parameter<int>("tf_retry_interval_ms", 500);

        // 默认相机外参（base_link -> camera_link 后备值）
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
        this->declare_parameter<std::vector<double>>("trajectory_color", {1.0, 0.0, 0.0});
    }

    // 安全访问vector元素，越界返回默认值
    static double vec_at(const std::vector<double>& v, size_t i, double def)
    {
        return i < v.size() ? v[i] : def;
    }

    // 从ROS参数服务器加载所有参数
    void load_parameters()
    {
        load_from_ros_params();
    }

    void load_from_ros_params()
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

        orb_.vocabulary_path = this->get_parameter("vocabulary_path").as_string();
        orb_.settings_path = this->get_parameter("settings_path").as_string();
        orb_.sensor_type_str = this->get_parameter("sensor_type").as_string();
        orb_.sensor_type = parse_sensor_type(orb_.sensor_type_str);
        orb_.use_viewer = this->get_parameter("use_viewer").as_bool();

        log_interval_ = this->get_parameter("log_interval_frames").as_int();
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
        trajectory_color_r_ = vec_at(traj_color, 0, 1.0);
        trajectory_color_g_ = vec_at(traj_color, 1, 0.0);
        trajectory_color_b_ = vec_at(traj_color, 2, 0.0);

        RCLCPP_INFO(this->get_logger(), "Loaded config from ROS parameters");
    }

    // 传感器类型字符串转枚举
    static ORB_SLAM3::System::eSensor parse_sensor_type(const std::string& s)
    {
        if (s == "MONOCULAR")     return ORB_SLAM3::System::MONOCULAR;
        if (s == "STEREO")        return ORB_SLAM3::System::STEREO;
        if (s == "RGBD")          return ORB_SLAM3::System::RGBD;
        if (s == "IMU_MONOCULAR") return ORB_SLAM3::System::IMU_MONOCULAR;
        if (s == "IMU_STEREO")    return ORB_SLAM3::System::IMU_STEREO;
        if (s == "IMU_RGBD")      return ORB_SLAM3::System::IMU_RGBD;
        return ORB_SLAM3::System::RGBD;
    }

    // 设置默认的base_link -> camera_link变换（无法从TF树获取时使用）
    void set_default_base_to_camera_transform()
    {
        base_to_camera_tf_msgs.header.frame_id = robot_frame_;
        base_to_camera_tf_msgs.child_frame_id = camera_frame_;
        base_to_camera_tf_msgs.transform.translation.x = default_tf_x_;
        base_to_camera_tf_msgs.transform.translation.y = default_tf_y_;
        base_to_camera_tf_msgs.transform.translation.z = default_tf_z_;

        tf2::Quaternion q;
        q.setRPY(default_tf_roll_, default_tf_pitch_, default_tf_yaw_);
        base_to_camera_tf_msgs.transform.rotation.x = q.x();
        base_to_camera_tf_msgs.transform.rotation.y = q.y();
        base_to_camera_tf_msgs.transform.rotation.z = q.z();
        base_to_camera_tf_msgs.transform.rotation.w = q.w();

        RCLCPP_WARN(this->get_logger(), "Using default base_link -> camera_link TF");
    }

    // 发布 world -> odom 静态TF
    void publish_static_world_to_odom()
    {
        tf2::Quaternion quat1, quat2, final_quat;
        quat1.setRPY(0.0, 0.0, 0.0); 
        quat2.setRPY(0.0, 0.0, 0.0);
        final_quat = quat2 * quat1;

        static tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_(this);

        geometry_msgs::msg::TransformStamped static_transform;
        static_transform.header.stamp = this->now();
        static_transform.header.frame_id = world_frame_;
        static_transform.child_frame_id = odom_frame_;
        static_transform.transform.translation.x = 0.0;
        static_transform.transform.translation.y = 0.0;
        static_transform.transform.translation.z = 0.0;
        static_transform.transform.rotation.x = final_quat.x();
        static_transform.transform.rotation.y = final_quat.y();
        static_transform.transform.rotation.z = final_quat.z();
        static_transform.transform.rotation.w = final_quat.w();

        static_tf_broadcaster_.sendTransform(static_transform);
    }

    // ================================================================
    // 同步回调：RGB + Depth 时间对齐后触发
    // ================================================================
    void sync_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_img_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_img_msg)
    {
        // 周期打印同步时间戳
        static int image_img_msg_count = 0;
        image_img_msg_count++;
        if (image_img_msg_count == log_interval_) {
            RCLCPP_INFO(this->get_logger(),
                        "Sync - RGB: %d.%09d, Depth: %d.%09d",
                        rgb_img_msg->header.stamp.sec,
                        rgb_img_msg->header.stamp.nanosec,
                        depth_img_msg->header.stamp.sec,
                        depth_img_msg->header.stamp.nanosec);
            image_img_msg_count = 0;
        }

        // ROS图像消息转cv::Mat（零拷贝，不复制数据）
        cv::Mat rgb_mat = cv::Mat(rgb_img_msg->height, rgb_img_msg->width, CV_8UC3,
                                  const_cast<unsigned char*>(&rgb_img_msg->data[0]));
        cv::Mat depth_mat = cv::Mat(depth_img_msg->height, depth_img_msg->width, CV_16UC1,
                                    const_cast<unsigned char*>(&depth_img_msg->data[0]));

        // 获取当前时间戳（微秒），ORB-SLAM3需要double微秒格式
        auto common_stamp = rgb_img_msg->header.stamp;
        double timestamp_us = static_cast<double>(common_stamp.sec) * 1e6 + static_cast<double>(common_stamp.nanosec) / 1e3;

        if (!rgb_mat.empty() && !depth_mat.empty()) {
            // 调用ORB-SLAM3 RGBD追踪
            Sophus::SE3f camera_pose = SLAM->TrackRGBD(rgb_mat, depth_mat, timestamp_us);
            if (is_valid_pose(camera_pose)) {
                // 将SLAM位姿转换（由于ORB-Slam3输出相机->世界，因此要取反）
                Eigen::Vector3f translation = camera_pose.inverse().translation();
                Eigen::Quaternionf quat = camera_pose.inverse().unit_quaternion();

                // 构建 odom -> camera 的TF变换（orbslam X右Y下Z前 -> ROS X前Y左Z上）
                tf2::Transform odom_to_camera_tf_;
                odom_to_camera_tf_.setOrigin(tf2::Vector3(
                    translation.z(), -translation.x(), -translation.y()));
                odom_to_camera_tf_.setRotation(tf2::Quaternion(
                    quat.z(), -quat.x(), -quat.y(), quat.w()));

                // 结合 camera -> base 变换得到 odom -> base
                tf2::Transform base_to_camera_tf_, camera_to_base_tf_;
                tf2::fromMsg(base_to_camera_tf_msgs.transform, base_to_camera_tf_);
                camera_to_base_tf_ = base_to_camera_tf_.inverse();
                tf2::Transform odom_to_base_tf_ = odom_to_camera_tf_ * camera_to_base_tf_;

                // 发布里程计、TF和轨迹
                publish_odometry(odom_to_base_tf_, rgb_img_msg->header.stamp);
                publish_TF(odom_to_base_tf_, rgb_img_msg->header.stamp);
                update_trajectory(odom_to_base_tf_, rgb_img_msg->header.stamp);
            }
        }
    }

    // 验证位姿是否有效（非NaN、旋转矩阵正交、行列式接近1）
    bool is_valid_pose(const Sophus::SE3f& pose)
    {
        Eigen::Matrix4f matrix = pose.matrix();
        if (!matrix.allFinite()) return false;

        Eigen::Matrix3f R = matrix.block<3, 3>(0, 0);
        Eigen::Matrix3f RRT = R * R.transpose();
        Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
        if ((RRT - I).norm() > 1e-3) return false;
        if (abs(R.determinant() - 1.0f) > 1e-3) return false;

        return true;
    }

    // 发布里程计消息（odom -> base_link）
    void publish_odometry(const tf2::Transform& odom_to_base_tf_,
                          const builtin_interfaces::msg::Time& stamp)
    {
        nav_msgs::msg::Odometry odom_to_base_pose_msgs;
        odom_to_base_pose_msgs.header.stamp = stamp;
        odom_to_base_pose_msgs.header.frame_id = odom_frame_;
        odom_to_base_pose_msgs.child_frame_id = robot_frame_;

        // 位置
        odom_to_base_pose_msgs.pose.pose.position.x = odom_to_base_tf_.getOrigin().x();
        odom_to_base_pose_msgs.pose.pose.position.y = odom_to_base_tf_.getOrigin().y();
        odom_to_base_pose_msgs.pose.pose.position.z = odom_to_base_tf_.getOrigin().z();
        // 姿态（四元数）
        odom_to_base_pose_msgs.pose.pose.orientation.x = odom_to_base_tf_.getRotation().x();
        odom_to_base_pose_msgs.pose.pose.orientation.y = odom_to_base_tf_.getRotation().y();
        odom_to_base_pose_msgs.pose.pose.orientation.z = odom_to_base_tf_.getRotation().z();
        odom_to_base_pose_msgs.pose.pose.orientation.w = odom_to_base_tf_.getRotation().w();

        set_odom_covariance(odom_to_base_pose_msgs);
        odom_publisher_->publish(odom_to_base_pose_msgs);

        // 同时发布 PoseStamped
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = odom_to_base_pose_msgs.header;
        pose_msg.pose = odom_to_base_pose_msgs.pose.pose;
        pose_publisher_->publish(pose_msg);
    }

    // 设置里程计协方差矩阵（6x6对角阵）
    void set_odom_covariance(nav_msgs::msg::Odometry& odom_msg)
    {
        std::fill(odom_msg.pose.covariance.begin(), odom_msg.pose.covariance.end(), 0.0);
        std::fill(odom_msg.twist.covariance.begin(), odom_msg.twist.covariance.end(), 0.0);

        // 位姿协方差 (x, y, z, roll, pitch, yaw)
        odom_msg.pose.covariance[0] = cov_pos_x_;
        odom_msg.pose.covariance[7] = cov_pos_y_;
        odom_msg.pose.covariance[14] = cov_pos_z_;
        odom_msg.pose.covariance[21] = cov_rot_x_;
        odom_msg.pose.covariance[28] = cov_rot_y_;
        odom_msg.pose.covariance[35] = cov_rot_z_;

        // 速度协方差（不使用IMU，设较大值表示不确定）
        odom_msg.twist.covariance[0] = cov_twist_linear_x_;
        odom_msg.twist.covariance[7] = cov_twist_linear_y_;
        odom_msg.twist.covariance[14] = cov_twist_linear_z_;
        odom_msg.twist.covariance[21] = cov_twist_angular_x_;
        odom_msg.twist.covariance[28] = cov_twist_angular_y_;
        odom_msg.twist.covariance[35] = cov_twist_angular_z_;
    }

    // 发布 odom -> base_link 的TF动态变换
    void publish_TF(const tf2::Transform& odom_to_base_tf_,
                    const builtin_interfaces::msg::Time& stamp)
    {
        geometry_msgs::msg::TransformStamped odom_to_base_tf_msgs;
        odom_to_base_tf_msgs.header.stamp = stamp;
        odom_to_base_tf_msgs.header.frame_id = odom_frame_;
        odom_to_base_tf_msgs.child_frame_id = robot_frame_;
        odom_to_base_tf_msgs.transform = tf2::toMsg(odom_to_base_tf_);
        tf_broadcaster_->sendTransform(odom_to_base_tf_msgs);
    }

    // 更新并发布SLAM轨迹（折线Marker）
    void update_trajectory(const tf2::Transform& odom_to_base_tf_,
                           const builtin_interfaces::msg::Time& stamp)
    {
        static visualization_msgs::msg::Marker trajectory;
        static int point_id = 0;

        // 首次初始化Marker属性
        if (point_id == 0) {
            trajectory.header.frame_id = odom_frame_;
            trajectory.header.stamp = stamp;
            trajectory.ns = "slam_trajectory";
            trajectory.id = 0;
            trajectory.type = visualization_msgs::msg::Marker::LINE_STRIP;
            trajectory.action = visualization_msgs::msg::Marker::ADD;
            trajectory.pose.orientation.w = 1.0;
            trajectory.scale.x = trajectory_line_width_;
            trajectory.color.r = trajectory_color_r_;
            trajectory.color.g = trajectory_color_g_;
            trajectory.color.b = trajectory_color_b_;
            trajectory.color.a = 1.0;
        }

        // 添加新轨迹点
        geometry_msgs::msg::Point point;
        point.x = odom_to_base_tf_.getOrigin().x();
        point.y = odom_to_base_tf_.getOrigin().y();
        point.z = odom_to_base_tf_.getOrigin().z();
        trajectory.points.push_back(point);
        trajectory.header.stamp = stamp;

        // 限制轨迹点数量，防止内存无限增长
        if (trajectory.points.size() > static_cast<size_t>(trajectory_max_points_)) {
            trajectory.points.erase(trajectory.points.begin());
        }

        trajectory_publisher_->publish(trajectory);
        point_id++;
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

    struct ORBConfig {
        std::string vocabulary_path;
        std::string settings_path;
        std::string sensor_type_str;
        ORB_SLAM3::System::eSensor sensor_type;
        bool use_viewer;
    };

    // ================================================================
    // 成员变量
    // ================================================================
    Topics topics_;
    Publishers publishers_;
    ORBConfig orb_;

    // 坐标框架
    std::string world_frame_;
    std::string odom_frame_;
    std::string camera_frame_;
    std::string robot_frame_;

    // 参数
    int log_interval_;
    int sync_queue_size_;
    int sync_max_interval_ms_;
    int tf_retry_count_;
    int tf_retry_interval_ms_;

    // 默认TF外参
    double default_tf_x_ = 0.2, default_tf_y_ = 0.0, default_tf_z_ = 0.1;
    double default_tf_roll_ = 0.0, default_tf_pitch_ = 0.0, default_tf_yaw_ = 0.0;

    // 协方差
    double cov_pos_x_ = 0.01, cov_pos_y_ = 0.01, cov_pos_z_ = 0.01;
    double cov_rot_x_ = 0.01, cov_rot_y_ = 0.01, cov_rot_z_ = 0.01;
    double cov_twist_linear_x_ = 0.1, cov_twist_linear_y_ = 0.1, cov_twist_linear_z_ = 0.1;
    double cov_twist_angular_x_ = 0.1, cov_twist_angular_y_ = 0.1, cov_twist_angular_z_ = 0.1;

    // 轨迹可视化
    double trajectory_line_width_;
    int trajectory_max_points_;
    double trajectory_color_r_, trajectory_color_g_, trajectory_color_b_;

    // ORB-SLAM3系统实例
    std::shared_ptr<ORB_SLAM3::System> SLAM = nullptr;

    // 订阅器
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_img_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_img_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // TF
    std::shared_ptr<tf2_ros::Buffer> base_to_camera_tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> base_to_camera_tf_listener_;
    geometry_msgs::msg::TransformStamped base_to_camera_tf_msgs;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // 发布器
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr trajectory_publisher_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ORB_SLAM3_ROS2>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
