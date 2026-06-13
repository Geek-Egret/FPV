/**
 * @file orbbec_camera.cpp
 * @brief Orbbec RGBD相机 ROS2 驱动节点
 *
 * 通过OrbbecSDK驱动相机，发布RGB图像、深度图像、点云和CameraInfo。
 * 同时发布 base_link -> camera_link 的静态TF变换。
 * 所有参数通过ROS2参数系统配置，由Python启动脚本传入；
 * 相机内参在初始化后从硬件读取并注册到参数服务器。
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <chrono>

// ROS2 核心
#include <rclcpp/rclcpp.hpp>
// 消息类型
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/header.hpp>
// TF2
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
// OpenCV
#include <opencv2/opencv.hpp>
// OrbbecSDK + Bridge封装
#include "libobsensor/ObSensor.hpp"
#include "orbbec_bridge.h"


class OrbbecCamera : public rclcpp::Node
{
public:
    OrbbecCamera() : Node("orbbec_camera")
    {
        // 1. 声明并加载ROS参数
        declare_all_parameters();
        load_parameters();

        // 2. 创建发布者（QoS配置为可靠传输）
        auto sensor_qos = rclcpp::SensorDataQoS();
        sensor_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);

        rgb_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            topics_.rgb, sensor_qos);
        rgb_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            topics_.rgb_info, sensor_qos);
        depth_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            topics_.depth, sensor_qos);
        depth_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            topics_.depth_info, sensor_qos);
        cloud_point_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            topics_.cloud_point, sensor_qos);

        // 3. 初始化相机硬件
        init_camera();

        // 4. 从硬件读取内参并注册到参数服务器
        declare_camera_parameters();

        // 5. 发布静态TF（base_link -> camera_link）
        static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        publish_static_tf();

        // 6. 创建定时器，定期采集图像
        sensor_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timer_interval_ms_),
            std::bind(&OrbbecCamera::sensor_timer_callback, this));

        RCLCPP_INFO(this->get_logger(), "Orbbec Camera node started");
        RCLCPP_INFO(this->get_logger(), "  RGB topic: %s", topics_.rgb.c_str());
        RCLCPP_INFO(this->get_logger(), "  Depth topic: %s", topics_.depth.c_str());
        RCLCPP_INFO(this->get_logger(), "  Cloud topic: %s", topics_.cloud_point.c_str());
        RCLCPP_INFO(this->get_logger(), "  Timer: %dms (%dHz)",
                    timer_interval_ms_, 1000 / timer_interval_ms_);
    }

private:
    // ================================================================
    // 参数声明 - 用户可调参数
    // ================================================================
    void declare_all_parameters()
    {
        // 发布话题
        this->declare_parameter<std::string>("rgb_topic", "rgb/image_raw");
        this->declare_parameter<std::string>("rgb_info_topic", "rgb/camera_info");
        this->declare_parameter<std::string>("depth_topic", "depth/image_raw");
        this->declare_parameter<std::string>("depth_info_topic", "depth/camera_info");
        this->declare_parameter<std::string>("cloud_point_topic", "cloud_point");

        // 坐标系
        this->declare_parameter<std::string>("camera_name", "camera_link");
        this->declare_parameter<std::string>("camera_optical_name", "camera_optical_frame");
        this->declare_parameter<std::string>("parent_frame", "base_link");

        // TF变换（相机在机器人上的安装位姿）
        this->declare_parameter<std::vector<double>>("tf_translation", {0.00345, 0.0, 0.0038});
        this->declare_parameter<std::vector<double>>("tf_rotation_rpy", {0.0, M_PI / 2.0, -M_PI / 2.0});

        // 定时器与点云
        this->declare_parameter<int>("timer_interval_ms", 16);
        this->declare_parameter<double>("pointcloud_min_distance", 1e-6);

        // 相机初始化分辨率与帧率
        this->declare_parameter<int>("camera.rgb_width", 640);
        this->declare_parameter<int>("camera.rgb_height", 480);
        this->declare_parameter<int>("camera.rgb_fps", 60);
        this->declare_parameter<int>("camera.depth_width", 640);
        this->declare_parameter<int>("camera.depth_height", 400);
        this->declare_parameter<int>("camera.depth_fps", 60);
    }

    // 从ROS参数服务器加载用户可调参数
    void load_parameters()
    {
        topics_.rgb = this->get_parameter("rgb_topic").as_string();
        topics_.rgb_info = this->get_parameter("rgb_info_topic").as_string();
        topics_.depth = this->get_parameter("depth_topic").as_string();
        topics_.depth_info = this->get_parameter("depth_info_topic").as_string();
        topics_.cloud_point = this->get_parameter("cloud_point_topic").as_string();

        camera_name_ = this->get_parameter("camera_name").as_string();
        camera_optical_name_ = this->get_parameter("camera_optical_name").as_string();
        parent_frame_ = this->get_parameter("parent_frame").as_string();

        auto tf_trans = this->get_parameter("tf_translation").as_double_array();
        tf_x_ = tf_trans.size() > 0 ? tf_trans[0] : 0.00345;
        tf_y_ = tf_trans.size() > 1 ? tf_trans[1] : 0.0;
        tf_z_ = tf_trans.size() > 2 ? tf_trans[2] : 0.0038;

        auto tf_rpy = this->get_parameter("tf_rotation_rpy").as_double_array();
        tf_roll_ = tf_rpy.size() > 0 ? tf_rpy[0] : 0.0;
        tf_pitch_ = tf_rpy.size() > 1 ? tf_rpy[1] : M_PI / 2.0;
        tf_yaw_ = tf_rpy.size() > 2 ? tf_rpy[2] : -M_PI / 2.0;

        timer_interval_ms_ = this->get_parameter("timer_interval_ms").as_int();
        min_distance_ = this->get_parameter("pointcloud_min_distance").as_double();

        rgb_w_ = this->get_parameter("camera.rgb_width").as_int();
        rgb_h_ = this->get_parameter("camera.rgb_height").as_int();
        rgb_fps_ = this->get_parameter("camera.rgb_fps").as_int();
        depth_w_ = this->get_parameter("camera.depth_width").as_int();
        depth_h_ = this->get_parameter("camera.depth_height").as_int();
        depth_fps_ = this->get_parameter("camera.depth_fps").as_int();
    }

    // 初始化Orbbec相机硬件
    // Camera构造参数: (enable_rgb, w, h, format, fps, enable_ir, w_ir, h_ir, format_ir, fps_ir, enable_depth, w_depth, h_depth, format_depth, fps_depth)
    void init_camera()
    {
        gemini = new Camera(true, rgb_w_, rgb_h_, OB_FORMAT_RGB888, rgb_fps_,
                            false, rgb_w_, rgb_h_, OB_FORMAT_Y8, rgb_fps_,
                            true, depth_w_, depth_h_, OB_FORMAT_Y12, depth_fps_);
        gemini->start();
        auto camera_param = gemini->pipe->getCameraParam();
        point_cloud.setCameraParam(camera_param);
        RCLCPP_INFO(this->get_logger(), "Camera initialized: RGB %dx%d@%d, Depth %dx%d@%d",
                    rgb_w_, rgb_h_, rgb_fps_, depth_w_, depth_h_, depth_fps_);
    }

    // 从相机硬件读取内参和畸变参数，注册到ROS参数服务器
    void declare_camera_parameters()
    {
        auto rgb_intrinsic = gemini->get_color_intrinsic();
        auto color_distortion = gemini->get_color_distortion();
        auto depth_intrinsic = gemini->get_depth_intrinsic();
        auto depth_distortion = gemini->get_depth_distortion();

        // RGB相机内参
        this->declare_parameter("rgb.width", rgb_intrinsic.width);
        this->declare_parameter("rgb.height", rgb_intrinsic.height);
        this->declare_parameter("rgb.fx", rgb_intrinsic.fx);
        this->declare_parameter("rgb.fy", rgb_intrinsic.fy);
        this->declare_parameter("rgb.cx", rgb_intrinsic.cx);
        this->declare_parameter("rgb.cy", rgb_intrinsic.cy);
        this->declare_parameter("rgb.k1", color_distortion.k1);
        this->declare_parameter("rgb.k2", color_distortion.k2);
        this->declare_parameter("rgb.p1", color_distortion.p1);
        this->declare_parameter("rgb.p2", color_distortion.p2);
        this->declare_parameter("rgb.k3", color_distortion.k3);

        // 深度相机内参（宽高使用RGB的，保证对齐）
        this->declare_parameter("depth.width", rgb_intrinsic.width);
        this->declare_parameter("depth.height", rgb_intrinsic.height);
        this->declare_parameter("depth.fx", depth_intrinsic.fx);
        this->declare_parameter("depth.fy", depth_intrinsic.fy);
        this->declare_parameter("depth.cx", depth_intrinsic.cx);
        this->declare_parameter("depth.cy", depth_intrinsic.cy);
        this->declare_parameter("depth.k1", depth_distortion.k1);
        this->declare_parameter("depth.k2", depth_distortion.k2);
        this->declare_parameter("depth.p1", depth_distortion.p1);
        this->declare_parameter("depth.p2", depth_distortion.p2);
        this->declare_parameter("depth.k3", depth_distortion.k3);

        // 加载到本地变量
        rgb_width_ = this->get_parameter("rgb.width").as_int();
        rgb_height_ = this->get_parameter("rgb.height").as_int();
        rgb_fx_ = this->get_parameter("rgb.fx").as_double();
        rgb_fy_ = this->get_parameter("rgb.fy").as_double();
        rgb_cx_ = this->get_parameter("rgb.cx").as_double();
        rgb_cy_ = this->get_parameter("rgb.cy").as_double();
        rgb_k1_ = this->get_parameter("rgb.k1").as_double();
        rgb_k2_ = this->get_parameter("rgb.k2").as_double();
        rgb_p1_ = this->get_parameter("rgb.p1").as_double();
        rgb_p2_ = this->get_parameter("rgb.p2").as_double();
        rgb_k3_ = this->get_parameter("rgb.k3").as_double();

        depth_width_ = this->get_parameter("depth.width").as_int();
        depth_height_ = this->get_parameter("depth.height").as_int();
        depth_fx_ = this->get_parameter("depth.fx").as_double();
        depth_fy_ = this->get_parameter("depth.fy").as_double();
        depth_cx_ = this->get_parameter("depth.cx").as_double();
        depth_cy_ = this->get_parameter("depth.cy").as_double();
        depth_k1_ = this->get_parameter("depth.k1").as_double();
        depth_k2_ = this->get_parameter("depth.k2").as_double();
        depth_p1_ = this->get_parameter("depth.p1").as_double();
        depth_p2_ = this->get_parameter("depth.p2").as_double();
        depth_k3_ = this->get_parameter("depth.k3").as_double();

        RCLCPP_INFO(this->get_logger(), "Camera params loaded");
        RCLCPP_INFO(this->get_logger(), "  RGB: %dx%d fx=%.1f fy=%.1f",
                    rgb_width_, rgb_height_, rgb_fx_, rgb_fy_);
        RCLCPP_INFO(this->get_logger(), "  Depth: %dx%d fx=%.1f fy=%.1f",
                    depth_width_, depth_height_, depth_fx_, depth_fy_);
    }

    // 发布 base_link -> camera_link 和 camera_link -> camera_optical_frame 静态TF
    void publish_static_tf()
    {
        // TF1: base_link -> camera_link（用户可调参数）
        tf2::Quaternion q1;
        q1.setRPY(tf_roll_, tf_pitch_, tf_yaw_);

        geometry_msgs::msg::TransformStamped tf_base_cam;
        tf_base_cam.header.stamp = this->now();
        tf_base_cam.header.frame_id = parent_frame_;
        tf_base_cam.child_frame_id = camera_name_;
        tf_base_cam.transform.translation.x = tf_x_;
        tf_base_cam.transform.translation.y = tf_y_;
        tf_base_cam.transform.translation.z = tf_z_;
        tf_base_cam.transform.rotation.x = q1.x();
        tf_base_cam.transform.rotation.y = q1.y();
        tf_base_cam.transform.rotation.z = q1.z();
        tf_base_cam.transform.rotation.w = q1.w();
        static_tf_broadcaster_->sendTransform(tf_base_cam);

        // TF2: camera_link -> camera_optical_frame（固定值，不可调）
        tf2::Quaternion q2;
        q2.setRPY(-M_PI_2, 0.0, -M_PI_2);

        geometry_msgs::msg::TransformStamped tf_cam_opt;
        tf_cam_opt.header.stamp = this->now();
        tf_cam_opt.header.frame_id = camera_name_;
        tf_cam_opt.child_frame_id = camera_optical_name_;
        tf_cam_opt.transform.translation.x = 0.0;
        tf_cam_opt.transform.translation.y = 0.0;
        tf_cam_opt.transform.translation.z = 0.0;
        tf_cam_opt.transform.rotation.x = q2.x();
        tf_cam_opt.transform.rotation.y = q2.y();
        tf_cam_opt.transform.rotation.z = q2.z();
        tf_cam_opt.transform.rotation.w = q2.w();
        static_tf_broadcaster_->sendTransform(tf_cam_opt);
    }

    // ================================================================
    // 定时器回调：采集一帧RGB、深度、点云并发布
    // ================================================================
    void sensor_timer_callback()
    {
        // 从相机获取一帧数据
        auto frame_set = gemini->get();
        if (frame_set == nullptr) return;

        // 统一时间戳
        auto common_stamp = this->get_clock()->now();
        rgb_msg_.header.stamp = common_stamp;
        rgb_info_msg_.header.stamp = common_stamp;
        depth_msg_.header.stamp = common_stamp;
        depth_info_msg_.header.stamp = common_stamp;
        cloud_point_msg_.header.stamp = common_stamp;

        // 提取各数据类型
        get_rgb_image(frame_set);
        get_rgb_camera_info();
        get_depth_image(frame_set);
        get_depth_camera_info();
        get_cloud_point(frame_set);

        // 发布所有数据
        rgb_publisher_->publish(rgb_msg_);
        rgb_info_publisher_->publish(rgb_info_msg_);
        depth_publisher_->publish(depth_msg_);
        depth_info_publisher_->publish(depth_info_msg_);
        cloud_point_publisher_->publish(cloud_point_msg_);
    }

    // 提取RGB图像并转为ROS消息（BGR8编码）
    void get_rgb_image(std::shared_ptr<ob::FrameSet> frame_set)
    {
        auto rgb_frame = frame_set->colorFrame();
        if (rgb_frame == nullptr) return;

        cv::Mat rgb_image = gemini->frame2mat(rgb_frame);
        rgb_msg_.header.frame_id = camera_name_;
        rgb_msg_.height = rgb_image.rows;
        rgb_msg_.width = rgb_image.cols;
        rgb_msg_.encoding = "bgr8";  // OpenCV默认BGR格式
        rgb_msg_.is_bigendian = false;
        rgb_msg_.step = static_cast<sensor_msgs::msg::Image::_step_type>(rgb_image.step);

        size_t size = rgb_image.step * rgb_image.rows;
        rgb_msg_.data.resize(size);
        memcpy(rgb_msg_.data.data(), rgb_image.data, size);
    }

    // 构建RGB相机内参信息（CameraInfo消息）
    void get_rgb_camera_info()
    {
        rgb_info_msg_.header.frame_id = camera_name_;
        rgb_info_msg_.height = rgb_height_;
        rgb_info_msg_.width = rgb_width_;
        rgb_info_msg_.distortion_model = "plumb_bob";  // Brown畸变模型

        // 内参矩阵 K (3x3 row-major)
        rgb_info_msg_.k = {rgb_fx_, 0.0, rgb_cx_, 0.0, rgb_fy_, rgb_cy_, 0.0, 0.0, 1.0};
        // 畸变参数 D (k1, k2, p1, p2, k3)
        rgb_info_msg_.d = {rgb_k1_, rgb_k2_, rgb_p1_, rgb_p2_, rgb_k3_};
        // 投影矩阵 P (3x4)
        rgb_info_msg_.p = {rgb_fx_, 0.0, rgb_cx_, 0.0, 0.0, rgb_fy_, rgb_cy_, 0.0, 0.0, 0.0, 1.0, 0.0};
        // 旋转矩阵 R (单位矩阵，主相机无旋转)
        rgb_info_msg_.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

        rgb_info_msg_.binning_x = 0;
        rgb_info_msg_.binning_y = 0;
        rgb_info_msg_.roi.x_offset = 0;
        rgb_info_msg_.roi.y_offset = 0;
        rgb_info_msg_.roi.height = 0;
        rgb_info_msg_.roi.width = 0;
        rgb_info_msg_.roi.do_rectify = false;
    }

    // 提取深度图像并转为ROS消息（mono16编码，单位mm）
    void get_depth_image(std::shared_ptr<ob::FrameSet> frame_set)
    {
        auto depth_frame = frame_set->depthFrame();
        if (depth_frame == nullptr) return;

        cv::Mat depth_image = gemini->frame2mat(depth_frame);
        depth_msg_.header.frame_id = camera_name_;
        depth_msg_.height = depth_image.rows;
        depth_msg_.width = depth_image.cols;
        depth_msg_.encoding = "mono16";  // 16位无符号整数，单位mm
        depth_msg_.is_bigendian = false;
        depth_msg_.step = static_cast<sensor_msgs::msg::Image::_step_type>(depth_image.step);

        size_t size = depth_image.step * depth_image.rows;
        depth_msg_.data.resize(size);
        memcpy(depth_msg_.data.data(), depth_image.data, size);
    }

    // 构建深度相机内参信息（CameraInfo消息）
    void get_depth_camera_info()
    {
        depth_info_msg_.header.frame_id = camera_name_;
        depth_info_msg_.height = depth_height_;
        depth_info_msg_.width = depth_width_;
        depth_info_msg_.distortion_model = "plumb_bob";

        depth_info_msg_.k = {depth_fx_, 0.0, depth_cx_, 0.0, depth_fy_, depth_cy_, 0.0, 0.0, 1.0};
        depth_info_msg_.d = {depth_k1_, depth_k2_, depth_p1_, depth_p2_, depth_k3_};
        depth_info_msg_.p = {depth_fx_, 0.0, depth_cx_, 0.0, 0.0, depth_fy_, depth_cy_, 0.0, 0.0, 0.0, 1.0, 0.0};
        depth_info_msg_.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

        depth_info_msg_.binning_x = 0;
        depth_info_msg_.binning_y = 0;
        depth_info_msg_.roi.x_offset = 0;
        depth_info_msg_.roi.y_offset = 0;
        depth_info_msg_.roi.height = 0;
        depth_info_msg_.roi.width = 0;
        depth_info_msg_.roi.do_rectify = false;
    }

    // 从深度帧生成点云（过滤无效点，坐标单位mm转m）
    void get_cloud_point(std::shared_ptr<ob::FrameSet> frame_set)
    {
        auto depthFrame = frame_set->depthFrame();
        if (depthFrame == nullptr) return;

        // 设置深度缩放和点云格式
        auto depthValueScale = depthFrame->getValueScale();
        point_cloud.setPositionDataScaled(depthValueScale);
        point_cloud.setCreatePointFormat(OB_FORMAT_POINT);

        // 生成点云
        std::shared_ptr<ob::Frame> frame = point_cloud.process(frame_set);
        int pointsSize = frame->dataSize() / sizeof(OBPoint);
        OBPoint* point = (OBPoint*)frame->data();

        // 过滤无效点（距离小于阈值的点）
        std::vector<OBPoint> validPoints;
        validPoints.reserve(pointsSize);
        for (int i = 0; i < pointsSize; i++) {
            if (point != nullptr &&
                (fabs(point->x) >= min_distance_ ||
                 fabs(point->y) >= min_distance_ ||
                 fabs(point->z) >= min_distance_)) {
                validPoints.push_back(*point);
            }
            point++;
        }

        // 构建ROS点云消息
        cloud_point_msg_.header.frame_id = camera_optical_name_;
        cloud_point_msg_.height = 1;  // 无序点云
        cloud_point_msg_.width = validPoints.size();

        // 定义字段：X, Y, Z
        sensor_msgs::PointCloud2Modifier modifier(cloud_point_msg_);
        modifier.setPointCloud2Fields(
            3,
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32);

        // 填充点云数据（毫米转米）
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_point_msg_, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_point_msg_, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_point_msg_, "z");

        for (size_t i = 0; i < validPoints.size(); ++i, ++iter_x, ++iter_y, ++iter_z) {
            *iter_x = validPoints[i].x / 1000.0;
            *iter_y = validPoints[i].y / 1000.0;
            *iter_z = validPoints[i].z / 1000.0;
        }

        RCLCPP_DEBUG(this->get_logger(), "Published %zu cloud points", validPoints.size());
    }

    // ================================================================
    // 数据结构
    // ================================================================
    struct Topics {
        std::string rgb;
        std::string rgb_info;
        std::string depth;
        std::string depth_info;
        std::string cloud_point;
    };
    Topics topics_;

    // ================================================================
    // 成员变量
    // ================================================================
    // 坐标系
    std::string camera_name_;
    std::string camera_optical_name_;
    std::string parent_frame_;

    // 相机初始化参数（分辨率、帧率）
    int rgb_w_, rgb_h_, rgb_fps_;
    int depth_w_, depth_h_, depth_fps_;

    // RGB相机内参（从硬件读取）
    int rgb_width_, rgb_height_;
    double rgb_fx_, rgb_fy_, rgb_cx_, rgb_cy_;
    double rgb_k1_, rgb_k2_, rgb_p1_, rgb_p2_, rgb_k3_;

    // 深度相机内参（从硬件读取）
    int depth_width_, depth_height_;
    double depth_fx_, depth_fy_, depth_cx_, depth_cy_;
    double depth_k1_, depth_k2_, depth_p1_, depth_p2_, depth_k3_;

    // TF外参（用户配置）
    double tf_x_, tf_y_, tf_z_;
    double tf_roll_, tf_pitch_, tf_yaw_;

    // 定时器与点云参数
    int timer_interval_ms_;
    double min_distance_;

    // 发布器
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr rgb_info_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_point_publisher_;

    // 定时器与TF
    rclcpp::TimerBase::SharedPtr sensor_timer_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

    // 消息缓存（复用避免重复分配内存）
    sensor_msgs::msg::Image rgb_msg_;
    sensor_msgs::msg::CameraInfo rgb_info_msg_;
    sensor_msgs::msg::Image depth_msg_;
    sensor_msgs::msg::CameraInfo depth_info_msg_;
    sensor_msgs::msg::PointCloud2 cloud_point_msg_;

    // 相机实例
    Camera* gemini = nullptr;
    ob::PointCloudFilter point_cloud;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OrbbecCamera>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
