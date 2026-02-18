// UNIX
#include <cstdlib>
#include <iostream>
#include <memory>
#include <chrono>
// ROS2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
// ORBBEC
#include "libobsensor/ObSensor.hpp"
#include "orbbec_bridge.h"


class OrbbecCamera : public rclcpp::Node
{
public:
    OrbbecCamera() : Node("orbbec_camera")
    {
        // 创建发布者 - 使用 QoS 策略确保兼容性
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
        auto sensor_qos = rclcpp::SensorDataQoS();
        sensor_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);  // 关键修改
        // int sensor_qos = 10;

        // 创建发布者
        rgb_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "rgb/image_raw", 
            sensor_qos
        );
        rgb_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            "rgb/camera_info", 
            sensor_qos
        );
        depth_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "depth/image_raw", 
            sensor_qos
        );
        depth_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            "depth/camera_info", 
            sensor_qos
        );

        cloud_point_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "cloud_point", 
            sensor_qos
        );

        // 初始化摄像头
        init_camera();

        // 声明相机参数
        declare_camera_parameters();
        // 发布相机TF
        publish_static_tf();
        
        // 创建定时器
        sensor_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(16),  // 60Hz
            std::bind(&OrbbecCamera::sensor_timer_callback, this)
        );
        
        RCLCPP_INFO(this->get_logger(), "Orbbec ROS2节点启动");
    }

private:
    void init_camera()
    {
        // 这里初始化Orbbec相机
        gemini = new Camera(true, 640, 480, OB_FORMAT_RGB888, 60, 
                            false, 640, 480, OB_FORMAT_Y8, 60, 
                            true, 640, 480, OB_FORMAT_Y12, 60);
        gemini -> start();
        auto camera_param = gemini -> pipe -> getCameraParam();
        point_cloud.setCameraParam(camera_param);
        // 实际使用时需要替换为Orbbec SDK的初始化代码
        RCLCPP_INFO(this->get_logger(), "初始化ORBBEC相机");
    }

    void declare_camera_parameters()
    {
        auto rgb_intrinsic = gemini -> get_color_intrinsic();
        auto color_distortion = gemini -> get_color_distortion();
        auto depth_intrinsic = gemini -> get_depth_intrinsic();
        auto depth_distortion = gemini -> get_depth_distortion();

        // 声明相机内参参数（可以使用默认值，后续通过标定更新）
        this->declare_parameter("camera_name", "camera_link");
        
        // 彩色相机内参（640x480分辨率示例，需要根据实际标定修改）
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
        
        // 深度相机内参
        this->declare_parameter("depth.width", rgb_intrinsic.width);      // 防止因高宽比不同导致rtabmap等无法使用，因此用rgb_intrinsic
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
        
        // 加载参数
        camera_name_ = this->get_parameter("camera_name").as_string();
        
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
        
        RCLCPP_INFO(this->get_logger(), "相机参数加载完成");
        RCLCPP_INFO(this->get_logger(), "RGB: %dx%d, fx=%.1f, fy=%.1f", 
                   rgb_width_, rgb_height_, rgb_fx_, rgb_fy_);
        RCLCPP_INFO(this->get_logger(), "Depth: %dx%d, fx=%.1f, fy=%.1f", 
                   depth_width_, depth_height_, depth_fx_, depth_fy_);
    }
    
    void publish_static_tf()
    {
        // 如果map和odom相同，可以发布一个恒等变换
        static tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_(this);
        // TF
        tf2::Quaternion quat1, quat2, final_quat;
        // 第一个旋转
        quat1.setRPY(0.0, M_PI/2, 0.0); 
        // quat1.setRPY(0.0, M_PI/2, 0.0); 
        // 第二个旋转
        quat2.setRPY(-M_PI/2, 0.0, 0.0); 
        // quat2.setRPY(0.0, 0.0, 0.0); 
        // 组合旋转：先绕X轴旋转，再绕Z轴旋转
        // 注意：四元数乘法顺序是 q_final = q2 * q1（先q1后q2）
        final_quat = quat2 * quat1;
        
        geometry_msgs::msg::TransformStamped static_transform;
        static_transform.header.stamp = this->now();
        static_transform.header.frame_id = "base_link";
        static_transform.child_frame_id = camera_name_;
        
        // 单位 M，先平移后旋转
        static_transform.transform.translation.x = 0.00345;
        static_transform.transform.translation.y = 0.0;
        static_transform.transform.translation.z = 0.0038;
        
        static_transform.transform.rotation.x = final_quat.x();
        static_transform.transform.rotation.y = final_quat.y();
        static_transform.transform.rotation.z = final_quat.z();
        static_transform.transform.rotation.w = final_quat.w();
        
        static_tf_broadcaster_.sendTransform(static_transform);
    }

    void sensor_timer_callback()
    {
        auto frame_set = gemini -> get();
        if(frame_set != nullptr){
            // 获取当前时间戳并同步时间戳
            auto common_stamp = this->get_clock()->now();
            // uint64_t sensor_ns = frame_set->timeStamp();
            // rclcpp::Time sensor_stamp(sensor_ns);
            rgb_msg_.header.stamp = common_stamp;
            rgb_info_msg_.header.stamp = common_stamp;
            depth_msg_.header.stamp = common_stamp;
            depth_info_msg_.header.stamp = common_stamp;
            cloud_point_msg_.header.stamp = common_stamp;

            // 获取数据
            get_rgb_image(frame_set);
            get_rgb_camera_info();
            get_depth_image(frame_set);
            get_depth_camera_info();
            get_cloud_point(frame_set);

            // 发布数据
            rgb_publisher_->publish(rgb_msg_);
            rgb_info_publisher_->publish(rgb_info_msg_);
            depth_publisher_->publish(depth_msg_);
            depth_info_publisher_->publish(depth_info_msg_);
            cloud_point_publisher_->publish(cloud_point_msg_);
        }
    }

    void get_rgb_image(std::shared_ptr<ob::FrameSet> frame_set)
    {
        auto rgb_frame = frame_set->colorFrame();
        if(rgb_frame != nullptr){
            cv::Mat rgb_image = gemini -> frame2mat(rgb_frame);
            
            // 设置消息头
            rgb_msg_.header.frame_id = camera_name_;
            
            // 设置图像参数
            rgb_msg_.height = rgb_image.rows;
            rgb_msg_.width = rgb_image.cols;
            rgb_msg_.encoding = "bgr8";  // OpenCV使用BGR格式
            rgb_msg_.is_bigendian = false;
            rgb_msg_.step = static_cast<sensor_msgs::msg::Image::_step_type>(
                rgb_image.step
            );
            
            // 复制图像数据
            size_t size = rgb_image.step * rgb_image.rows;
            rgb_msg_.data.resize(size);
            memcpy(rgb_msg_.data.data(), rgb_image.data, size);
        }
    }

    // 发布RGB相机信息
    void get_rgb_camera_info()
    {   
        // 设置消息头
        rgb_info_msg_.header.frame_id = camera_name_;  // 使用RGB光学坐标系
        
        // 设置图像尺寸
        rgb_info_msg_.height = rgb_height_;
        rgb_info_msg_.width = rgb_width_;
        
        // 设置相机模型（通常是plumb_bob）
        rgb_info_msg_.distortion_model = "plumb_bob";
        
        // 设置内参矩阵 K (3x3 row-major)
        rgb_info_msg_.k = {
            rgb_fx_, 0.0,     rgb_cx_,
            0.0,     rgb_fy_, rgb_cy_,
            0.0,     0.0,     1.0
        };
        
        // 设置畸变参数 D (5x1)
        rgb_info_msg_.d = {rgb_k1_, rgb_k2_, rgb_p1_, rgb_p2_, rgb_k3_};
        
        // 设置投影矩阵 P (3x4 row-major)
        rgb_info_msg_.p = {
            rgb_fx_, 0.0,     rgb_cx_, 0.0,
            0.0,     rgb_fy_, rgb_cy_, 0.0,
            0.0,     0.0,     1.0,     0.0
        };
        
        // 设置旋转矩阵 R (3x3 identity，因为是主相机)
        rgb_info_msg_.r = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        };
        
        // 设置binning和ROI（通常不使用）
        rgb_info_msg_.binning_x = 0;
        rgb_info_msg_.binning_y = 0;
        rgb_info_msg_.roi.x_offset = 0;
        rgb_info_msg_.roi.y_offset = 0;
        rgb_info_msg_.roi.height = 0;
        rgb_info_msg_.roi.width = 0;
        rgb_info_msg_.roi.do_rectify = false;
    }

    void get_depth_image(std::shared_ptr<ob::FrameSet> frame_set)
    {
        auto depth_frame = frame_set->depthFrame();
        if(depth_frame != nullptr){
            cv::Mat depth_image = gemini -> frame2mat(depth_frame);
                    
            // 设置消息头
            depth_msg_.header.frame_id = camera_name_;
            // 设置图像参数
            depth_msg_.height = depth_image.rows;
            depth_msg_.width = depth_image.cols;
            depth_msg_.encoding = "mono16";
            depth_msg_.is_bigendian = false;
            depth_msg_.step = static_cast<sensor_msgs::msg::Image::_step_type>(
                depth_image.step
            );
            
            // 复制图像数据
            size_t size = depth_image.step * depth_image.rows;
            depth_msg_.data.resize(size);
            memcpy(depth_msg_.data.data(), depth_image.data, size);
        }
    }

    // 发布深度相机信息
    void get_depth_camera_info()
    {
        // 设置消息头
        depth_info_msg_.header.frame_id = camera_name_;  // 使用深度光学坐标系
        
        // 设置图像尺寸
        depth_info_msg_.height = depth_height_;
        depth_info_msg_.width = depth_width_;
        
        // 设置相机模型
        depth_info_msg_.distortion_model = "plumb_bob";
        
        // 设置内参矩阵 K
        depth_info_msg_.k = {
            depth_fx_, 0.0,       depth_cx_,
            0.0,       depth_fy_, depth_cy_,
            0.0,       0.0,       1.0
        };
        
        // 设置畸变参数 D
        depth_info_msg_.d = {depth_k1_, depth_k2_, depth_p1_, depth_p2_, depth_k3_};
        
        // 设置投影矩阵 P
        depth_info_msg_.p = {
            depth_fx_, 0.0,       depth_cx_, 0.0,
            0.0,       depth_fy_, depth_cy_, 0.0,
            0.0,       0.0,       1.0,       0.0
        };
        
        // 设置旋转矩阵 R（如果是立体相机，这里需要设置与RGB相机的相对旋转）
        depth_info_msg_.r = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        };
        
        // 设置binning和ROI
        depth_info_msg_.binning_x = 0;
        depth_info_msg_.binning_y = 0;
        depth_info_msg_.roi.x_offset = 0;
        depth_info_msg_.roi.y_offset = 0;
        depth_info_msg_.roi.height = 0;
        depth_info_msg_.roi.width = 0;
        depth_info_msg_.roi.do_rectify = false;
    }

    void get_cloud_point(std::shared_ptr<ob::FrameSet> frame_set)
    {
        auto depthFrame = frame_set->depthFrame();
        if(depthFrame != nullptr){
            auto depthValueScale = depthFrame->getValueScale();
            point_cloud.setPositionDataScaled(depthValueScale);
            point_cloud.setCreatePointFormat(OB_FORMAT_POINT);
            // std::cout << depthValueScale << std::endl;
            std::shared_ptr<ob::Frame> frame = point_cloud.process(frame_set);
            int   pointsSize = frame->dataSize() / sizeof(OBPoint);
            OBPoint* point = (OBPoint *)frame->data();
            // int               validPointsCount = 0;
            static const auto min_distance     = 1e-6;
            // 收集有效点
            std::vector<OBPoint> validPoints;
            validPoints.reserve(pointsSize);
            // First pass: Count valid points (non-zero points)
            for(int i = 0; i < pointsSize; i++) {
                if(point != nullptr && (fabs(point->x) >= min_distance || fabs(point->y) >= min_distance || fabs(point->z) >= min_distance)) {
                    // std::cout << point->x << " " << point->y << " " << point->z << std::endl;
                    validPoints.push_back(*point);
                }
                point++;
            }

            // 设置消息头
            cloud_point_msg_.header.frame_id = camera_name_;
            
            // 设置点云字段（X, Y, Z）
            cloud_point_msg_.height = 1;  // 非有序点云
            cloud_point_msg_.width = validPoints.size();

            // // 定义点云字段
            sensor_msgs::PointCloud2Modifier modifier(cloud_point_msg_);
            modifier.setPointCloud2Fields(
                3,  // 字段数量
                "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                "z", 1, sensor_msgs::msg::PointField::FLOAT32
                // "timestamp", 1, sensor_msgs::msg::PointField::FLOAT32  // 使用"timestamp"字段名
            );
            
            // // 使用迭代器填充数据
            sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_point_msg_, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_point_msg_, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_point_msg_, "z");
            // sensor_msgs::PointCloud2Iterator<float> iter_timestamp(cloud_point_msg_, "timestamp");
            
            for (size_t i = 0; i < validPoints.size(); ++i, ++iter_x, ++iter_y, ++iter_z) {
                *iter_x = validPoints[i].x/1000.0;
                *iter_y = validPoints[i].y/1000.0;
                *iter_z = validPoints[i].z/1000.0;  
            }
                        
            RCLCPP_DEBUG(this->get_logger(), "发布点云，点数: %d", validPoints.size());
        }
    }

    // 添加私有成员变量
    std::string camera_name_;
    
    // RGB相机参数
    int rgb_width_, rgb_height_;
    double rgb_fx_, rgb_fy_, rgb_cx_, rgb_cy_;
    double rgb_k1_, rgb_k2_, rgb_p1_, rgb_p2_, rgb_k3_;
    
    // 深度相机参数
    int depth_width_, depth_height_;
    double depth_fx_, depth_fy_, depth_cx_, depth_cy_;
    double depth_k1_, depth_k2_, depth_p1_, depth_p2_, depth_k3_;

    // 添加发布者
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr rgb_info_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_point_publisher_;

    // 定时器
    rclcpp::TimerBase::SharedPtr tf_timer_;
    rclcpp::TimerBase::SharedPtr sensor_timer_;

    // 添加消息
    sensor_msgs::msg::Image rgb_msg_;
    sensor_msgs::msg::CameraInfo rgb_info_msg_;
    sensor_msgs::msg::Image depth_msg_;
    sensor_msgs::msg::CameraInfo depth_info_msg_;
    sensor_msgs::msg::PointCloud2 cloud_point_msg_;

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