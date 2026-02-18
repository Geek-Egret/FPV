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
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sophus/se3.hpp>
// ORB-SLAM3
#include "orb_slam3.h"

// 解决Eigen内存对齐问题（关键！）
#include <Eigen/Dense>
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SE3f)

// 定义同步策略类型（简化后续代码）
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;

class ORB_SLAM3_ROS2 : public rclcpp::Node
{
public:

    ORB_SLAM3_ROS2() : Node("orb_slam3")
    {
        world_frame_ = "map";                   // ORB-SLAM3的世界坐标系
        camera_frame_ = "camera_link";          // 相机坐标系
        robot_frame_ = "base_link";             // 基座坐标系
        odom_frame_ = "odom";                   // 中间里程计坐标系

        publish_static_map_to_odom();

        // 尝试获取 camera_link 到 base_link 的静态变换
        base_to_camera_tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        base_to_camera_tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*base_to_camera_tf_buffer_);
        try {
            // 等待TF可用
            base_to_camera_tf_buffer_->canTransform(camera_frame_, robot_frame_, tf2::TimePointZero, tf2::durationFromSec(1.0));
            base_to_camera_tf_msgs = base_to_camera_tf_buffer_->lookupTransform(camera_frame_, robot_frame_, tf2::TimePointZero);
            RCLCPP_INFO(this->get_logger(), "成功获取 base_link 到 %s 的静态变换", camera_frame_.c_str());
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "无法获取 base_link 到 %s 的变换: %s", 
                       camera_frame_.c_str(), ex.what());
            // 设置默认变换（如果获取失败）
            set_default_base_to_camera_transform();
        }

        // 创建message_filters订阅器
        rgb_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, "/rgb/image_raw");  // 图像话题
        depth_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, "/depth/image_raw");  // 图像话题

        SLAM = std::make_shared<ORB_SLAM3::System>("/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Thirdparty/ORB_Slam3/Vocabulary/ORBvoc.txt","/home/leeeezy/Workspace/GEEK-EGRET-FPV-SDK/Node/ORB_Slam3/Workspace/setting/orbbec_gemini.yaml",ORB_SLAM3::System::RGBD,false);

        // 创建同步器
        // 参数说明：队列大小、两个订阅器、节点上下文
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10),  // 队列大小10，可根据需求调整
            *rgb_img_sub_, 
            *depth_img_sub_
        );
        
        // 设置时间差阈值（可选，默认0.1秒）
        sync_->setMaxIntervalDuration(std::chrono::milliseconds(10));  // 允许100ms内的时间差
        
        // 创建发布者 - 使用 QoS 策略确保兼容性
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
        auto sensor_qos = rclcpp::SensorDataQoS();
        sensor_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);  // 关键修改
        
        // 发布
        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "orb_slam3/odom", 
            sensor_qos
        );
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "orb_slam3/pose", 
            sensor_qos
        );
        trajectory_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "orb_slam3/trajectory", 
            sensor_qos
        );

        // TF广播器
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        

        // 注册同步回调函数
        sync_->registerCallback(std::bind(
            &ORB_SLAM3_ROS2::sync_callback, 
            this, 
            std::placeholders::_1, 
            std::placeholders::_2)
        );

        RCLCPP_INFO(this->get_logger(), "同步订阅节点已启动！");
    }

private:
    void set_default_base_to_camera_transform()
    {
        base_to_camera_tf_msgs.header.frame_id = robot_frame_;
        base_to_camera_tf_msgs.child_frame_id = camera_frame_;
        
        // 这里设置默认的变换参数，根据您的实际安装位置调整
        base_to_camera_tf_msgs.transform.translation.x = 0.2;   // 假设相机在前方0.2米
        base_to_camera_tf_msgs.transform.translation.y = 0.0;   // 中心位置
        base_to_camera_tf_msgs.transform.translation.z = 0.1;   // 高度0.1米
        
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, 0.0);  // 假设无旋转
        base_to_camera_tf_msgs.transform.rotation.x = q.x();
        base_to_camera_tf_msgs.transform.rotation.y = q.y();
        base_to_camera_tf_msgs.transform.rotation.z = q.z();
        base_to_camera_tf_msgs.transform.rotation.w = q.w();
        
        RCLCPP_WARN(this->get_logger(), "使用默认的 base_link 到 camera_link 变换");
    }

    void publish_static_map_to_odom()
    {
        // TF
        tf2::Quaternion quat1, quat2, final_quat;
        // 第一个旋转
        quat1.setRPY(0.0, M_PI/2, 0.0); 
        // 第二个旋转
        quat2.setRPY(-M_PI/2, 0.0, 0.0); 
        // 组合旋转：先绕X轴旋转，再绕Z轴旋转
        // 注意：四元数乘法顺序是 q_final = q2 * q1（先q1后q2）
        final_quat = quat2 * quat1;

        // 如果map和odom相同，可以发布一个恒等变换
        static tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_(this);
        
        geometry_msgs::msg::TransformStamped static_transform;
        static_transform.header.stamp = this->now();
        static_transform.header.frame_id = "map";
        static_transform.child_frame_id = "odom";
        
        // 单位 M，先平移后旋转
        static_transform.transform.translation.x = 0.0;
        static_transform.transform.translation.y = 0.0;
        static_transform.transform.translation.z = 0.0;
        
        static_transform.transform.rotation.x = final_quat.x();
        static_transform.transform.rotation.y = final_quat.y();
        static_transform.transform.rotation.z = final_quat.z();
        static_transform.transform.rotation.w = final_quat.w();
        
        static_tf_broadcaster_.sendTransform(static_transform);
    }
    
    // 同步回调函数：两个消息时间戳匹配时触发
    void sync_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_img_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_img_msg)
    {
        static int image_img_msg_count = 0;
        image_img_msg_count++;
        if(image_img_msg_count == 300){
            // 打印同步后的时间戳
            RCLCPP_INFO(this->get_logger(), 
            "同步消息 - 彩色图时间戳: %ld.%09ld, 深度图时间戳: %ld.%09ld",
            rgb_img_msg->header.stamp.sec, rgb_img_msg->header.stamp.nanosec,
            depth_img_msg->header.stamp.sec, depth_img_msg->header.stamp.nanosec);
            image_img_msg_count = 0;
        }

        // msg 转为cv::Mat
        cv::Mat rgb_mat = cv::Mat(rgb_img_msg->height, rgb_img_msg->width, CV_8UC3, 
                    const_cast<unsigned char*>(&rgb_img_msg->data[0]));
        cv::Mat depth_mat = cv::Mat(depth_img_msg->height, depth_img_msg->width, CV_16UC1,
                    const_cast<unsigned char*>(&depth_img_msg->data[0]));

        // 获取消息时间戳
        auto common_stamp = this->get_clock()->now();
        double timestamp_us = common_stamp.seconds()*1e6;
        if(!rgb_mat.empty() && !depth_mat.empty()){
            Sophus::SE3f camera_pose = SLAM->TrackRGBD(rgb_mat,depth_mat,timestamp_us);
            if (is_valid_pose(camera_pose)) {
                // 从Sophus::SE3f中提取平移和旋转
                // 使用位姿的逆
                Eigen::Vector3f translation = camera_pose.inverse().translation();
                Eigen::Quaternionf quat = camera_pose.inverse().unit_quaternion();
                tf2::Transform odom_to_camera_tf_;
                odom_to_camera_tf_.setOrigin(tf2::Vector3(
                    translation.x(),
                    translation.y(),
                    translation.z()));
                odom_to_camera_tf_.setRotation(tf2::Quaternion(
                    quat.x(),
                    quat.y(),
                    quat.z(),
                    quat.w()));
                
                // 将 base_to_camera_tf_msgs 转为 tf2::Transform
                tf2::Transform camera_to_base_tf_;
                tf2::fromMsg(base_to_camera_tf_msgs.transform, camera_to_base_tf_);

                // 相乘得到 odom -> base
                tf2::Transform odom_to_base_tf_ = odom_to_camera_tf_ * camera_to_base_tf_;

                // 转换为ROS2消息并发布
                publish_odometry(odom_to_base_tf_, rgb_img_msg->header.stamp);
                publish_TF(odom_to_base_tf_, rgb_img_msg->header.stamp);
                update_trajectory(odom_to_base_tf_, rgb_img_msg->header.stamp);
            } 
        }
    }

    // 位姿是否合法
     bool is_valid_pose(const Sophus::SE3f& pose)
    {
        // 检查位姿是否有效（非零或非奇异）
        Eigen::Matrix4f matrix = pose.matrix();
        
        // 检查是否为无效值（NaN或无穷大）
        if (!matrix.allFinite()) {
            return false;
        }
        
        // 检查旋转部分是否为有效的旋转矩阵
        Eigen::Matrix3f R = matrix.block<3, 3>(0, 0);
        Eigen::Matrix3f RRT = R * R.transpose();
        Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
        
        // 检查是否接近单位矩阵
        if ((RRT - I).norm() > 1e-3) {
            return false;
        }
        
        // 检查行列式是否接近1
        if (abs(R.determinant() - 1.0f) > 1e-3) {
            return false;
        }
        
        return true;
    }
    
    // 发布里程计
    void publish_odometry(const tf2::Transform odom_to_base_tf_, const builtin_interfaces::msg::Time& stamp)
    {        
        // 创建里程计消息
        nav_msgs::msg::Odometry odom_to_base_pose_msgs;
        odom_to_base_pose_msgs.header.stamp = stamp;
        odom_to_base_pose_msgs.header.frame_id = odom_frame_;
        odom_to_base_pose_msgs.child_frame_id = robot_frame_;

        // 正确：Pose 使用 position
        odom_to_base_pose_msgs.pose.pose.position.x = odom_to_base_tf_.getOrigin().x();
        odom_to_base_pose_msgs.pose.pose.position.y = odom_to_base_tf_.getOrigin().y();
        odom_to_base_pose_msgs.pose.pose.position.z = odom_to_base_tf_.getOrigin().z();

        // 正确：四元数赋值
        odom_to_base_pose_msgs.pose.pose.orientation.x = odom_to_base_tf_.getRotation().x();
        odom_to_base_pose_msgs.pose.pose.orientation.y = odom_to_base_tf_.getRotation().y();
        odom_to_base_pose_msgs.pose.pose.orientation.z = odom_to_base_tf_.getRotation().z();
        odom_to_base_pose_msgs.pose.pose.orientation.w = odom_to_base_tf_.getRotation().w();    

        // 设置协方差（根据SLAM精度调整）
        set_odom_covariance(odom_to_base_pose_msgs);
        
        // 发布里程计
        odom_publisher_->publish(odom_to_base_pose_msgs);
        
        // 同时发布PoseStamped（可选）
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = odom_to_base_pose_msgs.header;                 
        pose_msg.pose = odom_to_base_pose_msgs.pose.pose;
        pose_publisher_->publish(pose_msg);
    }
    
    void set_odom_covariance(nav_msgs::msg::Odometry& odom_msg)
    {
        // 重置协方差矩阵
        std::fill(odom_msg.pose.covariance.begin(), odom_msg.pose.covariance.end(), 0.0);
        std::fill(odom_msg.twist.covariance.begin(), odom_msg.twist.covariance.end(), 0.0);
        
        // 设置位置协方差（根据SLAM的置信度调整）
        // 较小的值表示较高的置信度
        odom_msg.pose.covariance[0] = 0.01;  // x
        odom_msg.pose.covariance[7] = 0.01;  // y
        odom_msg.pose.covariance[14] = 0.01; // z
        
        // 设置旋转协方差
        odom_msg.pose.covariance[21] = 0.01; // 绕x轴旋转
        odom_msg.pose.covariance[28] = 0.01; // 绕y轴旋转
        odom_msg.pose.covariance[35] = 0.01; // 绕z轴旋转
        
        // 速度协方差（如果不使用IMU，可以设置较大值）
        odom_msg.twist.covariance[0] = 0.1;  // 线速度x
        odom_msg.twist.covariance[7] = 0.1;  // 线速度y
        odom_msg.twist.covariance[14] = 0.1; // 线速度z
        odom_msg.twist.covariance[21] = 0.1; // 角速度x
        odom_msg.twist.covariance[28] = 0.1; // 角速度y
        odom_msg.twist.covariance[35] = 0.1; // 角速度z
    }
    
    // 发布TF变换
    void publish_TF(const tf2::Transform odom_to_base_tf_, const builtin_interfaces::msg::Time& stamp)
    {
        
        // 创建TF变换
        geometry_msgs::msg::TransformStamped odom_to_base_tf_msgs;
        odom_to_base_tf_msgs.header.stamp = stamp;
        odom_to_base_tf_msgs.header.frame_id = odom_frame_;
        odom_to_base_tf_msgs.child_frame_id = robot_frame_;

        // 转回 TransformStamped
        odom_to_base_tf_msgs.transform = tf2::toMsg(odom_to_base_tf_);

        // 广播TF变换
        tf_broadcaster_->sendTransform(odom_to_base_tf_msgs);
    }
    
    void update_trajectory(const tf2::Transform odom_to_base_tf_, const builtin_interfaces::msg::Time& stamp)
    {
        // 可选：发布轨迹可视化
        static visualization_msgs::msg::Marker trajectory;
        static int point_id = 0;
        
        if (point_id == 0) {
            // 初始化轨迹marker
            trajectory.header.frame_id = odom_frame_;
            trajectory.header.stamp = stamp;
            trajectory.ns = "slam_trajectory";
            trajectory.id = 0;
            trajectory.type = visualization_msgs::msg::Marker::LINE_STRIP;
            trajectory.action = visualization_msgs::msg::Marker::ADD;
            trajectory.pose.orientation.w = 1.0;
            trajectory.scale.x = 0.05;  // 线宽
            trajectory.color.r = 1.0;
            trajectory.color.g = 0.0;
            trajectory.color.b = 0.0;
            trajectory.color.a = 1.0;
        }

        // 添加新的轨迹点
        geometry_msgs::msg::Point point;
        point.x = odom_to_base_tf_.getOrigin().x();
        point.y = odom_to_base_tf_.getOrigin().y();
        point.z = odom_to_base_tf_.getOrigin().z();
        
        trajectory.points.push_back(point);
        trajectory.header.stamp = stamp;
        
        // 限制轨迹点数量
        if (trajectory.points.size() > 1000) {
            trajectory.points.erase(trajectory.points.begin());
        }
        
        // 发布轨迹
        trajectory_publisher_->publish(trajectory);
        point_id++;
    }

    // 参数
    std::string world_frame_;   // ORB-SLAM3的世界坐标系（通常是"map"）
    std::string odom_frame_;    // 里程计坐标系（ROS中的odom框架）
    std::string camera_frame_;   // 相机坐标系
    std::string robot_frame_;   // 机器人坐标系

    // 订阅
    std::shared_ptr<ORB_SLAM3::System> SLAM = nullptr;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_img_sub_;        
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_img_sub_;    
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    // TF
    std::shared_ptr<tf2_ros::Buffer> base_to_camera_tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> base_to_camera_tf_listener_;
    geometry_msgs::msg::TransformStamped base_to_camera_tf_msgs;
    // 发布
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr trajectory_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;                            
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ORB_SLAM3_ROS2>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
