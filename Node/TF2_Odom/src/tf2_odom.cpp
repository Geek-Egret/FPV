#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <memory>
#include <chrono>
#include <cmath>

using namespace std::chrono_literals;

class TF2OdometryPublisher : public rclcpp::Node
{
public:
    TF2OdometryPublisher() : Node("tf2_odometry_publisher")
    {
        // 声明参数
        this->declare_parameter<std::string>("target_frame", "world");
        this->declare_parameter<std::string>("source_frame", "camera_optical_frame");
        this->declare_parameter<std::string>("odom_topic", "/vins_estimator/odometry");
        this->declare_parameter<std::string>("child_frame_id", "base_link");
        this->declare_parameter<double>("publish_rate", 30.0);
        
        // 获取参数
        target_frame_ = this->get_parameter("target_frame").as_string();
        source_frame_ = this->get_parameter("source_frame").as_string();
        odom_topic_ = this->get_parameter("odom_topic").as_string();
        child_frame_id_ = this->get_parameter("child_frame_id").as_string();
        double publish_rate = this->get_parameter("publish_rate").as_double();
        
        // 初始化TF2
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // 创建发布者
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            odom_topic_, 
            rclcpp::QoS(10).reliable()
        );
        
        // 创建定时器
        double timer_period = 1.0 / publish_rate;
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(timer_period),
            std::bind(&TF2OdometryPublisher::timerCallback, this)
        );
        
        // 初始化协方差矩阵
        setupCovariance();
        
        // 统计信息
        publish_count_ = 0;
        fail_count_ = 0;
        
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "TF2OdometryPublisher 已启动");
        RCLCPP_INFO(this->get_logger(), "  目标坐标系: %s", target_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  源坐标系: %s", source_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  子坐标系: %s", child_frame_id_.c_str());
        RCLCPP_INFO(this->get_logger(), "  输出话题: %s", odom_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  发布频率: %.1f Hz", publish_rate);
        RCLCPP_INFO(this->get_logger(), "========================================");
    }

private:
    void setupCovariance()
    {
        // 设置位姿协方差
        for (int i = 0; i < 36; ++i) {
            pose_covariance_[i] = 0.0;
            twist_covariance_[i] = 0.0;
        }
        
        // 位置协方差 (0.01)
        pose_covariance_[0] = 0.01;   // x
        pose_covariance_[7] = 0.01;   // y
        pose_covariance_[14] = 0.01;  // z
        
        // 姿态协方差 (0.01)
        pose_covariance_[21] = 0.01;  // roll
        pose_covariance_[28] = 0.01;  // pitch
        pose_covariance_[35] = 0.01;  // yaw
        
        // 速度协方差 (0.01)
        twist_covariance_[0] = 0.01;
        twist_covariance_[7] = 0.01;
        twist_covariance_[14] = 0.01;
        twist_covariance_[21] = 0.01;
        twist_covariance_[28] = 0.01;
        twist_covariance_[35] = 0.01;
    }
    
    bool getTransform(geometry_msgs::msg::TransformStamped& transform)
    {
        try {
            // 查找最新的变换
            transform = tf_buffer_->lookupTransform(
                target_frame_,
                source_frame_,
                tf2::TimePointZero  // 使用最新可用的变换
            );
            return true;
        } catch (const tf2::TransformException& ex) {
            fail_count_++;
            if (fail_count_ % 50 == 0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "TF查询失败 (%d次): %s", fail_count_, ex.what());
            }
            return false;
        }
    }
    
    void transformToOdometry(const geometry_msgs::msg::TransformStamped& transform,
                            nav_msgs::msg::Odometry& odom_msg)
    {
        // 设置header
        odom_msg.header.stamp = transform.header.stamp;
        odom_msg.header.frame_id = target_frame_;
        
        // 设置child_frame_id
        odom_msg.child_frame_id = child_frame_id_;
        
        // 设置位置
        odom_msg.pose.pose.position.x = transform.transform.translation.x;
        odom_msg.pose.pose.position.y = transform.transform.translation.y;
        odom_msg.pose.pose.position.z = transform.transform.translation.z;
        
        // 设置姿态
        odom_msg.pose.pose.orientation = transform.transform.rotation;
        
        // 设置协方差
        for (int i = 0; i < 36; ++i) {
            odom_msg.pose.covariance[i] = pose_covariance_[i];
            odom_msg.twist.covariance[i] = twist_covariance_[i];
        }
        
        // 速度信息留空（TF变换不包含速度）
        odom_msg.twist.twist.linear.x = 0.0;
        odom_msg.twist.twist.linear.y = 0.0;
        odom_msg.twist.twist.linear.z = 0.0;
        odom_msg.twist.twist.angular.x = 0.0;
        odom_msg.twist.twist.angular.y = 0.0;
        odom_msg.twist.twist.angular.z = 0.0;
    }
    
    void timerCallback()
    {
        geometry_msgs::msg::TransformStamped transform;
        
        // 获取TF变换
        if (!getTransform(transform)) {
            return;
        }
        
        // 检查是否是新的变换（避免重复发布）
        if (last_stamp_ == transform.header.stamp) {
            return;
        }
        
        // 转换为Odometry并发布
        nav_msgs::msg::Odometry odom_msg;
        transformToOdometry(transform, odom_msg);
        odom_pub_->publish(odom_msg);
        
        publish_count_++;
        last_stamp_ = transform.header.stamp;
        
        // 定期输出统计信息
        if (publish_count_ % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), 
                "已发布 %d 条消息, TF失败次数: %d",
                publish_count_, fail_count_);
        }
    }
    
private:
    // ROS2组件
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // 参数
    std::string target_frame_;
    std::string source_frame_;
    std::string odom_topic_;
    std::string child_frame_id_;
    
    // 协方差矩阵
    double pose_covariance_[36];
    double twist_covariance_[36];
    
    // 统计信息
    int publish_count_;
    int fail_count_;
    builtin_interfaces::msg::Time last_stamp_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TF2OdometryPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}