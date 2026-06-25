#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <quadrotor_msgs/msg/position_command.hpp>
#include <mavros_msgs/msg/position_target.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <chrono>
#include <cmath>

class PX4Bridge : public rclcpp::Node
{
public:
    PX4Bridge() : Node("px4_bridge"), has_cmd_(false), has_odom_(false)
    {
        declare_parameters();
        load_parameters();

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 10,
            std::bind(&PX4Bridge::odom_callback, this, std::placeholders::_1));
        vision_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            vision_topic_, 10);

        pos_cmd_sub_ = this->create_subscription<quadrotor_msgs::msg::PositionCommand>(
            pos_cmd_topic_, 10,
            std::bind(&PX4Bridge::pos_cmd_callback, this, std::placeholders::_1));
        setpoint_pub_ = this->create_publisher<mavros_msgs::msg::PositionTarget>(
            setpoint_topic_, 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&PX4Bridge::timer_callback, this));

        RCLCPP_INFO(this->get_logger(), "PX4 Bridge started");
    }

private:
    void declare_parameters()
    {
        this->declare_parameter<std::string>("odom_topic", "cuvslam/odom");
        this->declare_parameter<std::string>("vision_topic", "/mavros/vision_pose/pose");
        this->declare_parameter<std::string>("pos_cmd_topic", "/position_cmd");
        this->declare_parameter<std::string>("setpoint_topic", "/mavros/setpoint_raw/local");
        this->declare_parameter<std::string>("world_frame", "odom");
        this->declare_parameter<bool>("enable_odom_forward", true);
        this->declare_parameter<bool>("enable_cmd_forward", true);
        this->declare_parameter<bool>("feed_forward_vel", false);
        this->declare_parameter<bool>("feed_forward_acc", false);
        this->declare_parameter<bool>("feed_forward_yaw_rate", false);
    }

    void load_parameters()
    {
        odom_topic_ = this->get_parameter("odom_topic").as_string();
        vision_topic_ = this->get_parameter("vision_topic").as_string();
        pos_cmd_topic_ = this->get_parameter("pos_cmd_topic").as_string();
        setpoint_topic_ = this->get_parameter("setpoint_topic").as_string();
        world_frame_ = this->get_parameter("world_frame").as_string();
        enable_odom_ = this->get_parameter("enable_odom_forward").as_bool();
        enable_cmd_ = this->get_parameter("enable_cmd_forward").as_bool();
        ff_vel_ = this->get_parameter("feed_forward_vel").as_bool();
        ff_acc_ = this->get_parameter("feed_forward_acc").as_bool();
        ff_yaw_rate_ = this->get_parameter("feed_forward_yaw_rate").as_bool();
    }

    int compute_type_mask() const
    {
        int mask = 0;
        if (!ff_vel_)       mask |= 0b00111000;
        if (!ff_acc_)       mask |= 0b00111000000;
        if (!ff_yaw_rate_)  mask |= 0b10000000000;
        return mask;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        if (!enable_odom_) return;

        geometry_msgs::msg::PoseStamped vision;
        vision.header.stamp = msg->header.stamp;
        vision.header.frame_id = world_frame_;
        vision.pose.position = msg->pose.pose.position;
        vision.pose.orientation = msg->pose.pose.orientation;
        vision_pub_->publish(vision);

        latest_odom_ = *msg;
        has_odom_ = true;
    }

    void pos_cmd_callback(const quadrotor_msgs::msg::PositionCommand::SharedPtr msg)
    {
        if (!enable_cmd_) return;

        publish_setpoint(
            msg->position.x, msg->position.y, msg->position.z,
            msg->velocity.x, msg->velocity.y, msg->velocity.z,
            msg->acceleration.x, msg->acceleration.y, msg->acceleration.z,
            msg->yaw, msg->yaw_dot, msg->header.stamp);

        latest_cmd_ = *msg;
        has_cmd_ = true;
    }

    void timer_callback()
    {
        if (!enable_cmd_) return;

        auto now = this->now();

        if (has_cmd_)
        {
            publish_setpoint(
                latest_cmd_.position.x, latest_cmd_.position.y, latest_cmd_.position.z,
                latest_cmd_.velocity.x, latest_cmd_.velocity.y, latest_cmd_.velocity.z,
                latest_cmd_.acceleration.x, latest_cmd_.acceleration.y, latest_cmd_.acceleration.z,
                latest_cmd_.yaw, latest_cmd_.yaw_dot, now);
        }
        else if (has_odom_)
        {
            double qx = latest_odom_.pose.pose.orientation.x;
            double qy = latest_odom_.pose.pose.orientation.y;
            double qz = latest_odom_.pose.pose.orientation.z;
            double qw = latest_odom_.pose.pose.orientation.w;
            double siny_cosp = 2.0 * (qw * qz + qx * qy);
            double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
            double yaw = std::atan2(siny_cosp, cosy_cosp);

            publish_setpoint(
                latest_odom_.pose.pose.position.x,
                latest_odom_.pose.pose.position.y,
                latest_odom_.pose.pose.position.z,
                0, 0, 0, 0, 0, 0, yaw, 0, now);
        }
    }

    void publish_setpoint(double x, double y, double z,
                          double vx, double vy, double vz,
                          double ax, double ay, double az,
                          double yaw, double yaw_dot, rclcpp::Time stamp)
    {
        mavros_msgs::msg::PositionTarget sp;
        sp.header.stamp = stamp;
        sp.header.frame_id = world_frame_;
        sp.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;
        sp.type_mask = compute_type_mask();

        sp.position.x = x;
        sp.position.y = y;
        sp.position.z = z;
        sp.velocity.x = vx;
        sp.velocity.y = vy;
        sp.velocity.z = vz;
        sp.acceleration_or_force.x = ax;
        sp.acceleration_or_force.y = ay;
        sp.acceleration_or_force.z = az;
        sp.yaw = yaw;
        sp.yaw_rate = yaw_dot;

        setpoint_pub_->publish(sp);
    }

    std::string odom_topic_, vision_topic_, pos_cmd_topic_, setpoint_topic_, world_frame_;
    bool enable_odom_, enable_cmd_;
    bool ff_vel_, ff_acc_, ff_yaw_rate_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<quadrotor_msgs::msg::PositionCommand>::SharedPtr pos_cmd_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr vision_pub_;
    rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr setpoint_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    quadrotor_msgs::msg::PositionCommand latest_cmd_;
    nav_msgs::msg::Odometry latest_odom_;
    bool has_cmd_, has_odom_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PX4Bridge>());
    rclcpp::shutdown();
    return 0;
}
