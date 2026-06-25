from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = {
        "odom_topic": "cuvslam/odom",
        "vision_topic": "/mavros/vision_pose/pose",
        "pos_cmd_topic": "/position_cmd",
        "setpoint_topic": "/mavros/setpoint_raw/local",
        "world_frame": "odom",
        "enable_odom_forward": "true",
        "enable_cmd_forward": "true",
        "feed_forward_vel": "false",
        "feed_forward_acc": "false",
        "feed_forward_yaw_rate": "false",
    }
    return LaunchDescription([
        DeclareLaunchArgument(k, default_value=v) for k, v in args.items()
    ] + [
        Node(
            package="px4_bridge", executable="px4_bridge", name="px4_bridge",
            output="screen",
            parameters=[{k: LaunchConfiguration(k) for k in args}],
        )
    ])
