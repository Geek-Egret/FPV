from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'target_frame',
            default_value='world',
            description='Target frame for odometry'
        ),
        DeclareLaunchArgument(
            'source_frame',
            default_value='camera_optical_frame',
            description='Source frame to transform from'
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/vins_estimator/odometry',
            description='Output odometry topic'
        ),
        DeclareLaunchArgument(
            'child_frame_id',
            default_value='base_link',
            description='Child frame ID for odometry'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='30.0',
            description='Publishing rate in Hz'
        ),
        Node(
            package='tf2_odom',
            executable='tf2_odom',
            name='tf2_odom',
            output='screen',
            parameters=[{
                'target_frame': LaunchConfiguration('target_frame'),
                'source_frame': LaunchConfiguration('source_frame'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'child_frame_id': LaunchConfiguration('child_frame_id'),
                'publish_rate': LaunchConfiguration('publish_rate'),
            }]
        )
    ])