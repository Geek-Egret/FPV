#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # ============================================================
    # 参数配置 - 在此修改所有可调参数
    # ============================================================

    # --- 订阅话题 ---
    rgb_topic = "/rgb/image_raw"
    depth_topic = "/depth/image_raw"

    # --- 坐标系 ---
    world_frame = "world"
    camera_frame = "camera_link"
    robot_frame = "base_link"
    odom_frame = "odom"

    # --- 发布话题 ---
    odom_topic = "orb_slam3/odom"
    pose_topic = "orb_slam3/pose"
    trajectory_topic = "orb_slam3/trajectory"

    # --- ORB_SLAM3 核心设置 ---
    vocabulary_path = "/home/jetson/Workspace/FPV/Thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    settings_path = "/home/jetson/Workspace/FPV/Node/ORB_Slam3/setting/orbbec_gemini.yaml"
    sensor_type = "RGBD"  # MONOCULAR / STEREO / RGBD / IMU_MONOCULAR / IMU_STEREO / IMU_RGBD
    use_viewer = False

    # --- 同步器 ---
    sync_queue_size = 10
    sync_max_interval_ms = 100

    # --- TF ---
    tf_retry_count = 10
    tf_retry_interval_ms = 500

    # --- 默认相机外参 (base_link -> camera_link 后备) ---
    default_tf_translation = [0.2, 0.0, 0.1]
    default_tf_rpy = [0.0, 0.0, 0.0]

    # --- 协方差 ---
    pose_covariance = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    twist_covariance = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # --- 轨迹可视化 ---
    trajectory_line_width = 0.05
    trajectory_max_points = 1000
    trajectory_color = [1.0, 0.0, 0.0]  # RGB

    # --- 点云 ---

    # --- 日志 ---
    log_interval_frames = 300

    # ============================================================
    # 节点定义
    # ============================================================

    orb_slam3_node = Node(
        package="orb_slam3",
        executable="orb_slam3",
        name="orb_slam3",
        output="screen",
        parameters=[{
            "rgb_topic": rgb_topic,
            "depth_topic": depth_topic,
            "world_frame": world_frame,
            "camera_frame": camera_frame,
            "robot_frame": robot_frame,
            "odom_frame": odom_frame,
            "odom_topic": odom_topic,
            "pose_topic": pose_topic,
            "trajectory_topic": trajectory_topic,
            "vocabulary_path": vocabulary_path,
            "settings_path": settings_path,
            "sensor_type": sensor_type,
            "use_viewer": use_viewer,
            "sync_queue_size": sync_queue_size,
            "sync_max_interval_ms": sync_max_interval_ms,
            "tf_retry_count": tf_retry_count,
            "tf_retry_interval_ms": tf_retry_interval_ms,
            "default_tf_translation": default_tf_translation,
            "default_tf_rpy": default_tf_rpy,
            "pose_covariance": pose_covariance,
            "twist_covariance": twist_covariance,
            "trajectory_line_width": trajectory_line_width,
            "trajectory_max_points": trajectory_max_points,
            "trajectory_color": trajectory_color,
            "log_interval_frames": log_interval_frames,
        }],
    )

    return LaunchDescription([orb_slam3_node])
