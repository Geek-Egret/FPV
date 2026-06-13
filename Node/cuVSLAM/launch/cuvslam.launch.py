#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # ============================================================
    # cuVSLAM RGBD 节点参数 - 在此修改所有参数
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
    odom_topic = "cuvslam/odom"
    pose_topic = "cuvslam/pose"
    trajectory_topic = "cuvslam/trajectory"

    # --- 相机内参 (Orbbec Gemini FPV参数) ---
    camera_width = 640
    camera_height = 480
    camera_fx = 455.483
    camera_fy = 455.483
    camera_cx = 329.67
    camera_cy = 243.265

    # Brown畸变模型 (k1,k2,k3=径向, p1,p2=切向)
    # 全部设为0.0即使用Pinhole无畸变模型
    camera_k1 = 0.05382
    camera_k2 = -0.0703176
    camera_p1 = -0.0000575311
    camera_p2 = -0.000674939
    camera_k3 = 0.0

    # --- RGBD ---
    depth_scale_factor = 1000.0  # 深度值除数: mm转m用1000.0, float米用1.0

    # --- cuVSLAM 算法参数 ---
    use_denoising = False
    rectified_stereo = False
    enable_observations_export = True
    enable_landmarks_export = True
    num_desired_tracks = 400

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
    trajectory_color = [0.0, 1.0, 0.0]  # 绿色

    # --- 日志 ---
    log_interval_frames = 300

    # ============================================================
    # 节点定义
    # ============================================================

    cuvslam_node = Node(
        package="cuvslam",
        executable="cuvslam",
        name="cuvslam",
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
            "camera.width": camera_width,
            "camera.height": camera_height,
            "camera.fx": camera_fx,
            "camera.fy": camera_fy,
            "camera.cx": camera_cx,
            "camera.cy": camera_cy,
            "camera.k1": camera_k1,
            "camera.k2": camera_k2,
            "camera.p1": camera_p1,
            "camera.p2": camera_p2,
            "camera.k3": camera_k3,
            "depth.scale_factor": depth_scale_factor,
            "use_denoising": use_denoising,
            "rectified_stereo": rectified_stereo,
            "enable_observations_export": enable_observations_export,
            "enable_landmarks_export": enable_landmarks_export,
            "num_desired_tracks": num_desired_tracks,
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

    return LaunchDescription([cuvslam_node])
