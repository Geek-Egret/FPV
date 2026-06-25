#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # ============================================================
    # cuVSLAM RGBD Node Parameters - Configure everything here
    # ============================================================

    # --- Topics ---
    rgb_topic = "/rgb/image_raw"
    depth_topic = "/depth/image_raw"

    # --- Frames ---
    world_frame = "world"
    camera_frame = "camera_link"
    robot_frame = "base_link"
    odom_frame = "odom"

    # --- Published topics ---
    odom_topic = "cuvslam/odom"
    pose_topic = "cuvslam/pose"
    trajectory_topic = "cuvslam/trajectory"
    slam_pose_topic = "cuvslam/slam_pose"   # SLAM 后端校正位姿

    # --- Camera intrinsics (Orbbec Gemini FPV) ---
    camera_width = 640
    camera_height = 480
    camera_fx = 455.483
    camera_fy = 455.483
    camera_cx = 329.67
    camera_cy = 243.265
    # Brown distortion (k1/k2/k3 radial, p1/p2 tangential; all 0 = Pinhole)
    camera_k1 = 0.05382
    camera_k2 = -0.0703176
    camera_p1 = -0.0000575311
    camera_p2 = -0.000674939
    camera_k3 = 0.0

    # --- RGBD depth ---
    depth_scale_factor = 1000.0     # divisor: mm->m=1000.0, float-m=1.0
    depth_camera_id = 0

    # --- Odometry::Config ---
    use_gpu = True
    use_motion_model = True
    use_denoising = False
    rectified_stereo = False
    enable_observations_export = True
    enable_landmarks_export = True
    enable_final_landmarks_export = False
    async_sba = True
    max_frame_delta_s = 0.02
    ransac_filter = True
    kf_survivor_from_last_pct = 70.0
    kf_max_timedelta_s = 120
    num_desired_tracks = 400
    border_top = 0
    border_bottom = 0
    border_left = 0
    border_right = 0

    # --- Slam::Config ---
    slam_sync_mode = False
    slam_planar_constraints = False
    slam_enable_reading_internals = True
    slam_throttling_time_ms = 1000
    slam_retention_time_ms = 5000
    slam_max_map_size = 300
    # slam_map_cache_path = "/home/jetson/Workspace/FPV/Map/cuvslam_cache"

    # --- Synchronizer ---
    sync_queue_size = 10
    sync_max_interval_ms = 100

    # --- TF ---
    tf_retry_count = 10
    tf_retry_interval_ms = 500
    default_tf_translation = [0.2, 0.0, 0.1]
    default_tf_rpy = [0.0, 0.0, 0.0]

    # --- Covariance ---
    pose_covariance = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    twist_covariance = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # --- Trajectory visualization ---
    trajectory_line_width = 0.05
    trajectory_max_points = 1000
    trajectory_color = [0.0, 1.0, 0.0]

    # --- SLAM 后端位姿 ---
    use_slam_pose = False       # True=用SLAM校正位姿发布odom/TF, False=用原始VO

    # --- Logging ---
    log_interval_frames = 300

    # ============================================================
    # Node definition
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
            "slam_pose_topic": slam_pose_topic,
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
            "depth.camera_id": depth_camera_id,
            "use_gpu": use_gpu,
            "use_motion_model": use_motion_model,
            "use_denoising": use_denoising,
            "rectified_stereo": rectified_stereo,
            "enable_observations_export": enable_observations_export,
            "enable_landmarks_export": enable_landmarks_export,
            "enable_final_landmarks_export": enable_final_landmarks_export,
            "async_sba": async_sba,
            "max_frame_delta_s": max_frame_delta_s,
            "ransac_filter": ransac_filter,
            "kf_survivor_from_last_pct": kf_survivor_from_last_pct,
            "kf_max_timedelta_s": kf_max_timedelta_s,
            "num_desired_tracks": num_desired_tracks,
            "border_top": border_top,
            "border_bottom": border_bottom,
            "border_left": border_left,
            "border_right": border_right,
            "slam.sync_mode": slam_sync_mode,
            "slam.planar_constraints": slam_planar_constraints,
            "slam.enable_reading_internals": slam_enable_reading_internals,
            "slam.throttling_time_ms": slam_throttling_time_ms,
            "slam.retention_time_ms": slam_retention_time_ms,
            "slam.max_map_size": slam_max_map_size,
            "slam.map_cache_path": "",
            "use_slam_pose": use_slam_pose,
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
