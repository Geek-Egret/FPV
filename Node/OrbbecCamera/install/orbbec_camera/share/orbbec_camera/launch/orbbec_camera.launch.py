#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
import math


def generate_launch_description():
    # ============================================================
    # Orbbec Camera 节点参数 - 在此修改所有参数
    # ============================================================

    # --- 发布话题 ---
    rgb_topic = "rgb/image_raw"
    rgb_info_topic = "rgb/camera_info"
    depth_topic = "depth/image_raw"
    depth_info_topic = "depth/camera_info"
    cloud_point_topic = "cloud_point"

    # --- 坐标系 ---
    camera_name = "camera_link"
    camera_optical_name = "camera_optical_frame"
    parent_frame = "base_link"

    # --- TF变换 (base_link -> camera_link) ---
    tf_translation = [0.00345, 0.0, 0.0038]
    tf_rotation_rpy = [0.0 , 0.0, 0.0]
    # tf_rotation_rpy = [-math.pi / 2.0, math.pi / 2.0, 0.0]

    # --- 定时器 (ms, 越小频率越高) ---
    timer_interval_ms = 16  # ~60Hz

    # --- 点云最小距离过滤 (米) ---
    pointcloud_min_distance = 1e-6

    # --- 相机初始化参数 ---
    camera_rgb_width = 640
    camera_rgb_height = 480
    camera_rgb_fps = 60
    camera_depth_width = 640
    camera_depth_height = 400
    camera_depth_fps = 60

    # ============================================================
    # 节点定义
    # ============================================================

    orbbec_node = Node(
        package="orbbec_camera",
        executable="orbbec_camera",
        name="orbbec_camera",
        output="screen",
        parameters=[{
            "rgb_topic": rgb_topic,
            "rgb_info_topic": rgb_info_topic,
            "depth_topic": depth_topic,
            "depth_info_topic": depth_info_topic,
            "cloud_point_topic": cloud_point_topic,
            "camera_name": camera_name,
            "camera_optical_name": camera_optical_name,
            "parent_frame": parent_frame,
            "tf_translation": tf_translation,
            "tf_rotation_rpy": tf_rotation_rpy,
            "timer_interval_ms": timer_interval_ms,
            "pointcloud_min_distance": pointcloud_min_distance,
            "camera.rgb_width": camera_rgb_width,
            "camera.rgb_height": camera_rgb_height,
            "camera.rgb_fps": camera_rgb_fps,
            "camera.depth_width": camera_depth_width,
            "camera.depth_height": camera_depth_height,
            "camera.depth_fps": camera_depth_fps,
        }],
    )

    return LaunchDescription([orbbec_node])
