import argparse
import os

import numpy as np
import math

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind
import cv2

from drone import controller as sim
from drone import camera

KEY_DPOS = 0.1
KEY_DANGLE = 0.1

NUM_CYLINDERS = 8
NUM_BOXES = 6
CYLINDER_RING_RADIUS = 0.3
BOX_RING_RADIUS = 0.3

def main():
    parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Camera Visualization with Keyboard Teleop")
    parser.add_argument("-B", "--n_envs", type=int, default=0, help="Number of environments to replicate")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 3.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=viewer_options,
        show_viewer=True,
    )

    # for i in range(NUM_CYLINDERS):
    #     angle = 2 * np.pi * i / NUM_CYLINDERS
    #     x = CYLINDER_RING_RADIUS * np.cos(angle)*10
    #     y = CYLINDER_RING_RADIUS * np.sin(angle)*10
    #     scene.add_entity(
    #         gs.morphs.Cylinder(
    #             height=1.5,
    #             radius=0.3,
    #             pos=(x, y, 0.75),
    #             fixed=True,
    #         )
    #     )

    # for i in range(NUM_BOXES):
    #     angle = 2 * np.pi * i / NUM_BOXES + np.pi / 6
    #     x = BOX_RING_RADIUS * np.cos(angle)*10
    #     y = BOX_RING_RADIUS * np.sin(angle)*10
    #     scene.add_entity(
    #         gs.morphs.Box(
    #             size=(0.5, 0.5, 2.0 * (i + 1) / NUM_BOXES),
    #             pos=(x, y, 1.0),
    #             fixed=True,
    #         )
    #     )

    # entity_kwargs = dict(
    #     pos=(0.0, 0.0, 0.35),
    #     quat=(1.0, 0.0, 0.0, 0.0),
    #     fixed=True,
    # )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="ge_fpv.urdf",
            pos=(0.0, 0.0, 0.0),
        ),
    )
    drone.propellers_spin

    ##########################  sensor ##########################
    # depth
    depth = scene.add_camera(
        res=(640, 400),         # 分辨率（宽，高）
        pos=(0.0, 0.0, 0.5),         # 相机位置
        lookat=(1, 0, 0),       # 注视点
        fov=60,                 # 垂直视野角度，默认30度
        up=(0, 0, 1),           # 向上向量
        model="pinhole",        # 相机模型（pinhole或thinlens）
        GUI=True                # 图像显示
    )
    # # depth
    # sensor_kwargs = dict(
    #     entity_idx=drone.idx,
    #     pos_offset=(0.0425, 0.0, 0.0345),
    #     euler_offset=(0.0, 0.0, 0.0),
    #     return_world_frame=False,
    #     draw_debug=False,        # 显示点
    #     min_range=0.25,          # 最小检测范围 (m)
    #     max_range=2.5,         # 最大检测范围 (m)
    # )
    # depth = scene.add_sensor(gs.sensors.DepthCamera(pattern=gs.sensors.DepthCameraPattern(), **sensor_kwargs))

    ########################## build ##########################
    scene.build(n_envs=args.n_envs)

    ########################## control ##########################
    controller = sim.sim_controller(entity=drone, frame="base")

    ########################## viewer ##########################
    # viewer = scene.visualizer.viewer
    # # 让视角跟随无人机
    # viewer.follow_entity(
    #     entity=drone,
    #     fixed_axis=(None, None, 1.5),   # 高度固定在2米
    #     smoothing=0.1,
    #     fix_orientation=False
    # )

    ########################## init ##########################
    init_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    init_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def apply_pose_to_all_envs(pos_np: np.ndarray, quat_np: np.ndarray):
        if args.n_envs > 0:
            pos_np = np.expand_dims(pos_np, axis=0).repeat(args.n_envs, axis=0)
            quat_np = np.expand_dims(quat_np, axis=0).repeat(args.n_envs, axis=0)
        drone.set_pos(pos_np)
        drone.set_quat(quat_np)

    # Define control callbacks
    def reset_pose():
        apply_pose_to_all_envs(init_pos, gu.euler_to_quat(init_euler))

    # Register keybindings
    scene.viewer.register_keybinds(
        Keybind("reset", Key.BACKSLASH, KeyAction.HOLD, callback=reset_pose),
    )

    # Print controls
    print("[\\]: Reset")
    apply_pose_to_all_envs(init_pos, gu.euler_to_quat(init_euler))

    while(True):
        # 更新深度相机位姿
        pos_new, lookat_new, up_new = camera.update_pos(entity=drone, pos_offset=(0.0425, 0.0, 0.0345), forward=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)) # (0.0425, 0.0, 0.0345)
        depth.set_pose(
            pos = pos_new,
            lookat = lookat_new,
            up = up_new
        )
        rgb_img, depth_img, _, _ = depth.render(rgb=True, depth=True)
        # if rgb_img.dtype != np.uint8:
        #     rgb_img = (rgb_img * 255).astype(np.uint8)
        
        # # 转换BGR到RGB（OpenCV使用BGR）
        # rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        # # 显示RGB图像
        # cv2.imshow('RGB Camera View', rgb_img_bgr)
        # cv2.waitKey(1)
        controller.set_control_target(entity=drone, exp_vx=0.2, exp_vy=0.0, exp_vz=0.0, yaw_rate=0.0)
        rpm = controller.sim_control(0, 3, 2, 1)
        drone.set_propellels_rpm(rpm)
        scene.step()
        # viewer.update_following()


if __name__ == "__main__":
    main()
