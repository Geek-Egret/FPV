import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

from drone import controller as sim

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
    gs.init(backend=gs.cpu)

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
            file="urdf/drones/cf2x.urdf",
            pos=(0.0, 0.0, 1.0),

        ),
    )
    drone.propellers_spin

    ##########################  sensor ##########################
    # depth
    pos_offset = (0.0, 0.0, 0.1)
    sensor_kwargs = dict(
        entity_idx=drone.idx,
        pos_offset=pos_offset,
        euler_offset=(0.0, 0.0, 0.0),
        return_world_frame=False,
        draw_debug=False,        # 显示点
        min_range=0.25,          # 最小检测范围 (m)
        max_range=2.5,         # 最大检测范围 (m)
    )
    depth = scene.add_sensor(gs.sensors.DepthCamera(pattern=gs.sensors.DepthCameraPattern(), **sensor_kwargs))
    # imu
    end_effector = drone.get_link("base_link")
    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=drone.idx,
            link_idx_local=end_effector.idx_local,
            pos_offset=(0.0, 0.0, 0.0),
            # 传感器特性
            acc_cross_axis_coupling=(0.00, 0.00, 0.00),
            gyro_cross_axis_coupling=(0.00, 0.00, 0.00),
            acc_noise=(0.00, 0.00, 0.00),
            gyro_noise=(0.00, 0.00, 0.00),
            acc_random_walk=(0.000, 0.000, 0.000),
            gyro_random_walk=(0.000, 0.000, 0.000),
            delay=0.01,
            jitter=0.01,
            interpolate=True,
            draw_debug=True,
        )
    )

    ########################## build ##########################
    scene.build(n_envs=args.n_envs)

    init_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    init_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    target_pos = init_pos.copy()
    target_euler = init_euler.copy()

    def apply_pose_to_all_envs(pos_np: np.ndarray, quat_np: np.ndarray):
        if args.n_envs > 0:
            pos_np = np.expand_dims(pos_np, axis=0).repeat(args.n_envs, axis=0)
            quat_np = np.expand_dims(quat_np, axis=0).repeat(args.n_envs, axis=0)
        drone.set_pos(pos_np)
        drone.set_quat(quat_np)

    # Define control callbacks
    def reset_pose():
        target_pos[:] = init_pos
        target_euler[:] = init_euler

    def translate(index: int, is_negative: bool):
        target_pos[index] += (-1 if is_negative else 1) * KEY_DPOS

    def rotate(index: int, is_negative: bool):
        target_euler[index] += (-1 if is_negative else 1) * KEY_DANGLE

    # Register keybindings
    scene.viewer.register_keybinds(
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, False)),
        Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, True)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, True)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, False)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
        Keybind("roll_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(0, False)),
        Keybind("roll_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(0, True)),
        Keybind("pitch_up", Key.COMMA, KeyAction.HOLD, callback=rotate, args=(1, False)),
        Keybind("pitch_down", Key.PERIOD, KeyAction.HOLD, callback=rotate, args=(1, True)),
        Keybind("yaw_ccw", Key.O, KeyAction.HOLD, callback=rotate, args=(2, False)),
        Keybind("yaw_cw", Key.P, KeyAction.HOLD, callback=rotate, args=(2, True)),
        Keybind("reset", Key.BACKSLASH, KeyAction.HOLD, callback=reset_pose),
    )

    # Print controls
    print("Keyboard Controls:")
    print("[↑/↓/←/→]: Move XY")
    print("[j/k]: Down/Up")
    print("[n/m]: Roll CCW/CW")
    print("[,/.]: Pitch Up/Down")
    print("[o/p]: Yaw CCW/CW")
    print("[\\]: Reset")

    apply_pose_to_all_envs(target_pos, gu.euler_to_quat(target_euler))

    ########################## control ##########################
    controller = sim.sim_controller(drone)

    ang_ki = [0.0, 0.0, 0.0]

    ########################## viewer ##########################
    viewer = scene.visualizer.viewer
    # 让视角跟随无人机
    viewer.follow_entity(
        entity=drone,
        fixed_axis=(None, 1.5, 1.5),   # 高度固定在2米
        smoothing=0.1,
        fix_orientation=False
    )

    while(True):
        # depth.read_image()
        # apply_pose_to_all_envs(target_pos, gu.euler_to_quat(target_euler))
        pos = drone.get_pos()
        print(f"z: {pos[2].item():.6f}")
        controller.set_control_target(drone, exp_vx=0.5, exp_vy=0.0, exp_vz=0.1, yaw_rate=0.0)
        # if(pos[2].item() < 2):
        #     controller.set_control_target(drone, exp_vx=0.0, exp_vy=0.0, exp_vz=0.5, yaw_rate=0.0)
        # elif(pos[2].item() > 3):
        #     controller.set_control_target(drone, exp_vx=0.0, exp_vy=0.0, exp_vz=-0.5, yaw_rate=0.0)
        rpm = controller.sim_control(0, 3, 2, 1)
        drone.set_propellels_rpm(rpm)
        scene.step()
        viewer.update_following()


if __name__ == "__main__":
    main()