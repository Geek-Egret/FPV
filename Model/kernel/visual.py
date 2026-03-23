import genesis

import kernel.util as util

class visual():
    def __init__(self, urdf, init_pos, init_euler, device, batch_size):
        self._device = device
        genesis.init(backend=genesis.cpu)
        viewer_options = genesis.options.ViewerOptions(
            camera_pos=(1.0, 1.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=90,
            max_FPS=120,
        )
        self._scene = genesis.Scene(
            sim_options=genesis.options.SimOptions(
                dt=0.01,
            ),
            viewer_options=viewer_options,
            show_viewer=True,
        )
        plane = self._scene.add_entity(
            genesis.morphs.Plane(
                visualization=True,   # 显示地面
                collision=True        # 有碰撞效果
            ),
        )
        self._drone = self._scene.add_entity(
            morph=genesis.morphs.Drone(
                file=urdf,
                pos=init_pos.to('cpu').numpy(),
                euler=init_euler.to('cpu').numpy(),
            ),
        )
        self._scene.build(n_envs=batch_size)

    def step(self, pos, euler):
        self._drone.set_pos(pos)
        self._drone.set_quat(util.euler_to_quat(util.angle_to_rad(euler)))
        self._scene.step()