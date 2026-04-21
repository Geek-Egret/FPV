import genesis

import env.util as util

class visual():
    def __init__(self, urdf, device, init_pos, init_euler, batch_size):
        self._batch_size = batch_size
        if device == 'cuda':
            genesis.init(
                seed                = None,
                precision           = '32',
                debug               = False,
                eps                 = 1e-12,
                logging_level       = 'warning',
                backend             = genesis.cuda,
                theme               = 'dark',
                logger_verbose_time = False
            )
        else:
            genesis.init(
                seed                = None,
                precision           = '32',
                debug               = False,
                eps                 = 1e-12,
                logging_level       = 'warning',
                backend             = genesis.cpu,
                theme               = 'dark',
                logger_verbose_time = False
            )
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
                collision=False        # 有碰撞效果
            ),
        )
        self._drone = self._scene.add_entity(
            morph=genesis.morphs.Drone(
                file=urdf,
                pos=init_pos.clone().detach().to('cpu').numpy(),
                euler=init_euler.clone().detach().to('cpu').numpy(),
            ),
        )

    def add_sphere(self, x, y, z, R):
        self._scene.add_entity(
            genesis.morphs.Sphere(
                pos=(x, y, z),
                radius=R,
                fixed=True,
            )
        )

    def add_cylinder(self, x, y, z, R, H):
        self._scene.add_entity(
            genesis.morphs.Cylinder(
                height=H,
                radius=R,
                pos=(x, y, z),
                fixed=True,
            )
        )

    def add_box(self, x, y, z, L, W, H):
        self._scene.add_entity(
            genesis.morphs.Box(
                size=(x, y, z),
                pos=(L, W, H),
                fixed=True,
            )
        )

    def build(self):
        self._scene.build(n_envs=self._batch_size)

    def step(self, pos, euler):
        self._drone.set_pos(pos.clone().detach())
        self._drone.set_quat(util.euler_to_quat(util.deg_to_rad(euler.clone().detach())))
        self._scene.step()