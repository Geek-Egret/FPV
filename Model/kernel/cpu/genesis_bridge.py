import util
import genesis

class genesis_bridge:
    """
        camera_pos:相机位置:(x, y, z)
        camera_lookat:相机方向:(x, y, z)
        camera_fov:视场角:度
        max_FPS:最大FPS
        dt:步长:s
        device:设备:cpu/gpu/cuda
    """
    def __init__(self, camera_pos, camera_lookat, camera_fov, max_FPS, dt, device):
        if device == "cpu":
            genesis.init(backend=genesis.cpu)
        elif device == "cuda":
            genesis.init(backend=genesis.cuda)

        self._viewer_options = genesis.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=camera_fov,
            max_FPS=max_FPS,
        )
        self._scene = genesis.Scene(
            sim_options=genesis.options.SimOptions(
                dt=dt,
            ),
            viewer_options=self._viewer_options,
            show_viewer=True,
        )
        self._plane = self._scene.add_entity(
            genesis.morphs.Plane(),
        )
        self._device = device
        self._drones_list = []

    """
        urdf_path:URDF路径
        init_pos:初始位置:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        num:同一场景实体数量
    """
    def add_drone(self, urdf_path, init_pos, init_R, num):
        if num == 1:
            if self._device == 'cuda':
                pos = init_pos.cpu().numpy()
                euler = util.rad_to_angle(util.R_to_euler(init_R)).cpu().numpy()
            else:
                pos = init_pos.numpy()
                euler = util.rad_to_angle(util.R_to_euler(init_R)).numpy()
            drone = self._scene.add_entity(
                morph=genesis.morphs.Drone(
                    file=urdf_path,
                    pos=pos,
                    euler=euler
                ),
            )
            self._drones_list.append(drone)
        elif num > 1:
            for i in range(num):
                if self._device == 'cuda':
                    pos = init_pos[i].cpu().numpy()
                    euler = util.rad_to_angle(util.R_to_euler(init_R[i])).cpu().numpy()
                else:
                    pos = init_pos[i].numpy()
                    euler = util.rad_to_angle(util.R_to_euler(init_R[i])).numpy()
                drone = self._scene.add_entity(
                    morph=genesis.morphs.Drone(
                        file=urdf_path,
                        pos=pos,
                        euler=euler
                    ),
                )
                self._drones_list.append(drone)
        print(euler)
        self._num = num

    """
        构建场景
    """
    def build(self):
        self._scene.build(n_envs=1)

    """
        继续下一步
    """
    def step(self):
        self._scene.step()

    """
        next_pos:下一刻位置:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        next_R:下一刻旋转矩阵
    """
    def set_pos_R(self, next_pos, next_R):
        if self._num == 1:
            if self._device == 'cuda':
                self._drones_list[0].set_pos(next_pos.cpu().numpy())
                # drone.set_quat(ge_fpv.next_pos.cpu().numpy())
            else:
                self._drone.set_pos[0](next_pos.numpy())
        elif self._num > 1:
            for i in range(self._num):
                if self._device == 'cuda':
                    self._drones_list[i].set_pos(next_pos[i].cpu().numpy())
                    # drone.set_quat(ge_fpv.next_pos.cpu().numpy())
                else:
                    self._drones_list[i].set_pos(next_pos[i].numpy())
    