import genesis
import torch
import matplotlib.pyplot as plt
import numpy as np

import kernel.util as util

class geometry:
    """
        camera_pos:相机位置:(x, y, z)
        camera_lookat:相机方向:(x, y, z)
        camera_fov:视场角:度
        max_FPS:最大FPS
        show_viewer:显示
        dt:步长:s
        device:设备:cpu/gpu/cuda
    """
    def __init__(self, camera_pos, camera_lookat, camera_fov, max_FPS, show_viewer, dt, device):
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
            show_viewer=show_viewer,
        )
        self._plane = self._scene.add_entity(
            genesis.morphs.Plane(
                visualization=True,   # 显示地面
                collision=True        # 有碰撞效果
            ),
        )
        self._device = device
        self._drones_list = []
        self._depths_list = []


        NUM_CYLINDERS = 8
        NUM_BOXES = 6
        CYLINDER_RING_RADIUS = 0.3
        BOX_RING_RADIUS = 0.3
        for i in range(NUM_CYLINDERS):
            angle = 2 * np.pi * i / NUM_CYLINDERS
            x = CYLINDER_RING_RADIUS * np.cos(angle)*10
            y = CYLINDER_RING_RADIUS * np.sin(angle)*10
            self._scene.add_entity(
                genesis.morphs.Cylinder(
                    height=1.5,
                    radius=0.3,
                    pos=(x, y, 0.75),
                    fixed=True,
                )
            )

        for i in range(NUM_BOXES):
            angle = 2 * np.pi * i / NUM_BOXES + np.pi / 6
            x = BOX_RING_RADIUS * np.cos(angle)*10
            y = BOX_RING_RADIUS * np.sin(angle)*10
            self.a = self._scene.add_entity(
                genesis.morphs.Box(
                    size=(0.5, 0.5, 2.0 * (i + 1) / NUM_BOXES),
                    pos=(x, y, 1.0),
                    fixed=True,
                )
            )

        entity_kwargs = dict(
            pos=(0.0, 0.0, 0.35),
            quat=(1.0, 0.0, 0.0, 0.0),
            fixed=True,
        )

    """
        res_W:分辨率像素宽度
        res_H:分辨率像素高度
        init_pos:初始位置:torch.tensor([x, y, z], dtype=torch.double):m
        pos_offset:深度相机的pos相对于机器人pos的偏置:torch.tensor([x, y, z], dtype=torch.double):m
        lookat:
    """
    def add_depth(self, res_W, res_H, init_pos, pos_offset, lookat, up, fov_V, GUI):
        depth = self._scene.add_camera(
            res=(res_W, res_H),         # 分辨率（宽，高）
            pos=(0.0, 0.0, 0.5),         # 相机位置
            lookat=(1, 0, 0),       # 注视点
            fov=fov_V,                 # 垂直视野角度，默认30度
            up=(0, 0, 1),           # 向上向量
            model="pinhole",        # 相机模型（pinhole或thinlens）
            GUI=GUI                # 图像显示
        )

    """
        urdf_path:URDF路径
        drone_init_pos:无人机初始位置:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.double):m
        drone_init_R:无人机初始旋转向量
        res_W:分辨率像素宽度
        res_H:分辨率像素高度
        detph_init_pos:初始位置:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.double):m
        detph_pos_offset:深度相机的pos相对于机器人pos的偏置:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.double):m
        lookat:深度相机的朝向:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.double):m
        up:深度相机的上向向量:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.double):m
        fov_V:垂直视场角:度
        GUI:可视化使能
        num:同一场景实体数量
    """
    def add_drone(self, urdf_path, drone_init_pos, drone_init_R, res_W, res_H, init_pos, pos_offset, lookat, up, fov_V, GUI, num):
        self._num = num
        print(drone_init_R)
        if num == 1:
            if self._device == 'cuda':
                drone_init_pos_tensor = self._tensor_adapt(drone_init_pos, num, 3)
                drone_pos = drone_init_pos_tensor.cpu().numpy()
                drone_init_R_tensor = drone_init_R
                drone_euler = util.rad_to_angle(util.R_to_euler(drone_init_R_tensor)).cpu().numpy()
                init_pos_tensor = self._tensor_adapt(init_pos, num, 3)
                depth_pos = init_pos_tensor.cpu().numpy()
                self._depth_pos_offset = self._tensor_adapt(pos_offset, num, 3)
                lookat_tensor = self._tensor_adapt(lookat, num, 3)
                depth_lookat = lookat_tensor.cpu().numpy()
                up_tensor = self._tensor_adapt(up, num, 3)
                depth_up = up_tensor.cpu().numpy()
            else:
                drone_init_pos_tensor = self._tensor_adapt(drone_init_pos, num, 3)
                drone_pos = drone_init_pos_tensor.numpy()
                drone_init_R_tensor = drone_init_R
                drone_euler = util.rad_to_angle(util.R_to_euler(drone_init_R_tensor)).numpy()
                init_pos_tensor = self._tensor_adapt(init_pos, num, 3)
                depth_pos = init_pos_tensor.numpy()
                self._depth_pos_offset = self._tensor_adapt(pos_offset, num, 3)
                lookat_tensor = self._tensor_adapt(lookat, num, 3)
                depth_lookat = lookat_tensor.numpy()
                up_tensor = self._tensor_adapt(up, num, 3)
                depth_up = up_tensor.numpy()
            # 归一化前向向量
            lookat_tensor = util.tensor_nrom(lookat_tensor)
            print(f"lookat {lookat_tensor}")
            # 计算左向向量
            left_vec = torch.cross(up_tensor, lookat_tensor, dim=-1)
            left_vec = util.tensor_nrom(left_vec)
            print(f"lookat {left_vec}")
            # 校正上向向量
            up_tensor = torch.cross(lookat_tensor, left_vec, dim=-1)
            up_tensor = util.tensor_nrom(up_tensor)
            print(f"lookat {up_tensor}")
            # 构建深度相机的旋转矩阵
            depth_R = torch.stack([
                lookat_tensor,
                left_vec,
                up_tensor 
            ], dim=-1)
            # 计算深度相机到无人机的旋转矩阵
            self._depth_drone_R = torch.matmul(depth_R, drone_init_R.transpose(-1, -2))
            print(f"depth_drone: {depth_R}  {drone_init_R}  {self._depth_drone_R}")
            drone = self._scene.add_entity(
                morph=genesis.morphs.Drone(
                    file=urdf_path,
                    pos=drone_pos,
                    euler=drone_euler
                ),
            )
            depth = self._scene.add_camera(
                res=(res_W, res_H),        
                pos=depth_pos,         
                lookat=depth_lookat,     
                fov=fov_V,                
                up=depth_up,         
                model="pinhole",       
                GUI=GUI             
            )
            self._drones_list.append(drone)
            self._depths_list.append(depth)
        elif num > 1:
            for i in range(num):
                if self._device == 'cuda':
                    drone_init_pos_tensor = self._tensor_adapt(drone_init_pos, num, 3)
                    drone_pos = drone_init_pos_tensor[i].cpu().numpy()
                    drone_init_R_tensor = drone_init_R
                    drone_euler = util.rad_to_angle(util.R_to_euler(drone_init_R_tensor[i])).cpu().numpy()
                    init_pos_tensor = self._tensor_adapt(init_pos, num, 3)
                    depth_pos = init_pos_tensor[i].cpu().numpy()
                    self._depth_pos_offset = self._tensor_adapt(pos_offset, num, 3)
                    lookat_tensor = self._tensor_adapt(lookat, num, 3)
                    depth_lookat = lookat_tensor[i].cpu().numpy()
                    up_tensor = self._tensor_adapt(up, num, 3)
                    depth_up = up_tensor[i].cpu().numpy()
                else:
                    drone_init_pos_tensor = self._tensor_adapt(drone_init_pos, num, 3)
                    drone_pos = drone_init_pos_tensor[i].numpy()
                    drone_init_R_tensor = drone_init_R
                    drone_euler = util.rad_to_angle(util.R_to_euler(drone_init_R_tensor[i])).numpy()
                    init_pos_tensor = self._tensor_adapt(init_pos, num, 3)
                    depth_pos = init_pos_tensor[i].numpy()
                    self._depth_pos_offset = self._tensor_adapt(pos_offset, num, 3)
                    lookat_tensor = self._tensor_adapt(lookat, num, 3)
                    depth_lookat = lookat_tensor[i].numpy()
                    up_tensor = self._tensor_adapt(up, num, 3)
                    depth_up = up_tensor[i].numpy()
                # 归一化前向向量
                lookat_tensor = util.tensor_nrom(lookat_tensor)
                # 计算左向向量
                left_vec = torch.cross(lookat_tensor, up_tensor, dim=-1)
                left_vec = util.tensor_nrom(left_vec)
                # 校正上向向量
                up_tensor = torch.cross(lookat_tensor, left_vec, dim=-1)
                up_tensor = util.tensor_nrom(up_tensor)
                # 构建深度相机的旋转矩阵
                depth_R = torch.stack([
                    lookat_tensor,
                    left_vec,
                    up_tensor 
                ], dim=-1)
                # 计算深度相机到无人机的旋转矩阵
                self._depth_drone_R = torch.matmul(depth_R.unsqueeze(1), drone_init_R.transpose(-1, -2)).squeeze(1)
                drone = self._scene.add_entity(
                    morph=genesis.morphs.Drone(
                        file=urdf_path,
                        pos=drone_pos,
                        euler=drone_euler,
                        gravity=False
                    ),
                )
                depth = self._scene.add_camera(
                    res=(res_W, res_H),        
                    pos=depth_pos,         
                    lookat=depth_lookat,     
                    fov=fov_V,                
                    up=depth_up,         
                    model="pinhole",       
                    GUI=GUI             
                )
                self._drones_list.append(drone)
                self._depths_list.append(depth)
        self._num = num

    """
        构建场景
    """
    def build(self):
        self._scene.build(n_envs=0)

    """
        继续下一步
    """
    def step(self):
        distance = self.sdf_distance(self._next_pos, self.a.links[0].geoms[0])
        print(distance)
        rgb, depth, _, _ = self._depths_list[0].render(rgb=True, depth=True)
        # self._plt_imshow(rgb, depth)
        self._scene.step()

    """
        next_pos:下一刻位置:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.double):m
        next_R:下一刻旋转矩阵
    """
    def set_pos_R(self, next_pos, next_R):
        self._next_pos = next_pos
        self._next_R = next_R
        # 计算深度相机旋转矩阵
        depth_R = torch.matmul(next_R, self._depth_drone_R)
        print(f"depth R {depth_R[:, 0]} {depth_R[:, 1]} {depth_R[:, 2]}")
        # 计算深度相机位置
        pos_offset_world = torch.matmul(depth_R,  self._depth_pos_offset)
        pos_world = next_pos+pos_offset_world
        print(f"depth pos {pos_world}")
        if self._num == 1:
            if self._device == 'cuda':
                self._drones_list[0].set_pos(next_pos.cpu().numpy())
                self._drones_list[0].set_quat(util.R_to_quat(next_R).cpu().numpy())
                self._depths_list[0].set_pose(
                    pos = pos_world.cpu().numpy(),
                    lookat = (pos_world+depth_R[:, 0]).cpu().numpy(),
                    up = depth_R[:, 2].cpu().numpy()
                )
            else:
                self._drones_list[0].set_pos(next_pos.numpy())
                self._drones_list[0].set_quat(util.R_to_quat(next_R).numpy())
                self._depths_list[0].set_pose(
                    pos = pos_world.numpy(),
                    lookat = (pos_world+depth_R[:, 0]).numpy(),
                    up = depth_R[:, 2].numpy()
                )
        elif self._num > 1:
            for i in range(self._num):
                if self._device == 'cuda':
                    self._drones_list[i].set_pos(next_pos[i].cpu().numpy())
                    self._drones_list[i].set_quat(util.R_to_quat(next_R[i]).cpu().numpy())
                    self._depths_list[i].set_pose(
                        pos = pos_world[i].cpu().numpy(),
                        lookat = (pos_world+depth_R[i, :, 0]).cpu().numpy(),
                        up = depth_R[i, :, 2].cpu().numpy()
                    )
                else:
                    self._drones_list[i].set_pos(next_pos[i].numpy())
                    self._drones_list[i].set_quat(util.R_to_quat(next_R[i]).numpy())
                    self._depths_list[i].set_pose(
                        pos = pos_world[i].numpy(),
                        lookat = (pos_world[i]+depth_R[i, :, 0]).numpy(),
                        up = depth_R[i, :, 2].numpy()
                    )

    """
        SDF距离
        point_world:世界坐标系点:torch.tensor([x, y, z], dtype=torch.double):m
        geom:刚体:xxx.links.geoms
    """
    def sdf_distance(self, point_world, geom):
        if self._device == 'cuda':
            dist = geom.sdf_world(point_world.cpu().numpy(), recompute=False)
        else:
            dist = geom.sdf_world(point_world.numpy(), recompute=False)
        min_abs_dist = torch.abs(dist).min()
        return min_abs_dist
    
    """
        1阶张量阶度适配到2阶
        tensor:张量
        rows_num:适配目标行数
        cols_num:传入张量的正确列数
        返回：适配后张量
    """
    def _tensor_adapt(self, tensor, rows_num, cols_num):
        new_tensor = None
        # print(tensor)
        if tensor.dim() == 1 and tensor.shape[0] == cols_num and rows_num > 1:
            new_tensor = tensor.unsqueeze(0).expand(rows_num, cols_num)    # 新增行维度后拷贝
        elif tensor.dim() == 1 and tensor.shape[0] == cols_num and rows_num == 1:
            new_tensor = tensor
        elif tensor.dim() == 2 and tensor.shape[0] == rows_num and tensor.shape[1] == cols_num:
            new_tensor = tensor
        elif tensor.dim() == 2 and tensor.shape[1] != cols_num:
            try:
                raise ValueError(f"{tensor}的列数为{tensor.shape[0]}不等于指定列数{cols_num}")
            except ValueError as e:
                print(e)
        elif tensor.dim() == 1 and tensor.shape[0] != cols_num:
            try:
                raise ValueError(f"{tensor}的列数为{tensor.shape[0]}不等于指定列数{cols_num}")
            except ValueError as e:
                print(e)
        elif tensor.dim() == 2 and tensor.shape[0] != rows_num:
            try:
                raise ValueError(f"{tensor}的行数为{tensor.shape[0]},无法转换")
            except ValueError as e:
                print(e)
        
        return new_tensor
    
    def _plt_imshow(self, rgb_image, depth_image):
        plt.clf()  # 清除当前图像
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title('RGB')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze()
        plt.imshow(depth_image, cmap='jet')
        plt.title('Depth')
        plt.axis('off')

        plt.pause(0.01)  # 非阻塞显示