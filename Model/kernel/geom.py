import torch
import cv2
import numpy as np

import kernel.pid as pid
import kernel.util as util

"""
    将无人机作为质点，深度相机对无人机有一个pos_offset和euler_offset
"""
class geom:
    """
        @ GEOM初始化
        batch_size:并行数量
        device:训练设备:cpu/cuda
        dt:步长:s
        init_pos:初始位置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):m
        init_euler:初始姿态角度:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):度
        pos_offset:深度相机相较于无人机的位置偏置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):m
        euler_offset:深度相机相较于无人机的姿态偏置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):度
        mass:质量:kg
        T_max:最大推力:N
        ang_vel_max:最大角速度:[x,y,z]:度/s
        res_W:分辨率W
        res_H:分辨率H
        fov_H:水平视场角:度
        fov_V:垂直视场角:度
        min_depth:最近深度:m
        max_depth:最大深度:m
    """
    def __init__(self, batch_size, device, dt, init_pos, init_euler, 
                 pos_offset, euler_offset, mass, T_max, ang_vel_max,
                 res_W, res_H, fov_H, fov_V, min_depth, max_depth):
        self._threshold = 1e-8
        self._g = 9.81
        self._batch_size = batch_size
        self._device = device
        self._dt = dt
        self._init_drone_pos = self._adapt(init_pos).clone()
        self._init_drone_euler = util.angle_to_rad(self._adapt(init_euler)).clone()
        self._depth_pos_offset = self._adapt(pos_offset).clone()
        self._depth_euler_offset =util.angle_to_rad(self._adapt(euler_offset)).clone()
        self._mass = mass
        self._T_max = T_max
        self._ang_vel_max = ang_vel_max
        self._res_W = res_W#torch.tensor(res_W, dtype=torch.float, device=device)  
        self._res_H = res_H#torch.tensor(res_H, dtype=torch.float, device=device)  
        self._fov_H = torch.tensor(util.angle_to_rad(fov_H), dtype=torch.float, device=self._device, requires_grad=True)  
        self._fov_V = torch.tensor(util.angle_to_rad(fov_V), dtype=torch.float, device=self._device, requires_grad=True)  
        self._min_depth = min_depth
        self._max_depth = max_depth

        # 计算重力
        self._G = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True)
        z_axis = torch.full((self._batch_size,), -self._mass * self._g, device=self._device, requires_grad=True)
        self._G = torch.cat([self._G[:, :2], z_axis.unsqueeze(1)], dim=1)

        self._drone_pos = self._init_drone_pos
        self._drone_euler = self._init_drone_euler

        # 计算深度相机相对于世界坐标系的位姿
        self._drone_R = util.euler_to_R(self._init_drone_euler)
        self._drone_R = torch.where(torch.abs(self._drone_R) < self._threshold, torch.tensor(0.0, device=self._device), self._drone_R)
        self._depth_drone_R = util.euler_to_R(self._depth_euler_offset)
        self._depth_drone_R = torch.where(torch.abs(self._depth_drone_R) < self._threshold, torch.tensor(0.0, device=self._device), self._depth_drone_R)
        self._depth_R = torch.matmul(self._drone_R, self._depth_drone_R.transpose(-1, -2))
        self._depth_R = torch.where(torch.abs(self._depth_R) < self._threshold, torch.tensor(0.0, device=self._device), self._depth_R)
        self._depth_pos = self._init_drone_pos+torch.matmul(self._drone_R, self._depth_pos_offset.unsqueeze(-1)).squeeze(-1)
        self._depth_pos = torch.where(torch.abs(self._depth_pos) < self._threshold, torch.tensor(0.0, device=self._device), self._depth_pos)

        # 计算深度相机成像平面像素位置方向向量
        # 假设成像平面在传感器前方单位位置,传感器位姿和深度相机位姿一致
        half_width = torch.tan(self._fov_H/2)
        half_height = torch.tan(self._fov_V/2)
        y = torch.linspace(half_width, -half_width, self._res_W, device=self._device)
        z = torch.linspace(-half_height, half_height, self._res_H, device=self._device)
        yy, zz = torch.meshgrid(y, -z, indexing='xy')
        xx = torch.ones_like(yy)
        self._camera_pixel_dir = torch.stack([xx, yy, zz], dim=-1)  # 在相机坐标系下的像素方向向量

        # PID定义
        self._roll_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[0], self._device)
        self._pitch_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[1], self._device)
        self._yaw_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[2], self._device)

        # 定义
        self._acc = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).clone()
        self._vel = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).clone()
        self._ang_vel = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).clone()
        self._T = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).clone()
        self._spheres_list = []
        self._cylinders_list = []
        self._boxes_list = []
        self._depth = torch.zeros(self._batch_size, self._res_H, self._res_W, device=self._device, requires_grad=True).clone()   # 深度图
        self._is_collision = torch.zeros(self._batch_size, 1, device=self._device, dtype=torch.bool).clone()

    """
        @ GEOM添加球体
        x,y,z:球体中心位置:m
        R:球体形状:m
    """
    def add_sphere(self, x, y, z, R):
        self._spheres_list.append(torch.tensor([x, y, z, R], device=self._device, requires_grad=True))
        return len(self._spheres_list)-1

    """
        @ GEOM添加圆柱
        x,y,z:圆柱中心位置:m
        R,H:圆柱形状:m
    """
    def add_cylinder(self, x, y, z, R, H):
        self._cylinders_list.append(torch.tensor([x, y, z, R, H], device=self._device, requires_grad=True))
        return len(self._cylinders_list)-1

    """
        @ GEOM添加方块
        x,y,z:方块中心位置:m
        L,W,H:方块形状:m
    """
    def add_box(self, x, y, z, L, W, H):
        self._boxes_list.append(torch.tensor([x, y, z, L, W, H], device=self._device, requires_grad=True))
        return len(self._boxes_list)-1

    """
        @ GEOM执行一步
        act:动作(姿态角度,推力比例):
        T_att:推力衰减比例:0.0-1.0
    """
    def step(self, act, T_att, show_depth, idx):
        self._solver(act=act, T_att=T_att)
        self._render(show_depth=show_depth, idx=idx)

    """
        @ GEOM复位
    """
    def reset(self):
        self._drone_pos = self._init_drone_pos
        self._drone_euler = self._init_drone_euler
        # 计算深度相机相对于世界坐标系的位姿
        self._drone_R = util.euler_to_R(self._init_drone_euler)
        self._drone_R = torch.where(torch.abs(self._drone_R) < self._threshold, torch.tensor(0.0, device=self._device), self._drone_R)
        self._depth_drone_R = util.euler_to_R(self._depth_euler_offset)
        self._depth_drone_R = torch.where(torch.abs(self._depth_drone_R) < self._threshold, torch.tensor(0.0, device=self._device), self._depth_drone_R)
        self._depth_R = torch.matmul(self._drone_R, self._depth_drone_R.transpose(-1, -2))
        self._depth_R = torch.where(torch.abs(self._depth_R) < self._threshold, torch.tensor(0.0, device=self._device), self._depth_R)
        self._depth_pos = self._init_drone_pos+torch.matmul(self._drone_R, self._depth_pos_offset.unsqueeze(-1)).squeeze(-1)
        self._depth_pos = torch.where(torch.abs(self._depth_pos) < self._threshold, torch.tensor(0.0, device=self._device), self._depth_pos)

        # 计算深度相机成像平面像素位置方向向量
        # 假设成像平面在传感器前方单位位置,传感器位姿和深度相机位姿一致
        half_width = torch.tan(self._fov_H/2)
        half_height = torch.tan(self._fov_V/2)
        y = torch.linspace(half_width, -half_width, self._res_W, device=self._device)
        z = torch.linspace(-half_height, half_height, self._res_H, device=self._device)
        yy, zz = torch.meshgrid(y, -z, indexing='xy')
        xx = torch.ones_like(yy)
        self._camera_pixel_dir = torch.stack([xx, yy, zz], dim=-1)  # 在相机坐标系下的像素方向向量

        # PID定义
        self._roll_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[0], self._device)
        self._pitch_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[1], self._device)
        self._yaw_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[2], self._device)

        # 定义
        self._acc = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).clone()
        self._vel = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).clone()
        self._ang_vel = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).clone()
        self._T = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).clone()
        self._depth = torch.zeros(self._batch_size, self._res_H, self._res_W, device=self._device, requires_grad=True).clone()   # 深度图
        self._is_collision = torch.zeros(self._batch_size, 1, dtype=torch.bool, device=self._device).clone()

    """
        @ 无人机动力学求解器
        act:动作(姿态角度,推力比例):torch.tensor([[x,y,z,T], ...], dtype=torch.float, device=device, requires_grad=True):度,0.0-1.0
        T_att:推力衰减比例:0.0-1.0
    """
    def _solver(self, act, T_att):  
        # 计算合加速度
        self._T = self._drone_R[:, :, 2]*self._T_max*(1-T_att)*self._adapt(act)[:, 3].unsqueeze(1)
        sigma_force = self._T+self._G
        # 只为没有碰撞的无人机计算加速度，碰撞的无人机加速度为0.0
        self._acc = torch.where(self._is_collision, torch.tensor(0.0, device=self._device), sigma_force/self._mass)  
        # 计算速度
        self._vel = torch.where(self._is_collision, torch.tensor(0.0, device=self._device), self._vel+self._acc*self._dt)    
        # 计算位置
        self._drone_pos = self._drone_pos+self._vel*self._dt    # 注释加号后面的以实现定点调试旋转
        # PID计算角速度
        drone_euler = util.rad_to_angle(self._drone_euler)
        roll_vel = self._roll_pid.position(drone_euler[:, 0].unsqueeze(1), self._adapt(act)[:, 0].unsqueeze(1)).squeeze(1)
        pitch_vel = self._pitch_pid.position(drone_euler[:, 1].unsqueeze(1), self._adapt(act)[:, 1].unsqueeze(1)).squeeze(1)
        yaw_vel = self._yaw_pid.position(drone_euler[:, 2].unsqueeze(1), self._adapt(act)[:, 2].unsqueeze(1)).squeeze(1)
        self._ang_vel = torch.where(self._is_collision, torch.tensor(0.0, device=self._device), torch.stack([roll_vel, pitch_vel, yaw_vel], dim=1))   
        # 积分得到下一步姿态角度
        drone_euler= drone_euler+self._ang_vel*self._dt
        self._drone_euler = util.angle_to_rad(drone_euler)
        self._drone_R = util.euler_to_R(self._drone_euler)
        # 计算深度相机位姿
        self._depth_R = torch.matmul(self._drone_R, self._depth_drone_R.transpose(-1, -2))
        self._depth_pos = self._drone_pos+torch.matmul(self._drone_R, self._depth_pos_offset.unsqueeze(-1)).squeeze(-1)
        self._sphere_collision()
        self._ground_collision()    

    """
        @ 深度相机渲染深度图
    """
    def _render(self, show_depth, idx):
        self._depth = torch.zeros_like(self._depth)
        self._pixel_pos_dir = util.tensor_norm(torch.matmul(self._drone_R.unsqueeze(1).unsqueeze(1), self._camera_pixel_dir.unsqueeze(0).unsqueeze(-1)).squeeze(-1))
        self._ground_render()
        self._sphere_render()
        self._depth_validity_check()
        if show_depth:
            img = self._depth[idx, :, :].detach().cpu().numpy()
            # 2. 归一化到 0~255（深度图必须做这步）
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imshow("DEPTH VIEWER", img.astype(np.uint8))
            cv2.waitKey(1)

    """
        @ 深度相机地面渲染
    """
    def _ground_render(self):
        t = -self._depth_pos[:, 2].view(self._batch_size, 1, 1) / self._pixel_pos_dir[:, :, :, 2]
        mask_valid_t = t > 0  
        mask_update = (self._depth == 0) | (t < self._depth)    # 之前深度未被更新/当前深度比之前小
        final_mask = mask_valid_t & mask_update
        self._depth = torch.where(final_mask, t, self._depth)
    
    """
        @ 地面碰撞检测
    """
    def _ground_collision(self):
        D = self._drone_pos[:, 2].unsqueeze(-1)
        self._is_collision  = torch.where(self._is_collision, self._is_collision, torch.any(D < 0, dim=-1, keepdim=True))

    """
        @ 深度相机球体渲染
    """
    def _sphere_render(self):
        for i in range(len(self._spheres_list)):
            D = self._depth_pos-self._spheres_list[i][0:3]
            # 求根公式
            a = torch.square(torch.norm(self._pixel_pos_dir, dim=-1))
            b = 2*torch.sum(D.unsqueeze(1).unsqueeze(2)*self._pixel_pos_dir, dim=-1)#.unsqueeze(-1).unsqueeze(-1)
            c = (torch.square(torch.norm(D, dim=-1))-torch.square(torch.norm(self._spheres_list[i][3], dim=-1))).unsqueeze(-1).unsqueeze(-1)
            delta = torch.square(b)-4*a*c
            t_1 = torch.where(delta > 0.0, (-b+torch.sqrt(delta))/(2*a), torch.tensor(0.0, device=self._device))
            t_2 = torch.where(delta > 0.0, (-b-torch.sqrt(delta))/(2*a), torch.tensor(0.0, device=self._device))
            # 选取大于0且最小的/都小于0时为无穷大
            t = torch.where((t_1 > 0) & (t_2 > 0), torch.min(t_1, t_2), 
                    torch.where(t_1 > 0, t_1, 
                        torch.where(t_2 > 0, t_2, torch.tensor(-1.0, device=self._device))))
            z = self._depth_pos[..., 2].unsqueeze(-1).unsqueeze(-1)+self._pixel_pos_dir[..., 2]*t
            mask_valid_t = t > 0  
            mask_on_ground = z > 0
            mask_update = (self._depth == 0) | (t < self._depth)    # 之前深度未被更新/当前深度比之前小
            final_mask = mask_valid_t & mask_on_ground & mask_update
            self._depth = torch.where(final_mask, t, self._depth)
    
    """
        @ 球体碰撞检测
    """
    def _sphere_collision(self):
        for i in range(len(self._spheres_list)):
            D = torch.norm(self._drone_pos-self._spheres_list[i][0:3], dim=-1, keepdim=True)
            self._is_collision = torch.where(self._is_collision, self._is_collision, torch.any(D <= self._spheres_list[i][3], dim=-1, keepdim=True))

    """
        @ 深度相机有限高圆柱体渲染
    """
    # def _cylinder_render(self):

    """
        @ 深度相机有限高圆柱体碰撞检测
    """
    # def _cylinder_collision(self):

    """
        @ 深度相机有效检测
        t:计算得到的距离
    """
    def _depth_validity_check(self):
        # 求中心区域的平均值
        depth_center = self._depth[:, round(self._res_H*0.45):round(self._res_H*0.55), round(self._res_W*0.45):round(self._res_W*0.55)]
        depth_center_mean = torch.mean(depth_center, dim=(-2, -1))
        # 掩膜
        mask_vaild = depth_center_mean > self._min_depth
        mask_in_range = (self._depth >= self._min_depth) & (self._depth <= self._max_depth)
        final_mask = mask_vaild.unsqueeze(-1).unsqueeze(-1) & mask_in_range
        self._depth = torch.where(final_mask, self._depth, torch.tensor(0.0))

    """
        @ 适配张量维度到batch_size
    """
    def _adapt(self, tensor):
        if tensor.size(0) == 1 and tensor.size(0) != self._batch_size:
            repeat_times = [self._batch_size] + [1] * (tensor.dim() - 1)
            return tensor.repeat(repeat_times)
        elif tensor.size(0) == self._batch_size:
            return tensor
        else:
            raise Exception("[ERROR] tensor can't adapt to batchsize")

    @property
    def drone_pos(self):
        return self._drone_pos
    @property
    def drone_acc(self):
        return self._acc
    @property
    def drone_vel(self):
        return self._vel
    @property   # 度
    def drone_euler(self):
        return util.rad_to_angle(self._drone_euler)
    @property
    def drone_R(self):
        return self._drone_R
    @property
    def depth_pos(self):
        return self._depth_pos
    @property
    def depth_R(self):
        return self._depth_R
    @property
    def depth(self):
        return self._depth
    @property   # 得到各个无人机的碰撞状态张量:[batch_size, 1]
    def collision_state(self):
        return self._is_collision