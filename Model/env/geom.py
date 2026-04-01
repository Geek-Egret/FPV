import torch
import cv2
import numpy as np
from scipy import stats

import env.pid as pid
import env.util as util

"""
    将无人机作为质点,深度相机对无人机有一个pos_offset和euler_offset
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
        self._fov_H = torch.tensor(util.angle_to_rad(fov_H), dtype=torch.float, device=self._device)  
        self._fov_V = torch.tensor(util.angle_to_rad(fov_V), dtype=torch.float, device=self._device)  
        self._min_depth = min_depth
        self._max_depth = max_depth

        # 计算重力
        self._G = torch.zeros(self._batch_size, 3, device=self._device)
        z_axis = torch.full((self._batch_size,), -self._mass * self._g, device=self._device)
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
        self._depth = torch.zeros(self._batch_size, self._res_H, self._res_W, device=self._device).clone()   # 深度图
        self._distance = torch.full((self._batch_size, 1), float('inf'), device=self._device).clone()   # 最近距离
        self._is_collision = torch.zeros(self._batch_size, 1, device=self._device, dtype=torch.bool).clone()

    """
        @ GEOM添加球体
        x,y,z:球体中心位置:m
        R:球体形状:m
    """
    def add_sphere(self, x, y, z, R):
        self._spheres_list.append(torch.tensor([x, y, z, R], device=self._device))
        return len(self._spheres_list)-1

    """
        @ GEOM添加圆柱
        x,y,z:圆柱中心位置:m
        R,H:圆柱形状:m
    """
    def add_cylinder(self, x, y, z, R, H):
        self._cylinders_list.append(torch.tensor([x, y, z, R, H], device=self._device))
        return len(self._cylinders_list)-1

    """
        @ GEOM添加方块
        x,y,z:方块中心位置:m
        L,W,H:方块形状:m
    """
    def add_box(self, x, y, z, L, W, H):
        self._boxes_list.append(torch.tensor([x, y, z, L, W, H], device=self._device))
        return len(self._boxes_list)-1

    """
        @ GEOM执行一步
        act:动作(姿态角度,推力比例):
        T_att:推力衰减比例:0.0-1.0
        show_depth:是否使能显示深度图
        show_idx:显示的深度图索引
        noise:是否加入噪声
        noise_range:噪声范围
        black_hole_prob:深度图黑洞出现概率
    """
    def step(self, act, T_att, show_depth, show_idx, noise, noise_range, black_hole_prob):
        self._solver(act=act, T_att=T_att)
        self._render(show_depth=show_depth, show_idx=show_idx, noise=noise, noise_range=noise_range, black_hole_prob=black_hole_prob)

    """
        @ GEOM复位
        断开之前的计算图
    """
    def reset(self, init_pos, init_euler, domain_randomization):
        self._init_drone_pos = self._adapt(init_pos).clone()
        self._init_drone_euler = util.angle_to_rad(self._adapt(init_euler)).clone()
        self._drone_pos = self._init_drone_pos.detach().clone()
        self._drone_euler = self._init_drone_euler.detach().clone()
        # 计算深度相机相对于世界坐标系的位姿
        self._drone_R = util.euler_to_R(self._init_drone_euler).detach().clone()
        self._drone_R = torch.where(torch.abs(self._drone_R) < self._threshold, torch.tensor(0.0, device=self._device), self._drone_R).detach().clone()
        self._depth_drone_R = util.euler_to_R(self._depth_euler_offset).detach().clone()
        self._depth_drone_R = torch.where(torch.abs(self._depth_drone_R) < self._threshold, torch.tensor(0.0, device=self._device), self._depth_drone_R).detach().clone()
        self._depth_R = torch.matmul(self._drone_R, self._depth_drone_R.transpose(-1, -2)).detach().clone()
        self._depth_R = torch.where(torch.abs(self._depth_R) < self._threshold, torch.tensor(0.0, device=self._device), self._depth_R).detach().clone()
        self._depth_pos = self._init_drone_pos+torch.matmul(self._drone_R, self._depth_pos_offset.unsqueeze(-1)).squeeze(-1).detach().clone()
        self._depth_pos = torch.where(torch.abs(self._depth_pos) < self._threshold, torch.tensor(0.0, device=self._device), self._depth_pos).detach().clone()

        # PID定义
        self._roll_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[0], self._device)
        self._pitch_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[1], self._device)
        self._yaw_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[2], self._device)

        # 定义
        self._acc = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).detach().clone()
        self._vel = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).detach().clone()
        self._ang_vel = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).detach().clone()
        self._T = torch.zeros(self._batch_size, 3, device=self._device, requires_grad=True).detach().clone()
        self._depth = torch.zeros(self._batch_size, self._res_H, self._res_W, device=self._device).detach().clone()   # 深度图
        self._distance = torch.full((self._batch_size, 1), float('inf'), device=self._device).clone()   # 最近距离
        self._is_collision = torch.zeros(self._batch_size, 1, dtype=torch.bool, device=self._device).detach().clone()

        if domain_randomization:
            self._spheres_list.clear()
            self._cylinders_list.clear()
            self._boxes_list.clear()

    """
        @ GEOM构建
    """
    def build(self, show_depth, show_idx, noise, noise_range, black_hole_prob):
        self._render(show_depth=show_depth, show_idx=show_idx, noise=noise, noise_range=noise_range, black_hole_prob=black_hole_prob)

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
        self._acc = torch.where(self._is_collision, torch.zeros_like(self._acc), sigma_force/self._mass)  
        # 计算速度
        self._vel = torch.where(self._is_collision, torch.zeros_like(self._vel), self._vel+self._acc*self._dt)    
        # 计算位置
        self._drone_pos = self._drone_pos+self._vel*self._dt    # 注释加号后面的以实现定点调试旋转
        # PID计算角速度
        drone_euler = util.rad_to_angle(self._drone_euler)
        roll_vel = self._roll_pid.position(drone_euler[:, 0].unsqueeze(1), self._adapt(act)[:, 0].unsqueeze(1)).squeeze(1)
        pitch_vel = self._pitch_pid.position(drone_euler[:, 1].unsqueeze(1), self._adapt(act)[:, 1].unsqueeze(1)).squeeze(1)
        yaw_vel = self._yaw_pid.position(drone_euler[:, 2].unsqueeze(1), self._adapt(act)[:, 2].unsqueeze(1)).squeeze(1)
        self._ang_vel = torch.where(self._is_collision, torch.zeros_like(self._ang_vel), torch.stack([roll_vel, pitch_vel, yaw_vel], dim=1))   
        # 积分得到下一步姿态角度
        drone_euler= drone_euler+self._ang_vel*self._dt
        self._drone_euler = util.angle_to_rad(drone_euler)
        self._drone_R = util.euler_to_R(self._drone_euler)
        # 计算深度相机位姿
        self._depth_R = torch.matmul(self._drone_R, self._depth_drone_R.transpose(-1, -2))
        self._depth_pos = self._drone_pos+torch.matmul(self._drone_R, self._depth_pos_offset.unsqueeze(-1)).squeeze(-1)
        # 计算最近距离
        self._distance = torch.full((self._batch_size, 1), float('inf'), device=self._device).clone()
        self._sphere_distance()
        self._cylinder_distance()
        self._ground_distance()    

    """
        @ 深度相机渲染深度图
        show_depth:是否使能显示深度图
        show_idx:显示的深度图索引
        noise:是否加入噪声
        noise_range:噪声范围
        black_hole_prob:深度图黑洞出现概率
    """
    def _render(self, show_depth, show_idx, noise, noise_range, black_hole_prob):
        self._depth = torch.zeros_like(self._depth)
        self._pixel_dir = util.tensor_norm(torch.matmul(self._drone_R.unsqueeze(1).unsqueeze(1), self._camera_pixel_dir.unsqueeze(0).unsqueeze(-1)).squeeze(-1))
        self._ground_render(noise, noise_range, black_hole_prob)
        self._sphere_render(noise, noise_range, black_hole_prob)
        self._cylinder_render(noise, noise_range, black_hole_prob)
        self._depth_validity_check()
        if show_depth:
            img = self._depth[show_idx, :, :].detach().cpu().numpy()/self._max_depth
            # 2. 归一化到 0~255（深度图必须做这步）
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imshow("DEPTH VIEWER", img.astype(np.uint8))
            cv2.waitKey(1)

    """
        @ 深度相机地面渲染
    """
    def _ground_render(self, noise, noise_range, black_hole_prob):
        Rs_z = self._depth_pos[:, 2].detach().unsqueeze(1).unsqueeze(1)
        Rt_z = self._pixel_dir[:, :, :, 2].detach()
        t = -Rs_z / Rt_z
        mask_valid_t = t > 0  
        mask_update = (self._depth == 0) | (t <= self._depth)    # 之前深度未被更新/当前深度比之前小
        final_mask = mask_valid_t & mask_update
        self._depth = torch.where(final_mask, t, self._depth)
        mask_inf = (t > self._max_depth) & (self._max_depth == 0)    # 当前深度超过最大深度且原深度未被更新
        self._depth = torch.where(mask_inf, torch.tensor([float('inf')], dtype=torch.float32), self._depth) # 超过最大深度的部分设置为无穷大
        if noise:
            # 选取有效区域加入噪声
            mask_noise = (self._depth >= self._min_depth) & (self._depth <= self._max_depth)
            # 深度图传感器误差
            offset_noise = torch.clamp(torch.randn_like(self._depth)*(noise_range / 3), min=-noise_range, max=noise_range)  # 或其他随机分布
            self._depth[mask_noise] += offset_noise[mask_noise]
            # 深度图黑洞
            black_hole_noise = torch.randn_like(self._depth) < stats.norm.ppf(black_hole_prob)   # 每个位置有5%概率出现黑洞
            black_hole = torch.zeros_like(self._depth)
            self._depth[black_hole_noise] = black_hole[black_hole_noise]
    
    """
        @ 地面距离计算
    """
    def _ground_distance(self):
        D = self._drone_pos[:, 2].unsqueeze(-1)
        # 最近距离
        mask = D < self._distance
        self._distance = torch.where(mask, D, self._distance)
        # 碰撞检测
        collision_flag = torch.any(D.detach() < 0, dim=-1, keepdim=True)
        self._is_collision = self._is_collision | collision_flag


    """
        @ 深度相机球体渲染
    """
    def _sphere_render(self, noise, noise_range, black_hole_prob):
        for i in range(len(self._spheres_list)):
            Rs_xyz = self._depth_pos.detach().unsqueeze(1).unsqueeze(1)  # 射线起点XYZ
            Rt_xyz = self._pixel_dir.detach()    # 射线方向向量XYZ
            S_xyz = self._spheres_list[i][0:3].detach().unsqueeze(0).unsqueeze(0).unsqueeze(0) # 球体XYZ
            R = self._spheres_list[i][3].detach().unsqueeze(0).unsqueeze(0).unsqueeze(0)   # 球体半径
            # 求根公式
            a = torch.square(torch.norm(Rt_xyz, dim=-1, keepdim=True)).squeeze(-1)
            b = 2*torch.sum(Rt_xyz*(Rs_xyz-S_xyz), dim=-1, keepdim=True).squeeze(-1)
            c = (torch.square(torch.norm(Rs_xyz-S_xyz, dim=-1, keepdim=True))-torch.square(torch.norm(R, dim=-1, keepdim=True))).squeeze(-1)
            delta = torch.square(b)-4*a*c
            t_1 = torch.where(delta > 0.0, (-b+torch.sqrt(delta))/(2*a), torch.tensor(0.0, device=self._device))
            t_2 = torch.where(delta > 0.0, (-b-torch.sqrt(delta))/(2*a), torch.tensor(0.0, device=self._device))
            # 选取大于0且最小的/都小于0时为0.0
            t = torch.where((t_1 > 0) & (t_2 > 0), torch.min(t_1, t_2), 
                    torch.where(t_1 > 0, t_1, 
                        torch.where(t_2 > 0, t_2, torch.tensor(0.0, device=self._device))))
            z = self._depth_pos[..., 2].unsqueeze(-1).unsqueeze(-1)+self._pixel_dir[..., 2]*t
            mask_valid_t = t > 0  
            mask_on_ground = z > 0
            mask_update = (self._depth == 0) | (t < self._depth)    # 之前深度未被更新/当前深度比之前小
            final_mask = mask_valid_t & mask_on_ground & mask_update
            self._depth = torch.where(final_mask, t, self._depth)
            mask_inf = (t > self._max_depth) & (self._max_depth == 0)    # 当前深度超过最大深度且原深度未被更新
            self._depth = torch.where(mask_inf, torch.tensor([float('inf')], dtype=torch.float32), self._depth) # 超过最大深度的部分设置为无穷大
            if noise:
                # 选取有效区域加入噪声
                mask_noise = (self._depth >= self._min_depth) & (self._depth <= self._max_depth)
                # 深度图传感器误差
                offset_noise = torch.clamp(torch.randn_like(self._depth)*(noise_range / 3), min=-noise_range, max=noise_range)  # 或其他随机分布
                self._depth[mask_noise] += offset_noise[mask_noise]
                # 深度图黑洞
                black_hole_noise = torch.randn_like(self._depth) < stats.norm.ppf(black_hole_prob)   # 每个位置有5%概率出现黑洞
                black_hole = torch.zeros_like(self._depth)
                self._depth[black_hole_noise] = black_hole[black_hole_noise]
    
    """
        @ 球体距离计算
    """
    def _sphere_distance(self):
        for i in range(len(self._spheres_list)):
            R = self._spheres_list[i][3].unsqueeze(0) # 球体半径
            D = torch.norm(self._drone_pos-self._spheres_list[i][0:3], dim=-1, keepdim=True)-R
            # 最近距离
            mask = D < self._distance
            self._distance = torch.where(mask, D, self._distance)
            # 碰撞检测
            collision_flag = torch.any(D.detach() <= 0, dim=-1, keepdim=True)
            self._is_collision = self._is_collision | collision_flag

    """
        @ 深度相机有限高圆柱体渲染
    """
    def _cylinder_render(self, noise, noise_range, black_hole_prob):
        for i in range(len(self._cylinders_list)):
            Rs_xy = self._depth_pos[:, 0:2].detach().unsqueeze(1).unsqueeze(1) # 射线起点XY
            Rt_xy = self._pixel_dir[..., 0:2].detach() # 射线方向向量XY
            C_xy = self._cylinders_list[i][0:2].detach().unsqueeze(0).unsqueeze(0).unsqueeze(0) # 圆柱XY
            Rs_z = self._depth_pos[:, 2].detach().unsqueeze(1).unsqueeze(1)  # 射线起点Z
            Rt_z = self._pixel_dir[:, :, :, 2].detach()  # 射线方向向量Z
            C_z = self._cylinders_list[i][2].detach().unsqueeze(0).unsqueeze(0).unsqueeze(0) # 圆柱Z
            R = self._cylinders_list[i][3].detach().unsqueeze(0).unsqueeze(0).unsqueeze(0) # 圆柱半径
            H = self._cylinders_list[i][4].detach().unsqueeze(0).unsqueeze(0).unsqueeze(0) # 圆柱高度
            # 圆柱曲面
            # 求根公式
            a = torch.square(torch.norm(Rt_xy, dim=-1, keepdim=True)).squeeze(-1)
            b = 2*torch.sum((Rs_xy-C_xy)*Rt_xy, dim=-1, keepdim=True).squeeze(-1)
            c = (torch.square(torch.norm(Rs_xy-C_xy, dim=-1, keepdim=True))-torch.square(R)).squeeze(-1)
            delta = torch.square(b)-4*a*c
            t_1_1 = torch.where(delta > 0.0, (-b+torch.sqrt(delta))/(2*a), torch.tensor(0.0, device=self._device))
            t_1_2 = torch.where(delta > 0.0, (-b-torch.sqrt(delta))/(2*a), torch.tensor(0.0, device=self._device))
            # 选取大于0且最小的/都小于0时为0.0
            t_1 = torch.where((t_1_1 > 0) & (t_1_2 > 0), torch.min(t_1_1, t_1_2), 
                    torch.where(t_1_1 > 0, t_1_1, 
                        torch.where(t_1_2 > 0, t_1_2, torch.tensor(0.0, device=self._device))))
            z = self._depth_pos[..., 2].unsqueeze(-1).unsqueeze(-1)+self._pixel_dir[..., 2]*t_1
            mask_valid_t_1 = t_1 > 0  
            mask_on_ground_1 = z > 0
            mask_in_region_1 = (z > C_z-H/2) & (z < C_z+H/2)
            mask_update_1 = (self._depth == 0) | (t_1 < self._depth)    # 之前深度未被更新/当前深度比之前小
            final_mask_1 = mask_valid_t_1 & mask_on_ground_1 & mask_in_region_1 & mask_update_1
            self._depth = torch.where(final_mask_1, t_1, self._depth)
            mask_inf = (t_1 > self._max_depth) & (self._max_depth == 0)    # 当前深度超过最大深度且原深度未被更新
            self._depth = torch.where(mask_inf, torch.tensor([float('inf')], dtype=torch.float32), self._depth) # 超过最大深度的部分设置为无穷大
            # 判断无人机相对于圆柱的位置
            up_cylinder = Rs_z > C_z+H/2
            down_cylinder = Rs_z < C_z-H/2
            # 圆柱顶面
            for i in range(up_cylinder.size(0)):
                # 无人机在圆柱上方
                if up_cylinder[i, 0, 0]:
                    t_2 = ((C_z+H/2-Rs_z) / Rt_z)
                    R_t_1 = torch.norm((Rs_xy+Rt_xy*t_2.unsqueeze(-1))-C_xy, dim=-1)    # 计算半径
                    mask_valid_t_2 = t_2 > 0  
                    mask_on_ground_2 = C_z > 0
                    mask_in_region_2 = R_t_1 <= R
                    mask_update_2 = (self._depth == 0) | (t_2 < self._depth)    # 之前深度未被更新/当前深度比之前小
                    final_mask_2 = mask_valid_t_2 & mask_on_ground_2 & mask_in_region_2 & mask_update_2
                    self._depth = torch.where(final_mask_2, t_2, self._depth)
                    mask_inf = (t_2 > self._max_depth) & (self._max_depth == 0)    # 当前深度超过最大深度且原深度未被更新
                    self._depth = torch.where(mask_inf, torch.tensor([float('inf')], dtype=torch.float32), self._depth) # 超过最大深度的部分设置为无穷大

            # 圆柱底面
            for i in range(down_cylinder.size(0)):
                # 无人机在圆柱下方
                if down_cylinder[i, 0, 0]:
                    t_3 = ((C_z-H/2-Rs_z) / Rt_z)
                    R_t_2 = torch.norm((Rs_xy+Rt_xy*t_3.unsqueeze(-1))-C_xy, dim=-1)  # 计算半径
                    mask_valid_t_3 = t_3 > 0  
                    mask_on_ground_3 = C_z > 0
                    mask_in_region_3 = R_t_2 <= R
                    mask_update_3 = (self._depth == 0) | (t_3 < self._depth)    # 之前深度未被更新/当前深度比之前小
                    final_mask_3 = mask_valid_t_3 & mask_on_ground_3 & mask_in_region_3 & mask_update_3
                    self._depth = torch.where(final_mask_3, t_3, self._depth)
                    mask_inf = (t_3 > self._max_depth) & (self._max_depth == 0)    # 当前深度超过最大深度且原深度未被更新
                    self._depth = torch.where(mask_inf, torch.tensor([float('inf')], dtype=torch.float32), self._depth) # 超过最大深度的部分设置为无穷大
            
            if noise:
                # 选取有效区域加入噪声
                mask_noise = (self._depth >= self._min_depth) & (self._depth <= self._max_depth)
                # 深度图传感器误差
                offset_noise = torch.clamp(torch.randn_like(self._depth)*(noise_range / 3), min=-noise_range, max=noise_range)  # 或其他随机分布
                self._depth[mask_noise] += offset_noise[mask_noise]
                # 深度图黑洞
                black_hole_noise = torch.randn_like(self._depth) < stats.norm.ppf(black_hole_prob)   # 每个位置有5%概率出现黑洞
                black_hole = torch.zeros_like(self._depth)
                self._depth[black_hole_noise] = black_hole[black_hole_noise]

    """
        @ 深度相机有限高圆柱体距离计算
    """
    def _cylinder_distance(self):
        for i in range(len(self._cylinders_list)):
            drone_pos_xy = self._drone_pos[:, 0:2]  # 无人机XY
            C_xy = self._cylinders_list[i][0:2].unsqueeze(0) # 圆柱XY
            drone_pos_z = self._drone_pos[:, 2].unsqueeze(1)  # 无人机Z
            C_z = self._cylinders_list[i][2].unsqueeze(0) # 圆柱Z
            R = self._cylinders_list[i][3].unsqueeze(0) # 圆柱半径
            H = self._cylinders_list[i][4].unsqueeze(0) # 圆柱高度

            D_xy = torch.norm(drone_pos_xy-C_xy, dim=-1, keepdim=True)  # 计算无人机到圆柱中轴距离
            D_z = torch.norm(drone_pos_z-C_z, dim=-1, keepdim=True) # 计算无人机到圆柱中心平面距离
            
            mask_xy_out = D_xy > R   # 质点XY在圆柱外
            mask_z_out = D_z > H/2   # 质点Z在圆柱外

            mask_xy_z_in = (H/2-D_z) > (R-D_xy) # 在圆柱内部，质点更靠近弧面

            # 欧氏距离计算
            D = torch.where(
                mask_xy_out | mask_z_out,   
                torch.where(    # 当质点在圆柱外
                    mask_xy_out & mask_z_out,   
                    torch.sqrt(torch.square(D_xy-R)+torch.square(D_z-H/2)), # 当质点在上顶面上/下顶面下，弧面外
                    torch.where(
                        mask_xy_out,
                        D_xy-R, # mask_xy_out=True,mask_z_out=False
                        D_z-H/2 # mask_xy_out=False,mask_z_out=True
                    )
                ),
                torch.where(    # 当质点在圆柱内
                    mask_xy_z_in,
                    D_xy-R, # 更靠近弧面时，最近距离
                    D_z-H/2 # 更靠近上下顶面时，最近距离
                )
            )

            # 最近距离
            mask = D < self._distance
            self._distance = torch.where(mask, D, self._distance)
            # 碰撞检测
            collision_flag = torch.any(D.detach() <= 0, dim=-1, keepdim=True)
            self._is_collision = self._is_collision | collision_flag

    """
        @ 深度相机有效检测
        t:计算得到的距离
    """
    def _depth_validity_check(self):
        # 求中心区域的平均值
        depth_center = self._depth[:, round(self._res_H*0.45):round(self._res_H*0.55), round(self._res_W*0.45):round(self._res_W*0.55)]
        depth_center_mean = torch.mean(depth_center, dim=(-2, -1))
        # 掩膜
        mask_vaild = depth_center_mean > self._min_depth    # 中心区域的平均距离大于最小距离阈值
        mask_in_range = (self._depth >= self._min_depth) & (self._depth <= self._max_depth)
        final_mask = mask_vaild.unsqueeze(-1).unsqueeze(-1) & mask_in_range
        self._depth = torch.where(final_mask, self._depth, torch.tensor(0.0))
        mask_inf = (self._depth > self._max_depth)
        self._depth = torch.where(mask_inf, torch.tensor(0.0), self._depth) # 将无穷大的像素变为0

    """
        @ 适配张量维度到 batch_size
    """
    def _adapt(self, tensor):
        if tensor.size(0) == 1 and tensor.size(0) != self._batch_size:
            repeat_times = [self._batch_size] + [1] * (tensor.dim() - 1)
            return tensor.repeat(repeat_times)
        elif tensor.size(0) == self._batch_size:
            return tensor
        else:
            raise Exception("[ERROR] tensor can't adapt to batchsize")

    '''
        @ GEOM状态:有梯度
    '''
    @property
    def drone_pos(self):
        return self._drone_pos
    @property
    def drone_acc(self):
        return self._acc
    @property
    def drone_vel(self):
        return self._vel
    @property
    def drone_ang_vel(self):
        return self._ang_vel
    @property   # 度
    def drone_euler(self):
        self._drone_euler = util.R_to_euler(self._drone_R)
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
    def closest_distance(self):
        return self._distance
    '''
        @ GEOM状态:无梯度
    '''
    @property
    def depth(self):
        return self._depth
    @property   # 得到各个无人机的碰撞状态张量:[batch_size, 1]
    def collision_state(self):
        return self._is_collision