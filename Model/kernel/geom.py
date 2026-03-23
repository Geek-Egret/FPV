import torch

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
        safty_radius:无人机安全半径:m
        init_pos:初始位置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device):m
        init_euler:初始姿态角度:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device):度
        pos_offset:深度相机相较于无人机的位置偏置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device):m
        euler_offset:深度相机相较于无人机的姿态偏置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device):度
        mass:质量:kg
        T_max:最大推力:N
        ang_vel_max:最大角速度:[x,y,z]:度/s
        res_W:分辨率W
        res_H:分辨率H
        fov_H:水平视场角:度
        fov_V:垂直视场角:度
    """
    def __init__(self, batch_size, device, dt, safty_radius, init_pos, init_euler, 
                 pos_offset, euler_offset, mass, T_max, ang_vel_max,
                 res_W, res_H, fov_H, fov_V):
        self._threshold = 1e-8
        self._g = 9.81
        self._batch_size = batch_size
        self._device = device
        self._dt = dt
        self._safty_radius = safty_radius
        self._drone_pos = init_pos
        self._drone_euler = util.angle_to_rad(init_euler)
        self._depth_pos_offset = pos_offset
        self._depth_euler_offset =util.angle_to_rad(euler_offset)
        self._mass = mass
        self._T_max = T_max
        self._ang_vel_max = ang_vel_max
        self._res_W = res_W#torch.tensor(res_W, dtype=torch.float, device=device)  
        self._res_H = res_H#torch.tensor(res_H, dtype=torch.float, device=device)  
        self._fov_H = torch.tensor(util.angle_to_rad(fov_H), dtype=torch.float, device=device)  
        self._fov_V = torch.tensor(util.angle_to_rad(fov_V), dtype=torch.float, device=device)  

        # 计算重力
        self._G = torch.zeros(self._batch_size, 3)  
        self._G[:, 2] = -self._mass*self._g   

        # 计算深度相机相对于世界坐标系的位姿
        self._drone_R = util.euler_to_R(self._drone_euler)
        self._drone_R = torch.where(torch.abs(self._drone_R) < self._threshold, torch.tensor(0.0), self._drone_R)
        self._depth_drone_R = util.euler_to_R(self._depth_euler_offset)
        self._depth_drone_R = torch.where(torch.abs(self._depth_drone_R) < self._threshold, torch.tensor(0.0), self._depth_drone_R)
        self._depth_R = torch.matmul(self._drone_R, self._depth_drone_R.transpose(-1, -2))
        self._depth_R = torch.where(torch.abs(self._depth_R) < self._threshold, torch.tensor(0.0), self._depth_R)
        self._depth_pos = self._drone_pos+torch.matmul(self._drone_R, self._depth_pos_offset.unsqueeze(-1)).squeeze(-1)
        self._depth_pos = torch.where(torch.abs(self._depth_pos) < self._threshold, torch.tensor(0.0), self._depth_pos)

        # 计算深度相机成像平面像素位置
        # 假设成像平面在传感器前方单位位置,传感器位姿和深度相机位姿一致
        half_width = torch.tan(self._fov_H/2)
        half_height = torch.tan(self._fov_V/2)
        y = torch.linspace(-half_width, half_width, self._res_W, device=self._device)
        z = torch.linspace(-half_height, half_height, self._res_H, device=self._device)
        yy, zz = torch.meshgrid(y, -z, indexing='xy')
        xx = torch.ones_like(yy)
        self._pixel_pos = torch.stack([xx, yy, zz], dim=-1)
        self._pixel_pos = self._depth_pos+torch.matmul(self._drone_R, self._pixel_pos.unsqueeze(-1)).squeeze(-1)
        print(self._pixel_pos)

        # PID定义
        self._roll_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[0])
        self._pitch_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[1])
        self._yaw_pid = pid.pid(self._batch_size, 2.0, 0.0, 0.0, 0.0, self._ang_vel_max[2])

        # 定义
        self._acc = torch.zeros(self._batch_size, 3) 
        self._vel = torch.zeros(self._batch_size, 3) 
        self._ang_vel = torch.zeros(self._batch_size, 3) 
        self._T = torch.zeros(self._batch_size, 3)
        self._spheres_list = []
        self._cylinders_list = []
        self._boxes_list = []
        self._depth_ray = []

    """
        @ GEOM添加球体
        x,y,z:球体中心位置:m
        R:球体形状:m
    """
    def add_sphere(self, x, y, z, R):
        self._spheres_list.append([x, y, z, R])

    """
        @ GEOM添加圆柱
        x,y,z:圆柱中心位置:m
        R,H:圆柱形状:m
    """
    def add_cylinder(self, x, y, z, R, H):
        self._spheres_list.append([x, y, z, R, H])

    """
        @ GEOM添加方块
        x,y,z:方块中心位置:m
        L,W,H:方块形状:m
    """
    def add_box(self, x, y, z, L, W, H):
        self._boxes_list.append([x, y, z, L, W, H])

    """
        @ GEOM执行一步
        act:动作(姿态角度,推力比例):
        T_att:推力衰减比例:0.0-1.0
    """
    def step(self, act, T_att):
        self._solver(act=act, T_att=T_att)

    """
        @ 无人机动力学求解器
        act:动作(姿态角度,推力比例):torch.tensor([[x,y,z,T], ...], dtype=torch.float, device=device):度,0.0-1.0
        T_att:推力衰减比例:0.0-1.0
    """
    def _solver(self, act, T_att):  
        # 计算合加速度
        self._T = self._drone_R[:, 2]*self._T_max*(1-T_att)*act[:, 3].unsqueeze(1)
        sigma_force = self._T+self._G
        self._acc = sigma_force/self._mass  
        # 计算速度
        self._vel = self._vel+self._acc*self._dt
        # 计算位置
        self._drone_pos = self._drone_pos+self._vel*self._dt
        # PID计算角速度
        drone_euler = util.rad_to_angle(self._drone_euler)
        self._ang_vel[:, 0] = self._roll_pid.position(drone_euler[:, 0].unsqueeze(1), act[:, 0].unsqueeze(1))
        self._ang_vel[:, 1] = self._pitch_pid.position(drone_euler[:, 1].unsqueeze(1), act[:, 1].unsqueeze(1))
        self._ang_vel[:, 2] = self._yaw_pid.position(drone_euler[:, 2].unsqueeze(1), act[:, 2].unsqueeze(1))
        print(self._ang_vel)
        # 积分得到下一步姿态角度
        drone_euler[:, 0] = drone_euler[:, 0]+self._ang_vel[:, 0]*self._dt
        drone_euler[:, 1] = drone_euler[:, 1]+self._ang_vel[:, 1]*self._dt
        drone_euler[:, 2] = drone_euler[:, 2]+self._ang_vel[:, 2]*self._dt
        self._drone_euler = util.angle_to_rad(drone_euler)
        self._drone_R = util.euler_to_R(self._drone_euler)
        # 计算深度相机位姿
        self._depth_R = torch.matmul(self._drone_R, self._depth_drone_R.transpose(-1, -2))
        self._depth_pos = self._drone_pos+torch.matmul(self._drone_R, self._depth_pos_offset.unsqueeze(-1)).squeeze(-1)

    """
        @ 深度相机渲染深度图
    """
    def _render(self):
        # 假设成像平面在传感器前方单位位置
        half_width = torch.tan(self._fov_H/2)
        half_height = torch.tan(self._fov_V/2)
        x = torch.linspace(-half_width, half_width, self._res_W, device=self._device)
        y = torch.linspace(-half_height, half_height, self._res_H, device=self._device)
        xx, yy = torch.meshgrid(x, -y, indexing='xy')
        zz = -torch.ones_like(xx)
        dirs_camera = torch.stack([xx, yy, zz], dim=-1)
        dirs_camera = dirs_camera / torch.norm(dirs_camera, dim=-1, keepdim=True)
        # 旋转到世界坐标系
        dirs_world = torch.einsum('ij,khj->khi', self._depth_R, dirs_camera)

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