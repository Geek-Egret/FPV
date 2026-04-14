import torch
import random
from typing import Union

import util as util
import sensor as sensor

"""
    @ 无人机
    内部角度单位:rad
"""
class drone:
    """
        @ 无人机初始化
        device:运行设备:cpu/cuda
        init_pos:初始位置:torch.tensor([x,y,z], dtype=torch.float, device=device, requires_grad=True):m
        init_euler:初始姿态角度:torch.tensor([x,y,z], dtype=torch.float, device=device, requires_grad=True):度
        mass:质量:kg
        T_max:最大推力:N
        collision_radius:碰撞半径:m
    """
    def __init__(
        self,
        device: str,
        init_pos: torch.Tensor, 
        init_euler: torch.Tensor, 
        mass: float, 
        T_max: float, 
        collision_radius: float
    ):
        self.type = 'drone'

        self._device = device
        self._init_pos = init_pos.detach()
        self._init_euler = util.angle_to_rad(init_euler.detach())
        self.pos = None
        self.euler = None
        self.mass = mass
        self.T_max = T_max
        self.collision_radius = collision_radius

        self._g = 9.81
        self.sensor_list = []
        self.cloest_distance = torch.zeros(1, device=self._device, dtype=torch.float, requires_grad=True).detach()
        self.is_collision = False
        self.acc = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=True).detach()
        self.vel = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=True).detach()
        self.ang_vel = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=True).detach()
        self._G = torch.tensor([0.0, 0.0, -self.mass*self._g], dtype=torch.float, device=device, requires_grad=True).detach()

        self._pose_set(init_pos.detach(), self._init_euler.detach())
    
    """
        @ 无人机传感器绑定
        sensor:传感器类:closest_dist/depth/lidar/...
    """
    def sensor_bind(self, sensor: Union[sensor.depth]):
        self.sensor_list.append(dict(type=sensor.type, sensor=sensor))
        self._pose_set(self.pos, self.euler)
    
    """
        @ 无人机复位
    """
    def reset(self):
        self._pose_set(self._init_pos, self._init_euler)   
        self.cloest_distance = torch.zeros(1, device=self._device, dtype=torch.float, requires_grad=True).detach()
        self.is_collision = False
        self.acc = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=True).detach()
        self.vel = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=True).detach()
        self.ang_vel = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=True).detach()

    """
        @ 重力加速度设置,若不使用则默认为9.81
        g:重力加速度
    """
    def g_set(self, g: float):
        self._g = g
        self._G[2] = torch.tensor([0.0, 0.0, -self.mass*self._g], dtype=torch.float, device=device, requires_grad=True)
    
    """
        @ 无人机动力学解算
        mode:解算模式:'euler+T_rate'/acc+yaw'/'ang_vel+T_rate':姿态+推力率/加速度+yaw/角速度+推力率
        T_att_range:推力衰减率范围:{'min':min, 'max':max}:0-1.0
        act:控制动作:torch.tensor([a,b,c,d], dtype=torch.float, device=device, requires_grad=True)
        alpha_1_range:延迟范围:{'min':min, 'max':max}:0-1.0
        dt:步长:s
    """
    def solver(
        self, 
        mode: str,
        T_att_range: dict, 
        act: torch.Tensor,
        alpha_1_range: dict, 
        dt: float
    ):
        
        collision_mask = float(not self.is_collision)
        T_att = random.uniform(T_att_range['min'], T_att_range['max'])
        alpha_1 = random.uniform(alpha_1_range['min'], alpha_1_range['max'])

        match mode:
            case 'euler+T_rate':
                R = util.euler_to_R(self.euler)
                # 计算合加速度
                T = R[:, 2]*self.T_max*(1-T_att)*act[3]
                sigma_force = T+self._G
                # 只为没有碰撞的无人机计算加速度，碰撞的无人机加速度为0.0
                self.acc = sigma_force/self.mass
                acc = self.acc*collision_mask
                # 计算速度
                self.vel = self.vel+acc*dt
                self.vel = self.vel*collision_mask
                # 计算姿态
                euler = util.rad_to_angle(self.euler)
                euler_next = euler+alpha_1*(act[0:3]-euler)
                self.ang_vel = (euler_next-euler)/dt
                # 设置位姿
                self._pose_set(self.pos+self.vel*dt, util.angle_to_rad(euler_next))
            case 'acc+yaw':
                pass
            case 'ang_vel+T_rate':
                pass
        
    """
        @ 无人机位姿态设置
        pos:位置:torch.tensor([x,y,z], dtype=torch.float, device=device, requires_grad=True):m
        euler:姿态角度:torch.tensor([x,y,z], dtype=torch.float, device=device, requires_grad=True):Rad
    """
    def _pose_set(self, 
        pos: torch.Tensor,
        euler: torch.Tensor
    ):
        self.pos = pos
        self.euler = euler
        # 更新传感器位姿
        for sensor_dict in self.sensor_list:
            if sensor_dict['type'] == 'closest_dist':
                sensor_dict['sensor'].pos_set(self.pos)
                self.is_collision = sensor_dict['sensor'].is_collision
            if sensor_dict['type'] == 'depth':
                sensor_dict['sensor'].pose_set(self.pos, self.euler)

    

    
