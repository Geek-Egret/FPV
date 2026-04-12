import torch
import cv2
import numpy as np
from scipy import stats

import env.pid as pid
import env.util as util
import env.sensor as sensor

"""
    @ 无人机
"""
class drone:
    """
        @ 无人机初始化
        device:运行设备:cpu/cuda
        init_pos:初始位置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):m
        init_euler:初始姿态角度:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):度
        mass:质量:kg
        T_max:最大推力:N
        ang_vel_max:最大角速度:[x,y,z]:度/s
    """
    def __init__(
        self,
        device,
        init_pos, 
        init_euler, 
        mass, 
        T_max, 
        ang_vel_max,
        collision_radius
    ):
        self._device = device
        self._init_pos = init_pos
        self._init_euler = init_euler
        self.pose_set(init_pos, init_euler)
        self.mass = mass
        self.T_max = T_max
        self.ang_vel_max = ang_vel_max
        self.collision_radius = collision_radius
        self.sensor_list = []
    
    """
        @ 无人机传感器绑定
    """
    def sensor_bind(self, sensor):
        self.sensor_list.append(dict(sensor_type=sensor.sensor_type, sensor_class=sensor))
    
    """
        @ 无人机位姿重置
    """
    def pose_reset(self):
        self.pose_set(self._init_pos, self._init_euler)   
    
    """
        @ 无人机位姿态设置
    """
    def pose_set(self, pos, euler):
        self.pos = pos
        self.euler = euler
        # 更新传感器位姿
        for _sensor in self.sensor_list:
            if _sensor['sensor_type'] == 'depth':
                _sensor['sensor_class'].pose_set(self.pos, self.euler)

    

    
