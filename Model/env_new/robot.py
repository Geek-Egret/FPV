import torch
import cv2
import numpy as np
from scipy import stats

import env.pid as pid
import env.util as util

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
        collision_radius):
    