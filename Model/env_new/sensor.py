import torch
import cv2
import numpy as np
from scipy import stats

import env.pid as pid
import env.util as util

"""
    @ 深度相机
    init_pos:初始位置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):m
    init_euler:初始姿态角度:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):度
    mass:质量:kg
    T_max:最大推力:N
    ang_vel_max:最大角速度:[x,y,z]:度/s
"""
def add_depth_camera(
    self,
    base_link,
    pos_offset, 
    euler_offset,  
    res_W, 
    res_H, 
    fov_H, 
    fov_V, 
    min_depth, 
    max_depth):