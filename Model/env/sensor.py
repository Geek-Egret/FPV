import torch
import cv2
import numpy as np
from scipy import stats

import env.pid as pid
import env.util as util

"""
    @ 深度相机
"""
class depth:
    """
        @ 深度相机初始化
        device:运行设备:cpu/cuda
        pos_offset:深度相机相较于baselink的位置偏置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):m
        euler_offset:深度相机相较于baselink的姿态偏置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):度
        res_W:分辨率W
        res_H:分辨率H
        fov_H:水平视场角:度
        fov_V:垂直视场角:度
        min_depth:最近深度:m
        max_depth:最大深度:m
        noise_range:噪声范围
        black_hole_prob:深度图黑洞出现概率
    """
    def __init__(
        self,
        device,
        pos_offset, 
        euler_offset,  
        res_W, 
        res_H, 
        fov_H, 
        fov_V, 
        min_depth, 
        max_depth,
        noise_range,
        black_hole_prob
    ):
        self._device = device
        self.sensor_type = 'depth'
        self.pos = None
        self.R = None
        self.pos_offset = pos_offset
        self.R_offset = util.euler_to_R(euler_offset)
        self.R_offset = torch.where(torch.abs(self.R_offset) < 1e-8, torch.tensor(0.0, device=self._device), self.R_offset)
        self.res_W = res_W
        self.res_H = res_H
        self.fov_H = fov_H
        self.fov_V = fov_V
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.noise_range = noise_range
        self.black_hole_prob = black_hole_prob

    """
        @ 深度相机位姿设置
        pos:深度相机的位置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):m
        euler:深度相机的姿态:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):度
    """
    def pose_set(self, pos, euler):
        R = util.euler_to_R(euler)
        R = torch.where(torch.abs(R) < 1e-8, torch.tensor(0.0, device=self._device), R)
        self.pos = pos+torch.matmul(R, self.pos_offset.unsqueeze(-1)).squeeze(-1)
        self.pos = torch.where(torch.abs(self.pos) < 1e-8, torch.tensor(0.0, device=self._device), self.pos)
        self.R = torch.matmul(R, self.R_offset.transpose(-1, -2))
    
    