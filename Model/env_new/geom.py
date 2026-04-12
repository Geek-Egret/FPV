import torch
import cv2
import numpy as np
from scipy import stats

import env.pid as pid
import env.util as util

"""
    将机器人作为质点,深度相机对无人机有一个pos_offset和euler_offset
"""
class geom:
    """
        @ GEOM场景初始化
        batch_size:并行数量
        device:运行设备:cpu/cuda
        
        pos_offset:深度相机相较于无人机的位置偏置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):m
        euler_offset:深度相机相较于无人机的姿态偏置:torch.tensor([[x,y,z], ...], dtype=torch.float, device=device, requires_grad=True):度
        
        res_W:分辨率W
        res_H:分辨率H
        fov_H:水平视场角:度
        fov_V:垂直视场角:度
        min_depth:最近深度:m
        max_depth:最大深度:m
        collision_radius:碰撞半径:m
    """
    def __init__(
        self, 
        batch_size, 
        device):
        self._batch_size = batch_size
        self._device = device
    
    """
        @ GEOM场景添加机器人
        robot:机器人类(dron)
        sensor:传感器类(depth, lidar)
    """
    def add_robot(
        self,
        robot,
        sensor
    ):