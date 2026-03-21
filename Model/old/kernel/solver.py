import numpy as np
import torch 
import math

import kernel.util as util
import kernel.pid as pid

epsilon = 1e-12

"""
    坐标定义:ENU东北天,FLU前左上
    无人机构型:X四轴四桨
    所有成员变量都是ENU坐标系下的
"""
class solver:
    """
        dt:步长:s
        mass:质量:kg
        T_max:最大推力:N
        T_max_att:最大推力衰减率:0.0-1.0
        ang_vel_max:无人机最大角速度:rad/s
        collider_radius:球体碰撞体积半径:m
        init_pos:初始位置:torch.tensor([x, y, z], dtype=torch.double):m:ENU
        init_forward_vec:初始前向向量:torch.tensor([x, y, z], dtype=torch.double):m:ENU
        init_up_vec:初始向上向量:torch.tensor([x, y, z], dtype=torch.double):m:ENU
        wind_dir:风向向量:torch.tensor([x, y, z], dtype=torch.double):m:ENU
        wind_speed:风速:torch.tensor([speed], dtype=torch.double):m:ENU
        drag_1:线性阻力系数
        drag_2:平方阻力系数
        z_drag:垂直方向的特殊阻力系数
        device:设备:cpu/cuda
    """
    def __init__(self, dt, mass, T_max, T_max_att, ang_vel_max, collider_radius, init_pos, init_forward_vec, init_up_vec, wind_dir, wind_speed, drag_1, drag_2, z_drag, device):
        self._dt = dt
        self._mass = mass
        self._T_max = T_max-T_max*T_max_att
        self._ang_vel_max = ang_vel_max
        self.collider_radius = collider_radius
        self._pos = init_pos
        self._forward_vec = util.tensor_nrom(init_forward_vec)
        self._up_vec = util.tensor_nrom(init_up_vec)
        # 计算左向向量
        self._left_vec = util.tensor_nrom(torch.cross(self._up_vec, self._forward_vec, dim=-1))
        # 校正上向向量
        self._up_vec = util.tensor_nrom(torch.cross(self._forward_vec,  self._left_vec, dim=-1))
        # 构建旋转矩阵
        self._R = torch.stack([
            self._forward_vec, 
            self._left_vec,
            self._up_vec 
        ], dim=-1)
        self._wind_dir = util.tensor_nrom(wind_dir)
        self._wind_speed = self._wind_dir*wind_speed
        self._drag_1 = drag_1
        self._drag_2 = drag_2
        self._z_drag = z_drag
        self._device = device
        self._is_next_acc_set = False

        self._roll_pid = pid.pid(0.1, 0.0, 0.0, 0.0, ang_vel_max)
        self._pitch_pid = pid.pid(0.1, 0.0, 0.0, 0.0, ang_vel_max)
        self._yaw_pid = pid.pid(0.1, 0.0, 0.0, 0.0, ang_vel_max)
        self._acc = torch.zeros(3, dtype=torch.double, device=device)
        self._vel = torch.zeros(3, dtype=torch.double, device=device)
        self._G = torch.tensor([0.0, 0.0, -9.8*mass], dtype=torch.double, device=device)    # 重力

        self._current_euler = util.R_to_euler(self._R)

    
    """
        pred_quat:模型预测的无人机姿态四元数:torch.tensor([x, y, z, w], ...], dtype=torch.double):ENU
        thrust:模型预测的无人机推力:0.0-1.0
    """
    def step(self, pred_quat, thrust):
        pred_euler = util.quat_to_euler(pred_quat)
        self._current_euler[0] = self._current_euler[0]+self._roll_pid.position(self._current_euler[0], pred_euler[0])
        self._current_euler[1] = self._current_euler[1]+self._roll_pid.position(self._current_euler[1], pred_euler[1])
        self._current_euler[2] = self._current_euler[2]+self._roll_pid.position(self._current_euler[2], pred_euler[2])
        self._R = util.euler_to_R(self._current_euler)
        thrust_xyz = self._T_max*thrust*self._R[2, :]   # 计算推力在ENU xyz上的分量
        
        vel_rel_ENU = self._wind_speed-self._vel
        vel_rel_FLU = util.ENU_to_FLU(vel_rel_ENU, self._R)
        vel_rel_square_FLU = torch.sign(vel_rel_FLU)*torch.square(vel_rel_FLU)
        F_drag_1_FLU = vel_rel_FLU*self._z_drag*self._drag_1
        F_drag_2_FLU = vel_rel_square_FLU*self._z_drag*self._drag_2
        F_drag_1_ENU = util.FLU_to_ENU(F_drag_1_FLU, self._R)
        F_drag_2_ENU = util.FLU_to_ENU(F_drag_2_FLU, self._R)

        F_net = thrust_xyz+self._G+F_drag_1_ENU+F_drag_2_ENU    # 合力
        self._acc = F_net/self._mass    # 合加速度
        self._vel = self._vel+self._acc*self._dt
        self._pos += self._vel*self._dt
    
    """
        返回:下一步加速度
        参考坐标系为ENU坐标系
    """
    @property
    def next_acc(self):
        return self._acc
    
    """
        返回:下一步线速度
        参考坐标系为ENU坐标系
    """
    @property
    def next_vel(self):
        return self._vel

    """
        若执行了set_pred_acc,则返回下一步,否则返回当前
        返回:下一步位置
        参考坐标系为ENU坐标系
    """
    @property
    def next_pos(self):  
        return self._pos
    
    """
        若执行了set_pred_acc,则返回下一步,否则返回当前
        返回:下一步旋转矩阵
        参考坐标系为ENU坐标系
    """
    @property
    def next_R(self):
        return self._R