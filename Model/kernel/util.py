import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math


"""
    欧拉角->旋转矩阵
"""
def euler_to_R(euler, convention='zyx'):
    shape = euler.shape[:-1]
    euler = euler.reshape(-1, 3)
    
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    
    if convention == 'zyx':  # 航空顺序: yaw-pitch-roll
        R = torch.stack([
            cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr,
            sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr,
            -sp,   cp*sr,             cp*cr
        ], dim=-1).reshape(-1, 3, 3)
    
    elif convention == 'xyz':  # 机器人顺序
        R = torch.stack([
            cp*cy, -cp*sy, sp,
            cr*sy + sr*sp*cy, cr*cy - sr*sp*sy, -sr*cp,
            sr*sy - cr*sp*cy, sr*cy + cr*sp*sy, cr*cp
        ], dim=-1).reshape(-1, 3, 3)
    
    return R.reshape(*shape, 3, 3)

"""
    旋转矩阵->欧拉角
"""
def R_to_euler(R, convention='zyx'):
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    if convention == 'zyx':
        # 检查万向锁
        sy = torch.sqrt(R[..., 0, 0]**2 + R[..., 1, 0]**2)
        singular = sy < 1e-6
        
        roll = torch.zeros_like(sy)
        pitch = torch.zeros_like(sy)
        yaw = torch.zeros_like(sy)
        
        # 非奇异情况
        mask = ~singular
        roll[mask] = torch.atan2(R[mask, 2, 1], R[mask, 2, 2])
        pitch[mask] = torch.atan2(-R[mask, 2, 0], sy[mask])
        yaw[mask] = torch.atan2(R[mask, 1, 0], R[mask, 0, 0])
        
        # 奇异情况 (万向锁)
        roll[singular] = torch.atan2(R[singular, 0, 1], R[singular, 0, 2])
        pitch[singular] = torch.atan2(-R[singular, 2, 0], sy[singular])
        yaw[singular] = 0
        
        euler = torch.stack([roll, pitch, yaw], dim=-1)
    
    elif convention == 'xyz':
        sy = torch.sqrt(R[..., 1, 2]**2 + R[..., 2, 2]**2)
        singular = sy < 1e-6
        
        roll = torch.zeros_like(sy)
        pitch = torch.zeros_like(sy)
        yaw = torch.zeros_like(sy)
        
        mask = ~singular
        roll[mask] = torch.atan2(-R[mask, 1, 2], R[mask, 2, 2])
        pitch[mask] = torch.atan2(R[mask, 0, 2], sy[mask])
        yaw[mask] = torch.atan2(-R[mask, 0, 1], R[mask, 0, 0])
        
        roll[singular] = torch.atan2(-R[singular, 1, 2], R[singular, 2, 2])
        pitch[singular] = torch.atan2(R[singular, 0, 2], sy[singular])
        yaw[singular] = 0
        
        euler = torch.stack([roll, pitch, yaw], dim=-1)
    
    return euler.reshape(*shape, 3)

"""
    四元数->旋转矩阵
"""
def quat_to_R(q):
    shape = q.shape[:-1]
    q = q.reshape(-1, 4)
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.stack([
        1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w,
        2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y
    ], dim=-1).reshape(-1, 3, 3)
    
    return R.reshape(*shape, 3, 3)

"""
    旋转矩阵->四元数
"""
def R_to_quat(R):
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    batch_size = R.shape[0]
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    mask1 = trace > 0
    if mask1.any():
        s = 0.5 / torch.sqrt(trace[mask1] + 1.0)
        q[mask1, 0] = 0.25 / s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) * s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) * s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) * s
    
    mask2 = ~mask1
    if mask2.any():
        # 找到最大对角元素
        diag = torch.stack([
            R[mask2, 0, 0],
            R[mask2, 1, 1],
            R[mask2, 2, 2]
        ], dim=-1)
        max_diag, max_idx = torch.max(diag, dim=-1)
        
        for i in range(3):
            mask_i = (max_idx == i) & mask2
            if not mask_i.any():
                continue
            
            if i == 0:  # R[0,0] 最大
                s = 2.0 * torch.sqrt(1.0 + R[mask_i, 0, 0] - R[mask_i, 1, 1] - R[mask_i, 2, 2])
                q[mask_i, 0] = (R[mask_i, 2, 1] - R[mask_i, 1, 2]) / s
                q[mask_i, 1] = 0.25 * s
                q[mask_i, 2] = (R[mask_i, 0, 1] + R[mask_i, 1, 0]) / s
                q[mask_i, 3] = (R[mask_i, 0, 2] + R[mask_i, 2, 0]) / s
            
            elif i == 1:  # R[1,1] 最大
                s = 2.0 * torch.sqrt(1.0 + R[mask_i, 1, 1] - R[mask_i, 0, 0] - R[mask_i, 2, 2])
                q[mask_i, 0] = (R[mask_i, 0, 2] - R[mask_i, 2, 0]) / s
                q[mask_i, 1] = (R[mask_i, 0, 1] + R[mask_i, 1, 0]) / s
                q[mask_i, 2] = 0.25 * s
                q[mask_i, 3] = (R[mask_i, 1, 2] + R[mask_i, 2, 1]) / s
            
            else:  # R[2,2] 最大
                s = 2.0 * torch.sqrt(1.0 + R[mask_i, 2, 2] - R[mask_i, 0, 0] - R[mask_i, 1, 1])
                q[mask_i, 0] = (R[mask_i, 1, 0] - R[mask_i, 0, 1]) / s
                q[mask_i, 1] = (R[mask_i, 0, 2] + R[mask_i, 2, 0]) / s
                q[mask_i, 2] = (R[mask_i, 1, 2] + R[mask_i, 2, 1]) / s
                q[mask_i, 3] = 0.25 * s

    q = F.normalize(q, p=2, dim=-1)
    
    return q.reshape(*shape, 4)

"""
    四元数->欧拉角
"""
def quat_to_euler(q, convention='zyx'):
    R = quat_to_R(q)
    return R_to_euler(R, convention)

"""
    欧拉角->四元数
"""
def euler_to_quat(euler, convention='zyx'):
    R = euler_to_R(euler, convention)
    return R_to_quat(R)

"""
    弧度->角度
"""
def rad_to_angle(rad):
    return rad*180.0/math.pi

"""
    角度->弧度
"""
def angle_to_rad(angle):
    return angle*math.pi/180.0

"""
    东北天:世界->前左上:机器人
    ENU->FLU
"""
def ENU_to_FLU(ENU, R):
    if ENU.dim() == 1:
        FLU = torch.matmul(ENU, R)
    elif ENU.dim() > 1:
        FLU = torch.matmul(ENU.unsqueeze(1), R).squeeze(1)
    return FLU

"""
    前左上:机器人->东北天:世界
    FLU->ENU
"""
def FLU_to_ENU(FLU, R):
    if FLU.dim() == 1:
        ENU = torch.matmul(FLU, R.transpose(-1, -2))
    elif FLU.dim() > 1:
        ENU = torch.matmul(FLU.unsqueeze(1), R.transpose(-1, -2)).squeeze(1)
    return ENU

"""
    模型输出->东北天:世界
    MODEL->ENU
"""
def MODEL_to_ENU(MODEL, R):
    epsilon = 1e-12
    forward_vec = torch.where(torch.abs(R[:, 0]) < epsilon, torch.tensor(0.0), R[:, 0]) # 获取未与上向向量正交的前向向量
    forward_vec_x = torch.where(torch.abs(R[0, 0]) < epsilon, torch.tensor(0.0), R[0, 0]) 
    forward_vec_y = torch.where(torch.abs(R[0, 0]) < epsilon, torch.tensor(0.0), R[0, 0]) 