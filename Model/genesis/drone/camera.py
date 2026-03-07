import numpy as np
import genesis as gs
import genesis.utils.geom as gu
import torch

# 更新位姿
def update_pos(entity, pos_offset=(0.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    # 获取base坐标系相对于world的四元数、欧拉角和旋转矩阵
    base_quat = entity.get_quat()
    base_euler = gu.quat_to_xyz(base_quat) 
    R_base_world = gu.quat_to_R(base_quat)
    # 获取base坐标系位姿
    base_pos = entity.get_pos()
    # 转为tensor
    device = base_pos.device if hasattr(base_pos, 'device') else 'cpu'
    pos_offset_tensor = torch.tensor(pos_offset, dtype=torch.float32, device=device)
    forward_tensor = torch.tensor(forward, dtype=torch.float32, device=device)
    up_tensor = torch.tensor(up, dtype=torch.float32, device=device)
    # 计算相对于world的深度相机位姿
    forward_world = torch.matmul(R_base_world,  forward_tensor)
    forward_world = forward_world / torch.linalg.norm(forward_world)    # 归一化
    up_new= torch.matmul(R_base_world,  up_tensor)
    up_new = up_new / torch.linalg.norm(up_new)    # 归一化
    pos_offset_world = torch.matmul(R_base_world,  pos_offset_tensor)
    pos_new = base_pos+pos_offset_world
    lookat_new = pos_new + forward_world * 2.0

    return pos_new, lookat_new, up_new