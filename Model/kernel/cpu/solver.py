import numpy as np
import torch 
import util

"""
    坐标定义:ENU东北天,FLU前左上
    外部传入的向量参考 ENU
    内部会转换为 FLU
"""
class solver:
    """
        dt:步长:s
        mass:质量:kg
        collider_radius:球体碰撞体积半径:m
        init_pos:初始位置:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        init_forward_vec:初始前向向量:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        init_up_vec:初始向上向量:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        alpha:控制器执行延迟参数:0.0-1.0:越大延迟越高，飞机月大延迟越高
        wind_dir:风向向量:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        wind_speed:风速:torch.tensor([speed]/[[speed0], ...], dtype=torch.float32):m
        drag_1:线性阻力系数
        drag_2:平方阻力系数
        z_drag:垂直方向的特殊阻力系数
        num:飞机数量
        device:设备:cpu/gpu/cuda
    """
    def __init__(self, dt, mass, collider_radius, init_pos, init_forward_vec, init_up_vec, alpha, wind_dir, wind_speed, drag_1, drag_2, z_drag, num, device):
        self._dt = torch.tensor(dt, dtype=torch.float32, device=device)
        self._mass = torch.tensor(mass, dtype=torch.float32, device=device)
        self._collider_radius = collider_radius
        self._pos_ENU = self._tensor_adapt(init_pos, num, 3)
        init_forward_vec = self._tensor_adapt(init_forward_vec, num, 3)
        self._forward_vec_ENU = self._tensor_nrom(init_forward_vec)
        init_up_vec = self._tensor_adapt(init_up_vec, num, 3)
        self._up_vec_ENU = self._tensor_nrom(init_up_vec)
         # 计算左向向量
        self._left_vec_ENU = torch.cross(self._up_vec_ENU, self._forward_vec_ENU, dim=-1)
        self._left_vec_ENU = self._tensor_nrom(self._left_vec_ENU)
        # 校正上向向量
        self._up_vec_ENU = torch.cross(self._forward_vec_ENU,  self._left_vec_ENU, dim=-1)
        self._up_vec_ENU = self._tensor_nrom(self._up_vec_ENU)
        # 构建旋转矩阵
        self._R_ENU = torch.stack([
            self._forward_vec_ENU, 
            self._left_vec_ENU,
            self._up_vec_ENU 
        ], dim=-1)
        self._alpha = torch.tensor(alpha, dtype=torch.float32, device=device) 
        wind_dir = self._tensor_adapt(wind_dir, num, 3)
        self._wind_dir_ENU = self._tensor_nrom(wind_dir)
        self._wind_speed_ENU = self._tensor_adapt(wind_speed, num, 1)*self._wind_dir_ENU
        self._drag_1 = drag_1
        self._drag_2 = drag_2
        self._z_drag = self._tensor_adapt(torch.tensor([1.0, 1.0, z_drag], dtype=torch.float32, device=device), num, 3)
        self._num = num
        self._device = device

        self._acc_ENU = self._tensor_adapt(torch.zeros(3, dtype=torch.float32, device=device), num, 3)              # 执行加速度
        self._vel_ENU = self._tensor_adapt(torch.zeros(3, dtype=torch.float32, device=device), num, 3)              # 速度
        self._g_ENU = self._tensor_adapt(torch.tensor([0.0, 0.0, 9.8], dtype=torch.float32, device=device), num, 3) # 重力加速度
        self._is_next_acc_set = False
    
    """
        pred_acc:下一步预测的线加速度:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m/s^2
        加速度的参考坐标系是机体坐标系
        返回:是否设置成功
    """
    def step(self, pred_acc):
        # 一阶且列数量为3/二阶且列数量为3，行数量为设置的并行数量
        if (pred_acc.dim() == 1 and pred_acc.shape[0] == 3 and self.num == 1) or (pred_acc.dim() == 2 and pred_acc.shape[1] == 3 and pred_acc.shape[0] == self._num):
            # 计算预测加速度,一阶加速度延迟(已经包含了因为飞机转动惯量导致的加速度增加延迟)
            self._acc_pred_ENU = self._pred_acc_ENU*self._alpha+pred_acc*(1-self._alpha)+self._g_ENU
            # 计算空气阻力加速度
            self._vel_rel_ENU = self._vel_ENU-self._wind_speed_ENU  # 相对风速的速度     
            self._vel_rel_FLU = util.ENU_to_FLU(self._vel_rel_ENU, self._R_ENU) # 将ENU的速度转为FLU的速度
            self._vel_rel_square_FLU = torch.sign(self._vel_rel_FLU)*torch.square(self._vel_rel_FLU)    # 按元素求平方
            self._acc_drag_1_FLU = self._vel_rel_FLU*self._z_drag*self._drag_1/self._mass         # 层流/低速时的粘性阻力加速度
            self._acc_drag_2_FLU = self._vel_rel_square_FLU*self._z_drag*self._drag_2/self._mass  # 湍流/高速时的阻力加速度
            self._acc_drag_1_ENU = util.FLU_to_ENU(self._a_drag_1_FLU, self._R_ENU)
            self._acc_drag_2_ENU = util.FLU_to_ENU(self._a_drag_2_FLU, self._R_ENU)
            # print(f"{self._a_drag_1_ENU}\n{self._a_drag_2_ENU}")
            # 计算执行加速度:重力加速度+空气阻力加速度+预测加速度
            self._acc_ENU = self._acc_pred_ENU-self._g_ENU-self._acc_drag_1_ENU-self._acc_drag_2_ENU
            # 计算旋转矩阵
            self.up_vec_ENU = self._up_vec_ENU = self._tensor_nrom(self._acc_ENU)   # 由于执行加速度有且仅由推力驱动，且推力始终平行于上向向量，及上向向量方向与执行加速度方向相同
            
            # self._R_ENU = 
            # 计算速度
            self._vel_ENU = self._vel_ENU+self._acc_ENU*self._dt
            # 计算POS
            self._pos_ENU += self._vel_ENU*self._dt
            self._is_next_acc_set = True
        else:
            self._is_next_acc_set = False
        return self._is_next_acc_set
    
    """
        返回:下一步加速度
        参考坐标系为ENU坐标系
    """
    @property
    def next_acc(self):
        return self._acc_ENU
    
    """
        返回:下一步线速度
        参考坐标系为ENU坐标系
    """
    @property
    def next_vel(self):
        return self._vel_ENU

    """
        若执行了set_pred_acc,则返回下一步,否则返回当前
        返回:下一步位置
        参考坐标系为ENU坐标系
    """
    @property
    def next_pos(self):  
        return self._pos_ENU
    
    """
        若执行了set_pred_acc,则返回下一步,否则返回当前
        返回:下一步旋转矩阵
        参考坐标系为ENU坐标系
    """
    @property
    def next_R(self):
        return self._R_ENU
    
    # PRIVATE
    """
        1阶张量阶度适配到2阶
        tensor:张量
        rows_num:适配目标行数
        cols_num:传入张量的正确列数
        返回：适配后张量
    """
    def _tensor_adapt(self, tensor, rows_num, cols_num):
        new_tensor = None
        # print(tensor)
        if tensor.dim() == 1 and tensor.shape[0] == cols_num and rows_num > 1:
            new_tensor = tensor.unsqueeze(0).expand(rows_num, cols_num)    # 新增行维度后拷贝
        elif tensor.dim() == 1 and tensor.shape[0] == cols_num and rows_num == 1:
            new_tensor = tensor
        elif tensor.dim() == 2 and tensor.shape[0] == rows_num and tensor.shape[1] == cols_num:
            new_tensor = tensor
        elif tensor.dim() == 2 and tensor.shape[1] != cols_num:
            try:
                raise ValueError(f"{tensor}的列数为{tensor.shape[0]}不等于指定列数{cols_num}")
            except ValueError as e:
                print(e)
        elif tensor.dim() == 1 and tensor.shape[0] != cols_num:
            try:
                raise ValueError(f"{tensor}的列数为{tensor.shape[0]}不等于指定列数{cols_num}")
            except ValueError as e:
                print(e)
        elif tensor.dim() == 2 and tensor.shape[0] != rows_num:
            try:
                raise ValueError(f"{tensor}的行数为{tensor.shape[0]},无法转换")
            except ValueError as e:
                print(e)
        
        return new_tensor

    """
        张量归一化
        tensor:张量
        返回:归一化张量
    """
    def _tensor_nrom(self, tensor):
        tensor_norm = torch.norm(tensor, dim=-1, keepdim=True)
        new_tensor = tensor/(tensor_norm+1e-8) # 归一化
        return new_tensor