import numpy as np
import torch 
import util

"""
    坐标定义:右手坐标系,X向前,Y向左,Z向上
"""
class solver:
    """
        dt:步长:s
        mass:质量:kg
        collider_radius:球体碰撞体积半径:m
        init_pos:初始位置:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        init_forward_vec:初始前向向量:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        init_up_vec:初始向上向量:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        com_offset:质心偏移量:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        alpha:控制器执行延迟参数:0.0-1.0:越大延迟越高
        wind_dir:风向向量:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m
        wind_speed:风速:torch.tensor([speed]/[[speed0], ...], dtype=torch.float32):m
        num:并行数量
    """
    def __init__(self, dt, mass, collider_radius, init_pos, init_forward_vec, init_up_vec, com_offset, alpha, wind_dir, wind_speed, num):
        self.dt = dt
        self.mass = mass
        self.collider_radius = collider_radius
        self.pos = self._tensor_adapt(init_pos, num, 3)
        init_forward_vec = self._tensor_adapt(init_forward_vec, num, 3)
        self.forward_vec = self._tensor_nrom(init_forward_vec)
        init_up_vec = self._tensor_adapt(init_up_vec, num, 3)
        self.up_vec = self._tensor_nrom(init_up_vec)
         # 计算右向向量
        right = torch.cross(self.forward_vec, self.up_vec, dim=-1)
        right = self._tensor_nrom(right)
        # 校正上向向量
        self.up_vec = torch.cross(right,  self.forward_vec, dim=-1)
        self.up_vec = self._tensor_nrom(self.up_vec)
        # 构建旋转矩阵
        self.forward_R = torch.stack([
            right, 
            self.forward_vec,
            self.up_vec 
        ], dim=-1)
        self.com_offset = com_offset
        self.alpha = alpha
        wind_dir = self._tensor_adapt(wind_dir, num, 3)
        self.wind_dir = self._tensor_nrom(wind_dir)
        self.wind_speed = self._tensor_adapt(wind_speed, num, 1)*self.wind_dir
        self.num = num

        self._acc = 0.0
        self._vel = 0.0
    
    """
        pred_acc:下一步预测的线加速度:torch.tensor([x, y, z]/[[x, y, z], ...], dtype=torch.float32):m/s^2
        返回:是否设置成功
    """
    def set_pred_acc(self, pred_acc):
        # 一维且列数量为3
        if pred_acc.dim() == self.num and pred_acc.shape[0] == 3:
            # 一阶加速度延迟
            self._acc = self._acc*self.alpha+pred_acc*(1-self.alpha)
            self._vel = self._vel+self._acc*self.dt
            self._is_next_acc_set = True
        # 二维且列数量为3，行数量为设置的并行数量
        elif pred_acc.dim() == 2 and pred_acc.shape[1] == 3 and pred_acc.shape[0] == self.num:
            # 一阶加速度延迟
            self._acc = self._acc*self.alpha+pred_acc*(1-self.alpha)
            self._vel = self._vel+self._acc*self.dt
            self._is_next_acc_set = True
        else:
            self._acc = 0.0
            self._is_next_acc_set = False
        return self._is_next_acc_set
    
    """
        返回下一步加速度
    """
    @property
    def next_acc(self):
        return self._acc
    
    """
        返回下一步线速度
    """
    @property
    def next_vel(self):
        return self._vel

    """
        返回:下一步位置
    """
    @property
    def next_pos(self):
        if self._is_pred_acc_set and self._is_pred_vel_set:
            print("a")
        else:
            self.pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        return self.pos
    
    """
        返回:下一步旋转矩阵
    """
    # @property
    # def next_R(self):
    #     return self._pos
    
    """
        private
    """
    # def _get

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
        if tensor.dim() == 1 and tensor.shape[0] == cols_num:
            new_tensor = tensor.unsqueeze(0).expand(rows_num, cols_num)    # 新增行维度后拷贝
        if tensor.dim() == 2 and tensor.shape[1] != cols_num:
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
        if tensor.dim() == 2 and tensor.shape[0] == rows_num and tensor.shape[1] == cols_num:
            new_tensor = tensor
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