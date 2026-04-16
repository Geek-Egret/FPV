import torch
import cv2
import random
from scipy import stats

import env.util as util

"""
    @ 最近距离
"""
class cloest_dist:
    """
        @ 最近距离初始化
        device:运行设备:cpu/cuda
    """
    def __init__(self, device: str):
        self.type = 'closest_dist'

        self._device = device
        self.pos_prev = None
        self.pos = None
        self.distance = torch.full((1,), float('inf'), device=self._device)
        self.is_collision = False

    """
        @ 复位
    """
    def reset(self):
        self.distance = torch.full((1,), float('inf'), device=self._device)
        self.pos_prev = self.pos.detach()
        self.is_collision = False

    """
        @ 位置设置
        pos:质点位置:torch.tensor([x,y,z], dtype=torch.float, device=device, requires_grad=True):m
    """
    def pos_set(self, pos):
        if self.pos_prev is not None:
            self.pos_prev = self.pos.clone().detach()
        self.pos = pos
    
    """
        @ 最近距离计算
        is_ground_exist:地面是否存在
        collision_radius:碰撞半径:m
        sphere_list:渲染的球体列表
        cylinder_list:渲染的球体列表张量
    """
    def dist_calc(self, 
        is_ground_exist: bool,
        collision_radius: float,
        sphere_list: torch.Tensor, 
        cylinder_list: torch.Tensor, 
    ):
        """ 当地面存在 """
        if is_ground_exist:
            D = self.pos[2].unsqueeze(-1)
            # 最近距离
            mask = D < self.distance
            self.distance = torch.where(mask, D, self.distance)
            # 碰撞检测
            collision_flag = torch.any(D.detach() < collision_radius).item()
            self.is_collision = self.is_collision | collision_flag
        
        if torch.any(sphere_list != 0.0):
            self._sphere_dist(collision_radius, sphere_list)
        if torch.any(cylinder_list != 0.0):
            self._cylinder_distance(collision_radius, cylinder_list)
        
    """
        @ 球体距离计算
        collision_radius:碰撞半径:m
        sphere_list:渲染的球体列表张量
    """
    def _sphere_dist(self, collision_radius, sphere_list):
        C = sphere_list[:, 0:3] # 球心
        R = sphere_list[:, 3] # 球体半径
        D_all = torch.norm(self.pos-C, dim=-1)-R
        D = D_all.min(dim=0)[0]
        # 最近距离
        mask = D < self.distance
        self.distance = torch.where(mask, D, self.distance)
        # 流形碰撞检测
        D_sweeping_body = torch.zeros_like(D).detach()
        if self.pos_prev != None and torch.any(self.pos != self.pos_prev):
            w = (C - self.pos_prev).detach()           # 上一刻位置指向球心
            v = (self.pos - self.pos_prev).detach()    # 上一刻位置指向当前位置
            u = (C - self.pos).detach()                # 当前位置指向球心
            d = torch.sum(w*v, dim=-1)/torch.norm(v, dim=-1)    # 球心在两个位置连线直线上的投影到上一刻位置的距离，带符号
            mask_1 = d <= 0   # 最近距离是球心到上一刻位置
            mask_2 = d >= torch.norm(v, dim=-1) # 最近距离是球心到当前位置
            D_sweeping_body_all = torch.where(
                mask_1,
                torch.where(
                    mask_2,
                    D.detach(),   # 当前位置和上一刻位置重叠,不可能到这的
                    torch.norm(w, dim=-1)-R   #  最近距离是球心到上一刻位置
                ),
                torch.where(
                    mask_2,
                    torch.norm(u, dim=-1)-R,  #  最近距离是球心到当前位置
                    torch.sqrt(torch.square(torch.norm(w, dim=-1))-torch.square(d))-R
                )
            )
            D_sweeping_body = (D_sweeping_body_all.min(dim=0)[0]).detach()
        else:
            D_sweeping_body = D.detach()
        collision_flag = torch.any(D_sweeping_body.detach() <= collision_radius).item()
        self.is_collision = self.is_collision | collision_flag
    
    """
        @ 深度相机有限高圆柱体距离计算
        collision_radius:碰撞半径:m
        cylinder_list:渲染的球体列表张量
    """
    def _cylinder_distance(self, collision_radius, cylinder_list):
        pos_xy = self.pos[0:2]  # 无人机XY
        C_xy = cylinder_list[:, 0:2] # 圆柱XY
        pos_z = self.pos[2].unsqueeze(0)  # 无人机Z
        C_z = cylinder_list[:, 2] # 圆柱Z
        R = cylinder_list[:, 3] # 圆柱半径
        H = cylinder_list[:, 4] # 圆柱高度

        D_xy = torch.norm(pos_xy-C_xy, dim=-1)  # 计算质点到圆柱中轴距离
        D_z = torch.abs(pos_z-C_z) # 计算质点到圆柱中心平面距离
        
        mask_xy_out = D_xy > R   # 质点XY在圆柱外
        mask_z_out = D_z > H/2   # 质点Z在圆柱外

        mask_xy_z_in = (H/2-D_z) > (R-D_xy) # 在圆柱内部，质点更靠近弧面

        # 欧氏距离计算
        D_all = torch.where(
            mask_xy_out | mask_z_out,   
            torch.where(    # 当质点在圆柱外
                mask_xy_out & mask_z_out,   
                torch.sqrt(torch.square(D_xy-R)+torch.square(D_z-H/2)), # 当质点在上顶面上/下顶面下，弧面外
                torch.where(
                    mask_xy_out,
                    D_xy-R, # mask_xy_out=True,mask_z_out=False
                    D_z-H/2 # mask_xy_out=False,mask_z_out=True
                )
            ),
            torch.where(    # 当质点在圆柱内
                mask_xy_z_in,
                D_xy-R, # 更靠近弧面时，最近距离
                D_z-H/2 # 更靠近上下顶面时，最近距离
            )
        )
        D = D_all.min(dim=0)[0]

        # 最近距离
        mask = D < self.distance
        self.distance = torch.where(mask, D, self.distance)
        # 流形碰撞检测
        D_sweeping_body = torch.zeros_like(D).detach()
        if self.pos_prev != None and torch.any(self.pos != self.pos_prev):
            w = (C - self.pos_prev).detach()           # 上一刻位置指向球心
            v = (self.pos - self.pos_prev).detach()    # 上一刻位置指向当前位置
            u = (C - self.pos).detach()                # 当前位置指向球心
            d = torch.sum(w*v, dim=-1)/torch.norm(v, dim=-1)    # 球心在两个位置连线直线上的投影到上一刻位置的距离，带符号
            mask_1 = d <= 0   # 最近距离是球心到上一刻位置
            mask_2 = d >= torch.norm(v, dim=-1) # 最近距离是球心到当前位置
            D_sweeping_body_all = torch.where(
                mask_1,
                torch.where(
                    mask_2,
                    D.detach(),   # 当前位置和上一刻位置重叠,不可能到这的
                    torch.norm(w, dim=-1)-R   #  最近距离是球心到上一刻位置
                ),
                torch.where(
                    mask_2,
                    torch.norm(u, dim=-1)-R,  #  最近距离是球心到当前位置
                    torch.sqrt(torch.square(torch.norm(w, dim=-1))-torch.square(d))-R
                )
            )
            D_sweeping_body = (D_sweeping_body_all.min(dim=0)[0]).detach()
        else:
            D_sweeping_body = D.detach()
        collision_flag = torch.any(D_sweeping_body.detach() <= collision_radius).item()
        self.is_collision = self.is_collision | collision_flag
        


"""
    @ 深度相机
"""
class depth:
    """
        @ 深度相机初始化
        device:运行设备:cpu/cuda
        pos_offset:深度相机相较于baselink的位置偏置:torch.tensor([x,y,z], dtype=torch.float, device=device, requires_grad=False):m
        euler_offset:深度相机相较于baselink的姿态偏置:torch.tensor([x,y,z], dtype=torch.float, device=device, requires_grad=False):度
        res_W:分辨率W
        res_H:分辨率H
        fov_H:水平视场角:度
        fov_V:垂直视场角:度
        min_depth:最近深度:m
        max_depth:最大深度:m
        noise_range:噪声范围:['min':min, 'max':max]
        black_hole_prob:深度图黑洞出现概率
    """
    @torch.no_grad()
    def __init__(
        self,
        device: str,
        pos_offset: torch.Tensor, 
        euler_offset: torch.Tensor,  
        res_W: float, 
        res_H: float, 
        fov_H: float, 
        fov_V: float, 
        min_depth: float, 
        max_depth: float,
        noise_range: dict,
        black_hole_prob: float
    ):
        self.type = 'depth'

        self._device = device
        self.pos = None
        self.R = None
        self.pos_offset = pos_offset
        self.R_offset = util.euler_to_R(euler_offset)
        self.R_offset = torch.where(torch.abs(self.R_offset) < 1e-8, torch.tensor(0.0, dtype=torch.float, device=self._device), self.R_offset)
        self.res_W = res_W
        self.res_H = res_H
        self.fov_H = torch.tensor(util.angle_to_rad(fov_H), dtype=torch.float, device=self._device)  
        self.fov_V = torch.tensor(util.angle_to_rad(fov_V), dtype=torch.float, device=self._device)  
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.noise_range = random.uniform(noise_range['min'], noise_range['max'])
        self.black_hole_prob = black_hole_prob

        self.depth = torch.zeros(self.res_H, self.res_W, dtype=torch.float, device=self._device)   # 深度图

        # 计算深度相机成像平面像素位置方向向量
        # 假设成像平面在传感器前方单位位置,传感器位姿和深度相机位姿一致
        half_width = torch.tan(self.fov_H/2)
        half_height = torch.tan(self.fov_V/2)
        y = torch.linspace(half_width, -half_width, self.res_W, dtype=torch.float, device=self._device)
        z = torch.linspace(-half_height, half_height, self.res_H, dtype=torch.float, device=self._device)
        yy, zz = torch.meshgrid(y, -z, indexing='xy')
        xx = torch.ones_like(yy)
        self.camera_pixel_dir = torch.stack([xx, yy, zz], dim=-1)  # 在相机坐标系下的像素方向向量

    """
        @ 深度相机位姿设置
        pos:深度相机未偏移的位置:torch.tensor([x,y,z], dtype=torch.float, device=device, requires_grad=False):m
        euler:深度相机未偏移的姿态:torch.tensor([x,y,z], dtype=torch.float, device=device, requires_grad=False):Rad
    """
    @torch.no_grad()
    def pose_set(self, 
        pos: torch.Tensor, 
        euler: torch.Tensor
    ):
        R = util.euler_to_R(euler)
        R = torch.where(torch.abs(R) < 1e-8, torch.tensor(0.0, dtype=torch.float, device=self._device), R)
        self.pos = pos+torch.matmul(R, self.pos_offset.unsqueeze(-1)).squeeze(-1)
        self.pos = torch.where(torch.abs(self.pos) < 1e-8, torch.tensor(0.0, dtype=torch.float, device=self._device), self.pos)
        self.R = torch.matmul(R, self.R_offset.transpose(-1, -2))

    """
        @ 深度相机深度图渲染
        sphere_list:渲染的球体列表张量
        cylinder_list:渲染的球体列表张量
        euler:深度相机未偏移的姿态:torch.tensor([x,y,z], dtype=torch.float, device=device, requires_grad=False):Rad
    """
    @torch.no_grad()
    def render(self, 
        sphere_list: torch.Tensor, 
        cylinder_list: torch.Tensor, 
        euler: torch.Tensor
    ):
        self.depth = torch.zeros_like(self.depth)
        R = util.euler_to_R(euler)
        R = torch.where(torch.abs(R) < 1e-8, torch.tensor(0.0, dtype=torch.float, device=self._device), R)
        self.pixel_dir = util.tensor_norm(torch.matmul(R.unsqueeze(0).unsqueeze(0), self.camera_pixel_dir.unsqueeze(-1)).squeeze(-1))
        self._ground_render()
        if torch.any(sphere_list != 0.0):
            self._sphere_render(sphere_list)
        if torch.any(cylinder_list != 0.0):
            self._cylinder_render(cylinder_list)

        # 深度相机有效检测
        # 求中心区域的平均值
        depth_center = self.depth[round(self.res_H*0.45):round(self.res_H*0.55), round(self.res_W*0.45):round(self.res_W*0.55)]
        depth_center_mean = torch.mean(depth_center, dim=(-2, -1))
        # 掩膜
        mask_vaild = depth_center_mean > self.min_depth    # 中心区域的平均距离大于最小距离阈值
        mask_in_range = (self.depth >= self.min_depth) & (self.depth <= self.max_depth)
        final_mask = mask_vaild.unsqueeze(-1).unsqueeze(-1) & mask_in_range
        self.depth = torch.where(final_mask, self.depth, torch.tensor(0.0))
        mask_inf = (self.depth > self.max_depth)
        self.depth = torch.where(mask_inf, torch.tensor(0.0), self.depth) # 将无穷大的像素变为0


    """
        @ 地面渲染
    """
    @torch.no_grad()
    def _ground_render(self):
        Rs_z = self.pos[2]
        Rt_z = self.pixel_dir[... , 2]
        t = -Rs_z / Rt_z
        mask_valid_t = t > 0  
        mask_update = (self.depth == 0) | (t <= self.depth)    # 之前深度未被更新/当前深度比之前小
        final_mask = mask_valid_t & mask_update
        self.depth = torch.where(final_mask, t, self.depth)
        mask_inf = (t > self.max_depth) & (self.depth == 0)    # 当前深度超过最大深度且原深度未被更新
        self.depth = torch.where(mask_inf, torch.tensor([float('inf')], dtype=torch.float32, device=self._device), self.depth) # 超过最大深度的部分设置为无穷大
        if self.noise_range != 0.0:
            # 选取有效区域加入噪声
            mask_noise = (self.depth >= self.min_depth) & (self.depth <= self.max_depth)
            # 深度图传感器误差
            offset_noise = torch.clamp(torch.randn_like(self.depth)*(self.noise_range / 3), min=-self.noise_range, max=self.noise_range)  # 或其他随机分布
            self.depth = torch.where(mask_noise, self.depth+offset_noise, self.depth)
            # 深度图黑洞
            black_hole_noise = torch.randn_like(self.depth) < stats.norm.ppf(self.black_hole_prob)   # 每个位置有5%概率出现黑洞
            black_hole = torch.zeros_like(self.depth)
            self.depth = torch.where(black_hole_noise, self.depth+black_hole, self.depth)

    """
        @ 球体渲染
    """
    @torch.no_grad()
    def _sphere_render(self, sphere_list):
        Rs_xyz = self.pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)        # 射线起点XYZ
        Rt_xyz = self.pixel_dir.unsqueeze(0)                            # 射线方向向量XYZ
        S_xyz = sphere_list[:, 0:3].unsqueeze(1).unsqueeze(1)           # 球体XYZ
        R = sphere_list[:, 3].unsqueeze(1).unsqueeze(1).unsqueeze(1)    # 球体半径
        # 求根公式
        a = torch.square(torch.norm(Rt_xyz, dim=-1, keepdim=True)).squeeze(-1)
        b = 2*torch.sum(Rt_xyz*(Rs_xyz-S_xyz), dim=-1, keepdim=True).squeeze(-1)
        c = (torch.square(torch.norm(Rs_xyz-S_xyz, dim=-1, keepdim=True))-torch.square(torch.norm(R, dim=-1, keepdim=True))).squeeze(-1)
        delta = torch.square(b)-4*a*c
        t_1 = torch.where(delta > 0.0, (-b+torch.sqrt(delta))/(2*a), torch.tensor([float('inf')], dtype=torch.float32, device=self._device))
        t_2 = torch.where(delta > 0.0, (-b-torch.sqrt(delta))/(2*a), torch.tensor([float('inf')], dtype=torch.float32, device=self._device))
        # 选取大于0且最小的/都小于0时为无穷大
        t_all = torch.where((t_1 > 0) & (t_2 > 0), torch.min(t_1, t_2), 
                torch.where(t_1 > 0, t_1, 
                    torch.where(t_2 > 0, t_2, torch.tensor([float('inf')], dtype=torch.float32, device=self._device))))
        # 在所有渲染的深度图中选择最小的距离
        t = t_all.min(dim=0)[0]
        # 只渲染在地面上的部分
        z = self.pos[2].unsqueeze(-1).unsqueeze(-1)+self.pixel_dir[..., 2]*t
        mask_on_ground = z > 0
        mask_update = (self.depth == 0) | (t < self.depth)    # 之前深度未被更新/当前深度比之前小
        final_mask = mask_on_ground & mask_update
        self.depth = torch.where(final_mask, t, self.depth)
        mask_inf = (t > self.max_depth) & (self.depth == 0)    # 当前深度超过最大深度且原深度未被更新
        self.depth = torch.where(mask_inf, torch.tensor([float('inf')], dtype=torch.float32, device=self._device), self.depth) # 超过最大深度的部分设置为无穷大
        if self.noise_range != 0.0:
            # 选取有效区域加入噪声
            mask_noise = (self.depth >= self.min_depth) & (self.depth <= self.max_depth)
            # 深度图传感器误差
            offset_noise = torch.clamp(torch.randn_like(self.depth)*(self.noise_range / 3), min=-self.noise_range, max=self.noise_range)  # 或其他随机分布
            self.depth = torch.where(mask_noise, self.depth+offset_noise, self.depth)
            # 深度图黑洞
            black_hole_noise = torch.randn_like(self.depth) < stats.norm.ppf(self.black_hole_prob)   # 每个位置有5%概率出现黑洞
            black_hole = torch.zeros_like(self.depth)
            self.depth = torch.where(black_hole_noise, self.depth+black_hole, self.depth)

    """
        @ 圆柱体渲染
    """
    @torch.no_grad()
    def _cylinder_render(self, cylinder_list):
        Rs_xy = self.pos[0:2].unsqueeze(0).unsqueeze(0).unsqueeze(0)            # 射线起点XY
        Rt_xy = self.pixel_dir[..., 0:2].unsqueeze(0)                           # 射线方向向量XY
        C_xy = cylinder_list[:, 0:2].unsqueeze(1).unsqueeze(1)                  # 圆柱XY
        Rs_z = self.pos[2].unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 射线起点Z
        Rt_z = self.pixel_dir[..., 2].unsqueeze(0).unsqueeze(-1)                # 射线方向向量Z
        C_z = cylinder_list[:, 2].unsqueeze(1).unsqueeze(1).unsqueeze(1)        # 圆柱Z
        R = cylinder_list[:, 3].unsqueeze(1).unsqueeze(1).unsqueeze(1)          # 圆柱半径
        H = cylinder_list[:, 4].unsqueeze(1).unsqueeze(1).unsqueeze(1)          # 圆柱高度
        # 圆柱曲面
        # 求根公式
        a = torch.square(torch.norm(Rt_xy, dim=-1, keepdim=True)).squeeze(-1)
        b = 2*torch.sum((Rs_xy-C_xy)*Rt_xy, dim=-1, keepdim=True).squeeze(-1)
        c = (torch.square(torch.norm(Rs_xy-C_xy, dim=-1, keepdim=True))-torch.square(R)).squeeze(-1)
        delta = torch.square(b)-4*a*c
        t_1_1 = torch.where(delta > 0.0, (-b+torch.sqrt(delta))/(2*a), torch.tensor([float('inf')], dtype=torch.float32, device=self._device))
        t_1_2 = torch.where(delta > 0.0, (-b-torch.sqrt(delta))/(2*a), torch.tensor([float('inf')], dtype=torch.float32, device=self._device))
        # 选取大于0且最小的/都小于0时为无穷大
        t_1_all = torch.where((t_1_1 > 0) & (t_1_2 > 0), torch.min(t_1_1, t_1_2), 
                torch.where(t_1_1 > 0, t_1_1, 
                    torch.where(t_1_2 > 0, t_1_2, torch.tensor([float('inf')], dtype=torch.float32, device=self._device))))
        # 在所有渲染的深度图中选择最小的距离
        t_1 = t_1_all.min(dim=0)[0]
        # 只渲染在地面上的部分
        z = self.pos[2].unsqueeze(-1).unsqueeze(-1)+self.pixel_dir[..., 2]*t_1
        mask_on_ground_1 = z > 0
        mask_update_1 = (self.depth == 0) | (t_1 < self.depth)    # 之前深度未被更新/当前深度比之前小
        mask_in_region_1 = ((z > C_z-H/2) & (z < C_z+H/2)).squeeze(1).any(dim=0)  # 有限高圆柱，沿0维按位或
        final_mask_1 = mask_on_ground_1 & mask_update_1 & mask_in_region_1
        self.depth = torch.where(final_mask_1, t_1, self.depth)
        mask_inf = (t_1 > self.max_depth) & (self.depth == 0)    # 当前深度超过最大深度且原深度未被更新
        self.depth = torch.where(mask_inf, torch.tensor([float('inf')], dtype=torch.float32, device=self._device), self.depth) # 超过最大深度的部分设置为无穷大

        # 判断无人机相对于圆柱的位置
        up_cylinder = (Rs_z > C_z+H/2).squeeze(-1)
        down_cylinder = (Rs_z < C_z-H/2).squeeze(-1)
        # 圆柱顶面
        t_2_all = ((C_z+H/2-Rs_z) / Rt_z).squeeze(-1)
        # 在所有渲染的深度图中选择最小的距离
        t_2 = t_2_all.min(dim=0)[0]
        R_t_1 = torch.norm((Rs_xy+Rt_xy*t_2.unsqueeze(-1))-C_xy, dim=-1)    # 计算半径
        mask_valid_t_2 = t_2 > 0  
        # 只渲染在地面上的部分
        z = self.pos[2].unsqueeze(-1).unsqueeze(-1)+self.pixel_dir[..., 2]*t_2
        mask_on_ground_2 = z > 0
        mask_in_region_2 = (R_t_1 <= R.squeeze(-1)).any(dim=0)  
        mask_update_2 = (self.depth == 0) | (t_2 < self.depth)    # 之前深度未被更新/当前深度比之前小
        final_mask_2 = (mask_valid_t_2 & mask_on_ground_2 & mask_in_region_2 & mask_update_2 & up_cylinder).all(dim=0)
        self.depth = torch.where(final_mask_2, t_2, self.depth)
        mask_inf = (t_2 > self.max_depth) & (self.max_depth == 0)    # 当前深度超过最大深度且原深度未被更新
        self.depth = torch.where(mask_inf, torch.tensor([float('inf')], dtype=torch.float32, device=self._device), self.depth) # 超过最大深度的部分设置为无穷大

        # 圆柱底面
        t_3_all = ((C_z-H/2-Rs_z) / Rt_z).squeeze(-1)
        # 在所有渲染的深度图中选择最小的距离
        t_3 = t_3_all.min(dim=0)[0]
        R_t_1 = torch.norm((Rs_xy+Rt_xy*t_3.unsqueeze(-1))-C_xy, dim=-1)    # 计算半径
        mask_valid_t_3 = t_3 > 0  
        # 只渲染在地面上的部分
        z = self.pos[2].unsqueeze(-1).unsqueeze(-1)+self.pixel_dir[..., 2]*t_3
        mask_on_ground_2 = z > 0
        mask_in_region_2 = (R_t_1 <= R.squeeze(-1)).any(dim=0)  
        mask_update_2 = (self.depth == 0) | (t_3 < self.depth)    # 之前深度未被更新/当前深度比之前小
        final_mask_2 = (mask_valid_t_3 & mask_on_ground_2 & mask_in_region_2 & mask_update_2 & down_cylinder).all(dim=0)
        self.depth = torch.where(final_mask_2, t_3, self.depth)
        mask_inf = (t_3 > self.max_depth) & (self.max_depth == 0)    # 当前深度超过最大深度且原深度未被更新
        self.depth = torch.where(mask_inf, torch.tensor([float('inf')], dtype=torch.float32, device=self._device), self.depth) # 超过最大深度的部分设置为无穷大
        
        if self.noise_range != 0.0:
            # 选取有效区域加入噪声
            mask_noise = (self.depth >= self.min_depth) & (self.depth <= self.max_depth)
            # 深度图传感器误差
            offset_noise = torch.clamp(torch.randn_like(self.depth)*(self.noise_range / 3), min=-self.noise_range, max=self.noise_range)  # 或其他随机分布
            self.depth = torch.where(mask_noise, self.depth+offset_noise, self.depth)
            # 深度图黑洞
            black_hole_noise = torch.randn_like(self.depth) < stats.norm.ppf(self.black_hole_prob)   # 每个位置有5%概率出现黑洞
            black_hole = torch.zeros_like(self.depth)
            self.depth = torch.where(black_hole_noise, self.depth+black_hole, self.depth)

