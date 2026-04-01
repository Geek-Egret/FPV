import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import time

import env.geom as geom
import env.util as util
import env.visual as visual
import model

"""
    @ 适配张量维度到 batch_size
"""
def adapt(tensor, batch_size):
    if tensor.size(0) == 1 and tensor.size(0) != batch_size:
        repeat_times = [batch_size] + [1] * (tensor.dim() - 1)
        return tensor.repeat(repeat_times)
    elif tensor.size(0) == batch_size:
        return tensor
    else:
        raise Exception("[ERROR] tensor can't adapt to batchsize")
        
# 训练参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
episodes = 10000
steps = 250
batch_size = 3
# GEOM参数
dt = 0.01
init_pos = torch.tensor([[0.0, 0.0, 4.50],[0.0, 0.0, 6.50],[3.0, 0.0, 4.50]], dtype=torch.float, device=device, requires_grad=True)
init_euler = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float, device=device, requires_grad=True)
pos_offset = torch.tensor([[0.0425, 0.0, 0.0345]], dtype=torch.float, device=device, requires_grad=True)
euler_offset = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float, device=device, requires_grad=True)
mass = 0.33
T_max = 0.33*9.81
ang_vel_max = [90, 90, 90]
res_W = 80
res_H = 50
fov_H = 67.9
fov_V = 45.3
min_depth = 0.25
max_depth = 2.5
max_acc = 16.0
max_vel = 20.0
max_roll_pitch = 30.0
max_yaw = 180.0
safty_distance = 0.25

geom = geom.geom(
    batch_size=batch_size, device=device, dt=dt, init_pos=init_pos,
    init_euler=init_euler, pos_offset=pos_offset, euler_offset=euler_offset, 
    mass=mass, T_max=T_max, ang_vel_max=ang_vel_max, res_W=res_W, res_H=res_H, 
    fov_H=fov_H, fov_V=fov_V, min_depth=min_depth, max_depth=max_depth
)
visual = visual.visual(
    urdf="urdf/ge_fpv.urdf", device=device, init_pos=init_pos[0, :], 
    init_euler=init_euler[0, :], batch_size=0
)
# geom.add_sphere(0.0, 0.0, 0.0, 1.2)
# geom.add_sphere(0.0, 1.0, 0.0, 2.0)
# visual.add_sphere(0.0, 0.0, 0.0, 1.2)
# visual.add_sphere(0.0, 1.0, 0.0, 2.0)
geom.add_cylinder(0.1, 0.0, 7.6, 0.5, 4.0)
visual.add_cylinder(0.1, 0.0, 7.6, 0.5, 4.0)
visual.build()

for episode in range(episodes):
    for i in range(steps):
        # 环境计算
        geom.step(
            act=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float, device=device), 
            T_att=0.0, 
            show_depth=True, 
            show_idx=0, 
            noise=True, 
            noise_range=0.0, 
            black_hole_prob=0.0
        )
        visual.step(
            geom.drone_pos[2, ...].detach(), 
            geom.drone_euler[2, ...].detach()
        )
        print(geom.closest_distance)
        print(geom.collision_state)
        print("")