import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import env.geom as geom
import env.util as util
import env.visual as visual
import model

steps = 2000
batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
dt = 0.01
init_pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float, device=device, requires_grad=True)
init_euler = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float, device=device, requires_grad=True)
pos_offset = torch.tensor([[0.0425, 0.0, 0.0345]], dtype=torch.float, device=device, requires_grad=True)
euler_offset = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float, device=device, requires_grad=True)
mass = 0.33
T_max = 4*0.40*9.81
ang_vel_max = [90, 90, 90]
res_W = 80
res_H = 50
fov_H = 67.9
fov_V = 45.3
min_depth = 0.25
max_depth = 2.5
spheres_num = 5
spheres_xyzR_range = {
    "x_min": 0.5, "x_max": 10,
    "y_min": -3, "y_max": 3,
    "z_min": 0.2, "z_max": 2,
    "R_min": 0.05, "R_max": 0.3
}
cylinders_num = 15
cylinders_xyzRH_range = {
    "x_min": 0.5, "x_max": 10,
    "y_min": -3, "y_max": 3,
    "z_min": 0.2, "z_max": 2,
    "R_min": 0.05, "R_max": 0.3,
    "H_min": 1.0, "H_max": 5,
}     
T_att_range = {"T_att_min": 0.0, "T_att_max": 0.3}  
noise_range = {"noise_min": 0.0, "noise_max": 0.005}
black_hole_prob_range = {"prob_min": 0.0, "prob_max": 0.01}                                       

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
for i in range(spheres_num):
    x = random.uniform(spheres_xyzR_range["x_min"], spheres_xyzR_range["x_max"])
    y = random.uniform(spheres_xyzR_range["y_min"], spheres_xyzR_range["y_max"])
    z = random.uniform(spheres_xyzR_range["z_min"], spheres_xyzR_range["z_max"])
    R = random.uniform(spheres_xyzR_range["R_min"], spheres_xyzR_range["R_max"])
    geom.add_sphere(x, y, z, R)
    visual.add_sphere(x, y, z, R)
for i in range(cylinders_num):
    x = random.uniform(cylinders_xyzRH_range["x_min"], cylinders_xyzRH_range["x_max"])
    y = random.uniform(cylinders_xyzRH_range["y_min"], cylinders_xyzRH_range["y_max"])
    z = random.uniform(cylinders_xyzRH_range["z_min"], cylinders_xyzRH_range["z_max"])
    R = random.uniform(cylinders_xyzRH_range["R_min"], cylinders_xyzRH_range["R_max"])
    H = random.uniform(cylinders_xyzRH_range["H_min"], cylinders_xyzRH_range["H_max"])
    geom.add_cylinder(x, y, z, R, H)
    visual.add_cylinder(x, y, z, R, H)
geom.build(
    show_depth=True, 
    show_idx=0, 
    noise=True, 
    noise_range=random.uniform(noise_range["noise_min"], noise_range["noise_max"]), 
    black_hole_prob=random.uniform(black_hole_prob_range["prob_min"], black_hole_prob_range["prob_max"])
)
visual.build()

model = model.Old_Model()  # 先创建模型实例
# model.load_state_dict(torch.load('best/final.pth'))  # 再加载参数
checkpoint = torch.load('outputs/checkpoint_22.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])  # 从字典中提取模型参数
model.eval()

for step in range(steps):
    # 模型前向传播
    mean, _ = model(geom.depth, geom.drone_acc, geom.drone_euler, geom.drone_ang_vel)
    # 将采样结果映射到 角度：0-360 推力:0-1
    action = mean.clone()
    action[:, 0:3] = torch.sigmoid(mean[:, 0:3]) * 360 - 180
    action[:, 3] = torch.sigmoid(mean[:, 3])
    geom.step(
        act=action, 
        T_att=random.uniform(T_att_range["T_att_min"], T_att_range["T_att_max"]), 
        show_depth=True, 
        show_idx=0, 
        noise=True, 
        noise_range=random.uniform(noise_range["noise_min"], noise_range["noise_max"]), 
        black_hole_prob=random.uniform(black_hole_prob_range["prob_min"], black_hole_prob_range["prob_max"])
    )
    # geom.step(act=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float, device=device), 
    #           T_att=0.0, 
    #           show_depth=True, 
    #           show_idx=0, 
    #           noise=True, 
    #           noise_range=0.005, 
    #           black_hole_prob=0.01)
    visual.step(geom.drone_pos[0, ...].detach(), geom.drone_euler[0, ...].detach())
    print(action)
    print(geom.drone_pos)
    print(f"Running Steps: {step}")

    if torch.all(geom.collision_state == True):
        print("@ End\n")
        break