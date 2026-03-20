import torch
import math
import os

import env.util as util
import env.geom as geom

device = 'cpu'
dt = 0.002
drone_init_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
drone_init_euler = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double, device=device)
T_max = 4*0.4315*9.81
T_max_att = 0.0
ang_vel_max = torch.tensor([0.4, 0.4, 0.4], dtype=torch.double, device=device)
depth_pos_offset = torch.tensor([0.0425, 0.0, 0.0345], dtype=torch.double, device=device)
init_forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.double, device=device)
init_up_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
num = 1

wind_dir = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.double, device=device)
wind_speed = torch.tensor([0.0], dtype=torch.double, device=device)
depth_init_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double, device=device)



scene = geom.geom(camera_pos=(3.5, 3.5, 3.5), camera_lookat=(0.0, 0.0, 0.0), camera_fov=35, 
                    max_FPS=60, show_viewer=True, dt=dt, device=device)
scene.add_drone(urdf_path="urdf/ge_fpv.urdf", drone_init_pos=drone_init_pos, drone_init_euler=drone_init_euler, 
                T_max=T_max, T_max_att=T_max_att, ang_vel_max=ang_vel_max, res_W=640, res_H=480, 
                depth_pos_offset=depth_pos_offset, depth_euler_offset=drone_init_euler, 
                depth_fov_H=67.9, depth_fov_V=45.3, num=num) 
scene.build()

while True:
    scene.step(torch.tensor([0.2, 0.4, 0.3], dtype=torch.double, device=device), 0.19119351)    # 0.33*9.81
