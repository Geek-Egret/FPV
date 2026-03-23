import torch
import genesis

import kernel.geom as geom
import kernel.util as util
import kernel.visual as visual

batch_size = 1
device = 'cpu'
dt = 0.01
safty_radius = 0.25
init_pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float, device=device)
init_euler = torch.tensor([[0.0, 0.0, 90.0]], dtype=torch.float, device=device)
pos_offset = torch.tensor([[0.0425, 0.0, 0.0345]], dtype=torch.float, device=device)
euler_offset = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float, device=device)
mass = 0.33
T_max = 0.33*9.81# 4*0.40*9.81
ang_vel_max = [90, 90, 90]
res_W = 640
res_H = 400
fov_H = 67.9
fov_V = 45.3

geom = geom.geom(batch_size=batch_size, device=device, dt=dt, safty_radius=safty_radius, init_pos=init_pos,
                 init_euler=init_euler, pos_offset=pos_offset, euler_offset=euler_offset, mass=mass, T_max=T_max,
                 ang_vel_max=ang_vel_max, res_W=res_W, res_H=res_H, fov_H=fov_H, fov_V=fov_V)

visual = visual.visual(urdf="urdf/ge_fpv.urdf", init_pos=init_pos[0, :], init_euler=init_euler[0, :], device=device, batch_size=0)

for i in range(100000):
    geom.step(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float, device=device), 0.0)
    # print(geom.drone_pos)
    # print(geom.drone_euler)
    # print(geom.drone_acc)
    # print(geom.drone_vel)
    # print("")
    visual.step(geom.drone_pos, geom.drone_euler)
    # visual.step(init_pos, geom.drone_euler)
    
