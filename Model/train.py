import torch

import kernel.geom as geom
import kernel.util as util
import kernel.visual as visual

batch_size = 5
device = 'cpu'
dt = 0.01
safty_radius = 0.25
init_pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float, device=device, requires_grad=True)
init_euler = torch.tensor([[0.0, 0.0, 30.0]], dtype=torch.float, device=device, requires_grad=True)
pos_offset = torch.tensor([[0.0425, 0.0, 0.0345]], dtype=torch.float, device=device, requires_grad=True)
euler_offset = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float, device=device, requires_grad=True)
mass = 0.33
T_max = 0.33*9.81# 4*0.40*9.81
ang_vel_max = [90, 90, 90]
res_W = 640
res_H = 400
fov_H = 67.9
fov_V = 45.3
min_depth = 0.25
max_depth = 4.0

geom = geom.geom(batch_size=batch_size, device=device, dt=dt, safty_radius=safty_radius, init_pos=init_pos,
                 init_euler=init_euler, pos_offset=pos_offset, euler_offset=euler_offset, mass=mass, T_max=T_max,
                 ang_vel_max=ang_vel_max, res_W=res_W, res_H=res_H, fov_H=fov_H, fov_V=fov_V, min_depth=min_depth,
                 max_depth=max_depth)
geom.add_sphere(2.0, 1.0 ,1.0, 0.1)

visual = visual.visual(urdf="urdf/ge_fpv.urdf", init_pos=init_pos[0, :], init_euler=init_euler[0, :], device=device, batch_size=0)

for i in range(100000):
    geom.step(torch.tensor([[0.0, 0.0, 30.0, 1.0]], dtype=torch.float, device=device), 0.0, True)
    print(geom.drone_pos.grad_fn)
    print(geom.drone_euler.grad_fn)
    print(geom.drone_acc.grad_fn)
    print(geom.drone_vel.grad_fn)
    print(geom.depth.grad_fn)
    print("")
    
    visual.step(geom.drone_pos[0, ...], geom.drone_euler[0, ...])
    # visual.step(init_pos, geom.drone_euler)
    
