import torch
import math
import os

import solver
import util
import genesis_bridge as bg

device = 'cpu'
dt = 0.01
init_pos = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.double, device=device)
init_forward_vec = torch.tensor([1.0, 1.0, 0.0], dtype=torch.double, device=device)
init_up_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
wind_dir = torch.tensor([[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.double, device=device)
wind_speed = torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.double, device=device)
num = 4

# device = 'cpu'
# dt = 0.01
# init_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
# init_forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.double, device=device)
# init_up_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
# wind_dir = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.double, device=device)
# wind_speed = torch.tensor([0.0], dtype=torch.double, device=device)
# num = 1

init_pos_clone = init_pos.clone()

scene = bg.genesis_bridge(camera_pos=(3.5, 3.5, 3.5), camera_lookat=(0.0, 0.0, 0.0), camera_fov=35, max_FPS=60, dt=dt, device=device)
ge_fpv = solver.solver(dt=0.01, mass=0.33, collider_radius=0.1, init_pos=init_pos, init_forward_vec=init_forward_vec, init_up_vec=init_up_vec, alpha_1=0.2, alpha_2=0.2, wind_dir=wind_dir, wind_speed=wind_speed, drag_1=0.000, drag_2=0.000, z_drag=0.000, num=num, device=device)
scene.add_drone(urdf_path="ge_fpv.urdf", init_pos=init_pos, init_R=ge_fpv.next_R, num=num)
print(f"EULER: {util.rad_to_angle(util.R_to_euler(ge_fpv.next_R))}")
scene.build()

flag = False

while True:
    # if input("") == "":
    # pred_acc_FLU = torch.tensor([[0.3, 0.2, 0.0], [0.0, 0.1, 0.0], [0.0, 0.3, 0.0], [0.2, 0.0, 0.0]], dtype=torch.double, device=device)
    # pred_yaw_vel_FLU = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1], [0.0, 0.0, 0.0], [0.1, 0.1, 0.0]], dtype=torch.double, device=device)
    if ge_fpv.next_vel[0, 0] < 2 and flag == False:
        pred_acc_FLU = torch.tensor([0.5, 0.0, 0.0], dtype=torch.double, device=device)
    if ge_fpv.next_vel[0, 0] >= 2:
        flag = True
        pred_acc_FLU = torch.tensor([-0.5, 0.0, 0.0], dtype=torch.double, device=device)
    if ge_fpv.next_vel[0, 0] < -2 and flag == True:
        flag = False
        pred_acc_FLU = torch.tensor([0.5, 0.0, 0.0], dtype=torch.double, device=device)
    pred_yaw_vel_FLU = torch.tensor([0.0, 0.0, 0.0*math.pi/180.0], dtype=torch.double, device=device)

    ge_fpv.step(pred_acc_FLU, pred_yaw_vel_FLU)
    scene.set_pos_R(ge_fpv.next_pos, ge_fpv.next_R)
    # scene.set_pos_R(init_pos_clone, ge_fpv.next_R)
    # os.system('cls' if os.name == 'nt' else 'clear')
    # print(f"POS:{ge_fpv.next_pos}\nEULER:{util.rad_to_angle(util.R_to_euler(ge_fpv.next_R))}\nACC:{ge_fpv.next_acc}")
    scene.step()

    # print(ge_fpv.next_pos)


# print(f"pos:{ge_fpv.pos}")
# print(f"forward_pos:{ge_fpv.forward_vec}")
# print(f"up_vec:{ge_fpv.up_vec}")
# print(f"forward_R:{ge_fpv.forward_R}")
# print(f"forward_euler:{util.R_to_euler(ge_fpv.forward_R)}")
# print(f"wind_dir:{ge_fpv.wind_dir}")
# print(f"wind_speed:{ge_fpv.wind_speed}")