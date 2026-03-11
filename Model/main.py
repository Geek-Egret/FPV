import torch
import math
import os

import kernel.solver as solver
import kernel.util as util
import kernel.geometry as geometry
import kernel.env as env

# device = 'cpu'
# dt = 0.01
# init_pos = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.double, device=device)
# init_forward_vec = torch.tensor([4.0, 1.0, 0.0], dtype=torch.double, device=device)
# init_up_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
# wind_dir = torch.tensor([[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.double, device=device)
# wind_speed = torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.double, device=device)
# num = 4

device = 'cpu'
dt = 0.01
init_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
init_forward_vec = torch.tensor([1.0, 1.0, 0.0], dtype=torch.double, device=device)
init_up_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
wind_dir = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.double, device=device)
wind_speed = torch.tensor([5.0], dtype=torch.double, device=device)
depth_init_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double, device=device)
depth_pos_offset = torch.tensor([0.0425, 0.0, 0.0345], dtype=torch.double, device=device)
num = 1

pos_bind = init_pos.clone()

scene = geometry.geometry(camera_pos=(3.5, 3.5, 3.5), camera_lookat=(0.0, 0.0, 0.0), camera_fov=35, max_FPS=60, show_viewer=True, dt=dt, device=device)
fpv_solver = solver.solver(dt=0.01, mass=0.33, T_max=4*0.21*9.8, collider_radius=0.1, init_pos=init_pos, 
                        init_forward_vec=init_forward_vec, init_up_vec=init_up_vec, alpha_1=0.4, alpha_2=0.4, 
                        wind_dir=wind_dir, wind_speed=wind_speed, drag_1=0.005, drag_2=0.005, z_drag=0.005, 
                        num=num, device=device)
print(f"EULER: {util.rad_to_angle(util.R_to_euler(fpv_solver.next_R))}")
scene.add_drone(urdf_path="urdf/ge_fpv.urdf", drone_init_pos=init_pos, drone_init_R=fpv_solver.next_R, res_W=640, res_H=480, 
                init_pos=depth_init_pos, pos_offset=depth_pos_offset, lookat=init_forward_vec, up=init_up_vec, fov_V=45.3, 
                GUI=False, num=num)    

scene.build()

flag = False

while True:
    # if input("") == "":
    if fpv_solver.next_vel[0] < 1 and flag == False:
        pred_acc_model = torch.tensor([0.5, 0.0, 0.0], dtype=torch.double, device=device)
    if fpv_solver.next_vel[0] >= 1:
        flag = True
        pred_acc_model = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.double, device=device)
    if fpv_solver.next_vel[0] < -1 and flag == True:
        flag = False
        pred_acc_model = torch.tensor([0.5, 0.0, 0.0], dtype=torch.double, device=device)
    # pred_acc_model = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double, device=device)
    pred_yaw_vel = torch.tensor([0.0, 0.0, 0.0*math.pi/180.0], dtype=torch.double, device=device)

    pred_acc = util.MODEL_to_ENU(pred_acc_model, fpv_solver.next_R)

    fpv_solver.step(pred_acc, pred_yaw_vel)
    scene.set_pos_R(fpv_solver.next_pos, fpv_solver.next_R)
    # scene.set_pos_R(pos_bind, fpv_solver.next_R)
    # os.system('cls' if os.name == 'nt' else 'clear')
    # print(f"POS:{fpv_solver.next_pos}\nEULER:{util.rad_to_angle(util.R_to_euler(fpv_solver.next_R))}\nACC:{fpv_solver.next_acc}")
    scene.step()

    # print(f"pos:{fpv_solver.next_pos}")
    # print(f"euler:{util.rad_to_angle(util.R_to_euler(fpv_solver.next_R))}")
    # print(f"acc:{pred_acc}")
    # print(f"vel:{fpv_solver.next_vel}\n")
