import torch

import solver
import util
import genesis_bridge as bg

device = 'cuda'
dt = 0.01
init_pos = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
init_forward_vec = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32, device=device)
init_up_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
wind_dir = torch.tensor([[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float32, device=device)
wind_speed = torch.tensor([[3.0], [2.0], [1.0], [0.0]], dtype=torch.float32, device=device)
num = 4

# scene = bg.genesis_bridge(camera_pos=(3.5, 3.5, 3.5), camera_lookat=(0.0, 0.0, 0.0), camera_fov=35, max_FPS=240, dt=dt, device=device)
ge_fpv = solver.solver(dt=0.01, mass=0.33, collider_radius=0.1, init_pos=init_pos, init_forward_vec=init_forward_vec, init_up_vec=init_up_vec, alpha=0.2, wind_dir=wind_dir, wind_speed=wind_speed, drag_1=0.1, drag_2=0.1, z_drag=0.1, num=num, device=device)
# scene.add_drone(urdf_path="ge_fpv.urdf", init_pos=init_pos, init_R=ge_fpv.next_forward_R, num=num)
# scene.build()

for i in range(10):
    acc = torch.tensor([[5.0, 4.0, 3.0], [0.00, 0.0, 0.0], [0.0, 0.00, 0.0], [0.0, 0.00, 0.00]], dtype=torch.float32, device=device)
    ge_fpv.set_pred_acc(acc)
    # scene.set_pos_R(ge_fpv.next_pos, None)
    # scene.step()

    # print(ge_fpv.next_pos)


# print(f"pos:{ge_fpv.pos}")
# print(f"forward_pos:{ge_fpv.forward_vec}")
# print(f"up_vec:{ge_fpv.up_vec}")
# print(f"forward_R:{ge_fpv.forward_R}")
# print(f"forward_euler:{util.R_to_euler(ge_fpv.forward_R)}")
# print(f"wind_dir:{ge_fpv.wind_dir}")
# print(f"wind_speed:{ge_fpv.wind_speed}")