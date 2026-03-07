import torch

import solver
import util

init_pos = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 0.0]], dtype=torch.float32)
init_forward_vec = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)
init_up_vec = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float32)
com_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
wind_dir = torch.tensor([[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float32)
wind_speed = torch.tensor([[3.0], [2.0], [1.0], [0.0]], dtype=torch.float32)

ge_fpv = solver.solver(dt=0.01, mass=0.32, collider_radius=0.1, init_pos=init_pos, init_forward_vec=init_forward_vec, init_up_vec=init_up_vec, com_offset=com_offset, alpha=0.2, wind_dir=wind_dir, wind_speed=wind_speed, num=4)

for i in range(100):
    acc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 1.0, 0.0]], dtype=torch.float32)
    ge_fpv.set_pred_acc(acc)
    print(ge_fpv.next_acc)
    print(ge_fpv.next_vel)

# print(f"pos:{ge_fpv.pos}")
# print(f"forward_pos:{ge_fpv.forward_vec}")
# print(f"up_vec:{ge_fpv.up_vec}")
# print(f"forward_R:{ge_fpv.forward_R}")
# print(f"forward_euler:{util.R_to_euler(ge_fpv.forward_R)}")
# print(f"wind_dir:{ge_fpv.wind_dir}")
# print(f"wind_speed:{ge_fpv.wind_speed}")