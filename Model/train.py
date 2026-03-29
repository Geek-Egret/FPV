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

episodes = 10000
steps = 250
batch_size = 50
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
# 域随机化
spheres_num = 5
spheres_xyzR_range = {
    "x_min": 0.5, "x_max": 10,
    "y_min": -3, "y_max": 3,
    "z_min": 0.2, "z_max": 2,
    "R_min": 0.05, "R_max": 0.3
}
cylinders_num = 10
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
visual.build()

model = model.Model()
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
checkpoint_num = 0
last_checkpoint_episode = 0
best_mean_loss = 1e10
for episode in range(episodes):
    start = time.perf_counter()
    geom.reset(domain_randomization=True)
    # 随机添加障碍
    for i in range(spheres_num):
        x = random.uniform(spheres_xyzR_range["x_min"], spheres_xyzR_range["x_max"])
        y = random.uniform(spheres_xyzR_range["y_min"], spheres_xyzR_range["y_max"])
        z = random.uniform(spheres_xyzR_range["z_min"], spheres_xyzR_range["z_max"])
        R = random.uniform(spheres_xyzR_range["R_min"], spheres_xyzR_range["R_max"])
        geom.add_sphere(x, y, z, R)
    for i in range(cylinders_num):
        x = random.uniform(cylinders_xyzRH_range["x_min"], cylinders_xyzRH_range["x_max"])
        y = random.uniform(cylinders_xyzRH_range["y_min"], cylinders_xyzRH_range["y_max"])
        z = random.uniform(cylinders_xyzRH_range["z_min"], cylinders_xyzRH_range["z_max"])
        R = random.uniform(cylinders_xyzRH_range["R_min"], cylinders_xyzRH_range["R_max"])
        H = random.uniform(cylinders_xyzRH_range["H_min"], cylinders_xyzRH_range["H_max"])
        geom.add_cylinder(x, y, z, R, H)
    geom.build(
        show_depth=True, 
        show_idx=0, 
        noise=True, 
        noise_range=random.uniform(noise_range["noise_min"], noise_range["noise_max"]), 
        black_hole_prob=random.uniform(black_hole_prob_range["prob_min"], black_hole_prob_range["prob_max"])
    )
    obs = []
    for i in range(steps):
        reward = torch.zeros(batch_size, device=device)
        # 模型前向传播
        # print(geom.depth)
        # print(geom.drone_acc)
        # print(geom.drone_euler)
        # print(geom.drone_ang_vel)
        mean, std = model.forward(geom.depth, geom.drone_acc, geom.drone_euler, geom.drone_ang_vel)
        # print(act)
        # 重参数
        eps = torch.randn_like(mean)
        act_raw = mean+eps*std
        act = torch.zeros_like(act_raw)
        act[:, 0:3] = torch.sigmoid(act_raw[:, 0:3]) * 360 - 180
        act[:, 3] = torch.sigmoid(act_raw[:, 3])
        # 环境计算
        geom.step(
            act=act, 
            T_att=random.uniform(T_att_range["T_att_min"], T_att_range["T_att_max"]), 
            show_depth=True, 
            show_idx=0, 
            noise=True, 
            noise_range=random.uniform(noise_range["noise_min"], noise_range["noise_max"]), 
            black_hole_prob=random.uniform(black_hole_prob_range["prob_min"], black_hole_prob_range["prob_max"])
        )
        visual.step(
            geom.drone_pos[0, ...].detach(), 
            geom.drone_euler[0, ...].detach()
        )

        obs.append([geom.drone_pos, geom.drone_acc, geom.drone_ang_vel, geom.drone_euler, geom.collision_state])
        if torch.all(geom.collision_state == True):
            break

    # 计算损失
    loss_list = []
    for step_obs in obs:
        loss_list.append(
            2.0*torch.norm(step_obs[0]-init_pos, dim=-1) + \
            1.5*step_obs[4].int() + \
            (-1.0)*(1-step_obs[4].int()) + \
            (-0.08)*i*(1-step_obs[4].int())
        )
    batch_loss = torch.stack(loss_list).mean(dim=-1)
    loss = torch.mean(batch_loss)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # 当平均奖励大于之前的最佳平均奖励 && 全部没有碰撞
    if torch.mean(loss) < best_mean_loss and torch.all(~geom.collision_state):
        best_mean_loss = torch.mean(loss)
        checkpoint_num += 1
        last_checkpoint_episode = episode
        checkpoint = {
            'epoch': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss if 'loss' in locals() else None,
            'best_score': best_mean_loss,
            'model_mode': 'train' if model.training else 'eval',
        }
        torch.save(checkpoint, f'outputs/checkpoint_{checkpoint_num}.pth')
        print("Save Checkpoint")
    end = time.perf_counter()
    elapsed = end - start
    sep = "=" * 50
    print(f"""
    {sep}
    @ Episode: {episode:3d}/{episodes}
    @ Non Collision: {torch.count_nonzero(~geom.collision_state).item()}/{batch_size}
    @ Mean Loss: {torch.mean(batch_loss)}
    @ Min Loss: {torch.min(batch_loss)}
    @ Max Loss: {torch.max(batch_loss)}
    @ Best Mean Loss: {best_mean_loss}
    @ Last Checkpoint Episode: {last_checkpoint_episode}
    @ Duration Time: {elapsed}s
    {sep}
    """)

torch.save(model.state_dict(), "final.pth")
print("Save Final")