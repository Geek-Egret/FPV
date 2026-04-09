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
episodes = 50000
steps = 150
batch_size = 5
target_pos = adapt(torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float, device=device), batch_size=batch_size)
target_vel = adapt(torch.tensor([[1.5]], dtype=torch.float, device=device), batch_size=batch_size)
safty_distance = 0.3
gru_seq_len = 50    # GRU时序长度
vel_queue_len = 30   # 速度队列长度
H_dir_queue_len = 30    # 水平方向队列长度
pos_z_queue_len = 10    # 高度队列长度
# coef = {
#     "coef_vel": -0.1,    # 惩罚速度误差
#     "coef_move": 0.0,   # 奖励移动
#     "coef_H_dir": -0.01,    # 惩罚水平方向误差
#     "coef_pos_z": 0.0,    # 惩罚高度误差
#     "coef_distance_target": -1.0,   # 惩罚到目标点的距离    
#     "coef_distance_no_safty": -5.0,  # 惩罚不安全距离
#     "coef_alive": 1.0,  # 奖励存活
# }
coef = {
    "coef_vel": 0.0,    # 惩罚速度误差
    "coef_move": 0.0,   # 奖励移动
    "coef_H_dir": 0.0,    # 惩罚水平方向误差
    "coef_pos_z": -2.0,    # 惩罚高度误差
    "coef_distance_target": 0.0,   # 惩罚到目标点的距离    
    "coef_distance_no_safty": 0.0,  # 惩罚不安全距离
    "coef_alive": 1.0,  # 奖励存活
}
# GEOM参数
dt = 0.03
init_pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float, device=device, requires_grad=True)
prev_pos = init_pos.clone()
init_euler = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float, device=device, requires_grad=True)
pos_offset = torch.tensor([[0.0425, 0.0, 0.0345]], dtype=torch.float, device=device, requires_grad=True)
euler_offset = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float, device=device, requires_grad=True)
mass = 0.33
T_max = 4*0.40*9.81
ang_vel_max = [90, 90, 20]
collision_radius = 0.072
res_W = 40
res_H = 25
fov_H = 67.9
fov_V = 45.3
min_depth = 0.25
max_depth = 2.5
# 模型归一化参数
max_acc = 16.0
max_vel = 20.0
max_roll_pitch = 40.0
max_yaw = 30.0
# 域随机化
domain_randomization_enable = False
spheres_num = 5
spheres_xyzR_range = {
    "x_min": 0.5, "x_max": 4,
    "y_min": -2, "y_max": 2,
    "z_min": 1, "z_max":5,
    "R_min": 0.05, "R_max": 0.3
}
cylinders_num = 10
cylinders_xyzRH_range = {
    "x_min": 0.5, "x_max": 4,
    "y_min": -2, "y_max": 2,
    "z_min": 3, "z_max": 5,
    "R_min": 0.05, "R_max": 0.3,
    "H_min": 1.0, "H_max": 5,   # 不使用，由Z*2得
}     
T_att_range = {"T_att_min": 0.0, "T_att_max": 0.5}  
# noise_range = {"noise_min": 0.0, "noise_max": 0.005}
# black_hole_prob_range = {"prob_min": 0.0, "prob_max": 0.01}
noise_range = {"noise_min": 0.0, "noise_max": 0.0}
black_hole_prob_range = {"prob_min": 0.0, "prob_max": 0.0}

geom = geom.geom(
    batch_size=batch_size, device=device, dt=dt, init_pos=init_pos,
    init_euler=init_euler, pos_offset=pos_offset, euler_offset=euler_offset, 
    mass=mass, T_max=T_max, ang_vel_max=ang_vel_max, res_W=res_W, res_H=res_H, 
    fov_H=fov_H, fov_V=fov_V, min_depth=min_depth, max_depth=max_depth, collision_radius=collision_radius
)
visual = visual.visual(
    urdf="urdf/ge_fpv.urdf", device=device, init_pos=init_pos[0, :], 
    init_euler=init_euler[0, :], batch_size=0
)
geom.add_cylinder(1.2, 0.0, 2.0, 0.2, 2*2.0)
visual.add_cylinder(1.2, 0.0, 2.0, 0.2, 2*2.0)
geom.add_cylinder(1.1, 1.5, 2.0, 0.2, 2*2.0)
visual.add_cylinder(1.1, 1.5, 2.0, 0.2, 2*2.0)
geom.add_cylinder(1.1, -1.4, 2.0, 0.2, 2*2.0)
visual.add_cylinder(1.1, -1.4, 2.0, 0.2, 2*2.0)

# geom.add_cylinder(2.7, 0.7, 2.0, 0.2, 2*2.0)
# visual.add_cylinder(2.7, 0.7, 2.0, 0.2, 2*2.0)
# geom.add_cylinder(2.8, -0.8, 2.0, 0.2, 2*2.0)
# visual.add_cylinder(2.8, -0.8, 2.0, 0.2, 2*2.0)

# geom.add_cylinder(4.2, 0.0, 2.0, 0.2, 2*2.0)
# visual.add_cylinder(4.2, 0.0, 2.0, 0.2, 2*2.0)
# geom.add_cylinder(4.1, 1.6, 2.0, 0.2, 2*2.0)
# visual.add_cylinder(4.1, 1.6, 2.0, 0.2, 2*2.0)
# geom.add_cylinder(4.1, -1.5, 2.0, 0.2, 2*2.0)
# visual.add_cylinder(4.1, -1.5, 2.0, 0.2, 2*2.0)

geom.build(
    show_depth=True, 
    show_idx=0, 
    noise=True, 
    noise_range=random.uniform(noise_range["noise_min"], noise_range["noise_max"]), 
    black_hole_prob=random.uniform(black_hole_prob_range["prob_min"], black_hole_prob_range["prob_max"])
)
visual.build()

model = model.Model_GRU()
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
checkpoint_num = 0
last_checkpoint_episode = 0
best_mean_reward = -1e10
for episode in range(episodes):
    start = time.perf_counter()
    geom.reset(
        init_pos=init_pos,
        init_euler=init_euler,
        domain_randomization=domain_randomization_enable
    )
    # 随机添加障碍
    if domain_randomization_enable:
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
            H = random.uniform(cylinders_xyzRH_range["H_min"], cylinders_xyzRH_range["H_max"])  # 不使用，由Z*2得
            geom.add_cylinder(x, y, z, R, 2*z)
        geom.build(
            show_depth=True, 
            show_idx=0, 
            noise=True, 
            noise_range=random.uniform(noise_range["noise_min"], noise_range["noise_max"]), 
            black_hole_prob=random.uniform(black_hole_prob_range["prob_min"], black_hole_prob_range["prob_max"])
        )
    # 模型输入
    depth_norm_queue = [torch.zeros_like(geom.depth)]*gru_seq_len
    acc_norm_queue = [torch.zeros_like(geom.drone_acc)]*gru_seq_len
    euler_norm_queue = [torch.zeros_like(geom.drone_euler)]*gru_seq_len
    angle_vel_norm_queue = [torch.zeros_like(geom.drone_ang_vel)]*gru_seq_len
    target_vel_norm_queue = [torch.zeros_like(target_vel)]*gru_seq_len
    # 奖励计算
    vel_queue = []
    H_dir_queue = []
    pos_z_queue = []
    reward_list = []
    for i in range(steps):
        reward = torch.zeros(batch_size, device=device)
        # 归一化
        depth_norm = geom.depth / max_depth
        acc_norm = geom.drone_acc / max_acc
        euler_norm = geom.drone_euler / 180.0
        ang_vel_norm = geom.drone_ang_vel / torch.tensor(ang_vel_max, device=device, dtype=torch.float).unsqueeze(0)
        target_vel_norm = target_vel/max_vel
        # 时序化
        depth_norm_queue.append(depth_norm)
        if len(depth_norm_queue) > gru_seq_len:
            depth_norm_queue.pop(0)
        acc_norm_queue.append(acc_norm)
        if len(acc_norm_queue) > gru_seq_len:
            acc_norm_queue.pop(0)
        euler_norm_queue.append(euler_norm)
        if len(euler_norm_queue) > gru_seq_len:
            euler_norm_queue.pop(0)
        angle_vel_norm_queue.append(ang_vel_norm)
        if len(angle_vel_norm_queue) > gru_seq_len:
            angle_vel_norm_queue.pop(0)
        target_vel_norm_queue.append(target_vel_norm)
        if len(target_vel_norm_queue) > gru_seq_len:
            target_vel_norm_queue.pop(0)
        # 前向传播
        # mean, std = model.forward(
        #     torch.stack(depth_norm_queue, dim=1) , 
        #     torch.stack(acc_norm_queue, dim=1) , 
        #     torch.stack(euler_norm_queue, dim=1) , 
        #     torch.stack(angle_vel_norm_queue, dim=1) , 
        #     torch.stack(target_vel_norm_queue, dim=1) 
        # )
        act_raw, _ = model.forward(
            # torch.stack(depth_norm_queue, dim=1) , 
            torch.stack(acc_norm_queue, dim=1) , 
            torch.stack(euler_norm_queue, dim=1) , 
            torch.stack(angle_vel_norm_queue, dim=1) , 
            torch.stack(target_vel_norm_queue, dim=1) 
        )
        # # 重参数
        # eps = torch.randn_like(mean)
        # act_raw = mean+eps*std
        act = torch.zeros(batch_size, 4)
        # 映射
        act[:, 0:2] = torch.tanh(act_raw[:, 0:2])*max_roll_pitch
        act[:, 2] = 0.0
        act[:, 3] = torch.sigmoid(act_raw[:, 2])
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
        # 奖励      
        vel_queue.append(geom.drone_vel)
        if len(vel_queue) > vel_queue_len:
            vel_queue.pop(0)
        vel_avg = torch.stack(vel_queue).mean(dim=0)   # 计算平均速度
        H_dir_queue.append(geom.drone_vel)
        if len(H_dir_queue) > H_dir_queue_len:
            H_dir_queue.pop(0)
        H_dir_avg = torch.stack(H_dir_queue).mean(dim=0)   # 计算平均水平方向
        pos_z_queue.append(geom.drone_pos)
        if len(pos_z_queue) > pos_z_queue_len:
            pos_z_queue.pop(0)
        pos_z_avg = torch.stack(pos_z_queue).mean(dim=0)   # 计算平均高度
        target_vel_vec = util.tensor_norm(target_pos-geom.drone_pos)*target_vel # coef["coef_move"]*torch.norm(geom.drone_pos-prev_pos, dim=-1) + \
        # reward = (
        #     coef["coef_vel"]*torch.clamp(torch.norm(geom.drone_vel, dim=-1)-torch.norm(target_vel_vec, dim=-1), min=0.0) + \
        #     coef["coef_H_dir"]*torch.norm(util.tensor_norm(H_dir_avg)[:, 0:2]-geom.drone_R[:, 0:2, 0], dim=-1) + \
        #     coef["coef_pos_z"]*torch.norm((pos_z_avg[:, 2]-target_pos[:, 2]).unsqueeze(1), dim=-1) + \
        #     coef["coef_distance_target"]*(torch.norm((geom.drone_pos-target_pos), dim=-1)**2) + \
        #     coef["coef_distance_no_safty"]*(safty_distance-geom.closest_distance) + \
        #     coef["coef_alive"]
        # )
        reward = (
            coef["coef_vel"]*torch.clamp(torch.norm(geom.drone_vel, dim=-1)-torch.norm(target_vel_vec, dim=-1), min=0.0) + \
            coef["coef_H_dir"]*torch.norm(util.tensor_norm(H_dir_avg)[:, 0:2]-geom.drone_R[:, 0:2, 0], dim=-1) + \
            coef["coef_pos_z"]*torch.norm((pos_z_avg-target_pos).unsqueeze(1), dim=-1) + \
            coef["coef_distance_target"]*(torch.norm((geom.drone_pos-target_pos), dim=-1)**2) + \
            coef["coef_distance_no_safty"]*(safty_distance-geom.closest_distance) + \
            coef["coef_alive"]
        )
        prev_pos = geom.drone_pos.clone().detach()

        reward_list.append(reward)
        if torch.all(geom.collision_state == True):
            break

    # 计算折扣奖励
    discount_reward_list = []
    discount_reward = torch.zeros(batch_size, 1, device=device)
    # collision_rate = 0
    # if torch.count_nonzero(~geom.collision_state).item() == 0:
    #     collision_rate = batch_size+1
    # else:
    #     collision_rate = batch_size/torch.count_nonzero(~geom.collision_state).item()
    for reward in reversed(reward_list):
        # discount_reward = collision_rate*steps/(i+1)*reward+0.99*discount_reward
        discount_reward = steps/(i+1)*reward+0.99*discount_reward
        discount_reward_list.insert(0, discount_reward)
    mean_reward_list = torch.stack(discount_reward_list).mean(dim=-1)
    # 计算损失
    batch_loss = -mean_reward_list
    loss = torch.mean(batch_loss)
    # 反向传播
    optim.zero_grad()
    loss.backward()
    optim.step()
    # 当平均奖励大于之前的最佳平均奖励 && 全部没有碰撞
    if torch.mean(mean_reward_list) > best_mean_reward and torch.all(~geom.collision_state):
        best_mean_reward = torch.mean(mean_reward_list)
        checkpoint_num += 1
        last_checkpoint_episode = episode
        checkpoint = {
            'epoch': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss if 'loss' in locals() else None,
            'best_score': best_mean_reward,
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
    @ Mean Reward: {torch.mean(mean_reward_list)}
    @ Min Reward: {torch.min(mean_reward_list)}
    @ Max Reward: {torch.max(mean_reward_list)}
    @ Best Mean Reward: {best_mean_reward}
    @ Last Checkpoint Episode: {last_checkpoint_episode}
    @ Duration Time: {elapsed}s
    {sep}
    """)

torch.save(model.state_dict(), "final.pth")

print("Save Final")
