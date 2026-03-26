import torch
import torch.nn as nn
import torch.nn.functional as F

import kernel.geom as geom
import kernel.util as util
import kernel.visual as visual
import model

episodes = 2000
steps = 200

batch_size = 20
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
max_depth = 10.0

geom = geom.geom(batch_size=batch_size, device=device, dt=dt, init_pos=init_pos,
                 init_euler=init_euler, pos_offset=pos_offset, euler_offset=euler_offset, mass=mass, T_max=T_max,
                 ang_vel_max=ang_vel_max, res_W=res_W, res_H=res_H, fov_H=fov_H, fov_V=fov_V, min_depth=min_depth,
                 max_depth=max_depth)
geom.add_sphere(5.0, 0.0, 1.0, 1.0)
geom.add_sphere(5.0, 0.0, 2.2, 0.5)
geom.add_cylinder(3.0, 1.0, 1.0, 0.5, 2.0)
geom.add_cylinder(3.0, -0.7, 1.5, 0.3, 3.0)
# geom.add_sphere(4.0, 0.0 ,2.5, 0.5)
# geom.add_sphere(3.0, 0.0 ,2.5, 1.0)
# geom.add_sphere(4.0, 2.0 ,2.5, 1.5)
# geom.add_sphere(4.0, -1.0 ,2.5, 0.5)

visual = visual.visual(urdf="urdf/ge_fpv.urdf", device=device, init_pos=init_pos[0, :], init_euler=init_euler[0, :], batch_size=0)
visual.add_sphere(5.0, 0.0, 1.0, 1.0)
visual.add_sphere(5.0, 0.0, 2.2, 0.5)
visual.add_cylinder(3.0, 1.0, 1.0, 0.5, 2.0)
visual.add_cylinder(3.0, -0.7, 1.5, 0.3, 3.0)
# visual.add_sphere(5.0, 0.0 ,0.5, 2.0)
# visual.add_sphere(3.0, 0.0 ,2.5, 1.0)
# visual.add_sphere(4.0, 2.0 ,2.5, 1.5)
# visual.add_sphere(4.0, -1.0 ,2.5, 0.5)

visual.build()

model = model.Model()
optim = torch.optim.Adam(model.parameters(), lr=3e-4)

for episode in range(episodes):
    geom.reset()
    total_reward = 0
    episode_data = []
    for i in range(steps):
        # 模型前向传播
        mean, std = model.forward(geom.depth, geom.drone_acc, geom.drone_euler, geom.drone_ang_vel)
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.sample()
        log_prob_raw = dist.log_prob(action_raw).sum(-1)
        # 将采样结果映射到 角度：0-360 推力:0-1
        action = action_raw.clone()
        action[:, 0:3] = torch.sigmoid(action_raw[:, 0:3]) * 360 - 180
        action[:, 3] = torch.sigmoid(action_raw[:, 3])
        # 雅可比修正
        action_sigmoid = torch.sigmoid(action_raw)
        jacobian_ang = torch.log(360 * action_sigmoid[:, 0:3] * (1 - action_sigmoid[:, 0:3]))
        jacobian_thrust = torch.log(action_sigmoid[:, 3] * (1 - action_sigmoid[:, 3]))
        log_jacobian = (jacobian_ang.sum(-1) + jacobian_thrust)
        log_prob = log_prob_raw - log_jacobian  # 变换后的对数概率
        geom.step(act=torch.tensor(action, device=device), 
                T_att=0.0, 
                show_depth=True, 
                show_idx=0, 
                noise=True, 
                noise_range=0.005, 
                black_hole_prob=0.01)
        # geom.step(act=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float, device=device), 
        #           T_att=0.0, 
        #           show_depth=True, 
        #           show_idx=0, 
        #           noise=True, 
        #           noise_range=0.005, 
        #           black_hole_prob=0.01)

        if torch.all(geom.collision_state == True):
            print("RESET\n\n\n\n\n\n")
            break
        
        # 奖励/惩罚
        not_collision = 1-geom.collision_state.int()
        reward = (-0.5 * torch.norm(geom.drone_pos-init_pos, dim=-1)
                  +0.1*not_collision)
        total_reward += reward
        episode_data.append([log_prob, reward])
        visual.step(geom.drone_pos[0, ...].detach(), geom.drone_euler[0, ...].detach())
        # visual.step(init_pos, geom.drone_euler)
        # print("RUNNING")
        
    # 计算折扣回报
    discount_rewards = []
    discount_reward = torch.zeros_like(total_reward)
    for _, reward in reversed(episode_data):
        discount_reward = reward + 0.99 * discount_reward
        discount_rewards.insert(0, discount_reward)

    # # 转换为tensor并标准化
    # discounted_rewards = torch.stack([torch.tensor(rewards, device=device) 
    #                                 for rewards in discounted_rewards_per_batch]).T  # [steps, batch_size]

    # if discounted_rewards.shape[0] > 1:
    #     mean = discounted_rewards.mean(dim=0, keepdim=True)
    #     std = discounted_rewards.std(dim=0, keepdim=True) + 1e-8
    #     discounted_rewards = (discounted_rewards - mean) / std

    # 计算损失
    loss_list = []
    for (log_prob, _),discount_reward in zip(episode_data, discount_rewards):
        loss_list.insert(0, -log_prob * discount_reward)

    loss = torch.stack(loss_list).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"Episode {episode:3d}/{episodes} | Mean Reward: {torch.mean(total_reward)} | Min Reward: {torch.min(total_reward)} | Max Reward: {torch.max(total_reward)}")

torch.save(model.state_dict(), "final.pth")
print("✅ 训练完成！模型已保存！")