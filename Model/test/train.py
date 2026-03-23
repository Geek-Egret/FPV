import torch
import torch.nn as nn
import torch.nn.functional as F
import env.geom as geom
import env.util as util

# ===================== 你的参数 =====================
device = 'cpu'
dt = 0.005
drone_init_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double, device=device)
drone_init_euler = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double, device=device)
T_max = 4 * 0.4315 * 9.81
T_max_att = 0.0
ang_vel_max = torch.tensor([0.2, 0.2, 0.2], dtype=torch.double, device=device)
depth_pos_offset = torch.tensor([0.0425, 0.0, 0.0345], dtype=torch.double, device=device)
num = 1
num_cylinders = 1
cylinders = torch.tensor([2.0, 1.0, 0.0, 3.0, 0.5], dtype=torch.double, device=device)
num_boxes = 1
boxes = torch.tensor([1.0, 0.0, 0.0, 0.3, 0.5, 3.0], dtype=torch.double, device=device)

# ===================== 环境（纯手写，无TorchRL，零报错） =====================
class GenesisDroneEnv:
    def __init__(self):
        self.device = device
        self.geom = geom.geom(
            camera_pos=(3.5, 3.5, 3.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=35,
            max_FPS=1000,
            show_viewer=True,
            dt=dt,
            device=device
        )
        self.geom.add_cylinders(num_cylinders, cylinders)
        self.geom.add_boxes(num_boxes, boxes)
        self.geom.add_drone(
            urdf_path="urdf/ge_fpv.urdf",
            drone_init_pos=drone_init_pos,
            drone_init_euler=drone_init_euler,
            T_max=T_max,
            T_max_att=T_max_att,
            ang_vel_max=ang_vel_max,
            res_W=640,
            res_H=400,
            depth_pos_offset=depth_pos_offset,
            depth_euler_offset=drone_init_euler,
            depth_fov_H=67.9,
            depth_fov_V=45.3,
            num=num
        )
        self.geom.build()

    def reset(self):
        self.geom.reset()
        obs = {
            # 修复维度：增加通道维度 [1, H, W]
            "depth_img": self.geom.depth_img.squeeze(0).float().unsqueeze(0),
            "quat": self.geom.quat.squeeze(0).float(),
            "angular_vel": self.geom.ang_vel.squeeze(0).float(),
        }
        return obs

    def step(self, action):
        desired_quat = action[:4]
        desired_thrust = action[4]
        desired_quat = F.normalize(desired_quat, dim=-1)
        desired_thrust = torch.clamp(desired_thrust, 0, 1)
        desired_euler = util.quat_to_euler(desired_quat)
        collision = self.geom.step(desired_euler, desired_thrust)

        quat_now = self.geom.quat.squeeze(0).float()
        ang_vel = self.geom.ang_vel.squeeze(0).float()
        pos_now = self.geom.pos.squeeze(0).float()
        target_pos = torch.tensor([0.0, 0.0, 1.0], device=device)
        target_pos_range = torch.tensor([0.1, 0.1, 0.1], device=device)
        target_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        target_ang_vel = torch.tensor([0.0, 0.0, 0.0], device=device)

        pos_err = torch.norm(torch.abs(pos_now - target_pos)) - torch.norm(target_pos_range)
        quat_err = torch.norm(quat_now - target_quat)
        ang_vel_err = torch.norm(ang_vel - target_ang_vel)
        
        if pos_err > 0.0:
            reward = -pos_err # - 2.0*quat_err - 5.0*ang_vel_err - 0.5*pos_err
        else:
            reward = -20.0*pos_err
        if collision:
            reward -= 10.0

        obs = {
            "depth_img": self.geom.depth_img.squeeze(0).float().unsqueeze(0),
            "quat": quat_now,
            "angular_vel": self.geom.ang_vel.squeeze(0).float(),
        }
        done = collision
        return obs, reward, done, {}

# ===================== 模型 =====================
class DepthPoseActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,5,2,2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(),
            nn.Conv2d(128,256,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.pose_mlp = nn.Sequential(
            nn.Linear(7, 128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(256+128,512), nn.ReLU(),
            nn.Linear(512,512), nn.ReLU()
        )
        self.mean = nn.Linear(512, 5)
        self.log_std = nn.Parameter(torch.zeros(5))

    def forward(self, obs):
        d = obs["depth_img"]  # [1,400,640]
        q = obs["quat"]       # [4]
        w = obs["angular_vel"]# [3]
        
        # 增加 batch 维度
        if d.dim() == 3:
            d = d.unsqueeze(0)
            q = q.unsqueeze(0)
            w = w.unsqueeze(0)

        img_feat = self.cnn(d)
        pose_feat = self.pose_mlp(torch.cat([q, w], dim=-1))
        feat = self.fusion(torch.cat([img_feat, pose_feat], dim=-1))
        mean = self.mean(feat)
        std = torch.exp(self.log_std).clamp(0.01, 0.5)
        return mean, std

# ===================== 训练 =====================
if __name__ == "__main__":
    env = GenesisDroneEnv()
    model = DepthPoseActor().to(device).float()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("✅ 启动无人机悬停训练！无任何报错！")
    for episode in range(100000):
        obs = env.reset()
        total_reward = 0
        episode_data = []
        live_time = 0
        for step in range(400):
            mean, std = model(obs)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

            next_obs, reward, done, _ = env.step(action.squeeze())
            live_time += 1
            reward += 0.1*live_time
            total_reward += reward

            obs = next_obs
            
            episode_data.append((log_prob, reward))
            if done:
                break
        discounted_rewards = []
        cumulative_reward = 0
        for log_prob, reward in reversed(episode_data):
            cumulative_reward = reward + 0.99 * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(discounted_rewards) > 1:  # 避免除零
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        loss_list = []
        for (log_prob, reward), rewards in zip(episode_data, discounted_rewards):
            loss_list.append(-log_prob * rewards)  # 使用累积奖励R而不是即时reward
        loss = torch.stack(loss_list).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Episode {episode:3d} | Total Reward: {total_reward:.2f}")

    torch.save(model.state_dict(), "drone_hover_final.pth")
    print("✅ 训练完成！模型已保存！")
