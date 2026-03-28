import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    @ 模型
    输入:
        depth:深度:[batch_size, 50, 80]
        acc:加速度:[batch_size, 3]
        ang:角度:[batch_size, 3]
        ang_vel:角速度:[batch_size, 3]
    输出:
        act:姿态欧拉角/推力:[batch_size, 4]
        姿态欧拉角:-180-180
        推力:0-1.0
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 深度图像处理CNN
        # 输入: [batch, 1, 50, 80]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.LeakyReLU(), nn.MaxPool2d(2),  # [batch, 32, 12, 20]
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(), nn.MaxPool2d(2),  # [batch, 64, 3, 5]
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(),                  # [batch, 128, 1, 2]
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),                 # [batch, 256, 1, 1]
            nn.AdaptiveAvgPool2d(1), nn.Flatten()                    # [batch, 256]
        )
        
        # 姿态传感器MLP (acc, ang, ang_vel)
        # 输入: [batch, 9] (3+3+3)
        self.pose_mlp = nn.Sequential(
            nn.Linear(9, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )
        
        # 输出层：姿态欧拉角（roll, pitch, yaw）和推力
        self.output_layer = nn.Linear(512, 4)
        
        # 为姿态欧拉角添加tanh激活函数，将输出限制在[-1, 1]范围内
        # 然后通过缩放映射到[-180, 180]
        # 推力使用sigmoid激活函数，将输出限制在[0, 1]范围内
        
    def forward(self, depth, acc, ang, ang_vel):
        # 处理深度图像: 添加通道维度 [batch, 50, 80] -> [batch, 1, 50, 80]
        depth = depth.unsqueeze(1)
        depth_features = self.cnn(depth)  # [batch, 256]
        
        # 拼接姿态传感器数据 [batch, 9]
        pose_input = torch.cat([acc, ang, ang_vel], dim=-1)  # [batch, 9]
        pose_features = self.pose_mlp(pose_input)  # [batch, 128]
        
        # 融合特征
        fused_features = torch.cat([depth_features, pose_features], dim=-1)  # [batch, 384]
        fused_features = self.fusion(fused_features)  # [batch, 512]
        
        # 输出原始值
        output = self.output_layer(fused_features)  # [batch, 4]
        
        # 分离姿态欧拉角和推力
        euler_angles = output[:, :3]  # [batch, 3] - roll, pitch, yaw
        thrust = output[:, 3:4]       # [batch, 1] - thrust
        
        # 应用激活函数
        # 姿态欧拉角: 使用tanh限制在[-1,1]，然后缩放到[-180,180]
        euler_angles = torch.tanh(euler_angles) * 180.0
        
        # 推力: 使用sigmoid限制在[0,1]
        thrust = torch.sigmoid(thrust)
        
        # 拼接最终输出
        act = torch.cat([euler_angles, thrust], dim=-1)  # [batch, 4]
        
        return act