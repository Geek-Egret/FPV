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
        mean:姿态欧拉角/推力均值:[batch_size, 4]
        std:姿态欧拉角/推力标准差:[batch_size, 4]
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 深度图像处理CNN
        # 输入: [batch, 1, 50, 80]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.ReLU(), nn.MaxPool2d(2),  # [batch, 32, 12, 20]
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.MaxPool2d(2),  # [batch, 64, 3, 5]
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),                  # [batch, 128, 1, 2]
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),                 # [batch, 256, 1, 1]
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
        
        # 输出均值 (4个值: roll, pitch, yaw, thrust)
        self.mean = nn.Linear(512, 4)
        
        # 输出标准差 (4个值: 对应每个输出的标准差)
        self.log_std = nn.Parameter(torch.zeros(4))
        
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
        
        # 输出均值
        mean = self.mean(fused_features)  # [batch, 4]
        
        # 输出标准差 (使用exp确保标准差为正)
        std = torch.exp(self.log_std)  # [4]
        std = std.expand_as(mean)  # [batch, 4]
        
        return mean, std