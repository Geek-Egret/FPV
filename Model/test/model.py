import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import TensorDictModule


class DepthPoseControlNet(nn.Module):
    """
    TorchRL 自定义控制模型
    输入：深度图像 (1,640,400) + 姿态四元数 (4) + 角速度 (3)
    输出：期望姿态四元数 (4) + 期望推力 (1)
    """
    def __init__(self):
        super().__init__()
        
        # ====================== 1. 深度图像 CNN 特征提取 ======================
        # 输入: [B, 1, 640, 400]
        self.cnn = nn.Sequential(
            # 卷积层 1
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # [B,32,320,200]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B,32,160,100]
            
            # 卷积层 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B,64,80,50]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B,64,40,25]
            
            # 卷积层 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B,128,20,12]
            nn.ReLU(),
            
            # 卷积层 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B,256,10,6]
            nn.ReLU(),
            
            # 全局池化 + 展平 → 固定长度图像特征
            nn.AdaptiveAvgPool2d(1),  # [B,256,1,1]
            nn.Flatten()  # [B,256]
        )
        
        # ====================== 2. 姿态/角速度 MLP 特征提取 ======================
        # 输入: 四元数(4) + 角速度(3) = 7 维
        self.pose_mlp = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # ====================== 3. 多模态特征融合 ======================
        # 图像特征(256) + 姿态特征(128) = 384 维
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # ====================== 4. 双输出头 ======================
        # 输出 1：期望姿态四元数 (4维) → 归一化为单位四元数
        self.quat_head = nn.Linear(512, 4)
        
        # 输出 2：期望推力 (1维) → 非负输出
        self.thrust_head = nn.Linear(512, 1)

    def forward(self, depth_img, quat, angular_vel):
        """
        前向传播（严格匹配 TorchRL 输入格式）
        Args:
            depth_img: [B, 1, 640, 400]  深度图像
            quat: [B, 4]                 姿态四元数
            angular_vel: [B, 3]          角速度
        Returns:
            desired_quat: [B, 4]         期望姿态四元数（单位化）
            desired_thrust: [B, 1]       期望推力（非负）
        """
        # 1. 提取图像特征
        img_feat = self.cnn(depth_img)  # [B,256]
        
        # 2. 拼接姿态+角速度并提取特征
        pose_feat = self.pose_mlp(torch.cat([quat, angular_vel], dim=-1))  # [B,128]
        
        # 3. 特征融合
        fused_feat = self.fusion(torch.cat([img_feat, pose_feat], dim=-1))  # [B,512]
        
        # 4. 输出预测
        desired_quat = self.quat_head(fused_feat)
        desired_quat = F.normalize(desired_quat, dim=-1)  # 四元数必须单位化
        
        desired_thrust = self.thrust_head(fused_feat)
        desired_thrust = F.softplus(desired_thrust)  # 推力非负约束
        
        return desired_quat, desired_thrust


# ====================== TorchRL 集成包装（关键！） ======================
def make_custom_policy():
    """
    包装成 TorchRL 标准策略模块
    可直接用于 PPO/SAC 等算法
    """
    model = DepthPoseControlNet()
    
    # 定义输入输出键（与你的 TensorDict 键名对应）
    policy = TensorDictModule(
        model,
        in_keys=["depth_img", "quat", "angular_vel"],
        out_keys=["desired_quat", "desired_thrust"]
    )
    return policy
