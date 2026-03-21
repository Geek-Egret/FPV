import torch
from torch import nn

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class Model(nn.Module):
    def __init__(self, dim_obs=3, dim_action=4, img_height=400, img_width=640) -> None:
        super().__init__()
        
        # 动态计算卷积层输出尺寸，适配 400x640 输入
        self.img_height = img_height
        self.img_width = img_width
        
        self.stem = nn.Sequential(
            # 输入: [B, 1, 400, 640]
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2, bias=False),
            # 输出: [B, 32, 100, 160] ((400-8+4)/4+1=100, (640-8+4)/4+1=160)
            nn.LeakyReLU(0.05),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            # 输出: [B, 64, 50, 80] ((100-4+2)/2+1=50, (160-4+2)/2+1=80)
            nn.LeakyReLU(0.05),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            # 输出: [B, 128, 25, 40] ((50-3+2)/2+1=25, (80-3+2)/2+1=40)
            nn.LeakyReLU(0.05),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            # 输出: [B, 256, 13, 20] ((25-3+2)/2+1=13, (40-3+2)/2+1=20)
            nn.LeakyReLU(0.05),
            
            nn.Flatten(),
            # 展平: [B, 256 * 13 * 20 = 66560]
            nn.Linear(256 * 13 * 20, 512, bias=False),
            nn.LeakyReLU(0.05),
            nn.Linear(512, 256, bias=False),
        )
        
        # 向量投影层: 3 -> 256
        self.v_proj = nn.Linear(dim_obs, 256)
        self.v_proj.weight.data.mul_(0.5)
        
        # GRU层
        self.gru = nn.GRUCell(256, 256)
        
        # 输出层: 256 -> 4
        self.fc = nn.Linear(256, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)
        
        # 可选：添加dropout防止过拟合
        self.dropout = nn.Dropout(0.1)

    def reset(self):
        # 重置GRU隐藏状态
        self.hx = None

    def forward(self, x: torch.Tensor, v, hx=None):
        """
        Args:
            x: [batch_size, 1, 400, 640] 图像输入
            v: [batch_size, 3] 向量输入
            hx: [batch_size, 256] 或 None GRU隐藏状态
        Returns:
            act: [batch_size, 4] 动作输出
            None: 占位符
            hx: [batch_size, 256] 更新后的隐藏状态
        """
        # 检查输入尺寸
        assert x.shape[2:] == (self.img_height, self.img_width), \
            f"图像输入尺寸错误，期望({self.img_height}, {self.img_width})，实际{x.shape[2:]}"
        assert v.shape[1] == 3, f"向量输入维度错误，期望3，实际{v.shape[1]}"
        
        # 图像特征提取
        img_feat = self.stem(x)  # [B, 256]
        
        # 向量投影
        v_feat = self.v_proj(v)  # [B, 256]
        
        # 融合特征
        x = self.act(img_feat + v_feat)
        x = self.dropout(x)
        
        # GRU更新
        hx = self.gru(x, hx)
        
        # 输出动作
        act = self.fc(self.act(hx))
        
        return act, None, hx