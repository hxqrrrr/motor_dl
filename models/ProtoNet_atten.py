import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from models.ProtoNet import ProtoNet
from models.ProtoNet_backbone import backone
class AttentiveEncoder(nn.Module):
    """带有注意力机制的编码器"""
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        attention_type: str,
        dropout: float,
        backbone_type: str = 'cnn1d'
    ):
        super().__init__()
        
        # 确保dropout是float类型
        dropout = float(dropout)
        
        # 初始化backbone
        if backbone_type == 'motornet':
            self.backbone = backone(
                in_channels=in_channels,
                feature_dim=feature_dim,  # 只传递必要的参数
                dropout_rate=dropout,
                backbone_type=backbone_type
            )
        else:
            self.backbone = backone(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                feature_dim=feature_dim,
                dropout_rate=dropout,
                backbone_type=backbone_type
            )
        
        # 注意力模块
        if attention_type == 'channel':
            self.attention = ChannelAttention(feature_dim)  
        elif attention_type == 'spatial':
            self.attention = SpatialAttention()
        elif attention_type == 'cbam':
            self.attention = CBAM(feature_dim)
        elif attention_type == 'motorsignal':
            self.attention = MotorSignalAttention(in_channels)
        elif attention_type == 'no':
            self.attention = None
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征 [batch_size, in_channels, length]
            
        返回:
            features: 增强后的特征 [batch_size, feature_dim]
        """
        # 1. 先应用注意力机制
        if self.attention is not None:
            x = self.attention(x)  # [batch_size, in_channels, length]
            
        # 2. 然后通过backbone
        features = self.backbone(x)  # [batch_size, feature_dim]
        features = features.unsqueeze(-1)  # [batch_size, feature_dim, 1]
        
        return features.squeeze(-1)  # [batch_size, feature_dim]
    

class ChannelAttention(nn.Module):
    """通道注意力模块
    
    参数:
        in_channels (int): 输入特征的通道数
        reduction_ratio (int): 降维比例
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        
        # 确保降维后的通道数至少为1
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 共享MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征 [batch_size, channels, length]
            
        返回:
            out: 加权后的特征 [batch_size, channels, length]
        """
        b, c, _ = x.size()
        
        # 平均池化分支 [b, c, 1] -> [b, c]
        avg_out = self.avg_pool(x).squeeze(-1)
        avg_out = self.mlp(avg_out)
        
        # 最大池化分支 [b, c, 1] -> [b, c]
        max_out = self.max_pool(x).squeeze(-1)
        max_out = self.mlp(max_out)
        
        # 融合两个分支 [b, c] -> [b, c, 1]
        attention = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        
        return x * attention


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征 [batch_size, channels, length]
            
        返回:
            out: 加权后的特征 [batch_size, channels, length]
        """
        # 计算通道维度上的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接特征
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 空间注意力权重
        attention = self.sigmoid(self.conv(x_cat))
        
        return x * attention


class CBAM(nn.Module):
    """CBAM注意力模块：结合通道注意力和空间注意力"""
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 依次应用通道注意力和空间注意力
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x




class MotorSignalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.freq_attention = nn.Sequential(
            # 将输入通道数改为实际的通道数
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, groups=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=1),
            nn.ReLU()
        )
        
        self.phase_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, groups=1),
            nn.ReLU()
        )
        
        self.channel_interaction = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.size()
        
        # 1. 频域分析
        x_fft = torch.fft.rfft(x, dim=2)
        x_fft_mag = torch.abs(x_fft)  # 幅值谱
        
        # 幅值特征
        freq_mag = self.freq_attention(x_fft_mag)  # [b, 64, l//2+1]
        freq_mag = torch.mean(freq_mag, dim=2)  # [b, 64]
        
        # 相位特征
        phase_feat = self.phase_conv(x)  # [b, 64, l]
        phase_feat = torch.mean(phase_feat, dim=2)  # [b, 64]
        
        # 2. 通道间关系
        channel_feat = self.channel_interaction(
            torch.cat([freq_mag, phase_feat], dim=1)  # [b, 128]
        )  # [b, in_channels]
        
        # 3. 生成注意力权重
        attention = self.sigmoid(channel_feat).unsqueeze(-1)  # [b, in_channels, 1]
        
        return x * attention  # 返回加权后的特征
