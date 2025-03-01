import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from models.ProtoNet import ProtoNet


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


class CNN1D_Attention(nn.Module):
    """专门为注意力机制设计的1D-CNN网络
    
    参数:
        in_channels (int): 输入通道数
        hidden_dim (int): 隐藏层维度
        feature_dim (int): 输出特征维度
        dropout (float): Dropout比率
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        dropout: float = 0.5
    ):
        super(CNN1D_Attention, self).__init__()
        
        # 第一个卷积块 - 保持通道数不变，方便后续注意力机制
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # 第二个卷积块 - 同样保持通道数不变
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # 第三个卷积块 - 增加通道数
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征映射层
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取中间特征，用于应用注意力机制
        
        参数:
            x: 输入数据 [batch_size, in_channels, length]
            
        返回:
            features: 中间特征 [batch_size, hidden_dim, length/4]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    def get_final_features(self, x: torch.Tensor) -> torch.Tensor:
        """从中间特征继续处理得到最终特征
        
        参数:
            x: 中间特征 [batch_size, hidden_dim, length/4]
            
        返回:
            features: 最终特征 [batch_size, feature_dim]
        """
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.feature_layer(x)
        x = F.normalize(x, p=2, dim=1)  # L2归一化
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整的前向传播
        
        参数:
            x: 输入数据 [batch_size, in_channels, length]
            
        返回:
            features: 最终特征 [batch_size, feature_dim]
        """
        x = self.get_intermediate_features(x)
        x = self.get_final_features(x)
        return x


class AttentiveEncoder(nn.Module):
    """带有注意力机制的编码器"""
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        attention_type: str = 'cbam',
        dropout: float = 0.5
    ):
        super(AttentiveEncoder, self).__init__()
        
        # 使用新的CNN1D_Attention作为backbone
        self.backbone = CNN1D_Attention(in_channels, hidden_dim, feature_dim, dropout)
        
        # 注意力模块 - 使用hidden_dim作为通道数
        if attention_type == 'channel':
            self.attention = ChannelAttention(hidden_dim)
        elif attention_type == 'spatial':
            self.attention = SpatialAttention()
        elif attention_type == 'cbam':
            self.attention = CBAM(hidden_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征 [batch_size, in_channels, length]
            
        返回:
            features: 增强后的特征 [batch_size, feature_dim]
        """
        # 获取中间特征
        features = self.backbone.get_intermediate_features(x)  # [batch_size, hidden_dim, length/4]
        
        # 应用注意力
        attended_features = self.attention(features)
        
        # 继续处理得到最终特征
        final_features = self.backbone.get_final_features(attended_features)  # [batch_size, feature_dim]
        
        return final_features


class ProtoNetWithAttention(ProtoNet):
    """带有注意力机制的Prototypical Network
    
    参数:
        in_channels (int): 输入数据的通道数
        hidden_dim (int): 嵌入网络的隐藏层维度
        feature_dim (int): 最终特征向量的维度
        attention_type (str): 注意力类型 ('channel', 'spatial', 'cbam')
        backbone (str): 特征提取器的类型 ('cnn1d', 'cnn2d', 'lstm')
        distance_type (str): 距离度量方式 ('euclidean', 'cosine')
        dropout (float): Dropout比率
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        attention_type: str = 'cbam',
        backbone: str = 'cnn1d',
        distance_type: str = 'euclidean',
        dropout: float = 0.5
    ):
        super().__init__(in_channels, hidden_dim, feature_dim, backbone, distance_type)
        self.encoder = AttentiveEncoder(in_channels, hidden_dim, feature_dim, attention_type, dropout)

    def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor, 
                query_images: torch.Tensor) -> torch.Tensor:
        """
        参数:
            support_images: [batch_size, n_way * n_support, channels, length]
            support_labels: [batch_size, n_way * n_support]
            query_images: [batch_size, n_way * n_query, channels, length]
            
        返回:
            query_logits: [batch_size * n_way * n_query, n_way]
        """
        # 直接使用父类的forward方法，因为现在它已经支持batch处理
        return super().forward(support_images, support_labels, query_images) 