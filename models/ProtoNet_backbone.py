import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from models.ProtoNet import ProtoNet
import os
from datetime import datetime
from utils.visualize import visualize_attention
import time


class ChannelAttention(nn.Module):
    """通道注意力模块
    
    参数:
        in_channels (int): 输入特征的通道数
        reduction_ratio (int): 降维比例
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        
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
        
        # 平均池化分支
        avg_out = self.avg_pool(x).squeeze(-1)
        avg_out = self.mlp(avg_out)
        
        # 最大池化分支
        max_out = self.max_pool(x).squeeze(-1)
        max_out = self.mlp(max_out)
        
        # 融合注意力
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
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class TransformerEncoder(nn.Module):
    """优化的单头自注意力模块
    
    参数:
        in_channels (int): 输入特征的通道数
        hidden_dim (int, 可选): 查询、键、值的投影维度，默认为输入通道数
        dropout (float): dropout比率，默认0.1
    """
    def __init__(self, in_channels: int, hidden_dim: int = None, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_channels
        
        # 查询、键、值的线性变换层
        self.query = nn.Linear(in_channels, self.hidden_dim)
        self.key = nn.Linear(in_channels, self.hidden_dim)
        self.value = nn.Linear(in_channels, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, in_channels)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            nn.ReLU(),
            nn.Linear(4 * in_channels, in_channels)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.hidden_dim ** 0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入维度：[batch_size, channels, length]
        b, c, t = x.size()
        
        # 转换维度为 [batch_size, length, channels]
        x_orig = x.permute(0, 2, 1).contiguous()
        
        # 应用第一个层归一化
        x = self.norm1(x_orig)
        
        # 自注意力计算
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # 注意力分数和权重
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        attention_weights = self.dropout(F.softmax(attention_scores, dim=-1))
        
        # 应用注意力并投影
        attended = torch.matmul(attention_weights, v)
        attended = self.dropout(self.out_proj(attended))
        
        # 第一个残差连接
        x = x_orig + attended
        
        # 前馈网络和第二个残差连接
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        # 转回原始维度 [batch_size, channels, length]
        out = x.permute(0, 2, 1).contiguous()
        
        return out

class CNN1D_Attention(nn.Module):
    """专门为注意力机制设计的1D-CNN网络，基于EnhancedCNN1D_embed架构
    
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
        dropout: float = 0.1
    ):
        super(CNN1D_Attention, self).__init__()
        
        # 输入标准化层
        self.norm = nn.BatchNorm1d(in_channels)
        
        # 第一个卷积块 - 16通道，无BN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
        )
        
        # 第二个卷积块 - 32通道，有BN
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
        )
        
        # 第三个卷积块 - 64通道，有BN
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
        )
        
        # 第四个卷积块 - 64通道，有BN
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
        )
        
        # 第五个卷积块 - 64通道，有BN，无池化
        self.conv5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        
        )
        
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征映射层
        self.feature_layer = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(),
           
        )
        
    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取中间特征，用于应用注意力机制
        
        参数:
            x: 输入数据 [batch_size, in_channels, length]
            
        返回:
            features: 中间特征 [batch_size, 32, length/4]
        """
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    def get_final_features(self, x: torch.Tensor) -> torch.Tensor:
        """从中间特征继续处理得到最终特征
        
        参数:
            x: 中间特征 [batch_size, 32, length/4]
            
        返回:
            features: 最终特征 [batch_size, feature_dim]
        """
       
        x = self.conv4(x)
        x = self.conv5(x)
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
        
        # 注意力模块 - 使用64作为通道数（第三个卷积块的输出通道数）
        intermediate_channels = 64  # backbone的中间特征通道数
        
        if attention_type == 'channel':
            self.attention = ChannelAttention(intermediate_channels)
        elif attention_type == 'spatial':
            self.attention = SpatialAttention()
        elif attention_type == 'cbam':
            self.attention = CBAM(intermediate_channels)
        elif attention_type == 'transformer':
            self.attention = TransformerEncoder(intermediate_channels, hidden_dim, dropout)  # 使用intermediate_channels
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
        features = self.backbone.get_intermediate_features(x)  # [batch_size, 64, length/8]
        
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
        return super().forward(support_images, support_labels, query_images) 