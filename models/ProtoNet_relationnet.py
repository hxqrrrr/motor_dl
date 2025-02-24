import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from models.ProtoNet_attention import AttentiveEncoder

class CNN1D_embed(nn.Module):
    """用于Prototypical Network的1D-CNN嵌入网络
    
    参数:
        in_channels: 输入通道数
        hidden_dim: 隐藏层维度
        feature_dim: 输出特征维度
        dropout: Dropout比率
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        dropout: float = 0.5
    ):
        super(CNN1D_embed, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),  # 较小的dropout率
            nn.MaxPool1d(2)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),  # 中等的dropout率
            nn.MaxPool1d(2)
        )
        
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),  # 较大的dropout率
            nn.MaxPool1d(2)
        )

        # 添加自适应平均池化层
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征映射层
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)  # 最大的dropout率
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一个卷积块
        x = self.conv1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 特征映射
        x = self.feature_layer(x)
        
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        
        return x


class AttentionRelationModule(nn.Module):
    """基于注意力机制的关系模块
    
    参数:
        feature_dim: 输入特征维度
        hidden_dim: 注意力机制的隐藏层维度
    """
    def __init__(self, feature_dim: int, hidden_dim: int):
        super(AttentionRelationModule, self).__init__()
        
        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # 关系网络
        self.relation_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        """
        参数:
            query_features: [batch_size, n_query, feature_dim]
            support_features: [batch_size, n_support, feature_dim]
            
        返回:
            relation_scores: [batch_size, n_query, n_support]
        """
        batch_size = query_features.size(0)
        n_query = query_features.size(1)
        n_support = support_features.size(1)
        
        # 对每个batch单独处理
        relation_scores = []
        for i in range(batch_size):
            # 准备当前batch的特征
            q = query_features[i].unsqueeze(1)  # [n_query, 1, feature_dim]
            s = support_features[i].unsqueeze(1)  # [n_support, 1, feature_dim]
            
            # 计算注意力权重
            attn_output, _ = self.multihead_attn(
                q.transpose(0, 1),  # [1, n_query, feature_dim]
                s.transpose(0, 1),  # [1, n_support, feature_dim]
                s.transpose(0, 1)   # [1, n_support, feature_dim]
            )
            attn_output = attn_output.transpose(0, 1)  # [n_query, 1, feature_dim]
            
            # 准备关系网络的输入
            q_expanded = q.expand(-1, n_support, -1)  # [n_query, n_support, feature_dim]
            s_expanded = s.expand(n_query, -1, -1)    # [n_query, n_support, feature_dim]
            
            # 连接特征
            combined = torch.cat([
                q_expanded,
                attn_output.expand(-1, n_support, -1)
            ], dim=-1)  # [n_query, n_support, feature_dim*2]
            
            # 计算关系分数
            scores = self.relation_net(combined).squeeze(-1)  # [n_query, n_support]
            relation_scores.append(scores)
        
        # 堆叠所有batch的结果
        relation_scores = torch.stack(relation_scores)  # [batch_size, n_query, n_support]
        return relation_scores


class AllModel(nn.Module):
    """基于注意力机制的关系网络版本的Prototypical Network
    
    参数:
        in_channels (int): 输入数据的通道数
        hidden_dim (int): 嵌入网络的隐藏层维度
        feature_dim (int): 最终特征向量的维度
        backbone (str): 特征提取器的类型 ('cnn1d', 'cnn2d', 'lstm')
        distance_type (str): 距离度量方式 ('euclidean', 'cosine', 'relation')
        dropout (float): Dropout比率
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        backbone: str = 'cnn1d',
        distance_type: str = 'relation',
        dropout: float = 0.5
    ):
        super(AllModel, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.backbone = backbone
        self.distance_type = distance_type
        self.dropout = dropout
        
        # 初始化嵌入网络
        self.encoder = self._build_encoder()
        
        # 初始化关系模块
        if distance_type == 'relation':
            self.relation_module = AttentionRelationModule(feature_dim, hidden_dim)
    
    def _build_encoder(self) -> nn.Module:
        """构建特征提取器"""
        if self.backbone == 'cnn1d':
            return CNN1D_embed(
                self.in_channels, 
                self.hidden_dim, 
                self.feature_dim,
                self.dropout
            )
        elif self.backbone in ['channel', 'spatial', 'cbam']:
            return AttentiveEncoder(
                self.in_channels,
                self.hidden_dim,
                self.feature_dim,
                attention_type=self.backbone,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown backbone type: {self.backbone}")
        
    def _compute_prototypes(self, support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """计算每个类别的原型，支持batch处理
        
        参数:
            support_features: support set的特征向量 [batch_size, n_support, feature_dim]
            support_labels: support set的标签 [batch_size, n_support]
            
        返回:
            prototypes: 类别原型向量 [batch_size, n_way, feature_dim]
        """
        batch_size = support_features.size(0)
        n_way = len(torch.unique(support_labels[0]))  # 假设每个batch的类别数相同
        feature_dim = support_features.size(-1)
        device = support_features.device
        
        # 初始化原型tensor
        prototypes = torch.zeros(batch_size, n_way, feature_dim, device=device)
        
        # 对每个batch分别计算原型
        for i in range(batch_size):
            batch_features = support_features[i]  # [n_support, feature_dim]
            batch_labels = support_labels[i]      # [n_support]
            
            # 获取当前batch的类别
            classes = torch.unique(batch_labels)
            
            # 对每个类别计算原型
            for j, cls in enumerate(classes):
                mask = batch_labels == cls
                class_features = batch_features[mask]
                prototypes[i, j] = torch.mean(class_features, dim=0)
                
        return prototypes
        
    def _compute_distances(self, query_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """计算查询样本与原型之间的距离或关系分数
        
        参数:
            query_features: query set的特征向量 [batch_size, n_query, feature_dim]
            prototypes: 类别原型向量 [batch_size, n_way, feature_dim]
            
        返回:
            distances/scores: 距离矩阵或关系分数 [batch_size, n_query, n_way]
        """
        if self.distance_type == 'relation':
            # 使用注意力关系模块计算关系分数
            return self.relation_module(query_features, prototypes)
        
        batch_size = query_features.size(0)
        n_query = query_features.size(1)
        n_way = prototypes.size(1)
        
        if self.distance_type == 'euclidean':
            # 计算欧氏距离
            query_features = query_features.unsqueeze(2)
            prototypes = prototypes.unsqueeze(1)
            distances = torch.sum((query_features - prototypes) ** 2, dim=-1)
            distances = torch.sqrt(torch.clamp(distances, min=0.0))
            return distances
            
        elif self.distance_type == 'cosine':
            # 计算余弦相似度
            query_features = F.normalize(query_features, p=2, dim=-1)
            prototypes = F.normalize(prototypes, p=2, dim=-1)
            similarities = torch.bmm(query_features, prototypes.transpose(1, 2))
            return 1 - similarities
            
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor
    ) -> torch.Tensor:
        """前向传播，支持batch处理
        
        参数:
            support_images: support set图像 [batch_size, n_way * n_support, channels, length]
            support_labels: support set标签 [batch_size, n_way * n_support]
            query_images: query set图像 [batch_size, n_way * n_query, channels, length]
            
        返回:
            logits: 预测的类别概率 [batch_size, n_query, n_way]
        """
        batch_size = support_images.size(0)
        n_support = support_images.size(1)
        n_query = query_images.size(1)
        
        # 重新整理维度用于特征提取
        support_images = support_images.reshape(batch_size * n_support, *support_images.shape[2:])
        query_images = query_images.reshape(batch_size * n_query, *query_images.shape[2:])
        
        # 1. 提取特征
        support_features = self.encoder(support_images)
        query_features = self.encoder(query_images)
        
        # 恢复batch维度
        support_features = support_features.view(batch_size, n_support, -1)
        query_features = query_features.view(batch_size, n_query, -1)
        
        if self.distance_type == 'relation':
            # 2. 直接计算关系分数
            logits = self._compute_distances(query_features, support_features)
        else:
            # 2. 计算原型
            prototypes = self._compute_prototypes(support_features, support_labels)
            
            # 3. 计算距离
            distances = self._compute_distances(query_features, prototypes)
            
            # 4. 计算logits
            logits = -distances
        
        return logits
