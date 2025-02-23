

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from models.cnn1d import CNN1D_embed



class ProtoNet(nn.Module):
    """Prototypical Network模型，遵循原始论文架构
    
    参数:
        in_channels (int): 输入数据的通道数
        hidden_dim (int): 嵌入网络的隐藏层维度
        feature_dim (int): 最终特征向量的维度
        backbone (str): 特征提取器的类型 ('cnn1d', 'cnn2d', 'lstm')
        distance_type (str): 距离度量方式 ('euclidean', 'cosine')
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        backbone: str = 'cnn1d',
        distance_type: str = 'euclidean'
    ):
        super(ProtoNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.backbone = backbone
        self.distance_type = distance_type
        
        # 初始化嵌入网络
        self.encoder = self._build_encoder()
        
    def _build_encoder(self) -> nn.Module:
        """构建特征提取器"""
        if self.backbone == 'cnn1d':
            return CNN1D_embed(
                self.in_channels, 
                self.hidden_dim, 
                self.feature_dim
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
        """计算查询样本与原型之间的距离，支持batch处理
        
        参数:
            query_features: query set的特征向量 [batch_size, n_query, feature_dim]
            prototypes: 类别原型向量 [batch_size, n_way, feature_dim]
            
        返回:
            distances: 距离矩阵 [batch_size, n_query, n_way]
        """
        batch_size = query_features.size(0)
        n_query = query_features.size(1)
        n_way = prototypes.size(1)
        
        if self.distance_type == 'euclidean':
            # 计算欧氏距离
            # 重塑张量以便广播
            query_features = query_features.unsqueeze(2)  # [batch_size, n_query, 1, feature_dim]
            prototypes = prototypes.unsqueeze(1)         # [batch_size, 1, n_way, feature_dim]
            
            # 计算欧氏距离的平方
            distances = torch.sum((query_features - prototypes) ** 2, dim=-1)  # [batch_size, n_query, n_way]
            
            # 开根号得到真实距离
            distances = torch.sqrt(torch.clamp(distances, min=0.0))
            
        elif self.distance_type == 'cosine':
            # 计算余弦相似度
            # 首先对特征进行L2归一化
            query_features = F.normalize(query_features, p=2, dim=-1)  # [batch_size, n_query, feature_dim]
            prototypes = F.normalize(prototypes, p=2, dim=-1)         # [batch_size, n_way, feature_dim]
            
            # 计算余弦相似度
            similarities = torch.bmm(query_features, prototypes.transpose(1, 2))  # [batch_size, n_query, n_way]
            
            # 将相似度转换为距离：distance = 1 - similarity
            distances = 1 - similarities
            
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
            
        return distances
    
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
        # 1. 首先把所有样本展平到一个批次中
        support_images = support_images.reshape(batch_size * n_support, *support_images.shape[2:])
 
        query_images = query_images.reshape(batch_size * n_query, *query_images.shape[2:])
        
        
        # 1. 提取特征
        support_features = self.encoder(support_images)  # [batch_size * n_support, feature_dim]
        query_features = self.encoder(query_images)      # [batch_size * n_query, feature_dim]
        
        # 恢复batch维度
        support_features = support_features.view(batch_size, n_support, -1)  # [batch_size, n_support, feature_dim]
        query_features = query_features.view(batch_size, n_query, -1)       # [batch_size, n_query, feature_dim]
        
        # 2. 计算原型
        prototypes = self._compute_prototypes(support_features, support_labels)  # [batch_size, n_way, feature_dim]
        
        # 3. 计算距离
        distances = self._compute_distances(query_features, prototypes)  # [batch_size, n_query, n_way]
        
        # 4. 计算logits并保持batch维度结构
        if self.distance_type == 'euclidean':
            logits = -distances  # 距离越小，相似度越高
        else:  # cosine
            logits = -distances   # 余弦距离越小，相似度越高
            
        # 保持batch维度结构 [batch_size, n_query, n_way]
        return logits
