import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from models.ProtoNet_backbone import AttentiveEncoder
import random
from models.relation import RelationModule, RelationModuleWithAttention, SimpleConvRelationModule


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


class EnhancedCNN1D_embed(nn.Module):
    """优化版的CNN1D_embed，结合enhance的优点和embed的元学习特性
    
    参数:
        in_channels: 输入通道数
        hidden_dim: 隐藏层维度
        feature_dim: 输出特征维度
        use_l2_norm: 是否使用L2归一化
        dropout_rate: Dropout比率
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        use_l2_norm: bool = True,
        dropout_rate: float = 0.3
    ):
        super(EnhancedCNN1D_embed, self).__init__()
        
        # 输入标准化层
        self.norm = nn.BatchNorm1d(in_channels)
        
        # 第一个卷积块 - 16通道，无BN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第二个卷积块 - 32通道，有BN
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第三个卷积块 - 64通道，有BN
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第四个卷积块 - 64通道，有BN
        self.conv4 = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第五个卷积块 - 64通道，有BN，无池化
        self.conv5 = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU()
        )
        
        # 全局自适应平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征映射层
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.use_l2_norm = use_l2_norm
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用输入标准化
        x = self.norm(x)
        
        # 应用卷积块
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # 全局池化
        x = self.global_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 特征映射
        x = self.feature_layer(x)
        
        # 条件L2归一化
        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=1)
        
        return x


class AllModel(nn.Module):
    """基于注意力机制的关系网络版本的Prototypical Network
    
    参数:
        in_channels (int): 输入数据的通道数
        hidden_dim (int): 嵌入网络的隐藏层维度
        feature_dim (int): 最终特征向量的维度
        backbone (str): 特征提取器的类型 ('cnn1d', 'cnn2d', 'lstm', 'enhanced_cnn1d')
        distance_type (str): 距离度量方式 ('euclidean', 'cosine', 'relation', 'relation_selfattention')
        dropout (float): Dropout比率
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        backbone: str = 'cnn1d',
        distance_type: str = 'euclidean',
        dropout: float = 0.2
    ):
        """初始化模型
        
        参数:
            in_channels: 输入通道数
            hidden_dim: 隐藏层维度
            feature_dim: 特征维度
            backbone: 骨干网络类型，可选['cnn1d', 'channel', 'spatial', 'cbam', 'enhanced_cnn1d']
            distance_type: 距离度量类型，可选['euclidean', 'cosine', 'relation', 'relation_selfattention']
            dropout: Dropout比率
        """
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
            self.relation_module = SimpleConvRelationModule(feature_dim, hidden_dim, dropout)
        elif distance_type == 'relation_selfattention':
            self.relation_module = RelationModuleWithAttention(feature_dim, hidden_dim, dropout)
    
    def _build_encoder(self) -> nn.Module:
        """构建特征提取器"""
        if self.backbone == 'cnn1d':
            return CNN1D_embed(
                self.in_channels, 
                self.hidden_dim, 
                self.feature_dim,
                self.dropout
            )
        elif self.backbone == 'enhanced_cnn1d':
            return EnhancedCNN1D_embed(
                self.in_channels,
                self.hidden_dim,
                self.feature_dim,
                use_l2_norm=True,
                dropout_rate=self.dropout
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
        """计算每个类别的原型，支持batch处理"""
        batch_size = support_features.size(0)
        feature_dim = support_features.size(-1)
        device = support_features.device
        
        # 初始化原型tensor
        prototypes = []
        
        # 对每个batch分别计算原型
        for i in range(batch_size):
            batch_features = support_features[i]  # [n_support, feature_dim]
            batch_labels = support_labels[i]      # [n_support]
            
            # 获取当前batch的类别
            classes = torch.unique(batch_labels)
            
            # 检查类别数量
            n_way = len(classes)
            if n_way == 0:
                raise ValueError("当前批次没有有效的类别，无法计算原型。")
            
            # 初始化当前batch的原型
            current_prototypes = torch.zeros(n_way, feature_dim, device=device)
            
            # 对每个类别计算原型
            for j, cls in enumerate(classes):
                mask = batch_labels == cls
                class_features = batch_features[mask]
                current_prototypes[j] = torch.mean(class_features, dim=0)
            prototypes.append(current_prototypes)
        
        return torch.stack(prototypes)  # 返回 [batch_size, n_way, feature_dim]
        
    def _compute_distances(self, query_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """计算查询样本与原型之间的距离或关系分数
        
        参数:
            query_features: query set的特征向量 [batch_size, n_query, feature_dim]
            prototypes: 类别原型向量 [batch_size, n_way, feature_dim]
            
        返回:
            distances/scores: 距离矩阵或关系分数 [batch_size, n_query, n_way]
        """
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
        support_t: torch.Tensor,
        support_labels: torch.Tensor,
        query_t: torch.Tensor
    ) -> torch.Tensor:
        """前向传播，支持batch处理
        
        参数:
            support_t: 支持集数据 [batch_size, n_way * n_support, length]
            support_labels: 支持集标签 [batch_size, n_way * n_support]
            query_t: 查询集数据 [batch_size, n_way * n_query, length]
            
        返回:
            logits: 预测的类别概率 [batch_size, n_query, n_way]
        """
        batch_size = support_t.size(0)
        n_support = support_t.size(1)
        n_query = query_t.size(1)
        
        # 重新整理维度用于特征提取
        support_t = support_t.reshape(batch_size * n_support, *support_t.shape[2:])
        query_t = query_t.reshape(batch_size * n_query, *query_t.shape[2:])
        
        # 1. 提取特征
        support_features = self.encoder(support_t)
        query_features = self.encoder(query_t)
        
        # 恢复batch维度
        support_features = support_features.view(batch_size, n_support, -1)
        query_features = query_features.view(batch_size, n_query, -1)
        
        # 2. 计算原型 (对所有距离类型都计算)
        prototypes = self._compute_prototypes(support_features, support_labels)
        
        if self.distance_type in ['relation', 'relation_selfattention']:
            # 3. 使用关系网络计算查询样本与原型的关系分数
            logits = self.relation_module(query_features, prototypes)
        else:
            # 3. 计算距离
            distances = self._compute_distances(query_features, prototypes)
            
            # 4. 计算logits
            logits = -distances
        
        return logits

