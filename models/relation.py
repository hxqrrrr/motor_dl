import torch
import torch.nn as nn
import torch.nn.functional as F



class RelationModule(nn.Module):
    """简单的关系网络模块
    
    参数:
        feature_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        dropout: Dropout比率
    """
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.2):
        super(RelationModule, self).__init__()
        
        # 增强版关系网络
        self.relation_net = nn.Sequential(
            # 第一层
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            # 第二层
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            # 第三层
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout * 0.5),
            
            # 输出层
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 将关系分数归一化到[0,1]区间
        )
        
    def forward(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        """
        参数:
            query_features: [batch_size, n_query, feature_dim]
            support_features: [batch_size, n_way, feature_dim] - 现在是原型，而不是所有支持样本
            
        返回:
            relation_scores: [batch_size, n_query, n_way]
        """
        batch_size = query_features.size(0)
        n_query = query_features.size(1)
        n_way = support_features.size(1)  # 现在是类别数量
        
        # 特征归一化，提高稳定性
        query_features = F.normalize(query_features, p=2, dim=-1)
        support_features = F.normalize(support_features, p=2, dim=-1)
        
        # 准备特征对
        q_expanded = query_features.unsqueeze(2).expand(-1, -1, n_way, -1)    # [batch_size, n_query, n_way, feature_dim]
        s_expanded = support_features.unsqueeze(1).expand(-1, n_query, -1, -1)    # [batch_size, n_query, n_way, feature_dim]
        
        # 连接特征
        paired_features = torch.cat([q_expanded, s_expanded], dim=-1)  # [batch_size, n_query, n_way, feature_dim*2]
        
        # 重塑tensor以便输入关系网络
        paired_features = paired_features.view(-1, paired_features.size(-1))  # [batch_size*n_query*n_way, feature_dim*2]
        
        # 计算关系分数
        relation_scores = self.relation_net(paired_features)  # [batch_size*n_query*n_way, 1]
        
        # 重塑回原始维度
        relation_scores = relation_scores.view(batch_size, n_query, n_way)  # [batch_size, n_query, n_way]
        
        return relation_scores
class RelationModuleWithAttention(nn.Module):
    """基于注意力机制的关系网络模块
    
    参数:
        feature_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        dropout: Dropout比率
    """
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.2):
        super(RelationModuleWithAttention, self).__init__()
        
        # 自注意力机制
        self.query_transform = nn.Linear(feature_dim, hidden_dim)
        self.key_transform = nn.Linear(feature_dim, hidden_dim)
        self.value_transform = nn.Linear(feature_dim, hidden_dim)
        
        # 缩放因子
        self.scale_factor = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        
        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # 关系网络
        self.relation_net = nn.Sequential(
            # 第一层
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            # 第二层
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout * 0.5),
            
            # 输出层
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 将关系分数归一化到[0,1]区间
        )
        
    def apply_attention(self, query_features, support_features):
        """应用自注意力机制
        
        参数:
            query_features: [batch_size, n_query, feature_dim]
            support_features: [batch_size, n_way, feature_dim]
            
        返回:
            attended_query: [batch_size, n_query, hidden_dim]
            attended_support: [batch_size, n_way, hidden_dim]
        """
        batch_size = query_features.size(0)
        n_query = query_features.size(1)
        n_way = support_features.size(1)
        
        # 转换查询特征
        q_query = self.query_transform(query_features)  # [batch_size, n_query, hidden_dim]
        k_query = self.key_transform(query_features)    # [batch_size, n_query, hidden_dim]
        v_query = self.value_transform(query_features)  # [batch_size, n_query, hidden_dim]
        
        # 计算查询特征的自注意力
        attention_scores_query = torch.bmm(q_query, k_query.transpose(1, 2))  # [batch_size, n_query, n_query]
        attention_scores_query = attention_scores_query / self.scale_factor
        attention_weights_query = F.softmax(attention_scores_query, dim=-1)   # [batch_size, n_query, n_query]
        attended_query = torch.bmm(attention_weights_query, v_query)          # [batch_size, n_query, hidden_dim]
        
        # 转换支持特征
        q_support = self.query_transform(support_features)  # [batch_size, n_way, hidden_dim]
        k_support = self.key_transform(support_features)    # [batch_size, n_way, hidden_dim]
        v_support = self.value_transform(support_features)  # [batch_size, n_way, hidden_dim]
        
        # 计算支持特征的自注意力
        attention_scores_support = torch.bmm(q_support, k_support.transpose(1, 2))  # [batch_size, n_way, n_way]
        attention_scores_support = attention_scores_support / self.scale_factor
        attention_weights_support = F.softmax(attention_scores_support, dim=-1)     # [batch_size, n_way, n_way]
        attended_support = torch.bmm(attention_weights_support, v_support)          # [batch_size, n_way, hidden_dim]
        
        # 计算查询-支持交叉注意力
        cross_attention_scores = torch.bmm(q_query, k_support.transpose(1, 2))  # [batch_size, n_query, n_way]
        cross_attention_scores = cross_attention_scores / self.scale_factor
        
        return attended_query, attended_support, cross_attention_scores
        
    def forward(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        """
        参数:
            query_features: [batch_size, n_query, feature_dim]
            support_features: [batch_size, n_way, feature_dim] - 类别原型
            
        返回:
            relation_scores: [batch_size, n_query, n_way]
        """
        batch_size = query_features.size(0)
        n_query = query_features.size(1)
        n_way = support_features.size(1)
        feature_dim = query_features.size(-1)
        
        # 特征归一化，提高稳定性
        query_features = F.normalize(query_features, p=2, dim=-1)
        support_features = F.normalize(support_features, p=2, dim=-1)
        
        # 应用注意力机制
        attended_query, attended_support, cross_attention = self.apply_attention(query_features, support_features)
        
        # 准备特征对
        q_expanded = query_features.unsqueeze(2).expand(-1, -1, n_way, -1)      # [batch_size, n_query, n_way, feature_dim]
        s_expanded = support_features.unsqueeze(1).expand(-1, n_query, -1, -1)  # [batch_size, n_query, n_way, feature_dim]
        
        # 连接特征
        paired_features = torch.cat([q_expanded, s_expanded], dim=-1)  # [batch_size, n_query, n_way, feature_dim*2]
        
        # 重塑tensor以便输入特征转换层
        paired_features = paired_features.view(-1, feature_dim * 2)  # [batch_size*n_query*n_way, feature_dim*2]
        
        # 特征转换
        transformed_features = self.feature_transform(paired_features)  # [batch_size*n_query*n_way, hidden_dim]
        
        # 准备注意力特征
        attended_q = attended_query.unsqueeze(2).expand(-1, -1, n_way, -1)  # [batch_size, n_query, n_way, hidden_dim]
        attended_s = attended_support.unsqueeze(1).expand(-1, n_query, -1, -1)  # [batch_size, n_query, n_way, hidden_dim]
        
        # 重塑注意力特征
        attended_q = attended_q.reshape(-1, attended_q.size(-1))  # [batch_size*n_query*n_way, hidden_dim]
        attended_s = attended_s.reshape(-1, attended_s.size(-1))  # [batch_size*n_query*n_way, hidden_dim]
        
        # 连接转换后的特征和注意力特征
        combined_features = torch.cat([transformed_features, attended_q], dim=-1)  # [batch_size*n_query*n_way, hidden_dim*2]
        
        # 计算关系分数
        relation_scores = self.relation_net(combined_features)  # [batch_size*n_query*n_way, 1]
        
        # 重塑回原始维度
        relation_scores = relation_scores.view(batch_size, n_query, n_way)  # [batch_size, n_query, n_way]
        
        return relation_scores

class SimpleConvRelationModule(nn.Module):
    """简化版基于卷积的关系网络模块
    
    参数:
        feature_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        dropout: Dropout比率
    """
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.2):
        super(SimpleConvRelationModule, self).__init__()
        
        # 假设特征可以重塑为正方形
        self.feature_size = int(feature_dim ** 0.5)
        self.reshape_dim = self.feature_size ** 2
        
        # 如果特征维度不是完全平方数，使用线性层调整
        if self.reshape_dim != feature_dim:
            self.feature_adapter = nn.Linear(feature_dim, self.reshape_dim)
        else:
            self.feature_adapter = None
        
        # 简化的卷积关系网络 - 只有两层卷积
        self.relation_net = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第二层卷积
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 扁平化
            nn.Flatten(),
            
            # 全连接层
            nn.Linear(64 * self.feature_size * self.feature_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 将关系分数归一化到[0,1]区间
        )
        
    def forward(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        """
        参数:
            query_features: [batch_size, n_query, feature_dim]
            support_features: [batch_size, n_way, feature_dim] - 类别原型
            
        返回:
            relation_scores: [batch_size, n_query, n_way]
        """
        batch_size = query_features.size(0)
        n_query = query_features.size(1)
        n_way = support_features.size(1)
        
        # 特征归一化，提高稳定性
        query_features = F.normalize(query_features, p=2, dim=-1)
        support_features = F.normalize(support_features, p=2, dim=-1)
        
        # 如果需要，调整特征维度
        if self.feature_adapter is not None:
            query_features = self.feature_adapter(query_features)
            support_features = self.feature_adapter(support_features)
        
        # 准备特征对
        relation_pairs = []
        for i in range(n_way):
            for j in range(n_query):
                # 获取当前查询和支持特征
                q_feature = query_features[:, j, :]  # [batch_size, feature_dim]
                s_feature = support_features[:, i, :]  # [batch_size, feature_dim]
                
                # 重塑为图像格式
                q_feature = q_feature.view(batch_size, 1, self.feature_size, self.feature_size)
                s_feature = s_feature.view(batch_size, 1, self.feature_size, self.feature_size)
                
                # 连接特征作为通道
                paired = torch.cat([q_feature, s_feature], dim=1)  # [batch_size, 2, feature_size, feature_size]
                relation_pairs.append(paired)
        
        # 堆叠所有对
        relation_pairs = torch.stack(relation_pairs, dim=0)  # [n_way*n_query, batch_size, 2, feature_size, feature_size]
        
        # 重塑以便批量处理
        relation_pairs = relation_pairs.view(-1, 2, self.feature_size, self.feature_size)  # [n_way*n_query*batch_size, 2, feature_size, feature_size]
        
        # 计算关系分数
        relation_scores = self.relation_net(relation_pairs)  # [n_way*n_query*batch_size, 1]
        
        # 重塑回原始维度
        relation_scores = relation_scores.view(n_way, n_query, batch_size, -1)  # [n_way, n_query, batch_size, 1]
        relation_scores = relation_scores.permute(2, 1, 0, 3).squeeze(-1)  # [batch_size, n_query, n_way]
        
        return relation_scores

