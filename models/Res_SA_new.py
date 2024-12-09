# 导入必要的包
import h5py
import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# DataLoader
class h5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            return len(f['data'])

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            data = f['data'][index]
            label = f['labels'][index]
            label = torch.argmax(torch.tensor(label, dtype=torch.float32))# 将8位独热编码转换为数字
            return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 定义CNN1D模型
class CNN1D(nn.Module):
    def __init__(self, input_channels=5, sequence_length=5000, num_classes=8):
        super(CNN1D, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 计算全连接层的输入维度
        self.fc_input_dim = 256 * (sequence_length // 8)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        return x
    











class SelfAttention(nn.Module):
    '''
    自注意力机制原理:
    1. 将输入序列通过线性变换得到Q(查询),K(键值),V(数值)三个矩阵
    2. Q和K做点积得到注意力分数,表示每个位置对其他位置的关注度
    3. 注意力分数经过softmax归一化得到注意力权重
    4. 注意力权重与V相乘得到加权后的特征表示
    5. 最后通过前馈网络进一步处理特征
    '''
    def __init__(self, embed_size, heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # 嵌入维度
        self.heads = heads  # 注意力头数
        self.head_dim = embed_size // heads  # 每个头的维度
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"
        
        # 自注意力层的线性变换
        self.values = nn.Linear(embed_size, embed_size, bias=False)  # 值矩阵变换
        self.keys = nn.Linear(embed_size, embed_size, bias=False)    # 键矩阵变换
        self.queries = nn.Linear(embed_size, embed_size, bias=False) # 查询矩阵变换
        self.fc_out = nn.Linear(embed_size, embed_size)  # 输出特征变换
        
        # 前馈网络,用于进一步处理注意力输出
        self.ff_fc1 = nn.Linear(embed_size, embed_size * 4)  # 第一个全连接层扩展维度
        self.ff_fc2 = nn.Linear(embed_size * 4, embed_size)  # 第二个全连接层还原维度
        
        self.dropout = nn.Dropout(dropout)  # dropout防止过拟合
        
    def forward(self, x):
        # 自注意力计算
        N = x.shape[0]  # 批次大小
        length = x.shape[1]  # 序列长度
        
        # 生成Q,K,V矩阵并重塑维度为(batch, length, heads, head_dim)
        values = self.values(x).view(N, length, self.heads, self.head_dim)
        keys = self.keys(x).view(N, length, self.heads, self.head_dim)
        queries = self.queries(x).view(N, length, self.heads, self.head_dim)

        # 调整维度顺序为(batch, heads, length, head_dim)
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        # 计算注意力分数
        energy = torch.einsum("nqhd,nkhd->nqk", [queries, keys])  # Q和K的点积
        # 缩放并softmax归一化得到注意力权重
        attention = self.dropout(F.softmax(energy / (self.embed_size ** (1 / 2)), dim=2))

        # 注意力权重与V相乘得到输出
        attention_out = torch.einsum("nqk,nvhd->nqhd", [attention, values]).reshape(
            N, length, self.heads * self.head_dim
        )
        attention_out = self.fc_out(attention_out)  # 线性变换
        
        # 前馈网络处理
        ff_out = self.ff_fc2(self.dropout(F.relu(self.ff_fc1(attention_out))))
        
        return ff_out
    
class ResidualConv2DBlock(nn.Module):
    """
    残差卷积块,包含两个卷积层和残差连接
    原理:通过残差连接缓解深度网络的梯度消失问题,同时提升特征提取能力
    """
    def __init__(self, in_channels, out_channels, kernel_size=(1,5), stride=1, padding=(0,2),dropout=0.1):
        super(ResidualConv2DBlock, self).__init__()
        # 如果未指定padding,则根据kernel_size自动计算
        if padding is None:
            padding = kernel_size[0] // 2, kernel_size[1] // 2
            
        # 第一个卷积层,用于特征变换
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # 第二个卷积层,进一步提取特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        
        # 第一个BN-ReLU-Dropout序列
        # BatchNorm用于归一化,ReLU用于非线性变换,Dropout防止过拟合
        self.bnRelu1 = nn.Sequential(nn.BatchNorm2d(out_channels),
                                     nn.ReLU(),
                                     nn.Dropout(dropout))
        
        # 第二个BN-ReLU-Dropout序列                             
        self.bnRelu2 = nn.Sequential(nn.BatchNorm2d(out_channels),
                                     nn.ReLU(),
                                     nn.Dropout(dropout))
                                     
        # shortcut连接,用于将输入特征图变换为相同维度
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # 保存输入用于残差连接
        identity = x
        
        # 第一个卷积+BN+ReLU+Dropout
        out = self.conv1(x)
        out = self.bnRelu1(out)
        
        # shortcut分支处理
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        # 第一个残差连接
        out += identity
        
        # 保存中间结果用于第二个残差连接
        identity = out
        # 第二个卷积+BN+ReLU+Dropout
        out = self.conv2(out)
        out = self.bnRelu2(out)
        # 第二个残差连接
        out += identity
        
        return out

class SABlock(nn.Module):
    """自注意力模块
    通过自注意力机制捕获序列中的长距离依赖关系。
    主要包含以下步骤:
    1. 输入投影 - 将输入特征投影到注意力空间
    2. 多头自注意力 - 并行计算多个注意力头,增强特征提取能力 
    3. 残差连接和层归一化 - 缓解梯度消失问题并加速训练
    4. 输出投影 - 将特征映射回原始维度
    """
    def __init__(self, input_channels, sequence_length=1001, embed_size=256, num_heads=8, dropout=0.1):
        super(SABlock,self).__init__()
        
        self.embed_size = embed_size
        
        # 输入投影层:使用1x1卷积将输入通道数映射到embed_size维度
        # 相比全连接层,1x1卷积可以保持空间维度信息
        self.projection = nn.Sequential(
            nn.Conv1d(input_channels, embed_size, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout)  # dropout防止过拟合
        )
        
        # 多头自注意力层:并行计算多个注意力头,增强特征提取能力
        self.attention = SelfAttention(embed_size=embed_size, heads=num_heads, dropout=dropout)
        
        # 层归一化:归一化特征分布,加速训练收敛
        self.layer_norm1 = nn.LayerNorm([sequence_length, embed_size])
        self.layer_norm2 = nn.LayerNorm([sequence_length, embed_size])
        self.dropout = nn.Dropout(dropout)
        
        # 输出投影层:将特征映射回原始通道数
        self.output_projection = nn.Conv1d(embed_size, input_channels, kernel_size=1)

    def forward(self, x):
        # x输入维度: (batch_size, channels, 1, sequence_length)
        
        # 去掉冗余维度
        x = x.squeeze(2)  # (batch_size, channels, sequence_length)
        
        # 输入投影
        x = self.projection(x)  # (batch_size, embed_size, sequence_length)
        
        # 调整维度顺序以适配注意力机制
        x = x.transpose(1, 2)  # (batch_size, sequence_length, embed_size)
        
        # 自注意力处理 + 残差连接 + 层归一化
        attention_out = self.layer_norm1(x + self.dropout(self.attention(x)))
        out = self.layer_norm2(attention_out + self.dropout(attention_out))
        
        # 恢复原始维度顺序
        out = out.transpose(1, 2)  # (batch_size, embed_size, sequence_length)
        
        # 输出投影,恢复通道数
        out = self.output_projection(out)  # (batch_size, channels, sequence_length)
        
        # 添加维度以匹配输入格式
        out = out.unsqueeze(2)  # (batch_size, channels, 1, sequence_length)
        
        return out
class Res_SA(nn.Module):
    """
    Res-SA模型: 结合残差连接和自注意力机制的深度神经网络
    主要由以下模块组成:
    1. 初始卷积层: 提取基础特征
    2. 交替的残差卷积块和自注意力块: 逐步提取更高层特征,同时保持梯度流动
    3. 全局平均池化和全连接层: 特征整合与分类
    """
    def __init__(self, dropout, input_channels=4, sequence_length=1001, embed_size=256, num_heads=8, num_classes=16):
        super(Res_SA,self).__init__()
        # 初始卷积层:将输入信号转换为初始特征图
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(1,5), stride=1, padding=(0,2))
        
        # 自注意力块:在不同特征维度上捕获长程依赖关系
        self.sa_block1 = SABlock(input_channels=64, sequence_length=1001, embed_size=256, num_heads=8, dropout=dropout)
        self.sa_block2 = SABlock(input_channels=128, sequence_length=1001, embed_size=256, num_heads=8, dropout=dropout)
        self.sa_block3 = SABlock(input_channels=256, sequence_length=1001, embed_size=256, num_heads=8, dropout=dropout)
        
        # 残差卷积块:通过残差连接保持梯度流动,逐步增加特征通道数
        self.residual_conv2d_block1 = ResidualConv2DBlock(in_channels=64, out_channels=64, kernel_size=(1,5), stride=1, padding=(0,2),dropout=dropout)
        self.residual_conv2d_block2 = ResidualConv2DBlock(in_channels=64, out_channels=128, kernel_size=(1,5), stride=1, padding=(0,2),dropout=dropout)
        self.residual_conv2d_block3 = ResidualConv2DBlock(in_channels=128, out_channels=256, kernel_size=(1,5), stride=1, padding=(0,2),dropout=dropout)
        self.residual_conv2d_block4 = ResidualConv2DBlock(in_channels=256, out_channels=512, kernel_size=(1,5), stride=1, padding=(0,2),dropout=dropout)
        
        # 全局平均池化:压缩空间维度,保留通道信息
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # 分类头:将特征映射到类别空间
        self.output = nn.Sequential(
            nn.Linear(512, 128),  # 降维
            nn.ReLU(),  # 非线性激活
            nn.Dropout(dropout),  # 防止过拟合
            nn.Linear(128, num_classes)  # 最终分类
        )
        
    def forward(self, x):
        print(x.shape)
        # 添加维度以适配2D卷积
        x = x.unsqueeze(2)
        
        # 特征提取阶段
        x = self.conv1(x)  # 初始特征提取
        x = self.residual_conv2d_block1(x)  # 第一个残差块处理
        x = self.sa_block1(x)  # 第一个自注意力处理
        x = self.residual_conv2d_block2(x)  # 第二个残差块处理
        x = self.sa_block2(x)  # 第二个自注意力处理
        x = self.residual_conv2d_block3(x)  # 第三个残差块处理
        x = self.sa_block3(x)  # 第三个自注意力处理
        x = self.residual_conv2d_block4(x)  # 最后的残差块处理
        
        # 特征整合阶段
        x = self.avgpool(x)  # 全局特征池化
        print(x.shape)
        x = x.view(x.size(0), -1)  # 展平特征
        
        # 分类阶段
        x = self.output(x)  # 通过分类头得到最终预测
        return x
    
