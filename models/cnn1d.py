# 导入必要的包
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
# 定义CNN1D模型class CNN1D(nn.Module):

class CNN1D(nn.Module):
    def __init__(self, input_channels=1, sequence_length=12000, num_classes=12):
        super(CNN1D, self).__init__()
        self.sequence_length = sequence_length
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
            nn.Linear(self.fc_input_dim, 256),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        # 标准化
        self.norm = nn.BatchNorm1d(input_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 展平
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    def forward(self, x):
        batch_size, num_channels, length = x.size()
        channels_per_group = num_channels // self.groups
        # 将通道维度分组
        x = x.view(batch_size, self.groups, channels_per_group, length)
        # 交换组的顺序
        x = x.permute(0, 2, 1, 3).contiguous()
        # 展平回原始形状
        x = x.view(batch_size, num_channels, length)
        return x
    
class CNN1D_shuffle(nn.Module):
    def __init__(self, input_channels=1, sequence_length=12000, num_classes=10, groups=2):
        super(CNN1D_shuffle, self).__init__()
        self.sequence_length = sequence_length
        self.groups = groups

        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, groups=self.groups),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, groups=self.groups),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 通道混洗
        self.channel_shuffle = ChannelShuffle(groups=self.groups)
        # 计算全连接层的输入维度
        self.fc_input_dim = 256 * (sequence_length // 8)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.channel_shuffle(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN1D_KAT(nn.Module):
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
        # 连接层
        x = self.fc(x)
        return x