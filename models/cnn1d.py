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
        self.fc_input_dim = 256 * (sequence_length // 8)
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
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim,256),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        # 标准化
        self.norm = nn.BatchNorm1d(input_channels) 



    def forward(self, x):
        # 卷积层
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 展平
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