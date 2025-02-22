# 导入必要的包
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import time

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
        # 添加第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1, groups=self.groups),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 通道混洗
        self.channel_shuffle = ChannelShuffle(groups=self.groups)
        # 计算全连接层的输入维度
        self.fc_input_dim = 512 * (sequence_length // 16)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        residual = x  # 残差连接
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.channel_shuffle(x)
        print(x.shape)
        x += residual  # 添加残差
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

class ShuffleNet1DUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(ShuffleNet1DUnit, self).__init__()
        self.groups = groups
        mid_channels = out_channels // 4

        # 1x1 Grouped Convolution
        self.gconv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, groups=self.groups)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # Channel Shuffle
        self.shuffle = ChannelShuffle(groups=self.groups)

        # 3x3 Depthwise Convolution
        self.dwconv = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        self.bn2 = nn.BatchNorm1d(mid_channels)

        # 1x1 Grouped Convolution
        self.gconv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, groups=self.groups)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # Residual Connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x, return_times=False):
        times = {}
        
        start_time = time.time()
        residual = self.shortcut(x)
        times['shortcut'] = time.time() - start_time

        start_time = time.time()
        x = self.gconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        times['gconv1_bn1_relu'] = time.time() - start_time

        start_time = time.time()
        x = self.shuffle(x)
        times['shuffle'] = time.time() - start_time

        start_time = time.time()
        x = self.dwconv(x)
        x = self.bn2(x)
        times['dwconv_bn2'] = time.time() - start_time

        start_time = time.time()
        x = self.gconv2(x)
        x = self.bn3(x)
        times['gconv2_bn3'] = time.time() - start_time

        start_time = time.time()
        x += residual
        times['residual_addition'] = time.time() - start_time

        if return_times:
            return x, times
        return x

class ShuffleNet1DClassifier(nn.Module):
    def __init__(self, input_channels, num_classes, groups=1):
        super(ShuffleNet1DClassifier, self).__init__()
        self.unit = ShuffleNet1DUnit(input_channels, 64, groups)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, return_times=False):
        times = {}
        
        # ShuffleNet1DUnit forward
        if return_times:
            x, unit_times = self.unit(x, return_times=True)
            times.update(unit_times)
        else:
            x = self.unit(x)

        # Global pooling
        start_time = time.time()
        x = self.global_pool(x)
        times['global_pool'] = time.time() - start_time

        # Reshape
        start_time = time.time()
        x = x.view(x.size(0), -1)
        times['view'] = time.time() - start_time

        # Fully connected
        start_time = time.time()
        x = self.fc(x)
        times['fc'] = time.time() - start_time

        if return_times:
            return x, times
        return x

class CNN1D_embed(nn.Module):
    """用于Prototypical Network的1D-CNN嵌入网络
    
    参数:
        in_channels: 输入通道数
        hidden_dim: 隐藏层维度
        feature_dim: 输出特征维度
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
    ):
        super(CNN1D_embed, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(2)
        )
        
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2)
        )

        # 添加自适应平均池化层
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征映射层
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
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
