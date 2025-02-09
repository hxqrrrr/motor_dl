import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupConvNet(nn.Module):
    def __init__(self, num_groups=3):
        super(GroupConvNet, self).__init__()
        
        # 第一层：组卷积，输入通道为3，输出通道为12，卷积核大小为3x3，分成2组
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, groups=num_groups, padding=1)
        # 第二层：普通卷积，输入通道为12，输出通道为24，卷积核大小为3x3
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
        # 第三层：普通卷积，输入通道为24，输出通道为48，卷积核大小为3x3
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1)
        # 全连接层
        self.fc = nn.Linear(48 * 32 * 32, 10)  # 假设输入图像尺寸为 32x32

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层
        x = F.relu(self.conv2(x))  # 第二层
        x = F.relu(self.conv3(x))  # 第三层
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)  # 全连接层
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups  # 组数

    def forward(self, x):
        # 获取输入的尺寸 (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()

        # 如果通道数不能被组数整除，抛出异常
        assert channels % self.groups == 0, "Channels must be divisible by groups"

        # 计算每组的通道数
        group_channels = channels // self.groups

        # 将输入通道分成多个组并重新排列
        # 1. 按照组的顺序将通道拆分成多个部分
        x = x.view(batch_size, self.groups, group_channels, height, width)
        # 2. 重新排列通道顺序，使得不同组之间的信息可以流动
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # 3. 将分组后的张量展平回原始形状
        x = x.view(batch_size, channels, height, width)
        return x

class ShuffleNet(nn.Module):
    def __init__(self, num_groups=2):
        super(ShuffleNet, self).__init__()

        # 第一层：普通卷积
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1)
        # 第二层：组卷积 + 通道混洗
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1, groups=2)
        self.channel_shuffle = ChannelShuffle(groups=2)  # 通道混洗

        # 第三层：普通卷积
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=2, padding=1)

        # 全连接层
        self.fc = nn.Linear(96 * 8 * 8, 10)  # 假设输入图像尺寸为32x32

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层
        x = F.relu(self.conv2(x))  # 第二层，组卷积
        x = self.channel_shuffle(x)  # 通道混洗
        x = F.relu(self.conv3(x))  # 第三层
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)  # 全连接层
        return x

# 模型实例化
model = ShuffleNet(num_groups=3)  # 设置组数为3

# 输入数据：batch_size=8, 输入图像尺寸为32x32, 输入通道为3
input_data = torch.randn(8, 3, 32, 32)

# 前向传播
output = model(input_data)

print(output.shape)  # 输出形状 (8, 10)，假设是分类任务
