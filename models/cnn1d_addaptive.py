import torch
import torch.nn as nn

class AdaptiveCNN1D(nn.Module):
    def __init__(self, input_channels, num_classes, conv_filters=None, kernel_size=5, pool_kernel_size=2, dropout_rate=0.3):
        """
        自适应输入长度的 1D CNN 模型

        Args:
            input_channels (int): 输入信号的通道数
            num_classes (int): 分类类别数
            conv_filters (list of int): 卷积层滤波器数量列表，默认为 [64, 128, 256]
            kernel_size (int): 卷积核大小，默认为 5
            pool_kernel_size (int): 池化层核大小，默认为 2
            dropout_rate (float): Dropout 率，默认为 0.3
        """
        super(AdaptiveCNN1D, self).__init__()
        if conv_filters is None:
            conv_filters = [64, 128, 256]
        self.conv_layers = nn.ModuleList() # 使用 ModuleList 存储卷积层

        # 构建卷积层 (根据 conv_filters 列表动态构建)
        in_channels = input_channels
        for out_channels in conv_filters:
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'), # padding='same' 保证输出长度不变
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_kernel_size)
            )
            self.conv_layers.append(conv_layer) # 添加到 ModuleList
            in_channels = out_channels # 更新下一层的输入通道数

        self.dropout = nn.Dropout(dropout_rate)

        # **全局平均池化层 (Global Average Pooling)**：自适应不同输入长度的关键
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1) # 输出维度固定为 1

        # 全连接层 (分类头)
        self.fc1 = nn.Linear(conv_filters[-1], conv_filters[-1] // 2) # 输入维度为最后一个卷积层的通道数
        self.fc2 = nn.Linear(conv_filters[-1]//2, num_classes)

    def forward(self, x):
        """
        模型前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_channels, sequence_length)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, num_classes)，表示每个类别的预测 logits
        """
        # 1. 卷积层 + 池化层
        for conv_layer in self.conv_layers:
            x = conv_layer(x) # 逐层通过卷积层

        # 2. Dropout
        x = self.dropout(x)

        # 3. **全局平均池化 (Global Average Pooling)**：自适应不同输入长度的关键
        x = self.global_avg_pool(x) # 输出形状: (batch_size, 通道数, 1)
        x = x.squeeze(2)         # 去掉维度为 1 的维度，形状: (batch_size, 通道数)

        # 4. 全连接层分类
        x = self.fc1(x) # 形状: (batch_size, num_classes)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # 单元测试 (示例)
    batch_size = 2
    input_channels = 1
    sequence_length = 120000  # 测试不同的序列长度
    num_classes = 12

    # 创建一个 AdaptiveCNN1D 模型的实例
    model = AdaptiveCNN1D(input_channels=input_channels,
                           num_classes=num_classes,
                           conv_filters=[64, 128, 256], # 可以自定义卷积层滤波器数量
                           kernel_size=5,
                           pool_kernel_size=2,
                           dropout_rate=0.2)

    # 创建一个随机输入张量 (测试不同序列长度)
    input_tensor = torch.randn(batch_size, input_channels, sequence_length) # 形状: (batch_size, input_channels, sequence_length)

    # 模型前向传播
    output = model(input_tensor)

    # 打印输出形状
    print("Input tensor shape:", input_tensor.shape) # 预期: torch.Size([2, 1, 1200]) (序列长度可变)
    print("Output tensor shape:", output.shape)   # 预期: torch.Size([2, 10]) (batch_size, num_classes)

    # **计算模型总参数量**
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") # 打印总参数量 (带千位分隔符)


    # 检查输出是否为 NaN (Not a Number)
    if torch.isnan(output).any():
        print("Warning: Output contains NaN values!")
    else:
        print("Output does not contain NaN values.")

    print("AdaptiveCNN1D model test finished successfully!")