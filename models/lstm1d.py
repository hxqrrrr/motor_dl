import torch
import torch.nn as nn

class LSTM1D(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes, hidden_size=128, num_layers=3, dropout=0.3):
        """
        1D LSTM 模型 (展平所有时间步输出)

        Args:
            input_channels (int): 输入信号的通道数 (例如，对于单通道时间序列数据，input_channels=1)
            sequence_length (int): 输入序列的长度 (时间步数)
            num_classes (int): 分类类别数
            hidden_size (int): LSTM 隐藏层维度，默认为 64
            num_layers (int): LSTM 层数，默认为 2
            dropout (float): LSTM 和 Dropout 层的 Dropout 率，默认为 0.3
        """
        super(LSTM1D, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. 输入线性层：将输入通道数映射到 LSTM 的输入维度 (hidden_size)
        self.input_linear = nn.Linear(input_channels, hidden_size)

        # 2. LSTM 层
        self.lstm = nn.LSTM(input_size=hidden_size,  # LSTM 输入维度
                            hidden_size=hidden_size, # LSTM 隐藏层维度
                            num_layers=num_layers,  # LSTM 层数
                            batch_first=True,       # 输入数据的第一维是 batch size (batch_size, sequence_length, input_size)
                            dropout=dropout)        # Dropout

        # 3. 全连接层 (分类头)：将 LSTM 所有时间步的输出展平后进行分类
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * sequence_length, 256), # LSTM 输出展平后的维度: hidden_size * sequence_length
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        self.dropout = nn.Dropout(dropout) # Dropout 层 (可选，可以添加到全连接层之前或之后)


    def forward(self, x):
        """
        模型前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_channels, sequence_length)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, num_classes)，表示每个类别的预测 logits
        """
        # 1. 调整输入形状：(batch_size, input_channels, sequence_length) -> (batch_size, sequence_length, input_channels)
        x = x.transpose(1, 2)

        # 2. 线性变换：将输入通道数映射到 LSTM 的输入维度
        x = self.input_linear(x)

        # 3. **确保输入张量 x 是连续的 (contiguous)**，并转换为 float32 类型 (可选，但推荐)
        x = x.contiguous().float()

        # 4. LSTM 前向传播
        out, _ = self.lstm(x) # out 的形状: (batch_size, sequence_length, hidden_size)
        # _ 包含 LSTM 的 hidden state 和 cell state，这里我们通常不需要使用

        # 5. **展平 LSTM 的输出**：将 LSTM 所有时间步的输出展平成一个二维张量
        x = out.reshape(out.size(0), -1) # 形状: (batch_size, sequence_length * hidden_size)

        # 6. Dropout (可选，可以在全连接层之前添加)
        x = self.dropout(x) # 应用 Dropout

        # 7. 全连接层分类 (使用展平后的 LSTM 输出)
        x = self.fc(x) # 形状: (batch_size, num_classes)
        return x


if __name__ == '__main__':
    # 单元测试 (示例)
    batch_size = 2
    input_channels = 1
    sequence_length = 100
    num_classes = 10

    # 创建一个 LSTM1D 模型的实例
    model = LSTM1D(input_channels=input_channels,
                     sequence_length=sequence_length,
                     num_classes=num_classes,
                     hidden_size=32,
                     num_layers=2,
                     dropout=0.2)

    # 创建一个随机输入张量 (模拟一个 batch 的数据)
    input_tensor = torch.randn(batch_size, input_channels, sequence_length) # 形状: (batch_size, input_channels, sequence_length)

    # 模型前向传播
    output = model(input_tensor)

    # 打印输出形状
    print("Input tensor shape:", input_tensor.shape) # 预期: torch.Size([2, 1, 100])
    print("Output tensor shape:", output.shape)   # 预期: torch.Size([2, 10]) (batch_size, num_classes)

    # 检查输出是否为 NaN (Not a Number)
    if torch.isnan(output).any():
        print("Warning: Output contains NaN values!")
    else:
        print("Output does not contain NaN values.")

    print("LSTM1D model test finished successfully!")