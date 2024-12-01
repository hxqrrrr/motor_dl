
# 导入必要的包
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

# 设置随机种子
myseed = 6666  
# 设置随机种子以确保结果的可重复性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    # 为所有GPU设置随机种子
    torch.cuda.manual_seed_all(myseed)

# 数据集形状处理
def read_df(df,df_):
    '''
    df: 电流数据，形状为(1001,3)
    df_: 转矩数据，形状为(1001,1)
    处理后的数据集形状为(4,1001)
    '''
    df = df.iloc[:1001,1:].T
    df_=df_.iloc[:1001,1:].T
    df_append = pd.concat([df, df_], ignore_index=True, axis=0)
    return  df_append

class MyDataset(Dataset):
    '''
    定义数据集
    path: 数据集路径
    使用方法：dataset = MyDataset(path)
    即可读取同一文件夹下所有电流.csv和转矩.csv数据，并将其合并为一个数据集
    '''
    def __init__(self,path):
        super(MyDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith("电流.csv")])
        self.files_= sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith("转矩.csv")])
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self, idx):
        try:
            fname = self.files[idx]
            data = pd.read_csv(fname)
            data_ = pd.read_csv(self.files_[idx])
            
            # 数据集形状处理
            data_tensor = torch.tensor(read_df(data, data_).values, dtype=torch.float32)
                       # 检测数据集形状，维度为2且第一维值为4
            if data_tensor.ndim != 2 or data_tensor.shape[0] != 4:  # Adjust according to your needs
                print(f"Unexpected shape for {fname}: {data_tensor.shape}")
                return None, -1  # Indicate an error
    
        except Exception as e:
            print(f"Error reading file {fname}: {e}")
            return None, -1  # Handle error appropriately
    
        try:
            # 数据集标签处理，使用文件名中的数字切片作为标签
            label = int(fname.split("\\")[-1].split("_")[0])  # Use [-1] to get last part
        except Exception as e:
            print(f"Error extracting label from {fname}: {e}")
            label = -1  # Test has no label
    
        return data_tensor, label

def MyDataLoader(data_dir, batch_size, n_workers):
    """Generate dataloader"""
    dataset = MyDataset(data_dir)
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
    )
    return train_loader, valid_loader


#一维cnn模型
class OneDCNN(nn.Module):  
    '''
    一维cnn模型
    input_size: 输入数据长度
    num_classes: 类别数
    kernel_size: 卷积核大小
    为什么使用一维cnn模型：
    1. 一维cnn模型可以处理时间序列数据，而二维cnn模型只能处理图像数据。
    2. 我们的数据形状（batch_size,in_channels=4,sequence_length=1001），一维cnn模型可以处理这种数据
    3. 二维cnn要求输入数据形状为（batch_size,channels,height,width），无法处理我们的数据，需要修改形状后才能读取
    '''
    def __init__(self, input_size, num_classes, kernel_size=5):  
        super(OneDCNN, self).__init__()  
        # 第一个卷积层,输入通道4,输出通道16,卷积核大小kernel_size
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=kernel_size)  
        # 最大池化层,池化核大小为2
        self.pool = nn.MaxPool1d(kernel_size=2)  
        # 第二个卷积层,输入通道16,输出通道32,卷积核大小kernel_size
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size)  
        
        # 计算每层卷积后的输出尺寸
        # 第一次卷积后的尺寸 = 输入尺寸 - 卷积核大小 + 1
        conv1_output_size = (input_size - kernel_size + 1)  
        # 第一次池化后的尺寸 = 卷积输出尺寸 // 2
        conv1_pooled_size = conv1_output_size // 2  
        # 第二次卷积后的尺寸 = 第一次池化尺寸 - 卷积核大小 + 1
        conv2_output_size = (conv1_pooled_size - kernel_size + 1)  
        # 第二次池化后的尺寸 = 第二次卷积尺寸 // 2
        conv2_pooled_size = conv2_output_size // 2  
        
        # 全连接层
        # 第一个全连接层,输入维度为32*conv2_pooled_size,输出维度128
        self.fc1 = nn.Linear(32 * conv2_pooled_size, 128)  
        # 第二个全连接层,输入维度128,输出维度为类别数
        self.fc2 = nn.Linear(128, num_classes)  

    def forward(self, x):  
        # 第一次卷积+激活+池化
        x = self.pool(nn.functional.relu(self.conv1(x)))  
        # 第二次卷积+激活+池化
        x = self.pool(nn.functional.relu(self.conv2(x)))  
        # 展平张量,为全连接层准备
        x = x.view(x.size(0), -1)  
        # 第一个全连接层+激活
        x = nn.functional.relu(self.fc1(x))  
        # 第二个全连接层,输出预测结果
        x = self.fc2(x)  
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
    
#超参数调节部分
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize a model, and put it on the device specified.
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "定子匝间短路")
# 添加模型保存路径
model_save_path = "best_model.pth"

batch_size =64
n_epochs = 1000
patience = 500
lr=0.002
wramup_epochs=5
initial_lr=0.001
weight_decay=1e-5
dropout=0.1
model = Res_SA(dropout=dropout).to(device)
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
    betas=(0.9, 0.999)
)
# 创建调度器
scheduler = ReduceLROnPlateau(optimizer, 
                             mode='min',           # 监控指标是越小越好
                             factor=0.1,           # 学习率调整倍数
                             patience=100,           # 容忍多少个epoch指标不改善
                             verbose=True)         # 打印学习率变化信息



train_set = MyDataset(data_dir)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = MyDataset(data_dir)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# 参数记录在runs文件夹，使用tendorboard查看
writer = SummaryWriter("runs/fulltry8")


# 初始化
stale = 0
best_acc = 0

# 主循环
for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        datas, labels = batch
        
        #imgs = imgs.half()
       
    

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(datas.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        #print
        print(f"loss: {loss.item():.4f}")
        print(f"acc: {acc:.4f}")
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    writer.add_scalar('Loss/train', train_loss, epoch) 
    writer.add_scalar('Accuracy/train', train_acc, epoch)

    # ---------- Validation ----------
    model.eval()
    valid_loss = []
    valid_accs = []

    with torch.no_grad():
        for batch in valid_loader:
            datas, labels = batch
            logits = model(datas.to(device))
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    scheduler.step(valid_loss)  # 用验证损失来调整学习率
    # 保存最优模型
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, model_save_path)
        print(f'Saved model with acc: {best_acc:.4f}')
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f'No improvement for {patience} epochs, early stopping...')
            break

    writer.add_scalar('Loss/valid', valid_loss, epoch)
    writer.add_scalar('Accuracy/valid', valid_acc, epoch)

    print(f"Epoch {epoch+1}/{n_epochs}")
    print(f"Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")
    print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
