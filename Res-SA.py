# 设置实验名称
_exp_name = "sample"
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
    def __init__(self,path,files = None):
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
            print(data_tensor.shape)
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

    return train_loader, valid_loader, 



#一维cnn模型
class OneDCNN(nn.Module):  
    def __init__(self, input_size, num_classes, kernel_size=5):  
        super(OneDCNN, self).__init__()  
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=kernel_size)  
        self.pool = nn.MaxPool1d(kernel_size=2)  
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size)  
        
        # 计算有效的输入尺寸  
        conv1_output_size = (input_size - kernel_size + 1)  
        conv1_pooled_size = conv1_output_size // 2  
        conv2_output_size = (conv1_pooled_size - kernel_size + 1)  
        conv2_pooled_size = conv2_output_size // 2  
        
        # 为fc1定义尺寸  
        self.fc1 = nn.Linear(32 * conv2_pooled_size, 128)  
        self.fc2 = nn.Linear(128, num_classes)  

    def forward(self, x):  
        x = self.pool(nn.functional.relu(self.conv1(x)))  
        x = self.pool(nn.functional.relu(self.conv2(x)))  
        x = x.view(x.size(0), -1)  # Flatten the output  
        x = nn.functional.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x
    
#Res-SA模型
class ResSA(nn.Module):
    def __init__(self, input_size, num_classes,batch_size, kernel_size=3):
        super(ResSA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, kernel_size), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, kernel_size), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, kernel_size), padding=(0, 1))
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, kernel_size), padding=(0, 1))
        self.SA1 = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.SA2 = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.SA3 = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.fc1 = nn.Linear(32000, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.bnReLU1 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU())
        self.bnReLU2 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU())
        self.bnReLU3 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU())
        self.bnReLU4 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU())
        self.flatten = nn.Flatten()
        
        # 取消注释并正确初始化SA层
        self.SA1 = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.SA2 = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.SA3 = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        

    def forward(self, x):
        # x形状为(batch_size, 4, 1001)
        x = x.unsqueeze(1)
        # x形状为(batch_size, 1，4， 1001)
        print(x.shape)
        x =self.conv1(x)
        x = self.bnReLU1(x)
        #残差层1
        residual = x
        x = self.conv2(x)
        x = self.bnReLU1(x)
        x = x + residual
        x = self.bnReLU1(x)

        residual = x
        x = self.conv2(x)
        x = self.bnReLU1(x)
        x = x + residual
        x = self.bnReLU1(x)

        #SA层1
        batch_size, _, height, width = x.shape
        x = x.permute(0, 2, 1, 3)  # [batch_size, height, channels, width]
        x = x.reshape(batch_size * height, width, -1)  # [batch_size*height, width, channels]
        x, _ = self.SA1(x, x, x)
        x = x.reshape(batch_size, height, width, -1)  # 恢复原始形状
        x = x.permute(0, 3, 1, 2)  # 调整回所需的通道顺序
   
        x = self.pool(x)

        x = self.flatten(x)
  
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
    
    
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize a model, and put it on the device specified.

batch_size =1024
n_epochs = 1000
patience = 200
lr=0.005
weight_decay=1e-5
model = ResSA(input_size=1001, num_classes=16,batch_size=batch_size).to(device)
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
    betas=(0.9, 0.999)
)
train_set = MyDataset("定子匝间短路")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = MyDataset("定子匝间短路")
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


# 参数记录在runs文件夹，使用tendorboard查看
writer = SummaryWriter('runs/lr0.005_bs512_epoch2000_patience10000_lrdecay0.1_wd1e-5_seed6666')
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
