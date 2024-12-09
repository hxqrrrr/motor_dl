import h5py
from models.Res_SA_new import *
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from models.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter
from utils.config_loader import *

# 加载配置
config = load_config('config.json')

# 设置随机种子
set_random_seed(config['seed'])

# 设置数据集和加载器
train_loader, val_loader = setup_dataset(config)

# 设置训练环境
model, optimizer, scheduler, device = setup_training(config)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 初始化 SummaryWriter
writer = SummaryWriter()

# 初始化 best_val_loss 和 no_improve
best_val_loss = float('inf')
no_improve = 0

try:
    for epoch in range(config['training']['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_data, batch_label in train_loader:
            # 将数据移至GPU
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_label.size(0)
            train_correct += (predicted == batch_label).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_label in val_loader:
                # 将验证数据也移至GPU
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += batch_label.size(0)
                correct += (predicted == batch_label).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total

        # 打印当前epoch的训练情况
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Training Accuracy: {train_acc:.6f}')
        print(f'Validation Loss: {val_loss:.6f}')
        print(f'Validation Accuracy: {val_acc:.6f}')

        # 记录损失和准确率
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/valid', val_acc, epoch)

        # 学习率调整
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= config['training']['patience']:
                print(f"\n验证loss在{config['training']['patience']}个epoch内没有改善,停止训练")
                break

except KeyboardInterrupt:
    print('\n训练被手动中断')

# 关闭 SummaryWriter
writer.close()
