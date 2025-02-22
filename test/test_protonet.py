import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.ProtoNet import ProtoNet
from models.dataset import ProtoNetDataset, h5Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_acc = 0
    
    # 使用tqdm显示训练进度
    with tqdm(dataloader, desc="训练") as pbar:
        for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(pbar):
            # 将数据移到设备上
            support_x = support_x.squeeze(0).to(device)  # [n_way * n_support, channel=5, seq_len]
            support_y = support_y.squeeze(0).to(device)  # [n_way * n_support]
            query_x = query_x.squeeze(0).to(device)     # [n_way * n_query, channel=5, seq_len]
            query_y = query_y.squeeze(0).to(device)     # [n_way * n_query]
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(support_x, support_y, query_x)
            
            # 计算损失
            loss = torch.nn.functional.cross_entropy(logits, query_y)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 计算准确率
            pred = torch.argmax(logits, dim=1)
            acc = (pred == query_y).float().mean().item()
            
            # 更新统计信息
            total_loss += loss.item()
            total_acc += acc
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    return avg_loss, avg_acc

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        with tqdm(dataloader, desc="评估") as pbar:
            for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(pbar):
                # 将数据移到设备上
                support_x = support_x.squeeze(0).to(device)  # [n_way * n_support, channel=5, seq_len]
                support_y = support_y.squeeze(0).to(device)  # [n_way * n_support]
                query_x = query_x.squeeze(0).to(device)     # [n_way * n_query, channel=5, seq_len]
                query_y = query_y.squeeze(0).to(device)     # [n_way * n_query]
                
                # 前向传播
                logits = model(support_x, support_y, query_x)
                
                # 计算损失
                loss = torch.nn.functional.cross_entropy(logits, query_y)
                
                # 计算准确率
                pred = torch.argmax(logits, dim=1)
                acc = (pred == query_y).float().mean().item()
                
                # 更新统计信息
                total_loss += loss.item()
                total_acc += acc
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.4f}'
                })
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    return avg_loss, avg_acc

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 加载数据集
    h5_folder_path = "data/h5data"
    base_dataset = h5Dataset(folder_path=h5_folder_path)
    
    # 创建训练集和验证集
    train_dataset = ProtoNetDataset(
        base_dataset=base_dataset,
        n_way=5,           
        n_support=8,       
        n_query=15         # 增加查询样本数量
    )
    
    val_dataset = ProtoNetDataset(
        base_dataset=base_dataset,
        n_way=5,
        n_support=8,    
        n_query=15
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 创建模型
    model = ProtoNet(
        in_channels=5,      
        hidden_dim=32,
        feature_dim=64,
        backbone='cnn1d',
        distance_type='cosine'  # 改用余弦相似度
    ).to(device)
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.0001,         # 降低学习率
        weight_decay=0.01  # 添加L2正则化
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=20, 
        gamma=0.5
    )
    
    # 训练参数
    n_epochs = 100
    patience = 10  # 如果验证集性能10个epoch没有提升就停止
    patience_counter = 0
    best_val_acc = 0
    
    # 训练循环
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        
        # 学习率调整
        scheduler.step()
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_protonet.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    
