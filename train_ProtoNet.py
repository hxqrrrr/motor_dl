import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.ProtoNet import ProtoNet
from models.cnn1d import CNN1D
from models.dataset import ProtoNetDataset, h5Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_acc = 0
    
    with tqdm(dataloader, desc="训练", ncols=100, leave=True) as pbar:
        for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(pbar):
            # 将数据移到设备上
            support_x = support_x.to(device)  # [batch_size, n_support, channels, length]
            support_y = support_y.to(device)  # [batch_size, n_support]
            query_x = query_x.to(device)      # [batch_size, n_query, channels, length]
            query_y = query_y.to(device)      # [batch_size, n_query]
            
            optimizer.zero_grad()
            logits = model(support_x, support_y, query_x)  # [batch_size, n_query, n_way]
            
            # 重塑维度以计算损失
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                query_y.reshape(-1)
            )
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 计算准确率
            pred = logits.argmax(dim=-1)  # [batch_size, n_query]
            acc = (pred == query_y).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(
                f"loss: {loss.item():.3f}, acc: {acc:.3f}, lr: {current_lr:.2e}"
            )
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    return avg_loss, avg_acc

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        with tqdm(dataloader, desc="评估", ncols=100, leave=True) as pbar:
            for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(pbar):
                support_x = support_x.to(device)  # [batch_size, n_support, channels, length]
                support_y = support_y.to(device)  # [batch_size, n_support]
                query_x = query_x.to(device)      # [batch_size, n_query, channels, length]
                query_y = query_y.to(device)      # [batch_size, n_query]
                
                logits = model(support_x, support_y, query_x)  # [batch_size, n_query, n_way]
                
                # 重塑维度以计算损失
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    query_y.reshape(-1)
                )
                
                # 计算准确率
                pred = logits.argmax(dim=-1)  # [batch_size, n_query]
                acc = (pred == query_y).float().mean().item()
                
                total_loss += loss.item()
                total_acc += acc
                
                pbar.set_postfix_str(
                    f"loss: {loss.item():.3f}, acc: {acc:.3f}"
                )
    
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
    
    # 创建保存模型和图表的目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'runs/protonet_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据集
    h5_folder_path = "data/h5data"
    train_base_dataset = h5Dataset(folder_path=h5_folder_path, train=True)
    val_base_dataset = h5Dataset(folder_path=h5_folder_path, train=False)
    
    # 创建训练集和验证集
    train_dataset = ProtoNetDataset(
        base_dataset=train_base_dataset,
        n_way=8,  # 使用所有8个类别
        n_support=5,
        n_query=15
    )
    
    val_dataset = ProtoNetDataset(
        base_dataset=val_base_dataset,
        n_way=8,  # 验证时也使用8个类别
        n_support=5,
        n_query=15
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,  # 使用更大的batch_size
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2,  # 使用更大的batch_size
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型
    model = ProtoNet(
        in_channels=5,      
        hidden_dim=64,
        feature_dim=128,
        backbone='cnn1d',
        distance_type='euclidean'  # 使用欧氏距离
    ).to(device)
    
    # 训练参数
    n_epochs = 100
    patience = 15
    test_interval = 1
    patience_counter = 0
    best_val_acc = 0
    start_epoch = 0
    
    # 是否加载预训练模型继续训练
    pretrained_model_path = "runs/protonet_20250219_143734/best_model_0.8965.pth"  # 设置为None则从头训练
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"\n加载预训练模型: {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f"从epoch {start_epoch}继续训练，之前最佳验证集准确率: {best_val_acc:.4f}")
    
    # 创建优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,
        weight_decay=0.001
    )
    
    # 如果加载了预训练模型，也加载优化器状态
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 更新学习率（可选）
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001  # 使用较小的学习率继续训练
    
    # 使用StepLR，每30个epoch将学习率降低为原来的0.5
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.5
    )
    
    # 如果加载了预训练模型，也加载调度器状态
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 用于记录训练过程
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs = []
    
    # 训练循环
    for epoch in range(start_epoch, n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        
        # 记录训练指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 定期评估
        if (epoch + 1) % test_interval == 0:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
            
            # 记录验证指标
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            epochs.append(epoch)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                }, os.path.join(save_dir, f'best_model_{val_acc:.4f}.pth'))
                print(f"保存最佳模型，验证集准确率: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # 更新学习率
        scheduler.step()
        
        # 绘制训练过程图
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(epochs, val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_val_acc': val_acc,
    }, os.path.join(save_dir, 'final_model.pth'))
    
    print(f"训练完成！最佳验证集准确率: {best_val_acc:.4f}")
    
    
