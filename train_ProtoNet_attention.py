import sys
import os
import json
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
from models.ProtoNet_attention import ProtoNetWithAttention
from utils.utils import check_data_leakage, train_epoch, evaluate, plot_training_curves, save_training_info


if __name__ == "__main__":
    n_way = 5
    n_support = 5
    n_query = 10
    batch_size = 4
    n_epochs = 80
    patience = 15
    test_interval = 1
    pretrained_model_path = 'runs/protonet_attention_20250223_194815/best_model_val_acc_0.8670.pth'
    lr = 0.0001
    weight_decay = 0.001
    patience_counter = 0
    best_train_loss = float('inf')
    best_val_acc = 0
    start_epoch = 0

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 创建保存模型和图表的目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'runs/protonet_attention_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据集
    h5_folder_path = "data/h5data"
    train_base_dataset = h5Dataset(folder_path=h5_folder_path, train=True)
    val_base_dataset = h5Dataset(folder_path=h5_folder_path, train=False)
    
    # 检查数据泄露
    if check_data_leakage(train_base_dataset,val_base_dataset):
        print("请检查数据集划分，确保训练集和验证集没有重叠！")
        sys.exit(1)
    
    # 创建训练集和验证集
    train_dataset = ProtoNetDataset(
        base_dataset=train_base_dataset,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query
    )
    
    val_dataset = ProtoNetDataset(
        base_dataset=val_base_dataset,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 创建训练参数字典
    training_params = {
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'patience': patience,
        'test_interval': test_interval,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'model_config': {
            'in_channels': 5,
            'hidden_dim': 64,
            'feature_dim': 128,
            'backbone': 'cnn1d',
            'distance_type': 'euclidean',
            'attention_type': 'cbam'
        }
    }
    
    # 创建模型
    model = ProtoNetWithAttention(
        in_channels=5,      
        hidden_dim=64,
        feature_dim=128,
        backbone='cnn1d',
        distance_type='euclidean',
        attention_type='cbam'
    ).to(device)
    
    # 是否加载预训练模型继续训练
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"\n加载预训练模型: {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f"从epoch {start_epoch}继续训练，之前最佳验证集准确率: {best_val_acc:.4f}")
    
    # 创建优化器和学习率调度器
    optimizer = optim.Adam(  # 使用Adam优化器
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )
    
    # 如果加载了预训练模型，也加载优化器状态
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 更新学习率（可选）
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # 使用StepLR，每6个epoch将学习率降低为原来的0.5
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.3
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
        
        # 在验证集上评估
        if (epoch + 1) % test_interval == 0:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
            
            # 记录验证指标
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            epochs.append(epoch)
            
            # 检查验证准确率是否改善
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                model_info = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc
                }
                save_training_info(model_info, training_params, save_dir, is_best_model=True)
                print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"验证准确率在{patience}个评估周期内未改善，在epoch {epoch}停止训练")
                    break
        
        # 更新学习率
        scheduler.step()
        
        # 使用工具函数绘制训练曲线
        plot_training_curves(
            train_losses=train_losses,
            train_accs=train_accs,
            val_losses=val_losses,
            val_accs=val_accs,
            epochs=epochs,
            save_dir=save_dir
        )
    
    # 使用工具函数保存最终训练信息
    model_info = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_accs': val_accs
    }
    params_file = save_training_info(model_info, training_params, save_dir, is_best_model=False)
    
    print(f"训练完成！最佳验证集准确率: {best_val_acc:.4f}")
    print(f"训练参数已保存到: {params_file}")
    
    
