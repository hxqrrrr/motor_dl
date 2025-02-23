import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.dataset import ProtoNetDataset, h5Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
from utils.utils import check_data_leakage, train_epoch, evaluate, plot_training_curves, save_training_info, get_model
import argparse


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='元学习模型训练脚本')
    parser.add_argument('--model', type=str, default='protonet',
                      help='要训练的模型名称 (protonet 或 protonet_attention)')
    parser.add_argument('--pretrained', type=str, default=None,
                      help='预训练模型路径，例如: runs/protonet_20240223_194815/best_model_val_acc_0.8670.pth')
    args = parser.parse_args()

    # 设置超参数
    model_config = {
        'in_channels': 5,
        'hidden_dim': 64,
        'feature_dim': 128,
        'backbone': 'cnn1d',
        'distance_type': 'euclidean'
    }
    
    train_config = {
        'n_way': 5,
        'n_support': 5,
        'n_query': 10,
        'batch_size': 4,
        'n_epochs': 80,
        'patience': 15,
        'test_interval': 1
    }
    
    optimizer_config = {
        'lr': 0.001,
        'weight_decay': 0.0001,
        'step_size': 2,
        'gamma': 0.5
    }

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"使用模型: {args.model}")
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 创建保存模型和图表的目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'runs/{args.model}_{timestamp}'
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
        n_way=train_config['n_way'],
        n_support=train_config['n_support'],
        n_query=train_config['n_query']
    )
    
    val_dataset = ProtoNetDataset(
        base_dataset=val_base_dataset,
        n_way=train_config['n_way'],
        n_support=train_config['n_support'],
        n_query=train_config['n_query']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型
    model = get_model(
        model_name=args.model,
        **model_config
    ).to(device)
    
    # 初始化训练状态
    start_epoch = 0
    best_val_acc = 0
    patience_counter = 0
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=optimizer_config['step_size'],
        gamma=optimizer_config['gamma']
    )
    
    # 如果指定了预训练模型，加载模型和训练状态
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"\n加载预训练模型: {args.pretrained}")
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', checkpoint.get('final_val_acc', 0))
        print(f"从epoch {start_epoch}继续训练，之前最佳验证集准确率: {best_val_acc:.4f}")
    
    # 用于记录训练过程
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs = []
    
    # 训练循环
    for epoch in range(start_epoch, train_config['n_epochs']):
        print(f"\nEpoch {epoch+1}/{train_config['n_epochs']}")
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 在验证集上评估
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        
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
            save_training_info(model_info, train_config, save_dir, is_best_model=True)
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= train_config['patience']:
                print(f"验证准确率在{train_config['patience']}个评估周期内未改善，在epoch {epoch}停止训练")
                break
        
        scheduler.step()
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, epochs, save_dir)
    
    # 保存最终模型和训练信息
    model_info = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_accs': val_accs
    }
    params_file = save_training_info(model_info, train_config, save_dir, is_best_model=False)
    
    print(f"训练完成！最佳验证集准确率: {best_val_acc:.4f}")
    print(f"训练参数已保存到: {params_file}")
    
    
