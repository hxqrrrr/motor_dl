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
    parser.add_argument('--model', type=str, default='all_model',
                      help='要训练的模型名称 ( all_model)')
    parser.add_argument('--pretrained', type=str, default=None,
                      help='预训练模型路径，例如: runs/protonet_20240223_194815/best_model_val_acc_0.8670.pth')
    parser.add_argument('--in_channels', type=int, default=5,
                      help='输入通道数 (默认: 5)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='隐藏层维度 (默认: 64)')
    parser.add_argument('--feature_dim', type=int, default=128,
                      help='特征维度 (默认: 128)')
    parser.add_argument('--backbone', type=str, default='cnn1d',
                      choices=['cnn1d', 'channel', 'spatial', 'cbam'],
                      help='特征提取器类型: cnn1d, channel, spatial, cbam (默认: cnn1d)')
    parser.add_argument('--distance_type', type=str, default='euclidean',
                      choices=['euclidean', 'cosine', 'relation'],
                      help='距离度量方式: euclidean, cosine, relation (默认: euclidean)')
    args = parser.parse_args()

    # 设置超参数
    training_params = {

        'model': args.model,

        # 任务参数
        'n_way': 5,
        'n_support': 5,
        'n_query': 10,
        'batch_size': 4,
        
        # 训练超参数
        'n_epochs': 80,
        'patience': 15,
        'lr': 0.001,                # 学习率
        'weight_decay': 0.0001,     # 权重衰减
        
        # 学习率调度器参数
        'step_size': 3,             # 每3个epoch调整一次学习率
        'gamma': 0.5,               # 学习率衰减因子
        
        # 模型参数 - 从命令行参数获取
        'in_channels': args.in_channels,     # 输入通道数
        'hidden_dim': args.hidden_dim,       # 隐藏层维度
        'feature_dim': args.feature_dim,     # 特征维度
        'backbone': args.backbone,           # 骨干网络类型
        'distance_type': args.distance_type  # 距离度量方式
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
        n_way=training_params['n_way'],
        n_support=training_params['n_support'],
        n_query=training_params['n_query']
    )
    
    val_dataset = ProtoNetDataset(
        base_dataset=val_base_dataset,
        n_way=training_params['n_way'],
        n_support=training_params['n_support'],
        n_query=training_params['n_query']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型
    model = get_model(
        model_name=training_params['model'],
        in_channels=training_params['in_channels'],
        hidden_dim=training_params['hidden_dim'],
        feature_dim=training_params['feature_dim'],
        backbone=training_params['backbone'],
        distance_type=training_params['distance_type']
    ).to(device)
    
    # 初始化训练状态
    start_epoch = 0
    best_val_acc = 0
    patience_counter = 0
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_params['lr'],
        weight_decay=training_params['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_params['step_size'],
        gamma=training_params['gamma']
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
    for epoch in range(start_epoch, training_params['n_epochs']):
        print(f"\nEpoch {epoch+1}/{training_params['n_epochs']}")
        
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
                'best_val_acc': best_val_acc,
                'val_accs': val_accs,
                'in_channels': training_params['in_channels'],
                'hidden_dim': training_params['hidden_dim'],
                'feature_dim': training_params['feature_dim'],
                'backbone': training_params['backbone'],
                'distance_type': training_params['distance_type'],
                'lr': training_params['lr'],
                'weight_decay': training_params['weight_decay'],
                'step_size': training_params['step_size'],
                'gamma': training_params['gamma']
            }
            save_training_info(model_info, training_params, save_dir, is_best_model=True)
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= training_params['patience']:
                print(f"验证准确率在{training_params['patience']}个评估周期内未改善，在epoch {epoch}停止训练")
                break
        
        scheduler.step()
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, epochs, save_dir)
    
    # 保存最终模型和训练信息
    model_info = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_accs': val_accs,
        'in_channels': training_params['in_channels'],
        'hidden_dim': training_params['hidden_dim'],
        'feature_dim': training_params['feature_dim'],
        'backbone': training_params['backbone'],
        'distance_type': training_params['distance_type'],
        'lr': training_params['lr'],
        'weight_decay': training_params['weight_decay'],
        'step_size': training_params['step_size'],
        'gamma': training_params['gamma']
    }
    params_file = save_training_info(model_info, training_params, save_dir, is_best_model=False)
    
    print(f"训练完成！最佳验证集准确率: {best_val_acc:.4f}")
    print(f"训练参数已保存到: {params_file}")
    
    
