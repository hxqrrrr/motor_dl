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
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, SequentialLR


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
    parser.add_argument('--distance_type', type=str, default='cosine',
                      choices=['euclidean', 'cosine', 'relation'],
                      help='距离度量方式: euclidean, cosine, relation (默认: euclidean)')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout比率 (默认: 0.3)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                      choices=['step', 'cosine'],
                      help='学习率调度器类型: step, cosine (默认: cosine)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                      help='预热阶段的epoch数 (默认: 5)')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6,
                      help='预热阶段的初始学习率 (默认: 1e-6)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='目标学习率 (默认: 0.001)')
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
        'lr': args.lr,                # 目标学习率
        'weight_decay': 0.0001,     # 权重衰减
        'dropout': args.dropout,     # Dropout比率
        
        # 预热阶段参数
        'warmup_epochs': args.warmup_epochs,        # 预热epoch数
        'warmup_start_lr': args.warmup_start_lr,    # 预热初始学习率
        
        # 学习率调度器参数
        'scheduler': args.scheduler, # 调度器类型
        'step_size': 3,             # 每3个epoch调整一次学习率(仅用于step调度器)
        'gamma': 0.5,               # 学习率衰减因子(仅用于step调度器)
        'T_max': 50,                # 余弦退火周期(仅用于cosine调度器)
        'eta_min': 1e-6,            # 最小学习率(仅用于cosine调度器)
        
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
    save_dir = f'runs/{args.model}_{args.backbone}_{args.distance_type}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
    
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
        distance_type=training_params['distance_type'],
        dropout=training_params['dropout']
    ).to(device)
    
    # 初始化训练状态
    start_epoch = 0
    best_val_acc = 0
    patience_counter = 0
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_params['lr'],  # 使用目标学习率
        weight_decay=training_params['weight_decay']
    )
    
    # 创建预热调度器
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=training_params['warmup_start_lr'] / training_params['lr'],  # 从较小的学习率开始
        end_factor=1.0,  # 达到目标学习率
        total_iters=training_params['warmup_epochs']
    )
    
    # 创建主要训练阶段的学习率调度器
    if training_params['scheduler'] == 'step':
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_params['step_size'],
            gamma=training_params['gamma']
        )
    else:  # cosine
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_params['T_max'] - training_params['warmup_epochs'],  # 减去预热阶段的epoch数
            eta_min=training_params['eta_min']
        )
    
    # 组合预热和主要训练阶段的调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[training_params['warmup_epochs']]  # 在warmup_epochs后切换到主要调度器
    )
    
    # 如果指定了预训练模型，加载模型和训练状态
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"\n加载预训练模型: {args.pretrained}")
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', checkpoint.get('final_val_acc', 0))
        print(f"从epoch {start_epoch}继续训练，之前最佳验证集准确率: {best_val_acc:.4f}")
    
    # 用于记录训练过程
    train_losses = []
    train_accs = []
    train_recalls = []
    train_f1s = []
    val_losses = []
    val_accs = []
    val_recalls = []
    val_f1s = []
    epochs = []
    
    # 训练循环
    for epoch in range(start_epoch, training_params['n_epochs']):
        print(f"\nEpoch {epoch+1}/{training_params['n_epochs']}")
        
        # 训练一个epoch
        train_loss, train_acc, train_recall, train_f1 = train_epoch(model, train_loader, optimizer, device)
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, 召回率: {train_recall:.4f}, F1分数: {train_f1:.4f}")
        
        # 记录训练指标到 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        
        # 在验证集上评估
        val_loss, val_acc, val_recall, val_f1 = evaluate(model, val_loader, device)
        print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, 召回率: {val_recall:.4f}, F1分数: {val_f1:.4f}")
        
        # 记录验证指标到 TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        
        # 记录学习率
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
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
                'val_recalls': val_recalls,
                'val_f1s': val_f1s,
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
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}, 召回率: {val_recall:.4f}, F1分数: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= training_params['patience']:
                print(f"验证准确率在{training_params['patience']}个评估周期内未改善，在epoch {epoch}停止训练")
                break
        
        scheduler.step()
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, epochs, save_dir)
    
    # 关闭 TensorBoard writer
    writer.close()
    
    # 保存最终模型和训练信息
    model_info = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_accs': val_accs,
        'val_recalls': val_recalls,
        'val_f1s': val_f1s,
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
    
    
