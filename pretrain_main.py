import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ProtoNet import ProtoNet
from models.dataset import h5Dataset, ProtoNetDataset,SourceDomainDataset
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from models.ProtoNet_backbone import ProtoNetWithAttention
import h5py
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from sklearn.metrics import recall_score, f1_score
from utils.utils_train import check_data_leakage, train_epoch, evaluate, plot_training_curves, save_training_info, get_model, split_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用GPU 1

# 添加一个新函数，用于加载预训练的编码器
def load_pretrained_encoder(model, encoder_path, freeze=True):
    """
    加载预训练的编码器到模型中
    
    参数:
        model: 目标模型
        encoder_path: 预训练编码器的路径
        freeze: 是否冻结编码器参数
    
    返回:
        加载了预训练编码器的模型
    """
    if not os.path.exists(encoder_path):
        print(f"预训练编码器路径不存在: {encoder_path}")
        return model
    
    print(f"加载预训练编码器: {encoder_path}")
    
    # 加载预训练编码器的状态字典
    encoder_state_dict = torch.load(encoder_path)
    
    # 获取模型中的编码器
    if hasattr(model, 'encoder'):
        # 对于ProtoNet模型
        encoder = model.encoder
    elif hasattr(model, 'feature_extractor'):
        # 对于ProtoNetWithAttention模型
        encoder = model.feature_extractor
    else:
        print("无法确定模型中的编码器部分")
        return model
    
    # 尝试加载预训练权重
    try:
        # 直接加载可能会因为键名不匹配而失败
        encoder.load_state_dict(encoder_state_dict)
        print("成功加载预训练编码器权重")
    except Exception as e:
        print(f"加载预训练编码器权重时出错: {e}")
        print("尝试部分加载权重...")
        
        # 获取编码器的状态字典
        encoder_dict = encoder.state_dict()
        
        # 过滤掉不匹配的键
        pretrained_dict = {k: v for k, v in encoder_state_dict.items() if k in encoder_dict and encoder_dict[k].shape == v.shape}
        
        # 更新编码器的状态字典
        encoder_dict.update(pretrained_dict)
        encoder.load_state_dict(encoder_dict)
        
        print(f"成功加载 {len(pretrained_dict)}/{len(encoder_state_dict)} 个预训练参数")
    
    # 如果需要冻结编码器参数
    if freeze:
        print("冻结编码器参数")
        for param in encoder.parameters():
            param.requires_grad = False
    
    return model

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='源域元学习模型训练脚本')
    parser.add_argument('--model', type=str, default='all_model',
                      help='要训练的模型名称 (protonet 或 protonet_attention)')
    parser.add_argument('--pretrained', type=str, default=None,
                      help='预训练模型路径，例如: runs/source_domain_protonet_20240226_194815/best_model_val_acc_0.8670.pth')
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                      help='预训练编码器路径，例如: runs/encoder_pretrain_channel_20240301_123456/best_encoder_val_acc_95.50.pth')
    parser.add_argument('--freeze_encoder', action='store_true',
                      help='是否冻结预训练编码器的参数')
    parser.add_argument('--n_way', type=int, default=4, help='N-way分类任务中的类别数')
    parser.add_argument('--n_support', type=int, default=5, help='每个类别的支持集样本数')
    parser.add_argument('--n_query', type=int, default=10, help='每个类别的查询集样本数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--in_channels', type=int, default=5, help='输入通道数')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--feature_dim', type=int, default=64, help='特征维度')
    parser.add_argument('--backbone', type=str, default='cnn1d', 
                      choices=['cnn1d', 'channel', 'spatial', 'cbam', 'enhanced_cnn1d'],
                      help='骨干网络类型: cnn1d, channel, spatial, cbam, enhanced_cnn1d')
    parser.add_argument('--distance', type=str, default='euclidean', 
                      choices=['euclidean', 'cosine', 'relation', 'relation_selfattention'],
                      help='距离度量类型: euclidean, cosine, relation, relation_selfattention')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout比率')
    parser.add_argument('--scheduler', type=str, default='cosine',
                      choices=['step', 'cosine'],
                      help='学习率调度器类型: step, cosine')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热阶段的epoch数')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6, help='预热阶段的初始学习率')
    parser.add_argument('--lr', type=float, default=0.0001, help='目标学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--source_data', type=str, default='data/h5data/selected_data.h5', help='源域数据文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_episodes', type=int, default=200, help='训练时的episode数量')
    parser.add_argument('--acc_threshold', type=float, default=0.9, 
                      help='准确率阈值，达到此值后降低学习率')
    parser.add_argument('--lr_decay_factor', type=float, default=0.4, 
                      help='准确率达到阈值后的学习率衰减因子')
    args = parser.parse_args()

    # 设置训练参数
    training_params = {
        'model': args.model,
        
        # 任务参数
        'n_way': args.n_way,
        'n_support': args.n_support,
        'n_query': args.n_query,
        'batch_size': args.batch_size,
        
        # 训练超参数
        'n_epochs': args.epochs,
        'patience': 30,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        
        # 预热阶段参数
        'warmup_epochs': args.warmup_epochs,
        'warmup_start_lr': args.warmup_start_lr,
        
        # 学习率调度器参数
        'scheduler': args.scheduler,
        'step_size': 3,             # 每3个epoch调整一次学习率(仅用于step调度器)
        'gamma': 0.5,               # 学习率衰减因子(仅用于step调度器)
        'T_max': 50,                # 余弦退火周期(仅用于cosine调度器)
        'eta_min': 1e-6,            # 最小学习率(仅用于cosine调度器)
        
        # 模型参数
        'in_channels': args.in_channels,
        'hidden_dim': args.hidden_dim,
        'feature_dim': args.feature_dim,
        'backbone': args.backbone,
        'distance_type': args.distance,
        
        # 数据参数
        'source_data': args.source_data,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        
        # 添加新参数
        'acc_threshold': args.acc_threshold,
        'lr_decay_factor': args.lr_decay_factor,
        'pretrained_encoder': args.pretrained_encoder,
        'freeze_encoder': args.freeze_encoder,
    }

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"使用模型: {args.model}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建保存模型和图表的目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    encoder_suffix = "_pretrained_encoder" if args.pretrained_encoder else ""
    freeze_suffix = "_frozen" if args.freeze_encoder else ""
    save_dir = f'runs/source_domain_{args.model}_{args.backbone}_{args.distance}{encoder_suffix}{freeze_suffix}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
    
    # 只进行一次数据划分
    print(f"加载源域数据并进行划分: {args.source_data}")
    train_indices, test_indices = split_dataset(
        data_path=args.source_data,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # 创建训练集和测试集，使用各自的索引
    train_base_dataset = SourceDomainDataset(h5_file_path=args.source_data, indices=train_indices)
    val_base_dataset = SourceDomainDataset(h5_file_path=args.source_data, indices=test_indices)
    # 检查数据泄露
    if check_data_leakage(train_base_dataset,val_base_dataset):
        print("请检查数据集划分，确保训练集和验证集没有重叠！")
        sys.exit(1)
    # 创建训练集和验证集
    train_dataset = ProtoNetDataset(
        base_dataset=train_base_dataset,
        n_way=training_params['n_way'], 
        n_support=training_params['n_support'],
        n_query=training_params['n_query'],
        num_episodes=args.num_episodes  # 设置训练时的episode数量
    )
    
    val_dataset = ProtoNetDataset(
        base_dataset=val_base_dataset,
        n_way=training_params['n_way'],  
        n_support=training_params['n_support'],
        n_query=training_params['n_query'],
        num_episodes=args.num_episodes  # 设置验证时的episode数量
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
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
    )
    
    # 如果指定了预训练编码器，加载并可选冻结
    if args.pretrained_encoder and os.path.exists(args.pretrained_encoder):
        model = load_pretrained_encoder(model, args.pretrained_encoder, freeze=args.freeze_encoder)
    
    model = model.to(device)
    
    # 初始化训练状态
    start_epoch = 0
    best_val_acc = 0
    patience_counter = 0
    
    # 创建优化器 - 如果冻结了编码器，只优化非冻结参数
    if args.freeze_encoder and args.pretrained_encoder:
        # 只优化非冻结参数
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay']
        )
        print("优化器只更新非冻结参数")
    else:
        # 优化所有参数
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay']
        )
    
    # 创建预热调度器
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=training_params['warmup_start_lr'] / training_params['lr'],
        end_factor=1.0,
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
            T_max=training_params['T_max'] - training_params['warmup_epochs'],
            eta_min=training_params['eta_min']
        )
    
    # 组合预热和主要训练阶段的调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[training_params['warmup_epochs']]
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
    
    # 在训练循环开始前添加标志
    lr_reduced = False
    
    # 训练循环
    print(f"开始训练源域元学习模型...")
    for epoch in range(start_epoch, training_params['n_epochs']):
        print(f"\nEpoch {epoch+1}/{training_params['n_epochs']}")
        
        # 训练一个epoch
        train_loss, train_acc, train_recall, train_f1 = train_epoch(model, train_loader, optimizer, device,training_params['n_way'])
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
        
        # 当验证准确率达到阈值时，降低学习率以减缓训练速度
        if not lr_reduced and val_acc >= training_params['acc_threshold'] and scheduler.get_last_lr()[0] > 1e-5:
            # 降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * training_params['lr_decay_factor']
            print(f"验证准确率达到{val_acc:.4f}，降低学习率至 {optimizer.param_groups[0]['lr']:.6f}")
            
            # 更新学习率调度器
            if training_params['scheduler'] == 'cosine':
                # 对于余弦退火调度器，需要重新初始化
                main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=training_params['T_max'] - epoch,  # 剩余的epoch数
                    eta_min=training_params['eta_min']
                )
                # 更新调度器
                scheduler._schedulers[1] = main_scheduler
            
            # 设置标志，避免多次降低学习率
            lr_reduced = True
        
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
                'pretrained_encoder': args.pretrained_encoder,
                'freeze_encoder': args.freeze_encoder
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
        'final_val_acc': val_acc,
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
        'pretrained_encoder': args.pretrained_encoder,
        'freeze_encoder': args.freeze_encoder
    }
    params_file = save_training_info(model_info, training_params, save_dir, is_best_model=False)
    
    print(f"训练完成！最佳验证集准确率: {best_val_acc:.4f}")
    print(f"训练参数已保存到: {params_file}") 