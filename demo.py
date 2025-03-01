import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.dataset import ProtoNetDataset, h5Dataset, TaskProtoNetDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
from utils.utils import check_data_leakage, train_epoch, evaluate, plot_training_curves, save_training_info, get_model
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, SequentialLR

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
    parser.add_argument('--task_labels', type=int, default=1,
                      help='任务标签 (默认: 1)')
    args = parser.parse_args()
    if args.task_labels == 1:
        selected_labels = [0, 1, 2, 3]
    elif args.task_labels == 2:
        selected_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 设置超参数
    training_params = {
        'model': args.model,

        # 任务参数
        'n_way': len(selected_labels),
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
        'distance_type': args.distance_type,  # 距离度量方式
        'selected_labels': selected_labels
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
    folder_path = "data/h5data"
    
    source_dataset = h5Dataset(folder_path=folder_path, train=True)
    target_dataset = h5Dataset(folder_path=folder_path, train=False)

       # 检查数据泄露
    if check_data_leakage(source_dataset,target_dataset):
        print("请检查数据集划分，确保训练集和验证集没有重叠！")
        sys.exit(1)
    
    # 创建训练集和验证集
    source_train_dataset = TaskProtoNetDataset(
        base_dataset=source_dataset,
        n_way=training_params['n_way'],
        n_support=training_params['n_support'],
        n_query=training_params['n_query'],
        selected_labels=training_params['selected_labels'],
        is_train=True
    )
    
    source_val_dataset = TaskProtoNetDataset(
        base_dataset=target_dataset,
        n_way=training_params['n_way'],
        n_support=training_params['n_support'],
        n_query=training_params['n_query'],
        selected_labels=training_params['selected_labels'],
        is_train=False
    )
    
   
    
    # 创建数据加载器
    train_loader = DataLoader(
        source_train_dataset,
        batch_size=training_params['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        source_val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        num_workers=4
    )
    for support_x, support_y, query_x, query_y in train_loader:
        print("支持样本 (support_x):", support_x)
        print("支持标签 (support_y):", support_y)
        print("查询样本 (query_x):", query_x)
        print("查询标签 (query_y):", query_y)
        break  # 只获取一个 batch，之后退出循环