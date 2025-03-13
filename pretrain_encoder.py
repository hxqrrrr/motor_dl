import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ReduceLROnPlateau
from models.ProtoNet_backbone import AttentiveEncoder
# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设脚本在项目的二级目录下）
project_root = os.path.dirname(os.path.dirname(script_dir))

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from models.ProtoNet import ProtoNet, CNN1D_embed
from models.dataset import SourceDomainDataset
from utils.utils_train import split_dataset, check_data_leakage


# 创建自定义数据集包装器，用于标签映射
class MappedLabelDataset(Dataset):
    def __init__(self, dataset, label_mapping):
        self.dataset = dataset
        self.label_mapping = label_mapping
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        data, label = self.dataset[index]
        # 将原始标签映射到连续的索引
        mapped_label = torch.tensor(self.label_mapping[label.item()], dtype=torch.long)
        return data, mapped_label


# 添加CNN1D模型
class CNN1D(nn.Module):
    def __init__(self, input_channels=1, sequence_length=5000, feature_dim=64):
        super(CNN1D, self).__init__()
        self.sequence_length = sequence_length
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 计算全连接层的输入维度
        self.fc_input_dim = 256 * (sequence_length // 8) 
        # 全连接层 - 修改为输出feature_dim
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim)
        )
        # 标准化
        self.norm = nn.BatchNorm1d(input_channels)

    def forward(self, x):
        # 应用输入标准化
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 添加优化版的CNN1D_embed，结合enhance的优点和embed的元学习特性
class EnhancedCNN1D_embed(nn.Module):
    """优化版的CNN1D_embed，结合enhance的优点和embed的元学习特性
    
    参数:
        in_channels: 输入通道数
        hidden_dim: 隐藏层维度
        feature_dim: 输出特征维度
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        use_l2_norm: bool = True,
        dropout_rate: float = 0.3
    ):
        super(EnhancedCNN1D_embed, self).__init__()
        
        # 输入标准化层
        self.norm = nn.BatchNorm1d(in_channels)
        
        # 第一个卷积块 - 16通道，无BN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第二个卷积块 - 32通道，有BN
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第三个卷积块 - 64通道，有BN
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第四个卷积块 - 64通道，有BN
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第五个卷积块 - 64通道，有BN，无池化
        self.conv5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 全局自适应平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征映射层
        self.feature_layer = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.use_l2_norm = use_l2_norm
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用输入标准化
        x = self.norm(x)
        
        # 应用卷积块
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # 全局池化
        x = self.global_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 特征映射
        x = self.feature_layer(x)
        
        # 条件L2归一化
        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=1)
        
        return x


class EncoderClassifier(nn.Module):
    """
    将ProtoNet的特征提取器与分类头结合的模型
    用于监督学习训练特征提取器
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        feature_dim: int,
        num_classes: int,
        backbone: str = 'cnn1d',
        dropout: float = 0.5,
        use_l2_norm: bool = True  # 添加是否使用L2归一化的选项
    ):
        super(EncoderClassifier, self).__init__()
        
        # 初始化特征提取器
        if backbone == 'cnn1d':
            # 使用原始的CNN1D_embed
            self.encoder = CNN1D_embed(in_channels, hidden_dim, feature_dim)
        elif backbone == 'cnn1d_enhanced':
            # 使用增强版的CNN1D
            self.encoder = CNN1D(input_channels=in_channels, sequence_length=5000, feature_dim=feature_dim)
        elif backbone == 'cnn1d_embed_enhanced':
            # 使用优化版的CNN1D_embed
            self.encoder = EnhancedCNN1D_embed(
                in_channels, 
                hidden_dim, 
                feature_dim,
                use_l2_norm=use_l2_norm,
                dropout_rate=dropout
            )
        elif backbone == 'cbam':
            # 使用注意力机制的编码器
            self.encoder = AttentiveEncoder(
                in_channels,
                hidden_dim,
                feature_dim,
                attention_type='cbam'
            )
        elif backbone == 'channel':
            # 使用通道注意力的编码器
            self.encoder = AttentiveEncoder(
                in_channels,
                hidden_dim,
                feature_dim,
                attention_type='channel'
            )
        elif backbone == 'spatial':
            # 使用空间注意力的编码器
            self.encoder = AttentiveEncoder(
                in_channels,
                hidden_dim,
                feature_dim,
                attention_type='spatial'
            )
        else:
            raise ValueError(f"不支持的骨干网络类型: {backbone}")
        
        # 添加分类头，增加dropout
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        # 提取特征
        features = self.encoder(x)
        # 应用dropout
        features = self.dropout(features)
        # 分类
        logits = self.classifier(features)
        return logits
    
    def get_encoder(self):
        """返回训练好的编码器部分"""
        return self.encoder


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc="训练", ncols=100, leave=True) as pbar:
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # 将数据移到设备上
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            total_loss += loss.item()
            acc = 100. * correct / total
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(
                f"loss: {loss.item():.3f}, acc: {acc:.2f}%, lr: {current_lr:.2e}"
            )
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100. * correct / total
    
    return avg_loss, avg_acc


def evaluate(model, val_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc="评估", ncols=100, leave=True) as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 计算准确率
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                total_loss += loss.item()
                acc = 100. * correct / total
                pbar.set_postfix_str(
                    f"loss: {loss.item():.3f}, acc: {acc:.2f}%"
                )
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = 100. * correct / total
    
    return avg_loss, avg_acc


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


def save_model(model, optimizer, scheduler, epoch, best_val_acc, save_path):
    """保存模型"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_acc': best_val_acc
    }, save_path)


def save_encoder(model, save_path):
    """保存编码器部分"""
    encoder = model.get_encoder()
    torch.save(encoder.state_dict(), save_path)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='ProtoNet特征提取器监督学习训练脚本')
    parser.add_argument('--in_channels', type=int, default=5, help='输入通道数')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--feature_dim', type=int, default=64, help='特征维度')
    parser.add_argument('--backbone', type=str, default='cnn1d_enhanced', 
                      choices=['cnn1d', 'cnn1d_enhanced', 'cnn1d_embed_enhanced', 'channel', 'spatial', 'cbam'],
                      help='骨干网络类型: cnn1d, cnn1d_enhanced, cnn1d_embed_enhanced, channel, spatial, cbam')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout比率')
    parser.add_argument('--use_l2_norm', type=bool, default=True, help='是否使用L2归一化')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='权重衰减')
    parser.add_argument('--scheduler', type=str, default='plateau',
                      choices=['step', 'cosine', 'plateau'],
                      help='学习率调度器类型: step, cosine, plateau')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='预热阶段的epoch数')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-5, help='预热阶段的初始学习率')
    parser.add_argument('--source_data', type=str, 
                    default=os.path.join('/root', 'hxq/motor_dl/data/h5data/selected_data.h5'), 
                    help='源域数据文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'hxq/motor_dl/runs/encoder_pretrain_{args.backbone}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
    
    # 加载数据集
    print(f"加载源域数据并进行划分: {args.source_data}")
    train_indices, test_indices = split_dataset(
        data_path=args.source_data,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # 创建训练集和测试集
    train_dataset = SourceDomainDataset(h5_file_path=args.source_data, indices=train_indices)
    val_dataset = SourceDomainDataset(h5_file_path=args.source_data, indices=test_indices)
    
    # 检查数据泄露
    if check_data_leakage(train_dataset, val_dataset):
        print("请检查数据集划分，确保训练集和验证集没有重叠！")
        sys.exit(1)
    
    # 获取类别数量和标签映射
    with torch.no_grad():
        all_labels = []
        for _, label in train_dataset:
            all_labels.append(label.item())
        unique_labels = sorted(set(all_labels))
        num_classes = len(unique_labels)
        
        # 创建标签映射，将原始标签映射到连续的索引
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        print(f"数据集中的类别数量: {num_classes}")
        print(f"原始标签: {unique_labels}")
        print(f"标签映射: {label_to_idx}")
    
    # 使用标签映射包装数据集
    train_dataset_mapped = MappedLabelDataset(train_dataset, label_to_idx)
    val_dataset_mapped = MappedLabelDataset(val_dataset, label_to_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset_mapped,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 设置为0以避免多进程问题
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset_mapped,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 设置为0以避免多进程问题
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型
    model = EncoderClassifier(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        backbone=args.backbone,
        dropout=args.dropout,
        use_l2_norm=args.use_l2_norm
    )
    model = model.to(device)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建优化器
    optimizer = optim.AdamW(  # 使用AdamW优化器，更好的权重衰减实现
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),  # 默认值
        eps=1e-8  # 默认值
    )
    
    # 创建预热调度器
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_start_lr / args.lr,
        end_factor=1.0,
        total_iters=args.warmup_epochs
    )
    
    # 创建主要训练阶段的学习率调度器
    if args.scheduler == 'step':
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
    elif args.scheduler == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=1e-6
        )
    else:  # plateau
        # 使用ReduceLROnPlateau，根据验证集损失自动调整学习率
        main_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',  # 监控验证损失
            factor=0.5,  # 学习率减半
            patience=5,  # 5个epoch没有改善就降低学习率
            verbose=True,
            min_lr=1e-6
        )
    
    # 如果使用ReduceLROnPlateau，不需要SequentialLR
    if args.scheduler == 'plateau':
        # 先进行预热
        scheduler = warmup_scheduler
        use_plateau = True
    else:
        # 组合预热和主要训练阶段的调度器
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[args.warmup_epochs]
        )
        use_plateau = False
    
    # 保存训练参数
    params = vars(args)
    params['num_classes'] = num_classes
    params['label_mapping'] = label_to_idx
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # 初始化训练状态
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 训练循环
    print(f"开始训练特征提取器...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        
        # 记录训练指标到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 在验证集上评估
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
        
        # 记录验证指标到TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 绘制训练曲线
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_dir)
        
        # 检查验证准确率是否改善
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # 保存最佳模型
            save_model(
                model, 
                optimizer, 
                scheduler if not use_plateau else None, 
                epoch, 
                best_val_acc, 
                os.path.join(save_dir, f'best_model_val_acc_{val_acc:.2f}.pth')
            )
            
            # 保存编码器部分
            save_encoder(
                model,
                os.path.join(save_dir, f'best_encoder_val_acc_{val_acc:.2f}.pth')
            )
            
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"验证准确率在{args.patience}个评估周期内未改善，在epoch {epoch+1}停止训练")
                break
        
        # 更新学习率
        if use_plateau and epoch >= args.warmup_epochs:
            # 如果使用ReduceLROnPlateau，根据验证损失更新学习率
            main_scheduler.step(val_loss)
        elif epoch < args.warmup_epochs and use_plateau:
            # 预热阶段
            scheduler.step()
        else:
            # 使用其他调度器
            scheduler.step()
    
    # 关闭TensorBoard writer
    writer.close()
    
    # 保存最终模型
    save_model(
        model, 
        optimizer, 
        scheduler if not use_plateau else None, 
        epoch, 
        best_val_acc, 
        os.path.join(save_dir, 'final_model.pth')
    )
    
    # 保存最终编码器
    save_encoder(
        model,
        os.path.join(save_dir, 'final_encoder.pth')
    )
    
    print(f"训练完成！最佳验证集准确率: {best_val_acc:.2f}%")
    print(f"模型和训练参数已保存到: {save_dir}") 