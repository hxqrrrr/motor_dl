import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ProtoNet_attention import ProtoNetWithAttention
from models.cnn1d import CNN1D
from models.dataset import ProtoNetDataset, h5Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# 添加CUDA性能优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def train_episode(model, support_x, support_y, query_x, query_y, optimizer, device):
    """训练单个episode"""
    model.train()
    
    # 将数据移到设备上
    support_x = support_x.to(device)  # [batch_size, n_support, channels, length]
    support_y = support_y.to(device)  # [batch_size, n_support]
    query_x = query_x.to(device)      # [batch_size, n_query, channels, length]
    query_y = query_y.to(device)      # [batch_size, n_query]
    
    # 记录适应前的性能
    with torch.no_grad():
        init_logits = model(support_x, support_y, query_x)  # [batch_size, n_query, n_way]
        # 重塑维度以计算损失
        init_loss = F.cross_entropy(init_logits.reshape(-1, init_logits.size(-1)), query_y.reshape(-1))
        init_acc = (init_logits.argmax(dim=-1) == query_y).float().mean().item()
    
    # 训练这个episode
    optimizer.zero_grad()
    logits = model(support_x, support_y, query_x)  # [batch_size, n_query, n_way]
    # 重塑维度以计算损失
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), query_y.reshape(-1))
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 计算最终性能
    final_acc = (logits.argmax(dim=-1) == query_y).float().mean().item()
    
    return {
        'loss': loss.item(),
        'init_acc': init_acc,
        'final_acc': final_acc,
        'improvement': final_acc - init_acc
    }

def evaluate_model(model, val_loader, device, n_test_episodes=100):
    """评估模型在多个episode上的性能"""
    model.eval()
    episode_metrics = []
    
    with torch.no_grad():
        for _ in range(n_test_episodes):
            support_x, support_y, query_x, query_y = next(iter(val_loader))
            
            support_x = support_x.to(device)  # [batch_size, n_support, channels, length]
            support_y = support_y.to(device)  # [batch_size, n_support]
            query_x = query_x.to(device)      # [batch_size, n_query, channels, length]
            query_y = query_y.to(device)      # [batch_size, n_query]
            
            logits = model(support_x, support_y, query_x)  # [batch_size, n_query, n_way]
            # 重塑维度以计算损失
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), query_y.reshape(-1))
            acc = (logits.argmax(dim=-1) == query_y).float().mean().item()
            
            episode_metrics.append({
                'loss': loss.item(),
                'acc': acc
            })
    
    # 计算平均指标和置信区间
    mean_loss = np.mean([m['loss'] for m in episode_metrics])
    mean_acc = np.mean([m['acc'] for m in episode_metrics])
    acc_std = np.std([m['acc'] for m in episode_metrics])
    conf_interval = 1.96 * acc_std / np.sqrt(n_test_episodes)
    
    return {
        'mean_loss': mean_loss,
        'mean_acc': mean_acc,
        'conf_interval': conf_interval
    }

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
    save_dir = f'runs/protonet_attention_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据集
    h5_folder_path = "data/h5data"
    train_base_dataset = h5Dataset(folder_path=h5_folder_path, train=True)
    val_base_dataset = h5Dataset(folder_path=h5_folder_path, train=False)
    
    # 创建训练集和验证集
    train_dataset = ProtoNetDataset(
        base_dataset=train_base_dataset,
        n_way=5,  # 每个episode随机选择5个类别
        n_support=5,
        n_query=15
    )
    
    val_dataset = ProtoNetDataset(
        base_dataset=val_base_dataset,
        n_way=5,  # 验证时使用相同的设置
        n_support=5,
        n_query=15
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # 从4增加到16
        shuffle=True,
        num_workers=8,  # 从4增加到8
        pin_memory=True,  # 添加这个参数
        persistent_workers=True  # 添加这个参数
)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=True,  # 验证时也随机采样
        num_workers=4
    )
    
    # 创建模型
    model = ProtoNetWithAttention(
        in_channels=5,      
        hidden_dim=64,
        feature_dim=128,
        attention_type='channel',
        backbone='cnn1d',
        distance_type='euclidean'
    ).to(device)
    
    # 训练参数
    n_episodes = 2000  # 总episode数
    eval_interval = 10  # 每1000个episode评估一次
    patience = 5  # 早停耐心值
    best_val_acc = 0
    patience_counter = 0
    
    # 创建优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,
        weight_decay=0.001
    )
    
    # 使用余弦退火重启
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=500,  # 每1000个episode重启一次
        T_mult=2,  # 每次重启后周期翻倍
        eta_min=1e-5
    )
    
    # 用于记录训练过程
    train_metrics = []
    val_metrics = []
    eval_episodes = []
    
    # 训练循环
    pbar = tqdm(train_loader, desc="训练Batches")
    for batch_idx, episode_data in enumerate(pbar):
        # 1. 训练一个batch
        metrics = train_episode(model, *episode_data, optimizer, device)
        train_metrics.append(metrics)
        
        # 更新学习率
        scheduler.step()
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix_str(
            f"loss: {metrics['loss']:.3f}, "
            f"acc: {metrics['final_acc']:.3f}, "
            f"imp: {metrics['improvement']:.3f}, "
            f"lr: {current_lr:.2e}"
        )
        
        # 2. 定期评估
        current_episode = batch_idx * train_loader.batch_size
        if (current_episode + 1) % eval_interval == 0:
            val_results = evaluate_model(model, val_loader, device)
            val_metrics.append(val_results)
            eval_episodes.append(current_episode)

            print(f"\nEpisode {current_episode+1}/{n_episodes}")
            print(f"验证集 - 准确率: {val_results['mean_acc']:.4f} "
                  f"± {val_results['conf_interval']:.4f}")
            
            # 保存最佳模型
            if val_results['mean_acc'] > best_val_acc:
                best_val_acc = val_results['mean_acc']
                patience_counter = 0
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                }, os.path.join(save_dir, f'best_model_{best_val_acc:.4f}.pth'))
                print(f"保存最佳模型，验证集准确率: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at episode {episode}")
                    break
            
            # 绘制训练过程图
            plt.figure(figsize=(15, 5))
            
            # 绘制损失曲线
            plt.subplot(1, 3, 1)
            plt.plot([m['loss'] for m in train_metrics[-eval_interval:]], 
                    label='Train Loss', alpha=0.5)
            plt.plot(eval_episodes, [m['mean_loss'] for m in val_metrics], 
                    label='Val Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            # 绘制准确率曲线
            plt.subplot(1, 3, 2)
            plt.plot([m['final_acc'] for m in train_metrics[-eval_interval:]], 
                    label='Train Acc', alpha=0.5)
            plt.plot(eval_episodes, [m['mean_acc'] for m in val_metrics], 
                    label='Val Acc')
            plt.xlabel('Episode')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training and Validation Accuracy')
            
            # 绘制改进程度曲线
            plt.subplot(1, 3, 3)
            plt.plot([m['improvement'] for m in train_metrics[-eval_interval:]], 
                    label='Improvement')
            plt.xlabel('Episode')
            plt.ylabel('Improvement')
            plt.title('Episode Improvement')
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'training_curves.png'))
            plt.close()
    
    # 保存最终模型
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_val_acc': val_results['mean_acc'],
    }, os.path.join(save_dir, 'final_model.pth'))
    
    print(f"\n训练完成！")
    print(f"最佳验证集准确率: {best_val_acc:.4f}")
    print(f"最终验证集准确率: {val_results['mean_acc']:.4f} ± {val_results['conf_interval']:.4f}")
    
    
