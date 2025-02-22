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
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(dataloader, desc="评估") as pbar:
            for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(pbar):
                # 将数据移到设备上
                support_x = support_x.to(device)  # [batch_size, n_way * n_support, channel=5, seq_len]
                support_y = support_y.to(device)  # [batch_size, n_way * n_support]
                query_x = query_x.to(device)     # [batch_size, n_way * n_query, channel=5, seq_len]
                query_y = query_y.to(device)     # [batch_size, n_way * n_query]
                
                # 前向传播
                logits = model(support_x, support_y, query_x)  # [batch_size, n_query, n_way]
                
                # 重塑维度以计算损失和准确率
                logits = logits.view(-1, logits.size(-1))  # [batch_size * n_query, n_way]
                query_y = query_y.view(-1)                 # [batch_size * n_query]
                
                # 计算损失
                loss = torch.nn.functional.cross_entropy(logits, query_y)
                
                # 计算准确率
                pred = torch.argmax(logits, dim=1)
                acc = (pred == query_y).float().mean().item()
                
                # 保存预测结果
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(query_y.cpu().numpy())
                
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
    
    return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)

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
    base_dataset = h5Dataset(folder_path=h5_folder_path, train=False)  # 使用测试集进行评估
    
    # 获取数据集中的类别信息
    all_labels = set()
    for i in range(len(base_dataset)):
        _, label = base_dataset[i]
        all_labels.add(label.item())
    print(f"\n数据集中的类别数量: {len(all_labels)}")
    print(f"可用的类别: {sorted(list(all_labels))}")
    
    # 测试不同配置
    test_configs = [
        {'n_way': 5, 'n_support': 5, 'n_query': 15},
        {'n_way': 5, 'n_support': 8, 'n_query': 15},
        {'n_way': 8, 'n_support': 5, 'n_query': 15},  # 使用所有8个类别
        {'n_way': 8, 'n_support': 8, 'n_query': 15},
    ]
    
    # 创建模型
    model = ProtoNet(
        in_channels=5,
        hidden_dim=64,
        feature_dim=128,
        backbone='cnn1d',
        distance_type='euclidean'
    ).to(device)
    
    # 加载预训练模型
    model_path = "runs/protonet_20250219_143734/best_model_0.8965.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n=== ProtoNet Benchmark 测试结果 ===")
    print(f"模型路径: {model_path}")
    print("\n各配置测试结果：")
    
    # 测试每个配置
    for config in test_configs:
        print(f"\n{config['n_way']}-way {config['n_support']}-shot测试:")
        
        # 创建测试数据集
        test_dataset = ProtoNetDataset(
            base_dataset=base_dataset,
            n_way=config['n_way'],
            n_support=config['n_support'],
            n_query=config['n_query']
        )
        
        # 创建数据加载器
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # 评估
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, device)
        
        # 打印结果
        print(f"平均损失: {test_loss:.4f}")
        print(f"平均准确率: {test_acc:.4f}")
        
        # 计算每个类别的准确率
        print("\n各类别准确率:")
        for i in range(config['n_way']):
            class_mask = labels == i
            n_samples = np.sum(class_mask)
            if n_samples > 0:
                class_acc = (preds[class_mask] == labels[class_mask]).mean()
                print(f"类别 {i}: {class_acc:.4f} (样本数: {n_samples})")
            else:
                print(f"类别 {i}: 无样本")
        print("-" * 50)
    
    
