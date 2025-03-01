import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score
import seaborn as sns
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ProtoNet import ProtoNet
from models.ProtoNet_attention import ProtoNetWithAttention
from models.dataset import ProtoNetDataset

def get_model(model_name, in_channels, hidden_dim, feature_dim, backbone, distance_type, dropout=0.3):
    """
    根据模型名称返回相应的模型实例
    """
    model_dict = {
        'protonet': ProtoNet,
        'protonet_attention': ProtoNetWithAttention
    }
    
    if model_name not in model_dict:
        raise ValueError(f"不支持的模型名称: {model_name}。支持的模型有: {list(model_dict.keys())}")
    
    if model_name == 'protonet':
        return model_dict[model_name](
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            backbone=backbone,
            distance_type=distance_type
        )
    else:
        return model_dict[model_name](
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            backbone=backbone,
            distance_type=distance_type,
            dropout=dropout
        )

class SourceDomainDataset(Dataset):
    """源域数据集类"""
    def __init__(self, h5_file_path, indices=None):
        self.h5_file_path = h5_file_path
        
        # 加载数据
        with h5py.File(h5_file_path, 'r') as f:
            self.data = f['data'][:]
            self.labels = f['labels'][:]
            
            # 打印数据信息
            print(f"数据形状: {self.data.shape}")
            print(f"标签形状: {self.labels.shape}")
            print(f"唯一标签: {np.unique(self.labels)}")
        
        # 如果提供了索引，则只使用这些索引对应的数据
        self.indices = indices if indices is not None else np.arange(len(self.data))
        
        print(f"数据集大小: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        idx = self.indices[index]
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels[idx]


def split_dataset(data_path, train_ratio=0.8, seed=42):
    """
    将数据集划分为训练集和测试集，确保只进行一次划分
    
    参数:
        data_path: 数据文件路径
        train_ratio: 训练集比例
        seed: 随机种子
    
    返回:
        train_indices: 训练集索引
        test_indices: 测试集索引
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 加载数据
    with h5py.File(data_path, 'r') as f:
        labels = f['labels'][:]
    
    # 分层采样
    unique_labels = np.unique(labels)
    train_indices = []
    test_indices = []
    
    print("\n数据集划分情况:")
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)  # 随机打乱
        split_idx = int(len(label_indices) * train_ratio)
        
        train_indices.extend(label_indices[:split_idx])
        test_indices.extend(label_indices[split_idx:])
        
        print(f"类别 {label}:")
        print(f"  - 总样本数: {len(label_indices)}")
        print(f"  - 训练集: {split_idx} 个样本")
        print(f"  - 测试集: {len(label_indices) - split_idx} 个样本")
    
    # 随机打乱索引
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    print(f"\n训练集大小: {len(train_indices)}")
    print(f"测试集大小: {len(test_indices)}")
    
    return train_indices, test_indices

def evaluate_model(model, dataloader, device, n_way):
    """评估模型并返回详细指标"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (support_images, support_labels, query_images, query_labels) in enumerate(tqdm(dataloader, desc="评估中")):
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            # 前向传播
            logits = model(support_images, support_labels, query_images)
            
            # 计算预测结果
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=range(n_way))
    
    # 生成分类报告
    report = classification_report(all_labels, all_preds, labels=range(n_way), zero_division=0)
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'true_labels': all_labels
    }

def plot_confusion_matrix(cm, n_way, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_way), yticklabels=range(n_way))
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='源域元学习模型测试脚本')
    parser.add_argument('--model_path', type=str, required=True,
                      help='模型路径，例如: runs/source_domain_protonet_cnn1d_euclidean_20240226_194815/best_model_val_acc_0.8670.pth')
    parser.add_argument('--test_data', type=str, default='data/h5data/selected_data.h5',
                      help='测试数据文件路径')
    parser.add_argument('--n_way', type=int, default=4,
                      help='N-way分类任务中的类别数')
    parser.add_argument('--n_support', type=int, default=5,
                      help='每个类别的支持集样本数')
    parser.add_argument('--n_query', type=int, default=15,
                      help='每个类别的查询集样本数')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='批次大小')
    parser.add_argument('--num_test_episodes', type=int, default=100,
                      help='测试的episode数量')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                      help='训练集比例，用于确保与训练时使用相同的数据划分')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子，用于确保与训练时使用相同的数据划分')
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        return
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path)
    
    # 获取模型参数
    model_params = {
        'model_name': checkpoint.get('model_name', 'protonet'),
        'in_channels': checkpoint.get('in_channels', 5),
        'hidden_dim': checkpoint.get('hidden_dim', 64),
        'feature_dim': checkpoint.get('feature_dim', 64),
        'backbone': checkpoint.get('backbone', 'cnn1d'),
        'distance_type': checkpoint.get('distance_type', 'euclidean'),
        'dropout': checkpoint.get('dropout', 0.3)
    }
    
    print("模型参数:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = get_model(
        model_name=model_params['model_name'],
        in_channels=model_params['in_channels'],
        hidden_dim=model_params['hidden_dim'],
        feature_dim=model_params['feature_dim'],
        backbone=model_params['backbone'],
        distance_type=model_params['distance_type'],
        dropout=model_params['dropout']
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 加载测试数据并进行划分
    print(f"加载测试数据并进行划分: {args.test_data}")
    _, test_indices = split_dataset(
        data_path=args.test_data,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # 创建测试数据集，只使用测试集索引
    test_base_dataset = SourceDomainDataset(h5_file_path=args.test_data, indices=test_indices)
    
    # 创建测试数据集
    test_dataset = ProtoNetDataset(
        base_dataset=test_base_dataset,
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        num_episodes=args.num_test_episodes
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 评估模型
    print("开始评估模型...")
    results = evaluate_model(model, test_loader, device, args.n_way)
    
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/source_domain_test_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存结果
    print("\n评估结果:")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")
    print("\n分类报告:")
    print(results['classification_report'])
    
    # 绘制混淆矩阵
    plot_confusion_matrix(results['confusion_matrix'], args.n_way, f"{results_dir}/confusion_matrix.png")
    
    # 保存结果到文件
    with open(f"{results_dir}/results.txt", 'w') as f:
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"测试数据: {args.test_data}\n")
        f.write(f"N-way: {args.n_way}\n")
        f.write(f"N-support: {args.n_support}\n")
        f.write(f"N-query: {args.n_query}\n")
        f.write(f"测试episodes数量: {args.num_test_episodes}\n\n")
        
        f.write("模型参数:\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n评估结果:\n")
        f.write(f"准确率: {results['accuracy']:.4f}\n")
        f.write(f"召回率: {results['recall']:.4f}\n")
        f.write(f"F1分数: {results['f1']:.4f}\n\n")
        
        f.write("分类报告:\n")
        f.write(results['classification_report'])
    
    print(f"\n结果已保存到: {results_dir}")

if __name__ == "__main__":
    main() 