import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.ProtoNet import ProtoNet
from models.dataset import ProtoNetDataset, h5Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from utils.utils import check_data_leakage, train_epoch, evaluate, get_model


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ProtoNet Benchmark')
    parser.add_argument('--model_name', type=str, default='protonet_attention', help='模型名称')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_path', type=str, default='data/h5data', help='数据集路径')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 加载数据集
    h5_folder_path = 'data/h5data'
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
    
    # 创建并加载模型
    model = get_model(
        model_name=args.model_name,
        in_channels=5,
        hidden_dim=64,
        feature_dim=128,
        backbone='cnn1d',
        distance_type='euclidean'
    ).to(device)
    
    # 加载预训练模型
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n=== ProtoNet Benchmark 测试结果 ===")
    print(f"模型名称: {args.model_name}")
    print(f"模型路径: {args.model_path}")
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
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        # 打印结果
        print(f"平均损失: {test_loss:.4f}")
        print(f"平均准确率: {test_acc:.4f}")
        print("-" * 50)
    
    
