import sys
import os
import argparse
import json
from datetime import datetime
import re
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


def load_model_params(json_path):
    """从 JSON 文件中提取模型参数"""
    with open(json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    hidden_dim = params.get('hidden_dim', 64)  # 默认值为64
    feature_dim = params.get('feature_dim', 128)  # 默认值为128
    return hidden_dim, feature_dim

def find_best_model_paths(runs_dir):
    best_model_paths = []

    # 遍历 runs 目录下的每个二级文件夹
    for model_dir in os.listdir(runs_dir):
        model_path = os.path.join(runs_dir, model_dir)
        
        if os.path.isdir(model_path):
            max_accuracy = -1
            best_model_path = None
            
            # 遍历该二级文件夹中的所有文件
            for file in os.listdir(model_path):
                if file.endswith('.pth'):
                    # 使用正则表达式提取准确度
                    match = re.search(r'_(\d+\.\d+)\.pth$', file)
                    if match:
                        accuracy = float(match.group(1))
                        # 更新最大准确度和对应的文件路径
                        if accuracy > max_accuracy:
                            max_accuracy = accuracy
                            best_model_path = os.path.join(model_path, file)
            
            if best_model_path:
                best_model_paths.append(best_model_path)

    return best_model_paths

def test_model(model_path, data_path, model_name, device):
    # 从模型路径中提取 training_params.json 的路径
    training_params_path = os.path.join(os.path.dirname(model_path), 'training_params.json')
    
    # 加载模型参数
    hidden_dim, feature_dim = load_model_params(training_params_path)

    # 创建并加载模型
    model = get_model(
        model_name=model_name,
        in_channels=5,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        backbone='cbam',
        distance_type='euclidean',
        dropout=0.3
    ).to(device)

    # 加载预训练模型
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载数据集
    base_dataset = h5Dataset(folder_path=data_path, train=False)  # 使用测试集进行评估

    # 测试不同配置
    test_configs = [
        {'n_way': 5, 'n_support': 5, 'n_query': 15},
        {'n_way': 5, 'n_support': 8, 'n_query': 15},
        {'n_way': 8, 'n_support': 5, 'n_query': 15},
        {'n_way': 8, 'n_support': 8, 'n_query': 15},
    ]

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
        test_loss, test_acc, test_recall, test_f1 = evaluate(model, test_loader, device)
        
        # 打印结果
        print(f"平均损失: {test_loss:.4f}")
        print(f"平均准确率: {test_acc:.4f}")
        print(f"平均召回率: {test_recall:.4f}")
        print(f"平均F1分数: {test_f1:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ProtoNet Benchmark')
    parser.add_argument('--model_name', type=str, default='all_model', help='模型名称')
    parser.add_argument('--model_path', type=str, help='模型路径')  # 这里不再是required
    parser.add_argument('--data_path', type=str, default='data/h5data', help='数据集路径')
    parser.add_argument('--test_all_models', action='store_true', help='是否测试所有模型')
    args = parser.parse_args()
    # 使用示例
    runs_directory = 'runs'  # 请根据实际路径修改
    # 创建benchmark文件夹
    os.makedirs('benchmark', exist_ok=True)
    
    # 获取当前时间作为结果文件的标识
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备结果字典
    results = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "timestamp": timestamp,
        "configs": []
    }

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
    
    if args.test_all_models:
        best_model_paths = find_best_model_paths(runs_directory)
        for model_path in best_model_paths:
            print(f"正在测试模型: {model_path}")
            test_model(model_path, args.data_path, args.model_name, device)
    elif args.model_path:
        print(f"使用提供的模型路径进行测试: {args.model_path}")
        test_model(args.model_path, args.data_path, args.model_name, device)
    else:
        print("请提供模型路径或使用 --test_all_models 参数进行测试所有模型。")
    
    
