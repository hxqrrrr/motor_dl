import sys
import os
import argparse
import json
from datetime import datetime
import re
import h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.ProtoNet import ProtoNet
from models.dataset import ProtoNetDataset, h5Dataset, SourceDomainDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from utils.utils_train import check_data_leakage, train_epoch, evaluate, get_model, split_dataset
from utils.utils_benchmark import find_best_model_paths, test_model


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='目标域元学习模型测试脚本')
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--target_data', type=str, default='data/h5data/remaining_data.h5', help='目标域数据路径 (h5文件或文件夹)')
    parser.add_argument('--model_name', type=str, default='all_model', help='模型名称')
    parser.add_argument('--test_all_models', action='store_true', help='是否测试所有模型')
    parser.add_argument('--num_episodes', type=int, default=100, help='测试的episode数量')
    parser.add_argument('--force_cpu', action='store_true', help='强制使用CPU，即使有GPU可用')
    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小，GPU模式下可以设置更大')
    parser.add_argument('--optimize_memory', action='store_true', help='优化GPU内存使用')
    args = parser.parse_args()
    
    # 创建benchmark文件夹
    os.makedirs('benchmark', exist_ok=True)
    
    # 获取当前时间作为结果文件的标识
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("\n警告: 未检测到可用的CUDA设备。如果您想使用GPU，请确保：")
        print("1. 您的系统有NVIDIA GPU")
        print("2. 已正确安装CUDA和cuDNN")
        print("3. 已安装支持CUDA的PyTorch版本")
        print("当前将使用CPU进行测试，这可能会很慢。\n")
    
    # 设置设备
    if args.force_cpu:
        device = torch.device('cpu')
        print(f"已强制使用CPU")
    else:
        device = torch.device('cuda' if cuda_available else 'cpu')
        if cuda_available:
            print(f"使用设备: GPU ({torch.cuda.get_device_name(0)})")
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 显示GPU内存信息
            if hasattr(torch.cuda, 'memory_allocated'):
                print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # 如果启用了内存优化
            if args.optimize_memory:
                print("已启用GPU内存优化")
                # 设置内存分配器
                if hasattr(torch.cuda, 'memory_stats'):
                    torch.cuda.memory_stats(device=None)
                # 使用确定性算法
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                # 启用cudnn benchmark以提高性能
                torch.backends.cudnn.benchmark = True
        else:
            print(f"使用设备: CPU")
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 测试配置
    test_configs = [
        {'n_way': 4, 'n_support': 1, 'n_query': 15},
        {'n_way': 4, 'n_support': 5, 'n_query': 15},
        {'n_way': 4, 'n_support': 10, 'n_query': 15},
        {'n_way': 4, 'n_support': 20, 'n_query': 15},
    ]
    
    # 创建保存目录
    save_dir = f'benchmark/target_domain_test_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    if args.test_all_models:
        # 测试所有模型
        runs_directory = 'runs'
        best_model_paths = find_best_model_paths(runs_directory)
        
        if not best_model_paths:
            print("未找到任何模型文件！")
            sys.exit(1)
        
        print(f"找到 {len(best_model_paths)} 个模型进行测试")
        
        for model_path in best_model_paths:
            # 获取模型完整路径，去掉"runs"部分
            model_full_path = model_path
            if "runs/" in model_full_path:
                model_save_path = model_full_path.split("runs/")[1]
            elif "runs\\" in model_full_path:
                model_save_path = model_full_path.split("runs\\")[1]
            else:
                model_save_path = model_full_path
                
            # 直接使用模型保存路径，不需要额外的目录
            model_results = test_model(
                model_path=model_path,
                target_data_path=args.target_data,
                device=device,
                test_configs=test_configs,
                num_episodes=args.num_episodes,
                batch_size=args.batch_size,
                save_dir=f'benchmark/{model_save_path}',
                optimize_memory=args.optimize_memory
            )
            all_results.append(model_results)
    
    elif args.model_path:
        # 测试单个模型
        # 获取模型完整路径，去掉"runs"部分
        model_full_path = args.model_path
        if "runs/" in model_full_path:
            model_save_path = model_full_path.split("runs/")[1]
        elif "runs\\" in model_full_path:
            model_save_path = model_full_path.split("runs\\")[1]
        else:
            model_save_path = model_full_path
            
        model_results = test_model(
            model_path=args.model_path,
            target_data_path=args.target_data,
            device=device,
            test_configs=test_configs,
            num_episodes=args.num_episodes,
            batch_size=args.batch_size,
            save_dir=f'benchmark/{model_save_path}',
            optimize_memory=args.optimize_memory
        )
        all_results.append(model_results)
    
    else:
        print("请提供模型路径或使用 --test_all_models 参数进行测试所有模型。")
        sys.exit(1)
    
    # 保存所有结果的汇总
    with open(os.path.join(save_dir, 'all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n所有测试完成，结果已保存到: {save_dir}")
    
    
