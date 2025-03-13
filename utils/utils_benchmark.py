import os
import re
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils_train import get_model
from models.dataset import SourceDomainDataset, h5Dataset, ProtoNetDataset
from utils.utils_train import split_dataset
from datetime import datetime
from torch.utils.data import DataLoader
from utils.utils_train import evaluate

def load_model_params(model_path):
    """从模型文件和训练参数JSON文件中提取模型参数"""
    # 加载模型文件
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    training_params = checkpoint.get('training_params', {})
    
    # 尝试从模型文件中直接获取参数
    model_params = {
        'in_channels': training_params.get('in_channels', 5),
        'hidden_dim': training_params.get('hidden_dim', 16),
        'feature_dim': training_params.get('feature_dim', 128),
        'backbone': training_params.get('backbone', 'cnn1d'),
        'distance_type': training_params.get('distance_type', 'euclidean'),
        'model': training_params.get('model', 'all_model')
    }
    
    # 尝试从训练参数文件中获取更多信息
    training_params_path = os.path.join(os.path.dirname(model_path), 'training_params.json')
    if os.path.exists(training_params_path):
        with open(training_params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
            
        # 更新模型参数
        model_params.update({
            'in_channels': params.get('in_channels', model_params['in_channels']),
            'hidden_dim': params.get('hidden_dim', model_params['hidden_dim']),
            'feature_dim': params.get('feature_dim', model_params['feature_dim']),
            'backbone': params.get('backbone', model_params['backbone']),
            'distance_type': params.get('distance_type', model_params['distance_type']),
            'model': params.get('model', model_params['model'])
        })
    
    return model_params

def find_best_model_paths(runs_dir):
    """查找runs目录下每个模型文件夹中的最佳模型"""
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

def test_model(
    model_path, 
    target_data_path, 
    device, 
    test_configs, 
    num_episodes, 
    batch_size=4, 
    save_dir='benchmark', 
    optimize_memory=False,
    visualize=True
):
    """测试模型在目标域上的性能"""
    print(f"\n正在测试模型: {model_path}")
    
    # 加载模型参数
    model_params = load_model_params(model_path)
    
    print("模型参数:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    # 加载模型
    checkpoint = torch.load(model_path)
    training_params = checkpoint.get('training_params', {})
    
    model = get_model(
        model_name=model_params['model'],
        in_channels=model_params['in_channels'],
        hidden_dim=model_params['hidden_dim'],
        feature_dim=model_params['feature_dim'],
        backbone=model_params['backbone'],
        distance_type=model_params['distance_type'],
        visualize=visualize
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 如果使用GPU，启用半精度浮点数以节省内存
    if device.type == 'cuda' and optimize_memory:
        model = model.half()  # 转换为FP16
        print("已启用FP16半精度计算以优化GPU内存使用")

    # 加载目标域数据
    print(f"加载目标域数据: {target_data_path}")
    
    # 检查目标数据是文件还是文件夹
    if os.path.isfile(target_data_path) and target_data_path.endswith(('.h5', '.h5data')):
        # 如果是单个h5文件，使用SourceDomainDataset
        train_indices, test_indices = split_dataset(
            data_path=target_data_path,
            train_ratio=0.2,  # 在目标域上，我们主要关注测试性能，所以只用20%作为训练集
            seed=42
        )
        base_dataset = SourceDomainDataset(h5_file_path=target_data_path, indices=test_indices)
    else:
        # 如果是文件夹，使用h5Dataset
        base_dataset = h5Dataset(folder_path=target_data_path, train=False)
    
    # 创建结果目录
    if save_dir is None:
        # 获取模型完整路径，去掉"runs"部分
        model_full_path = model_path
        if "runs/" in model_full_path:
            model_save_path = model_full_path.split("runs/")[1]
        elif "runs\\" in model_full_path:
            model_save_path = model_full_path.split("runs\\")[1]
        else:
            model_save_path = model_full_path
            
        save_dir = f'benchmark/{model_save_path}'
    
    # 确保目录存在，包括所有父目录
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存测试结果
    results = {
        "model_path": model_path,
        "target_data": target_data_path,
        "model_params": model_params,
        "configs": []
    }

    # 测试每个配置
    for config in test_configs:
        print(f"\n{config['n_way']}-way {config['n_support']}-shot测试:")
        
        # 创建测试数据集
        test_dataset = ProtoNetDataset(
            base_dataset=base_dataset,
            n_way=config['n_way'],
            n_support=config['n_support'],
            n_query=config['n_query'],
            num_episodes=num_episodes
        )
        
        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=torch.cuda.is_available() and True
        )
        
        # 评估 - 添加n_way参数
        test_loss, test_acc, test_recall, test_f1 = evaluate(
            model, 
            test_loader, 
            device,
            config['n_way']  # 添加这个参数
        )
        
        # 打印结果
        print(f"平均损失: {test_loss:.4f}")
        print(f"平均准确率: {test_acc:.4f}")
        print(f"平均召回率: {test_recall:.4f}")
        print(f"平均F1分数: {test_f1:.4f}")
        
        # 收集结果
        config_result = {
            "n_way": config['n_way'],
            "n_support": config['n_support'],
            "n_query": config['n_query'],
            "loss": test_loss,
            "accuracy": test_acc,
            "recall": test_recall,
            "f1": test_f1
        }
        
        results["configs"].append(config_result)
        
        print("-" * 50)
        
        # 如果使用GPU，清理缓存
        if device.type == 'cuda' and optimize_memory:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_allocated'):
                print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # 保存结果到JSON文件
    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"测试结果已保存到: {save_dir}")
    return results