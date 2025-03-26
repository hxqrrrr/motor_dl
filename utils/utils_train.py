import torch
import numpy as np
import h5py
from tqdm import tqdm
from models.dataset import h5Dataset
from models.ProtoNet import ProtoNet
from models.ProtoNet_backbone import ProtoNetWithAttention
from models.ProtoNet_all_model import  AllModel
import matplotlib.pyplot as plt
import os
import json


def get_model(model_name, in_channels, hidden_dim, feature_dim, backbone, distance_type, dropout=0.5, visualize=True, no_attention_vis=False):
    """
    根据模型名称返回相应的模型实例
    
    参数:
        model_name: 模型名称，可选['protonet', 'protonet_attention', 'all_model']
        in_channels: 输入通道数
        hidden_dim: 隐藏层维度
        feature_dim: 特征维度
        backbone: 骨干网络类型，可选['cnn1d', 'channel', 'spatial', 'cbam','transformer']
        distance_type: 距离度量类型，可选['euclidean', 'cosine', 'relation', 'relation_selfattention']
        dropout: Dropout比率
        visualize (bool): 是否启用注意力可视化
        no_attention_vis (bool): 是否禁用注意力可视化
    
    返回:
        model: 模型实例
    """
    model_dict = {

        'protonet_attention': ProtoNetWithAttention,
        'all_model': AllModel
    }
    
    if model_name not in model_dict:
        raise ValueError(f"不支持的模型名称: {model_name}，可用的模型有: {list(model_dict.keys())}")
    
    # 检查距离类型是否有效
    valid_distance_types = ['euclidean', 'cosine', 'relation', 'relation_selfattention']
    if distance_type not in valid_distance_types:
        raise ValueError(f"不支持的距离类型: {distance_type}，可用的距离类型有: {valid_distance_types}")
    
    if backbone in ['channel', 'spatial', 'cbam', 'enhanced_cnn1d','transformer']:
        model = ProtoNetWithAttention(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            attention_type=backbone,
            backbone='cnn1d',
            distance_type=distance_type,
            dropout=dropout
        )
        
        # 在创建模型后设置注意力模块的可视化
        if hasattr(model.encoder, 'attention'):
            model.encoder.attention.visualize = visualize
        if hasattr(model.encoder, 'channel_attention'):
            model.encoder.channel_attention.visualize = visualize
        if hasattr(model.encoder, 'spatial_attention'):
            model.encoder.spatial_attention.visualize = visualize
    else:
        print("backbone not supported")
    
    return model

def train_epoch(model, train_loader, optimizer, device, n_way):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_acc = 0
    total_recall = 0
    total_f1 = 0
    n_batches = len(train_loader)
    
    # 只在第一个batch进行可视化
    first_batch = True
    
    with tqdm(train_loader, desc="训练", ncols=100, leave=True) as pbar:
        for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(pbar):
            # 临时关闭除第一个batch外的所有可视化
            if not first_batch:
                def disable_visualization(module):
                    if hasattr(module, 'visualize'):
                        module.visualize = False
                model.apply(disable_visualization)
            
            # 将数据移到设备上
            support_x = support_x.to(device)  # [batch_size, n_support, channels, length]
            support_y = support_y.to(device)  # [batch_size, n_support]
            query_x = query_x.to(device)      # [batch_size, n_query, channels, length]
            query_y = query_y.to(device)      # [batch_size, n_query]

            
            optimizer.zero_grad()
            logits = model(support_x, support_y, query_x)  # [batch_size, n_query, n_way]
            

            
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                query_y.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            
            # 计算指标
            pred = logits.argmax(dim=-1)  # [batch_size, n_query]
            acc = (pred == query_y).float().mean().item()
            
            # 计算每个类别的TP、FP、FN
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            for c in range(n_way):
                pred_c = pred == c
                target_c = query_y == c
                tp = (pred_c & target_c).sum().item()
                fp = (pred_c & ~target_c).sum().item()
                fn = (~pred_c & target_c).sum().item()
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            # 计算召回率和F1分数
            recall = total_tp / (total_tp + total_fn + 1e-8)
            precision = total_tp / (total_tp + total_fp + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            total_loss += loss.item()
            total_acc += acc
            total_recall += recall
            total_f1 += f1
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(
                f"loss: {loss.item():.3f}, acc: {acc:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, lr: {current_lr:.2e}"
            )
            
            # 恢复可视化设置
            if not first_batch:
                def restore_visualization(module):
                    if hasattr(module, 'visualize'):
                        module.visualize = True
                model.apply(restore_visualization)
            
            first_batch = False
    
    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    avg_recall = total_recall / n_batches
    avg_f1 = total_f1 / n_batches
    
    return avg_loss, avg_acc, avg_recall, avg_f1

def evaluate(model, val_loader, device, n_way):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_acc = 0
    total_recall = 0
    total_f1 = 0
    n_batches = len(val_loader)
    
    # 临时关闭所有可视化
    def disable_visualization(module):
        if hasattr(module, 'visualize'):
            module._visualize_backup = module.visualize  # 备份当前状态
            module.visualize = False
    model.apply(disable_visualization)
    
    with torch.no_grad():
        with tqdm(val_loader, desc="评估", ncols=100, leave=True) as pbar:
            for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(pbar):
                support_x = support_x.to(device)  # [batch_size, n_support, channels, length]
                support_y = support_y.to(device)  # [batch_size, n_support]
                query_x = query_x.to(device)      # [batch_size, n_query, channels, length]
                query_y = query_y.to(device)      # [batch_size, n_query]
                
                logits = model(support_x, support_y, query_x)  # [batch_size, n_query, n_way]
                
                # 重塑维度以计算损失
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    query_y.reshape(-1)
                )
                
                # 计算指标
                pred = logits.argmax(dim=-1)  # [batch_size, n_query]
                acc = (pred == query_y).float().mean().item()
                
                # 计算每个类别的TP、FP、FN
                total_tp = 0
                total_fp = 0
                total_fn = 0
                
                for c in range(n_way):
                    pred_c = pred == c
                    target_c = query_y == c
                    tp = (pred_c & target_c).sum().item()
                    fp = (pred_c & ~target_c).sum().item()
                    fn = (~pred_c & target_c).sum().item()
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                
                # 计算召回率和F1分数
                recall = total_tp / (total_tp + total_fn + 1e-8)
                precision = total_tp / (total_tp + total_fp + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                total_loss += loss.item()
                total_acc += acc
                total_recall += recall
                total_f1 += f1
                
                pbar.set_postfix_str(
                    f"loss: {loss.item():.3f}, acc: {acc:.3f}, recall: {recall:.3f}, f1: {f1:.3f}"
                )
    
    # 恢复可视化设置
    def restore_visualization(module):
        if hasattr(module, '_visualize_backup'):
            module.visualize = module._visualize_backup
            delattr(module, '_visualize_backup')
    model.apply(restore_visualization)
    
    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    avg_recall = total_recall / n_batches
    avg_f1 = total_f1 / n_batches
    
    return avg_loss, avg_acc, avg_recall, avg_f1

def check_data_leakage(train_base_dataset,val_base_dataset):
    """检查训练集和验证集之间是否存在数据泄露"""
    print("\n正在检查数据泄露...")
    # 获取训练集和验证集的所有样本标识符
    train_samples = set(train_base_dataset.get_all_samples())
    val_samples = set(val_base_dataset.get_all_samples())
    
    # 检查交集
    intersection = train_samples.intersection(val_samples)
    if len(intersection) > 0:
        print(f"警告：发现数据泄露！训练集和验证集有{len(intersection)}个重复样本")
        print("重复样本示例：")
        for sample in list(intersection)[:5]:  # 只显示前5个重复样本
            print(f"  - {sample}")
        return True
    print("数据集检查通过，未发现数据泄露。")
    return False

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, epochs, save_dir):
    """
    绘制训练过程的损失和准确率曲线。
    
    Args:
        train_losses (list): 训练损失列表
        train_accs (list): 训练准确率列表
        val_losses (list): 验证损失列表
        val_accs (list): 验证准确率列表
        epochs (list): 验证轮次列表
        save_dir (str): 保存图表的目录路径
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def save_training_info(model_info, training_params, save_dir, is_best_model=False):
    """
    保存训练信息和模型。
    
    Args:
        model_info (dict): 包含模型状态等信息的字典
        training_params (dict): 训练参数字典
        save_dir (str): 保存目录路径
        is_best_model (bool): 是否为最佳模型，默认为False
    """
    # 保存训练参数到JSON文件
    params_file = os.path.join(save_dir, 'training_params.json')
    with open(params_file, 'w') as f:
        json.dump(training_params, f, indent=4)
    
    # 确定模型保存路径
    if is_best_model:
        model_path = os.path.join(save_dir, f'best_model_val_acc_{model_info["best_val_acc"]:.4f}.pth')
    else:
        model_path = os.path.join(save_dir, 'final_model.pth')
    
    # 保存模型
    save_dict = {
        'epoch': model_info['epoch'],
        'model_state_dict': model_info['model_state_dict'],
        'optimizer_state_dict': model_info['optimizer_state_dict'],
        'scheduler_state_dict': model_info['scheduler_state_dict'],
        'training_params': training_params
    }
    
    # 添加验证准确率信息
    if is_best_model:
        save_dict['best_val_acc'] = model_info['best_val_acc']
    else:
        save_dict['final_val_acc'] = model_info['val_accs']
    
    torch.save(save_dict, model_path)
    return params_file

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