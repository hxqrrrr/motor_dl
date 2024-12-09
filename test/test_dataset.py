import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dataset import h5Dataset, KATDataset
import torch
from torch.utils.data import DataLoader

def test_h5_dataset(folder_path):
    """测试h5数据集的读取"""
    print("\n开始测试h5数据集...")
    try:
        # 创建数据集实例
        dataset = h5Dataset(folder_path)
        print(f"数据集大小: {len(dataset)}")
        
        # 测试数据加载器
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # 获取一个批次的数据
        batch_data, batch_labels = next(iter(dataloader))
        
        print(f"批次数据形状: {batch_data.shape}")
        print(f"批次标签形状: {batch_labels.shape}")
        print(f"数据类型: {batch_data.dtype}")
        print(f"标签类型: {batch_labels.dtype}")
        print(f"标签值示例: {batch_labels}")
        
    except Exception as e:
        print(f"h5数据集测试失败: {str(e)}")

def test_kat_dataset(folder_path):
    """测试KAT数据集的读取"""
    print("\n开始测试KAT数据集...")
    try:
        # 创建数据集实例
        dataset = KATDataset(folder_path)
        print(f"数据集大小: {len(dataset)}")
        
        # 测试数据加载器
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # 获取一个批次的数据
        batch_data, batch_labels = next(iter(dataloader))
        
        print(f"批次数据形状: {batch_data.shape}")
        print(f"批次标签形状: {batch_labels.shape}")
        print(f"数据类型: {batch_data.dtype}")
        print(f"标签类型: {batch_labels.dtype}")
        print(f"标签值示例: {batch_labels}")
        
    except Exception as e:
        print(f"KAT数据集测试失败: {str(e)}")

if __name__ == "__main__":
    # 测试h5数据集
    h5_folder_path = "data/h5data"
    test_h5_dataset(h5_folder_path)
    
    # 测试KAT数据集
    kat_folder_path = "data/KAT"
    test_kat_dataset(kat_folder_path)

