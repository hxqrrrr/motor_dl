import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dataset import h5Dataset, KATDataset, ProtoNetDataset
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
        dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
        
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
def test_proto_dataset(folder_path):
    # 首先创建基础数据集
    base_dataset = h5Dataset(folder_path=folder_path)

    # 创建ProtoNet数据集
    protonet_dataset = ProtoNetDataset(
        base_dataset=base_dataset,
        n_way=5,           
        n_support=8,       
        n_query=15         
    )

    # 创建数据加载器
    dataloader = DataLoader(
        protonet_dataset,
        batch_size=1,      # 通常设为1，因为每个batch就是一个完整的episode
        shuffle=True
    )

    # 使用数据加载器
    for support_x, support_y, query_x, query_y in dataloader:
        # support_x: [1, n_way * n_support, ...]
        # support_y: [1, n_way * n_support]
        # query_x:   [1, n_way * n_query, ...]
        # query_y:   [1, n_way * n_query]
        print(f"支持集数据形状: {support_x.shape}")
        print(f"支持集标签形状: {support_y.shape}")
        print(f"查询集数据形状: {query_x.shape}")
        print(f"查询集标签形状: {query_y.shape}")
if __name__ == "__main__":
    # 测试h5数据集
    h5_folder_path = "data/h5data"
    
    # 测试KAT数据集
    kat_folder_path = "data/katdata"
    
    # 测试ProtoNet数据集
    test_h5_dataset(h5_folder_path)

