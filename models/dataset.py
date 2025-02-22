import h5py
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os
from data.dataloader.cwru_dataset import CWRUDataset
import numpy as np

# DataLoader
class h5Dataset(Dataset):
    def __init__(self, folder_path, train=True, train_ratio=0.8, seq_len=2048):
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.train = train
        print(f"正在加载文件夹: {folder_path}")
        
        # 获取h5文件列表
        self.h5_files = [f for f in os.listdir(folder_path) if f.endswith(('.h5', '.h5data'))]
        if not self.h5_files:
            raise ValueError(f"在 {folder_path} 中没有找到任何 .h5 或 .h5data 文件")
        
        # 加载数据
        file_path = os.path.join(folder_path, self.h5_files[0])
        with h5py.File(file_path, 'r') as f:
            self.data = f['data'][:]
            self.labels = f['labels'][:]
            # 打印标签信息
            print("\n原始标签形状:", self.labels.shape)
            print("标签示例（前5个）:")
            print(self.labels[:5])
            if 'labelName' in f:
                label_names = f['labelName'].asstr()[:]
                print("\n标签名称:", label_names)
        
        # 将one-hot标签转换为类别索引
        self.labels = torch.argmax(torch.tensor(self.labels, dtype=torch.float32), dim=1).numpy()
        
        # 分层采样
        unique_labels = np.unique(self.labels)
        train_indices = []
        test_indices = []
        
        print("\n数据集划分情况:")
        for label in unique_labels:
            label_indices = np.where(self.labels == label)[0]
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
        
        # 计算标准化参数（只使用训练数据）
        if train:
            self.mean = np.mean(self.data, axis=0, keepdims=True)
            self.std = np.std(self.data, axis=0, keepdims=True) + 1e-10
        else:
            # 如果是测试集，使用原始数据计算训练集的统计信息
            train_data = self.data[train_indices]
            self.mean = np.mean(train_data, axis=0, keepdims=True)
            self.std = np.std(train_data, axis=0, keepdims=True) + 1e-10
        
        # 对数据进行标准化
        self.data = (self.data - self.mean) / self.std
        
        # 选择相应的数据集
        if train:
            self.data = self.data[train_indices]
            self.labels = self.labels[train_indices]
        else:
            self.data = self.data[test_indices]
            self.labels = self.labels[test_indices]
        
        # 将标签转换为tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        print(f"\n{('训练' if train else '测试')}集信息:")
        unique_labels = torch.unique(self.labels)
        print(f"包含的类别: {unique_labels.tolist()}")
        for label in unique_labels:
            count = (self.labels == label).sum().item()
            print(f"类别 {label}: {count} 个样本")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32), \
               self.labels[index].clone().detach()


class KATDataset(Dataset):
    def __init__(self, folder_path):
        """
        初始化KAT数据集
        Args:
            folder_path (str): 文件夹路径
        """
        self.folder_path = folder_path
        print(f"加载文件夹: {folder_path}")
        # 遍历文件夹，找到所有.mat结尾的文件
        self.mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        
    def __len__(self):
        """返回数据集中的样本总数"""
        print(f"数据集中的样本总数: {len(self.mat_files)}")
        return len(self.mat_files)
    
    def __getitem__(self, index):
        """
        获取指定索引的样本和标签
        Args:
            index (int): 样本索引
        Returns:
            tuple: (样本数据, 标签)
        """
        # 根据索引获取.mat文件的路径
        file_path = os.path.join(self.folder_path, self.mat_files[index])
        # 加载.mat文件
        data = sio.loadmat(file_path)
        # 假设.mat文件中的数据和标签的键名分别为'data'和'label'
        sample = torch.tensor(data['data'], dtype=torch.float32)  # 预期形状: (样本数, 通道数, 序列长度)
        label = torch.tensor(data['label'], dtype=torch.long)  # 预期形状: (样本数,)
        
        return sample, label


class ProtoNetDataset(Dataset):
    def __init__(self, base_dataset, n_way, n_support, n_query):
        """
        ProtoNet数据集包装器
        
        参数:
            base_dataset: 基础数据集 (h5Dataset 或 KATDataset)
            n_way: 每个episode中的类别数
            n_support: 每个类别的support样本数
            n_query: 每个类别的query样本数
        """
        self.base_dataset = base_dataset
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        
        # 按类别组织数据
        self.label_to_indices = {}
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            if label.item() not in self.label_to_indices:
                self.label_to_indices[label.item()] = []
            self.label_to_indices[label.item()].append(idx)
            
        self.labels = list(self.label_to_indices.keys())
        
    def __len__(self):
        # 返回可能的episode数量
        return 1000  # 可以设置为更大的数字
        
    def __getitem__(self, index):
        """
        返回一个episode的数据
        
        返回:
            support_x: 支持集数据 [n_way * n_support, ...]
            support_y: 支持集标签 [n_way * n_support]
            query_x: 查询集数据 [n_way * n_query, ...]
            query_y: 查询集标签 [n_way * n_query]
        """
        # 随机选择n_way个类别
        selected_classes = torch.randperm(len(self.labels))[:self.n_way]
        
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        
        for i, class_idx in enumerate(selected_classes):
            # 获取当前类别的所有样本索引
            current_class = self.labels[class_idx]
            indices = self.label_to_indices[current_class]
            
            # 随机选择support和query样本
            selected_indices = torch.randperm(len(indices))
            support_indices = selected_indices[:self.n_support]
            query_indices = selected_indices[self.n_support:self.n_support + self.n_query]
            
            # 收集support样本
            for idx in support_indices:
                x, _ = self.base_dataset[indices[idx]]
                support_x.append(x)
                support_y.append(i)  # 使用0到n_way-1作为新的类别标签
                
            # 收集query样本
            for idx in query_indices:
                x, _ = self.base_dataset[indices[idx]]
                query_x.append(x)
                query_y.append(i)
                
        # 转换为tensor
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y)
        
        return support_x, support_y, query_x, query_y


class DynamicWayProtoNetDataset(Dataset):
    def __init__(self, base_dataset, start_way, end_way, n_support, n_query, way_interval=1):
        """
        支持动态n_way的ProtoNet数据集
        
        参数:
            base_dataset: 基础数据集
            start_way: 起始类别数
            end_way: 结束类别数
            n_support: 每个类别的support样本数
            n_query: 每个类别的query样本数
            way_interval: n_way的增长间隔
        """
        self.base_dataset = base_dataset
        self.start_way = start_way
        self.end_way = end_way
        self.n_support = n_support
        self.n_query = n_query
        self.way_interval = way_interval
        
        # 计算总的way数量
        self.n_way_steps = (end_way - start_way) // way_interval + 1
        
        # 按类别组织数据
        self.label_to_indices = {}
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            if label.item() not in self.label_to_indices:
                self.label_to_indices[label.item()] = []
            self.label_to_indices[label.item()].append(idx)
            
        self.labels = list(self.label_to_indices.keys())
        
    def __len__(self):
        # 每个n_way配置返回固定数量的episode
        return 1000 * self.n_way_steps
        
    def __getitem__(self, index):
        # 确定当前的n_way
        way_step = index // 1000
        current_n_way = self.start_way + way_step * self.way_interval
        
        # 随机选择current_n_way个类别
        selected_classes = torch.randperm(len(self.labels))[:current_n_way]
        
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        
        for i, class_idx in enumerate(selected_classes):
            # 获取当前类别的所有样本索引
            current_class = self.labels[class_idx]
            indices = self.label_to_indices[current_class]
            
            # 随机选择support和query样本
            selected_indices = torch.randperm(len(indices))
            support_indices = selected_indices[:self.n_support]
            query_indices = selected_indices[self.n_support:self.n_support + self.n_query]
            
            # 收集support样本
            for idx in support_indices:
                x, _ = self.base_dataset[indices[idx]]
                support_x.append(x)
                support_y.append(i)
                
            # 收集query样本
            for idx in query_indices:
                x, _ = self.base_dataset[indices[idx]]
                query_x.append(x)
                query_y.append(i)
                
        # 转换为tensor
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y)
        
        # 不再返回current_n_way
        return support_x, support_y, query_x, query_y



