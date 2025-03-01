import h5py
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os
import numpy as np
from typing import Dict, List, Tuple
import random

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
        
        # 获取文件路径
        self.file_path = os.path.join(folder_path, self.h5_files[0])
        
        # 打开H5文件获取必要信息，然后关闭
        with h5py.File(self.file_path, 'r') as f:
            # 获取数据和标签的形状
            data_shape = f['data'].shape
            labels_shape = f['labels'].shape
            
            # 打印标签信息
            print("\n原始标签形状:", labels_shape)
            print("标签示例（前5个）:")
            print(f['labels'][:5])
            if 'labelName' in f:
                label_names = f['labelName'].asstr()[:]
                print("\n标签名称:", label_names)
            
            # 将one-hot标签转换为类别索引 - 只处理标签数据
            labels_sample = f['labels'][:]  # 这里需要加载标签数据
            self.labels_indices = torch.argmax(torch.tensor(labels_sample, dtype=torch.float32), dim=1).numpy()
        
        # 分层采样
        unique_labels = np.unique(self.labels_indices)
        train_indices = []
        test_indices = []
        
        print("\n数据集划分情况:")
        for label in unique_labels:
            label_indices = np.where(self.labels_indices == label)[0]
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
        
        # 计算标准化参数 - 只使用训练数据的一部分样本计算统计信息
        sample_size = min(1000, len(train_indices))  # 使用最多1000个样本计算统计信息
        sample_indices = np.random.choice(train_indices, sample_size, replace=False)
        
        # 加载样本数据计算统计信息
        sample_data = []
        with h5py.File(self.file_path, 'r') as f:
            for i in sample_indices:
                sample_data.append(f['data'][i])
        sample_data = np.array(sample_data)
        
        self.mean = np.mean(sample_data, axis=0, keepdims=True)
        self.std = np.std(sample_data, axis=0, keepdims=True) + 1e-10
        
        # 选择相应的数据集索引
        self.indices = train_indices if train else test_indices
        
        # 将标签索引转换为tensor
        self.labels = torch.tensor([self.labels_indices[i] for i in self.indices], dtype=torch.long)
        
        print(f"\n{('训练' if train else '测试')}集信息:")
        unique_labels = torch.unique(self.labels)
        print(f"包含的类别: {unique_labels.tolist()}")
        for label in unique_labels:
            count = (self.labels == label).sum().item()
            print(f"类别 {label}: {count} 个样本")
        
        # 预加载数据到内存中
        print(f"预加载数据到内存中，共 {len(self.indices)} 个样本...")
        self.data_cache = []
        with h5py.File(self.file_path, 'r') as f:
            for idx in self.indices:
                # 读取数据并标准化
                data = (f['data'][idx] - self.mean) / self.std
                self.data_cache.append(torch.tensor(data, dtype=torch.float32))
        print("数据预加载完成")
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # 直接从缓存中获取数据
        return self.data_cache[index], self.labels[index].clone().detach()

    def get_all_samples(self):
        """
        获取数据集中所有样本的唯一标识符
        返回：包含所有样本标识符的列表
        """
        # 使用数据和标签的组合作为唯一标识符
        identifiers = []
        for i in range(len(self.indices)):
            # 使用数据的哈希值和标签作为唯一标识符
            data = self.data_cache[i]
            data_hash = hash(data.numpy().tobytes())
            label = self.labels[i].item()
            identifier = f"{data_hash}_{label}"
            identifiers.append(identifier)
        return identifiers


class SourceDomainDataset(Dataset):
    """源域数据集类，支持多进程加载"""
    def __init__(self, h5_file_path, indices=None):
        self.h5_file_path = h5_file_path
        
        # 只打开文件获取信息，然后关闭
        with h5py.File(h5_file_path, 'r') as f:
            # 获取数据形状和标签信息
            self.data_shape = f['data'].shape
            self.labels_shape = f['labels'].shape
            # 获取唯一标签
            labels_data = f['labels'][:]
            if len(labels_data.shape) > 1 and labels_data.shape[1] > 1:
                # 如果是one-hot编码，转换为类别索引
                labels_indices = np.argmax(labels_data, axis=1)
            else:
                # 如果已经是类别索引
                labels_indices = labels_data
            unique_labels = np.unique(labels_indices)
            
            # 打印数据信息
            print(f"数据形状: {self.data_shape}")
            print(f"标签形状: {self.labels_shape}")
            print(f"唯一标签: {unique_labels}")
        
        # 如果提供了索引，则只使用这些索引对应的数据
        self.indices = indices if indices is not None else np.arange(self.data_shape[0])
        
        # 缓存标签数据，避免每次都读取文件
        with h5py.File(h5_file_path, 'r') as f:
            # 只加载需要的标签
            labels_data = f['labels'][:]
            if len(labels_data.shape) > 1 and labels_data.shape[1] > 1:
                # 如果是one-hot编码，转换为类别索引
                self.labels_cache = np.array([np.argmax(labels_data[i]) for i in self.indices])
            else:
                # 如果已经是类别索引
                self.labels_cache = np.array([labels_data[i] for i in self.indices])
            
            # 预加载数据到内存中
            print(f"预加载数据到内存中，共 {len(self.indices)} 个样本...")
            self.data_cache = []
            for idx in self.indices:
                self.data_cache.append(f['data'][idx])
            print("数据预加载完成")
        
        print(f"数据集大小: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        # 直接从缓存中获取数据和标签
        data = self.data_cache[index]
        label = self.labels_cache[index]
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        
    def get_all_samples(self):
        """
        获取数据集中所有样本的唯一标识符
        返回：包含所有样本标识符的列表
        """
        # 使用数据和标签的组合作为唯一标识符
        identifiers = []
        for i in range(len(self.indices)):
            # 使用数据的哈希值和标签作为唯一标识符
            data = self.data_cache[i]
            label = self.labels_cache[i]
            data_hash = hash(data.tobytes())
            identifier = f"{data_hash}_{label}"
            identifiers.append(identifier)
        return identifiers


class ProtoNetDataset(Dataset):
    def __init__(self, base_dataset, n_way, n_support, n_query, num_episodes=100):
        """
        ProtoNet数据集包装器
        
        参数:
            base_dataset: 基础数据集 (h5Dataset 或 KATDataset)
            n_way: 每个episode中的类别数
            n_support: 每个类别的support样本数
            n_query: 每个类别的query样本数
            num_episodes: 生成的episode数量，默认为100
        """
        self.base_dataset = base_dataset
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.num_episodes = num_episodes
        
        # 按类别组织数据
        self.label_to_indices = {}
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
            
        self.labels = list(self.label_to_indices.keys())
        
        # 检查数据是否足够
        self._check_data_sufficiency()
        
        # 打印数据集信息
        self._print_dataset_info()
        
        # 预生成所有episode的类别选择
        # 这样可以确保多进程加载时每个worker看到相同的episode结构
        np.random.seed(42)  # 使用固定种子以确保可重现性
        self.episodes = []
        for _ in range(self.num_episodes):
            # 如果类别数量不足，则使用所有可用类别
            if len(self.labels) < self.n_way:
                episode_classes = self.labels
            else:
                episode_classes = np.random.choice(self.labels, self.n_way, replace=False)
            self.episodes.append(episode_classes)
        np.random.seed(None)  # 重置随机种子
        
        # 预加载所有数据，避免在多进程中使用h5py文件句柄
        self._preload_data()
    
    def _preload_data(self):
        """预加载所有数据到内存中，避免在多进程中使用h5py文件句柄"""
        print("预加载数据到内存中...")
        self.data_cache = {}
        for label, indices in self.label_to_indices.items():
            self.data_cache[label] = []
            for idx in indices:
                x, _ = self.base_dataset[idx]
                self.data_cache[label].append(x)
        print("数据预加载完成")
    
    def _check_data_sufficiency(self):
        """检查每个类别的样本是否足够构建元学习任务"""
        for label, indices in self.label_to_indices.items():
            if len(indices) < self.n_support + self.n_query:
                print(f"警告: 类别 {label} 的样本数量 ({len(indices)}) 不足以构建元学习任务 (需要 {self.n_support + self.n_query} 个样本)")
    
    def _print_dataset_info(self):
        """打印数据集信息"""
        print(f"\nProtoNetDataset 信息:")
        print(f"  - 类别数量: {len(self.labels)}")
        print(f"  - N-way: {self.n_way}")
        print(f"  - N-support: {self.n_support}")
        print(f"  - N-query: {self.n_query}")
        print(f"  - Episode数量: {self.num_episodes}")
        
        # 打印每个类别的样本数量
        print("\n类别样本分布:")
        for label in self.labels:
            print(f"  - 类别 {label}: {len(self.label_to_indices[label])} 个样本")
    
    def __len__(self):
        # 返回可能的episode数量
        return self.num_episodes
        
    def __getitem__(self, index):
        """
        获取一个episode
        
        参数:
            index: episode索引
            
        返回:
            support_x: 支持集样本
            support_y: 支持集标签
            query_x: 查询集样本
            query_y: 查询集标签
        """
        # 使用预生成的类别选择
        episode_classes = self.episodes[index]
        
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        
        for i, cls in enumerate(episode_classes):
            # 获取当前类别的所有样本
            samples = self.data_cache[cls]
            
            # 确保有足够的样本
            if len(samples) < self.n_support + self.n_query:
                # 如果样本不足，则使用重复采样
                selected_indices = np.random.choice(len(samples), self.n_support + self.n_query, replace=True)
            else:
                # 随机选择不重复的样本
                selected_indices = np.random.choice(len(samples), self.n_support + self.n_query, replace=False)
            
            # 分割为支持集和查询集
            support_indices = selected_indices[:self.n_support]
            query_indices = selected_indices[self.n_support:self.n_support + self.n_query]
            
            # 获取样本和标签
            for idx in support_indices:
                x = samples[idx]
                support_x.append(x)
                support_y.append(i)  # 使用类别的索引作为标签
                
            for idx in query_indices:
                x = samples[idx]
                query_x.append(x)
                query_y.append(i)  # 使用类别的索引作为标签
        
        # 转换为tensor
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y, dtype=torch.long)
        
        return support_x, support_y, query_x, query_y

class TaskProtoNetDataset(Dataset):
    def __init__(self, base_dataset, n_way, n_support, n_query, selected_labels, selected_classes=None, is_train=True):
        self.base_dataset = base_dataset
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.selected_labels = selected_labels
        self.selected_classes = selected_classes
        self.is_train = is_train

        # 按类别组织数据
        self.label_to_indices = self._organize_labels()
        self.labels = list(self.label_to_indices.keys())

        # 打印标签信息
        self._print_label_info()

        # 划分训练集和验证集
        self.data_indices = self._split_dataset()  

    def _organize_labels(self):
        label_to_indices = {}
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            if label.item() in self.selected_labels:
                if label.item() not in label_to_indices:
                    label_to_indices[label.item()] = []
                label_to_indices[label.item()].append(idx)
        return label_to_indices

    def _print_label_info(self):
        print(f"可用标签: {self.labels}")
        print(f"标签范围: {min(self.labels)} 到 {max(self.labels)}")

    def _split_dataset(self):
        indices = []
        for label in self.labels:
            label_indices = self.label_to_indices[label]
            split_idx = int(len(label_indices) * 0.8)  # 80%作为训练集
            if self.is_train:
                indices.extend(label_indices[:split_idx])
            else:
                indices.extend(label_indices[split_idx:])
        return indices

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.data_indices):
            raise IndexError("索引超出范围")
        idx = self.data_indices[index]
        return self.get_episode_data(idx)

    def get_episode_data(self, idx):
        _, label = self.base_dataset[idx]
        label = label.item()

        # 检查标签是否在 selected_labels 中
        if label not in self.selected_labels:
            raise ValueError(f"标签 {label} 不在选择的标签中")

        # 获取当前类别的所有样本索引
        indices = self.label_to_indices[label]
        if len(indices) < self.n_support + self.n_query:
            raise ValueError("样本数量不足以满足支持和查询的要求")

        # 随机选择支持和查询样本
        selected_classes = self.selected_classes if self.selected_classes else [label]
        support_x, support_y, query_x, query_y = [], [], [], []

        for selected_class in selected_classes:
            if selected_class not in self.label_to_indices:
                raise ValueError(f"选择的类别 {selected_class} 不在数据集中")
            class_indices = self.label_to_indices[selected_class]
            selected_indices = torch.randperm(len(class_indices))
            support_indices = selected_indices[:self.n_support]
            query_indices = selected_indices[self.n_support:self.n_support + self.n_query]

            # 收集支持样本和查询样本
            support_x_class, support_y_class = self._collect_samples(class_indices, support_indices, selected_class)
            query_x_class, query_y_class = self._collect_samples(class_indices, query_indices, selected_class)

            support_x.append(support_x_class)
            support_y.append(support_y_class)
            query_x.append(query_x_class)
            query_y.append(query_y_class)

        return torch.cat(support_x), torch.cat(support_y), torch.cat(query_x), torch.cat(query_y)

    def _collect_samples(self, indices, selected_indices, label):
        x_samples = []
        y_samples = []
        for idx in selected_indices:
            x, _ = self.base_dataset[indices[idx]]
            x_samples.append(x)
            y_samples.append(label)  # 使用原始标签
        return torch.stack(x_samples), torch.tensor(y_samples)



class EpisodeGenerator(Dataset):
    def __init__(self, dataset: h5Dataset, n_way: int, n_support: int, n_query: int, selected_labels: List[int]):
        """
        初始化剧集生成器。
        
        参数:
            dataset: 一个 h5Dataset 实例。
            k: 每集的类数。
            n: 支持集中每个类的示例数。
            m: 查询集中每个类的示例数。
        """
        self.dataset = dataset  # 存储数据集
        self.n_way = n_way  # 每集的类数
        self.n_support = n_support  # 支持集中每个类的示例数
        self.n_query = n_query  # 查询集中每个类的示例数
        self.selected_labels = selected_labels
        # 划分支持集和查询集
        self.support_set, self.query_set = self.split_dataset()

    def split_dataset(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        划分支持集和查询集。
        
        返回:
            支持集和查询集的字典。
        """
        support_set = {}
        query_set = {}
        
        # 获取所有标签
        unique_labels = np.unique(self.dataset.labels)
        
        for cls in unique_labels:
            # 获取当前类别的所有样本索引
            cls_indices = [i for i, label in enumerate(self.dataset.labels) if label == cls]
            # 随机选择 n 个示例作为支持集
            support_examples = np.random.choice(cls_indices, self.n_support, replace=False)
            # 从剩余的示例中随机选择 m 个示例作为查询集
            remaining_examples = list(set(cls_indices) - set(support_examples))
            query_examples = np.random.choice(remaining_examples, self.n_query, replace=False)
            
            support_set[cls.item()] = support_examples.tolist()
            query_set[cls.item()] = query_examples.tolist()
        
        return support_set, query_set

    def __len__(self):
        """返回数据集中可能的剧集数量。"""
        return 100

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据索引返回支持集和查询集的样本。
        
        参数:
            index: 要获取的剧集索引。
        
        返回:
            支持集样本和查询集样本的元组。
        """
        # 生成一个剧集
        support_set, support_labels, query_set, query_labels = self.generate_episode()
        
        return support_set, support_labels, query_set, query_labels

    def generate_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        classes = random.sample(self.selected_labels, self.n_way)
        
        support_set = []
        support_labels = []
        query_set = []
        query_labels = []
        
        for cls in classes:
            support_indices = self.support_set[cls]
            query_indices = self.query_set[cls]
            
            for idx in support_indices:
                x, _ = self.dataset[idx]
                support_set.append(x)
                support_labels.append(cls)
            
            for idx in query_indices:
                x, _ = self.dataset[idx]
                query_set.append(x)
                query_labels.append(cls)
        
        return torch.stack(support_set), torch.tensor(support_labels), torch.stack(query_set), torch.tensor(query_labels)







