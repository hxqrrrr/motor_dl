import h5py
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os

# DataLoader
class h5Dataset(Dataset):
    def __init__(self, folder_path, train=True, train_ratio=0.8):
        self.folder_path = folder_path
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
            
        # 计算划分点
        total_samples = len(self.data)
        split_idx = int(total_samples * train_ratio)
        
        # 根据train参数选择数据集部分
        if train:
            self.data = self.data[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.data = self.data[split_idx:]
            self.labels = self.labels[split_idx:]
            
        self.labels = torch.argmax(torch.tensor(self.labels, dtype=torch.float32), dim=1)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32), \
               torch.tensor(self.labels[index], dtype=torch.long)


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

