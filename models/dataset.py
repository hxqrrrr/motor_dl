import h5py
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os

# DataLoader
class h5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            return len(f['data'])

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            data = f['data'][index]
            label = f['labels'][index]
            label = torch.argmax(torch.tensor(label, dtype=torch.float32))# 将8位独热编码转换为数字
            return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


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

