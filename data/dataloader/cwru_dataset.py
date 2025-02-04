import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class CWRUDataset(Dataset):
    def __init__(self, data_dir, signal_length=120000, transform=None,downsample_ratio=1):
        """
        初始化CWRU数据集
        :param data_dir: 数据目录路径
        :param signal_length: 统一信号长度
        :param transform: 可选的数据变换
        """
        self.data_dir = data_dir
        self.signal_length = signal_length
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        self.scaler = StandardScaler()
        self.downsample_ratio = downsample_ratio
        # 预加载数据
        self.signals, self.labels = self._load_data()
        
    
    def _load_data(self):
        signals = []
        labels = []
        
        for filename in self.file_list:
            try:
                # 加载.mat文件
                data = sio.loadmat(os.path.join(self.data_dir, filename))
                
                # 提取信号
                signal = self._get_signal(data, filename)
                if signal is None:
                    continue
                    
                # 获取标签
                label = self._get_12class_label(filename)
                if label is None:
                    continue
                    
                # 统一信号长度
                if len(signal) > self.signal_length:
                    signal = signal[:self.signal_length]
                else:
                    signal = np.pad(signal, (0, max(0, self.signal_length - len(signal))))
                
                signals.append(signal)
                labels.append(label)
            
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                continue
        
        # 转换为numpy数组并标准化
        signals = np.array(signals)
        self.scaler.fit(signals)
        signals = self.scaler.transform(signals)
        
        return signals, np.array(labels)
    
    def _get_signal(self, data, filename):
        """
        获取振动信号，处理不同键名格式
        :param data: 加载的.mat文件数据
        :param filename: 文件名
        :return: 振动信号（如果找到），否则返回None
        """
        # 可能的键名格式
        possible_keys = [
            f'X{filename.split(".")[0]}_DE_time',  # 标准格式
            'DE_time',                             # 简化格式
            's1',                                  # 特殊格式
            f'X{filename.split(".")[0]}',          # 可能存在的其他格式
            'X097_DE_time',                        # 某些文件可能使用固定名称
            'X099_DE_time'
        ]
        
        # 尝试获取信号
        for key in possible_keys:
            if key in data:
                return data[key].flatten()
        
        return None
    def _get_12class_label(self, filename):
        """
        根据文件名获取12分类标签
        :param filename: 文件名
        :return: 标签 (0-11)
        """
        # 去掉文件扩展名和前缀
        file_id_str = filename.split('.')[0]
        
        # 去掉前缀'W'（如果存在）
        if file_id_str.startswith('W'):
            file_id_str = file_id_str[1:]
        
        # 转换为整数
        try:
            file_id = int(file_id_str)
        except ValueError:
            print(f"无法解析文件名: {filename}")
            return None
        
        # 故障尺寸分类
        fault_size = (file_id // 1000) % 10  # 获取千位数
        if fault_size == 0:
            fault_size = file_id // 100  # 处理3位数的文件编号
        
        # 负载分类
        load = (file_id % 100 - 1) % 4  # 根据最后两位数字计算负载
        
        # 计算12分类标签
        # 0-3: 0.007", 4-7: 0.014", 8-11: 0.021", 12-15: 0.028"
        label = (fault_size - 1) * 4 + load
        
        return label
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]  # 获取 NumPy array 信号
        label = self.labels[idx]

        # **下采样 (如果 downsample_ratio > 1)**
        if self.downsample_ratio > 1:
            signal = signal[::self.downsample_ratio]


        # 转换为torch tensor并增加维度 (在下采样之后)
        signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)  # 先转为 Tensor
        label_tensor = torch.tensor(label).long()

        # 应用变换（如果有）
        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        return signal_tensor, label_tensor

# 使用示例
if __name__ == "__main__":
    data_dir = 'C:/Users/luoji/PycharmProjects/motor_dl/data/CWRU/12k Drive End Bearing Fault Data'
    dataset = CWRUDataset(data_dir, downsample_ratio=4)
    
    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 示例：获取一个batch的数据
    for signals, labels in dataloader:
        print(f"Batch signals shape: {signals.shape}")
        print(f"Batch labels: {labels}")
        break
