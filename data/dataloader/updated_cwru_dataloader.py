# -*- coding: utf-8 -*-
"""mckn_cnn_dataset.py (Final Complete Version - 10 Class, Sliding Window, DE Signal Loading, Main Test)"""
import sys
import os
import time

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # For visualization in main test
from utils.vmd import extract_vmd_features


        # 数据集处理逻辑：
        # 1. 初始化数据集类，设置数据目录、信号长度、下采样比例、类别数量、窗口大小和步长等参数。
        # 2. 预加载数据，包括滑动窗口增强。
        # 3. 在 _load_data 方法中：
        #    a. 遍历数据目录中的每个类名，检查其是否在10类映射中。
        #    b. 对于每个.mat文件，加载驱动端振动信号。
        #    c. 获取信号对应的10类标签。
        #    d. 处理完整信号，转置并展平。
        #    e. 对正常状态进行下采样。
        #    f. 直接测量数据长度，统一信号长度。
        #    g. 先下采样，再将信号分割为窗口。
        #    h. 打印每条信号分割成的窗口数量。
        # 4. 将所有信号转换为numpy数组并标准化。
        # 5. 在 __getitem__ 方法中，获取指定索引的信号和标签，进行下采样，转换为torch张量并添加维度，应用变换（如果有）。
        # 6. 在 show_dataset_info 方法中，显示数据集的详细信息，包括样本大小、样本形状、批次数据形状等。
        # 7. 创建数据加载器，分割数据集为训练集和验证集。





class CWRUDataset(Dataset):
    def __init__(self, data_dir, batch_size, transform=None, scale=True, downsample_ratio=1, num_classes=10, window_size=600, stride=600,vmd_mode=False,k=3):
        """
        初始化 CWRUDataset（MCKD-CNN 论文复现 - 10 类，滑动窗口，驱动端信号）
        :param data_dir: 数据目录的路径
        :param transform: 可选的数据变换
        :param downsample_ratio: 下采样比例（默认：1 - 不进行下采样）
        :param num_classes: 类别数量（默认：10）
        :param window_size: 滑动窗口大小（例如，240）
        :param stride: 滑动窗口步长（例如，60）
        :param vmd_mode: 是否启用 VMD 模式
        :param k: VMD 的 k 值
        """
        super(CWRUDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.downsample_ratio = downsample_ratio
        self.num_classes = num_classes
        self.window_size = window_size
        self.stride = stride
        self.vmd_mode = vmd_mode
        self.k = k
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        self.batch_norm = nn.BatchNorm1d(1)
        self.scaler = StandardScaler()
        self.scale = scale
        # 10 Class Label Mapping (One-Hot Encoding) - Clearly labeled fault conditions
        self.class_labels_10class = {
            'Normal': 0,  # 0: 正常状态
            'B007': 1,  # 1: 滚动体故障 (0.007")
            'B014': 2,  # 2: 滚动体故障 (0.014")
            'B021': 3,  # 3: 滚动体故障 (0.021")
            'IR007': 4,  # 4: 内圈故障 (0.007")
            'IR014': 5,  # 5: 内圈故障 (0.014")
            'IR021': 6,  # 6: 内圈故障 (0.021")
            'OR007': 7,  # 7: 外圈故障 (0.007")
            'OR014': 8,  # 8: 外圈故障 (0.014")
            'OR021': 9  # 9: 外圈故障 (0.021")
        }


        # Preload data (including sliding window augmentation)
        self.signals, self.labels = self._load_data()
        self.show_dataset_info(self.batch_size)

    def _load_data(self):
        # 初始化数据存储
        all_signals = []
        all_labels = []
        class_names = sorted(os.listdir(self.data_dir))
        
        for class_name in class_names:
            # 检查类名是否在10类映射中
            if class_name not in self.class_labels_10class:
                continue  # 跳过不在10类映射中的类
            class_dir = os.path.join(self.data_dir, class_name)
            
            for filename in os.listdir(class_dir):
                if filename.endswith('.mat'):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        data = sio.loadmat(filepath)

                        # 加载驱动端振动信号
                        de_time_keys = [key for key in data.keys() if key.endswith('_DE_time')]
                        if de_time_keys:
                            signal_data = data.get(de_time_keys[0])
                        else:
                            print(f"Warning: 未找到驱动端振动信号变量 (_DE_time) 在 {filename} 中，跳过。")
                            continue

                        if signal_data is None:
                            print(f"Warning: 驱动端振动信号数据在 {filename} 中为 None，跳过。")
                            continue

                        # 获取10类标签
                        label = self._get_10class_label(class_name)
                        if label is None:
                            continue

                        # 处理完整信号
                        if not isinstance(signal_data, np.ndarray):
                            signal_data = np.array(signal_data)
                            print("signal_data 不是 np 数组，已转换。")
                        full_signal = signal_data.transpose().flatten()  # 转置并展平

                        # 直接使用完整信号
                        signal = full_signal

                        # 检查信号原始长度是否大于测量的信号长度
                        if len(full_signal) > len(signal):
                            print(f"信号 {filename} 的原始长度 {len(full_signal)} 大于测量的信号长度 {len(signal)}")

                        # 先下采样
                        if self.downsample_ratio > 1:
                            signal = signal[::self.downsample_ratio]

                        # 将信号分割为窗口
                        for start_idx in range(0, len(signal), self.stride):
                            if len(signal) - start_idx < self.window_size:
                                break
                            windowed_signal = signal[start_idx:start_idx + self.window_size]
                            all_signals.append(windowed_signal)
                            all_labels.append(label)

                        # 打印每条信号分割成的窗口数量
                        num_windows = (len(signal) - self.window_size) // self.stride + 1
                        print(f"信号 {filename} 分割成 {num_windows} 个窗口")
                    except Exception as e:
                        print(f"Error loading or processing {filename}: {e}")
                        continue
        if self.vmd_mode:
            print("开始 VMD 特征提取...")
            start_time = time.time()
            # 使用 GPU 加速进行 VMD 特征提取
            all_signals = [extract_vmd_features(signal, K=self.k, use_gpu=True) for signal in all_signals]
            end_time = time.time()
            print(f"VMD 特征提取完成，用时：{end_time - start_time:.2f}秒")
            print(f"VMD 特征维度：{all_signals[0].shape}")
            
        # 转换为numpy数组并标准化
        all_signals_np = np.array(all_signals, dtype=np.float64)
        print("转换为 np 数组")

        if self.scale:
            self.scaler.fit(all_signals_np)
            all_signals_np = self.scaler.transform(all_signals_np)
            print("数据已标准化")
        return all_signals_np, np.array(all_labels)


    def _get_10class_label(self, class_name):
        """Get 10-class label from class name"""
        return self.class_labels_10class.get(class_name)


    def __len__(self):

        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        # Downsample (applied AFTER windowing)
        if self.downsample_ratio > 1:
            signal = signal[::self.downsample_ratio]

        # Convert to torch tensor and add dimension
        signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)

        label_tensor = torch.tensor(label).long()

        # Apply transform (if any)
        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        return signal_tensor, label_tensor

    def show_dataset_info(self, batch_size):
        """显示数据集的详细信息，包括样本大小、样本形状、批次数据形状等"""
        # 打印数据集基本信息
        print(f"数据集大小: {len(self)}")
        print(f"标签数量: {self.num_classes}")
        print(f"窗口大小: {self.window_size}")
        print(f"窗口步长: {self.stride}")
        print(f"下采样比例: {self.downsample_ratio}")

        # 获取一个数据样本
        signal, label = self[0]

        # 打印样本的形状和标签
        print(f"样本信号形状: {signal.shape}")
        print(f"样本标签: {label}")

        # 创建数据加载器
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=True)

        # 获取一个批次的数据
        batch_signals, batch_labels = next(iter(dataloader))

        # 打印批次数据的形状
        print(f"批次信号形状: {batch_signals.shape}")
        print(f"批次标签形状: {batch_labels.shape}")

if __name__ == '__main__':
    # -------------  Simple Dataset Test  -------------
    print("-----  Simple CWRUDataset Test  -----")

    # 设置数据目录和参数
    data_dir = 'data/CWRU_10Class_Verified'
    batch_size = 20
    window_size = 600
    downsample_ratio = 2
    stride = 600

    # 创建数据集实例
    dataset = CWRUDataset(data_dir,  window_size=window_size, stride=stride, downsample_ratio=downsample_ratio,batch_size=batch_size,vmd_mode=True,k=3)





