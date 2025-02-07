# -*- coding: utf-8 -*-
"""mckn_cnn_dataset.py (Final Complete Version - 10 Class, Sliding Window, DE Signal Loading, Main Test)"""
import os
import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # For visualization in main test

class CWRUDataset(Dataset):
    def __init__(self, data_dir, signal_length=1200,signal_count_per_label=1000, transform=None,scale=True, downsample_ratio=1, num_classes=10,
                 window_size=240, stride=60):
        """
        Initializes CWRUDataset (MCKD-CNN Paper Replication - 10 Class, Sliding Window, DE Signal)
        :param data_dir: Path to data directory
        :param signal_length: Uniform signal length (e.g., 1200)
        :param transform: Optional data transforms
        :param downsample_ratio: Downsampling ratio (default: 1 - no downsampling)
        :param num_classes: Number of classes (default: 10)
        :param window_size: Sliding window size (e.g., 240)
        :param stride: Sliding window stride (e.g., 60)
        """
        super(CWRUDataset, self).__init__()
        self.data_dir = data_dir
        self.signal_length = signal_length
        self.signal_count_per_label = signal_count_per_label
        self.transform = transform
        self.downsample_ratio = downsample_ratio
        self.num_classes = num_classes
        self.window_size = window_size
        self.stride = stride
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

    def _load_data(self):
        all_signals = []
        all_labels = []
        class_names = sorted(os.listdir(self.data_dir))
        count = 0
        lst_of_num_sample = []
        for class_name in class_names:
            if class_name not in self.class_labels_10class:
                continue  # Skip classes not in 10-class mapping
            print(count,class_name)
            class_dir = os.path.join(self.data_dir, class_name)
            for filename in os.listdir(class_dir):
                num_signals_for_label = 0
                if filename.endswith('.mat'):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        data = sio.loadmat(filepath)

                        # **Corrected Signal Loading: Load Drive End Vibration Signal (_DE_time)**
                        de_time_keys = [key for key in data.keys() if key.endswith('_DE_time')]
                        if de_time_keys:
                            signal_data = data.get(de_time_keys[0])# Load Drive End signal
                        else:
                            print(
                                f"Warning: Drive End vibration signal variable (_DE_time) not found in {filename}, skipping.")
                            continue

                        if signal_data is None:
                            print(f"Warning: Drive End vibration signal data is None in {filename}, skipping.")
                            continue

                        label = self._get_10class_label(class_name)
                        if label is None:
                            continue

                        # **Process the full signal directly (NO Sliding Window)**
                        if not isinstance(signal_data, np.ndarray):
                            signal_data = np.array(signal_data)
                            print("signal_data was not a np array, it has been converted")
                        full_signal = signal_data.transpose().flatten()  # 转置并展平
                        # full_signal = signal_data.flatten()  # 转置并展平
                        if class_name == 'Normal':
                            ratio = len(full_signal) // self.signal_length
                            full_signal = full_signal[: : ratio]
                            print(full_signal.shape,"修复过的normal")
                        # Uniform signal length (applied to the full signal)
                        if len(full_signal) > self.signal_length:
                            signal = full_signal[:self.signal_length]
                        else:
                            signal = np.pad(full_signal, (0, max(0, self.signal_length - len(full_signal))))
                        print(signal.shape)
                        for start_idx in range(0, len(signal), self.stride):
                            if len(signal)-start_idx < self.window_size:
                                break
                            windowed_signal = signal[start_idx:start_idx + self.window_size]
                            print(windowed_signal.shape)
                            all_signals.append(windowed_signal)
                            all_labels.append(label)
                            num_signals_for_label += 1
                            if num_signals_for_label >= self.signal_count_per_label:
                                break
                        count+=1
                    except Exception as e:
                        print(f"Error loading or processing {filename}: {e}")
                        continue
        # Convert to numpy arrays and standardize (applied to ALL signals)
        all_signals_np = np.array(all_signals,dtype=np.float64)
        print("converted to np array")
        if self.scale is True:
            self.scaler.fit(all_signals_np)
            print("fitted")
            all_signals_np = self.scaler.transform(all_signals_np)
            print("scaled")
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



def create_dataloader_mckn(data_dir, batch_size, signal_length=1200, downsample_ratio=1, num_workers=0, pin_memory=True, val_ratio=0.2,
                           window_size=240, stride=60):
    """创建 MCKN_CNN 数据加载器"""
    print("creating dataloader mckn")
    dataset = CWRUDataset(data_dir, signal_length=signal_length, downsample_ratio=downsample_ratio, num_classes=10, # num_classes=10
                            window_size=window_size, stride=stride) # 传递 window_size 和 stride

    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size # **调整： 先定义 val_size， 再定义 train_size**
    trainset, valset = random_split(dataset, [train_size, val_size]) # **修正： 现在 train_size 已经被定义， 不会报错**

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )

    return train_loader, val_loader



if __name__ == '__main__':
    # -------------  Main Function Test  -------------
    print("-----  CWRUDataset Main Function Test  -----")

    data_dir = 'C:/Users/luoji/PycharmProjects/motor_dl/data/CWRU_10Class_Verified'  # **<-- VERY IMPORTANT: Set YOUR DATA DIRECTORY HERE! **
    batch_size = 16
    signal_length = 120000
    downsample_ratio = 1 # Test downsampling ratio 4
    window_size = 2400   # Test window size 240
    stride = 240        # Test stride 60

    # Create CWRUDataset instance with sliding window and downsampling
    dataset = CWRUDataset(data_dir, signal_length=signal_length, downsample_ratio=downsample_ratio,
                            window_size=window_size, stride=stride, num_classes=10) # num_classes=10 for 10-class test

    # Print dataset size (number of windowed signals)
    print(f"Dataset size (number of windowed signals): {len(dataset)}") # Expect: Much larger than original .mat files

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Get a batch of data
    example_batch = next(iter(dataloader))
    signals, labels = example_batch

    # Print Batch signals and labels shapes
    print("Batch signals shape:", signals.shape)# Expect: [batch_size, 1, signal_length // downsample_ratio] (e.g., [32, 1, 300])
    print("Batch labels shape:", labels.shape)    # Expect: [batch_size]
    print("Example Batch labels:", labels)    # Print example batch labels


    # Visualization Check (Plot waveforms of first 3 samples - optional but HIGHLY recommended)
    num_samples_to_plot = min(3, batch_size)
    plt.figure(figsize=(10, 3 * num_samples_to_plot))
    for i in range(num_samples_to_plot):
        signal = signals[i].squeeze().numpy()
        label = labels[i].item()

        plt.subplot(num_samples_to_plot, 1, i + 1)
        plt.plot(signal)
        plt.title(f"Sample {i+1}, Label: {label} (Class: {dataset.labels[i]})") # Show original class label

    plt.tight_layout()
    plt.show() # Show plot


    # -------------  .mat File Data Inspection (Example for ONE file) -------------
    print("\n-----  .mat File Data Inspection (Example for ONE file) -----")

    example_mat_filepath = 'C:/Users/luoji/PycharmProjects/motor_dl/data/CWRU_10Class_Verified/B007/118_0.mat'
    example_mat_data = sio.loadmat(example_mat_filepath)

    print("Keys in .mat file:", example_mat_data.keys())

    # Check shape and dtype of Drive End vibration signal variable ('X118_DE_time' - EXAMPLE - adjust if needed)
    if 'X118_DE_time' in example_mat_data: # **<-- MODIFY VARIABLE NAME IF NEEDED! **
        signal_data_example_de = example_mat_data['X118_DE_time'] # **<-- MODIFY VARIABLE NAME IF NEEDED! **
        print("Shape of 'X118_DE_time' variable:", signal_data_example_de.shape) # **<-- MODIFY VARIABLE NAME IF NEEDED! **
        print("Data type of 'X118_DE_time' variable:", signal_data_example_de.dtype) # **<-- MODIFY VARIABLE NAME IF


