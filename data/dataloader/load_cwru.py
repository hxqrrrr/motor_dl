import scipy.io as sio
import numpy as np
import os

def load_cwru_dataset(data_dir):
    """
    加载CWRU轴承数据集
    :param data_dir: 包含.mat文件的目录路径
    :return: (signals, labels) - 信号数据和对应标签
    """
    signals = []
    labels = []
    
    # 遍历数据目录
    for filename in os.listdir(data_dir):
        if filename.endswith('.mat'):
            filepath = os.path.join(data_dir, filename)
            
            # 加载.mat文件
            mat_data = sio.loadmat(filepath)
            
            # 提取振动信号
            de_key = f'X{filename.split(".")[0]}_DE_time'  # 构建驱动端振动信号键名
            if de_key in mat_data:
                signal = mat_data[de_key].reshape(-1)
                
                # 统一信号长度（例如取前120000个点）
                signal_length = 120000  # 你可以根据需要调整这个值
                if len(signal) > signal_length:
                    signal = signal[:signal_length]
                else:
                    # 如果信号较短，可以填充0或截断
                    signal = np.pad(signal, (0, max(0, signal_length - len(signal))))
                
                signals.append(signal)
                
                # 根据文件名生成标签
                label = int(filename.split('.')[0])
                labels.append(label)
    
    # 转换为numpy数组
    signals = np.array(signals)
    labels = np.array(labels)
    
    return signals, labels

# 使用示例
data_dir = 'data/CWRU/12k Drive End Bearing Fault Data'
signals, labels = load_cwru_dataset(data_dir)
print(f"加载了 {len(signals)} 个样本")
print(f"信号形状: {signals.shape}")
print(f"标签形状: {labels.shape}")