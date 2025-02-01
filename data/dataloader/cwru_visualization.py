import matplotlib.pyplot as plt
import numpy as np
from cwru_dataset import CWRUDataset
# 定义中文标签映射
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
label_names = {
    0: "正常-0HP",
    1: "正常-1HP", 
    2: "正常-2HP",
    3: "正常-3HP",
    4: "0.007英寸内圈-0HP",
    5: "0.007英寸内圈-1HP",
    6: "0.007英寸内圈-2HP",
    7: "0.007英寸内圈-3HP",
    8: "0.014英寸内圈-0HP",
    9: "0.014英寸内圈-1HP",
    10: "0.014英寸内圈-2HP",
    11: "0.014英寸内圈-3HP",
    12: "0.021英寸内圈-0HP",
    13: "0.021英寸内圈-1HP",
    14: "0.021英寸内圈-2HP",
    15: "0.021英寸内圈-3HP"
}

def get_chinese_label(label_id):
    """根据标签ID获取中文标签"""
    return label_names.get(label_id, "未知标签")

def plot_signals(dataset, num_samples=5):
    """
    可视化CWRU数据集中的信号
    :param dataset: CWRUDataset实例
    :param num_samples: 要显示的样本数量
    """
    # 创建子图
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
    
    # 如果只有一个样本，将axes转换为列表
    if num_samples == 1:
        axes = [axes]
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # 绘制信号
    for i, idx in enumerate(indices):
        signal, label = dataset[idx]
        
        # 转换为numpy数组
        signal = signal.numpy()
        
        # 绘制信号
        axes[i].plot(signal, linewidth=0.5)
        axes[i].set_title(f"样本 {idx} - 标签 {label}")
        axes[i].set_xlabel("采样点")
        axes[i].set_ylabel("幅值")
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_spectrum(dataset, num_samples=5):
    """
    可视化信号的频谱
    :param dataset: CWRUDataset实例
    :param num_samples: 要显示的样本数量
    """
    # 创建子图
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
    
    # 如果只有一个样本，将axes转换为列表
    if num_samples == 1:
        axes = [axes]
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # 绘制频谱
    for i, idx in enumerate(indices):
        signal, label = dataset[idx]
        
        # 转换为numpy数组
        signal = signal.numpy()
        
        # 计算FFT
        fft = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), d=1/12000)  # 采样频率12kHz
        
        # 绘制频谱
        axes[i].plot(freq[:len(freq)//2], np.abs(fft[:len(fft)//2]), linewidth=0.5)
        axes[i].set_title(f"样本 {idx} - 标签 {label}")
        axes[i].set_xlabel("频率 (Hz)")
        axes[i].set_ylabel("幅值")
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 创建数据集
    data_dir = 'data/CWRU/12k Drive End Bearing Fault Data'
    dataset = CWRUDataset(data_dir)
    
    # 可视化时域信号
    print("时域信号可视化：")
    plot_signals(dataset, num_samples=5)
    
    # 可视化频域信号
    print("频域信号可视化：")
    plot_spectrum(dataset, num_samples=5)