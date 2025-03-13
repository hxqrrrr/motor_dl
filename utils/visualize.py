import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Union, List, Dict
import seaborn as sns

def visualize_attention(
    x: torch.Tensor,
    attention: torch.Tensor,
    attention_type: str,
    save_dir: str = "attention_vis",
    step_name: Optional[str] = None,
    channels: Optional[List[str]] = None,
    sample_idx: int = 0
) -> None:
    """
    可视化注意力权重
    
    参数:
        x: 输入数据 [batch_size, channels, length]
        attention: 注意力权重
            - 通道注意力: [batch_size, channels, 1]
            - 空间注意力: [batch_size, 1, length]
            - CBAM: 字典包含两种注意力
        attention_type: 注意力类型 ('channel', 'spatial', 'cbam')
        save_dir: 保存目录
        step_name: 步骤名称（可选）
        channels: 通道名称列表（可选）
        sample_idx: 要可视化的样本索引
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备数据
    x_np = x[sample_idx].cpu().detach().numpy()
    
    if attention_type == 'channel':
        visualize_channel_attention(x_np, attention[sample_idx].cpu().detach().numpy(), 
                                  save_dir, timestamp, step_name, channels)
    
    elif attention_type == 'spatial':
        visualize_spatial_attention(x_np, attention[sample_idx].cpu().detach().numpy(),
                                  save_dir, timestamp, step_name)
    
    elif attention_type == 'cbam':
        # 假设CBAM attention是一个字典，包含channel和spatial attention
        visualize_cbam_attention(x_np, 
                               attention['channel'][sample_idx].cpu().detach().numpy(),
                               attention['spatial'][sample_idx].cpu().detach().numpy(),
                               save_dir, timestamp, step_name, channels)

def visualize_channel_attention(x_np, attention_np, save_dir, timestamp, step_name=None, channels=None):
    """可视化通道注意力"""
    n_channels = x_np.shape[0]
    
    plt.figure(figsize=(15, 10))
    
    # 1. 原始信号
    plt.subplot(3, 1, 1)
    plt.title("Original Signal")
    for i in range(n_channels):
        label = f"Channel {i}" if channels is None else channels[i]
        plt.plot(x_np[i], label=label)
    plt.legend()
    
    # 2. 注意力权重
    plt.subplot(3, 1, 2)
    plt.title("Channel Attention Weights")
    channel_indices = range(n_channels)
    weights = attention_np.squeeze()
    
    # 使用条形图显示权重
    bars = plt.bar(channel_indices, weights)
    
    # 添加权重值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.xlabel("Channel")
    plt.ylabel("Weight")
    if channels:
        plt.xticks(channel_indices, channels, rotation=45)
    
    # 3. 加权后的信号
    plt.subplot(3, 1, 3)
    plt.title("Weighted Signal")
    weighted_signal = x_np * attention_np
    for i in range(n_channels):
        label = f"Channel {i}" if channels is None else channels[i]
        plt.plot(weighted_signal[i], label=label)
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    name = f"channel_attention_{step_name}_{timestamp}.png" if step_name else f"channel_attention_{timestamp}.png"
    plt.savefig(os.path.join(save_dir, name))
    plt.close()

def visualize_spatial_attention(x_np, attention_np, save_dir, timestamp, step_name=None):
    """可视化空间注意力"""
    plt.figure(figsize=(15, 10))
    
    # 1. 原始信号强度
    plt.subplot(3, 1, 1)
    plt.title("Original Signal Intensity")
    plt.imshow(x_np, aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.ylabel("Channel")
    
    # 2. 空间注意力权重
    plt.subplot(3, 1, 2)
    plt.title("Spatial Attention Weights")
    plt.plot(attention_np.squeeze())
    plt.xlabel("Time Step")
    plt.ylabel("Attention Weight")
    
    # 3. 加权后的信号
    plt.subplot(3, 1, 3)
    plt.title("Weighted Signal")
    weighted_signal = x_np * attention_np.T
    plt.imshow(weighted_signal, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weighted Amplitude')
    plt.ylabel("Channel")
    
    # 保存图像
    plt.tight_layout()
    name = f"spatial_attention_{step_name}_{timestamp}.png" if step_name else f"spatial_attention_{timestamp}.png"
    plt.savefig(os.path.join(save_dir, name))
    plt.close()

def visualize_cbam_attention(x_np, channel_attention_np, spatial_attention_np, 
                           save_dir, timestamp, step_name=None, channels=None):
    """可视化CBAM注意力"""
    plt.figure(figsize=(15, 15))
    
    # 1. 原始信号
    plt.subplot(4, 1, 1)
    plt.title("Original Signal")
    for i in range(x_np.shape[0]):
        label = f"Channel {i}" if channels is None else channels[i]
        plt.plot(x_np[i], label=label)
    plt.legend()
    
    # 2. 通道注意力权重
    plt.subplot(4, 1, 2)
    plt.title("Channel Attention Weights")
    weights = channel_attention_np.squeeze()
    bars = plt.bar(range(len(weights)), weights)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    if channels:
        plt.xticks(range(len(weights)), channels, rotation=45)
    
    # 3. 空间注意力权重
    plt.subplot(4, 1, 3)
    plt.title("Spatial Attention Weights")
    plt.plot(spatial_attention_np.squeeze())
    plt.xlabel("Time Step")
    plt.ylabel("Weight")
    
    # 4. 最终加权信号
    plt.subplot(4, 1, 4)
    plt.title("Final Weighted Signal")
    weighted_signal = x_np * channel_attention_np * spatial_attention_np.T
    for i in range(weighted_signal.shape[0]):
        label = f"Channel {i}" if channels is None else channels[i]
        plt.plot(weighted_signal[i], label=label)
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    name = f"cbam_attention_{step_name}_{timestamp}.png" if step_name else f"cbam_attention_{timestamp}.png"
    plt.savefig(os.path.join(save_dir, name))
    plt.close()
