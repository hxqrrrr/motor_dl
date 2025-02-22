import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft, ifft
import torch
import matplotlib.pyplot as plt
import time
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

def vmd(signal, alpha, tau, K, DC=False, init=False, tol=1e-7, use_gpu=True):
    """
    变分模态分解的Python实现（支持GPU加速）
    
    参数:
    signal: 输入信号
    alpha: 带宽约束
    tau: 噪声容限
    K: 分解模态数
    DC: 是否包含直流分量
    init: 是否初始化中心频率
    tol: 收敛容限
    use_gpu: 是否使用GPU加速
    
    返回:
    u: 分解得到的模态
    u_hat: 模态的傅里叶变换
    omega: 中心频率
    """
    # 检查是否可以使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    
    # 如果输入是numpy标量，转换为数组
    if isinstance(signal, (np.float32, np.float64)):
        signal = np.array([signal])
    elif isinstance(signal, (int, float)):
        signal = np.array([signal])
    
    # 确保信号是一维数组
    signal = np.asarray(signal)
    if signal.ndim != 1:
        signal = signal.flatten()
    
    # 添加小量以避免除零
    eps = np.finfo(float).eps
    
    # 信号长度
    N = len(signal)
    
    # 计算采样频率
    fs = 1.0
    T = N/fs
    t = np.arange(N)/fs
    
    # 将信号和相关数组转换为PyTorch张量
    signal_tensor = torch.from_numpy(signal).to(device)
    f_mirror = torch.arange(-N//2, N//2, device=device, dtype=torch.float32)/T
    
    # 对信号进行傅里叶变换
    f_signal = torch.fft.fft(signal_tensor)
    f_signal = torch.fft.fftshift(f_signal)
    
    # 初始化
    u_hat = torch.zeros((K, N), dtype=torch.complex64, device=device)
    omega = torch.zeros((K, 1), device=device)
    
    if init:
        for i in range(K):
            omega[i] = (0.5/K)*(i)
    else:
        omega = torch.rand((K, 1), device=device)
        
    # 双重算子
    lambda_hat = torch.zeros(N, dtype=torch.complex64, device=device)
    
    # 存储上一次迭代的结果
    u_hat_old = torch.zeros_like(u_hat) + eps
    omega_old = torch.zeros_like(omega)
    
    # 迭代更新
    n = 0
    uDiff = tol + 1
    while (uDiff > tol and n < 500):  # 最大迭代500次
        n = n + 1
        
        # 更新第一个模态到第K个模态
        for i in range(K):
            # 计算其他模态的和
            sum_uk = torch.zeros(N, dtype=torch.complex64, device=device)
            for j in range(K):
                if j != i:
                    sum_uk = sum_uk + u_hat[j]
            
            # 更新模态
            denom = 1 + alpha*(f_mirror - omega[i])**2 + eps
            u_hat[i] = (f_signal - sum_uk + lambda_hat/2)/denom
            
            # 更新中心频率
            num = torch.sum(f_mirror[N//4:3*N//4]*torch.abs(u_hat[i, N//4:3*N//4])**2)
            den = torch.sum(torch.abs(u_hat[i, N//4:3*N//4])**2) + eps
            omega[i] = num/den
        
        # 双重更新
        lambda_hat = lambda_hat + tau*(torch.sum(u_hat, dim=0) - f_signal)
        
        # 计算收敛条件
        uDiff = torch.sum(torch.abs(u_hat - u_hat_old)**2)/(torch.sum(torch.abs(u_hat_old)**2) + eps)
        omega_diff = torch.sum(torch.abs(omega - omega_old)**2)/(torch.sum(torch.abs(omega_old)**2) + eps)
        
        # 保存当前值作为下一次迭代的旧值
        u_hat_old = u_hat.clone()
        omega_old = omega.clone()
    
    # 重构信号
    u = torch.zeros((K, N), device=device)
    u_hat = torch.fft.ifftshift(u_hat, dim=1)
    
    for i in range(K):
        u[i] = torch.real(torch.fft.ifft(u_hat[i]))
    
    # 计算重构误差
    recon = torch.sum(u, dim=0)
    error = torch.mean((recon - signal_tensor)**2)
    
    # 转回CPU和numpy
    u = u.cpu().numpy()
    u_hat = u_hat.cpu().numpy()
    omega = omega.cpu().numpy()
    error = error.cpu().numpy()
    
    return u, u_hat, omega, error

def apply_vmd_features(signal, alpha=2000, tau=0, K=3, use_gpu=True):
    """
    signal: 输入信号
    alpha: 带宽约束参数（默认2000）
        - 控制分解的带宽
        - 值越大，分解得到的模态带宽越窄
        - 值越小，分解得到的模态带宽越宽
    
    tau: 噪声容限参数（默认0）
        - 控制对噪声的敏感度
        - 值越大，对噪声越不敏感
        - 值越小，分解越精细
    
    K: 分解模态数（默认3）
        - 将信号分解成几个分量
        - 需要根据实际信号特征选择
    """
    # 应用VMD
    u, _, omega, error = vmd(signal, alpha, tau, K, use_gpu=use_gpu)
    
    # 提取特征
    features = []
    
    for i in range(K):
        imf = u[i, :]
        # 时域特征
        features.extend([
            np.mean(imf),           # 均值
            np.std(imf),            # 标准差
            np.max(imf),            # 最大值
            np.min(imf),            # 最小值
            np.sum(imf**2),         # 能量
            np.mean(np.abs(imf)),   # 平均幅值
        ])
        
        # 频域特征
        fft_imf = np.abs(fft(imf))
        features.extend([
            np.mean(fft_imf),       # 频域均值
            np.std(fft_imf),        # 频域标准差
            np.max(fft_imf),        # 频域最大值
            omega[i][0]             # 中心频率
        ])
    
    # 添加重构误差
    features.append(error)
    
    return np.array(features)

def     extract_vmd_features(signals, **vmd_params):
    """
    批量处理信号并提取VMD特征（支持GPU加速）
    
    参数:
    signals: 输入信号数组 [n_samples, signal_length]
    vmd_params: VMD参数
    
    返回:
    features: 特征矩阵 [n_samples, n_features]
    """
    features = []
    for signal in signals:
        feat = apply_vmd_features(signal, **vmd_params)
        features.append(feat)
    return np.array(features) 

def batch_vmd(signals, alpha, tau, K, batch_size=32, use_gpu=True):
    """
    批量处理信号的 VMD 分解
    
    参数:
    signals: 输入信号数组 [n_samples, signal_length]
    alpha: 带宽约束
    tau: 噪声容限
    K: 分解模态数
    batch_size: 批处理大小
    use_gpu: 是否使用 GPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    n_samples = len(signals)
    results = []
    
    for i in range(0, n_samples, batch_size):
        batch = signals[i:min(i+batch_size, n_samples)]
        batch_tensor = torch.from_numpy(batch).to(device)
        
        # 批量处理 VMD
        batch_results = []
        for signal in batch_tensor:
            u, u_hat, omega, error = vmd(signal.cpu().numpy(), alpha, tau, K, use_gpu=use_gpu)
            batch_results.append((u, u_hat, omega, error))
        
        results.extend(batch_results)
        
        if (i + batch_size) % 1000 == 0:
            print(f"已处理 {i + batch_size}/{n_samples} 个样本")
    
    return results

def batch_extract_vmd_features(signals, **vmd_params):
    """
    批量提取 VMD 特征
    
    参数:
    signals: 输入信号数组 [n_samples, signal_length]
    vmd_params: VMD 参数
    """
    batch_size = vmd_params.pop('batch_size', 32)
    use_gpu = vmd_params.pop('use_gpu', True)
    
    # 使用批处理进行 VMD 分解
    results = batch_vmd(signals, batch_size=batch_size, use_gpu=use_gpu, **vmd_params)
    
    # 提取特征
    features = []
    for u, _, omega, error in results:
        feat = []
        for i in range(len(u)):
            imf = u[i]
            # 时域特征
            feat.extend([
                np.mean(imf),
                np.std(imf),
                np.max(imf),
                np.min(imf),
                np.sum(imf**2),
                np.mean(np.abs(imf))
            ])
            
            # 频域特征
            fft_imf = np.abs(fft(imf))
            feat.extend([
                np.mean(fft_imf),
                np.std(fft_imf),
                np.max(fft_imf),
                omega[i][0]
            ])
        
        # 添加重构误差
        feat.append(error)
        features.append(feat)
    
    return np.array(features)

def generate_test_signal(t):
    """生成测试信号：包含多个频率分量"""
    # 高频信号
    s1 = 1.0 * np.cos(2.0 * np.pi * 2.0 * t)
    # 中频信号
    s2 = 0.5 * np.cos(2.0 * np.pi * 1.0 * t)
    # 低频信号
    s3 = 0.3 * np.cos(2.0 * np.pi * 0.3 * t)
    # 添加噪声
    noise = 0.1 * np.random.randn(len(t))
    
    # 合成信号
    signal = s1 + s2 + s3 + noise
    return signal, [s1, s2, s3]

def plot_decomposition(t, original_signal, components, u, error):
    """绘制分解结果"""
    plt.figure(figsize=(12, 8))
    
    # 绘制原始信号
    plt.subplot(5, 1, 1)
    plt.plot(t, original_signal)
    plt.title('原始信号')
    plt.grid(True)
    
    # 绘制真实分量
    plt.subplot(5, 1, 2)
    for i, comp in enumerate(components):
        plt.plot(t, comp, label=f'真实分量 {i+1}')
    plt.legend()
    plt.title('真实信号分量')
    plt.grid(True)
    
    # 绘制VMD分解结果
    plt.subplot(5, 1, 3)
    for i in range(u.shape[0]):
        plt.plot(t, u[i, :], label=f'VMD分量 {i+1}')
    plt.legend()
    plt.title(f'VMD分解结果 (重构误差: {error:.6f})')
    plt.grid(True)
    
    # 绘制重构信号
    plt.subplot(5, 1, 4)
    reconstructed = np.sum(u, axis=0)
    plt.plot(t, original_signal, 'b', label='原始信号')
    plt.plot(t, reconstructed, 'r--', label='重构信号')
    plt.legend()
    plt.title('信号重构对比')
    plt.grid(True)
    
    # 绘制误差
    plt.subplot(5, 1, 5)
    plt.plot(t, original_signal - reconstructed)
    plt.title('重构误差')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # 生成测试信号
    t = np.linspace(0, 10, 1000)
    signal, components = generate_test_signal(t)
    
    print("开始VMD分解...")
    start_time = time.time()
    
    # 应用VMD
    K = 4  # 分解模态数
    alpha = 2000  # 带宽约束
    tau = 0  # 噪声容限
    
    u, u_hat, omega, error = vmd(signal, alpha, tau, K)
    
    end_time = time.time()
    print(f"VMD分解完成，用时: {end_time - start_time:.2f} 秒")
    print(f"重构误差: {error:.6f}")
    
    # 提取特征
    features = apply_vmd_features(signal, alpha=alpha, tau=tau, K=K)
    print("\nVMD特征:")
    print("-" * 40)
    feature_names = [
        "均值", "标准差", "最大值", "最小值", "能量", "平均幅值",
        "频域均值", "频域标准差", "频域最大值", "中心频率"
    ]
    
    for i in range(K):
        print(f"\n模态 {i+1} 特征:")
        for j, name in enumerate(feature_names):
            print(f"{name}: {features[i*len(feature_names) + j]:.6f}")
    
    print(f"\n重构误差: {features[-1]:.6f}")
    
    # 绘制结果
    plot_decomposition(t, signal, components, u, error)

if __name__ == "__main__":
    main() 