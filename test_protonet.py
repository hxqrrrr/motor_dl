import torch
import numpy as np
from models.ProtoNet import ProtoNet

def generate_sinusoidal_data(n_samples: int, sequence_length: int, frequency: float, noise_level: float = 0.1) -> torch.Tensor:
    """生成带有噪声的正弦波数据
    
    参数:
        n_samples: 样本数量
        sequence_length: 序列长度
        frequency: 正弦波频率
        noise_level: 噪声水平
        
    返回:
        数据张量 [n_samples, 1, sequence_length]
    """
    t = np.linspace(0, 10, sequence_length)
    data = []
    
    for _ in range(n_samples):
        # 生成正弦波
        signal = np.sin(2 * np.pi * frequency * t)
        # 添加随机噪声
        noise = np.random.normal(0, noise_level, sequence_length)
        # 合并信号和噪声
        sample = signal + noise
        data.append(sample)
    
    # 转换为tensor并添加通道维度
    return torch.FloatTensor(data).unsqueeze(1)

def test_protonet():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 模型参数
    in_channels = 1  # 输入通道数
    hidden_dim = 64  # 隐藏层维度
    feature_dim = 32  # 特征维度
    sequence_length = 100  # 序列长度
    
    # 创建ProtoNet模型
    model = ProtoNet(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        backbone='cnn1d',
        distance_type='euclidean'
    )
    
    # 数据参数
    n_classes = 5  # 类别数
    n_support = 5  # 每个类别的support样本数
    n_query = 3    # 每个类别的query样本数
    
    # 为每个类别生成不同频率的数据
    frequencies = [0.5, 1.0, 1.5, 2.0, 2.5]  # 不同类别的频率
    support_images = []
    query_images = []
    
    for freq in frequencies:
        # 生成support set数据
        support_data = generate_sinusoidal_data(n_support, sequence_length, freq)
        support_images.append(support_data)
        
        # 生成query set数据
        query_data = generate_sinusoidal_data(n_query, sequence_length, freq)
        query_images.append(query_data)
    
    # 合并数据
    support_images = torch.cat(support_images, dim=0)
    query_images = torch.cat(query_images, dim=0)
    
    # 生成标签
    support_labels = torch.tensor([i for i in range(n_classes) for _ in range(n_support)])
    query_labels = torch.tensor([i for i in range(n_classes) for _ in range(n_query)])
    
    # 将模型设置为评估模式
    model.eval()
    
    print("测试数据形状：")
    print(f"Support set: {support_images.shape}")
    print(f"Support labels: {support_labels.shape}")
    print(f"Query set: {query_images.shape}")
    print(f"Query labels: {query_labels.shape}")
    
    # 可视化第一个类别的数据
    print("\n第一个类别的数据示例（前两个support样本）：")
    print("Support sample 1:", support_images[0, 0, :10].numpy())
    print("Support sample 2:", support_images[1, 0, :10].numpy())
    
    # 前向传播
    with torch.no_grad():
        logits, prototypes = model(support_images, support_labels, query_images)
    
    print("\n模型输出形状：")
    print(f"Logits: {logits.shape}")
    print(f"Prototypes: {prototypes.shape}")
    
    # 计算预测结果
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == query_labels).float().mean()
    
    print("\n预测结果：")
    print(f"Predictions: {predictions}")
    print(f"True labels: {query_labels}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # 测试余弦距离
    print("\n测试余弦距离：")
    model.distance_type = 'cosine'
    with torch.no_grad():
        logits_cos, prototypes_cos = model(support_images, support_labels, query_images)
    
    predictions_cos = torch.argmax(logits_cos, dim=1)
    accuracy_cos = (predictions_cos == query_labels).float().mean()
    
    print(f"Cosine distance accuracy: {accuracy_cos:.4f}")

    # 打印每个类别的准确率
    print("\n每个类别的准确率：")
    for class_idx in range(n_classes):
        mask = query_labels == class_idx
        class_acc = (predictions[mask] == query_labels[mask]).float().mean()
        print(f"类别 {class_idx} (频率 {frequencies[class_idx]} Hz): {class_acc:.4f}")

if __name__ == "__main__":
    test_protonet() 