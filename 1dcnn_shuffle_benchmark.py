import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataloader.updated_cwru_dataloader import CWRUDataset
from models.cnn1d import ShuffleNet1DClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from torch.profiler import profile, record_function, ProfilerActivity
import torch.profiler
from pathlib import Path
import os


# Train and evaluate functions
def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps, profiler=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for i, (signals, labels) in enumerate(train_loader):
        if profiler is not None:
            profiler.step()
            
        # 确保数据在正确的设备上
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(signals)
        loss = criterion(outputs, labels) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # 优化器更新
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item()

        if i >= 100:  # 只分析前100个batch
            break
            
    return total_loss / len(train_loader)

def count_parameters(model):
    """计算模型每一层的参数量并进行可视化分析"""
    total_params = 0
    trainable_params = 0
    print("\n模型参数分析:")
    print("-" * 40)
    print("层级参数统计:")
    
    # 按层统计参数
    layer_params = {}
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        if parameter.requires_grad:
            trainable_params += param
            # 提取层名（去除子层级）
            layer_name = name.split('.')[0]
            layer_params[layer_name] = layer_params.get(layer_name, 0) + param
        total_params += param
    
    # 打印每层的参数量和占比
    for layer_name, params in layer_params.items():
        percentage = (params / trainable_params) * 100
        print(f"{layer_name:20s}: {params:8,d} 参数 ({percentage:5.1f}%)")
    
    print("\n参数量总结:")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"总参数量: {total_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")  # 假设每个参数占4字节
    
    return total_params

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    batch_times = []  # 记录每个batch的处理时间
    
    with torch.no_grad():
        for signals, labels in val_loader:
            batch_start = time.time()
            # 确保数据在正确的设备上
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            batch_times.append(time.time() - batch_start)

    val_acc = 100 * correct / total
    avg_batch_time = sum(batch_times) / len(batch_times)
    inference_speed = len(val_loader.dataset) / sum(batch_times)
    
    return val_loss / len(val_loader), val_acc, avg_batch_time, inference_speed

# Initialize model weights
def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

def show_cuda_info():
    # CUDA 诊断信息
    print("\nCUDA 环境检查:")
    print("-" * 40)
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
        print(f"设备数量: {torch.cuda.device_count()}")
        print(f"当前设备索引: {torch.cuda.current_device()}")
    
    # 确保模型和数据都在正确的设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    if device.type == "cuda":
        print(f"显存使用情况:")
        print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
        print(f"当前占用: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
        print(f"缓存占用: {torch.cuda.memory_reserved() / 1024**2:.0f} MB")

# Main block
if __name__ == "__main__":
    show_cuda_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.005
    accumulation_steps = 4
    sequence_length = 360

    # Data preparation
    data_dir = 'data/CWRU_10Class_Verified'
    dataset = CWRUDataset(data_dir, transform=None,batch_size=batch_size, scale=True, downsample_ratio=1, num_classes=10, window_size=sequence_length, stride=360)
    
    scaler = StandardScaler()
    dataset.signals = scaler.fit_transform(dataset.signals)
    
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)

    # 打印数据集信息
    print("\n数据集信息：")
    print("=" * 50)
    print(f"总样本数: {len(dataset)}")
    print(f"训练集样本数: {len(train_idx)}")
    print(f"验证集样本数: {len(val_idx)}")
    print(f"Batch大小: {batch_size}")
    print(f"理论训练batch数: {len(train_idx) // batch_size}")
    print("=" * 50)

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx), 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # 使用多进程加载数据
        pin_memory=True,  # 使用固定内存加速数据传输
        persistent_workers=True,  # 保持工作进程存活
        prefetch_factor=2  # 预取因子
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx), 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Model setup
    model = ShuffleNet1DClassifier(input_channels=1, num_classes=10, groups=1)
    if torch.cuda.is_available():
        model = model.cuda()  # 使用 cuda() 而不是 to(device)
        # 启用 cudnn 基准测试模式
        torch.backends.cudnn.benchmark = True
    model.apply(init_weights)

    # 打印模型结构和参数量
    print("\n模型结构和参数统计：")
    print("=" * 50)
    count_parameters(model)
    print("=" * 50)

    # 进行一次推理来显示每层的时间统计
    print("\n首次推理时间统计：")
    print("=" * 50)
    with torch.no_grad():
        test_input = torch.randn(batch_size, 1, sequence_length).to(device)
        _, times = model(test_input, return_times=True)
        for key, value in times.items():
            print(f"{key} 时间: {value:.6f} seconds")
    print("=" * 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    # 创建profiler输出目录
    log_dir = Path("./runs/profiler")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n开始性能分析...")
    print("=" * 50)
    
    # 使用更简单的profiler配置
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(log_dir)),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        # 进行训练和性能分析
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps, profiler=prof)
    
    # 打印性能分析结果
    print("\n性能分析结果:")
    print("=" * 50)
    
    print("\n1. CPU 操作分析 (前10个最耗时的操作):")
    print("-" * 40)
    cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    print(cpu_table)
    
    if torch.cuda.is_available():
        print("\n2. GPU 操作分析:")
        print("-" * 40)
        print("2.1 按总 CUDA 时间排序 (前5个最耗时的操作):")
        cuda_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=5)
        print(cuda_table)
        
        print("\n2.2 GPU 内存使用情况:")
        memory_table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5)
        print(memory_table)
    
    print("\n3. 内存分配分析:")
    print("-" * 40)
    print("3.1 CPU 内存分配 (前5个最大的内存分配操作):")
    memory_table = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=5)
    print(memory_table)
    
    # 计算关键性能指标
    events = prof.key_averages()
    total_cpu_time = sum(event.cpu_time_total for event in events)
    
    print("\n4. 性能总结:")
    print("-" * 40)
    print(f"总 CPU 时间: {total_cpu_time/1000:.2f} ms")
    
    if torch.cuda.is_available():
        cuda_events = [event for event in events if event.device_type == 'cuda']
        if cuda_events:
            cuda_time = sum(event.self_cuda_time_total for event in cuda_events)
            print(f"总 CUDA 时间: {cuda_time/1000:.2f} ms")
            print(f"CPU/CUDA 时间比: {total_cpu_time/cuda_time:.2f}")
    
    # 分析潜在的性能瓶颈
    print("\n5. 性能瓶颈分析:")
    print("-" * 40)
    
    # 找出最耗时的操作
    sorted_events = sorted(events, key=lambda x: x.cpu_time_total, reverse=True)
    if len(sorted_events) > 0:
        top_event = sorted_events[0]
        print(f"最耗时的操作: {top_event.key}")
        print(f"- CPU 时间: {top_event.cpu_time_total/1000:.2f} ms")
        print(f"- 占总时间比例: {(top_event.cpu_time_total/total_cpu_time)*100:.1f}%")
    
    # 分析内存使用
    sorted_by_memory = sorted(events, key=lambda x: x.self_cpu_memory_usage, reverse=True)
    if len(sorted_by_memory) > 0:
        top_memory = sorted_by_memory[0]
        print(f"\n最大内存使用的操作: {top_memory.key}")
        print(f"- 内存使用: {top_memory.self_cpu_memory_usage/1024/1024:.2f} MB")
    
    print("\n6. 优化建议:")
    print("-" * 40)
    print("- 模型参数量小，适合部署在资源受限的环境")
    if torch.cuda.is_available():
        cuda_events = [event for event in events if event.device_type == 'cuda']
        if cuda_events:
            cuda_time = sum(event.self_cuda_time_total for event in cuda_events)
            if total_cpu_time > cuda_time * 2:
                print("- CPU 处理时间显著大于 GPU 时间，考虑优化数据加载和预处理")
            if any(event.self_cuda_memory_usage > 1e9 for event in cuda_events):
                print("- GPU 内存使用较大，考虑使用梯度检查点或减小批次大小")
    
    print("\n性能分析完成")
    print("=" * 50)
    print(f"\n性能分析数据已保存到: {log_dir}")
    print("要查看详细的性能分析，请运行:")
    print(f"tensorboard --logdir={log_dir}")
    print("\n提示: 在 TensorBoard 中，您可以:")
    print("1. 查看详细的操作时间线")
    print("2. 分析内存使用趋势")
    print("3. 查看 GPU 利用率")
    print("4. 检查操作间的依赖关系")
    
    print("\n开始正常训练...")
    print("=" * 50)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train and evaluate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps)
        val_loss, val_acc, avg_batch_time, inference_speed = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Logging
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch [{epoch + 1}/{num_epochs}] 用时: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save model
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')

    # 模型评估和性能统计
    print("\n模型性能评估:")
    print("=" * 50)
    
    # 1. 参数效率分析
    params_count = count_parameters(model)
    
    # 2. 推理速度测试
    val_loss, val_acc, avg_batch_time, inference_speed = evaluate(model, val_loader, criterion, device)
    
    print("\n推理性能:")
    print("-" * 40)
    print(f"验证集准确率: {val_acc:.2f}%")
    print(f"平均批次处理时间: {avg_batch_time*1000:.2f} ms")
    print(f"推理速度: {inference_speed:.0f} 样本/秒")
    
    print("\n模型效率指标:")
    print("-" * 40)
    print(f"参数量/准确率比: {params_count/val_acc:.0f} (参数数量/准确率百分比)")
    print(f"准确率/处理时间比: {val_acc/(avg_batch_time*1000):.2f} (%/ms)")
    
    print("\n结论:")
    print("-" * 40)
    print(f"该模型使用 {params_count:,} 个参数")
    print(f"在 {avg_batch_time*1000:.2f} ms 内完成单个批次处理")
    print(f"达到 {val_acc:.2f}% 的分类准确率")