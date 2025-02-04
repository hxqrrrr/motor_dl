import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataloader.cwru_dataset import CWRUDataset
from models.cnn1d import CNN1D
from sklearn.model_selection import train_test_split
import numpy as np
import time
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (signals, labels) in enumerate(train_loader):
        signals = signals.to(device)
        labels = labels.to(device)

        with autocast():
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # 使用梯度累积

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / (len(train_loader) / accumulation_steps)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    return val_loss / len(val_loader), val_acc

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


if __name__ == "__main__":
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 超参数设置
    num_epochs = 50
    batch_size = 10
    learning_rate = 0.00001  # 尝试更大的学习率
    accumulation_steps = 4  # 梯度累积步数
    sequence_length = 30000

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    data_dir = 'data/CWRU/12k Drive End Bearing Fault Data'
    dataset = CWRUDataset(data_dir, downsample_ratio=4, truncate_length=sequence_length)
    scaler = StandardScaler()
    dataset.signals = scaler.fit_transform(dataset.signals)
    # 划分训练集和验证集
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = CNN1D(sequence_length=sequence_length, input_channels=1, num_classes=12).to(device)
    model.apply(init_weights)
    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 混合精度训练
    scaler = GradScaler()

    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练阶段
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps)

        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 学习率调度
        scheduler.step(val_loss)

        # 打印训练信息
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存模型
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')