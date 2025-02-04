import torch
from torch.utils.data import DataLoader
from data.dataloader.cwru_dataset import CWRUDataset
from models.lstm1d import LSTM1D
from sklearn.model_selection import train_test_split
import numpy as np
import time
from torch.cuda.amp import GradScaler, autocast

if __name__ == "__main__":
    # 设置随机种子保证可重复性

    # 超参数设置
    num_epochs = 50
    batch_size = 2
    learning_rate = 0.001
    early_stop_patience = 5
    accumulation_steps = 4  # 梯度累积步数

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # 数据加载
    data_dir = 'data/CWRU/12k Drive End Bearing Fault Data'
    dataset = CWRUDataset(data_dir,downsample_ratio=4)
    print(len(dataset))
    # 划分训练集和验证集
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = LSTM1D(input_channels=1, sequence_length=30000,num_classes=12).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 创建优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 创建损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 混合精度训练
    scaler = GradScaler()

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        # 训练阶段
        optimizer.zero_grad()
        for i, (signals, labels) in enumerate(train_loader):
            signals = signals.to(device)
            labels = labels.to(device)
            with autocast():
                outputs = model(signals)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

        # 验证阶段
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

        # 计算平均损失和准确率
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        # 学习率调度
        scheduler.step()

        # 打印训练信息
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_LSTM_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break