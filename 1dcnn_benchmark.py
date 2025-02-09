import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataloader.updated_cwru_dataloader import CWRUDataset,create_dataloader_mckn
from models.cnn1d import CNN1D,CNN1D_shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import time
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler


# Train and evaluate functions
def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (signals, labels) in enumerate(train_loader):
        signals, labels = signals.to(device), labels.to(device)

        # Forward pass
        outputs = model(signals)
        loss = criterion(outputs, labels) / accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    return val_loss / len(val_loader), val_acc

# Initialize model weights
def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

# Main block
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    num_epochs = 100
    batch_size = 20
    learning_rate = 0.001
    accumulation_steps = 4
    sequence_length = 240

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    data_dir = 'data/CWRU_10Class_Verified'
    dataset = CWRUDataset(data_dir, signal_length=1200, signal_count_per_label=1000, 
                          transform=None, scale=True, downsample_ratio=1, num_classes=10,
                          window_size=240, stride=60)
    
    scaler = StandardScaler()
    dataset.signals = scaler.fit_transform(dataset.signals)
    
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    # Model setup
    # model = CNN1D_shuffle(sequence_length=sequence_length, input_channels=1, num_classes=10).to(device)
    model = CNN1D(sequence_length=sequence_length, input_channels=1, num_classes=10).to(device)
    model.apply(init_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # Train and evaluate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Logging
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save model
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')