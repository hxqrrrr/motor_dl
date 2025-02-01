import torch
from torch.utils.data import DataLoader
from data.dataloader.cwru_dataset import CWRUDataset

if __name__ == "__main__":
    data_dir = 'data/CWRU/12k Drive End Bearing Fault Data'
    dataset = CWRUDataset(data_dir)
    
    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 示例：获取一个batch的数据
    for signals, labels in dataloader:
        print(f"Batch signals shape: {signals.shape}")
        print(f"Batch labels: {labels}")
        