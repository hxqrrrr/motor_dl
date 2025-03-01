import h5py
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
# 获取当前文件所在目录的上两级目录（项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
print(f"添加到Python路径: {project_root}")

# 现在可以导入models模块
from models.dataset import h5Dataset

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前脚本目录: {current_dir}")

# Set PYTHONPATH
os.environ["PYTHONPATH"] = project_root

# 直接从原始h5文件加载数据
print("\n直接从原始h5文件加载数据...")
original_h5_file = os.path.join(current_dir, "data.h5")

if not os.path.exists(original_h5_file):
    print(f"错误：找不到原始数据文件 {original_h5_file}")
    sys.exit(1)

print(f"处理文件: {original_h5_file}")
with h5py.File(original_h5_file, 'r') as f:
    if 'data' in f and 'labels' in f:
        data = f['data'][:]
        labels = f['labels'][:]
        print(f"  数据形状: {data.shape}")
        print(f"  标签形状: {labels.shape}")
        
        # 将标签转换为单一类别索引
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            print("将one-hot标签转换为类别索引...")
            indices = np.argmax(labels, axis=1)
        else:
            indices = labels
        
        print("标签索引形状:", indices.shape)
        
        # 筛选指定标签的数据
        selected_labels = [0, 2, 4, 6]
        print(f"\n筛选标签为 {selected_labels} 的数据...")
        selected_mask = np.isin(indices, selected_labels)
        selected_data = data[selected_mask]
        selected_indices = indices[selected_mask]
        remaining_data = data[~selected_mask]
        remaining_indices = indices[~selected_mask]
        
        print("筛选后数据形状:")
        print("选中数据形状:", selected_data.shape)
        print("选中标签形状:", selected_indices.shape)
        print("剩余数据形状:", remaining_data.shape)
        print("剩余标签形状:", remaining_indices.shape)
        
        # 保存到 HDF5 文件
        print("\n保存数据到HDF5文件...")
        selected_output_file = os.path.join(current_dir, 'selected_data.h5')
        with h5py.File(selected_output_file, 'w') as f_out:
            f_out.create_dataset('data', data=selected_data)
            f_out.create_dataset('labels', data=selected_indices)
            print(f"已保存选中数据到 {selected_output_file}")
        
        remaining_output_file = os.path.join(current_dir, 'remaining_data.h5')
        with h5py.File(remaining_output_file, 'w') as f_out:
            f_out.create_dataset('data', data=remaining_data)
            f_out.create_dataset('labels', data=remaining_indices)
            print(f"已保存剩余数据到 {remaining_output_file}")
        
        # 打开 HDF5 文件验证
        with h5py.File(selected_output_file, 'r') as f_verify:
            # 查看文件中的所有数据集
            print("\n验证 selected_data.h5:")
            print("数据集名称:", list(f_verify.keys()))
            
            # 读取数据和标签
            verify_data = f_verify['data'][:]
            verify_labels = f_verify['labels'][:]
            
            print("数据形状:", verify_data.shape)
            print("标签形状:", verify_labels.shape)
            
            # 获取唯一标签
            unique_labels = np.unique(verify_labels)
            print("唯一标签:", unique_labels)
            
            # 统计每个标签的样本数
            for label in unique_labels:
                count = np.sum(verify_labels == label)
                print(f"标签 {label}: {count} 个样本")
        
        print("\n数据处理完成!")
    else:
        print("错误：数据文件中没有'data'或'labels'数据集")