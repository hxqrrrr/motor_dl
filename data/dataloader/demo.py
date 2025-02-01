import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 加载.mat文件
file_path = r"C:\Users\hxq11\Downloads\C-CWRUdata10.mat"
data = sio.loadmat(file_path)

# 查看文件中的键
print("文件包含的键：", data.keys())

# 检查并打印每个键对应的数据信息
for key in data.keys():
    if key.startswith('__'):  # 跳过元数据键
        continue
    
    print(f"\n正在分析键: {key}")
    value = data[key]
    
    if isinstance(value, np.ndarray):
        print(f"数据类型: {value.dtype}")
        print(f"数组形状: {value.shape}")
        
        # 如果是2D数组，绘制第一个样本
        if len(value.shape) == 2:
            plt.figure(figsize=(12, 6))
            plt.plot(value[0])
            plt.title(f'{key} 第一个样本')
            plt.xlabel('采样点')
            plt.ylabel('幅值')
            plt.grid(True)
            plt.show()
            
            # 打印统计信息
            print(f"样本数量: {value.shape[0]}")
            print(f"信号长度: {value.shape[1]}")
            print(f"最大值: {value.max()}")
            print(f"最小值: {value.min()}")
            print(f"均值: {value.mean()}")
    else:
        print(f"非数组数据: {type(value)}")
