# 小样本学习项目

这是一个基于PyTorch实现的小样本学习（Few-shot Learning）项目，主要包含原型网络（Prototypical Networks）及其改进版本的实现。

## 项目结构

```
.
├── data/               # 数据集目录
├── models/            # 模型定义
├── utils/             # 工具函数
├── runs/              # 训练日志和模型保存
├── test/              # 测试脚本
└── img/               # 文档图片
```

## 主要特性

- 支持多种深度学习模型：
  - 原型网络 (ProtoNet)
  - 带注意力机制的原型网络 (ProtoNet with Attention)
  - 1D-CNN
  - 1D-LSTM
- 提供完整的训练和评估流程
- 支持模型预训练和继续训练
- 提供详细的训练过程可视化
- 支持多种距离度量方式

## 环境要求

请确保您的环境满足以下要求：

```bash
torch>=1.8.0
numpy
h5py
matplotlib
tqdm
```

完整的依赖列表请参见 `requirements.txt`。

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备数据：
   - 将数据集放在 `data/h5data` 目录下

3. 训练模型：
```bash
# 训练基础原型网络
python train.py --model protonet

# 训练带注意力机制的原型网络
python train.py --model protonet_attention

# 使用预训练模型继续训练
python train.py --model protonet --pretrained runs/protonet_XXXXXX/best_model.pth
```

4. 评估模型：
```bash
python benchmark.py --model_path runs/protonet_XXXXXX/best_model.pth
```

## 主要配置参数

### 模型配置
- `in_channels`: 输入通道数
- `hidden_dim`: 隐藏层维度
- `feature_dim`: 特征维度
- `backbone`: 主干网络类型 ('cnn1d' 或 'lstm1d')
- `distance_type`: 距离度量类型 ('euclidean' 或 'cosine')

### 训练配置
- `n_way`: N-way分类数
- `n_support`: 支持集样本数
- `n_query`: 查询集样本数
- `batch_size`: 批次大小
- `n_epochs`: 训练轮数
- `patience`: 早停耐心值
- `test_interval`: 测试间隔

### 优化器配置
- `lr`: 学习率
- `weight_decay`: 权重衰减
- `step_size`: 学习率调整步长
- `gamma`: 学习率衰减因子

## 文件说明

- `train.py`: 主训练脚本
- `train_ProtoNet.py`: 原型网络训练脚本
- `train_ProtoNet_attention.py`: 带注意力机制的原型网络训练脚本
- `benchmark.py`: 模型基准测试脚本
- `models/dataset.py`: 数据集处理
- `utils/utils.py`: 工具函数

## 训练结果

训练过程中的日志、模型和可视化结果将保存在 `runs` 目录下，格式为：
```
runs/
└── model_name_timestamp/
    ├── best_model.pth
    ├── final_model.pth
    ├── training_curves.png
    └── params.json
```

## 注意事项

1. 确保数据集格式正确且已正确放置
2. 训练前检查GPU内存是否充足
3. 可以通过修改配置参数调整模型性能
4. 建议使用GPU进行训练以获得更好的性能

## 许可证

本项目采用 MIT 许可证

# 电机故障诊断深度学习项目

## 项目主页：[链接](https://www.notion.so/13d42872c05480b88ec4ef624a233933?pvs=4)

## 项目介绍：
- 本项目使用深度学习方法对电机故障进行诊断分类
- 本框架是一个利用json文件在不同数据集测试不同模型性能的平台

---

## 工具


- 可视化工具使用[tensorboard](https://www.tensorflow.org/tensorboard?hl=zh-cn)

- TensorBoard教程：https://kuanhoong.medium.com/how-to-use-tensorboard-with-pytorch-e2b84aa55e67


----
## 数据集说明
目前包含两种数据集:
### 1. H5数据集:

- 形状: (样本数, 通道数=5, 序列长度=5000)
- 采样率: 25000Hz
- 采样时长: 0.2s/样本
- 通道说明: 2个振动信号(水平/垂直) + 3个三相电流信号
- 标签格式: 8位独热编码
  - NORMAL: [1,0,0,0,0,0,0,0] - 正常
  - SC: [0,1,0,0,0,0,0,0] - 匝间短路
  - HR: [0,0,1,0,0,0,0,0] - 高阻故障
  - RB: [0,0,0,1,0,0,0,0] - 转子故障
  - BF-I: [0,0,0,0,1,0,0,0] - 轴承内圈故障
  - BF-O: [0,0,0,0,0,1,0,0] - 轴承外圈故障
  - BF-R: [0,0,0,0,0,0,1,0] - 轴承滚动体故障
  - BF-C: [0,0,0,0,0,0,0,1] - 轴承保持架故障

### 2. KAT格式数据集:
- 文件格式: .mat文件
- 数据结构: 
  - data: 振动采样数据
  - label: 对应的标签信息（0: 健康, 1: 内圈故障, 2: 外圈故障）
- 采样频率: 64 kHz
- 样本数量: 每个工况文件包含300个样本（每种类型100个）
- 工况说明:
  - KATData0.mat: 转速1500rpm, 负载转矩0.7Nm, 径向力1000N
  - KATData1.mat: 转速900rpm, 负载转矩0.7Nm, 径向力1000N
  - KATData2.mat: 转速1500rpm, 负载转矩0.1Nm, 径向力1000N
  - KATData3.mat: 转速1500rpm, 负载转矩0.7Nm, 径向力400N

-----

## 支持模型：



------

## train.py：

- 训练的起点（）

-----

## 项目结构

### models 文件夹
- `cnn1d.py`: 定义了一维卷积神经网络 (CNN1D)，用于处理一维信号数据。包含多个卷积层和全连接层，适合于特征提取和分类任务。

- `dataset.py`: 定义了数据集加载类，包括 H5 和 KAT 数据集的读取和预处理。

### utils 文件夹
- `config_loader.py`: 提供配置文件的加载功能，支持从 JSON 文件中读取模型和训练参数。

----------------

#### Res-SA架构图：

![Res-SA架构图](https://github.com/hxqrrrr/motor_dl/blob/main/img/%E7%94%B3%E6%8A%A5%E4%B9%A6%20(1)-20.png)

---

#### TODO：

- [x] 迁移学习：[notion介绍](https://nutritious-cruiser-d7d.notion.site/14f42872c05480438277e6166262efa6?pvs=4)

- [x] 残差-自注意方案（主要）：把目前的cnn改成残差-自注意，类似transformer

-------

#### 有用的资料：

pytorh入门：https://github.com/hunkim/PyTorchZeroToAll

transformer入门：https://nutritious-cruiser-d7d.notion.site/learn-path-transformers-11d42872c054803596faee1f411525f6?pvs=4

hangingface ：https://huggingface.co/

