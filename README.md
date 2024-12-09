# 电机故障诊断深度学习项目

## 项目主页：[链接](https://www.notion.so/13d42872c05480b88ec4ef624a233933?pvs=4)

## 项目介绍：
本项目使用深度学习方法对电机故障进行诊断分类。主要包含以下故障类型:

正常状态

匝间短路

高阻故障

转子故障

轴承故障(内圈、外圈、滚动体、保持架)

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

### 1.cnn1d：

### 2.resnet1d：

### 3.res-sa:

----

## 配置环境：

可以使用anaconda直接导入motor_env.yaml
```
#bash
conda create -n motor python=3.10.14 
conda activate motor
conda install pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

------

## train.py：

- 训练的起点（）

-----

## 项目结构

### models 文件夹
- `cnn1d.py`: 定义了一维卷积神经网络 (CNN1D)，用于处理一维信号数据。包含多个卷积层和全连接层，适合于特征提取和分类任务。
- `res_sa.py`: <!--上一阶段实现的残差自注意力网络 (Res-SA)-->
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

