# motor_dl：大创电机项目仓库

#### 项目主页：[链接](https://www.notion.so/13d42872c05480b88ec4ef624a233933?pvs=4)

#### 项目介绍：

- 可视化工具使用[tensorboard](https://www.tensorflow.org/tensorboard?hl=zh-cn)

- TensorBoard教程：https://kuanhoong.medium.com/how-to-use-tensorboard-with-pytorch-e2b84aa55e67


----

#### 配置环境：
可以使用anaconda直接导入motor_env.yaml
```
#bash
conda create -n motor python=3.10.14 
conda activate motor
conda install pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

------

## models文件夹：

#### **Res-SA.py**

- 残差自注意架构类

- 适应原有的数据集

#### Res_SA_new.py

- 构建的残差自注意类
- 适应[新的数据集](https://github.com/hxqrrrr/motor_dl/blob/main/data.h5)
  - 形状：（数据数，通道数=5，步长=5000）
  - 详细见[手册说明](https://github.com/hxqrrrr/motor_dl/blob/main/%E8%AF%B4%E6%98%8E.txt)

----------------

#### Res-SA架构图：

![Res-SA架构图](https://github.com/hxqrrrr/motor_dl/blob/main/img/%E7%94%B3%E6%8A%A5%E4%B9%A6%20(1)-20.png)

---

#### TODO：

- [ ] 迁移学习：[notion介绍](https://nutritious-cruiser-d7d.notion.site/14f42872c05480438277e6166262efa6?pvs=4)

- [x] 残差-自注意方案（主要）：把目前的cnn改成残差-自注意，类似transformer

-------

#### 有用的资料：

pytorh入门：https://github.com/hunkim/PyTorchZeroToAll

transformer入门：https://nutritious-cruiser-d7d.notion.site/learn-path-transformers-11d42872c054803596faee1f411525f6?pvs=4

hangingface ：https://huggingface.co/

