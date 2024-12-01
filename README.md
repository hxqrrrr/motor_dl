# motor_dl：大创电机项目仓库

#### 项目主页：[链接](https://www.notion.so/13d42872c05480b88ec4ef624a233933?pvs=4)

#### 项目介绍：

定子匝间短路文件夹：电流故障数据集，两个一组

cnn.ipynb:能运行的cnn训练

可视化工具使用[tensorboard](https://www.tensorflow.org/tensorboard?hl=zh-cn)

TensorBoard教程：https://kuanhoong.medium.com/how-to-use-tensorboard-with-pytorch-e2b84aa55e67

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

#### **Res-SA.py**

可以直接使用的残差自注意架构

正确率在85~90，需要调参

----------------

#### Res-SA架构图：

![Res-SA架构图](https://gitee.com/xingzhixin5/picture/raw/master/申报书 (1)-20.png)

---

#### TODO：

- [ ] 迁移学习：[notion介绍](https://nutritious-cruiser-d7d.notion.site/14f42872c05480438277e6166262efa6?pvs=4)

- [x] 残差-自注意方案（主要）：把目前的cnn改成残差-自注意，类似transformer

-------

#### 有用的资料：

pytorh入门：https://github.com/hunkim/PyTorchZeroToAll

transformer入门：https://nutritious-cruiser-d7d.notion.site/learn-path-transformers-11d42872c054803596faee1f411525f6?pvs=4

hangingface ：https://huggingface.co/

