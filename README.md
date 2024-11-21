# motor_dl：大创电机项目仓库

#### 项目主页：

https://www.notion.so/13d42872c05480b88ec4ef624a233933?pvs=4

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

![](https://github.com/hxqrrrr/motor_dl/blob/main/img/~%244vsoaee.png)

---

#### TODO：

- [ ] 残差-自注意方案（主要）：把目前的cnn改成残差-自注意，类似transformer

- [ ] 音频信号方案（废弃）：https://nutritious-cruiser-d7d.notion.site/CNN-11542872c0548060983fc6b582e3b89a?pvs=4



-------

#### 有用的资料：

pytorh入门：https://github.com/hunkim/PyTorchZeroToAll

transformer入门：https://nutritious-cruiser-d7d.notion.site/learn-path-transformers-11d42872c054803596faee1f411525f6?pvs=4

hangingface ：https://huggingface.co/

