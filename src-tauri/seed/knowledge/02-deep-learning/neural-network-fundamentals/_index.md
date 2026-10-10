# 神经网络基础

> 分类：**深度学习** → **神经网络基础**
> 路径：`resources/knowledge/02-deep-learning/neural-network-fundamentals`

本目录梳理深度学习最底层的三个支柱：从生物神经元启发的感知机，到把多个神经元堆叠成多层感知机（MLP），再到训练神经网络的反向传播算法与梯度机制。最后系统比较主流激活函数、损失函数与它们在不同任务中的搭配。

## 文章目录

- [感知机与多层感知机（MLP）](./perceptron-and-mlp.md) — 生物神经元模型、单层感知机的局限、XOR 问题、MLP 架构、通用逼近定理、PyTorch 实现。
- [反向传播算法](./backpropagation.md) — 计算图、链式法则、BPTT 推导、四大 BP 方程、矩阵化、向量化实现、梯度检验、PyTorch autograd。
- [激活函数与损失函数](./activation-functions-and-loss.md) — Sigmoid/Tanh/ReLU 族、softmax、多分类交叉熵、MSE/CE 梯度量级分析、Focal Loss。

## 三方资料

- [BatchNorm 论文](./三方资料/batchnorm.pdf) — Ioffe & Szegedy 2015 通过 mini-batch 归一化加速深层网络训练。
- [Dropout 论文](./三方资料/dropout.pdf) — Srivastava et al. 2014 通过随机失活神经元防止过拟合。
