# 神经网络基础

> 分类：**深度学习** → **神经网络基础**
> 路径：`resources/knowledge/02-deep-learning/neural-network-fundamentals`

本目录梳理深度学习最底层的三个支柱：从生物神经元启发的感知机，到把多个神经元堆叠成多层感知机（MLP），再到训练神经网络的反向传播算法与梯度机制。最后系统比较主流激活函数、损失函数与它们在不同任务中的搭配。

## 文章目录

- [感知机与多层感知机（MLP）](./perceptron-and-mlp.md) — 生物神经元模型、单层感知机的局限、XOR 问题、MLP 架构、通用逼近定理、PyTorch 实现。
- [反向传播算法](./backpropagation.md) — 计算图、链式法则、BPTT 推导、四大 BP 方程、矩阵化、向量化实现、梯度检验、PyTorch autograd。
- [激活函数与损失函数](./activation-functions-and-loss.md) — Sigmoid/Tanh/ReLU 族、softmax、多分类交叉熵、MSE/CE 梯度量级分析、Focal Loss。

## 收录范围

- 教材与讲义（Markdown / PDF）
- 论文（PDF）
- 讲稿与笔记（Markdown / Word / PPTX）

## 命名约定

- 每个子主题一个文件夹，文件夹命名用 kebab-case。
- 每个文件夹下放一个 `_index.md` 作为目录索引（阶段 2 由索引生成器读取）。
- 文件名建议：`YYYY-MM-DD-<title>.md` 或原文件名。
