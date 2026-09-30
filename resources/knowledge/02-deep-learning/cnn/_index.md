# 卷积神经网络（CNN）

> 分类：**深度学习** → **卷积神经网络**
> 路径：`resources/knowledge/02-deep-learning/cnn`

本目录覆盖卷积神经网络从基础到前沿的完整脉络：先讲清楚卷积、池化、感受野这些核心算子，再沿着 LeNet → AlexNet → VGG → GoogLeNet → ResNet 这条主线看经典架构，最后讨论 ConvNeXt、EfficientNet 等"Transformer 时代"之后 CNN 如何演化以及与 ViT 的取舍。

## 文章目录

### 基础与经典

- [CNN 基础：从卷积操作到 PyTorch 实现](./cnn-fundamentals.md) — 为什么 MLP 处理图像不行、卷积/池化/感受野、通道、特征层级、PyTorch 完整 CIFAR-10 示例、MLP vs CNN 参数对比。
- [经典 CNN 架构：从 LeNet 到 ResNet](./classic-cnn-architectures.md) — LeNet-5、AlexNet、VGG、GoogLeNet/Inception、ResNet 残差连接、迁移学习 PyTorch 实践。

### 现代趋势与选型

- [现代 CNN：ConvNeXt、EfficientNet 与移动端](./modern-cnn-trends.md) — ViT 冲击、深度可分离卷积、MobileNet v1/v2/v3、EfficientNet 复合缩放、ConvNeXt 现代 ResNet、骨干网络选型指南。

## 收录范围

- 教材与讲义（Markdown / PDF）
- 论文（PDF）
- 讲稿与笔记（Markdown / Word / PPTX）

## 命名约定

- 每个子主题一个文件夹，文件夹命名用 kebab-case。
- 每个文件夹下放一个 `_index.md` 作为目录索引（阶段 2 由索引生成器读取）。
- 文件名建议：`YYYY-MM-DD-<title>.md` 或原文件名。
