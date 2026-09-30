# 深度学习训练技巧

> 分类：**深度学习** → **训练技巧**
> 路径：`resources/knowledge/02-deep-learning/training-techniques`

本目录汇集把神经网络"训出来、训得快、训得稳"所需要的全部工程经验：从优化器与学习率调度的算法原理，到正则化与归一化抑制过拟合的机制，再到混合精度与分布式训练支撑大模型规模化。这三块共同决定了一个模型能否在合理时间内收敛到可用精度。

## 文章目录

### 优化与调度

- [优化器与学习率调度：从 SGD 到 AdamW](./optimizers-and-scheduling.md) — SGD+Momentum、Nesterov、AdaGrad、RMSProp、Adam 偏差修正、AdamW 解耦权重衰减、Step/Exponential/Cosine/Warmup 调度、batch size 缩放规则、PyTorch 实战。

### 泛化与稳定

- [正则化与归一化](./regularization-and-normalization.md) — L1/L2、Dropout、Early Stopping、Mixup、Label Smoothing、BatchNorm/LayerNorm/InstanceNorm/GroupNorm 对比、Pre-LN vs Post-LN、PyTorch 残差块模板。

### 规模化训练

- [混合精度与分布式训练](./mixed-precision-and-distributed.md) — FP16/BF16 精度对比、`autocast` + `GradScaler`、AdamW 显存拆解、DP/DDP/FSDP/ZeRO-1/2/3/张量并行/流水线并行、梯度检查点、`torchrun` 启动模板。

## 收录范围

- 教材与讲义（Markdown / PDF）
- 论文（PDF）
- 讲稿与笔记（Markdown / Word / PPTX）

## 命名约定

- 每个子主题一个文件夹，文件夹命名用 kebab-case。
- 每个文件夹下放一个 `_index.md` 作为目录索引（阶段 2 由索引生成器读取）。
- 文件名建议：`YYYY-MM-DD-<title>.md` 或原文件名。
