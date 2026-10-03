# 开源仓库：训练优化技术

## NVIDIA Apex

- 仓库：https://github.com/NVIDIA/apex
- 简介：Apex 提供 FP16 / BF16 混合精度训练、分布式同步 BN 与 fused 优化器，是论文 *Mixed Precision Training* 的官方实现参考。
- 适用：需要手动控制 loss scaling 的混合精度训练。

## DeepSpeed (Microsoft)

- 仓库：https://github.com/microsoft/DeepSpeed
- 简介：实现 ZeRO（1/2/3）、offload、3D 并行（DP+TP+PP）、Mixture-of-Experts 训练，是大模型训练的事实标准之一。
- 适用：10B+ 级别模型训练、显存优化、流水线并行。

## Optuna

- 仓库：https://github.com/optuna/optuna
- 简介：基于贝叶斯/进化/剪枝策略的超参搜索框架，配合 PyTorch Lightning / Keras 等 callback 自动剪枝。
- 适用：学习率、batch size、正则化系数搜索。
