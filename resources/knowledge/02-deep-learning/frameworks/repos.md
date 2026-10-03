# 开源仓库：深度学习框架

## PyTorch

- 仓库：https://github.com/pytorch/pytorch
- 简介：动态图 + Python-first 的深度学习框架，学术界事实标准。`torch.compile` 与 TorchInductor 正在缩小与 JAX/TF 的性能差距。
- 适用：研究、训练中小型模型、生产部署（TorchServe）。

## TensorFlow

- 仓库：https://github.com/tensorflow/tensorflow
- 简介：Google 维护的端到端 ML 平台；含 Keras 高层 API、TF Lite 端侧推理、TF.js 浏览器推理、TFX 流水线。
- 适用：大规模生产部署、移动端、浏览器端落地。

## JAX

- 仓库：https://github.com/google/jax
- 简介：Google 出品的 NumPy 兼容加速器，结合 `jit` / `grad` / `vmap` / `pmap` 实现函数式自动微分；Flax/Optax/Pax 构成其上层生态。
- 适用：TPU 训练、科学计算、需要细粒度控制梯度与并行的研究。
