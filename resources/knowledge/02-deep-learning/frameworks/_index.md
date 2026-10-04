# 深度学习框架

> 分类：**深度学习** → **框架**
> 路径：`resources/knowledge/02-deep-learning/frameworks`

本目录横向对比当前深度学习三大主流框架：PyTorch（研究界事实标准）、TensorFlow/Keras（工业部署成熟）、JAX（函数式 + 高性能编译）。理解它们的 API 风格、执行模型与适用场景，是工程师选型与跨团队协作的基础。

## 文章目录

- [PyTorch 基础：从张量到训练循环](./pytorch-fundamentals.md) — 动态计算图、Tensor/nn.Module/optim/DataLoader/autograd 五大抽象、训练循环骨架、设备管理、`state_dict` 持久化、`torch.compile` 编译模式、迁移学习、生态系统。
- [TensorFlow 与 JAX 对比](./tensorflow-and-jax.md) — TF 1.x → 2.x 演进、`tf.keras` Sequential/Functional、`tf.data` 流水线、SavedModel/TF Serving/Lite/JS、JAX 四件套（grad/jit/vmap/pmap）、Flax/Haiku、决策流程图、ONNX/HuggingFace 互通。

## 三方资料

- [PyTorch 论文](./三方资料/pytorch.pdf) — Paszke et al. 2019 介绍 PyTorch 的设计与实现，张量 + 动态计算图。
- [TensorFlow 论文](./三方资料/tensorflow.pdf) — Abadi et al. 2016 介绍 TensorFlow 大规模机器学习系统的设计。
