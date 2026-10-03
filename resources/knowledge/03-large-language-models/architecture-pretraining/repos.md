# 开源仓库：LLM 架构与预训练

## Hugging Face Transformers

- 仓库：https://github.com/huggingface/transformers
- 简介：当前最主流的预训练模型实现库，覆盖 BERT/GPT/T5/LLaMA/Qwen/DeepSeek 等数百种架构，配套 `Trainer` / `pipeline` / `generate` 等高层 API。
- 适用：模型微调、推理、跨架构实验。

## Megatron-LM (NVIDIA)

- 仓库：https://github.com/NVIDIA/Megatron-LM
- 简介：实现张量并行 + 流水线并行 + 序列并行的 GPT 类模型大规模训练框架，是 LLaMA / BLOOM 等模型的训练底座之一。
- 适用：10B+ 模型从零预训练、学习分布式训练工程。

## nanoGPT (Andrej Karpathy)

- 仓库：https://github.com/karpathy/nanoGPT
- 简介：300 行代码可训练 GPT-2 级别模型的极简实现，是理解 GPT 训练循环最干净的教材。
- 适用：教学、动手实现 GPT 训练流程。
