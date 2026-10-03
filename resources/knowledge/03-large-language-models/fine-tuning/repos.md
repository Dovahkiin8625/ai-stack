# 开源仓库：LLM 微调

## PEFT (Hugging Face)

- 仓库：https://github.com/huggingface/peft
- 简介：LoRA / Prefix Tuning / IA³ / AdaLoRA 等参数高效微调方法的标准实现，配合 Transformers 无缝使用。
- 适用：单卡/小显存微调 7B–70B 模型。

## TRL

- 仓库：https://github.com/huggingface/trl
- 简介：Transformer Reinforcement Learning 库，提供 SFT、DPO、PPO、ORPO 等对齐训练流程，论文 DPO 的官方实现之一。
- 适用：SFT → 偏好对齐（DPO/PPO）一站式训练。

## bitsandbytes

- 仓库：https://github.com/TimDettmers/bitsandbytes
- 简介：8-bit / 4-bit 量化算子（LLM.int8()、NF4/QLoRA），显著降低微调与推理显存。
- 适用：QLoRA 微调、量化推理。

## LLaMA-Factory (hiyouga)

- 仓库：https://github.com/hiyouga/LLaMA-Factory
- 简介：覆盖 100+ 模型的统一微调框架，支持 LoRA / QLoRA / 全参 / 多模态 / DPO，开箱即用。
- 适用：快速复现/对比多种微调方法。
