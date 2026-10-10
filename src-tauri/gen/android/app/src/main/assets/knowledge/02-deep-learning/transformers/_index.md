# Transformer

> 分类：**深度学习** → **Transformer**
> 路径：`resources/knowledge/02-deep-learning/transformers`

本目录沿着 Transformer 这条主线展开：从注意力机制的直觉与公式，到完整 encoder-decoder 架构的逐层拆解，再到 BERT/GPT/T5 三种预训练范式的分歧，最后落到 GPT-3 与 LLaMA 系列的具体工程取舍——位置编码、归一化、激活函数、注意力优化，这些也是今天所有主流开源 LLM 的共同模板。

## 文章目录

### 基础与架构

- [注意力机制详解](./attention-mechanism-explained.md) — Q/K/V 直觉、缩放点积注意力、$\sqrt{d_k}$ 推导、多头注意力、3-token 手工演算示例。
- [Transformer 架构详解](./transformer-architecture.md) — encoder-decoder 整体结构、位置编码（Sinusoidal/RoPE）、残差 + LayerNorm、Masked Self-Attention、端到端前向传播。

### 范式与演进

- [GPT 与 BERT 的演化](./gpt-and-bert-evolution.md) — MLM 与 CLM 的差异、encoder-only / decoder-only / encoder-decoder 三种范式、为什么 decoder-only 赢得主流。
- [GPT vs LLaMA：现代开源 LLM 的架构差异](./gpt-vs-llama-architecture.md) — 绝对位置编码 vs RoPE、LayerNorm vs RMSNorm、GeLU vs SwiGLU、MHA vs GQA、KV Cache 推理优化。

## 三方资料

- [Attention Is All You Need](./三方资料/Attention_Is_All_You_Need.pdf) — Vaswani et al. 2017 Transformer 原始论文，奠基之作。
- [Stable Diffusion 讲解稿](./三方资料/sd1.pptx) — Latent Diffusion 模型的图文讲解。
- [Transformer 综述讲义](./三方资料/transformer.docx) — Transformer 架构详细讲义（Word 文档）。
