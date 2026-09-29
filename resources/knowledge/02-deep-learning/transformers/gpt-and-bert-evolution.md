# GPT 与 BERT 的演化

自 2017 年 Transformer 诞生起，整个 NLP 领域围绕一个问题展开：**预训练时应该让模型做什么任务？** 这个问题催生了三大范式：仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5/BART）。本文梳理它们的设计差异、训练目标，以及为何今天的大模型几乎都走上了"decoder-only"的路。

## 一、BERT：双向编码器

Google 2018 年发布的 BERT 采用**仅编码器**结构。它的核心创新是**掩码语言模型（Masked Language Modeling, MLM）**：随机遮住输入中 15% 的 token，让模型根据左右两侧上下文预测被遮住的词。

辅助任务还包括**下一句预测（NSP）**：判断两段文本是否相邻，强化句间关系建模。

由于 encoder 天然允许每个位置看到完整上下文，BERT 特别适合理解类任务：文本分类、问答、命名实体识别。Google 后续也推出了 RoBERTa、ALBERT、DeBERTa 等改进。

## 二、GPT：自回归解码器

OpenAI 的 GPT 系列采用**仅解码器**结构。预训练目标是经典的**因果语言模型（Causal Language Modeling, CLM）**：给定前文 $x_1, \dots, x_{t-1}$，预测下一个 token $x_t$：

```math
\mathcal{L}_{\text{CLM}} = -\sum_t \log P_\theta(x_t \mid x_{<t})
```

这种目标天然契合生成任务——只要逐 token 自回归采样即可。GPT-2 / GPT-3 通过把规模放大到 175B 参数，展示了**少样本学习**的涌现能力。

## 三、T5 / BART：编码器-解码器

Google 的 T5 与 Facebook 的 BART 采用完整的 encoder-decoder。训练目标多样：

- 文本 span corruption（类似 MLM 但生成式）
- 翻译、摘要等监督任务
- 去噪自编码

它们擅长"输入 → 输出"的转换类任务，例如翻译、摘要、问答生成。

## 四、为什么 decoder-only 赢了

到 2020 年后，社区几乎一致转向 decoder-only。原因有几条：

1. **统一性**：CLM 一个目标既能做理解又能做生成，而 MLM 只能填空。
2. **规模效率**：decoder-only 的 in-context learning 随规模呈涌现式增长。
3. **推理效率**：decoder 自回归生成可直接复用 KV cache，无需额外处理 encoder 输出。
4. **生态惯性**：开源社区（Llama、Mistral、Qwen）几乎全是 decoder-only，下游工具链（vLLM、TGI）也围绕它优化。

但 encoder-only 也有自己的位置：BERT 类小模型在分类、检索 embedding 上至今仍是性价比最高的选择。

## 五、范式对比表

| 范式 | 代表模型 | 预训练目标 | 擅长任务 |
| --- | --- | --- | --- |
| Encoder-only | BERT, RoBERTa | MLM + NSP | 分类、检索、NER |
| Decoder-only | GPT, LLaMA | CLM | 生成、对话、in-context learning |
| Encoder-Decoder | T5, BART | Span corruption | 翻译、摘要、seq2seq |

## 小结

MLM 与 CLM 的差异看似微小，却塑造了两种截然不同的模型行为。今天的大模型主流选择是 decoder-only + CLM，但理解 BERT 这条线，仍然是理解"模型如何表征语言"的关键基础。
