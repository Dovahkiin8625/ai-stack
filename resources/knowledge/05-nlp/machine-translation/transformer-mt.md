# Transformer 机器翻译：从原论文到多语言大模型

2017 年 Vaswani et al. 发表 "Attention Is All You Need"，用纯 attention 替换 RNN/CNN encoder-decoder，把 WMT'14 EN-DE BLEU 推到 28.4（base）/ 28.5（big），同时训练速度比 GNMT 快 5-10 倍。这一架构迅速成为 NMT 的事实标准，并衍生出 mBART、mT5、NLLB-200 等多语言大模型。本文深入剖析 Transformer for MT 的架构细节，并讨论它在多语言、低资源、长文档等场景的扩展。

## 一、回顾：Transformer 原论文

### 1.1 整体架构

Transformer 沿用了 Seq2Seq 的 encoder-decoder 结构，但把 LSTM 全部替换为 **Multi-Head Self-Attention** + **FFN**：

```
Encoder (×N): [Self-Attention → Add&Norm → FFN → Add&Norm]
Decoder (×N): [Masked Self-Attn → Cross-Attn → Add&Norm → FFN → Add&Norm]
```

**关键设计**：

1. **Encoder Self-Attention**：每个位置看所有位置（双向）。
2. **Decoder Masked Self-Attention**：每个位置只看自己 + 之前的位置（causal mask）。
3. **Decoder Cross-Attention**：query 来自 decoder，key/value 来自 encoder——让 decoder "关注" 源序列。
4. **残差 + LayerNorm**：每个子层都是 `LayerNorm(x + Sublayer(x))`（Post-Norm），但 LLaMA 等现代 LLM 用 Pre-Norm。
5. **位置编码**：正弦位置编码或可学习位置编码。

### 1.2 Multi-Head Attention

把 $d_{\text{model}}$ 维拆成 $h$ 头，每头独立做 attention：

$$
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O
$$
$$
\text{head}_i = \text{Attention}(Q W_Q^{(i)}, K W_K^{(i)}, V W_V^{(i)})
$$
$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

直觉：不同头可以学不同关系——一个头关注句法依赖，一个头关注共指，一个头关注长距离。

### 1.3 缩放因子 $\sqrt{d_k}$

当 $d_k$ 大时，$Q K^\top$ 的方差是 $d_k$，softmax 容易进入饱和区（梯度接近 0）。除以 $\sqrt{d_k}$ 把方差归一。

### 1.4 FFN

逐位置两层全连接 + ReLU：

$$
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$

典型 $W_1$ 把维度从 $d$ 升到 $4d$，$W_2$ 再降回 $d$。

## 二、Transformer for MT 的关键改进

### 2.1 Pre-Norm vs Post-Norm

原论文用 Post-Norm：`LayerNorm(x + Sublayer(x))`。深层训练不稳定（warmup 阶段需要小心）。

现代 LLM 用 **Pre-Norm**：`x + Sublayer(LayerNorm(x))`——训练更稳定，warmup 更短。代价是最终激活值的尺度略大，需要 final LayerNorm。

### 2.2 学习率与 warmup

Transformer 训练对学习率极敏感。原论文使用 **Noam warmup**：

$$
\text{lr} = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})
$$

前 4000 步线性 warmup，之后按 step$^{-0.5}$ 衰减。

### 2.3 Label Smoothing

对真实标签做软化：

$$
y' = (1 - \epsilon) \cdot \mathbf{1}_{y} + \epsilon / V
$$

$\epsilon = 0.1$ 显著提升 BLEU，但 perplexity 变差——label smoothing 抑制过自信，让模型学得更"温和"。

### 2.4 BPE 子词切分

Transformer 直接用 BPE / WordPiece 子词切分（详见 [[text-representation/subword-tokenization]]），把词表控制在 32K-64K，覆盖几乎所有文本，避免 OOV。

### 2.5 Back-Translation（Sennrich et al., 2016）

用目标语言单语语料反向翻译扩充训练集：

1. 训练一个 EN→DE 模型。
2. 用 EN→DE 把目标语言（DE）单语语料翻译成伪源（EN）。
3. 把伪源 + 真目标语料加入训练集。

Back-Translation 在低资源 MT 上能把 BLEU 提升 3-5 个点。

## 三、大规模多语言翻译

### 3.1 mBART（Liu et al., 2020）

mBART 是 BART 的多语言扩展，在 25 种语言的 CC-25 单语语料上做 **denoising autoencoding** 预训练：

预训练任务：

- **Token masking**：随机 mask 35% 的 token。
- **Token deletion**：随机删 20% 的 token。
- **Text infilling**：用单个 mask 替换一段连续 span。
- **Sentence permutation**：打乱句子顺序。
- **Document rotation**：选一个 token 作为文档开头。

mBART 通过预训练获得了 25 种语言的**通用语义表示**，在低资源 MT 上微调就能达到当时 SOTA。

### 3.2 mT5（Xue et al., 2021）

Google 的 mT5 是 T5 的多语言版本：

- **100 种语言** mC4 语料。
- **统一 text-to-text 框架**：所有任务（分类、QA、翻译、摘要）都视为"输入文本 → 输出文本"。
- **13B 参数版本**：在 xTREME、xGLUE 等多语言基准上 SOTA。

mT5 的翻译 prompt：

```
Translate to French: <source> → <target>
```

零样本就能翻译 100+ 语言对。

### 3.3 NLLB-200（NLLB Team, 2022）

Meta 的"No Language Left Behind"是专门为低资源语言设计的大模型：

- **200 种语言**（包括许多低资源语言如 Kimbundu、Ligurian）。
- **Conditional Compute + Sparse MoE**：用 MoE 提高低资源语言的质量。
- **FLORES-200 评测**：覆盖 200 种语言的翻译评测。

NLLB-200 在 Kpelle ↔ English 等极端低资源对上的 BLEU 超过人类基线 10+ 个点，是 MT 包容性研究的里程碑。

### 3.4 多语言共享词表的关键

多语言模型的难点是**如何平衡词表分配**：

- 高资源语言（中文、英文）需要更多 token。
- 低资源语言（斯瓦希里语）可能只有几千个 token。

解决方案：

- **SentencePiece + sampling alpha**：`--input_sentence_size=10000000 --shuffle_input_sentence=true --train_extremely_large_corpus=true`。
- **Byte-level BPE**：从字节出发，自动按语言频率分配 token。
- **Shared embeddings**：所有语言共享同一词表 + embedding 矩阵（mBERT 风格）。

## 四、长文档翻译

### 4.1 问题

标准 Transformer 的最大序列长度是 512（受位置编码限制）。处理长文档（法律合同、书籍翻译）必须扩展：

- **Transformer-XL**（Dai et al., 2019）：跨段落的循环机制，但需要重训。
- **Relative Position Encoding**（Shaw et al., 2018）：用相对位置替代绝对位置，可外推到更长。
- **Longformer**（Beltagy et al., 2020）：局部窗口 + 全局 token attention，复杂度 $O(n)$。
- **BigBird**（Zaheer et al., 2020）：稀疏 attention（random + window + global）。
- **Streaming**：实时翻译流式输入，滑动窗口处理。

### 4.2 上下文感知翻译

文档级翻译需要"上下文感知"——同一个词在不同上下文应有不同翻译。代表工作：

- **Document-Level NMT**（Maruf et al., 2019）：用 hierarchical attention 编码跨句上下文。
- **Context-Aware Decoding**（Liu et al., 2020）：decoder 同时看当前句和上下文，目标函数的差值（context-aware delta）作为解码引导。

## 五、低资源机器翻译

### 5.1 Pivot Translation（枢轴翻译）

低资源语言 A ↔ B 直接翻译数据稀缺，可以借助高资源语言 C：

$$
\text{A} \to \text{C} \to \text{B}
$$

通过 C 作为"枢轴"生成 A → C 和 C → B 训练数据。

### 5.2 Zero-Shot Translation

多语言模型如 mBART、mT5 训练时只用 EN→X 与 X→EN 数据，推理时直接生成 Y→Z（从未见过的方向）。零样本性能通常弱于有监督，但比 pivot 翻译简单。

### 5.3 Few-Shot Translation

GPT-4 / Claude 等大模型用 in-context learning：

```python
prompt = """Translate English to Swahili.

English: How are you?
Swahili: Habari yako?
English: Good morning.
Swahili: Habari za asubuhi?
English: I love programming.
Swahili: """
```

GPT-4 在 5-shot Swahili 翻译上的 BLEU 接近监督基线。Claude 3 Opus 在 100+ 低资源语言上接近人类水平。

### 5.4 主动学习与数据增强

- **Active Learning**：模型选择最有价值的句子请求人工翻译。
- **Synthetic Data**：用大模型生成合成平行语料。
- **Data Augmentation**：word replacement、back-translation、forward-translation。

## 六、解码策略

### 6.1 Beam Search

Transformer MT 的标准解码策略：

```
beam = [(<BOS>, log_prob=0)]
for step in range(max_len):
    candidates = []
    for prefix, score in beam:
        if prefix ends with <EOS>:
            candidates.append((prefix, score))
            continue
        logits = model(prefix)
        top_k = topk(logits, k=beam_size)
        for token, lp in top_k:
            candidates.append((prefix + [token], score + lp))
    beam = topk(candidates, k=beam_size, key=lambda x: x[1] / len(x[0]) ** alpha)
```

长度惩罚 $\alpha$ 通常 0.6-1.0。

### 6.2 Length-Normalized Beam Search

简单按长度归一化对短句不利：

$$
\text{score} = \frac{\log P(\mathbf{y} \mid \mathbf{x})}{(5 + |\mathbf{y}|)^\alpha / (5 + 1)^\alpha}
$$

Wu et al.（2016）的 Google 公式。

### 6.3 Diverse Beam Search

普通 Beam Search 容易退化成"重复候选"。Diverse Beam Search 强制不同 beam 关注不同子空间：

```
beam_groups = split beam into G groups
for each group g:
    score = lm_score - diversity_penalty * similarity(other_groups)
    pick top beam_size/G
```

多样性惩罚项防止各组生成相同候选。

### 6.4 Sampling vs Beam Search

MT 领域 Beam Search 仍占主导，但 Sampling-based 解码（top-k、top-p）在创意文本生成中常用。MT 主要关心"准确性"，Beam Search 的"全局最优"假设仍然有效。

## 七、PyTorch 实现：Transformer MT

```python
import torch
import torch.nn as nn
from torch.nn import Transformer


class TransformerMT(nn.Module):
    """PyTorch nn.Transformer 封装：源/目标语言翻译。"""

    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 512,
                 nhead: int = 8, num_layers: int = 6, dim_ff: int = 2048,
                 dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def _add_pos(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(positions))

    def encode(self, src, src_pad_mask):
        # src: (B, T_src), src_pad_mask: (B, T_src) True=pad
        src_emb = self._add_pos(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        return memory

    def decode(self, tgt, memory, src_pad_mask, tgt_pad_mask, tgt_mask):
        tgt_emb = self._add_pos(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.out_proj(out)

    @staticmethod
    def causal_mask(T, device):
        return torch.triu(torch.full((T, T), float("-inf")), diagonal=1).to(device)

    def forward(self, src, tgt_in, src_pad_mask, tgt_pad_mask):
        memory = self.encode(src, src_pad_mask)
        tgt_mask = self.causal_mask(tgt_in.size(1), tgt_in.device)
        return self.decode(tgt_in, memory, src_pad_mask, tgt_pad_mask, tgt_mask)
```

训练循环与 Seq2Seq 类似，但 PyTorch 的 `nn.Transformer` 已经封装好所有 mask。

## 八、工程经验

### 8.1 训练稳定性

- **学习率 warmup**：前 4000 步线性 warmup，避免初期梯度爆炸。
- **梯度裁剪**：`clip_grad_norm_(model.parameters(), 1.0)`。
- **Adam / AdamW**：$\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$。
- **Label smoothing**：$\epsilon=0.1$。

### 8.2 推理加速

- **KV-cache**：推理时缓存已计算的 K、V，避免重复编码。
- **量化**：FP16 / INT8 量化，吞吐量提升 2-3 倍。
- **Batched beam search**：同 batch 不同 beam 并行解码。
- **Speculative decoding**：用小模型预生成 token，大模型验证。

### 8.3 多语言部署

- **模型并行**：单模型 100+ 语言，可用 tensor parallelism 拆分到多卡。
- **任务路由**：根据输入语言选择子模型（或不同 head）。
- **Adapter**：每个语言对独立 Adapter，共享主模型。

## 九、当前 SOTA 与前沿

### 9.1 WMT 评测（2024-2025）

| 模型 | EN-DE | EN-FR | ZH-EN |
| --- | --- | --- | --- |
| Transformer-Big (2017) | 28.4 | 41.0 | - |
| mBART (2020) | - | - | 24.4 |
| NLLB-200 (2022) | 30+ | 45+ | 28+ |
| GPT-4 (2024) | 33+ | 48+ | 30+ |
| 人类专家 | 35+ | 50+ | 32+ |

LLM 已经接近甚至超越专家水平。

### 9.2 趋势

1. **LLM 主导**：GPT-4 / Claude / Gemini 用 in-context learning 翻译，质量接近专用模型。
2. **多模态翻译**：图像 + 文本 + 语音的联合翻译。
3. **同声传译**：实时流式翻译 + 上下文等待。
4. **个性化翻译**：根据用户偏好调整风格（正式 / 口语 / 学术）。

## 十、Transformer for MT 的精神遗产

Transformer MT 不仅改写了翻译，更奠定了**预训练大模型时代**的架构基础：

- **GPT 系列** = Transformer decoder + 自回归预训练。
- **BERT** = Transformer encoder + MLM 预训练。
- **T5** = Transformer encoder-decoder + span corruption。
- **LLaMA / Qwen** = 改进的 Transformer decoder。

可以说，Transformer MT 是**整个生成式 AI 革命的起点**。

## 小结

| 模型 | 时间 | 创新 | WMT EN-DE BLEU |
| --- | --- | --- | --- |
| Moses | 2007 | 短语统计 | 22+ |
| Seq2Seq | 2014 | LSTM + attention | 23+ |
| GNMT | 2016 | 深层 LSTM | 24+ |
| Transformer | 2017 | Self-attention | 28.4 |
| mBART | 2020 | 多语言预训练 | 30+ |
| NLLB-200 | 2022 | 200 语言 SOTA | 30+ |
| GPT-4 | 2024 | LLM zero/few-shot | 33+ |

Transformer 机器翻译的演进折射出 NLP 整体的进步：从手工特征 → 端到端学习 → 预训练 + 微调 → 通用大模型。这一架构不仅是机器翻译的事实标准，更奠定了 LLaMA、GPT 等大模型的基石。下一篇我们将进入**文本生成**，解码策略与生成质量评估。
