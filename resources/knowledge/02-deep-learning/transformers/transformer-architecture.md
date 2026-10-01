# Transformer 架构：从 Encoder-Decoder 到端到端走读

上一篇 [attention-mechanism-explained.md](attention-mechanism-explained.md) 解决了"如何对齐任意两个位置"的问题。本文把注意力作为核心组件，搭建出原论文 *Attention Is All You Need* (Vaswani et al., 2017) 的完整 encoder-decoder 结构，并系统介绍位置编码、Add&Norm 子层、前馈网络、Pre/Post-LN、训练 vs 推理的差异，最后用 "I love you → 我爱你" 走一遍端到端流程。下一篇 [gpt-and-bert-evolution.md](gpt-and-bert-evolution.md) 会基于这个"完整 Transformer"，说明预训练目标如何把家族分成 BERT、GPT、T5 三条路。

## 一、整体架构一览

```text
                        Transformer (Encoder-Decoder)
═══════════════════════════════════════════════════════════════

  Source         ┌──────────────── Encoder Stack ────────────────┐
  (I love you)   │  ┌─────── Encoder Block ──────┐                │
       │         │  │ Embed + PosEnc             │                │
       ▼         │  │   ▼                        │                │
   ┌───────┐     │  │ Multi-Head Self-Attn       │                │
   │  src  │ ──► │  │ Add & LayerNorm            │                │
   │embed+ │     │  │   ▼                        │                │
   │ pos   │     │  │ Feed-Forward (FFN)         │                │
   └───┬───┘     │  │ Add & LayerNorm            │                │
       │         │  └────────────────────────────┘ × N           │
       │         │              ▼                                │
       │         │       encoder output ─────────────┐            │
       │         └───────────────────────────────────┼────────────┘
       │                                             │
       │           ┌──────────────── Decoder Stack ── ┼────────────┐
       │           │  ┌─────── Decoder Block ──────┐ │            │
       │           │  │ Embed + PosEnc             │ │            │
       │           │  │   ▼                        │ │            │
       │           │  │ Masked Self-Attn            │ │            │
       │           │  │ Add & LayerNorm            │ │            │
       │           │  │   ▼                        │ │            │
       │           │  │ Cross-Attention (Q←dec, ───┘ │            │
   Target        │  │                  K,V←enc)      │            │
   (我爱你)       │  │ Add & LayerNorm            │             │
       │         │  │   ▼                        │             │
       ▼         │  │ Feed-Forward (FFN)         │             │
   ┌───────┐     │  │ Add & LayerNorm            │             │
   │ tgt   │ ──► │  └────────────────────────────┘ × N        │
   │embed+ │     │              ▼                            │
   │ pos   │     │         Linear (vocab)                    │
   └───┬───┘     │              ▼                            │
       │         │          Softmax → token                   │
       ▼         └────────────────────────────────────────────┘
   tokens out
```

左半部分是 encoder（编码源序列），右半部分是 decoder（生成目标序列），中间靠 cross-attention 把信息从 encoder 传给 decoder。

## 二、Embedding + Positional Encoding

模型本身对位置无感（attention 是集合操作），需要显式注入位置信息。

### 2.1 Token Embedding

```python
self.tok_emb = nn.Embedding(vocab_size, d_model)
x = self.tok_emb(tokens)             # (B, T, d_model)
# 乘以 √d_model 是原始论文的做法，理由：
# positional encoding 的量级 ~1，embedding 在小 vocab 时量级偏小，
# 乘以 √d_model 让两者在同一数量级。
x = x * (d_model ** 0.5)
```

### 2.2 Sinusoidal Positional Encoding

原论文使用一组不同频率的正弦/余弦：

$$
PE_{(pos,\, 2i)}   = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

其中 $pos$ 是 token 位置，$i$ 是维度索引。

**$10000^{2i/d}$ 的几何直觉**：

```text
维度 i 越低 → 周期越短（高频，区分相邻位置）
维度 i 越高 → 周期越长（低频，区分远距离位置）

例如 d=4:
  dim 0: 周期 ≈ 2π        → 每步都变，能编码相邻
  dim 2: 周期 ≈ 2π·100    → 大约 100 步变一次
```

最终位置编码矩阵每一行（一个位置）是一个 $d_{\text{model}}$ 维向量，相加到 token embedding 上：

```text
position 0:  [ 0.000, 1.000, 0.000, 1.000, ...]   ─┐
position 1:  [ 0.841, 0.540, 0.010, 0.999, ...]    ├─ 加到
position 2:  [ 0.909, -0.416, 0.020, 0.998, ...]   │  token emb
...                                              ─┘
```

**关键性质**：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性组合——这让模型能通过相对位置 $k$ 推断"距离"，是 attention 能学习相对位置的根因。

### 2.3 RoPE（Rotary Position Embedding）

LLaMA、Mistral 等现代 LLM 不再用 sinusoidal，而是**把 Q、K 向量按位置做复数旋转**：

$$
\text{RoPE}(x,\, pos) = x \cdot e^{i\, pos \cdot \theta_l}
$$

直观上：把 $q, k$ 看成复数，对位置 $pos$ 旋转一个角度 $pos \cdot \theta_l$，再求内积 $\langle \text{RoPE}(q_i, i),\ \text{RoPE}(k_j, j) \rangle$。由于**旋转的代数性质**，最终结果只依赖于 $(i - j)$——天然编码**相对位置**。具体实现：
- 对 $q, k$ 的相邻两维做 $\begin{pmatrix} \cos & -\sin \\ \sin & \cos \end{pmatrix}$ 旋转。
- 不同维度对（pair）用不同频率 $\theta_l = 10000^{-2l/d}$，与 sinusoidal 类似。
- **优势**：长度外推能力强——模型在 4K 训练后能较好处理 8K 甚至更长。

### 2.4 ALiBi

BLOOM 等模型使用更简单的方案：直接把**与距离成正比的偏置**加到 attention logits 上：

$$
\text{score}_{ij} = q_i^\top k_j - m \cdot |i - j|
$$

$m$ 是一个固定的"斜率"，无需学习。优势是极其简单、显存友好，缺点是建模能力上限低于 RoPE。

## 三、子层结构：Add & Norm

每个 sublayer 的标准形式（Post-LN）为：

$$
\text{output} = \text{LayerNorm}\bigl( x + \text{Sublayer}(x) \bigr)
$$

`Sublayer` 可以是 Multi-Head Attention 或 Feed-Forward。**残差连接 + LayerNorm** 让信息可绕过非线性直接传向输出，是训练深层网络的关键（详见 ResNet 系列文章）。

### 3.1 Pre-LN vs Post-LN

```text
   Post-LN（原论文）                Pre-LN（现代主流）
   ─────────────────               ─────────────────
   y = LN(x + Sublayer(x))        y = x + Sublayer(LN(x))
```

| 形式 | 训练稳定性 | 表现上限 | 代表模型 |
|---|---|---|---|
| Post-LN | 早期层梯度易爆，需 warmup | 同参数下略好 | 原 Transformer、BERT |
| Pre-LN | 稳定，可省略 warmup | 略差但差距小 | GPT-2/3、LLaMA、PaLM |

现代 LLM **几乎全部 Pre-LN**——深层（80+ 层）训练稳定性优先。代价是表示幅度逐层累积，需要在最后加一个 final LN。

## 四、Encoder Block

每个 encoder block 由两个子层堆叠而成：

```text
         x  (B, T, d_model)
          │
          ├─────────────── 残差
          ▼
   Multi-Head Self-Attention
          │
          ├─────────────── 残差
          ▼
   LayerNorm
          │
          ▼
       output
```

完整前向公式：

$$
x' = \text{LayerNorm}\bigl(x + \text{MHA}(x)\bigr)     \quad\text{(Pre-LN)}
$$

$$
y   = \text{LayerNorm}\bigl(x' + \text{FFN}(x')\bigr)
$$

Encoder 内的 self-attention 是**双向**的——每个位置可以看其他所有位置，没有 mask。堆叠 $N$ 层（原始论文 $N=6$），最后一层输出作为 encoder 的最终表示，送给 decoder 的 cross-attention。

## 五、Decoder Block

Decoder block 比 encoder 复杂：三个子层而不是两个。

```text
         y  (已生成 token)
          │
          ▼
   Masked Self-Attention        ← 看到当前与过去
          │
          ▼
   Cross-Attention              ← Q 来自这里, K/V 来自 encoder
          │  (Q=decoder, K,V=encoder_output)
          ▼
   Feed-Forward (FFN)
          │
          ▼
       output (再喂给下一层 / 线性投影到 vocab)
```

### 5.1 Masked Self-Attention

第 $t$ 步只能看到 $1 \dots t$，使用 causal mask：

$$
\text{CausalMask}_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
$$

$$
\text{masked-Attn} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
$$

矩阵下三角有效、上三角被屏蔽——这是 GPT 系列自回归生成的核心机制（[attention-mechanism-explained.md](attention-mechanism-explained.md) 第六节给了详细示例）。

### 5.2 Cross-Attention

Decoder 用自己的隐状态作为 $Q$，Encoder 的输出作为 $K, V$：

$$
Q = y\, W^Q,\quad K = \text{enc}\, W^K,\quad V = \text{enc}\, W^V
$$

$$
\text{CrossAttn}(y,\, \text{enc}) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

直觉：**$Q$ 是"我（解码器）现在想知道什么"，$K, V$ 是"源文里有什么"**——cross-attention 让解码器每一步都重新检索源文的相关片段。翻译模型在这一步真正"读懂"源文。

### 5.3 三个注意力的对比

| 子层 | $Q$ | $K, V$ | mask |
|---|---|---|---|
| Encoder 自注意力 | encoder 输入 | encoder 输入 | 无 |
| Decoder masked 自注意力 | decoder 输入 | decoder 输入 | causal |
| Decoder cross 注意力 | decoder 隐状态 | encoder 输出 | 无 |

## 六、Feed-Forward Network（FFN）

每个 attention 子层后接一个**位置独立**的全连接网络：

$$
\text{FFN}(x) = \text{Activation}(x W_1 + b_1)\, W_2 + b_2
$$

原始论文用 ReLU、内层维度 $d_{ff} = 4 d_{\text{model}}$（"expand 4× → contract"）：

```text
   x  ──►  Linear(d_model, 4·d_model)  ──►  ReLU  ──►  Linear(4·d_model, d_model)
   (B, T, d_model)                          (B, T, 4·d_model)         (B, T, d_model)
```

直觉："**FFN 是模型存知识的地方**"——每个 token 的表示经过两层线性，相当于用一个键值表检索"对应的事实"。可以用 key-value 记忆的角度解释：

$$
\text{FFN}(x) \approx \sum_{f=1}^{d_{ff}} \mathbb{1}[x \approx k_f] \cdot v_f
$$

$d_{ff}$ 越大，"记忆槽"越多。这就是为什么 LLM 容量与 FFN 维度高度相关。现代 LLM 常用 **SwiGLU**（带门控），详见 [gpt-vs-llama-architecture.md](gpt-vs-llama-architecture.md)。

## 七、Layer Normalization

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中 $\mu, \sigma^2$ 是**沿特征维度**（$d_{\text{model}}$）计算的均值与方差，每个 token 独立归一化。

**为什么不用 BatchNorm**：

- BN 在 batch 维度算统计量，对变长序列不友好（不同 batch 内同一位置的 token 数量差异大）。
- BN 在推理时要用 running stats，而 LLM 推理 batch size 经常为 1。
- LN 与 attention 的"逐位置"哲学匹配：每个位置有自己的 normalization。

## 八、堆叠与输出头

把 $N$ 个 block 叠起来（$N$ 在不同模型中差异巨大）：

| 模型 | $N$ | $d_{\text{model}}$ | Heads |
|---|---|---|---|
| Transformer-base (2017) | 6 | 512 | 8 |
| BERT-base | 12 | 768 | 12 |
| BERT-large | 24 | 1024 | 16 |
| GPT-3 175B | 96 | 12288 | 96 |
| LLaMA-2 70B | 80 | 8192 | 64 |
| GPT-4 (估计) | 120+ | ~12800 | ~100 |

最终 decoder 输出经过一个线性投影 + softmax 得到词表上的概率分布：

$$
P(\text{next token}) = \text{softmax}(W_{\text{out}} \cdot h_T)
$$

其中 $W_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times |V|}$，通常与输入 embedding **权重共享**（"tied embeddings"），节省参数。

## 九、端到端示例：I love you → 我爱你

为了把所有部件串起来，考虑一次简化翻译：

```text
源文: "I love you"      (T_src = 3)
目标: "我爱你"           (T_tgt = 3, 假设已知)
```

**训练阶段**（teacher forcing）：

```text
1) 源文 embedding + PE → (3, d_model)  →  送入 encoder × N 层
                                          →  encoder_output: (3, d_model)

2) 目标 "BOS 我 爱 你" (右移一位) → embedding + PE → (4, d_model)
   ↓
   Decoder masked self-attention: (4, d_model) — 因果 mask
   ↓
   Decoder cross-attention: Q ← 上一步, K,V ← encoder_output
   ↓
   Decoder FFN × N 层
   ↓
   Linear(vocab) + softmax → (4, |V|) 概率分布
   ↓
   与真实目标 "我 爱 你 EOS" 算 cross-entropy loss
```

**推理阶段**（自回归生成）：

```text
step 1: 输入 "BOS"         → 模型预测 "我"
step 2: 输入 "BOS 我"      → 模型预测 "爱"
step 3: 输入 "BOS 我 爱"   → 模型预测 "你"
step 4: 输入 "BOS 我 爱 你" → 模型预测 "EOS" → 停止
```

训练时所有 token 一次性算完（teacher forcing），推理时只能串行——这就是为什么需要 **KV cache**：把之前步的 $K, V$ 缓存起来，避免每步重算。详见 [gpt-vs-llama-architecture.md](gpt-vs-llama-architecture.md)。

## 十、训练 vs 推理：两套不同的执行模式

| 维度 | 训练 | 推理 |
|---|---|---|
| 输入 | 完整序列（teacher forcing） | 逐步生成（自回归） |
| 注意力 | 全序列一次算完 | 只能看到已生成 + 缓存 |
| Mask | 静态 mask | KV cache + 增量 mask |
| 优化 | 反向传播 + AdamW | 无梯度，前向采样 |
| 并行 | 完全并行（GPU 友好） | 串行（需要 KV cache 优化） |

训练用 teacher forcing 让所有位置同时算 loss，大幅加速；推理时 KV cache 是节省算力的关键工程——也是 GQA、MQA 等技术大放异彩的地方。

## 小结

Transformer 通过"embedding + 位置编码 + 多头注意力 + FFN + Add&Norm"的反复堆叠，配合 encoder-decoder 双塔结构，把任意序列到任意序列的映射用一个统一的、可并行的、可学习的架构解决。位置编码（sinusoidal / RoPE / ALiBi）解决"顺序感知"，Pre-LN + RMSNorm 解决"深层训练稳定"，FFN 提供"知识存储"，cross-attention 把 encoder 信息注入 decoder。训练时 teacher forcing 让所有 token 并行算 loss，推理时 KV cache 让自回归生成可行。这个"完整 Transformer"是后续所有 NLP 大模型的母版——下一篇 [gpt-and-bert-evolution.md](gpt-and-bert-evolution.md) 将基于它讨论预训练目标如何把家族分成 BERT、GPT、T5 三条路线，以及为什么 decoder-only 最终成为主流。