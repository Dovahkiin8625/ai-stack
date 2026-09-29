# Transformer 架构详解

上一篇文章介绍了注意力机制，本文把它"装进"完整的 Transformer 中。我们会逐层拆解 encoder、decoder、关键子模块（位置编码、层归一化、残差连接、前馈网络），最后走一遍端到端的前向传播。

## 一、整体结构

```
                ┌──────────────────────┐
                │       Encoder        │
                │  ┌────────────────┐  │
input tokens → │ Embedding + Pos │  │
                │  └─────┬────────┘  │
                │        ▼            │
                │  ┌─────────────┐    │
                │  │ Multi-Head  │    │ × N
                │  │ Self-Attn   │    │
                │  └─────┬───────┘    │
                │  ┌─────▼───────┐    │
                │  │ Feed-Forward│    │
                │  └─────────────┘    │
                └─────────┬───────────┘
                          │  context
                          ▼
                ┌──────────────────────┐
                │       Decoder        │
                │  Masked Self-Attn    │
                │  Cross-Attn (Q←K,V)  │
                │  Feed-Forward        │
                └──────────────────────┘
                          ▼
                       Output
```

编码器把输入序列压成富含上下文的表示；解码器一边自回归生成，一边通过 cross-attention "回头看"编码器输出。

## 二、位置编码

注意力是置换不变的——它天然忽略 token 顺序。我们必须显式注入位置信息。

- **正弦位置编码（Sinusoidal）**：用不同频率的正弦/余弦函数生成位置向量，公式：

```math
PE_{(pos, 2i)}   = \sin(pos / 10000^{2i/d})
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
```

- **RoPE（旋转位置编码）**：现代 LLM 常用，通过把 Q/K 视作复数并按位置旋转，把相对位置信息编码进点积。

## 三、子模块组合

每个 encoder/decoder block 都包含以下关键子层（顺序以原始论文为参考）：

1. **多头自注意力**：见上一篇文章。
2. **残差连接 + 层归一化**：$\text{LayerNorm}(x + \text{Sublayer}(x))$，缓解梯度消失、加速收敛。
3. **逐位置前馈网络（FFN）**：两层全连接 $+\ \text{ReLU}$（或 GeLU/SwiGLU）。它提供非线性，是模型记忆事实的关键。

## 四、Masked Self-Attention

解码器在生成第 $t$ 个 token 时，不能"看到"未来的 token。做法是在 softmax 之前对未来位置加上一个很大的负数：

```math
\text{mask}(M_{ij}) = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
```

这样训练和推理都自回归地逐位预测。

## 五、端到端前向传播（以翻译为例）

1. **源语言 token 化**：例如 "I love you" → $[I, \text{love}, you]$。
2. **embedding + 位置编码**：得到 $X \in \mathbb{R}^{3 \times d}$。
3. **编码器**：经 N 个 block 处理，得到上下文表示 $C$。
4. **解码器输入**：已生成的目标 token（推理时为当前序列；训练时用 teacher forcing）。
5. **Masked self-attention**：目标序列内部依赖建模。
6. **Cross-attention**：以解码器状态为 Q，$C$ 提供 K/V。
7. **FFN + 残差 + LayerNorm**。
8. **线性 + Softmax**：输出下一个 token 的概率分布。
9. **重复 5-8** 直到生成 EOS 标记。

## 小结

Transformer = 位置编码 + 多头注意力 + FFN + 残差 + LayerNorm。这套组合既可并行训练，又能捕捉长距离依赖，是它统治 NLP 领域十年的关键。下一篇我们会沿着这条主线，比较 GPT、BERT 与 LLaMA 的演进。
