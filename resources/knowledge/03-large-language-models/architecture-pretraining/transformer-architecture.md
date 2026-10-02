# Transformer 架构详解：从注意力机制到现代 LLM

Transformer 是过去七年 NLP 乃至整个深度学习最重要的架构创新。它的核心——**Self-Attention**——让序列中任意两个位置都能以 $O(1)$ 的"路径长度"直接交互，彻底摆脱了 RNN 的串行依赖。本文从注意力机制的数学推导出发，逐层展开 Multi-Head、位置编码、FFN、归一化策略与现代 LLM 的工程取舍，最后用 PyTorch 从零实现一个最小可用的 Multi-Head Self-Attention。

## 一、为什么要"注意力"

在 Transformer 出现之前，序列建模主要靠 RNN / LSTM。它的根本问题是**信息必须沿时间步串行传递**：第 $t$ 步的隐藏状态要"看"第 $1$ 步，必须经过 $t-1$ 次矩阵乘法。距离越远，信号越弱——这就是所谓的"长程依赖"难题。

Self-Attention 的核心思想很简单：**让序列中任意两个位置直接相连**。给定输入序列 $X = (x_1, \dots, x_n)$，每个位置都能用一个加权平均汇聚所有位置的信息，权重由内容本身决定：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

其中 $Q, K, V$ 是同一输入经不同线性投影得到的"查询 / 键 / 值"。物理直觉：每个 token 用 $Q$ 去"问"所有 token 的 $K$，匹配度越高，从对应 $V$ 中拿到的信息越多。

## 二、Scaled Dot-Product Attention 的数学细节

逐元素展开公式：

1. **投影**：$Q = X W_Q$, $K = X W_K$, $V = X W_V$，其中 $W_Q, W_K \in \mathbb{R}^{d \times d_k}$, $W_V \in \mathbb{R}^{d \times d_v}$。
2. **打分**：$\text{scores} = Q K^{\top} \in \mathbb{R}^{n \times n}$，第 $i$ 行第 $j$ 列表示位置 $i$ 对位置 $j$ 的关注度。
3. **缩放**：除以 $\sqrt{d_k}$。原因是当 $d_k$ 大时，$Q K^{\top}$ 的方差是 $d_k$，softmax 会进入梯度极小的饱和区；除以 $\sqrt{d_k}$ 让方差归一。
4. **掩码（可选）**：在自回归场景下，需要把位置 $j > i$ 的分数置 $-\infty$，确保位置 $i$ 看不到未来。
5. **加权求和**：$\text{softmax}$ 后乘 $V$，得到每个位置融合全局信息的表示。

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top} + M}{\sqrt{d_k}}\right) V
$$

其中 $M$ 是上三角为 $-\infty$、下三角与对角为 $0$ 的掩码矩阵。

## 三、Multi-Head Attention：分头看不同子空间

把 $d$ 维的 $Q, K, V$ 拆成 $h$ 个头（每头 $d_k = d/h$），各自独立做一次 attention，最后拼接再投影回来：

$$
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O
$$
$$
\text{head}_i = \text{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)})
$$

直觉：**不同头可以学不同的关系**——一个头追踪句法依赖，另一个头捕捉共指关系，还有一个头关注相邻位置。参数量与单头相近，但表达能力大幅提升。LLaMA-3-8B 的 $d=4096$, $h=32$，每头 $d_k=128$。

## 四、位置编码：让模型感知"顺序"

Self-Attention 本身是**置换不变**的——把输入序列打乱，输出也会以同样方式打乱。但语言是有顺序的，必须显式注入位置信息。三种主流方案：

### 1. Sinusoidal（原始 Transformer, 2017）

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}),\quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
$$

优点：可以"外推"到训练时未见过的长度。缺点：实际效果不如可学习编码与 RoPE。

### 2. RoPE（Rotary Position Embedding, Su et al. 2021）

把位置信息编码成对 $Q, K$ 向量的**旋转变换**：

$$
\text{RoPE}(q_m, m) = R_m q_m,\quad R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
$$

其中 $\theta$ 随维度变化。RoPE 的关键性质是**相对位置依赖只与 $(m-n)$ 有关**，且自然支持长度外推。现代 LLM（LLaMA、Mistral、Qwen）几乎全部采用 RoPE。

### 3. ALiBi（Attention with Linear Biases, Press et al. 2022）

不修改 $Q, K$，而是在 attention 分数上加一个与距离成正比的负偏置：

$$
\text{score}_{i,j} = q_i^{\top} k_j - \alpha \cdot |i - j|
$$

简单、对长上下文友好，但表达能力一般。

## 五、FFN、归一化与 Pre/Post-Norm

Transformer block 的另一关键是**逐位置前馈网络（FFN）**：

$$
\text{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2
$$

通常 $W_1$ 把维度从 $d$ 升到 $4d$，$W_2$ 再降回来。在现代 LLM 里，这个 $4d$ 中间层也是 MoE 化的主要改造点。

**归一化策略**：

- **Post-Norm**（原始 Transformer）：`LayerNorm(x + Sublayer(x))`，训练深层网络不稳。
- **Pre-Norm**（GPT-2 / LLaMA）：`x + Sublayer(LayerNorm(x))`，梯度顺畅，几乎所有现代 LLM 都用 Pre-Norm。
- **RMSNorm**（LLaMA / Mistral）：去掉了 LayerNorm 的均值中心化，只保留缩放，计算更快、效果相当。

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot g,\quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$

## 六、现代 LLM 的 KV-cache 友好结构（LLaMA）

推理时每个 token 都要复用历史的 K、V，所以现代 LLM 的设计都**优先照顾 KV cache**：

1. **Pre-Norm + RMSNorm**：训练稳定、归一化计算量小。
2. **RoPE**：相对位置编码，与 KV cache 兼容。
3. **GQA / MQA**：把 $h$ 个 query 头分成 $g$ 组共用 KV（`num_kv_heads < num_heads`），把 KV 显存砍 $h/g$ 倍。
4. **SwiGLU FFN**：把 ReLU 换成带门控的 $\text{SwiGLU}(x) = \text{SiLU}(x W_1) \odot (x W_3)$，参数略增但效果更好。
5. **无 bias**：LLaMA 在所有线性层上去掉了 bias，进一步省参数与算力。

## 七、PyTorch 实现：最小 Multi-Head Self-Attention

下面是从零实现的带 causal mask 的 Multi-Head Self-Attention，可直接 `python mhsa.py` 跑通：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """最小可用的 Multi-Head Self-Attention（LLaMA 风格：RoPE + GQA 可选）。"""

    def __init__(self, d_model: int, num_heads: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 一体化投影，最后再 split 成多头（实现更高效）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 因果 mask：上三角为 -inf
        mask = torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        # 投影并切多头：(B, T, D) -> (B, H, T, Dh)
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # Scaled dot-product attention（手写版，便于看清结构）
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)   # (B, H, T, T)
        scores = scores + self.causal_mask[:T, :T]                       # 因果掩码
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                      # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)             # (B, T, D)
        return self.o_proj(out)


# 烟测：构造输入，跑一次 forward
if __name__ == "__main__":
    torch.manual_seed(0)
    mhsa = MultiHeadSelfAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 64, 512)            # (batch=2, seq=64, dim=512)
    y = mhsa(x)
    print("output shape:", y.shape)        # torch.Size([2, 64, 512])
```

代码里几个值得注意的点：

- **一体化投影再 split**：把 `(B, T, D)` 投影到 `D` 后用 `view` 切成 `(B, H, T, Dh)`，比三个独立线性层更高效。
- **`causal_mask` 用 buffer 注册**：不参与梯度、随模型 `.to(device)` 自动迁移。
- **`scaled = scores / sqrt(Dh)`**：在量化训练中也可以换成 `1/sqrt(Dh)` 作为乘数融合到 QK 投影里。

把这段塞进 `LayerNorm -> Attention -> Residual -> LayerNorm -> SwiGLU FFN -> Residual`，就是一个最小的 decoder block。把它重复 $L$ 次，就是一个完整的 decoder-only LLM。

## 小结

Transformer 的关键洞察是**用 attention 替代 recurrence**，让任意两个位置以常数路径长度直接相连；Multi-Head 让不同子空间并行学不同关系；Pre-Norm + RMSNorm + RoPE + GQA 是当前 LLaMA 系模型的"四件套"，既稳定训练，又对 KV-cache 友好。从零实现一个 MHA 不到 50 行代码，但要真正"好用"，仍要靠位置编码、归一化与并行策略的工程细节共同托底。下一篇我们将看到这种架构在不同**预训练目标**下能学到什么——从 BERT 的 MLM 到 GPT 的 CLM。
