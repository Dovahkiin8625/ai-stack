# 注意力机制详解：从直觉到多头与掩码

注意力（Attention）是 Transformer 唯一真正的"新东西"。本文承接 [transformer-architecture.md](transformer-architecture.md) 中提到的"注意力是核心"，系统讲解为什么需要注意力、Query/Key/Value 的信息检索类比、Scaled Dot-Product Attention 的数学推导、为什么除以 $\sqrt{d_k}$、多头注意力的设计哲学与复杂度、注意力掩码、以及一个完整的 3-token 手算示例。下一篇 [gpt-vs-llama-architecture.md](gpt-vs-llama-architecture.md) 会再回到多头的工程优化（GQA）来谈节省显存。

## 一、为什么需要注意力：RNN 的长程瓶颈

在 Transformer 出现之前，序列建模的主力是 RNN/LSTM：

```text
        h1       h2       h3       h4       h5
x1 ──►  ●  ──►  ●  ──►  ●  ──►  ●  ──►  ●  ──► …
        ↑        ↑        ↑        ↑        ↑
      递归压缩   递归压缩   递归压缩   递归压缩
      (信息每步被压进固定向量)
```

每个 $h_t$ 都是过去所有信息的"压缩"——一个固定维度的隐状态。当句子中**主语和谓语相距 50 个 token**（比如定语从句、长篇技术文档），信息需要经过 50 次乘法递归才能"传到"对应位置，几乎注定梯度消失或语义失真。

**注意力的核心思想**：与其把所有历史压成一个向量，不如让当前位置**直接查询**历史任意位置，按相关性加权聚合——距离不再是障碍，路径长度是 $O(1)$。

举一个具体的"长程依赖"例子：

```text
"The cat, which had been sleeping under the warm afternoon sun
 that filtered through the half-open window in the living room
 of the old house that my grandmother grew up in, ____ (ate)."

主语 "cat" 与动词相隔约 30 个 token
RNN：必须沿链式传递 30 次才能对齐
Attention：动词位置直接 query 主语位置，一次对齐
```

## 二、Query / Key / Value：信息检索的类比

注意力可以理解为一次"软查询"。把序列想成一个**图书馆**：

- **每本书有**：书名（**Key**）、内容（**Value**）。
- **你带着**一个问题（**Query**）走进图书馆。
- **流程**：拿你的 Query 跟每本书的 Key 比相似度 → 决定"借阅比例" → 按比例混合所有 Value。

```text
        Query              Keys                  Values
       (问题)             (索引)                 (内容)

       "什么是 RMSNorm"  ──►  "AdamW"            ──►  AdamW 论文…
                          ──►  "RMSNorm"          ──►  RMSNorm 论文…  (高相似)
                          ──►  "Transformer"      ──►  架构综述…
                          ──►  "BPE"              ──►  分词论文…

相似度计算          softmax 归一化              加权求和
 q · k_i            α_i = exp(q·k_i)/Σexp     output = Σ α_i · v_i
```

**形式化**：

- Query $Q \in \mathbb{R}^{n \times d_k}$：当前要"提问"的表示。
- Keys $K \in \mathbb{R}^{n \times d_k}$：所有位置提供的"索引"。
- Values $V \in \mathbb{R}^{n \times d_v}$：所有位置真正提供的"内容"。

在 self-attention 里，$Q, K, V$ **都来自同一序列**（只是经过不同的线性投影）；在 cross-attention 里 $Q$ 来自一个序列、$K, V$ 来自另一个序列——这是后面 [transformer-architecture.md](transformer-architecture.md) 中解码器的关键。

## 三、Scaled Dot-Product Attention：完整推导

最常用的注意力形式是 **Scaled Dot-Product Attention**，定义为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

逐项拆解：

```text
输入：
  Q  (n × d_k)    ──► "查询"
  K  (m × d_k)    ──► "索引"
  V  (m × d_v)    ──► "内容"

步骤 1: 相似度矩阵
  S = Q · Kᵀ       ──► (n × m) 矩阵，每个 (i, j) 是 query_i 与 key_j 的点积

步骤 2: 缩放
  S' = S / √d_k    ──► 让数值进入 softmax 的"好区间"

步骤 3: 行 softmax（按 key 维度归一化）
  A = softmax(S')  ──► 每行是一个概率分布 α_{i,j}

步骤 4: 加权聚合
  output = A · V   ──► (n × d_v)，每行是所有 value 的加权平均
```

直观上，$S_{ij} = q_i^\top k_j$ 衡量"位置 $j$ 的内容对位置 $i$ 有多相关"；softmax 把相关性变成概率；最后用概率去 mix values 得到输出。

## 四、为什么除以 $\sqrt{d_k}$：方差与梯度

这一项不是"看起来无害"的工程细节，而是**训练稳定性的关键**。

**方差分析**（假设 $q, k$ 的每个分量独立、零均值、方差 1）：

$$
q_i \cdot k_j = \sum_{l=1}^{d_k} q_{i,l}\, k_{j,l}
$$

$$
\mathbb{E}[q_i \cdot k_j] = 0,\quad
\text{Var}(q_i \cdot k_j) = d_k
$$

所以点积的**标准差**是 $\sqrt{d_k}$。当 $d_k = 64$（典型），点积典型值落在 $[-16, 16]$ 量级，进入 softmax 的"饱和区"。

**Softmax 饱和问题**：

```text
logits      0   1   2   3
softmax     0.21 0.27 0.34 0.18     ← 健康区间，梯度活跃

logits      0   8  16  24
softmax     ~0  ~0  ~0  ~1.0        ← 一个位置接近 one-hot
                                  ← 其他位置梯度 ≈ 0
                                  ← "赢者通吃"，训练停滞
```

除以 $\sqrt{d_k}$ 后，logits 的标准差回到 ~1，softmax 不再饱和，**梯度能均匀地流向所有位置**。这一项是 Vaswani et al. (2017) 实验验证的关键 tricks 之一。

## 五、多头注意力：不同子空间看不同关系

单个注意力头只能捕捉"一种相关性"。但语言中**语法关系**（主谓、动宾、修饰）、**语义关系**（共指、反义、同义）、**位置关系**（相邻、远距）需要不同的几何结构。**多头注意力**让模型在不同子空间里并行做多次注意力。

```text
        输入 X
          │
    ┌─────┼─────┬─────┬─────┐         head 1: 语法
    │     │     │     │     │
    ▼     ▼     ▼     ▼     ▼         head 2: 共指
  W_q¹  W_q²  W_q³  W_q⁴  W_q⁵
    │     │     │     │     │
    ▼     ▼     ▼     ▼     ▼         head 3: 远距
  Attn¹ Attn² Attn³ Attn⁴ Attn⁵        …
    │     │     │     │     │
    └─────┼─────┼─────┼─────┘
          │     │     │
          Concat → W_O → 输出
```

**形式化**：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)
$$

**参数量权衡**：把 $d_{\text{model}}$ 维的特征拆成 $h$ 个 $d_k = d_{\text{model}} / h$ 维的头，每个头的 FLOPs 是原来的 $1/h$，但要做 $h$ 次——**总 FLOPs 与单头 $d_{\text{model}}$ 维注意力几乎相同**，却换来了"多种关系"的表示能力。这是计算上"几乎免费"的升级。

实践中常见 $h$：

| 模型 | $d_{\text{model}}$ | $h$ | $d_k = d_v$ |
|---|---|---|---|
| Transformer-base | 512 | 8 | 64 |
| BERT-large | 1024 | 16 | 64 |
| GPT-3 175B | 12288 | 96 | 128 |
| LLaMA-2 70B | 8192 | 64 | 128 |

## 六、注意力掩码：让模型"看不到"某些位置

掩码把 softmax 的输入中"不该看"的位置强行置为 $-\infty$，softmax 后概率为 0：

$$
\text{masked-softmax}(x)_i = \frac{\exp(x_i + m_i)}{\sum_j \exp(x_j + m_j)},\quad m_i \in \{0,\ -\infty\}
$$

两类常见掩码：

### 6.1 Padding Mask

序列通常被补到统一长度（`<pad>` token）。注意力不应跨 token 计算相关性：

```text
序列 (有 pad):  [我, 爱, NLP, <pad>, <pad>]

mask 矩阵:
[ 0,  0,  0, -∞, -∞]
[ 0,  0,  0, -∞, -∞]
[ 0,  0,  0, -∞, -∞]
[ 0,  0,  0,  0, -∞]
[ 0,  0,  0,  0,  0 ]
```

### 6.2 Causal Mask（解码器自回归用）

生成时第 $t$ 个位置不能"偷看"第 $t+1$ 之后的未来：

```text
位置 i=1   2   3   4
   1 [  0, -∞, -∞, -∞]   ← 第1个 token 只看自己
   2 [  0,  0, -∞, -∞]   ← 第2个 token 看 1,2
   3 [  0,  0,  0, -∞]   ← 第3个 token 看 1,2,3
   4 [  0,  0,  0,  0 ]   ← 第4个 token 看 1,2,3,4
```

矩阵形式 $M_{ij} = -\infty$ 当 $j > i$，否则 $0$。**注意**：causal mask 把 $Q K^\top$ 矩阵变成下三角，推理时配合 KV cache 一次生成一个 token。

## 七、完整手算示例：3 个 token、$d_k = 2$

设三个 token 的 query/key/value（已经过 embedding + 投影）：

```text
       d_k=2
q1 = [1, 0]    k1 = [1, 0]    v1 = [1, 0]
q2 = [0, 1]    k2 = [0, 1]    v2 = [0, 2]
q3 = [1, 1]    k3 = [1, 1]    v3 = [1, 1]
```

**步骤 1：$Q K^\top$**

```text
       k1=[1,0]   k2=[0,1]   k3=[1,1]
q1=[1,0]  1          0          1
q2=[0,1]  0          1          1
q3=[1,1]  1          1          2
```

**步骤 2：除以 $\sqrt{d_k} = \sqrt{2} \approx 1.414$**

```text
S' =  [ 0.71    0       0.71 ]
      [ 0       0.71    0.71 ]
      [ 0.71    0.71    1.41 ]
```

**步骤 3：行 softmax**（用 $\exp$ 然后归一化）

```text
行 1:  exp = [2.03, 1.00, 2.03]   → α = [0.42, 0.16, 0.42]
行 2:  exp = [1.00, 2.03, 2.03]   → α = [0.20, 0.40, 0.40]
行 3:  exp = [2.03, 2.03, 4.11]   → α = [0.25, 0.25, 0.50]
```

**步骤 4：$A V$**（逐行加权求和 values）

```text
o1 = 0.42·[1,0] + 0.16·[0,2] + 0.42·[1,1]  =  [0.84, 0.74]
o2 = 0.20·[1,0] + 0.40·[0,2] + 0.40·[1,1]  =  [0.60, 1.20]
o3 = 0.25·[1,0] + 0.25·[0,2] + 0.50·[1,1]  =  [0.75, 1.00]
```

**直觉解读**：

- $q_1$ 与 $k_1$、$k_3$ 都高度相关（$k_1 = q_1$、$k_3$ 包含 $q_1$）→ 输出主要由 $v_1, v_3$ 决定。
- $q_3$ 与所有 key 都相关，且 $k_3 \cdot q_3 = 2$ 最大 → 输出权重更集中在 $v_3$。
- 每个输出都是 values 的加权混合——这就是"软聚合"。

## 八、Self-Attention vs Cross-Attention

| 类型 | $Q$ 来源 | $K, V$ 来源 | 用途 |
|---|---|---|---|
| Self-Attention | 当前序列 | 当前序列 | 编码器、解码器自回归 |
| Cross-Attention | 解码器当前步 | 编码器输出 | 翻译 / seq2seq 的"读源文"步骤 |
| Masked Self-Attention | 当前序列（被 mask） | 当前序列 | 解码器自回归生成 |

三种形式数学上完全一致（只是 $Q/K/V$ 的来源不同 + 是否有 mask），却承担完全不同的语义角色。

## 九、复杂度分析：$O(n^2 d)$ 的著名问题

```text
S = Q Kᵀ                  → n × m 次乘加     (n² 主导)
A = softmax(S)            → O(n²)
output = A V              → n × m × d 次乘加  (n² d 主导)
```

总复杂度：$O(n^2 \cdot d)$，**关于序列长度 $n$ 是平方**。

直观对比：

| 模型 | 计算模式 | 关于 $n$ |
|---|---|---|
| RNN | $h_t = f(h_{t-1}, x_t)$ | $O(n)$ 串行、$O(1)$ per-step 并行差 |
| 1D CNN | 窗口 $k$ | $O(n k)$ 全并行 |
| Self-Attention | 全连接 | $O(n^2)$ 但**完全并行** |

长序列时 $n^2$ 爆炸是 Transformer 的"原罪"，催生了大量研究（Longformer、Linformer、Performer、Mamba 等稀疏/线性注意力）。这也是为什么 LLaMA 等模型如此关心 KV cache 大小——见 [gpt-vs-llama-architecture.md](gpt-vs-llama-architecture.md)。

## 十、PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape                       # (B, T, d_model)
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # 现在 q,k,v: (B, n_heads, T, d_k)

        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, h, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = attn @ v                                          # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


# 或直接用 PyTorch 内置（生产级实现，含 flash attention 等优化）
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)
x = torch.randn(2, 64, 512)             # (B=2, T=64, d=512)
out, attn_weights = mha(x, x, x)        # Q=K=V=x  (self-attention)
print(out.shape)                        # torch.Size([2, 64, 512])
```

几个 PyTorch 工程细节：

- `batch_first=True` 让输入输出格式为 `(B, T, D)`，更直观。
- `view(..., n_heads, d_k).transpose(1, 2)` 把 head 维提到 batch 维前面，所有 head 一次矩阵乘搞定。
- `masked_fill(mask == 0, -inf)` 实现 padding / causal mask。
- 生产环境应使用 `torch.nn.functional.scaled_dot_product_attention`，它会自动选择最优实现（FlashAttention 等）。

## 小结

注意力机制让序列中任意两个位置以 $O(1)$ 路径长度直接对齐，彻底解决了 RNN 的长程瓶颈。Scaled Dot-Product Attention 通过 $Q K^\top / \sqrt{d_k}$ 计算相关性、softmax 归一化、加权聚合 $V$，其中除以 $\sqrt{d_k}$ 是为了控制 logits 方差、避免 softmax 饱和。多头注意力把表示分到 $h$ 个子空间并行计算多个关系，几乎不增加 FLOPs 但大幅提升表示能力。掩码机制让模型按需"看不见"某些位置，是 padding 处理与自回归解码的关键。复杂度的 $O(n^2)$ 既是大模型长上下文的核心瓶颈，也是众多稀疏/线性注意力研究的发力点。本文是 Transformer 的"原子操作"——下一篇 [transformer-architecture.md](transformer-architecture.md) 将把这些注意力子层组装成完整的 encoder-decoder 架构。