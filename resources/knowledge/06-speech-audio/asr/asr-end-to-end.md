# 端到端 ASR：Transformer-Transducer 与 Conformer

CTC 把对齐"折叠"成单调的，对长句、远场、口音仍显笨拙。一个朴素的问题是：**如果允许解码端自由地"回头看"声学序列的任意位置，会不会更好？** 这正是 **LAS（Listen-Attend-Spell, 2015, Chorowski et al.）** 的核心直觉——用一个注意力机制在声学编码器输出上做软对齐。再往后，**Graves 2012 提出的 RNN-T** 把"流式"和"对齐"结合得更优雅，而 **Conformer（Gulati et al., 2020）** 则把 CNN 与 Self-Attention 结合，在几乎所有公开榜上登顶。

## 一、LAS：基于注意力的编解码 ASR

### 1.1 整体结构

LAS 由三部分组成：

- **Listener（编码器）**：通常是 Pyramid-BiLSTM 或 Transformer，把 Fbank 序列 $(x_1, \dots, x_T)$ 压成高层表示 $H = (h_1, \dots, h_{T'})$。
- **Attention**：在解码步 $t$ 计算 $h_j$ 的权重 $\alpha_{t,j}$，得到上下文 $c_t$。
- **Speller（解码器）**：自回归 LSTM/Transformer，按 $(y_1, \dots, y_{t-1})$ 与 $c_t$ 预测 $y_t$。

### 1.2 软对齐

解码端在第 $t$ 步对编码器 $T'$ 个位置计算分数：

$$
e_{t,j} = \phi(v^\top \tanh(W_s s_{t-1} + W_h h_j))
$$

softmax 得注意力权重 $\alpha_{t,j}$，上下文向量：

$$
c_t = \sum_{j=1}^{T'} \alpha_{t,j} h_j
$$

直觉：**第 $t$ 个标签"看向"声学序列中相关的那一段**，这就是"软对齐"。相比 CTC 的单调对齐，软对齐可跳跃、可重复，因此 LAS 在长句上更鲁棒。

### 1.3 训练目标即 seq2seq 的最大似然

$$
\mathcal{L}_{\text{LAS}} = -\log P(Y \mid X) = -\sum_{t=1}^{U} \log P(y_t \mid y_{<t}, X)
$$

LAS 不需要任何对齐标注——但**这是有代价的**：（1）attention 软对齐是隐式的，模型可能"偷懒"把所有注意力放到某几个 token 上（**attention collapse**）；（2）LAS 几乎无法流式（解码必须看到完整声学序列）。

## 二、RNN-Transducer：流式 + 端到端

### 2.1 三大模块

RNN-T（Graves, 2012）把 LAS 的问题正面解决：

- **Encoder（声学编码器）**：把 $X = (x_1, \dots, x_T)$ 编码为 $H = (h_1, \dots, h_T)$，**保持原长**（不像 Pyramid-BiLSTM 那样下采样）。
- **Predictor（语言预测器）**：自回归 LSTM，输入历史标签 $y_{1:u-1}$，输出 $p_u$。
- **Joint network**：在每个 $(t, u)$ 位置融合声学和语言信号：

$$
z_{t,u} = \tanh(W_h h_t + W_p p_u + b), \quad P(y \mid t, u) = \text{softmax}(W_z z_{t,u})
$$

### 2.2 二维网格与"双流"输出

RNN-T 的输出空间是一个 $T \times U$ 的二维网格。**每一步可以走两步**：

- **横向（emit blank）**：消耗一帧声学，不输出新标签。
- **纵向（emit label）**：消耗一个标签，不消耗声学帧。

这与 CTC 的关键区别：**RNN-T 把"是否消耗新标签"显式地放进模型里**，让对齐和分类解耦——这是它比 CTC 更鲁棒的根本原因。

### 2.3 训练目标：与所有变量对齐路径求和

$$
P(Y \mid X) = \sum_{\text{path} \in \mathcal{B}^{-1}(Y)} \prod_{(t,u) \in \text{path}} P(\cdot \mid t, u)
$$

直接枚举路径不可行。用**前向后向**：

$$
\alpha(t, u) = \alpha(t-1, u) \, P(\epsilon \mid t-1, u) + \alpha(t, u-1) \, P(y_u \mid t, u-1)
$$

$$
\beta(t, u) = \beta(t+1, u) \, P(\epsilon \mid t, u) + \beta(t, u+1) \, P(y_{u+1} \mid t, u)
$$

归一化后：

$$
\mathcal{L}_{\text{RNN-T}} = -\log P(Y \mid X) = -\log \beta(1, 0)
$$

工程上 PyTorch 的 `torchaudio.functional.rnnt_loss` 已高度优化，支持 GPU 并行。

### 2.4 RNN-T 的流式性

RNN-T 天然**流式**：每读入一帧 $x_t$ 即可增量计算，无需等完整句结束。LAS 做不到——LAS 的解码依赖整个 $H$。这也是工业流式 ASR（Google 的 Speech-to-Text 流式、Apple Siri）几乎都用 RNN-T 系的原因。

## 四、Conformer：CNN + Self-Attention 的卷积增强

### 4.1 为什么 Self-Attention 还不够

Transformer 在 NLP 登顶，但纯 Self-Attention 在语音上有个**频率偏置缺失**问题：自注意力是"全局加权"，但局部时频模式（谐波、瞬态）天然适合 CNN。**Conformer（Gulati et al., 2020, Google）** 把这两者放在同一 block 里。

### 4.2 Conformer Block 结构

每个 Conformer block 由四个子模块组成（按顺序）：

1. **Feed-Forward Module（前 1/2）**：实现 Macaron FFN，扩张-收缩非线性。
2. **Multi-Head Self-Attention（MHSA）**：相对位置编码，全局依赖。
3. **Convolution Module**：1D 卷积捕获局部时频结构。
4. **Feed-Forward Module（后 1/2）**：再一个 FFN。

最终用**Sandwich 残差**：

$$
\tilde{x} = x + \frac{1}{2} \text{FFN}(x) + \text{MHSA}(x) + \text{Conv}(x) + \frac{1}{2} \text{FFN}(x)
$$

### 4.3 相对位置编码

正弦位置编码加在输入上时存在**长度外推问题**。相对位置编码（Shaw et al., 2018）把位置偏置放进 attention 的 Q-K 内积中：

$$
e_{ij} = \frac{x_i W^\top_Q (x_j W_K + r_{i-j})}{\sqrt{d}}
$$

$r_{i-j}$ 是一个学习型相对位置嵌入。Conformer 用**正弦相对位置编码**（Transformer-XL 风格），可外推到训练集没见过的长度。

### 4.4 PyTorch 实现：Conformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Macaron FFN: 扩张 → Swish → 收缩 → Dropout。"""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),                       # Swish 激活
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):                     # x: (B, T, D)
        return self.net(x)


class ConvModule(nn.Module):
    """1D 卷积模块：捕获局部时频模式。"""

    def __init__(self, dim: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, dim * 2, 1)   # pointwise
        self.dw = nn.Conv1d(dim, dim, kernel_size,
                            padding=kernel_size // 2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # x: (B, T, D)
        x = self.norm(x).transpose(1, 2)       # (B, D, T)
        x = F.glu(self.pw1(x), dim=1)          # GLU: 通道二等分取一半
        x = self.dw(x)
        x = self.bn(x).silu()
        x = self.pw2(x)
        x = self.drop(x).transpose(1, 2)
        return x


class RelativeMHSA(nn.Module):
    """简化的多头自注意力 + 相对位置偏置。"""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, max_rel: int = 500):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.rel_bias = nn.Embedding(2 * max_rel + 1, heads)
        self.drop = nn.Dropout(dropout)
        self.max_rel = max_rel

    def forward(self, x):                      # x: (B, T, D)
        B, T, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)     # (B, T, D) × 3
        q, k, v = [t.view(B, T, self.heads, -1).transpose(1, 2) for t in qkv]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, T)

        # 相对位置偏置：i-j 范围 [-T+1, T-1]，重映射到 [0, 2T-2]
        rel = torch.arange(T, device=x.device)
        rel_idx = (rel[None, :] - rel[:, None]).clamp(-self.max_rel, self.max_rel)
        rel_idx = rel_idx + self.max_rel
        bias = self.rel_bias(rel_idx).permute(2, 0, 1).unsqueeze(0)  # (1, H, T, T)
        attn = attn + bias

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.drop(self.out(out))


class ConformerBlock(nn.Module):
    """单个 Conformer block（前 FFN + MHSA + Conv + 后 FFN）。"""

    def __init__(self, dim: int = 144, heads: int = 4,
                 kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ff1 = FeedForward(dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = RelativeMHSA(dim, heads, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.conv = ConvModule(dim, kernel_size, dropout=dropout)
        self.norm4 = nn.LayerNorm(dim)
        self.ff2 = FeedForward(dim, dropout=dropout)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):                     # (B, T, D)
        x = x + 0.5 * self.ff1(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.conv(self.norm3(x))
        x = x + 0.5 * self.ff2(self.norm4(x))
        return self.out(x)
```

代码里几个**非显然的设计**：

- **GLU（Gated Linear Unit）** 在 ConvModule 中把通道一分为二、一半乘 sigmoid 当门控——这能让模型学到"哪些频带该保留"。
- **Macaron FFN（半步残差）**：相比标准 Transformer 的一步 FFN，半步 FFN 让模型更易训练（梯度更稳定）。
- **相对位置偏置用 Embedding**：把 $i-j$ 离散化为 $-T+1, \dots, T-1$，每个相对距离独立 Embedding。**未实现** 的相对位置编码用正弦函数则更平滑，外推性更好。

## 五、训练技巧：SpecAugment 与速度扰动

### 5.1 SpecAugment

SpecAugment（Park et al., 2019, Google）在 Fbank 频谱上做三种数据增强：

- **Time warping**：在时间轴上做小幅随机扭曲。
- **Time masking**：随机遮蔽连续 $T$ 帧（强制模型学习长时依赖）。
- **Frequency masking**：随机遮蔽连续 $F$ 个频带（强制模型学习频带间互补）。

直观：**让模型永远只看到"不完整"的频谱，迫使它学到更鲁棒的模式**。SpecAugment 在 Librispeech 上把 WER 降低 10%-20% 几乎是免费的代价。

### 5.2 Speed Perturbation

将波形按时速 0.9, 1.0, 1.1 重采样，**等价于数据三倍**。注意：变速不会改变 Fbank 的相对结构，但会改变每帧的物理时长——下游帧数要相应调整。

### 5.3 Word-piece 与 Sentence-piece

CTC / RNN-T 在字符级训练会丢失词级信号；用 **Sentence-piece**（子词单元）可以兼顾：

- 词表大小可控（通常 1000-10000）。
- OOV 不存在。
- 训练速度更快（序列更短）。

## 六、与传统 Pipeline 的对比

| 维度 | GMM-HMM Pipeline | CTC | RNN-T | Conformer |
| --- | --- | --- | --- | --- |
| 模块数 | 4-5（特征/字典/声学/解码/语言模型） | 1 | 1 | 1 |
| 训练目标 | 多模块独立 MLE | CTC Loss | CTC | CTC + Attention |
| 流式 | 全 | 半 | 全 | 半/可 |
| 长句表现 | 退化 | 退化 | 鲁棒 | 好 |
| 主流代表 | Kaldi | DeepSpeech | Google STT | Google USM |

Conformer 在工业大规模 ASR 上几乎是 2020-2024 年的"标配"——但其缺点也很明显：**计算量大、对低资源语言不友好、长尾语时常常"幻觉"**。这催生了 **Whisper** 和 **LLM-based ASR** 的新趋势，下一篇将深入讨论。

## 小结

| 方法 | 关键创新 | 优势 | 局限 |
| --- | --- | --- | --- |
| LAS | 注意力软对齐 | 长句鲁棒 | 无法流式 |
| RNN-T | 二维 grid + 双流输出 | 流式、端到端 | 训练复杂 |
| Conformer | CNN + Self-Attention | 局部 + 全局 | 计算量大 |
| SpecAugment | 频谱掩蔽 | 鲁棒提升 | 增强超参敏感 |

端到端 ASR 把多模块流水线压成了单一可微系统，让深度网络从"语音特征抽取器"升级为"语义对齐器"。但端到端不是终点——大规模弱监督预训练（Whisper）、LLM 融合（USM、SALMONN）正在把 ASR 推向新的范式。