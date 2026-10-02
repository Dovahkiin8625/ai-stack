# 预训练目标：从 MLM 到 CLM

Transformer 给出了"用什么结构读序列"，但回答"模型到底学什么"的是**预训练目标**。同一个 Transformer decoder，在 MLM 目标下会变成 BERT，在 CLM 目标下会变成 GPT。目标函数决定了模型看到的"监督信号"，进而决定了它擅长什么。本文梳理六种主流预训练目标的数学形式、典型代表与适用场景，并给出 CLM/MLM 损失对比与一个小型数据采样器。

## 一、为什么目标函数决定模型能力

把语言模型写成条件概率的形式：

$$
p(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t}, \text{context})
$$

不同的预训练目标，本质是在选择**对哪些 $x_t$ 计算条件概率、用什么 context**。这个选择决定了：

- 模型能看到哪一侧的上下文（双向 / 单向 / 前缀）。
- 模型会学到什么能力（理解 / 生成）。
- 推理时怎么用（判别式 / 生成式）。

下面六种目标，是过去七年最具影响力的范式。

## 二、CLM（Causal Language Modeling, GPT 风格）

让模型从左到右逐 token 预测下一个：

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

**代表**：GPT-2/3/4、LLaMA、Mistral、Qwen。**特点**：天然适合生成式任务；推理时一次前向就能逐 token 自回归。**缺点**：每个 token 只能看到上文，下文信息完全浪费。

## 三、MLM（Masked Language Modeling, BERT 风格）

随机遮住约 15% 的 token，让模型用**双向**上下文预测被遮部分：

$$
\mathcal{L}_{\text{MLM}} = -\sum_{t \in \text{masked}} \log p_\theta(x_t \mid x_{\setminus t})
$$

其中 $x_{\setminus t}$ 是除位置 $t$ 外的全序列。**实现细节**：被选中位置中 80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持原样——这 10% / 10% 是为了让模型在微调时也能识别真实 token 分布。**代表**：BERT、RoBERTa、ALBERT。**缺点**：预训练与微调不一致（微调时没有 `[MASK]`），且不适合直接做生成。

## 四、Prefix LM（Prefix Language Modeling）

把一段 prefix 内的 token 设为**双向可见**，prefix 之后仍按 CLM 单向：

$$
\mathcal{L}_{\text{Prefix}} = -\sum_{t > L_p} \log p_\theta(x_t \mid x_{<t})\quad\text{with}\quad x_{1..L_p} \text{ 全连接}
$$

**代表**：UniLM 1/2/3、T5（span-corruption 可以看成 prefix LM 的特例）。**特点**：兼顾理解与生成；适合 NLU+NLG 联合任务。

## 五、Permutation LM（XLNet 风格）

把序列的一个随机排列 $\pi$ 当作生成顺序，但**位置编码**保持原序：

$$
\mathcal{L}_{\text{Perm}} = -\sum_{t=1}^{T} \log p_\theta(x_{\pi_t} \mid x_{\pi_{<t}})
$$

物理实现需要"two-stream attention"等 trick 才能让每个位置看到正确的上文集合。**代表**：XLNet。**特点**：理论上比 BERT 更优雅（无 [MASK]、能看到双向上下文），但实现复杂，工程上没能跑赢纯 CLM + 规模。

## 六、ELECTRA 的 RTD（Replaced Token Detection）

不用 MLM 的"重建"思路，而是让一个小**生成器**把部分 token 替换成看起来合理但实际是错的，再让**判别器**判断每个位置是否被替换：

$$
\mathcal{L}_{\text{RTD}} = -\sum_{t=1}^{T} \left[ y_t \log D(x_t) + (1 - y_t) \log(1 - D(x_t)) \right]
$$

其中 $y_t \in \{0, 1\}$ 是"是否被替换"的标签。**特点**：每个 token 都贡献监督信号（不像 MLM 只有 15%），sample-efficiency 极高——ELECTRA-Small 在 GLUE 上能逼近 BERT-Base。**缺点**：生成器会引入额外成本。

## 七、T5 的 Span-Corruption 与 UL2

### Span-Corruption（T5）

把若干段连续 span 替换为 sentinel token（每个 span 一个独立 sentinel），让模型按顺序生成被遮的 span：

```text
原文:  Thank you for inviting me to your party last week.
输入:  Thank you <X> me to your <Y> week.
输出:  <X> for inviting <Y> party last <Z>
```

监督信号比 MLM 更密集（一次预测一个完整短语而非单 token），学习更高效。

### UL2（Tay et al. 2022）

把 R-denoiser（prefix LM）、X-denoiser（span corruption）、S-denoiser（严格前缀）等多种目标**融合**到一个模型，用 mode token 控制，让模型同时具备理解与生成能力。它的论文还引入了"模式切换训练"，效果上更像"什么目标都学一点"。

## 八、CLM 与 MLM 的损失函数对比代码

下面的代码演示了两种目标在 PyTorch 中的实现差异：

```python
import torch
import torch.nn.functional as F


def clm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    CLM: 预测下一个 token。
    logits: (B, T, V), labels: (B, T)，labels[:, t] = x[:, t+1]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    MLM: 预测被 mask 的 token。
    labels: (B, T)，非 mask 位置为 -100（被 ignore）
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


# 简单烟测
B, T, V = 2, 8, 100
logits = torch.randn(B, T, V, requires_grad=True)
labels_clm = torch.randint(0, V, (B, T))
labels_mlm = labels_clm.clone()
mask = torch.rand(B, T) < 0.15
labels_mlm[~mask] = -100

print("CLM loss:", clm_loss(logits, labels_clm).item())
print("MLM loss:", mlm_loss(logits, labels_mlm).item())
```

注意 `ignore_index=-100` 是 PyTorch 交叉熵的默认约定——非 mask 位置用 `-100` 屏蔽，是 HuggingFace Transformers 训练 MLM 时的标准做法。

## 九、预训练数据的采样策略

光有目标还不够，**怎么采样训练样本**对最终能力影响巨大。一个简单的"打包 + 按比例采样"实现：

```python
import random
from typing import List


class MultilingualSampler:
    """多源语料按比例采样。sources = [(name, path, ratio), ...]"""

    def __init__(self, sources: List[tuple], chunk_size: int = 2049):
        self.sources = sources
        self.chunk_size = chunk_size
        # 预读所有文本到内存（真实场景用 mmap）
        self.buckets = []
        for name, path, ratio in sources:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.buckets.append((name, text, ratio))
        self.weights = [s[2] for s in sources]

    def sample_chunk(self) -> str:
        # 按权重选语种
        name, text, _ = random.choices(self.buckets, weights=self.weights, k=1)[0]
        max_start = max(0, len(text) - self.chunk_size)
        start = random.randint(0, max_start)
        return text[start: start + self.chunk_size]


# 使用
sampler = MultilingualSampler([
    ("zh", "data/zh.txt", 0.5),
    ("en", "data/en.txt", 0.4),
    ("code", "data/code.txt", 0.1),
])
batch = [sampler.sample_chunk() for _ in range(8)]
```

实际生产中还会做：

1. **质量过滤**：用 fastText classifier 或 perplexity filter 去掉低质文本。
2. **去重**：exact dedup + MinHash fuzzy dedup，避免重复片段主导训练。
3. **打包（Packing）**：把多个短文档拼到固定 `seq_len`，减少 padding 浪费。
4. **温度采样**：用 `p_i^(1/T)` 控制多语种 / 多领域的相对比例（$T<1$ 提升尾部权重）。

## 十、目标选择速查表

| 目标 | 上下文 | 适合任务 | 代表模型 |
|---|---|---|---|
| CLM | 单向 | 生成、对话、零样本推理 | GPT、LLaMA、Mistral |
| MLM | 双向 | 分类、NER、检索 | BERT、RoBERTa |
| Prefix LM | prefix 内双向 | NLU + NLG 联合 | UniLM、T5 |
| Span-corruption | 双向 | 通用 seq2seq | T5、UL2 |
| RTD | 双向 | 高效判别 | ELECTRA |
| Permutation | 双向 | 优雅但工程复杂 | XLNet |

**经验法则**：今天绝大多数"前沿"大模型都是 **decoder-only + CLM**，原因不是 MLM 不好，而是 CLM + 规模 + RLHF 在生成与推理任务上表现更直接，部署也最简单（一个 `model.generate` 走天下）。

## 小结

预训练目标是"模型从语料中学什么"的根本定义。CLM 把语言建模压缩成"猜下一个 token"——简单却能 scale 到千亿参数；MLM 用双向信号换取更强的理解力；ELECTRA 用替换检测把监督密度拉到 100%；T5 / UL2 把多种目标融合，让一个模型兼具理解与生成。今天的 LLM 几乎都站在 CLM 的肩膀上，但了解其它目标的优劣，能帮助你在微调或垂直训练时做出更合理的目标设计。
