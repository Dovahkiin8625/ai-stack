# 上下文相关词向量：从 ELMo 到 BERT 的表示革命

静态词向量（Word2Vec、GloVe）有一个根本缺陷：**一个词只有一个向量**。"bank" 在 "river bank" 和 "commercial bank" 中语义完全不同，模型却只能给同一个向量。这一痛点催生了**上下文相关词向量**（contextual embeddings）——同一个词在不同上下文里拥有不同表示。本文沿着 ELMo（2018）→ BERT（2018）→ GPT（2018）→ RoBERTa（2019）的时间线，剖析它们的架构差异、预训练目标与适用场景。

## 一、为什么需要"上下文相关"

考虑两个句子：

- "He sat on the **river bank**."
- "She went to the **commercial bank**."

静态词向量给 `bank` 一个固定的 $\mathbf{v}_{\text{bank}}$，模型无法从向量里看出"这是河岸还是银行"。但人类读这两个句子时，对 `bank` 的理解截然不同——因为**上下文**改变了它的含义。

直觉上，理想的表示应该满足：

$$
\mathbf{h}_{\text{bank}} = f(w_{\text{bank}}, w_{i-2}, w_{i-1}, w_{i+1}, w_{i+2}, \dots)
$$

即 `bank` 的向量不仅取决于自身，还取决于整个句子的所有 token。

## 二、ELMo：双向 LSTM 的浅层融合

ELMo（Embeddings from Language Models, Peters et al., 2018）的核心思路是：**用一个双向语言模型分别编码"从左到右"和"从右到左"的上下文**。

### 2.1 架构

ELMo 的底层是一个多层（典型 $L=2$）双向 LSTM。给定句子 $(t_1, t_2, \dots, t_N)$：

**前向 LSTM** 建模 $P(t_k \mid t_1, \dots, t_{k-1})$：

$$
\overrightarrow{\mathbf{h}}_k^{(j)} = \text{LSTM}^{(j)}_{\rightarrow}(\overrightarrow{\mathbf{h}}_{k-1}^{(j)}, [\overrightarrow{\mathbf{h}}_k^{(j-1)}; \mathbf{e}_{t_k}])
$$

**后向 LSTM** 建模 $P(t_k \mid t_{k+1}, \dots, t_N)$：方向反过来。

最终每个位置的表示是**所有层加权和**：

$$
\text{ELMo}_k = \gamma \sum_{j=0}^{L} s_j \mathbf{h}_k^{(j)}, \quad s_j = \text{softmax}(\mathbf{w}_s)_j
$$

其中 $\mathbf{h}_k^{(0)} = \mathbf{e}_{t_k}$（字符级 CNN 编码），$\mathbf{h}_k^{(j)}$ 是第 $j$ 层 BiLSTM 的拼接 $[\overrightarrow{\mathbf{h}}_k^{(j)}; \overleftarrow{\mathbf{h}}_k^{(j)}]$，$\gamma$、$s_j$ 是任务相关的可学习缩放与权重。

### 2.2 ELMo 的特点与局限

- **特征式（feature-based）**：ELMo 的输出作为下游模型的额外输入，下游可以**冻结**它。它对算力紧张的场景友好——BERT-base 要 110M 参数微调，ELMo 可以"挂上去"用。
- **双向但不够深**：只有 2 层 LSTM，远不如 BERT 的 12 层 Transformer 信息整合能力。
- **任务相关融合权重**：$s_j$ 需要在下游任务上重新学习，相当于加了一个小型超参搜索。

## 三、BERT：双向 Transformer 的 MLM

BERT（Bidirectional Encoder Representations from Transformers, Devlin et al., 2018）彻底改变了 NLP 的预训练范式——它用**掩码语言模型**（Masked Language Model, MLM）+ **下一句预测**（Next Sentence Prediction, NSP）作为预训练任务，得到的表示直接刷新 11 个 NLP 任务的 SOTA。

### 3.1 架构

BERT 的 backbone 是 Transformer encoder（与 Vaswani et al. 2017 完全一致），只是层数更深：

| 模型 | 层数 | 隐藏维度 | 头数 | 参数量 |
| --- | --- | --- | --- | --- |
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

输入由三部分 embedding 相加得到：

$$
\mathbf{e}_i = \mathbf{E}_{\text{token}}(t_i) + \mathbf{E}_{\text{segment}}(s_i) + \mathbf{E}_{\text{pos}}(i)
$$

`segment` 让 BERT 能处理句子对（问题 + 段落），`pos` 是可学习的位置编码（最大 512）。

### 3.2 MLM：掩码语言模型

随机选择 15% 的 token 做处理：

- 80% 的时间替换为 `[MASK]`。
- 10% 的时间替换为随机词。
- 10% 的时间保持原词不动。

只对这些被选中的位置计算交叉熵损失。这种"偶尔替换为随机词"的策略是为了缓解**预训练-微调 mismatch**——微调时不会出现 `[MASK]`，所以模型不能只依赖 `[MASK]` 标记。

### 3.3 NSP：下一句预测

50% 的样本是真实的"句子 A + 句子 B"对，50% 是随机拼接的句子 B。模型预测 B 是不是 A 的真实下一句。NSP 在 RoBERTa 之后被证明**几乎无用**——它更多在捕捉主题相似度，而非真正的句子连贯性。

### 3.4 输出与微调

BERT 的输出是每个位置的上下文向量 $\mathbf{h}_i \in \mathbb{R}^d$。下游任务有四种典型用法：

1. **分类**：取 `[CLS]` 位置的向量 $\mathbf{h}_{\text{[CLS]}}$，过全连接层。
2. **序列标注**：取每个位置的 $\mathbf{h}_i$，过全连接层预测标签。
3. **问答**：取问题 + 段落拼接，预测答案的 `start` 和 `end` 位置。
4. **句对匹配**：把两个句子拼接，做分类。

## 四、GPT：单向 Transformer 的自回归预训练

GPT（Generative Pre-Training, Radford et al., 2018）与 BERT 的关键区别是**预训练目标**——GPT 用标准的**自回归语言模型**（Causal Language Model, CLM）：

$$
\mathcal{L} = -\sum_{i=1}^{N} \log P(t_i \mid t_1, \dots, t_{i-1})
$$

GPT 用 Transformer **decoder**（带 causal mask），所以每个位置只能看到自己和之前的 token。代价是无法直接得到"双向"的表示——但 GPT 的优势是**天生适合生成**：自回归解码天然契合。

### 4.1 GPT-1 / GPT-2 / GPT-3 / GPT-4 的演进

- **GPT-1 (2018)**：117M 参数，12 层。证明"生成式预训练 + 任务微调"可行。
- **GPT-2 (2019)**：1.5B 参数，48 层。Zero-shot 任务迁移能力首次显现。
- **GPT-3 (2020)**：175B 参数，96 层。In-context learning（少样本提示）成为主流范式。
- **GPT-4 (2023)**：多模态、长上下文、强化学习微调。

GPT 系列把"生成"作为主战场，而 BERT 系列更擅长"理解"任务。**两者并非对立，而是互补**——BERT 派擅长分类、QA、NER 等"读"的任务；GPT 派擅长对话、写作、代码等"写"的任务。

## 五、RoBERTa、ALBERT、ELECTRA 等改进

### 5.1 RoBERTa（Liu et al., 2019）

Facebook 对 BERT 的系统性 ablation 得出几条关键改进：

1. **更大数据**：从 16GB 扩到 160GB。
2. **更长训练**：更多步数 + 更大 batch。
3. **去掉 NSP**：改为连续长文档采样。
4. **动态 MLM 掩码**：每轮训练重新生成掩码，提升数据效率。

RoBERTa 在 GLUE 上比 BERT 高出 2-3 个点，验证了"**训练量比模型创新更重要**"。

### 5.2 ALBERT（Lan et al., 2019）

通过**跨层参数共享**和**嵌入层分解**大幅降低参数量：

- 嵌入矩阵分解：把词表 $V \times H$ 拆成 $V \times E + E \times H$（$E \ll H$）。
- 跨层共享：12 层 Transformer 共享同一组参数。

ALBERT-base 只有 12M 参数，但效果接近 BERT-base（110M），是参数高效模型的重要里程碑。

### 5.3 ELECTRA（Clark et al., 2020）

提出**替换 token 检测**（Replaced Token Detection, RTD）替代 MLM：

1. 训练一个小的生成器 $G$，把 15% 的 token 替换为生成器认为合理的词。
2. 训练判别器 $D$ 判断每个位置是否被替换。

$$
\mathcal{L}_D = -\sum_i \left[ y_i \log D(x_i) + (1-y_i) \log (1 - D(x_i)) \right]
$$

ELECTRA 的关键洞察：**判别所有位置**（不是 15% 的 masked 位置），让每个 token 都有梯度信号，训练效率比 MLM 高 4 倍。

## 六、如何选型

| 任务 | 推荐模型 |
| --- | --- |
| 文本分类 / NER / 抽取式 QA | BERT, RoBERTa, ELECTRA |
| 多标签分类 / 句子对匹配 | SBERT, SimCSE |
| 文本生成 / 对话 / 代码 | GPT-2/3/4, LLaMA, Qwen |
| 长文档理解 | Longformer, BigBird, LED |
| 多语言 | XLM-R, mBERT, mDeBERTa |
| 中文 | Chinese-BERT, ERNIE, RoFormer |

## 七、PyTorch：从零实现一个简化 BERT block

```python
import torch
import torch.nn as nn
import math


class BertSelfAttention(nn.Module):
    """BERT 的多头自注意力（与 LLaMA 不同：这里用 Post-Norm 与绝对位置编码）。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, T, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, T, T)
        if attention_mask is not None:
            scores = scores + attention_mask  # mask: pad=0 → -inf
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                       # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj_drop(self.out(out))


class BertBlock(nn.Module):
    """一个 BERT block：MHA → Post-Norm → FFN → Post-Norm"""

    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int, dropout: float = 0.1):
        super().__init__()
        self.attn = BertSelfAttention(hidden_size, num_heads, dropout)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_ln = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        x = self.attn_ln(x + self.attn(x, attention_mask))
        x = self.ffn_ln(x + self.ffn(x))
        return x
```

注意 BERT 用的是 **Post-Norm**（LayerNorm 在残差后），与 LLaMA 等现代 LLM 的 Pre-Norm 相反。Post-Norm 训练更深网络时容易不稳定，所以现代模型几乎都用 Pre-Norm——这是 BERT 之后学术界反思的产物。

## 八、上文表示与 Sentence Embedding

BERT 的 `[CLS]` 句向量质量并不好——直接对所有 token 向量取平均（mean pooling）通常更优。Sentence-BERT（Reimers & Gurevych, 2019）则用**孪生网络 + 对比学习**专门训练句向量，我们将在下一篇详细讨论。

## 小结

| 模型 | 架构 | 预训练目标 | 擅长 |
| --- | --- | --- | --- |
| ELMo | BiLSTM × 2 | 双向 LM | 特征式迁移 |
| BERT | Transformer enc | MLM + NSP | 理解、分类 |
| GPT | Transformer dec (causal) | CLM | 生成 |
| RoBERTa | Transformer enc | MLM（动态） | 通用理解 |
| ELECTRA | Transformer enc | RTD | 高效训练 |

从 Word2Vec 到 BERT，NLP 的表示学习走过了一条**"上下文越来越丰富"**的道路——从静态到上下文、从浅到深、从单向到双向。BERT 之后，理解类任务的范式基本稳定，而生成类任务还在被 GPT、LLaMA 系列快速重写。下一篇我们将看到如何把这些上下文向量聚合成高质量的**句子/段落级表示**，用于检索、聚类、对比学习等场景。
