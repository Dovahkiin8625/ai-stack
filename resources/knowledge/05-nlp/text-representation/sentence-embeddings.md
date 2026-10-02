# 句子级表示：从 Sentence-BERT 到 SimCSE 的对比学习范式

BERT 让我们拿到了高质量的 token 级上下文向量，但**句子级**表示并没有"开箱即用"的解决方案——直接用 `[CLS]` 或者平均所有 token 的向量，得到的句子嵌入在语义相似度任务上效果都很差。原因是 BERT 的**各向异性（anisotropy）**：向量空间被挤压到一个狭窄的锥形区域，任意两个句子（即使语义无关）的余弦相似度都偏高。本文沿着 Sentence-BERT（2019）→ SimCSE（2021）→ Contriever（2022）的时间线，剖析对比学习如何"撑开"这个空间、得到真正可用的句子向量。

## 一、为什么 BERT 句向量不好用

直觉上，"猫在沙发上睡觉"和"小狗在垫子上打盹"语义相近，它们的句向量应该余弦相似度高；"量子力学奠基人"和"今天天气真好"语义无关，相似度应该低。但实验发现：

- BERT `[CLS]` 向量在 STS（Semantic Textual Similarity）任务上与人类标注的 Pearson 相关性只有约 0.30-0.40。
- 任意两个随机句子的余弦相似度集中在 0.6-0.9，几乎无法区分。

原因主要有三个：

1. **预训练目标不匹配**：MLM 让 token 向量表达"被预测的下一个词"，而非"句子的整体语义"。
2. **频率偏差**：高频 token 占据向量空间的主要区域，低频 token 被推到外圈，挤压了真正的语义方向。
3. **Pooling 策略不佳**：`[CLS]` 仅在 NSP 任务上训练过，并非设计来代表整句；mean pooling 略好但仍受 token 频率影响。

## 二、Sentence-BERT：孪生网络 + NLI 微调

Sentence-BERT（Reimers & Gurevych, 2019）是第一个把 BERT 训练成"高质量句向量"的系统方法。

### 2.1 核心思路

用**孪生网络**（Siamese Network）结构：

1. 两个相同的 BERT（共享权重）分别编码句子 $u$ 和 $v$，得到句向量 $\mathbf{u}, \mathbf{v}$（mean pooling 或 `[CLS]`）。
2. 用**有监督**的句子对分类/回归目标训练。

最成功的训练数据是 **SNLI + Multi-NLI**（自然语言推理数据集），共约 100 万对"前提-假设"，标签为 entailment（蕴含）/ neutral（中性）/ contradiction（矛盾）。Sentence-BERT 用三种目标函数：

**分类目标**：把 $[\mathbf{u}, \mathbf{v}, |\mathbf{u}-\mathbf{v}|]$ 拼接过全连接层，3 分类。

**回归目标**：直接用 $\cos(\mathbf{u}, \mathbf{v})$ 与人工标注的相似度分数（0-5）做 MSE。

**Triplet 目标**：给定锚点 $a$、正例 $p$、负例 $n$，最小化：

$$
\mathcal{L} = \max\!\left(0, \; d(\mathbf{a}, \mathbf{p}) - d(\mathbf{a}, \mathbf{n}) + \text{margin}\right)
$$

直觉：让"蕴含对"的距离小于"矛盾对"的距离，间距至少为 margin。

### 2.2 效果

Sentence-BERT 在 STS-Benchmark 上把 Pearson 相关性从 BERT 的 0.47 提升到 0.85，推理速度也快了 10000 倍——因为可以直接用向量召回，不必每对句子都过 BERT。

## 三、SimCSE：无监督对比学习的里程碑

SimCSE（Gao et al., 2021）证明：**只用无监督数据，也能训练出超越 Sentence-BERT 的句向量**。它分无监督（SimCSE-unsup）和有监督（SimCSE-sup）两个版本。

### 3.1 无监督 SimCSE：Dropout 作为"数据增强"

核心洞察极其优雅：**同一个句子过两次 BERT（使用不同的随机 dropout mask），得到的两个向量互为正例**。

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j^+) / \tau)}
$$

其中 $\mathbf{h}_i^+$ 是同一个句子 $\mathbf{x}_i$ 用不同 dropout mask 重新编码得到的向量，$\mathbf{h}_j^+$ 是同 batch 中其他句子的"对偶"。温度系数 $\tau$ 通常设为 0.05。

直觉：Dropout 让 BERT 对同一个句子产生略有不同的表示，这模拟了"同一语义的不同表达"。迫使模型把这两个表示拉近，自然就学到了**对噪声鲁棒、对语义敏感**的句子嵌入。

无监督 SimCSE 在 STS-B 上达到 0.768，已经和有监督 Sentence-BERT（0.85）很接近。

### 3.2 有监督 SimCSE：用 NLI 监督对比

有监督 SimCSE 用 NLI 数据构造对比样本：

- 锚点 $a$：NLI 的"前提"。
- 正例 $p$：蕴含的"假设"（语义相关）。
- **硬负例** $n$：矛盾的"假设"（语义相关但对立）。

$$
\mathcal{L} = -\log \frac{e^{\text{sim}(\mathbf{h}_a, \mathbf{h}_p)/\tau}}{e^{\text{sim}(\mathbf{h}_a, \mathbf{h}_p)/\tau} + e^{\text{sim}(\mathbf{h}_a, \mathbf{h}_n)/\tau}}
$$

硬负例（矛盾）比随机负例难得多，迫使模型学更精细的语义差异。有监督 SimCSE 在 STS-B 上达到 0.841，超过 Sentence-BERT。

## 四、对比学习的理论直觉

对比学习的核心假设是：**向量空间中的距离反映语义距离**。InfoNCE 损失（SimCSE 用的是它的温度化变体）本质上在最大化**互信息下界**——把正例对的互信息推到最大，把其他对推到最小。

为什么对比学习能"撑开"各向异性的空间？考虑两个极端：

- 没有任何约束时，所有句向量退化成同一个点（最优解），互信息为 0。
- 加上 InfoNCE 后，必须让正例对距离近、负例对距离远。这等价于让向量在球面上**均匀分布**——这是各向同性的几何意义。

工程上，温度 $\tau$ 控制"难度"：

- $\tau \to 0$：分布变尖锐，模型对难负例极度敏感。
- $\tau \to \infty$：分布变平坦，几乎不区分负例。
- $\tau = 0.05$ 是经验最优（小数据）/ $0.1$ 是大模型常用。

## 五、Contriever：大规模检索预训练

Contriever（Izacard et al., 2022）把对比学习扩展到**无监督检索**——给定 query，召回归档语料中相关的文档。它的训练数据来自维基百科 + 互联网：

- **正例对**：文档中的相邻段落（5 句窗口）。
- **负例对**：同 batch 中的随机段落。

模型结构与 SimCSE 类似，但 query 和 document 用**两个独立的 BERT**编码（dual encoder），推理时可以离线把所有文档预编码成向量，大幅加速检索。

Contriever 的扩展版 Contriever-MSMARCO 在 BEIR 基准上达到 zero-shot 检索的 SOTA，是后续 RAG 系统的重要 backbone。

## 六、其他重要工作

### 6.1 DPR（Dense Passage Retrieval, Karpukhin et al., 2020）

Google 提出的双塔检索模型，用 QA 数据训练：

- 正例：包含答案的段落。
- 负例：BM25 难负例 + 同 batch 内随机段落。

DPR 在 Natural Questions 上把 top-20 召回率从 BM25 的 59% 提升到 78%，证明了 dense retrieval 在开放域 QA 上的威力。

### 6.2 BGE / M3E / BCE 中文句向量

中文领域也有大量优质模型：

- **BGE（BAAI General Embedding）**：智源研究院出品，中英文双语，在 C-MTEB 中文榜单长期 SOTA。
- **M3E（Moka Massive Mixed Embedding）**：开源中文 embedding，适合中等规模部署。
- **BCE（Beijing Academy of AI Embedding）**：百度出品，强调检索场景。

### 6.3 Instructor：任务指令引导

Instructor（Su et al., 2023）用一个**任务指令**（如 "Represent this sentence for retrieval: "）拼接到输入，让同一个模型能根据指令生成不同语义的向量。这把"一个 embedding 适配所有任务"变成了"按需生成 embedding"。

## 七、PyTorch 实现：简易 SimCSE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SimCSE(nn.Module):
    """无监督 SimCSE：dropout 作为隐式数据增强。"""

    def __init__(self, model_name: str = "bert-base-uncased", temperature: float = 0.05):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.temperature = temperature

    def forward(self, input_ids, attention_mask):
        # 同一个 batch 跑两次 encoder，得到 (B, D) 与 (B, D)
        emb1 = self._encode(input_ids, attention_mask)
        emb2 = self._encode(input_ids, attention_mask)
        return self._contrastive_loss(emb1, emb2)

    def _encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling（排除 pad）
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        emb = summed / counts
        return F.normalize(emb, dim=-1)   # L2 normalize，余弦相似度等价点积

    def _contrastive_loss(self, emb1, emb2):
        sim = torch.matmul(emb1, emb2.T) / self.temperature   # (B, B)
        labels = torch.arange(emb1.size(0), device=emb1.device)
        loss = (F.cross_entropy(sim, labels) +
                F.cross_entropy(sim.T, labels)) / 2
        return loss


# 烟测
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SimCSE()
    texts = ["hello world", "good morning", "completely unrelated"]
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    print("loss:", model(**batch).item())
```

注意几个关键点：

- **Mean pooling + L2 normalize**：这是 SimCSE 的标配组合。
- **对称损失**：`sim` 和 `sim.T` 都做 cross_entropy，等价于"互换正负例"。SimCSE 论文里就是这么做的。
- **训练时打开 dropout，推理时关闭**：这是"dropout 作为数据增强"的精髓——编码两次时 dropout 模式不同，相当于生成两个略有差异的正例对。

## 八、评估与基准

### 8.1 STS（Semantic Textual Similarity）

经典基准，句子对+人工标注 0-5 分相似度。报告 Pearson/Spearman 相关性。**STS-B**（STS Benchmark）是最常用的子集。

### 8.2 C-MTEB（Chinese Massive Text Embedding Benchmark）

智源研究院发布的中文 embedding 评测套件，覆盖分类、聚类、检索、重排、STS 等 6 大类任务，共 50+ 数据集。在 C-MTEB 上 SOTA 的模型，中文场景下基本可以放心使用。

### 8.3 BEIR

包含 18 个零样本检索任务的英语基准，涵盖科学、新闻、生物医学等多种领域。Contriever、DPR、ColBERT 都在 BEIR 上评测。

## 九、工程经验

1. **向量归一化**：检索场景**必须**做 L2 normalize，否则点积与余弦差距大。
2. **向量维度**：768 维是 BERT 默认；如果存储紧张，可以用 PCA 或 Matryoshka 训练降到 256 / 128。
3. **训练数据量**：1 万对高质量 NLI > 100 万对网络爬取的"伪正例"。
4. **硬负例挖掘**：随机负例训练到后期收益递减，必须主动挖掘难负例（同 topic 但不同实体等）。
5. **任务对齐**：如果下游是检索，就在检索数据上训练；如果是聚类，就在 NLI + STS 上训练——不要"通用 embedding 适配一切"。

## 小结

| 方法 | 训练数据 | 核心机制 | STS-B Pearson |
| --- | --- | --- | --- |
| BERT `[CLS]` | - | - | ~0.30 |
| Sentence-BERT | NLI | 孪生 + 分类 | 0.85 |
| SimCSE-unsup | Wikipedia | Dropout 数据增强 | 0.768 |
| SimCSE-sup | NLI + 硬负例 | 对比损失 | 0.841 |
| Contriever | Wikipedia | Dual encoder | 检索场景 SOTA |

句子级表示是从"词级别"到"应用级别"的关键一跳——没有它，RAG、向量检索、文本聚类等下游任务都无法高效运转。从 Sentence-BERT 的有监督微调，到 SimCSE 的对比学习，再到大模型时代的指令驱动 embedding，这条线一直在朝着"更好用、更通用、更便宜"的方向演进。下一篇我们将沿着这一基础进入**信息抽取**：如何用上下文向量识别命名实体、抽取关系、识别事件。
