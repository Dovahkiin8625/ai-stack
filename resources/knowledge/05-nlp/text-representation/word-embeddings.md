# 词向量：从 Word2Vec 到 GloVe 的分布式表示

在深度学习时代之前，文本必须先被表示成"数字"才能喂给模型。最朴素的 One-Hot 把每个词表示成 $V$ 维稀疏向量（$V$ 是词表大小），但它有三个致命问题：**维度爆炸**（$V$ 经常上百万）、**正交无相似度**（"猫"和"狗"的点积永远是 0）、**无法泛化**（新词必须重新进入词表）。Word2Vec（2013）和 GloVe（2014）的核心思想是——**用一个低维稠密向量，让语义相近的词在向量空间里也相近**。这就是分布式假设（Distributional Hypothesis）："上下文相似的词，语义也相似"。

## 一、分布式假设与 SVD 路线

Harris（1954）和 Firth（1957）的"词的语义由其上下文决定"是所有词向量方法的理论基石。最早的工程实现走的是**计数路线**：构造"词-上下文"共现矩阵 $M \in \mathbb{R}^{V \times V}$，然后对 $M$ 做 SVD 截断分解。

设 $M_{ij}$ 是词 $w_i$ 与上下文 $c_j$ 在语料中共同出现的次数。SVD 把 $M = U \Sigma V^\top$ 分解，取 $U$ 的前 $d$ 列作为 $d$ 维词向量：

$$
\hat{M}_{ij} = \mathbf{u}_i^\top \mathbf{v}_j
$$

SVD 路线简单、可解释，但有三个工程缺陷：（1）共现矩阵是 $V^2$ 量级，百万词表就要 TB 级内存；（2）SVD 计算复杂度 $O(V^3)$；（3）对高频词过度敏感（"the"、"is" 出现几十万次，会主导整个向量空间）。**预测路线**（Word2Vec、GloVe）则直接从局部上下文窗口中学习稠密向量，绕开了显式矩阵分解。

## 二、Word2Vec：两种预测架构

Mikolov 等人 2013 年提出的 Word2Vec 是 NLP 时代最经典的词向量方法。它包含两个对称的架构：

### 2.1 CBOW（Continuous Bag-of-Words）

用**上下文窗口内的词预测中心词**。设窗口大小为 $c$，上下文词为 $w_{t-c}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+c}$，CBOW 把它们对应的向量取平均，再过一个 softmax 预测中心词 $w_t$：

$$
P(w_t \mid \text{context}) = \frac{\exp(\bar{\mathbf{v}}_t^\top \mathbf{v}_{w_t})}{\sum_{j=1}^{V} \exp(\bar{\mathbf{v}}_t^\top \mathbf{v}_j)}
$$

其中 $\bar{\mathbf{v}}_t = \frac{1}{2c} \sum_{k \in \{-c,\dots,c\}\setminus\{0\}} \mathbf{v}_{w_{t+k}}$。

### 2.2 Skip-gram

CBOW 的"反向"——**用中心词预测上下文窗口内的每个词**：

$$
P(w_{t+k} \mid w_t) = \frac{\exp(\mathbf{v}_{w_t}^\top \mathbf{v}_{w_{t+k}})}{\sum_{j=1}^{V} \exp(\mathbf{v}_{w_t}^\top \mathbf{v}_j)}
$$

Skip-gram 在小语料下表现更好，因此工业界更常用。

### 2.3 负采样（Negative Sampling）

直接优化 softmax 的分母 $\sum_{j=1}^{V} \exp(\cdot)$ 需要遍历整个词表，对百万词表完全不可行。负采样把多分类问题转换为二分类：给定一对 $(w, c)$，判断 $c$ 是不是 $w$ 的真实上下文。训练时从词表中按 $\text{Unigram}^{3/4}$ 采样 $K$ 个"假上下文"作为负例，损失函数变为：

$$
\mathcal{L} = -\log \sigma(\mathbf{v}_w^\top \mathbf{v}_{c^+}) - \sum_{k=1}^{K} \mathbb{E}_{c^- \sim P_n} \log \sigma(-\mathbf{v}_w^\top \mathbf{v}_{c^-})
$$

直觉：让真实上下文对的向量内积尽量大，让随机采样的"噪声对"内积尽量小。$K$ 通常取 5-20。负采样把每步的计算复杂度从 $O(V)$ 降到 $O(K)$，这是 Word2Vec 能在大语料上训练的工程关键。

## 三、GloVe：全局统计 + 局部预测

Word2Vec 只用局部滑动窗口，忽略了**整个语料中的全局共现统计**。GloVe（Pennington et al., 2014）把两者结合：先用整个语料统计共现矩阵 $X$，其中 $X_{ij}$ 是词 $j$ 出现在词 $i$ 上下文中的次数，然后直接拟合**共现比值**。

GloVe 的核心观察是：词向量 $\mathbf{w}_i, \mathbf{w}_j$ 的差 $\mathbf{w}_i - \mathbf{w}_j$ 经过某个函数 $F$ 后，应该能反映 $P_{ik} / P_{jk}$（词 $i$、$j$ 与探测词 $k$ 的共现概率之比）。最终损失函数为：

$$
\mathcal{L} = \sum_{i,j=1}^{V} f(X_{ij}) \left( \mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2
$$

其中 $f(x)$ 是一个截断权重函数，避免高频词（如 "the"）主导训练：

$$
f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}
$$

典型超参 $\alpha=0.75, x_{\max}=100$。最终每个词的表示是 $\mathbf{w}_i$ 和 $\tilde{\mathbf{w}}_i$ 的平均或拼接。GloVe 在 2014-2018 年间是文本分类、相似度任务的标配词向量。

## 四、FastText、子词信息与 OOV

Word2Vec 和 GloVe 都把每个词当作"原子单位"。**FastText**（Bojanowski et al., 2017）则把每个词拆成 3-6 gram 字符 n-gram，向量是这些子词向量的和。例如 `where` 拆成 `<wh, whe, her, ere, re>`（含边界符号 `<` 和 `>`）。这样做有两大好处：

1. **OOV 友好**：训练时未见过的罕见词甚至新造词，也可以由子词向量拼出。
2. **形态学信息**：英语的 `-ing`、`-ed`，德语的复杂复合词，俄语的不同词缀，都能通过子词共享信号。

$$
\mathbf{v}_w = \sum_{g \in G_w} \mathbf{z}_g
$$

$G_w$ 是词 $w$ 的所有字符 n-gram 集合（含词本身）。FastText 在形态学丰富的语言（德语、芬兰语、土耳其语）上表现尤其突出。

## 五、词向量的经典"测试场"：语义类比

Mikolov 提出的词向量类比（word analogy）任务至今仍是快速 sanity check 的标准：

$$
\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}
$$

找到与 $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}}$ 余弦相似度最高的词 $w^*$（排除输入词），如果答案恰好是 `queen`，则该词向量"理解了"性别关系。GoogleNews 预训练的 Word2Vec 在这个测试上能达到约 60% 准确率。

## 六、PyTorch 实现：Skip-gram + 负采样

下面是一个最小但能跑通的 Skip-gram with Negative Sampling 实现（约 80 行），可直接 `python skipgram.py` 训练：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import random


class SkipGramNS(nn.Module):
    """Skip-gram with Negative Sampling。"""

    def __init__(self, vocab_size: int, embedding_dim: int = 100):
        super().__init__()
        # 输入嵌入 = 中心词向量；输出嵌入 = 上下文词向量
        self.input_emb = nn.Embedding(vocab_size, embedding_dim)
        self.output_emb = nn.Embedding(vocab_size, embedding_dim)
        # 两个 embedding 都用均匀分布初始化
        nn.init.uniform_(self.input_emb.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.output_emb.weight)

    def forward(self, center: torch.Tensor, context: torch.Tensor, neg: torch.Tensor):
        """
        center:  (B,)       中心词 id
        context: (B,)       真实上下文词 id
        neg:     (B, K)     负采样词 id
        """
        v_c = self.input_emb(center)              # (B, D)
        v_o = self.output_emb(context)            # (B, D)
        v_n = self.output_emb(neg)                # (B, K, D)

        pos_score = (v_c * v_o).sum(dim=-1)                      # (B,)
        neg_score = torch.bmm(v_n, v_c.unsqueeze(-1)).squeeze(-1) # (B, K)

        # 真实上下文要正分（sigmoid→1），负采样要负分（sigmoid→0）
        pos_loss = F.logsigmoid(pos_score)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)
        return -(pos_loss + neg_loss).mean()


def build_unigram_dist(counter: Counter, power: float = 0.75) -> torch.Tensor:
    """负采样按 unigram^0.75 采样，避免低频词完全采不到。"""
    vocab = list(counter.keys())
    counts = torch.tensor([counter[w] for w in vocab], dtype=torch.float)
    probs = counts.pow(power)
    probs /= probs.sum()
    return vocab, probs


# 烟测：构造小语料，跑一次 forward
if __name__ == "__main__":
    sentences = ["the cat sat on the mat".split(),
                 "the dog sat on the rug".split(),
                 "cat and dog are friends".split()]
    counter = Counter(w for s in sentences for w in s)
    word2id = {w: i for i, w in enumerate(counter.keys())}
    vocab, probs = build_unigram_dist(counter)
    model = SkipGramNS(vocab_size=len(vocab), embedding_dim=20)
    # 随机采样一组 (center, context, neg)
    center = torch.tensor([word2id["cat"]])
    context = torch.tensor([word2id["sat"]])
    neg = torch.multinomial(probs, num_samples=5, replacement=True).unsqueeze(0)
    print("loss:", model(center, context, neg).item())
```

代码里两个值得展开的细节：

- **双 Embedding**：`input_emb` 是"作为中心词时的向量"，`output_emb` 是"作为上下文时的向量"。两者分别学，推理时只用 `input_emb`；也可以推理时把两者平均，效果通常更好。
- **负采样的 0.75 次方**：原始论文发现 unigram 直接采样会让高频词（"the"）被反复选作负例、把低频词"挤出"训练。用 $P^{0.75}$ 抬高低频词的采样概率，能显著提升低频词质量。

## 七、静态词向量的局限

Word2Vec / GloVe / FastText 有一个共同的局限：**一个词只有一个向量**。这导致"bank"（河岸 / 银行）、"apple"（公司 / 水果）这类多义词无法在不同语境下区分。这一痛点直接催生了**上下文相关词向量**——ELMo（2018）、BERT（2018）、GPT（2018），我们将在下一篇中系统讨论。

此外，词向量还存在**各向异性（anisotropy）**：高频词聚集在原点附近，低频词被"推到"外圈，使得向量空间的方向并不均匀。这一现象在后续 Sentence-BERT、SimCSE 的对比学习中得到了专门处理。

## 小结

| 方法 | 路线 | 特点 | 代表场景 |
| --- | --- | --- | --- |
| One-Hot | 离散 | 稀疏正交、维度爆炸 | 仅作 baseline |
| SVD | 计数 | 全局统计、可解释 | 小语料教学 |
| Word2Vec | 预测 | 局部窗口、负采样高效 | 工业界标配 |
| GloVe | 计数 + 预测 | 共现比值、训练快 | 大语料预训练 |
| FastText | 子词 | OOV 友好、形态学 | 多语言/罕见词 |

静态词向量把 NLP 从"符号派"推进到了"分布式语义"时代，但一个词一义的固有缺陷，最终催生了 BERT 式的上下文编码——下一篇我们将看到 ELMo 和 BERT 如何让同一个词在不同句子里拥有不同的向量。
