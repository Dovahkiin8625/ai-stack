# BiLSTM-CRF：深度学习时代的序列标注经典模型

2015 年前后，深度学习开始席卷 NLP。Huang 等人（2015）提出的 **BiLSTM-CRF** 是这个转折点的标志性工作——它把双向 LSTM 作为发射分数的"特征提取器"，再把 CRF 作为"标签依赖建模器"，在 CoNLL-2003 等经典 NER 数据集上首次把 F1 推到 90+，并成为此后 5 年工业界 NER 的事实标准。本文从架构细节、损失函数到 PyTorch 实现完整剖析 BiLSTM-CRF，并讨论它的优势与边界。

## 一、为什么需要 BiLSTM + CRF

HMM 时代我们只能用一个字符到一个标签的发射概率。深度学习让我们学到**任意特征的发射分数**，但朴素 BiLSTM + Softmax 有两个问题：

1. **标签独立性假设**：每个位置的标签预测独立于其他位置，忽略"B-PER 后面几乎肯定是 I-PER 或 O"这类强结构约束。
2. **曝光偏差**：训练时每步看到真实上一个标签，推理时却看到自己预测的——分布不匹配。

CRF 层的引入恰好解决这两个问题：
- 通过转移矩阵 $A$ 显式建模相邻标签依赖。
- 全局归一化让训练目标考虑整个序列。

## 二、BiLSTM-CRF 架构

模型分为三层：

```
输入字符序列 → 字符 Embedding → BiLSTM → 发射分数 → CRF → 标签序列
```

### 2.1 字符 Embedding 层

把每个 token 从 one-hot 映射为低维稠密向量：

$$
\mathbf{e}_i = \mathbf{E}[x_i] \in \mathbb{R}^{d_{\text{emb}}}
$$

实际工程中，常常**同时叠加词向量与字符级 CNN/BiLSTM**：

$$
\mathbf{e}_i = [\mathbf{E}_{\text{word}}(x_i); \mathbf{h}_{\text{char}}(x_i)]
$$

字符级编码能处理 OOV，对罕见词和中文尤其有用。

### 2.2 BiLSTM 编码层

双向 LSTM 分别从左到右和从右到左读取序列：

$$
\overrightarrow{\mathbf{h}}_i = \text{LSTM}_{\rightarrow}(\overrightarrow{\mathbf{h}}_{i-1}, \mathbf{e}_i)
$$
$$
\overleftarrow{\mathbf{h}}_i = \text{LSTM}_{\leftarrow}(\overleftarrow{\mathbf{h}}_{i+1}, \mathbf{e}_i)
$$

拼接得到双向表示：

$$
\mathbf{h}_i = [\overrightarrow{\mathbf{h}}_i; \overleftarrow{\mathbf{h}}_i] \in \mathbb{R}^{2d_{\text{hid}}}
$$

BiLSTM 的核心价值：**每个位置同时看到左右两侧的上下文**。例如 "Apple **Inc.** was founded in 1976"，看到 `Inc.` 就能联系到前面的 `Apple`，直接判断 `Inc.` 是组织名结尾。

### 2.3 发射分数层

把 BiLSTM 的输出投影到标签空间：

$$
P_{i, k} = \mathbf{W}_O^\top \mathbf{h}_i + b_k, \quad P \in \mathbb{R}^{n \times K}
$$

其中 $K$ 是标签数，$P_{i, k}$ 表示位置 $i$ 标签为 $k$ 的"未归一化分数"。

### 2.4 CRF 解码层

把 $P$ 作为发射分数，配上可学习的转移矩阵 $A \in \mathbb{R}^{K \times K}$：

$$
\text{score}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} \left( A_{y_{i-1}, y_i} + P_{i, y_i} \right)
$$

训练时最大化真实路径的对数似然（详见上篇），推理时维特比解码。

## 三、损失函数详解

BiLSTM-CRF 的训练目标：

$$
\mathcal{L} = -\sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}} \log P(\mathbf{y} \mid \mathbf{x})
$$

其中：

$$
\log P(\mathbf{y} \mid \mathbf{x}) = \text{score}(\mathbf{x}, \mathbf{y}) - \log \sum_{\mathbf{y}'} \exp \text{score}(\mathbf{x}, \mathbf{y}')
$$

第二项是配分函数的对数，由前向算法计算。直觉上：

- **真实路径的分数** $\text{score}(\mathbf{x}, \mathbf{y})$ 要尽量高。
- **所有路径的总分** $\log Z(\mathbf{x})$ 要尽量低（让真实路径独占鳌头）。

这两项的差就是负对数似然的负数。训练时反向传播同时更新 BiLSTM 参数（影响 $P$）和 CRF 参数（$A$）。

## 四、PyTorch 实现：完整 BiLSTM-CRF

下面是一个端到端可训练的 BiLSTM-CRF（已在上篇的基础上封装好），可直接 `python bilstm_crf.py` 跑通：

```python
import torch
import torch.nn as nn
from bilstm_crf_module import CRF  # 复用上篇的 CRF 实现


class BiLSTMCRF(nn.Module):
    """字符级 BiLSTM-CRF，可用于中文分词 / POS / NER。"""

    def __init__(self, vocab_size: int, num_tags: int,
                 embedding_dim: int = 100, hidden_dim: int = 200,
                 num_layers: int = 1, dropout: float = 0.5, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags)
        self.dropout = nn.Dropout(dropout)

    def _features(self, x, mask):
        emb = self.embedding(x)
        emb = self.dropout(emb)
        # 关键：pack_padded_sequence 让 LSTM 跳过 pad
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.hidden2tag(out)

    def forward(self, x, mask, tags=None):
        emissions = self._features(x, mask)
        if tags is not None:
            return self.crf(emissions, tags, mask)        # 训练：返回 loss
        return self.crf.decode(emissions, mask)           # 推理：返回标签序列


# 烟测
if __name__ == "__main__":
    vocab_size, num_tags = 1000, 5
    model = BiLSTMCRF(vocab_size, num_tags)
    x = torch.randint(1, vocab_size, (2, 10))
    mask = (x != 0).long()
    tags = torch.randint(0, num_tags, (2, 10))
    print("loss:", model(x, mask, tags).item())
    print("decode:", model(x, mask))                       # (B, T)
```

实现中几个易错的细节：

1. **pack_padded_sequence**：让 LSTM 跳过 pad 位置，否则 BiLSTM 会把 pad 当成正常 token 学习，反而降低质量。
2. **mask 双重作用**：既是 LSTM 的"长度"，也是 CRF 的"有效位"。
3. **CRF 的 `decode` 与 `forward` 共用一个发射分数**：训练和推理无缝衔接。

## 五、数据增强与训练技巧

### 5.1 标签方案选择

- **BIO**：最常用，对绝大多数任务够用。
- **BIOES**：B-/I-/E-/S 四种 + O，能表达"单位词实体"，更精细但参数更多。CoNLL 评测默认 BIOES。

### 5.2 Dropout 的位置

- **Embedding Dropout**：随机将整个 token 的 embedding 置 0。
- **LSTM 层间 Dropout**：多层 LSTM 之间。
- **输出 Dropout**：发射分数前。

典型配置：embedding dropout 0.5、LSTM 层间 0.3。

### 5.3 字符增强（Lexicon Feature）

中文 NER 常用——引入**词汇匹配特征**：扫描输入字符串与外部词典，对每个字符标记"是否在某词典的某个词中"。Gao et al.（2005）的 lexicon feature 至今仍是中文 NER baseline 的标配。

### 5.4 训练细节

- **学习率**：1e-3 起步 + Adam。
- **梯度裁剪**：`torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)`——LSTM 训练必备。
- **Early stopping**：在 dev 集上 F1 连续 5 个 epoch 不升就停。

## 六、CoNLL-2003 实验结果（历史对照）

| 模型 | CoNLL-2003 English F1 |
| --- | --- |
| HMM baseline | ~78 |
| CRF（手动特征） | ~84 |
| BiLSTM | ~85.5 |
| BiLSTM-CRF | **90.94** |
| BiLSTM-CNN-CRF | ~91.2 |
| BERT-base | ~92.8（无 CRF） |
| BERT-base + CRF | **~93.5** |

BiLSTM-CRF 在 2015-2017 年间是 NER 的事实标准，BERT 出现后又让 F1 涨了 2-3 个点。

## 七、BiLSTM-CRF 的局限与替代

### 7.1 长序列问题

LSTM 的"长程依赖"在超过 200 token 后明显衰减。CoNLL-2003 的句子平均只有 14 个 token，所以问题不严重；但处理法律文书、医学记录（上千 token）时，BiLSTM 不如 Transformer。

### 7.2 训练速度

BiLSTM 必须串行计算，训练一个 epoch 比同参数量 Transformer 慢 5-10 倍。工业界大规模数据上 BERT-base 比 BiLSTM-CRF 更受青睐。

### 7.3 替代方案

- **BERT-CRF**：用 BERT 替换 BiLSTM，发射分数质量更高，但模型更大。
- **Pointer Network**：把 NER 视为 span 预测而非序列标注，对嵌套实体更友好。
- **GlobalPointer / TPLinker**：用矩阵分类实现嵌套 NER。
- **GLOBALPOINTER-Nested**：苏剑林 2022 年的工作，把嵌套实体识别视为 $n \times n$ 的分类矩阵，在 CLUENER 等中文嵌套数据集上达到 SOTA。

## 八、BiLSTM-CRF 的精神遗产

即便 BERT 时代 NER 已不再首选 BiLSTM，BiLSTM-CRF 留下的方法论仍是后续工作的基石：

1. **Encoder-Decoder 分解**：encoder 学上下文表示，CRF 学标签依赖。这一思路被扩展到 BERT-CRF、RoFormer-CRF。
2. **显式结构建模**：用概率图模型把"领域知识"（标签转移合法性）注入深度学习。
3. **端到端训练**：发射分数和转移分数联合优化，避免 pipeline 误差传播。

## 小结

BiLSTM-CRF 是 NLP "深度学习时代" 的开山之作之一。它用 BiLSTM 学上下文、用 CRF 学标签依赖，两者的组合在 2015-2018 年间把 NER 等序列标注任务推到了前所未有的高度。即使在 BERT / GPT 时代，理解 BiLSTM-CRF 的设计哲学仍然是每个 NLP 工程师的必修课——它把"深度特征提取"与"显式结构建模"清晰地区分开来，这一原则在后续所有序列标注模型中都以不同形式延续。下一篇我们将看到 BERT-CRF 如何把发射分数从 BiLSTM 升级为 Transformer encoder。
