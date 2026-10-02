# 序列标注基础：从 HMM 到 CRF 的概率图模型

序列标注（Sequence Labeling）是 NLP 最基础的任务之一——给句子中的每个 token 打上一个标签。典型应用包括中文分词（CWS）、词性标注（POS）、命名实体识别（NER）、槽位填充（Slot Filling）。本文从最朴素的 HMM 出发，经过 MEMM，最终到达序列标注的"老牌 SOTA"——条件随机场（CRF）。这三类模型构成了理解后续 BiLSTM-CRF、BERT-CRF 的理论基础。

## 一、序列标注任务的设定

给定输入序列 $\mathbf{x} = (x_1, x_2, \dots, x_n)$，输出对应的标签序列 $\mathbf{y} = (y_1, y_2, \dots, y_n)$。每个 $y_i$ 来自预定义的标签集合，例如：

- **BIO 标注**：B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, O
- **BIOES 标注**：B-/I-/E-/S-/O 五种，更精细但建模更复杂

模型的目标是学习 $P(\mathbf{y} \mid \mathbf{x})$，并对每个输入找最可能的标签序列：

$$
\mathbf{y}^* = \arg\max_{\mathbf{y}} P(\mathbf{y} \mid \mathbf{x})
$$

三类模型对这个概率做了不同的分解假设。

## 二、HMM：生成式模型

隐马尔可夫模型（HMM）是序列标注的经典生成式模型。它对联合概率做因式分解：

$$
P(\mathbf{x}, \mathbf{y}) = P(y_1) \prod_{i=2}^{n} P(y_i \mid y_{i-1}) \prod_{i=1}^{n} P(x_i \mid y_i)
$$

三项分别是：

- **初始概率** $P(y_1)$：句首标签的先验。
- **转移概率** $P(y_i \mid y_{i-1})$：上一个标签是 $y_{i-1}$ 时，当前标签是 $y_i$ 的概率。
- **发射概率** $P(x_i \mid y_i)$：标签 $y_i$ 产生字符 $x_i$ 的概率。

预测时用维特比算法（Viterbi）动态规划，时间复杂度 $O(n \cdot |Y|^2)$，其中 $|Y|$ 是标签数。

### 2.1 HMM 的优缺点

**优点**：模型小（参数只跟标签数有关，与词表大小解耦），训练快（Baum-Welch / 监督频次统计），可解释。

**缺点**：

1. **强独立性假设**：观测独立性 $P(x_i \mid y_i, x_{j \neq i}, y_{j \neq i}) = P(x_i \mid y_i)$ 假设当前字符只与当前标签有关，忽略上下文。
2. **生成式**：必须建模 $P(\mathbf{x})$，但我们只关心 $P(\mathbf{y} \mid \mathbf{x})$，浪费容量。
3. **特征工程受限**：只能使用"当前 token 是什么"这类离散特征，难以融入拼写、词形等复杂信号。

## 三、 MEMM：判别式的第一步

最大熵马尔可夫模型（MEMM, McCallum et al., 2000）把 HMM 的生成式改为判别式，直接建模：

$$
P(\mathbf{y} \mid \mathbf{x}) = \prod_{i=1}^{n} P(y_i \mid y_{i-1}, \mathbf{x}, i)
$$

每个位置的标签只依赖"前一个标签 + 整个输入 + 当前位置"。这带来两大优势：

1. **判别式**：不浪费容量在 $P(\mathbf{x})$ 上，数据利用率高。
2. **任意特征**：$P(y_i \mid y_{i-1}, \mathbf{x}, i)$ 可以用任意复杂特征函数（当前词的拼写、前后词的字符、词性等），不再受发射概率约束。

### 3.1 Label Bias Problem

MEMM 的致命缺陷是**标签偏置问题**（Label Bias Problem）。原因在于局部归一化——每个 $P(y_i \mid y_{i-1}, \mathbf{x}, i)$ 都归一化到 1，于是当某个状态的后继选择很少时（无论后续路径分数多低），它都会"贪婪地"选择分数高的转移。结果是模型倾向于走"低熵"路径，而不是全局最优。

直觉例子：假设 MEMM 在某个位置 $y_{i-1} = \text{B-PER}$ 下，无论后面是什么 $y_i$，分数都差不多（因为观测 $\mathbf{x}$ 没有提供足够信号）。MEMM 会倾向于转移概率高的标签，而忽略整句的全局合理性。

## 四、CRF：解决标签偏置

条件随机场（CRF, Lafferty et al., 2001）通过**全局归一化**解决了 MEMM 的问题：

$$
P(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \prod_{i=1}^{n} \psi_i(y_{i-1}, y_i, \mathbf{x}, i)
$$

其中：

- $\psi_i(y_{i-1}, y_i, \mathbf{x}, i)$ 是**势函数**（unnormalized score），可以任意设计。
- $Z(\mathbf{x}) = \sum_{\mathbf{y}'} \prod_i \psi_i(y'_{i-1}, y'_i, \mathbf{x}, i)$ 是**配分函数**（partition function），对所有可能的标签序列求和。

CRF 的关键设计：**对整个序列做归一化，而不是每个位置单独归一化**。这避免了标签偏置——因为某个位置的"优势"会被其他位置的"竞争"稀释。

### 4.1 线性链 CRF（Linear-chain CRF）

序列标注中最常用的形式是**线性链 CRF**——只考虑相邻标签的一阶依赖：

$$
\psi_i(y_{i-1}, y_i, \mathbf{x}, i) = \exp\!\left( \mathbf{W}^{\top} \mathbf{f}(y_{i-1}, y_i, \mathbf{x}, i) \right)
$$

其中 $\mathbf{f}(\cdot)$ 是特征函数向量，$\mathbf{W}$ 是权重。

实际工程中，我们通常把势函数拆成两部分：

$$
\psi_i(y_{i-1}, y_i, \mathbf{x}, i) = \exp\!\left( A_{y_{i-1}, y_i} + P_{i, y_i} \right)
$$

- $A \in \mathbb{R}^{|Y| \times |Y|}$ 是**转移矩阵**，可学习。
- $P_{i, y_i}$ 是**发射分数**，通常来自神经网络的输出（如 BiLSTM 或 BERT 的 token 表示）。

### 4.2 CRF 的训练与解码

**训练**：最大化对数似然：

$$
\mathcal{L} = \log P(\mathbf{y}^* \mid \mathbf{x}) = \sum_i (A_{y^*_{i-1}, y^*_i} + P_{i, y^*_i}) - \log Z(\mathbf{x})
$$

$Z(\mathbf{x})$ 通过**前向算法**（forward algorithm）动态规划计算，时间复杂度 $O(n \cdot |Y|^2)$。

**解码**：维特比算法找最优路径：

$$
\mathbf{y}^* = \arg\max_{\mathbf{y}} \sum_i (A_{y_{i-1}, y_i} + P_{i, y_i})
$$

CRF 训练和推理的时间复杂度都是 $O(n \cdot |Y|^2)$，对现代 NLP 任务完全够用。

## 五、CRF 相比 HMM/MEMM 的优势总结

| 模型 | 类型 | 归一化 | 特征 | 标签偏置 | 性能 |
| --- | --- | --- | --- | --- | --- |
| HMM | 生成式 | 局部 | 离散 | 严重 | 弱 |
| MEMM | 判别式 | 局部 | 任意 | 严重 | 中 |
| CRF | 判别式 | 全局 | 任意 | 无 | 强 |

CRF 的"全局归一化 + 任意势函数"组合，是它成为序列标注 SOTA 长达 10 年的根本原因。直到 BiLSTM-CRF（2015）和 BERT-CRF（2018）出现，这一格局才被打破——但 CRF 层至今仍是序列标注的标配。

## 六、PyTorch 实现：字符级中文分词的 CRF

下面是一个最小可用的 CRF 实现，可作为后续 BiLSTM-CRF 的基础模块：

```python
import torch
import torch.nn as nn


class CRF(nn.Module):
    """线性链 CRF。训练时计算 -log P(y*|x)；推理时维特比解码。"""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # 转移矩阵 A[i, j]：从 i 转移到 j 的分数
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        """
        emissions: (B, T, K)  来自 encoder 的发射分数
        tags:      (B, T)     真实标签 id
        mask:      (B, T)     1=有效，0=pad
        返回 -log P(tags|emissions)
        """
        B, T, K = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0, tags[:, 0]]
        for t in range(1, T):
            cur_emission = emissions[:, t, tags[:, t]]
            cur_transition = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (cur_emission + cur_transition) * mask[:, t]
        score = score + self.end_transitions[tags[torch.arange(B), mask.sum(dim=1) - 1]]
        partition = self._log_partition(emissions, mask)
        return -(score - partition).mean()

    def _log_partition(self, emissions, mask):
        """前向算法计算 log Z(x)。"""
        B, T, K = emissions.shape
        log_alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]   # (B, K)
        for t in range(1, T):
            # broadcast: (B, K, 1) + (K, K) + (B, 1, K)
            score = log_alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            log_alpha = torch.logsumexp(score, dim=1) * mask[:, t].unsqueeze(1) + \
                        log_alpha * (1 - mask[:, t]).unsqueeze(1)
        return torch.logsumexp(log_alpha + self.end_transitions.unsqueeze(0), dim=1)

    def decode(self, emissions, mask):
        """维特比解码，返回 (B,) 最优标签序列。"""
        B, T, K = emissions.shape
        log_delta = self.start_transitions.unsqueeze(0) + emissions[:, 0]    # (B, K)
        backpointers = []
        for t in range(1, T):
            score = log_delta.unsqueeze(2) + self.transitions.unsqueeze(0)   # (B, K, K)
            best_tag = score.argmax(dim=1)                                    # (B, K)
            backpointers.append(best_tag)
            log_delta = (score.gather(1, best_tag.unsqueeze(1)).squeeze(1) +
                         emissions[:, t]) * mask[:, t].unsqueeze(1) + \
                        log_delta * (1 - mask[:, t]).unsqueeze(1)
        # 回溯
        best_path = [log_delta.argmax(dim=-1)]
        for bp in reversed(backpointers):
            best_path.append(bp[torch.arange(B), best_path[-1]])
        best_path.reverse()
        return best_path
```

关键设计：

- **log-sum-exp 数值稳定性**：直接用 `torch.logsumexp` 避免上溢/下溢。
- **mask 处理**：pad 位置的转移分数乘 0，避免影响路径得分。
- **CRF 训练时间复杂度** $O(T \cdot K^2)$：对常见标签数（5-50）非常快。

## 七、CRF 的局限与替代

CRF 在序列标注上风光了 10 年，但有两个局限：

1. **依赖特征工程 / encoder 质量**：势函数只接受外部提供的"发射分数" $P_{i, y_i}$，本身不学表示。
2. **解码慢**：维特比 $O(T \cdot K^2)$ 在 $K$ 大（如 100+ 标签的多任务标注）时成为瓶颈。

现代做法几乎都是**encoder + CRF head**：encoder 用 BiLSTM 或 BERT 学发射分数，CRF 层负责标签转移约束。下一篇我们将看到 BiLSTM-CRF 如何在经典 NER 数据集 CoNLL-2003 上达到 91+ F1，超越纯 CRF 和纯 BiLSTM。

## 小结

| 模型 | 时间 | 核心假设 | 关键缺陷 |
| --- | --- | --- | --- |
| HMM | 1970s | 观测独立、生成式 | 强独立性 |
| MEMM | 2000 | 判别式、局部归一化 | 标签偏置 |
| CRF | 2001 | 判别式、全局归一化 | 需设计势函数 |

HMM、CRF 是 NLP 时代序列标注的"基本语法"——它们用概率图模型的语言，把"句子 → 标签"这件事形式化为可计算、可优化的数学问题。掌握这三类模型的概率分解与解码算法，不仅是面试高频考点，更是理解 BiLSTM-CRF、BERT-CRF 的必要前提。下一篇我们将看到深度学习如何与 CRF 结合，把 NER 推向新的高度。
