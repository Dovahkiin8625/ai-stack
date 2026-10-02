# BERT 时代序列标注：从 BERT-CRF 到 PL-Marker 的 span 表示

BiLSTM-CRF 在 2018 年前是序列标注的事实标准，但 BERT 的出现把整个范式推向了预训练 + 微调。本文沿着 BERT-CRF（2018）→ BERT-Softmax（2019）→ GlobalPointer / TPLinker（2020-2022）→ PL-Marker（2022）的时间线，剖析 Transformer 时代序列标注的最新进展，以及如何处理嵌套实体、多语言、文档级标注等开放问题。

## 一、BERT-CRF：用 Transformer encoder 替换 BiLSTM

最直接的升级——把 BiLSTM-CRF 中的 BiLSTM 换成 BERT，发射分数 $P_{i, y_i}$ 来自 BERT 的最后层向量：

$$
\mathbf{h}_i = \text{BERT}(\mathbf{x})_i, \quad P_{i, k} = \mathbf{W}_O^\top \mathbf{h}_i + b_k
$$

再套上 CRF 层即可。看似只是替换 backbone，效果却跃升 2-3 个 F1。

### 1.1 为什么 BERT 比 BiLSTM 强

- **更深的上下文建模**：12 层 Transformer，每一层都有全局 attention + FFN；BiLSTM 只有 2 层，且长距离依赖衰减严重。
- **预训练语料**：BERT 用 33 亿字符训练，BiLSTM-CRF 从零开始学——尤其在小数据 NER 上，预训练的收益巨大。
- **并行计算**：Transformer 一次 forward 算完所有位置；BiLSTM 必须串行，训练慢一个数量级。

### 1.2 实际效果

CoNLL-2003 English NER：

| 模型 | F1 |
| --- | --- |
| BiLSTM-CRF（Rei, 2017） | 90.94 |
| BERT-base + Softmax | 92.80 |
| BERT-base + CRF | **93.50** |
| BERT-large + CRF | 95.30 |

CoNLL-2003 中文（OntoNotes）：

| 模型 | F1 |
| --- | --- |
| BiLSTM-CRF + lexicon | 75.7 |
| BERT-base + CRF | 81.8 |
| Chinese-BERT-wwm-ext + CRF | **83.4** |

CRF 仍然有用——在 BERT 顶上加一层 CRF，平均涨 0.3-0.5 个 F1。

## 二、Softmax vs CRF：何时选哪个

加 CRF 不是"永远更好"。两种方案的本质区别：

| | Softmax | CRF |
| --- | --- | --- |
| 标签依赖 | 隐式（BERT 自己学） | 显式（转移矩阵） |
| 训练目标 | 逐 token 交叉熵 | 序列对数似然 |
| 解码速度 | 快（并行 argmax） | 慢（维特比） |
| 适用 | 标签转移约束弱 | 标签转移约束强 |

**经验法则**：

- 标签数少（5-20）、BIO 标注：用 CRF，能稳涨 0.3-0.5 F1。
- 标签数多（>50）或嵌套实体：用 Softmax 或更复杂的 span 模型。
- 推理延迟敏感：Softmax。

工程上有一个**免费午餐**：先训练 BERT-Softmax，看在 dev 上的错误类型——如果出现大量"非法转移"（如 I-PER 紧跟 B-LOC），加 CRF；如果错误主要是边界偏移，则不必加。

## 三、嵌套命名实体：超越 BIO

BIO 标注有一个根本限制：**一个 token 只能属于一个实体**。但现实语料中嵌套实体大量存在，例如：

> "清华大学计算机系" 中，"清华大学" 是机构，"清华" 也是机构（短名）。

BIO 标注只能给 `清` 一个标签，要么是 B-ORG 要么是 I-ORG，无法同时表达两个实体。这就是**嵌套 NER**（Nested NER）问题。

### 3.1 评测基准

- **ACE 2004/2005**：英文嵌套 NER 标准评测。
- **Genia**：生物医学嵌套实体。
- **CLUENER / Weibo**：中文嵌套场景。
- **DuIE**：百度中文关系抽取，含有嵌套结构。

### 3.2 嵌套 NER 的三类方案

#### (a) 区域分类（Region Classification）

把每个 span $(i, j)$（$i \leq j$）当作一个候选实体，预测它的类型。最朴素的实现：

$$
P(\text{type} = t \mid i, j) = \text{MLP}([\mathbf{h}_i; \mathbf{h}_j; \mathbf{h}_i - \mathbf{h}_j; \mathbf{h}_i \odot \mathbf{h}_j])
$$

复杂度 $O(n^2)$，但可以剪枝（只对长度 ≤ L 的 span 评估）。代表工作：Span-based 模型（Lee et al., 2017）。

#### (b) GlobalPointer

苏剑林 2022 年提出的 GlobalPointer 把嵌套 NER 视为**实体头-尾配对问题**。对每个类型 $t$，定义 $n \times n$ 的打分矩阵 $S_t$，其中 $S_t[i, j]$（$i \leq j$）表示类型 $t$ 的实体从 $i$ 到 $j$ 的分数。训练目标用**多标签圆环 Loss**：

$$
\mathcal{L}_t = \log \sum_{(i,j) \in P_t} e^{S_t[i,j]} + \log \sum_{(i,j) \in Q_t} e^{-S_t[i,j]}
$$

$P_t$ 是类型 $t$ 的真实实体集合，$Q_t$ 是同长度其他类型的所有 span。直觉：让真实实体的分数尽量高，其他所有候选的分数尽量低。

GlobalPointer 的优势：

1. **统一处理嵌套与非嵌套**：矩阵对角线以上全是候选，无须枚举类型。
2. **类别不互斥**：每个实体类型独立打分，可以同时给 (3, 5) 打 `ORG` 和 (3, 3) 打 `ORG-SHORT` 两个标签。
3. **推理时直接取 top-k 个分数最高的 span**，效率高。

实际效果：CLUENER 中文嵌套 NER F1 突破 70%，比 BERT-Softmax 高 5+ 个点。

#### (c) TPLinker / TPLinker-plus

Wang et al.（2020）提出 TPLinker（Token Pair Linking），把实体识别视为三类**token pair 关系**：

- **EH-ET**（Entity Head to Entity Tail）：实体的起止配对。
- **SH-OH**（Same Entity, Other Head to Head）：同一实体的不同 head。
- **ST-OT**（Same Entity, Tail to Other Tail）：同一实体的不同 tail。

训练时只对实体内/外的 token pair 打标签，解码时通过 EH-ET + SH-OH + ST-OT 关系还原出所有实体。优势：能识别**不连续实体**（如 "the [United States] of America"）。

TPLinker-plus 通过**相邻 token pair 的距离特征**进一步提升，ACE2005 上 F1 达到 87.7%。

#### (d) PL-Marker：方向感知的 span 表示

PL-Marker（Pseudo Label-aware Marker, Ye et al., 2022）是 2022 年的 SOTA。它通过**伪标记 marker** 显式注入"实体的方向信息"：

```
输入:  [CLS]  中国  驻  [实体开始]  美  国  [实体结束]  大  使 馆  [SEP]
```

在句首和句尾插入 marker token，让 BERT 知道"接下来/已经是一个实体"。推理时枚举所有可能的 marker 位置，分类 span 类型。

PL-Marker 在 ACE2004/2005、Genia 等嵌套 NER 上首次超越人类标注水平。

## 四、多语言与跨语言 NER

### 4.1 多语言预训练模型

- **mBERT**（multilingual BERT, 2018）：100 种语言，共享词表 + 共享 Transformer。在 XLM-R 出现前是事实标准。
- **XLM-RoBERTa**（Conneau et al., 2020）：用 CC-100（100 种语言 2.5TB 文本）训练，在 XTREME 评测上把多语言 NER / QA 推到新高度。
- **mDeBERTa-v3**：用 ELECTRA-style 预训练，多语言 SOTA。

### 4.2 跨语言迁移

跨语言 NER 的典型场景：英语有大量标注，目标语言（中文、西班牙语）几乎没有标注。两种主流方案：

- **零样本迁移**：直接用 mBERT 在英语 NER 上微调，推理时换中文，依赖词表共享 + 跨语言对齐。
- **翻译增强**：把目标语言翻译成英语，用英语 NER 模型预测，再把答案对齐回去。

经验上：

- 词表重叠度高的语言对（中 ↔ 英、日 ↔ 韩）零样本效果好。
- 词法差异大的语言对（英 ↔ 阿、中 ↔ 越）需要翻译增强或少量目标语言标注。

## 五、文档级与对话级 NER

经典 NER 是句子级的，但实际业务中常常需要处理整篇文档：

- **文档级 NER**：同一实体在文档不同位置出现，要正确链接（coreference）。SciBERT-DOC 是代表。
- **对话级 NER**：客服对话中，"那台 iPhone" 里的 "iPhone" 是产品实体，指代上文某款手机。对话状态跟踪（DST）数据集（如 MultiWOZ）的槽位识别即此类问题。

文档级 NER 的常见做法：

1. 把整文档拼接输入 BERT，加段嵌入区分句子。
2. 在最后层用**全局 attention** 链接同一实体的不同出现。

## 六、PyTorch：BERT-CRF 微调

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from bilstm_crf_module import CRF  # 复用上一篇的 CRF


class BertCRF(nn.Module):
    """BERT + CRF：用 BERT 的 token 表示作为发射分数。"""

    def __init__(self, model_name: str, num_tags: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, input_ids, attention_mask, token_type_ids=None, tags=None):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        emissions = self.classifier(self.dropout(out.last_hidden_state))
        if tags is not None:
            return self.crf(emissions, tags, attention_mask)   # 训练
        return self.crf.decode(emissions, attention_mask)     # 推理


# 训练循环（伪代码）
from transformers import AdamW, get_linear_schedule_with_warmup
model = BertCRF("bert-base-chinese", num_tags=11)  # BIOES + O = 11
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

for epoch in range(3):
    for batch in train_loader:
        loss = model(**batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

训练要点：

- **学习率**：BERT 微调一般 2e-5 - 5e-5，比 BiLSTM-CRF 的 1e-3 小两个数量级。
- **梯度裁剪**：1.0 是 BERT 训练的事实标准——不加容易训飞。
- **Epoch**：3-5 个 epoch 足够，再多会过拟合（BERT 容量大）。
- **warmup**：前 10% 步数线性 warmup，让 Adam 稳定。

## 七、低资源与主动学习

实际项目中，标注 NER 数据昂贵。以下是几个常见策略：

### 7.1 半监督学习

- **Noisy Student**：先用少量标注数据训一个 teacher，预测大量未标注数据，把高置信度样本加入训练，迭代训练 student。
- **Self-Training + Curriculum**：从易到难逐步扩大训练集。

### 7.2 主动学习

让模型选择"最有价值"的样本请求人工标注：

- **不确定性采样**：选模型预测置信度最低的样本。
- **多样性采样**：选与已有标注最不相似的样本。
- **委员会查询**：训练多个模型，选它们分歧最大的样本。

### 7.3 提示学习（Prompt-based NER）

用生成式 LLM 做 NER——把任务形式化为"填空"：

```
北京是[ORG:城市/国家]之一。
```

模型生成的位置 + 类型就是实体。GPT-3.5 / ChatGPT 在标准 NER 上 F1 大约 80-85%（不及微调 BERT），但在低资源场景下优势明显——可以靠提示词灵活加入新类型。

## 八、评估与常见陷阱

### 8.1 Span-level 评估

NER 必须按 **span** 评估，而非 token-level——边界正确才算对：

- 微平均 F1（Micro-F1）：把所有句子拼接计算全局 F1。
- 严格匹配：实体的起止 + 类型完全一致才算对。
- 松弛匹配：起止一致、类型不一致也算 0.5。

### 8.2 常见错误类型

| 错误类型 | 占比 | 改进方法 |
| --- | --- | --- |
| 边界偏移 | ~30% | 加 CRF / span-based 模型 |
| 类型混淆 | ~25% | 多任务学习 + entity linking |
| 嵌套漏检 | ~20% | GlobalPointer / TPLinker |
| OOV | ~15% | 字符增强 / 字符级模型 |
| 嵌套过检 | ~10% | 后处理约束 |

## 小结

序列标注经历了 **HMM → CRF → BiLSTM-CRF → BERT-CRF → Span-based / PL-Marker** 的清晰演进。当前（2026）的 SOTA 实践：

| 场景 | 推荐模型 |
| --- | --- |
| 通用 NER，数据量充足 | BERT-CRF 或 RoBERTa-CRF |
| 嵌套 NER | GlobalPointer / PL-Marker |
| 文档级 / 多语言 | mDeBERTa / XLM-R |
| 长文本（>512 token） | Longformer + CRF |
| 低资源 | 提示学习 / 半监督 |
| 工业部署（延迟敏感） | DistilBERT-CRF |

序列标注看似"基础任务"，但它把 NLP 的核心挑战——**结构化预测、上下文建模、长程依赖**——压缩在一个看似简单的框架里。掌握 BiLSTM-CRF 的概率图直觉和 BERT-CRF 的端到端微调，是每位 NLP 工程师的必修课。下一篇我们将进入**信息抽取**领域：从单个 token 的标签，扩展到实体、关系、事件的联合抽取。
