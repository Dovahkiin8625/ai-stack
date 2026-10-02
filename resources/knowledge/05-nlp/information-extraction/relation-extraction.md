# 关系抽取：从模板匹配到 LLM 的实体关系建模

如果说命名实体识别（NER）是从文本中识别"实体是什么"，那么**关系抽取**（Relation Extraction, RE）就是回答"实体之间有什么关系"。给定句子"马云于 1999 年在杭州创办了阿里巴巴"，NER 找出"马云"（PER）和"阿里巴巴"（ORG），而 RE 要进一步识别"创始人"关系。关系抽取是构建知识图谱、自动问答、智能搜索的核心组件。本文沿"模板 → 监督学习 → 远程监督 → 预训练 → LLM"的时间线，剖析 RE 的核心方法与最新进展。

## 一、任务定义与分类

### 1.1 任务形式

给定句子 $\mathbf{x}$ 和句子中的一对实体 $(e_1, e_2)$（位置已由 NER 给出），预测它们之间的关系 $r$：

$$
r^* = \arg\max_{r \in \mathcal{R} \cup \{\text{None}\}} P(r \mid \mathbf{x}, e_1, e_2)
$$

$\mathcal{R}$ 是预定义关系集，None 表示无关系。

### 1.2 三种任务设定

| 设定 | 输入 | 输出 | 评测 |
| --- | --- | --- | --- |
| 句子级 RE | 单句 + 一对实体 | 关系标签 | TACRED, SemEval-2010 Task 8 |
| 文档级 RE | 整篇文档 + 多对实体 | 关系标签（核心指代、跨句推理） | DocRED, HacRED |
| 开放域 RE | 单句 / 文档 | 任意关系（自由文本） | FewRel, OIE 基准 |

### 1.3 关系分类体系

- **FewRel**（Han et al., 2018）：100 类关系，每类 700 个实例，用于少样本 RE。
- **TACRED**（Zhang et al., 2017）：42 类关系 + None，106K 句子。
- **NYT-10**（Riedel et al., 2010）：远程监督生成的关系分类数据，53 类。
- **DocRED**（Yao et al., 2019）：96 类关系，跨句推理。
- **百度 DuIE**：中文关系抽取，49 类 + 嵌套。

## 二、基于模板与规则

### 2.1 模板匹配

最早期的方法是手动定义关系模板：

```
[PER] 出生于 [LOC]    → born_in(PER, LOC)
[PER] 任职于 [ORG]    → works_for(PER, ORG)
[PER] 与 [PER] 结婚   → spouse(PER, PER)
```

优点：精确率高、可解释。缺点：召回低、跨语言迁移差、维护成本高。

### 2.2 Hearst 模式与 bootstrapping

Hearst（1992）提出用**模式学习**自动扩展：

1. 从少量种子关系对开始（如 "Barack Obama - Hawaii" → born_in）。
2. 找到包含这些对的句子，抽取上下文模式。
3. 用模式在语料中找更多候选对。
4. 迭代直至收敛。

代表系统：**DIPRE**（Brin, 1999）、**Snowball**（Agichtein & Gravano, 2000）。优点：自动扩展关系库。缺点：语义漂移（semantic drift）、难以处理长距离依赖。

## 三、监督学习时代：特征工程与神经网络

### 3.1 经典特征工程

传统 RE 把问题转为**句子分类**——对句子 $\mathbf{x}$ 和实体对 $(e_1, e_2)$ 设计特征：

- **词汇特征**：实体类型、实体之间的词、实体的依存路径。
- **句法特征**：POS、依存句法、句法树上的最短路径。
- **实体特征**：NER 类型、Wikipedia 链接、实体对的共现统计。

代表：**Mintz**（2009）、**Rink & Harabagiu**（2010）、**Surdeanu et al.**（2012）的多实例学习。

### 3.2 CNN / RNN 时代

深度学习让 RE 不再依赖手工特征：

- **CNN-RE**（Zeng et al., 2014）：对句子做卷积 + max pooling + softmax 分类。
- **BiLSTM-RE**（Zhang et al., 2015）：用 BiLSTM 编码句子，实体位置特征拼接。
- **Attention-RE**（Zhou et al., 2016）：在 BiLSTM 顶上加 attention 突出实体对相关 token。

代表模型：

- **PCNN（Piecewise CNN, Zeng et al., 2015）**：把句子按实体对分成三段（左实体 / 中间 / 右实体），分别 max-pool，捕获局部信息。
- **Attention-based BiLSTM**：在 attention 权重上学"实体对的语义相关片段"。

### 3.3 实体位置编码

RE 的一个关键设计：**如何让模型知道"实体在哪里"**。常用方法：

- **Position Embedding**：每个 token 相对实体 1/2 的距离，作为额外 embedding 加到 token embedding 上。
- **Entity Marker**：在实体前后插入特殊 token `[E1] ... [/E1] [E2] ... [/E2]`，让 BERT 自己学位置。
- **Typed Marker**：实体的 NER 类型也作为 marker，让模型先知道类型再判断关系。

## 四、BERT-RE：用 Transformer 编码关系

BERT 的出现让 RE 性能大幅跃升。典型方案：

### 4.1 实体标记 + 句对分类

```python
# 输入构造
"[CLS] [E1] 马斯克 [/E1] 访问了 [E2] 中国 [/E2] [SEP]"
```

模型对 `[CLS]` 向量做分类，预测实体 1 和实体 2 之间的关系。

效果（TACRED）：

| 模型 | F1 |
| --- | --- |
| BiLSTM + 位置特征 | 65.7 |
| BERT-base | 67.2 |
| BERT-base + entity marker | **70.1** |
| BERT-large + marker | 71.5 |

### 4.2 实体对向量拼接

另一种做法是同时用 `[CLS]` 与实体首尾向量拼接：

$$
\mathbf{h}_{\text{rep}} = [\mathbf{h}_{\text{[CLS]}}; \mathbf{h}_{e_1^s}; \mathbf{h}_{e_1^e}; \mathbf{h}_{e_2^s}; \mathbf{h}_{e_2^e}]
$$

这种"多向量融合"在 ALBERT、RoBERTa 上常带来 1-2 个 F1 提升。

### 4.3 关系分类专用预训练

- **ERNIE**（Baidu, 2019）：百度提出，把知识图谱中的实体关系作为预训练任务。
- **KEPLER**（Wang et al., 2021）：把知识图谱嵌入与 MLM 联合训练。
- **K-BERT**（Liu et al., 2020）：在输入中直接注入知识三元组。

这些方法在 TACRED、FewRel 等专业关系分类上有 2-3 个 F1 提升。

## 五、远程监督：解决标注数据稀缺

### 5.1 基本思路

人工标注关系数据昂贵。**Mintz et al.（2009）** 提出远程监督（distant supervision）：

> **核心假设**：如果知识图谱中存在关系 $r(e_1, e_2)$，那么任何同时包含 $e_1$ 和 $e_2$ 的句子都表达关系 $r$。

举例：知识图谱有 `born_in(Barack Obama, Hawaii)`，那么所有同时包含 "Barack Obama" 和 "Hawaii" 的句子都被假设为 `born_in` 关系的训练样本。

### 5.2 多实例学习

远程监督的核心问题是**噪声**——同一对实体可能在不同句子中表达不同关系：

- "Obama 在 Hawaii 出生" → born_in
- "Obama 访问了 Hawaii" → visit
- "Obama 谈论了 Hawaii 旅游业" → None

如果不处理，模型会把"Obama 谈论..."学成 `born_in`。**多实例学习**（Multi-Instance Learning, MIL）把每个"实体对-句子袋"作为一个 bag，bag 至少包含一个正确的标签：

$$
\mathcal{L} = -\log P(r \mid \text{bag}) = -\log \frac{\exp(g(e_1, e_2, r))}{\sum_{r'} \exp(g(e_1, e_2, r'))}
$$

其中 $g$ 是某种聚合（如 max、attention、mean）：

$$
g(e_1, e_2, r) = \max_{x_i \in \text{bag}} \text{NN}(x_i, e_1, e_2)_r
$$

### 5.3 句子级注意力

Lin et al.（2016）提出**句子级 attention**，给每个句子不同权重：

$$
g(e_1, e_2, r) = \sum_{i} \alpha_i \cdot \text{NN}(x_i, e_1, e_2)_r
$$
$$
\alpha_i = \frac{\exp(e_i^\top \mathbf{a})}{\sum_j \exp(e_j^\top \mathbf{a})}
$$

直觉：模型自动学习"哪一句最可能表达真实关系"，给高分句更高权重。

### 5.4 远程监督的现代实践

- **预训练 + 远程微调**：先在大规模远程监督数据（如 NYT-10）上预训练，再在人工标注的小数据集上微调。
- **强化学习去噪**：用 RL 选择"高置信句子"作为训练集。
- **生成式伪标签**：用 LLM 对远程监督的句子重新标注，去除噪声。

## 六、文档级关系抽取

### 6.1 任务定义

文档级 RE 处理跨句推理：

> "马云于 1964 年出生在杭州。他在 1999 年创办了阿里巴巴。"

单看第一句，`马云 - 杭州` 是 `born_in`。但要识别 `马云 - 阿里巴巴` 的 `founder_of` 关系，必须**结合跨句信息**。

### 6.2 DocRED 数据集

Yao et al.（2019）提出 DocRED：5056 篇维基百科文章，96 类关系，~150K 个关系实例。约 40% 的关系需要跨句推理。

### 6.3 代表模型

- **BERT-doc**：把整篇文档（截断到 512 token）输入 BERT，文档图卷积（DGCNN）做关系分类。
- **CorefBERT**：先做 coreference resolution，把文档中的代词链接到实体名。
- **ATLOP**（Zhou et al., 2021）：自适应阈值与局部上下文池化，DocRED dev F1 突破 63。
- **KD-DocRE**（Ma et al., 2022）：把文档结构（章节、句子位置）作为先验知识注入。

文档级 RE 在 2024-2026 年间快速进步，主要得益于图神经网络与 LLM 的结合。

## 七、少样本与零样本关系抽取

### 7.1 FewRel

FewRel（Han et al., 2018）是少样本 RE 的标准基准：100 类关系，每类仅 16 个有标注样本。代表方法：

- **原型网络**：对每类关系，计算支持集样本向量的均值作为"原型"，查询样本与原型的距离决定分类。
- **GNN-based**：把支持集 + 查询集建成图，用 GNN 聚合。
- **Prompt-based**：把 RE 视为"完形填空"，用 GPT-3 风格的提示。

### 7.2 零样本 RE

零样本 RE 用关系名称的自然语言描述作为输入：

```python
prompt = """判断以下句子中两个实体的关系。

句子：{}年，特斯拉 CEO 马斯克访问中国。
实体 1：马斯克（PER）
实体 2：中国（LOC）

可选关系：
- born_in: 出生地
- visit: 访问
- work_for: 任职
- ceo_of: CEO 与公司
- located_in: 位于

关系："""
```

GPT-3.5 在零样本 RE 上 F1 约 50-60%，Claude 3 Opus 约 70%，仍不及微调监督模型。但 LLM 的优势是**支持任意新关系**——只要在提示词里写好名字和描述。

## 八、开放信息抽取（OIE）

OIE 不依赖预定义关系集，而是直接从句子中抽取 $(arg1, rel, arg2)$ 三元组：

- "Obama was born in Hawaii" → (Obama; was born in; Hawaii)
- "Apple acquired Beats in 2014" → (Apple; acquired; Beats); (Beats; was acquired in; 2014)

代表系统：**ReVerb**（Fader et al., 2011）、**OpenIE 5**（Saha & Mausam, 2018）、**MinIE**（Gashteovski et al., 2017）。

OIE 在工业界主要用于：
- 自动构建知识图谱候选三元组。
- 与远程监督结合，覆盖长尾关系。
- 为下游 QA 系统提供"事实库"。

LLM 时代 OIE 范式复兴：用 ChatGPT 直接生成结构化三元组，质量大幅超过传统 OIE 系统。

## 九、PyTorch 实现：BERT 句子级 RE

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BertRE(nn.Module):
    """BERT 句子级关系抽取。"""

    def __init__(self, model_name: str, num_relations: int, marker_dim: int = 768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        config = self.bert.config
        # 可学习的 entity marker
        self.e1_start = nn.Parameter(torch.randn(marker_dim) * 0.02)
        self.e1_end = nn.Parameter(torch.randn(marker_dim) * 0.02)
        self.e2_start = nn.Parameter(torch.randn(marker_dim) * 0.02)
        self.e2_end = nn.Parameter(torch.randn(marker_dim) * 0.02)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_relations)

    def forward(self, input_ids, attention_mask, e1_start_pos, e1_end_pos, e2_start_pos, e2_end_pos):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B, T, D)

        # 注入 entity marker
        B = h.size(0)
        h = h.clone()
        h[torch.arange(B), e1_start_pos] += self.e1_start
        h[torch.arange(B), e1_end_pos] += self.e1_end
        h[torch.arange(B), e2_start_pos] += self.e2_start
        h[torch.arange(B), e2_end_pos] += self.e2_end

        # 用 [CLS] 做关系分类
        cls = self.dropout(h[:, 0])
        return self.classifier(cls)
```

训练时对实体位置做 BIO 标注，再用 `tokenizer` 的 offset mapping 把标签对齐到 subword。

## 十、工程实践与挑战

### 10.1 NER + RE pipeline vs 联合抽取

- **Pipeline**：先 NER 后 RE。实现简单，但误差传播（NER 错误会污染 RE）。
- **联合抽取**：同时识别实体与关系。代表方案：
  - **NovelTagging**（Zheng et al., 2017）：把关系编码到实体标签中（如 `B-PER-∧works_for`），用单一序列标注器。
  - **Table Filling**：把关系建模成 $n \times n$ 表格，每个单元格是关系类型。
  - **CasRel**（Wei et al., 2020）：先识别主体，再对每个主体识别所有可能的关系和客体。
  - **TPLinker**（见 [[transformer-tagging]]）：把实体对关系视为 token pair 关系，统一处理嵌套与重叠。

联合抽取在重叠三元组场景（如一个实体参与多个关系）下显著优于 pipeline。

### 10.2 评估与错误分析

RE 的常见错误：

| 错误类型 | 占比 | 改进方法 |
| --- | --- | --- |
| 关系类型混淆 | ~30% | 加入关系描述 / 更多训练数据 |
| 实体边界错 | ~25% | 用更强的 NER 模型 |
| 长距离指代 | ~20% | 文档级 RE |
| 噪声数据 | ~15% | 远程监督去噪 |
| 复杂句法 | ~10% | 句法特征、Tree-LSTM |

### 10.3 工业部署

- **服务化**：BERT-base RE 在 A100 上 ~5000 句/秒。
- **缓存**：相同实体对的结果可缓存（如 `Apple-创始人` 在多篇文章中常出现）。
- **多任务**：NER + RE + EL（Entity Linking）联合部署，共享 BERT 编码。

## 十一、未来方向

1. **LLM 主导 RE**：提示学习逐步取代微调监督——LLM 在少样本/零样本 RE 上优势明显。
2. **多模态 RE**：从图像 + 文本联合抽取关系（如从新闻图片和图说中识别"人物合影"）。
3. **因果关系抽取**：识别"X 导致 Y"、"X 阻碍 Y"等因果关系，是医疗、经济学的核心需求。
4. **图谱补全**：把 RE 与知识图谱嵌入结合，自动发现缺失关系。

## 小结

| 时代 | 方法 | 优势 | 局限 |
| --- | --- | --- | --- |
| 1990s | 模板匹配 | 高精确率 | 低召回、维护贵 |
| 2000s | Bootstrapping | 自动扩展 | 语义漂移 |
| 2010-2014 | 特征 + SVM/MIL | 监督学习 | 特征工程 |
| 2014-2018 | CNN/BiLSTM + Attention | 端到端 | 长距离、文档级弱 |
| 2018-2024 | BERT-RE / 远程监督 | 预训练 + 微调 | 标注数据仍贵 |
| 2024+ | LLM 提示 / 联合抽取 | 零样本新关系 | 推理慢、定位难 |

关系抽取是从"识别实体"到"理解世界"的飞跃——它把孤立的实体连接成语义网络，让机器开始理解"谁做了什么、对谁、在何时何地"。下一篇我们将进入**事件抽取**：识别更复杂的动作与状态变化，从静态知识跃迁到动态事件。
