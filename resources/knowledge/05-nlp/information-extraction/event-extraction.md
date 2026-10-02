# 事件抽取：从 ACE 到 Doc2Dial 的复杂结构预测

如果说命名实体识别（NER）告诉我们"文本中有谁、有哪些地方"，关系抽取（RE）告诉我们"他们之间有什么关系"，那么**事件抽取**（Event Extraction, EE）则更进一步——识别"文本中发生了什么事件、涉及哪些参与者、发生在何时何地"。事件抽取是构建动态知识图谱、舆情监控、智能投研、临床决策支持等应用的核心组件，也是 NLP 任务中结构最复杂、难度最高的之一。本文从经典 ACE 评测出发，深入剖析事件抽取的核心方法与最新进展。

## 一、任务定义

### 1.1 事件的基本组成

事件抽取通常包含 4 个子任务：

1. **事件检测（Trigger Identification）**：识别事件触发词（trigger）。例如"收购"、"出生"、"地震"通常是触发词。
2. **事件分类（Event Type Classification）**：判断事件属于哪个类型（如"商业收购"、"自然灾害"）。
3. **论元抽取（Argument Extraction）**：识别事件中的参与者、时间和地点等论元。
4. **论元角色分类（Argument Role Classification）**：判断每个论元扮演什么角色（如"收购方"、"被收购方"、"金额"）。

举例：

> "2024 年 4 月 1 日，特斯拉 CEO 马斯克宣布以 560 亿美元收购推特。"

抽取结果：

```json
{
  "event_type": "Business.MergeAcquisition",
  "trigger": "收购",
  "arguments": [
    {"entity": "特斯拉", "role": "Acquirer"},
    {"entity": "马斯克", "role": "Acquirer.Representative"},
    {"entity": "推特", "role": "Target"},
    {"entity": "560 亿美元", "role": "Price"},
    {"entity": "2024 年 4 月 1 日", "role": "Time"}
  ]
}
```

### 1.2 评测基准

| 基准 | 语言 | 类型数 | 论元角色数 | 特点 |
| --- | --- | --- | --- | --- |
| ACE 2005 | 英 | 33 | 35 | 经典标准评测 |
| TAC KBP | 英、中 | 38 | 18 | 跨语言、跨域 |
| MAVEN | 英 | 168 | - | 大规模事件检测 |
| FewEvent | 英 | 100 | - | 少样本事件检测 |
| DuEE 1.0/2.0 | 中 | 65/119 | - | 中文事件抽取 |
| Doc2Dial | 中 | 11 | - | 对话场景事件抽取 |

### 1.3 三大挑战

1. **多事件**：一段文本可能同时包含多个事件，且事件之间可能存在因果、共指、时间序列等复杂关系。
2. **跨句与隐含事件**：有时事件论元分布在不同句子（如代词指代、省略主语）。
3. **触发词歧义**：同一个词在不同语境下触发不同类型事件。"苹果公司发布了新产品"中"发布"是产品发布；"央行发布了新的利率政策"中"发布"是政策宣布。

## 二、经典方法：基于特征与模式

### 2.1 ACE 2005 的范式

ACE 2005 提出 pipeline 式事件抽取：

```
输入句子 → 触发词识别 → 触发词分类 → 论元识别 → 论元角色分类
```

每个子任务被视为分类问题：

- **触发词识别**：每个 token 二分类（是触发词 / 不是）。
- **触发词分类**：多分类（33 种 ACE 事件类型）。
- **论元识别**：每个实体二分类（是该事件的论元 / 不是）。
- **论元角色分类**：多分类（35 种论元角色）。

### 2.2 特征工程

经典系统（如 UIUC 的联合模型）依赖大量人工特征：

- **词汇特征**：词形、词性、lemma、形态。
- **句法特征**：依存路径、短语结构树。
- **实体特征**：NER 类型、Wikipedia 链接。
- **触发词-论元距离**：token 距离、依存树距离、句子距离。

特征工程的天花板：F1 在 ACE 2005 上约 55-58%。

### 2.3 联合模型

为了减少 pipeline 误差传播，研究者提出**联合模型**：

- **联合概率模型**：把触发词与论元联合建模（如结构化 SVM、ILP 约束）。
- **依存树结构**：用依存句法把触发词与论元的关系建模成树结构。

代表工作：

- **Li et al.（2013）**：用结构化感知机联合抽取触发词与论元。
- **Nguyen & Grishman（2014）**：用 RNN + skip features 联合事件抽取。

## 三、深度学习时代

### 3.1 DMCNN（动态多池化 CNN）

Chen et al.（2015）提出 **DMCNN**，首次用 CNN + dynamic multi-pooling 做事件抽取：

1. 用 CNN 编码句子，得到 token 级表示。
2. **动态多池化**：根据触发词位置，把句子分成 2 段（左 / 右）分别 max-pool。
3. 触发词分类与论元分类分别预测。

dynamic pooling 的关键：标准 max-pool 只保留一个最强特征，而事件信息分散在触发词两侧（如"X 收购 Y"），分两段 pool 能保留两侧的关键证据。

### 3.2 JRNN（Joint RNN）

Nguyen et al.（2016）用 Bi-RNN 联合建模触发词与论元：

```
h_i = BiRNN(emb_i, h_{i-1}, h_{i+1})
trigger_label_i = softmax(W_t h_i + b_t)
argument_label_{i,j} = softmax(W_a [h_i; h_j] + b_a)
```

优势：

- 共享底层表示，触发词与论元互相影响。
- BiRNN 编码左右上下文，缓解长距离问题。

### 3.3 框架语义与图模型

- **Graph LSTM**：把依存句法树转成图，事件论元沿最短路径传播信息。
- **HMEAE**（Hierarchical Modular Event Argument Extraction, Wang et al., 2019）：模块化神经网络，分别建模"句子编码 → 论元识别 → 角色分类"。

## 四、BERT 时代的事件抽取

### 4.1 BERT-EE

把事件抽取的每个子任务套在 BERT 上：

1. **触发词分类**：每个 token 的 BERT 输出过 softmax 分类 trigger。
2. **论元角色分类**：用实体起止的 BERT 输出拼接 $\mathbf{h}_{e}$，分类 role。

效果（ACE 2005）：

| 模型 | Trigger F1 | Argument F1 |
| --- | --- | --- |
| DMCNN | 67.7 | 52.1 |
| JRNN | 73.9 | 55.4 |
| BERT-base | 72.3 | 54.7 |
| BERT + 结构 | **75.7** | **58.9** |

BERT 的上下文表示确实更强，但需要配合**结构化约束**才能达到 SOTA。

### 4.2 联合抽取模型

#### (a) CasEE（Cascade Encoding, Sheng et al., 2021）

把事件抽取视为级联：

```
输入 → 触发词识别 → 触发词分类 → (对每个 trigger) 论元识别 → 论元分类
```

每一步的输出影响下一步的编码。例如已知触发词是"收购"，则论元分类 head 应该更关注"金额"、"收购方"、"被收购方"等 role。

#### (b) OneIE（Lin et al., 2020）

把事件抽取视为**全局联合图**：

- 每个节点是 token 或实体。
- 边表示 trigger-argument 关系。
- 用 ILP 约束保证全局一致性（如"同一 token 不能同时是 trigger 和非 trigger"）。

OneIE 在 ACE 2005 中英双语上首次 trigger F1 突破 72，argument F1 突破 58。

#### (c) GAIL（Generative Adversarial Imitation Learning）

用 GAN 的方式做事件抽取——生成器预测事件结构，判别器判断预测是否符合真实分布。

### 4.3 基于生成的 EE

把事件抽取视为**结构化生成**：

```python
prompt = """从以下文本中抽取事件，返回 JSON。

文本：2024 年 4 月 1 日，特斯拉 CEO 马马斯克宣布以 560 亿美元收购推特。

事件类型：Business.MergeAcquisition

JSON："""
```

LLM 直接生成结构化 JSON。GPT-4 / Claude 3 Opus 在 ACE 2005 上：

- Trigger 识别 F1：~80
- Argument 抽取 F1：~65

已超过微调 BERT 模型，但仍受限于**位置准确性**与**幻觉问题**。

## 五、文档级事件抽取

### 5.1 任务定义

文档级 EE 处理跨句、跨段的事件：

> "4 月 1 日，特斯拉宣布收购推特。"
> "5 月 15 日，这笔交易获得欧盟批准。"
> "6 月 10 日，欧盟反垄断部门发布调查报告。"

整篇文档包含**多个相关事件**，需要：

1. 跨句论元链接（"这笔交易" 指代上一句的收购）。
2. 事件共指（多个句子表达同一事件）。
3. 事件时序与因果（先收购，后批准）。

### 5.2 ChFinAnn / Doc2Event 数据集

- **ChFinAnn**：中文金融公告，包含公司公告中的多个事件。
- **Doc2Event**（Zheng et al., 2021）：英文维基百科 + 金融新闻，文档级事件抽取。
- **MAVEN-ERE**：大规模事件关系抽取。

### 5.3 代表方法

- **Doc2Event**（2021）：用异构图（节点 = token/实体，边 = 句法/共指/邻接），用 GNN 聚合信息。
- **EA2E**（2022）：用预训练 + 文档结构 + 事件模板。
- **RATE**（2023）：检索增强的事件抽取，用 Wikipedia 检索补充事件知识。

## 六、开放域事件抽取

### 6.1 任务定义

开放域 EE 不依赖预定义的事件类型，而是从文本中**自动发现**新事件类型。

### 6.2 流水线

1. **候选事件识别**：用触发词检测、句法模式找到潜在事件。
2. **事件类型归纳**：对候选事件聚类，归纳出新类型。
3. **论元角色归纳**：发现每种类型的常见论元。

代表工作：

- **Libra**（Huang & Ji, 2020）：用层级聚类 + 自动命名事件类型。
- **FrameNet 扩展**：用 FrameNet 的语义框架自动标注。
- **LLM 主导**：用 ChatGPT / GPT-4 直接生成事件类型与论元描述。

### 6.3 工业实践

- **新闻事件流**：实时监控 1000+ 媒体，自动发现新事件。
- **金融事件流**：从公告中抽取并购、增减持、业绩预告等，自动归类。
- **医疗事件流**：从电子病历抽取诊断、用药、手术事件，辅助决策。

## 七、少样本与零样本事件抽取

### 7.1 少样本 EE

FewEvent（2019）是少样本事件检测的标准基准：100 类事件，每类仅 N 个样本（典型 N=5）。代表方法：

- **原型网络**：对每类事件计算支持集样本的均值作为原型，查询样本与原型距离决定分类。
- **Matching Networks**：注意力机制匹配支持集与查询。
- **GNN-FewShot**：把支持集与查询集建成图，用 GNN 传播信息。

### 7.2 零样本 EE

用事件类型名称的自然语言描述作为提示：

```python
prompt = """判断以下句子是否包含 "自然灾害-地震" 事件。

事件描述：地震是指地壳快速释放能量过程中造成的振动。

句子：日本本州东岸近海发生 6.2 级地震。

是否触发：是（trigger：地震）"""
```

LLM 在零样本事件检测上 F1 约 40-50%，远不及微调监督，但提供了**冷启动**的快速路径。

## 八、多模态事件抽取

文本只是事件的载体之一。结合图像、视频、表格、传感器数据能显著提升抽取质量。

### 8.1 任务场景

- **新闻**：文本 + 现场图片，识别事件 + 标注关键视觉证据。
- **体育**：比赛解说 + 视频帧，识别进球、犯规、换人。
- **社交媒体**：文本 + 图片 + 元数据（位置、时间），识别事件类型。

### 8.2 代表方法

- **WASE**（Yang et al., 2021）：用 CLIP 编码图像，文本与图像联合编码。
- **Multimodal Event Extraction**：用 ViLBERT、CLIP 等多模态预训练。
- **OntoEvent**：百度 2023 年的多模态事件抽取基准。

## 九、PyTorch 实现：BERT 触发词 + 论元联合抽取

```python
import torch
import torch.nn as nn
from transformers import AutoModel


class BertJointEE(nn.Module):
    """BERT 联合事件抽取：同时预测 trigger 和 argument role。"""

    def __init__(self, model_name: str, num_event_types: int, num_roles: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        D = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        # Trigger head: 每个 token 分类为 (none, event_type_1, event_type_2, ...)
        self.trigger_head = nn.Linear(D, num_event_types + 1)
        # Argument head: 给定 trigger 位置 h_t，预测每个实体起止位置的角色
        self.arg_head = nn.Linear(D * 2, num_roles + 1)

    def forward(self, input_ids, attention_mask, entity_start, entity_end, trigger_positions=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)

        # Trigger 分类
        trigger_logits = self.trigger_head(h)              # (B, T, num_types+1)
        if trigger_positions is None:
            return trigger_logits

        # Argument 分类：对每个 trigger，取其表示 h_t，与每个实体表示拼接
        trigger_rep = h[torch.arange(h.size(0)), trigger_positions]   # (B, D)
        ent_rep = (h[torch.arange(h.size(0)), entity_start] +
                   h[torch.arange(h.size(0)), entity_end]) / 2       # (B, D)
        arg_rep = torch.cat([trigger_rep, ent_rep], dim=-1)            # (B, 2D)
        arg_logits = self.arg_head(arg_rep)                            # (B, num_roles+1)
        return trigger_logits, arg_logits
```

训练时：

- **Trigger 损失**：所有 token 的 trigger 分类交叉熵。
- **Argument 损失**：已知 trigger 位置下，每个实体是/不是该事件论元的二分类。

## 十、工程实践与挑战

### 10.1 评估

- **Trigger F1**：trigger 词 + 事件类型正确才计为正确。
- **Argument F1**：argument 词 + 角色正确才计为正确。
- **Event F1**：trigger + 全部 arguments 都正确才计为正确（最严格）。

### 10.2 常见错误

| 错误类型 | 占比 | 改进方法 |
| --- | --- | --- |
| Trigger 漏检 | ~25% | 多模型集成 + 词典 |
| Trigger 类型错 | ~20% | 上下文信息、句法 |
| Argument 边界错 | ~20% | 更好的 NER |
| Argument 角色错 | ~20% | 全局联合、约束 |
| 跨句事件 | ~15% | 文档级模型 |

### 10.3 工业部署

- **金融公告**：实时抽取并购、增减持事件，自动入库。
- **新闻事件流**：监控 1000+ 媒体，自动聚类同一事件的不同报道。
- **医疗**：从电子病历抽取疾病诊断、手术操作、用药事件。

## 十一、未来方向

1. **LLM 主导的事件抽取**：用 ChatGPT / GPT-4 做冷启动 + 监督微调做量产。
2. **多模态融合**：文本 + 图像 + 视频 + 时间戳的联合事件抽取。
3. **事件演化追踪**：识别同一事件的多个时间点报道，构建事件演化图谱。
4. **因果事件链**：识别"A 导致 B"、"B 阻碍 C"等因果链，是医疗、经济学的核心。
5. **事件知识图谱**：把抽取的事件构建成动态图谱，支撑问答、推理、决策。

## 小结

| 时代 | 方法 | 优势 | 局限 |
| --- | --- | --- | --- |
| 1990s-2000s | 规则模板 | 高精确率 | 低召回、维护贵 |
| 2005-2014 | 联合模型 + 特征 | 结构化推理 | 特征工程 |
| 2015-2018 | DMCNN / JRNN | 端到端 | 长距离、复杂结构 |
| 2018-2022 | BERT + OneIE | 预训练 + 联合图 | 数据稀缺 |
| 2022+ | LLM + 文档级 + 多模态 | 零样本新类型 | 推理慢、幻觉 |

事件抽取是 NLP 信息抽取的"珠穆朗玛峰"——它要求模型同时理解句法、语义、上下文与外部知识，且输出是高度结构化的复杂对象。从 ACE 2005 到 MAVEN，从单句到文档，从纯文本到多模态，事件抽取的演进折射出 NLP 整体的发展脉络。掌握事件抽取，你就掌握了结构化 NLP 输出的"完整拼图"。
