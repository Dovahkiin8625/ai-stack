# 命名实体识别：从规则到 LLM 的完整演进

命名实体识别（Named Entity Recognition, NER）是信息抽取的"原子操作"——它从非结构化文本中识别出预定义类型的实体片段（人名、地名、机构名、日期、金额等），是后续关系抽取、事件抽取、实体链接、知识图谱构建的基础。本文沿"规则 → 统计学习 → 深度学习 → 提示学习"的时间线，剖析 NER 的核心挑战与工程实践。

## 一、任务定义与评测

### 1.1 任务形式

给定输入句子 $\mathbf{x} = (x_1, \dots, x_n)$，输出实体集合：

$$
\mathcal{E} = \{ (s, e, t) \mid s \leq e,\ t \in \mathcal{T} \}
$$

其中 $(s, e)$ 是实体的起止位置，$t$ 是类型。常见类型集：

- **粗粒度**（3 类）：PER（人名）、LOC（地名）、ORG（机构名）。
- **细粒度**（OntoNotes 5, 18 类）：PER, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, MONEY, PERCENT, QUANTITY, ORDINAL, CARDINAL.
- **垂直领域**：医学（疾病、药物、症状、解剖部位）、金融（股票代码、上市公司、监管机构）、法律（法条、案件、当事人）。

### 1.2 评测指标

- **Micro-F1**：把所有句子的实体合并计算 F1，最常用。
- **Span 严格匹配**：实体的起止 + 类型完全一致才算正确。
- **边界 F1**：仅评估边界，不区分类型——用于类型缺失场景。

### 1.3 主要评测基准

| 基准 | 语言 | 类别数 | 实体数 | 特点 |
| --- | --- | --- | --- | --- |
| CoNLL-2003 | 英/德/西/荷 | 4 | ~35K | 经典英文 NER |
| OntoNotes 5 | 英/中/阿 | 18 | ~160K | 多语言细粒度 |
| MSRA | 中 | 3 | ~50K | 中文新闻 |
| Weibo / CLUENER | 中 | 8-10 | ~20K | 社交媒体 |
| BC5CDR / NCBI | 英 | 1-2 | ~30K | 生物医学 |
| CCKS 2018 | 中 | 6 | ~15K | 医疗电子病历 |

## 二、早期方法：基于规则与词典

### 2.1 规则系统

NER 最早的工业实现是基于规则的，例如：

- **正则表达式**：匹配日期、电话、邮箱、身份证号等。
- **触发词词典**：维护"出生于"、"任职于"、"毕业于"等模式词。
- **上下文规则**：如"称谓 + 1-4 字中文 → 人名"。

代表系统：**ANNIE**（GATE 的一部分）、**LT-XML2**。

优点：精确率高、对结构化模式（如编号、日期）覆盖好、可解释。缺点：召回率低、维护成本高、跨领域迁移困难。

### 2.2 半结构化词典匹配

字典/词典匹配是规则法的延伸。给定词典 $\mathcal{D}$（如 Wikipedia 标题），对每个文本片段做最长匹配：

$$
(s, e) = \arg\max_{i \leq j} \left\{ j - i \mid (x_i, \dots, x_j) \in \mathcal{D} \right\}
$$

工具：**Ahocorasick**（多模式匹配，$O(n + \text{matches})$）、**FlashText**（百万级词典匹配秒级完成）。

词典匹配的弱点：**召回严重依赖词典覆盖率**。现代工业 NER 几乎都用神经网络，但仍会把词典作为特征（lexicon feature）注入——尤其在中文场景。

## 三、统计学习时代

### 3.1 HMM / MEMM / CRF

详见 [[hmm-crf]] 一文。HMM 在 90 年代是主流，2000 年代后被 CRF 取代。经典特征：

- **字符级别**：词性、词形、是否大写、字符 n-gram。
- **词典特征**：当前词是否在某个词典中出现。
- **上下文特征**：前后 1-2 个词的标签 / 词性。

### 3.2 经典工作

- **Stanford NER**（Finkel et al., 2005）：CRF + 人工特征，CoNLL-2003 F1 达 86.86。
- **Illinois NER**（Ratinov & Roth, 2009）：CRF + 大量外部词典 + 规则后处理。
- **SENNA**（Collobert et al., 2011）：神经网络 NER 的开山之作，用窗口式 CNN。

## 四、深度学习时代

### 4.1 BiLSTM-CRF

详见 [[bilstm-crf-ner]] 一文。BiLSTM-CRF 在 2015-2017 年是 NER 的事实标准，CoNLL-2003 上 F1 首次突破 90。

### 4.2 BERT-NER

详见 [[transformer-tagging]] 一文。BERT + Softmax 或 BERT + CRF 把 F1 推到 92+，是当前（2026）工业部署的主流方案。

### 4.3 中文 NER 的特殊处理

中文 NER 与英文的关键差异：

1. **没有显式分词边界**：必须由模型自动判断"清华大学"是一整个词还是"清华 / 大学"两个词。
2. **字符 vs 词汇**：纯字符 NER 损失词级信息；引入 lexicon feature 显著提升。
3. **嵌套现象更常见**：百度 DuIE 数据集中 30%+ 实体是嵌套的。

经典方案：

- **字符 + 词向量**：Zhang & Yang (2018) 的 Lattice LSTM，把词典匹配的所有可能词作为额外输入。
- **Lex-BERT**：在 BERT 输入中嵌入"该字符是否属于某个词典中的某词"的特征。
- **Soft-Lexicon**（Ma et al., 2020）：把字符的词典信息压缩为 4 个向量（B、M、E、S），效果优于 Lattice LSTM 且并行。

## 五、嵌套 NER

详见 [[transformer-tagging]] 一文。代表方案：

- **GlobalPointer**：苏剑林 2022 年的工作，矩阵打分，CLUENER F1 突破 70%。
- **TPLinker / TPLinker-plus**：基于 token pair 关系，可处理不连续实体。
- **PL-Marker**：2022 年的 SOTA，引入方向感知的伪标记。

## 六、低资源 NER

工业中常遇到标注数据极少的场景（几十条标注）。解决方案：

### 6.1 跨域迁移

在大规模源域（如 OntoNotes）训练，目标域（如医疗）只微调几层或几轮：

- **冻结底层**：只微调 BERT 的最后 2 层 + CRF。
- **Adapter**：在 BERT 各层插入小型可训练 Adapter，参数量 < 5%。
- **LoRA**：在 attention 权重上加低秩分解，只训 < 1% 参数。

### 6.2 字典增强 + 远程监督

用大型词典（如 Wikipedia 标题、百度百科）在未标注语料上做远程监督：

1. 用词典匹配所有候选实体。
2. 用 NER 模型打分，高分作为"伪标签"。
3. 在伪标签 + 少量人工标注上训练。

### 6.3 提示学习 NER

把 NER 视为生成任务：

```
输入：北京是中国的首都。
提示：北京是[ORG:国家/城市]的首都。
输出：城市
```

模型在 mask 位置生成的 token 就是实体类型。GPT-3.5 / ChatGPT 在 MSRA 上 F1 约 80%，远不如微调 BERT；但**对零样本类型定义的支持极好**——只要在提示词里写好"什么是新类型"，模型就能识别。

## 七、LLM 时代的 NER 范式

### 7.1 提示词工程

主流范式：用 ChatGPT / GPT-4 / Claude 通过结构化提示做 NER：

```python
prompt = """从以下句子中提取命名实体，返回 JSON 数组。

实体类型：PER（人名）、LOC（地名）、ORG（机构名）、DATE（日期）

句子：{}年{}月，特斯拉 CEO 马斯克访问了中国上海。

输出格式：[{"text": "...", "type": "...", "start": int, "end": int}]

JSON："""
```

主流 LLM 在标准 NER 上达到：

| 模型 | CoNLL-2003 F1 | MSRA F1 |
| --- | --- | --- |
| GPT-3.5 | ~85 | ~75 |
| GPT-4 | ~91 | ~85 |
| Claude 3 Opus | ~92 | ~88 |
| 微调 BERT-base | 92.8 | 94+ |

LLM 在通用 NER 上已接近微调 BERT，但对长尾类型、低资源语言仍有差距。工业实践通常是 **LLM + BERT 双轨**：LLM 做快速原型与冷启动，BERT 做高精度量产部署。

### 7.2 结构化输出与后处理

LLM 的 NER 输出经常需要后处理：

1. **JSON 解析失败**：用正则或 AST 修复常见错误（缺逗号、缺括号）。
2. **位置偏移**：用字符串匹配定位，避免 LLM 输出错的 start/end。
3. **类型不一致**：用白名单强制转换。

工具：`guardrails-ai`、`instructor`（强制 schema）、`outlines`（正则约束生成）。

## 八、PyTorch 实战：BERT 中文 NER 完整流程

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset

# 1. 准备数据
dataset = load_dataset("json", data_files="msra_train.json", field="data")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def tokenize_and_align(examples):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
    labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(word_labels[word_idx])
            else:
                # subword 的后续片段用 -100 忽略，或用相同标签
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_datasets = dataset.map(tokenize_and_align, batched=True)

# 2. 训练
model = AutoModelForTokenClassification.from_pretrained("bert-base-chinese", num_labels=11)

args = TrainingArguments(
    output_dir="bert-crf-ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["validation"],
                  tokenizer=tokenizer)
trainer.train()
```

实现要点：

- **subword 对齐**：BERT 的 WordPiece 切分会把一个汉字切成多个 token，必须把标签正确对齐到第一个 subword。
- **`-100` 是 PyTorch 默认忽略标签**：subword 后续片段和 pad 都用 `-100`，避免影响损失。
- **`evaluation_strategy="epoch"`**：每 epoch 在 dev 集上评估，保留最优 checkpoint。

## 九、NER 的工程化挑战

### 9.1 速度

工业部署关心吞吐量。BERT-base 单卡 A100 上：

- **Batch 32, seq 128**：~5000 句/秒（Softmax）、~3000 句/秒（CRF）。
- **DistilBERT**：提速 60%，精度损失 < 2%。
- **ONNX / TensorRT 量化**：再提速 2-3 倍。
- **稀疏注意力**：Longformer、BigBird 处理 >512 token 文档时必要。

### 9.2 多任务标注

工业 NER 常需要识别多种类型：

- 单一模型 + 多类型标签（11-30 类）：CRF 仍可胜任。
- **多任务学习**：NER + POS + Chunking 共享底层 Transformer，顶部各任务独立 head，互相正则化。
- **联合学习**：NER + Entity Linking 一起训练，端到端优化。

### 9.3 持续学习

NER 系统上线后常常遇到**新类型 / 新领域**：

- **增量训练**：在新数据上继续训练，但有"灾难性遗忘"风险——需要 replay 旧数据或加 EWC（Elastic Weight Consolidation）。
- **AdapterFusion**：为每个领域训练独立 Adapter，推理时根据输入动态融合。

## 十、未来方向

1. **统一标注框架**：LLM 把 NER、关系抽取、事件抽取统一为"生成结构化 JSON"，可能取代任务专用模型。
2. **多模态 NER**：结合图像（视频帧、产品图）、表格（财务报表）、语音（会议录音）的多模态 NER。
3. **主动学习与人在回路**：让模型主动选样请求标注，最大化标注 ROI。
4. **零样本类型发现**：从非结构化文本中自动发现新实体类型（如生物医学领域的新基因名）。

## 小结

| 时代 | 方法 | 优势 | 局限 |
| --- | --- | --- | --- |
| 1980s-90s | 规则 + 词典 | 高精确率 | 低召回、维护贵 |
| 1990s-2010s | HMM/CRF | 概率框架、特征灵活 | 强独立性假设 |
| 2015-2018 | BiLSTM-CRF | 端到端、性能跃升 | 长序列、长尾类型 |
| 2018-2024 | BERT-CRF / GlobalPointer | 预训练 + 微调 | 训练贵、标注需求高 |
| 2024+ | LLM 提示 | 零样本新类型、灵活 | 位置定位不准、贵 |

NER 作为 NLP 的"老兵"任务，从规则到 LLM 完整地经历了"特征工程 → 端到端 → 预训练 → 通用智能"的演进。当前（2026）的最佳实践：**LLM 快速冷启动 + 微调 BERT 量产部署 + CRF 后处理保障结构**。下一篇我们将进入关系抽取：从单个实体，扩展到实体之间的语义关系。
