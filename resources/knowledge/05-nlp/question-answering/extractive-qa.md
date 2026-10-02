# 抽取式问答：从 SQuAD 到 BERT for QA 的 span 预测范式

抽取式问答（Extractive Question Answering）是 NLP 中最经典的任务之一——给定一段上下文和一个问题，模型从上下文中找到（抽取）答案片段。与自由生成答案不同，抽取式 QA 假设答案一定是原文的某个 span，这大大简化了任务，也使得 BERT 时代的预训练-微调范式取得巨大成功。本文深入剖析 SQuAD、MRC 范式、BERT for QA 的实现细节，以及后续的改进方向。

## 一、任务定义与评测

### 1.1 任务形式

给定：

- **段落** $\mathbf{p} = (p_1, \dots, p_m)$：包含答案的文本。
- **问题** $\mathbf{q} = (q_1, \dots, q_n)$：用户提出的问题。

输出：

- **答案起止位置** $(s, e)$，$s \leq e$，答案就是 $\mathbf{p}_{s:e}$。

### 1.2 三大评测基准

| 基准 | 语言 | 规模 | 特点 |
| --- | --- | --- | --- |
| SQuAD 1.1 | 英 | 100K 问题 | 答案必在文中 |
| SQuAD 2.0 | 英 | 150K 问题 | 加入不可答 |
| CMRC 2018 | 中 | 20K 问题 | 中文 SQuAD |
| Natural Questions | 英 | 300K 问题 | 真实 Google 查询 |
| TriviaQA | 英 | 650K 问题 | 远程监督 |

### 1.3 评测指标

- **Exact Match (EM)**：预测答案与标准答案完全一致（已标准化）的比例。
- **F1 分数**：预测与标准答案的 token 重合 F1。

$$
\text{F1} = 2 \cdot \frac{P \cdot R}{P + R}, \quad P = \frac{|\text{pred} \cap \text{gold}|}{|\text{pred}|}, \quad R = \frac{|\text{pred} \cap \text{gold}|}{|\text{gold}|}
$$

对每个问题取多个参考答案的最大 F1，最后取平均。

### 1.4 SQuAD 2.0 的不可答问题

SQuAD 2.0（Rajpurkar et al., 2018）加入**不可答问题**——段落中不包含答案：

```
段落：苹果公司于 1976 年由史蒂夫·乔布斯创立。
问题：苹果公司的现任 CEO 是谁？
答案：不可答
```

模型必须同时预测"是否有答案" + "答案位置"。这一改动让评测更接近真实场景——用户经常问"段落里没有的问题"。

## 二、MRC 范式：从 LSTM 到 BERT

### 2.1 经典 BiDAF

Seo et al．（2016）的 BiDAF（Bidirectional Attention Flow）是 MRC 时代的标杆：

1. **Embedding 层**：字符级 + 词级 embedding。
2. **Encoder 层**：双向 LSTM 编码段落与问题。
3. **Attention 层**：双向注意力——段落→问题、问题→段落。
4. **Modeling 层**：另一层 BiLSTM 融合 attention 输出。
5. **Output 层**：预测答案的 start 与 end 位置。

BiDAF 在 SQuAD 1.1 上 EM 达 67.7、F1 达 77.3，是 2017 年前的 SOTA。

### 2.2 BERT for QA 的范式

BERT 把 MRC 推向新高度。Devlin et al.（2019）的 BERT-SQuAD 实现：

```python
输入：[CLS] question [SEP] paragraph [SEP]
```

把问题和段落拼接输入 BERT，每个位置的隐藏状态 $\mathbf{h}_i$ 同时编码了"它在段落中"和"它与问题的关系"。

**预测 start**：

$$
P_{\text{start}}(i) = \frac{\exp(\mathbf{w}_{\text{start}}^\top \mathbf{h}_i)}{\sum_j \exp(\mathbf{w}_{\text{start}}^\top \mathbf{h}_j)}
$$

**预测 end**：

$$
P_{\text{end}}(i) = \frac{\exp(\mathbf{w}_{\text{end}}^\top \mathbf{h}_i)}{\sum_j \exp(\mathbf{w}_{\text{end}}^\top \mathbf{h}_j)}
$$

损失函数：

$$
\mathcal{L} = -\log P_{\text{start}}(s^*) - \log P_{\text{end}}(e^*)
$$

推理时取 $P_{\text{start}} \cdot P_{\text{end}}$ 最大的 $(s, e)$ 对，且 $s \leq e$。

### 2.3 效果飞跃

SQuAD 1.1 上的演进：

| 模型 | 时间 | EM | F1 |
| --- | --- | --- | --- |
| 人类 | - | 82.3 | 91.2 |
| BiDAF | 2017 | 67.7 | 77.3 |
| BiDAF + Self-Attention | 2018 | 72.1 | 81.1 |
| BERT-base | 2018 | 80.8 | 88.5 |
| BERT-large | 2018 | 84.1 | 90.9 |
| BERT-large + 集成 | 2019 | 87.4 | 93.2 |
| RoBERTa-large | 2019 | 88.5 | 94.6 |
| ALBERT | 2020 | 89.3 | 95.0 |

BERT 让 MRC 性能首次接近人类水平。

## 三、不可答问题（SQuAD 2.0）

### 3.1 解决方案

预测答案时还要判断"是否可答"：

$$
P_{\text{has\_ans}} = \sigma(\mathbf{w}_0^\top \mathbf{h}_{\text{[CLS]}})
$$

训练时：

- 可答：$\text{target} = 1$，正常预测 $(s, e)$。
- 不可答：$\text{target} = 0$，让 $s$ 和 $e$ 都指向 `[CLS]`。

损失：

$$
\mathcal{L} = -\log P_{\text{has\_ans}}(y^*) - \mathbb{1}[y^*=1] \left[ \log P_{\text{start}}(s^*) + \log P_{\text{end}}(e^*) \right]
$$

### 3.2 不可答训练技巧

- **Null Score Difference**（阈值分类器）：

$$
\text{null\_score} = \mathbf{S}_0 + \mathbf{E}_0 \quad \text{（s 和 e 都在 [CLS] 的分数）}
$$
$$
\text{best\_non\_null} = \max_{i \neq \text{[CLS]}} \mathbf{S}_i + \mathbf{E}_i
$$
$$
\text{has\_answer} = (\text{best\_non\_null} - \text{null\_score}) > \delta
$$

$\delta$ 在 dev 集上调优。

- **对抗训练**：生成"看似可答但实际不可答"的问题。
- **数据增强**：把可答问题的答案替换为 random span，构造不可答样本。

## 四、PyTorch 实现：BERT QA

```python
import torch
import torch.nn as nn
from transformers import AutoModel


class BertQA(nn.Module):
    """BERT for SQuAD-style Extractive QA。"""

    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        D = self.bert.config.hidden_size
        self.qa_outputs = nn.Linear(D, 2)  # start & end logits
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                start_positions=None, end_positions=None):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        h = self.dropout(out.last_hidden_state)
        logits = self.qa_outputs(h)                  # (B, T, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)     # (B, T)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            return (start_loss + end_loss) / 2

        return start_logits, end_logits

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, token_type_ids=None,
                n_best_size=20, max_answer_length=30):
        start_logits, end_logits = self(input_ids, attention_mask, token_type_ids)
        # 取 start * end 的 top-k 个 span
        start_probs = start_logits.softmax(-1)
        end_probs = end_logits.softmax(-1)
        # 简单实现：取 top-k start，top-k end，组合
        candidates = []
        for i in range(input_ids.size(0)):
            start_idx = start_probs[i].topk(n_best_size).indices
            end_idx = end_probs[i].topk(n_best_size).indices
            for s in start_idx:
                for e in end_idx:
                    if s <= e and (e - s + 1) <= max_answer_length:
                        score = (start_probs[i, s] * end_probs[i, e]).item()
                        candidates.append((s.item(), e.item(), score))
        return sorted(candidates, key=lambda x: x[2], reverse=True)
```

关键设计：

- **单层分类头**：start 和 end 共享同一隐藏状态，仅用不同权重。
- **subword 对齐**：训练时把答案的起止对齐到 subword 序列。
- **`max_answer_length`**：限制答案长度，过长反而有噪声。

## 五、长文档 QA

### 5.1 问题

BERT 最大输入 512 token，但许多文档（法律、研究论文）远超 512。需要专门技术。

### 5.2 Sliding Window

最朴素的方案：滑动窗口分段，每段独立预测答案，最后合并：

```
文档 (5000 token) → 切分 (10 × 512) → 各段预测 → 合并答案
```

合并策略：取分数最高的答案，或用 NMS 去重。

代表工作：**DocumentQA**（Chen et al., 2017）。

### 5.3 Longformer / BigBird

改造 attention 让模型支持长输入：

- **Longformer**：window + dilated + global attention，复杂度 $O(n)$。
- **BigBird**：random + window + global attention。

Longformer-base（4096 token）在 Natural Questions 上 F1 达 56，适合文档级 QA。

### 5.4 Hierarchical 模型

把文档分成段，先段内 QA，再段间聚合：

1. 每段独立 BERT QA。
2. 用 cross-segment attention 聚合信息。
3. 重新预测答案。

代表：**SDNet**（Zhu et al., 2018）、**HAT**（Chen et al., 2020）。

### 5.5 检索式方法

对于超长文档（整本书、整个网站），用**两阶段**：

1. **段落检索**：找到最相关的 100-1000 个段落。
2. **段落 QA**：在候选段落上做 BERT QA。

详见 [[open-domain-qa-rag]] 一文。

## 六、多跳 QA（Multi-Hop QA）

### 6.1 任务定义

多跳 QA 需要跨多个段落推理：

```
段落 1：A 出生于北京。
段落 2：B 是 A 的老师。
段落 3：B 工作于清华。

问题：B 在哪里工作？
答案：需要先从段落 2 找到 B 与 A 的关系，再从段落 3 找到 B 的工作地。
```

### 6.2 HotpotQA

Yang et al．（2018）的 HotpotQA 是多跳 QA 的标准评测：

- **98,168 个问题**，平均每问题需 2.4 跳推理。
- **支持事实**（supporting facts）标注：哪几个句子参与了推理。
- **评测指标**：QA F1 + 支持事实 F1 + 联合准确率。

### 6.3 代表方法

#### (a) 链式推理（Chain-of-Reasoning）

先识别第一跳答案，再基于它找第二跳：

```
问题 → 第一跳文档 → 中间答案 → 第二跳文档 → 最终答案
```

代表：**CogQA**（Ding et al., 2019）、**ChainNet**。

#### (b) 图网络（Graph Neural Networks）

把文档中实体 / 句子建图，用 GNN 传播信息：

- **DFGN**（Xiao et al., 2019）：dynamic fusion graph + GNN。
- **HGN**（Fang et al., 2020）：hierarchical graph network。
- **SAE**（Tu et al., 2020）：select, answer, explain。

#### (c) Decomposable Attention

把问题分解为子问题，逐个回答：

```
问题：B 在哪里工作？
子问题 1：B 是谁？
子问题 2：B 在哪里？
```

代表：**DecompRC**（Min et al., 2019）。

### 6.4 当前 SOTA

- **Beam Retrieval + FiD**（Ye et al., 2021）：beam search 检索 + Fusion-in-Decoder。
- **Hindsight**（Armen et al., 2022）：用 beam 搜索发现推理路径。
- **BeamDR**（Yoran et al., 2022）：把多跳视为文档检索问题。

## 七、多语言与跨语言 QA

### 7.1 数据集

- **XQuAD**（Artetxe et al., 2020）：SQuAD 1.1 翻译为 11 种语言。
- **MLQA**（Lewis et al., 2020）：7 种语言，跨语言对齐问题。
- **TyDi QA**（Clark et al., 2020）：11 种类型学多样的语言。

### 7.2 跨语言迁移

- **mBERT**：直接微调 mBERT on SQuAD，零样本迁移到其他语言。
- **XLM-R**：在 XLM-R 上微调 + translate-train（把目标语言翻译成源语言训练）。
- **COREC**（Liu et al., 2022）：用对比学习对齐多语言表示。

### 7.3 实战经验

- 高资源语言（en ↔ zh）零样本 mBERT 即可达到 F1 70+。
- 低资源语言（sw ↔ en）需要 translate-train 或 few-shot 微调。
- 多语言混合训练可能引入语言混淆。

## 八、对抗与鲁棒性

### 8.1 对抗样本

Jia & Liang（2017）的"AddSent"攻击：在 SQuAD 段落末尾加干扰句：

```
段落：... 最后重要的是，[干扰句] 答案：X
```

模型可能被误导，把 X 错认为答案。BERT 在这种对抗下准确率下降 20+ 个点。

### 8.2 鲁棒训练

- **数据增强**：用回译、实体替换、问题改写扩充训练集。
- **对抗训练**（FGSM / PGD）：训练时加入扰动。
- **RTT**（Round-Trip Translation）：原文 → 中文 → 英文，让模型见过"等价但不同"的段落。
- **Mixup**：在 embedding 层做样本混合。

### 8.3 不变性测试

QA 模型应当对**不改变语义的扰动**保持不变：

- 同义替换："哪个国家" → "哪国"。
- 段落内句子顺序调整。
- 答案实体的指代表达替换。

实际测试显示 BERT 在这些扰动下性能下降 5-15%，仍有改进空间。

## 九、QA 与其他任务的统一

### 9.1 Unified QA

Khashabi et al.（2020）的 Unified QA 把 QA、阅读理解、分类、QA-over-KB 等任务统一为"输入文本-输出文本"：

```
输入：[context] [question]
输出：answer
```

跨任务训练后模型学会多种格式的问答。

### 9.2 FLAN-T5

Wei et al.（2022）的 FLAN 把 QA 视为指令微调的任务之一，训练后 T5 模型能按指令切换任务。

### 9.3 ChatGPT / GPT-4 的 QA

现代 LLM 的 QA 不限于抽取：

- 直接生成答案。
- 处理不可答（"我无法回答"）。
- 多轮澄清（"你指的是哪个？"）。

但在严格抽取场景，专用 BERT 模型仍更可靠。

## 十、应用与工程

### 10.1 文档问答系统

```
用户问题
   ↓
段落检索（BM25 / Embedding）
   ↓
Top-K 候选段落
   ↓
BERT QA 抽取答案
   ↓
答案 + 引用展示
```

工业部署关心：

- **延迟**：BERT QA 在 GPU 上 50-100ms / 问题。
- **召回 vs 精度**：检索 top-K 影响答案质量。
- **多模态 QA**：表格 + 文本 + 图像的联合 QA。

### 10.2 FAQ 系统

把 FAQ 作为 QA 微调数据：

- 用户问题 → 历史 FAQ 答案。
- 用 BERT 做相似度排序 + 答案抽取。

### 10.3 客服机器人

QA + NLU + Slot Filling + 对话管理 + 答案生成 = 完整客服。

## 十一、未来方向

1. **多模态 QA**：图像 + 文本 + 表格的联合 QA。
2. **多跳多模态**：跨图像、跨段落的复杂推理。
3. **可解释 QA**：模型必须给出推理路径。
4. **对抗鲁棒 QA**：稳定抵抗各种扰动。
5. **生成式 QA**：见 [[generative-qa]] 一文。

## 小结

| 时代 | 模型 | 关键创新 | SQuAD EM |
| --- | --- | --- | --- |
| 2016 | BiDAF | 双向 attention | 67.7 |
| 2017 | r-net, FusionNet | 多跳 attention | 75 |
| 2018 | BERT-base | 预训练 | 80.8 |
| 2019 | BERT-large + 集成 | 模型集成 | 87.4 |
| 2020 | ALBERT | 参数共享 | 89.3 |
| 2020+ | Longformer | 长文档 | 89+ |

抽取式 QA 是 NLP "BERT 时代" 最有代表性的成功案例——预训练 + 微调范式把 MRC 推向人类水平。但抽取式有局限：答案必须存在、必须连续、必须是原文片段。下一篇我们将进入生成式 QA，看 T5、FLAN、ChatGPT 如何突破这些限制。
