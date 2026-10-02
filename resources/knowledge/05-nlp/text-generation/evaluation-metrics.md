# 文本生成评估：从 BLEU 到 LLM-as-Judge 的指标体系

文本生成评估是 NLP 最难的问题之一——"什么是好文本"本身就难以定义。从 BLEU、ROUGE 等 n-gram 重合指标，到 BERTScore、COMET 等语义嵌入指标，再到最近的 LLM-as-Judge 和人类评估的结合，本文系统梳理当前主流的评估方法，并讨论各方法的局限与适用场景。

## 一、为什么评估难

### 1.1 内在矛盾

- **多样性与准确性的张力**：同一句话可以有多种正确表达，n-gram 匹配会惩罚"等价但不同"的情况。
- **质量的多个维度**：流畅度、相关性、创造性、安全性、风格一致性——单一指标难以兼顾。
- **主观性**：人类对"好文本"的判断有文化、年龄、专业背景差异。

### 1.2 黄金标准

人类评估仍是文本生成的"黄金标准"，但成本高、不可重现、主观性强。理想的自动指标应满足：

1. **与人类判断高度相关**。
2. **可计算、可重现**。
3. **解释性强**：分数高低代表什么。
4. **公平**：对生成模型无偏。

## 二、N-gram 重合指标

### 2.1 BLEU（Papineni et al., 2002）

最经典的机器翻译指标。计算 n-gram（n=1,2,3,4）精确率的几何平均，再乘以短句惩罚（BP）：

$$
\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^{N} \frac{1}{N} \log p_n \right)
$$

其中 $p_n$ 是 n-gram 精确率：

$$
p_n = \frac{\sum_{C \in \text{Candidates}} \sum_{\text{ngram} \in C} \text{Count}_{\text{clip}}(\text{ngram})}{\sum_{C \in \text{Candidates}} \sum_{\text{ngram} \in C} \text{Count}(\text{ngram})}
$$

$\text{Count}_{\text{clip}}$ 限制每个 n-gram 最多在参考翻译中出现的次数。

### 2.2 BLEU 的局限

- **不区分语义**："猫在沙发上"和"狗在床上"虽然意思不同，但 BLEU 仍可能高。
- **不评估流畅度**：纯靠 n-gram 重合。
- **不评估语法**：句法错误可能不影响 BLEU。
- **短句偏好**：BP 部分缓解但未根除。

### 2.3 ROUGE（Lin, 2004）

主要用于**摘要评估**，与 BLEU 对偶：

- **ROUGE-N**：n-gram 召回率。
- **ROUGE-L**：最长公共子序列（LCS）F1。
- **ROUGE-W**：加权 LCS。
- **ROUGE-S**：Skip-Bigram。

$$
\text{ROUGE-L} = \frac{(1 + \beta^2) R_{\text{lcs}} P_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 P_{\text{lcs}}}
$$

ROUGE 强调**召回率**（参考摘要中多少被覆盖），适合评估摘要"是否包含了要点"。

### 2.4 METEOR（Banerjee & Lavie, 2005）

METEOR 引入同义词、词干匹配：

$$
\text{METEOR} = F_{\text{mean}} \cdot (1 - \text{Penalty})
$$

直觉：精确率 × 召回率（harmonic mean）× 碎片惩罚。METEOR 在多个任务上与人类相关性高于 BLEU。

### 2.5 chrF（Popović, 2015）

字符 n-gram 的 F1，对形态学丰富语言（捷克语、芬兰语、阿拉伯语）更友好：

$$
\text{chrF}_\beta = (1 + \beta^2) \frac{P_{\text{chr}} \cdot R_{\text{chr}}}{\beta^2 P_{\text{chr}} + R_{\text{chr}}}
$$

## 三、语义嵌入指标

### 3.1 BERTScore（Zhang et al., 2020）

用 BERT 的 token 表示计算余弦相似度：

**精确率**：

$$
P_{\text{BERT}} = \frac{1}{|\hat{\mathbf{y}}|} \sum_{\hat{y}_j \in \hat{\mathbf{y}}} \max_{y_i \in \mathbf{y}} \mathbf{x}_i^\top \hat{\mathbf{x}}_j
$$

**召回率**：

$$
R_{\text{BERT}} = \frac{1}{|\mathbf{y}|} \sum_{y_i \in \mathbf{y}} \max_{\hat{y}_j \in \hat{\mathbf{y}}} \mathbf{x}_i^\top \hat{\mathbf{x}}_j
$$

$\mathbf{x}_i, \hat{\mathbf{x}}_j$ 是 BERT 编码后的 token 向量。

直觉：候选翻译的每个 token 在参考中找到最相似的，召回 / 精确按此计算。BERTScore 能识别同义替换（如 "big" ↔ "large"），与人类相关性比 BLEU 高 20-30%。

### 3.2 BLEURT（Sellam et al., 2020）

基于 BERT 的**监督**指标：

1. 预训练 BERT。
2. 用合成数据预训练（成对的"原句 + 句子 + 分数"）。
3. 在 WMT 人类评分上微调。

BLEURT 学习了"句子质量"的人类判断，胜过 BLEU 和 BERTScore。

### 3.3 COMET（Rei et al., 2020）

跨语言优化的评估指标，用 XLM-R 编码源 + 候选 + 参考：

- **COMET-MQM**：用专业 MQM 标注训练。
- **COMET-HTER**：用 HTER 标注训练。
- **COMET-KI**：解释性版本，给出错误位置。

COMET 在 WMT 与人类评分的 Pearson 相关性约 0.85，远超 BLEU 的 0.4-0.5。

### 3.4 MoverScore（Zhao et al., 2019）

基于 **Earth Mover's Distance**（Wasserstein 距离）的句子相似度。把候选与参考的 token 表示视为分布，计算"搬运"成本。MoverScore 在 STS 上与人类相关性高于 BLEU。

## 四、任务特定指标

### 4.1 摘要：FactCC、QAGS、SummaC

- **FactCC**（Kryściński et al., 2020）：用 NLI 分类器判断摘要与原文是否事实一致。
- **QAGS**（Wang et al., 2020）：用 QA 模型从摘要生成问题，回到原文找答案，匹配判断事实性。
- **SummaC**（Laban et al., 2022）：用 NLI 模型分段判断，跨段聚合。

### 4.2 翻译：chrF、COMET、bleurt

机器翻译通常用 chrF + COMET 组合：

- chrF 提供字符级精度。
- COMET 提供语义级评估。
- 两者差异大时往往说明翻译有特定问题。

### 4.3 故事 / 创意写作

- **STORY-LEVEL COHERENCE**：用 GPT-4 评分 1-5。
- **Liu et al.（2023）的 StoryFr**：用事件链连贯性评估。
- **Plot-Entity Consistency**：实体一致性评估。

### 4.4 对话

- **USR**（Mehri & Eskenazi, 2020）：多维度评估（informative、coherent、consistent）。
- **FED**（Mehri & Eskenazi, 2020）：用 dialoGPT 等对话模型评估。
- **DST Eval**：对话状态跟踪任务的 slot 准确率、联合准确率。

## 五、LLM-as-Judge

### 5.1 基本范式

用强 LLM（GPT-4、Claude 3 Opus、Gemini 1.5 Pro）作为"评判者"，给生成文本打分：

```python
prompt = """你是一个专业的文本质量评估员。请按 1-5 分评估以下文本的质量。

评估维度：
1. 流畅度：语法是否正确，表达是否自然。
2. 相关性：是否回答了 prompt。
3. 创造性：是否有新意。

Prompt：{}
生成文本：{}

输出格式：
流畅度: <1-5>
相关性: <1-5>
创造性: <1-5>
总分: <1-5>
"""
```

### 5.2 代表方法

- **AlpacaEval**（Li et al., 2023）：用 GPT-4 比较两个模型的回复，胜率作为指标。
- **MT-Bench**（Zheng et al., 2023）：多轮对话评估，GPT-4 打分。
- **Chatbot Arena**：人类投票 + Elo 评分（详见下文）。
- **LLM-as-a-judge**（Zheng et al., 2023）：系统性研究 GPT-4 作为评判者。

### 5.3 LLM-as-Judge 的优势

- **与人类高度相关**：GPT-4 与人类评分的相关性达 0.85-0.95。
- **可解释**：可以要求 LLM 给出理由。
- **可扩展**：无需人工标注。
- **多维度**：可同时评估流畅度、相关性、安全性等。

### 5.4 LLM-as-Judge 的局限

#### (a) 位置偏差（Position Bias）

模型倾向于偏好放在前面的候选：

```
候选 A: ...
候选 B: ...
```

GPT-4 偏好 A 的概率约 55-60%。缓解：随机交换位置、多次评估取平均。

#### (b) 长度偏差（Length Bias）

LLM 倾向于偏好更长的回复，即使更长不等于更好。

#### (c) 自我偏好（Self-Enhancement）

LLM 倾向于给"自己生成的"或"同系列模型的"更高分。GPT-4 评估 GPT-4 输出显著高于人类评估。

#### (d) 评分尺度不一致

不同 prompt 下，1-5 分的含义不同，难以跨任务比较。

### 5.5 改进方案

- **多 LLM 投票**：GPT-4 + Claude + Gemini 多模型投票。
- **Logit-Based Scoring**：用 token logit 而非文本打分。
- **Pairwise 比较**：替代绝对评分，降低主观性。
- **Rubric-Based**：明确评分维度与标准。

## 六、人类评估方法论

### 6.1 直接评分（Direct Assessment）

评估者按 1-5 或 1-7 分制独立打分：

- 单评估者简单快速，但主观性大。
- 多评估者平均可降低噪声，但成本高。

Graham et al.（2015）的"群体质量评估"建议：

- 每个样本至少 3 个评估者。
- 移除离群分（>2 SD）。
- 报告 95% 置信区间。

### 6.2 Pairwise 比较

评估者在两个候选中选更好的：

```
生成 A: ...
生成 B: ...
评估：A 更好 / B 更好 / 平局
```

优势：

- 比绝对评分更稳定（避免尺度偏差）。
- 适合"哪个更好"的问题。

劣势：

- K 个模型两两比较需 $K(K-1)/2$ 次评估。
- 平局定义模糊。

### 6.3 排名（Ranked Comparison）

评估者对 K 个候选排序。比 pairwise 收集更多信息，但评估成本高。

### 6.4 Likert 多维度评分

按维度评分：

- 流畅度（1-5）
- 相关性（1-5）
- 创造性（1-5）
- 安全性（1-5）

提供细粒度反馈，是 NLP 学术论文的标准评估方式。

## 七、Chatbot Arena 与 Elo 评分

### 7.1 Chatbot Arena（Zheng et al., 2023）

LMSYS 推出的对话评估平台：

1. 用户与两个匿名模型对话。
2. 投票哪个更好。
3. 用 Elo 评分（国际象棋同款）计算模型排名。

截至 2026 年，Chatbot Arena 收集了超过 500 万次投票，是 LLM 评估的事实标准之一。

### 7.2 Elo 评分

Elo 起源于国际象棋评分：

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$
$$
R_A \leftarrow R_A + K \cdot (S_A - E_A)
$$

$E_A$ 是预期胜率，$S_A$ 是实际得分（1=胜，0.5=平，0=负），$K$ 是更新步长（典型 16-32）。

优势：

- 分数随时间动态调整。
- 不需所有模型两两比较。

局限：

- 早期投票少时分数不稳定。
- 投票偏差（如 GPT-4 粉丝多）。

### 7.3 Bradley-Terry 模型

更严格的数学框架：

$$
P(A > B) = \frac{e^{R_A}}{e^{R_A} + e^{R_B}}
$$

最大似然估计所有模型的真实分数。

## 八、基准评测套件

### 8.1 MT-Bench / Vicuna QA

80 个多轮对话问题，覆盖写作、推理、数学、编码等。GPT-4 作为评判。

### 8.2 AlpacaEval

805 个问题，比较模型与 GPT-4 / GPT-4 Turbo 的胜率。

### 8.3 MMLU

57 个学科的多选题（数学、法律、医学、计算机等），评估模型的"通识知识"。

### 8.4 GSM8K / MATH

数学推理基准。GSM8K 是小学数学，MATH 是高中数学。

### 8.5 HumanEval / MBPP / LiveCodeBench

代码生成基准。HumanEval 是经典，LiveCodeBench 是动态更新版本。

### 8.6 TruthfulQA / HaluEval

幻觉与事实性评估。判断模型输出是否包含错误信息。

## 九、特定任务的最佳实践

| 任务 | 推荐评估组合 |
| --- | --- |
| 机器翻译 | chrF + COMET + 人类 spot check |
| 摘要 | ROUGE-L + BERTScore + FactCC + 人类评估 |
| 对话 | USR + MT-Bench + Chatbot Arena |
| 故事创作 | BERTScore + LLM-as-Judge + 人类评估 |
| 推理 | GSM8K / MATH + Self-Consistency + 人类评估 |
| 代码生成 | HumanEval / MBPP + pass@k + 人工 review |
| 安全 / 毒性 | RealToxicityPrompts + Detoxify + 人工 |

## 十、评估的陷阱与反思

### 10.1 数据污染（Data Contamination）

模型可能在预训练时见过评测数据，导致分数虚高。典型表现：

- 模型能完美复述评测集原题答案。
- 不同 prompt 都能给出相同的"标准答案"。

缓解：

- 动态基准（LiveCodeBench、Chatbot Arena）。
- 私有测试集。
- 抗污染 prompt 变体。

### 10.2 Goodhart's Law

"当一个指标成为目标时，它就不再是好指标"。模型会针对指标"过拟合"——专门优化 BLEU、AlpacaEval 等分数，而非真正提升质量。

缓解：

- 多指标组合评估。
- 定期更新基准。
- 保留人类评估作为终极评判。

### 10.3 评测成本的指数增长

2023-2024 年间，各机构发布的 benchmark 评估成本激增：

- MMLU 5-shot：~30 分钟 / 模型。
- MT-Bench + AlpacaEval：~2-3 小时 / 模型。
- 完整评估套件：~10-20 小时 / 模型。

模型越大、benchmark 越多，成本难以承受。轻量级评估（如 AlpacaEval Lite）成为必要。

## 十一、未来方向

1. **Process Reward Model**：评估推理过程而非仅最终答案。
2. **Self-Evolving Benchmarks**：模型能解决后自动更新的动态基准。
3. **Multi-Modal Evaluation**：文本 + 图像 + 视频的联合评估。
4. **Causal Evaluation**：评估生成文本对后续决策的影响，而非仅静态质量。

## 小结

| 指标 | 时间 | 核心 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| BLEU / ROUGE | 2002/2004 | n-gram 重合 | 简单、可重现 | 不区分语义 |
| BERTScore | 2020 | 语义嵌入 | 识别同义 | 慢 |
| COMET / BLEURT | 2020 | 监督语义 | 高相关性 | 需训练 |
| FactCC / QAGS | 2020 | NLI / QA | 事实性 | 任务特定 |
| LLM-as-Judge | 2023 | GPT-4 打分 | 高相关性 | 偏差、成本 |
| Chatbot Arena | 2023 | 人类投票 | 真实偏好 | 慢、不可重现 |

文本生成评估的演进折射出 NLP 整体的进步：**从字面匹配 → 语义匹配 → 任务特定 → 人类偏好 → 多元评估**。当前（2026）的最佳实践：

1. **多指标组合**：n-gram + 语义 + 任务特定。
2. **LLM-as-Judge + 人类 spot check**：低成本与高可靠性的平衡。
3. **动态基准**：避免数据污染。
4. **公开 + 私有测试集**：既可重现又防过拟合。

掌握评估方法，你就能客观衡量 LLM 的真实进步，而非被营销数字迷惑。
