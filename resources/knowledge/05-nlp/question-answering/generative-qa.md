# 生成式问答：从 T5 到指令微调的开放式 QA

抽取式 QA 假设答案必是原文片段，但现实中许多问题需要**综合、抽象、推理**才能回答——"为什么地球是圆的？"、"如何学习机器学习？"、"总结这篇文章"。生成式问答（Generative QA）让模型自由生成答案，从 RAG 到指令微调 LLM，这一范式正在成为现代 QA 系统的事实标准。本文深入剖析 T5、BART、指令微调、FLAN、ChatGPT 的生成式 QA 实现，以及其在开放域、对话、多模态场景的应用。

## 一、从抽取到生成：为什么需要转变

### 1.1 抽取式 QA 的局限

| 局限 | 例子 |
| --- | --- |
| 答案不在原文中 | "今天的天气"（无段落） |
| 答案跨多个 span | "总结这个文档" |
| 答案需要抽象 | "这篇文章的主旨是什么" |
| 答案需要推理 | "为什么量子力学很难" |

抽取式 QA 假设答案 $a \in \text{paragraph}$，但**许多真实问题不满足这一假设**。

### 1.2 生成式 QA 的优势

- **答案自由**：可以综合多个段落 / 多篇文档。
- **抽象能力**：可以总结、归纳、推理。
- **多任务统一**：QA、摘要、对话都能用"输入文本 → 输出文本"。

### 1.3 三大挑战

1. **幻觉（Hallucination）**：模型生成看似合理但错误的内容。
2. **可验证性**：难以判断答案是否正确。
3. **长度控制**：不同问题需要不同长度的答案。

## 二、Seq2Seq QA：从 LSTM 到 Transformer

### 2.1 SQuAD 1.1 上的早期工作

经典的生成式 QA 方法用 Seq2Seq 架构：

1. Encoder 编码段落 + 问题。
2. Decoder 自回归生成答案。

但直接 Seq2Seq 在 SQuAD 上不如抽取式，因为 Seq2Seq 容易"生成错误细节"。

### 2.2 Pointer-Generator Network

See et al.（2017）的 Pointer-Generator 混合"复制"和"生成"：

$$
P(w) = p_{\text{gen}} P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{i: w_i = w} a_i
$$

$p_{\text{gen}}$ 是"生成 vs 复制"的开关，$a_i$ 是 attention 权重。

直觉：模型可以**复制原文片段**（如专有名词），也可以**自由生成**（如连词、抽象词）。

Pointer-Generator 在摘要和 QA 上都取得突破，特别适合"答案部分在原文、部分需要总结"的场景。

## 三、BART / T5：统一 Seq2Seq 预训练

### 3.1 BART（Lewis et al., 2020）

BART 是 denoising autoencoder 预训练的 Seq2Seq 模型：

**预训练任务**：

- **Token masking**：随机 mask token。
- **Token deletion**：随机删 token。
- **Text infilling**：用单个 mask 替换连续 span。
- **Sentence permutation**：打乱句子顺序。
- **Document rotation**：选 token 作为文档开头。

Encoder 类似 BERT，Decoder 类似 GPT。BART 在生成任务上特别强。

### 3.2 T5（Raffel et al., 2020）

T5 把**所有 NLP 任务统一为 text-to-text**：

```
翻译："translate English to German: That is good." → "Das ist gut."
分类："cola sentence: The course is jumping well." → "acceptable"
QA："question: What is the capital of France? context: ..." → "Paris"
摘要："summarize: <article>" → "<summary>"
```

预训练用 span corruption：

- 随机 mask 15% 的 span（平均长度 3）。
- Decoder 预测被 mask 的 token。

T5 在生成任务上的优势：

- **统一框架**：一个模型处理所有任务。
- **可扩展**：11B 版本在 SuperGLUE 超过人类。
- **指令友好**：用 prefix 区分任务。

### 3.3 BART/T5 在 QA 上的应用

#### (a) 抽象 QA

给定段落 + 问题，BART/T5 生成答案：

```
输入：context: [段落] question: [问题]
输出：[自由生成的答案]
```

训练数据：

- **Natural Questions**：把多个参考答案合并 / 抽象。
- **NarrativeQA**：从故事中生成答案。
- **TriviaQA**：自由生成。

#### (b) 多文档 QA

把多个段落拼接输入，模型综合生成答案：

```python
inputs = " ".join([f"context{i}: {ctx}" for i, ctx in enumerate(contexts)])
inputs += f" question: {question}"
```

代表工作：

- **FiD（Fusion-in-Decoder）**：每个段落独立编码，Decoder 融合。
- **RAG**：检索 + 生成端到端（详见 [[open-domain-qa-rag]]）。

#### (c) HotpotQA 等多跳 QA

用生成式模型综合多个段落的答案：

```
输入：context1: ... context2: ... context3: ...
     question: ...
输出：基于多段落的推理答案 + 解释
```

## 四、指令微调 LLM 的 QA 范式

### 4.1 FLAN：多任务指令微调

Wei et al.（2022）的 FLAN 在 T5 上做大规模指令微调：

- **60+ 数据集**，覆盖 12 类任务。
- **每个数据集多个 prompt 模板**。
- 训练后 T5 能 zero-shot 完成新任务。

QA 任务在 FLAN 中以多种形式出现：

```
"question: [问题] context: [段落]" → "answer"
"answer this question: [问题]" → "[答案]"
"根据以下内容回答问题：[段落] 问题：[问题]" → "[答案]"
```

### 4.2 InstructGPT / ChatGPT

Ouyang et al．（2022）的 InstructGPT 三阶段训练：

1. **SFT**：人工写高质量问答对微调 GPT-3。
2. **Reward Model**：人类对多个回复排序，训练 reward model。
3. **PPO**：用 RL 优化"人类偏好"。

ChatGPT 进一步加入 RLHF + 多轮对话优化。

### 4.3 现代生成式 QA 的最佳实践

```python
import openai

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个专业的 AI 助手，回答问题时请引用来源。"},
        {"role": "user", "content": "量子力学的主要奠基人有哪些？他们的核心贡献是什么？"}
    ],
    temperature=0.3,
    max_tokens=800,
)
```

工程要点：

- **temperature=0-0.3**：QA 强调准确性，避免随机性。
- **引用来源**：在 prompt 中要求模型标注引用段落，降低幻觉。
- **多轮澄清**：用户问题不明确时主动反问。

## 五、长文档生成式 QA

### 5.1 挑战

整本书 / 整份文档（百万 token）远超 LLM 的上下文窗口（GPT-4 Turbo 128K，Claude 3 Opus 200K）。

### 5.2 检索增强生成（RAG）

最主流的方案：

1. **段落切分**：把文档切分（典型 512 token）。
2. **Embedding 检索**：用 dense retriever 找 top-K 相关段落。
3. **Prompt 构造**：把 top-K 段落 + 问题拼接到 prompt。
4. **LLM 生成**：用 LLM 基于上下文生成答案。

代表工作：

- **REALM**（Guu et al., 2020）：端到端训练 retriever + reader。
- **DPR**（Karpukhin et al., 2020）：dual encoder 检索。
- **RAG**（Lewis et al., 2020）：BART + DPR 端到端。
- **Atlas**（Izacard et al., 2022）：retriever + LLM 联合训练。

详见 [[open-domain-qa-rag]] 一文。

### 5.3 长上下文 LLM

直接把文档装入超长上下文：

- **GPT-4 Turbo**：128K token。
- **Claude 3 Opus**：200K token。
- **Gemini 1.5 Pro**：1M token。
- **Yi-200K、LLaMA-3-400K**：开源超长模型。

超长上下文省去了检索，但成本高、精度未必更好（"中间丢失"问题）。

### 5.4 Hierarchical QA

层次化处理：

```
文档 → 章节摘要 → 段落 QA → 段落答案合并 → 文档级答案
```

每层独立处理，适合超长文档（书籍、技术手册）。

## 六、多模态生成式 QA

### 6.1 任务定义

输入包含图像 + 文本，输出自由生成答案：

- **VQA**（Visual QA）："图中有几只狗？"
- **DocVQA**：文档图像 + 文字，"这个表格的总收入是多少？"
- **ChartQA**：图表问答。

### 6.2 代表方法

- **VisualGPT / BLIP-2**：图像 encoder + LLM。
- **LLaVA**：视觉指令微调。
- **GPT-4V / Claude 3 Opus**：原生多模态。

代表评测：

- **OK-VQA**：需要外部知识的 VQA。
- **VQAv2**：基础 VQA。
- **DocVQA**：文档图像。

## 七、对话式 QA

### 7.1 任务定义

多轮对话中回答用户问题：

```
用户：中国的首都是什么？
助手：北京。
用户：那里有什么著名大学？
助手：北京有北京大学、清华大学...
```

### 7.2 关键技术

- **对话状态跟踪（DST）**：识别当前对话的意图与槽位。
- **历史建模**：把多轮对话拼接到上下文。
- **指代消解**：识别"那里"指代"北京"。
- **话题切换**：检测用户是否切换话题。

### 7.3 代表系统

- **ChatGPT / Claude / Gemini**：通用对话系统。
- **RAG + 对话**：检索增强的对话 QA。
- **MultiWOZ**：任务型对话标准评测。

## 八、幻觉问题与缓解

### 8.1 幻觉类型

| 类型 | 例子 |
| --- | --- |
| **事实幻觉** | 编造不存在的论文 / 数据 |
| **推理幻觉** | 推理链看似合理但实际错误 |
| **指代幻觉** | 错误的实体指代 |
| **时序幻觉** | 时间错误（如事件顺序） |

### 8.2 评估

- **TruthfulQA**：判断模型输出是否真实。
- **HaluEval**：幻觉分类与评估。
- **HHEM**（Hallucination Evaluation Model）：专门评估幻觉。

### 8.3 缓解方案

#### (a) 检索增强（RAG）

让模型基于真实文档生成，大幅降低事实幻觉。详见 [[open-domain-qa-rag]]。

#### (b) 引用标注

在 prompt 中要求模型标注引用：

```
基于以下文档回答问题。请在答案中标注引用。

文档：[段落]
问题：[问题]
答案：[答案（带 [1][2] 引用标记）]
```

模型必须给每个事实标注引用，便于验证。

#### (c) Self-Check

让模型在生成后**自我检查**：

```python
def answer_with_self_check(question, context):
    answer = llm(f"基于上下文回答：{question}\n上下文：{context}")
    check = llm(f"检查以下答案是否有事实错误：\n问题：{question}\n答案：{answer}")
    if "无错误" in check:
        return answer
    else:
        return llm(f"基于以下反馈重新回答：{check}")
```

#### (d) Constitutional AI

用 self-critique 让模型自己去除幻觉：

```
模型生成答案 → 模型评估"是否有幻觉" → 如有则重生成
```

代表：Anthropic Claude 的 Constitutional AI。

#### (e) 不确定性表达

训练模型在不确定时说"我不知道"：

```python
prompt = """基于以下文档回答问题。如果文档没有足够信息，请回答"我不知道"，不要编造。

文档：{context}
问题：{question}
答案："""
```

在 SQuAD 2.0 的不可答训练数据上微调，模型学会"承认不知道"。

## 九、领域特定生成式 QA

### 9.1 医疗 QA

- **MedQA**：USMLE 医学考试。
- **PubMedQA**：生物医学文献 QA。
- **ChatDoctor / Med-PaLM**：医疗 LLM。

关键技术：

- 检索 PubMed / UpToDate 等权威来源。
- 拒绝超出能力范围的医学建议。
- 引用专业医学指南。

### 9.2 法律 QA

- **LegalBench**：法律推理评测。
- **LawGPT / ChatLaw**：中文法律 LLM。

关键技术：

- 引用法条（"根据《合同法》第 X 条..."）。
- 不预测具体案件结果。
- 提示用户咨询专业律师。

### 9.3 金融 QA

- **FinQA**：金融表格推理。
- **BloombergGPT**：金融专用 LLM。

关键技术：

- 结合表格数据 + 文本。
- 计算精度（数值问题用代码执行）。
- 实时市场数据检索。

## 十、评测基准

### 10.1 Natural Questions（NQ）

Google 真实查询 + Wikipedia 段落。要么给 span 答案，要么"无答案"。

### 10.2 TriviaQA

650K 远程监督的 trivia 问题。评测开放域 QA。

### 10.3 ELI5

长答案（150 字以上）开放域问题，要求生成详细解释。

### 10.4 HotpotQA / 2WikiMultiHopQA

多跳 QA，需要综合多个段落。

### 10.5 Chatbot Arena / MT-Bench

实际对话评估，含 QA 任务。

### 10.6 当前 SOTA（2024-2025）

| 基准 | GPT-4 | Claude 3 Opus | Gemini 1.5 Pro |
| --- | --- | --- | --- |
| NQ | 87% | 86% | 85% |
| TriviaQA | 90% | 89% | 88% |
| HotpotQA | 70% | 72% | 68% |
| MT-Bench | 8.9 | 9.0 | 8.8 |

LLM 在大多数 QA 基准上已超过人类水平（NQ 上人类 ~85%）。

## 十一、工程经验

### 11.1 Prompt 设计

```python
qa_prompt = """你是一个专业的问答助手。请基于以下原则回答：

1. 只使用提供的信息回答，不要编造。
2. 如果文档没有答案，明确说"我不知道"。
3. 简洁但完整，避免冗余。
4. 用中文回答，除非用户用其他语言提问。

文档：
{context}

问题：{question}

答案："""
```

### 11.2 性能优化

- **缓存**：相同问题的答案可缓存（5-30 分钟）。
- **批处理**：多问题并行推理。
- **流式输出**：用户等待时先返回部分答案。
- **检索优先**：先检索再生成，比直接让 LLM 答更可靠。

### 11.3 多模型路由

```python
def smart_qa(question):
    if is_simple_factoid(question):
        return extractive_qa_model(question)      # 快、准
    elif needs_reasoning(question):
        return chain_of_thought_llm(question)     # CoT
    elif needs_external_info(question):
        return rag_system(question)              # RAG
    else:
        return general_llm(question)             # 通用
```

不同类型问题用不同模型 / 流程。

## 十二、未来方向

1. **Self-Improving QA**：模型从错误中自动学习。
2. **Multi-Agent QA**：多模型协作，互相验证。
3. **Causal QA**：理解因果关系而非表面相关。
4. **Personalized QA**：根据用户历史定制答案。
5. **Long-Horizon Reasoning**：处理需要多步、多文档、长时间推理的复杂问题。

## 小结

| 时代 | 方法 | 关键 | 局限 |
| --- | --- | --- | --- |
| 2017 | Pointer-Generator | 复制 + 生成 | 仍基于原文 |
| 2019-2020 | BART / T5 | 统一生成 | 需 SFT |
| 2022 | FLAN / InstructGPT | 指令微调 | 数据贵 |
| 2023+ | ChatGPT / Claude / GPT-4 | RLHF + 多任务 | 幻觉、成本 |

生成式 QA 是 LLM 时代的事实标准——它把"问答"从"找片段"扩展到"理解 + 推理 + 综合 + 生成"。当前（2026）的最佳实践：

1. **RAG 为基础**：减少幻觉。
2. **指令微调 LLM 为核心**：灵活处理多种问题。
3. **专业领域垂直化**：医疗 / 法律 / 金融的专用模型。
4. **多模型协作**：不同模型各司其职，互相验证。

下一篇我们将深入探讨 RAG 这一核心架构：从检索到生成的全栈优化。
