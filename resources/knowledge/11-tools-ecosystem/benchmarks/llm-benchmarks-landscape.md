# LLM 评测基准全景：从 MMLU 到 Chatbot Arena

LLM 评测是判断模型能力的"度量衡"——但评测本身充满陷阱：数据污染让分数虚高、单一指标掩盖真实能力、静态基准无法跟上模型迭代。本文梳理主流评测基准的分类（知识、推理、代码、数学、人类偏好），剖析每个 benchmark 的设计意图与局限，并讨论 Chatbot Arena 等动态评测的兴起。

## 一、为什么 LLM 评测如此困难

LLM 评测面临三个根本性挑战：

1. **能力多维**：一个模型需要同时具备知识、推理、对话、代码、安全等多种能力——单一分数无法刻画。
2. **数据污染**：模型可能在预训练中见过测试题，导致分数虚高。GPT-4 在 HumanEval 上 67%，但 SWE-bench 仅 1.7%——差异来自评测任务的"训练可见度"。
3. **能力 vs 对齐**：模型可能"知道答案"但"不会好好回答"——这需要单独评测（遵循指令、对话友好、不幻觉）。

评测基准的设计直接影响我们对模型能力的判断。

## 二、知识类基准

### MMLU（Massive Multitask Language Understanding）

**57 个学科**、15978 道选择题，从小学数学到专业法学：

```python
# MMLU 题目示例
question = """A research group wants to study the effect of a new drug on blood pressure.
They randomly divide 200 patients into a treatment group and a control group.
The treatment group receives the new drug while the control group receives a placebo.
After 4 weeks, they measure the blood pressure of all patients.
What type of study is this?

A) Case-control study
B) Cohort study  
C) Cross-sectional study
D) Randomized controlled trial"""

# 选项 A/B/C/D，模型选 D
```

**特点**：覆盖广泛、答案固定、易自动化评分。**局限**：纯选择题无法评测生成能力；中文场景需用 CMMLU（中文版）。

### CMMLU（中国版）

涵盖 **67 个学科**，包括中医、中国法律、马克思主义等本土知识：

```python
# CMMLU 子领域
domains = [
    "Chinese Literature", "Chinese History", "Chinese Philosophy",
    "Traditional Chinese Medicine", "Chinese Law", 
    "Computer Science", "Mathematics", "Engineering",
]
```

### ARC、HellaSwag、TruthfulQA

- **ARC-Challenge**：7-9 年级科学推理，测试常识推理。
- **HellaSwag**：句子补全，看似简单但 LLM 早期只到 30%。
- **TruthfulQA**：专门测试幻觉——模型会编造看似合理但虚假的答案。

## 三、推理类基准

### GSM8K（Grade School Math）

**8K 个小学数学应用题**，需要多步推理：

```python
question = """Natalia sold clips to 48 of her friends in April, and then she sold half 
as many clips in May. How many clips did Natalia sell altogether in April and May?"""

# 模型需要：
# step 1: April = 48
# step 2: May = 48 / 2 = 24  
# step 3: total = 48 + 24 = 72
```

**关键评测能力**：多步算术推理。Chain-of-thought 让 GPT-3 从 ~10% 跃升到 ~50%。

### MATH

**12500 个高难度数学题**，含 AMC/AIME 风格竞赛题：

```python
# MATH 比 GSM8K 难得多
question = """Find the number of integers n, 1 ≤ n ≤ 100, such that 
n^2 + n + 1 is divisible by 7."""
# 答案需要完整推导过程
```

需要符号推理能力，单纯 next-token 预测很难做好。

### BBH（Big Bench Hard）

从 BIG-Bench 200+ 任务中挑出的 **23 个最难的**，模型在 chain-of-thought 之前表现差。

## 四、代码类基准

### HumanEval（OpenAI, 2021）

**164 个手写编程题**，每题有单元测试：

```python
def humaneval_problem():
    return {
        "prompt": '''
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    ''',
        "tests": [
            "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False",
            "assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True",
        ],
    }

# pass@1: 一次生成通过的概率
# pass@100: 生成 100 次至少一次通过的概率
```

**局限**：题量小（164 道）、题目简单、模型很快饱和。

### MBPP（Mostly Basic Python Problems）

**974 道**基础 Python 题，覆盖更广。

### LiveCodeBench

**持续更新**——从 LeetCode、Codeforces、AtCoder 收集新题，带时间戳防止污染：

```python
# LiveCodeBench 题目带时间戳
{
    "contest": "leetcode-2024-03",
    "date": "2024-03-15",
    "difficulty": "Medium",
    "prompt": "...",
    "tests": [...],
}
```

### SWE-bench（最难）

**真实 GitHub issue**——给模型 issue 描述 + 仓库，模型修改代码修复：

```python
{
    "issue": "TypeError when calling save() with a custom encoder",
    "repo": "django/django",
    "commit_before": "abc123",
    "commit_after": "def456",
    "test_patch": "...",
}
```

GPT-4 仅 **1.7% pass@1**，是当前最难、最真实的代码评测。SWE-Agent、OpenHands 等研发在 SWE-bench Verified 上达到 ~20%。

## 五、多任务与综合基准

### MMLU-Pro

MMLU 的**升级版**——10 选项、推理题、抵御纯猜测：

```python
# MMLU 原版: 4 选项, 25% 随机正确率
# MMLU-Pro: 10 选项, 10% 随机正确率 + 需推理
```

### Big-Bench

**200+ 任务**，覆盖翻译、问答、推理、常识等，由 450+ 研究者贡献。

### GLUE / SuperGLUE

NLP 经典基准，被 LLM 时代超越——GPT-4 在 SuperGLUE 上超人类。

## 六、人类偏好基准

### Chatbot Arena（LMSYS, 2023）

**真实用户盲测**——Elo 评分系统，类似国际象棋等级分：

```python
# 用户看到两个匿名回答，投票哪个更好
┌─────────────────┬─────────────────┐
│   Model A       │   Model B       │
│   response      │   response      │
│                 │                 │
│  ◯ A is better  │  ◯ B is better  │
│  ◯ Tie          │  ◯ Same         │
│  ◯ Both bad     │                 │
└─────────────────┴─────────────────┘

# 后台基于投票计算 Elo 排名
```

**优势**：抗数据污染（用户问什么不可预测）、测真实偏好。

**局限**：受提示工程影响；用户群体偏向英语+特定话题。

### MT-Bench

**80 道多轮对话**问题，GPT-4 作为评判（"LLM-as-judge"）：

```python
questions = [
    {"turn": 1, "question": "Write a story about a dragon"},
    {"turn": 2, "question": "Now add a twist ending"},
    # 模型需要在两轮都保持一致与高质量
]
```

### AlpacaEval

**805 道指令**，比较模型输出与 GPT-4 的胜率（自动化版本）。

## 七、多模态基准

### MMMU（多模态 MMLU）

跨学科的图像+文本选择题，覆盖艺术、设计、医学等。

### VQA、OK-VQA、ScienceQA

视觉问答类基础基准。

### MMBench、MMStar

中文与综合的多模态基准。

## 八、安全与对齐基准

### HH-RLHF（Hugging Face）

人类偏好数据，用于 RLHF 训练。

### Toxicity（Real Toxicity Prompts）

测试模型生成有毒内容的概率。

### BBQ（Bias Benchmark）

9 类社会偏见的问答评测。

## 十、数据污染：评测的"暗礁"

数据污染是 LLM 评测最大的隐患：

```python
# 假设 2023 年发布 HumanEval 测试集
# 2024 年的模型训练数据时间截止 2024 年初
# 那么模型可能见过 HumanEval 的题目，导致分数虚高

# 检测污染的方法
def detect_contamination(model_outputs, benchmark_questions):
    """通过题目字符串匹配、BERT 等大型指标检查输出与训练数据的相似度。"""
    suspicious = []
    for question, output in zip(benchmark_questions, model_outputs):
        # 方法 1: 精确字符串匹配
        if question.exact_text in output:
            suspicious.append(question)
        # 方法 2: 高 n-gram 重叠
        if ngram_overlap(question, output) > 0.8:
            suspicious.append(question)
    return suspicious
```

**缓解方式**：
- 持续更新的 benchmark（LiveCodeBench、Chatbot Arena）。
- 数据集标注发布时间戳，过滤训练时间之前的数据。
- 测试集分片（held-out 子集）。

## 九、评测方法论

### LLM-as-Judge

用 GPT-4 等强模型评判其他模型的输出：

```python
JUDGE_PROMPT = """Compare the following two responses to the question.
Response 1: {response_a}
Response 2: {response_b}
Which is better? Output "1", "2", or "tie"."""

# 优点: 自动化、低成本
# 局限: GPT-4 偏好 verbose 回答;对自身有偏好;对长回答不利
```

### 人类评估 vs 自动化：

| 维度 | 自动化 | 人类 |
|---|---|---|
| 成本 | 低 | 高 |
| 速度 | 快 | 慢 |
| 一致性 | 高 | 低 |
| 细微差异 | 弱 | 强 |

## 十二、小结

LLM 评测没有"金标准"基准——**MMLU 测知识广度、HumanEval 测代码能力、SWE-bench 测真实工程、Chatbot Arena 测人类偏好**。每个 benchmark 都只是模型能力的一个切片。工业实践中通常需要**多基准组合 + 内部业务评测**才能形成完整判断。下一篇我们将讨论**评测方法论本身**：如何设计一个能区分模型能力且不被污染的 benchmark？