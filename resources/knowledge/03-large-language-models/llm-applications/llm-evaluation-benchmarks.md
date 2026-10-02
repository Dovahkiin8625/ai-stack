# LLM 评测基准：MMLU、SWE-bench 与中文对齐评测

如何判断一个新模型“真的更好”？靠人工排排坐打显然不可持续。LLM 评测基准是今天对比报告与仲裁分析的唯一语言。本文梳理主流 benchmark、各自的偏差与陷阱，并给出 lm-evaluation-harness 跑 MMLU 的最小命令。

## 一、为什么需要基准

三大动机：

1. **学术可比较**：A 在 MMLU 上 88%、B 在 85% 之间——可以为论文、PR 、市场带来可重复信号。
2. **训练 signal**：可在 RLHF、SFT、DPO 中以 benchmark 作为一个 objective。
3. **能力诊断**：评测拆分到领域的能力补集，揭示方向性提升。

但“基准也是有偏向的”：

- HumanEval 补全短，可能高估“未加重点”的能力。
- 训练集污染：热门模型见过/被评测预言过原题。
- 商用 / 闭源模型可能已针对基准做了“变种优化”。

## 二、通用知识与推理基准

### MMLU 与 MMLU-Pro

MMLU 涵盖 57 个学科（STEM、社科、人文等），题目都是 4 选 1：

- **MMLU**：原版 14k 题，难度跨度从初中到职业考试。
- **MMLU-Pro**：升级为 12k 题、10 选项、引入推理 trap、“避免随机猜测与单一调用记忆”。

### GSM8K / MATH

- **GSM8K**：8500 道中小学数学问答。需要多步推理。
- **MATH**：12.5k 题，难度高于 GSM8K，包含 LaTeX 表达。

表现提升路径：CoT prompting → self-consistency → specialized verifiers。

### ARC

ARC (AI2 Reasoning Challenge) 是面向科学常识推理的多项选择题。ARC-Challenge 是其中较难的子集。

## 三、代码基准

| 基准 | 适用场景 | 评测方式 |
| --- | --- | --- |
| HumanEval | 单函数 | pass@k |
| MBPP | 极简编程题 | pass@k |
| LiveCodeBench | 近期竞赛题 | pass@1 + hidden test |
| SWE-bench | 真实 GitHub issue | repo-level tests |
| RepoBench | 跨文件补全 | pass@1 |

代码基准的挑战是数据污染：HumanEval 原题在 Stack Overflow 上有答案，Free2023 复现都可能出现能“记住答案”的现象。LiveCodeBench 推广就是周期性更新题库。

## 四、中文基准

中文场景里以下点特别关键：

- **C-Eval**：52 个学科、13942 题。面向中国中考、高考、职业考试。
- **CMMLU**：67 个学科、11528 题。与 C-Eval 互补，中文为主、英文为辅。
- **MMCU**：多任务多领域的中文评测集。

中文基准里“原题被评测集仓”同样重要。

## 五、对齐 / 偏好基准

对齐是三个基准起不了作用的：

- **MT-Bench**：80 个多轮对话问题 + LLM-as-judge。打分错率排序。
- **AlpacaEval**：模型输出 vs 参考输出，由 GPT-4 打分。胜率作为指标。
- **Arena-Hard**：Chatbot Arena 的高频难问题。Elo rating 打分。

$$
\text{EloWinRate}_{model} = \frac{\sum_{\text{battles}} \mathbb{1}[\text{model wins}]}{|\text{battles}|}
$$

Elo Elo 能减轻排名偏好，但依然会被“多个胜诉”掩盖。

## 六、长上下文基准

随着 128k、200k 甚至 1M context 出现，“长上下文能力”评估被广泛重提：

- **LongBench**：双语、21 个数据集。LLM、前 RAG、检索、记号多种任务。
- **RULER**：13 个任务的“累计评测模式”，包括“在指定下文中查 N 个 marker”、“括号抽取”等。
- **Needle-in-a-Haystack**：单个 needle 在指定上下文中检索。
- **SCROLLS**：7 个长上下文任务。

注意：实测“长上下文”是“能检索但不总是能推理”——这是现行 RULER 类 benchmark 重点解决的。

## 七、领域评测集构造方法

为什么要“自造 benchmark”：

1. 现成基准无法覆盖业务场景。
2. 现有基准训练污染越来越重。
4. 业务领域需求未被现有 benchmark 覆盖。

设计原则：

- **覆盖面广**：从“场景 × 难点”二维交叉得到采样点。
- **不污染训练集**：避免使用可能被用于训练的样本。
- **可追踪溯源**：每道题有明确“来源 / 标注人 / 标注时间”。
- **静态 vs 动态混合**：静态保口护、动态拼市均。

## 八、用 lm-eval-harness 跑 MMLU 子集

```bash
# 安装
pip install lm-eval

# 跑 HuggingFace 上的任意模型 / 默认是 greedy
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16 \
    --tasks mmlu_high_school_computer_science,mmlu_anatomy \
    --num_fewshot 5 \
    --batch_size 8 \
    --output_path ./results/qwen7b_mmlu
```

跑完会产出 `results.json`，示例结构如下：

```json
{
  "results": {
    "mmlu_high_school_computer_science": {
      "acc,none": 0.78, "acc_stderr,none": 0.021
    },
    "mmlu_anatomy": {
      "acc,none": 0.71, "acc_stderr,none": 0.024
    }
  }
}
```

控制变量：温度为 0、固定 prompt、固定 few-shot，才能可重复。

## 九、LLM-as-Judge

很多“准对齐”任务（MT-Bench、AlpacaEval、Arena-Hard）依赖“LLM-as-judge”。代码思路：

```python
from openai import OpenAI
client = OpenAI()

JUDGE_TPL = """请严格对比 A、B 谁更好。
问题：{a_prompt}
A：{a_answer}
B：{b_answer}
只输出 A / B / Tie。"""

def judge(prompt, a, b):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": JUDGE_TPL.format(
            a_prompt=prompt, a_answer=a, b_answer=b)}],
        temperature=0,
    )
    return r.choices[0].message.content.strip()
```

LLM-as-judge 会引入三种偏差：

- **位置偏差**：A 总胜 B。随机换位取平均能消除。
- **长度偏差**：偏好更长答案。需 length-coverage。
- **自我偏好**：用 GPT-4 评判 GPT-4 输出会偏高自己。

## 十、避免“刷分”

发表/汇报中单一直滥用点：

1. **多设几个独立 benchmark，防止单个被漏。
2. **隐藏任务，明确可不可以刷分。
3. **人工验证子集。LLM-as-judge 与人工 judge 同时上，明确高一致性区间。
4. **公开评测脚本/参数，让结果可重复。

## 小结

LLM benchmark 是“度量衡”的本身。不打分本身也会被“刷分”。可信的评估需要在多个独立 benchmark 上跑、报告独立任务的 stderr、并有静态测评集作为“口护”。下一篇是这篇的“后续”：如何从 benchmark 出发优化 LLM 应用的单位成本。