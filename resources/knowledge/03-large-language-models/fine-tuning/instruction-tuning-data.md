# Instruction Tuning 与数据工程

"Garbage in, garbage out"——再先进的对齐算法，没有高质量的指令数据也出不来好模型。本文系统讲解主流指令格式、Self-Instruct 数据合成、质量过滤与去污染，并给出一条完整的指令数据构造 pipeline。

## 一、主流指令数据格式

指令微调的标准形式是 **(instruction, input, output)** 三元组，主流数据集大致可以分为三类。

### 1. 单轮指令（Alpaca 风格）

Stanford Alpaca 是最先流行的开源指令数据集，每条样本包含指令、可选输入、模型输出：

```json
{
  "instruction": "把下列句子改写成更正式的书面语。",
  "input": "我觉得这个方案还行，可以试试。",
  "output": "我认为该方案具有可行性，建议予以试行。"
}
```

特点：单轮、长度短、覆盖广。

### 2. 多轮对话（ShareGPT 风格）

ShareGPT 把 ChatGPT 的真实多轮对话导出，天然带角色与上下文：

```json
{
  "conversations": [
    {"from": "human", "value": "帮我写一个快排"},
    {"from": "gpt",   "value": "好的，下面是 Python 实现：..."},
    {"from": "human", "value": "能不能加上注释？"},
    {"from": "gpt",   "value": "当然，每行都加上说明：..."}
  ]
}
```

特点：多轮、上下文关联、贴近真实使用。

### 3. 大规模对话（UltraChat / OpenHermes）

- **UltraChat**：覆盖多种话题与风格，超过百万轮。
- **OpenHermes**：基于多个开源数据集混合，质量过滤更严。
- **WizardLM Evol-Instruct**：用 LLM 把简单指令"进化"成复杂变体（加约束、加步骤、加深度）。

## 二、数据合成：Self-Instruct 与 Evol-Instruct

人工标注贵且慢，主流做法是**让 LLM 自己造数据**，再由人或规则过滤。

### Self-Instruct 流程

```
            ┌─────────────────────────────┐
            │   1. 准备种子任务（~100 条）  │
            └──────────────┬──────────────┘
                           ▼
            ┌─────────────────────────────┐
            │ 2. LLM 基于种子生成新指令    │
            │    (批量、多种子采样)        │
            └──────────────┬──────────────┘
                           ▼
            ┌─────────────────────────────┐
            │ 3. LLM 给每条指令写回答      │
            └──────────────┬──────────────┘
                           ▼
            ┌─────────────────────────────┐
            │ 4. 规则 + 模型过滤           │
            │    (去重 / 长度 / 质量 / 安全) │
            └──────────────┬──────────────┘
                           ▼
            ┌─────────────────────────────┐
            │ 5. 合并种子 → 扩充训练集      │
            └─────────────────────────────┘
```

### Evol-Instruct

Self-Instruct 生成的指令偏简单。Evol-Instruct 在此之上加 **"进化"步骤**——把一条简单指令改写成复杂变体：

- **加约束**："写一首诗" → "写一首七言绝句，押韵，主题是秋日黄昏"。
- **加步骤**："解释 X" → "分三步详细解释 X，并各举一例"。
- **加深度**："介绍 Y" → "对比 Y 和 Z 的异同，结合 2020 年以来的研究"。

效果：用 Evol-Instruct 训练出的 WizardLM 在多个 benchmark 上明显超过 Self-Instruct 基线。

## 三、质量 vs 数量：经验法则

行业经验大致如下：

| 数据量级 | 适配场景 | 关键 |
| --- | --- | --- |
| **< 1k** | 极小数据微调，格式对齐 | 几乎要 100% 人工精标 |
| **1k – 10k** | 单一垂直任务 | 70% 人工 + 30% LLM 合成 |
| **10k – 100k** | 多任务 / 多风格 SFT | 大量 LLM 合成 + 严格过滤 |
| **> 100k** | 通用对话基座 | 多源混合 + 启发式清洗 |

经验法则：

- **质量先于数量**：5k 条精标数据 > 100k 条脏数据。
- **多样性 ≥ 长度**：覆盖 50 种任务类型 > 1 种任务的 50 个变体。
- **去重必须做**：同一指令的不同 paraphrase 太多会引入模式坍缩。

## 四、清洗与过滤 pipeline

下面是一条常用的 **构造 pipeline** 伪代码，涵盖合成 → 过滤 → 划分 → 验证：

```python
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import hashlib, re

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def gen_with_llm(prompts: list[str]) -> list[str]:
    """调用大模型批量生成（vllm / OpenAI / DeepSeek ...）"""
    ...

def is_low_quality(text: str) -> bool:
    if len(text) < 20 or len(text) > 4000:       # 长度过滤
        return True
    if re.search(r"(http|www\.|<\s*script)", text, re.I):  # 简单安全过滤
        return True
    if text.count("\n") / max(len(text), 1) > 0.3:  # 行密度过高（多为 boilerplate）
        return True
    return False

def dedup(records: list[dict]) -> list[dict]:
    """基于 instruction 的 n-gram 哈希去重"""
    seen, out = set(), []
    for r in records:
        h = hashlib.md5(r["instruction"].lower().encode()).hexdigest()
        if h in seen: continue
        seen.add(h); out.append(r)
    return out

# Step 1: 用 Self-Instruct 生成原始数据
seeds = load_dataset("json", data_files="seeds.jsonl", split="train")
new_instructions = gen_with_llm([
    f"请基于以下示例生成 5 条相似但不同的中文指令：\n{s['instruction']}"
    for s in seeds
])

# Step 2: 给每条指令生成回答
records = [
    {"instruction": inst, "output": ans}
    for inst, ans in zip(new_instructions, gen_with_llm(new_instructions))
]

# Step 3: 过滤
records = [r for r in records if not is_low_quality(r["instruction"] + r["output"])]
records = dedup(records)

# Step 4: 去污染（去掉与评测集近似的问题）
def contaminated(r, eval_set_keywords):
    return any(k in r["instruction"] for k in eval_set_keywords)

eval_keywords = ["MMLU", "GSM8K", "IFEval", "..."]   # 评测集关键词
records = [r for r in records if not contaminated(r, eval_keywords)]

# Step 5: 划分 train / val / test
dataset = Dataset.from_list(records).train_test_split(test_size=0.05, seed=42)
val_test = dataset["test"].train_test_split(test_size=0.5, seed=42)
dataset["val"], dataset["test"] = val_test["train"], val_test["test"]
dataset.save_to_disk("data/instruct_v1")
```

## 五、多轮对话与长度过滤

多轮对话要小心两件事：

1. **长度爆炸**：把整个对话拼起来后，可能超过模型的最大上下文。要按 token 长度（如 4096）截断。
2. **角色混淆**：训练与推理的 chat template 必须一致。常见 ChatML / Llama-3 / Mistral 各自不同，混用会让模型"忘角色"。

```python
def format_chatml(messages, tokenizer):
    text = ""
    for m in messages:
        if m["role"] == "user":
            text += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
        else:
            text += f"<|im_start|>assistant\n{m['content']}<|im_end|>\n"
    return text
```

## 小结

数据工程是 SFT / DPO 的胜负手：格式决定能不能跑，质量决定上限，多样性决定泛化。Self-Instruct + Evol-Instruct + 严格过滤，已经可以低成本地构造 10 万级的高质量指令集；关键是**不要跳过过滤和去污染这两步**，否则再大的数据量都是噪声。下一篇会讨论如何评估微调后的模型——避免"loss 降了但能力没涨"的错觉。
