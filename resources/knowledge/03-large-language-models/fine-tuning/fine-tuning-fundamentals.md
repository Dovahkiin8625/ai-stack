# Fine-tuning 基础：什么时候微调、数据准备与训练流程

预训练模型已经是"通才"，但要让它在垂直任务上稳定发挥，往往还差一步：**微调（fine-tuning）**。本文回答三个最常被问到的问题——什么时候该微调、数据要如何准备，以及用 HuggingFace Transformers 跑一次完整微调的最小代码长什么样。

## 一、什么时候应该微调

微调不是免费的午餐。它需要数据、算力和工程时间。下图给出一个常见的**决策树**：

```
问 1: 模型已经能回答，但是答得不准 / 不稳定？
  ├── 否 → 用 prompt engineering 改写 prompt。
  └── 是
       ├── 问 2: 答案需要"私有 / 实时"的事实？
       │    ├── 是 → 先 RAG（外挂知识库），再看是否需要微调。
       │    └── 否
       │         ├── 问 3: 错误是"格式 / 风格"问题（如不输出 JSON、爱啰嗦）？
       │         │    └── 是 → SFT（小规模指令微调）通常就够了。
       │         └── 问 4: 错误是"复杂推理 / 知识深度"问题？
       │              └── 是 → 考虑更大模型，或先 prompt + RAG，实在不行再考虑全参数微调。
       └── 问 5: 数据是否能稳定更新？
            ├── 否 → 微调会很快过时，优先 RAG。
            └── 是 → 进入微调流程。
```

简而言之：

- **Prompt engineering** 成本最低，优先尝试。
- **RAG** 解决"知识"问题，不改变模型行为。
- **SFT**（监督微调）解决"格式 / 风格 / 简单任务对齐"。
- **RLHF / DPO**（见下一篇）解决"复杂偏好 / 安全"问题。
- **全参数微调** 只在数据、算力、目标都清晰时再做。

## 二、数据准备：质量 > 数量

数据集是微调最关键的旋钮。先记住几条经验法则：

1. **数量**：简单的格式对齐任务 500–2k 条样本即可；领域任务一般 5k–50k 条；想要明显改变模型行为（如新语言、新领域）通常需要 50k+。
2. **质量**：宁可 1k 条人工精标的，也不要 100k 条机器乱标的。噪声数据是过拟合和能力退化的最大来源。
3. **多样性**：覆盖目标场景下的各种输入形态（长短、领域、口吻）。只见过一种问法的模型，遇到变体就崩。
4. **去重 / 去污染**：训练集不能出现评测集原题或近似题；同一任务的样本之间要避免逐字重复（不然模型会"背样本"）。

数据集格式上，**指令微调**一般采用 instruction / input / output 三元组：

```json
{
  "instruction": "把下面的中文句子翻译成英文。",
  "input": "今天天气不错，我们去公园散步吧。",
  "output": "The weather is nice today, let's take a walk in the park."
}
```

没有 `input` 时退化为 `instruction + output`。

### 划分 train / val / test

- **train**：训练参数。
- **val（开发集）**：挑超参、选 checkpoint、看曲线。
- **test**：最终汇报指标，**训练过程中严禁偷看**。

常见比例：8 : 1 : 1 或 9 : 0.5 : 0.5。如果总样本只有几百条，就用 5 / 折交叉验证。

### 过拟合检测

观察 val loss 与 train loss：

- 两者同步下降 → 健康。
- train loss 下降、val loss 上升 → **过拟合**，应早停 / 加正则 / 减 epoch。
- 两者都不动 → 学习率过大或模型太小。

## 三、训练流程：用 Transformers 微调一个 7B 模型

下面给出用 `transformers` + `trl` 微调一个 7B 级别模型的**最小可运行**示例。包含数据加载、LoRA 配置、Trainer 训练。

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

# 1. 加载模型与分词器
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    load_in_4bit=True,                # QLoRA: 4bit 量化加载
)

# 2. 准备数据：以 instruction / input / output 三元组为例
dataset = load_dataset("json", data_files="data/instructions.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

def format_example(ex):
    if ex.get("input"):
        prompt = f"### 指令:\n{ex['instruction']}\n\n### 输入:\n{ex['input']}\n\n### 回答:\n"
    else:
        prompt = f"### 指令:\n{ex['instruction']}\n\n### 回答:\n"
    return {"text": prompt + ex["output"] + tokenizer.eos_token}

dataset = dataset.map(format_example)

# 3. LoRA 配置（详见 lora-peft.md）
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# 4. 训练参数
args = TrainingArguments(
    output_dir="ckpts/qwen7b-sft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # 等效 batch=16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="none",
)

# 5. 训练
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,
)
trainer.train()
```

> 显存估算：7B 模型 + QLoRA 4bit + 序列 2048 + batch=2 ≈ **14 GB** 显存（A100 / 4090 24G 可跑）；如要去掉量化、跑全参数微调，预算会到 **60 GB 以上**。

## 四、最容易踩的几个坑

1. **学习率用错了量级**。全参数微调用 `1e-5 ~ 5e-5`，LoRA 用 `1e-4 ~ 3e-4`，跨量级混用几乎一定翻车。
2. **对话模板不一致**。训练时用 ChatML，推理时换 Llama-3 prompt，输出格式会乱。训练与推理必须用**同一套 chat template**。
3. **不设 early stop**。Loss 早期见好就收，过几个 epoch 就会过拟合到指令措辞上。
4. **评测集被污染**。模型"高分"是因为见过原题，不是真的学会。
5. **不保存 tokenizer / generation_config**。换环境后才发现 `<|im_end|>` 之类特殊 token 不见了。

## 小结

微调的本质是**用一个相对小的、领域相关的数据集，把通用模型的行为微调到一个具体目标**。先决定要不要微调（prompt → RAG → SFT → RLHF），再保证数据质量与多样性，最后用一致的训练 / 推理模板与合适的超参跑完整流程。下一篇会深入 LoRA / QLoRA 等参数高效微调方法，把显存需求从 60GB 降到 16GB。
