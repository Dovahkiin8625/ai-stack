# LoRA 与参数高效微调（PEFT）

全参数微调一个 7B 模型需要 60+ GB 显存，对个人开发者和小团队几乎不可行。**参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）**通过冻结大部分原始权重、只训练一小撮新参数，把显存需求压到消费级显卡也能承受的范围。本文重点讲解 LoRA 与 QLoRA，并给出一份能直接跑的代码示例。

## 一、为什么需要 PEFT

设预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$。全参数微调需要为 $W_0$ 的每个元素都存一份梯度与优化器状态，显存开销随参数线性增长。

PEFT 的核心观察是：**任务相关的权重变化往往处在低秩子空间**。也就是说，要拟合一个下游任务，并不需要真的去更新整个 $W_0$，只需要给 $W_0$ 加一个"小补丁"就够了。

## 二、LoRA：低秩适配

LoRA（Low-Rank Adaptation）把权重更新 $\Delta W$ 分解成两个低秩矩阵的乘积：

$$
W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A
$$

其中：

- $W_0 \in \mathbb{R}^{d \times k}$：**冻结**的预训练权重。
- $A \in \mathbb{R}^{r \times k}$，$B \in \mathbb{R}^{d \times r}$：可训练参数，$r \ll \min(d, k)$。
- $\alpha$：**缩放因子**，控制 adapter 对输出的影响强度。
- $r$（rank）：**秩**，越小越省参数，但表达力也越弱。

直觉上：$A$ 先把输入投影到 $r$ 维空间，$B$ 再把它投回原始维度，$W_0$ 的"主要信息"被冻结，只学一个"任务特定的偏移量"。

训练时只更新 $A, B$，$W_0$ 保持不变；推理时可以把 $BA$ 合并回 $W_0$，**不增加任何额外延迟**。

### 关键超参：$\alpha$ 和 $r$

- **$r$**：常见取 4、8、16、32、64。经验上 SFT 用 8–16 就够，需要学新语言或大幅改变风格时考虑 32–64。
- **$\alpha$**：和 $r$ 配对。`lora_alpha / r` 决定了 adapter 的有效"学习率"。常见的配比是 $\alpha = 2r$，例如 `r=16, alpha=32`。
- **target_modules**：在注意力层上一般选 `q_proj, k_proj, v_proj, o_proj`；要更强表达可以加 `gate_proj, up_proj, down_proj`（MLP 层）。

## 三、QLoRA：4bit 量化 + LoRA

QLoRA 在 LoRA 之上又压了一道：**把冻结的 $W_0$ 用 4bit 量化加载**。它由三个关键技巧组成：

1. **NF4（4-bit NormalFloat）**：一种对正态分布权重最优的 4bit 数据类型。比普通 INT4 信息损失更小。
2. **双量化（Double Quantization）**：对量化的常数再做一次量化，进一步省显存。
3. **分页优化器（Paged Optimizer）**：在 GPU 显存紧张时自动卸载到 CPU，避免 OOM。

效果：7B 模型从 ~14GB（FP16）降到 ~5GB（4bit），加 LoRA 训练总共 **~14 GB**，单张 RTX 4090 / A6000 即可跑。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 量化
    bnb_4bit_compute_dtype="bfloat16",   # 计算用 bf16
    bnb_4bit_use_double_quant=True,     # 双量化
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)  # 冻结 / 转 kbit 适配

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出类似：trainable params: 16,384,000 || all params: 7,628,000,000 || trainable%: 0.21%
```

## 四、其它常用 PEFT 方法

| 方法 | 思路 | 参数量级 | 适用场景 |
| --- | --- | --- | --- |
| **LoRA** | $W' = W + BA$ | 0.1% – 1% | 通用 SFT / 偏好学习 |
| **QLoRA** | LoRA + 4bit 加载 | 同上 | 显存紧、消费卡 |
| **AdaLoRA** | 自适应分配不同模块的 rank | 自适应 | 多任务、超大模型 |
| **IA³** | 给激活乘一个可学习向量 | 0.01% | 极致省显存、轻量适配 |
| **Prefix Tuning** | 在 K/V 前拼接可学习 prefix | 0.1% | 生成式任务 |
| **(IA)³ vs LoRA** | IA³ 训练更快、效果略差 | 0.01% | 极小数据场景 |

选择经验：

- **数据少 / 想快出活**：LoRA `r=8~16`，仅注意力层。
- **数据多 / 想深改风格**：LoRA `r=32~64`，加 MLP 层。
- **显存极紧张**：QLoRA + IA³。

## 五、显存对比（7B 模型，序列 2048）

| 方式 | 模型权重 | 优化器 | 激活 + 梯度 | 总计（近似） |
| --- | --- | --- | --- | --- |
| 全参数 FP32 | 28 GB | 56 GB | 8 GB | ~92 GB |
| 全参数 BF16 | 14 GB | 28 GB | 4 GB | ~46 GB |
| LoRA (BF16) | 14 GB | ~1 GB | 4 GB | ~19 GB |
| QLoRA (NF4) | ~5 GB | ~1 GB | 4 GB | ~10 GB |
| QLoRA + 8bit 优化器 | ~5 GB | ~0.5 GB | 4 GB | ~10 GB |

> 经验数字，实际随 batch size、序列长度、是否 gradient checkpointing 浮动。

## 六、合并与部署

训练完后有两种使用方式：

```python
# 方式 1：保存 adapter，推理时动态加载
model.save_pretrained("ckpts/qwen7b-lora")

# 方式 2：把 LoRA 合并回基础模型，得到一个独立的 checkpoint（部署更方便）
merged = model.merge_and_unload()
merged.save_pretrained("ckpts/qwen7b-merged", safe_serialization=True)
```

合并后的模型与原模型形状一致，可以用 `vllm`、`sglang`、`TGI` 等推理框架零改造部署。

## 小结

LoRA 通过把权重更新约束在低秩空间，把可训练参数压到 0.1%–1% 级别；QLoRA 进一步把冻结部分量化到 4bit，使 7B 级别的微调在单卡上可行。$\alpha$ 与 $r$ 是核心旋钮，target_modules 决定覆盖深度。下一篇会进入 RLHF / DPO——用偏好数据继续把模型对齐到人类意图上。
