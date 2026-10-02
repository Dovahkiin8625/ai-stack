# RLHF 与对齐：从 SFT 到 DPO

SFT 让模型"会说"，但要让它"说得好"——更安全、更符合人类偏好——还需要**对齐（alignment）**。本文梳理从经典 RLHF 到 DPO / ORPO / KTO 的演化路径，配以 TRL 库的最小可运行代码，并指出几个常见的工程坑。

## 一、为什么要"对齐"

预训练 + SFT 之后的模型仍然存在几个问题：

- **有害输出**：可能被诱导说出不该说的话。
- **冗长啰嗦**：能答对但总要先讲一通套话。
- **指令不服从**：对 prompt 的细微差别反应差。
- **幻觉**：在不确定时仍"自信"地编造。

这些都不能只靠 SFT 解决，因为它们是**偏好的**而非**知识的**问题。RLHF（Reinforcement Learning from Human Feedback）就是用人类偏好信号来直接优化模型的"回答方式"。

## 二、经典 RLHF：SFT → RM → PPO

经典三阶段：

### 阶段 1：SFT（监督微调）

用人工撰写的高质量 (prompt, response) 样本对基础模型做监督微调。这一步产出 **SFT 模型**，相当于一个"听话的基线"。

### 阶段 2：Reward Model（RM）

对同一个 prompt 采样多个回答，由人类标注员**两两比较**哪个更好。训练一个 RM $r_\phi(x, y)$，让它对"人类更偏好的回答"打高分。

损失函数（Bradley-Terry 模型）：

$$
\mathcal{L}_{\text{RM}} = -\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

其中 $y_w$ 是被偏好回答，$y_l$ 是被拒绝回答。

### 阶段 3：PPO 强化学习

用 PPO（Proximal Policy Optimization）优化策略 $\pi_\theta$：

$$
\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x, y)\right] - \beta \cdot \text{KL}\big(\pi_\theta(y|x) \| \pi_{\text{SFT}}(y|x)\big)
$$

直观解释：

- 第一项：用 RM 打分，鼓励高分回答。
- 第二项：KL 散度惩罚，防止模型为了刷高分而"跑偏"到 RM 不熟悉的区域。

### PPO 的工程坑

RLHF 之所以"难"，主要难在 PPO：

1. **四个模型同时在显存里**：策略、参考策略、价值网络、奖励模型。7B 级别至少需要 4×14GB ≈ 56GB。
2. **训练不稳定**：reward hacking、KL 崩塌、policy collapse 是家常便饭。
3. **调参敏感**：学习率、KL 系数 $\beta$、reward 缩放……任何一个错都会让模型"变笨"或"变怪"。

## 三、DPO：直接偏好优化

DPO（Direct Preference Optimization）是一个"绕过 RM 和 PPO"的优雅替代。核心思路：

> 如果我们最终想要的是"模型在偏好数据上的似然"，为什么不**直接用偏好数据做监督学习**？

DPO 损失：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

直观解释：让偏好回答相对于参考模型的"似然比"**变大**，让拒绝回答的"似然比"**变小**。

相比 PPO 的优势：

- **不需要 RM**：省一个模型。
- **不需要在线采样**：训练和 SFT 一样稳定。
- **不需要 4 个模型**：策略 + 参考策略即可。

### DPO 代码示例（TRL）

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# 偏好数据：chosen / rejected pair
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

args = DPOConfig(
    output_dir="ckpts/qwen7b-dpo",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,                # DPO 一般比 SFT 低一个量级
    beta=0.1,                          # 温度，控制偏离 ref 的程度
    max_length=1024,
    bf16=True,
    logging_steps=20,
    save_strategy="epoch",
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)
trainer.train()
```

## 四、DPO 之后的演化

| 方法 | 核心改动 | 优点 |
| --- | --- | --- |
| **IPO** | 用平方损失替代 log-sigmoid，避免 DPO 的过拟合 | 数据噪声大时更稳 |
| **KTO** | 不需要成对偏好，单条回答 + 标"好/坏"即可 | 标注成本低 |
| **ORPO** | SFT + 偏好一体，淘汰参考模型 | 训练更简单 |
| **SimPO** | 用长度归一化的 reward | 更适合长文本 |

如果偏好数据稀疏，**KTO** 最省事；如果想一套流程跑完 SFT + 偏好，**ORPO** 最简洁；如果追求效果上限，**DPO / SimPO** 仍是主力。

## 五、避坑清单

1. **$\beta$ 不要太大**：会让模型"贴回"参考模型，学不动。
2. **$\beta$ 不要太小**：模型会"胡来"，生成乱码或刷分。
3. **chosen / rejected 长度差太大**：DPO 会退化成"长度优化器"。必要时做长度归一化或裁剪。
4. **参考模型用错了**：一定要用 **SFT 后的**模型当 ref，不能用 base 模型。
5. **reward hacking**：训练后期，模型可能学会用奇怪格式（极长、emoji 堆叠）骗 RM 分数。要在训练集和验证集上定期抽检。

## 小结

RLHF 是把"人类偏好"刻进模型的关键步骤。经典 RLHF（SFT → RM → PPO）效果上限高，但工程复杂；**DPO** 通过把偏好问题转化为监督学习，把门槛降到与 SFT 相当的水平，目前已是中小团队首选。下一篇会讨论 instruction 数据工程——再好的对齐算法，也得先有好数据。
