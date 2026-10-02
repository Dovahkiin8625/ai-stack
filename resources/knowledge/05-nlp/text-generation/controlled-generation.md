# 受控文本生成：从 PPLM 到 RLHF 的可控性技术

语言模型的"自由生成"经常与人类意图脱节——模型可能偏离主题、产生有害内容、风格不一致。受控文本生成（Controllable Text Generation）通过多种机制让模型**按指定属性生成**：主题、情感、风格、长度、关键词、结构。本文沿着 PPLM（2020）→ GeDi（2020）→ FUDGE（2021）→ RLHF（2022）→ Constitutional AI（2022）的时间线，剖析受控生成的核心技术与最新进展。

## 一、任务定义与挑战

### 1.1 任务形式

给定：

- **Prompt** $\mathbf{x}$：用户提供的内容或任务。
- **控制条件** $\mathbf{c}$：期望的属性（主题、情感、长度、关键词）。

生成：

$$
\mathbf{y}^* = \arg\max_{\mathbf{y}} \; P(\mathbf{y} \mid \mathbf{x}, \mathbf{c}) \cdot Q(\mathbf{y})
$$

$Q(\mathbf{y})$ 是质量项（如流利度），$P$ 是满足控制的条件概率。

### 1.2 三大挑战

1. **质量-控制 trade-off**：过度强调控制条件会导致生成质量下降（如生成不自然的文本）。
2. **属性耦合**：多个控制条件之间可能冲突（如同时要求"幽默"和"严肃"）。
3. **组合泛化**：训练时见过的属性组合，推理时如何处理未见过的新组合？

## 二、训练时控制：Conditional Training

### 2.1 最朴素方案

训练时把控制条件作为输入前缀：

```python
prompt = "情感:积极。主题:科技。文章:"
```

模型在 $\mathbf{x} = [\text{control\_prefix}; \text{content\_prompt}]$ 条件下训练。推理时只需指定控制前缀。

代表工作：

- **CTRL**（Keskar et al., 2019）：用控制码作为训练前缀，覆盖风格、领域、实体类型等。
- **Prefix-Tuning**（Li & Liang, 2021）：训练一段连续向量作为前缀，冻结原模型。
- **Prompt-Tuning**（Lester et al., 2021）：把前缀当作可学习参数。

优点：实现简单、训练快。缺点：每个新属性都要重新训练或微调。

### 2.2 指令微调（Instruction Tuning）

把所有任务统一为"指令-输出"格式：

```
[INST] 把以下句子翻译成英文：你好世界 [/INST]
Hello world
```

训练后模型学会了**按指令控制输出**。代表：

- **FLAN**（Wei et al., 2022）：1000+ 任务的指令微调。
- **T0**（Sanh et al., 2022）：多任务指令微调。
- **InstructGPT**（Ouyang et al., 2022）：人工标注指令 + GPT-3 微调。

指令微调是当前（2026）LLM 的标准训练步骤，但单独的指令微调不能保证**细粒度控制**（如情感、风格）。

## 三、解码时控制：PPLM 与 GeDi

### 3.1 PPLM（Plug and Play Language Models）

Dathathri et al．（2020）的 PPLM 在**不重新训练 LLM** 的情况下，通过 attribute model 引导生成：

核心步骤：

1. 用 LLM 生成 token。
2. 用 attribute model（属性分类器，如情感分类器）计算当前隐藏状态 $\mathbf{h}_t$ 的属性分数。
3. 计算 attribute loss 的梯度 $\nabla_{\mathbf{h}} L_{\text{attr}}$。
4. **沿梯度方向更新 $\mathbf{h}_t$**，然后继续生成。

直觉：让 LLM 在"语言模型流利度"与"属性控制"之间保持平衡。

$$
\mathbf{h}_t \leftarrow \mathbf{h}_t + \alpha \cdot \nabla_{\mathbf{h}} L_{\text{attr}}(\mathbf{h}_t)
$$

PPLM 的优点：

- 无需训练 LLM，只需训练小型 attribute model。
- 可以组合多个属性模型。

缺点：

- 慢：每步都要反向传播到 attribute model。
- 质量不稳定：梯度更新可能破坏流利度。

### 3.2 GeDi（Guided Diffusion）

GeDi（Krause et al., 2020）用**两个小型分类器**（positive / negative）引导生成：

$$
P(y \mid \text{context}) \propto P_{\text{LM}}(y \mid \text{context}) \cdot \left[ \frac{P_\theta(\text{class} = + \mid \text{context}, y)}{P_\theta(\text{class} = - \mid \text{context}, y)} \right]^\gamma
$$

$\gamma$ 是引导强度。

直觉：每步生成时，优先选择"positive 分类器认为符合属性、negative 分类器认为不符合属性"的 token。GeDi 比 PPLM 速度快约 10 倍。

### 3.3 FUDGE（Future Discriminators for Generation）

Yao et al.（2021）的 FUDGE 用**未来判别器**预测"如果生成 token $y$，后续文本属于某属性的概率"：

$$
P(y \mid \text{context}) \propto P_{\text{LM}}(y \mid \text{context}) \cdot P_\phi(\text{attr} \mid \text{context}, y)
$$

训练 FUDGE 时只需 attribute model，推理时每步重新计算。优点：

- 比 PPLM 快 5-10 倍。
- 可以集成多个属性判别器。

### 3.4 解码时控制的局限

PPLM、GeDi、FUDGE 都是**无训练或轻训练**的方法，优点是不改变 LLM，但缺点是：

- 每步都需额外计算，吞吐量下降 2-5 倍。
- 控制精度有限（受限于 attribute model 质量）。
- 多属性组合时易冲突。

## 四、训练时强化：RLHF

### 4.1 经典 RLHF 三阶段

Ouyang et al.（2022）的 InstructGPT 提出经典 RLHF 流程：

**Step 1：监督微调（SFT）**

人工写高质量（prompt, response）对，微调 LLM。得到 SFT 模型 $\pi_{\text{SFT}}$。

**Step 2：奖励建模（Reward Model）**

收集比较数据：同一 prompt 的多个回复，人类标注哪个更好。训练 reward model $r_\phi(\mathbf{x}, \mathbf{y})$：

$$
\mathcal{L} = -\log \sigma(r_\phi(\mathbf{x}, \mathbf{y}_w) - r_\phi(\mathbf{x}, \mathbf{y}_l))
$$

$\mathbf{y}_w$ 是更好的回复，$\mathbf{y}_l$ 是较差的。

**Step 3：PPO 强化学习**

用 PPO 算法优化：

$$
\max_\pi \; \mathbb{E}_{\mathbf{x}, \mathbf{y} \sim \pi} [r_\phi(\mathbf{x}, \mathbf{y})] - \beta \cdot \text{KL}(\pi \| \pi_{\text{SFT}})
$$

直觉：奖励高，但不要离 SFT 模型太远（KL 惩罚）。

### 4.2 RLHF 的核心优势

- **直接优化人类偏好**：不再依赖手工设计的属性损失。
- **细粒度控制**：能学到"什么算好回复"的细微差异。
- **可扩展**：增加新偏好只需更多比较数据。

### 4.3 RLHF 的局限

- **奖励黑客**（Reward Hacking）：模型学到了"如何骗过 reward model"而非"真正变好"。
- **训练不稳定**：PPO 调参复杂，KL 系数难定。
- **标注成本**：每条比较标注需 1-3 分钟人工。

## 五、RLHF 的现代化：DPO 与 SimPO

### 5.1 DPO（Direct Preference Optimization）

Rafailov et al.（2023）的 DPO 去掉 reward model，直接用偏好数据优化：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left( \beta \log \frac{\pi_\theta(\mathbf{y}_w \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_w \mid \mathbf{x})} - \beta \log \frac{\pi_\theta(\mathbf{y}_l \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_l \mid \mathbf{x})} \right)
$$

$\pi_{\text{ref}}$ 是参考模型（通常是 SFT 模型）。

直觉：让"好回复"的相对似然增长，"差回复"的相对似然降低。DPO 实质上是把 PPO 的"奖励 → 策略更新"两阶段压缩为单阶段直接优化。

DPO 的优势：

- 训练稳定（标准监督学习 + 交叉熵）。
- 无需训练 reward model。
- 效果接近 RLHF，但训练简单 10 倍。

### 5.2 SimPO（Simple Preference Optimization）

Meng et al.（2024）的 SimPO 进一步简化：

$$
\mathcal{L}_{\text{SimPO}} = -\log \sigma\left( \frac{\beta}{|\mathbf{y}|} \log \pi_\theta(\mathbf{y}_w \mid \mathbf{x}) - \frac{\beta}{|\mathbf{y}|} \log \pi_\theta(\mathbf{y}_l \mid \mathbf{x}) - \gamma \right)
$$

**去掉参考模型**，直接用长度归一化的对数概率差。$\gamma$ 是 margin。

SimPO 在 AlpacaEval、MT-Bench 等评测上达到或超越 DPO，且训练更简单。

### 5.3 IPO、KTO、ORPO 等变种

- **IPO**（Azar et al., 2023）：解决 DPO 在偏好过强时的过拟合。
- **KTO**（Ethayarajh et al., 2024）：用单个标注（好/坏）替代成对比较。
- **ORPO**（Hong et al., 2024）：结合 SFT 和偏好优化，无需参考模型。

## 六、Constitutional AI：自监督的可控生成

Anthropic 的 Constitutional AI（Bai et al., 2022）让模型**自我批评**：

**Step 1：监督学习**

模型对 red-team prompt 生成多个回复，按宪法原则（如"不要有害"、"不要歧视"）选出最好的。

**Step 2：RLAIF（RL from AI Feedback）**

让 AI 评估多个回复的好坏，生成偏好对，训练 reward model，再 PPO 训练。

直觉：用 AI 替代人类评估偏好，降低标注成本。Anthropic 在 Claude 系列中广泛使用 Constitutional AI。

## 七、Prompt Engineering 与结构化控制

### 7.1 System Prompt

现代 LLM 都支持 system prompt：

```python
messages = [
    {"role": "system", "content": "你是一个专业的科技作家，写作风格简洁严谨。"},
    {"role": "user", "content": "介绍一下 Transformer。"},
]
```

System prompt 不参与用户对话，但**引导模型的整体行为**。

### 7.2 Few-Shot 提示

在 prompt 中加入示例，让模型按示例格式生成：

```python
prompt = """将以下句子改写为正式风格。

例 1:
原文: 这个东西超好用！
改写: 此产品具有优秀的用户体验。

例 2:
原文: 这个方案不太行。
改写: 此方案尚需进一步完善。

原文: 这个想法贼有意思。
改写: """
```

Few-shot 是最简单有效的控制方法，但占用 token 多、灵活性差。

### 7.3 JSON Schema 与工具调用

让模型按结构化 schema 生成：

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "抽取以下文本的实体。"}],
    response_format={"type": "json_schema", "schema": {
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": {"type": "string"}},
            "types": {"type": "array", "items": {"type": "string"}}
        }
    }}
)
```

JSON schema 把"控制"从语义层降到结构层，确保输出可解析。

## 八、Adapter 与 LoRA 的细粒度控制

### 8.1 Conditional Adapter

为每个属性训练一个独立 Adapter（详见 [[lora-peft]]）：

- Adapter_Positive：让模型生成积极情感。
- Adapter_Negative：让模型生成消极情感。
- Adapter_Formal：让模型生成正式风格。

推理时按需加载：

```python
base_model + Adapter_Positive = "积极情感生成器"
base_model + Adapter_Formal = "正式风格生成器"
base_model + Adapter_Positive + Adapter_Formal = "积极且正式"
```

Adapter 的优势：

- 单一基座，多种控制。
- 训练成本低（每个 Adapter < 1% 参数）。
- 可组合（多个 Adapter 线性叠加）。

### 8.2 Prompt-Tuning 的细粒度控制

Lester et al．（2021）提出的 Prompt-Tuning 训练一段软前缀作为控制条件。每个属性对应一段软前缀，推理时拼接。

Li & Liang（2021）的 Prefix-Tuning 更进一步，训练每层的 prefix。

## 九、约束解码（Constrained Decoding）

### 9.1 关键词约束

生成时强制包含/避免某些词：

```python
from transformers import LogitsProcessor

class KeywordConstraint(LogitsProcessor):
    def __init__(self, keyword, must_appear=True):
        self.keyword = keyword
        self.must_appear = must_appear

    def __call__(self, input_ids, scores):
        if self.must_appear and self.keyword not in decode(input_ids):
            # 增加 keyword 第一个 token 的 logits
            scores[:, self.keyword[0]] += 5.0
        return scores
```

HuggingFace 的 `ForceWordsLogitsProcessor` 是这一思路的官方实现。

### 9.2 结构约束

让模型按 JSON、YAML、代码语法生成：

- **JSON Schema**：强制键值对必须符合 schema。
- **正则约束**：`outlines` 库支持正则表达式约束生成。
- **CFG（上下文无关文法）**：用语法指导解码（`guidance`、`lm-format-enforcer`）。

工具：

- **Guidance**：Microsoft 出品，支持 JSON schema + CFG。
- **Outlines**：支持正则 + JSON + CFG。
- **LMQL**：用类 SQL 语言描述生成约束。

## 十、特定任务的控制

### 10.1 情感控制

经典任务：把积极文本改写为消极（或反之）。

| 方法 | 优点 | 缺点 |
| --- | --- | --- |
| Prompt："改写为消极情感" | 简单 | 不稳定 |
| SFT + 情感标签 | 稳定 | 需标注数据 |
| GeDi / FUDGE | 无需训练 | 慢 |
| RLHF | 高质量 | 标注贵 |

### 10.2 主题控制

让模型按指定主题生成（如"科技"、"体育"）。

代表方法：

- **Topic-Prompt**：训练每个主题一个软前缀。
- **Topic-Conditioned Generation**：在 SFT 时加入主题标签。
- **Topical Attention**：强制 attention 集中到主题相关 token。

### 10.3 长度控制

让生成文本长度严格在指定范围内。

实现：

- **Length Token**：在 prompt 中加入 `<|len: 100|>`。
- **Soft Length Penalty**：解码时按剩余长度调整 logits。
- **Constrained Decoding**：截断或继续生成直到指定长度。

### 10.4 风格控制

模仿特定作者或文风（鲁迅、海明威、学术论文）。

代表工作：

- **Style Transfer**：从源风格转换到目标风格。
- **Author Style Learning**：训练时加入作者 ID。
- **Style-conditioned RLHF**：用风格标注训练 reward model。

## 十一、评估

### 11.1 自动指标

- **属性分类准确率**：用属性分类器评估生成文本的属性（情感、主题、风格）。
- **流利度**：用语言模型 perplexity 评估。
- **多样性**：Distinct-N、Self-BLEU。
- **任务完成率**：是否满足硬约束（包含关键词、长度在范围内）。

### 11.2 人类评估

- **Pairwise 比较**：A vs B，人类选择哪个更"符合控制条件"。
- **Likert 评分**：1-5 分制评估多个维度（流畅度、控制度、新颖度）。

## 十二、未来方向

1. **Multi-Turn Control**：多轮对话中持续控制属性，避免"prompt 漂移"。
2. **Personalized Control**：根据用户偏好动态调整控制策略。
3. **Self-Correcting Generation**：模型生成后自我评估、自我修正（Constitutional AI 思路）。
4. **Tool-Augmented Generation**：用工具（搜索、计算）辅助生成，提升可控性。

## 小结

| 时代 | 方法 | 核心思路 | 适用 |
| --- | --- | --- | --- |
| 2019 | CTRL / Conditional | 训练时控制 | 固定控制集 |
| 2020-21 | PPLM / GeDi / FUDGE | 解码时引导 | 无需训练 |
| 2021 | Instruction Tuning | 任务指令 | 通用控制 |
| 2022 | RLHF | 人类偏好 | 高质量通用 |
| 2023-24 | DPO / SimPO | 直接偏好优化 | 简化训练 |
| 2024+ | Constitutional AI | 自我评估 | 道德 / 安全 |

受控生成的核心演进：从"训练时嵌入控制" → "解码时引导" → "RLHF 直接优化偏好" → "自我批评"。当前（2026）的 SOTA 实践：**RLHF / DPO 训练 + 解码时约束 + 结构化输出（JSON）**。掌握这些技术，你就能构建"听话"的 LLM 应用。
