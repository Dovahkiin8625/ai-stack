# 去偏见技术：从数据增强到解码干预

理解偏见来源（bias-types）与公平性指标（fairness-metrics）之后，下一步是**缓解**。LLM 时代去偏见技术在三个层面展开：**数据层**（让训练数据更均衡）、**训练层**（让模型表征更中立）、**推理层**（让输出更可控）。本文系统梳理每一层的代表方法，给出代码示例，并讨论**去偏见的根本局限**——很多时候它只是把"显性偏见"变成"隐性偏见"。

## 一、去偏见的三个层级

```text
┌─────────────────────────────────────────────────────┐
│  数据层（Pre-processing）                              │
│  重新采样 / 增广 / 标签修正 / 公平 prompt 设计            │
└─────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────┐
│  训练层（In-processing）                               │
│  正则项 / 对抗训练 / RLHF 公平约束 / DRO                 │
└─────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────┐
│  推理层（Post-processing）                             │
│  去偏见解码 / 输出过滤 / self-debias prompting          │
└─────────────────────────────────────────────────────┘
```

每一层都有代价：**数据层便宜但效果有限；训练层有效但成本高；推理层灵活但难通用**。

## 二、数据层去偏见

### 1. 数据平衡（Re-balancing）

最简单的策略是**欠采样/过采样**让各群体在训练集中均衡：

```python
import random
from collections import defaultdict

def rebalance_dataset(examples, group_key="gender"):
    """
    把 examples 按 group_key 分组，对少数群体过采样到多数群体规模。
    """
    buckets = defaultdict(list)
    for ex in examples:
        buckets[ex[group_key]].append(ex)
    
    max_size = max(len(v) for v in buckets.values())
    balanced = []
    for group, items in buckets.items():
        # 重复采样补齐
        balanced.extend(items)
        n_extra = max_size - len(items)
        balanced.extend(random.choices(items, k=n_extra))
    
    random.shuffle(balanced)
    return balanced
```

**代价**：过采样少数群体可能**过拟合到该群体的表面模式**。

### 2. 反事实数据增广（Counterfactual Data Augmentation, CDA）

Lu et al. (2020) 在性别偏见缓解中提出：**把"男/女"代词互换，生成新样本**：

```python
GENDER_PAIRS = [
    ("he", "she"), ("him", "her"), ("his", "hers"),
    ("himself", "herself"), ("男", "女"), ("他", "她"),
    ("先生", "女士"), ("Mr.", "Ms."),
]

def swap_gender(text: str) -> str:
    """把文本中的性别词替换成对偶词。"""
    new_text = text
    for a, b in GENDER_PAIRS:
        new_text = new_text.replace(a, f"__TMP_{b}__").replace(b, a).replace(f"__TMP_{b}__", b)
    return new_text


# 例子
print(swap_gender("The doctor said he would help the patient."))
# "The doctor said she would help the patient."
```

把 `(原始, 交换后)` 都加入训练，模型被迫学到"性别与职业的虚假关联是不必要的"。

**局限**：只对**显性偏见**（代词、称谓）有效，对**隐性偏见**（"CEO 走路自信"这类隐性男性化描述）无效。

### 3. 公平 Prompt 设计

让标注员或合成数据生成器**使用多样化的人物设定**：

```python
DIVERSE_PROMPTS = [
    "Write a story about a {profession}. Use a {gender} {pronoun} as the main character.",
    "Recommend a book for a {age}-year-old {gender} who enjoys {interest}.",
    ...
]
```

填充时**均匀采样**各种 (profession, gender, age, ethnicity) 组合，让训练数据本身多样。

### 4. 偏见词表过滤

维护一个**敏感词 / 刻板短语**词表，过滤或替换训练样本中的对应片段：

```python
STEREOTYPE_PATTERNS = [
    (r"\b(aggressive|emotional|irrational)\s+(woman|girl)\b", "person"),
    (r"\b(thug|criminal)\s+(black|African[- ]American)\b", "person"),
    (r"\b(terrorist|extremist)\s+Muslim\b", "person"),
]

import re

def filter_stereotypes(text: str) -> str:
    for pattern, replacement in STEREOTYPE_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text
```

**局限**：词表更新永远追不上语言演变，且容易过度修正。

## 三、训练层去偏见

### 1. Sentence-Debias（投影去偏）

Liang et al. (2020) 提出：**识别 bias subspace，把表征投影到正交子空间**：

```python
import torch
import numpy as np
from sklearn.decomposition import PCA

def identify_bias_direction(embeddings_a, embeddings_b):
    """
    给定"he"与"she"的词向量集合，找出 bias direction。
    假设：(mean(b) - mean(a)) 即为 bias 主方向。
    """
    mean_a = np.mean(embeddings_a, axis=0)
    mean_b = np.mean(embeddings_b, axis=0)
    bias_direction = mean_b - mean_a
    return bias_direction / np.linalg.norm(bias_direction)


def project_away(embedding, bias_direction):
    """把单个向量减去 bias_direction 上的投影。"""
    proj = np.dot(embedding, bias_direction) * bias_direction
    return embedding - proj
```

对 LLM 应用时，可以：
1. 收集所有"性别词"的 token embedding。
2. 计算 bias direction。
3. 在模型推理前**修改 embedding**（减去 bias 投影）。

**局限**：需要事先知道"偏见方向"；但 bias 经常是**多方向**的；过度投影会损伤语义。

### 2. 对抗去偏（Adversarial Debiasing）

Zhang et al. (2018) 加一个**对抗器**试图从表征预测受保护属性，**主模型则学让对抗器失败**：

```python
class DebiasModel(nn.Module):
    def __init__(self, encoder, classifier, adversary):
        super().__init__()
        self.encoder = encoder     # 主模型
        self.classifier = classifier
        self.adversary = adversary  # 试图预测群体

    def forward(self, x):
        h = self.encoder(x)
        y = self.classifier(h)
        g_pred = self.adversary(h)
        return y, g_pred


def train_step(model, x, y_true, g_true, lambda_adv=0.5):
    y_pred, g_pred = model(x)
    
    # 主任务损失
    task_loss = F.cross_entropy(y_pred, y_true)
    # 最大化对抗器损失（梯度反转层实现）
    adv_loss = F.cross_entropy(g_pred, g_true)
    
    # 让主模型 loss 最小化，对抗器 loss 最大化
    loss = task_loss - lambda_adv * adv_loss
    loss.backward()
    optimizer.step()
    return task_loss.item(), adv_loss.item()
```

数学形式：$\min_\theta \max_\phi \; \mathcal{L}_{\text{task}}(\theta) - \lambda \mathcal{L}_{\text{adv}}(\theta, \phi)$——梯度反转层（GRL）让它能正常反传。

**局限**：
- 训练不稳定（min-max 博弈）。
- 对抗器只能"已知"的群体属性；如果有未列出的群体，仍然有偏见。
- LLM 规模下训练成本极高。

### 3. DRO（Distributionally Robust Optimization）

不是去偏见，而是**让模型在最差的群体子分布上表现良好**：

$$
\min_\theta \; \sup_{q \in \mathcal{Q}} \; \mathbb{E}_{(x,y) \sim q}[\mathcal{L}(\theta; x, y)]
$$

其中 $\mathcal{Q}$ 是与真实分布"接近"的分布集合（通常用 KL 散度约束）。直观：**不要只为"主流群体"优化，也要照顾少数群体**。

```python
def dro_loss(model, batches_by_group, eta=0.1):
    """
    batches_by_group: dict[group] -> list of batches
    用对偶变量实现 DRO。
    """
    losses = {g: compute_avg_loss(model, b) for g, b in batches_by_group.items()}
    # 最大加权
    weights = torch.softmax(torch.stack(list(losses.values())) / eta, dim=0)
    worst_case = sum(w * l for w, l in zip(weights, losses.values()))
    return worst_case
```

**优点**：理论上保证 worst-group 性能。**缺点**：权重更新复杂、超参敏感。

### 4. RLHF 中加公平约束

在 RM 训练时同时考虑公平性：

$$
\mathcal{L}_{\text{RM-fair}} = \mathcal{L}_{\text{RM}} + \alpha \cdot \text{Disparity}(\{r(x, y_i)\}_i, \text{group})
$$

把"群体间 reward 差距"作为惩罚项。在 RLHF 阶段加入公平性约束，让最终策略的输出分布更均衡。

## 四、推理层去偏见

### 1. Self-Debias Prompting（自检式 prompt）

Schick et al. (2021) 提出 **Self-Diagnosis** 与 **Self-Debiasing**：让模型先**评估**自己的输出是否有偏见，再决定是否重写：

```python
SELF_DEBIAS_PROMPT = """你是一个注重公平的助手。在回答前，请先评估你的回答是否包含对任何群体的刻板印象或偏见。

用户问题：{question}
你的回答：{answer}

评估：
1. 这个回答是否对特定性别、种族、宗教、年龄等群体有刻板印象？
2. 如果有，请指出并改写一个更中立的版本。

中立版本："""

def self_debias(model, question, initial_answer):
    full_prompt = SELF_DEBIAS_PROMPT.format(
        question=question, answer=initial_answer
    )
    return model.generate(full_prompt)
```

效果上能让模型**主动规避**自己产生的偏见，但对**隐性偏见**仍无能为力。

### 2. 公平性引导解码（Fairness-guided Decoding）

在 token 选择阶段引入公平性约束：

```python
def fair_decoding_step(logits, bias_detector, threshold=0.8, penalty=2.0):
    """
    如果下一个 token 触发高偏见概率，降低它的 logits。
    """
    bias_prob = bias_detector(logits.unsqueeze(0)).item()
    if bias_prob > threshold:
        logits = logits / penalty  # 软惩罚
    return logits


# 生成循环
for step in range(max_length):
    logits = model(next_token_ids)
    logits = fair_decoding_step(logits, bias_detector)
    next_token = sample(logits)
    next_token_ids = torch.cat([next_token_ids, next_token], dim=-1)
```

**优点**：无需重新训练；可控制强度。**缺点**：bias_detector 本身可能有偏见；推理延迟。

### 3. 输出过滤与重写

训练一个**毒性与偏见检测器**对模型输出做后处理：

```python
class OutputFilter:
    def __init__(self, bias_classifier, rewrite_model):
        self.clf = bias_classifier
        self.rewriter = rewrite_model
    
    def __call__(self, text):
        # 1) 检测偏见分数
        bias_score = self.clf(text)
        
        # 2) 超阈值则重写
        if bias_score > 0.5:
            rewrite_prompt = (
                f"以下文本包含对特定群体的偏见（分数 {bias_score:.2f}）。"
                f"请改写为中立版本：\n\n{text}\n\n中立版本："
            )
            return self.rewriter.generate(rewrite_prompt)
        return text
```

工业级 LLM API（OpenAI、Anthropic）都内置类似过滤管线。

### 4. Constrained Decoding（约束解码）

让模型在生成时**强制遵守某些规则**。例如：

- 不出现刻板短语列表（通过 Trie 拒绝）。
- 必须提到"个体差异"、"具体情况具体分析"等中性短语（概率偏置）。
- 职业描述必须用"他/她"或"他们"等中性代词。

实现：在 decoding 时把禁用 token 的 logits 设为 $-\infty$，或把鼓励 token 的 logits 加偏置。

## 五、组合策略与实践

实际工业系统通常**多层组合**：

```text
数据层:
  - 训练数据平衡 + 反事实增广
  - 偏见词表过滤
  
训练层:
  - SFT + DPO（避免 RM 偏差）
  - RLHF 中加入公平约束
  
推理层:
  - Self-debias prompt
  - Constrained decoding
  - 输出过滤与重写
  
监控层:
  - 实时 BBQ / StereoSet 评估
  - 用户反馈循环
```

## 六、去偏见的根本局限

### 1. 隐性偏见无法消除

**隐性偏见（implicit bias）**指通过委婉语、场景设定等传递的偏见：

> "那位优雅的女士优雅地走入会议室"

"优雅地走入会议室"长期与女性共现，模型仍会学到关联。简单的代词替换解决不了。

### 2. 偏见从显性转向隐性

Gonen & Goldberg (2019) 证明：**投影去偏后，bias 仍可从最近邻词中恢复**。模型把偏见从"主方向"挪到"次方向"——表面中立，实际仍在。

### 3. 文化相对性

"公平"在不同文化下定义不同。Anthropic 宪法原则带有西方自由主义色彩，套用到集体主义文化可能造成新的不公。

### 4. 公平-性能权衡

强制公平往往牺牲**总体准确率**，且代价主要由弱势群体承担（他们被错过的概率上升）。

### 5. 可解释性难题

**无法验证**模型"真的"没有偏见——只能验证**已知偏见**被缓解。新偏见可能涌现而我们不知道。

## 七、去偏见技术的评估方法

### 1. 内部评估

| 基准 | 衡量 |
|---|---|
| StereoSet | 4 类偏见的 stereotype 倾向 |
| CrowS-Pairs | 9 类偏见的最小对差异 |
| BBQ | ambig + disambig 双语境 |
| WinoGender | 性别代词消解 |
| BOLD | 5 类人群的生成质量 |

### 2. 外部评估（任务相关）

- **招聘场景**：让模型给简历打分，看群体间分数差异。
- **医疗场景**：让模型推荐治疗方案，看群体间方案差异。
- **教育场景**：让模型评估作文质量，看方言 / 族裔差异。

### 3. 长期监控

- **生产环境日志**：不同群体的对话质量、回答长度、refusal 率。
- **用户投诉**：按群体聚类看投诉热点。
- **定期再评估**：每月跑 StereoSet / BBQ 追踪指标变化。

## 八、去偏见 vs 取消对齐

去偏见的目标是**减少不必要的群体差异**，不是**取消对群体的关注**。两个常见错误：

1. **过度去偏**：模型对真实存在的差异也"装看不见"，反而误导用户（如医疗场景忽略性别差异）。
2. **取消对齐**：把所有群体特征视为"刻板印象"一律删除，丧失有用的先验。

**正确做法**：去偏见是**减少不公正差异**，不是**消除所有差异**。

## 九、给工程团队的清单

1. **数据审计**：发布前对训练数据做性别 / 种族 / 地域分布审计。
2. **多基准评估**：Stereotype + BBQ + RealToxicityPrompts + 自建任务相关基准。
3. **A/B 测试**：不同用户群体间的实际使用差异。
4. **可解释性**：用 probing classifier 检查中间层是否仍编码偏见信息。
5. **用户反馈循环**：让用户能标记"这个回答不公平"，定期 retrain。
6. **透明披露**：在 model card 中说明已知偏见与缓解措施。

## 小结

去偏见是一个**多层、长期、没有银弹**的工程问题。数据层平衡与增广便宜但效果有限；训练层正则与对抗有效但成本高；推理层灵活但难通用。更关键的是**去偏见有根本局限**——隐性偏见、偏见迁移、文化相对性都让"完全公平"成为伪命题。**真正的做法是持续评估 + 透明披露 + 让受影响群体参与设计**。下一篇我们从"偏见"转向另一个对齐核心话题——**interpretability**：当我们说模型"学到了偏见"时，它在内部到底学到了什么？
