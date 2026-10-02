# Few-shot 与 In-Context Learning：从示例中学习

不微调也能让模型"学会"新任务——这就是 few-shot prompting 的威力。OpenAI GPT-3 论文第一次系统展示了 in-context learning（ICL）：把若干示例塞进 prompt，模型就能在"无梯度更新"的情况下学会模式。本文讲清 ICL 的本质、示例的选择策略，以及少样本在情感分类任务上的效果演进。

## 一、什么是 In-Context Learning

In-Context Learning 指模型不更新权重，仅通过 prompt 中给定的若干 `(input, output)` 示例"当场学会"任务。GPT-3 论文把 ICL 拆成三种设置：

- **Zero-shot**：prompt 里只有任务描述，没有任何示例。
- **One-shot**：1 个示例。
- **Few-shot**：通常 10-100 个示例（受上下文窗口限制）。

直觉上，ICL 让 LLM 在 inference 阶段**隐式地学到了一个临时任务向量**——给定上文中的示例，模型被"诱导"出对应任务的条件分布。

## 二、ICL 为什么有效：两个视角

### 1. 频率学派视角

ICL 可以看作**隐式贝叶斯推断**：模型在预训练阶段见过大量"任务 + 示例 + 答案"的元模式（meta-pattern）。给定 $k$ 个示例，模型做的事相当于**估计任务的隐变量 $p$**，然后按 $p$ 下的条件分布生成：

$$
P(y \mid x, \mathcal{D}_{k}) = \int P(y \mid x, p)\, P(p \mid \mathcal{D}_{k})\, dp
$$

其中 $\mathcal{D}_k = \{(x_1, y_1), \dots, (x_k, y_k)\}$。这就是为什么"示例要典型、要覆盖典型模式"。

### 2. 工程视角

更实际的经验：示例的作用是**约束输出格式 + 限定语义边界 + 校准难度预期**。当输出格式复杂（如自定义 JSON）时，示例几乎是"必需"的；当任务歧义大时，示例能压方差。

## 三、示例数量：越多越好吗

一般规律：

- **0→1 个示例**：跳变最大。Zero-shot 在格式任务上经常"自由发挥"。
- **1→3-5 个**：通常仍能稳定提升。
- **5→10+ 个**：边际收益递减，且**挤占上下文窗口**。

经验范围：**分类/抽取任务 4-8 个示例起步**，**生成任务 2-4 个示例**就够——太多示例反而会"埋没"用户问题。

## 四、示例的 6 条选择策略

不是"随便挑几个"就能跑出好效果。实践中：

1. **典型性**：选最能代表任务"标准答案"的示例。
2. **多样性**：覆盖各类边界场景（短/长/含噪声/含特殊符号）。
3. **正确性**：示例标签错一个，模型可能全部学坏。
4. **顺序**：把"简单+标准"的放前面**，复杂+边缘的放后面**（这是经验规律，但任务不同可能相反）。
5. **格式一致性**：示例格式必须与最终期望输出严格一致，否则模型"模仿错"。
6. **避免过长示例**：示例本身占用 token，且增加注意力分散。

## 五、对比实验：情感分类 0/1/5-shot

下面用 OpenAI API 演示同一个情感分类任务在不同 shot 下的效果。任务：把英文评论分类为 positive / neutral / negative。

```python
import os
from openai import OpenAI

client = OpenAI()
LABELS = ["positive", "neutral", "negative"]

def classify(text: str, examples: list[tuple[str, str]]) -> str:
    """examples 元素为 (text, label) 对"""
    example_block = "\n".join(
        f"Text: {t}\nLabel: {l}" for t, l in examples
    )
    prompt = f"""Classify the sentiment of each text. Possible labels: positive, neutral, negative.

{example_block}

Text: {text}
Label:"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    return resp.choices[0].message.content.strip().lower()

# 评估集
TEST = [
    ("Battery lasts two days, then I'd recommend this phone.", "positive"),
    ("Shipping was slow but the product itself is fine.", "neutral"),
    ("It broke after a week, total waste of money.", "negative"),
    ("The camera is amazing, takes beautiful photos.", "positive"),
]

# 候选示例池
POOL = [
    ("Best purchase I've made all year!", "positive"),
    ("Does the job, nothing special.", "neutral"),
    ("Stopped working after three days, furious.", "negative"),
    ("Comfortable, fast shipping, would buy again.", "positive"),
    ("Mediocre at best, expected more for the price.", "negative"),
]

# 1-shot: 仅 1 个 positive 示例
one_shot = [POOL[0]]
# 5-shot: 取前 5 个覆盖多类
single = POOL[:5]

zero_res = [classify(t, []) for t, _ in TEST]
one_res  = [classify(t, one_shot) for t, _ in TEST]
five_res = [classify(t, single) for t, _ in TEST]

def acc(preds, gold):
    return sum(p.startswith(g) for p, g in zip(preds, gold)) / len(gold)

print(f"zero-shot acc = {acc(zero_res, [g for _, g in TEST]):.2f}")
print(f"1-shot   acc = {acc(one_res,  [g for _, g in TEST]):.2f}")
print(f"5-shot   acc = {acc(five_res, [g for _, g in TEST]):.2f}")
```

预期结果（会因模型版本而异，但总体上 1-shot 就能修复 zero-shot 的"自由发挥"，例如模型可能输出 `Positive` `Positive.` `pos.` 等大小写/标点不一致）。

## 七、Few-shot vs Fine-tuning 的取舍

| 维度 | Few-shot | Fine-tuning |
| --- | --- | --- |
| 成本 | 低（仅推理 token） | 高（GPU + 数据 + 训练） |
| 灵活性 | 高（改 prompt 即改行为） | 低（需重新训练） |
| 数据需求 | 极少（5-50 条） | 中等（数百-数万） |
| 性能上限 | 中 | 通常更高 |
| 延迟 | 较高（prompt 长） | 低（蒸馏后更小） |
| 适合场景 | 任务多变、PoC | 任务固定、量大 |

## 八、常见坑

1. **示例与真实分布不一致**：示例全是长句，测试时来短句，模型效果骤降。
2. **示例标签错误**：1 个错标签可能拉低整体准确率 5-10 个百分点。
3. **示例顺序把难例放第一**：很多场景下，模型对"最近邻示例"权重更大。
4. **示例占满上下文**：把用户问题挤出注意力窗口，模型只见示例不见问题。

## 小结

Few-shot 是 prompt 工程的"重武器"：用 1-4 个精心挑选的示例就能显著提升格式一致性、压缩方差、校准难度预期。下一篇文章我们会看一个更强的推理增强技术——Chain-of-Thought。