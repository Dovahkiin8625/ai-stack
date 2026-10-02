# Chain-of-Thought 与推理增强

LLM 在算术、符号推理、多步逻辑上常常"翻车"——不是因为它不会，而是因为它**跳步**：默认直接给答案，跳过了推理过程。Chain-of-Thought（CoT）让模型把中间步骤显式写出来，推理能力就会大幅提高。本文讲清 CoT 的基本思想、Zero-shot/Few-shot/自一致性，并演示 GSM8K 风格的数学题效果。

## 一、为什么 LLM 需要 Chain-of-Thought

直觉上，LLM 在每一步都只是一个"下一 token 预测器"，它对**单步预测**很强，但对**多步推理**容易丢链：一个错中间步骤会导致后续全部错。CoT 的核心是**把多步问题拆成串行单步预测**，每一步都在上下文中"验证一次"。

$$
\text{Autoregressive 错率} \sim \prod_{t=1}^{T} (1 - \epsilon) \approx (1 - \epsilon)^T
$$

每步 $\epsilon$ 的小错率，多步后累积成大错率。CoT 把错误"暴露在上下文中"——错的中间步反而给后续纠错提供了 hook。

## 二、Zero-shot CoT：一句"Let's think step by step"

2022 年 Kojima 等人发现一个惊人现象：在 prompt 末尾加一句 *"Let's think step by step"*（让我们一步一步思考），就能让模型自动展开推理链，无需任何示例。

```text
Q: A bakery sold 23 cupcakes in the morning and 18 in the afternoon.
   How many cupcakes did they sell in total?

A: Let's think step by step.
   The bakery sold 23 in the morning and 18 in the afternoon.
   Total = 23 + 18 = 41.
   So the answer is 41 cupcakes.
```

直觉上，这一句话相当于**激活了模型在预训练中见过的"解题文本"先验**。

## 三、Few-shot CoT：用示例教会推理格式

Zero-shot CoT 经常"自己跑偏"——它会展开推理但推理链未必最优。Few-shot CoT 通过 $k$ 个 `(问题, 推理过程, 答案)` 示例，让模型模仿"高质量推理"。

Wei 等人 2022 年的经典 prompt：

```text
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 each = 6. 5 + 6 = 11.
   The answer is 11.

Q: ... (用户问题)
A:
```

## 四、Self-Consistency：让模型多次推理再投票

CoT 还有一种典型失败：模型在**第一条推理链上犯错**。Self-Consistency（Wang et al., 2022）的关键思想是**用采样温度 > 0 跑** $k$ 次，取**最终答案的众数**：

```python
from collections import Counter
from openai import OpenAI

client = OpenAI()

def solve(question: str, k: int = 7) -> str:
    prompt = f"""Solve the problem step by step, ending with "The answer is X".

Q: {question}
A:"""
    answers = []
    for _ in range(k):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        text = resp.choices[0].message.content
        # 提取最终数字
        for tok in text.split(".")[::-1]:
            for w in tok.split():
                w = w.strip(",:;")
                if w.isdigit():
                    answers.append(int(w))
                    break
    # 众数即最终答案
    return Counter(answers).most_common(1)[0][0]
```

Self-Consistency 在 GSM8K 上比单次 CoT 提升 **5-15 个百分点**，几乎成为数学/代码生成任务的默认技巧。

## 五、Least-to-Most：把难题递归拆解

Least-to-Most Prompting（Zhou et al., 2022）针对**组合泛化**问题（问题比训练见过的更长/更复杂）。它分两阶段：

1. **Decompose**：让模型把问题拆成子问题序列 $[q_1, q_2, \dots, q_n]$。
2. **Solve**：依次回答每个子问题，把上一步答案喂给下一步。

```text
Q: The cafes in Paris serve 12 pastries each. If 3 cafes opened in
   London serving 5 more each, how many total pastries are served
   in both cities?

Decompose:
1. How many pastries does each Paris cafe serve?
2. How many pastries does each London cafe serve?
3. How many cafes are there in total?
4. What's the total pastries?

Solve:
1. 12
2. 12 + 5 = 17
3. 3 (Paris) + 3 (London) = 6
4. (3 * 12) + (3 * 17) = 36 + 51 = 87
```

## 六、GSM8K 风格完整示例

下面用一个 GSM8K 风格的数学题，演示 CoT 与 Self-Consistency 的实际效果。

```python
COT_PROMPT = """Solve the following math problem. Show your reasoning step by step,
and put the final answer after "The answer is " followed by just the number.

Q: Natalia sold clips to 48 of her friends in April, and then she sold half
   as many clips in May. How many clips did Natalia sell altogether in
   April and May?
A: In April she sold 48 clips. In May she sold half of 48 = 24 clips.
   Total = 48 + 24 = 72. The answer is 72.

Q: {question}
A:"""

def cot_solve(question: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": COT_PROMPT.format(question=question)}],
        temperature=0,
        max_tokens=512,
    )
    return resp.choices[0].message.content

print(cot_solve("A farmer has 12 chickens. 5 of them laid eggs today. "
                "Each chicken that laid eggs produced 3 eggs. "
                "How many eggs were collected?"))
# 期望: Step-by-step reasoning... The answer is 15.
```

## 七、CoT 的局限

- **任务类型相关**：逻辑/算术/代码类受益大；纯抽取/分类类几乎无收益。
- **推理链质量不稳**：Few-shot CoT 的上限取决于示例的推理质量。
- **token 成本**：推理链平均占 200-1000 token，延迟与成本都上升。
- **幻觉推理**：模型可能给出"听起来合理但错误"的中间步骤。

## 八、CoT 适用场景速查

| 任务 | 推荐 |
| --- | --- |
| 多步算术 / 文字题 | Few-shot CoT + Self-Consistency |
| 代码生成 / 调试 | CoT 显式列步骤 |
| 复杂逻辑判断 | Least-to-Most 或 CoT |
| 简单分类 / 抽取 | 不需要 CoT，反而拖累 |

CoT 是"推理增强"的入门砖，下一篇的 ReAct / Tree-of-Thoughts / Reflexion 会把它推向 agent 范式。