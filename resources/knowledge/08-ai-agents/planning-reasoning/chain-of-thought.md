# Chain-of-Thought：让 LLM "一步步思考"

Chain-of-Thought（CoT, Wei et al. 2022）是 LLM 推理能力提升的里程碑式发现——只要在 prompt 中加一句"Let's think step by step"，模型在算术、常识、符号推理任务上的准确率就能大幅提升。本文深入 CoT 的原理、变体（Zero-Shot / Few-Shot / Self-Consistency）、工程实践与边界。

## 一、为什么需要 CoT

### 1.1 朴素 prompt 的局限

```text
Q:  Roger 有 5 个网球。他又买了 2 罐网球，每罐 3 个。他现在有多少个？
A:  11

朴素回答：11（错误，标准答案）
```

直觉上人类也会算错，但稍加思考就能得到 11。但 LLM 直接答 "11" 不是因为"会"，而是因为训练数据中有大量类似 "5 + 2 + 3 = ?" 的简单题。

### 1.2 CoT 的核心思想

让模型**显式生成中间推理步骤**，而不是直接跳跃到答案。

```text
Q:  Roger 有 5 个网球。他又买了 2 罐网球，每罐 3 个。他现在有多少个？
A:  Roger 起初有 5 个。每罐 3 个，他买了 2 罐，所以是 6 个。5 + 6 = 11。
    答案：11
```

## 二、CoT 的范式

### 2.1 Zero-Shot CoT

```python
prompt = """
Q: Roger 有 5 个网球。他又买了 2 罐网球，每罐 3 个。他现在有多少个？
A: Let's think step by step.
"""

# 输出：
# "Roger 起初有 5 个。他买了 2 罐，每罐 3 个，所以新增 2*3=6 个。
#  5 + 6 = 11。所以答案是 11。"
```

关键是那句话：**"Let's think step by step"**。Kojima et al. 2022 的论文发现这句"咒语"能解锁 LLM 的推理能力。

### 2.2 Few-Shot CoT（更强大）

提供几个示例，让模型模仿推理风格：

```python
prompt = """
Q: 咖啡店有 23 个苹果，用了 20 个做午餐，又买了 6 个，还有几个？
A: 咖啡店起初有 23 个苹果。用掉 20 个后剩 3 个。又买了 6 个，所以有 9 个。答案是 9。

Q: Jessica 有 6 块糖，给了朋友 4 块，又从妈妈那里得到 10 块，她现在有多少块？
A: Jessica 起初有 6 块糖。给了朋友 4 块后剩 2 块。又从妈妈那里得到 10 块，所以有 12 块。答案是 12。

Q: Roger 有 5 个网球。他又买了 2 罐网球，每罐 3 个。他现在有多少个？
A: 
"""
```

Wei et al. 2022 证明 Few-Shot CoT 在 GSM8K（数学应用题）上把准确率从 18% 提升到 57%。

### 2.3 Self-Consistency（Wang et al. 2022）

同一个问题采样多个 CoT 推理路径，投票选最一致的答案：

```python
from collections import Counter

def self_consistency(llm, prompt, n_samples: int = 5, temperature: float = 0.7):
    answers = []
    for _ in range(n_samples):
        response = llm.invoke(prompt, temperature=temperature)
        # 从 response 中提取最终答案
        answer = extract_final_answer(response)
        answers.append(answer)

    return Counter(answers).most_common(1)[0][0]
```

直觉：如果是"碰巧答对"，多个推理路径给出不同答案；如果是"真推理"，不同路径会收敛到同一答案。

**GSM8K 上进一步提升到 74%**。

### 2.4 Least-to-Most（Zhou et al. 2022）

把复杂问题分解为子问题，逐个解决：

```text
Q: 桌子上有红色、蓝色、绿色球各 5 个。再添加 3 个红色球，红色球总数是多少？

Step 1 (子问题 1)：桌子上有多少个红色球？→ 5
Step 2 (子问题 2)：添加 3 个红色球后是多少？→ 5 + 3 = 8
最终答案：8
```

```python
def least_to_most(llm, question):
    # 第一步：分解
    subtasks = llm.invoke(f"把以下问题分解为子问题：\n{question}")
    subtasks = parse_list(subtasks)

    # 第二步：逐个解决
    context = []
    for subtask in subtasks:
        answer = llm.invoke(
            f"已知：{context}\n回答：{subtask}"
        )
        context.append((subtask, answer))

    # 第三步：汇总
    return llm.invoke(f"已知：{context}\n回答原问题：{question}")
```

## 三、CoT 的底层机制

### 3.1 为什么 CoT 起作用？

学界有几种解释：

**解释 1：更多计算 = 更多思考**

Transformer 是恒定深度（不考虑层数）。但 CoT 让模型在多个 token 上"展开"推理，相当于增加了有效计算量：

```text
直接答: x → y    (1 步)
CoT:     x → a → b → c → y  (4 步)

后者相当于深度 4× 的模型
```

**解释 2：序列化分解复杂问题**

复杂推理需要"中间变量"在内存中暂存。CoT 把它显式写在纸面上。

**解释 3：训练分布的模仿**

LLM 训练数据中已经见过大量"分步骤思考"的文本（教科书、习题解答）。CoT prompt 触发这种分布。

### 3.2 CoT 不是万能的

```text
- 对简单任务：CoT 可能引入错误（多一步多错一次机会）
- 对模型能力依赖：太弱的模型（< 7B）CoT 反而效果差
- 对推理能力边界：GSM8K 上 100B+ 模型 CoT 准确率 80%+，但 OOD 题仍失败
- 对幻觉放大：CoT 中间步骤可能"自信地错"
```

## 四、CoT 的工程实现

### 4.1 标准 Few-Shot CoT 模板

```python
COT_TEMPLATE = """
{examples}

Q: {question}
A: Let's think step by step.
"""

EXAMPLES = """
Q: 一个披萨被切成 8 块，3 个人平分，每人吃几块？
A: 8 块披萨 3 个人平分。8 ÷ 3 = 2.67 块。答案是约 2.67 块。

Q: 一列火车 2 小时行驶 240 公里，平均速度多少？
A: 速度 = 距离 / 时间 = 240 / 2 = 120 公里/小时。答案是 120。
"""
```

### 4.2 答案提取

```python
import re

def extract_final_answer(response: str) -> str:
    """从 CoT 响应中提取最终答案"""
    # 模式 1: "答案是 X"
    match = re.search(r"答案[是为]?\s*[:：]?\s*([^\n。]+)", response)
    if match:
        return match.group(1).strip()

    # 模式 2: "Final Answer: X"
    match = re.search(r"Final Answer[:：]\s*([^\n]+)", response)
    if match:
        return match.group(1).strip()

    # 模式 3: 数字（兜底）
    numbers = re.findall(r"-?\d+\.?\d*", response)
    return numbers[-1] if numbers else response.strip()
```

### 4.3 与 Tool Use 结合

CoT 与工具调用协同：

```python
def cot_with_tools(llm, question):
    cot_prompt = f"问题：{question}\n让我们一步步思考。需要计算时使用 calculator 工具。"

    while True:
        response = llm.invoke_with_tools(cot_prompt)
        if response.has_tool_call:
            # 执行 tool，把结果拼回 prompt
            result = execute_tool(response.tool_call)
            cot_prompt += f"\n{response.text}\n工具结果：{result}\n继续："
        else:
            return extract_final_answer(response.text)
```

### 4.4 CoT 的 Token 成本

CoT 通常让响应从 ~50 tokens 涨到 ~500 tokens，成本 10×：

```python
def cost_aware_cot(llm, question, complexity_threshold=0.5):
    """简单问题不用 CoT，复杂问题才用"""
    if estimate_complexity(question) < complexity_threshold:
        return llm.invoke(question)              # 直接答
    else:
        return llm.invoke(question + "\nLet's think step by step.")  # 用 CoT
```

## 五、CoT 的进阶变体

### 5.1 Zero-Shot CoT 的多种"咒语"

```text
"Let's think step by step."                        (Kojima 2022)
"Take a deep breath and work through this step by step."  (Google, 2024 - 提示词工程)
"Let's think about this logically."               (变体)
"First, let's understand the problem..."           (变体)
```

### 5.2 Auto-CoT（Zhang et al. 2022）

自动生成 CoT 示例：

```python
def auto_cot(question, llm):
    # 1. 用 Zero-Shot CoT 答
    response = llm.invoke(f"Q: {question}\nA: Let's think step by step.")

    # 2. 把这个 response 作为未来问题的 few-shot 示例
    add_to_example_pool(question, response)
```

### 5.3 Multimodal CoT

CoT 应用于多模态（图像 + 文本）：

```python
prompt = """
[图像：手写的数学题]
请先描述你在图像中看到了什么，然后一步步推理答案。
"""
```

### 5.4 Tree of Thoughts（Yao et al. 2023）

CoT 是单链推理，ToT 探索多条路径（见 tree-of-thought.md）。

## 六、CoT 在生产中的最佳实践

### 6.1 何时用 CoT

| 任务 | 是否用 CoT | 原因 |
|---|---|---|
| 简单分类 | 否 | 增加 token 无收益 |
| 数学应用题 | 是 | 显著提升准确率 |
| 多步推理 | 是 | 中间步骤是必须的 |
| 代码生成 | 视情况 | CoT 可帮助规划，但代码本身要简洁 |
| 摘要 | 否 | CoT 反而稀释摘要 |
| 多跳问答 | 是 | 帮助检索 + 推理 |

### 6.2 调试 CoT 失败

```python
def debug_cot(question, expected_answer):
    cot_response = llm.invoke(question + "\nLet's think step by step.")
    actual = extract_final_answer(cot_response)

    if actual != expected_answer:
        # 1. 看中间步骤哪里出错
        print("CoT trace:", cot_response)

        # 2. 尝试 Self-Consistency
        sc_answer = self_consistency(llm, question)
        if sc_answer == expected_answer:
            print("Self-Consistency 修复了")

        # 3. 尝试更强的模型或 Few-Shot
```

### 6.3 CoT 与温度

```text
- 数学题：temperature = 0（确定性）
- 多样性任务：temperature = 0.7 + Self-Consistency
- 创意写作：temperature = 1.0
```

## 七、局限与未来

### 7.1 已知局限

- **数学能力天花板**：GSM8K 上 95% 后停滞，再大模型也难突破。
- **组合泛化差**：训练分布内的组合能答，分布外失败。
- **幻觉放大**：CoT 中间步骤可能自信地错。
- **长度敏感**：过长 CoT 容易迷失。

### 7.2 未来方向

- **Verifier**：训练一个 verifier 模型给 CoT 步骤打分。
- **Tool-augmented CoT**：CoT 中间步骤可以调工具（计算、搜索）。
- **Self-Refine**：CoT 后让模型反思自己的推理。
- **Process Reward Model (PRM)**：OpenAI o1 类推理，每步奖励。

## 小结

CoT 是 LLM 推理能力的第一性突破：用"显式中间步骤"释放模型潜在的推理能力。Few-Shot CoT、Self-Consistency、Least-to-Most 等变体进一步提升效果。生产中要按任务复杂度决定是否启用 CoT，并配合工具调用、self-consistency 进一步提升准确率。CoT 不是银弹，但对数学、逻辑、多步推理任务是性价比最高的优化。下一篇我们将深入 **tree-of-thought**——CoT 的"分支升级版"。
