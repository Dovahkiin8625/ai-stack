# 高级提示技术：ReAct、Tree of Thoughts、Reflection

Chain-of-Thought 让模型"会推理"，但真正的复杂任务常常需要**推理 + 行动 + 反思**的组合。本文介绍四个代表性强力技术：ReAct、Tree of Thoughts、Self-Refine、Reflexion，以及让 prompt 自己进化的 APE。它们把 LLM 从"单次回答者"推进到"具备规划与纠错能力的 agent"。

## 一、ReAct：把推理和行动绑成一个循环

ReAct（Reason + Acting, Yao et al. 2022）的核心观察：**推理和行动应该是交替进行的**。模型先"想一步"（Thought），决定调用哪个工具（Action），从工具得到观察（Observation），再进入下一轮 Thought。

```
Question -> Thought1 -> Action1 -> Observation1
        -> Thought2 -> Action2 -> Observation2
        -> ... -> Final Answer
```

```python
TOOLS = {
    "search":       "输入查询字符串，返回相关摘要列表",
    "calculator":    "输入算术表达式，返回计算结果",
}

REACT_PROMPT = """你是一名严谨的研究助手，可以使用以下工具：
{tool_desc}

请按以下格式严格回答每一步：

Thought: <你的推理>
Action: <工具名>[<输入>]
Observation: <系统填入>
... (循环 Thought/Action/Observation)
Thought: 我现在可以回答了。
Final Answer: <答案>

问题：{question}
{history}"""

import re

def react(question: str, llm, max_steps: int = 6) -> str:
    history = ""
    for step in range(max_steps):
        prompt = REACT_PROMPT.format(
            tool_desc="\n".join(f"- {k}: {v}" for k, v in TOOLS.items()),
            question=question,
            history=history,
        )
        out = llm(prompt)            # 模型返回 Thought/Action
        history += out + "\n"
        m = re.search(r"Action:\s*(\w+)\[(.*?)\]", out)
        if not m:
            break
        tool, arg = m.group(1), m.group(2)
        # 真实系统在这里调用工具
        obs = f"Observation: <{tool}({arg}) 的结果>\n"
        history += obs + "\n"
    return history
```

ReAct 让模型能在"我不知道"与"去查该库"之间切换。在 HotpotQA / Fever 等多跳推理任务上，ReAct 比纯 CoT 提升 **8-15 个百分点**。

## 二、Tree of Thoughts：让模型"分叉思考"

CoT 是线性链条，Self-Consistency 是多条链投票。Tree of Thoughts（Yao et al. 2023）把这两件事进一步推：模型在每个 Thought 节点**生成多个候选**，并由一个 evaluator（bubble 启发式或 LLM 评分）打分；通过 BFS/DFS 选出最优路径。

```text
                 Q: 4 升水如何用 3 升与 5 升桶装出？
                        |
              +---------+---------+
              |                   |
         T1: 先装满 5L       T2: 先装满 3L
         score=0.3           score=0.7
              |                   |
        +-----+-----+        +----+----+
        |           |        |         |
   A1: 倒入3L     A2:倒掉  B1:倒入5L  B2:倒掉
   score=0.4      0.2      0.6        0.1
```

实现上常借助 LangGraph 或自定义递归：

```python
from typing import Callable

def tree_of_thoughts(question: str, generate: Callable, evaluate: Callable, k=3, depth=4):
    frontier = [{"state": "Q: " + question, "score": 0.0, "path": []}]
    for d in range(depth):
        next_frontier = []
        for node in frontier:
            cands = [generate(node["state"]) for _ in range(k)]
            scored = [{"state": c, "score": evaluate(c), "path": node["path"] + [c]} for c in cands]
            next_frontier.extend(scored)
        frontier = sorted(next_frontier, key=lambda x: -x["score"])[:k]
    return frontier[0]
```

ToT 适合**搜索 / 规划 / 组合优化**类任务（如 24 点游戏、数独、命题逻辑证明），但 token 成本是 CoT 的 $k^{\text{depth}}$ 倍。

## 三、Self-Refine：用上一轮输出改写自己

Self-Refine（Chen et al., 2023）让模型在生成初稿后**自我批判**：

```
Initial draft -> Feedback -> Refined draft -> Feedback -> ... -> Final output
```

```python
REFINE_PROMPT = """你的任务是检查下面的回答，给出具体改进建议。
原始问题：{q}
回答：{problem}
输出 JSON：
{{"issues": ["..."], "rewrite_needed": true}}
"""

REWRITE_PROMPT = """根据下面的反馈重写回答，保持原意但消除问题。
问题：{q}
原回答：{a}
反馈：{fb}
重写后："""

def self_refine(q: str, initial: str, llm) -> str:
    for _ in range(3):  # 最多 3 轮
        fb = llm(REFINE_PROMPT.format(q=q, problem=initial))
        if '"rewrite_needed": false' in fb:
            break
        initial = llm(REWRITE_PROMPT.format(q=q, a=initial, fb=fb))
    return initial
```

经验：Self-Refine 在**长文本生成（摘要 / 对话 / 翻译）**上效果显著，但对短分类任务收益小。

## 四、Reflexion：用语言"反思"形成经验

Reflexion（Shinn et al., 2023）把 Self-Refine 推进一步：把多轮反思沉淀为**口头记忆**，下次遇到类似问题直接调用。

```text
Attempt 1 -> Failure -> Reflection: "我没考虑边界 N=0"
Attempt 2 -> Failure -> Reflection: "应先验证输入再split"
Attempt 3 -> Success -> 沉淀为 memory[0]
```

工程实现通常用一个"反思缓冲"（reflection buffer）：

```python
class ReflexionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.memory = []  # 累积的教训

    def reflect(self, task: str, trace: str, score: float):
        if score > 0.5:
            return
        lesson = self.llm(f"任务：{task}\n执行：{trace}\n失败原因与下次如何改进：")
        self.memory.append(lesson)

    def plan(self, task: str) -> str:
        prior = "\n".join(f"- {m}" for m in self.memory[-5:])
        return self.llm(f"过去的教训：\n{prior}\n\n新任务：{task}\n下一步计划：")
```

Reflexion 在 HumanEval / ALFWorld 等任务上比 ReAct 又提升 5-10 个百分点。

## 五、APE：让 LLM 自动生成 prompt

APE（Automatic Prompt Engineering, Zhou et al., 2022）把 prompt 设计本身当成一个**优化问题**：

1. 用 LLM 对原始 prompt 生成 $k$ 个候选改写。
2. 在评估集上跑每个候选，挑得分最高的进入下一轮。
3. 类似 prompt 进化，迭代 $n$ 轮。

```python
def ape(initial_prompt: str, eval_set, llm, scorer, rounds=3, pop=8):
    population = [initial_prompt] + [
        llm(f"请改写这条 prompt 以获得更高质量输出：\n{initial_prompt}\n改写：")
        for _ in range(pop - 1)
    ]
    for r in range(rounds):
        scores = [scorer(p, eval_set) for p in population]
        top = sorted(zip(population, scores), key=lambda x: -x[1])[:pop // 2]
        population = [p for p, _ in top]
        # 生成新变体
        for _ in range(pop // 2):
            seed = population[0]  # 简化：基于最强变体突变
            child = llm(f"基于以下 prompt 生成新变体，保持意图但改进表达：\n{seed}\n新 prompt：")
            population.append(child)
    return max(zip(population, [scorer(p, eval_set) for p in population]), key=lambda x: x[1])
```

APE 适合**业务 prompt 已经定型、还想再榨一两个百分点**的场景。

## 六、技术对比与选型

| 技术 | 核心思想 | 提升最大的场景 | 成本 |
| --- | --- | --- | --- |
| ReAct | Thought-Action-Observation 循环 | 多工具问答 | 中 |
| ToT | 多分支 + 评分剪枝 | 搜索/规划/证明 | 高 |
| Self-Refine | 自我批判 + 改写 | 长文本生成 | 中 |
| Reflexion | 反思沉淀为记忆 | 多轮复杂任务 | 中高 |
| APE | 自动进化 prompt | 成熟 prompt 微调 | 高 |

## 小结

从 CoT 到 ReAct / ToT / Reflexion / APE，可以看到一条清晰的脉络：**让模型不仅"想一步"，而是"想-做-看-改"形成闭环**。在生产中，最具 ROI 的通常仍是 ReAct + 工具调用——因为它最贴近真实业务流。下一篇会把视角拉高到 prompt 工程实战：评估、灰度、成本控制与安全。