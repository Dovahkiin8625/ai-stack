# Agent 基础：从反应式到自主智能体

"Agent"是 AI 领域最被滥用、也最被误解的术语之一。它可以指代任何从简单恒温器到 AutoGPT 的程序。本文给出 Agent 的严格定义、关键属性分类、认知架构演进谱系，以及在 LLM 时代下"智能体"概念的重新审视。

## 一、Agent 的严格定义

Russell & Norvig 在《Artificial Intelligence: A Modern Approach》中给出经典定义：

> **Agent** 是任何能通过**传感器（sensors）**感知其**环境（environment）**，并通过**执行器（actuators）**作用于环境的实体。

```text
                 ┌──────────────┐
                 │ Environment  │
                 └──────┬───────┘
                        │ 感知 (Percept)
                        ↓
    执行动作 (Action)  ┌──────────────┐
       ←─────────────│    Agent     │
                       └──────────────┘
```

把"Agent"映射到 LLM 时代：

| 经典概念 | LLM 时代对应 |
|---|---|
| 传感器 | 用户输入、tool 返回、检索结果 |
| 执行器 | 生成文本、调用 tool |
| 环境 | 文件系统、API、Web、数据库 |
| 理性（rationality） | 选择使目标最大化的动作 |

## 二、Agent 的属性谱

Wooldridge & Jennings 提出"强弱 Agent"的区分：

### 弱 Agent（Weak Notion）

满足最小条件即可：

```text
- 自主性 (autonomy)        无外部指令也能行动
- 社会性 (social ability)   与其他 agent 通信
- 反应性 (reactivity)       感知并响应环境
- 主动性 (pro-activeness)  主动发起目标导向的行为
```

恒温器算弱 Agent（只有反应性），智能音箱算强一些（多了社会性）。

### 强 Agent（Strong Notion）

```text
- 知识 / 信念 (knowledge / belief)
- 意图 (intention)
- 欲望 (desire)
- 理性 (rationality)
- 心智状态 (mental state)
```

强 Agent 通常用 **BDI（Belief-Desire-Intention）架构**建模。

## 三、Agent 的环境分类

| 维度 | 范围 | 例子 |
|---|---|---|
| **可观察性** | 完全 ↔ 部分 | 棋盘（完全）vs 真实世界（部分） |
| **确定性** | 确定 ↔ 随机 | 扫地机器人 vs 推荐系统 |
| **回合性** | 回合 ↔ 连续 | 下棋 vs 自动驾驶 |
| **静态 / 动态** | 静态 ↔ 动态 | 解方程 vs 实时对话 |
| **离散 / 连续** | 离散 ↔ 连续 | 文本生成 vs 机器人控制 |
| **Agent 数量** | 单 ↔ 多 | AlphaGo vs 电商竞价 |

LLM Agent 通常处于"部分可观察 + 动态 + 多 agent"的环境，最复杂的一类。

## 四、认知架构演进

### 4.1 反应式（Reactive）

```text
感知 ──→ IF-THEN 规则 ──→ 动作
```

- 没有内部状态
- 极快、可解释
- 例：Brooks 的 Subsumption Architecture、ELIZA（聊天机器人鼻祖）

```python
def reactive_agent(percept: str) -> str:
    if "hello" in percept.lower():
        return "Hi, how can I help?"
    if "bye" in percept.lower():
        return "Goodbye!"
    return "I see."
```

### 4.2 慎思式（Deliberative）

```text
感知 ──→ 世界模型 ──→ 规划 ──→ 动作
```

- 维护内部世界模型
- 推理、规划
- 例：STRIPS planner、早期 GPS（General Problem Solver）

### 4.3 混合式（Hybrid）

反应式 + 慎思式，常见于现代 LLM Agent：

```text
高层（慢）：LLM 规划 "我应该先搜索、再比较、再写"
   ↓
低层（快）：工具调用、API、规则执行
```

例：**LangChain ReAct Agent**、**AutoGPT**、**Claude with Tools** 都是这种架构。

### 4.4 BDI 架构

```text
Beliefs:  关于世界的事实       "用户在问退款政策"
Desires:  想达成的目标         "提供准确、礼貌的回答"
Intentions: 已承诺的计划       "先检索 FAQ，再总结"
```

```prolog
% 伪代码
belief(user_intent, refund_inquiry).
desire(provide_help).
intend(search_faq, [refund]).

% BDI 推理循环
while not goal_achieved:
    perceive(env)
    update_beliefs()
    deliberate()           % 从 desires 中选出 intentions
    plan(intention)
    execute(action)
```

## 五、LLM Agent 的范式转换

LLM 时代下 Agent 范式发生了根本变化：

| 维度 | 传统 Agent | LLM Agent |
|---|---|---|
| 知识表示 | 符号逻辑、一阶谓词 | 自然语言 + embedding |
| 推理 | 定理证明、规划算法 | LLM 内化 + Chain-of-Thought |
| 规划 | 经典 AI planner | LLM 生成 todo list |
| 学习 | 监督 / 强化学习 | In-context + fine-tune |
| 工具 | 预编程 API | LLM 自主决定调用哪个工具 |
| 可解释性 | 中等 | 弱（黑盒推理） |

### 现代 LLM Agent 的最小骨架

```python
class LLMAgent:
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools            # 函数注册表
        self.memory = memory          # 历史

    def step(self, observation):
        # 1. 把 observation 加进 memory
        self.memory.add(observation)

        # 2. LLM 思考："看到 X，应该调 Y 还是回复？"
        action = self.llm.decide(
            history=self.memory.retrieve(),
            tools=self.tools.descriptions(),
            observation=observation,
        )

        # 3. 执行 action（调 tool 或生成回复）
        result = self.execute(action)

        # 4. 把结果加进 memory
        self.memory.add(result)
        return result

    def run(self, max_steps: int = 10):
        for _ in range(max_steps):
            obs = self.perceive()
            result = self.step(obs)
            if self.is_done(result):
                return result
        return "未完成"
```

## 六、Agent 的评估难题

评估 Agent 比评估分类器难得多：

```text
- 任务完成率（成功率）
- 步数效率（用多少步完成任务）
- 工具选择准确率（该用的工具都用了，没用不该用的）
- 鲁棒性（异常输入下的表现）
- 可解释性（每步决策是否可追溯）
- 安全性（不会破坏环境 / 误删数据）
```

常用基准：

- **SWE-bench**：软件工程任务（修复 GitHub issue）
- **WebArena**：浏览器操作（订机票、买东西）
- **GAIA**：通用助手任务
- **τ-bench / AgentBench**：综合 agent 能力
- **OSWorld**：操作系统操作

## 七、争议与思考

1. **LLM Agent 真有"意图"吗？** 还是只是看起来有？这关系到哲学层面的"中文房间"。
2. **自主性的边界**：Agent 该多大程度被授权？AutoGPT 删错文件的风险怎么防？
3. **可解释性 vs 能力**：能力越强的 Agent 越黑盒，但生产环境需要可解释。
4. **多 Agent 系统的涌现**：超出设计意图的群体行为是否可控？

## 小结

Agent 是 AI 最古老、最广泛的概念之一。LLM 时代给了它新的实现范式——从符号推理转向"自然语言 + 工具调用"。理解 Agent 需要从认知架构（反应式 / 慎思式 / BDI / 混合）出发，结合 LLM 特性重新审视。下一步我们将深入 **planning-reasoning**——Agent 如何思考、如何分解复杂任务。
