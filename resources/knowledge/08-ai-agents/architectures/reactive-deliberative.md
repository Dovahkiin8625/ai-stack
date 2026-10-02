# 反应式 vs 慎思式：Agent 认知架构深度对比

经典 AI 把 Agent 架构分成**反应式（Reactive）**与**慎思式（Deliberative）**两大流派。前者快、后者聪明；前者无需世界模型，后者必须维护内部状态。LLM Agent 时代下，这场争论有了新答案——大多数生产系统走"**混合架构**"：高层用 LLM 慎思、低层用反应式执行。本文深入两种架构的数学建模、代表实现、典型缺陷，以及如何把它们组合成现代 Agent。

## 一、反应式 Agent：感知-动作直接映射

### 1.1 数学模型

$$
\text{Action} = f(\text{Percept})
$$

不需要历史、不需要世界模型，每次响应都是当前感知的纯函数。

### 1.2 Brooks 的 Subsumption Architecture

MIT 的 Rodney Brooks 1986 年提出，颠覆传统 AI 假设：

```text
Layer 3: 探索行为           ← 最高优先级
Layer 2: 路径规划
Layer 1: 避障
Layer 0: 避碰
```

```python
class SubsumptionAgent:
    """分层反应式机器人架构示意"""
    def __init__(self):
        self.layers = [
            self.avoid_collision,    # 0: 必须执行
            self.wander,             # 1: 默认行为
        ]

    def step(self, sensors):
        for layer in self.layers:
            action = layer(sensors)
            if action is not None:
                return action
        return None    # 无动作

    def avoid_collision(self, sensors):
        if sensors["front"] < 0.3:
            return "turn_left"
        return None     # 让下一层处理

    def wander(self, sensors):
        return "move_forward"
```

特点：

- **没有世界模型**：直接读传感器 → 输出动作。
- **分层抑制**：高层可以"覆盖"低层行为。
- **鲁棒**：单层失败不致命。
- **缺点**：缺乏规划，不能为远期目标做长期序列。

### 1.3 反应式在 LLM Agent 中的体现

Tool call 层的执行就是反应式：

```python
def react_layer(tool_name: str, args: dict) -> dict:
    """已决定调哪个工具，直接执行，不解释为什么"""
    return TOOLS[tool_name](**args)
```

## 二、慎思式 Agent：世界模型 + 规划

### 2.1 数学模型

慎思式 Agent 维护一个内部世界模型：

$$
M_{t+1} = \text{Update}(M_t, \text{Percept}_t)
$$

然后在 $M$ 上做规划：

$$
\text{Plan} = \text{Planner}(M_t, \text{Goal})
$$

$$
\text{Action} = \text{Execute}(\text{Plan}, t)
$$

### 2.2 STRIPS / PDDL 经典规划

STRIPS（Stanford Research Institute Problem Solver）用一阶谓词描述状态：

```prolog
% 初始状态
state([on(a, b), on(b, table), clear(a), clear(table)]).
goal(on(a, table), on(b, a)).

% 操作
action(move(X, Y),
    precond([clear(X), on(X, Z)]),
    effect([on(X, Y), not(on(X, Z))])
).
```

求解器用搜索算法（A*、BFS）找从初始状态到目标状态的动作序列。

### 2.3 慎思式在 LLM Agent 中的体现

```python
class DeliberativeLLMAgent:
    def step(self, observation: str):
        # 1. 更新世界模型
        self.world_model.update(observation)

        # 2. 用 LLM 生成计划
        plan = self.llm.invoke(f"""
            当前世界状态：{self.world_model}
            目标：{self.goal}
            请生成一个分步计划。
        """)

        # 3. 执行第一步
        first_step = plan.steps[0]
        result = self.execute(first_step)

        # 4. 更新计划（如果失败，重新规划）
        self.plan = plan.update(first_step, result)
        return result
```

典型例子：**Plan-and-Execute**、**Reflexion**、**Tree of Thoughts** 都是慎思式。

## 三、对比：反应式 vs 慎思式

| 维度 | 反应式 | 慎思式 |
|---|---|---|
| **响应延迟** | 极低（μs 级） | 高（秒级，含规划） |
| **可解释性** | 高（直接映射） | 中（计划可读，但 LLM 推理黑盒） |
| **长期规划** | 弱 | 强 |
| **环境扰动** | 鲁棒 | 脆弱（计划过时） |
| **资源消耗** | 低 | 高（推理开销） |
| **学习** | 简单规则可学习 | 需要世界模型训练 |
| **适用** | 反应 / 实时 | 复杂多步任务 |

## 四、混合架构：现代 Agent 的标准选择

### 4.1 三层混合架构

```text
            ┌────────────────────────────┐
  Layer 3   │ 慎思层（Deliberative）       │ ← LLM 规划、反思
            │   - 长期目标                  │
            │   - 子目标分解                │
            ├────────────────────────────┤
  Layer 2   │ 决策层（Sequencing）         │ ← 决定当前 step
            │   - 选哪个工具                │
            │   - 处理异常                  │
            ├────────────────────────────┤
  Layer 1   │ 反应层（Reactive）           │ ← 工具直接执行
            │   - HTTP 请求                │
            │   - 数据库查询                │
            │   - 确定性逻辑                │
            └────────────────────────────┘
```

### 4.2 ReAct：典型的混合范式

```python
def react_step(llm, observation):
    # 慎思部分
    thought = llm.invoke(f"""
        观察：{observation}
        历史思考：{thoughts}
        请输出下一步思考和行动。
    """)

    # 反应部分
    action_match = parse_action(thought)
    if action_match:
        tool, args = action_match
        return execute_tool(tool, args)        # 反应式执行
    else:
        return thought                       # 最终回答
```

### 4.3 LangGraph 的状态机视角

```python
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    plan: list[str]
    current_step: int
    history: list

def plan_node(state):        # 慎思：生成计划
    plan = llm.invoke(f"任务：{state['task']}, 给我 5 步计划")
    return {"plan": plan.split("\n"), "current_step": 0}

def execute_node(state):     # 反应：执行一步
    action = state["plan"][state["current_step"]]
    result = tools[action]()
    return {"history": state["history"] + [result], "current_step": state["current_step"] + 1}

def reflect_node(state):     # 慎思：反思是否需要重新规划
    if should_replan(state):
        new_plan = llm.invoke(f"基于历史 {state['history']} 重新规划")
        return {"plan": new_plan, "current_step": 0}
    return state

g = StateGraph(AgentState)
g.add_node("plan", plan_node)
g.add_node("execute", execute_node)
g.add_node("reflect", reflect_node)
g.add_edge("plan", "execute")
g.add_edge("execute", "reflect")
g.add_conditional_edges("reflect", lambda s: "execute" if s["current_step"] < len(s["plan"]) else END)
```

### 4.4 生产级混合架构：RASPA

NVIDIA 的 RASPA（Reactive and Selective Planning Agent）：

```text
1. ReAct 决定下一步动作（反应）
2. 如果连续 K 步都在重复 / 失败，触发 Plan（慎思）
3. Plan 生成子目标，回到 ReAct
4. 直到 Final Answer
```

实现：

```python
class RASPAAgent:
    def step(self, obs):
        # 反应层
        action = self.react(obs)
        self.history.append(action)

        # 选择性触发规划
        if self.should_replan():
            new_plan = self.plan(self.history)
            self.plan_steps = new_plan
        return action

    def should_replan(self) -> bool:
        # 启发式：连续 N 步重复 / 失败 / plan 完成
        return self.consecutive_failures >= 3 or self.cycle_detected()
```

## 五、设计取舍

| 场景 | 推荐 |
|---|---|
| **实时对话 / 流式响应** | 反应式优先 |
| **多步复杂任务（订机票、写报告）** | 慎思式 |
| **混合（最常见）** | 三层架构 |
| **环境高度动态** | 反应式 + 定期重规划 |
| **资源严重受限（边缘设备）** | 反应式 |
| **需要严格可解释** | 慎思式（计划可读） |

## 六、典型失败模式

### 6.1 反应式过度

```text
症状：每次都调同一个工具，循环跑死
根因：没有"记忆 + 反思"机制
对策：加 Plan node、定期 summarization
```

### 6.2 慎思式僵化

```text
症状：计划生成后遇到环境变化，不调整
根因：计划与感知脱钩
对策：执行每步后重新评估
```

### 6.3 混合架构的协调问题

```text
症状：慎思层规划很完美，反应层执行不了
根因：抽象级别不匹配
对策：慎思层用工具能力描述作为约束
```

## 小结

反应式与慎思式各有取舍。LLM 时代下，**混合架构** 是事实标准——慎思层用 LLM 做规划与反思，反应层用确定性代码执行工具。关键是在两层之间设好"何时规划、何时执行"的判断逻辑，避免反应式陷入循环、避免慎思式僵化。下一篇我们将进入 **memory-systems**——Agent 如何跨时间保留信息。
