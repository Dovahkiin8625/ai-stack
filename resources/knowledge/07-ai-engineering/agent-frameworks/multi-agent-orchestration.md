# 多 Agent 编排：Supervisor、Swarms 与角色分工

单个 Agent 能解决"调一两个工具"的问题，但面对"调研 → 写作 → 审稿 → 发布"这种多阶段任务时，单 Agent 容易"既要又要还要"导致规划混乱。多 Agent 编排把任务拆给多个**专精**的 Agent，再用**协调者**或**共享消息总线**组织它们。本文梳理主流多 Agent 模式：Supervisor、Swarms（群智）、Debate（辩论）、CrewAI / AutoGen 的工程实现。

## 一、为什么需要多 Agent

单 Agent 的瓶颈：

1. **上下文过载**：系统 prompt 里塞太多工具、太多指令，模型注意力稀释。
2. **专精度不足**：同一个模型既写代码又做翻译，效果都打折。
3. **角色冲突**：规划、执行、审稿逻辑相互干扰。

多 Agent 通过**职责拆分**和**结构化通信**解决这些问题。N 个专精 Agent 通常胜过 1 个"全能" Agent。

## 二、Supervisor 模式：层级式

最经典的多 Agent 架构。一个 **Supervisor（调度者）**负责拆解任务、分发给 Worker、汇总结果：

```text
                ┌──→ Researcher Agent ──┐
                │                        │
User Query ──→ Supervisor ──┼──→ Writer Agent ──┼──→ Final Output
                │                        │
                └──→ Reviewer Agent ─────┘
```

实现要点：

- Supervisor 的 system prompt 描述"我有几个 worker、各自能做什么"。
- 每轮对话 Supervisor 决定下一步 dispatch 给哪个 worker。
- Worker 完成子任务后把结果回填给 Supervisor。

LangGraph 里可以直接用 `StateGraph` 描述：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    messages: list
    next: str

def supervisor(state):
    # 调 LLM 决定下一个 worker
    decision = llm.invoke(...)
    return {"next": decision}

def researcher(state): ...
def writer(state): ...
def reviewer(state): ...

g = StateGraph(State)
g.add_node("supervisor", supervisor)
g.add_node("researcher", researcher)
g.add_node("writer", writer)
g.add_node("reviewer", reviewer)
g.add_conditional_edges("supervisor", lambda s: s["next"],
    {"researcher": "researcher", "writer": "writer", "reviewer": "reviewer", "FINISH": END})
for w in ["researcher", "writer", "reviewer"]:
    g.add_edge(w, "supervisor")
g.set_entry_point("supervisor")
app = g.compile()
```

Supervisor 模式适合**强层级、流程清晰**的任务（写报告、数据分析 pipeline）。

## 三、Swarms 模式：去中心化

与 Supervisor 相反，Swarms 让所有 Agent **平等**，通过**共享消息总线**通信：

```text
Agent A ──┐
          ├──→ Shared Bus ──┼──→ Agent C
Agent B ──┘                 │
                            └──→ Agent D
```

每个 Agent 监听消息，根据自己的专长决定是否响应。OpenAI 的 Swarm 框架、CrewAI 的 Hierarchical 模式都采用此思路。

优势：

- 没有单点故障（Supervisor 挂了整个系统瘫）。
- 容易扩展（新 Agent 直接订阅总线即可）。

劣势：

- 难以保证全局一致性。
- 调试困难（消息流向不直观）。

## 四、Debate 模式：对抗性

让两个或多个 Agent **就同一问题辩论**，迫使它们互相反驳，最终收敛到更好的答案：

```text
Proposer Agent  ──→ 主张
                              ↘
                            Judge Agent
                              ↗
Critic Agent    ──→ 反驳
```

适用场景：

- 决策类问题（"应不应该投资 X 公司"）。
- 内容审核（一个生成、一个挑刺）。
- 代码 review（一个写、一个测）。

工程上要限制辩论轮数，并设计"何时收敛"的判定。

## 五、CrewAI 与 AutoGen 的工程抽象

### CrewAI：以"角色 + 任务 + 团队"为核心

```python
from crewai import Agent, Task, Crew

researcher = Agent(role="研究员", goal="搜集事实", backstory="经验丰富的调查记者")
writer = Agent(role="作家", goal="写文章", backstory="擅长把复杂信息讲清楚")

t1 = Task(description="调研 X 主题", agent=researcher)
t2 = Task(description="基于调研写一篇文章", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[t1, t2], process=Process.sequential)
result = crew.kickoff()
```

CrewAI 把"流程（sequential / hierarchical）"和"Agent 协作"做了高层抽象，适合快速搭 PoC。

### AutoGen：对话式多 Agent

微软的 AutoGen 把 Agent 抽象成**能互相发消息的对象**：

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config={...})
user = UserProxyAgent("user", code_execution_config={"work_dir": "coding"})

user.initiate_chat(assistant, message="写一个爬虫，抓取 example.com 标题")
```

AutoGen 的杀手锏是 **code execution**——Agent 可以真的执行 Python 并把 traceback 发回给 LLM，让它自己 debug。

## 六、生产环境的考量

1. **成本放大**：N 个 Agent 意味着 N 倍 token。要监控 aggregate cost per task。
2. **可观测性**：用 LangSmith / Phoenix trace 每条消息，看哪个 Agent 是瓶颈。
3. **一致性**：跨 Agent 的"事实"要可追溯，否则容易出现 Agent A 说"A 是 X"、Agent B 说"A 是 Y"的矛盾。
4. **人机协同**：关键节点保留 human-in-the-loop，例如审稿 Agent 写完后让用户确认再发布。
5. **失败兜底**：单个 Agent 失败不能让整个编排崩溃，要设计 retry / fallback。

## 小结

多 Agent 编排是单 Agent 在**复杂任务**下的自然延伸。Supervisor 适合层级清晰的任务，Swarms 适合去中心化协作，Debate 适合需要对抗性思考的场景。CrewAI、AutoGen、LangGraph 各有侧重——选型时看**任务结构**（是否有明确的流程）与**团队能力**（是否需要执行代码）。工程上要把 token 成本、可观测性、human-in-the-loop 作为一等公民来设计，否则 Agent 越多，系统越脆弱。
