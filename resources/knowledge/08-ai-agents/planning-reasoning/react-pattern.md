# ReAct 模式：推理 + 行动循环

Chain-of-Thought 让 LLM 思考，Tool Use 让 LLM 行动。**ReAct**（Reason + Act, Yao et al. 2022）把两者结合在同一个循环里，让 LLM 在"思考-行动-观察"的迭代中解决多步问题。本文深入 ReAct 的核心循环、prompt 模板、实现、与反思模式的对比，以及在 LangChain / LangGraph 中的工程实现。

## 一、ReAct 的核心思想

### 1.1 CoT 与 Tool Use 的局限

```text
CoT：思考，但只能"自言自语"
  Q: 一座城市有 50 万人口，年增长 3%，5 年后多少人？
  A: 50万 × 1.03^5 ≈ 57.96 万
  问题：算错了（实际 1.03^5 = 1.15927）

Tool Use：行动，但不思考为什么调
  调 calc(50万 × 1.03^5) → 579,636
  问题：模型不知道为什么调这个计算

ReAct：思考 + 行动 + 观察循环
  Thought: 需要计算复合增长
  Action: calc(500000 * 1.03^5)
  Observation: 579636.73
  Thought: 现在可以给出答案
  Final Answer: 5年后人口约 57.96 万
```

### 1.2 ReAct 解决的核心问题

```text
1. 信息不足：模型需要外部信息（搜索、知识库）
2. 计算能力不足：模型算术差
3. 验证缺失：模型需要外部验证（数据库）
4. 长期任务：多步子任务
5. 决策依赖：每步决策基于上一步结果
```

## 二、ReAct 的循环结构

### 2.1 标准循环

```text
For each step:
    1. Thought:   模型生成"我现在要做什么"
    2. Action:    模型决定调哪个工具、传什么参数
    3. Observation: 工具返回结果
    4. 把 (Thought, Action, Observation) 加入 history
    5. 检查是否完成
```

```python
class ReActAgent:
    def __init__(self, llm, tools, max_steps: int = 10):
        self.llm = llm
        self.tools = tools                  # 函数注册表
        self.max_steps = max_steps
        self.history = []                   # 累积 (Thought, Action, Observation)

    def run(self, question: str) -> str:
        self.history = []
        prompt = self._build_prompt(question)

        for step in range(self.max_steps):
            response = self.llm.invoke(prompt)
            self.history.append(response)

            # 解析 Thought + Action
            thought, action = self._parse(response)

            if action is None:
                # 没有 Action = Final Answer
                return self._extract_final(response)

            # 执行 Action
            observation = self._execute(action)
            self.history.append(("Observation", observation))

            # 重新构建 prompt（包含完整 history）
            prompt = self._build_prompt(question)

        return "达到最大步数"

    def _parse(self, response: str) -> tuple[str, dict | None]:
        # 解析 "Thought: ... Action: tool_name(args)"
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", response, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action = None
        if action_match:
            action = {
                "tool": action_match.group(1),
                "args": self._parse_args(action_match.group(2)),
            }
        return thought, action

    def _execute(self, action: dict):
        tool = self.tools[action["tool"]]
        return tool(**action["args"])
```

### 2.2 ReAct Prompt 模板

```text
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:
```

### 2.3 Few-Shot 示例

```text
Question: 特朗普出生在哪里？
Thought: 我需要搜索关于特朗普的信息。
Action: search
Action Input: 特朗普 出生地
Observation: 唐纳德·特朗普 1946 年 6 月 14 日出生于纽约市皇后区。
Thought: 我现在知道答案了。
Final Answer: 特朗普出生于纽约市皇后区。

Question: 2020 年 NBA 总冠军是哪支队？
Thought: 我需要查 NBA 2020 总冠军。
Action: search
Action Input: 2020 NBA 总冠军
Observation: 2020 年 NBA 总冠军是洛杉矶湖人队。
Thought: 我现在知道答案了。
Final Answer: 2020 年 NBA 总冠军是洛杉矶湖人队。

Question: {你的问题}
Thought:
```

## 三、ReAct vs Function Calling

### 3.1 Function Calling 的"沉默"模式

OpenAI 的 Function Calling 让模型直接输出结构化 tool_call，不暴露思考过程：

```python
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    tools=tools_schemas,
)
# resp.choices[0].message.tool_calls
# 模型没说为什么调这个工具
```

优点：高效、可控。缺点：调试困难、模型不能基于"思考"调整。

### 3.2 ReAct 的"显式思考"模式

```python
# 文本形式：模型先思考，再行动
"""
Thought: 用户想知道天气。我应该调用 get_weather 工具。
Action: get_weather(city="北京")
"""
```

优点：可解释、易调试。缺点：token 多、对 prompt 格式敏感。

### 3.3 现代实践：两者结合

```python
def hybrid_react_step(llm, history, tool_schemas):
    # 1. 让模型先"思考"
    thinking_prompt = f"基于 history {[-2:]}，下一步应该怎么做？"
    reasoning = llm.invoke(thinking_prompt)

    # 2. 基于推理结果，决定调哪个工具（用 Function Calling）
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": reasoning + "\n\n现在执行："},
            *history,
        ],
        tools=tool_schemas,
    )
    return response
```

或用"thought + tool_call" 的统一格式：

```python
# 让 LLM 在 Function Calling 输出前先输出 "reasoning"
# 然后再 tool_call
```

## 四、ReAct 的失败模式

### 4.1 循环陷阱

```text
Step 1: Thought: 调 search("X")
        Action: search("X")
Step 2: Observation: 没找到
        Thought: 换个词调 search
        Action: search("Y")
Step 3: Observation: 没找到
        Thought: 再换 search
        Action: search("Z")
...
```

对策：

```python
class ReActAgentWithLoopDetection(ReActAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_history = []

    def run(self, question):
        for step in range(self.max_steps):
            response = self.llm.invoke(...)
            action = self._parse_action(response)

            # 检测：相同 action 重复？
            if self.action_history.count(action) >= 2:
                # 强制重新规划
                response = self.llm.invoke(
                    f"你之前重复调用了 {action}。请尝试完全不同的策略。"
                )
                action = self._parse_action(response)

            self.action_history.append(action)
```

### 4.2 错误传播

```text
Step 1: Action: calc("a + b")  ← 字符串参数错
        Observation: 错误
Step 2: 模型基于错误 Observation 继续推理 → 全部错
```

对策：

```python
def safe_execute(self, action):
    try:
        return self.tools[action["tool"]](**action["args"])
    except Exception as e:
        return f"工具执行错误：{e}。请检查参数。"
```

### 4.3 Hallucinated Tools

模型可能"调"一个不存在的工具：

```python
# 防御
def execute(self, action):
    if action["tool"] not in self.tools:
        return f"工具 '{action['tool']}' 不存在。可用工具：{list(self.tools.keys())}"
    ...
```

## 五、ReAct 的进阶模式

### 5.1 Reflexion（自我反思）

```python
class ReflexionAgent(ReActAgent):
    def run(self, question, max_reflections=2):
        for attempt in range(max_reflections):
            answer = super().run(question)
            if self._is_correct(answer):
                return answer

            # 反思失败原因
            critique = self.llm.invoke(f"分析这个答案为什么不对：\n{answer}\n原问题：{question}")
            self.history.append(("Reflection", critique))

        return answer
```

Shinn et al. 2023 论文报告，Reflexion 在 HumanEval 上提升 10%。

### 5.2 ReAct + Memory

```python
class ReActWithMemory(ReActAgent):
    def __init__(self, *args, long_term_memory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = long_term_memory

    def run(self, question):
        # 检索相关历史经验
        past_lessons = self.memory.retrieve(question, top_k=3)
        enriched_question = f"历史教训：\n{past_lessons}\n\n现在问题：{question}"
        return super().run(enriched_question)
```

### 5.3 Multi-Agent ReAct

```python
class MultiAgentReAct:
    def __init__(self, agents: dict[str, ReActAgent]):
        self.agents = agents

    def run(self, question):
        # 调度：先 search agent，再 analysis agent
        context = self.agents["searcher"].run(question)
        answer = self.agents["analyzer"].run(
            f"基于以下信息：\n{context}\n原问题：{question}"
        )
        return answer
```

## 六、ReAct 的实现框架

### 6.1 LangChain 实现

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,
    handle_parsing_errors=True,
)

result = agent_executor.invoke({"input": "北京今天天气怎么样？"})
```

### 6.2 LangGraph 实现（更灵活）

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class ReActState(TypedDict):
    question: str
    history: list
    final_answer: str | None

def thought_node(state):
    response = llm.invoke(state["history"])
    return {"history": state["history"] + [response]}

def action_node(state):
    action = parse_action(state["history"][-1])
    result = execute_tool(action)
    return {"history": state["history"] + [("Observation", result)]}

def should_continue(state):
    return "final" if state.get("final_answer") else "action"

g = StateGraph(ReActState)
g.add_node("thought", thought_node)
g.add_node("action", action_node)
g.add_conditional_edges("thought", should_continue,
                          {"action": "action", "final": END})
g.add_edge("action", "thought")
g.set_entry_point("thought")
app = g.compile()
```

## 七、生产级 ReAct 优化

### 7.1 Token 控制

```python
class TokenAwareReAct(ReActAgent):
    def run(self, question):
        for step in range(self.max_steps):
            if self.token_count() > MAX_TOKENS:
                # 触发压缩
                self.history = self.compress_history(self.history)

            response = self.llm.invoke(...)
            ...
```

### 7.2 并行工具调用

```python
async def parallel_actions(self, actions):
    """多个独立的工具并行执行"""
    tasks = [self.tools[a["tool"]](**a["args"]) for a in actions]
    return await asyncio.gather(*tasks)
```

### 7.3 错误恢复

```python
def robust_react_step(self, action):
    for attempt in range(3):
        try:
            result = self.tools[action["tool"]](**action["args"])
            return result
        except Exception as e:
            if attempt < 2:
                # 让 LLM 修正参数
                corrected = self.llm.invoke(
                    f"工具 '{action['tool']}' 调用失败：{e}\n原参数：{action['args']}\n请修正："
                )
                action = parse_action(corrected)
            else:
                return f"工具最终失败：{e}"
```

## 八、ReAct 的替代方案

### 8.1 OpenAI o1 / o3 类推理模型

```text
o1 模型把 ReAct 集成到训练里：
  - 内部"思考"更长、更深
  - 但思考过程不可见（hidden reasoning）
  - 工具调用仍然显式

效果：在数学、代码任务上大幅超过 ReAct
代价：API 贵 3-10×
```

### 8.2 Plan-and-Execute

```python
class PlanExecuteAgent:
    def run(self, question):
        # 1. LLM 先生成完整计划
        plan = self.llm.invoke(f"为问题 '{question}' 生成步骤计划：")
        steps = parse_list(plan)

        # 2. 按计划逐步执行（每步可能用不同工具）
        context = ""
        for step in steps:
            result = self.execute_step(step, context)
            context += f"\n{step}: {result}"

        return context
```

与 ReAct 区别：先有完整计划 vs 边走边规划。

### 8.3 DSPy 的声明式 Agent

```python
import dspy

class ReActAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct("question -> answer", tools=[...])
```

## 九、评估 ReAct

```text
- 任务成功率：多少任务成功完成？
- 平均步数：完成任务的平均工具调用次数
- 工具选择准确率：每步是否选了正确的工具
- 路径效率：与最优路径相比多走了几步
- 失败模式分布：循环？幻觉？错误传播？
```

基准：AgentBench、SWE-bench、WebArena、HotpotQA。

## 小结

ReAct 把"思考"和"行动"统一在同一个循环中，是 LLM Agent 的基础模式。CoT 提供推理能力，Tool Use 提供行动能力，ReAct 让两者协同——每步推理决定下一步行动，每步行动的结果反馈给下一步推理。生产中要防御循环、错误传播、幻觉工具等问题，并配合 Memory、Reflection、Multi-Agent 等增强模式。o1/o3 类推理模型是 ReAct 思路的"内化"——把推理循环从 prompt 层移到训练层。下一篇我们将进入 **tool-use** 系列——从 Function Calling 协议到代码执行、网页浏览。
