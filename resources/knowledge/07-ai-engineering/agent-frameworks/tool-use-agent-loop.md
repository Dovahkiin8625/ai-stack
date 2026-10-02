# Tool Use 与 Agent Loop：ReAct、Function Calling 与自主决策

LLM 本身只能"说话"。Agent 的核心是让模型**自主决定**何时调用外部工具（搜索、计算、查询数据库、执行代码），并把工具返回结果再次喂给模型，最终得到一个能解决多步问题的答案。本文梳理 Tool Use 的两种范式、ReAct 推理模式、Function Calling 协议，以及一个最小可跑的 Agent Loop。

## 一、为什么需要 Tool Use

纯 LLM 有三大盲区：

1. **事实陈旧**：训练数据有截止日期。
2. **不会计算**：6×7、复杂单位换算经常出错。
3. **碰不到私有系统**：无法直接查数据库、调 API。

Tool Use 让 LLM 退化成"调度器"——决定**调哪个工具、传什么参数**，工具返回结构化结果后再继续思考。这把"语言能力"与"行动能力"解耦，模型只需擅长规划。

## 二、Function Calling：现代协议

OpenAI 在 2023 年推出的 **Function Calling** 已成为业界事实标准。模型不是直接生成自然语言调用，而是输出**结构化的 JSON**，声明要调用哪个函数及参数：

```json
{
  "name": "get_weather",
  "arguments": "{\"city\": \"Beijing\"}"
}
```

OpenAI / Anthropic / Google 都兼容这套协议。开发者只需声明工具 schema：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    }
]

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
)
tool_call = resp.choices[0].message.tool_calls[0]
# ChatCompletionMessageToolCall(id='...', function=Function(name='get_weather', arguments='{"city": "Beijing"}'), ...)
```

返回结构里有 `tool_call_id`，要把它回填进 messages 一起送回去，模型才能"看到"工具结果。

## 三、ReAct：推理 + 行动循环

ReAct（Reason + Act, Yao et al. 2022）是经典的 agent 范式。模型在每一步交替生成两件事：

1. **Thought**：用自然语言写出"我现在的判断"。
2. **Action**：调用某个工具。

工具返回 **Observation** 后，模型再循环，直到产出 **Final Answer**：

```text
Thought: 用户问北京天气，我需要先调 get_weather 工具。
Action: get_weather(city="Beijing")
Observation: {"temp": 22, "condition": "晴"}
Thought: 现在有数据了，可以回答了。
Final Answer: 北京今天晴，气温 22°C。
```

Function Calling 把 ReAct 的"行动"部分结构化，但"推理"仍由模型在每一步隐式完成。

## 四、最小 Agent Loop 实现

下面是一个 50 行级别的 ReAct Agent：

```python
import json
from openai import OpenAI

client = OpenAI()

TOOLS = {
    "get_weather": lambda city: {"temp": 22, "condition": "晴"},
    "calculate": lambda expr: eval(expr),  # 生产环境请用 sympy
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        },
    },
]

SYSTEM = """你是一名助手。能用工具就用工具，最终给出 Final Answer。"""

def run_agent(user_query: str, max_steps: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_query},
    ]
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # 没有 tool_calls，说明模型已经给出最终答案
        if not msg.tool_calls:
            return msg.content

        # 执行每个 tool_call
        for tc in msg.tool_calls:
            fn = TOOLS[tc.function.name]
            args = json.loads(tc.function.arguments)
            result = fn(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    return "Agent 超过最大步数，未完成。"

print(run_agent("北京今天天气怎么样？另外 22 + 20 等于多少？"))
# 北京今天晴，气温 22°C。22 + 20 = 42。
```

工程上要再加：超时、retry、tool_call_id 一致性校验、token 计数、人工兜底。

## 五、复杂场景的坑

1. **无限循环**：模型可能反复调同一个工具。用 `max_steps` 截断，并检测重复模式。
2. **幻觉参数**：模型可能编造不存在的字段。在工具侧做 schema 校验。
3. **错误传播**：工具抛异常要把错误回填给模型，让它决定 fallback，不能直接崩。
4. **Token 爆炸**：每轮 prompt 都叠加历史，要做上下文压缩或截断。

## 六、OpenAI vs Anthropic 的 Tool 协议差异

| 维度 | OpenAI | Anthropic |
|---|---|---|
| 协议名 | Function Calling / Tools | Tool Use |
| 触发字段 | `tools=[...]` | `tools=[...]` |
| 调用信号 | `tool_calls` 数组 | `content` 中含 `type: tool_use` 块 |
| 返回方式 | `role: tool` 消息 | `role: user` 中含 `tool_result` 块 |
| 多工具并行 | 一条消息可含多个 tool_calls | 单条消息可含多个 tool_use 块 |
| 结构化输出 | `response_format: json_schema` | 同上 |

LangChain / LlamaIndex 抽象掉了这些差异，开发者写一份代码可以跨 provider 跑。

## 小结

Tool Use 把 LLM 从"聊天机器人"升级为"调度器"：模型负责规划与推理，工具负责执行与事实。Function Calling 协议让这一切变成结构化的 JSON 流，而 ReAct 框架把"思考-行动-观察"的循环具象化。一个生产级 Agent 还需要循环截断、错误回填、token 控制、可观测性等工程护栏。下一篇我们将进入 **多 Agent 编排**——单个 Agent 不够用时，如何让多个 Agent 协作解决更复杂的问题。
