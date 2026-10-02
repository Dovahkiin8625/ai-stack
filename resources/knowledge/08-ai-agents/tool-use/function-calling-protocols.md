# Function Calling 协议：从 JSON Schema 到跨厂商兼容

Function Calling 是 LLM 接入外部工具的"标准协议"。OpenAI 在 2023 年 6 月推出后，Anthropic、Google、Meta、Mistral 几乎全部采用。本文深入 Function Calling 的 JSON Schema 设计、各家协议对比、错误处理、安全考量，以及如何构建可复用的工具注册中心。

## 一、Function Calling 的本质

LLM 不直接执行代码，而是**输出结构化 JSON 声明**：

```json
{
  "name": "get_weather",
  "arguments": "{\"city\": \"Beijing\"}"
}
```

这个 JSON 告诉调用方"我决定调哪个函数、传什么参数"。调用方（应用代码）负责真正执行。这种**决策与执行分离**的设计让 LLM 与外部世界解耦。

```text
LLM: "我决定调 get_weather(city=Beijing)"
 ↓
应用代码: get_weather(city=Beijing) → 返回结果
 ↓
LLM: 基于结果继续推理
```

## 二、JSON Schema 规范

### 2.1 基本结构

OpenAI / Anthropic 的工具声明都遵循类似的 Schema：

```json
{
  "type": "function",
  "function": {
    "name": "search_documents",
    "description": "在企业知识库中搜索相关文档",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "搜索关键词"
        },
        "top_k": {
          "type": "integer",
          "description": "返回结果数",
          "default": 5
        },
        "filter_category": {
          "type": "string",
          "enum": ["tech", "news", "finance"],
          "description": "按类别过滤"
        }
      },
      "required": ["query"]
    }
  }
}
```

### 2.2 类型支持

| JSON Schema 类型 | Python 类型 | 用途 |
|---|---|---|
| `string` | str | 文本 |
| `integer` | int | 整数 |
| `number` | float | 浮点 |
| `boolean` | bool | 布尔 |
| `array` | list | 列表 |
| `object` | dict | 嵌套对象 |
| `enum` | - | 限定值 |
| `oneOf` | - | 多种类型 |
| `anyOf` | - | 任一类型 |

### 2.3 高级特性

#### `oneOf`（联合类型）

```json
{
  "location": {
    "oneOf": [
      {"type": "object", "properties": {"city": {"type": "string"}}},
      {"type": "object", "properties": {"coords": {"type": "object", "properties": {"lat": {"type": "number"}, "lng": {"type": "number"}}}}}
    ]
  }
}
```

#### `description` 的艺术

`description` 是给 LLM 看的"自然语言文档"，**质量决定调用准确率**：

```json
// 不好
{"description": "query parameter"}

// 好
{"description": "用户的搜索关键词。建议用短语而非完整句子。要包含核心概念。"}

// 更好（带边界）
{"description": "用户的搜索关键词。建议用短语而非完整句子。必须包含至少一个名词；不接受问句形式。"}
```

**经验法则**：description 应该和"函数 docstring"同等对待，写得清楚能提升 10-20% 调用准确率。

## 三、OpenAI 协议

### 3.1 完整调用流程

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                },
                "required": ["city"]
            }
        }
    }
]

# 1. 发起请求
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
)

message = response.choices[0].message
print(f"Model reasoning: {message.content}")              # 模型可能输出自然语言
print(f"Tool calls: {message.tool_calls}")                # 结构化工具调用

# 2. 处理 tool_calls
if message.tool_calls:
    for tool_call in message.tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        # 3. 执行（这里是应用代码，不是 LLM）
        result = get_weather(**fn_args)

        # 4. 把结果回填 messages
        messages.append(message)                          # 模型的消息
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,                 # 必须一致
            "content": json.dumps(result),
        })

    # 5. 第二次调用：让模型基于工具结果生成最终回复
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    print(final.choices[0].message.content)
```

### 3.2 关键细节

- `tool_call_id`：每个 tool_call 有唯一 ID，必须原样回填。
- 并行调用：模型可一次输出多个 tool_calls（同一消息里）。
- `tool_choice`：控制工具调用行为（"auto"、"required"、"none"、指定函数名）。
- Structured Output：OpenAI 2024 年新增 `response_format: json_schema`，让模型直接输出 JSON 而非工具调用。

## 四、Anthropic 协议

Anthropic 的协议语义相似但语法不同：

### 4.1 调用

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "查询某城市当前天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
)

# 处理 tool_use block
for block in response.content:
    if block.type == "tool_use":
        result = get_weather(**block.input)
        # 把结果回填（注意是 user role，content 中含 tool_result）
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            ],
        })

# 第二次调用
final = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=messages,
)
```

### 4.2 OpenAI vs Anthropic 对比

| 维度 | OpenAI | Anthropic |
|---|---|---|
| 工具 schema | `type: function` + `function.parameters` | 直接 `name`, `input_schema` |
| 调用信号 | `tool_calls` 数组 | `content` 中含 `tool_use` 块 |
| 返回方式 | `role: tool` | `role: user` + `content: tool_result` |
| 并行调用 | 一条消息多 tool_calls | 单条 message 多 tool_use 块 |
| Schema 验证 | 严格 | 较灵活 |
| Structured Output | `response_format: json_schema` | 工具调用方式 |

## 五、Google Gemini 协议

```python
import google.generativeai as genai

model = genai.GenerativeModel('gemini-1.5-pro',
                               tools=[get_weather_function])

chat = model.start_chat()
response = chat.send_message("北京天气？")

# Gemini 把工具调用放在 parts 里
for part in response.parts:
    if part.function_call:
        result = get_weather(**dict(part.function_call.args))
        response = chat.send_message(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=part.function_call.name,
                        response={"result": result},
                    )
                )]
            )
        )
```

## 六、跨厂商兼容层

生产系统往往要支持多厂商。抽象层是关键：

```python
class ToolCall:
    id: str
    name: str
    arguments: dict

class UnifiedToolProvider:
    """统一的工具调用接口，屏蔽 OpenAI / Anthropic / Gemini 差异"""

    def __init__(self, provider: str):
        self.provider = provider

    def format_tools(self, tools: list[dict]) -> list[dict]:
        if self.provider == "openai":
            return [{"type": "function", "function": t} for t in tools]
        elif self.provider == "anthropic":
            return [{"name": t["name"], "description": t["description"],
                     "input_schema": t["parameters"]} for t in tools]
        # ...

    def parse_tool_calls(self, response) -> list[ToolCall]:
        if self.provider == "openai":
            return [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in response.choices[0].message.tool_calls
            ]
        elif self.provider == "anthropic":
            return [
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                )
                for block in response.content if block.type == "tool_use"
            ]
```

## 七、工具注册中心

生产环境的工具数量可能上百个。注册中心统一管理：

```python
class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, func, name=None, description=None, schema=None):
        """装饰器或显式注册"""
        tool = Tool(
            name=name or func.__name__,
            func=func,
            description=description or func.__doc__,
            schema=schema or self._auto_schema(func),
        )
        self.tools[tool.name] = tool
        return tool

    def get_schema_for_llm(self) -> list[dict]:
        """生成给 LLM 的工具 schema 列表"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            }
            for t in self.tools.values()
        ]

    def execute(self, name: str, args: dict) -> Any:
        tool = self.tools[name]
        return tool.func(**args)

registry = ToolRegistry()

@registry.register
def get_weather(city: str, unit: str = "celsius") -> dict:
    """查询某城市当前天气"""
    return requests.get(f"https://api.weather.com/{city}").json()
```

## 八、错误处理

### 8.1 工具执行失败

```python
def robust_execute(name, args):
    try:
        return registry.execute(name, args)
    except ValidationError as e:
        return {"error": "参数验证失败", "details": str(e)}
    except TimeoutError:
        return {"error": "工具超时"}
    except Exception as e:
        return {"error": "工具异常", "type": type(e).__name__}

# 把错误回填给 LLM，让它决定下一步
response = client.chat.completions.create(
    messages=[
        *history,
        {"role": "tool", "tool_call_id": tool_call.id,
         "content": json.dumps(robust_execute(...))},
    ]
)
```

### 8.2 模型生成非法参数

```python
def validate_args(tool_call, schema):
    try:
        validate(instance=json.loads(tool_call.function.arguments), schema=schema)
    except ValidationError as e:
        # 让 LLM 重新生成
        return {"error": "参数不符 schema", "details": e.message}
```

### 8.3 工具不可用

```python
class ToolRegistryWithFallback(ToolRegistry):
    def execute(self, name, args):
        if name not in self.tools:
            return {"error": f"工具 '{name}' 不存在", "available": list(self.tools.keys())}
        try:
            return super().execute(name, args)
        except Exception as e:
            return {"error": str(e)}
```

## 九、安全考量

### 9.1 危险工具的护栏

```python
DANGEROUS_TOOLS = {"delete_file", "drop_table", "transfer_money"}

def execute(self, name, args):
    if name in DANGEROUS_TOOLS:
        # 1. 二次确认
        if not self.user_confirmed(name, args):
            return {"error": "需要用户确认"}

        # 2. 审计日志
        audit_log.record(name=name, args=args, agent_id=AGENT_ID, user=USER_ID)

    return super().execute(name, args)
```

### 9.2 注入防御

```python
def sanitize_args(args: dict) -> dict:
    """防止 SQL 注入、路径穿越、命令注入"""
    sanitized = {}
    for k, v in args.items():
        if isinstance(v, str):
            # 检测危险模式
            if re.search(r";\s*(DROP|DELETE|UPDATE)", v, re.IGNORECASE):
                raise SecurityError(f"潜在 SQL 注入：{v}")
            if ".." in v or v.startswith("/etc"):
                raise SecurityError(f"潜在路径穿越：{v}")
        sanitized[k] = v
    return sanitized
```

### 9.3 速率限制

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=100, period=60)
def call_external_api():
    ...
```

## 十、性能优化

### 10.1 并行工具调用

```python
async def parallel_execute(tool_calls: list[ToolCall]) -> list[Any]:
    tasks = [
        async_execute(tc.name, tc.arguments)
        for tc in tool_calls
    ]
    return await asyncio.gather(*tasks)
```

### 10.2 Token 优化

工具 schema 占用 context，过多工具会浪费 token：

```python
def select_relevant_tools(query, all_tools, top_k=5):
    """只把相关工具发给 LLM"""
    # 用 embedding 检索相关工具
    query_emb = embedding_model.encode(query)
    tool_embs = embedding_model.encode([t.description for t in all_tools])
    scores = cosine_similarity([query_emb], tool_embs)[0]
    top_indices = np.argsort(scores)[-top_k:]
    return [all_tools[i] for i in top_indices]
```

## 小结

Function Calling 是 LLM 接入外部工具的标准协议。OpenAI / Anthropic / Google 各有语法差异，但语义相似——LLM 输出结构化 JSON 表示"调哪个工具、传什么参数"，应用代码负责执行。生产系统需要：跨厂商抽象层、统一工具注册中心、健壮的错误处理、安全护栏、性能优化。下一篇我们将进入 **code-execution-agents**——让 Agent 自己写并执行代码。
