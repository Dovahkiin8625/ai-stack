# LLM 应用架构模式：API 集成、缓存与队列

把一个能"和 LLM 对话"的 demo 变成能扛住生产流量的服务，是一件工程上不平凡的事。本文梳理 LLM 应用最常见的几类架构组件——API 网关、Function Calling、缓存、队列、流式输出与可观测性——并给出一个最小可运行的 FastAPI 网关代码。

## 一、典型分层架构

一个生产级 LLM 应用通常长这样：

```
[Web / App]
    │
    ▼
[API Gateway]         鉴权、路由、限流
    │
    ▼
[LLM Service]   ──► [Vector DB]   （RAG）
    │           ──► [Tools / Functions] (Function Calling)
    ▼
[Cache]   [Queue]   [Observability]
```

各层关注点：

- **Gateway** 负责"对外契约"：统一鉴权、用户配额、流式响应适配。
- **LLM Service** 负责"业务逻辑"：拼 prompt、调模型、调用工具。
- **Vector DB** 提供长记忆与领域知识。
- **Tools** 让模型"伸手操作世界"——查订单、发邮件、写数据库。
- **队列 / 缓存** 解决突发流量与重复请求。
- **可观测性** 让"延迟、错误、token 用量"一目了然。

## 二、Function Calling / Tool Use

Function Calling 把"调一个本地函数"这件事交给模型来决策。协议上大致是：

```jsonc
// 1) 我们告诉模型有哪些可用工具
{"name": "get_weather", "description": "查某城市天气",
 "parameters": {"type": "object",
   "properties": {"city": {"type": "string"}}, "required": ["city"]}}

// 2) 模型决定要不要调用、参数是什么
{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Beijing"}}]}

// 3) 我们执行函数，把结果回填给模型
// 4) 模型再产出最终回复
```

下面是一段最小代码（伪 SDK，逻辑等价于 OpenAI / Anthropic tool use）：

```python
import json
from openai import OpenAI

client = OpenAI()
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查某城市当前天气",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]},
    },
}]

def get_weather(city: str) -> str:
    return f"{city} 今天晴，25°C"  # 实际可接天气 API

def chat(user_msg: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_msg}],
        tools=tools,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = get_weather(**args)
            # 把工具结果回填，让模型组织自然语言回复
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": user_msg},
                    msg,
                    {"role": "tool", "tool_call_id": tc.id,
                     "content": result},
                ],
                tools=tools,
            )
        return resp.choices[0].message.content
    return msg.content
```

工程上要点：

- **强约束参数 schema**：用 JSON Schema 描述，参数错误率显著下降。
- **幂等工具**：模型可能对同一工具调函数重复请求。
- **超时与降级**：工具内部一定要有超时，否则会拖垮整个请求。

## 三、Prompt 缓存

LLM 调用贵，且重复问题很多。常见两层缓存：

1. **Exact-match cache**：用 (model, prompt hash) 做 key。命中率不高，但实现极简。
2. **Semantic cache**：把 prompt embedding 存起来，新查询和库内余弦相似度超过阈值就直接复用。

简单实现：

```python
import hashlib, time
_cache: dict[str, tuple[float, str]] = {}

def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()

def llm_call(model: str, prompt: str, ttl: int = 600) -> str:
    k = cache_key(model, prompt)
    if k in _cache:
        ts, val = _cache[k]
        if time.time() - ts < ttl:
            return val
    out = real_llm(model, prompt)         # 实际模型调用
    _cache[k] = (time.time(), out)
    return out
```

> 注意：很多 provider（OpenAI / Anthropic / DeepSeek）已经在**服务端**实现了 prompt cache，会自动识别长 system / 长前缀，命中可省 70%+ 成本。代码层缓存命中率低、且容易污染，慎用。

## 四、限流与队列

LLM 的"成本 = token 数 × 单价"，因此**限流是必选项**，而不是锦上添花。常用算法是 token bucket。

设令牌以速率 $r$ 补满，桶容量为 $B$。一个请求消耗一个令牌，桶空则拒绝。直观上：短时突发可放过，长期不超过 $r$。

工程实现里常用 Redis + Lua 做分布式令牌桶，或直接用 API 网关（Kong / Envoy + ratelimit filter）。

## 五、流式输出（SSE / WebSocket）

用户对"逐字出现"的体感延迟远低于"等几秒出全文"。两种主流方式：

- **SSE（Server-Sent Events）**：单向流，HTTP/1.1 友好，断线重连简单。OpenAI / Anthropic 的 streaming 接口都是 SSE。
- **WebSocket**：双向流，适合需要服务端主动推送、又需要客户端回传控制信号的 chat 场景。

FastAPI 里一行就能开 SSE：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/chat/stream")
def chat_stream(q: str):
    def gen():
        for token in stream_llm(q):
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": "[DONE]"}
    return StreamingResponse(gen(), media_type="text/event-stream")
```

## 六、可观测性

最起码要记录四件事：

1. **请求级**：prompt 长度、模型、温度、用户 ID。
2. **响应级**：生成 token 数、首字延迟（TTFT）、总延迟、是否流式。
3. **错误级**：超时、4xx/5xx、工具执行失败。
4. **成本级**：按 prompt 或 model、按用户/团队的 token 用量。

落地时：

- **日志**：结构化 JSON（`structlog`）。
- **Tracing**：OpenTelemetry，串联 gateway → service → upstream。
- **Token 统计**：每次 LLM 调用都从 usage 字段读出，落到 ClickHouse / BigQuery。

## 七、FastAPI 最小可用 LLM 网关

```python
import time, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
log = logging.getLogger("llm-gateway")
client = OpenAI()

class ChatReq(BaseModel):
    user_id: str
    messages: list
    model: str = "gpt-4o-mini"

@app.post("/v1/chat")
def chat(req: ChatReq):
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=req.model,
            messages=req.messages,
            temperature=0.2,
        )
        usage = resp.usage
        log.info("llm_call",
                 extra={"uid": req.user_id, "model": req.model,
                        "in_t": usage.prompt_tokens,
                        "out_t": usage.completion_tokens,
                        "latency_ms": int((time.perf_counter()-t0)*1000)})
        return {"reply": resp.choices[0].message.content,
                "usage": {"in": usage.prompt_tokens,
                          "out": usage.completion_tokens}}
    except Exception as e:
        log.exception("llm_error", extra={"uid": req.user_id})
        raise HTTPException(status_code=502, detail=str(e))
```

加上鉴权中间件、Redis 限流、Prometheus `/metrics`，就是生产可用了。**架构的本质就是**把 LLM 当成一个**外部昂贵依赖**，按它之前的成本来设计——超时、限流、缓存、可观测，缺一不可。

## 小结

LLM 应用架构的关键词是：**网关层管流控、服务层管业务、缓存层省钱、可观测层保命**。Function Calling 让模型"长出手脚"、缓存与限流让单 token 成本可承受、流式输出让用户体验更顺。下一篇会聊这些架构之上最常见的应用形态——chatbot。