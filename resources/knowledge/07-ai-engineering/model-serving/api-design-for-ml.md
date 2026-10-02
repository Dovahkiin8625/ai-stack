# ML API 设计：REST、gRPC、Streaming 与 OpenAI 兼容协议

模型服务化对外暴露的接口，看似是"加一层 FastAPI 就完事"，实际涉及协议选择（REST vs gRPC vs WebSocket）、请求/响应 schema、流式 vs 非流式、超时与取消、限流与配额。设计良好的 API 既能让上游集成简单，又能在故障时优雅降级。本文从协议选型、Schema 约定、流式响应、错误处理四个维度展开。

## 一、协议选型：REST vs gRPC vs WebSocket

| 协议 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| **REST / JSON** | 易调试、跨语言、生态广 | 文本冗长、无强 schema、流式不友好 | 对外 API、低 QPS、跨团队 |
| **gRPC** | 高性能、强 schema（Protobuf）、双向流 | 学习曲线、调试困难、防火墙不友好 | 内部服务间高频调用 |
| **WebSocket** | 双向实时、低开销 | 长连接管理复杂、需要心跳 | 实时对话、流式响应 |
| **SSE**（Server-Sent Events） | 单向流、HTTP 友好、断线可恢复 | 单向 | LLM token 流式输出 |
| **OpenAI 兼容** | 上游零学习成本、生态成熟 | 功能受限（受 OpenAI 协议约束） | LLM 服务的事实标准 |

LLM 服务几乎都采用 **OpenAI 兼容协议**——vLLM、TGI、LMDeploy、SGLang 默认都提供 `/v1/chat/completions`。这样上游可以用 OpenAI SDK 无缝迁移。

## 二、OpenAI 兼容协议

### 2.1 非流式

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b",
    "messages": [
      {"role": "user", "content": "介绍 vLLM"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "llama3-8b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "vLLM 是 UC Berkeley 推出的高性能 LLM 推理引擎..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 47,
    "total_tokens": 65
  }
}
```

### 2.2 流式（SSE）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ "stream": true, "...": "..." }'
```

响应是 `text/event-stream`：

```text
data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"vLLM "},"index":0}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"是"},"index":0}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"高性能"},"index":0}]}

data: [DONE]
```

每个 chunk 都带 `delta` 字段——客户端只解析增量、拼接即可。

## 三、自定义 REST API 设计

如果不是 LLM，而是图像分类、推荐、CV 检测等，可以自定义 REST。下面以分类为例。

### 3.1 路由设计

```text
POST   /v1/models/{model_id}/predict       单次预测
POST   /v1/models/{model_id}/predict/batch 批量预测
GET    /v1/models/{model_id}/schema         模型 schema 自描述
GET    /v1/models                           列出可用模型
GET    /healthz                             健康检查
GET    /metrics                             Prometheus 指标
```

### 3.2 Schema（OpenAPI）

```python
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    instances: list[dict] = Field(..., description="输入实例列表")
    parameters: dict | None = Field(default=None, description="推理参数")

class Prediction(BaseModel):
    scores: list[float] = Field(..., description="各类别概率")
    label: str = Field(..., description="预测标签")

class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
```

### 3.3 批量与单次合一

```python
@app.post("/v1/models/{model_id}/predict")
async def predict(model_id: str, req: PredictRequest):
    model = registry.get(model_id)
    preds = model.predict_batch(req.instances, **(req.parameters or {}))
    return {"predictions": preds, "model_version": model.version}
```

支持单条：

```json
{ "instances": [{"image_url": "https://..."}] }
```

也支持多条：

```json
{ "instances": [{"image_url": "..."}, {"image_url": "..."}, ...] }
```

## 四、流式响应：SSE vs WebSocket

### 4.1 SSE 实现（推荐 LLM 流式）

```python
from fastapi.responses import StreamingResponse
import asyncio

async def token_stream(prompt: str):
    async for token in llm.stream_generate(prompt):
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0)  # 让出事件循环

@app.post("/v1/chat/stream")
async def stream_chat(req: ChatRequest):
    return StreamingResponse(
        token_stream(req.prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 禁用 Nginx buffering
        },
    )
```

### 4.2 WebSocket 实现（双向）

```python
from fastapi import WebSocket

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    while True:
        msg = await ws.receive_json()
        prompt = msg["prompt"]

        async for token in llm.stream_generate(prompt):
            await ws.send_json({"token": token})

        await ws.send_json({"finish_reason": "stop"})
```

WebSocket 适合**双向交互**场景（用户能中途打断、修改 prompt）。

## 五、超时、取消、重试

### 5.1 服务端超时分级

```python
@app.middleware("http")
async def timeout_middleware(request, call_next):
    try:
        return await asyncio.wait_for(
            call_next(request),
            timeout=request_timeout(request),
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": "inference_timeout", "latency_ms": 30_000},
        )

def request_timeout(req):
    if "/stream" in req.url.path:
        return 60.0    # 流式长
    return 5.0         # 普通请求
```

### 5.2 客户端取消

```python
async def stream_chat(req: ChatRequest, request: Request):
    async def gen():
        async for token in llm.stream_generate(req.prompt):
            if await request.is_disconnected():
                llm.cancel()         # 通知 worker 停
                break
            yield f"data: {token}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

### 5.3 客户端重试策略

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
async def chat_with_retry(prompt: str):
    return await client.chat(prompt)
```

**幂等性**：重试必须保证服务侧是幂等的——同 prompt 不产生副作用。如果有状态（写库、扣费），用 `idempotency_key`。

## 六、限流、配额与多租户

```python
from fastapi import HTTPException, Depends

@app.post("/v1/chat")
async def chat(req: ChatRequest, user: User = Depends(authenticate)):
    if not quota_checker.allow(user.id, tokens=count_tokens(req.prompt)):
        raise HTTPException(429, "rate limit exceeded")

    response = await llm.generate(req)
    quota_checker.consume(user.id, tokens=response.usage.total_tokens)
    return response
```

多租户隔离的三种粒度：

- **QPS 限流**：每用户每秒最多 N 次请求。
- **Token 配额**：每用户每天最多 M tokens（适合 LLM）。
- **GPU 占用**：每租户最多 N 张卡（硬隔离）。

## 七、可观测性埋点

```python
from prometheus_client import Counter, Histogram

REQ_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Inference latency",
    labelnames=["model", "status"],
)
TOKENS_GENERATED = Counter(
    "model_tokens_generated_total",
    "Total tokens generated",
    labelnames=["model"],
)

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    with REQ_LATENCY.labels(model=req.model, status="ok").time():
        response = await llm.generate(req)
    TOKENS_GENERATED.labels(model=req.model).inc(response.usage.completion_tokens)
    return response
```

## 小结

ML API 设计的核心是**协议标准化 + 流式友好 + 故障优雅**。LLM 服务首推 **OpenAI 兼容 + SSE 流式**，复用成熟生态；自定义 ML 服务用 **REST + OpenAPI 描述**，兼顾易用性。生产级 API 必须有超时、取消、重试、限流、配额、可观测性全套能力。下一篇我们将进入 **批处理与调度**——如何用 vLLM 的 continuous batching、动态调度等机制把 GPU 利用率推到极致。
