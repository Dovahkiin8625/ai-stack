# ML 可观测性栈：Metrics、Logs、Traces 整合

传统 SRE 的可观测性三支柱是 **Metrics（指标）、Logs（日志）、Traces（追踪）**。ML 系统的可观测性在此基础上还要加入 **model-specific signals（数据漂移、embedding 分布、token 成本）**。本文梳理三支柱在 ML 系统中的具体形态、OpenTelemetry 标准化整合、LLM 应用 trace 的特殊性，以及典型工具栈选型。

## 一、三支柱在 ML 中的形态

```text
                ┌──────────────────────────────────────┐
                │         ML System Observability      │
                └──────────────────────────────────────┘
                     │              │              │
                ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
                │ Metrics │    │  Logs   │    │ Traces  │
                └────┬────┘    └────┬────┘    └────┬────┘
                     │              │              │
            QPS, latency    请求详情,     端到端调用链
            GPU util        模型输出      LLM/tool/retriever
            Token cost      异常 stack    各阶段耗时
            PSI drift       用户反馈     token 级 timing
```

## 二、Metrics：聚合数字

Metrics 是"**预聚合**"的数值，存储成本低、查询快。适合监控、告警、仪表盘。

### 2.1 常见指标

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Counter: 只增不减
REQ_TOTAL = Counter("model_request_total", "Total requests", ["model", "status"])
TOKEN_TOTAL = Counter("model_tokens_total", "Total tokens", ["model", "direction"])

# Histogram: 分布（自动算分位数）
LATENCY = Histogram(
    "model_latency_seconds",
    "Inference latency",
    labelnames=["model", "phase"],    # phase=prefill|decode
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
)

# Gauge: 当前值
GPU_UTIL = Gauge("gpu_utilization_ratio", "GPU SM util", ["gpu_id"])
GPU_MEM = Gauge("gpu_memory_used_bytes", "GPU VRAM used")

# Summary: 类似 histogram，但算分位数更准
TTFT = Summary("model_ttft_seconds", "Time to first token")
```

### 2.2 业务指标

```python
USER_SATISFACTION = Gauge("user_satisfaction_24h", "24h average rating", ["model"])
BUSINESS_KPI = Counter("business_kpi_total", "Business KPI", ["event_type"])

@app.post("/v1/feedback")
async def feedback(req: FeedbackRequest):
    USER_SATISFACTION.labels(model=req.model).set(req.rating)
    BUSINESS_KPI.labels(event_type="positive_feedback").inc()
    return {"ok": True}
```

### 2.3 暴露端点

```python
from prometheus_client import make_asgi_app

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

Prometheus / Grafana Alloy 定期 scrape。

## 三、Logs：详细记录

Logs 是"**离散事件**"，含完整上下文。适合调试、审计、根因分析。

### 3.1 结构化日志

```python
import structlog

log = structlog.get_logger()

@app.post("/v1/chat")
async def chat(req: ChatRequest, user: User = Depends(auth)):
    log.info("chat_request",
        user_id=user.id,
        model=req.model,
        prompt_len=len(req.prompt),
        prompt_hash=hashlib.md5(req.prompt.encode()).hexdigest()[:16],   # 不存原文
    )
    try:
        response = await llm.generate(req)
        log.info("chat_response",
            user_id=user.id,
            model=req.model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            latency_ms=response.latency_ms,
        )
        return response
    except Exception as e:
        log.error("chat_failed",
            user_id=user.id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise
```

**关键原则**：

- **结构化（JSON）**：便于 ELK / Loki 检索。
- **不要存敏感数据**：原文 prompt / 回复用 hash 或截断。
- **关联 ID**：每个请求一个 `request_id`，贯穿整条链路。

### 3.2 LLM 专用日志

LLM 应用必须存：

```python
log.info("llm_call",
    request_id=request_id,
    model=req.model,
    messages=[{"role": m.role, "content": m.content} for m in req.messages],
    response=response.content,
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens,
    temperature=req.temperature,
    tool_calls=[...],
)
```

便于事后排查"为什么这个回答这么奇怪"。

## 四、Traces：调用链路

Trace 记录"一个请求在多个服务间的完整路径与耗时"。LLM 应用尤其需要 trace，因为请求往往经过 `retriever → reranker → prompt_assembler → llm → post_processor` 多个环节。

### 4.1 OpenTelemetry 标准化

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# 配置 tracer
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# 自动 instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

tracer = trace.get_tracer(__name__)
```

### 4.2 手工 span

```python
@app.post("/v1/chat")
async def chat(req: ChatRequest):
    with tracer.start_as_current_span("rag_chat") as span:
        span.set_attribute("user.id", req.user_id)
        span.set_attribute("model", req.model)

        with tracer.start_as_current_span("retriever") as sub:
            docs = await retriever.search(req.prompt, top_k=5)
            sub.set_attribute("docs.count", len(docs))

        with tracer.start_as_current_span("llm_generate") as sub:
            sub.set_attribute("model", req.model)
            response = await llm.generate(build_prompt(req.prompt, docs))
            sub.set_attribute("tokens.in", response.usage.prompt_tokens)
            sub.set_attribute("tokens.out", response.usage.completion_tokens)

    return response
```

### 4.3 LLM 特殊 trace

LLM 生成是长过程，单一 span 看不到 token-level 进度。用 `add_event` 记录中间事件：

```python
with tracer.start_as_current_span("llm_generate") as span:
    span.add_event("llm_start", {"model": req.model})
    async for chunk in llm.stream_generate(prompt):
        span.add_event("token", {"token_id": chunk.token_id, "latency_ms": chunk.latency_ms})
    span.add_event("llm_end", {"total_tokens": total})
```

trace UI（Jaeger / Tempo / Arize Phoenix）可以看到：

```text
rag_chat (1.2s)
├── retriever (80ms)          docs=5
├── reranker (30ms)
├── llm_generate (1000ms)     tokens_in=1200, tokens_out=200
│   ├── llm_start
│   ├── token × 200
│   └── llm_end
└── post_process (50ms)
```

## 五、Metrics + Logs + Traces 关联

通过 `request_id` / `trace_id` 关联三者：

```python
import uuid
from opentelemetry import trace

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    request_id = str(uuid.uuid4())
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")

    log.info("chat_start",
        request_id=request_id,
        trace_id=trace_id,                 # 关联 metrics/logs/traces
        user_id=req.user_id,
    )

    REQ_TOTAL.labels(model=req.model, status="started").inc()

    try:
        response = await llm.generate(req)
        REQ_TOTAL.labels(model=req.model, status="ok").inc()
        LATENCY.labels(model=req.model, phase="total").observe(response.latency_ms)
        log.info("chat_success", request_id=request_id, trace_id=trace_id, ...)
        return response
    except Exception as e:
        REQ_TOTAL.labels(model=req.model, status="error").inc()
        log.error("chat_failed", request_id=request_id, trace_id=trace_id, error=str(e))
        raise
```

排查流程：

1. Grafana 看 metric 发现 P99 latency 飙升。
2. 拉出错误 request_id 列表。
3. 用 trace_id 到 Jaeger 看完整调用链。
4. 找到瓶颈 span（如 retriever 慢了）。

## 六、LLM 时代的可观测性工具

| 工具 | 类型 | 特点 |
|---|---|---|
| **Prometheus** | metrics | Pull 模型、生态成熟 |
| **Grafana** | 可视化 | 仪表盘 + 告警 |
| **Loki** | logs | 与 Grafana 深度集成 |
| **Tempo / Jaeger** | traces | OpenTelemetry 兼容 |
| **ELK / EFK** | logs | 全文搜索强 |
| **Arize Phoenix** | LLM trace | token-level 可视化 |
| **LangSmith** | LangChain | LLM 应用专属 |
| **WhyLabs** | ML 监控 | data drift + model |
| **Datadog** | 一体化 | 商业 SaaS |
| **Helicone** | LLM proxy | 轻量 LLM 可观测 |

## 七、典型架构

```text
                ┌─────────────────────────┐
                │      ML Application     │
                │  (FastAPI + LangChain)  │
                └────────────┬────────────┘
                             │ OTel SDK
                ┌────────────┴────────────┐
                ↓                          ↓
        ┌──────────────┐          ┌──────────────┐
        │ Prometheus   │          │ OTel         │
        │ (scrape      │          │ Collector    │
        │  /metrics)   │          │ (OTLP)       │
        └──────┬───────┘          └──────┬───────┘
               ↓                          ↓
        ┌──────────────┐          ┌──────────────┐
        │ Grafana      │          │ Tempo /      │
        │ (dashboards) │          │ Elasticsearch│
        └──────────────┘          └──────┬───────┘
                                        ↓
                                ┌──────────────┐
                                │ Grafana      │
                                │ (correlate)  │
                                └──────────────┘
```

## 八、可观测性反模式

1. **指标全打**：每个变量都 log，存储爆。要按"业务关键度"分级。
2. **日志只输出到 stdout**：无法搜索。要走 ELK / Loki。
3. **trace 缺失**：只看到 metric 升高，不知道为什么。trace 是定位根因的关键。
4. **不关联三者**：metrics / logs / traces 各管各的，排查要"人肉 join"。
5. **没有 request_id**：所有日志是平行的，无法串成一条链路。

## 小结

ML 可观测性 = **Metrics（聚合）+ Logs（细节）+ Traces（路径）**，加上 LLM 时代的专属信号（embedding 漂移、token 成本、tool 调用）。三者通过 `trace_id` 关联，构成完整观测体系。OpenTelemetry 是行业标准，工具栈选型上 Prometheus + Grafana + Tempo/Jaeger 是开源事实组合，商业方案（Datadog、LangSmith）适合预算充足。下一篇我们将进入 **向量数据库**——RAG、语义搜索背后的核心技术。
