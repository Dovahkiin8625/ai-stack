# ML 监控基础：从指标到告警的全栈体系

模型上线只是开始。生产环境的输入分布会随时间漂移（COVID 前后的消费行为）、上游数据 pipeline 可能出错（缺失列、量纲变化）、模型本身可能退化（推理服务延迟飙升、GPU OOM）。ML 监控的目标是**早期发现这些问题**，把业务损失降到最低。本文梳理 ML 监控的三大维度（性能、数据、运营）、关键指标、告警策略与典型工具栈。

## 一、监控的三大维度

```text
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 性能监控     │  │ 数据监控     │  │ 运营监控     │
│ (Performance)│  │ (Data)       │  │ (Operational)│
│              │  │              │  │              │
│ 准确率       │  │ 输入分布     │  │ 延迟         │
│ 业务指标     │  │ 特征分布     │  │ QPS          │
│ 用户反馈     │  │ 异常值       │  │ 错误率       │
│              │  │ 漂移检测     │  │ GPU 利用率   │
│              │  │              │  │ 成本         │
└──────────────┘  └──────────────┘  └──────────────┘
```

**性能监控**问的是"模型对不对"，**数据监控**问的是"输入数据变了没有"，**运营监控**问的是"服务健康不健康"。三类问题需要不同的工具与方法。

## 二、性能监控：模型准不准

### 2.1 在线指标

最理想的监控是直接看**业务指标**：CTR、转化率、用户留存。但业务指标往往延迟大、被多因素干扰。

```python
# 业务指标埋点
@app.post("/v1/chat")
async def chat(req: ChatRequest):
    response = await llm.generate(req)
    metrics.labels(model=req.model, status="ok").inc()
    await log_business_event(user_id=req.user_id, event="chat_success")
    return response
```

需要与下游数据 pipeline（A/B 报表、用户行为日志）打通，定期聚合看趋势。

### 2.2 离线反馈

对分类 / 推荐类模型，可以延迟收集 ground truth：

```python
# 用户点击 → 实际标签
async def log_click(user_id, item_id, clicked):
    if clicked:
        # 标记为正样本
        await db.execute(
            "UPDATE predictions SET actual_label=true WHERE user_id=%s AND item_id=%s",
            (user_id, item_id),
        )
```

每天聚合，计算 **online accuracy / AUC**。和离线测试集的指标对比，发现漂移。

### 2.3 用户反馈

LLM 应用常见"点赞 / 点踩"：

```python
@app.post("/v1/feedback")
async def feedback(req: FeedbackRequest):
    metrics.labels(model=req.model, rating=req.rating).inc()
    await db.insert_feedback(req)
    return {"status": "ok"}
```

聚合后看：

- 好评率趋势
- 单个 prompt 模板的反馈
- 高频低分 prompt 聚类

## 三、数据监控：输入漂移

### 3.1 特征分布漂移

训练时 feature $X$ 的均值是 100、方差是 10；上线后某天突然均值变成 200、方差 50——输入分布变了，模型性能可能崩。

```python
from prometheus_client import Histogram

FEATURE_DISTRIBUTION = Histogram(
    "input_feature_value",
    "Distribution of input feature X",
    buckets=[0, 10, 50, 100, 200, 500, 1000],
)

@app.post("/v1/predict")
async def predict(req: PredictRequest):
    FEATURE_DISTRIBUTION.observe(req.feature_x)
    return await model.predict(req)
```

Grafana 上画出 histogram，看分位数漂移。

### 3.2 PSI / KS 检验

**Population Stability Index（PSI）** 是金融风控的经典方法：

$$
\text{PSI} = \sum_i (p_i^{\text{new}} - p_i^{\text{base}}) \cdot \ln\!\left(\frac{p_i^{\text{new}}}{p_i^{\text{base}}}\right)
$$

```python
import numpy as np

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    expected_counts = np.histogram(expected, breakpoints)[0] + 1e-6
    actual_counts   = np.histogram(actual, breakpoints)[0] + 1e-6
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct   = actual_counts / actual_counts.sum()
    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

# PSI < 0.1:  无漂移
# 0.1-0.25: 轻微漂移
# > 0.25:    严重漂移
```

KS 检验（Kolmogorov-Smirnov）也可以用于连续特征分布对比。

### 3.3 Embedding 漂移

LLM 应用特别关心 embedding 漂移——用户问题分布变了，RAG 检索效果可能崩。

```python
# 把每条 query 的 embedding 平均向量存入时序库
@app.post("/v1/chat")
async def chat(req: ChatRequest):
    emb = embedding_model.encode(req.prompt)
    EMB_MEAN_VECTOR.observe(emb.mean())       # 看一阶矩
    EMB_STD_VECTOR.observe(emb.std())         # 看二阶矩
    return await llm.generate(req)
```

更严谨：跟踪 embedding 主成分（PCA 前 k 个方向）的偏移。

## 四、运营监控：服务健康

### 4.1 四大黄金指标

借鉴 Google SRE 的"四大黄金信号"：

| 指标 | 公式 | 目标 |
|---|---|---|
| **Latency** | P50/P95/P99 响应时间 | < SLO |
| **Traffic** | QPS / RPM | 监控趋势 |
| **Errors** | 4xx/5xx 比例 | < 0.1% |
| **Saturation** | GPU 利用率 / 显存占用 | < 85% |

### 4.2 LLM 专用指标

```text
TTFT (Time To First Token)         # 用户感知延迟
TPOT (Time Per Output Token)       # 生成速度
generation_tokens_total            # 总 token 输出
prompt_tokens_total                # 总 token 输入
prefix_cache_hit_rate              # prefix cache 命中率
gpu_cache_usage_perc               # KV cache 显存占用
queue_wait_time                    # 排队时长
```

vLLM、TGI、TensorRT-LLM 都自带 Prometheus `/metrics` 端点。

### 4.3 成本监控

```python
COST_PER_REQUEST = Counter(
    "model_cost_usd_total",
    "Cumulative inference cost",
    labelnames=["model"],
)

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    response = await llm.generate(req)
    cost = response.usage.total_tokens * COST_PER_1K_TOKENS[req.model] / 1000
    COST_PER_REQUEST.labels(model=req.model).inc(cost)
    return response
```

按租户 / 模型 / 时间聚合，看成本曲线。

## 五、告警策略

### 5.1 静态阈值

```yaml
groups:
- name: ml_service_alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.99, rate(model_latency_seconds_bucket[5m])) > 2.0
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "P99 延迟 > 2s"
```

### 5.2 动态阈值（更稳）

```python
# 同比上周 / 环比上周
def alert_if_drop(metric: str, threshold: float = 0.05) -> bool:
    today = get_metric(metric, days=1)
    last_week = get_metric(metric, days=7, offset=7)
    return (today - last_week) / last_week < -threshold
```

避免大促 / 周末 / 工作日的周期性变化导致误报。

### 5.3 多窗口多阈值（MWM）

```yaml
- alert: AccuracyDrop
  expr: |
    (online_accuracy{env=prod} < 0.85)
    and (
      (online_accuracy{env=prod} < 0.85) unless on() (online_accuracy{env=prod} offset 1h > 0.85)
    )
  for: 30m
```

需要"持续 X 时间 + 显著变化"才触发，避免毛刺误报。

## 六、监控工具栈

| 工具 | 定位 | 特点 |
|---|---|---|
| **Prometheus** | 指标采集 | Pull 模型、PromQL、生态成熟 |
| **Grafana** | 可视化 | 仪表盘、告警 |
| **Evidently AI** | ML 漂移检测 | 数据 + 模型 drift |
| **WhyLabs** | ML 监控 SaaS | LLM-aware |
| **Arize Phoenix** | LLM observability | trace + eval |
| **LangSmith** | LangChain 生态 | LLM 应用专属 |
| **Weights & Biases** | 实验 + 监控 | 一体化 |
| **OpenTelemetry** | trace 标准 | 跨语言 |

## 七、典型仪表盘

```text
┌────────────────────────────────────┐
│ Latency (P50 / P95 / P99)          │
│  ─── P50  ── P95  ── P99          │
├────────────────────────────────────┤
│ QPS / Errors                       │
│  ─── QPS   ── 5xx rate            │
├────────────────────────────────────┤
│ GPU Util / VRAM / KV cache %      │
│  ─── util  ── vram ── kv cache    │
├────────────────────────────────────┤
│ Drift (PSI / KS / Embedding shift) │
│  ─── PSI   ── KS                  │
├────────────────────────────────────┤
│ Cost / 1k requests / day           │
│  ─── cost                          │
└────────────────────────────────────┘
```

## 小结

ML 监控是"模型上生产"之后的质量保障。性能监控关心"模型对不对"，数据监控关心"输入变了没"，运营监控关心"服务健康否"。三类监控配合 Prometheus + Grafana + 告警规则，构成完整的可观测体系。LLM 时代要额外关注 embedding 漂移、token 成本、prefix cache 命中率等专属指标。下一篇我们将深入 **漂移检测**——具体的算法与工程实现。
