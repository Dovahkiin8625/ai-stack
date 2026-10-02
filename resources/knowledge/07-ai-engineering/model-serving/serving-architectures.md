# 模型服务化架构：在线、异步、批处理三种范式

模型部署到生产环境时，"如何接收请求、返回结果"看起来简单，实则有三种本质不同的架构：**在线推理（低延迟同步）、异步推理（任务队列）、批处理（高吞吐离线）**。选错架构，要么把 GPU 烧光，要么让用户体验崩塌。本文梳理三种范式的适用场景、工程实现，以及 LLM 时代下的混合架构。

## 一、三种服务范式

| 范式 | 延迟要求 | 吞吐量 | 典型场景 |
|---|---|---|---|
| **在线推理（Online）** | 100ms-2s | 中 | 搜索、推荐、对话、实时分类 |
| **异步推理（Async）** | 分钟-小时 | 中-高 | 图像生成、长文档摘要、批量标注 |
| **批处理（Batch）** | 小时-天 | 极高 | 离线报表、风控打分、批量 embedding |

## 二、在线推理：同步 HTTP / gRPC

在线推理是用户感知最强的链路——点开 APP 就要等结果。

### 典型架构

```text
                ┌──────────────────┐
Client ──HTTPS──→ API Gateway      │
                │ (限流 / 鉴权)   │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ 模型服务集群      │
                │ (vLLM / Triton) │
                │ ┌──────────────┐ │
                │ │ K8s Pod ×N  │ │
                │ │ + HPA       │ │
                │ └──────────────┘ │
                └────────┬─────────┘
                         ↓
                  Feature Store / Cache
```

### 实现：FastAPI + vLLM

```python
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=1)

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    outputs = llm.generate([req.prompt], params)
    return {"response": outputs[0].outputs[0].text}
```

### 关键能力

- **流式响应**：SSE / WebSocket，让用户看到 token 逐字生成。
- **批处理合并**：用 vLLM 的 continuous batching 提升吞吐。
- **熔断降级**：下游超时返回默认结果，避免雪崩。
- **A/B 测试**：网关层按用户 ID 哈希分配流量。

## 三、异步推理：任务队列 + Worker

当任务耗时较长（>10s）或需要重试时，用异步架构：API 立即返回 `task_id`，Worker 后台执行。

### 典型架构

```text
Client ──POST /tasks──→ API Server ──→ Message Queue (Kafka / Redis)
                                            ↓
                                    Worker Pool (Celery / Ray)
                                            ↓
                                    Model Serving (Triton)
                                            ↓
                                    Result Store (Redis / DB)
Client ──GET /tasks/{id}──→ API Server ──→ Result Store
```

### 实现：Celery + Redis

```python
from celery import Celery
import time

app = Celery("tasks", broker="redis://redis:6379", backend="redis://redis:6379")

@app.task(bind=True, max_retries=3)
def long_predict(self, image_url: str) -> dict:
    # 下载图片
    image = download(image_url)
    # 跑模型（可能很慢）
    result = model.predict(image)
    return {"label": result.label, "score": result.score}

# API 层：入队 + 查状态
from fastapi import FastAPI

@app.post("/api/predict")
async def predict(req: PredictRequest):
    task = long_predict.delay(req.image_url)
    return {"task_id": task.id}

@app.get("/api/predict/{task_id}")
async def get_result(task_id: str):
    task = long_predict.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return {"status": "done", "result": task.result}
    return {"status": task.state}
```

### 适用场景

- 文生图（Stable Diffusion，单图 5-30s）
- 视频理解（分钟级）
- 长文档摘要（百万 token 级）
- 批量 embedding

### 调度策略

- **优先级队列**：付费用户优先。
- **公平调度**：每用户每分钟最多 N 个任务。
- **成本上限**：限速避免 GPU 过载。

## 四、批处理：MapReduce 模式

当任务是"把这一亿条记录都跑一遍模型"时，用批处理：

```text
Input (S3/Parquet) ──→ Spark/Flink ──→ Model Worker ×N ──→ Output (Parquet)
```

### 实现：Spark + PyTorch

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
import pandas as pd
import torch

spark = SparkSession.builder.appName("batch_predict").getOrCreate()

# 加载模型（每个 executor 加载一次）
_model = None
def get_model():
    global _model
    if _model is None:
        _model = torch.jit.load("model.pt")
        _model.eval()
    return _model

@pandas_udf("string")
def predict_udf(texts: pd.Series) -> pd.Series:
    model = get_model()
    with torch.inference_mode():
        embeddings = model.encode(texts.tolist())
    return pd.Series([str(e.tolist()) for e in embeddings])

df = spark.read.parquet("s3://data/articles/")
df_with_emb = df.withColumn("embedding", predict_udf(df["text"]))
df_with_emb.write.parquet("s3://output/articles_with_emb/")
```

### 适用场景

- 离线报表：用户行为预测、风险打分。
- 全量 embedding：把语料库全部向量化。
- 反向索引构建：把候选集全部打分排序。
- 周期性 ETL：每天凌晨跑一次。

### 优化技巧

- **GPU 集群调度**：YARN / K8s 把 task 调度到带 GPU 的节点。
- **模型 worker pool**：每节点一份模型权重，避免重复加载。
- **数据分片 + 并行度**：partition 数 ≈ GPU 总数 × 2-4。
- **失败重试**：每个 task 独立，局部失败不影响整体。

## 五、混合架构：现代 LLM 应用

复杂 LLM 应用通常三种范式混用：

```text
用户请求 ──→ 在线 API（聊天补全）
                ↓
            异步任务（文档解析、向量化）
                ↓
            批处理（知识库全量重建）
```

例：RAG 系统

1. **在线**：用户问题 → embedding 检索 → LLM 生成回答（<2s）。
2. **异步**：用户上传文档 → OCR / 解析 → embedding 入库（<1min）。
3. **批处理**：每周全量重建知识库索引（小时级）。

## 六、选型决策

```text
                ┌──────────────────┐
                │ 任务延迟要求？    │
                └─────────┬────────┘
                          │
            ┌─────────────┼─────────────┐
            ↓             ↓             ↓
        < 2s         2s - 1h        > 1h / 周期性
            │             │             │
        在线 API     异步队列        批处理
        vLLM/Triton  Celery/RQ      Spark/Ray
            │             │             │
        ┌───┴───┐         │             │
        │流量大？│         │             │
        └───┬───┘         │             │
        Yes│ No           │             │
            │ │           │             │
        K8s  单机         │             │
        HPA  FastAPI      │             │
                         │             │
                    Worker 池      GPU 集群
                    + 队列         调度
```

## 七、关键工程能力

1. **可观测性**：所有三种范式都要接 Prometheus + Grafana。
2. **幂等性**：异步 / 批处理任务必须可重入。
3. **资源隔离**：在线 API 不能被批处理任务抢占 GPU。
4. **配额管理**：每租户 quota，避免一个用户打满集群。
5. **成本归因**：把 GPU 时长归到具体业务 / 用户。

## 小结

模型服务化没有"银弹架构"。在线、异步、批处理三种范式对应不同延迟与吞吐需求，往往需要组合使用。LLM 应用尤其要把"实时对话"和"离线索引构建"分开调度，否则一次全量重建就把实时 API 拖垮。下一篇我们将进入 **API 设计 for ML**——REST、gRPC、WebSocket、流式响应的取舍。
