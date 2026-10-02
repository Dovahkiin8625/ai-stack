# Feature Store：让特征可复用、可追溯、训练-服务一致

在传统 ML 系统里，特征工程是最大的人力成本。一个用户平均有上百个特征（年龄、消费力、活跃度、兴趣标签），这些特征往往**离线训练用 Hive/Spark 算，在线服务又用 Flink 重新算一遍**——两边对不上，模型效果立刻崩。Feature Store 把"特征定义、计算、存储、查询"抽象成一个统一系统，专门解决"训练-服务一致性"和"特征复用"两大痛点。

## 一、为什么需要 Feature Store

经典矛盾：

```text
离线训练（Spark, 批）           在线服务（Flink, 流）
─────────────────────         ────────────────────
user_features_v1.parquet       Redis cache key: hash
join 5 张大宽表                实时算近 7 天消费
耗时 4 小时                     耗时 < 50 ms
```

两套代码、两套 schema、两套时间口径——**结果几乎一定不一致**。线上推理时特征与训练分布不匹配，模型性能直接腰斩。

Feature Store 的目标就是**定义一次特征，同时被离线训练和在线服务查询使用**。

## 二、核心架构

```text
                   ┌─────────────┐
                   │ Feature Repo │  ← 特征定义（SQL/Python）
                   └──────┬──────┘
                          │
       ┌──────────────────┼──────────────────┐
       ↓                  ↓                  ↓
  ┌─────────┐       ┌──────────┐       ┌─────────┐
  │ Offline │       │ Online   │       │ Registry │
  │ Store   │       │ Store    │       │         │
  │(Parquet/│       │(Redis/   │       │Catalog  │
  │ Iceberg)│       │ DynamoDB)│       │         │
  └─────────┘       └──────────┘       └─────────┘
       ↑                  ↑
   训练任务            在线推理
```

三大组件：

- **Feature Registry**：所有特征的元数据中心（schema、版本、owner、文档）。
- **Offline Store**：存历史特征（Parquet/Iceberg），供离线训练。
- **Online Store**：存最新特征（Redis/DynamoDB），供在线推理延迟 < 10 ms 查询。

## 三、典型实现：Feast

[Feast](https://feast.dev/) 是当前最流行的开源 feature store。下面演示一个最小例子。

### 1. 定义特征

```python
# features.py
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

user = Entity(name="user_id", value_type=String)

user_stats_source = FileSource(
    path="s3://bucket/user_stats.parquet",
    timestamp_field="event_timestamp",
)

user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="lifetime_value", dtype=Float64),
        Field(name="last_30d_purchases", dtype=Int64),
    ],
    source=user_stats_source,
    online=True,
)
```

### 2. 物化到 Online Store

```bash
feast apply                     # 注册 feature view
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### 3. 离线训练查询

```python
from feast import FeatureStore
fs = FeatureStore(repo_path=".")

# 拿到训练样本：把 label 表与特征 join
training_df = fs.get_historical_features(
    entity_df=labels_df,                          # 包含 user_id 和 event_timestamp
    feature_refs=["user_stats:age", "user_stats:lifetime_value"],
).to_df()
```

### 4. 在线推理查询

```python
# 实时：按 user_id 拿最新特征
features = fs.get_online_features(
    features=["user_stats:age", "user_stats:lifetime_value"],
    entity_rows=[{"user_id": "u_123"}],
).to_dict()
# {"user_id": ["u_123"], "age": [28], "lifetime_value": [1234.5]}
```

同一份 `FeatureView` 定义，离线和在线都消费——这就是**训练-服务一致性**。

## 四、Point-in-Time Join

离线训练最容易踩的坑：**特征穿越未来**。如果用今天算的全量特征去训练昨天的样本，相当于"开了天眼"。

Feast 用 **point-in-time correct join** 解决：对每条样本 `(user_id, event_timestamp)`，只取该时刻之前已知的特征值。

```sql
-- 概念上等价于
SELECT s.user_id, s.event_timestamp, f.age, f.lifetime_value
FROM labels s
JOIN user_stats f
  ON s.user_id = f.user_id
 AND f.event_timestamp <= s.event_timestamp  -- 不许穿越！
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s.user_id, s.event_timestamp
  ORDER BY f.event_timestamp DESC
) = 1                                       -- 取最近一条
```

训练数据和生产数据用**同一份 join 逻辑**，从根本上避免穿越。

## 五、Streaming 特征

上面是批处理。生产里很多特征需要**实时更新**（用户最近一次点击、当前 session 时长）。Feast 通过 Spark / Flink 物化流式特征：

```python
from feast import FeatureView, StreamSource
from feast.infra.materialization.contrib.spark.spark_materialization_job import SparkMaterializationJob

click_stream_fv = FeatureView(
    name="user_click_stream_1h",
    entities=[user],
    schema=[
        Field(name="click_count_1h", dtype=Int64),
        Field(name="last_click_time", dtype=String),
    ],
    source=StreamSource(
        name="kafka_clicks",
        kafka_bootstrap_servers="localhost:9092",
        topic="user_clicks",
    ),
    online=True,
)
```

流式特征落 Redis 后，在线推理可以拿到秒级新鲜度的特征。

## 六、其他 Feature Store 方案

| 方案 | 部署 | 特点 |
|---|---|---|
| **Feast** | 开源，自托管 | 灵活、Python-first、偏在线场景 |
| **Tecton** | 商业（SaaS） | 企业级、与 Databricks 深度集成 |
| **Hopsworks** | 开源/商业 | Feature Store + ML Platform 一体 |
| **Databricks Feature Store** | 商业 | 与 Spark / MLflow 深度整合 |
| **AWS SageMaker Feature Store** | 商业 | 与 SageMaker 训练 / 推理整合 |

云厂商的方案适合"全家桶用户"，开源方案适合自建 + 灵活定制的团队。

## 七、Feature Store 不是银弹

不要为了用而用。Feature Store 适合：

- ✅ 多个团队复用同一组特征
- ✅ 离线 / 在线一致性成为痛点
- ✅ 特征数量大、变更频繁

不适合：

- ❌ 一次性 PoC（直接 Pandas 算）
- ❌ 仅离线训练、不上线
- ❌ 特征只有几个

## 小结

Feature Store 解决的是 ML 系统最被低估的问题——**特征一致性**。一个统一特征定义，同时被离线训练、在线推理、流式更新消费，从根本上消除训练-服务偏差。开源方案 Feast 适合自建，企业级方案（Tecton / SageMaker）适合预算充足、上规模的生产环境。下一篇我们将进入 **实验追踪**——如何让模型迭代可重现、可对比。
