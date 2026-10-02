# 主流向量库对比：FAISS、Pinecone、Qdrant、Milvus、Weaviate、pgvector

向量检索算法只是底层。真正在生产环境跑的，是把索引、分片、持久化、过滤、API、运维整合起来的**向量数据库**。本文对比六款主流方案——FAISS、Qdrant、Milvus、Weaviate、Pinecone、pgvector——从架构、特性、性能、生态四个维度展开，帮助你在不同场景下做选型。

## 一、横向对比总览

| 维度 | FAISS | Qdrant | Milvus | Weaviate | Pinecone | pgvector |
|---|---|---|---|---|---|---|
| **类型** | 库 | 数据库 | 数据库 | 数据库 | SaaS | Postgres 扩展 |
| **部署** | 自建 | 自建 / 云 | 自建 / 云 | 自建 / 云 | 仅云 | Postgres |
| **索引** | IVF/HNSW/PQ | HNSW | IVF/HNSW/PQ | HNSW | 自研 | HNSW/IVF |
| **过滤** | 不支持 | 原生 | 原生 | 原生 | 原生 | SQL |
| **混合检索** | 弱 | 强 | 强 | 强 | 中 | 强 |
| **分布式** | 需自实现 | 内置 | 内置（强） | 内置 | 内置 | PG 集群 |
| **易用性** | 中 | 高 | 中 | 高 | 极高 | 高 |
| **性能 (10M)** | 极高 | 高 | 高 | 中-高 | 高 | 中 |
| **生态** | Meta | 开源 | LF | 开源 | 商业 | Postgres |

## 二、FAISS：Meta 出品的算法库

FAISS（Facebook AI Similarity Search）是底层 C++ / Python 库，**不是数据库**——不持久化、不支持过滤、不带 API。

### 2.1 适用场景

- 一次性离线构建索引
- 单机内存够用（10M 级 1024 维）
- 需要极致速度

### 2.2 最小示例

```python
import faiss
import numpy as np

D = 768
N = 1_000_000
corpus = np.random.randn(N, D).astype("float32")
faiss.normalize_L2(corpus)                  # cosine → 内积

# 构建 HNSW 索引
index = faiss.IndexHNSWFlat(D, M=32, metric=faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.add(corpus)

# 持久化
faiss.write_index(index, "index.faiss")

# 查询
index = faiss.read_index("index.faiss")
query = np.random.randn(1, D).astype("float32")
faiss.normalize_L2(query)
D, I = index.search(query, k=10)
```

### 2.3 优缺点

| 优点 | 缺点 |
|---|---|
| 速度最快（>10K QPS 单机） | 无持久化 API（要自己存） |
| 算法最全 | 无元数据过滤 |
| Meta 持续维护 | 横向扩展需自实现 |
| 适合离线 / 嵌入式 | 无集群、无副本 |

**定位**：FAISS 是"算法引擎"，不是"数据库"。生产 RAG 通常把它包装在 Pinecone / Qdrant / Milvus 里。

## 三、Qdrant：Rust 写的现代开源向量库

Qdrant 是 Rust 实现的开源向量数据库，主打**性能 + 易用 + 过滤**。

### 3.1 启动

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 3.2 Python SDK

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")

# 创建 collection
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# 写入（带 payload）
client.upsert(
    collection_name="docs",
    points=[
        PointStruct(id=1, vector=[0.1, 0.2, ...], payload={"title": "doc1", "category": "tech"}),
        PointStruct(id=2, vector=[0.3, 0.1, ...], payload={"title": "doc2", "category": "news"}),
    ],
)

# 检索（带过滤）
hits = client.search(
    collection_name="docs",
    query_vector=[0.15, 0.18, ...],
    query_filter=Filter(must=[
        FieldCondition(key="category", match=MatchValue(value="tech")),
    ]),
    limit=10,
)
```

### 3.3 特色

- **Payload 过滤**：标量字段直接参与过滤，无需 pre/post filter。
- **多向量**：一条记录可存多个向量（图文多模态）。
- **稀疏 + 稠密混合**：原生支持 BM25 + 向量混合检索。
- **Rust 性能**：内存占用低，查询延迟稳定。

### 3.4 适用场景

RAG、推荐、多模态检索、中小规模（10M 级）。

## 四、Milvus：分布式能力最强的开源方案

Milvus（LF AI & Data 基金会）专为**超大规模**设计，分片 / 副本 / 多租户都是一等公民。

### 4.1 架构

```text
              ┌──────────────────┐
              │      Proxy       │  ← 无状态，路由 + 鉴权
              └────────┬─────────┘
                       ↓
         ┌─────────────┼─────────────┐
         ↓             ↓             ↓
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │QueryNode│  │QueryNode│  │QueryNode│   ← 检索
    │DataNode │  │DataNode │  │DataNode │   ← 存储
    └─────────┘  └─────────┘  └─────────┘
         ↑             ↑             ↑
         └─────────────┼─────────────┘
                       ↓
              ┌──────────────────┐
              │    Meta Store    │  ← etcd
              └──────────────────┘
```

### 4.2 部署方式

- **Milvus Lite**：Python 内嵌，适合单机 PoC。
- **Milvus Standalone**：单节点 Docker。
- **Milvus Distributed**：K8s Helm chart，分片 + 副本。

### 4.3 适用场景

- 10M - 100B 向量
- 强水平扩展需求
- 已有 K8s 运维能力

### 4.4 优缺点

| 优点 | 缺点 |
|---|---|
| 分布式能力最强 | 运维复杂（多组件） |
| 多副本 / 多租户 / 分区 | 资源占用相对大 |
| GPU 加速支持 | 学习曲线陡 |

## 五、Weaviate：自带 ML 模型的 GraphQL 数据库

Weaviate 的独特之处是**把向量化内置**：插入文本即可自动调用 embedding 模型生成向量。

### 5.1 特色

```python
import weaviate

client = weaviate.connect_to_local()

# 定义带 vectorizer 的 schema
client.collections.create(
    name="Article",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
    ],
)

# 插入原始文本，Weaviate 自动向量化
articles = client.collections.get("Article")
articles.data.insert({"title": "...", "content": "..."})

# 用 GraphQL 查询
response = articles.query.near_text(query="...", limit=10)
```

### 5.2 适用场景

- 想少写代码、让 DB 自动管理向量化
- GraphQL 友好
- 多模态（text2vec、img2vec、multi2vec）

### 5.3 优缺点

| 优点 | 缺点 |
|---|---|
| 一体化（向量化 + 存储 + 检索） | 性能不如 Qdrant |
| 内置多模态支持 | 大规模扩展性中等 |
| GraphQL 查询 | 商业版才有部分高级特性 |

## 六、Pinecone：完全托管的 SaaS

Pinecone 是 AWS 风格的"**完全托管**"——不用运维，开箱即用，按 pod 计费。

### 6.1 一行启动

```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("docs")

# upsert
index.upsert(vectors=[("id1", [0.1, ...], {"category": "tech"})])

# 查询
results = index.query(
    vector=[0.15, ...],
    top_k=10,
    filter={"category": {"$eq": "tech"}},
    include_metadata=True,
)
```

### 6.2 优缺点

| 优点 | 缺点 |
|---|---|
| 零运维 | 成本高（按存储 + QPS 计费） |
| 自动扩缩容 | 数据出域（合规风险） |
| 稳定 SLA | 算法不透明 |
| 企业级 SLA | vendor lock-in |

### 6.3 适用场景

- 不想运维向量库
- 流量波动大、需要弹性
- 预算充足

## 七、pgvector：Postgres 扩展

如果数据已经在 Postgres 里，pgvector 是阻力最小的方案——SQL 直接做向量检索。

### 7.1 用法

```sql
-- 启用
CREATE EXTENSION vector;

-- 建表
CREATE TABLE docs (
    id BIGSERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(768)
);

-- 建索引
CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops);

-- 查询（最近邻）
SELECT id, title, 1 - (embedding <=> :query) AS similarity
FROM docs
ORDER BY embedding <=> :query
LIMIT 10;
```

### 7.2 优缺点

| 优点 | 缺点 |
|---|---|
| 与业务数据同库 | 性能弱于专用向量库 |
| SQL 简单 | 大规模（10M+）较吃力 |
| 无需额外组件 | 索引选择少（HNSW / IVF） |
| 事务一致 | 备份 / 恢复要管 PG |

### 7.3 适用场景

- 数据规模小（< 5M）
- 已经在用 Postgres
- 需要 join 业务表

## 八、选型决策

```text
数据规模 / 场景              │ 推荐
─────────────────────────────┼─────────────
< 100K + 一次性 / 离线       │ FAISS / hnswlib
< 5M + 已有 Postgres        │ pgvector
5M - 50M + 想少运维         │ Pinecone / Weaviate Cloud
5M - 50M + 自建             │ Qdrant（最均衡）
10M - 1B + 分布式           │ Milvus
多模态 + 自动向量化          │ Weaviate
强混合检索（标量 + 向量）    │ Qdrant / Milvus
```

## 九、实战陷阱

1. **过滤 vs 召回率**：post-filter 可能在 top-K 里找不到满足过滤的。Qdrant 的 pre-filter 配合 HNSW 通常效果最好。
2. **索引参数与数据规模不匹配**：HNSW 的 M/ef 太大/太小都会让 recall 下降；IVF 的 nlist 太小簇太松，太大又慢。
3. **元数据没建索引**：`category = 'tech'` 没建索引会全表扫。
4. **冷启动延迟**：HNSW 索引需要 warm up，刚启动时前几条请求慢。
5. **向量归一化**：cosine 距离要求向量 L2-norm，否则 recall 大降。
6. **embedding 模型更换**：换 embedding 模型必须重建索引，旧索引的向量分布变了。

## 小结

向量库选型没有"银弹"。FAISS 是引擎，pgvector 是扩展，Qdrant / Weaviate 是开箱即用的开源，Milvus 是分布式王者，Pinecone 是零运维 SaaS。生产 RAG 系统的常见选择是 **Qdrant**（均衡）或 **Milvus**（大规模）。无论选哪个，都要关注召回率、过滤性能、运维成本三者的权衡。下一篇我们将深入 **RAG 中的向量索引**——具体怎么把文档切成块、怎么选 embedding、怎么评估检索质量。
