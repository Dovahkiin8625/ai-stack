# 向量检索基础：从余弦相似度到 ANN 算法

RAG、语义搜索、推荐系统——这些应用背后都离不开一个核心数据结构：**向量索引**。给定 N 个 1024 维向量，找出与查询向量最近的 top-K，朴素做法是 $O(ND)$ 全量比对，但 N 达到百万级就完全不可行。本文梳理向量距离度量、近似最近邻（ANN）算法家族、HNSW / IVF / PQ 三大主流索引，以及工程选型要点。

## 一、距离度量

### 1.1 余弦相似度（Cosine Similarity）

```text
similarity = (A · B) / (||A|| × ||B||)
```

- **范围**：[-1, 1]，归一化后 [0, 1]
- **直观**：衡量**方向**一致性，忽略大小
- **适用**：文本 embedding（已 L2-norm）

### 1.2 点积（Inner Product / Dot Product）

```text
similarity = A · B
```

- **范围**：(-∞, +∞)
- **适用**：未归一化向量、或与向量大小相关时

### 1.3 欧氏距离（Euclidean / L2）

```text
distance = ||A - B||₂ = sqrt(Σ(Aᵢ - Bᵢ)²)
```

- **范围**：[0, +∞)，越小越相似
- **适用**：图像 embedding、低维稠密向量

### 1.4 选型

| Embedding 类型 | 推荐距离 |
|---|---|
| OpenAI text-embedding-3 | cosine |
| BGE / M3E | cosine |
| Sentence-BERT | cosine |
| CLIP 图文 | cosine |
| 推荐系统隐向量 | dot product |
| 图像检索 (raw pixel) | L2 |

大多数 embedding 模型在训练时就把向量归一化，所以 cosine 与 dot product 等价。

## 二、精确检索 vs 近似检索

### 2.1 精确最近邻（Exact KNN）

```python
import torch

def exact_topk(query: torch.Tensor, corpus: torch.Tensor, k: int = 10):
    """O(N × D) 时间复杂度"""
    # query: (D,), corpus: (N, D)
    scores = corpus @ query                       # (N,)
    topk_scores, topk_indices = torch.topk(scores, k)
    return topk_indices, topk_scores
```

精确，但 N=10M 时单条查询要 ~40s（FP32，CPU）。

### 2.2 近似最近邻（ANN）

放弃**严格**的 top-K，换**概率上**接近 top-K 的结果：

- **Recall@10 = 0.95**：95% 概率在 top-10 里包含真实 top-1
- 速度提升 **100-1000×**

ANN 算法家族：

```text
树形:    KD-Tree, Ball-Tree, Annoy
哈希:    LSH (Locality-Sensitive Hashing)
图:      HNSW (Hierarchical Navigable Small World)
量化:    IVF (Inverted File), PQ (Product Quantization)
混合:    IVF-PQ, HNSW-PQ, ScaNN
```

## 三、HNSW：图索引的事实标准

### 3.1 核心思想

HNSW（Hierarchical Navigable Small World, Malkov & Yashunin 2016）把所有向量组织成一个**多层图**：

```text
Layer 3:    稀疏入口
Layer 2:    ↓                ↓
Layer 1:    ↓ ──── →  ──── ↓
Layer 0:    密集连接，每节点有几十条边
```

- **顶层**：少量"高速公路"节点，跨越距离大。
- **底层**：密集连接，邻居距离近。

### 3.2 检索过程

```text
1. 从顶层入口节点开始
2. 在当前层贪心搜索最近邻
3. 把当前层的最近邻当作下一层入口
4. 逐层下钻到 Layer 0
5. 在 Layer 0 做精细搜索，返回 top-K
```

类比：先坐飞机跨国，再开车进城，最后步行找具体地址。

### 3.3 性能参数

```python
import hnswlib

# 索引参数
M = 16                  # 每节点边数，越大越精确但越慢
ef_construction = 200   # 构建时的搜索宽度，越大索引质量越高
ef = 50                 # 查询时的搜索宽度，越大越精确

index = hnswlib.Index(space="cosine", dim=768)
index.init_index(
    max_elements=N,
    ef_construction=ef_construction,
    M=M,
)
index.add_items(embeddings, ids=list(range(N)))
index.set_ef(ef)        # 查询时的 ef

labels, distances = index.knn_query(query_vec, k=10)
```

### 3.4 优缺点

| 优点 | 缺点 |
|---|---|
| 查询延迟极低（O(log N)） | 内存占用大（每节点 ~M × 8 字节） |
| 召回率高（>95% 容易做到） | 写入较慢（要维护图结构） |
| 实现简单 | 不支持删除（一些变种支持） |
| 不需训练 | 1M 维 1024 向量约 ~2-3 GB |

## 四、IVF：倒排索引

### 4.1 核心思想

```text
1. 用 KMeans 把所有向量聚成 nlist 个簇（如 4096）
2. 每条向量归属一个簇
3. 查询时：先找最近 nprobe 个簇中心，再在这些簇内精确比对
```

```python
import faiss

quantizer = faiss.IndexFlatL2(D)
index = faiss.IndexIVFFlat(quantizer, D, nlist=4096, metric=faiss.METRIC_L2)
index.train(corpus)               # 必须先 train（KMeans）
index.add(corpus)
index.nprobe = 32                 # 查询时搜索的簇数，越大越精确

D, I = index.search(query, k=10)
```

### 4.2 优缺点

| 优点 | 缺点 |
|---|---|
| 内存占用小 | 召回率比 HNSW 低 |
| 训练后可静态发布 | 必须 train（要 sample） |
| 支持增量添加 | nprobe / nlist 需要调 |

## 五、PQ：乘积量化压缩

### 4.1 核心思想

把 D 维向量切成 M 个子段，每段单独做 KMeans 量化（如 256 个 centroid）。原始向量用一个 M 字节的码字表示：

```text
原始向量:  [v₁, v₂, ..., v₇₆₈]      3072 bytes (FP32)
PQ-8 编码:  [c₁, c₂, ..., c₈]        8 bytes
压缩比:    384×
```

```python
m = 8                              # 子段数
nbits = 8                          # 每段 256 个 centroid
index = faiss.IndexIVFPQ(quantizer, D, nlist=4096, m=m, nbits=nbits)
index.train(corpus)
index.add(corpus)
```

### 4.2 优缺点

| 优点 | 缺点 |
|---|---|
| 内存压缩 10-100× | 查询精度下降 |
| 适合超大规模（10 亿级） | 训练时间较长 |
| 可与 HNSW / IVF 叠加 | 距离计算是"非对称"，需小心 |

## 六、混合索引：HNSW + PQ / IVFPQ

生产环境通常**组合多种技术**：

```python
# Faiss: HNSW + PQ
index = faiss.IndexHNSWPQ(D, M=16, nbits=8, pq_m=32)

# Milvus: 多字段组合索引（向量 + 标量过滤）
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_PQ",
    "params": {"nlist": 4096, "m": 16, "nbits": 8},
}
```

```text
组合                │ 内存    │ 召回   │ 速度
────────────────────┼─────────┼────────┼─────
HNSW (FP32)        │ 大      │ 高     │ 快
HNSW + PQ          │ 小 4×   │ 中     │ 快
IVF (FP32)         │ 中      │ 中     │ 中
IVF + PQ           │ 小 10×  │ 中低   │ 中
ScaNN (ANNOY-like) │ 中      │ 高     │ 快
```

## 七、过滤搜索（Filtered Search）

真实场景常需要**带条件的向量检索**：

```sql
SELECT * FROM docs
WHERE category = '技术'        -- 元数据过滤
  AND created_at > '2024-01-01'
ORDER BY embedding <=> :query  -- 向量距离
LIMIT 10;
```

实现策略：

1. **Pre-filtering**：先按标量过滤，再向量检索。简单但 recall 受限。
2. **Post-filtering**：先向量检索，再过滤。可能返回 < K 结果。
3. **Hybrid**：索引同时支持向量 + 标量（Milvus、Qdrant、Weaviate 原生支持）。

```python
# Qdrant 示例
hits = client.search(
    collection_name="docs",
    query_vector=query_emb,
    query_filter=Filter(must=[
        FieldCondition(key="category", match=MatchValue(value="技术")),
        FieldCondition(key="created_at", range=Range(gte="2024-01-01")),
    ]),
    limit=10,
)
```

## 八、工程选型矩阵

| 规模 | 推荐 |
|---|---|
| < 100K | Faiss / hnswlib in-memory |
| 100K - 10M | Qdrant / Weaviate / Milvus（单节点） |
| 10M - 1B | Milvus / Pinecone / Weaviate（集群） |
| > 1B | 分片 + 量化（PQ）+ GPU 加速 |

## 小结

向量检索是 RAG、推荐、语义搜索的"心脏"。距离度量（cosine / L2 / IP）按 embedding 训练方式选；ANN 算法在召回率、内存、速度之间做权衡——HNSW 是事实标准，IVF-PQ 适合超大规模，元数据过滤是生产必需。下一篇我们将进入 **主流向量库对比**——Qdrant、Milvus、Pinecone、Weaviate、pgvector 各自的特点与适用场景。
