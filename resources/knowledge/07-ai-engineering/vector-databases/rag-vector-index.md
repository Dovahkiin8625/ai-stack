# RAG 中的向量索引：从文档切块到检索评估

RAG 系统把"私有知识 + LLM 推理"结合起来，而向量索引是连接两者的"桥梁"。索引质量决定了 RAG 的上限——再强大的 LLM，检索到错的文档也只能生成错的答案。本文深入 RAG 向量索引的全链路：文档切块、embedding 选择、索引构建、混合检索、查询改写、评估指标。

## 一、RAG 索引的整体流程

```text
文档集合                        用户查询
   │                               │
   ↓                               ↓
┌──────────┐                  ┌──────────┐
│ 解析 / OCR│                  │ 改写     │
└─────┬────┘                  └─────┬────┘
      ↓                             ↓
┌──────────┐                  ┌──────────┐
│ 切块     │                  │ Embedding│
└─────┬────┘                  └─────┬────┘
      ↓                             ↓
┌──────────┐                  ┌──────────┐
│ Embedding│                  │ 向量检索 │
└─────┬────┘                  └─────┬────┘
      ↓                             ↓
┌──────────┐                  ┌──────────┐
│ 向量索引 │  ─── 检索 ──→   │ 重排(可选)│
└──────────┘                  └─────┬────┘
                                    ↓
                              ┌──────────┐
                              │ Prompt   │
                              │ 组装     │
                              └─────┬────┘
                                    ↓
                              ┌──────────┐
                              │ LLM 生成 │
                              └──────────┘
```

下面按链路逐段展开。

## 二、文档解析与切块

### 2.1 文档类型

| 类型 | 工具 |
|---|---|
| **PDF** | PyMuPDF, pdfplumber, Unstructured |
| **Word** | python-docx, Unstructured |
| **HTML** | BeautifulSoup, Trafilatura |
| **Markdown** | mistune, markdown-it-py |
| **代码** | tree-sitter, LangChain 的 CodeTextSplitter |
| **表格** | Camelot, Tabula |

### 2.2 切块策略

固定长度切块最简单，但常切坏语义。更智能的方法：

#### 递归字符切块（推荐默认）

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # 字符数（≈ 200 token）
    chunk_overlap=64,      # 重叠，避免切断
    separators=["\n\n", "\n", "。", "，", " ", ""],   # 优先按段落、句号切
)
chunks = splitter.split_documents(docs)
```

#### 语义切块（高级）

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
)
chunks = splitter.split_documents(docs)
```

按 embedding 相似度的拐点切——语义完整性更好，但慢且贵。

#### 结构化切块（针对 Markdown / HTML）

按 H1/H2/H3 标题 + 段落切：

```python
from langchain.text_splitter import MarkdownTextSplitter

splitter = MarkdownTextSplitter(chunk_size=512, chunk_overlap=64)
```

每个 chunk 保留 `headers` 元数据，方便后续引用。

### 2.3 表格与代码的特殊处理

```python
# 表格保留完整（不切）
# 用 markdown 表达或单独的 chunk_type="table"

# 代码按函数/类切
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language

python_splitter = RecursiveCharacterTextSplitter.from_language(
    Language.PYTHON,
    chunk_size=512,
    chunk_overlap=64,
)
```

### 2.4 元数据

每个 chunk 保留的元数据：

```python
chunk.metadata = {
    "source": "s3://docs/llm.pdf",
    "page": 12,
    "section": "3.2 Transformer 架构",
    "chunk_type": "text",         # text / table / code
    "created_at": "2024-01-15",
    "version": "v3",
}
```

**元数据决定引用质量**——下游 prompt 里能告诉 LLM"答案来自 P12 第 3 节"。

## 三、Embedding 模型选型

### 3.1 关键指标

```text
MTEB (Massive Text Embedding Benchmark) 平均分数
  衡量 56 个数据集的综合检索 / 分类 / 聚类能力

embedding 维度
  768 / 1024 / 1536，越大表达力强但成本高

最大输入长度
  BGE-m3 支持 8192 token，OpenAI 支持 8191
```

### 3.2 主流模型对比

| 模型 | 维度 | MTEB | 多语 | 长文本 | 备注 |
|---|---|---|---|---|---|
| **OpenAI text-embedding-3-large** | 3072 | 64.6 | 中 | 8K | 闭源、贵、稳定 |
| **BGE-large-zh-v1.5** | 1024 | 高 | 中 | 512 | 中文首选 |
| **BGE-m3** | 1024 | 高 | 强 | 8K | 多语 + 长文本 |
| **Cohere embed-v3** | 1024 | 65 | 强 | 8K | 多语 + 检索强 |
| **GTE-Qwen2** | 1536 | 65 | 中 | 32K | 阿里开源 |
| **mxbai-embed-large** | 1024 | 高 | 弱 | 512 | 英文强 |
| **E5-mistral-7b** | 4096 | 高 | 中 | 32K | 大模型、SOTA |

### 3.3 选型建议

- **英文为主 + 预算充足** → OpenAI text-embedding-3-large
- **中文为主** → BGE-large-zh-v1.5 或 BGE-m3
- **多语 + 长文档** → BGE-m3 或 Cohere embed-v3
- **自托管 / 数据敏感** → BGE / GTE / mxbai
- **极致效果** → E5-mistral-7b（需要 GPU）

## 四、向量索引构建

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    optimizers_config={"indexing_threshold": 10000},  # 1w 条后建索引
)

# 批量写入
points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={
            "text": chunks[i].text,
            "source": chunks[i].metadata["source"],
            "section": chunks[i].metadata["section"],
        },
    )
    for i in range(len(chunks))
]
client.upsert(collection_name="docs", points=points, batch_size=256)
```

**生产技巧**：

- 分批写入（每批 256-1024 条），避免内存爆。
- 用 `payload` 存 chunk 原文 + 元数据。
- 用 `payload_index` 给过滤字段建索引。

## 五、混合检索：向量 + BM25

向量检索擅长**语义匹配**，但对**关键词**（人名、产品代号、专业术语）效果差。BM25 反之。

**Hybrid Search** 把两者结合：

```python
# Qdrant 原生支持
from qdrant_client.models import FusionQuery, Prefetch

results = client.query_points(
    collection_name="docs",
    prefetch=[
        Prefetch(query=query_dense, using="dense", limit=20),
        Prefetch(query=query_sparse, using="sparse", limit=20),
    ],
    query=FusionQuery(fusion="rrf"),       # Reciprocal Rank Fusion
    limit=10,
)
```

**RRF** 把两路排名按公式合并：

$$
\text{RRF}(d) = \sum_{r \in \{dense, sparse\}} \frac{1}{k + \text{rank}_r(d)}
$$

$k=60$ 是常用默认值。生产上 hybrid 通常比单路召回率高 5-15%。

## 六、查询改写与扩展

原始用户问题往往不是最佳检索形式。常见改写：

### 6.1 HyDE（Hypothetical Document Embeddings）

让 LLM 先"想象"一段答案，用这段答案的 embedding 去检索：

```python
hyde_prompt = "请用一段话回答：{question}"
hypothetical = llm.invoke(hyde_prompt.format(question=user_query))
results = vector_store.search(embedding(hypothetical))
```

对短查询特别有效。

### 6.2 Multi-Query

让 LLM 生成 3-5 个不同措辞的查询，分别检索后合并：

```python
multi_query_prompt = "请把下面问题改写成 5 个不同表述：\n{question}"
queries = llm.invoke(multi_query_prompt).split("\n")
all_results = [vector_store.search(embedding(q)) for q in queries]
merged = rrf_merge(all_results)
```

### 6.3 Step-back Prompting

让 LLM 先抽象一步（"这个问题属于哪一类？"），用抽象 query 检索背景知识，再结合原始问题检索细节。

### 6.4 Query Rewriting with History

多轮对话场景：把"它解决了什么？"改写成"Transformer 解决了什么问题？"。

```python
rewrite_prompt = """基于对话历史，把用户最后的问题改写成独立、完整的检索 query。
对话：{history}
问题：{question}
改写后的 query："""
rewritten = llm.invoke(rewrite_prompt)
```

## 七、重排序（Re-ranking）

先用向量粗召 top-K=50，再用 Cross-Encoder 模型精排：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large")

pairs = [[query, doc.text] for doc in candidates]
scores = reranker.predict(pairs)
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:5]
```

**为什么需要 Reranker**：

- Bi-Encoder（embedding）独立编码 query / doc，向量空间有损。
- Cross-Encoder 把 query + doc 联合编码，更准但慢。

典型 pipeline：

```text
向量粗召 top-50  →  Cross-Encoder 精排  →  top-5 给 LLM
recall ≈ 0.9        precision ≈ 0.95       context 质量高
```

BGE-reranker-large、cohere-rerank-3、jina-rerank 都是常用选项。

## 八、评估：检索质量

### 8.1 评估指标

| 指标 | 含义 | 公式 |
|---|---|---|
| **Recall@K** | top-K 中相关文档占比 | relevant ∩ retrieved / relevant |
| **MRR** | 第一个相关文档的倒数排名均值 | mean(1/rank_first_relevant) |
| **NDCG@K** | 考虑相关性的归一化折损累积增益 | 复杂，见 sklearn |
| **Hit Rate** | top-K 是否包含任意相关文档 | 1 if hit else 0 |

### 8.2 评估数据集

生产中通常要手工标注：

```json
{
  "query": "如何重训 embedding 模型？",
  "relevant_docs": ["doc_123", "doc_456"],
  "irrelevant_docs": ["doc_789"]
}
```

抽样 200-500 条 query + 人工标注 ground truth，是评估召回率的最小可用集。

### 8.3 端到端评估

检索只是中间环节。最终看 **Answer Quality**：

```python
# LLM-as-Judge
judge_prompt = """请评估下面回答对用户问题的质量：
- 是否事实正确
- 是否完整
- 是否简洁

问题：{question}
参考：{reference}
回答：{answer}
评分（1-5）："""

scores = []
for qa in eval_set:
    score = llm.invoke(judge_prompt.format(...))
    scores.append(score)
print("avg:", sum(scores) / len(scores))
```

工具：**RAGAS**、**ARES**、**TruLens** 提供半自动化的 RAG 评估。

## 九、运维注意事项

1. **embedding 模型版本锁定**：换 embedding 必须重建索引。
2. **chunk 大小与 query 长度匹配**：query 短（如人名）配小块；query 长（段落理解）配大块。
3. **索引重建**：定期增量更新（按 source / 时间），旧数据不删直接版本化。
4. **监控**：跟踪 recall 趋势、chunk 被检索频次、LLM 反馈率。
5. **冷文档**：长期不被检索的 chunk 可能是 dead weight，定期归档。

## 小结

RAG 向量索引是"文档"到"可检索知识"的转化管道。切块策略决定语义完整性，embedding 模型决定语义表达力，混合检索和重排提升召回率，查询改写提升 query 质量，端到端评估是质量保证。所有这些环节都要可监控、可评估、可迭代——RAG 系统是数据 + 算法 + 工程三者结合的产物，单点优化都可能被另一点的短板抵消。
