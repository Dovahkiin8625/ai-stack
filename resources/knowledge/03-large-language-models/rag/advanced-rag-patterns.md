# 进阶 RAG 模式

朴素 RAG 在很多业务上"够用"，但要追求更高的召回率、答案准确率或复杂推理能力，就需要更聪明的检索与编排策略。本文介绍五种被广泛验证有效的进阶模式，并简述它们何时值得使用。

## 一、Re-ranking（重排序）

朴素检索只用单一相似度（cosine 或点积）排序，对短查询与长文档之间的语义鸿沟常常束手无策。Re-ranking 在初步召回的 top-k 候选上，再用一个**更强的 cross-encoder** 给每对 (query, chunk) 打一个更精细的相关性分。

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
hits = retriever.search(query, top_k=20)
scores = reranker.predict([(query, h.text) for h in hits])
hits = sorted(zip(hits, scores), key=lambda x: -x[1])[:5]
```

**何时用**：检索质量是瓶颈、延迟预算允许（cross-encoder 比 bi-encoder 慢 10-100 倍）。

## 二、混合检索（BM25 + Dense）

BM25 是经典的稀疏检索算法，基于词频与文档长度，对精确术语、人名、产品型号特别敏感；dense retrieval 擅长语义匹配但对稀有词不友好。混合检索对两者结果做加权融合：

```math
\text{score}(q, d) = \alpha \cdot \text{BM25}(q, d) + (1 - \alpha) \cdot \text{cosine}(q, d)
```

**何时用**：领域含有大量专业术语或代码标识符。

## 三、查询改写 / HyDE

用户的查询常常很口语化甚至有错别字，导致检索不到。常见补救：

- **Query rewriting**：用 LLM 把用户原始问题改写成更适合检索的形式，例如补充同义词、拆分复合问题。
- **HyDE（Hypothetical Document Embeddings）**：让 LLM 先凭空生成一篇"假想的"答案文档，再把这段文本送进 embedding 检索。它的直觉是：答案空间里的文档比问题空间里的查询更接近目标文档。

```python
hyde_doc = llm(f"请根据问题写一段可能包含答案的短文：{query}")
hits = retriever.search(hyde_doc, top_k=5)
```

**何时用**：用户查询很短或噪声大。

## 四、多跳检索（Multi-hop Retrieval）

有些问题需要串联多个文档才能回答。例如"2018 年那次收购后，被收购公司的 CEO 后来加入了哪家初创公司？"——必须先找到收购记录，再找到该 CEO 的去向。

实现方式：

1. 让 LLM 在每一步判断"还需要什么信息"，再针对性检索。
2. 把前几步检索到的中间结论拼进下一轮查询。

**何时用**：复杂分析任务、研究类问答。

## 五、Agentic RAG（ReWOO / ReAct）

当问题需要组合多种工具（检索 + 计算 + API 调用）时，可以让 LLM 担任"调度者"，按 ReAct 或 ReWOO 范式决定下一步动作。

- **ReAct**：让模型在"思考 → 行动 → 观察"循环中推进。
- **ReWOO**：把所有动作先"规划"出来，再批量执行，最后由 LLM 综合。ReWOO 节省中间轮推理成本。

**何时用**：开放域问答、研究助手、企业知识工作流。

## 六、模式对比

| 模式 | 主要收益 | 主要代价 | 典型场景 |
| --- | --- | --- | --- |
| Re-ranking | 提升 top-k 精度 | 增加延迟 | 高质量问答 |
| 混合检索 | 兼顾术语与语义 | 需维护两套索引 | 法律 / 代码 / 医疗 |
| Query rewriting | 提升召回 | 一次额外 LLM 调用 | 短查询 / 口语化 |
| HyDE | 解决查询-文档 gap | LLM 调用 + 噪声 | 文档库小且杂 |
| Multi-hop | 支持复杂推理 | 多轮检索成本 | 研究类问题 |
| Agentic RAG | 多工具协同 | 延迟与可控性 | 复杂工作流 |

## 小结

从朴素 RAG 到 agentic RAG，本质是在"让模型多花一点算力，换更高的答案质量"和"延迟预算"之间权衡。下一篇我们会讨论如何**评估**这些系统的输出，避免"看起来好、实际上没改进"的陷阱。
