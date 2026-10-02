# 开放域 QA 与 RAG：从信息检索到检索增强生成的完整架构

开放域问答（Open-Domain QA）回答"任何问题"，不限定上下文。传统 QA 系统依赖 IR（信息检索）找到相关文档，再做 MRC；现代 RAG（Retrieval-Augmented Generation）更进一步，让 LLM 在生成答案时实时检索外部知识。本文从 BM25、DPR 出发，深入剖析 RAG 的完整架构、关键挑战与最新进展。

## 一、任务定义

### 1.1 Open-Domain QA

给定大规模文档集合（如 Wikipedia 全体 600 万篇文章）和任意用户问题，输出答案。与机器阅读理解（MRC）的关键区别：

| | MRC | Open-Domain QA |
| --- | --- | --- |
| 上下文 | 给出 | 需检索 |
| 输入长度 | 短（≤ 512 token） | 长（百万文档） |
| 系统 | QA 模型 | 检索 + QA |
| 难度 | 中 | 高 |

### 1.2 RAG 的核心动机

LLM 的三大局限：

1. **知识截止**：模型训练后无法获取新信息。
2. **幻觉**：模型可能编造事实。
3. **专业领域**：通用 LLM 在医学、法律等专业问题表现差。

RAG 通过**检索外部知识**缓解这些问题，让 LLM "带着资料开卷考"。

## 二、传统方法：DrQA 与基于 IR 的 QA

### 2.1 DrQA（Chen et al., 2017）

Facebook 2017 年的 DrQA 是开放域 QA 的开山之作：

```
[问题] → [Document Retriever (TF-IDF)] → [Top-K 段落] → [Document Reader (BiLSTM)] → [答案]
```

**Retriever**：TF-IDF + bigram features 在 Wikipedia 上检索 top-K 段落。

**Reader**：BiLSTM + attention 编码问题-段落对，预测答案起止。

DrQA 在 SQuAD 上 F1 达 28.4（远低于当时的 SOTA 70+），但证明了"检索 + 阅读"的可行性。

### 2.2 BERTserini（Yang et al., 2019）

用 BERT 替换 BiLSTM Reader：

```
[问题] → [Anserini BM25] → [Top-K 段落] → [BERT QA] → [答案]
```

Anserini 是基于 Lucene 的学术 IR 工具包。BERTserini 在 Natural Questions 上 F1 达 36，推动了"神经 Reader + 经典 Retriever"的混合架构。

### 2.3 TF-IDF / BM25 的局限

- **词袋假设**：忽略词序、语义。
- **词汇不匹配**：用户用口语查询，文档用书面语。
- **长尾词**：稀有词 TF-IDF 高但区分力低。

## 三、Dense Passage Retrieval（DPR）

### 3.1 核心思想

Karpukhin et al.（2020）的 DPR 用**双塔 BERT**编码查询与段落，用向量相似度检索：

$$
\text{score}(q, p) = \mathbf{q}^\top \mathbf{p}
$$

$\mathbf{q}, \mathbf{p}$ 是 BERT 的 `[CLS]` 向量（或 mean pooling）。

### 3.2 训练数据

用 QA 数据集作为训练信号：

- **正例**：包含答案的段落。
- **负例**：BM25 检索的 top 不含答案的段落（同 batch 内其他正例也作为 in-batch negatives）。

损失：

$$
\mathcal{L} = -\log \frac{\exp(\mathbf{q}^\top \mathbf{p}^+)}{\exp(\mathbf{q}^\top \mathbf{p}^+) + \sum_j \exp(\mathbf{q}^\top \mathbf{p}_j^-)}
$$

### 3.3 DPR 的效果

Natural Questions：

| 方法 | Top-20 段落召回率 |
| --- | --- |
| BM25 | 59% |
| DPR | **78%** |

DPR 把段落检索从 59% 提升到 78%，是 dense retrieval 的里程碑。

### 3.4 DPR 的局限

- **训练数据**：需要问答对监督信号。
- **独立编码**：query 与 passage 编码时无交互，可能漏掉复杂语义。
- **Index 大**：Wikipedia 2100 万段落需 ~50GB 索引。

## 四、ColBERT 与晚期交互

### 4.1 Late Interaction

Khattab & Zaharia（2020）的 ColBERT 用**晚期交互**替代 DPR 的早期池化：

- **每个 token 单独编码**为向量。
- **相似度** = token 级 max-sim 之和：

$$
S(q, p) = \sum_{i=1}^{n} \max_{j=1}^{m} \mathbf{q}_i^\top \mathbf{p}_j
$$

直觉：每个 query token 找到 passage 中最相似的 token，累加得分。比 DPR 的单向量更精细。

### 4.2 ColBERT 的工程创新

- **离线压缩**：passage 端向量量化、聚类。
- **PLAID 索引**：用近似最近邻（ANN）加速。
- **Multi-vector 检索**：保留细粒度语义。

### 4.3 ColBERTv2

Santhanam et al.（2022）的 ColBERTv2 进一步优化：

- 残差压缩（Residual Compression）。
- 跨编码蒸馏提升质量。

ColBERTv2 在 BEIR 基准上达到 BEIR SOTA。

## 五、RAG 的端到端架构

### 5.1 Lewis et al．（2020）的 RAG

经典 RAG 模型包括：

**Retriever**：DPR（冻结）
**Generator**：BART（可训练）

训练时**端到端**：

$$
\mathbf{y}^* = \arg\max_{\mathbf{y}} \sum_{z \in \text{Top-K}(p(\cdot | q))} P_{\text{BART}}(\mathbf{y} \mid q, z) \cdot P_{\text{DPR}}(z \mid q)
$$

前向时检索 top-K 段落，每个段落独立作为 BART 的输入，BART 生成答案；后向时把段落似然的梯度回传到 retriever。

### 5.2 推理模式

| 模式 | 描述 |
| --- | --- |
| RAG-Sequence | 同一文档生成完整答案，再加权 |
| RAG-Token | 每个 token 可以来自不同文档 |

RAG-Token 更灵活，RAG-Sequence 更稳定。

### 5.3 REALM

Guu et al.（2020）的 REALM 首次端到端训练 retriever + reader：

1. 用 BERT 做 retriever（不是双塔），可以端到端优化。
2. 用 Masked LM 预测特定 token 来挑选段落。
3. 训练中**周期性重建索引**（每 500 步），因为 retriever 在变。

REALM 在 Open-QA 上 F1 比 DPR + 冻结 retriever 高 3-5 个点。

### 5.4 Atlas（Izacard et al., 2022）

Meta 的 Atlas 把 retriever + LLM 联合训练：

- **Retriever**：Contriever（无监督）+ 监督微调。
- **LLM**：LLaMA、Atlas-11B 等。
- **联合训练**：retriever 损失 + LM 损失加权。

Atlas-11B 在 Natural Questions 上达到 60+ EM，**接近 GPT-3.5 + RAG**。

## 六、现代 RAG 的全栈架构

### 6.1 完整 Pipeline

```
[用户问题] 
   ↓
[Query 改写 / HyDE / Multi-Query]
   ↓
[Retriever]
   ├── Dense (DPR / ColBERT / BGE)
   ├── Sparse (BM25 / SPLADE)
   └── Hybrid (RRF / Reciprocal Rank Fusion)
   ↓
[Top-K 段落]
   ↓
[Reranker] (Cross-Encoder)
   ↓
[Top-N 段落 (N << K)]
   ↓
[Context Compression / Re-ranking]
   ↓
[LLM Generation]
   ↓
[Self-Check / Citation / Verification]
   ↓
[最终答案 + 引用]
```

### 6.2 关键组件详解

#### (a) Query 改写

原始查询往往不够清晰，需要改写：

- **HyDE**（Gao et al., 2022）：让 LLM 生成"假设性答案"，用其 embedding 检索。
- **Multi-Query**：生成多个查询变体，分别检索后合并。
- **Step-Back Prompting**：抽象查询（如 "1999 年马云创立的公司" → "阿里巴巴的历史"）。
- **Query 分解**：复杂查询拆解为子查询。

#### (b) 检索器

- **BM25**：经典稀疏检索。
- **DPR / Contriever / BGE**：双塔 dense retrieval。
- **ColBERT / ColBERTv2**：多向量 late interaction。
- **SPLADE**（Formal et al., 2021）：稀疏 + dense 混合，可解释。
- **E5 / BGE-M3**：多语言、多任务统一。
- **BGE-reranker**：专门训练的交叉编码 reranker。

#### (c) Reranker

把 retriever 召回的 top-K (e.g. 100) 段落 rerank 到 top-N (e.g. 5)：

- **Cross-Encoder**：query-passage 拼接过 BERT，输出相关性分数。
- **MonoBERT / MonoT5**：监督训练的 reranker。
- **LLM Reranker**：用 GPT-4 / Claude 作为 reranker。

代表模型：

- **Cohere Rerank**（商业 API）。
- **BGE-reranker-large**（开源）。
- **Jina Reranker**（多语言）。

#### (d) Context Compression

LLM 上下文有限，需要压缩段落：

- **LongLLMLingua**（Jiang et al., 2023）：训练一个小模型识别"问题相关" token，剪掉其他。
- **REPLUG**（Shi et al., 2023）：把检索段落当作"soft prompt"加权融合。
- **Selective Context**（Li et al., 2023）：用 perplexity 选择性保留 token。

#### (e) Generation

最终用 LLM 生成：

- **Prompt 模板**：包括问题、检索段落（带序号）、指令。
- **Streaming**：流式输出。
- **Citation**：让模型标注引用。

#### (f) Self-Verification

生成后验证：

- **Fact-checking**（Bohnet et al., 2022）：用 NLI 判断生成与检索段落是否一致。
- **Self-RAG**（Asai et al., 2023）：模型在生成时插入"反思 token"，决定是否需要再检索。
- **Verify-and-Edit**（Zhao et al., 2023）：生成后用外部知识验证，编辑错误。

### 6.3 Self-RAG（Asai et al., 2023）

Self-RAG 训练模型**在生成时反思**：

```
[Retrieve token] → [IsRel] → [IsSup] → [IsUse]
```

- **Retrieve**：决定是否检索（避免冗余）。
- **IsRel**：检索段落是否相关。
- **IsSup**：生成段落是否被段落支持。
- **IsUse**：整体答案是否有用。

训练数据用 GPT-4 自动生成，模型学会"批判性思考"。

### 6.4 CRAG（Corrective RAG）

Yan et al.（2024）的 CRAG 在检索后加入"校正"步骤：

1. 检索 top-K 段落。
2. 用轻量评估器判断每段质量（Correct / Incorrect / Ambiguous）。
3. **Correct**：直接用。
4. **Incorrect**：用 Web 搜索替代。
5. **Ambiguous**：内部 + 外部结合。

CRAG 在 PopQA 等基准上比朴素 RAG 提升 10+ 个点。

## 七、RAG 评测

### 7.1 评测维度

| 维度 | 评估 |
| --- | --- |
| 检索质量 | Recall@K、nDCG、MRR |
| 答案正确性 | EM、F1、Accuracy |
| 答案忠实度 | 是否引用了检索段落 |
| 答案相关性 | 是否回答了问题 |
| 流畅度 | 人类评估 |

### 7.2 评测基准

- **Natural Questions**：Google 真实查询。
- **TriviaQA**：通用 trivia。
- **PopQA**：长尾知识（人名、地名、日期）。
- **HotpotQA**：多跳推理。
- **BEIR**：18 个检索任务。
- **RAGAS**：专门评估 RAG 系统的框架。

### 7.3 RAGAS 框架

RAGAS（Es et al., 2023）从四个维度评估 RAG：

- **Context Relevance**：检索段落与问题相关。
- **Faithfulness**：答案是否被段落支持。
- **Answer Relevance**：答案是否回答了问题。
- **Context Recall**：是否检索到包含答案的段落。

每个维度用 LLM-as-Judge 自动评估。

## 八、Advanced RAG 模式

### 8.1 GraphRAG（Edge et al., 2024）

Microsoft 的 GraphRAG 把文档构建成**知识图谱**：

1. **实体抽取**：从段落中识别人名、地名、概念。
2. **关系抽取**：识别实体之间的关系。
3. **图谱构建**：实体作为节点，关系作为边。
4. **社区检测**：用 Leiden 算法发现密集子图。
5. **摘要生成**：每个社区生成摘要。
6. **检索 + 生成**：基于图谱的语义检索。

优势：

- 解决"全局问题"（"数据集中的主要主题是什么"）。
- 处理跨段落、跨文档的复杂关系。
- 提供结构化、可解释的检索。

### 8.2 Agentic RAG

Agentic RAG 让 LLM Agent 自主决定检索策略：

```python
class RAGAgent:
    def answer(self, question):
        plan = self.llm(f"规划回答此问题的步骤：{question}")
        for step in plan.steps:
            if step.need_search:
                docs = self.search(step.query)
                context = self.rerank(docs, step.query)
            answer = self.llm(f"基于上下文回答：{step.query}\n上下文：{context}")
        return self.synthesize(all_answers)
```

代表：

- **ReAct**：推理 + 行动。
- **AutoGen / LangGraph**：多 Agent 协作。
- **Open Deep Search**：开源 Agentic RAG。

### 8.3 Modular RAG

把 RAG 拆成可组合的模块，每个模块可独立优化：

```
[Query Understanding] → [Retrieval] → [Reranking] → [Compression] → [Generation] → [Verification]
```

每个模块可以用不同模型，可独立 AB 测试。

### 8.4 Long-Context RAG

超长上下文 LLM（GPT-4 Turbo、Claude 3 Opus）可以"塞下"整个文档：

```python
prompt = f"""文档：{entire_document}

问题：{question}

答案："""
```

优势：

- 不用检索，省去复杂管道。
- 简单。

劣势：

- 成本高（百万 token 输入 ~$10）。
- 中间信息丢失（"lost in the middle" 现象）。

实际工程：**超长上下文 + 检索结合**，用检索快速定位，再用上下文深入理解。

## 九、RAG vs Fine-tuning

### 9.1 何时用 RAG

- **知识频繁更新**：新闻、股价、政策。
- **需要可引用**：法律、医疗建议。
- **低成本**：避免训练大模型。
- **可解释**：能展示答案来源。

### 9.2 何时用 Fine-tuning

- **风格 / 格式定制**：希望模型按特定风格回答。
- **任务特定**：分类、提取等结构化任务。
- **高质量训练数据**：有大量标注对。
- **降低延迟**：避免每次检索。

### 9.3 RAG + Fine-tuning

最佳实践是**两者结合**：

1. **Fine-tune** LLM 让其理解 RAG 格式、生成风格。
2. **RAG** 提供最新、准确的知识。

代表：Self-RAG、RA-DIT（Anil et al., 2023）把两者联合训练。

## 十、PyTorch 工具与框架

### 10.1 主流 RAG 框架

- **LangChain**：最流行的 LLM 应用框架，支持 RAG、Agent、工具调用。
- **LlamaIndex**：专注 RAG / 文档问答。
- **Haystack**（deepset）：开源 RAG 框架，工业级。
- **txtai**：轻量级向量数据库 + LLM。
- **DSPy**：自动优化 RAG prompt。

### 10.2 简化 RAG 实现

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 1. Embedding & Index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 2. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model="gpt-4", temperature=0),
    chain_type="stuff",  # 把所有段落塞入 prompt
    retriever=retriever,
    return_source_documents=True,
)

result = qa_chain({"query": "量子力学的主要奠基人是谁？"})
print(result["result"])
print("Sources:", [doc.metadata for doc in result["source_documents"]])
```

### 10.3 完整 RAG 评估

```python
from ragas import evaluate
from ragas.metrics import (
    context_relevancy,
    faithfulness,
    answer_relevancy,
    context_recall,
)

result = evaluate(
    dataset,           # 包含 question, answer, contexts, ground_truth
    metrics=[
        context_relevancy,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)
print(result)
```

## 十一、工程经验

### 11.1 文档切分

- **切分粒度**：512-1024 token，每段 50-100 token overlap。
- **语义感知切分**：用段落标题、Markdown header 切分。
- **多粒度索引**：同时索引段落、句子、文档级摘要。

### 11.2 Embedding 模型选择

| 模型 | 维度 | 速度 | 质量 |
| --- | --- | --- | --- |
| text-embedding-3-small (OpenAI) | 1536 | 中 | 高 |
| BGE-large-en-v1.5 | 1024 | 快 | 高 |
| BGE-M3 | 1024 | 快 | 高（多语言） |
| M3E | 768 | 快 | 中 |
| E5-large-v2 | 1024 | 快 | 高 |

### 11.3 Reranker 必要性

朴素 retrieval 的 top-5 未必最优。Reranker 能把"看起来相关"提升到"真正相关"：

```
Retriever top-100 → Reranker top-5
```

工业实践中 Reranker 是必须的，能提升 5-15 个 nDCG。

### 11.4 延迟与成本

典型 RAG 延迟：

```
Query 改写: 0.5s
Embedding: 0.1s
检索 (FAISS): 0.05s
Rerank: 1-2s
LLM 生成: 2-10s
总计: 4-13s
```

优化：

- **缓存**：相同 / 相似 query 复用答案。
- **预生成 Embedding**：离线把文档全 embedding。
- **并行**：检索 + rerank 并行。
- **异步**：用户等待时先返回部分答案。

## 十二、未来方向

1. **Multi-Modal RAG**：图像 + 视频 + 表格联合检索。
2. **Knowledge Graph RAG**：结构化知识 + 向量检索。
3. **Self-Improving RAG**：从用户反馈中自动优化。
4. **Federated RAG**：跨组织、跨数据源的检索。
5. **Real-Time RAG**：实时索引新数据（如新闻、社交媒体）。

## 小结

| 时代 | 方法 | 关键 |
| --- | --- | --- |
| 2017 | DrQA | TF-IDF + BiLSTM |
| 2019 | BERTserini | BM25 + BERT QA |
| 2020 | DPR | 双塔 dense retrieval |
| 2020 | RAG | 端到端训练 retriever + generator |
| 2022 | Atlas | 大模型 + dense retrieval |
| 2023 | Self-RAG | 反思式 RAG |
| 2024 | GraphRAG | 知识图谱增强 |

RAG 是 LLM 应用的"操作系统"——它把语言模型的"知识截止"与"幻觉"两大痛点大幅缓解，是当前（2026）的事实标准。掌握 RAG 的全栈架构，你就能构建可靠、灵活、可扩展的智能问答系统。
