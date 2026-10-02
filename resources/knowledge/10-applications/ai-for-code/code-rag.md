# 代码 RAG：用检索增强代码生成

LLM 的训练数据有截止时间、单个 repo 不在训练集、长上下文会"中间遗忘"——这些问题让单纯的 prompt-based 代码生成**不够准确**。**代码 RAG（Retrieval-Augmented Code Generation）** 把"检索"与"生成"结合：先用 embedding 模型从代码库中**检索**最相关的片段，再让 LLM 在这些片段的上下文中生成代码。本文系统介绍代码 RAG 的核心组件、embedding 模型、检索策略，以及与 Agent 模式的组合。

## 一、为什么代码生成需要 RAG

LLM 直接生成的局限：

1. **训练数据陈旧**：模型不知道 2024 年发布的 API。
2. **私有代码库**：公司内部代码不会出现在预训练数据中。
3. **跨文件依赖**：长上下文下模型容易"丢失"早期信息。
4. **API 误用**：模型可能编造**不存在的 API**（hallucination）。
5. **风格不一致**：不同团队有不同的命名、架构风格。

RAG 的解决方案：**让模型"先查资料再回答"**——把相关代码片段当作上下文。

## 二、代码 RAG 的整体流程

```text
┌────────────────────────────────────────────────────────────┐
│  1. Indexing（索引阶段，离线）                                │
│     - 切分代码（按文件、函数、类、chunk）                       │
│     - 用 embedding 模型编码                                   │
│     - 存入向量数据库                                          │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  2. Query（查询阶段，在线）                                   │
│     - 用户 prompt（如 "如何调用 payment API？"）              │
│     - 编码为向量                                              │
│     - 从向量库 top-K 检索                                    │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  3. Augmented Generation（增强生成）                          │
│     - 把检索到的 chunks 与 prompt 拼接                        │
│     - LLM 生成代码                                           │
│     - 可选：去重 / 排序 / 重排                                │
└────────────────────────────────────────────────────────────┘
```

## 三、代码切分：保留语义边界

代码切分不能简单按字符数切——必须保留**语法/语义单元**：

### 1. 按函数 / 类切（最常用）

```python
import ast

def split_python_by_functions(code: str) -> list[dict]:
    """把 Python 代码按函数/类切分。"""
    tree = ast.parse(code)
    chunks = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            chunks.append({
                "type": type(node).__name__,  # "FunctionDef" / "ClassDef"
                "name": node.name,
                "code": ast.unparse(node),
                "lineno": node.lineno,
                "docstring": ast.get_docstring(node),
                "dependencies": [n.id for n in ast.walk(node) 
                                 if isinstance(n, ast.Name)],
            })
    
    return chunks
```

**优点**：每个 chunk 语义完整，可独立检索。
**缺点**：大文件切得太多，类间依赖丢失。

### 2. Tree-sitter 通用切分

Tree-sitter 支持 **40+ 语言**的统一语法树：

```python
from tree_sitter_languages import get_language, get_parser

def split_with_tree_sitter(code: str, language: str = "python") -> list[dict]:
    """用 tree-sitter 切分任意语言。"""
    parser = get_parser(language)
    tree = parser.parse(bytes(code, "utf8"))
    
    chunks = []
    def visit(node, parent=None):
        # 函数 / 方法 / 类 → 切为独立 chunk
        if node.type in ("function_definition", "method_definition", "class_definition"):
            chunks.append({
                "type": node.type,
                "name": extract_name(node),
                "code": code[node.start_byte:node.end_byte],
                "parent": parent,
            })
        else:
            for child in node.children:
                visit(child, parent=node)
    
    visit(tree.root_node)
    return chunks
```

**优势**：跨语言一致、保留语法结构。

### 3. 滑动窗口（次选）

```python
def sliding_window(code: str, chunk_size=512, overlap=64) -> list[str]:
    """简单的滑动窗口切分。"""
    tokens = code.split()  # 简化：按空白
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunks.append(" ".join(tokens[i:i + chunk_size]))
    return chunks
```

**缺点**：可能切在函数中间，语义不完整。

## 四、代码 Embedding 模型

代码 embedding 是 RAG 的核心——决定检索质量。

### 1. OpenAI text-embedding-3

通用 embedding，对代码**一般**。

### 2. CodeBERT / GraphCodeBERT

Microsoft 的代码专用 embedding：

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("microsoft/codebert-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def embed_code(text: str) -> torch.Tensor:
    """用 CodeBERT 编码代码。"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # [CLS]
```

### 3. StarCoder Embedding（BigCode, 2023）

基于 StarCoderBase 训练的 embedding，在 CodeSearchNet 上达到 SOTA。

### 4. Voyage Code（商业）

Voyage AI 的 code-2 / code-3，闭源但效果领先。

### 5. BGE-M3

BAAI 的多语言 embedding，也支持代码。

### 评测：CodeSearchNet

| 模型 | Ruby | JavaScript | Go | Python | Java | PHP |
|---|---|---|---|---|---|---|
| CodeBERT | 67.9 | 62.0 | 88.2 | 67.2 | 67.6 | 62.8 |
| GraphCodeBERT | 70.3 | 64.4 | 89.7 | 69.2 | 69.4 | 64.9 |
| StarCoderEmb | 73.0 | 68.5 | 92.0 | 73.0 | 71.5 | 67.0 |

数字代表"代码片段与对应 docstring 的检索 MRR@10"。

## 五、检索策略

### 1. 语义检索（Dense Retrieval）

```python
import faiss

# 1) 把代码 chunks 编码并建立索引
chunk_embeddings = np.array([embed(c["code"]) for c in chunks])
index = faiss.IndexFlatIP(chunk_embeddings.shape[1])  # 内积
index.add(chunk_embeddings)

# 2) 查询
def retrieve(query: str, k=5) -> list[dict]:
    q_emb = embed(query).numpy()
    distances, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]
```

### 2. 混合检索（Hybrid Search）

结合**稀疏（BM25）+ 密集（向量）**：

```python
from rank_bm25 import BM25Okapi

def hybrid_retrieve(query: str, chunks: list[dict], k=5, alpha=0.5):
    # 1) BM25 分数
    tokenized = [c["code"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())
    
    # 2) Dense 分数
    dense_scores = cosine_sim(query_emb, [embed(c["code"]) for c in chunks])
    
    # 3) 加权融合
    combined = alpha * dense_scores + (1 - alpha) * bm25_scores
    top_k = np.argsort(combined)[-k:][::-1]
    return [chunks[i] for i in top_k]
```

**优势**：BM25 抓精确匹配，dense 抓语义——互补。

### 3. 重排序（Reranking）

初检返回 50 个候选，再用 cross-encoder 重排：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[dict], top_k=5):
    pairs = [[query, c["code"]] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
```

**效果**：通常让最终 top-5 精度提升 **20~30%**。

### 4. 多跳检索（Multi-hop）

复杂任务需要**多步检索**：

```python
def multi_hop_retrieve(query: str, max_hops=3):
    """迭代检索，每次基于上轮结果扩展。"""
    context = []
    for hop in range(max_hops):
        results = retrieve(query + " ".join(context), k=5)
        context.extend([r["code"] for r in results])
        
        # 检查是否足够
        if is_query_satisfied(query, context):
            break
    
    return context
```

## 六、Prompt 构造

把检索到的 chunks 与 query 拼接：

```python
def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """构造带 RAG 上下文的 prompt。"""
    context = "\n\n---\n\n".join([
        f"# File: {c['path']}\n{c['code']}" 
        for c in retrieved_chunks
    ])
    
    return f"""参考以下代码片段回答用户问题：

{context}

---

用户问题: {query}

请基于参考代码给出回答。"""
```

**关键点**：
- 标注每个 chunk 的来源（文件路径）。
- 控制总长度（避免超过模型上下文窗口）。
- 让模型**显式说明**使用了哪些引用。

## 七、代码 RAG 的进阶模式

### 1. 代码-自然语言混合检索

```python
# 文档检索
doc_results = retrieve(query, k=3, type="doc")

# 代码检索
code_results = retrieve(query, k=5, type="code")

# 合并
all_results = doc_results + code_results
```

### 2. 上下文压缩

检索到的 chunks 可能有大量无关内容——压缩：

```python
def compress_context(chunks: list[dict], max_tokens=2000) -> str:
    """用 LLM 压缩 chunks 到关键信息。"""
    combined = "\n\n".join([c["code"] for c in chunks])
    prompt = f"""以下是从代码库检索到的片段。请压缩到 {max_tokens} token，
仅保留与用户查询相关的内容。

代码片段：
{combined}

压缩版本："""
    return llm.generate(prompt)
```

### 3. 检索增强 + Agent

RAG 与 Agent 组合——模型自主决定何时检索：

```python
def code_agent(user_query: str):
    """代码 agent：自主检索 + 生成。"""
    context = []
    plan = llm.generate(f"为以下任务设计步骤：{user_query}")
    
    for step in parse_plan(plan):
        if step.action == "search":
            results = retrieve(step.query)
            context.extend(results)
        elif step.action == "generate":
            code = llm.generate(
                build_prompt(step.query, context)
            )
            return code
        elif step.action == "test":
            run_tests()
```

## 八、评估代码 RAG

### 1. 检索质量

| 指标 | 含义 |
|---|---|
| **Recall@K** | 前 K 个中包含正确答案的比例 |
| **MRR** | 第一个正确答案的倒数排名 |
| **NDCG** | 考虑相关性的排序质量 |

### 2. 端到端生成质量

```python
# 在 RepoBench / SWE-bench 检索增强版上评测
# 或自建基准：内部代码库的 Q&A 对
```

### 3. 幻觉率

```python
def hallucination_rate(predictions, ground_truth):
    """统计"幻觉 API / 函数"的比例。"""
    hallucinations = 0
    for pred, truth in zip(predictions, ground_truth):
        # 检测 pred 中调用了哪些 API
        pred_apis = extract_api_calls(pred)
        truth_apis = extract_api_calls(truth)
        # 多余的 / 不存在的 = 幻觉
        if pred_apis - truth_apis:
            hallucinations += 1
    return hallucinations / len(predictions)
```

## 九、工业实践

### GitHub Copilot Enterprise

- 索引整个组织的代码库（含私有仓库）。
- 多语言、多仓库检索。
- 与 IDE 深度集成。

### Cursor

- "Codebase Indexing"：自动索引工作区。
- "@codebase" 命令：在 prompt 中引用代码库。
- 多文件编辑与 diff 预览。

### Sourcegraph Cody

- 强大的代码搜索 + AI。
- 支持大型 monorepo。
- 多语言支持。

### Continue.dev

- 开源的 VS Code / JetBrains AI 助手。
- 可配置 RAG pipeline。

## 十、工程挑战

### 1. 索引更新

代码不断变化——索引必须**增量更新**：

```python
class CodeIndex:
    def update(self, file_path: str):
        """单个文件更新。"""
        new_chunks = split(file_path)
        # 删除旧的，添加新的
        self.index.remove_by_file(file_path)
        self.index.add(new_chunks)
```

### 2. 大仓库性能

大型 monorepo（如 Google 的 20 亿行代码）——全量索引耗时。

**方案**：
- 分布式 embedding 服务。
- 仅索引**频繁修改**的文件（活跃代码占比通常 < 5%）。
- 多级索引（hot/warm/cold）。

### 3. 上下文窗口限制

128K context 能装下大多数代码，但**长上下文会稀释**——模型更关注中间内容。

**方案**：
- Rerank + Top-K，控制质量。
- 让模型先**总结**检索结果再回答。

### 4. 安全与许可

RAG 引入了**额外风险**——返回的私有代码可能包含敏感信息。

**方案**：
- 检索时过滤敏感文件（`.env`、`credentials`）。
- 输出前再过滤。
- 审计日志。

## 十一、未来方向

1. **结构化检索**：基于 AST 相似度而非文本相似度。
2. **跨语言检索**：Rust 代码搜"类似 Go 的实现"。
3. **演化检索**：跟踪代码的历史版本与提交信息。
4. **交互式检索**：模型边生成边决定"还需要看什么"。
5. **端到端训练**：把 embedding 与生成联合训练。

## 小结

代码 RAG 是解决"模型不知道私有代码 / 最新 API"的关键技术——**embedding + 切分 + 检索 + rerank + prompt 构造**构成完整 pipeline。代码 embedding 模型（CodeBERT、StarCoderEmb、Voyage Code）针对代码语义优化；混合检索与重排序显著提升精度；多跳与 Agent 模式让模型自主决定检索策略。**真正的工业系统**（Copilot Enterprise、Cursor、Sourcegraph）把 RAG、Agent、IDE 集成在一起。下一篇我们将看到代码 Agent——**让 LLM 自主完成多步编程任务**，从补全到端到端工程实现。
