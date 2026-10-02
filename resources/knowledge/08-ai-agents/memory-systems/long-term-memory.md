# 长期记忆：向量数据库、Memory Bank 与跨会话知识

短期记忆解决了"当前会话"的问题，但 Agent 还必须**跨会话**保留知识——用户的偏好、历史任务的教训、关键事实。长期记忆（Long-Term Memory, LTM）的核心是把信息**持久化存储、按语义检索、按需注入 context**。本文深入 LTM 的存储结构（向量库、知识图谱、SQL）、检索算法、写入策略与生产实践。

## 一、为什么需要长期记忆

短期记忆的局限：

```text
- 单次会话关闭后清空
- 容量受 context window 限制
- 用户的偏好、历史不能跨会话复用
- Agent 每次都"重新认识"用户

长期记忆要做到：
  - 跨会话持久化
  - 按语义检索相关条目
  - 自动管理（写入 / 遗忘 / 更新）
  - 与短期记忆无缝衔接
```

## 二、长期记忆的存储结构

### 2.1 向量存储（最主流）

把每条记忆编码成 embedding 存入向量库，检索时按相似度 top-K：

```python
import chromadb

chroma = chromadb.PersistentClient(path="./memory_db")
collection = chroma.get_or_create_collection("user_memory")

# 写入
collection.add(
    documents=["用户喜欢简洁回答，不喜欢冗长解释",
              "用户正在学习 LLM，基础是 PyTorch"],
    metadatas=[{"type": "preference"}, {"type": "context"}],
    ids=["mem_001", "mem_002"],
)

# 检索
results = collection.query(
    query_texts=["回答应该多详细？"],
    n_results=3,
)
# [用户喜欢简洁回答，不喜欢冗长解释, ...]
```

**优点**：语义检索、自然语言友好。**缺点**：无法表达复杂关系。

### 2.2 知识图谱

把记忆组织成实体-关系图：

```python
from py2neo import Graph, Node, Relationship

g = Graph("bolt://localhost:7687")

alice = Node("User", name="Alice")
topic = Node("Topic", name="LLM")
framework = Node("Framework", name="PyTorch")

g.create(alice)
g.create(Relationship(alice, "INTERESTED_IN", topic))
g.create(Relationship(alice, "USES", framework))
g.create(Relationship(framework, "USED_FOR", topic))
```

查询：

```cypher
MATCH (u:User)-[r:USES]->(f:Framework)-[:USED_FOR]->(t:Topic)
WHERE u.name = "Alice"
RETURN f.name, t.name
```

**优点**：表达关系、可推理。**缺点**：构建成本高。

### 2.3 结构化 SQL / KV

```sql
CREATE TABLE user_memory (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64),
    key VARCHAR(128),
    value JSONB,
    created_at TIMESTAMP,
    expires_at TIMESTAMP,
    importance INT
);

CREATE INDEX idx_user_key ON user_memory(user_id, key);
```

适合显式的属性记忆（"用户的名字是 Alice"、"用户的邮箱是 ..."）。

### 2.4 混合存储

生产系统往往混合多种存储：

```text
SQL：结构化用户属性（name, email, tier）
向量库：语义记忆（"用户上次讨论过 X 主题"）
知识图谱：实体关系（"用户 → 公司 → 行业"）
KV 缓存：会话级临时状态
```

## 三、Memory 的写入策略

### 3.1 显式写入

用户或开发者主动调用：

```python
memory.add("user_prefers", "concise_answers")
memory.add("user_skill", "PyTorch")
```

### 3.2 隐式写入（更智能）

Agent 在对话中自动提取"值得记住"的信息：

```python
EXTRACT_PROMPT = """从以下对话中提取值得长期记住的信息：
- 用户偏好（喜欢什么风格）
- 关键事实（用户身份、目标）
- 重要约定（"下次再聊"的事项）

只输出 JSON 数组：[{"type": ..., "content": ...}, ...]
不要提取琐碎信息。"""

def extract_and_store(conversation: list[dict]):
    response = llm.invoke(EXTRACT_PROMPT + "\n\n对话：\n" + format_msgs(conversation))
    memories = json.loads(response)
    for m in memories:
        memory.add(m["type"], m["content"])
```

### 3.3 周期性总结

每隔 N 轮或 N 分钟，把短期记忆压缩写入长期：

```python
def periodic_consolidation():
    recent_msgs = short_term.get_all()
    summary = llm.invoke(f"总结以下对话中需要长期保留的信息：\n{recent_msgs}")
    long_term.add(summary, importance=evaluate_importance(summary))
```

## 四、Memory 的检索与注入

### 4.1 检索时机

- **每轮对话前**：把相关 LTM 注入 system prompt。
- **工具调用前**：检索历史 tool 用法。
- **任务规划前**：检索过去的类似任务经验。

### 4.2 检索实现

```python
class MemoryAugmentedAgent:
    def __init__(self, llm, short_term, long_term):
        self.llm = llm
        self.short_term = short_term
        self.long_term = long_term

    def get_system_prompt(self) -> str:
        # 检索相关长期记忆
        user_query = self.short_term.last_user_message()
        relevant_memories = self.long_term.retrieve(user_query, top_k=5)

        memory_section = "\n".join(f"- {m.content}" for m in relevant_memories)
        return f"""
        你是一名助手。
        
        [长期记忆中关于用户的相关信息]
        {memory_section}
        
        请基于以上信息回答用户。
        """

    def step(self, user_msg: str) -> str:
        self.short_term.add("user", user_msg)

        # 检索 LTM
        memories = self.long_term.retrieve(user_msg, top_k=3)
        context = self.get_system_prompt()

        # 调 LLM
        response = self.llm.invoke(context + user_msg)

        # 写入 LTM（隐式）
        self.extract_and_store(user_msg, response)

        self.short_term.add("assistant", response)
        return response
```

### 4.3 注入格式

三种常见方式：

**方式 A：直接插入 system prompt**

```text
[已知用户偏好]
- 用户喜欢简洁回答
- 用户正在学 LLM
- 用户痛点是工程化落地
```

**方式 B：作为工具返回**

```python
def recall(query: str) -> str:
    """从长期记忆中检索信息"""
    results = long_term.retrieve(query, top_k=5)
    return "\n".join(m.content for m in results)

agent.tools.append(recall)
```

**方式 C：作为 RAG 上下文**

把 LTM 当作文档集合，让 retriever 选 top-K。

## 五、Memory Bank：MemGPT 的 LTM 抽象

MemGPT（UC Berkeley, 2023）提出**分层 memory 架构**：

```text
┌──────────────────────────────────────┐
│  Main Context (= Context Window)      │
│  ┌────────────┐  ┌────────────────┐  │
│  │ System     │  │ Working Context │  │
│  │ Instructions│  │ (动态注入)      │  │
│  └────────────┘  └────────────────┘  │
└──────────────┬───────────────────────┘
               ↓ function calls
┌──────────────────────────────────────┐
│  External Context (= Long-Term)       │
│  - Recall Storage (向量库)            │
│  - Archival Storage (KV / DB)         │
└──────────────────────────────────────┘
```

Agent 通过**显式 function call** 在主上下文和外部存储间搬运数据：

```python
def memory_move_to_context(memory_id: str):
    """从 LTM 加载到主 context"""
    content = archival_storage.get(memory_id)
    working_context.append(content)

def memory_search(query: str, top_k: int = 5):
    """向量检索相关记忆"""
    results = recall_storage.search(query, top_k=top_k)
    return results

def memory_archive(content: str):
    """从主 context 写入 LTM"""
    memory_id = archival_storage.put(content)
    return memory_id
```

LLM 主动决定**何时读、读什么、什么时候写、写什么**——这是"memory as a tool"的范式。

## 六、Memory 衰减与遗忘

人脑会遗忘，AI Memory 系统也需要"清理"机制：

### 6.1 时间衰减

```python
import math

def recency_weight(created_at: datetime, now: datetime, half_life_days: float = 30) -> float:
    age_days = (now - created_at).days
    return math.exp(-age_days * math.log(2) / half_life_days)
```

越久远的记忆权重越低，半衰期 30 天意味着 30 天后权重减半。

### 6.2 重要度评分

```python
def importance(content: str) -> float:
    # 用 LLM 评估
    prompt = f"评估以下信息对用户的长期价值（1-10）：\n{content}"
    return float(llm.invoke(prompt))
```

重要度高的记忆（"用户父亲过世"）永不衰减；琐碎记忆（"今天天气不错"）快速衰减。

### 6.3 容量上限

```python
def evict_if_over_capacity():
    memories = long_term.list_all()
    if len(memories) > MAX_MEMORIES:
        # 按 retention_score = importance × recency 排序
        scored = [(importance(m) * recency_weight(m), m) for m in memories]
        scored.sort()
        # 删除最差的
        to_remove = scored[:len(memories) - MAX_MEMORIES]
        for score, m in to_remove:
            long_term.delete(m.id)
```

## 七、安全与隐私

长期记忆带来新风险：

```text
- 用户隐私：长期存储 = 数据泄露风险大
- 跨用户泄漏：vector search 可能返回其他用户的信息
- 不当内容：恶意 prompt 让 Agent 存错误信息
- 监管：GDPR "right to be forgotten"
```

工程对策：

1. **用户隔离**：每个用户独立 collection，加 ACL。
2. **PII 过滤**：写入前用 NER 模型识别并脱敏。
3. **删除 API**：`memory.delete(user_id, memory_id)`，向量库做级联删除。
4. **加密**：LTM 静态加密（AES-256），传输 TLS。
5. **审计日志**：谁在什么时候读写过。

## 八、典型系统对比

| 系统 | 存储 | 检索 | 写入策略 |
|---|---|---|---|
| **MemGPT** | Recall + Archival | vector + KV | LLM 主动 |
| **ChatGPT Memory** | 向量库 | 语义检索 | 隐式提取 |
| **LangChain Memory** | 多种 backend | 灵活 | 显式 + 隐式 |
| **CrewAI** | SQLite | 全文 | 显式 |
| **Letta (ex-Berkeley)** | Postgres + pgvector | hybrid | LLM 主动 |

## 小结

长期记忆让 Agent 从"金鱼"（7 秒记忆）变成"大象"（永久记住）。核心架构是**短期压缩 → 长期存储 → 语义检索 → 按需注入**。生产系统要叠加衰减、容量管理、用户隔离、PII 过滤。MemGPT 的"memory as a tool"是当前最优雅的范式，让 LLM 主动管理自己的记忆。下一篇我们将进入 **episodic-semantic-memory**——人类记忆分类对 AI Agent 的启示。
