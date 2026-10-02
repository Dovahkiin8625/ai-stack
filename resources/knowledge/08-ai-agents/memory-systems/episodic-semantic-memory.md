# 情景记忆与语义记忆：从人类认知到 Agent 设计

认知心理学把人类记忆分为多种类型：**情景记忆（episodic）**、**语义记忆（semantic）**、**程序性记忆（procedural）**。这套分类对 AI Agent 设计极具启发——不同类型的信息需要不同的存储、检索、使用方式。本文梳理人类记忆分类对 Agent 设计的映射，以及工程实现路径。

## 一、Tulving 的记忆分类

Endel Tulving 1972 年提出多重记忆模型，区分情景记忆与语义记忆：

| 类型 | 内容 | 例 |
|---|---|---|
| **情景记忆（Episodic）** | 个人经历的事件 | "我昨天去过咖啡馆" |
| **语义记忆（Semantic）** | 通用事实知识 | "巴黎是法国首都" |
| **程序性记忆（Procedural）** | 技能、习惯 | "骑自行车" |

后来扩展加入：

| 类型 | 内容 | 例 |
|---|---|---|
| **工作记忆（Working）** | 当前任务上下文 | 算术中间步骤 |
| **感知记忆（Sensory）** | 原始感官输入 | 当前视野 |
| **前瞻记忆（Prospective）** | 未来要做的事 | "记得买牛奶" |
| **元记忆（Metamemory）** | 关于记忆的知识 | "我记得这个" |

## 二、LLM Agent 的记忆映射

### 2.1 当前 LLM 的"原生记忆"

```text
- 预训练权重 = 语义记忆 + 程序性记忆（静态、压缩、无限）
- Context window = 工作记忆 + 感知（动态、有限）
- 没有真正的情景记忆（除非外挂）
```

LLM 本身是"语义记忆体"，但缺乏**个人化**的情景记忆。

### 2.2 Agent 工程的映射

| 人类记忆 | Agent 实现 |
|---|---|
| 情景记忆 | 时间戳事件日志（"用户 2024-01-15 问了 X"） |
| 语义记忆 | 向量库中的事实条目（"用户喜欢咖啡"） |
| 程序性记忆 | Tools、skills、policy |
| 工作记忆 | Context window + KV cache |
| 感知记忆 | 当前对话 / 工具返回 |
| 前瞻记忆 | Todo list / scheduler |
| 元记忆 | "我对这个问题的把握"（置信度） |

## 三、情景记忆：Agent 的"个人日记"

### 3.1 设计原则

情景记忆存储"事件"而非"事实"：

```python
@dataclass
class EpisodicMemory:
    event_id: str
    timestamp: datetime
    actor: str                  # 谁（agent / user）
    action: str                 # 做了什么
    context: dict               # 当时的环境
    outcome: dict | None        # 结果
    embedding: list[float]      # 用于检索
```

```python
episodic_db.add(EpisodicMemory(
    event_id="evt_001",
    timestamp=datetime(2024, 1, 15, 14, 30),
    actor="user",
    action="asked_about_LLM_quantization",
    context={"topic": "AI", "intent": "learning"},
    outcome={"helpful": True, "satisfaction": 4},
))
```

### 3.2 检索：情景 vs 语义

情景记忆的检索往往需要**多维度过滤**：

```python
# 找出"上个月关于量化的话题"
results = episodic_db.query(
    time_range=(datetime.now() - timedelta(days=30), datetime.now()),
    topic="quantization",
    actor="user",
    limit=10,
)

# 找出"用户满意度 < 3 的事件"
results = episodic_db.query(
    outcome={"satisfaction": {"$lt": 3}},
)
```

```sql
-- SQL 实现
SELECT * FROM episodic_memory
WHERE timestamp > NOW() - INTERVAL '30 days'
  AND topic = 'quantization'
  AND actor = 'user'
ORDER BY timestamp DESC
LIMIT 10;
```

### 3.3 应用：Agent 自我反思

```python
def reflect_on_failure(current_task: str):
    """遇到新任务时，先查历史失败案例"""
    similar_past = episodic_db.search_by_embedding(current_task, top_k=5)

    prompt = f"""
    类似的历史事件：
    {format_events(similar_past)}

    现在的新任务是：{current_task}

    基于历史教训，我应该注意什么？
    """
    return llm.invoke(prompt)
```

类比人类的"前事不忘，后事之师"。

## 四、语义记忆：Agent 的"百科全书"

### 4.1 特点

- **去情景化**：脱离时间地点的"事实"。
- **跨事件一致**："用户喜欢咖啡"这条记忆不依赖具体会话。
- **高度抽象**：可能是"用户兴趣图谱"。

### 4.2 实现

```python
@dataclass
class SemanticMemory:
    concept: str
    attributes: dict
    source_events: list[str]      # 从哪些情景记忆提炼出来的
    confidence: float              # 0-1
    last_updated: datetime
    embedding: list[float]
```

```python
semantic_db.add(SemanticMemory(
    concept="user_coffee_preference",
    attributes={"drink": "latte", "sugar": "less", "frequency": "daily"},
    source_events=["evt_001", "evt_023", "evt_089"],
    confidence=0.9,
    last_updated=datetime.now(),
))
```

### 4.3 从情景到语义的提炼

定期把多条情景记忆"消化"为语义记忆：

```python
def consolidate_to_semantic():
    """周期性：从近期情景中提取语义事实"""
    recent_events = episodic_db.get_last_n_days(30)

    prompt = f"""
    以下是用户近 30 天的事件：
    {format_events(recent_events)}

    请提取可总结的事实（用户偏好、习惯、特征），
    输出 JSON：[{{"concept": ..., "attributes": {{...}}, "confidence": 0.0-1.0}}]
    """

    new_facts = json.loads(llm.invoke(prompt))

    for fact in new_facts:
        # 检查是否与已有事实冲突
        existing = semantic_db.find(fact["concept"])
        if existing and existing.attributes != fact["attributes"]:
            # 冲突：用更新版本的 fact，confidence 高者优先
            if fact["confidence"] > existing.confidence:
                semantic_db.update(fact)
        else:
            semantic_db.add(fact)
```

这是**记忆巩固**（memory consolidation）的算法实现。

## 五、程序性记忆：Agent 的"肌肉记忆"

### 5.1 形式

程序性记忆 = 工具 + 策略 + 工作流模板。

```python
# 显式存储"如何处理退款请求"的工作流
PROCEDURE_REFUND = """
1. 查询订单状态（tool: get_order）
2. 如果订单在 30 天内：
   a. 调用退款 API（tool: refund_order）
   b. 发送确认邮件（tool: send_email）
3. 否则：
   a. 解释退款政策
   b. 升级到人工
"""

procedural_db.add(name="refund", body=PROCEDURE_REFUND)
```

### 5.2 自动学习

Agent 可以从历史成功案例中**总结新流程**：

```python
def learn_procedure(task: str, successful_trace: list[dict]):
    """从成功 trace 中提炼可复用流程"""
    prompt = f"""
    任务：{task}
    执行 trace：
    {format_trace(successful_trace)}

    请提炼一个可复用的流程模板（步骤化、参数化）。
    """
    procedure = llm.invoke(prompt)
    procedural_db.add(name=task, body=procedure)
```

### 5.3 Skills 库：现代程序性记忆

Anthropic 的 "Skills"、OpenAI 的 "Custom GPTs" 都是把程序性记忆工具化的体现：

```yaml
name: sql-analyst
description: 从自然语言生成 SQL 查询
tools:
  - get_schema
  - execute_query
steps:
  - 解析用户意图
  - 调用 get_schema 获取相关表
  - 生成 SQL
  - 校验 SQL 语法
  - 执行 + 解释结果
```

## 六、前瞻记忆：Agent 的"待办"

前瞻记忆 = 未来要做的事。

```python
@dataclass
class ProspectiveMemory:
    task: str
    trigger: str           # 触发条件（时间 / 事件）
    deadline: datetime | None
    priority: int
    context: dict
```

```python
prospective_db.add(ProspectiveMemory(
    task="用户上次要求：明天上午 9 点发会议纪要",
    trigger="2026-10-03T09:00:00",
    deadline=datetime(2026, 10, 3, 9, 0),
    priority=3,
    context={"user_id": "alice", "meeting_id": "m_123"},
))
```

实现：

```python
# Scheduler 每分钟检查
def check_prospective():
    now = datetime.now()
    due = prospective_db.find_due(now)
    for task in due:
        execute_task(task)
        prospective_db.remove(task.id)
```

LLM Agent 也可以显式"提醒未来自己"：

```python
@tool
def schedule_task(task: str, trigger: str, priority: int = 3):
    """安排未来的任务"""
    prospective_db.add(task=task, trigger=trigger, priority=priority)
    return "已安排"
```

## 七、元记忆：知道自己知道什么

元记忆是"对自己记忆的认识"。LLM Agent 的一种实现：

```python
def confidence_in_answer(question: str, retrieved_context: list) -> float:
    """根据检索质量估计置信度"""
    if not retrieved_context:
        return 0.2
    avg_similarity = mean(c.similarity for c in retrieved_context)
    return min(0.95, avg_similarity)

# 用法
confidence = confidence_in_answer(user_query, retrieved_docs)
if confidence < 0.4:
    response = "我不太确定，但据我所知..."
elif confidence > 0.8:
    response = "根据资料显示..."
```

更高级的"知道我不知道"——这是真正智能的体现：

```python
METAPROMPT = """
评估你对以下问题的把握（1-10）：
- 你是否有相关知识？
- 检索到的信息是否一致？
- 是否涉及推测？
只输出一个整数。
"""
```

## 八、生产系统的分层架构

```text
┌────────────────────────────────────────┐
│ Working Memory (Context Window)        │ ← LLM 直接看到
│   + 最近 N 条消息                       │
│   + 检索到的 top-K 语义/情景记忆        │
└───────────────┬────────────────────────┘
                ↕ function calls (read/write)
┌────────────────────────────────────────┐
│ Episodic Memory (事件日志)              │
│   - 时间戳 + actor + action + outcome   │
│   - 用于反思、经验检索                  │
└───────────────┬────────────────────────┘
                ↕ periodic consolidation
┌────────────────────────────────────────┐
│ Semantic Memory (向量库)                │
│   - 概念 + 属性 + 置信度               │
│   - 注入 context 提供个性化            │
└────────────────────────────────────────┘
┌────────────────────────────────────────┐
│ Procedural Memory (Skills / Tools)    │
│   - 工作流模板                          │
│   - 工具描述 + 调用模式                  │
└────────────────────────────────────────┘
┌────────────────────────────────────────┐
│ Prospective Memory (Scheduler)         │
│   - 未来任务、提醒                      │
│   - 由 LLM 主动安排                     │
└────────────────────────────────────────┘
```

## 九、设计取舍

| 维度 | 偏情景 | 偏语义 |
|---|---|---|
| **检索方式** | 时间 + 维度过滤 | 语义相似度 |
| **存储量** | 大（每事件一条） | 小（精炼后） |
| **更新频率** | 高 | 低 |
| **适用** | 历史回溯、反思 | 个性化、当前任务上下文 |

混合使用效果最佳：情景是"原材料"，语义是"成品"。

## 小结

人类记忆分类为 LLM Agent 设计提供了清晰框架。**情景记忆**记录事件、**语义记忆**提炼事实、**程序性记忆**封装技能、**工作记忆**承载当前任务、**前瞻记忆**管理未来。生产系统需要把这五层架构协同起来，让 Agent 不仅能"对话"，还能"成长"——每次交互都让它更懂用户、更擅长任务。下一篇我们将进入 **multi-agent**——当单个 Agent 不够用时如何协作。
