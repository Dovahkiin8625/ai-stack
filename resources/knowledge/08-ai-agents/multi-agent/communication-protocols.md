# 多 Agent 通信协议：从消息总线到 FIPA ACL

多 Agent 系统（MAS, Multi-Agent System）让多个自主实体协作完成复杂任务。**通信协议**是协作的基础——Agent 之间用什么格式、什么语义、什么机制交换信息？本文梳理从消息总线到智能体通信语言（FIPA ACL）、现代 LLM 多 Agent 框架的协议设计，以及生产环境的工程考量。

## 一、为什么需要通信协议

多个 Agent 协作时面临的问题：

```text
- 谁负责哪个子任务？
- 如何告诉别人"我完成了 X"？
- 如何请求别人的能力？
- 如何共享知识而不重复计算？
- 如何保证通信不被滥用？

→ 通信协议 = 标准化这一切
```

## 二、协议分层模型

参考 ISO/OSI 模型的简化：

```text
应用层   │ FIPA-ACL, KQML, JSON over HTTP
         │ "我要订一张明天去北京的机票"
─────────┼───────────────────────────────────
对话层   │ FIPA Conversation Protocols
         │ request, query, propose, accept
─────────┼───────────────────────────────────
传输层   │ TCP, HTTP/2, WebSocket, gRPC
         │ 字节传输
```

## 三、经典协议：FIPA ACL

FIPA（Foundation for Intelligent Physical Agents）制定了智能体通信标准。

### 3.1 消息结构

```text
(inform
    :sender    agent1
    :receiver  agent2
    :content   "上海今天天气晴，25°C"
    :language  sl             ; 语义语言
    :ontology  weather        ; 本体
    :protocol  fipa-request
    :conversation-id conv-001
    :reply-with  msg-001
)
```

关键字段：

- **Performative（言语行为）**：消息意图
- **Sender / Receiver**：通信双方
- **Content**：实际内容（任意格式）
- **Ontology**：共享的概念词汇表
- **Protocol**：对话协议（如 fipa-request）

### 3.2 常见 Performative

| Performative | 含义 |
|---|---|
| **inform** | 通知事实 |
| **request** | 请求执行动作 |
| **query-if** | 询问命题真假 |
| **query-ref** | 询问对象 |
| **propose** | 提议方案 |
| **accept-proposal** | 接受提议 |
| **reject-proposal** | 拒绝提议 |
| **subscribe** | 订阅事件 |
| **agree** | 同意请求 |
| **refuse** | 拒绝请求 |
| **failure** | 报告失败 |

### 3.3 对话协议

FIPA 定义了几种典型对话流程：

#### FIPA-Request

```text
Requester                Responder
    │                        │
    │ ──── request ────────→ │
    │                        │ (decide)
    │ ←── agree/refuse ───── │
    │                        │
    │ ←── inform ───────────  │ (执行结果)
    │     OR                 │
    │ ←── failure ────────── │
```

#### FIPA-Contract-Net（任务招标）

```text
Manager                  Contractors
    │                        │
    │ ── cfp (招标) ───────→ │ ×N
    │                        │
    │ ←─ propose ─────────── │ ×N (投标)
    │                        │
    │ ── accept-proposal ──→ │ (中标)
    │ ── reject-proposal ──→ │ ×(N-1)
    │                        │
    │ ←─ inform ──────────── │ (执行结果)
```

类似拍卖：Manager 发标，多个 Contractor 投标，最优者中标执行。

## 四、KQML（更早的协议）

KQML（Knowledge Query and Manipulation Language）是 FIPA ACL 的前身。

```lisp
(ask-all
  :content      (price IBM ?price)
  :receiver     stock-server
  :language     LISP
  :ontology     NYSE-TICKS
)
```

KQML 与 FIPA ACL 思想相似，已被后者取代。

## 五、LLM 时代的多 Agent 协议

经典 ACL 语义精确但学习曲线陡。LLM 时代多 Agent 协议更轻量：

### 5.1 自然语言消息

```python
# 最简单的协议：纯文本
message = {
    "from": "researcher_agent",
    "to": "writer_agent",
    "content": """
        调研完成。找到 3 个关键点：
        1. Transformer 用 attention 替代 RNN
        2. Pre-Norm 比 Post-Norm 稳定
        3. KV cache 通过 GQA 节省显存
        
        请基于此写一篇文章。
    """,
}
```

LLM 能直接理解，无需严格的语义层。

### 5.2 结构化 + 自然语言混合

```python
message = {
    "from": "researcher_agent",
    "to": "writer_agent",
    "type": "task_handoff",
    "task": "write_article",
    "input": {
        "topic": "Transformer 架构",
        "key_points": [...],
    },
    "instructions": "目标读者是工程师，需要含 PyTorch 代码示例",
}
```

结构化字段 + 自然语言 instructions 是 LLM Agent 的最佳实践。

### 5.3 工具调用即通信

现代框架把 Agent 通信实现为"互相调用工具"：

```python
# Researcher Agent 把工具注册给 Writer Agent
@tool
def submit_findings(findings: list[str], sources: list[str]) -> None:
    """提交研究发现给 Writer Agent"""
    writer_agent.input_queue.put({
        "type": "findings",
        "content": findings,
        "sources": sources,
    })
```

## 六、消息总线：底层基础设施

### 6.1 实现选项

| 技术 | 优点 | 缺点 |
|---|---|---|
| **Redis Pub/Sub** | 简单、低延迟 | 无持久化、无消息回溯 |
| **RabbitMQ** | 可靠、灵活 routing | 较重 |
| **Kafka** | 高吞吐、持久化、可回放 | 学习曲线 |
| **NATS** | 极轻量、pub/sub 友好 | 持久化弱 |
| **ZeroMQ** | 嵌入式库、无中心 | 需自实现可靠性 |

### 6.2 Kafka 实现

```python
from kafka import KafkaProducer, KafkaConsumer
import json

producer = KafkaProducer(
    bootstrap_servers="kafka:9092",
    value_serializer=lambda v: json.dumps(v).encode(),
)

def send_message(to: str, content: dict):
    producer.send(f"agent.{to}", {
        "from": AGENT_ID,
        "content": content,
        "timestamp": time.time(),
    })

def receive_messages() -> Iterator[dict]:
    consumer = KafkaConsumer(
        f"agent.{AGENT_ID}",
        bootstrap_servers="kafka:9092",
        value_deserializer=lambda v: json.loads(v.decode()),
    )
    for msg in consumer:
        yield msg.value
```

### 6.3 消息路由

```python
# 主题路由
producer.send("agent.writer.task_request", message)

# 直接消息
producer.send(f"agent.{target_id}.inbox", message)

# 广播
producer.send("agent.broadcast.all", message)
```

## 七、对话状态管理

多 Agent 对话需要状态追踪：

```python
class Conversation:
    id: str
    participants: list[str]
    protocol: str              # "fipa-request", "contract-net", ...
    state: dict
    history: list[Message]

class ConversationManager:
    def __init__(self):
        self.conversations: dict[str, Conversation] = {}

    def get_or_create(self, conv_id, participants, protocol):
        if conv_id not in self.conversations:
            self.conversations[conv_id] = Conversation(
                id=conv_id,
                participants=participants,
                protocol=protocol,
                state={},
                history=[],
            )
        return self.conversations[conv_id]

    def validate_message(self, conv: Conversation, msg: Message) -> bool:
        """检查消息是否合法（在该 protocol 状态下）"""
        return PROTOCOLS[conv.protocol].is_valid(conv.state, msg)
```

## 八、安全与权限

多 Agent 通信的安全挑战：

```text
1. 身份认证：消息确实来自声称的 Agent？
2. 授权：Agent A 能调用 Agent B 的工具 X 吗？
3. 消息加密：敏感内容不能明文传输
4. 审计：谁在什么时候说过什么？
5. 速率限制：防止某个 Agent 消息洪水
```

### 8.1 JWT 鉴权

```python
import jwt

def sign_message(agent_id: str, message: dict) -> str:
    return jwt.encode(
        {"agent_id": agent_id, "ts": time.time(), "msg_hash": hash(message)},
        SECRET_KEY,
        algorithm="HS256",
    )

def verify_message(token: str, message: dict) -> str | None:
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    if payload["msg_hash"] == hash(message):
        return payload["agent_id"]
    return None
```

### 8.2 RBAC

```python
PERMISSIONS = {
    "researcher_agent": ["search_web", "read_documents"],
    "writer_agent": ["read_documents", "submit_article"],
    "reviewer_agent": ["read_documents", "approve_article", "reject_article"],
}

def can_call(agent_id: str, tool: str) -> bool:
    return tool in PERMISSIONS.get(agent_id, [])
```

## 九、生产级考量

### 9.1 消息丢失 vs 重复

- **至少一次**：消息可能重复（消费侧去重）。
- **最多一次**：消息可能丢失（适用不重要通知）。
- **恰好一次**：难实现，通常靠"幂等 + 去重"近似。

### 9.2 死信队列

```python
# 处理无法路由或解析失败的消息
consumer.send("dlq.failed_messages", message)
```

### 9.3 监控

```python
# 关键指标
- 消息吞吐量
- 端到端延迟（producer → consumer）
- 队列深度（积压情况）
- 失败 / 重试率
- 每个 Agent 的消息速率
```

## 十、典型实现对比

| 框架 | 协议 | 传输 | 适合 |
|---|---|---|---|
| **FIPA-JADE** | FIPA ACL | HTTP / RMI | 学术研究 |
| **AutoGen** | 自然语言 + JSON | 进程内 | LLM Agent 实验 |
| **CrewAI** | 任务列表 + 上下文 | 进程内 | 中等规模协作 |
| **LangGraph** | 状态图 + 消息 | 进程内 | 复杂编排 |
| **AgentVerse** | Blackboard | 数据库 | 大规模仿真 |

## 小结

多 Agent 通信协议从经典 FIPA ACL 的严格语义，走向 LLM 时代的"自然语言 + 结构化 JSON"。无论哪种风格，**身份、消息格式、对话状态、权限、可靠性**是五大约束。生产系统应该用成熟消息总线（Kafka / NATS）做底层传输，对话状态由框架（AutoGen / CrewAI）维护，安全通过 JWT + RBAC 实现。下一篇我们将深入 **cooperation-mechanisms**——Agent 之间如何协作。
