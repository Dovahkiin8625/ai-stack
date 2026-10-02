# 协作机制：任务分配、共识算法与团队涌现

多 Agent 系统要解决的核心问题是"**怎么让一群自主的 Agent 一起完成单个 Agent 做不了的事**"。这涉及任务如何分配、冲突如何解决、信息如何共享、信任如何建立。本文梳理多 Agent 协作的核心机制——任务分配、市场机制、投票、黑板模型、共识算法，以及 LLM 多 Agent 系统特有的协作模式。

## 一、协作的层次

```text
合作层级             │ 机制               │ 经典场景
─────────────────────┼────────────────────┼──────────────────
被动合作             │ 共享环境           │ 蜂群觅食
主动协商             │ 通信 + 协议        │ 合同网
团队协作             │ 角色分工 + 任务流   │ 软件团队
群体智能             │ 涌现 + 自组织      │ 蚁群算法
竞争 + 协作          │ 博弈 + 信誉         │ 拍卖 / 投票
```

## 二、任务分配：合同网协议（Contract Net）

合同网（Contract Net Protocol, Smith 1980）是最经典的任务分配机制，类似招标投标：

```text
Manager                Contractors
   │                       │
   │ ── call-for-proposal →  │ ×N
   │                       │ (各自评估)
   │ ←── propose ───────── │ ×N
   │                       │
   │ (评估所有投标)         │
   │                       │
   │ ── award ──────────── → │ (中标)
   │ ── reject ──────────── → │ ×(N-1)
```

### 2.1 实现

```python
class ContractNetManager:
    def __init__(self, contractors: list[Agent]):
        self.contractors = contractors

    def assign_task(self, task: Task) -> Agent:
        # 1. 招标
        proposals = []
        for contractor in self.contractors:
            proposal = contractor.propose(task)
            proposals.append(proposal)

        # 2. 评估与选择
        best = max(proposals, key=lambda p: (
            p.bid_price * 0.4 +
            p.estimated_quality * 0.4 +
            p.estimated_time * 0.2,
        ))

        # 3. 中标 + 通知
        for proposal in proposals:
            if proposal.agent_id == best.agent_id:
                proposal.agent.accept(task)
            else:
                proposal.agent.reject(task)

        return best.agent
```

### 2.2 适用场景

- 任务可分解为子任务
- 多个 Contractor 能力重叠
- 评估标准可量化（成本、质量、时间）

## 三、市场机制：基于价格的资源分配

让 Agent 像市场参与者一样通过价格信号分配资源：

```python
class MarketMechanism:
    def __init__(self):
        self.order_book = {"buy": [], "sell": []}

    def place_order(self, agent_id: str, side: str, resource: str, qty: int, price: float):
        self.order_book[side].append({
            "agent": agent_id, "resource": resource,
            "qty": qty, "price": price,
        })

    def match(self):
        """连续双向拍卖"""
        for buy in self.order_book["buy"]:
            for sell in self.order_book["sell"]:
                if buy["price"] >= sell["price"]:
                    # 成交
                    qty = min(buy["qty"], sell["qty"])
                    price = (buy["price"] + sell["price"]) / 2
                    self.execute(buy["agent"], sell["agent"], qty, price)
```

LLM Agent 中可用于：

- 多 Agent 竞争 GPU 资源
- 多 Agent 抢答问题（出价最高者回答）

## 四、投票与共识

### 4.1 多数投票

```python
def majority_vote(agents: list[Agent], question: str) -> Any:
    votes = [agent.vote(question) for agent in agents]
    return Counter(votes).most_common(1)[0][0]
```

简单，但容易被"恶意多数"操纵。

### 4.2 加权投票（基于信誉）

```python
def weighted_vote(agents: list[Agent], question: str) -> Any:
    votes = []
    weights = []
    for agent in agents:
        v = agent.vote(question)
        w = reputation_system.get_score(agent.id)
        votes.append(v)
        weights.append(w)
    return weighted_majority(votes, weights)
```

信誉高的 Agent 票数权重更大。

### 4.3 拜占庭容错（BFT）共识

当 Agent 不可信时，需要 PBFT 之类的协议：

```text
PBFT 流程：
1. Client 发请求到 Primary
2. Primary 广播 Pre-Prepare
3. 所有节点发送 Prepare
4. 收到 2f+1 个 Prepare 后发送 Commit
5. 收到 2f+1 个 Commit 后执行
6. 回复 Client
```

适用场景：金融多 Agent 系统、强一致性需求。

### 4.4 LLM Multi-Agent 的"辩论"

```python
def debate_consensus(agents: list[Agent], question: str, rounds: int = 3) -> str:
    """多 Agent 辩论达到共识"""
    opinions = [agent.initial_opinion(question) for agent in agents]

    for round in range(rounds):
        # 每个 Agent 看其他人观点，更新自己的
        new_opinions = []
        for i, agent in enumerate(agents):
            others = [opinions[j] for j in range(len(agents)) if j != i]
            new_op = agent.update_opinion(question, others, round)
            new_opinions.append(new_op)
        opinions = new_opinions

    # 最终用多数或 LLM 仲裁
    return majority_or_arbitrate(opinions)
```

Du et al. 2023 的论文证明，多 Agent 辩论能提升 LLM 的事实准确率 5-10%。

## 五、黑板模型：共享工作区

Blackboard 模式让所有 Agent 共享一个"工作区"，各自订阅感兴趣的事件：

```text
                BlackBoard
            ┌──────────────┐
   Agent A ─→│   共享状态    │←─ Agent B
            │  (任务、结果) │
            └──────────────┘
                  ↑
                  │
               Agent C
```

### 5.1 实现

```python
class Blackboard:
    def __init__(self):
        self.posts: dict[str, list] = defaultdict(list)
        self.subscribers: dict[str, list[Agent]] = defaultdict(list)

    def post(self, topic: str, data: dict, author: str):
        post_id = uuid4()
        self.posts[topic].append({
            "id": post_id, "data": data, "author": author,
            "timestamp": time.time(),
        })
        for agent in self.subscribers[topic]:
            agent.notify(topic, self.posts[topic][-1])

    def subscribe(self, topic: str, agent: Agent):
        self.subscribers[topic].append(agent)
```

### 5.2 适用场景

- 多专家系统（医疗诊断：症状 → 检验 → 影像 → 病理 → 综合）
- 信息检索：不同 Agent 关注不同源，汇总到黑板

### 5.3 CrewAI 的实现

CrewAI 的"任务上下文共享"本质就是黑板：

```python
crew = Crew(agents=[researcher, writer, editor],
            tasks=[research_task, write_task, edit_task])
# edit_task 自动拿到 write_task 的输出作为 context
```

## 六、角色分工：团队协作模式

### 6.1 经典团队模式

```text
Chief → Worker ×N:    一对多管理
Swarm → Swarm:        完全平等（蚁群）
Relay:                顺序接力（接力赛）
Hub-Spoke:            中心化（联邦学习）
```

### 6.2 LLM Multi-Agent 角色

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="高级研究员",
    goal="搜集和验证关于 {topic} 的最新信息",
    backstory="你是一位有 10 年经验的科技记者，擅长快速验证信息",
)

analyst = Agent(
    role="数据分析师",
    goal="基于 {topic} 数据分析趋势",
    backstory="你擅长从数据中提取洞见",
)

writer = Agent(
    role="技术作家",
    goal="把研究和分析写成易懂文章",
    backstory="你是《连线》杂志的特约撰稿人",
)

reviewer = Agent(
    role="事实核查员",
    goal="确保文章没有事实错误",
    backstory="你曾任 Nature 期刊编辑",
)

task1 = Task(description="调研 {topic}", agent=researcher)
task2 = Task(description="分析趋势", agent=analyst)
task3 = Task(description="写文章", agent=writer, context=[task1, task2])
task4 = Task(description="审稿", agent=reviewer, context=[task3])

crew = Crew(
    agents=[researcher, analyst, writer, reviewer],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential,
)
```

### 6.3 动态角色 vs 静态角色

| 维度 | 静态角色 | 动态角色 |
|---|---|---|
| **预设** | 启动时定义 | 任务运行时决定 |
| **灵活度** | 低 | 高 |
| **可预测性** | 高 | 中 |
| **适用** | 流程化任务 | 探索性任务 |

动态角色让 Supervisor Agent 根据当前状态分配新角色：

```python
def dynamic_role_assignment(state):
    if state["current_data"] == "raw_html":
        return Agent(role="html_parser")
    elif state["needs"] == "fact_check":
        return Agent(role="fact_checker")
```

## 七、协作中的涌现行为

群体智能的关键是"局部简单 + 全局涌现"：

```text
例：蚁群算法
  单只蚂蚁：随机游走 + 信息素
  整体行为：最短路径涌现
  
  LLM 多 Agent 类比：
    单个 Agent：本地规则 + LLM 决策
    整体：可能涌现出设计者未预料的协作模式
```

工程建议：

1. **小步快跑**：先验证 2 Agent 协作，扩到 5、10、20。
2. **监控涌现**：观察"集体行为是否符合预期"。
3. **约束上限**：用 max_steps、token 预算等限制意外循环。
4. **可解释性**：记录每个 Agent 的每条消息，便于回溯。

## 八、失败模式与防御

```text
1. Agent 互相等死锁
   - 防御：超时 + fallback
2. 重复劳动
   - 防御：明确的 task ownership
3. 通信洪水
   - 防御：消息路由 + 速率限制
4. 信誉污染
   - 防御：信誉系统审计
5. 整体行为偏离目标
   - 防御：定期 Supervisor 反思
```

## 九、典型框架对比

| 框架 | 协作模式 | 通信方式 | 适用规模 |
|---|---|---|---|
| **CrewAI** | 角色 + 任务链 | 上下文传递 | 2-10 Agents |
| **AutoGen** | 群聊 | 自然语言 | 2-20 Agents |
| **LangGraph** | 状态图 | 显式 message | 1-50 Agents |
| **AgentVerse** | 黑板 + 招募 | 共享存储 | 10-100 Agents |
| **ChatDev** | 软件公司模拟 | 阶段对话 | 5-20 Agents |
| **MetaGPT** | SOP 流水线 | 文档传递 | 5-30 Agents |

## 小结

多 Agent 协作的核心机制：**任务分配**（合同网、市场）、**共识达成**（投票、辩论、BFT）、**信息共享**（黑板、消息总线）、**角色分工**（团队、SOP）。LLM 时代下，简单的"自然语言对话 + 上下文传递"已能满足大多数场景。生产系统要重点防御失败模式（死锁、消息洪水、行为偏离），并保留完整的可观测性。下一篇我们将进入 **competitive-game-theory**——Agent 之间的竞争与博弈。
