# 分布式系统基础

## 一、分布式系统的本质挑战

单机系统假设组件要么"正确"要么"崩溃"。分布式系统的麻烦在于**部分失败**（partial failure）：一部分节点可能挂了、网络可能分区、消息可能延迟或丢失，而其他节点还在跑。**Leslie Lamport** 经典论文定义了分布式系统的 8 个不可能难题（"如果网络不可靠，能达成什么？"）。

几个根因：

- **网络不可靠**：包可能丢失、重复、乱序、延迟（"在数据中心内部 1 ms，跨地域 50 ms"）。
- **时钟不同步**：没有全局时钟，事件顺序只能靠逻辑时钟（Lamport 时钟、向量时钟）。
- **节点宕机**：硬件故障、OOM、内核 panic 都可能让节点失联。

ML 工程后果：

- 单卡慢会让整组 NCCL all-reduce 变慢（"straggler 问题"）。
- checkpoint 保存必须支持部分节点失败。
- 推理服务必须能优雅摘除挂掉的 worker。

## 二、CAP 定理

Eric Brewer 的 CAP 定理指出，分布式系统最多只能同时满足以下三者中的两个：

- **C（Consistency）**：所有节点看到的数据一致。
- **A（Availability）**：每个请求都能收到（非错误）响应。
- **P（Partition tolerance）**：网络分区时系统仍能运行。

由于网络分区在现实中不可避免（P 必须），现实系统只能在 C 和 A 之间权衡：

- **CP 系统**：选择一致性放弃可用性。ZooKeeper、etcd 是典型——分区期间拒绝写。
- **AP 系统**：选择可用性放弃强一致性。Cassandra、DynamoDB——分区期间继续接受写，可能有冲突。

ML 工程里的"分布式数据库"通常选 AP（如向量数据库 Weaviate、Milvus 多数模式）——服务级别一致性比绝对一致更重要。

## 三、一致性模型

按强度递减：

- **线性一致性（Linearizability）**：所有操作看起来按某个全局顺序执行，且与真实时间一致。最强但代价高。
- **顺序一致性（Sequential Consistency）**：所有操作按某个全局顺序执行，但允许与真实时间有偏差。
- **因果一致性（Causal Consistency）**：有因果关系的操作顺序一致；无因果关系的操作可任意。
- **最终一致性（Eventual Consistency）**：没有新写入后，所有副本最终会一致。DynamoDB、Cassandra 默认。

ML 工程场景：

- **分布式 checkpoint**：要线性一致——否则两个 worker 写到一半崩溃，恢复时可能拿到混合状态。
- **Embedding 服务**：最终一致即可——允许短时间不一致换高可用。
- **推理 KV-cache**：单请求内一致；多请求间不必一致（甚至不应该——会泄漏上下文）。

## 四、共识算法

当多个节点需要就"某个值"达成一致时，需要共识算法。最著名的是 **Paxos** 和它的简化版 **Raft**。

Raft 把共识拆成三个子问题：

1. **Leader 选举**：节点超时未收到心跳就变成候选，发起选举；多数票当选。
2. **日志复制**：Leader 收到写请求，把日志条目复制到多数节点后提交。
3. **安全性**：已提交的日志项永远保留。

$$
\text{多数派} = \lfloor n/2 \rfloor + 1 \quad (n \text{ 个节点})
$$

只要多数节点活着就能继续工作。3 节点容忍 1 故障，5 节点容忍 2 故障。

工程实现：

- **etcd**（Kubernetes 用）、**Consul**、**ZooKeeper**：分布式配置与 leader 选举。
- **Kafka Controller**：用 ZooKeeper/KRaft 选主。
- **Ray**（分布式 ML）：用 GCS（Global Control Store）做调度，类似 etcd。

分布式训练中：

- **NCCL**：用 ring/tree all-reduce，不需要 leader。
- **PS（Parameter Server）架构**：PS 是 leader，worker 是 follower——实际上是用客户端-服务端实现协调。
- **Megatron-LM 张量并行**：进程组 + barrier 同步。

## 五、数据复制与分片

**复制（Replication）**：同一份数据存多份。

- **同步复制**：写完所有副本才返回。一致性强、延迟高。
- **异步复制**：写主副本即返回，从副本最终追上。延迟低、可能丢数据。

**分片（Sharding）**：把数据按 key 拆到不同节点。

- **哈希分片**：`hash(key) % N`，均匀但扩缩容麻烦。
- **范围分片**：按 key 区间切分，适合范围查询但容易热点。
- **一致性哈希**：环形 hash，扩缩容只影响相邻节点，被 DynamoDB / Cassandra / Redis Cluster 广泛采用。

$$
\text{consistent\_hash}(k) = \text{hash}(k) \mod 2^{32}, \quad k \text{ 落到顺时针下一个虚拟节点}
$$

ML 工程中：

- **Embedding 表分片**：100 GB embedding 用哈希分到 8 张卡，每张只负责 12.5 GB——这就是 **ZeRO-3 / FSDP** 的核心。
- **特征存储**：Feast、Tecton 等用范围分片按时间分区。
- **向量库分片**：Milvus 把向量按 IVF 聚类分片，每个 pod 负责部分聚类。

## 六、通信模型

**同步通信**：调用方阻塞直到收到响应。RPC 是典型。

**异步通信**：消息发送后立即返回，接收方在某个时间点处理。消息队列（Kafka、RabbitMQ）是典型。

**最终一致 vs 强一致**：异步通信通常意味着最终一致。

ML 工程：

- **NCCL AllReduce**：每张卡发梯度、收所有其他卡的梯度，$\Theta(p \cdot s)$ 通信量。
- **gRPC 推理服务**：客户端发请求、阻塞等响应。流式接口可用 SSE / WebSocket。
- **消息队列做任务调度**：Celery / RQ / Ray 把推理任务分给 worker。

## 八、容错与一致性协议

**故障检测**：心跳（heartbeat）。节点超时无响应即视为故障。

**副本切换**：主从切换需在新主上能继续服务。Raft / Paxos 保证不丢已提交数据。

**幂等性**：操作执行多次和一次结果相同。ML 训练中"重复算一个 batch"是幂等的；"扣减账户余额"不是——需要状态机 + 幂等 token。

**限流与降级**：流量超过系统能力时丢弃请求（429）或返回降级结果（如 LLM 推理用小模型兜底）。

## 九、分布式训练中的"分布式系统"问题

分布式训练是分布式系统的具体场景，常见模式：

**数据并行**：每张卡有完整模型副本，看到不同 mini-batch。

$$
g_{\text{global}} = \frac{1}{p} \sum_{i=1}^{p} g_i, \quad \theta \leftarrow \theta - \eta \cdot g_{\text{global}}
$$

- AllReduce 通信量 $\Theta(p \cdot s)$，带宽效率高。
- PyTorch DDP、FSDP-zero-1/2 都属此类。

**模型并行（张量并行）**：把单层切到多卡。

- 每个 matmul 拆成两个小 matmul，通信量 $\Theta(s)$。
- Megatron-LM 用此。

**流水线并行**：不同层在不同的卡，mini-batch 切成 micro-batch 流过。

- 气泡率：$\frac{p-1}{m+p-1}$（$m$ 是 micro-batch 数）。
- PipeDream、GPipe 是典型。

**3D 并行**：数据 + 张量 + 流水线同时用，GPT-3 / LLaMA 训练标配。

**ZeRO / FSDP**：把优化器状态、梯度、参数分别切分到多卡，节省显存，但需要 all-gather 通信——本质上是"分片式数据并行"。

```python
# PyTorch FSDP 简化
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

## 十、常见故障与防御

- **Straggler**：某张卡慢拖累整个 step。监控 per-step 耗时，定期剔除慢节点。
- **NaN/Inf 传播**：一个 rank 梯度 NaN 会污染 all-reduce 结果。用 `nan_check` 或重新初始化。
- **Checkpoint 一致性**：用 barrier 保证所有 rank 都写入同一 epoch 才推进。
- **网络分区**：训练过程中若有节点失联，所有 rank 应在 timeout 内检测到并决定 abort or continue。

```python
# NCCL watchdog：超时自动 abort
torch.distributed.init_process_group(
    backend='nccl',
    timeout=timedelta(minutes=30),  # 超过 30 分钟无 collective 视为超时
)
```

## 小结

分布式大模型训练是分布式系统的"高强度版"应用：网络抖动会被放大成 step 抖动，节点故障必须快速检测与恢复，通信模式（all-reduce、all-gather、reduce-scatter）直接决定可扩展性。掌握 CAP、一致性、共识、复制/分片这些基础概念后，再去看 NCCL / FSDP / Megatron 的源码与论文，你会发现它们的设计都遵循了同样的分布式系统原则——"在不可靠的硬件上构建可靠的系统"。这是从"用 PyTorch 跑通训练"到"在 1000 张卡上稳定训练 70B 模型"的认知跨越。