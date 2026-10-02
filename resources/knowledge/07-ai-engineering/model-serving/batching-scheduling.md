# 推理调度：动态批处理、KV Cache 与请求调度策略

GPU 推理贵——一张 A100 一小时几十块。但实际上大部分团队的 GPU 利用率不到 30%：要么请求稀疏、要么请求长度差异巨大、要么 KV cache 设计糟糕。本文深入推理调度的核心机制：**continuous batching、KV cache 优化、prefix caching、speculative decoding、请求调度策略**，以及如何在 vLLM / TGI 中把它们调起来。

## 一、为什么需要调度优化

LLM 推理是**显存受限 + 计算受限**的混合瓶颈：

```text
Memory-bound:
  - 模型权重（8B FP16 ≈ 16 GB, 70B INT4 ≈ 40 GB）
  - KV cache（per-token, 随并发增长）
  - Activation

Compute-bound:
  - Prefill 阶段（长 prompt，attention O(n²)）
  - 大 batch decode 阶段
```

调度优化的目标是**最大化吞吐、压低延迟**，即在给定 GPU 上同时跑尽量多请求、且每个请求都尽快返回。

## 二、Batching 演进史

### 2.1 Static Batching（朴素方式）

把 N 个请求拼成一个 batch，等**全部完成**才返回。问题：长请求拖累短请求（head-of-line blocking），GPU 大量空转。

### 2.2 Dynamic Batching（动态拼批）

每收到 K 个请求或每 Δt 时间，把请求拼成 batch。**仍然等所有完成**，但减少了"等待集齐"的时间。

### 2.3 Continuous Batching（vLLM / TGI）

每生成 1 个 token 后，重新评估：已完成的请求退出，新请求可立即加入。这是当前 LLM 服务的事实标准。

```text
Static batching：  [====A====][B==]
Dynamic batching： [==A==][B][=C=]
Continuous：      [A][A][A][BA][BA][CB][CB][CB][C]    灵活交错
```

**效果**：吞吐提升 2-10×（vLLM 论文报告 23×）。

## 三、KV Cache 与 PagedAttention

### 3.1 为什么 KV Cache 是瓶颈

LLM 解码时每个 token 都要保存历史的 K、V 矩阵：

$$
\text{KV cache per token} = 2 \cdot \text{num\_layers} \cdot \text{num\_kv\_heads} \cdot \text{head\_dim} \cdot \text{bytes}
$$

对 LLaMA-3-8B（32 层、8 KV 头、head_dim=128、BF16）：

```python
2 * 32 * 8 * 128 * 2 = 131_072 bytes/token ≈ 128 KB/token
```

一个 4K context 的请求要 512 MB KV cache，1024 并发就是 512 GB——直接 OOM。

### 3.2 PagedAttention（vLLM 2023）

传统做法：给每个请求**预分配**一个最大长度的连续 KV 块。浪费严重——预分配 4K 但只用了 100 token，等于浪费 97% 的显存。

PagedAttention 把 KV cache 切成**固定大小页**（类比 OS 虚拟内存分页）：

```text
请求 A 持有页: [P3, P7, P12, P1]    ← 不连续，但有页表
请求 B 持有页: [P2, P5, P9]
请求 C 持有页: [P3, P8]              ← 共享页（A 和 C 共享 prompt 前缀）
```

好处：

1. **显存利用率 90%+**（接近理论上限）。
2. **prefix sharing**：相同前缀的请求可共享 KV 页。
3. **无碎片**：固定大小页避免外部碎片。

vLLM 默认开启。

### 3.3 GQA / MQA（Grouped/Multi-Query Attention）

进一步减小 KV cache：让所有 Q 头共用 KV 头。

```text
MHA: 32 Q heads, 32 KV heads      → 32× 显存
MQA: 32 Q heads,  1 KV head       →  1× 显存
GQA: 32 Q heads,  8 KV heads      →  8× 显存
```

LLaMA-2-70B、Mixtral、Llama-3 都用 GQA。

## 四、Prefix Caching：性价比之王

很多场景有大量重复前缀：

- **System prompt**：每条请求都带同样的 500-token system。
- **多轮对话**：上轮所有 token 都要重新计算 KV。
- **RAG 检索结果**：同一篇文档可能被多次检索。

vLLM 的 prefix cache：

```bash
vllm serve ... --enable-prefix-caching
```

实现：维护一个**全局 radix tree**，每个节点的 KV cache 存一次。新请求的 prompt 跟树做最长前缀匹配，命中的部分直接复用 KV。

命中率参考：

| 场景 | 命中率 | 节省 |
|---|---|---|
| 高重复 system prompt | 60-90% | 30-50% token 成本 |
| 多轮 agent | 40-70% | 20-40% |
| 独立用户问题 | <10% | 价值有限 |

监控方法：

```bash
curl http://localhost:8000/metrics | grep prefix_cache
# vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total
```

## 五、Chunked Prefill

长 prompt（10K+ token）做 prefill 时，单次 attention 矩阵会爆显存，且会阻塞其他请求的 decode step。

**Chunked prefill**：把长 prompt 切成小块，与 decode 请求混合跑。

```bash
vllm serve ... --enable-chunked-prefill --max-num-batched-tokens 8192
```

效果：长请求不阻塞短请求，P99 延迟更稳定。

## 六、Speculative Decoding

小模型先"草拟"K 个 token，大模型一次验证。理论无损加速。

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

对**输出短**或**模式化**的请求（分类、JSON 抽取）效果特别好。

## 七、请求调度策略

### 7.1 FCFS（先来先服务）

简单，但长请求拖累短请求。

### 7.2 Priority 调度

按租户等级 / token 配额分配优先级。

```python
# vLLM 暂不支持完整 priority，但可用 chunked prefill + 多实例分租
# SGLang 原生支持 priority queue
```

### 7.3 Preemption（抢占）

高优先级请求到达时，把低优先级请求 swap 到 CPU。

```bash
vllm serve ... --swap-space 4   # 4 GB CPU swap
```

适用场景：付费用户优先。

### 7.4 Deadline-based（截止时间）

SLO 要求 P99 < 500ms。调度器按剩余时间排序，先服务快超时的请求。

```python
# 概念示意
requests.sort(key=lambda r: r.deadline - now)
```

## 八、生产参数调优清单

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \    # 显存利用率上限
    --max-model-len 32768 \             # 最大 context
    --max-num-seqs 256 \                # 最大并发请求数
    --max-num-batched-tokens 8192 \     # chunked prefill 阈值
    --enable-prefix-caching \           # prefix cache
    --enable-chunked-prefill \          # chunked prefill
    --swap-space 4 \                    # CPU swap (GB)
    --block-size 16 \                   # KV cache 页大小
    --served-model-name llama3-8b
```

## 九、监控关键指标

| 指标 | 含义 | 目标 |
|---|---|---|
| **TTFT** | Time To First Token | < 200 ms (1k prompt) |
| **TPOT** | Time Per Output Token | < 50 ms |
| **Throughput** | aggregate tok/s | 越高越好 |
| **GPU util** | SM 占用率 | > 80% |
| **KV cache usage** | 显存 KV cache 占比 | < 90% |
| **Prefix cache hit rate** | 命中比例 | > 50%（按场景） |
| **Queue wait time** | 请求排队时间 | < 100 ms |

## 小结

推理调度是 LLM 服务的"性能心脏"。**Continuous batching + PagedAttention + prefix caching + chunked prefill** 是当前的事实标准组合，能把单卡吞吐推到极限。生产环境要按业务场景（长 prompt / 短 prompt / 多轮 / RAG）调对应参数，并用 Prometheus 全方位监控。下一篇我们将进入 **监控**——上线之后如何发现漂移、异常、性能退化。
