# 推理服务化：vLLM、TGI 与生产部署

把模型权重压到 INT4、把 KV cache 装进分页显存、把自回归变成可草拟验证——这些都只是"引擎"。真正的生产系统还要管请求调度、动态批处理、流量突峰、可观测性。本文对比主流推理服务框架，给出 vLLM 的端到端部署 + 监控示例。

## 二、主流框架对比

| 框架 | 维护方 | 核心特性 | 适用场景 |
|---|---|---|---|
| **vLLM** | UC Berkeley / Anyscale | PagedAttention、continuous batching、prefix caching、speculative decoding | 通用 GPU 服务，OpenAI 兼容 |
| **TGI** | HuggingFace | Rust 核心，token 级 streaming，动态 batching | HF 生态、HuggingFace Hub 模型 |
| **TensorRT-LLM** | NVIDIA | 极致 kernel 优化、FP8/INT4、in-flight batching | NVIDIA GPU，追求极致单卡吞吐 |
| **LMDeploy** | 商汤 | Turbomind 引擎、动态分割、量化集成 | 中文场景、移动端 |
| **SGLang** | LMSYS | RadixAttention、structured output | agent/多轮 API 场景 |
| **llama.cpp / Ollama** | 社区 / Ollama | CPU + Apple Silicon，纯本地 | 个人/边缘部署 |

经验法则：

- **多卡 GPU 服务** → vLLM（生态最大、文档最好）
- **极致单卡吞吐 + 不在乎部署麻烦** → TensorRT-LLM
- **CPU / Apple Silicon** → llama.cpp / Ollama
- **HF 模型快速上线** → TGI

## 二、动态批处理与请求调度

不同框架对 batching 的实现名不一样，本质思想一致：

1. **请求进入 waiting queue**。
2. **调度器**每 N ms 拉一批请求进入 running（continuous batching：已完成请求随时退出，新请求随时加入）。
3. **GPU kernel** 一次性处理这批请求的 decode step。
4. **token-level streaming**：每个请求每生成一个 token 立即 push 给客户端。

调度策略一般支持：

- FCFS（先来先服务）：公平但长 prompt 拖累整批。
- **Preemptive / priority**：高优先级请求可抢占（vLLM 当前是 swap/preempt，复杂策略仍在演进）。
- **Chunked prefill**：把超长 prompt 切成 chunk 喂进去，避免"老 prompt 整段卡住新请求"。

## 三、vLLM 启动 OpenAI 兼容服务

```bash
# 安装：pip install vllm
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5 \
    --swap-space 4 \
    --served-model-name llama3-8b
```

客户端像 OpenAI 一样调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# 非流式
resp = client.chat.completions.create(
    model="llama3-8b",
    messages=[{"role": "user", "content": "用 100 字介绍 vLLM"}],
    max_tokens=200,
    temperature=0.7,
)
print(resp.choices[0].message.content)

# 流式（SSE）
stream = client.chat.completions.create(
    model="llama3-8b",
    messages=[{"role": "user", "content": "写一首五言绝句"}],
    max_tokens=200,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

## 四、张量并行与多卡部署

单卡装不下时，用 `--tensor-parallel-size N`（vLLM）或 `tp=N`（TGI）：

```bash
# 70B 模型在 4× A100 80G
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8192
```

也可以叠加流水线并行 `--pipeline-parallel-size N` 或量化到 INT4 把单卡需求压下来。

## 五、Prefix Caching 的命中率

在 RAG、多轮对话、长 system prompt 的场景下，prefix cache 是性价比最高的优化：

```bash
vllm serve ... --enable-prefix-caching
```

命中率与流量模式强相关：

```text
高重复 system prompt    命中率 60-90%,  多探节省 30-50% token 成本
多轮 agent 场景        命中率 40-70%
随机独立用户问题       命中率 <10%,  价值有限
```

监控命中率的方法：

```bash
curl http://localhost:8000/metrics | grep prefix_cache
# vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total
```

## 六、生产级关注点

1. **健康检查**：暴露 `/health` 返回 200；`/v1/models` 列出服务中的模型。
2. **Prometheus 指标**：vLLM、TGI、TensorRT-LLM 都自带 `/metrics`。关键指标：

    ```text
    vllm:request_success_total
    vllm:prompt_tokens_total
    vllm:generation_tokens_total
    vllm:e2e_request_latency_seconds
    vllm:time_to_first_token_seconds
    vllm:gpu_cache_usage_perc
    vllm:prefix_cache_hits_total
    ```

3. **弹性**：Kubernetes 上跑一个 Deployment + HPA，自定义 metric 用 `vllm:num_requests_waiting`。
4. **请求级隔离**：用 API 网关按租户限流，避免"某个用户发了 50k token 的巨 prompt 把所有人拖死"。
5. **冷启动**：模型加载需要几十秒到几分钟（70B 4-bit 约 90s）；用 sidecar 预热或常驻避免冷启。

## 八、典型指标参考值

LLaMA-3-8B + A100 80G，TP=1，启用 prefix cache + chunked prefill：

```text
TTFT (1k prompt)        ≈ 80-150 ms
TPOT                   ≈ 25-40 ms
单请求 throughput       ≈ 30-40 tok/s
32 并发 aggregate       ≈ 1000-1400 tok/s
64 并发 aggregate       ≈ 1600-2000 tok/s
显存占用               ≈ 28 GB（权重 8 + KV ~8 + overhead）
```

## 小结

推理服务化是"模型权重"和"业务流量"之间的中间件层。**vLLM** 因其 PagedAttention、continuous batching、prefix caching、speculative decoding、OpenAI API 等完整能力，已经成为多数团队的事实默认。在它之上再叠 HPA 量化、spec speculative、prefix cache、监控告警，才能把"模型能力"变成"稳定的服务能力"。本系列至此把 LLM 推理优化的**存储（PagedAttention）+ 算法（量化 / 投机解码）+ 调度（vLLM / TGI）**三个维度都串起来了。