# LLM 推理基础：从 latency 到 throughput

LLM 推理是当前 AI 工程化的"主战场"。一个 7B 模型在 A100 上每秒能吐多少字、能撑多少并发用户，决定了产品能不能上生产。本文从最朴素的自回归生成讲起，剖析 prefill/decode 两阶段、显存/算力的不同瓶颈，以及 TTFT、TPOT、throughput 这一组核心指标。

## 一、自回归推理是什么

LLM 的本质是"给定上文，预测下一个 token"。因此生成一段长度为 $n$ 的文本，需要**串行地**跑 $n$ 次前向：

$$
y_t = \arg\max P(\cdot \mid x_1, x_2, \dots, x_t, y_1, \dots, y_{t-1})
$$

每一次解码都依赖上一步的结果，不能并行。这种"串行依赖"是后续所有优化（KV cache、speculative decoding、continuous batching）的根源。

## 二、Prefill vs Decode：两个截然不同的阶段

一次请求通常被切成两个阶段：

| 阶段 | 输入 | 计算特性 | 瓶颈 |
|---|---|---|---|
| **Prefill** | 整段 prompt（一次性）| 计算密集，attention 矩阵很大 | compute-bound，受限于 FLOPS |
| **Decode** | 每次只新增 1 个 token | 读多写少，batch 内极短序列 | memory-bound，受限于显存带宽 |

一个直观的数字：在一张 A100（80GB）上跑 LLaMA-3-8B（FP16），
- Prefill：~1500 tokens/s（受 FLOPS 限制）
- Decode：~120 tokens/s（受 HBM 带宽 ~2 TB/s 限制）

也就是说"生成"速度比"读 prompt"慢一个数量级。这也是为什么 streaming 输出比等全文生成感觉快很多。

## 三、最朴素的推理代码

用 HuggingFace Transformers 写一个最小可用的推理：

```python
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)
model.eval()

prompt = "解释一下什么是 KV cache："
inputs = tok(prompt, return_tensors="pt").to("cuda")

t0 = time.perf_counter()
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,   # 启用 KV cache
    )
dt = time.perf_counter() - t0

new = out.shape[-1] - inputs["input_ids"].shape[-1]
print(f"生成 {new} tokens, 用时 {dt:.2f}s, "
      f"throughput = {new/dt:.1f} tok/s")
```

`use_cache=True` 是关键——它会把每一步的 K/V 张量张量缓存起来，避免每一步重算整个 prompt 的 attention。

## 四、TTFT、TPOT 与 TPS

工业上有三个最常被提到的延迟/吞吐指标：

- **TTFT (Time To First Token)**：从请求到第一个字出现。直接由 Prefill 阶段决定。对聊天体感影响最大。
- **TPOT (Time Per Output Token)**：平均每个新 token 的生成时间（不含第一个）。由 Decode 阶段决定，决定流式的丝滑度。
- **TPS / Throughput**：系统级别的每秒 token 数（往往是多个请求合并后的）。

粗略估算 LLaMA-3-8B FP16 在 A100 上的可达水平：

```text
TTFT (1k tokens prompt)  ≈  0.7 s
TPOT                    ≈  25-40 ms
单请求 throughput       ≈  25-40 tok/s
32 并发 aggregate throughput  ≈  800-1200 tok/s
```

经验法则**：batch 越大，aggregate throughput 越高，但单请求 TPOT 也会变长——这是后端调度要做的核心权衡。

## 五、内存瓶颈的真实来源

LLM 推理的显存占用主要有三块：

1. **模型权重**：例如 FP16 下 8B 模型占 ~16 GB。
2. **KV cache**：随序列长度线性增长，$O(n \cdot d \cdot L \cdot 2)$（K 和 V 两个张量），后续文章会详细算。
3. **激活值**：Prefill 阶段与 batch 相关；Decode 阶段很小。

**经验法则**：当 batch + 序列长度变大，KV cache 很快会超过权重本身——这也是 vLLM、PagedAttention 这类系统的根本动机。

## 小结

LLM 推理的核心矛盾是：自回归的**串行依赖**让它天然"慢"，而 Prefill / Decode 两阶段的不同特性让优化策略必须分而治之。下一篇我们会深入 KV cache 的显存计算与 PagedAttention——这是 vLLM 改变 LLM 服务格局的关键技术。