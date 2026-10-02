# Speculative Decoding 与推理加速

自回归 LLM 解码最大的浪费是：**每一步只生成 1 个 token，却要做一次完整的前向**。一个 7B 模型跑 100 个 token，就要跑 100 次 forward。Speculative Decoding 的核心想法是：让一个**便宜的小模型**先"草拟"几个候选 token，再让**目标大模型一次性验证**，结果上等价、但算力放大。

## 一、串行瓶颈再回顾

Decode 阶段在 GPU 上是 memory-bound。LLaMA-3-8B 在 A100 上的 ~30 ms/token 主要花在"从显存把权重搬到 SM"上，**而非"真算"**。换句话说，GPU 算力在大部分时间空转。

朴素的 decode timeline：

```text
step 1:  prompt + 0    → 1
step 2:  prompt + 0,1 → 2
step 3:  prompt + 0,1,2 → 3
...

每步加载 16 GB 权重，只产出 1 个 token。
```

Speculative Decoding 试图打破"一步一个"的串行假设。

## 二、核心思想：小草稿 + 大验证

设大模型 $M_q$，小模型 $M_p$。每个 round：

1. 用 $M_p$ 自回归生成 $K$ 个草稿 token：$\tilde{x}_1, \dots, \tilde{x}_K$。
2. 把这 $K$ 个 token **一次性拼成一条新 prompt**，喂给 $M_q$，只跑 **1 次** forward，得到 $K+1$ 个位置的 logits。
3. 按概率**逐个**接受 / 拒绝草稿 token。

接受规则（Leviathan et al., 2023）：

$$
\text{接受 } \tilde{x}_i \iff r < \min\!\left(1,\ \frac{q(\tilde{x}_i \mid \cdot)}{p(\tilde{x}_i \mid \cdot)}\right)
$$

$r \sim U[0,1]$。拒绝时，从修正后的分布 $q'(\cdot) = \mathrm{norm}(\max(0, q - p))$ 中重采样一个 token 作为新草稿起点。

数学保证：接受-拒绝 scheme 下，输出分布与只用 $M_q$ 采样**严格等价**，没有任何精度损失。

## 三、伪代码

```python
def speculative_step(model_q, draft_q, context, k=4):
    # 1. 小模型草拟 k 个 token
    draft = []
    p_dist = []
    x = context[:, -1:]
    for _ in range(k):
        logits = draft_q(x)             # 小模型 forward
        p = softmax(logits[:, -1])
        x_i = sample(p)
        draft.append(x_i)
        p_dist.append(p)
        x = torch.cat([x, x_i], dim=1)

    # 2. 大模型一次 forward，验证 k+1 个位置
    big_input = torch.cat([context, *draft], dim=1)
    q_logits = model_q(big_input)[:, -k-1:]   # 取最后 k+1 个位置
    q_dist = [softmax(l) for l in q_logits]

    # 3. 逐 token 接受/拒绝
    accepted = []
    for i in range(k):
        r = torch.rand(1).item()
        ratio = q_dist[i][draft[i]] / (p_dist[i][draft[i]] + 1e-8)
        if r < ratio.item():
            accepted.append(draft[i])
        else:
            # 重采样：q'(x) = norm(max(0, q - p))
            diff = torch.clamp(q_dist[i] - p_dist[i], min=0)
            diff = diff / diff.sum()
            x_new = torch.multinomial(diff, 1)
            accepted.append(x_new)
            break
    else:
        # 全部接受，额外加一个 bonus token
        bonus = torch.multinomial(q_dist[-1], 1)
        accepted.append(bonus)

    return torch.cat(accepted, dim=1)
```

## 四、加速比分析

加速比近似为：

$$
S \approx \frac{K + 1}{K \cdot c_p / c_q + 1}
$$

其中 $c_p / c_q$ 是草稿模型 / 大模型的单 token 耗时比。

直觉：

- 草稿模型相对越快 → 加速越大。
- $K$ 越大 → 大模型被复用越充分；但 $K$ 太大时草稿出错率升高，**有效接受长度** $E[\text{accepted}]$ 会饱和。
- 经验上 **EAGLE-2** 在 70B target / 1B draft 上能拿到 ~2.8× 加速，文本质量与原模型不可区分。

| 配置 | 接受长度 | 加速比 |
|---|---|---|
| 70B target + 1B draft, K=5 | ~3.0 | 2.5-2.8× |
| 13B target + 1B draft, K=5 | ~2.2 | 1.8-2.0× |
| 8B target + 0.5B draft, K=4 | ~2.5 | 2.0-2.3× |

## 五、衍生工作

- **Medusa**：把草稿模型换成一组并行的 MLP 头（"多头预测器"），免去单独跑小模型的开销。
- **EAGLE / EAGLE-2**：在特征层做线性近似 + 动态树状验证（Token Tree），通常比 vanilla Speculative Decoding 再快 1.3-1.8×。
- **Lookahead Decoding**：用 Jacobi 迭代的思路在生成的同时并行试探后续 token，不依赖外部草稿模型。
- **Self-Speculative**（如 Medusa / EAGLE 自家做法）：草稿与大模型同源，进一步压低工程门槛。

## 六、与其他优化的兼容性

Speculative Decoding 是**正交**的：

- 与 **PagedAttention / Continuous batching** 兼容（vLLM 已原生支持）。
- 与 **INT4/AWQ 量化**兼容：草稿与目标都可分别量化。
- 与 **Prefix caching** 兼容：草稿阶段不需要 KV cache 共享。

vLLM 启动示例：

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5 \
    --tensor-parallel-size 4
```

## 小结

Speculative Decoding 用"小模型并行草拟 + 大模型批量确认"突破了自回归"一步一 token"的硬约束，并且在数学上**严格保持输出分布**。这是当前大模型低延迟 GPU 服务的事实加速手段之一，能稳定拿到 2-3× 的端到端提速。下一篇我们将进入**推理服务化**的实战——怎么用 vLLM / TGI / TensorRT-LLM 在生产中把上述所有优化整合起来。