# GPT vs LLaMA 架构：现代 LLM 的四大工程支柱

前三篇 [attention-mechanism-explained.md](attention-mechanism-explained.md)、[transformer-architecture.md](transformer-architecture.md)、[gpt-and-bert-evolution.md](gpt-and-bert-evolution.md) 解决了"注意力是什么、Transformer 长什么样、为什么 decoder-only 成为主流"。本文进入 decoder-only **内部**，对比 GPT-3（2020）与 LLaMA 系列（2023+）的工程细节：位置编码、Normalization、激活函数、KV cache 优化。把这些"看起来微小"的差异加起来，就是开源大模型追上闭源大模型的关键。本文承接前三篇，最后给出**现代 LLM 的四大支柱**（RoPE + RMSNorm + SwiGLU + GQA）。

## 一、同一范式，不同细节

GPT-3 与 LLaMA 看起来"很像"——都是 decoder-only Transformer + causal LM 预训练。但打开工程细节，差异巨大：

```text
GPT-3 (2020)             LLaMA-2 (2023)
─────────────────        ─────────────────
绝对位置 embedding       RoPE（旋转位置）
LayerNorm                RMSNorm（Pre-LN）
GeLU                     SwiGLU
MHA（每头独立 K/V）      GQA（K/V 分组共享）
上下文 2K                上下文 4K
300B tokens              2T tokens
```

逐项展开。

## 二、位置编码：Absolute → RoPE → 长度外推

### 2.1 GPT-3：绝对位置 Embedding

```python
self.pos_emb = nn.Embedding(max_len, d_model)   # 每一行是一个位置向量
x = self.tok_emb(tokens) + self.pos_emb(positions)
```

**问题**：位置 0..2047 各自学一个向量，没见过的长度（> 2048）直接乱掉——**长度外推能力几乎为零**。

### 2.2 RoPE：通过旋转注入相对位置

LLaMA 系列使用 RoPE，把 $q, k$ 投影到 2D 平面上按位置旋转：

$$
\text{RoPE}(x,\, m) = x \cdot e^{i\, m\theta_l}
$$

具体到维度对 $(x_{2l}, x_{2l+1})$：

$$
\begin{pmatrix} x'_{2l} \\ x'_{2l+1} \end{pmatrix}
=
\begin{pmatrix} \cos(m\theta_l) & -\sin(m\theta_l) \\ \sin(m\theta_l) & \cos(m\theta_l) \end{pmatrix}
\begin{pmatrix} x_{2l} \\ x_{2l+1} \end{pmatrix}
$$

不同维度用不同频率 $\theta_l = 10000^{-2l/d}$。

**关键性质**：attention 内积只依赖于相对距离 $(m - n)$：

$$
\langle \text{RoPE}(q, m),\ \text{RoPE}(k, n) \rangle
= \text{Re}\!\left[ q\, k^*\, e^{i(m - n)\theta} \right]
$$

证明思路：旋转角度之差进入复指数 → 只剩 $(m - n)$。

**为什么外推更好**：模型从训练长度内学到的"相对距离编码模式"在更长序列上仍然成立——只是被外推到没见过的距离，所以 LLaMA-2 训练 4K 后能勉强处理 8K-16K，再加 YaRN 等插值就能到 100K+。

### 2.3 长度扩展技术（简表）

| 技术 | 思路 | 代表 |
|---|---|---|
| RoPE 插值 (YaRN, NTK-aware) | 把高频 $\theta_l$ 重新分配 | LLaMA-2 长上下文版 |
| ALiBi | 显式加 $-\alpha \cdot |i - j|$ 偏置 | BLOOM |
| Linear Scaling | 把 RoPE 频率除以 $s$ | LLaMA 早期长上下文 |
| Position Interpolation | 在 RoPE 上做线性插值 | LLaMA-2 32K |

## 三、Normalization：LayerNorm → RMSNorm

### 3.1 LayerNorm 公式

$$
\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中 $\mu, \sigma^2$ 沿 $d$ 维算。

### 3.2 RMSNorm：去掉均值中心化

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
$$

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x) + \epsilon}
$$

**关键差异**：不做均值减法，只缩放方差，不做中心化。

### 3.3 为什么有效

- **可学习重参数** $\gamma$ 已经能恢复"位置偏移"——网络自己会学。
- 实测：质量几乎不下降（perplexity 差距在 0.01 量级），**算力节省约 7-10%**（少一次减法 + 减少依赖）。
- 在 Pre-LN + 深层 LLM 场景特别合适：残差流的均值本身就是稳定的，多余的中心化只是浪费。

### 3.4 总结

| 维度 | LayerNorm | RMSNorm |
|---|---|---|
| 计算 | 减均值 + 缩放方差 | 只缩放 |
| 速度 | 基线 | 快 7-10% |
| 质量 | 基线 | 几乎一致 |
| 代表 | GPT-3、BERT、T5 | **LLaMA、Mistral、Qwen、Gemma** |

## 四、激活函数：GeLU → SwiGLU

### 4.1 GeLU（GPT-3 用）

$$
\text{GeLU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\!\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

是 ReLU 的平滑近似，在 $x=0$ 附近不"硬截断"，保留小幅梯度。

### 4.2 SwiGLU（LLaMA 用）

SwiGLU 在 FFN 里加了一个**门控分支**——两个并行投影，一个过 Swish、一个不过，最后按元素乘：

$$
\text{SwiGLU}(x) = \text{Swish}_\beta(x W_1) \odot (x W_3)
$$

$$
\text{FFN}_{\text{SwiGLU}}(x) = \bigl(\text{Swish}_\beta(x W_1) \odot x W_3\bigr)\, W_2
$$

```text
        x
        │
   ┌────┼────┐
   ▼         ▼
  W₁         W₃
   │         │
 Swish       │
   │         │
   └────⊗────┘
        │
        ▼
        W₂
        │
        ▼
      output
```

- $W_1, W_3$ 各自 $d \to d_{ff}$，$W_2$ 把 $d_{ff}$ 投回 $d$。
- Swish$(x) = x \cdot \sigma(\beta x)$（$\beta$ 可学习或固定 1）。
- **总参数量** $W_1 + W_2 + W_3 = 3 d d_{ff}$，所以 LLaMA 调整 $d_{ff}$ 来保持总参数与 GeLU 版本一致（$2 d d_{ff}$ → $d_{ff} \to \tfrac{2}{3}$ 缩放，或保持 $d_{ff}$ 但参数变多）。

**经验收益**：固定参数量下，SwiGLU 比 GeLU 在 LLM 上带来约 **1 个 perplexity 点的改善**——在 70B 量级上是非常显著的差距。

| 激活 | 公式 | 代表模型 |
|---|---|---|
| ReLU | $\max(0, x)$ | 原始 Transformer |
| GeLU | $x \Phi(x)$ | GPT-2/3、BERT、T5 |
| SwiGLU | $\text{Swish}(xW_1) \odot xW_3$ | **LLaMA、Mistral、PaLM** |
| GeGLU | $\text{GeLU}(xW_1) \odot xW_3$ | Gemma 部分版本 |

## 五、Attention 变体：MHA → MQA → GQA

KV cache 占据 LLM 推理的主要显存（详见第七节）。减少 KV cache 是性能优化最重要的方向之一。

### 5.1 MHA（Multi-Head Attention）

```text
Q: h 个独立 head, 每个 d_k 维        → 共 h·d_k = d
K: h 个独立 head                     → 同上
V: h 个独立 head                     → 同上
```

KV cache 大小：每层 $2 \cdot h \cdot d_k \cdot T$。

### 5.2 MQA（Multi-Query Attention）

```text
Q: h 个独立 head                    ──► 计算独立
K, V: 共享 1 组                       ──► 所有 head 用同一份 K/V
```

KV cache 立刻减少为 $1/h$（LLaMA-2 70B 是 64×，节省极大）。代价：质量下降 0.5-1 perplexity 点。

### 5.3 GQA（Grouped-Query Attention）

中间方案：**$h_q$ 个 Q head 共享 $h_{kv}$ 组 K/V**：

```text
h_q = 64, h_kv = 8:   Q head 0-7 共享 K/V_1, head 8-15 共享 K/V_2, ...

LLaMA-2 70B:   h_q = 64, h_kv = 8
LLaMA-3 70B:   h_q = 64, h_kv = 8
Mistral:       h_q = 32, h_kv = 8
```

数学上：

$$
\text{head}_i^{(q)} = \text{softmax}\!\left(\frac{q_i \, k_{\lfloor i/h_q \cdot h_{kv} \rfloor}^\top}{\sqrt{d_k}}\right) v_{\lfloor i/h_q \cdot h_{kv} \rfloor}
$$

KV cache 大小：每层 $2 \cdot h_{kv} \cdot d_k \cdot T$。当 $h_{kv} \ll h_q$ 时，几乎接近 MQA 的内存，但保留 MHA 的质量。

### 5.4 三个对比

| 形式 | Q heads | K/V heads | KV cache 占比 | 质量 | 使用 |
|---|---|---|---|---|---|
| MHA | $h$ | $h$ | 100% | 基线 | GPT-3、原始 Transformer |
| MQA | $h$ | 1 | $1/h$ | -0.5~-1 ppl | PaLM、Falcon |
| **GQA** | $h$ | $g \ll h$ | $g/h$ | 与 MHA 几乎一致 | **LLaMA-2/3、Mistral、Qwen** |

## 六、上下文与数据规模演进

```text
GPT-3 (2020):         2K 上下文,    300B tokens,  175B 参数
LLaMA-1 (2023):       2K 上下文,    1.4T tokens,  7B/13B/65B
LLaMA-2 (2023):       4K 上下文,    2T tokens,    7B/13B/70B
LLaMA-3 (2024):       8K 训练 → 128K 部署,  15T tokens, 8B/70B/405B
GPT-4 (2023):         8K → 128K 上下文 (估计)
Claude 3 (2024):      200K → 1M 上下文
Gemini 1.5 (2024):    1M-10M 上下文
```

数据规模从 300B 到 15T tokens（约 50×），上下文从 2K 到 1M（500×）——而这一切都建立在同一个 decoder-only Transformer 之上。

## 七、KV Cache 深度剖析

### 7.1 为什么需要 KV Cache

自回归生成时，第 $t$ 步的 attention 需要所有历史的 $K_{\leq t}, V_{\leq t}$。**朴素实现**每步重算前面所有 token 的 $K, V$，复杂度 $O(T^2 d)$。**KV cache** 只算当前 token 的 $K, V$，旧的从缓存读：

```text
朴素:    第 1 步算 1 个, 第 2 步算 2 个, ..., 第 T 步算 T 个  → O(T²)
KV cache: 第 t 步只算 1 个新 K, V, 旧的从缓存读              → O(T)
```

### 7.2 KV Cache 显存公式

$$
\text{KV size} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot \text{seq\_len} \cdot \text{bytes\_per\_elem}
$$

每个 token 每层需要存 K、V 两份张量。

### 7.3 具体数字（LLaMA-2 70B, fp16, 2 bytes/elem）

```text
参数:
  n_layers = 80
  n_kv     = 8   (GQA)
  d_head   = 128
  bytes    = 2   (fp16)

KV 大小 = 2 × 80 × 8 × 128 × seq_len × 2 bytes
       = 327,680 × seq_len bytes
       ≈ 0.32 MB × seq_len

4K 上下文:    0.32 × 4096  ≈ 1.3 GB
32K 上下文:   0.32 × 32768 ≈ 10.5 GB
128K 上下文:  0.32 × 131072 ≈ 42 GB   ← 已超过模型权重本身 (140 GB)
1M 上下文:    0.32 × 1048576 ≈ 335 GB  ← 单卡绝对放不下
```

**核心问题**：超长上下文时，KV cache 比模型权重还大——这正是 GQA、量化、PagedAttention 等技术发力的地方。

### 7.4 KV 优化技术

| 技术 | 思路 | 收益 |
|---|---|---|
| **GQA / MQA** | 多 Q 头共享 K/V | KV 大小降 4-32× |
| **KV 量化 (8-bit / 4-bit)** | 把 K/V 压到 int8/int4 | 2-4× 节省 |
| **PagedAttention (vLLM)** | 像 OS 那样分页管理 KV | 提升吞吐 5-24× |
| **Sliding Window Attention** | 只保留最近 $W$ 个 token 的 KV | 显存固定 |
| **FlashAttention** | 融合 kernel，减少 IO | 速度 2-4× |
| **Multi-Token Prediction / Speculative** | 一次生成多个 token | 延迟降低 |

### 7.5 朴素实现 vs KV Cache（PyTorch 简版）

```python
# 朴素：每步把所有历史重新送进 attention
def naive_step(model, tokens_so_far):
    return model(tokens_so_far)  # O(T²)

# KV cache：每步只送新 token，旧的 K/V 从 cache 读
def kv_step(model, new_token, kv_cache):
    q = model.W_q(new_token)                  # (B, 1, d)
    k_new = model.W_k(new_token)              # (B, 1, d)
    v_new = model.W_v(new_token)              # (B, 1, d)
    kv_cache.k = torch.cat([kv_cache.k, k_new], dim=1)
    kv_cache.v = torch.cat([kv_cache.v, v_new], dim=1)
    attn = q @ kv_cache.k.transpose(-2, -1) / sqrt(d_k)
    attn = softmax(attn, dim=-1)
    return attn @ kv_cache.v
```

PyTorch 的 `nn.MultiheadAttention` 默认**不**管理 KV cache；生产环境用 `transformers` 的 `generate(model, use_cache=True)` 或 vLLM/TensorRT-LLM。

## 八、生产 Serving 栈

所有主流 LLM 推理框架都建立在"GQA + fp16 + PagedAttention + FlashAttention"这四个支柱上：

| 框架 | 关键特性 |
|---|---|
| **vLLM** | PagedAttention，吞吐极高 |
| **TGI (HuggingFace)** | 多 GPU + 连续 batching |
| **TensorRT-LLM** | NVIDIA 极致 kernel 优化 |
| **llama.cpp** | CPU / 边缘部署，量化支持 |
| **SGLang** | RadixAttention + structured generation |
| **DeepSpeed-MII** | 微软的 zero-overhead serving |

它们都支持 GQA、fp16/bf16、FlashAttention、量化（INT8/INT4/AWQ/GPTQ），本质上是同一个公式的不同工程实现。

## 九、完整对比：GPT-3 / LLaMA-1 / LLaMA-2 / LLaMA-3

| 维度 | GPT-3 (2020) | LLaMA-1 (2023) | LLaMA-2 (2023) | LLaMA-3 (2024) |
|---|---|---|---|---|
| 位置编码 | 绝对 embedding | RoPE | RoPE | RoPE |
| Normalization | LayerNorm (Pre-LN) | RMSNorm (Pre-LN) | RMSNorm (Pre-LN) | RMSNorm (Pre-LN) |
| 激活 | GeLU | SwiGLU | SwiGLU | SwiGLU |
| Attention | MHA | MHA | MHA (7B/13B) / GQA (70B) | GQA |
| FFN expand | 4× | ~2.7× (3 proj) | ~2.7× | ~2.7× |
| Token 数 | 300B | 1.4T | 2T | 15T |
| 上下文 | 2K | 2K | 4K | 8K 训练 / 128K 部署 |
| Vocab | ~50K (BPE) | 32K (SentencePiece) | 32K (SP) | 128K (SP) |
| 最大模型 | 175B | 65B | 70B | 405B |
| 开源 | 否 | 是 | 是（受限） | 是 |

把每一行打开看，**LLaMA 系列在所有细节上都做了优化**，合起来就是"开源追平闭源"。

## 十、现代 LLM 的四大支柱

最后给一个口诀：**RoPE + RMSNorm + SwiGLU + GQA**——这四件事几乎定义了"2024 年的现代 LLM"：

```text
   ┌─────────────────────────────────────────────────────┐
   │             现代 LLM 的四大支柱                       │
   ├─────────────────────────────────────────────────────┤
   │                                                     │
   │  1. RoPE      长度外推友好，训练短、推理长           │
   │                                                     │
   │  2. RMSNorm   去均值中心化，快 7-10%，质量不变        │
   │                                                     │
   │  3. SwiGLU    门控 FFN，perplexity -1 点             │
   │                                                     │
   │  4. GQA       共享 K/V，KV cache 降 4-8×            │
   │                                                     │
   └─────────────────────────────────────────────────────┘
```

几乎所有 2023 年之后发布的主流开源 LLM（LLaMA-2/3、Mistral、Qwen、Yi、DeepSeek、Gemma）都同时采用这四项——把它们看作事实标准即可。

## 小结

GPT-3 与 LLaMA 系列共享同一个 decoder-only Transformer 骨架，但工程细节差异显著。LLaMA 用 RoPE 替代绝对位置编码获得长度外推能力，用 RMSNorm 替代 LayerNorm 省下 7-10% 算力，用 SwiGLU 替代 GeLU 获得约 1 个 perplexity 点的提升，用 GQA 替代 MHA 把 KV cache 压缩数倍。这些"看起来微小"的改动合起来，是开源 LLM 在 2023-2024 年追上闭源 GPT-4 级别的关键。理解这四个支柱，也就能读懂现在任何开源大模型的配置表——剩下的是训练数据、scaling 策略、对齐微调等"非架构"层面的工程。这一系列（attention → architecture → 范式演化 → 工程支柱）覆盖了现代 LLM 架构层面的核心知识，可以作为继续学习稀疏注意力、MoE、状态空间模型等更高级主题的基础。