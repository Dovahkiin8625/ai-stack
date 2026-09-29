# GPT vs LLaMA：现代开源 LLM 的架构差异

GPT-3 之后，开源社区以 Meta 的 LLaMA 系列为代表，迅速把"decoder-only + 自回归"的范式推向工业级。本文聚焦四组核心架构差异：位置编码、归一化、激活函数、注意力优化，并附上 GPT-3 / LLaMA-2 / LLaMA-3 的简要对比表。

## 一、位置编码：绝对 vs RoPE

- **GPT-3 使用绝对位置编码**：每个位置对应一段可学习的 embedding，与 token embedding 相加。问题是对训练时没见过的长度泛化能力弱。
- **LLaMA 使用 RoPE（Rotary Position Embedding）**：把 Q、K 视作复数并按位置旋转，使得两点积天然包含相对位置信息。RoPE 对长度外推更友好，也是今天几乎所有主流开源 LLM 的默认选择。

## 二、归一化：LayerNorm vs RMSNorm

GPT 系列使用经典 LayerNorm：

```math
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
```

LLaMA 使用 **RMSNorm**：

```math
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}
```

RMSNorm 去掉了均值中心化步骤，只缩放。在数学上仍能稳定训练，且计算更快，对 LLM 这种"小算力大参数"的场景非常合适。

## 三、激活函数：GeLU vs SwiGLU

- **GPT-3 FFN**：GeLU 激活，结构为 $\text{FFN}(x) = W_2 \cdot \text{GeLU}(W_1 x + b_1) + b_2$。
- **LLaMA FFN**：使用 **SwiGLU**，三组权重：

```math
\text{SwiGLU}(x) = \text{Swish}_\beta(xW_1) \otimes (xW_2) \cdot W_3
```

其中 $\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$，$\otimes$ 是逐元素乘。SwiGLU 引入一个"门控"分支，让信息流更可控，论文与开源实现都报告了稳定的质量提升。

## 四、注意力：MHA vs GQA

- **GPT-3** 使用标准多头注意力（MHA），每个 head 都有独立的 Q/K/V 投影。
- **LLaMA-2** 在 70B 规模仍使用 MHA。
- **LLaMA-3 与更新版本** 采用 **GQA（Grouped-Query Attention）**：Q 头数 $h_q$ 较大，K/V 头数 $h_{kv}$ 较小（通常 $h_{kv} = h_q / 4$ 或 $h_q / 8$）。

直观上，K/V 投影是显存和 KV cache 的主要开销；GQA 用更少的 K/V 头数换来几乎不损失的质量。推理时 KV cache 大小可减少 4-8 倍，长上下文推理吞吐显著提升。

## 五、模型对比

| 项 | GPT-3 (2020) | LLaMA-2 (2023) | LLaMA-3 (2024) |
| --- | --- | --- | --- |
| 位置编码 | 绝对 | RoPE | RoPE |
| 归一化 | LayerNorm | RMSNorm | RMSNorm |
| FFN 激活 | GeLU | SwiGLU | SwiGLU |
| 注意力 | MHA | MHA (70B) / GQA | GQA |
| 上下文长度 | 2K | 4K | 8K+ |
| 训练 token 数 | ~300B | ~2T | ~15T |

## 六、KV Cache 与推理

GQA 与上述其他优化共同支撑了一个事实：**推理时 KV cache 的体积决定吞吐**。当上下文从 4K 扩到 128K，KV cache 是显存的主要消耗项。GQA + 8-bit KV 量化 + PagedAttention（vLLM）这类组合，已成为生产级 LLM 服务的事实标准。

## 小结

RoPE、RMSNorm、SwiGLU、GQA 共同构成了"现代开源 LLM"的标志特征。它们的设计哲学都是：**用更便宜的结构换可衡量的质量或吞吐改进**。这套组合拳也是 Mistral、Qwen、DeepSeek 等后续开源模型共同采纳的路线。
