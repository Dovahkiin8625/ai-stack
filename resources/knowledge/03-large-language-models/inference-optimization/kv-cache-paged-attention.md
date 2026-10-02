# KV Cache 与 PagedAttention

自回归 LLM 推理的最大瓶颈不是算力，而是 KV cache 的显存占用与管理方式。本文先算一笔账，看 KV cache 究竟占多少内存；再介绍 PagedAttention 如何借鉴操作系统的虚拟内存思路，把显存碎片率从 60%+ 降到 <4%；最后展示一个 vLLM 的启动例子。

## 一、KV cache 的内存账

对 transformer 的每一层，自回归解码时都要保存历史 token 的 K 和 V 张量。记：

- $L$：层数
- $d$：每头维度
- $h$：注意力头数
- $n$：当前序列长度
- $B$：batch size
- 精度：$s$ 字节（FP16 = 2, INT8 KV = 1）

则 KV cache 的总占用为：

$$
\text{KV size} = 2 \cdot B \cdot n \cdot L \cdot h \cdot d \cdot s
$$

以 LLaMA-3-8B 为例（$L=32, h=32, d=128, s=2$）：

```text
单请求 8k context:
    2 · 1 · 8192 · 32 · 32 · 128 · 2
  ≈ 17.2 GB  (≈ 模型权重本身的大小)

单请求 32k context:
    ≈ 68.7 GB  ← 单卡根本装不下，必须 PagedAttention 或外推
```

也就是说，在长上下文 + 大 batch 场景下，**KV cache 反而是显存的主要消费者**。

## 二、Naive 分配的痛点

最朴素的实现会给每个请求预分配一块**最大长度**的连续显存：

```python
# 伪代码
cache_k = torch.empty(max_len, num_layers, batch, num_kv_heads, head_dim)
cache_v = ...
```

这带来三个问题：

1. **浪费**：实际生成往往到不了 max_len，预分配的尾部完全空着。
2. **碎片**：每个请求的长度不一样（256、512、1024…），连续分配会产生大量不连续的小块。
3. **调度困难**：为了放进一张卡，系统只能"按最坏情况预留"，造成 GPU 利用率不到 30%。

## 三、PagedAttention：虚拟内存的思路

vLLM（Kwon et al., SOSP 2023）提出的 **PagedAttention** 把操作系统虚拟内存的范式搬了过来：

- 把 KV cache 切成**固定大小的 block**（默认 16 个 token 一块）。
- 每个请求维护一张**块表（block table）**，把逻辑块映射到物理块。
- 物理块在显存中是连续或不连续的都没关系，注意力计算时按块表索引。

```text
请求 A 物理块: [blk 17, blk 9, blk 42, blk 3]   ← 不连续也无所谓
请求 B 物理块: [blk 17, blk 9]                   ← 与 A 共享前两个 block（前缀共享）
请求 C 物理块: [blk 22, blk 23, blk 24]
```

带来的好处：

1. **碎片率从 60%+ 降到 <4%**（vLLM 论文实测）。
2. **支持 prefix caching**：相同 system prompt 的请求自动共享前缀 block，显存再省 30%~80%（例如 RAG、长 system 指令）。
3. **连续批处理（continuous batching）**：每个 step 只调度"还在跑"的请求，已结束请求的完成新请求可以马上填进去，GPU 几乎永远满载。

## 四、Block Manager 与调度

vLLM 内部用 BlockManager 维护三类块：

- **free blocks**：空闲物理块池。
- **cached blocks**：命中 prefix cache、可直接复用的块。
- **running blocks**：当前正在被某个请求写入/读取的块。

调度器在每个 step：

1. 把已完成的请求块归还到 free pool。
2. 把等待队列里的新请求分配 block（优先复用 cached）。
3. 把所有 running 请求的 block 表交给 GPU kernel，按页表方式做 attention。

## 五、用 vLLM 跑一个长上下文模型

```bash
# 安装：pip install vllm
# 启动一个 OpenAI 兼容服务
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --block-size 16 \
    --port 8000
```

客户端调用：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "用 200 字总结 KV cache 的本质"}],
    max_tokens=200,
    stream=False,
)
print(resp.choices[0].message.content)
```

几个关键 flag：

- `--enable-prefix-caching`：打开前缀缓存。
- `--block-size 16`：每个物理块装 16 个 token，可根据序列长度分布调。
- `--gpu-memory-utilization 0.92`：vLLM 最多吃 92% 显存，剩下的留给激活值。

## 小结

PagedAttention 借鉴操作系统虚拟内存，把 KV cache 从"按最大长度预分配连续内存"变成"按需分页 + 块表映射"。这让 vLLM 在相同硬件上能服务 2-4× 的并发请求，且几乎无碎片。下一篇我们将讨论另一条主线——**量化**，从 INT8 到 INT4、NF4、GGUF，把显存压力再砍掉 2-4×。