# 分布式训练：DP / TP / PP / ZeRO / FSDP

7B 模型用一张 A100 还能装下，70B 就需要 8 卡 H100，到了 500B 级别的训练就是上千张 GPU 协同工作的问题。**分布式训练**就是回答：如何把一个模型 + 一次前向 + 反向 + 优化，拆到多张卡甚至多台机器上高效完成。本文系统梳理 DP / TP / PP / ZeRO / FSDP / AMP 的核心思想，并给出 PyTorch FSDP 包装一个 7B 模型的最小代码与 Megatron-LM 风格 TP 切分示意。

## 一、为什么需要分布式训练

先算一笔账。以 LLaMA-3 风格模型为例（参数量 $N$，token 维度 $d$）：

| 项 | 显存占用（FP16/BF16） |
|---|---|
| 权重 | $2N$ 字节 |
| 优化器状态（Adam：m, v）| $8N$ 字节 |
| 梯度 | $2N$ 字节 |
| 激活值（与 batch、seq、layer 相关）| $O(B \cdot L \cdot s \cdot d)$ |
| **合计（仅静态）** | **$12N$ 字节** |

70B 模型 $12N \approx 840$ GB——单卡完全装不下。**分布式训练的核心就是把这 $12N$ 拆给多张卡**，并尽量减少通信开销。

## 二、数据并行（DP / DDP）

最朴素的并行：每张卡各放一份**完整模型副本**，跑不同的 mini-batch，然后同步梯度。

```text
GPU 0: model_w   ↘
GPU 1: model_w   →  all-reduce(grad)  →  opt.step()
GPU 2: model_w   ↗
GPU 3: model_w   ↗
```

数学：

$$
g = \frac{1}{K}\sum_{k=1}^{K} g_k,\quad \theta \leftarrow \theta - \eta \cdot g
$$

每个 GPU 用同一 batch 的不同切片算梯度，all-reduce 后平均。**优点**：实现简单，几乎线性扩展。**缺点**：每张卡都存完整模型 + 优化器 + 梯度，70B 单卡还是放不下。

PyTorch `DistributedDataParallel (DDP)` 是工业标准，配合 `torchrun` 一行启动：

```bash
torchrun --nproc_per_node=8 train.py     # 8 卡 DDP
```

## 三、张量并行（TP）

把"一个张量"沿某一维切到多卡。**Megatron-LM 风格**的做法：在 transformer block 内对 QKV 投影与 FFN 做列并行 / 行并行切分：

```text
QKV projection (d -> 3d):   按输出维度切，每卡算 3d/N 份
attention output (d -> d):  按输入维度切，每卡算 d/N 份
FFN up (d -> 4d):           按输出维度切
FFN down (4d -> d):         按输入维度切
```

每张卡算自己那份的 attention，但 attention 计算本身需要**全量序列**——所以要在进入 attention 之前做一次 all-gather 还原出完整 K/V。

**通信开销**：每个 transformer block 约 $4$ 次 all-reduce，规模与 $d$ 成正比，与 batch、seq 无关。**优点**：通信模式简单、对激活值影响小；与 PP 结合是 GPT-3 / LLaMA-3 训练的标准做法。

一个 Megatron 风格的示意（去掉 bias、用 2 卡 TP=2）：

```python
import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    """按输出维度切：Y = X @ W，每卡算一部分 Y。"""
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert out_features % world_size == 0
        self.local_out = out_features // world_size
        self.weight = nn.Parameter(torch.randn(self.local_out, in_features))
        self.world_size = world_size
        self.rank = rank

    def forward(self, x):
        # x: (..., in_features) -> local: (..., in_features)
        return x @ self.weight.t()           # (..., local_out)


class RowParallelLinear(nn.Module):
    """按输入维度切：每卡算部分输出再 all-reduce 求和。"""
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        assert in_features % world_size == 0
        self.local_in = in_features // world_size
        self.weight = nn.Parameter(torch.randn(out_features, self.local_in))
        self.world_size = world_size

    def forward(self, x):
        # x: (..., local_in) -> partial: (..., out_features)
        out = x @ self.weight.t()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


class TPMLP(nn.Module):
    """TP=2 的 MLP：up (col) -> GeLU -> down (row)。"""
    def __init__(self, d, hidden, world_size, rank):
        super().__init__()
        self.up = ColumnParallelLinear(d, hidden, world_size, rank)
        self.down = RowParallelLinear(hidden, d, world_size)

    def forward(self, x):
        # x 已经在各卡上一致（来自 attention 的 all-gather）
        h = torch.nn.functional.gelu(self.up(x))
        return self.down(h)
```

实际生产用 NVIDIA `Megatron-LM` 或 `tensor_parallel`（DeepSpeed / PyTorch 原生 `torch.distributed.tensor`）即可，无需从零写。

## 四、流水线并行（PP）

把模型按层切开，每张卡放连续几层。前向时数据像流水线一样穿过各卡，反向时反向传播：

```text
GPU 0: layer 0..7
GPU 1: layer 8..15
GPU 2: layer 16..23
GPU 3: layer 24..31
```

**朴素 PP 的问题**：流水线 bubble——GPU 0 算完要等 GPU 3 跑完才能开始反向。**GPipe** 把 mini-batch 切成 micro-batches 注入流水线填充 bubble；**PipeDream / 1F1B** 进一步把前向与反向交替，降低激活值存储。**通信开销**：只在相邻 stage 边界传 activation，比 TP 低一个数量级。**缺点**：模型必须严格按层切分，灵活性差。

## 五、ZeRO：把"状态"切到多卡

ZeRO（Zero Redundancy Optimizer, Microsoft 2020）针对 DDP 的显存冗余——每张卡都存完整权重 + 优化器状态 + 梯度。ZeRO 分三阶段切它们：

| 阶段 | 切什么 | 单卡显存 | 通信量 |
|---|---|---|---|
| ZeRO-1 | 优化器状态 | $2N + 2N + 8N/K$ | 与 DDP 相同 |
| ZeRO-2 | + 梯度 | $2N + 10N/K$ | 与 DDP 相同 |
| ZeRO-3 | + 权重 | $12N/K$ | **额外通信**（前向时按需 gather 权重） |

$K$ 是并行度。**ZeRO-3** 等价于把模型按参数切片摊到多卡——这就是 FSDP。

## 六、FSDP：PyTorch 原生的 ZeRO-3

`torch.distributed.fsdp.FullyShardedDataParallel` 是 PyTorch 对 ZeRO-3 的官方实现。给定一个 HuggingFace 模型，加一行就能包装：

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM
from functools import partial

# 1) 初始化进程组
torch.distributed.init_process_group("nccl")
rank = torch.distributed.get_rank()
torch.cuda.set_device(rank % torch.cuda.device_count())

# 2) 加载模型（meta 设备先建结构，再 materialize，节省初始化显存）
model_name = "meta-llama/Meta-Llama-3-8B"
with torch.device("meta"):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# 3) FSDP wrap 策略：按 transformer 层 wrap，通信粒度与计算粒度对齐
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
my_auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)

# 4) Mixed precision：参数/通信用 BF16，reduce 用 FP32 保证精度
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)

model = FullyShardedDataParallel(
    model,
    device_id=torch.cuda.current_device(),
    auto_wrap_policy=my_auto_wrap_policy,
    mixed_precision=mp_policy,
    sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,  # ZeRO-3
)

# 5) 训练循环（DDP 接口相同）
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
for batch in dataloader:
    ids = batch["input_ids"].cuda()
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    optim.step(); optim.zero_grad()
```

启动：

```bash
torchrun --nproc_per_node=8 train_fsdp.py
```

`FULL_SHARD` 即 ZeRO-3；`SHARD_GRAD_OP` 即 ZeRO-2；`NO_SHARD` 退化为 DDP。

## 七、激活检查点（Activation Checkpointing）

前向时保存所有激活值用于反向，是显存大头之一。**激活检查点**用"用时间换空间"：只保存 layer 边界的激活值，反向时重算中间层。

PyTorch 中只需一行：

```python
from torch.utils.checkpoint import checkpoint_sequential

# 把若干个 sublayer 包成 segment，前向只保留 segment 边界的激活值
model.layers = checkpoint_sequential(model.layers, segments=4, use_reentrant=False)
```

**经验**：4~8 个 segment 是常用甜点；太多则重算开销盖过显存收益。

## 八、混合精度训练（AMP / BF16）

混合精度训练用 FP16 或 BF16 存权重与梯度，用 FP32 维护优化器状态与 master 权重：

```text
forward  : FP16 / BF16
loss     : FP32（加 loss scaling 避免 FP16 下溢）
gradient : FP16 / BF16 + dynamic loss scaling
update   : FP32 master weights + FP32 optimizer state
```

LLaMA-3 / Mistral 全程用 **BF16**——它没有 FP16 的下溢问题，硬件上 A100 / H100 / 4090 都有原生 BF16 张量核支持。代码上 PyTorch 的 `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` 即可。

## 九、3D 并行：DP + TP + PP 一起用

GPT-3 175B、LLaMA-3 405B 这类大模型训练都靠**3D 并行**：

```text
PP (pipeline)         把模型按层切到 8 个 stage
  └─ TP (tensor)      每个 stage 内再用 8 卡张量并行
      └─ DP (data)    每组 TP 内再复制多份做数据并行
```

例如 `8 (PP) × 8 (TP) × 4 (DP) = 256 卡` 配置，每张卡只放 ~ 模型大小 / (PP × TP)。LLaMA-3-405B 训练用 ~16000 张 H100，配置正是这种 3D 并行加上 ZeRO-1 + BF16 + activation checkpoint。

## 十、序列并行与专家并行

序列并行（Sequence Parallel, SP）把 LayerNorm / Dropout 这类与序列维度相关的算子沿 seq 切，进一步省激活值显存。在 Megatron-LM 里，SP 与 TP 联合使用，把 transformer 内 "attn 计算 + LayerNorm + dropout" 这一段的 seq 维度切到 TP group 内各卡。

MoE 模型还会引入**专家并行（Expert Parallel, EP）**：把不同专家放到不同卡，所有 token 通过 all-to-all 路由到对应专家。Mixtral 8x7B 训练时常用 `TP=2, EP=4` 之类的组合。

## 十一、实践选型

| 模型规模 | 推荐策略 |
|---|---|
| <= 3B | 单机 DDP / FSDP-1 |
| 3B ~ 13B | FSDP-2 / FSDP-3，单机 8 卡 |
| 13B ~ 70B | FSDP-3 多机；或 3D 并行 (TP+PP) |
| 70B ~ 200B | 3D 并行 + ZeRO-1 + 激活检查点 |
| 200B+ | 3D 并行 + EP（若 MoE）+ 复杂流水线调度 |

## 小结

分布式训练不是"一张卡放不下就复制几份"那么简单。**DDP** 是基础（切数据），**TP** 解决单层过大（切张量），**PP** 解决整网络过大（切层），**ZeRO/FSDP** 把"状态"也切掉（切优化器/梯度），**激活检查点** 用时间换空间，**BF16** 进一步压显存带宽。真正的工业级大模型训练往往是 DP + TP + PP + ZeRO + AMP + Checkpoint 的联合优化。下一篇我们会讨论一个更宏观的问题——**为什么大就是好**：scaling laws、emergent abilities，以及它们对架构与训练策略的反向约束。
