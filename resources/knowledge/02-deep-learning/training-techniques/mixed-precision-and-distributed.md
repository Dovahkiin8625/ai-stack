# 混合精度训练与分布式训练：让大模型训得起、训得快

现代深度学习（特别是 LLM）训练离不开两件事：**降低单步成本**（混合精度）与**摊到多张卡上**（分布式训练）。前者把显存与算力压到接近一半，后者让"单卡放不下"的模型也能跑起来。本文先讲 FP32/FP16/BF16 的差异与 autocast + GradScaler，再用 DataParallel → DistributedDataParallel → FSDP → DeepSpeed ZeRO → 张量并行的演进路径，配合 PyTorch 实战。

## 一、为什么需要混合精度

GPU 的浮点算力与显存带宽与数据精度强相关：

- **FP32**（4 字节）：训练默认精度，数值稳定但吃显存与算力。
- **FP16**（2 字节）：算力翻倍、显存减半，但动态范围窄，训练常溢出/下溢。
- **BF16**（2 字节）：与 FP32 同 8 位指数位（范围大），尾数短（精度低）——深度学习的"甜点精度"。

| 类型 | 总位数 | 指数位 | 尾数位 | 范围（量级） | 典型用途 |
|---|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | 传统训练、损失/优化器状态 |
| FP16 | 16 | 5 | 10 | $\pm 6.5 \times 10^{4}$ | 推理、半精度训练 |
| BF16 | 16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | **LLM 训练首选** |

BF16 的范围与 FP32 相同，**不需要 loss scaling 就能稳定训练**——这就是 LLM 训练普遍切到 BF16 的根本原因。

## 二、FP16 混合精度训练：autocast + GradScaler

PyTorch 用 `torch.cuda.amp` 提供两件套：

- **`autocast`**：自动把算子切到 FP16（矩阵乘、卷积），保留 FP32（损失、softmax、归一化）。
- **`GradScaler`**：动态放大 loss，避免小梯度在 FP16 下溢出/下溢。

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model     = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler    = GradScaler()

for x, y in loader:
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)

    with autocast(dtype=torch.float16):
        logits = model(x)
        loss   = F.cross_entropy(logits, y)

    scaler.scale(loss).backward()   # 放大 loss 后反向
    scaler.unscale_(optimizer)     # 反向放大梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)         # 若溢出则跳过此次 step
    scaler.update()                # 动态调整 scale 因子
```

**Loss scaling 的直观**：FP16 能表示的最小正数约 $6 \times 10^{-8}$，而小梯度可能是 $10^{-10}$——直接训练会下溢成 0。把 loss 乘一个 $S = 2^{15}$ 左右的因子，梯度也被等比放大，反向完再除回去。

## 三、BF16 训练：更简单

BF16 不需要 loss scaling：

```python
with autocast(dtype=torch.bfloat16):
    logits = model(x)
    loss   = F.cross_entropy(logits, y)

loss.backward()      # 直接反向，无需 scaler
optimizer.step()
```

NVIDIA Ampere 架构（A100/A10/4090）开始原生支持 BF16。如果你的硬件支持，**优先用 BF16**——既快又稳，几乎不需要额外调参。

## 四、显存节省量化

以 AdamW 训练为例，单参数需要的总存储（参数 + 梯度 + AdamW 状态）：

| 精度 | 参数 | 梯度 | AdamW ($m, v$) | 合计/参数 |
|---|---|---|---|---|
| FP32 | 4 B | 4 B | 8 B | **16 B** |
| BF16/FP16 | 2 B | 2 B | 8 B（通常 FP32 维护） | **12 B** |

一个 7B 模型用 AdamW 训练，仅优化器状态就需要 $7 \times 10^9 \times 12B \approx 84$ GB——**单卡 A100（80 GB）几乎放不下**，必须用分布式或 ZeRO/FSDP。

## 五、分布式训练总览

主流方案分两条路：

- **数据并行**：每张 GPU 跑同一份模型副本，各自吃一份数据，只同步梯度（DP/DDP/FSDP/ZeRO）。
- **模型并行**：把模型本身切到多卡（张量并行 TP / 流水线并行 PP）。

下面逐个介绍。

## 六、DataParallel（DP）：最简单的多卡

`nn.DataParallel(model)` 把单进程包装成多卡：

```python
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

**优点**：API 简单，零样板代码。  
**缺点**：单进程——主卡汇总梯度再分发，通信与计算重叠差；主卡显存占用偏高；Python GIL 瓶颈；不支持多机。

只适合"快速跑起来"或笔记本多卡 debug。生产训练请用 DDP。

## 七、DistributedDataParallel（DDP）：生产首选

DDP **每个 GPU 起一个独立进程**，各持一份模型副本，分别做前向/反向，**只同步梯度**。通信靠 `torch.distributed`（NCCL 后端）。

**Ring All-Reduce 思想**：把 N 个 GPU 排成一个环，每个 GPU 同时向下一个邻居发数据并接收上一邻居的数据——$\log_2 N$ 轮就能让所有 GPU 拿到完整聚合结果，通信量与 GPU 数无关。

**DDP 启动模板（`torchrun` 风格）**：

```python
# train_ddp.py
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=world, rank=rank)
    torch.cuda.set_device(rank)

    model = MyModel().cuda(rank)
    model = DDP(model, device_ids=[rank])

    loader = DataLoader(ds, sampler=DistributedSampler(ds, shuffle=True))

    for epoch in range(epochs):
        loader.sampler.set_epoch(epoch)        # 重要！保证 shuffle 不同
        for x, y in loader:
            loss = model(x.cuda(rank), y.cuda(rank))
            loss.backward()
            opt.step(); opt.zero_grad()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

启动方式：`torchrun --nproc_per_node=4 train_ddp.py`；跨机加 `--nnodes`、`--node_rank`、`--master_addr`、`--master_port`。

**Bucket 通信优化**：DDP 不每层都触发通信，而是把同层的梯度攒到 bucket 里，等 bucket 满了再做 all-reduce——把通信与反向重叠，效率接近"零通信开销"。

## 八、FSDP：把模型本身切碎

DDP 假设每张卡能装下完整模型。**当模型大到一张卡放不下**（如 70B LLM），需要 FSDP / ZeRO-3：把参数、梯度、优化器状态都切分到所有 GPU。

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision, BackwardPrefetch
)

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    MyModel(),
    mixed_precision=mp_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device(),
)
```

FSDP 内部三步：

1. **All-Gather** 参数 → 前向算完立即释放。
2. **Reduce-Scatter** 梯度 → 各自保留自己那份。
3. **All-Gather** 优化器状态 → 更新参数。

显存占用大致从"$\text{model} \times N$"降到"$\text{model} / N$"——线性伸缩。

## 九、DeepSpeed ZeRO：分阶段切分

DeepSpeed 把切分做成三个阶段，可叠加：

| 阶段 | 切分对象 | 显存节省 | 适用 |
|---|---|---|---|
| ZeRO-1 | 优化器状态 ($\text{opt} / N$) | ~4× | 显存瓶颈在 Adam 状态 |
| ZeRO-2 | + 梯度 ($\text{grad} / N$) | ~8× | 进一步压梯度 |
| ZeRO-3 | + 参数 ($\text{param} / N$) | ~$N$× | 巨型模型，配合 FSDP 等价 |

**ZeRO-1 ≈ DDP**，**ZeRO-3 ≈ FSDP**。PyTorch 原生 FSDP 与 DeepSpeed ZeRO-3 性能与功能高度对齐，按团队习惯二选一即可。

## 十、张量并行与流水线并行（模型并行）

当模型大到一张节点都放不下，需要把**模型本身**拆到多卡：

**张量并行（Tensor Parallel）**：把单层的大矩阵乘切成多份，例如把 MLP 切到 4 张卡上各算一部分再求和。Megatron-LM、Transformer Engine 的核心思想。

**流水线并行（Pipeline Parallel）**：把不同层放到不同卡上，前向/反向按"流水线"接力。代表：GPipe、PipeDream。

模型并行的工程复杂度远高于数据并行，需要处理切分点、通信与流水线气泡（bubble）。当代 LLM 训练常见组合：**3D 并行**（数据并行 + 张量并行 + 流水线并行）。

## 十一、辅助显存优化技术

- **梯度累积**：把大 batch 拆成多个小 micro-batch，累计梯度再 step。
- **梯度检查点（Gradient Checkpointing）**：反向时重新计算部分激活，省一半以上激活显存，代价是训练慢 30%。

```python
from torch.utils.checkpoint import checkpoint_sequential
x = checkpoint_sequential(self.blocks, segments=4, input=x, use_reentrant=False)
```

- **CPU Offload**：把暂时不用的优化器状态卸到 CPU 内存。
- **激活 Offload**（ZeRO-Offload）：把激活也卸到 CPU。

## 十二、各方案对比与选型

| 方案 | 通信开销 | 显存节省 | 适用规模 | 复杂度 |
|---|---|---|---|---|
| 单卡 | — | 1× | 实验/小模型 | 低 |
| DataParallel | 高（主卡瓶颈） | 1× | 4 卡 debug | 低 |
| DDP | 低（NCCL ring） | 1× | 单机能装下的模型 | 中 |
| FSDP / ZeRO-3 | 中 | $\sim N$× | 单机/多机大模型 | 中 |
| ZeRO-1/2 | 低 | 4~8× | 显存瓶颈在 opt | 中 |
| 张量并行 (TP) | 高（all-reduce 多） | $\sim T$× | 单层太大 | 高 |
| 流水线并行 (PP) | 中 | $\sim P$× | 巨型模型 | 高 |
| 3D 并行 (DP+TP+PP) | 综合 | 极大 | 70B+ LLM 训练 | 高 |

**何时升一档**：

1. **DDP** 是默认起点，能装下就用 DDP。
2. 显存吃紧但模型仍能装下单卡 → 先试 ZeRO-1/2 或 offload。
3. 单卡装不下 → **FSDP / ZeRO-3**。
4. 单节点装不下 → FSDP 多节点或 3D 并行。
5. 训练万亿参数 LLM → 必然是 3D 并行 + ZeRO + 序列并行（Megatron-DeepSpeed 路线）。

## 十三、监控与诊断

分布式训练排错时关注：

- **吞吐（tokens/s 或 samples/s）**：直接反映效率。
- **GPU 显存峰值**：`torch.cuda.max_memory_allocated()`。
- **梯度/参数 norm**：检测异常。
- **NCCL 日志**：`NCCL_DEBUG=INFO` 可以查通信瓶颈。
- **桶利用率**：`TORCH_DISTRIBUTED_DEBUG=DETAIL`。

```python
if step % 100 == 0 and rank == 0:
    print(f"step {step}  loss {loss.item():.4f}  "
          f"throughput {samples_per_sec:.0f}  "
          f"peak_mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
```

## 小结

混合精度（BF16/FP16+AMP）让训练"又快又省"——LLM 默认 BF16，CNN 默认 FP16 + GradScaler。分布式训练是一条清晰的进阶路径：**单卡 → DataParallel → DDP → FSDP/ZeRO-3 → 3D 并行**。先按 DDP 起步，模型大到装不下就升级 FSDP，再大就上 3D 并行。配合梯度累积、梯度检查点、offload 等辅助技巧，可以把训练规模从"几张卡的小模型"推到"上千张卡的千亿参数 LLM"。掌握这套选型逻辑，比死记任何具体 API 都重要。
