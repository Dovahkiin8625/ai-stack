# 量化技术：INT8 / INT4 / GGUF / AWQ / GPTQ

把 FP16/BF16 的模型降到 INT8、INT4 是 LLM 落地的关键一步：8B 模型从 16 GB 砍到 4 GB（INT4），直接能在 4090、苹果 M2 16G 显存、笔记本上跑。本文梳理量化家族树、各自的核心思想，并给出 BitsAndBytes 4-bit 加载的最小代码。

## 一、量化分类速览

| 维度 | 选项 | 含义 |
|---|---|---|
| **训练时机** | PTQ（Post-Training） vs QAT（Quantization-Aware Training） | 训完再压 vs 训练中模拟量化 |
| **对称性** | 对称 vs 非对称 | 零点是否独立学习 |
| **粒度** | per-tensor / per-channel / per-group | 一组张量共用一个 scale 还是每个通道/每 32/64 元素各一组 |
| **比特数** | INT8 / INT4 / INT3 / FP8 / NF4 | 越低越省，但精度损失越大 |
| **代表性方案** | SmoothQuant、GPTQ、AWQ、NF4（BNB）、GGUF（llama） | 各对应不同的数学思路 |

实务中 PTQ 居多，因为不需要重训。下面重点说 PTQ 这条线。

## 二、核心数学

把一个 FP 张量量化到 INT8，最简单的对称 per-channel 公式：

$$
x_q = \text{clip}\!\left(\text{round}\!\left(\frac{x}{s}\right),\ -127,\ 127\right),\quad s = \frac{\max|x|}{127}
$$

反量化时再乘回去 $x \approx x_q \cdot s$。**误差来源**主要是 `round` 和 `clip` 引入的有偏估计。

GPTQ / AWQ 进一步用**二阶信息**或**激活分布**做更精细的 scale 设计，把 4-bit 的精度损失控制在可接受范围。

## 三、INT8：SmoothQuant

LLM 的激活值存在"异常值"——少数几个维度数值特别大，其余接近 0。直接 per-tensor 量化会因这几个 outlier 把整个 scale 拉爆。

SmoothQuant 的做法：在数学上等价地把 activation 的难度**转移**给权重：

$$
\mathbf{Y} = (\mathbf{X} \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot \mathbf{W})
$$

选一个合适的 $s$，让 $X/s$ 平滑好量化，同时让 $s \cdot W$ 仍在权重可量化范围内。基本上"白送"的 ~2× 显存收益，精度几乎不掉。

## 五、INT4：GPTQ 与 AWQ

### GPTQ（2023）

基于二阶 Hessian 信息（Optimal Brain Quantization 的现代版），按列逐层把误差最小化传播下去。一次 calibration（~128 条样本）+ 几分钟就能压一个 7B 模型到 4-bit。

### AWQ（Activation-aware Weight Quantization, 2023）

观察到"重要的权重往往对应激活值大的通道"。做法：

1. 用少量校准数据统计每个 channel 的平均激活幅度。
3. 找出 top 1% ~ 3% 的"salient channels"，**保留 FP16**；或者用一个 scale $s$ 把这些通道放大后再量化。

直觉上：把"难量化的点"用更宽的位宽保护起来，其余通道省着量化。

## 五、NF4 与 BitsAndBytes

BitsAndBytes（BNB）提出 **NF4（NormalFloat 4-bit）**：基于正态分布设计的一组 16 个量化点，比均匀 INT4 在 LLM 权重分布上更优。

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 关键
    bnb_4bit_quant_type="nf4",               # 用 NF4 而不是 FP4
    bnb_4bit_compute_dtype=torch.bfloat16,   # 计算仍用 BF16
    bnb_4bit_use_double_quant=True,          # 对 scale 再做一次量化，进一步省
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb_config,
    device_map="cuda",
)
```

内存收益对比（LLaMA-3-8B，A100）：

| 精度 | 权重占用 | 相对速度 | 备注 |
|---|---|---|---|
| FP16 | ~16 GB | 1.0× | 基准 |
| INT8 (SmoothQuant) | ~8 GB | 1.5-2.0× | 几乎不掉点 |
| INT4 (GPTQ / AWQ) | ~4 GB | 2.0-3.0× | 1-3% 精度损失 |
| NF4 (BNB) | ~4 GB | 2.0-3.0× | 简单易上手，无须校准 |

## 六、GGUF（llama.cpp）

GGUF 是 llama.cpp 生态的量化格式，目标平台包括 **CPU、Apple Silicon、低显存 GPU**。常用类型：

```text
Q2_K    ≈ 2.5 bit  极小模型/便携
Q4_K_M  ≈ 4.5 bit  推荐默认（精度/体积甜点）
Q5_K_M  ≈ 5.5 bit  精度优先
Q6_K    ≈ 6.5 bit  几乎无损
Q8_0    ≈ 8 bit    无损
```

用 llama.cpp 一行命令量化：

```bash
./llama-quantize ./models/llama3-8b-f16.gguf ./models/llama3-8b-q4.gguf Q4_K_M
```

加载：

```bash
./llama-cli -m ./models/llama3-8b-q4.gguf \
    -p "解释 AWQ 的核心思想" \
    -n 200 -t 8
```

## 七、工程取舍

| 场景 | 推荐 |
|---|---|
| 4090 / A100 GPU 服务 | AWQ-INT4 + vLLM |
| 显存紧张 / 笔记本 | NF4（BNB）或 GGUF Q4 |
| CPU 服务 | GGUF Q4_K_M |
| 极致无损 | FP16/BF16 + SmoothQuant-INT8 |

## 小结

量化是 LLM 推理优化的"性价比之王"——4-bit 通常能拿到 2-3× 的速度、4× 的显存降幅，精度只掉 1-3%。不同方案的取舍是：BNB 最易上手，ATFP8 + SmoothQuant 最稳，AWQ/GPTQ 在 GPU 上综合最优，GGUF 跨平台能力最强。量化+流水线+插件式优化可以叠加。下一篇我们会聊**投机解码（Speculative Decoding）**——在不降低精度的前提下，进一步压低延迟。