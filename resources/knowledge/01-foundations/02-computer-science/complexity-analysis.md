# 复杂度分析

## 一、渐近记号

**大 O**：上界。$f(n) = O(g(n))$ 表示存在常数 $c, n_0$，使得 $n \ge n_0$ 时 $f(n) \le c \cdot g(n)$。

**大 Ω**：下界。$f(n) = \Omega(g(n))$。

**大 Θ**：紧界。$f(n) = \Theta(g(n))$ 当且仅当它既是 $O$ 又是 $\Omega$。

**小 o**：严格小于。$f(n) = o(g(n))$ 表示 $f/g \to 0$。

我们通常用大 O 描述"最坏情况复杂度"。但在 ML 中**期望复杂度**（如哈希表平均 $O(1)$）往往更贴近实际。

## 二、主定理：分治算法的复杂度

很多 ML 算法（FFT、卷积、归并排序）都是分治结构。**主定理**告诉我们，对于递推

$$T(n) = a\, T(n/b) + f(n)$$

其中 $a$ 是子问题数、$n/b$ 是子问题规模、$f(n)$ 是合并成本：

- 若 $f(n) = O(n^{\log_b a - \epsilon})$，则 $T(n) = \Theta(n^{\log_b a})$（递归主导）。
- 若 $f(n) = \Theta(n^{\log_b a})$，则 $T(n) = \Theta(n^{\log_b a} \log n)$。
- 若 $f(n) = \Omega(n^{\log_b a + \epsilon})$，则 $T(n) = \Theta(f(n))$（合并主导）。

经典例子：

- **归并排序**：$T(n) = 2 T(n/2) + O(n)$，$a=2, b=2, f(n)=n$，与第二档匹配，$T(n) = \Theta(n \log n)$。
- **二分搜索**：$T(n) = T(n/2) + O(1)$，$a=1, b=2, f(n)=1$，第一档给出 $T(n) = \Theta(\log n)$。
- **Karatsuba 大整数乘**：$T(n) = 3 T(n/2) + O(n)$，$a=3, b=2$，$\log_b a = \log_2 3 \approx 1.585 > 1$，$T(n) = \Theta(n^{1.585})$，比 $O(n^2)$ 朴素法更快。

## 三、平摊分析

有些操作偶尔贵、多数便宜——例如哈希表扩容时把所有元素重哈希一遍。**平摊复杂度**统计 $n$ 个操作的总开销除以 $n$：

- **哈希表 push**：扩容均摊 $O(1)$。
- **动态数组 push**（Python `list.append`）：均摊 $O(1)$。
- **二项堆 / Fibonacci 堆**：insert 均摊 $O(1)$，decrease-key 均摊 $O(1)$，extract-min $O(\log n)$。

ML 中典型场景：

- **CUDA kernel 启动**：每次 kernel 启动几微秒，把"小操作"合并成 batch 能摊薄启动开销。
- **梯度累积**：每 $K$ 个 micro-batch 才做一次优化器 step，相当于把 $K$ 次前向+反向的开销摊到一次更新。

## 四、空间复杂度

空间复杂度关心**额外**内存：

- **线性回归闭式解** $w = (X^\top X)^{-1} X^\top y$：需要 $O(nd)$ 存 $X$，$O(d^2)$ 算 $X^\top X$。
- **反向传播**：需要保存所有中间激活用于反向计算，空间复杂度 $O(L \cdot d)$（$L$ 是层数）。**梯度检查点（gradient checkpointing）** 只保存部分激活、计算时再重算，空间降为 $O(\sqrt{L \cdot d})$。
- **Transformer 注意力矩阵**：$O(n^2)$，序列长度 4096 时单头注意力矩阵 16K×16K 个 float = 1 GB。**FlashAttention** 通过分块和重计算把空间降到 $O(n)$。

```python
# gradient checkpointing：用时间换空间
from torch.utils.checkpoint import checkpoint
class TinyBlock(torch.nn.Module):
    def forward(self, x):
        # 标准：保存中间激活
        # out = self.subblock(x)
        # checkpointing：不保存，反向时重算
        out = checkpoint(self.subblock, x, use_reentrant=False)
        return out
```

## 五、并行与分布式复杂度

当问题并行化后，单机复杂度变为：

$$T_{\text{parallel}}(n, p) = T_{\text{serial}}(n) / p + T_{\text{comm}}(p) + T_{\text{sync}}$$

其中 $p$ 是并行度。通信/同步成本决定可扩展性上限。ML 训练对应：

- **数据并行**：每张卡算梯度 $\to$ all-reduce $\to$ 同步。通信量 $\Theta(p \cdot s)$。
- **模型并行**：把模型切到多卡，每卡负责部分层。通信量 $\Theta(n \cdot d)$（前向/反向之间）。
- **流水线并行**：把不同 micro-batch 流水化到不同 stage，能掩盖气泡。
- **张量并行**：把单个矩阵乘切块，通信量 $\Theta(d^2 / p)$。

**Amdahl 定律**提醒我们：串行部分再小也限制总加速比。如果 5% 的步骤必须串行，理论最大加速比就是 20×。

```math
S_{\max}(p) = \frac{1}{f + (1-f)/p}
```

## 六、随机算法与期望复杂度

很多 ML 算法（蒙特卡洛、随机优化、采样）有随机性。分析时分两层：

- **最坏情况**：罕见事件也考虑（如 Rademacher 复杂度）。
- **期望情况**：平均意义上的复杂度。

```python
# 蒙特卡洛估计 π：期望 O(N)，最坏无界
import random
def estimate_pi(N):
    inside = 0
    for _ in range(N):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1.0:
            inside += 1
    return 4 * inside / N

# 收敛速度 O(1/sqrt(N))：误差 ε 需要 N ≈ 1/ε² 个样本
print(estimate_pi(100_000))   # ≈ 3.141...
```

**Hoeffding 不等式**：独立有界随机变量的平均偏离期望超过 $\epsilon$ 的概率 $\le 2 e^{-2 n \epsilon^2}$。这告诉我们：要把蒙特卡洛估计误差从 0.01 降到 0.001，样本量要乘 100。

## 七、ML 工程中的复杂度估算实践

几个常见场景的"心算公式"：

- **矩阵乘** $A_{m \times k} \cdot B_{k \times n}$：FLOPs $= 2 m k n$，内存 $m k + k n + m n$。
- **Transformer 自注意力**（batch $B$, seq $n$, head $d$）：FLOPs $B n^2 d$（QK^T + attn·V），内存 $B n^2$（注意力矩阵）。
- **Transformer MLP**：FLOPs $B n d^2 \cdot 8$（两次线性 + GeLU + 投影）。
- **反卷积** $C_{\text{in}} \to C_{\text{out}}$，kernel $K \times K$，输出 $H \times W$：FLOPs $2 C_{\text{in}} C_{\text{out}} K^2 H W$。

```python
# 估算 GPT-2 一次前向的 FLOPs ≈ 2 × 参数数 × token 数
# 参数量 N，一次前向 token 数 T
N, T = 1.5e9, 1024           # GPT-2 XL, 1K 上下文
flops = 2 * N * T
print(f"FLOPs: {flops/1e12:.2f} T")  # 约 3.07 TFLOPs
# A100 理论 312 TFLOPS(fp32) / 624 TFLOPS(TF32)
# 时间 ≈ 3 / 624 ≈ 5 ms（理想）
```

**训练 FLOPs 估算**：每个 token 训练需约 $6N$ FLOPs（前向 $2N$ + 反向 $4N$）。这是 Kaplan/McCandlish 2020 的标度律结论。

## 八、P、NP 与可计算性

虽然深度学习很多问题（如最优架构搜索）理论上是 NP 难，但实践中我们用启发式 + 近似解决。复杂度类层次：

- **P**：多项式时间可解。
- **NP**：解可在多项式时间验证（如 SAT、子集和）。
- **NP-hard**：至少和 NP 中最难问题一样难。
- **NP-complete**：既属 NP 又 NP-hard。
- **PSPACE**：多项式空间（国际象棋博弈）。
- **EXPTIME**：指数时间。

ML 里的复杂度讨论：

- **Transformer 的表达能力**：理论上是 $TC^0$，弱于图灵完备；很多问题（如递归函数、长度外推）它根本学不会。
- **架构搜索**：NAS 搜索空间是组合的，暴力枚举不可行，需用强化学习、可微搜索（DARTS）。
- **混合整数规划**：某些 NAS / 量化方案用求解器得到理论最优解。

## 小结

复杂度分析是工程判断的"度量衡"：大 O 给出渐近上界、主定理分析分治、平摊分析处理间歇性昂贵操作、空间复杂度量化显存压力、并行模型描述分布式训练。把这套语言内化，你就能在写代码前估算"是否可行""是否要并行""是否要用 checkpoint"，而不是等跑起来才发现"显存炸了"。这正是从"会写代码"到"会做工程"的分水岭。