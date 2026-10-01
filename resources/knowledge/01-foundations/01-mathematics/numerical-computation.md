# 数值计算与稳定性

## 一、IEEE 754 浮点数：机器里的"实数"

现代深度学习几乎只用两种精度：

- **fp32**（单精度）：1 位符号 + 8 位指数 + 23 位尾数，约 7 位有效十进制数字。
- **fp16**（半精度）：1 + 5 + 10 位，约 3 位有效数字。

浮点不是实数——它有**机器精度** $\epsilon$：满足 $1 + \epsilon = 1$ 的最小正数。在 fp32 下 $\epsilon \approx 1.19 \times 10^{-7}$，fp16 下约 $9.77 \times 10^{-4}$。

```python
import torch, numpy as np
print(torch.finfo(torch.float32).eps)   # 1.1920929e-07
print(torch.finfo(torch.float16).eps)   # 9.7656e-04

# fp16 下做累加很容易出问题
x = torch.tensor([1.0] * 10000, dtype=torch.float16)
print(x.sum())           # 9998.0 左右，不是精确的 10000
print(x.sum(dtype=torch.float32))  # 10000.0，更稳
```

大数"吃掉"小数也是经典坑：

```python
a = 1.0
for _ in range(50):
    a = a + 1e-9      # a 仍然是 1.0
print(a)               # 1.0 —— 1e-9 在 fp64 下能加，在 fp32 下也勉强；fp16 完全丢失
```

## 二、上溢 / 下溢与 log-sum-exp

直接计算 softmax 在大 logits 下会**上溢**：

```python
logits = torch.tensor([1000.0, 1001.0, 1002.0])
print(torch.softmax(logits, dim=-1))   # tensor([nan, nan, nan])
```

标准修复是减去最大值（"max trick"），但更一般也更稳定的写法是 **log-sum-exp**：

$$
\text{LSE}(\mathbf{x}) = \log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m},\quad m = \max_i x_i
\]

```python
def log_softmax(x):
    m = x.max()
    s = torch.log(torch.exp(x - m).sum())
    return (x - m) - s

print(log_softmax(torch.tensor([1000.0, 1001.0, 1002.0])))
# tensor([-2.4076, -1.4076, -0.4076])
$$

PyTorch 的 `F.log_softmax`、`cross_entropy` 内部已经做了 log-sum-exp 处理，**不要自己再 softmax 后取 log**——那样既慢又不稳。

## 三、矩阵条件数与线性方程组稳定性

求解 $A \mathbf{x} = \mathbf{b}$ 时，**条件数** $\kappa(A) = \|A\| \cdot \|A^{-1}\|$ 衡量"右端微小扰动引起解的放大倍数"：

- $\kappa$ 接近 1：良态。
- $\kappa \gg 1$：病态，数值解可能完全不可信。

```python
import numpy as np
A = np.array([[1.0, 1.0], [1.0, 1.0000001]])
b = np.array([2.0, 2.0000001])
print(np.linalg.cond(A))          # 1.6e7 —— 极端病态
print(np.linalg.solve(A, b))      # 看起来正常，但稍微扰动 b 结果就剧变
```

这与 ML 关系密切：

- **梯度消失/爆炸**：深度网络反向传播等价于连续乘以权重矩阵，条件数极大。
- **Adam 的分母** $\sqrt{v_t} + \epsilon$ 实际是在改善优化问题的条件数。
- **BatchNorm / LayerNorm**：通过标准化激活，把每层输入的条件数控制住。

## 四、梯度消失与梯度爆炸

深层网络中，反向传播是连乘雅可比矩阵：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} \prod_{\ell=2}^{L} \frac{\partial \mathbf{h}_\ell}{\partial \mathbf{h}_{\ell-1}}
$$

若每个雅可比的谱范数 $\rho_\ell < 1$，梯度按指数缩小（**消失**）；若 $\rho_\ell > 1$，按指数放大（**爆炸**）。常见缓解手段：

- **激活函数**：ReLU 替代 sigmoid（正区间导数恒为 1）。
- **归一化**：BatchNorm、LayerNorm、GroupNorm。
- **残差连接**：$\mathbf{h}_{\ell+1} = \mathbf{h}_\ell + F_\ell(\mathbf{h}_\ell)$ 让雅可比近似有恒等映射，保留梯度通路。
- **梯度裁剪**：$\|\nabla\| > C$ 时按比例缩放。
- **合适的初始化**：Xavier / Kaiming 初始化让激活和梯度方差逐层大致守恒。

```python
# Kaiming 初始化（PyTorch 默认 Conv2d）
torch.nn.Conv2d(in_ch, out_ch, kernel_size=3).weight  # 默认 kaiming_uniform_

# Xavier 初始化
torch.nn.init.xavier_uniform_(linear.weight)
```

## 五、混合精度训练

用 fp16 做前向和梯度计算能省一半显存、提速 1.5–3 倍，但 fp16 数值范围窄、易下溢。标准做法：

1. **维护 fp32 主权重**（"master weights"），每步把 fp32 → fp16 做前向。
2. **损失缩放**（loss scaling）：把 loss 乘以 $S$，让梯度也放大 $S$ 倍，避免在 fp16 下溢。反向传播后除以 $S$ 再更新。
3. **跳过/裁剪** fp16 下的 inf/nan 梯度。

```python
scaler = torch.amp.GradScaler('cuda')
for x, y in loader:
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**bf16**（bfloat16，1+8+7 位）是 Google 提出的折中：与 fp32 同 8 位指数（数值范围一致），但只有 7 位尾数（精度低于 fp16）。bf16 无需 loss scaling，对 Transformer 训练非常友好——很多新一代框架默认 bf16。

## 六、常见数值陷阱清单

- **log(0) = -inf**：概率计算要加 $\epsilon$ 或 clamp。
- **除以接近 0 的数**：Adam 的分母有 $\epsilon$ 就是为此。
- **Softmax 后取 log**：`F.log_softmax(x)` 优于 `torch.log(F.softmax(x))`。
- **大 Batch + 大学习率**：loss 表面更平坦但容易尖峰，配合 warmup + cosine。
- **不同 dtype 混算**：fp16 权重与 fp32 优化器状态不能直接相加，必须先转回。
- **整数索引溢出**：embedding 表很大时用 `int64` 索引。

```python
# 反例：先 softmax 再 log
logits = torch.tensor([1000.0, 1001.0])
torch.log(torch.softmax(logits, dim=-1))    # 可能产生 -inf
# 正例：直接 log_softmax
F.log_softmax(logits, dim=-1)               # 数值安全
```

## 小结

数值计算不是"玄学"——它有清晰的数学：IEEE 754 决定表示范围，条件数决定稳定性，log-sum-exp 让你在大 logits 下不爆炸，混合精度 + loss scaling 在显存与稳定之间取得平衡。掌握这些，你就能写出"在不同硬件、不同 batch size、不同模型规模下都跑得稳"的训练代码，也能更快定位"训练过程中突然出现 NaN"这类棘手 bug。