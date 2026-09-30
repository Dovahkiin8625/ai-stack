# 优化理论基础

## 一、优化问题的标准形式

一般优化问题可以写成：

```math
\min_{\mathbf{x} \in \mathcal{X}} \; f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \le 0,\; h_j(\mathbf{x}) = 0
```

其中 $f$ 是目标函数，$g_i$ 是不等式约束，$h_j$ 是等式约束，$\mathcal{X}$ 是变量的可行域。机器学习中 $\mathbf{x}$ 通常是模型参数 $\theta$，$f$ 是经验风险 $\frac{1}{N}\sum_{i=1}^N \ell(\theta; x_i, y_i)$。

当没有约束、且 $\mathcal{X} = \mathbb{R}^n$ 时，问题退化为无约束优化——这正是深度网络训练的主流形态。

## 二、凸 vs 非凸：为什么这么重要

一个集合 $C$ 是**凸集**，若对任意 $x, y \in C$ 和 $\lambda \in [0,1]$，都有 $\lambda x + (1-\lambda) y \in C$。一个函数 $f$ 是**凸函数**，若 $f(\lambda x + (1-\lambda) y) \le \lambda f(x) + (1-\lambda) f(y)$。

凸优化的核心性质：**任何局部极小都是全局极小**。这意味着：

- 线性回归、逻辑回归、SVM 的训练在理论上"很干净"，总能找到全局最优。
- 深度网络的损失是高度非凸的，存在大量鞍点和局部极小。这并不妨碍训练取得好效果（多数局部极小在泛化性能上差不多），但要求我们使用带动量、自适应学习率等"穿越"鞍点的技巧。

```python
import numpy as np
import matplotlib.pyplot as plt

# 凸函数：抛物线
f_convex   = lambda x: (x - 2) ** 2
# 非凸函数：双井
f_nonconvex = lambda x: x**4 - 4*x**2 + 0.5*x

x = np.linspace(-3, 3, 400)
plt.plot(x, f_convex(x), label='convex: (x-2)^2')
plt.plot(x, f_nonconvex(x), label='nonconvex: x^4 - 4x^2 + 0.5x')
plt.legend(); plt.show()
# 非凸曲线在 x ≈ -1.4 处有一个"伪极小"，但真正的全局极小在 x ≈ 1.9
```

## 三、Lagrange 与 KKT：带约束的极值

当模型有约束时（例如"权重范数 ≤ 1"做对抗鲁棒性，"输出分布接近先验"做 RLHF），我们需要 KKT（Karush-Kuhn-Tucker）条件。

构造 Lagrangian：

```math
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \mu_j h_j(\mathbf{x})
```

KKT 条件指出，最优解 $\mathbf{x}^*$ 必须满足：

1. **平稳性**：$\nabla_{\mathbf{x}} \mathcal{L} = 0$
2. **原始可行性**：$g_i(\mathbf{x}^*) \le 0,\; h_j(\mathbf{x}^*) = 0$
3. **对偶可行性**：$\lambda_i \ge 0$
4. **互补松弛**：$\lambda_i \cdot g_i(\mathbf{x}^*) = 0$

直观理解：约束要么"不起作用"（$g_i < 0$ 且 $\lambda_i = 0$），要么"紧贴边界"（$g_i = 0$ 且 $\lambda_i > 0$，相当于施加了一个推力）。

在 SVM 中，最大间隔分类问题通过 KKT 化简后，绝大多数样本的 $\lambda_i = 0$（不活跃），只有支持向量的 $\lambda_i > 0$——这正是 SVM 高效的根源。

## 四、一阶方法：SGD 与其变体

**随机梯度下降**是深度学习的"主力"：

```math
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta L(\theta_t; \mathcal{B}_t)
```

其中 $\mathcal{B}_t$ 是当前 mini-batch。SGD 的几个改进方向：

**Momentum**（动量）：把历史梯度指数加权累加，平滑震荡。

```math
v_{t+1} = \beta v_t + \nabla L(\theta_t), \quad \theta_{t+1} = \theta_t - \eta v_{t+1}
```

直觉：把梯度看作"力"，动量相当于"惯性"，能帮助穿越狭窄山谷和鞍点。

**Adam**（Adaptive Moment Estimation）：同时维护一阶矩 $m_t$ 和二阶矩 $v_t$ 的指数滑动平均，并用 $\sqrt{v_t}$ 缩放每个参数的学习率：

```math
m_{t+1} = \beta_1 m_t + (1-\beta_1) g_t,\quad
v_{t+1} = \beta_2 v_t + (1-\beta_2) g_t^2
```

```math
\hat{m}_{t+1} = \frac{m_{t+1}}{1-\beta_1^{t+1}},\quad
\hat{v}_{t+1} = \frac{v_{t+1}}{1-\beta_2^{t+1}}
```

```math
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}
```

其中 $\hat{m}, \hat{v}$ 是偏差修正。Adam 对稀疏梯度友好，是 Transformer 训练的事实标准。**AdamW** 在 Adam 基础上把权重衰减从梯度里解耦到参数更新里，对大模型预训练更稳定。

```python
# PyTorch 中 AdamW 的等价更新
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

## 五、二阶方法：为什么很少直接用

**Newton 法**利用 Hessian 矩阵 $H = \nabla^2 f$：

```math
\theta_{t+1} = \theta_t - H^{-1} \nabla f(\theta_t)
```

收敛速度是二次的（解附近的步数按平方递减），但 Hessian 是 $n \times n$ 矩阵，存储和求逆都是 $O(n^3)$，对亿级参数的模型完全不可行。

**拟 Newton 法**（L-BFGS）只维护低秩近似，内存 $O(n)$，但在大模型训练中仍不如 Adam 实用。

实践中，**二阶信息主要用于分析**——例如用损失景观的 Hessian 特征值理解泛化与锐度（Sharpness-Aware Minimization, SAM），而非直接优化。

## 六、学习率调度与训练技巧

- **Warmup**：训练初期学习率从 0 线性升到目标值，避免前几步梯度不稳定破坏参数。
- **Cosine decay**：$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$，平滑衰减。
- **梯度裁剪**：当 $\|\nabla L\| > C$ 时缩放到 $C$，防止梯度爆炸（尤其在 RNN、Transformer 中）。
- **混合精度训练**：用 fp16 做前向和梯度计算，fp32 维护主权重，节省显存、加速训练。

```python
scaler = torch.cuda.amp.GradScaler()
for x, y in loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

## 小结

优化理论把"训练模型"这件事从黑盒变成了可分析、可调控的工程：凸性决定能否找到全局最优，KKT 帮助处理约束，一阶方法（SGD/Momentum/Adam）是工业界主力，二阶信息主要用于理论分析。掌握这一组概念，你就具备了"看懂并改写训练循环"的能力。