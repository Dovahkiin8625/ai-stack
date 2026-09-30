# 反向传播：神经网络的"求导引擎"

## 一、为什么需要反向传播

训练神经网络就是最小化损失 $L(\theta)$。梯度下降需要 $\nabla_\theta L$，但直接对每个参数"数值微分"在 $N$ 个样本上扫一遍，复杂度是 $O(|\theta| \times N)$——对百万级参数根本不可行。

**反向传播（Backpropagation）** 由 Rumelhart、Hinton、Williams 在 1986 年系统化，本质是**链式法则的高效应用**：在前向传播时记录中间结果，反向时从输出到输入一次把梯度"传"回去，复杂度与一次前向传播同阶 $O(|\theta|)$。

## 二、链式法则回顾

设 $y = f(g(x))$，则：

```math
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
```

多元与向量版本类似：Jacobian 矩阵按链式相乘。反向传播做的事就是：**从最外层往里，每一步乘上局部导数**。

## 三、计算图：把函数表达成 DAG

把前向传播拆成基本运算的有向无环图（DAG），反向传播就是沿 DAG 反向遍历。

例：$f(x, y, z) = (x + y) \cdot \sigma(z)$，其中 $a = x + y$, $b = \sigma(z)$, $f = a \cdot b$：

```text
前向传播（从左到右）

    x ──┐
        ├──► a = x + y ──┐
    y ──┘                ├──► f = a · b
                       │
    z ──────► b = σ(z) ─┘

反向传播（从右到左，逐节点累加 ∂f/∂·）
```

设最终 $\partial f / \partial f = 1$，则：

- $\partial f / \partial a = b$
- $\partial f / \partial b = a$
- $\partial f / \partial z = a \cdot \sigma'(z)$
- $\partial f / \partial x = \partial f / \partial a = b$
- $\partial f / \partial y = b$

每条边的局部导数在前向时就可以算出来，反向时只需做"乘 + 累加"。

## 四、一个两层网络的完整例子

考虑：

```math
h = \sigma(W_1 x + b_1),\quad \hat{y} = W_2 h + b_2,\quad L = \tfrac{1}{2}(\hat{y} - y)^2
```

**前向**（假设批量大小 1）：

```text
x ──► z1 = W1·x + b1 ──► h = σ(z1) ──► z2 = W2·h + b2 ──► ŷ ──► L = ½(ŷ−y)²
```

**反向**：从 $\partial L / \partial L = 1$ 出发，按链式法则往回传：

```math
\delta_2 = \frac{\partial L}{\partial z_2} = (\hat{y} - y)               \quad \text{（输出误差）}
```

```math
\delta_1 = \frac{\partial L}{\partial z_1} = (W_2^\top \delta_2) \odot \sigma'(z_1) \quad \text{（隐藏误差）}
```

梯度：

```math
\frac{\partial L}{\partial W_2} = \delta_2 \cdot h^\top,\quad \frac{\partial L}{\partial b_2} = \delta_2
```

```math
\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^\top,\quad \frac{\partial L}{\partial b_1} = \delta_1
```

直觉：**每一层的误差** $\delta_l$ 就是"我对下游错误该负多少责"，乘上自己的输入得到对参数的梯度。

## 五、四个核心方程（BPD）

对于第 $l$ 层，定义误差信号 $\delta^{(l)} = \partial L / \partial z^{(l)}$（预激活处的梯度），有四个等价表述：

```math
\text{BP1（输出层误差）} \quad \delta^{(L)} = \nabla_{\hat{y}} L \odot \sigma'_L(z^{(L)})
```

```math
\text{BP2（反向传播）} \quad \delta^{(l)} = \left( W^{(l+1)\top} \delta^{(l+1)} \right) \odot \sigma'_l(z^{(l)})
```

```math
\text{BP3（权重梯度）} \quad \frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \, a^{(l-1)\top}
```

```math
\text{BP4（偏置梯度）} \quad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
```

**BP1** 从损失开始；**BP2** 把误差"传"回上一层（注意是 $W^\top$ 而非 $W$）；**BP3/4** 把误差乘上对应输入得到参数梯度。这就是"反向传播"名字的由来——误差信号 $\delta$ 像波一样从输出层往输入层传播。

## 七、矩阵形式 vs 元素形式

**元素形式**（教学清晰）：

```math
\delta^{(l)}_i = \sum_j W^{(l+1)}_{ji} \delta^{(l+1)}_j \cdot \sigma'(z^{(l)}_i)
```

**矩阵形式**（实现高效）：对每个样本独立计算，最后用批量平均（或求和）。

```python
# 假设 batch 形状 (B, d_l)
delta_l = (W_l_plus_1.T @ delta_l_plus_1) * sigma_prime(z_l)   # (B, d_l)
grad_W_l = delta_l.T @ a_{l-1} / B                              # (d_l, d_{l-1})
grad_b_l = delta_l.mean(dim=0)                                  # (d_l,)
```

PyTorch / TensorFlow 的 autograd 正是按这种矩阵粒度组织的——避免逐元素循环。

## 九、梯度消失与爆炸

观察 BP2：$\delta^{(l)} = W^{(l+1)\top} \delta^{(l+1)} \odot \sigma'(z^{(l)})$。如果每一层都把误差放大（或缩小）一个常数因子 $\rho$，传到第 $l$ 层就是 $\rho^{L-l}$ 的量级：

- **$\rho > 1$**：误差被指数放大 → **梯度爆炸** → 训练 loss 突然变成 NaN。
- **$\rho < 1$**：误差被指数缩小 → **梯度消失** → 前几层几乎不更新。

叠加 sigmoid $\sigma'(z) \le 0.25$ 这一"天然衰减因子"，传统 sigmoid 网络很难训练深——这是 2010 年之前深度网络难以加深的关键原因。

### 训练曲线上看症状

```text
loss
 │  ╲
 │   ╲─── 正常（平滑单调下降）
 │
 │
 └─────────────────── step

vs

 │   ╱╲
 │  ╱  ╲╱╲──╱╲
 │ ╱        ╲─── 梯度爆炸：loss 抖动 / NaN
 │
 └─────────────────── step

vs

 │─────╲
 │      ╲____________ 梯度消失：loss 在高位平台，迟迟不下降
 │
 └─────────────────── step
```

### 解决方案

- **更好的激活**（ReLU 家族：$\sigma'(z) = \mathbf{1}_{z>0}$，梯度不会饱和）。
- **归一化**（BatchNorm、LayerNorm）：把激活值拉回"合理区间"。
- **残差连接**（ResNet）：$a^{(l+1)} = a^{(l)} + \mathcal{F}(a^{(l)})$，梯度多一条"短路"路径。
- **梯度裁剪**（Gradient Clipping）：`torch.nn.utils.clip_grad_norm_`。
- **合适的初始化**：Xavier / Kaiming 让每层前向方差、反向方差都接近 1。

## 十、梯度检查：用数值微分验证 autograd

写自己的 autograd 时几乎一定会有 bug。**梯度检查（gradient checking）** 用数值微分作为"金标准"：

```python
import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in np.nditer(np.index_exp[x.shape], flags=['multi_index']):
        idx = x.multi_index
        old = x[idx]
        x[idx] = old + eps
        fx_plus = f(x)
        x[idx] = old - eps
        fx_minus = f(x)
        x[idx] = old
        grad[idx] = (fx_plus - fx_minus) / (2 * eps)
    return grad

def analytic_gradient(x):
    # 例如 f(x) = (x * 3 + 1)^2，df/dx = 6(x*3+1)
    return 6 * (x * 3 + 1)

x = np.random.randn(3)
g_num = numerical_gradient(lambda v: ((v*3+1)**2).sum(), x)
g_ana = analytic_gradient(x)
print('相对误差:', np.linalg.norm(g_num - g_ana) / np.linalg.norm(g_num + 1e-8))
# 通常 < 1e-6 表示实现正确
```

工程实践中只在小模型 / 小数据上跑梯度检查，因为数值微分是 $O(|\theta|)$ 次前向。**确认通过后再切回 analytic 梯度**。

## 十一、PyTorch autograd：自动建图

现代框架把"前向建图 + 反向传播"全部自动化。我们对比"手算"与"自动"两种方式：

```python
import torch

# === 自动 autograd：PyTorch 在 forward 时建图，.backward() 时反向 ===
x = torch.randn(4, 784, requires_grad=True)
W1 = torch.randn(784, 256, requires_grad=True)
b1 = torch.zeros(256, requires_grad=True)
W2 = torch.randn(256, 10,  requires_grad=True)
b2 = torch.zeros(10,  requires_grad=True)

h = torch.relu(x @ W1 + b1)
logits = h @ W2 + b2
y = torch.tensor([3, 7, 1, 9])
loss = torch.nn.functional.cross_entropy(logits, y)

loss.backward()    # 自动反向：计算所有 requires_grad=True 的 tensor 的梯度
print(W1.grad.shape, b2.grad.norm())   # (784, 256), scalar

# === 手动验证：手算 BP2 看梯度对不对 ===
# 我们手算 dL/dW2 = (softmax(logits) - one_hot(y)).T @ h / B
with torch.no_grad():
    probs = torch.softmax(logits, dim=1)
    onehot = torch.nn.functional.one_hot(y, num_classes=10).float()
    manual_grad_W2 = h.t() @ (probs - onehot) / 4
print('autograd vs manual 偏差:', (W2.grad - manual_grad_W2).abs().max())
```

```text
autograd vs manual 偏差: tensor(1.4e-07)   # 几乎只受浮点精度限制
```

这说明 autograd 与手算 BP 完全等价——但**前者你不用维护梯度公式**，改网络结构也无需重写求导。

### autograd 关键技巧

- `with torch.no_grad():` 上下文内不建图，省内存（推理 / 参数更新时）。
- `tensor.detach()` 切断梯度链。
- `loss.backward(retain_graph=True)` 保留中间节点，可多次反向（GAN、强化学习常用）。
- `register_hook` 在反向传播时检查 / 截断梯度。

## 十二、反向传播的几点"反直觉"

1. **梯度并不"消失"为零**，而是相对于参数尺度极小——Adam、RMSProp 这类自适应优化器能缓解但无法根治。
2. **梯度 ≠ 参数更新**。优化器（SGD、Momentum、Adam）拿到梯度后才决定怎么更新；二者可以独立设计。
3. **二阶导数（Hessian）也可以通过 BP 算**，但代价是 $O(|\theta|^2)$ 内存——所以 L-BFGS 等二阶方法仅用于小模型。
4. **梯度检查的数值微分不能与 autograd 同图混用**，否则会污染 requires_grad 状态。

## 小结

反向传播是"**链式法则 + 计算图反向遍历**"的高效实现：先前向记录中间结果，再从损失往输入"传"梯度，每一步只需乘局部 Jacobian。四个核心方程（BPD）给出了 $\delta$、$W$、$b$ 梯度的统一表述，是矩阵化实现的基础。梯度消失 / 爆炸源于多层连乘，是深度网络难训练的根本原因，需靠 ReLU、归一化、残差连接等手段缓解。PyTorch 的 autograd 把这一切自动化——你只需写 forward，`loss.backward()` 就能拿到所有梯度，并用梯度检查作为正确性"金标准"。掌握反向传播的直觉后，你能更精准地诊断训练失败（loss 不降？→ 检查梯度；NaN？→ 梯度爆炸），而不是盲目调参。