# 激活函数与损失函数

## 一、为什么必须有非线性激活

设想把 $L$ 层"线性"网络堆起来：

```math
a^{(L)} = W^{(L)} W^{(L-1)} \cdots W^{(1)} x + \text{合并后的偏置}
```

这本质还是一个**线性变换**——再多层也只能拟合一条直线/平面/超平面，无法表示 XOR、圆、曲线等任何非线性关系。

**万能近似定理的前提是"非线性"**：没有激活函数的"深度"是假深度。激活函数决定了网络能表示什么、梯度能传多远。

## 二、Sigmoid：经典但有饱和问题

```math
\sigma(z) = \frac{1}{1 + e^{-z}},\quad \sigma'(z) = \sigma(z)(1 - \sigma(z))
```

```text
   σ(z)
 1.0 ┤          ___________________
     │       /
 0.5 ┤-----/--------------------- z
     │   /
 0.0 ┤__/
     -6    -3    0    3    6
```

特性：

- **输出范围** $(0, 1)$，可解释为概率。
- **平滑、可微**，但**不以零为中心**——这会让后一层的梯度更新呈现"之字形"，收敛变慢。
- **梯度消失**：当 $|z| > 4$，$\sigma'(z) < 0.01$，梯度几乎为零，深层训练困难。
- **不为零均值** → 后一层的输入总是正，$w$ 梯度符号恒定 → 只能"之字"逼近最优。

**仍在使用的场景**：

- 二分类**输出层**（与 BCE 损失搭配）。
- LSTM/GRU 中的**门控**（配合 tanh 把信号压到 $[-1, 1]$）。
- 注意力机制中**注意力分数**做归一（现已多被 softmax 替代）。

## 三、Tanh：零中心但仍饱和

```math
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}},\quad \tanh'(z) = 1 - \tanh^2(z)
```

```text
  tanh(z)
  1.0 ┤         ___________________
      │       /
  0.0 ┤------/------------------- z
      │   /
 -1.0 ┤__/
      -3    -1    0    1    3
```

- **零中心**：相比 sigmoid，Tanh 的输出均值接近 0，下一层梯度符号不再恒定。
- **仍是挤压型**：$|z| > 2$ 时梯度仍很小，深层仍有消失风险。

Tanh 在 RNN 中作为隐状态激活比 sigmoid 更常用；在 Transformer 中则几乎被 GeLU/ReLU 取代。

## 四、ReLU：深度学习的"工业标准"

```math
\text{ReLU}(z) = \max(0, z),\quad \text{ReLU}'(z) = \mathbf{1}_{z>0}
```

```text
  ReLU(z)
    │       /
    │      /
    │     /
    │    /
  ──┼───/─────── z
    │
```

特性：

- **正区间梯度恒为 1**，不会饱和——这是它能加速深度网络训练的关键。
- **计算极简**：一次比较 + 一次乘法（反向时）。
- **稀疏激活**：约一半神经元输出为 0，给网络引入隐式正则。

**Dying ReLU 问题**：若某神经元在所有训练样本上都 $z \le 0$，它就永远输出 0、永远不被更新——"死了"。常见诱因：学习率过大、偏置初始化不当。**解决**：Leaky ReLU、PReLU、ELU，或监控 dead neuron 比例。

```python
import torch
import torch.nn.functional as F

# 看 dead 比例
with torch.no_grad():
    h = F.relu(z)
    dead = (h == 0).float().mean()
    print(f'dead neuron ratio: {dead:.2%}')
```

## 五、ReLU 家族：解决 Dying 问题

| 函数 | 公式 | 特点 |
|---|---|---|
| **Leaky ReLU** | $\max(\alpha z, z)$，$\alpha \approx 0.01$ | 负区间有微小梯度，几乎不增参数量 |
| **PReLU** | $\max(\alpha z, z)$，$\alpha$ **可学** | 多个超参，多见于旧论文 |
| **ELU** | $z$ 若 $z>0$，$\alpha(e^z-1)$ 否则 | 负值平滑趋近 $-\alpha$，均值更接近 0 |
| **GeLU** | $z \cdot \Phi(z)$，$\Phi$ 标准正态 CDF | Transformer 默认（GPT、BERT），平滑且非零 |
| **Swish / SiLU** | $z \cdot \sigma(z)$ | Google 提出，深度模型略优于 ReLU |
| **Mish** | $z \cdot \tanh(\text{softplus}(z))$ | 自正则、平滑，YOLOv4 等用 |

```python
import torch.nn as nn
act = {
    'relu':     nn.ReLU(),
    'leaky':    nn.LeakyReLU(0.01),
    'prelu':    nn.PReLU(),
    'gelu':     nn.GELU(),                # approx='none' 更精确
    'silu':     nn.SiLU(),                # == Swish
    'mish':     nn.Mish(),
}
```

经验选择：

- **视觉 CNN**：ReLU 仍是默认（快、稳）。
- **Transformer / LLM**：**GeLU** 事实标准。
- **强化学习 / 生成模型**：SiLU / Mish 偶有惊喜。
- **遇到 dead neuron**：Leaky ReLU 是最便宜的修复。

## 六、输出激活：softmax 与 sigmoid

**二分类**：$\hat{y} = \sigma(z)$，输出一个标量概率。

**多分类（互斥）**：

```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

- 把 $K$ 个 logits 归一到"概率分布"（和为 1，互斥）。
- 直接与 **Cross-Entropy** 损失搭配最稳定。

**多标签（非互斥）**：每个类别独立二分类，$\hat{y}_i = \sigma(z_i)$。

```python
# 二分类
logit = model(x)                       # (B,)
prob  = torch.sigmoid(logit)           # (B,)
loss  = F.binary_cross_entropy(prob, y.float())

# 多分类
logits = model(x)                      # (B, K)
log_prob = F.log_softmax(logits, dim=1)
loss = F.nll_loss(log_prob, y)          # 或 F.cross_entropy(logits, y)
```

## 七、回归损失：MSE / MAE / Huber

**均方误差（MSE）**：

```math
L_{\text{MSE}} = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2
```

- 对异常值敏感（平方放大误差），梯度 $\propto$ 误差——远离最优点时步长大，可能震荡。
- 与高斯噪声假设一致：$y \mid x \sim \mathcal{N}(f(x), \sigma^2)$ 时，MSE 是 MLE。

**平均绝对误差（MAE）**：

```math
L_{\text{MAE}} = \frac{1}{N}\sum_i |\hat{y}_i - y_i|
```

- 对异常值鲁棒，但 $z=0$ 处不可导，训练末期梯度恒为 1 难收敛。

**Huber 损失**：小误差用平方、大误差用绝对值，平滑过渡。

```math
L_\delta(r) = \begin{cases}
\tfrac{1}{2} r^2 & |r| \le \delta \\
\delta(|r| - \tfrac{1}{2}\delta) & |r| > \delta
\end{cases}
```

回归任务上 Huber 通常更稳。

## 八、分类损失：交叉熵

**二分类交叉熵（BCE）**：

```math
L = -\frac{1}{N}\sum_i \left[ y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]
```

**多分类交叉熵**：

```math
L = -\frac{1}{N}\sum_i \sum_{k=1}^K y_{i,k} \log \hat{y}_{i,k}
```

其中 $\hat{y}_{i,k}$ 是 softmax 输出。**Cross-Entropy = LogLikelihood 的负数**——在多项分布假设下，最小化 CE 等价于 MLE。

```python
# PyTorch 三种写法等价
loss = F.cross_entropy(logits, y)                 # logits + 整数标签
loss = F.nll_loss(F.log_softmax(logits, 1), y)    # 显式两步
loss = -(F.log_softmax(logits, 1)[torch.arange(B), y]).mean()  # 手写版
```

数值稳定小技巧：`F.cross_entropy` 内部把 softmax + log + nll 合并（log-sum-exp 技巧），直接吃 logits，不会因 $e^{z}$ 溢出。

## 九、为什么分类用 CE + Softmax，不用 MSE + Sigmoid

直觉地，"分类问题应该用交叉熵"，但为什么？

设 $\hat{y} = \sigma(z)$，真实标签 $y \in \{0, 1\}$，$L_{\text{MSE}} = \tfrac{1}{2}(\hat{y} - y)^2$，对 $z$ 求梯度：

```math
\frac{\partial L_{\text{MSE}}}{\partial z} = (\hat{y} - y)\,\hat{y}(1-\hat{y})
```

当 $\hat{y} \to 0$ 或 $\hat{y} \to 1$（即预测很"自信"）时，$\hat{y}(1-\hat{y}) \to 0$ —— **梯度消失**，学习停滞，即便预测错了也无法纠正。

换成 sigmoid 输出 + 交叉熵：

```math
L_{\text{BCE}} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]
```

```math
\frac{\partial L_{\text{BCE}}}{\partial z} = \hat{y} - y
```

梯度就是**预测与真值的差**，**与 sigmoid 自身的饱和无关**。这就是"CE + Sigmoid"组合抗饱和的根本原因——也是 softmax + CE 的等价好性质。

简单规则：**分类用 CE，回归用 MSE**。不要混搭。

## 十、类别不平衡与 Focal Loss

正负样本 1:1000 时，普通 CE 会让模型全部预测为负类。常见处理：

- **加权 CE**：每类一个权重，少数类权重大。

```math
L = -\alpha_{y} \log \hat{y}_y
```

- **Focal Loss**（RetinaNet）：在 CE 上再乘 $(1 - \hat{y})^\gamma$，让"已经分对的样本"梯度变小。

```math
L_{\text{focal}} = -(1 - \hat{y})^\gamma \log \hat{y}
```

- **OHEM / class-balanced sampling**：训练时只取难样本 / 少数类过采样。

## 十一、完整代码片段：分类与回归

```python
import torch, torch.nn as nn, torch.nn.functional as F

# === 回归：MSE，输出层线性激活 ===
reg_model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1)            # logits
)
x = torch.randn(32, 20)
y = torch.randn(32, 1)
pred = reg_model(x)
loss_reg = F.mse_loss(pred, y)

# === 二分类：BCEWithLogits，输出层线性 ===
bin_model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 1)
)
y_bin = torch.randint(0, 2, (32, 1)).float()
logits = bin_model(x)
loss_bin = F.binary_cross_entropy_with_logits(logits, y_bin)   # 数值稳定

# === 多分类：CrossEntropy，输出层线性 ===
multi_model = nn.Sequential(
    nn.Linear(20, 64), nn.GELU(),
    nn.Linear(64, 10)
)
y_multi = torch.randint(10, (32,))
logits = multi_model(x)
loss_multi = F.cross_entropy(logits, y_multi)
```

注意**最后一层都是线性**（无激活），把激活交给损失函数处理——`cross_entropy`、`binary_cross_entropy_with_logits` 内部会做 softmax/sigmoid + log，保证数值稳定。

## 十二、激活与损失对照表

| 任务 | 输出激活 | 损失 | 备注 |
|---|---|---|---|
| 二分类 | sigmoid（损失内含） | BCEWithLogitsLoss | 单标量输出 |
| 多分类（互斥） | softmax（损失内含） | CrossEntropyLoss | logits + 整数标签 |
| 多标签 | sigmoid（每个标签） | BCEWithLogitsLoss | 每类独立 |
| 回归（高斯） | 线性 | MSELoss / SmoothL1 | 默认起点 |
| 回归（鲁棒） | 线性 | Huber / MAE | 含异常值时用 |
| 计数/泊松 | 线性 + softplus | PoissonNLLLoss | 离散计数 |

## 十三、常见陷阱

1. **输出层加了 softmax 又用 cross_entropy**：双重 softmax，概率被压扁、训练变慢甚至不收敛。`F.cross_entropy` 期望 logits 输入。
2. **回归任务最后用了 sigmoid**：把输出压到 $(0, 1)$，模型永远学不到范围外的真值。除非真值就在这个区间，否则用线性输出。
3. **多标签任务用 softmax**：softmax 强制各类和为 1，会让"猫 + 狗"互相抑制。改用每类独立的 sigmoid。
4. **类别极不平衡时直接用 CE**：所有样本都被预测为多数类。加权、Focal、采样三选一。
5. **ReLU 输出层**：会"截断"负值预测；回归且真值有正有负时用线性输出。

## 小结

激活函数决定网络的**表达能力与梯度传播**：ReLU 家族是深度学习的默认，Transformer 偏好 GeLU；Sigmoid/Tanh 因饱和问题已被挤到"门控 + 输出"角色。损失函数与任务强绑定：**回归用 MSE/Huber，分类用 CE**——并通过"损失内含激活"获得数值稳定性。理解"CE 配 Softmax"在梯度上更优的根本原因（不再饱和）后，你会避免常见的 MSE+Sigmoid 配错组合，并在类别不平衡时主动想到加权 CE 或 Focal Loss。这两者共同决定了网络能不能学、能不能稳、能不能泛化。