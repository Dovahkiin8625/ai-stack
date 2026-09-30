# 正则化与归一化：控制过拟合与层间分布

训练深度网络最常遇到的两类问题是「训练集表现好、验证集差」（过拟合）和「梯度消失/爆炸、训练不收敛」（层间分布漂移）。前者靠正则化解决，后者靠归一化解决。本文系统梳理 L1/L2、Dropout、早停、Label Smoothing 等正则技巧，以及 BatchNorm/LayerNorm/InstanceNorm/GroupNorm 的差异与适用场景，最后给出 PyTorch 实战。

## 一、正则化总览

正则化的目标是在**偏差-方差**之间取平衡——限制模型复杂度或对训练过程注入噪声，让训练误差和验证误差都尽可能小。

| 手段 | 作用对象 | 典型强度 |
|---|---|---|
| L1 / L2 / Elastic Net | 参数 | $\lambda \in [10^{-4}, 10^{-1}]$ |
| Dropout / DropPath | 激活 | $p \in [0.1, 0.5]$ |
| 早停 (Early Stopping) | 训练流程 | patience = 5~20 |
| 数据增强 | 输入分布 | 任务相关 |
| Label Smoothing | 标签分布 | $\epsilon \in [0.05, 0.1]$ |
| 权重约束 / SAM | 参数 | 视任务 |

下面逐项展开。

## 二、L1 / L2 / Elastic Net：参数层面的正则

把参数"大小"加入损失函数：

```math
L_{\text{reg}}(\theta) = L(\theta) + \lambda \cdot \Omega(\theta)
```

**L2 正则（Ridge / Weight Decay）**：$\Omega(\theta) = \|\theta\|_2^2$。

- 贝叶斯视角：等价于参数先验为高斯分布。
- 梯度：$\nabla \Omega = 2\theta$，每步把权重"拉向 0"。
- 效果：权重普遍小、平滑、稠密。

**L1 正则（Lasso）**：$\Omega(\theta) = \|\theta\|_1 = \sum_i |\theta_i|$。

- 贝叶斯视角：等价于参数先验为 Laplace 分布。
- 几何：把可行域约束在 $\ell_1$ 菱体内，菱形顶点更易落在坐标轴上。
- 效果：**稀疏**——很多权重变 0，自动做特征选择。

**Elastic Net**：$\Omega = \alpha \|\theta\|_1 + (1-\alpha) \|\theta\|_2^2$。

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)
enet  = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
```

**深度学习中的 weight decay**：直接传给优化器。**AdamW 把 weight decay 从梯度中解耦**（详见优化器篇），对 Transformer 训练更稳定：

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=0.1
)
```

## 三、Dropout：训练时随机丢弃激活

训练时每个神经元以概率 $p$ 被置零：

```math
\mathbf{h}^{(l)} = \sigma(W^{(l)} \cdot (\mathbf{m}^{(l)} \odot \mathbf{h}^{(l-1)}) + b^{(l)})
```

其中 $\mathbf{m}^{(l)} \sim \text{Bernoulli}(1-p)$。

**为什么有效**：

1. **阻止共适应**：不让某些神经元只依赖另一些神经元"搭便车"。
2. **隐式集成**：每次前向是一个随机子网络，最终预测是这些子网络的几何平均。
3. **注入噪声**：等价于一种数据噪声，提供隐式正则。

**Inverted Dropout**（PyTorch 的实现方式）：训练时除以 $1-p$，推理时直接用，无需额外缩放：

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.drop = nn.Dropout(p=0.5)        # 训练置零 + 自动除以 (1-p)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))

model.eval()    # 推理时 Dropout 自动关闭
```

**Dropout 率的经验值**：输入层 $p \in [0, 0.1]$、隐藏层 $p \in [0.3, 0.5]$、输出层 $p = 0$。

**变体**：

| 名称 | 随机对象 | 典型场景 |
|---|---|---|
| Dropout | 激活 | MLP/CNN |
| SpatialDropout | 卷积通道整组置零 | CNN |
| DropPath / Stochastic Depth | 残差分支 | ResNet、ViT |

## 四、早停：性价比最高的"动态正则"

监控验证损失，连续若干 epoch 不再下降就停：

```python
best_val = float('inf')
patience, stale = 5, 0
for epoch in range(max_epochs):
    train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'best.pt')
        stale = 0
    else:
        stale += 1
        if stale >= patience:
            break
```

经验上"早停 + AdamW + Dropout"是 Transformer 训练的三件套组合。

## 五、数据增强与 Label Smoothing

**数据增强**：从源头扩大样本，等价于正则化。

- **图像**：随机裁剪、翻转、色彩抖动、Cutout、Mixup、Cutmix。
- **文本**：同义词替换、回译、随机插入/删除。

**Mixup** 把两张图按 $\lambda$ 线性混合，标签也按 $\lambda$ 混合：

```math
\tilde{x} = \lambda x_i + (1-\lambda) x_j,\quad
\tilde{y} = \lambda y_i + (1-\lambda) y_j
```

**Label Smoothing**：把 one-hot 标签 $[0, 1, 0]$ 替换为带均匀噪声的软标签：

```math
y^{\text{LS}}_i = (1 - \epsilon) \cdot y_i + \epsilon / K
```

阻止模型过度自信、提升校准度，对图像分类与语言模型都极有效：

```python
loss = F.cross_entropy(logits, target, label_smoothing=0.1)
```

## 六、其他正则技巧

- **Noise Injection**：输入或权重加高斯噪声。SGD 本身的小 batch 噪声就是一种隐式正则。
- **Adversarial Training**（FGSM / PGD）：用梯度构造最坏扰动，提升鲁棒性。
- **SAM（Sharpness-Aware Minimization）**：寻找损失平坦极小值，$\min_\theta \max_{\|\epsilon\| \le \rho} L(\theta + \epsilon)$。

```math
\min_\theta \max_{\|\epsilon\| \le \rho} L(\theta + \epsilon)
```

## 七、归一化：控制层间分布漂移

深度网络训练时，每层输入的分布会持续变化（Internal Covariate Shift），导致上层要不断适应——这就是「归一化」要解决的问题。

**Batch Normalization（BN）**：在 batch 维度上做归一化，是 CNN 的标配：

```math
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}},\quad
y = \gamma \hat{x} + \beta
```

$\mu_B$、$\sigma_B^2$ 是当前 mini-batch 上每个通道的均值与方差，$\gamma$、$\beta$ 是可学习缩放/偏移。

**BN 的两个隐患**：小 batch 下统计量噪声大（batch=1 方差为 0）；不适合 RNN——不同时间步统计量无法共享。

```python
nn.BatchNorm1d(num_features)   # (B, C, L)
nn.BatchNorm2d(num_features)   # (B, C, H, W)
```

**Layer Normalization（LN）**：在特征维度上做归一化，**不依赖 batch**，Transformer 的标配：

```math
\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}},\quad y = \gamma \hat{x} + \beta
```

$\mu_L$、$\sigma_L^2$ 是对**单个样本的所有特征**求均值方差。

```python
nn.LayerNorm(normalized_shape)   # Transformer 的标配
```

**Instance Normalization（IN）**：对每个样本、每个通道单独归一化，**风格迁移**领域的标配——直接抹去实例的对比度/风格信息。

**Group Normalization（GN）**：把通道分成 $G$ 组，每组内做 LN——BN 与 LN 的折中，**检测/分割小 batch 场景**的首选。

```python
nn.GroupNorm(num_groups=8, num_channels=32)
```

## 八、四种归一化对比

| 名称 | 归一化维度 | 统计量来源 | 典型场景 | 依赖 batch |
|---|---|---|---|---|
| BatchNorm | (B, H, W) 每通道 | mini-batch | CNN | 是 |
| LayerNorm | (C, H, W) 单样本 | 单样本 | Transformer | 否 |
| InstanceNorm | (H, W) 单样本单通道 | 单样本单通道 | 风格迁移 | 否 |
| GroupNorm | (C/G, H, W) 单样本分组 | 单样本分组 | 检测/分割 | 否 |

示意：把 $(B, C, H, W)$ 张量看成一摞卡片，每种归一化"扫过"的轴不同：

```
BN:  沿 B 求均值方差，每通道一组统计量
LN:  沿 C,H,W 求均值方差，每个样本一组
IN:  沿 H,W 求均值方差，每个 (B,C) 一组
GN:  把 C 分 G 组，组内沿 (C/G,H,W) 求均值方差
```

## 九、Pre-LN vs Post-LN

把归一化放在注意力/FFN 之前（Pre-LN）还是之后（Post-LN），对训练稳定性影响极大：

- **Post-LN**（原 Transformer）：LN 在残差之后。深层时梯度小，需要 warmup。
- **Pre-LN**（GPT-2、LLaMA）：LN 在子层之前。梯度干净，深层更稳定，几乎不需要精细 warmup。

现代 LLM 几乎默认 Pre-LN：

```python
class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp  = MLP(d)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

## 十、什么时候用什么正则与归一化

| 任务 | 推荐组合 |
|---|---|
| 浅层 MLP / 线性模型 | L1/L2 + 早停 |
| CNN 图像分类 | BN + Dropout + 数据增强 + weight decay |
| 目标检测/分割（小 batch） | GN + 数据增强 |
| Transformer / LLM | Pre-LN + AdamW + Label Smoothing + 早停 |
| 风格迁移 | InstanceNorm |
| 强过拟合风险 | Dropout + 数据增强 + L2 + 早停 |

## 小结

正则化通过**约束参数**（L1/L2）、**注入噪声**（Dropout、数据增强、Label Smoothing）或**控制训练流程**（早停）来抑制过拟合；归一化通过**控制层间输入分布**让优化更稳定、允许更大学习率。深度学习的事实配方是：CNN 用 BN + Dropout + 数据增强，Transformer 用 LN + AdamW + Label Smoothing。先把这些"默认件"配齐，再视任务需要引入 SAM、Mixup 等高级技巧——能覆盖绝大多数训练场景。
