# 偏差-方差与正则化

## 一、偏差-方差分解

监督学习的目标是最小化期望泛化误差 $\mathbb{E}[(y - f(x))^2]$。把 $f(x)$ 用不同训练集训练得到的"平均预测" $\bar{f}(x)$ 表示，可以分解为：

$$
\mathbb{E}\!\left[ (y - f(x))^2 \right] = \underbrace{(\bar{f}(x) - y)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(f(x) - \bar{f}(x))^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}
$$

直觉：

- **偏差**：模型平均预测与真实值之间的差距，反映**模型假设的强弱**。线性模型对真实非线性关系的偏差大。
- **方差**：不同数据集训练出的模型差异，反映**模型对数据的敏感度**。深度网络、k-NN 在小数据上方差大。
- **不可约噪声**：数据本身的随机性，无法通过模型消除。

模型复杂度与误差的关系（经典图）：

```
总误差 ──────╮
              │ ╲      ↑ 方差
              │  ╲    │
              │   ╲  ─┘
              │    ╲
              │     ╲
              │      ╲___
              │     ╱
              │   ╱   ↑ 偏差
              └────────────→ 模型复杂度
                最优点
```

**关键洞察**：最优模型复杂度是偏差与方差的**平衡点**，不是越大越好。

```python
# 用 sklearn 生成数据演示
from sklearn.model_selection import learning_curve
import numpy as np
train_sizes, train_scores, val_scores = learning_curve(
    estimator, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error'
)
```

## 二、过拟合与欠拟合

- **过拟合（high variance）**：训练误差 << 验证误差。模型"记住"训练集但学不到规律。
- **欠拟合（high bias）**：训练误差 ≈ 验证误差但都很高。模型能力不足。

诊断与处理：

| 现象 | 处理 |
|---|---|
| 训练 99% / 验证 65% | 加正则、增数据、降复杂度、Dropout |
| 训练 70% / 验证 68% | 升复杂度、换模型、增特征 |
| 训练 99% / 验证 80%（差距大） | 早停、增数据、Label smoothing |
| 训练 80% / 验证 60%（都一般） | 特征工程、换损失、检查数据质量 |

## 三、L1 / L2 / Elastic Net：最经典的正则

把模型参数 $\theta$ 的"大小"加入损失：

$$
L_{\text{reg}}(\theta) = L(\theta) + \lambda \cdot \Omega(\theta)
$$

**L2 正则（Ridge / Weight Decay）**：$\Omega(\theta) = \|\theta\|_2^2$。

- 几何：把可行域约束在 $\ell_2$ 球内。
- 效果：让权重普遍小、平滑、稠密。
- 梯度：$\nabla \Omega = 2\theta$，在每次更新时"缩小"权重。

**L1 正则（Lasso）**：$\Omega(\theta) = \|\theta\|_1 = \sum_i |\theta_i|$。

- 几何：把可行域约束在 $\ell_1$ 菱体内，菱形顶点更可能落在坐标轴上。
- 效果：**稀疏**——很多权重变 0，自动做特征选择。
- 梯度：$\nabla \Omega = \text{sign}(\theta)$。

**Elastic Net**：$\Omega = \alpha \|\theta\|_1 + (1-\alpha)\|\theta\|_2^2$，结合两者优点。

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)
enet  = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
```

**$\lambda$ 选取**：从大到小扫描，配合交叉验证。`scikit-learn` 的 `RidgeCV`、`LassoCV` 自动做。

**深度学习中的 L2**：也叫 **weight decay**。PyTorch 中：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

注意 AdamW 把 weight decay 从梯度中解耦，对 Transformer 训练更稳定。

## 四、Dropout：神经网络的"集成"技巧

训练时每个神经元以概率 $p$ 被随机置零：

$$
\mathbf{h}^{(l)} = \sigma(W^{(l)} \cdot (\mathbf{m}^{(l)} \odot \mathbf{h}^{(l-1)}) + b^{(l)})
$$

其中 $\mathbf{m}^{(l)} \sim \text{Bernoulli}(1-p)$。

**为什么有效**：

1. 阻止神经元共适应（co-adaptation），让每个神经元独立有用。
2. 等价于**模型集成**——每次前向是一个不同的子网络，最终预测是它们的平均。
3. 实现简单，计算便宜。

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))
```

**推理时**：不用 Dropout，但要把权重乘以 $1-p$（"inverted dropout"已自动做了）。

**变体**：

- **DropConnect**：随机置零权重而非激活。
- **Spatial Dropout**：卷积特征图整个通道置零。
- **DropPath / Stochastic Depth**：残差网络中随机跳过整个分支。
- **DropBlock**：卷积特征图中按块置零。

## 五、数据增强：从源头扩大数据

训练时对输入做随机变换，生成新的"训练样本"：

- **图像**：随机裁剪、翻转、旋转、色彩抖动、Cutout、Mixup、Cutmix。
- **文本**：同义词替换、回译、随机插入/删除。
- **音频**：加噪、变速、变调、SpecAugment（频谱遮蔽）。

```python
from torchvision import transforms
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

**Mixup** 把两张图按 $\lambda$ 线性混合，标签也按 $\lambda$ 混合：

$$
\tilde{x} = \lambda x_i + (1-\lambda) x_j,\quad \tilde{y} = \lambda y_i + (1-\lambda) y_j
$$

**Cutmix** 用一张图的矩形区域替换另一张图的对应区域，按面积比例混合标签。两者都能显著提升泛化、降低过拟合。

## 六、早停（Early Stopping）

最简单却最有效的正则化：在验证集性能开始下降时停止。

```python
best_val = float('inf')
patience, stale = 5, 0
for epoch in range(max_epochs):
    train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(model)
        stale = 0
    else:
        stale += 1
        if stale >= patience:
            break   # 触发早停
```

深度学习中"早停 + AdamW + Dropout"是常见的"三件套"正则化组合。

## 七、其他常用正则技术

**BatchNorm / LayerNorm**：归一化激活，控制层间分布，等价于一种隐式正则。

**Label Smoothing**：把硬标签 $[0, 1, 0]$ 变成 $[0.05, 0.9, 0.05]$，防止模型对预测过度自信。分类任务标准技巧。

$$
y^{\text{LS}}_i = (1 - \epsilon) \cdot y_i + \epsilon / K
$$

**Noise Injection**：输入加高斯噪声或权重加噪（"SGD 本身就提供隐式正则"）。

**Adversarial Training**（FGSM / PGD）：用梯度构造最坏扰动，提升鲁棒性。某种意义上是"最聪明的数据增强"。

**SAM（Sharpness-Aware Minimization）**：寻找损失平坦极小值：

$$
\min_\theta \max_{\|\epsilon\| \le \rho} L(\theta + \epsilon)
$$

## 八、容量、样本量与泛化界

经典 PAC 理论给出大致关系：

$$
\text{泛化误差} \lesssim O\!\left( \sqrt{\frac{C \cdot \log N}{N}} \right)
$$

$C$ 是模型容量（VC 维、Rademacher 复杂度）。直观：

- 模型越复杂，$C$ 越大。
- 样本越多，泛化越好。
- 增加数据可以"赎买"更复杂的模型。

这是"scale law"的思想源头——更大的模型 + 更多的数据 = 更好的泛化。但需配合合适的正则化，否则大模型会过拟合。

## 九、实用指南：什么正则用哪个？

| 场景 | 推荐 |
|---|---|
| 线性模型 / 浅层 MLP | L1/L2、ElasticNet |
| 深度 MLP / CNN | Dropout + 数据增强 + 早停 |
| Transformer | AdamW + Label Smoothing + 早停 |
| 小数据 + 复杂模型 | Dropout + 数据增强 + L2 + 早停 |
| 大数据 + 大模型 | 几乎不需要 Dropout，靠数据规模与合适架构 |

**经验**：先把 L2、Dropout、早停、数据增强这"四大件"用好，再考虑更高级技巧（SAM、Adversarial）。

## 小结

偏差-方差分解是理解泛化的"根公式"；正则化通过限制模型复杂度或注入噪声来平衡偏差与方差。L1/L2 是经典线性正则，Dropout 是神经网络的事实标准，数据增强是从源头扩大样本，早停是性价比最高的"动态正则"。掌握这套组合拳，你就能在面对过拟合时游刃有余——而不是盲目堆数据或换更大的模型。