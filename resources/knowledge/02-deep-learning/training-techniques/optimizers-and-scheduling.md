# 优化器与学习率调度：从 SGD 到 AdamW

优化器的选择几乎决定了神经网络训练的成败：再好的架构配上不合适的优化器，也只会震荡不收敛或卡在鞍点。本文系统梳理从朴素 SGD 到 AdamW 的演进、SGD+Momentum 与 Nesterov 的物理直觉、AdaGrad/RMSProp/Adam 的自适应机制，以及 step/exponential/cosine/warmup 等学习率调度策略。最后给出 PyTorch 实战代码与 batch size 缩放、梯度累积等工程经验。

## 一、朴素 SGD 与它的两个问题

最朴素的随机梯度下降用单个 mini-batch 的梯度更新参数：

$$
w_{t+1} = w_t - \eta \cdot \nabla L(w_t)
$$

直觉是「沿最陡方向下山」，但实践中会暴露两个问题：

1. **病态曲率下走之字形**。当损失函数在不同方向上曲率差异巨大（一个方向陡、一个方向平）时，SGD 会反复在陡方向震荡，沿平方向缓慢前进，效率极低。
2. **陷入局部极小/鞍点**。深度网络的损失面充满鞍点，梯度接近零时朴素 SGD 直接停步，无法逃离。

```
SGD 在病态曲率上的轨迹              期望的轨迹
                                     →
      ↑                               ↑
       ↗                               ↗
      ↑ ↘                             →
       ↗                              →
      ↑ ↘
       ↗
   ─────→ 损失  ─────→              平滑下降到极小
   振荡但整体下降很慢                沿"variance"小的方向走
```

学习率 $\eta$ 的设置也很微妙——太大易震荡/发散，太小则训练极慢，且对所有参数共用同一个 $\eta$ 几乎注定不够灵活。

## 二、SGD with Momentum：给小球加惯性

动量法引入一个速度变量 $v_t$，把梯度看成"加速度"：

$$
v_t = \beta v_{t-1} + \nabla L(w_t)
w_{t+1} = w_t - \eta \cdot v_t
$$

常用 $\beta = 0.9$ 或 $0.99$。

**物理直觉**：把优化过程想成一个**在损失面上滚动的小球**。当前梯度提供瞬时加速度，但小球还有之前的速度惯性——这让小球能：

- 越过浅鞍点（梯度方向变化但速度方向不变）。
- 在陡方向积累反向速度，平滑掉之字形振荡。
- 在梯度变小的地方继续向前冲一段。

```
        损失面示意
       ╱╲      ╱╲
      ╱  ╲    ╱  ╲
     ╱    ╲──╱    ╲   ← 小球沿低曲率方向积累动量
    ╱     ╲╱      ╲
   ╱      ●        ╲
  ╱  起点      终点  ╲
```

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=False,   # 这里先关
)
```

## 三、Nesterov Accelerated Gradient（NAG）

朴素 Momentum「先沿累积速度走、再在到达点算梯度」，Nesterov 改成「先**预估**沿累积速度走一步、再在预估点算梯度修正」：

$$
v_t = \beta v_{t-1} + \nabla L(w_t - \eta \beta v_{t-1})
w_{t+1} = w_t - \eta \cdot v_t
$$

直觉：与其被惯性"推过头"，不如先用惯性"看一眼未来"，再根据未来梯度做修正——这是**lookahead**思想的最早实现之一。NAG 在凸问题上享有 $O(1/t^2)$ 的理论收敛率，比朴素 Momentum 的 $O(1/t)$ 更快。

```python
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, nesterov=True
)
```

## 四、AdaGrad：每个参数独立自适应学习率

SGD 与 Momentum 对所有参数共用一个 $\eta$，但**稀疏特征对应的参数很少被更新，稠密特征对应的参数常被更新**，理应有不同的步长。AdaGrad 为每个参数维护一个**梯度平方的累加和**：

$$
G_t = \sum_{\tau=1}^{t} g_\tau^2,\qquad
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t
$$

含义：被频繁更新的参数，$G_t$ 大，步长自动变小；不常更新的参数，$G_t$ 小，步长自动变大。适合**稀疏数据**（NLP 词嵌入、推荐系统）。

**致命缺陷**：$G_t$ 单调递增，学习率会一路衰减到接近 0，训练后期几乎不再更新——这就是 RMSProp 要解决的问题。

## 五、RMSProp：用指数滑动平均修复 AdaGrad

RMSProp 把 AdaGrad 的"全量累加"换成**指数加权移动平均（EMA）**，让历史梯度平方的影响指数衰减：

$$
v_t = \beta v_{t-1} + (1-\beta) g_t^2,\qquad
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t
$$

常用 $\beta = 0.99$。这样学习率能"自适应但不持续衰减"，是非凸深度网络的事实标准之一。

## 六、Adam：动量 + RMSProp 的合体

Adam 把 Momentum（梯度一阶矩）与 RMSProp（梯度二阶矩）合到一起，并加上**偏差修正**抵消初始零偏置：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t       \quad\text{（一阶矩估计）}
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2     \quad\text{（二阶矩估计）}
\hat{m}_t = \frac{m_t}{1 - \beta_1^t},\quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}          \quad\text{（偏差修正）}
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

默认 $\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。Adam 因其「默认即好用」的特性，迅速成为深度学习的默认优化器。

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
```

## 七、AdamW：把 weight decay 从梯度里解耦

Adam + L2 正则其实有一个微妙问题：L2 项的梯度会被 $\sqrt{\hat{v}_t}$ 一起缩放，**对不同参数正则强度不一致**。AdamW（Loshchilov & Hutter, 2019）直接把 weight decay 从梯度中拿掉，单独加在参数上：

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t - \eta \lambda \theta_t
$$

好处：

- weight decay 强度对所有参数一致。
- 训练更稳定，LLM 训练几乎都默认 AdamW。
- 与 LayerNorm、Pre-LN Transformer 配合极好。

```python
# Transformer / LLM 训练的标准配方
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,             # LLM 常用 1e-4 ~ 3e-4
    betas=(0.9, 0.95),
    weight_decay=0.1,    # LLM 偏好较大的 weight decay
)
```

## 八、常用优化器对比

| 优化器 | 额外状态/参数 | 核心思想 | 适用场景 |
|---|---|---|---|
| SGD | 无 | 沿负梯度下降 | 小模型、CV 经典网络、强凸问题 |
| SGD+Momentum | $v$ | 累积历史梯度，平滑震荡 | 几乎所有 CV 任务 |
| Nesterov SGD | $v$ | lookahead 修正 | 凸问题、想再快一点 |
| AdaGrad | $G$ | 每参数自适应步长 | 稀疏数据、NLP embedding |
| RMSProp | $v$（EMA） | AdaGrad 的指数平均版 | RNN、非平稳目标 |
| Adam | $m, v$ | 一阶+二阶矩 + 偏差修正 | 默认首选，通用 |
| AdamW | $m, v$ | Adam + 解耦 weight decay | **LLM/Transformer 训练标准** |

## 九、学习率调度：决定收敛曲线形状

即使选对了优化器，**学习率曲线**也至关重要。常见策略：

| 调度器 | 公式/行为 | 何时用 |
|---|---|---|
| Step Decay | 每 N epoch 乘 $\gamma$ | CV baseline，简单稳定 |
| Exponential | $\eta_t = \eta_0 \cdot \gamma^t$ | 长训练，平滑衰减 |
| Cosine Annealing | $\eta_t = \eta_{\min} + (\eta_0 - \eta_{\min}) \cdot \frac{1+\cos(\pi t/T)}{2}$ | 几乎所有现代训练 |
| Warmup | 前 $T_w$ 步线性从 0 升到 $\eta_0$ | LLM/大模型必配 |
| Cosine + Warmup | 两者组合 | 当前事实标准 |

**Cosine + Warmup** 几乎是当代 LLM 与视觉模型训练的默认搭配。Warmup 让优化器在前几百到几千步逐步进入正常工作状态，避免初期梯度爆炸；之后余弦退火让学习率平滑下降，末期做精细收敛。

ASCII 图示（横轴 step，纵轴 lr）：

```
lr
 │
 │       ╱─────╮
 │      ╱       ╲
 │     ╱         ╲
 │    ╱           ╲
 │   ╱             ╲
 │  ╱               ╲
 │ ╱                 ╲
 │╱                   ╲___
 └─────────────────────────→ step
   warmup    cosine annealing
   (0→η_max) (η_max → η_min)
```

```python
from torch.optim.lr_scheduler import (
    LambdaLR, CosineAnnealingLR, LinearLR, SequentialLR
)

warmup_steps = 1000
total_steps  = 10000
base_lr      = 3e-4

warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0,
                  total_iters=warmup_steps)
cosine = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps,
                           eta_min=base_lr * 0.01)
sched  = SequentialLR(opt, [warmup, cosine], milestones=[warmup_steps])

for step, batch in enumerate(loader):
    train_step(batch)
    sched.step()
```

## 十、Batch Size 缩放与梯度累积

**线性缩放规则**（Goyal et al., 2017, *Accurate, Large Minibatch SGD*）：当 batch size 扩大到 $k$ 倍时，学习率也线性扩大到 $k$ 倍。ImageNet 训练从 batch=256 扩到 batch=8192 时，lr 从 0.1 提到 0.8。

但实际硬件与显存常常不允许直接用超大 batch，可用**梯度累积**模拟：

```python
# effective_batch = micro_batch * accum_steps
accum = 8
opt.zero_grad()
for i, (x, y) in enumerate(loader):
    out = model(x.to(device))
    loss = criterion(out, y.to(device)) / accum
    loss.backward()       # 梯度累加
    if (i + 1) % accum == 0:
        opt.step()        # 等效一次大批次更新
        opt.zero_grad()
        sched.step()
```

注意：

- 累积时 BN 的统计量仍在每个 micro-batch 上算，**不要把 BN 跨 micro-batch 累积**。
- LayerNorm 不受影响。
- 较新研究（*Don't Decay the Learning Rate, Increase the Batch Size*）发现 batch 增大时调小 weight decay 效果更好。

## 十一、实战经验

- **Transformer/LLM**：AdamW + (lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)) + cosine + warmup + gradient clipping（`torch.nn.utils.clip_grad_norm_`）。
- **CV ResNet 家族**：SGD + momentum=0.9 + step decay（baseline 依然胜出），或 AdamW + cosine。
- **GAN**：Adam + lr=2e-4 + betas=(0.5, 0.999) 是经验最优（SGD+Momentum 通常崩）。
- **小数据集**：优先考虑 weight decay + 早停 + 小 lr，避免大步长在窄极小间跳。
- **监控梯度**：打印 `grad_norm` 与 `param_norm`，发现梯度爆炸/消失立即介入。

## 小结

优化器演进的主线是「共享学习率 → 动量平滑 → 自适应步长 → 解耦正则 → 与学习率调度协同」。朴素 SGD 至今仍是许多 CV baseline 的首选，AdamW 已成为 Transformer/LLM 训练的事实标准。配合 cosine + warmup 的学习率曲线与合理的 batch size 缩放，就能覆盖绝大多数深度学习训练任务。掌握这些「默认配方」之后，再针对具体问题微调——盲目换优化器收益有限，扎实监控（loss/grad norm/val 曲线）才是优化的根本。
