# 循环神经网络（RNN）基础

## 一、为什么序列数据需要专门架构

现实世界中很多数据天然是序列的——文本、语音、股票价格、视频帧、传感器读数。这些数据有三个 MLP 难以处理的特性：

- **变长**：句子长度不固定，MLP 的输入维度必须固定。
- **顺序敏感**：`"我打他"` 和 `"他打我"` 词语相同但语义相反，MLP 把输入"压平"后丢失了顺序。
- **长程依赖**：句子 `"昨天那个穿红裙子的、帮我修过电脑的女孩___来了"` 中，主语"女孩"和动词"来了"相隔很远。模型需要在数十步外仍能保留信息。

CNN 通过"局部连接 + 池化"获得空间平移不变性；RNN 则通过**递归**（同一组参数在时间步上反复使用）显式建模时间依赖。本文从最基础的 Vanilla RNN 开始，建立对序列建模的直觉。

## 二、Vanilla RNN：最朴素的循环单元

### 2.1 单元定义

RNN 在每个时间步 $t$ 用同一个函数更新隐状态：

$$
h_t = \tanh\!\left( W_{hh}\, h_{t-1} + W_{xh}\, x_t + b \right)
$$

- $x_t \in \mathbb{R}^{d}$：当前步输入向量。
- $h_t \in \mathbb{R}^{h}$：隐状态，相当于网络的"记忆"。
- $W_{xh} \in \mathbb{R}^{h \times d}$：输入到隐状态的投影。
- $W_{hh} \in \mathbb{R}^{h \times h}$：**隐状态到自身的递归权重**——这是 RNN 区别于 MLP 的核心。
- $b \in \mathbb{R}^{h}$：偏置。

输出层（任务相关）通常是 $y_t = W_{hy} h_t$。

### 2.2 时间展开图

把递归沿时间轴"画平"，得到一个等价的**前馈网络**（同一组权重被复用）：

```text
       x_1         x_2         x_3         x_4
       │           │           │           │
       ▼           ▼           ▼           ▼
      [RNN]────h_1 [RNN]────h_2 [RNN]────h_3 [RNN]───► ...
       ▲           ▲           ▲           ▲
       │           │           │           │
       h_0=0       h_1         h_2         h_3
       │           │           │           │
       y_1         y_2         y_3         y_4
```

每个方块都是**同一个网络**（同一套 $W_{xh}, W_{hh}, b$），只是输入不同。展开是理解 BPTT 的关键——把循环转成"很深的"前馈网络。

```python
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, d_in, d_hid):
        super().__init__()
        self.W_xh = nn.Linear(d_in, d_hid, bias=False)
        self.W_hh = nn.Linear(d_hid, d_hid, bias=False)
        self.b    = nn.Parameter(torch.zeros(d_hid))
    def forward(self, x):           # x: (B, T, d_in)
        B, T, _ = x.shape
        h = torch.zeros(B, self.W_hh.weight.shape[0], device=x.device)
        outputs = []
        for t in range(T):
            h = torch.tanh(self.W_xh(x[:, t]) + self.W_hh(h) + self.b)
            outputs.append(h)
        return torch.stack(outputs, dim=1), h
```

## 三、通过时间的反向传播（BPTT）

损失在每个时间步可加，例如 $L = \sum_t \ell(y_t, \hat{y}_t)$。对 $h_t$ 的梯度需要沿时间链反传：

$$
\frac{\partial L}{\partial h_t} \;=\; \frac{\partial L}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}
$$

而

$$
\frac{\partial h_k}{\partial h_{k-1}} \;=\; \text{diag}\!\left(1 - \tanh^2(\cdot)\right) \cdot W_{hh}
$$

每多走一步，就**乘一次 $W_{hh}$**（再乘一个 $\le 1$ 的 Jacobian 因子）。

### 3.1 截断 BPTT

完整 BPTT 在长序列上显存和数值都不可行，实际做法是**截断（truncated）BPTT**：每 $K$ 步（如 $K=64$）做一次梯度更新，梯度只在窗口内反传。代价：超过 $K$ 步的依赖无法被梯度直接学习，需要其他机制（LSTM 的 cell state、信息流上的短接）补偿。

## 四、梯度消失与爆炸：RNN 的根本缺陷

直觉：设 $\|W_{hh}\| \approx \rho$，则梯度幅值大致按 $\rho^{T-t}$ 缩放。

- $\rho < 1$：每步衰减，**远距离梯度消失**，$T-t$ 大时几乎为 0。
- $\rho > 1$：每步放大，**远距离梯度爆炸**，几步后变成 `NaN`。
- $\rho \approx 1$：理论上能保留，但实际很难精确调到。

```text
梯度幅值（log）

   │
   │  ×
   │   ×        ←  ρ > 1（爆炸，截断后失真）
   │    ××
   │      ×××
   │
   │  ──────────  ρ = 1（理想，几乎做不到）
   │
   │  ··
   │    ····
   │       ······
   │            ········      ←  ρ < 1（消失，远处几乎无信号）
   └──────────────────────► 时间步
```

**为什么换 ReLU 也救不了**：ReLU 让 Jacobian 是 0 或 1，能缓解消失，但只要 $W_{hh}$ 谱范数 $> 1$，仍会**爆炸**；只要 $< 1$，仍会**消失**。ReLU 改的是"激活函数的 Jacobian"，没有改"矩阵被反复乘"这一本质。

### 4.1 梯度裁剪：爆炸的临时解

实际训练中几乎必加：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

直觉：把所有参数梯度按 $\|g\|$ 缩放到不超过阈值 $\tau$，保留方向、抑制爆炸。它**不解决消失**，但让网络至少能训起来。

### 4.2 合理的初始化

- **正交初始化** $W_{hh} \sim \text{Orthogonal}$：让 $\rho \approx 1$，配合 $\tanh$ 时表现稳定。
- 遗忘偏置 $b$ 设为 1（对应 LSTM 的 forget bias），减少初始遗忘。

## 五、用 PyTorch 实现多层 RNN + 变长序列

真实场景里一个 batch 内序列长度不一，直接 padding 会浪费计算且引入虚假"零信号"。PyTorch 用 `PackedSequence` 把短序列压紧：

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNClassifier(nn.Module):
    def __init__(self, vocab, d_emb=64, d_hid=128, n_layers=2, n_classes=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_emb, padding_idx=0)
        self.rnn = nn.RNN(d_emb, d_hid, num_layers=n_layers,
                          batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(d_hid, n_classes)
    def forward(self, ids, lengths):
        # ids: (B, T_max), lengths: (B,)
        emb = self.emb(ids)
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)            # h: (n_layers, B, d_hid)
        logits = self.fc(h[-1])              # 取最后一层最终隐状态
        return logits
```

`pack_padded_sequence` 把每个序列的有效步数告诉 RNN，跳过 padding 步；`pad_packed_sequence` 再把输出对齐回统一张量。注意 `lengths` 必须是**降序**或配合 `enforce_sorted=False` 使用。

## 六、复制任务：Vanilla RNN 的失败案例

**任务**：输入一段符号序列（如 `a, b, c, ...`），延迟若干步后原样输出。延迟越长，要求保留的信息越久。

```text
输入:   a  b  c  d  e  f  g  h  i  j  (空白)  (空白)  ...
标签:   _  _  _  _  _  _  _  _  _  _   a      b      c
                                          ↑
                                       10 步后开始输出
```

实验观察（Hochreiter & Schmidhuber 1997 经典结论）：

| 延迟步数 | Vanilla RNN | LSTM |
|---|---|---|
| $\le 10$ | ✅ 多数能学会 | ✅ |
| $20$ | ⚠️ 训练失败 / 准确率随机 | ✅ |
| $50+$ | ❌ 完全学不会 | ⚠️ 也开始困难 |

原因正如第四节：梯度在 $\rho^k$ 下指数衰减，远处 token 的信号传不到输出端。这正是 LSTM 通过**加性细胞状态**解决的——下一篇文章展开。

## 七、小结

Vanilla RNN 用同一组参数在时间上递归，把序列建模变成"沿时间展开的深度网络"。它的核心缺陷是**梯度消失/爆炸**：每经过一步梯度就乘一次 $W_{hh}$，$k$ 步后梯度幅值大致按 $\|W_{hh}\|^k$ 变化，使长程依赖无法学习。工程上用正交初始化、tanh 激活、梯度裁剪、截断 BPTT 能让短序列训练稳定，但**根本的解决需要架构创新**——下一篇文章介绍的 LSTM 与 GRU 通过"门控 + 加性记忆"绕过这个问题。理解 RNN 的失败模式，是理解为什么注意力机制最终会取代它（Transformer 那篇）的关键铺垫。
