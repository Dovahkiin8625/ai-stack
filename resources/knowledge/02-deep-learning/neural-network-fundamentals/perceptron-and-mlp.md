# 感知机与多层感知机（MLP）

## 一、从生物神经元到人工神经元

大脑中约 860 亿个神经元通过突触彼此相连，每个神经元接收来自其他神经元的电信号，当累计电位超过阈值时"放电"并向下游传递。1943 年 McCulloch 与 Pitts 把这一过程抽象成第一个**人工神经元模型**：

```text
        x1 ── w1 ─┐
        x2 ── w2 ─┤
        …        ├──► Σ ──► σ(·) ──► y
        xd ── wd ─┘   +
                   b
```

- **输入** $x \in \mathbb{R}^d$：来自上一层神经元的信号。
- **权重** $w \in \mathbb{R}^d$：突触"强度"，可正可负。
- **偏置** $b$：激活阈值。
- **激活函数** $\sigma$：决定是否放电。

这个简化抓住了三个关键：**线性加权、阈值判定、并行组合**。它也成了所有现代神经网络的最小单元。

## 二、单层感知机：能解什么、不能解什么

单层感知机是上面模型的"训练版本"——1958 年 Rosenblatt 提出感知机学习算法，用误分类样本更新权重：

$$
y = \sigma(w^\top x + b),\quad w \leftarrow w + \eta\, y (y_{\text{true}} - y)\, x
$$

其中 $\sigma$ 用阶跃函数（预测 +1 / -1）。它能用**梯度下降**等价的方式收敛——前提是数据线性可分。

### 2.1 XOR 困局

1969 年 Minsky 与 Papert 在《Perceptrons》中指出：单层感知机**解不了 XOR**。

```text
   x2
    │     o (0,0) → 0
    │     o (1,1) → 0
 1  │
    │  × (0,1) → 1
    │  × (1,0) → 1
    └───────────── x1
```

正负样本无法被一条直线分开。换成更一般的非线性函数（如 XOR）也不行——因为**单层感知机本质上是一个线性分类器**。

这一困局让神经网络研究陷入近 20 年低谷，直到 80 年代**反向传播 + 多层结构**带来复兴。

## 三、多层感知机（MLP）：堆叠解决问题

把多个感知机"层"堆叠起来，每层的输出作为下一层的输入，就得到**多层感知机（Multi-Layer Perceptron, MLP）**，也叫**前馈神经网络（Feedforward NN）**：

```text
        输入层           隐藏层            输出层
       (d 维)         (h1, h2 维)         (c 维)

  x1 ──► o ────────► o ──────► o ────────► o ──► y1
                   ╱   ╲   ╱   ╲
  x2 ──► o ─────► o ──► o ──► o ──────► o ──► y2
                   ╲   ╱   ╲   ╱
  …       …         …         …
  xd ──► o ────────► o ──────► o
```

每一层都是"线性变换 + 非线性激活"的复合。把足够多这样的层叠起来，网络就能表示任意复杂的函数——这正是**万能近似定理**的结论。

## 四、通用近似定理：为什么"足够宽"就够了

**定理（Cybenko 1989, Hornik 1991）**：一个具有**单隐藏层**、使用任意**挤压型**激活函数（如 sigmoid）的前馈网络，只要隐藏单元数量足够多，就能以任意精度逼近紧集上的任意连续函数。

直觉（不严谨版本）：

- 每个隐藏神经元可以看作在输入空间切出"一个凸区域"——比如 sigmoid 在 $w^\top x + b = 0$ 附近从 0 跳到 1。
- $H$ 个神经元就能用**凸区域组合**拟合任意形状的边界。
- 例如拟合 $f(x) = \sin(x)$：用 $H = 50$ 个 sigmoid "阶跃"叠加，效果已经很接近。

**重要限制**：

- 定理说"存在"足够宽的网络，但不保证我们能**训练**到它——梯度下降可能找不到。
- 实际中"深而窄"通常比"浅而宽"更易优化、更参数高效。
- 万能近似是**存在性**，不是**泛化保证**——网络也可能过拟合训练集。

## 五、前向传播：从输入到输出

对于一个有 $L$ 个带权层的 MLP，每一层的计算是：

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} \quad (\text{预激活})
$$

$$
a^{(l)} = \sigma_l(z^{(l)}) \quad (\text{激活})
$$

其中 $a^{(0)} = x$，$a^{(L)}$ 是最终输出。整个网络就是这些仿射 + 激活的复合：

$$
f(x) = \sigma_L\!\left( W^{(L)} \sigma_{L-1}\!\left( W^{(L-1)} \cdots \sigma_1(W^{(1)} x + b^{(1)}) + b^{(L-1)} \right) + b^{(L)} \right)
$$

这就是**前向传播（forward pass）**。每一层把上一层表示"投影 + 扭曲"一下，最终给出预测。

## 六、用 PyTorch 搭一个 MLP

我们搭一个两层 MLP 在 MNIST 上做手写数字分类：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, d_in=784, d_hidden=256, d_out=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                    # (B, 784)
            nn.Linear(d_in, d_hidden),       # (B, 256)
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),   # 第二隐藏层
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),      # logits (B, 10)
        )
    def forward(self, x):
        return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST('.', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('.', train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)        # 自带 softmax + log
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f'epoch {epoch} loss={loss.item():.4f}')
```

训练 5 个 epoch 通常就能在 MNIST 测试集上达到 97%+ 准确率。

## 七、MLP 的几何直觉

每一层都在做两件事：

1. **仿射变换** $W^{(l)} a^{(l-1)} + b^{(l)}$：旋转 + 拉伸 + 平移输入空间。
2. **激活** $\sigma(\cdot)$：在某些方向上"弯折"空间——把直线变成曲线。

经过若干层，原始空间中无法线性分割的数据，在最后几层变得线性可分。这就是深度学习的"逐层提特征"思想。

```text
 原始空间            隐藏层 1           隐藏层 2          输出空间
 (纠缠)            (部分展开)         (基本可分)        (线性分割)

  o o  × ×           o o                o o                  ○ ○
  o o  × ×    W1     o o     W2       o   o    W3        ─────────
  × ×  o o  ────►  ×   ×    ────►   ×   ×      ────►    × × × ×
  × ×  o o          × ×             × ×
```

## 八、MLP 的优势与短板

**擅长**：

- **表格数据**（特征数 $d$ 在 10–1000 量级、样本 $10^3$–$10^6$）：当数据没有明显的空间 / 时序结构时，MLP 配合特征工程和正则化往往就能取得不错的基线。
- **小规模函数拟合**：已知函数形式但参数未知时。
- **作为通用"嵌入 + 分类头"**：CNN/RNN/Transformer 的最后几层通常是 MLP。

**短板**：

- **图像**：像素间的**空间局部性**与**平移不变性**被 MLP 完全忽略，需要巨量参数才能勉强学到（CNN 解决）。
- **序列**：MLP 把序列"压平"，丢失**时序顺序**信息（RNN/Transformer 解决）。
- **高维稀疏输入**（如 One-hot 词向量）：参数量爆炸。
- **归纳偏置弱**：必须从大量数据中"自己发现"结构，样本少时不如带偏置的模型。

## 九、从 MLP 到现代架构

MLP 的核心思想——**线性 + 非线性 + 堆叠**——被所有现代架构继承：

- **CNN**：把"全连接"换成"卷积"，引入**空间局部性 + 平移不变性**偏置。
- **RNN / LSTM**：把"无状态"换成"递归"，引入**时序依赖**偏置。
- **Transformer**：把"逐位置 MLP"换成"注意力 + 位置感知"，处理**长程依赖**。
- **残差网络（ResNet）**：在 MLP 基础上加**跨层短路**，让 100+ 层网络可训练。

所以无论架构多复杂，你都会在最后几层看到熟悉的 `nn.Linear` + `nn.ReLU`。MLP 是所有深度学习模型的"骨架"。

## 小结

感知机是神经网络的最小单元，单层只能解线性可分问题（XOR 即反例）。把多层堆叠成 MLP，配合非线性激活，理论上能逼近任意连续函数（万能近似定理），实际中用反向传播 + 梯度下降训练。MLP 适合表格数据，是 CNN/RNN/Transformer 的基础构件，但在图像、序列、稀疏高维输入上需要带更强归纳偏置的架构。理解 MLP 这一"骨架"，再看任何现代架构都会更容易——它们无非是在 MLP 上加入了特定的"结构偏置"。