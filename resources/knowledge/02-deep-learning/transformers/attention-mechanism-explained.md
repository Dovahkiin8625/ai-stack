# 注意力机制详解

Transformer 是过去十年最具影响力的神经网络架构，而注意力机制（Attention）是它的核心。本文从直觉出发，推导缩放点积注意力公式，并通过一个 3-token 的小例子演示注意力权重是如何计算出来的。

## 一、Query、Key、Value 的直觉

把注意力想象成"信息检索"：

- **Query（查询，Q）**：当前 token 想问的问题。
- **Key（键，K）**：每个 token 能提供的"标签"。
- **Value（值，V）**：每个 token 真正承载的信息。

计算过程：先用 $Q$ 去和每个 $K$ 比相似度，得到一组权重，再用这些权重对 $V$ 做加权平均。整个过程可以并行化，是 Transformer 相对 RNN 的最大优势。

## 二、缩放点积注意力公式

```math
\text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
```

步骤拆解：

1. 计算 $Q K^\top$，得到一个形状为 $(\text{seq\_len}, \text{seq\_len})$ 的相似度矩阵。
2. 除以 $\sqrt{d_k}$，避免点积过大导致 softmax 饱和。
3. 按行做 softmax，得到归一化权重。
4. 用权重乘 $V$，得到每个位置的加权和。

## 三、为什么需要除以 $\sqrt{d_k}$

假设 $Q$、$K$ 的每个分量是独立零均值、方差为 1 的随机变量，则 $q \cdot k$ 的方差约为 $d_k$。当 $d_k$ 较大（如 64、128）时，点积会落在很大的数值范围，让 softmax 输出接近 one-hot，反向传播梯度也会变得极小。除以 $\sqrt{d_k}$ 后方差回到 $\sim 1$，训练更稳定。

## 四、多头注意力

单一注意力头只能捕捉一种关系模式；多头注意力把 $Q, K, V$ 切成 $h$ 份独立投影，让每份各自做注意力，再拼接回原维度：

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
```

其中 $\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$。

直觉上不同头可以学习不同语法或语义关系，例如一个头关注局部依赖，另一个头关注长距离指代。

## 五、3-token 小例子

设三个 token，每个 $d_k = 2$，初始 $Q = K = V$：

$$Q = K = V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$$

1. $QK^\top = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}$
2. 除以 $\sqrt{2}$：数值约为 $\begin{pmatrix} 0.71 & 0 & 0.71 \\ 0 & 0.71 & 0.71 \\ 0.71 & 0.71 & 1.41 \end{pmatrix}$
3. 行 softmax（以第一行为例）：$\text{exp}(0.71)\approx 2.03$，$\text{exp}(0) = 1$，$\text{exp}(0.71)\approx 2.03$，归一化得 $(0.40,\ 0.20,\ 0.40)$。
4. 用这组权重乘 $V$，得到 token 1 的新表示 $(0.40 \cdot 1 + 0.20 \cdot 0 + 0.40 \cdot 1,\ 0.40 \cdot 0 + 0.20 \cdot 1 + 0.40 \cdot 1) = (0.80,\ 0.60)$。

可以看到，token 1 的新表示融合了自身（权重 0.40）和 token 3（权重 0.40）的内容，仅少量受 token 2 影响。

## 小结

Q/K/V 让模型学会"按需查找"；缩放让训练稳定；多头让模型并行学习多种关系。掌握这些，你就已经理解了 Transformer 的"心脏"。
