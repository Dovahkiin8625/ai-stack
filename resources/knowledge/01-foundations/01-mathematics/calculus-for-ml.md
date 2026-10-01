# 机器学习中的微积分

微积分是"学习"的数学基础。模型训练本质是寻找让损失函数最小的参数，而"怎么走"由梯度决定。本文覆盖导数、偏导、梯度、链式法则，并通过小例子演示梯度下降如何工作。

## 一、导数与偏导数

单变量函数 $f(x)$ 的导数 $f'(x)$ 描述 $f$ 在 $x$ 处的瞬时变化率：

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

当函数依赖多个变量时，我们用偏导数 $\partial f / \partial x_i$ 描述在某个坐标方向上的变化率。把所有偏导数拼起来，就是

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)
$$

这是 ML 中最常见的梯度形式。

## 二、链式法则

深度学习的关键在于：模型是层层嵌套的复合函数。链式法则告诉我们如何"把梯度传回去"。

若 $y = f(g(x))$，则

$$
\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

多变量版本：若 $\mathbf{y} = f(\mathbf{u}), \mathbf{u} = g(\mathbf{x})$，则

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{u}} \cdot \frac{\partial \mathbf{u}}{\partial \mathbf{x}}
$$

这就是反向传播（Backpropagation）的数学原理。

## 三、梯度下降

梯度下降通过迭代更新参数来最小化损失 $L(\theta)$：

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

其中 $\eta$ 是学习率。直观上，梯度指向函数上升最快的方向，所以减去梯度就走到了下降最快的方向。

以 $L(\theta) = (\theta - 3)^2$ 为例，$\nabla L = 2(\theta - 3)$。从 $\theta_0 = 0$、$\eta = 0.1$ 开始：

```
step 0: theta = 0.00,  loss = 9.00
step 1: theta = 0.60,  loss = 5.76
step 2: theta = 1.08,  loss = 3.69
...
step 20: theta ≈ 2.94, loss ≈ 0.0036
```

可以看到 $\theta$ 不断逼近 3，损失不断下降。

## 四、常见陷阱与改进

- **学习率过大**：参数会在最优值附近震荡甚至发散。
- **学习率过小**：收敛极慢。
- **局部极小 / 鞍点**：高维非凸函数中常见；带动量（Momentum）、Adam 等优化器能缓解。
- **梯度消失 / 爆炸**：深层网络会出现，使用残差连接、归一化、合适的激活函数可以改善。

## 五、激活函数与梯度流

激活函数 $\sigma(x)$ 的导数直接决定反向传播时梯度的"乘子"。一个经典的例子是 sigmoid：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

由于 $\sigma'(x) \le 0.25$，梯度经过多层 sigmoid 后会指数级缩小——这就是 sigmoid 深层网络中"梯度消失"的根源之一。ReLU 在正区间导数为常数 1，因此在现代网络中很大程度上缓解了这个问题。GeLU、SwiGLU 等更平滑的替代品则在保持梯度通畅的同时引入轻微非线性。理解这一点，是从"模型能跑"过渡到"模型训得稳"的关键一步。

## 小结

偏导给出方向，梯度把所有方向整合起来，链式法则让深层网络可训练，梯度下降负责实际更新，激活函数的导数决定了梯度能否"流到"底层参数。理解了这五件事，你就具备了"读懂训练循环"的数学基础。
