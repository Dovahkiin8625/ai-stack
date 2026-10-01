# 信息论深入

## 一、回顾：熵与 KL 散度

设 $P$ 为离散分布，**熵** $H(P)$ 度量 $P$ 的"内在不确定性"：

$$
H(P) = -\sum_x P(x) \log P(x)
$$

**KL 散度**（相对熵）度量两个分布的差异：

$$
D_{\mathrm{KL}}(P \Vert Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = -H(P) + H(P, Q)
$$

其中 $H(P, Q) = -\sum_x P(x) \log Q(x)$ 是交叉熵。注意 KL **不是**对称的——$D_{\mathrm{KL}}(P \Vert Q) \ne D_{\mathrm{KL}}(Q \Vert P)$，前者是"用 $Q$ 编码 $P$ 时多花的比特数"的期望。

## 二、互信息：变量间的依赖

**互信息**（Mutual Information）度量两个随机变量 $X, Y$ 共享的信息量：

$$
I(X; Y) = D_{\mathrm{KL}}\!\left( P_{(X,Y)} \;\Vert\; P_X \otimes P_Y \right) = H(X) - H(X \mid Y)
$$

直觉：如果知道 $Y$ 能让 $X$ 的不确定性降低多少，$I(X;Y)$ 就是那个"降低量"。

互信息的关键性质：

- $I(X; Y) = I(Y; X)$（对称）。
- $I(X; Y) \ge 0$，当且仅当 $X, Y$ 独立时为 0。
- $I(X; Y) \le \min(H(X), H(Y))$。

**条件互信息** $I(X; Y \mid Z)$ 度量"在已知 $Z$ 后，$X$ 还额外从 $Y$ 获得多少信息"，在因果推断中用于识别条件独立结构。

## 三、信息瓶颈：表示学习的理论视角

Tishby 等人提出的**信息瓶颈（IB）**框架把学习表示 $\mathbf{z}$ 看作一个有约束优化问题：

$$
\min_{P_{\mathbf{z}\mid \mathbf{x}}} \; I(\mathbf{z}; \mathbf{x}) - \beta \cdot I(\mathbf{z}; y)
$$

目标：表示 $\mathbf{z}$ 要尽可能压缩输入 $\mathbf{x}$（最小化 $I(\mathbf{z}; \mathbf{x})$），同时保留与标签 $y$ 相关的信息（最大化 $I(\mathbf{z}; y)$）。$\beta$ 控制压缩-预测权衡。

这个视角解释了深度网络训练中的**信息平面**现象：训练早期 $I(\mathbf{z}; \mathbf{x})$ 快速上升（拟合），后期缓慢下降（压缩/泛化）。虽然实证细节有争议，但 IB 框架仍是理解表示学习的重要思想工具。

## 四、对比学习与 InfoNCE

**对比学习**的核心目标：让正样本对的表示相似、负样本对的表示远离。InfoNCE 损失用一个分类视角统一了多种对比方法：

$$
\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\!\left[ \log \frac{\exp(s(\mathbf{z}_i, \mathbf{z}_i^+) / \tau)}{\sum_{j=1}^{K} \exp(s(\mathbf{z}_i, \mathbf{z}_j) / \tau)} \right]
$$

其中 $s(\cdot, \cdot)$ 是余弦相似度，$\tau$ 是温度，$K$ 是负样本数。InfoNCE 与互信息的下界紧密相关：

$$
I(\mathbf{z}; \mathbf{z}^+) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}}
$$

直观：负样本越多，$K$ 越大，下界越紧，学到的表示越好。这也是为什么 CLIP 训练时使用 32 768 个负样本。

```python
import torch
import torch.nn.functional as F

def infonce(z1, z2, temperature=0.1):
    # z1, z2: (B, D)，同一样本的两个增强视图
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature          # (B, B)
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)    # InfoNCE ≈ 交叉熵
```

## 五、信道容量与率失真

把信息论扩展到"传输"场景：信源 $X$、信道 $P(Y|X)$、接收方收到 $Y$。**信道容量**是信道能可靠传输的最大信息率：

$$
C = \max_{P_X} \; I(X; Y)
$$

这是香农第二定理的基础：只要传输速率 $R < C$，总存在编码使错误率任意小。

**率失真理论**研究"在允许失真 $D$ 的条件下，压缩的最小比特率"：
$$
R(D) = \min_{P_{\hat{X}|X}:\; \mathbb{E}[d(X,\hat{X})] \le D} \; I(X; \hat{X})
$$

率失真解释了为什么我们可以对图像做有损压缩（人眼允许一定失真）、为什么 LLM 可以做 KV-cache 量化（轻微的失真换大量显存）。**扩散模型的 ELBO** 也可以从率失真角度理解：在每一步加噪与去噪之间寻找率失真最优的中间表示。

## 六、信息论在决策树与强化学习中的应用

**决策树中的信息增益**：每次分裂选特征 $A$ 使分裂后子集的**熵**下降最多，等价于最大化信息增益 $I(Y; A)$。这就是 ID3/C4.5 算法的核心。

**强化学习中的策略梯度**：REINFORCE、PPO 等算法的目标可以重写为 $\mathbb{E}_\tau[\nabla_\theta \log \pi_\theta(a|s) \cdot R(\tau)]$，其中的对数概率项可以理解为"策略 $\pi$ 的熵"的负梯度项——加上**熵正则** $\beta H(\pi(\cdot|s))$ 能鼓励探索，避免策略过早坍缩到单一动作。

```python
# PPO 中常用的熵正则
entropy = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
loss = policy_loss - 0.01 * entropy
```

## 小结

从熵出发，我们得到 KL 散度（分布差异）、互信息（变量依赖）、信息瓶颈（表示压缩-预测权衡）、InfoNCE（互信息下界与对比学习）；再到率失真（有损压缩与生成模型）和信道容量（通信极限）。这些工具既是信息论本身的骨架，也是理解现代 ML 几乎所有子领域的"通用钥匙"。