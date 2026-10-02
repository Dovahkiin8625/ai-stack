# DPO：直接偏好优化

RLHF 流程复杂（训 RM、跑 PPO、4 个模型协作）、显存昂贵、训练不稳定。**DPO（Direct Preference Optimization, Rafailov et al. 2023）**通过一个巧妙的推导证明：**RLHF 的最优策略可以闭式表达为偏好数据的监督学习目标**。这意味着我们能完全跳过 RM 和 PPO，直接用 `(prompt, chosen, rejected)` 三元组做监督训练——更简单、更稳定、效果相当甚至更好。本文推导 DPO 的核心方程，给出 PyTorch 实现，并讨论 IPO、KTO 等后续变体。

## 一、从 RLHF 目标到闭式解

RLHF 的最终目标是：

$$
\max_\pi \; \mathbb{E}_{x, y \sim \pi}\big[r(x, y)\big] - \beta\,\mathrm{KL}\big(\pi(\cdot\mid x) \,\|\, \pi_{\text{ref}}(\cdot\mid x)\big)
$$

这是一个带 KL 约束的 RL 问题。最优策略 $\pi^*$ 满足（对单个 prompt $x$）：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)
$$

其中 $Z(x) = \sum_{y'} \pi_{\text{ref}}(y' \mid x) \exp\!\big(\tfrac{1}{\beta} r(x, y')\big)$ 是配分函数（无法直接计算）。

取对数，把奖励反解出来：

$$
r(x, y) = \beta \log \frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)
$$

## 二、Bradley-Terry 与 DPO 损失

把上式代入 Bradley-Terry 偏好模型：

$$
P(y_w \succ y_l \mid x) = \sigma\!\big(r(x, y_w) - r(x, y_l)\big)
$$

由于 $\log Z(x)$ 在 $y_w$ 和 $y_l$ 之间相减抵消，**$Z(x)$ 消失了**！得到 DPO 损失：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

用 $\beta \log\frac{\pi_\theta}{\pi_{\text{ref}}}$ 表示"隐式奖励"。直觉：**让 chosen 回答在 $\pi_\theta$ 下的概率相对 $\pi_{\text{ref}}$ 比 rejected 提升得更多**。

## 三、DPO 与 RLHF 的对比

| 维度 | RLHF | DPO |
|---|---|---|
| 奖励模型 | 必须训一个 RM | 不需要 RM |
| 强化学习 | 必须跑 PPO | 直接监督学习 |
| 模型数 | 4 个（policy/ref/value/RM） | 2 个（policy/ref） |
| 显存 | 高 | 低（接近 SFT） |
| 训练稳定性 | 较敏感（PPO 超参） | 稳定（SFT 风格） |
| 在线采样 | 需要（PPO 用当前策略 rollout） | 不需要（用离线数据） |
| 效果 | 强 | 相当甚至更好（多数 benchmark） |

DPO 也支持**在线版本（Online DPO / Iterative DPO）**：每轮用当前策略采样新偏好对再训练，效果可逼近 PPO。

## 四、PyTorch 实现

```python
import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,   # (B,): π_θ(y_w | x) 的 log prob（按 token 求和）
    policy_rejected_logp: torch.Tensor, # (B,): π_θ(y_l | x)
    ref_chosen_logp: torch.Tensor,      # (B,): π_ref(y_w | x)
    ref_rejected_logp: torch.Tensor,    # (B,): π_ref(y_l | x)
    beta: float = 0.1,
    label_smoothing: float = 0.0,
):
    """
    DPO 损失 + 可选 SFT 辅助项（pi_lm_alpha）。
    """
    # 隐式奖励差
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp

    # 偏好 logits：chosen 比 rejected 多出的对数比
    logits = beta * (pi_logratios - ref_logratios)

    # 标准 DPO 损失
    loss = -F.logsigmoid(logits) * (1 - label_smoothing) \
           - F.logsigmoid(-logits) * label_smoothing
    return loss.mean()


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    计算序列级 log p(y | x)，对 label 位置的 token 求和并按 mask 取平均。
    logits: (B, T, V), labels: (B, T), mask: (B, T)
    """
    logp = F.log_softmax(logits, dim=-1)             # (B, T, V)
    per_token = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    per_token = per_token * mask                     # 把 padding / prompt 位置置 0
    return per_token.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)   # 按有效 token 数归一
```

训练循环几乎是 SFT 的模板——只是同时算 chosen 与 rejected 的 logprob，调用 `dpo_loss` 反传即可。

## 五、超参与常见技巧

### $\beta$（温度系数）

- $\beta$ 大 → 策略贴近 $\pi_{\text{ref}}$，保守。
- $\beta$ 小 → 激进偏离，可能过拟合偏好数据。
- 推荐起步：**$\beta = 0.1$**（LLaMA-3、Mixtral-OFFICIAL 配方）。

### Length Normalization

直接对 token 求和会让模型偏长回答（长回答 log prob 天然更负）。两种处理：

1. **按长度归一化**：`logp / len`（如上代码所示）。
2. **加 length penalty**：在 reward 里显式减去 $\lambda \cdot |y|$。

### Label Smoothing

`label_smoothing=0.1` 让目标不是"非 0 即 1"，避免过拟合尖锐偏好。Mistral-7B-DPO 用过这个 trick。

### NLL 辅助项（pi_lm_alpha）

有些实现加一个 SFT 损失项防止偏离：

$$
\mathcal{L} = \mathcal{L}_{\text{DPO}} + \alpha \cdot \mathcal{L}_{\text{SFT}}
$$

`alpha` 通常很小（0.05~0.1）。

## 六、DPO 的局限与变体

### 1. 分布偏移（Distribution Shift）

DPO 用的是 $\pi_{\text{ref}}$，但策略在训完会偏离它。这导致**训练目标与生成时的分布**不一致，长尾偏好被低估。

**解决**：Online DPO / Iterative DPO——每轮用当前策略重新采样偏好对。

### 2. 偏好"过度拟合"

DPO 容易出现 chosen vs rejected 差距过大但实际质量下降的现象（**reward hacking**）。

**解决**：IPO（Azar et al. 2023）用平方损失替代 log-sigmoid，约束隐式奖励差的方差，避免极端偏移：

$$
\mathcal{L}_{\text{IPO}} = \big(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)} - \frac{1}{2\beta}\big)^2
$$

### 3. 没有"负面偏好"

DPO 把数据看作"chosen > rejected"的成对比较。**KTO（Kahneman-Tversky Optimization, Ethayarajh et al. 2024）**直接用单个回答的"好/坏"标签，结合前景理论（人对损失的敏感度高于收益）：

$$
\mathcal{L}_{\text{KTO}} = \lambda_y \big(1 - \sigma(\beta r_\theta(x, y))\big) \quad \text{（对好回答）}
$$

KTO 在数据稀疏时效果更稳，且**不需要成对**——只需单条 (prompt, answer, good/bad) 标签。

### 4. 多目标对齐

需要同时优化"有帮助"与"无害"时，单个 RM 不够。**Multi-Reward DPO / MO-DPO** 把多个标量奖励合并，或在不同 RM 上分别 DPO 后 merge。

## 七、什么时候用 DPO vs RLHF？

| 场景 | 推荐 |
|---|---|
| 已有高质量偏好对，想快速对齐 | **DPO** |
| 需要在线采样 / 主动探索 | RLHF（PPO） |
| 算力紧张、显存有限 | **DPO** |
| 多目标（helpfulness + safety） | RLHF（多 RM 加权）或 ORPO |
| 偏好数据含大量噪声 | IPO / KTO |
| 已有 SFT 模型想进一步对齐 | **DPO**（最常见路线） |

## 八、DPO 在工业界的实践

- **Mistral-7B-Instruct-DPO**：在 Mixtral-OFFICIAL 偏好的 100K 对上做 DPO，效果优于纯 SFT。
- **Meta LLaMA-3-It**：SFT → rejection sampling → 两轮 DPO，达到与 RLHF 相当的水平但训练成本降一个数量级。
- **Zephyr-7B**：HuggingFace 用纯 DPO 在 7B 模型上复现 ChatGPT 风格对话。
- **Intel / NeuralChat**：在企业数据上 DPO，3 小时对齐 7B 模型到客服场景。

## 小结

DPO 用一个反直觉的推导证明：**显式 RM 与 RL 不是必需的**——把隐式奖励表达成 $\pi_\theta / \pi_{\text{ref}}$ 的对数比，配分函数 $Z(x)$ 在偏好对中自然抵消。训练变得像 SFT 一样简单，效果却逼近甚至超过 RLHF。它的局限（分布偏移、reward hacking）催生了 IPO、KTO、Online DPO 等变体。今天工业界做对齐时，**SFT → DPO** 已经是最常见的最小流程。下一篇我们看另一个方向的简化——**Constitutional AI** 与 RLAIF：用 LLM 自己当奖励模型，把"人类标注"替换成"原则驱动"。
