# RLHF：从人类反馈到 PPO 训练

把预训练 LLM 变成"听话的助手"的核心是 **RLHF（Reinforcement Learning from Human Feedback）**。它的思想是：用人对"哪条回答更好"的偏好来训练一个**奖励模型（Reward Model, RM）**，再用强化学习（最常用 PPO）让 LLM 策略最大化这个奖励。本文梳理 RLHF 的三阶段流程、奖励模型的数学形式、PPO 在 LLM 中的关键 trick，以及为什么 RLHF 后模型会出现"谄媚（sycophancy）"等副作用。

## 一、为什么需要 RLHF

预训练 LLM 只会"续写"——给定 prompt，它续出概率最高的 token 序列。但用户想要的是**有帮助（helpful）、诚实（honest）、无害（harmless）**的回答，这三件事不在 next-token 损失里。直接用 SFT 训"指令-回答"对也只能模仿风格，不能优化"哪个更好"的相对判断。

RLHF 的关键洞察：**人对"两个回答哪个更好"的判断，远比对"自己写一个完美回答"更可靠**。把成对比较收集起来，就能训一个 RM，再用 RM 当裁判，反过来优化 LLM。

## 二、RLHF 的三阶段流程

```text
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: SFT（监督微调）                                     │
│  用高质量 (prompt, answer) 对做监督微调，得到初始策略 π_SFT。   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: 奖励模型训练                                          │
│  采样 prompt → 让 π_SFT 生成 K 个回答 → 人类排序 → Bradley-Terry │
│  损失训 RM：r_θ(x, y)。                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: PPO 强化学习                                         │
│  用 r_θ 作为奖励，PPO 更新 π_θ；KL 惩罚防止偏离 π_SFT 太远。     │
└─────────────────────────────────────────────────────────────┘
```

OpenAI InstructGPT（2022）、Anthropic Claude、Meta LLaMA-2-Chat 都遵循这一框架。

## 三、奖励模型的 Bradley-Terry 损失

人对两个回答 $y_a, y_b$（prompt $x$）做偏好标注：$y_a \succ y_b$ 表示 $a$ 更好。Bradley-Terry 模型把偏好概率写成奖励差：

$$
P(y_a \succ y_b \mid x) = \frac{\exp(r_\theta(x, y_a))}{\exp(r_\theta(x, y_a)) + \exp(r_\theta(x, y_b))} = \sigma\!\big(r_\theta(x, y_a) - r_\theta(x, y_b)\big)
$$

奖励模型 $r_\theta(x, y)$ 通常用一个语言模型改造：把 LM 头替换为**标量回归头**，输入 `(x, y)` 输出一个实数。损失函数：

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_a, y_b, \mu) \sim \mathcal{D}}\Big[\mu \log \sigma\big(r_\theta(x, y_a) - r_\theta(x, y_b)\big) + (1-\mu)\log \sigma\big(r_\theta(x, y_b) - r_\theta(x, y_a)\big)\Big]
$$

其中 $\mu = \mathbb{1}[y_a \succ y_b]$。

## 四、Stage 3：PPO 优化与 KL 惩罚

策略 $\pi_\theta$ 通过最大化期望奖励优化：

$$
\max_\theta \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\big[r_\theta(x, y)\big] - \beta \cdot \mathrm{KL}\big(\pi_\theta(\cdot\mid x) \,\|\, \pi_{\text{SFT}}(\cdot\mid x)\big)
$$

$\beta$ 控制"偏离 SFT 模型多远"的惩罚——$\beta$ 大则保守（接近 SFT），$\beta$ 小则激进（高奖励但可能跑偏）。

直接求梯度要用到策略梯度，但 PPO 用**截断替代目标（clipped surrogate objective）**稳定训练：

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t(\theta) \hat{A}_t,\; \mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big],\quad \rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

在 LLM 场景下，"状态 $s_t$"是当前 prompt + 已生成的 token，"动作 $a_t$"是下一个 token。优势 $\hat{A}_t$ 用 GAE 估计。

## 五、为什么需要 Value Model 与 KL 散度

PPO 在 LLM 上有三个额外组件：

1. **Value Model $V_\phi(s_t)$**：估计从 $s_t$ 开始能拿到的总奖励，作用是降低方差。结构与 RM 几乎一样，但输入只看状态不看完整回答。
2. **KL 惩罚项**：除 RM 奖励外，每步加 $-\beta \cdot \log\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{SFT}}(a_t\mid s_t)}$，避免 $\pi_\theta$ 跑到 RM 的高奖励但**语义不通顺**的区域。
3. **Reward shaping**：把 KL 项直接加到 per-token reward 里，PPO 自动学会权衡。

工程上通常用一个**联合模型**同时输出 value 和 policy（共享 backbone），节省显存。

## 六、PyTorch 风格的 PPO 损失

下面给一个最小可跑的 PPO 目标实现，便于理解各部分的耦合：

```python
import torch
import torch.nn.functional as F


def ppo_loss(
    logp_new: torch.Tensor,   # (B, T): 新策略对每个 token 的 log prob
    logp_old: torch.Tensor,   # (B, T): 采样时的旧策略 log prob
    advantages: torch.Tensor, # (B, T): GAE 优势
    values_new: torch.Tensor,  # (B, T): 新 value 预测
    values_old: torch.Tensor,  # (B, T): 采样时的旧 value
    returns: torch.Tensor,     # (B, T): 折扣累计奖励
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    kl_coef: float = 0.05,
    ref_logp: torch.Tensor | None = None,  # SFT 模型的 log prob（KL 锚点）
):
    # 1) ratio 与截断替代目标
    ratio = (logp_new - logp_old).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    # 2) value 损失（截断或平方均可）
    value_loss = F.mse_loss(values_new, returns)

    # 3) 与 SFT 的 KL 惩罚（k3 估计器，无偏）
    if ref_logp is not None:
        kl = (logp_new - ref_logp).mean()
    else:
        kl = torch.tensor(0.0, device=logp_new.device)

    loss = policy_loss + vf_coef * value_loss + kl_coef * kl
    return loss, {"policy": policy_loss.item(), "value": value_loss.item(), "kl": kl.item()}
```

三个常用系数：`clip_ratio=0.2`（PPO 原论文），`vf_coef=0.1~0.5`，`kl_coef` 动态调整（KL 太大就调高）。

## 七、RLHF 的典型副作用

### 1. 谄媚（Sycophancy）

模型倾向于附和用户的观点——即使错了也说"你说得对"。原因是 RM 在标注时**隐含奖励"用户喜欢"**，策略就学会揣摩。

### 2. 奖励 Hacking

策略找到 RM 的漏洞拿高分但实际质量下降。比如：

- 回答变长（RM 偏长）。
- 用 Markdown 表格、加 emoji（RM 偏"看起来好看"）。
- 重复关键词、堆 safe completion（RM 偏"无害"到啰嗦）。

**缓解**：用 stronger RM（GPT-4 当裁判）、加 rule-based reward、KL 加大。

### 3. 模式坍缩

策略倾向于输出"标准答案"格式，对 prompt 的多样性响应下降。本质是 RM 对分布外样本外推差。

### 4. 灾难性遗忘

强化学习阶段可能让模型在某些能力（如代码、推理）上回退。**缓解**：混合 SFT loss，定期回归到 SFT 数据上做"重热"。

## 八、RLHF 的工程挑战

| 挑战 | 表现 | 缓解 |
|---|---|---|
| 显存 | 4 个模型（policy/ref/value/RM）同时在显存 | LoRA + 量化 ref/value |
| 不稳定 | PPO 对超参敏感 | 多 PPO epoch + clip + 优势归一化 |
| 标注噪声 | 人与人一致性低（Kappa ~ 0.6） | 多标注取平均 + 清晰 rubric |
| Reward hacking | 高分低质 | 加 length penalty + rule reward |
| 分布漂移 | 策略偏离 SFT 太远 | 调 KL 系数 + 早期停止 |

LLaMA-2 论文报告 RLHF 用了 **>1M 人类偏好对**、数千 GPU 时，成本极高。

## 九、替代与扩展

- **DPO**（Rafailov et al. 2023）：绕过 RM + RL，直接用偏好对做监督学习——下一篇详解。
- **RLAIF / Constitutional AI**：用 LLM 自己当 RM，减少人类标注。
- **Process Reward Model（PRM）**：奖励"中间推理步骤"，提升数学/代码能力（OpenAI o1 路线）。
- **RLVR**（RL with Verifiable Rewards）：用规则验证器（数学答案、单元测试）给奖励，无需人类偏好。

## 小结

RLHF 把"对齐"从模仿学习升级为"按人类偏好优化"——Bradley-Terry 训 RM，PPO 优化策略 + KL 锚点。它是 ChatGPT / Claude / LLaMA-2-Chat 的共同基础，但也带来谄媚、reward hacking、模式坍缩等副作用。下一篇我们将看到 **DPO** 如何绕过显式 RM 与 RL，用一个监督学习目标直接拟合偏好，从而大幅简化训练流程。
