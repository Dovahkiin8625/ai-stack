# 强化学习基础：从 MDP 到 PPO

## 一、MDP：强化学习的数学骨架

**马尔可夫决策过程（Markov Decision Process）**是一个五元组 $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$：

- $\mathcal{S}$：状态空间。
- $\mathcal{A}$：动作空间。
- $P(s'|s, a)$：转移概率——在状态 $s$ 采取动作 $a$ 后到 $s'$ 的概率。
- $r(s, a)$：奖励函数。
- $\gamma \in [0, 1]$：折扣因子，越未来奖励折扣越重。

智能体的目标是学习**策略** $\pi(a|s)$，最大化期望累计折扣奖励：

```math
J(\pi) = \mathbb{E}_\tau\!\left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right],\quad \tau = (s_0, a_0, s_1, a_1, \dots)
```

**Markov 性**：下一状态只依赖当前 $(s, a)$，与历史无关。

```python
# OpenAI Gymnasium 风格的环境示例
import gymnasium as gym
env = gym.make('CartPole-v1')
obs, info = env.reset()                         # obs: 当前状态
action = env.action_space.sample()              # 随机动作
obs, reward, terminated, truncated, info = env.step(action)
```

## 二、价值函数与贝尔曼方程

**状态价值函数**：从状态 $s$ 出发、按策略 $\pi$ 行动的期望累计奖励：

```math
V^\pi(s) = \mathbb{E}_\pi\!\left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \,\Big|\, s_0 = s \right]
```

**动作价值函数**：从状态 $s$ 出发、先执行 $a$、再按 $\pi$：

```math
Q^\pi(s, a) = \mathbb{E}_\pi\!\left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \,\Big|\, s_0 = s, a_0 = a \right]
```

两者关系：

```math
Q^\pi(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P}\!\left[ V^\pi(s') \right]
```

```math
V^\pi(s) = \mathbb{E}_{a \sim \pi}\!\left[ Q^\pi(s, a) \right]
```

**贝尔曼期望方程**给出递归关系：

```math
V^\pi(s) = \sum_a \pi(a|s) \!\left[ r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]
```

**最优价值函数**：

```math
V^*(s) = \max_\pi V^\pi(s),\quad Q^*(s, a) = \max_\pi Q^\pi(s, a)
```

满足**贝尔曼最优方程**：

```math
V^*(s) = \max_a \!\left[ r(s,a) + \gamma \mathbb{E}_{s'}\!\left[ V^*(s') \right] \right]
```

## 三、动态规划：基于模型的解法

当 $P, r$ 已知时，可用 DP 直接求解：

**策略迭代**：先固定 $\pi$ 算 $V^\pi$（**策略评估**），再对每个 $s$ 取 $\arg\max_a Q^\pi(s, a)$ 更新 $\pi$（**策略改进**），重复直到 $\pi$ 不变。

**价值迭代**：直接迭代贝尔曼最优方程：

```math
V_{k+1}(s) = \max_a\!\left[ r(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]
```

复杂度都是 $O(|\mathcal{S}|^2 |\mathcal{A}|)$，**只能用于小状态空间**（如棋盘游戏）。

## 四、模型无关的强化学习

实际场景往往不知道 $P$（如 Atari 游戏），只能用**采样**与环境交互估计价值。

### 4.1 MC vs TD

**蒙特卡洛（Monte Carlo）**：跑完一整条轨迹，用实际回报估计：

```math
V(s) \leftarrow V(s) + \alpha\,[G_t - V(s)]
```

$G_t = \sum_{k=t}^T \gamma^{k-t} r_k$。**无偏但方差大**。

**时序差分（TD）**：用单步奖励 + 估计的下一步价值：

```math
V(s) \leftarrow V(s) + \alpha\,[r + \gamma V(s') - V(s)]
```

**有偏差（bootstrapping）但方差小**。

**n 步 TD** 折中：$V(s_t) \leftarrow V(s_t) + \alpha\,[G_{t:t+n} - V(s_t)]$，$G_{t:t+n} = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^n V(s_{t+n})$。

### 4.2 Q-Learning

直接学 $Q(s, a)$：

```math
Q(s, a) \leftarrow Q(s, a) + \alpha\!\left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
```

**off-policy**——评估策略可以是 $\epsilon$-greedy 但更新时假设 max。这是 **Deep Q-Network (DQN)** 的基础。

DQN 关键技巧：

- **经验回放**（Replay Buffer）：打破样本时间相关性。
- **目标网络**（Target Network）：$Q$ 的目标网络周期性更新，稳定训练。
- **双 Q-Learning**（Double DQN）：解耦动作选择与价值估计，缓解过估计。

```python
import torch, torch.nn as nn, random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x): return self.net(x)

def train_dqn():
    policy_net = DQN(4, 2)
    target_net = DQN(4, 2)
    optim = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = deque(maxlen=10000)
    gamma = 0.99
    # ... rollout, sample batch, compute Bellman target, backward
```

## 五、策略梯度：直接优化策略

有些任务动作空间连续或极大（如机器人控制），价值函数难求，直接参数化策略：

```math
\pi_\theta(a|s) \;\text{是一个神经网络},\quad \theta \text{ 是参数}
```

**REINFORCE** 目标：

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ R(\tau) \right],\quad R(\tau) = \sum_t \gamma^t r_t
```

策略梯度定理：

```math
\nabla_\theta J(\theta) = \mathbb{E}_\tau\!\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]
```

直观：让"回报高的动作出现概率变大"，按 $G_t$ 加权。

```python
# REINFORCE 简版
log_prob = policy.log_prob(action)
loss = -log_prob * G_t
loss.backward()
```

**基线（baseline）**：减去 $V(s_t)$ 或平均回报，降低方差：

```math
\nabla_\theta J = \mathbb{E}\!\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot (G_t - b(s)) \right]
```

$b(s) = V^\pi(s)$ 是常见选择，称为"优势函数" $A(s, a) = Q(s, a) - V(s)$。

## 六、Actor-Critic：策略 + 价值联合

**Actor-Critic** 同时学策略（Actor）和价值函数（Critic）：

- **Actor**：$\pi_\theta(a|s)$，按优势方向更新。
- **Critic**：$V_\phi(s)$ 或 $Q_\phi(s, a)$，用 TD/蒙特卡洛回归。

```math
\theta \leftarrow \theta + \eta \nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)
```

```math
\phi \leftarrow \phi - \eta \nabla_\phi (A(s, a))^2 \quad \text{（优势预测的 MSE 损失）}
```

**A2C**（Advantage Actor-Critic）：同步版本。

**A3C**：异步多 worker 版本（已被 A2C 取代）。

## 七、PPO：工业级默认算法

**PPO（Proximal Policy Optimization）** 是 OpenAI 2017 提出的 Actor-Critic 改进，至今仍是事实标准（ChatGPT 的 RLHF 也用它）。

核心创新：**限制策略更新幅度**，避免一次更新太猛导致策略崩溃。

**Clipped Surrogate Objective**：

```math
L^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[ \min\!\left( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
```

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略概率比。clip 在 $[1-\epsilon, 1+\epsilon]$（如 $\epsilon=0.2$）截断。

直觉：不让 $r_t$ 偏离 1 太远，否则梯度被"剪掉"，更新被限制。

```python
import torch
import torch.nn.functional as F

def ppo_loss(log_probs, old_log_probs, advantages, clip=0.2):
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    return -torch.min(surr1, surr2).mean()

def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # Generalized Advantage Estimation
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        next_v = values[t+1] if t < len(rewards)-1 else 0
        delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages
```

**PPO 的工程要点**：

- 多 epoch 同一批数据复用（on-policy 但采样效率高）。
- GAE 计算 advantage。
- 加熵正则 $\beta H(\pi)$ 鼓励探索。
- Value function clipping、梯度裁剪。
- 学习率 1e-4 ~ 3e-4。

## 八、稀疏奖励与探索

很多任务的奖励信号非常稀疏（如围棋胜利才 +1），需要**鼓励探索**：

- **$\epsilon$-greedy**：以小概率随机动作（DQN）。
- **熵正则**：在目标中加 $-\beta H(\pi(\cdot|s))$ 鼓励动作分布均匀。
- **好奇心驱动**（ICM）：用预测误差作为内在奖励。
- **RND**（Random Network Distillation）：随机目标网络 + 预测网络，预测误差作为内在奖励。
- **课程学习**：从易到难安排训练任务。

## 九、RLHF 与现代 LLM 对齐

**RLHF（Reinforcement Learning from Human Feedback）**三步：

1. **SFT**（Supervised Fine-Tuning）：用人工标注的 (prompt, response) 对微调 LLM。
2. **奖励模型**（Reward Model）：用人类偏好标注（A vs B 哪个更好）训练一个标量奖励模型 $r_\phi$。
3. **PPO**：用 $r_\phi$ 作为奖励函数，对 LLM 做 RL 优化。

```math
L_{\text{RL}} = \mathbb{E}\!\left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]
```

第二项是**KL 散度正则**，防止 LLM 偏离 SFT 模型太远（防止"奖励黑客"）。

**现代替代**：**DPO**（Direct Preference Optimization）绕过显式奖励模型，直接用偏好数据做监督学习，公式简洁且稳定。

```math
L_{\text{DPO}} = -\log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
```

其中 $y_w, y_l$ 是人类标注的优选 / 劣选回答。

## 十、关键算法的演进谱系

```text
DP（模型已知）
  ↓
MC / TD（无模型）
  ↓
Q-Learning / SARSA（值函数方法）
  ↓
DQN（深度 Q-Learning + 经验回放 + 目标网络）
  ↓
Policy Gradient (REINFORCE)
  ↓
Actor-Critic (A2C)
  ↓
TRPO / PPO（信任域 / 裁剪约束）
  ↓
SAC（最大熵 + 连续动作）
  ↓
RLHF / DPO（LLM 对齐）
```

## 十一、强化学习的工程挑战

- **样本效率低**：百万次环境交互才学到一个简单任务。
- **训练不稳定**：种子敏感、超参敏感。
- **奖励工程**：奖励设计不合理会陷入局部最优（"wireheading"）。
- **sim2real gap**：仿真训练迁移到真实机器人困难。
- **安全性**：探索过程可能损坏硬件（机器人训练需安全约束）。

## 十二、与监督学习的对比

| 维度 | 监督学习 | 强化学习 |
|---|---|---|
| 数据来源 | 静态标注 | 智能体与环境交互产生 |
| 反馈 | 立即、确定 | 延迟、稀疏、随机 |
| 评估 | 训练/测试集 | 累计回报 |
| 数据分布 | i.i.d. | 强时序相关 |
| 探索 | 不需要 | 必须（探索-利用权衡） |

## 小结

强化学习的核心是"**在不确定性下学习长期最优行为**"：MDP 提供数学框架，价值函数刻画"未来收益"，策略梯度直接优化动作分布，Actor-Critic 结合两者，PPO 通过裁剪约束让训练稳定。从 Atari 到机器人，从 AlphaGo 到 ChatGPT 对齐，PPO 几乎无处不在。掌握这套框架后，你会理解现代 LLM 对齐的算法本质——RLHF/DPO 仍然是"奖励驱动 + 策略优化"的强化学习思想，只不过奖励由人类偏好训练得到。