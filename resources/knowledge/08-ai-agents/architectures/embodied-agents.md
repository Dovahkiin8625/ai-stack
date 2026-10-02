# 具身 Agent：从感知到物理世界的智能

"具身（embodiment）"是 AI 领域一个深刻的哲学主张：**真正的智能必须在物理世界中、通过身体与环境交互才能涌现**。LLM Agent 处理的是数字世界（文本、API），但具身 Agent 处理的是物理世界（机器人、自动驾驶、IoT）。本文深入具身认知理论、感知-动作循环、sim-to-real 迁移，以及 LLM × Robotics 的最新交叉。

## 一、具身认知（Embodied Cognition）

### 1.1 核心命题

```text
传统认知科学：
  思维 = 大脑内的符号计算（与身体无关）

具身认知：
  思维 = 大脑 + 身体 + 环境的耦合涌现
  智能 = 行动 + 感知 + 环境的动力学
```

经典论据：

- 婴儿通过**抓握、爬行**理解"持久性"概念
- 概念"重"是抽象的，但隐喻映射到"理解负担重"
- 抽象空间推理（如方向）依赖**身体感官**（左/右）

### 1.2 对 AI 的启示

```text
符号 AI 失败：把"理解语言"当成脱离语境的符号操作 → 不接地
具身 AI 成功：把"理解"看成"能在环境中正确行动" → 接地

例：让 LLM 看"苹果"的照片 vs 让机器人拿起苹果 → 后者真正"知道"苹果的重量、触感
```

## 二、感知-动作循环

### 2.1 经典模型

```text
Percept → World Model → Plan → Action → Environment
   ↑                                            │
   └────────────────────────────────────────────┘
                    反馈
```

具身 Agent 每一步都从环境中获得**新感知**（视觉、触觉、听觉、本体感觉），更新世界模型，再决定下一个动作。

### 2.2 视觉导航示例

```python
class VisionNavAgent:
    """具身导航 Agent：移动到目标位置"""
    def __init__(self):
        self.rgb_buffer = deque(maxlen=4)        # 4 帧历史
        self.position = (0, 0)
        self.map = OccupancyGrid()                # SLAM 地图

    def step(self, rgb: np.ndarray, depth: np.ndarray, gps: tuple):
        # 1. 感知：更新 buffer + 地图
        self.rgb_buffer.append(rgb)
        self.map.update(depth, gps)
        self.position = gps

        # 2. 决策：用视觉策略选下一步动作
        state = self.encode(rgb_buffer, self.map, self.position)
        action = self.policy(state)               # forward / left / right

        # 3. 执行：发到机器人控制
        self.controller.execute(action)
        return action
```

### 2.3 反馈的"延迟"

物理动作有真实延迟（10ms-1s），且**不可撤销**——你不能"撤回"掉下悬崖的动作。

LLM Agent 可以"撤回"任何文本动作（重新生成），但机器人踩下油门后只能等下一步。

## 三、世界模型

### 3.1 什么是世界模型

$$
\hat{s}_{t+1} = f_\theta(s_t, a_t)
$$

世界模型 $f_\theta$ 预测"在状态 $s_t$ 执行动作 $a_t$，下一状态会是什么"。

```text
感知 ──→ 编码 ──→ s_t ──→ f_θ ──→ ŝ_{t+1}  (预测)
                                       ↓
                                      损失 = ||s_{t+1} - ŝ_{t+1}||
```

### 3.2 LLM 作为世界模型

有趣的新趋势：用 LLM 当世界模型的"语义层"：

```python
class LLMWorldModel:
    """LLM 预测动作后果（语义级）"""
    def predict(self, state_desc: str, action: str) -> str:
        prompt = f"""
        当前状态：{state_desc}
        执行动作：{action}
        预测下一状态（描述）：
        """
        return self.llm.invoke(prompt)

# 例
state = "机器人在厨房，桌上有苹果"
next_state = model.predict(state, "把苹果拿到客厅")
# "机器人在客厅，手里握着苹果"
```

比传统 SLAM 更通用，但精度和速度不及专门模型。

## 四、Sim-to-Real 迁移

### 4.1 为什么需要仿真

```text
现实训练成本：
  - 1 台 Boston Dynamics Spot ≈ 75,000 USD
  - 摔倒维修：每次数千美元
  - 现实训练时长：1 周 = 数万元成本

仿真训练：
  - Isaac Sim / MuJoCo / Isaac Gym 免费
  - 1 天可跑 1000× 真实时长
  - 失败无成本
```

### 4.2 Reality Gap

仿真训练的策略在真实世界**直接迁移**往往失败：

```text
原因：
  - 物理引擎近似（摩擦、碰撞、空气阻力）
  - 传感器噪声模型不准确
  - 视觉渲染与真实相机差异
  - 未建模的动力学

对策：
  - Domain Randomization（仿真中随机化参数）
  - Domain Adaptation（real data fine-tune）
  - Real2Sim2Real（先从真实建仿真）
  - System Identification（标定物理参数）
```

### 4.3 Domain Randomization 示例

```python
# 训练时：每 episode 随机化物理参数
def env_randomized_episode():
    env.set_param("friction", uniform(0.1, 1.5))
    env.set_param("mass", uniform(0.5, 2.0))
    env.set_param("lighting", random_lighting())
    env.set_param("camera_fov", uniform(60, 100))
    return env
```

### 4.4 主流仿真器

| 仿真器 | 优势 | 适用 |
|---|---|---|
| **Isaac Sim** (NVIDIA) | GPU 加速、逼真渲染 | 机器人学习 |
| **MuJoCo** | 物理精度高 | 控制研究 |
| **PyBullet** | 易用、轻量 | 入门、教学 |
| **Habitat** | 室内导航 | 具身导航 |
| **CARLA** | 自动驾驶 | 驾驶仿真 |
| **Gazebo** | ROS 集成 | 工业机器人 |

## 五、LLM × Robotics

### 5.1 PaLM-E

Google 的 PaLM-E 把视觉、文本、控制融合到一个大模型：

```python
class PaLMEAgent:
    def step(self, image, instruction):
        # 输入：图像 + 文本指令
        # 输出：动作序列
        return self.model.generate(image, instruction)
```

### 5.2 RT-2（Robotics Transformer）

把机器人动作离散化为 token，和自然语言 token 一起训练：

```text
文本 token: "把苹果放到桌上"
动作 token: <ACTION> move_to(apple) <ACTION> grasp <ACTION> move_to(table) <ACTION> release
```

### 5.3 SayCan

让 LLM 输出"应该做什么"，配合 affordance function 输出"能做什么"，交集是实际动作：

$$
\text{Action} = \text{LLM}(\text{instruction}) \cap \text{Affordance}(\text{state})
$$

```python
def saycan_step(llm, affordance_fn, instruction, world_state):
    # LLM 给所有可能动作打分
    candidate_actions = list_affordance(world_state)
    scores = llm.score_actions(instruction, candidate_actions)

    # Affordance function 给出"能否执行"的概率
    can_execute = affordance_fn(world_state, candidate_actions)

    # 选择 LLM 评分 × 可执行概率最大的
    final_score = [s * p for s, p in zip(scores, can_execute)]
    return candidate_actions[argmax(final_score)]
```

### 5.4 工业落地：Figure 01 / Optimus / 1X Neo

2024-2025 年具身机器人进入工业：

```text
Figure 01 (Figure AI + OpenAI):
  - 多模态：视觉 + 语音 + 动作
  - 工厂装配演示

Tesla Optimus:
  - 重复性任务（搬运、装配）
  - 自家工厂试点

1X Neo (1X Technologies):
  - 家用场景
  - 远程遥操作 + 自主执行
```

## 六、安全与对齐

具身 Agent 的安全风险比纯文本 Agent 高得多：

```text
文本 Agent 的 worst case：吐出不雅内容
具身 Agent 的 worst case：
  - 撞人
  - 摔坏价值数千美元的设备
  - 误操作工业系统导致人身伤害
```

应对：

1. **安全屏障（safety filter）**：动作先过规则检查再执行。
2. **力 / 速度限制**：物理上限设置。
3. **应急停止**：操作员一键 kill switch。
4. **仿真验证**：新策略先在仿真中跑 100 万次。
5. **保守启动**：初期只在受控环境、低风险任务中部署。

## 七、未来方向

1. **VLA 模型**（Vision-Language-Action）：把视觉、语言、动作统一到一个大模型。
2. **世界模型的视频生成**：用 Sora 类模型预测"动作后的视频"，作为规划依据。
3. **多模态触觉 / 力反馈**：让 Agent "感觉到"自己在抓什么。
4. **群体机器人**：多 Agent 协作搬运、装配。
5. **仿真 + 真实双循环**：Sim-to-Real-to-Sim 持续迭代。

## 小结

具身 Agent 把 AI 从"理解语言"延伸到"在物理世界正确行动"。核心挑战是世界模型、感知-动作闭环、Sim-to-Real 迁移、安全约束。LLM + Robotics 的融合（PaLM-E、RT-2、SayCan）正在打破传统机器人学的边界，但具身智能的真正爆发还需要物理交互数据、仿真精度、安全机制的协同进步。
