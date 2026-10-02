# Sim-to-Real 迁移：从仿真到真实机器人

**Sim-to-Real 迁移** 是机器人学习的核心挑战——在仿真器训练的策略部署到真实机器人时，性能往往**急剧下降**。这个差距（sim2real gap）来自物理建模、感知噪声、控制差异等多个层面。本文系统介绍 sim2real 的核心问题、域随机化、域适应、系统识别、真实世界微调等关键技术，以及工业实践（Isaac Lab、MuJoCo、SAPIEN）。

## 一、为什么需要 Sim-to-Real

### 1. 真实机器人数据的成本

```python
REAL_ROBOT_COSTS = {
    "hardware": "工业机械臂 $50K+",
    "maintenance": "易损件、定期校准",
    "time": "一次实验 30min~数小时",
    "human": "遥操作员 $50~200/h",
    "safety": "需要安全围栏",
    "data_throughput": "1 个机器人 vs 仿真 10000 个并行",
}
```

**结论**：真实机器人数据**极其昂贵**——1 小时真实数据 ≈ 10000 GPU 小时的仿真数据。

### 2. 仿真的优势

```python
SIMULATION_ADVANTAGES = {
    "speed": "可加速到 1000x 真实时间",
    "parallelism": "10000+ 实例同时训练",
    "safety": "失败不损坏硬件",
    "labeling": "完美 ground truth（位置、力矩）",
    "reproducibility": "完全可复现",
}
```

### 3. 仿真的代价：Sim2Real Gap

```text
训练: 仿真，1M 步，成本 $100
部署: 真实，性能下降 50%
```

**问题**：这种 gap 让仿真训练价值大打折扣。

## 二、Sim2Real Gap 的根源

### 1. 物理差距

```python
PHYSICS_GAPS = {
    "friction": "库仑摩擦模型难以精确",
    "contact": "刚体接触假设 vs 实际柔性接触",
    "deformation": "可形变物体（衣物、绳子）难仿真",
    "fluid": "液体难以精确仿真",
    "wear": "关节磨损、齿轮间隙",
    "temperature": "温度对电机影响",
    "latency": "通信延迟、控制周期不稳定",
}
```

### 2. 感知差距

```python
PERCEPTION_GAPS = {
    "image_noise": "仿真完美 vs 真实有噪声、运动模糊",
    "lighting": "仿真恒定 vs 真实变化",
    "sensor_delay": "真实传感器有延迟",
    "depth": "RGB-D 相机对反光、透明物体失效",
    "textures": "仿真纹理与真实差异",
    "occlusion": "自遮挡、传感器遮挡",
    "lens_distortion": "鱼眼、畸变",
}
```

### 3. 动力学差距

```python
DYNAMICS_GAPS = {
    "mass_inertia": "物体实际参数难测",
    "actuator_dynamics": "电机响应非线性、滞后",
    "joint_backlash": "齿轮间隙",
    "cable_routing": "线缆影响运动",
    "temperature_effect": "电机温度影响性能",
    "battery": "电池电压变化",
}
```

## 三、域随机化（Domain Randomization）

### 1. 核心思想

**让仿真足够多样，策略自然泛化**：

```python
def randomize_physics():
    """随机化物理参数。"""
    return {
        "friction": np.random.uniform(0.1, 1.5),
        "mass": np.random.uniform(0.5, 2.0),
        "com_offset": np.random.uniform(-0.05, 0.05, size=3),
        "motor_strength": np.random.uniform(0.8, 1.2),
        "joint_damping": np.random.uniform(0.0, 0.5),
    }


def randomize_visual():
    """随机化视觉参数。"""
    return {
        "light_intensity": np.random.uniform(0.3, 1.5),
        "light_position": np.random.uniform(-2, 2, size=3),
        "light_color": np.random.uniform(0.5, 1.5, size=3),
        "camera_position": np.random.uniform(-0.05, 0.05, size=3),
        "camera_noise": np.random.uniform(0, 0.05),
        "background": random.choice([...]),
        "textures": random.choice([...]),
    }


class RandomizedEnv:
    """每次 reset 时随机化环境。"""
    def reset(self):
        self.physics = randomize_physics()
        self.visual = randomize_visual()
        return self.observe()
```

### 2. 随机化的范围

**随机化范围太小 → 欠泛化**
**随机化范围太大 → 任务变得无法学习**

```python
def adaptive_randomization(performance_history):
    """自适应随机化——根据策略表现调整范围。"""
    # 如果策略在某个范围表现好 → 扩大范围
    # 如果表现差 → 缩小范围
    ...
```

### 3. 代表工作

```python
DOMAIN_RAND_PAPERS = {
    "OpenAI et al. 2017": "DQN + sim2real for Rubik's cube",
    "Tobin et al. 2017": "Domain randomization for object detection",
    "Peng et al. 2018": "Sim2real for locomotion",
    "Akkaya et al. 2019": "Solving Rubik's cube with a robot hand",
}
```

## 四、域适应（Domain Adaptation）

### 1. 核心思想

**对齐仿真与真实分布**——而非只靠随机化。

### 2. 视觉域适应

#### CycleGAN：把仿真图像变成"真实风格"

```python
class CycleGAN:
    """把仿真图像风格迁移到真实。"""
    def __init__(self):
        self.G_sim2real = Generator()  # sim → real
        self.G_real2sim = Generator()  # real → sim
        self.D_sim = Discriminator()    # 判别 sim
        self.D_real = Discriminator()   # 判别 real
    
    def train_step(self, sim_image, real_image):
        # 1) 对抗损失
        # 让 G_sim2real 的输出被 D_real 接受
        fake_real = self.G_sim2real(sim_image)
        d_loss = -log(D_real(real_image)) - log(1 - D_real(fake_real))
        
        # 2) 循环一致性损失
        # real → sim → real 应该等于 original real
        reconstructed = self.G_sim2real(self.G_real2sim(real_image))
        cycle_loss = F.l1_loss(reconstructed, real_image)
        
        # 3) 总损失
        loss = d_loss + lambda_cycle * cycle_loss
```

#### 现实差距归一化

```python
def reality_gap_normalization(obs):
    """观测归一化——消除仿真与真实分布差异。"""
    # 用真实样本的均值/方差归一化
    # 不是用仿真样本的统计量
    return (obs - real_mean) / (real_std + 1e-6)


def collect_real_distribution(robot, n_samples=1000):
    """从真实机器人收集样本分布。"""
    samples = []
    for _ in range(n_samples):
        obs = robot.observe_random_state()
        samples.append(obs)
    return np.stack(samples)
```

### 3. 特征级域适应

```python
class DomainInvariantFeatures(nn.Module):
    """学 sim/real 不变的特征。"""
    def __init__(self):
        self.encoder = CNN()
        self.domain_classifier = nn.Linear(...)
    
    def forward(self, x):
        features = self.encoder(x)
        domain_pred = self.domain_classifier(features)
        return features, domain_pred
    
    def loss(self, features, domain_pred, true_domain):
        # 1) 任务损失
        task_loss = self.task_head(features, ...)
        # 2) 域分类损失（梯度反转）
        domain_loss = F.cross_entropy(domain_pred, true_domain)
        # 3) 训练 encoder 让 domain_classifier 失败
        return task_loss - lambda_d * domain_loss
```

## 五、系统识别（System Identification）

### 1. 核心思想

**用真实数据校准仿真器参数**——缩小物理 gap。

```python
def identify_robot_params():
    """从真实轨迹估计仿真器参数。"""
    # 1) 收集：实际控制 + 实际轨迹
    real_data = collect_real_trajectories(n=100)
    
    # 2) 优化仿真器参数让仿真轨迹 ≈ 实际轨迹
    params = {
        "friction": 0.5,
        "mass": 1.0,
        "com": [0, 0, 0],
        "motor_strength": 1.0,
    }
    
    optimizer = torch.optim.Adam([params], lr=0.01)
    for step in range(n_optim_steps):
        # 仿真在当前 params 下运行
        sim_traj = simulate(params, real_data["controls"])
        # 计算与真实轨迹差距
        loss = F.mse_loss(sim_traj, real_data["trajectories"])
        # 更新参数
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    return params
```

### 2. 域随机化中的系统识别

```python
class AutoDR:
    """自动域随机化（AutoDR）。"""
    def __init__(self, env):
        self.env = env
        self.param_range = {
            "friction": [0.1, 2.0],
            "mass": [0.5, 2.0],
            # ...
        }
    
    def update_ranges(self, success_rate):
        """根据真实机器人成功率调整参数范围。"""
        for param in self.param_range:
            if success_rate[param] > 0.9:
                # 策略在这个范围表现好 → 扩大
                self.param_range[param].expand()
            elif success_rate[param] < 0.3:
                # 表现差 → 缩小
                self.param_range[param].shrink()
```

### 3. 残差动力学建模

```python
class ResidualDynamicsModel:
    """仿真 + 残差模型——仿真器 + 学到的真实差距。"""
    def __init__(self, sim_model, residual_net):
        self.sim = sim_model
        self.residual = residual_net  # NN 学习 sim_real gap
    
    def forward(self, state, action):
        # 仿真器预测
        sim_next = self.sim(state, action)
        # 残差
        residual = self.residual(state, action)
        return sim_next + residual
```

## 六、真实世界微调（Real-World Fine-tuning）

### 1. Sim-to-Real + Real Fine-tuning

```python
class SimToRealFineTune:
    """先在仿真训练，再在真实微调。"""
    def __init__(self, policy):
        self.policy = policy
    
    def train(self, n_real_episodes=20):
        # 1) 仿真训练（已训练好）
        # policy.learn_sim(n_steps=1M)
        
        # 2) 真实微调
        for episode in range(n_real_episodes):
            traj = self.collect_real_traj()
            loss = self.policy.bc_loss(traj)
            self.policy.update(loss)
        
        return self.policy
```

### 2. 人类示范数据增强

```python
def augment_with_human_demos(policy, human_demos):
    """用人类遥操作示范微调。"""
    # 1) 仿训训过 base policy
    # 2) 用人类示范数据 fine-tune
    for demo in human_demos:
        loss = F.mse_loss(policy(demo["obs"]), demo["action"])
        policy.update(loss)
```

### 3. RL Fine-tuning in Real World

```python
class RealWorldRL:
    """在真实机器人上做 RL——安全第一。"""
    def __init__(self, robot, policy, safety_checker):
        self.robot = robot
        self.policy = policy
        self.safety = safety_checker
    
    def step(self):
        # 1) 策略选动作
        action = self.policy.act(self.observe())
        
        # 2) 安全检查
        if not self.safety.is_safe(action):
            action = self.safety.safest_action()
        
        # 3) 真实执行
        self.robot.execute(action)
        
        # 4) 收集经验
        self.update_policy(reward=compute_reward())
```

代表：**NVIDIA Isaac** 的 sim2real RL。

### 4. 人类参与学习

```python
class HumanInTheLoopLearning:
    """人类参与学习——失败时人类接管。"""
    def step(self):
        action = self.policy.act(obs)
        
        try:
            self.robot.execute(action)
        except Exception:
            # 失败时人类接管
            human_action = self.human.teleop()
            self.record_human_demo(obs, human_action)
            
            # 用人类示范更新
            self.policy.update(obs, human_action)
```

代表：**SERL**（Sample-efficient real-world RL）。

## 七、关键工具与平台

### 1. NVIDIA Isaac Lab

```python
class IsaacLabTraining:
    """Isaac Lab 训练 pipeline。"""
    def __init__(self):
        # GPU 并行仿真
        self.num_envs = 4096
    
    def train(self, task, n_steps=10_000_000):
        # 1) 创建任务
        env = IsaacLabEnv(task)
        
        # 2) 训练
        policy = SAC(env)
        policy.learn(n_steps, callback=DomainRandomizationCallback())
        
        # 3) 部署到真实
        self.deploy(policy, real_robot=...)
```

### 2. MuJoCo + MuJoCo MPC

```python
class MuJoCoTraining:
    """用 MuJoCo 仿真训练。"""
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("robot.xml")
    
    def train_mpc(self):
        # MPC（Model Predictive Control）——不直接学策略
        # 实时规划
        ...
```

### 3. SAPIEN / ManiSkill

```python
class SAPIENTraining:
    """操作任务专用仿真器。"""
    def __init__(self):
        self.sim = sapien.Engine()
    
    def setup_manipulation_task(self, task_name):
        """设置操作任务。"""
        task = TASKS[task_name]  # "OpenDrawer", "PickObject", ...
        return task
```

### 4. Habitat / AI2-THOR

```python
class HabitatNavTraining:
    """室内导航仿真。"""
    def __init__(self):
        self.env = HabitatEnv(config="indoor_nav.yaml")
    
    def train_visual_nav(self):
        # 训练"从 A 到 B"导航
        ...
```

### 5. Drone Simulators

```python
DRONE_SIMULATORS = {
    "AirSim": "Microsoft，无人机/自动驾驶",
    "Flightmare": "ETH，多无人机",
    "GymFC": "飞行控制",
    "CrazyFlie Sim": "小型四旋翼",
}
```

## 八、Sim2Real 中的常见失败与解决

### 1. 训练时不动的物体在真实会动

```python
# 问题：仿真中杯子很稳 → 训练用手稳稳抓
# 现实：杯子可能在滑动 → 抓空

# 解决：在仿真中加入物体动力学随机性
def randomize_object_dynamics():
    return {
        "object_friction": np.random.uniform(0.1, 1.5),
        "object_initial_vel": np.random.uniform(0, 0.1),
    }
```

### 2. 真实相机延迟

```python
# 解决：在仿真中加入感知延迟
class DelayedObsWrapper:
    def __init__(self, env, delay_steps=3):
        self.env = env
        self.delay_steps = delay_steps
        self.history = collections.deque(maxlen=delay_steps)
    
    def step(self, action):
        obs = self.env.step(action)
        self.history.append(obs)
        return self.history[0]  # 旧观测
```

### 3. 关节实际限位与仿真不同

```python
# 解决：仿真中扩大关节限位
def randomize_joint_limits(nominal_low, nominal_high, slack=0.1):
    return {
        "low": nominal_low - slack,
        "high": nominal_high + slack,
    }
```

### 4. 仿真器步长不匹配

```python
# 问题：仿真 1000 Hz，真实 100 Hz
# 解决：让策略在仿真也用 100 Hz
# 或者在仿真随机化 dt
def randomize_timestep():
    return np.random.uniform(0.001, 0.011)  # 90 Hz ~ 1000 Hz
```

## 九、特定领域的 Sim2Real

### 1. 抓取

```python
class GraspSim2Real:
    """6-DoF 抓取的 sim2real。"""
    def randomize(self):
        return {
            # 视觉
            "lighting": random_lighting(),
            "background": random_background(),
            "camera_pose_noise": random_camera_noise(),
            # 物理
            "object_friction": random_friction(),
            "object_mass": random_mass(),
            # 抓取
            "gripper_friction": random_gripper_friction(),
            "grasp_point_jitter": random_grasp_jitter(),
        }
```

代表：**Dex-Net**、**Contact-GraspNet**、**GG-CNN**。

### 2. 四足 / 双足运动

```python
class LocomotionSim2Real:
    """机器人运动控制。"""
    def randomize(self):
        return {
            # 机器人参数
            "motor_strength": random_motor_strength(),
            "joint_damping": random_damping(),
            "payload_mass": random_payload(),
            # 地形
            "friction": random_friction(),
            "slope": random_slope(),
            "obstacles": random_obstacles(),
            # 感知
            "imu_noise": random_imu_noise(),
            "latency": random_latency(),
        }
```

代表：**ANYmal**、**Spot Mini**、**Cassie**。

### 3. 自动驾驶

```python
class DrivingSim2Real:
    """自动驾驶 sim2real。"""
    def randomize(self):
        return {
            # 视觉
            "weather": random_weather(),
            "time_of_day": random_time(),
            "traffic_density": random_traffic(),
            # 物理
            "vehicle_mass": random_mass(),
            "tire_friction": random_friction(),
            # 通信
            "v2x_delay": random_v2x_delay(),
        }
```

代表：**CARLA**、**Waymax**、**nuScenes**。

## 十、Sim2Real 的根本局限

### 1. 复杂物理难以仿真

```python
HARD_TO_SIMULATE = {
    "deformable_objects": "衣物、绳子、海绵",
    "fluids": "液体、气体",
    "contact_rich": "复杂接触、摩擦",
    "soft_robots": "柔性机器人",
    "human_interaction": "与人交互的复杂场景",
}
```

### 2. 长视野任务

```python
# 仿真: 完美执行无错误
# 真实: 累计误差 → 偏离任务
```

### 3. 未建模的动力学

```python
# 仿真中假设电机线性
# 真实电机：非线性、滞后、温升
```

**结论**：Sim2Real 不是万能——某些领域需要**真实数据为主**。

## 十一、混合方法：仿真 + 真实

### 1. Sim-and-Real Co-training

```python
class SimRealCoTraining:
    """仿真 + 真实联合训练。"""
    def __init__(self):
        self.sim_buffer = ReplayBuffer(capacity=10_000_000)
        self.real_buffer = ReplayBuffer(capacity=10_000)
    
    def train_step(self):
        # 采样 50% 仿真 + 50% 真实
        sim_batch = self.sim_buffer.sample(batch=128)
        real_batch = self.real_buffer.sample(batch=128)
        
        # 训练
        loss = self.policy.loss(sim_batch) + self.policy.loss(real_batch)
        self.policy.update(loss)
```

### 2. Offline RL in Real Data

```python
class OfflineRealRL:
    """只用真实数据训练（不在线交互）。"""
    def __init__(self, real_dataset):
        self.data = real_dataset  # 离线收集的真实轨迹
    
    def train(self, n_steps=100_000):
        policy = SAC.pretrain(self.data, n_steps)
        return policy
```

代表：**CQL**、**IQL**、**DT**（Decision Transformer）。

## 十二、给 Sim2Real 团队的清单

1. **评估 sim2real gap**：先量化差距大小。
2. **域随机化起步**：从基础物理参数随机化开始。
3. **真实数据关键**：必须有少量真实数据来验证。
4. **安全部署**：所有 sim2real 部署都需要安全检查。
5. **持续微调**：部署后持续在真实数据上微调。
6. **硬件差异**：硬件磨损会让 gap 持续变化。
7. **域随机化范围**：从保守开始，根据性能调整。
8. **系统识别**：对精度要求高的场景（装配）必要。
9. **混合训练**：仿真 + 真实数据混合训练更稳。
10. **失败模式分析**：分析部署失败的样本，反馈到仿真。

## 小结

Sim-to-Real 是机器人学习的**核心挑战**——仿真器的速度优势与真实差距并存。**关键技术**包括域随机化（让仿真足够多样）、域适应（对齐 sim/real 分布）、系统识别（校准仿真器参数）、真实微调（用少量真实数据纠偏）。**代表平台** NVIDIA Isaac Lab、MuJoCo、SAPIEN 让大规模训练成为可能。**Sim2Real 没有银弹**——不同任务需要不同组合。**真正成功的 sim2real** 是"仿真加速 + 真实精修 + 持续迭代"的闭环。下一篇我们将看到机器人的核心能力——**机械臂操作**：让机器人完成复杂物理任务。
