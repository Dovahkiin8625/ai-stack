# 机器人操作：从抓取到灵巧手

**机器人操作（Robot Manipulation）** 是机器人学的核心挑战——让机器人完成抓取、推动、装配、折叠等物理任务。从亚马逊仓库的 Kiva 机器人，到 Tesla 的 Optimus 拧螺丝，再到 π₀ 叠衣服，操作技术已经从"专用"走向"通用"。本文系统介绍操作的核心任务（抓取、推动、装配、灵巧手）、核心算法（Diffusion Policy、ACT、6-DoF Grasp）、代表系统，以及从 demo 到生产的转化路径。

## 一、操作的核心任务

```text
┌────────────────────────────────────────────────────────┐
│  Level 1: 简单抓取                                       │
│     Pick-and-place：从 A 抓放到 B                          │
│     工业应用最成熟                                        │
└────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│  Level 2: 接触丰富操作                                    │
│     推动、插入、装配                                      │
│     需要精确力控                                          │
└────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│  Level 3: 形变体操作                                     │
│     折叠衣物、揉面团、绑线                                 │
│     难以建模                                              │
└────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│  Level 4: 双手机器人                                      │
│     装配、做饭、叠衣服                                    │
│     长视野任务                                            │
└────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│  Level 5: 灵巧手                                         │
│     多指手部操作                                         │
│     人类水平灵巧度                                        │
└────────────────────────────────────────────────────────┘
```

每一级都比上一级**难度指数级增加**。

## 二、操作任务分类

### 1. 按动作粒度

```python
MANIPULATION_TASKS = {
    "pick": "简单抓取",
    "place": "放置",
    "push": "推动",
    "slide": "滑动",
    "insert": "插入",
    "screw": "拧螺丝",
    "fold": "折叠",
    "pour": "倒",
    "stir": "搅拌",
    "wipe": "擦拭",
    "cut": "切割",
    "stake": "打桩",
}
```

### 2. 按物体类型

| 物体类型 | 难度 | 例子 |
|---|---|---|
| **刚性规则物体** | 易 | 杯子、盒子 |
| **关节物体** | 中 | 抽屉、剪刀、笔记本电脑 |
| **形变物体** | 难 | 衣物、毛巾、海绵 |
| **流体** | 极难 | 水、油 |
| **粉体** | 极难 | 沙、面粉 |

### 3. 按时间视野

```text
短视野 (1-10 秒):
  抓取、放置、推

中视野 (10 秒-1 分钟):
  倒水、拧瓶盖、折叠一张纸

长视野 (1 分钟 - 数小时):
  做一道菜、整理房间、装配合金
```

## 三、核心算法

### 1. 6-DoF 抓取

```python
class GraspNet(nn.Module):
    """6-DoF 抓取网络。"""
    def __init__(self):
        self.pointnet = PointNetEncoder()
        self.grasp_head = GraspHead()
    
    def forward(self, point_cloud):
        """
        输入: 点云 (N, 3)
        输出: 多个 6-DoF 抓取姿态
        """
        # 1) 点云编码
        features = self.pointnet(point_cloud)
        # 2) 抓取预测
        grasps = self.grasp_head(features)
        # 每个抓取: [position(3), orientation(rotation matrix 3x3), width, score]
        return grasps


class GraspHead(nn.Module):
    """输出多个抓取候选。"""
    def __init__(self, n_grasps=100):
        super().__init__()
        self.mlp = nn.Sequential(...)
        # 输出 6-DoF 抓取 + 置信度
        self.head = nn.Linear(..., 7)  # 3 位置 + 4 四元数
    
    def forward(self, features):
        return self.head(features)  # (N_grasps, 7)
```

代表：**GraspNet**（NVIDIA）、**Contact-GraspNet**、**GG-CNN**。

### 2. Diffusion Policy

**当前操作领域的 SOTA**：

```python
class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy: 用 Diffusion 生成动作序列。
    论文: Chi et al., 2023
    """
    def __init__(self, action_dim, obs_dim, n_action_steps=8):
        super().__init__()
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        
        # Condition encoder
        self.cond_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        # 1D UNet for diffusion
        self.unet = ConditionalUNet1D(
            in_channels=action_dim,
            cond_dim=256,
            diffusion_step_embed_dim=128,
        )
    
    def forward(self, obs):
        """
        obs: 当前观测
        返回: 未来 n_action_steps 步的动作序列
        """
        # 1) Condition 编码
        cond = self.cond_encoder(obs)
        
        # 2) 噪声动作序列
        noisy_action = torch.randn(1, self.n_action_steps, self.action_dim)
        
        # 3) 迭代去噪
        for t in reversed(range(self.n_diffusion_steps)):
            noisy_action = self.unet(noisy_action, t, cond)
        
        return noisy_action
```

**优势**：
- 生成多模态动作（多种合法解）。
- 平滑（动作序列一起生成）。
- 稳定训练。

**代表应用**：
- **Push-T**：推动任务。
- **复杂操作**：插孔、装配。
- **长视野任务**：做饭、装配。

### 3. ACT（Action Chunking Transformer）

```python
class ACT(nn.Module):
    """
    ACT: Action Chunking Transformer。
    论文: Zhao et al., 2023
    """
    def __init__(self):
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.action_head = nn.Linear(d_model, action_dim)
    
    def forward(self, images, instruction):
        # 1) 编码多视角图像
        features = self.encoder(images, instruction)
        
        # 2) 解码为动作 chunk（10-100 步）
        action_chunk = self.decoder(features)
        
        # 3) 输出动作
        return self.action_head(action_chunk)
```

**核心创新**：动作分块——一次生成 10~100 步，避免单步误差累积。

### 4. Implicit Policy

```python
class ImplicitPolicy(nn.Module):
    """Implicit Policy: 用能量函数表示策略。"""
    def __init__(self):
        self.energy_net = nn.Sequential(...)
    
    def act(self, state):
        # 找动作使能量最低
        action = self.optimize(state)
        return action
```

代表：**Implicit Behaviour Cloning (IBC)**。

### 5. Diffusion-based Grasping

```python
class GraspDiffusion(nn.Module):
    """用 Diffusion 生成 6-DoF 抓取。"""
    def __init__(self):
        self.unet = UNet3D()
    
    def forward(self, point_cloud, condition=None):
        # 1) 把点云体素化
        voxel = voxelize(point_cloud)
        # 2) Diffusion 生成 grasp heatmap
        for t in reversed(range(n_steps)):
            voxel = self.unet(voxel, t)
        return voxel  # 包含抓取概率
```

代表：**Contact-GraspNet**、**GraspDiffusion**。

## 四、Diffusion Policy 实战

### 1. 数据收集

```python
class DemonstrationCollector:
    """收集演示数据。"""
    def __init__(self, robot, teleop_device):
        self.robot = robot
        self.teleop = teleop_device  # VR 控制器 / 键盘 / 拖动
    
    def collect_episode(self, task_instruction):
        obs_list = []
        action_list = []
        
        obs = self.robot.reset()
        while not self.robot.task_done():
            # 人通过 teleop 设备控制
            action = self.teleop.get_action()
            obs_list.append(obs)
            action_list.append(action)
            
            obs = self.robot.step(action)
        
        return {
            "instruction": task_instruction,
            "observations": obs_list,
            "actions": action_list,
        }
```

### 2. 训练

```python
def train_diffusion_policy(policy, dataset, n_epochs=1000):
    """训练 Diffusion Policy。"""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    
    for epoch in range(n_epochs):
        for batch in dataset:
            obs = batch["observations"]
            actions = batch["actions"]
            
            # 1) 随机采样扩散 timestep
            t = torch.randint(0, n_diffusion_steps, (batch_size,))
            
            # 2) 加噪
            noise = torch.randn_like(actions)
            noisy_actions = add_noise(actions, noise, t)
            
            # 3) 预测噪声
            noise_pred = policy.unet(noisy_actions, t, obs)
            
            # 4) MSE 损失
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
```

### 3. 部署

```python
class DiffusionPolicyDeployment:
    """Diffusion Policy 部署。"""
    def __init__(self, policy, robot):
        self.policy = policy
        self.robot = robot
    
    def step(self):
        obs = self.robot.observe()
        action_chunk = self.policy(obs)  # 8 步动作
        
        # 逐步执行（带 temporal ensembling 平滑）
        for i in range(8):
            action = temporal_ensemble(action_chunk, i)
            self.robot.step(action)
```

### 4. 时序集成（Temporal Ensembling）

```python
def temporal_ensemble(action_chunks, step_idx):
    """
    用指数加权平均融合多个 action chunk。
    """
    weights = [0.8 ** i for i in range(len(action_chunks))]
    weights = np.array(weights) / sum(weights)
    
    # 加权平均
    ensembled = sum(
        w * chunk[step_idx] 
        for w, chunk in zip(weights, action_chunks)
    )
    return ensembled
```

**效果**：动作更平滑，减少 jitter。

## 五、关键操作任务

### 1. Pick-and-Place（最成熟）

```python
class PickAndPlace:
    """经典抓放任务。"""
    def run(self):
        # 1) 感知：检测目标物体位置
        obj_pose = self.vision.detect_object()
        
        # 2) 规划：到抓取位姿
        grasp_pose = self.grasp_planner.compute_grasp(obj_pose)
        
        # 3) 执行：移动 + 抓取
        self.arm.move_to(grasp_pose.pre_grasp)
        self.arm.move_to(grasp_pose.grasp)
        self.gripper.close()
        self.arm.lift()
        
        # 4) 放到目标位置
        place_pose = self.target_pose
        self.arm.move_to(place_pose)
        self.gripper.open()
        self.arm.retreat()
```

### 2. 接触丰富操作

#### 装配

```python
class PegInHole:
    """插孔任务。"""
    def run(self):
        # 1) 视觉定位
        hole_pose = self.vision.detect_hole()
        peg_pose = self.arm.current_pose()
        
        # 2) 接近 + 对齐
        self.arm.move_to(hole_pose.approach)
        
        # 3) 接触 + 力控
        while not self.is_inserted():
            force = self.ft_sensor.read()
            # 根据力反馈调整
            correction = self.compliance_controller(force)
            self.arm.move_delta(correction)
    
    def compliance_controller(self, force):
        """阻抗控制：根据外力调整动作。"""
        # F = K * Δx + D * Δv
        # 简单版本：把外力转化为位置修正
        return force * self.compliance_gain
```

#### 推动

```python
class PushingPolicy:
    """推动策略。"""
    def run(self):
        # 推动是 contact-rich 任务
        # Diffusion Policy 在 Push-T 上 SOTA
        
        obs = self.robot.observe()  # 包含物体位置
        action = self.diffusion_policy(obs)
        self.robot.step(action)
```

### 3. 形变体操作（最前沿）

```python
class DeformableManipulation:
    """形变物体操作（衣物、毛巾）。"""
    def run(self):
        # 关键挑战：
        # - 状态空间巨大
        # - 难以观测
        # - 难以仿真
        
        # 方法 1: 用关键点表示形变状态
        keypoints = self.vision.detect_keypoints(cloth)
        
        # 方法 2: 用 Diffusion Policy
        action = self.diffusion_policy(obs)  # 含 cloth 状态
        
        # 方法 3: 双手机器人协同
        left_action, right_action = self.dual_arm_policy(obs)
```

代表：**Aloha**（斯坦福低成本双臂）、**π₀**（Physical Intelligence）。

### 4. 灵巧手操作

```python
class DexterousHandPolicy:
    """多指灵巧手控制。"""
    def __init__(self):
        self.hand = AllegroHand()  # 16-DoF
        self.policy = DiffusionPolicy(
            action_dim=16,  # 16 个关节
        )
    
    def run(self):
        # 1) 灵巧手感知
        tactile = self.hand.read_tactile_sensors()
        proprio = self.hand.read_joint_positions()
        
        # 2) 灵巧操作
        joint_cmd = self.policy(obs)
        self.hand.move(joint_cmd)
```

代表：**Shadow Hand**（24-DoF）、**Allegro**（16-DoF）、**LEAP Hand**（低成本）。

## 六、仿真与基准

### 1. 仿真器

```python
MANIPULATION_SIMULATORS = {
    "Isaac Sim": "NVIDIA, GPU 并行",
    "MuJoCo": "经典物理",
    "SAPIEN": "操作任务专用",
    "PyBullet": "轻量",
    "Gazebo": "ROS 集成",
    "ManiSkill2/3": "Berkeley, 视觉操作",
    "RoboSuite": "Stanford, 操作",
    "RLBench": "CoppeliaSim",
}
```

### 2. 基准

```python
MANIPULATION_BENCHMARKS = {
    "RoboNet": "多种操作任务视频数据集",
    "BridgeData": "UC Berkeley, 跨场景操作",
    "Open X-Embodiment": "Google, 22 种机器人",
    "DROID": "Stanford + Google, 大规模操作",
    "MimicGen": "NVIDIA, 仿真数据生成",
    "ManiSkill2": "操作 benchmark",
}
```

### 3. 仿真数据生成

```python
class MimicGenDataPipeline:
    """从少量演示生成大量仿真数据。"""
    def __init__(self, source_demos):
        self.source = source_demos  # 10~50 个专家演示
    
    def generate_dataset(self, n_trajectories=10000):
        """通过场景变换生成大规模数据。"""
        generated = []
        for i in range(n_trajectories):
            # 1) 随机化场景
            scene = self.randomize_scene()
            
            # 2) 用 source demo 通过运动规划生成轨迹
            traj = self.kinematic_mimic(scene, self.source)
            
            generated.append(traj)
        return generated
```

## 七、感知融合

### 1. 多模态感知

```python
class MultiModalObservation:
    """多模态操作感知。"""
    def __init__(self):
        self.rgb_camera = RGBD()           # 主相机
        self.wrist_camera = RGBD()         # 手腕相机
        self.tactile_sensor = Tactile()    # 触觉
        self.ft_sensor = ForceTorque()     # 力矩
        self.proprio = JointState()        # 关节状态
    
    def get_observation(self):
        return {
            "rgb": self.rgb_camera.read(),
            "depth": self.rgb_camera.read_depth(),
            "wrist_rgb": self.wrist_camera.read(),
            "tactile": self.tactile_sensor.read(),
            "force_torque": self.ft_sensor.read(),
            "proprio": self.proprio.read(),
        }
```

### 2. 触觉感知

```python
class TactileProcessing:
    """触觉信号处理。"""
    def __init__(self):
        # DIGIT, GelSight 等触觉传感器
        self.sensor = GelSightSensor()
        self.tacnet = TactileNet()
    
    def get_observation(self):
        raw = self.sensor.read()  # 触觉图像
        features = self.tacnet(raw)  # 嵌入向量
        return features
    
    def detect_slip(self):
        """检测滑动——触觉特有信号。"""
        tactile = self.sensor.read()
        # 用滑动检测模型
        return self.slip_detector(tactile)
```

### 3. 力矩感知

```python
class ForceTorqueControl:
    """力矩反馈控制。"""
    def __init__(self):
        self.ft_sensor = ForceTorqueSensor()
    
    def compliant_move(self, target_pose):
        """顺应控制——遇到外力时顺应而非硬推。"""
        while not at_target():
            force = self.ft_sensor.read()
            
            # 阻抗控制
            pos_error = target_pose - self.arm.current_pose()
            torque = self.stiffness @ pos_error + self.damping @ velocity
            
            # 减去外力影响
            torque -= force
            
            self.arm.apply_torque(torque)
```

## 八、代表系统

### 1. π₀（Physical Intelligence, 2024）

**通用机器人操作基础模型**：

```text
能力:
- 叠衣服
- 装洗碗机
- 做早餐（煎鸡蛋）
- 整理房间
- 装箱

参数: ~3B（推测）
训练数据: 多种机器人平台
```

### 2. Aloha（Stanford）

**低成本双臂遥操作平台**：

```python
class AlohaSetup:
    """Aloha：低成本双臂。"""
    def __init__(self):
        # 2 个低成本机械臂
        self.left_arm = AlohaArm()
        self.right_arm = AlohaArm()
        # 2 个低成本夹爪
        self.left_gripper = AlohaGripper()
        self.right_gripper = AlohaGripper()
        # 4 个相机
        self.cameras = [AlohaCamera() for _ in range(4)]
    
    def collect_demo(self, task):
        # 遥操作员操作两个手臂
        ...
```

### 3. Mobile ALOHA / ALOHA 2

```text
Mobile ALOHA:
- ALOHA + 移动底盘
- 自主做饭、做家务

ALOHA 2 (2024):
- 更稳定硬件
- 更好的遥操作体验
- 大量演示数据
```

### 4. Figure 01 / Optimus 操作能力

```python
class Figure01Manipulation:
    """Figure 01 操作能力。"""
    def demo_tasks(self):
        return [
            "pick cup and place",
            "open drawer",
            "give object to human",
            "fold towel",
        ]
```

## 九、操作的关键挑战

### 1. 长视野任务

```text
短任务: pick-and-place (5 秒) — 简单
中任务: 装配 (1 分钟) — 中等
长任务: 做饭 (30 分钟) — 极难
```

**挑战**：误差累积、状态空间爆炸、任务分支。

**解决**：
- 分层：high-level planner + low-level controller。
- Memory：保存任务进度。
- Foundation models。

### 2. 形变物体

```text
当前局限:
- 衣物折叠成功率 < 50%
- 流体操作基本不可用
- 复杂接触动力学难以建模
```

### 3. 双手机器人

```python
class BimanualCoordination:
    """双手机器人协调——复杂。"""
    def run(self):
        # 关键挑战：
        # - 双臂之间的碰撞避免
        # - 协调动作（左手扶，右手拧）
        # - 任务分配（哪个手做什么）
```

### 4. 实时性

```python
# 操作要求：100Hz 控制
# Diffusion Policy：30Hz 推理
# 解决：模型蒸馏、量化
```

## 十、未来的方向

### 1. 基础模型路线

```python
class ManipulationFoundationModel:
    """通用操作基础模型。"""
    def __init__(self):
        # 跨多种机器人
        # 多种任务
        # 大规模数据
        ...
```

代表：**π₀**、**Figure Helix**、**DexVLG**。

### 2. 力控与触觉融合

```python
class ForceAwarePolicy:
    """力觉感知策略。"""
    def forward(self, rgb, force, instruction):
        return self.policy(rgb, force, instruction)
```

### 3. 视频预测与世界模型

```python
class WorldModelManipulation:
    """用世界模型预测动作后果。"""
    def forward(self, state, action):
        next_state_pred = self.world_model(state, action)
        return next_state_pred
```

### 4. 人机协作

```text
未来工厂:
- 人做精细操作
- 机器人做重活
- 协作完成复杂任务
```

## 十一、给操作团队的清单

1. **从 pick-and-place 开始**：最成熟，迭代快。
2. **用 Diffusion Policy**：当前 SOTA 范式。
3. **收集高质量演示**：人工遥操作为主。
4. **仿真生成数据**：用 MimicGen 等工具扩大数据。
5. **多视角相机**：覆盖盲区。
6. **触觉感知**：滑动检测等关键。
7. **力控**：装配任务必需。
8. **安全围栏**：物理保护。
9. **失败模式分析**：分析失败场景，迭代改进。
10. **渐进式任务**：从易到难。

## 小结

机器人操作是 AI 与机器人学的**核心交叉**——从简单的抓放，到装配、折叠、双手机器人、灵巧手。**核心技术**包括 6-DoF 抓取、Diffusion Policy、ACT、多模态感知融合。**当前 SOTA** 集中在简单 Pick-and-Place 和中等难度的接触丰富任务，**前沿挑战**是形变体操作、灵巧手、长视野任务。**真正成功的操作机器人** 不是"什么都能做"，而是**特定任务上稳定、可靠、安全**。**未来**是基础模型路线、世界模型、人机协作。三篇文章覆盖了 robotics 的核心：具身智能、Sim-to-Real、操作。10-applications 章节至此完成——下一篇（也是本批次最后一篇）我们将转向 **11-tools-ecosystem**。
