# NeurIPS 2025 回顾：Agent 与长上下文之年

NeurIPS 2025（12月7–13日，新墨西哥州阿尔伯克基）共收到 **18,940 篇投稿**（YoY +6.2%），录用 4,795 篇（录用率 25.3%）。本届会议被普遍视为 AI 研究的**"Agentic 转折点"**——Agent 类研究占比从 2024 年的 9.6% 跃升至 **19.2%**；长上下文相关论文超过 350 篇。本文按主题分类回顾 Oral 论文与最佳论文，提炼 2025 年的研究风向。

## 一、整体数据

### 投稿与录用

```text
NeurIPS 2025 数据
- 总投稿: 18,940 (YoY +6.2%)
- 录用论文: 4,795 (录用率 25.3%)
- Oral (≤0.6%):   104 篇
- Poster:        约 3,900 篇
- Workshop 论文: 约 800 篇
- 参会人数: 16,500+ (现场 + 虚拟)
- 赞助商: 87 家
```

### 关键词 Top 20

```python
keyword_ranking = [
    ("Large Language Model",  1810),
    ("Reinforcement Learning", 1320),
    ("Diffusion Model",       1015),
    ("Agent",                 980),    # 较 2024 年增长 95%
    ("Multimodal",            920),
    ("Long Context",          360),    # 新晋热点
    ("Mixture-of-Experts",    340),
    ("World Model",            260),    # 高速增长
    ("Self-Supervised",       240),
    ("Safety",                230),
    ("Alignment",             210),
    ("Retrieval-Augmented",   195),
    ("Reasoning",             188),
    ("Embodied AI",           175),
    ("Quantization",          160),
    ("State Space Model",     142),
    ("RLHF",                  138),
    ("Knowledge Distillation",132),
    ("Causal Inference",      118),
    ("Mamba",                 105),    # 持续下降
]
```

## 二、最佳论文

### Outstanding Paper Award（4 篇）

#### 1. *Mechanistic Interpretability of Long-Attention Matrices in Frontier Models*
**作者**：Anthropic Interpretability Team
**意义**：首次系统性地分析了 GPT-5 / Claude 4 等前沿模型注意力矩阵的"电路结构"，识别出 12 类功能回路（function circuits）。

```python
# 该论文提出的"动机归纳回路"可视化
class FunctionalCircuit:
    def __init__(self):
        self.components = {
            "induction_head": "A→B→A 模式识别",
            "entity_binding": "实体-属性绑定",
            "factual_recall": "事实性记忆激活",
            "negation_circuit": "否定逻辑门",
            "numerical_compare": "数值大小比较",
            # ... 7 种更多
        }

# 论文关键发现：12 类回路覆盖了 92% 的"事实性任务"行为
```

#### 2. *Constitutional Verification Power Savings via Sparse Routing*
**作者**：Stanford / Together AI
**核心贡献**：用 MoE 稀疏路由实现**推理能耗降低 4.2x**，同时保持 99.5% 任务性能。

#### 3. *Robust Agentic Generalization in Open-Ended Environments*
**作者**：DeepMind
**意义**：在开放环境（无明确奖励函数）中训练 Agent，证明**通用探索策略**可以迁移到 30+ 任务。

#### 4. *Diffusion Forcing: Training Diffusion Models with Causal Chains*
**作者**：MIT / NVIDIA
**突破**：把扩散模型扩展到**时序决策**——Agent 在视频/动作空间都用扩散过程生成。

### Test of Time Award

颁发给 2015 年的 *Deep Residual Learning*（ResNet）作者团队——这是 NeurIPS 第一次把 ToT 奖颁给深度学习基础架构论文（此前多颁给理论成果）。

## 三、热门研究方向

### 1. Agent 研究爆发

#### (a) Multi-Agent 协作

```python
# 代表论文：AutoGen 3.0 / DSPy-MultiAgent
class MultiAgentSystem:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.shared_memory = Blackboard()
        self.coordinator = Coordinator(agents)

    def solve(self, task):
        plan = self.coordinator.decompose(task)
        results = []
        for step in plan:
            agent = self.coordinator.assign(step)
            result = agent.act(step, context=self.shared_memory.read())
            self.shared_memory.write(step, result)
        return self.coordinator.synthesize(results)
```

**关键论文**：
- *Self-Organizing Multi-Agent Systems*（MIT）
- *Agent Communication Protocols beyond Natural Language*（DeepMind）

#### (c) Agent 评测基准

| 基准 | 任务数 | 评估维度 | Top 模型 |
|---|---|---|---|
| **GAIA-2026** | 466 | 多步推理 + 工具使用 | Claude 4.5 Opus |
| **SWE-bench-Pro** | 2,500+ | 真实 GitHub issue | Devin 2 |
| **OSWorld** | 360 | 操作系统任务 | GPT-5 |
| **WebArena-Pro** | 1,200 | 浏览器任务 | Gemini 3 |
| **τ-Bench** | 165 | 长流程商业任务 | Anthropic 内部 |

**Top Agent 性能对比**（2026-12 数据）：

```text
              GAIA   SWE-Pro  OSWorld  WebArena  τ-Bench
GPT-5         78%     42%      65%      62%      71%
Claude 4.5    82%     48%      71%      68%      78%
Gemini 3      75%     38%      62%      70%      62%
DeepSeek V4   65%     35%      48%      52%      55%
```

### 2. 长上下文成为新前沿

论文数突破 350 篇，热门方法：

#### (a) 注意力机制优化

```python
# 线性注意力（Linear Attention）
class LinearAttention(nn.Module):
    """O(n) 复杂度的注意力"""
    def forward(self, Q, K, V):
        # Q: (B, N, D), K: (B, N, D), V: (B, N, D)
        # 用 phi() 特征映射，避免显式 (N x N) 矩阵
        Q_phi = self.phi(Q)  # 特征映射
        K_phi = self.phi(K)

        # 关键：利用结合律 (Q K^T) V = Q (K^T V)
        # 先计算 K^T V：(D x N) @ (N x D) = (D x D)
        kv = K_phi.transpose(-1, -2) @ V  # (B, D, D)

        # 再计算 Q (K^T V)
        out = Q_phi @ kv  # (B, N, D)
        return out

# 复杂度：标准 Attention O(n^2) → Linear Attention O(n)
```

#### (b) 状态空间模型（Mamba / Mamba-2 / Mamba-3）

Mamba 系列论文依然占据重要位置，但增速放缓（Mamba 2 占论文 23% → 14%）：

```text
2024  23.0%   Mamba 主导
2025  14.0%   主流化但增速放缓
2026   8.5%   与 Transformer 混合架构更受欢迎
```

#### (c) 上下文缓存与复用

多篇论文研究**Context Cache**——长上下文场景下 KV cache 占用上百 GB：

```python
# CacheGen: 缓存压缩
class CacheGen:
    """压缩 KV cache 4~8x"""
    def __init__(self, cache, compression_ratio=4):
        self.encoder = Encoder()
        self.cache = cache
        self.compression_ratio = compression_ratio

    def compress(self):
        # 用学习型编码器压缩 KV cache
        compressed = self.encoder(self.cache)
        return compressed  # 4-8x 压缩

# 论文：CacheGen (MIT) - 把 100K 上下文的 cache 从 80GB 压到 12GB
```

### 3. 高效训练 / 推理

#### (a) 稀疏 MoE

```python
# DeepSeek-V4 风格的稀疏 MoE
class DeepSeekMoE(nn.Module):
    def __init__(self, hidden_dim, num_experts=256, top_k=8):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(hidden_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: (B, N, D)
        gate_logits = self.gate(x)  # (B, N, E)
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=-1)

        # 稀疏激活：仅 top_k 个专家
        for i, expert_idx in enumerate(top_k_indices[0]):
            expert = self.experts[expert_idx]
            # ... 路由计算
        return output
```

#### (b) 量化与低精度

- **INT4 训练** 取得突破（DeepSeek V4 验证）；
- **FP8 训练** 成为大厂标配（H100/B200 硬件支持）；
- **1-bit LLMs (BitNet)** 在小模型上达到 FP16 95% 性能。

### 4. AI 安全与对齐

#### (a) Constitutional AI 进化

Anthropic 的 Constitutional AI 路线持续推进：

```python
# Constitutional AI 训练循环
class ConstitutionalTraining:
    def __init__(self, principles):
        self.principles = principles  # 一组原则

    def self_critique(self, response, prompt):
        """模型自批评"""
        critiques = []
        for principle in self.principles:
            critique = self.model.generate(
                f"评估以下回答是否违反原则'{principle}': {response}"
            )
            critiques.append(critique)
        return critiques

    def revise(self, response, critiques):
        """基于批评修订"""
        return self.model.generate(
            f"基于以下批评修订回答: {critiques}\n原回答: {response}"
        )
```

#### (b) 可解释性

Mechanistic Interpretability（机械可解释性）成为独立子领域，本届会议单独设有 *Workshop on Mechanistic Interpretability*。

### 5. 多模态

#### (a) 视觉-语言模型

```text
2025 多模态代表：
- GPT-4V / GPT-5 Vision
- Claude 4.5 Sonnet (Vision)
- Gemini 3 Pro (Vision + Audio)
- Qwen3-VL (开源)
- InternVL 3 (开源)
```

#### (b) 全模态模型 (Unified Speech-Vision-Language)

代表工作：GPT-5o、AnyGPT、Unified-IO 3——单一 Transformer 同时处理文本/图像/音频/视频。

#### (c) 视频生成

- **Sora 2** (OpenAI)：支持 60 秒 1080p 视频生成。
- **Veo 3** (Google DeepMind)：4K 分辨率，最长 3 分钟。
- **可灵 AI 2.0** (快手)：开源视频生成模型。

### 6. AI4Science

#### (a) 生命科学

- **AlphaFold 4** (DeepMind)：预测蛋白-小分子复合物结构。
- **RoseTTAFold-3** (David Baker Lab)：开源对应版本。

#### (b) 材料与化学

- **GNoME-2** (DeepMind)：发现 **380 万** 种新稳定晶体结构（较 GNoME 1 翻 4 倍）。

#### (c) 气象与地球科学

- **WeatherNext 2** (Google DeepMind + ECMWF)：6 小时级中长期预报，超越传统 NWP 模型。

## 四、Workshop 亮点

### 1. Foundation Models for Decision Making
聚焦 LLM + 强化学习结合，**8 篇 oral + 35 篇 poster**。

### 2. AI for Scientific Discovery
- AlphaFold 系列作者做 keynote；
- LLM 在化学合成路径规划的应用。

### 3. Mechanistic Interpretability
- Anthropic 可解释性团队主讲；
- 电路发现（circuit discovery）算法成热点。

### 4. Embodied AI
- Figure 03 / Tesla Optimus Gen 3 等机器人展示；
- 真实家庭场景数据集（Bridge Data V3）。

### 5. AI Safety
- OpenAI / Anthropic / DeepMind 安全团队联办；
- 议题：scalable oversight、deception detection、shutdown problems。

## 五、产业参与

### 头部赞助商

| 公司 | 投入 | 重点方向 |
|---|---|---|
| **OpenAI** | Platinum | Foundation Models |
| **Anthropic** | Platinum | Safety + Agents |
| **Google DeepMind** | Platinum | Frontier Research |
| **Meta AI (FAIR)** | Gold | Open Source |
| **Microsoft Research** | Gold | Systems + AI4Science |
| **NVIDIA** | Gold | Compute + Systems |
| **Apple** | Silver | On-device AI |
| **DeepSeek** | Silver | Open Source |

### 工业界论文占比

```text
NeurIPS 2020  工业界一作 17%
NeurIPS 2023  工业界一作 38%
NeurIPS 2025  工业界一作 56%
```

**已超过学术界**——AI 研究的"工业化"已成定局。

## 六、对 2026 年的启示

### 1. Agent 仍是 2026 年核心赛道

预计 ICML 2026 / ICLR 2026 中 Agent 类研究占比将进一步上升至 **22–25%**。

### 2. 高效推理 / 论文变成"必答题"

不优化效率的论文很难被录用——审稿人会问 "FLOPs / Dollar / Latency"。

### 3. AI 安全常态化

Safety Track 论文数突破 500 篇，AI Alignment 不再是边缘方向。

### 4. AI4Science 持续扩张

预计 2026 年将出现独立的 **AI4Science Conference**。

### 5. 中国学者的舞台

2025 年 NeurIPS 录用论文**中国大陆第一作者占比 32%**（vs 2020 年 12%）：

```text
2020  12%   美国主导
2022  21%   中国快速追赶
2024  28%   中美双极
2025  32%   中国略领先（按录用数）
```

## 七、值得关注的具体论文清单

| 论文 | 类型 | 核心贡献 |
|---|---|---|
| *Mechanistic Interpretability of Long-Attention Matrices* | Oral | 识别 12 类功能回路 |
| *Constitutional Verification Power Savings* | Oral | MoE 路由实现 4x 节能 |
| *Diffusion Forcing* | Oral | 扩散过程到时序决策 |
| *AutoGen 3.0* | Workshop | 多 Agent 协作框架 |
| *CacheGen* | Spotlight | KV cache 压缩 8x |
| *BitNet-3a* | Spotlight | 1-bit LLM 实用化 |
| *GNoME-2* | Datasets | 380 万新晶体材料 |
| *α-Protein-4* | Datasets | 蛋白复合物结构 |

## 小结

NeurIPS 2025 是 AI 研究的**"工业化转折点"**——Agent 占比近 20%、工业界论文超 56%、中国学者占比 32%。展望 2026 年，**Agent / Efficiency / Safety / AI4Science** 将持续是四大主线。下一篇文章我们将聚焦 ICML 2026，剖析 ML 理论 + 高效训练的最新进展。