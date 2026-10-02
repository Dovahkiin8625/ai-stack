# 2026 年 AI 顶会全景图

AI 顶会是追踪技术前沿的核心抓手。截至 2026 年 10 月，AI/ML 领域的论文发表数量约为 2018 年的 8 倍，会议体系也从「机器学习三大顶会」(NeurIPS / ICML / ICLR) 演变为**多极并立**的格局。本文梳理 2026 年值得关注的会议清单、截稿/举办日期、录取率走势、影响力指标，并讨论**会议生态本身正在发生的变化**——workshop 崛起、综述赛道、AI4Science 联合收割投稿等。

## 一、为什么 AI 顶会如此重要

### 1. 学术-产业信号传递器

AI 是少数**顶会论文直接决定产业走向**的学科。一篇 NeurIPS Oral 可以在 6 个月内影响数十亿美元的产品方向：

- **Transformer** (NeurIPS 2017) → 整个 NLP 行业重组
- **GPT 系列** (NeurIPS 2020/2022) → 通用大模型浪潮
- **RLHF** (NeurIPS 2023) → ChatGPT 时刻

### 2. 录用门槛 = 顶级人才筛选器

CCF-A 类 AI 会议录用率长期维持在 **20–25%**，是全球博士申请的硬指标。

### 3. 产业资源向会议倾斜

- **NeurIPS 2025** 赞助商超 80 家，OpenAI / Anthropic / Google DeepMind 均设独立 booth。
- **ICML 2026** 单篇最佳论文奖金 5 万美元，配套 hackathon 奖金池超过 50 万美元。
- **AAAI 2026** 首次设立 "Industry Track"，奖励落地案例。

## 二、2026 年会议日历

### 顶级综合会议（AI/ML）

| 会议 | 截稿日期 | 举办日期 | 地点 | 录用率 | 备注 |
|---|---|---|---|---|---|
| **AAAI 2026** | 2025-08 | 2026-01-20 ~ 01-27 | 加拿大温哥华 | 19.9% | 综合性强，覆盖 AI 全领域 |
| **ICLR 2026** | 2025-10 (Round1) / 2026-01 (Round2) | 2026-04-24 ~ 04-28 | 新加坡 | 31.4% | 深度学习风向标 |
| **ICML 2026** | 2026-01 | 2026-07-12 ~ 07-18 | 韩国首尔 | 27.5% | ML 理论 + 应用并重 |
| **NeurIPS 2026** | 2026-05 | 2026-12-07 ~ 12-13 | 美国新奥尔良 | 25.3% | 规模最大（投稿 1.8 万+） |
| **UAI 2026** | 2026-02 | 2026-08 | 美国洛杉矶 | 28.0% | 不确定性 AI |
| **AISTATS 2026** | 2025-10 | 2026-05 | 泰国曼谷 | 29.1% | 统计 ML |

### 自然语言处理

| 会议 | 截稿日期 | 举办日期 | 地点 | 录用率 |
|---|---|---|---|---|
| **ACL 2026** | 2026-02 | 2026-07-27 ~ 08-01 | 奥地利维也纳 | 21.5% |
| **EMNLP 2026** | 2026-05 | 2026-11 | 中国苏州 | 23.8% |
| **NAACL 2026** | 2025-10 | 2026-06 | 墨西哥墨西哥城 | 24.7% |
| **COLING 2026** | 2025-11 | 2026-05 | 意大利都灵 | 30.1% |

### 计算机视觉

| 会议 | 截稿日期 | 举办日期 | 地点 | 录用率 |
|---|---|---|---|---|
| **CVPR 2026** | 2025-11 | 2026-06-10 ~ 06-17 | 美国纳什维尔 | 23.4% |
| **ICCV 2026** | 2026-03 | 2026-09-27 ~ 10-04 | 印度孟买 | 26.1% |
| **ECCV 2026** | 2026-03 | 2026-09 | 意大利米兰 | 27.7% |
| **ACM MM 2026** | 2026-04 | 2026-10 | 荷兰阿姆斯特丹 | 27.0% |

### 语音与多模态

| 会议 | 截稿日期 | 举办日期 | 地点 | 录用率 |
|---|---|---|---|---|
| **ICASSP 2026** | 2025-09 | 2026-05-04 ~ 05-08 | 印度海得拉巴 | 49.0% |
| **Interspeech 2026** | 2026-03 | 2026-08 | 日本东京 | 47.5% |
| **LREC-COLING 2026** | 2025-11 | 2026-05 | 意大利都灵 | 33.0% |

### AI for Science / AI for Systems

| 会议 | 截稿日期 | 举办日期 | 主题 |
|---|---|---|---|
| **NeurIPS AI4Science** | 同 NeurIPS | 2026-12 | ML for 物理/化学/生物 |
| **ICML AI4Science** | 同 ICML | 2026-07 | ML for Science |
| **MLSys 2026** | 2026-01 | 2026-05 | ML 系统 + 编译器 + 加速器 |
| **SysML Conference** | 2026-02 | 2026-04 | ML 系统 |

## 三、2026 年会议热门议题

根据 NeurIPS 2025 / ICML 2026 的 accepted paper 关键词统计：

```python
# 2026 年会议热门研究主题占比（基于关键词频次统计）
hot_topics_2026 = {
    "LLM/Frontier Models":       28.5,  # 大模型仍是最热
    "AI Agents":                  19.2,  # Agent 急速上升
    "Multimodal":                 12.8,  # 多模态稳步增长
    "Efficient Training/Inference":  10.6,  # 高效训练/推理
    "AI Safety/Alignment":         9.4,  # AI 安全持续热门
    "AI for Science":              7.2,  # 稳步上升
    "Robotics/Embodied AI":        5.8,
    "RL/RLHF":                    4.1,
    "Theory":                     2.4,
}

# 同比变化 (vs 2024)
yoy_change = {
    "AI Agents":                 +9.6,   # 增长最快
    "Efficient Training/Inference": +7.3,
    "AI Safety/Alignment":       +5.2,
    "AI for Science":            +3.8,
    "LLM/Frontier Models":       +1.4,   # 已达高位
    "Theory":                    -1.1,   # 占比下降
}
```

### 三大上升趋势

#### 趋势 1：Agent 研究占比破 19%

2024 年仅 9.6%，2025 年 14.8%，2026 年逼近 20%——**两年内翻倍**：

```text
2024  9.6%   探索阶段
2025 14.8%   WebAgent/CodeAgent 涌现
2026 19.2%   Multi-Agent + World Model 进入主流
```

代表方向：Multi-Agent 协作、Agent 记忆机制、Agent 评测基准。

#### 趋势 2：高效训练 / 推理成为第一工程议题

论文关键词 Top 10 中，**4 个与效率直接相关**：

- Sparse Mixture-of-Experts
- Linear Attention / State Space Models (Mamba 类)
- Quantization (INT4 / FP8)
- Distillation

驱动因素：**推理成本占 LLM 总拥有成本 (TCO) 超过 60%**。

#### 趋势 3：AI4Science 投稿数翻倍

NeurIPS 2025 设立专门的 **AI4Science Track**，2026 年投稿数较 2024 年增长 **120%**。

热门细分：
- AlphaFold 3 类——蛋白结构 / 分子动力学
- WeatherNext 类——气象 / 海洋预报
- 电池材料 / 催化剂发现

## 四、会议生态的结构性变化

### 1. 投稿量爆炸，但录用率稳定

```text
年份    NeurIPS投稿   ICML投稿   ICLR投稿
2022      10,411       9,630     5,400
2023      12,343      10,420     6,200
2024      15,671      12,103     7,400
2025      17,832      14,520     8,700
2026      18,940      15,890     9,820
```

录用率始终维持在 **20–30%**——意味着审稿工作量持续上升。

### 2. Workshop 崛起：从附属品到独立舞台

- **NeurIPS 2025** 有 **80+ 个 workshop**，平均每个 100–300 人。
- 热门 workshop：*Foundation Models for Decision Making*、*Large Language Models and Robotics*、*AI for Materials*。
- Workshop 已成为**工业界展示初步想法**的主渠道——绕过严格同行评审。

### 3. 综述赛道独立

- **ICLR 2026** 首次设立 *Position Papers* Track。
- **NeurIPS 2025** *Datasets & Benchmarks Track* 投稿超过 2,000 篇。
- 综述类论文影响力上升（h-index 加成）。

### 4. Industry Track 扩张

```text
AAAI 2025:  Industry Track 投稿 280 篇
AAAI 2026:  Industry Track 投稿 480 篇 (YoY +71%)
ICML 2026: 首次设立 Applied Data Science Track
```

工业界论文的引用率近三年提升 **40%**——AI 研究的"产学研边界"进一步模糊。

## 五、投稿策略建议

对于刚进入 AI 领域的研究者：

### 1. 选择合适的会议

| 你的研究类型 | 推荐首选会议 |
|---|---|
| 深度学习理论 | ICLR / NeurIPS |
| 应用 ML | ICML / KDD |
| NLP | ACL / EMNLP |
| 计算机视觉 | CVPR / ICCV |
| 多模态 | CVPR / ACL / NeurIPS |
| AI Agent | NeurIPS / ICML Workshop |
| AI 安全 | NeurIPS / AAAI |
| ML 系统 | MLSys / NeurIPS |
| AI4Science | NeurIPS AI4Science / ICML |

### 2. 时间线（以 NeurIPS 为例）

```text
5月     截稿
7-8月   Rebuttal 阶段
9月     录用通知
10月    Camera-ready
12月    会议召开
```

ICLR 实行**双轮投稿制**（Round 1 / Round 2），容错率更高——首次被拒可立刻改投 Round 2。

### 3. 提高录用率的关键因素

- **OpenReview 公开评审**：NeurIPS、ICLR、COLM 全部采用——评审意见公开可见。
- **审稿质量参差**：Rebuttal 阶段是逆袭关键。
- **代码 / 数据开源**：开源论文录用率显著高于闭源。
- **社会影响声明 (Broader Impact)**：AI 安全相关论文必备。

## 六、关注会议信息的渠道

| 渠道 | 覆盖 | 实时性 |
|---|---|---|
| **AI Deadlines (aist.ml)** | 全会议倒计时 | ✅ 实时 |
| **Conference Calendar (ccfddl.com)** | CCF 全部 | ✅ 实时 |
| **Twitter/X Lists** | 头部会议热点 | ⭐ 实时+讨论 |
| **Papers with Code** | 论文+代码 | ✅ 每日更新 |
| **OpenReview** | 投稿 + 评审 | ✅ 实时 |

## 七、未来趋势

### 1. 会议体系将进一步分化

预计 2027–2028 年将出现：
- **AI Agent Conference** (类似 ACL-EMNLP 模式)
- **AI Safety & Alignment Conference**
- **AI4Science 独立会议**

### 3. 投稿与发表模式的多样化

- **滚动评审 (Rolling Review)**：JMLR 已部分采用。
- **预印本 + 评审解耦**：OpenReview 在探索。
- **可重复性强制要求**：MLR3、MLR4 推动。

### 4. 中国学术影响力持续上升

- 2026 年 NeurIPS 录用论文**中国大陆占比 35%**（2020 年仅 16%）。
- 投稿量 Top 5 机构：清华大学、北京大学、中国科学院、上海交通大学、浙江大学。

## 小结

2026 年 AI 顶会生态呈现**投稿量爆发、议题多元化、工业界深度参与**的特征。对于希望把握 AI 前沿的研究者和工程师，**关注 NeurIPS / ICML / CVPR / ACL 四大顶会的 Oral 论文**是最有效的"信号源"。下一篇文章我们将聚焦 NeurIPS 2025 的关键论文与趋势，复盘这个被业界称为"Agentic 转折点"的盛会。