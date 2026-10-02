# 传统 TTS：拼接合成到参数合成

在深度学习彻底改变语音合成（Text-to-Speech, TTS）之前，TTS 系统经历了两个清晰的工程阶段：**拼接合成（Concatenative Synthesis）** 和**统计参数语音合成（Statistical Parametric Speech Synthesis, SPSS）**。前者把预先录制的大量语音片段拼接起来，自然度极高但灵活性差；后者用统计模型把语言学参数映射到声学参数，再用声码器（vocoder）合成波形，虽然自然度略逊但参数可控、便于个性化。本篇从 TTS 的三段式流水线出发，沿着"前端 → 声学模型 → 声码器"的脉络，梳理 1990 年代到 2010 年代中期这一长达二十多年的技术演化。

## 一、TTS 的三段式流水线

无论是传统还是现代 TTS，一个完整的语音合成系统都可以拆成三个模块：

1. **文本分析 / 前端（Text Analysis / Front-End）**：把原始文本转为带音素、重音、停顿、韵律边界的语言学表示（**字位（grapheme）→ 音素（phoneme）→ 韵律（prosody）**）。
2. **声学模型（Acoustic Model）**：从语言学特征预测声学参数。最早期是查表 + 规则，随后是 HMM，2010 年代后是 DNN。
3. **声码器（Vocoder）**：把声学参数（基频 F0、频谱包络、非周期成分）转换为最终的时域波形。

$$
\mathbf{y} = \text{Vocoder}\Big( \text{AcousticModel}(\text{Frontend}(\mathbf{x})) \Big)
$$

其中 $\mathbf{x}$ 是输入文本，$\mathbf{y}$ 是采样率通常为 16 kHz 或 22.05 kHz 的 PCM 波形。前端决定了"系统能不能正确发音"，声学模型决定了"听起来像不像目标说话人"，声码器决定了"信号听起来自不自然、是否有杂音"——三者任何一个薄弱，整体自然度都会受影响。

## 二、拼接合成（Concatenative Synthesis）

拼接合成是 1990 年代商用 TTS（如 AT&T Bell Labs 的 `Natural TTS`、DECtalk、后来的 Nuance Vocalizer）的主流方案。它的核心思想非常简单：**录一个非常大的语音数据库，把句子切成音素或半音素（diphone）单元，运行时挑出最合适的单元按目标韵律拼接**。理论上这种"重放式"合成能保留真人发音的微观细节，自然度上限极高。

### 2.1 单元选择（Unit Selection）

拼接系统的关键是**单元库（unit inventory）** 和**目标代价（target cost）/ 连接代价（join cost）** 的联合搜索。每个候选单元（通常是 diphone，比如 `#a-t#`，即 a→t 的过渡）都有一组上下文标签：前一个音素、后一个音素、重音位置、所在音节、词性、在韵律词中的位置等。给定一个目标韵律描述，系统要做两件事：

- **目标代价 $C^t(u, t)$**：候选单元 $u$ 与目标位置 $t$ 在上下文标签上的差异，例如期望的 F0 是 120 Hz、实际是 90 Hz，差 30 Hz 就给一个惩罚。
- **连接代价 $C^j(u_i, u_{i+1})$**：相邻两个候选单元在波形层面的不连续性，例如相位不连续、频谱跳变都会给惩罚。

最终代价是两者加权和，搜索问题是找到一条从 $u_1$ 到 $u_N$ 的路径，使总代价最小：

$$
\hat{\mathbf{u}} = \arg\min_{\mathbf{u}} \sum_{i=1}^{N} C^t(u_i, t_i) + \sum_{i=1}^{N-1} C^j(u_i, u_{i+1})
$$

### 2.2 Viterbi 搜索

这是一个典型的**最优路径搜索**，状态空间是 $|\mathcal{U}|^N$（$\mathcal{U}$ 是单元库大小，$N$ 是目标句子音素数），用动态规划可在 $O(N \cdot |\mathcal{U}|^2)$ 内找到全局最优，这就是 Viterbi 算法：

```python
import numpy as np


def viterbi_unit_selection(target_costs, join_costs):
    """
    target_costs: (N, U)  N 个目标位置 × U 个候选单元
    join_costs:   (U, U)  任意两个单元的连接代价
    """
    N, U = target_costs.shape
    # dp[i, u] = 到第 i 个位置选单元 u 的最小累积代价
    dp = np.full((N, U), np.inf)
    bp = np.full((N, U), -1, dtype=np.int64)  # backpointer

    dp[0] = target_costs[0]
    for i in range(1, N):
        # 对每个候选 u，dp[i-1] + join_costs[:, u] + target_costs[i, u] 取最小
        cost = dp[i - 1][:, None] + join_costs + target_costs[i][None, :]
        dp[i] = cost.min(axis=0)
        bp[i] = cost.argmin(axis=0)

    # 回溯最优路径
    path = np.zeros(N, dtype=np.int64)
    path[-1] = dp[-1].argmin()
    for i in range(N - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    return path
```

### 2.3 波形拼接与遗留问题

选好单元后，把对应的 PCM 切片按时间顺序用 PSOLA（Pitch-Synchronous Overlap-Add）算法或简单的 cross-fade 拼接，再做一次时长规整（time-warping）以匹配目标韵律。

拼接合成的优势是**自然度极高**（因为波形本身来自真人），但缺点非常致命：

- **数据量爆炸**：商用系统动辄 10-50 小时的单元库，录制、标注成本极高。
- **韵律僵硬**：合成句子的韵律必须能从单元库中"拼出来"，否则会出现不自然的跳变。
- **个性化难**：换一个说话人就要重录全部单元库。
- **拼接伪影**：单元交界处的相位、频谱不连续会被人耳敏锐捕捉。

正因为这些缺点，研究界在 2000 年代中期开始向**参数合成（Parametric Synthesis）** 转型。

## 三、参数合成与声码器特征

参数合成的核心思想是：先把语音分解成一组**可控的声学参数**，再用统计模型从文本预测这些参数，最后用声码器从参数重建波形。优势是模型可训练、参数空间小、可灵活控制。

### 3.1 STRAIGHT 与源-滤波器模型

语音的产生可以近似为**源-滤波器模型（Source-Filter Model）**：声带振动产生激励（source），经过声道（vocal tract）滤波得到最终频谱。1997 年 Kawahara 等人提出的 **STRAIGHT（Speech Transformation and Representation using Adaptive Interpolation of weiGHTed spectrogram）** 是这一时期的代表性工具，它通过对周期图做三次平滑来抹掉周期性的谐波结构，得到一个"干净的"频谱包络：

$$
|X(\omega, t)|_{\text{STRAIGHT}} = \sqrt{ \frac{\sum_\tau w(\tau - t) \cdot |X(\omega, \tau)|^2}{\sum_\tau w(\tau - t)} }
$$

其中 $w(\tau - t)$ 是以 $t$ 为中心、对周期图做 F0 自适应平滑的窗函数。STRAIGHT 把语音分解为三个独立可操纵的参数：

| 参数 | 物理含义 | 维度 / 频率 |
| --- | --- | --- |
| $F_0(t)$ | 基频 | 1D，曲线 |
| $\log |X(\omega, t)|$ | 频谱包络 | 高维（FFT bin × 帧） |
| $A_p(\omega, t)$ | 非周期成分（aperiodicity） | 高维 |

2016 年提出的 **WORLD（Yoshimura et al., 2016）** 是 STRAIGHT 的开源改进版，把提取过程拆成 DIO / CheapTrick / PLATINUM 三个独立模块，速度比 STRAIGHT 快几十倍，至今仍是 ESPnet 等开源 TTS 工具的默认声码器。

### 3.2 为什么参数化表示重要

参数化表示让 TTS 可以做很多拼接合成做不到的事：

- **时长控制**：直接对 $F_0$ 和时长做仿射变换，可以"加速 1.2 倍"或"把基频抬高 10 Hz"。
- **风格迁移**：把源说话人的频谱参数换成目标说话人的，就完成了声音转换（voice conversion）。
- **数据效率**：相比拼接合成需要 50 小时录音，参数合成的声学模型通常 5-10 小时数据就能训练出可用的合成质量。

代价是：**频谱包络的细节在参数化过程中会丢失**，合成波形的自然度上限被声码器锁死。早期声码器（如简单 inverse FFT + 激励脉冲）合成的声音听起来"嗡嗡的"，像隔着一层纱——这也是后来神经声码器（WaveNet、HiFi-GAN）的核心改进方向。

## 四、统计参数语音合成（SPSS）的 HMM 时代

### 4.1 HMM 声学模型

2000 年代初，Tokuda、Zen 等人推动的 **HMM-based Speech Synthesis System（HTS）** 成为 SPSS 的事实标准。HTS 把语音的每个状态（state）建模成 HMM，对 F0 用多空间分布（MSD）建模以处理清音 / 浊音切换，对频谱和时长用连续 HMM。文本到语音的合成过程是：

1. **训练**：从语音库提取 F0 / 频谱 / 非周期成分 + 对应的音素 / 韵律标签，用 Baum-Welch 训练 HMM，再用 Embedded Training 让所有音素共享一个状态空间。
2. **合成（参数生成）**：从文本上下文得到一条状态序列后，用 HMM 的均值 / 方差 + 全局方差（Global Variance, GV）做最大似然参数生成（MLPG），得到平滑的 F0 / 频谱轨迹。

参数生成的核心是最小化下式：

$$
\mathcal{L}(\mathbf{c}, \mathbf{o}) = \sum_{t=1}^{T} \Big( -\log \mathcal{N}(\mathbf{o}_t \mid \boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t) \Big) + w_{\text{GV}} \sum_{d=1}^{D} \frac{(\sigma_d - \bar{\sigma}_d)^2}{2 \bar{\sigma}_d^2}
$$

其中 $\mathbf{c}$ 是状态序列，$\mathbf{o}_t$ 是第 $t$ 帧参数，$\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t$ 是 HMM 的均值与协方差。GV 项强制预测参数的方差与训练集的全局方差接近，避免 MLPG 把频谱"压平"成过于平滑的曲线——这是 HTS 系统的关键 trick。

### 4.2 DNN 替换 HMM

2013 年前后，**深度神经网络（DNN）** 开始替换 HTS 里的 HMM，把声学模型从生成式改为判别式：用 DNN 直接从语言学特征预测 F0 / 频谱 / 非周期成分的均值与方差，再交给 MLPG + 声码器重建波形。这套框架后来被 Zen 等人整理成 **Merlin 工具包**（2015 年前后），成为 SPSS 的开源标杆。

```python
import torch
import torch.nn as nn


class DNNAcousticModel(nn.Module):
    """
    Merlin 风格的浅层 DNN 声学模型：
    输入 linguistic features (二进制 + 数值)，
    输出 F0 (1D) + 频谱 (BAP/MGC, 60D) + 非周期 (5D)。
    """

    def __init__(self, in_dim: int, lf0_dim: int = 1, mgc_dim: int = 60, bap_dim: int = 5):
        super().__init__()
        out_dim = lf0_dim + mgc_dim + bap_dim
        # Merlin 默认 6 层全连接 + sigmoid 激活 + 线性输出层
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x):
        # x: (B, T, in_dim)  时间展开的 linguistic features
        return self.net(x)   # (B, T, out_dim)
```

DNN 模型相比 HMM 的提升来自两点：（1）DNN 可以学到高度非线性的特征交互，韵律预测的准确率显著提高；（2）DNN 输出天然是确定性映射，不需要 HMM 那套 EM 训练。但 DNN-MLP 仍然有"过平滑（over-smoothing）"问题——频谱被预测得太"平均"，合成声音闷闷的。

## 五、优缺点对比与历史定位

| 维度 | 拼接合成 | HMM/DNN 参数合成 |
| --- | --- | --- |
| 数据量 | 10-50 小时单元库 | 5-10 小时标注语音 |
| 自然度上限 | 接近真人 | 受声码器限制，较低 |
| 韵律可控性 | 差（数据驱动） | 强（参数可调） |
| 个性化成本 | 重新录制全套 | 重新训练模型 |
| 训练难度 | 无（查表） | 中（EM / SGD） |
| 推理速度 | 快（波形直读） | 受声码器影响 |
| 声音"机械感" | 拼接处偶发卡顿 | 全局偏闷 |
| 代表工具 | AT&T Natural Voices、DECtalk | HTS、Merlin |

拼接合成在自然度上的优势让它在 2000 年代前的车载、客服等场景占据主流，但它不可扩展的本质使它在 2010 年代被 SPSS 取代。而 SPSS 又被 2017 年后的神经 TTS（Tacotron 2、FastSpeech）取代——下一篇我们将看到神经 TTS 如何把"自然度"和"可控性"第一次同时做到。

## 小结

| 阶段 | 核心思想 | 代表工具 | 主要问题 |
| --- | --- | --- | --- |
| 拼接合成 | 大库单元 + Viterbi 选最优路径 | AT&T Natural Voices | 数据量、可控性差 |
| STRAIGHT/WORLD | 源-滤波器 + 参数化表示 | WORLD vocoder | 参数化损失细节 |
| HTS / SPSS | HMM 声学模型 + MLPG 参数生成 | HTS、Merlin | 过平滑、自然度瓶颈 |
| DNN-SPSS | DNN 替换 HMM | Merlin | 仍受声码器天花板 |

传统 TTS 的三段式流水线（前端 / 声学模型 / 声码器）在 2017 年被 Tacotron 2 首次打通为端到端架构，而声码器也从 STRAIGHT / WORLD 进化到了 WaveNet / HiFi-GAN。下一篇将详细讨论神经 TTS 的两大里程碑：自回归的 Tacotron 2 与非自回归的 FastSpeech 系列。
