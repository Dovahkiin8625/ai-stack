# VITS 与端到端神经声码器

FastSpeech 2 把声学模型做到了实时，但它仍然是一个"两阶段系统"——先预测梅尔频谱，再交给独立训练的声码器（WaveNet、WaveRNN、HiFi-GAN）合成波形。两阶段系统的隐性问题在于：**梅尔频谱是有损的中间表示**，它丢弃了相位信息和短时细节，声码器必须"猜"出这些信息，这就限制了自然度上限。2021 年 Jaehyeon Kim 等人提出的 **VITS（Variational Inference with adversarial learning for end-to-end Text-to-Speech）** 把声学模型、声码器、时长预测全部塞进一个模型，并联合训练，把端到端 TTS 的自然度推到了新的 SOTA。本篇从神经声码器的演化出发，重点拆解 HiFi-GAN 与 VITS 的设计。

## 一、神经声码器的演化

### 1.1 WaveNet：自回归神经声码器

2016 年 Google DeepMind 的 **WaveNet**（van den Oord et al., 2016）首次证明：**用 dilated causal conv + softmax 直接预测 16-bit PCM 样本，可以生成接近真人的语音**。其条件分布建模为：

$$
p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_{1}, \dots, x_{t-1})
$$

通过指数级增长的 dilation（1, 2, 4, …, 512）让感受野覆盖整段语音，每层配合 gated activation：

$$
\mathbf{z} = \tanh(\mathbf{W}_{f} * \mathbf{x}) \odot \sigma(\mathbf{W}_{g} * \mathbf{x})
$$

WaveNet 开辟了神经声码器时代，但它**每秒钟要生成 16,000 次**自回归推理，RTF 远大于 1，无法实时。

### 1.2 WaveRNN：RNN + 稀疏化加速

2018 年的 **WaveRNN**（Kalchbrenner et al., 2018）把 WaveNet 的 dilated CNN 换成单层 GRU，并采用**双 softmax 技巧**——把 16-bit 样本拆成高 8 bit 和低 8 bit 各做一次 softmax，配合 weight pruning 在 CPU 上达到实时。但 RNN 自回归本质不变，训练不稳定、串行延迟仍是问题。

### 1.3 Parallel WaveGAN / NSF：并行非自回归声码器

2019 年的 **Parallel WaveGAN**（Yamamoto et al., 2019）用 **GAN + 多分辨率 STFT 损失** 把声码器变成了**完全并行**——一次性生成整段波形。NSF（Neural Source-Filter）则把源-滤波器的物理结构嵌入网络，用显式的 F0 + 谐波噪声建模把音质推向新高。但这些方法合成质量仍略逊于 WaveNet。

### 1.4 HiFi-GAN：多周期 / 多尺度判别器

2020 年 Jungil Kong 等人提出的 **HiFi-GAN**（Kong et al., 2020）一举成为新一代 TTS 的标配声码器。它的核心贡献是**两种针对音频周期结构的判别器**：

- **多周期判别器（Multi-Period Discriminator, MPD）**：用若干个原始周期 $p \in \{2, 3, 5, 7, 11\}$ 的 1D 卷积捕捉不同谐波周期的模式。
- **多尺度判别器（Multi-Scale Discriminator, MSD）**：在原始波形 + 不同上采样尺度的波形上各自跑一个判别器，捕捉**跨尺度的连续性**。

生成器是一个**多感受野融合**的卷积网络，每一层用转置卷积做上采样 + 残差块（带多尺度融合 MRF）：

```python
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """HiFi-GAN 生成器中的残差块：两条不同 kernel 的 dilated conv 残差。"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                      dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                      dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.tanh(c1(x)) * torch.sigmoid(c2(x))  # gated activation
            x = x + xt
        return x


class HiFiGANGenerator(nn.Module):
    """简化版 HiFi-GAN 生成器：4 个上采样块 + MRF。"""

    def __init__(self, in_channels: int = 80, upsample_rates=(8, 8, 2, 2)):
        super().__init__()
        self.upsample = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        ch = 512
        self.pre_conv = nn.Conv1d(in_channels, ch, kernel_size=7, padding=3)
        for i, r in enumerate(upsample_rates):
            cur_ch = ch // (2 ** (i + 1))
            # 转置卷积上采样
            self.upsample.append(nn.ConvTranspose1d(
                ch if i == 0 else cur_ch * 2, cur_ch,
                kernel_size=r * 2, stride=r, padding=r // 2,
            ))
            # 每个上采样后挂 3 个不同 kernel / dilation 的残差块
            self.resblocks.append(nn.ModuleList([
                ResBlock1D(cur_ch, k, d) for k, d in
                [(3, (1, 3)), (7, (1, 3)), (11, (1, 3))]
            ]))
        self.post_conv = nn.Conv1d(cur_ch, 1, kernel_size=7, padding=3)

    def forward(self, mel):
        # mel: (B, n_mels, T_mel)  T_mel 帧梅尔
        x = self.pre_conv(mel)
        for up, resblk in zip(self.upsample, self.resblocks):
            x = up(x)
            outs = [r(x) for r in resblk]
            x = sum(outs) / len(outs)
        return torch.tanh(self.post_conv(x))
```

生成器损失除了对抗损失，还包含 **Mel-Spectrogram Loss（梅尔重建损失）**——把生成波形重新算成梅尔频谱和 ground-truth 比 L1：

$$
\mathcal{L}_{\text{mel}}(G) = \mathbb{E}_{x, z} \Big[ \| \phi(x) - \phi(G(z)) \|_1 \Big]
$$

其中 $\phi$ 是预训练的特征提取器（一般是同一组 STFT + 梅尔滤波器组）。Mel loss 把"听起来像不像"的判别问题转成"梅尔是否一致"的可微信号，是 HiFi-GAN 能在 1-2 天训练出 SOTA 音质的关键 trick。

## 二、VITS：把声学模型与声码器合一

VITS 的核心思想是：**声学模型预测的隐变量本身就是声码器的输入，何必再绕一道梅尔频谱？** 它把 TTS 拆成三件可联合训练的事：

1. **后验编码器（Posterior Encoder）**：从真实波形提取 latent $\mathbf{z}$。
2. **解码器（Decoder）**：从 latent $\mathbf{z}$ 重建波形（替代声码器）。
3. **条件先验（Conditional Prior）**：从文本 token 预测 latent 的分布 $\mathcal{N}(\boldsymbol{\mu}_\theta(\text{text}), \boldsymbol{\sigma}_\theta(\text{text}))$。

### 2.1 变分下界（ELBO）

设 $\mathbf{x}$ 是真实波形，$\mathbf{c}$ 是文本条件，$\mathbf{z}$ 是隐变量。VITS 优化**变分下界**：

$$
\log p(\mathbf{x} \mid \mathbf{c}) \ge \underbrace{\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x} \mid \mathbf{z})]}_{\text{重构（声码器重建）}}
\underbrace{- D_{\text{KL}}\big( q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p_\theta(\mathbf{z} \mid \mathbf{c}) \big)}_{\text{KL 对齐}}
$$

但 VITS 直接优化这个 ELBO 不太稳，作者额外加了**对抗训练**——用判别器 $D$ 把 ELBO 中难以优化的重构项替换为对抗损失，最终损失是：

$$
\mathcal{L}_{\text{VITS}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{adv}}(G, D)
$$

其中 $\mathcal{L}_{\text{adv}}$ 和 HiFi-GAN 一样用 LSGAN / Hinge Loss 形式。

### 2.2 标准化流（Normalizing Flow）

直接让 $p_\theta(\mathbf{z} \mid \mathbf{c})$ 和 $q_\phi(\mathbf{z} \mid \mathbf{x})$ 对齐非常困难，因为真实语音的 latent 分布远比标准高斯复杂。VITS 在文本条件分支后挂了 **4 层 Affine Coupling Flow** 把简单高斯分布变换为复杂分布：

$$
\mathbf{z}_k = \mathbf{z}_{k-1} \odot \exp(\mathbf{s}_k(\mathbf{z}_{k-1})) + \mathbf{t}_k(\mathbf{z}_{k-1})
$$

每个 flow block 学习一个可逆变换，理论上可以拟合任意复杂分布。Flow 让 KL 项能更紧地约束生成过程，从而显著提升合成稳定性。

### 2.3 随机时长预测器（Stochastic Duration Predictor）

FastSpeech 系列用 MSE 监督时长，本质是"回归一个点估计"。VITS 把时长也建模为**随机变量**——给定文本特征，预测时长的高斯分布 $\mathcal{N}(\mu_\theta, \sigma_\theta)$，训练时通过重参数化采样。这带来两个直接好处：

- **韵律多样性**：推理时可以从同一段文本采出不同长度的合成语音，天然支持多模态韵律。
- **训练稳定性**：相比 MSE 回归，连续分布上的 KL 优化对异常值更鲁棒。

### 2.4 解码器：HiFi-GAN 风格的 GAN 声码器

VITS 的 Decoder 直接沿用 HiFi-GAN 生成器结构，但输入从梅尔频谱换成了 latent $\mathbf{z}$。配合 MPD / MSD 判别器，整套模型从 latent → waveform 的映射**完全在模型内部、联合训练**，跳过了独立的声码器阶段。

## 三、VITS 2：对抗时长、说话人条件、风格迁移

2022 年提出的 **VITS 2** 在 VITS 基础上做了若干工程改进：

- **对抗时长预测器（Adversarial Duration Predictor, ADP）**：用判别器区分"模型预测的时长分布"与"真实的时长分布"，让训练目标与人类标注分布更对齐。
- **说话人条件归一化（Speaker-Conditional Normalization）**：把说话人 embedding 通过 FiLM（Feature-wise Linear Modulation）注入解码器的每一层，实现 zero-shot 声音克隆。
- **文本线性变换**：在 encoder 后加一层线性投影，提升对长句和罕见词的鲁棒性。
- **风格迁移**：在 latent 中解耦出"内容"和"风格"两个维度，可以把说话人 A 的内容用说话人 B 的风格说出来。

VITS 2 在 LJSpeech 上达到了 4.57 MOS（与真人差距 < 0.1），是当时最接近真人的单说话人端到端 TTS。

## 四、端到端 vs 两阶段的范式之争

VITS 和 HiFi-GAN 代表了 TTS 的两种主流范式：

| 维度 | 端到端（VITS / VITS 2） | 两阶段（FastSpeech 2 + HiFi-GAN） |
| --- | --- | --- |
| 训练稳定性 | 较难（联合优化声码器 + 声学模型） | 易（两个模型分别训练） |
| 自然度 | 略高（绕过梅尔损失） | 接近天花板 |
| 推理延迟 | 单次前向，更低 | 两段前向 |
| 可控性 | 弱（latent 难解释） | 强（梅尔 + F0 / energy 可调） |
| 多说话人扩展 | 容易（FiLM 注入） | 需要重训声码器 |
| 部署成本 | 单一模型 | 两套模型、需同步 |
| 工程友好度 | 调参困难 | 成熟开源方案多 |

工业界目前的实践是**两阶段为主**（FastSpeech 2 + HiFi-GAN 更稳定、可控、便于做产品化定制），但端到端模型（VITS 系列）在音质上仍占优势，且随着对抗训练技术成熟，**端到端正在成为新的 SOTA 路线**。

## 五、VITS 在低资源 / 跨语言场景下的应用

VITS 的端到端结构让它在以下场景特别受欢迎：

- **小数据 TTS**：相比 FastSpeech 2 需要 5-10 小时数据，VITS 在 1-2 小时数据上就能合成出可懂的语音。
- **跨语言合成**：解耦的 latent 让 VITS 可以把一种语言的韵律迁移到另一种语言。
- **实时语音克隆**：配合说话人编码器（如 Resemblyzer、d-vector），VITS 可以实现 10 秒音频完成声音克隆。

## 小结

| 模块 | 关键创新 | 作用 |
| --- | --- | --- |
| HiFi-GAN | MPD + MSD + Mel Loss | 高质量并行声码器 |
| VITS 后验编码器 | latent 表征语音 | 替代梅尔频谱 |
| VITS 条件先验 | 文本→latent 分布 | 端到端生成 |
| VITS 标准化流 | Affine Coupling Flow | 拟合复杂先验分布 |
| VITS 随机时长 | 分布采样而非点估计 | 韵律多样、稳定训练 |
| VITS 2 ADP | 对抗时长预测器 | 对齐人类标注分布 |
| VITS 2 说话人条件 | FiLM 注入 | 多说话人 / 克隆 |

VITS 系列把 TTS 的"声学模型 + 声码器"彻底打通，但隐变量空间的不可解释性仍限制了细粒度韵律控制。下一篇我们将看到：当 LLM 把"文本"和"语音 token"统一到同一个序列建模框架时，TTS 的范式会发生怎样的又一次剧变——从 VALL-E 把 TTS 当作条件语言模型，到 SpeechGPT / Qwen2-Audio 实现语音与文本的一体化建模。
