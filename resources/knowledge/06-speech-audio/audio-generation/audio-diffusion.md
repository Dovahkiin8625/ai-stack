# 音频扩散模型：从 WaveGrad 到 AudioLDM

把一段 24 kHz 的语音展开成序列，每秒就有 24000 个采样点，相当于把一个 1024 token 的句子放大 25 倍。**音频生成**因此成为生成式模型最"苛刻"的试炼场：模型要在毫秒级的波形细节和分钟级的语义结构之间同时把控质量、一致性和速度。从 2016 年自回归的 **WaveNet**（van den Oord et al., 2016）到 2023 年的 **AudioLDM 2**（Liu et al., 2023），扩散模型（Diffusion Model）已成为这一领域的事实主流。本篇从任务定义出发，沿着"自回归慢速 → GAN → 频域扩散 → 潜在扩散"的工程演化路径，剖析音频扩散的核心思想与代表工作。

## 一、音频生成任务的四大子问题

在动手选模型之前，必须先澄清"要生成什么"。音频生成不是一个任务，而是一族任务，对波形建模的难度差异极大：

1. **语音增强（Speech Enhancement）**：给定带噪语音 $y = x + n$，恢复干净语音 $x$。输入输出都是 16 kHz 波形，本质是回归问题。
2. **文本转语音（TTS）**：把文字 $T$ 映射到语音 $x$，长度由文本决定。代表性工作有 Tacotron 2（Shen et al., 2018）、FastSpeech 2（Ren et al., 2020）。
3. **声音事件生成（Sound Event / Foley）**：给定类别标签或文本描述，生成对应声音，如"狗叫"、"雨声"、"汽车鸣笛"。
4. **音乐生成（Music Generation）**：给定文本、风格标签或 MIDI，生成数分钟级别的复调音乐。涉及长程结构、和声与节奏，将在下一篇单独讨论。

不同任务的频率范围差异巨大：语音 8 kHz 即可懂、音乐需要 44.1 kHz 才能保留高频细节；这意味着**采样率**与**波形长度**直接决定了模型的显存和算力预算。

## 二、早期音频生成：自回归慢速时代

### 2.1 WaveNet：因果卷积的逐点采样

WaveNet（van den Oord et al., 2016）把音频生成视为**自回归密度估计**：

$$
p(x) = \prod_{t=1}^{T} p(x_t \mid x_{<t})
$$

每个 $x_t$ 用 softmax 预测下一个采样的 256 类 $\mu$-law 量化值。WaveNet 用**膨胀因果卷积（dilated causal convolution）** 扩大感受野，每层 dilation 指数增长：

$$
d_l = 2^{l \bmod L}, \quad L = 10
$$

使得 30 层网络的感受野达到 1024 步，对应约 60 ms 的上下文。WaveNet 在 TTS 上首次达到接近真人 MOS，但**推理慢到 1 秒音频需要数十秒计算**，因为必须逐点采样。

### 2.2 SampleRNN 与 WaveRNN

SampleRNN（Mehri et al., 2017）把音频分成多尺度帧，每层 RNN 在不同时间分辨率上建模。WaveRNN（Kalchbrenner et al., 2018）则用单层 GRU + 稀疏化的双 softmax 输出，把推理时间压缩到实时的 4 倍，仍不可实用。**自回归路线的核心瓶颈：$O(T)$ 顺序采样无法并行**。这为后来的 GAN、Flow、Diffusion 路线埋下伏笔。

## 三、扩散模型基础：从图像到波形

### 3.1 前向过程与反向过程

DDPM（Ho et al., 2020）定义一个 $T$ 步马尔可夫前向过程，逐步把数据 $x_0$ 加噪成高斯噪声：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t \mathbf{I})
$$

记 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，则任意时刻的闭式采样为：

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

训练时，网络 $\epsilon_\theta(x_t, t)$ 预测加入的噪声，损失函数为：

$$
\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]
$$

推理时从 $x_T \sim \mathcal{N}(0, \mathbf{I})$ 出发，用学习到的反向过程一步步去噪。

### 3.2 Score Matching 与等价性

DDPM 的 $\epsilon$-预测与**分数匹配（Score Matching）**（Hyvärinen, 2005）有精确等价性：$\epsilon_\theta(x_t, t) \propto \nabla_{x_t} \log p_t(x_t)$，即学习"对数密度的梯度"。这一观点在 Yang et al.（2023）的 DiffFlow 综述里被系统阐明，也是后续 **Consistency Models**（Song et al., 2023）单步生成的理论基础。

### 3.3 图像扩散 vs 音频扩散

图像扩散以 256×256 RGB 图为对象，维度约 200K；音频以 24 kHz 单声道为例，每秒就是 24000 维的标量序列。直接迁移图像 U-Net 会出现两个问题：（1）显存随序列长度线性爆炸；（2）每秒采样率梯度对应毫秒级时序结构，必须保留高频细节。**频域扩散**成为主流解决方案。

## 四、频域扩散：DiffWave / WaveGrad / SpecGrad

### 4.1 DiffWave（2020）

DiffWave（Kong et al., 2020）保留 DDPM 的时域扩散，但在网络结构上做了两处音频适配：

- 用 **1D U-Net** 替代 2D U-Net，卷积核沿时间轴滑动；
- 引入 **Spectrogram Augmentation**：对梅尔谱做随机时频掩码，避免过拟合。

训练目标与 DDPM 完全一致，DiffWave 在 TTS 和声音事件合成上 MOS 接近 WaveNet，但推理要 50-100 步采样。

### 4.2 WaveGrad（2020）

WaveGrad（Chen et al., 2020）把音频扩散做成**条件式**声码器：输入梅尔谱 $M$，输出波形 $x$。关键贡献是**变步数推理**：通过分析 $\bar{\alpha}_t$ 的信噪比，把推理步数从 1000 压缩到 6 步仍能保持质量，推理速度首次接近实时。这是音频扩散走向实用的转折点。

### 4.3 SpecGrad（2021）

SpecGrad（Koizumi et al., 2021）观察到：直接在波形上加噪再反推，会让网络浪费容量去建模频谱细节。SpecGrad 的思路是**改用对数梅尔谱上的扩散**，再用一个 Griffin-Lim 风格的相位重建器生成波形。在工业级 TTS 上首次做到 MOS 4.5+，被 LINE/NAVER 的商用系统采用。

### 4.4 三者对比

| 方法 | 扩散空间 | 步数 | 推理速度 | 适用场景 |
| --- | --- | --- | --- | --- |
| DiffWave | 波形 | ~50 | 0.05×实时 | 通用音频 |
| WaveGrad | 波形 | ~6 | 1×实时 | TTS 声码器 |
| SpecGrad | 对数梅尔 | ~50 | 1×实时 | TTS 商用 |

频域扩散的代价是**相位信息丢失**：梅尔谱压缩把高频相位丢掉了，需要额外的相位重建或对抗损失补回来。

## 五、AudioLDM：CLAP + 潜在扩散的范式

把文本变成音频，最朴素的做法是 TTS + 音效合成。但这要求先有"文本-声音事件"配对数据，标注极贵。**AudioLDM**（Liu et al., 2023）借鉴 Stable Diffusion 的思路：用一个跨模态嵌入空间把文本和音频对齐，然后在**该空间的潜变量上做扩散**。

### 5.1 CLAP：对比语言-音频预训练

CLAP（Contrastive Language-Audio Pre-training, Elizalde et al., 2023）借鉴 CLIP（Radford et al., 2021）的双塔结构：

$$
\mathcal{L}_{\text{CLAP}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\text{sim}(t_i, a_i)/\tau)}{\sum_{j} \exp(\text{sim}(t_i, a_j)/\tau)}
$$

其中 $t_i, a_i$ 分别是文本和音频的嵌入，$\tau$ 是温度。训练后，CLAP 文本编码器可作为音频扩散模型的**条件编码器**，无需任何配对标注即可用文本控制音频生成。

### 5.2 Latent Diffusion

AudioLDM 不在原始波形上做扩散，而用一个预训练的 **VAE** 把音频压缩到潜变量 $z \in \mathbb{R}^{C \times T/200}$（约 200×压缩率），再在 $z$ 上跑 LDM（Rombach et al., 2022）。这样做带来三方面收益：

1. **显存下降 200×**，24 kHz 音频从 24000 维降到约 120 维；
2. **速度提升 200×**，扩散步数不再受 $T$ 限制；
3. **复用图像 LDM 的超参**，跨模态迁移成本低。

### 5.3 AudioLDM 2

AudioLDM 2（Liu et al., 2023）进一步用 **Audio-MAE** 把音频、文本、图像统一编码到同一语义空间，使得"一段文字 + 一张图片"共同控制音频生成成为可能。

## 六、PyTorch 关键模块：频域扩散的最小实现

下面是一个最小但能跑通的频域扩散训练循环（约 60 行核心代码），展示前向加噪、损失计算、推理采样的标准写法：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class GaussianDiffusion:
    """标准 DDPM 调度器，封装 β_t、α̅_t 与采样流程。"""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        # 线性调度 β_t，图像与音频扩散最常用
        self.beta = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)        # ᾱ_t

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """闭式采样：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε"""
        a = self.alpha_bar.to(x0.device)[t].sqrt().unsqueeze(-1)
        b = (1 - self.alpha_bar.to(x0.device)[t]).sqrt().unsqueeze(-1)
        return a * x0 + b * noise

    def training_loss(self, model, x0: torch.Tensor, cond: torch.Tensor):
        """单步训练损失：L = ||ε - ε_θ(x_t, t, cond)||²"""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t, cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def ddpm_sample(self, model, shape, cond, device):
        """DDPM 反向采样：从 x_T ~ N(0,I) 逐步去噪"""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            pred_noise = model(x, torch.full((shape[0],), t, device=device), cond)
            alpha_t = self.alpha.to(device)[t]
            alpha_bar_t = self.alpha_bar.to(device)[t]
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * pred_noise
            ) + self.beta.to(device)[t].sqrt() * z
        return x


class ConditionalUNet1D(nn.Module):
    """1D U-Net：对波形直接做扩散，适合 DiffWave 风格。"""

    def __init__(self, in_ch: int = 1, base_ch: int = 64, time_dim: int = 128, cond_dim: int = 512):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                       nn.Linear(time_dim, time_dim))
        self.cond_proj = nn.Linear(cond_dim, time_dim)
        # 下采样路径：1 → 64 → 128 → 256
        self.down1 = nn.Conv1d(in_ch, base_ch, 5, padding=2)
        self.down2 = nn.Conv1d(base_ch, base_ch * 2, 5, padding=2, stride=2)
        self.down3 = nn.Conv1d(base_ch * 2, base_ch * 4, 5, padding=2, stride=2)
        # 中间块
        self.mid = nn.Conv1d(base_ch * 4, base_ch * 4, 3, padding=1)
        # 上采样路径：256 → 128 → 64 → 1
        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.out = nn.Conv1d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t, cond):
        # t 与 cond 拼成时间-条件向量
        t_emb = sinusoidal_embedding(t, 128).to(x.device)
        t_emb = self.time_mlp(t_emb) + self.cond_proj(cond.mean(dim=-1))
        h1 = F.silu(self.down1(x))
        h2 = F.silu(self.down2(h1))
        h3 = F.silu(self.down3(h2))
        h = F.silu(self.mid(h3)) + t_emb.unsqueeze(-1)
        h = F.silu(self.up2(h)) + h2
        h = F.silu(self.up1(h)) + h1
        return self.out(h)


def sinusoidal_embedding(t: torch.Tensor, dim: int):
    """标准 Transformer 位置编码：t → R^dim"""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / half))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# 烟测：对数梅尔谱作为条件 cond，波形作为目标 x0
if __name__ == "__main__":
    diffusion = GaussianDiffusion(T=200)        # 音频任务用较小 T
    model = ConditionalUNet1D(in_ch=1)
    x0 = torch.randn(2, 1, 24000)              # 2 条 1 秒 24 kHz 音频
    cond = torch.randn(2, 80, 100)              # 80 维梅尔谱 × 100 帧
    print("loss:", diffusion.training_loss(model, x0, cond).item())
```

代码里几个值得展开的工程细节：

- **`alpha_bar` 的累积**：直接 `cumprod` 一次性算出所有 $\bar{\alpha}_t$，采样时按索引查表，避免每步重新计算。
- **频域条件**：把梅尔谱投影到与时间嵌入同维度后相加，扩散网络只需一个统一的 `cond` 输入。AudioLDM 把 `cond` 换成 CLAP 文本嵌入即可。
- **步数选择**：训练用 1000 步（$\beta$ 调度稳定），推理可裁剪到 50 步用 DDIM（Song et al., 2020）加速。

## 七、音频扩散的优缺点与未来方向

**优点**：（1）质量高，MOS 多次超过自回归与 GAN；（2）训练稳定，对抗损失无需调参；（3）条件灵活，可以拼接文本、频谱、类别标签。

**缺点**：（1）推理慢：50-100 步顺序去噪仍难实时；（2）显存大：波形 24 kHz × 60 秒就是 1.44M 维，必须靠潜在扩散或频域扩散压缩；（3）相位难建模：波形生成本质是相位敏感的，而梅尔谱压缩丢相位。

未来方向包括：**Consistency Models**（Song et al., 2023）做单步生成、**Latent Consistency**（Luo et al., 2023）把潜在空间与一致性训练结合、**Flow Matching**（Lipman et al., 2023）替代 DDPM 调度器。当前研究热点是"高质量 + 实时"的统一框架。

## 小结

| 方法 | 扩散空间 | 步数 | 推理速度 | 代表场景 |
| --- | --- | --- | --- | --- |
| WaveNet | 波形自回归 | — | 0.01× | 历史里程碑 |
| DiffWave | 波形 | 50 | 0.05× | 通用音频 |
| WaveGrad | 波形 | 6 | 1× | TTS 声码器 |
| SpecGrad | 对数梅尔 | 50 | 1× | 商用 TTS |
| AudioLDM | 潜在空间 | 200 | 0.5× | 文本→音频 |

音频扩散让"用自然语言描述声音"成为可能，但 **质量 vs 速度** 的权衡仍是工程核心。下一篇将聚焦在更复杂的子任务——**音乐生成**——以及 Jukebox、MusicLM、MusicGen 的范式之争。
