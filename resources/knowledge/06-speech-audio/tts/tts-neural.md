# 神经 TTS：Tacotron 与 FastSpeech 系列

传统 TTS 把"语言学特征 → 声学参数 → 波形"切成三段，每一段都有独立的目标函数和独立的误差累积。2017 年 Wang 等人提出 **Tacotron 2**（Shen et al., 2017）首次把前两段打通，用一个端到端 Seq2Seq 模型直接从字符预测梅尔频谱，再交给 WaveNet 神经声码器生成波形，自然度一举超过所有传统系统。紧接着 2019 年微软的 **FastSpeech**（Ren et al., 2019）和 2020 年的 **FastSpeech 2**（Ren et al., 2020）解决了 Tacotron 系列的自回归推理慢、注意力不稳定等痛点，标志着神经 TTS 进入"高质量 + 高效率"时代。

## 一、Tacotron 2：端到端梅尔频谱预测

### 1.1 整体架构

Tacotron 2 由三部分组成：**编码器（Encoder）**、**带注意力的解码器（Decoder）**、**后处理网络（Post-Net）**。输入是字符序列，输出是 80 通道梅尔频谱帧序列，最后再交给独立的 WaveNet 声码器合成波形：

```
字符序列 → [Encoder] → 编码序列
                          ↓
                       [Attention]
                          ↓
梅尔频谱 ← [Decoder] ← 解码状态
                          ↓
                  [Post-Net (残差精修)]
```

### 1.2 编码器：字符嵌入 + CBHG

编码器先把每个字符映射成 512 维嵌入，再过一个 **CBHG（Convolution Bank + Highway + Bidirectional GRU）** 模块。CBHG 是 Tacotron 系列的关键积木，作用是同时捕捉**局部 n-gram 模式**和**长距离上下文**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBHG(nn.Module):
    """Tacotron 2 中的 CBHG 模块：Conv1D bank → max-pool → highway → BiGRU."""

    def __init__(self, hidden_dim: int, K: int = 8, projections_dim: int = 128):
        super().__init__()
        # 1) Conv1D bank: K 个不同 kernel size 的卷积并列
        self.conv_bank = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k + 1,
                      padding=k // 2, bias=False)
            for k in range(K)
        ])
        # 2) 池化 + 投影
        self.pool = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.proj1 = nn.Conv1d(hidden_dim * (K + 1), projections_dim, kernel_size=3, padding=1)
        self.proj2 = nn.Conv1d(projections_dim, hidden_dim, kernel_size=3, padding=1)
        # 3) Highway 网络：4 层
        self.highway = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(4)
        ])
        # 4) BiGRU
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T) for Conv1d
        outs = [F.relu(c(x)) for c in self.conv_bank]   # K 个 bank
        outs.append(self.pool(x))                         # 1 个 pool
        h = torch.cat(outs, dim=1)                        # (B, (K+1)*C, T)
        h = F.relu(self.proj1(h))
        h = self.proj2(h)                                 # (B, C, T)
        h = h.transpose(1, 2)                             # (B, T, C)

        # Highway residual
        for layer in self.highway:
            gate = torch.sigmoid(layer(h))
            h = gate * layer(h) + (1 - gate) * h
        # BiGRU 输出维度是 2*hidden_dim，这里再投影回 hidden_dim
        h, _ = self.gru(h)
        # 论文里 proj 之后再过 BiGRU；简化版本直接返回 BiGRU 输出
        return h
```

### 1.3 解码器：自回归 + 位置敏感注意力

解码器每一步从**上一个梅尔频谱帧**预测**下一个梅尔频谱帧**（teacher forcing），用一个 2 层 GRU + 全连接输出层。每 2 个解码步生成 1 帧梅尔（r=2），加速收敛。关键创新是引入 **Location-Sensitive Attention**——把上一时刻的注意力权重 $\alpha_{t-1}$ 通过一个卷积后再喂给注意力打分函数：

$$
e_{t,i} = \mathbf{v}^\top \tanh\big( \mathbf{W}_s \mathbf{s}_{t-1} + \mathbf{W}_h \mathbf{h}_i + \mathbf{U} * \alpha_{t-1} + \mathbf{b} \big)
$$

其中 $*$ 表示卷积。这样做的好处是：**模型能"记住"自己上一时刻注意到哪里**，避免自回归注意力常见的"跳过某些字符"或"重复某些字符"错误。

### 1.4 停止令牌（Stop Token）

Tacotron 2 的解码器每一步还会输出一个 **sigmoid 标量** $\zeta_t$，表示"是否该停止"。训练时用 binary cross-entropy，推理时第一个 $\zeta_t > 0.5$ 的位置即为句子结束。这一机制让模型可以处理变长输出，不需要预先指定梅尔帧数。

### 1.5 后处理网络（Post-Net）

解码器输出的梅尔频谱再过一个 5 层卷积的 Post-Net 做**残差精修**：把解码输出 + Post-Net 输出作为最终预测。直觉是 Post-Net 专门负责修正解码器"过平滑"的低频细节，显著提升合成自然度。

## 二、Tacotron 2 的简化 PyTorch 实现

下面是一个能跑通的极简版本（去掉了大部分工程细节，保留核心结构）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationSensitiveAttention(nn.Module):
    """Tacotron 2 位置敏感注意力。"""

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 128):
        super().__init__()
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=False)
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.U = nn.Conv1d(1, attn_dim, kernel_size=3, padding=1, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, s_prev, enc, alpha_prev):
        # s_prev:    (B, dec_dim)  上一时刻解码状态
        # enc:       (B, T, enc_dim)
        # alpha_prev:(B, T)  上一时刻注意力权重
        alpha_conv = self.U(alpha_prev.unsqueeze(1)).transpose(1, 2)  # (B, T, attn_dim)
        score = self.v(torch.tanh(
            self.W_s(s_prev).unsqueeze(1) + self.W_h(enc) + alpha_conv
        )).squeeze(-1)                                                # (B, T)
        alpha = F.softmax(score, dim=-1)
        context = (enc * alpha.unsqueeze(-1)).sum(dim=1)              # (B, enc_dim)
        return context, alpha


class Tacotron2(nn.Module):
    """极简版 Tacotron 2：Encoder + Attention + Decoder + Post-Net."""

    def __init__(self, vocab_size: int, n_mels: int = 80,
                 enc_dim: int = 256, dec_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.encoder = nn.Sequential(
            nn.Conv1d(256, enc_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(enc_dim),
            nn.ReLU(),
            nn.Conv1d(enc_dim, enc_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(enc_dim),
            nn.ReLU(),
            nn.LSTM(enc_dim, enc_dim // 2, batch_first=True, bidirectional=True),
        )
        self.attention = LocationSensitiveAttention(enc_dim, dec_dim)
        self.decoder_rnn = nn.LSTMCell(dec_dim + n_mels, dec_dim)
        self.mel_proj = nn.Linear(dec_dim + enc_dim, n_mels)
        self.stop_proj = nn.Linear(dec_dim + enc_dim, 1)
        # 简化版 Post-Net：2 层卷积
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Conv1d(512, n_mels, kernel_size=5, padding=2),
        )

    def forward(self, text, mel_target=None):
        # text: (B, T_text) 字符 id 序列
        x = self.embedding(text).transpose(1, 2)  # (B, 256, T)
        enc = self.encoder(x)                      # (B, T, enc_dim)

        B, T_enc, _ = enc.size()
        T_mel = mel_target.size(1) if mel_target is not None else 400
        mel_prev = torch.zeros(B, 80, device=enc.device)
        alpha = torch.zeros(B, T_enc, device=enc.device)
        s = torch.zeros(B, 256, device=enc.device)
        c = torch.zeros(B, 256, device=enc.device)

        mel_outs, stop_outs = [], []
        for t in range(T_mel):
            gru_in = torch.cat([mel_prev, c], dim=-1)
            s, c = self.decoder_rnn(gru_in, (s, c))
            context, alpha = self.attention(s, enc, alpha)
            out = torch.cat([s, context], dim=-1)
            mel_t = self.mel_proj(out)
            stop_t = torch.sigmoid(self.stop_proj(out))
            mel_outs.append(mel_t)
            stop_outs.append(stop_t)
            # teacher forcing：用真实梅尔作为下一步输入
            mel_prev = (mel_target[:, t] if mel_target is not None else mel_t)
        mel_outs = torch.stack(mel_outs, dim=1)  # (B, T_mel, 80)

        # Post-Net 残差精修
        mel_post = mel_outs + self.postnet(mel_outs.transpose(1, 2)).transpose(1, 2)
        return mel_post, torch.stack(stop_outs, dim=1).squeeze(-1)
```

## 三、FastSpeech：非自回归 + Duration Predictor

Tacotron 2 在自然度上达到了历史新高，但有两个工程痛点：

1. **推理慢**：自回归必须逐帧生成，一句 10 秒的音频（100 帧梅尔）要跑 100 次解码，在 GPU 上 RTF（Real-Time Factor）仍然偏大，无法实时流式合成。
2. **注意力失败**：注意力偶尔会跳过字符、重复字符、或提前停止（babbling / hallucination），鲁棒性差。

2019 年微软提出的 **FastSpeech** 用**非自回归（Non-Autoregressive, NAR）** 范式彻底解决这两个问题。它的核心思想是：**时长是显式可预测的**——先用一个 duration predictor 预测每个 token 对应多少帧梅尔，再把编码器输出"按时长复制"成解码器长度的序列，最后一次性并行解码所有梅尔帧。

### 3.1 Length Regulator

FastSpeech 的关键模块是 **Length Regulator（LR）**，把编码序列 $\mathbf{H} = [\mathbf{h}_1, \dots, \mathbf{h}_T]$ 按预测的时长 $\mathbf{d} = [d_1, \dots, d_T]$ 扩展成解码序列：

$$
\mathbf{H}'_i = \mathbf{h}_{\mathcal{R}(i)}, \quad \text{其中 } \mathcal{R}(i) = k \text{ 满足 } \sum_{j=1}^{k-1} d_j < i \le \sum_{j=1}^{k} d_j
$$

```python
def length_regulator(enc, durations):
    """
    enc:       (B, T_text, D)
    durations: (B, T_text)  每个 token 对应的梅尔帧数（向上取整）
    return:    (B, T_mel, D)  T_mel = durations.sum(1)
    """
    outs = []
    for b in range(enc.size(0)):
        expanded = torch.repeat_interleave(enc[b], durations[b], dim=0)
        outs.append(expanded)
    return torch.stack([F.pad(o, (0, 0, 0, max_mel - o.size(0)))[:max_mel]
                        for o, max_mel in zip(
                            outs, [max(o.size(0) for o in outs)])])
```

### 3.2 Duration Predictor 的训练

Duration Predictor 是个 2 层卷积 + ReLU 的小网络，从编码器隐状态预测每个 token 对应的梅尔帧数。训练时的 ground-truth 时长来自**强制对齐（forced alignment）**——用一个外部工具（如 MFA / Montreal Forced Aligner）把训练数据里的"字符-梅尔"对齐结果取出来。损失函数：

$$
\mathcal{L}_{\text{dur}} = \text{MSE}(\hat{\mathbf{d}}, \mathbf{d}) + \text{MSE}(\log \hat{\mathbf{d}}, \log \mathbf{d})
$$

加 $\log$ 项是为了让小数值（比如 2 帧）有足够的梯度。

### 3.3 整体损失

FastSpeech 用 **FFT (Feed-Forward Transformer)** Block 同时做编码和解码，损失包含三项：

$$
\mathcal{L} = \mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{post-mel}} + \mathcal{L}_{\text{dur}}
$$

推理时**完全并行**：duration predictor 一次输出所有时长，length regulator 一次扩展，所有梅尔帧同步生成。

## 四、FastSpeech 2：直接在 ground-truth 上训练

FastSpeech 有一个微妙但重要的缺陷：**它要靠一个外部的强制对齐工具来获取时长标签**。强制对齐在嘈杂语音、罕见语言、儿童语音上经常出错，导致时长监督信号本身有噪声。2020 年的 **FastSpeech 2** 干脆把"教师模型"整个砍掉，直接在 ground-truth 声学特征上做回归。

### 4.1 三类信息的显式建模

FastSpeech 2 让模型同时预测 **时长（duration）、基频（pitch）、能量（energy）**，并把这三者显式加回声学特征：

$$
\mathcal{L} = \mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{dur}} + \mathcal{L}_{F_0} + \mathcal{L}_{E}
$$

能量定义为 $|x_t|^2$（梅尔频谱帧的 L2 范数）。这种"显式 + 多任务"的做法有两个直接收益：

- **时长标签不需要外部对齐器**：用 teacher 模型内部提取的时长即可。
- **韵律可控**：推理时可以手动调整 $F_0$ 或能量，直接改变合成语音的情感、语速、响度。

### 4.2 FastSpeech 2s：联合训练声码器

FastSpeech 2s 进一步在同一个模型里加一个 WaveNet 风格的解码头，**直接从梅尔预测原始波形样本**，跳过独立训练的声码器。但这条路线在自然度上仍输给"梅尔 + HiFi-GAN"的两阶段方案，因此工业界主流仍是 FastSpeech 2 + HiFi-GAN。

## 五、速度对比：自回归 vs 非自回归

一个直观的对比（基于 V100 GPU，单句约 5 秒）：

| 模型 | 范式 | 推理 RTF | 自然度 MOS | 训练稳定性 |
| --- | --- | --- | --- | --- |
| Tacotron 2 + WaveNet | 自回归 | ~0.5 | 4.53 | 注意力偶发失败 |
| Tacotron 2 + WaveRNN | 自回归 | ~0.2 | 4.50 | 同上 |
| **FastSpeech** | NAR | ~0.02 | 4.00 | 稳定 |
| **FastSpeech 2** | NAR | ~0.02 | 4.30 | 稳定 |
| FastSpeech 2 + HiFi-GAN | NAR | ~0.01 | 4.50+ | 稳定 |

FastSpeech 2 把 RTF 拉到了 0.01 量级（即合成 1 秒音频只需 10 ms 计算），相比 Tacotron 2 提速约 **30-50 倍**，让实时流式 TTS 在 GPU 上成为可能。FastSpeech 2 + HiFi-GAN 的组合至今仍是工业界低延迟 TTS 的主流默认配置。

## 小结

| 模型 | 范式 | 关键创新 | 主要优势 | 主要局限 |
| --- | --- | --- | --- | --- |
| Tacotron 2 | 自回归 | CBHG + Location-Sensitive Attention + Stop Token | 自然度里程碑 | 推理慢、注意力不稳定 |
| Tacotron (1) | 自回归 | Char-LSTM + Griffin-Lim | 端到端第一版 | 自然度差 |
| FastSpeech | NAR | Duration Predictor + Length Regulator | 30x 加速、稳定 | 需外部对齐工具 |
| FastSpeech 2 | NAR | 多任务（pitch/energy）+ ground-truth 训练 | 免对齐、可控 | 仍需独立声码器 |

Tacotron 2 与 FastSpeech 系列把 TTS 从"三段流水线"推进到"端到端 Seq2Seq / NAR Transformer"，但声码器仍是独立的瓶颈。下一篇我们将看到 VITS 如何把声码器也一并端到端化，以及 HiFi-GAN 这类 GAN 声码器如何把波形合成质量推到接近真人的水平。
