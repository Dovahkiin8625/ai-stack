# 音频 Tokenizer：EnCodec 与 SoundStream

过去三年大语言模型最重要的范式迁移之一是"**万物皆可 Token**"：把图像、音频、视频都离散成 token 序列，让 Transformer 统一处理。OpenAI 的 AudioGPT、Google 的 AudioPaLM、Meta 的 MusicGen，本质上都依赖一个共同的底层组件——**音频 Tokenizer**。它把连续波形 $x \in \mathbb{R}^T$ 编码成离散码本索引序列 $\mathbf{c} \in \{1, \dots, V\}^L$，再由解码器恢复回波形。这一篇解析 RVQ 量化机制、SoundStream 与 EnCodec 的工程权衡、语义 vs 声学的码本之争，以及 DAC 的最新进展。

## 一、为什么需要音频 Tokenizer？

### 1.1 与 LLM 接口的必然要求

离散 token 是 LLM 的天然语言：自回归 Transformer 必须输出"下一个 token 的概率分布"，**对连续向量输出束手无策**。一旦把音频离散化，就可以让 GPT-style 模型直接预测音频 token，复用成熟的 LLM 训练栈。AudioLM（Borsos et al., 2023）和 VALL-E（Wang et al., 2023）都依赖 SoundStream / EnCodec 这类 tokenizer。

### 1.2 压缩率与计算可行性

24 kHz 单声道音频每秒 24000 个 float32 采样，3 分钟的歌曲就是 1.04 GB 的 FP32 张量。Transformer 的 $O(L^2)$ 注意力完全无法处理。EnCodec 把音频压到 75 Hz（24 kHz ÷ 320 步），3 分钟只需 13500 个 token，**压缩率 320×**——这是 LLM 处理音频的前提。

### 1.3 语义保留与重建质量的权衡

理想的 tokenizer 应同时满足：（1）高压缩率；（2）重建后听感无损；（3）token 序列在语义层面对人类可解释（如"鼓点"、"钢琴"）。**声学码本**（EnCodec）擅长 (1)(2)，**语义码本**（HuBERT）擅长 (3)，二者结合是当前研究热点。

## 二、RVQ：残差向量量化的多码本结构

### 2.1 VQ-VAE 的基础

VQ-VAE（van den Oord et al., 2017）把编码器输出 $z_e(x)$ 通过最近邻查找映射到码本 $\mathcal{C} = \{e_1, \dots, e_V\}$：

$$
z_q = e_{k^*}, \quad k^* = \arg\min_{k} \|z_e - e_k\|^2
$$

损失函数为：

$$
\mathcal{L} = \|x - \hat{x}\|^2 + \|\text{sg}[z_e] - z_q\|^2 + \beta \|z_e - \text{sg}[z_q]\|^2
$$

其中 `sg` 是 stop-gradient。第一项是重建损失，第二项更新码本，第三项更新编码器。

### 2.2 RVQ 的多码本分层

单层 VQ 的码本容量有限（$V=512$ 时每帧只能区分 512 种模式）。**RVQ（Residual Vector Quantization）**（Zeghidour et al., 2021）把量化残差递归：

$$
z_q = \sum_{k=1}^{K} e_{c^{(k)}}, \quad c^{(k)} = \arg\min_{k} \| r^{(k-1)} - e_k \|^2
$$

其中 $r^{(0)} = z_e$，$r^{(k)} = r^{(k-1)} - e_{c^{(k)}}$ 是第 $k$ 层之后的残差。直觉上：第一层捕捉粗粒度结构（能量包络），第二层捕捉纹理，第三层捕捉高频细节……最后一层捕捉最精细的采样点抖动。

### 2.3 比特率计算

每帧输出 $K$ 个码本索引，每码本 $V$ 个值，用 $\log_2 V$ bit。EnCodec 默认 $K=32, V=1024$：

$$
\text{bitrate} = 50 \text{ Hz} \times 32 \times 10 \text{ bit} = 16 \text{ kbps}
$$

这是 MP3 128 kbps 的 1/8，但仍能达到接近透明的语音质量。**码本越多，重建越好，比特率也越高**——这是 SoundStream 与 EnCodec 调参的核心维度。

## 三、SoundStream：全卷积 + RVQ + 判别器

### 3.1 架构

SoundStream（Zeghidour et al., 2021, Google）首次把"**全卷积编码器 + RVQ + 流式解码**"组合成完整流水线：

- **编码器**：4 层下采样卷积（步幅 2/4/5/8，总压缩率 320×），输出 $z_e \in \mathbb{R}^{C \times L}$；
- **量化器**：$K$ 层 RVQ，每层独立码本 $V=1024$；
- **解码器**：5 层上采样转置卷积，对称恢复；
- **判别器**：HiFi-GAN 风格的多周期 + 多尺度判别器，对抗训练。

### 3.2 损失函数

SoundStream 的训练损失由三部分组成：

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{cmt}} \mathcal{L}_{\text{cmt}}
$$

- **重建损失**：多尺度梅尔谱 L1 距离而非简单 MSE，因为人耳对梅尔尺度敏感；
- **对抗损失**：用判别器区分重建波形与真实波形；
- **承诺损失（commitment loss）**：把编码器输出"拉向"所选码本向量，避免编码器不更新。

### 3.3 流式推理

SoundStream 的所有卷积都是**因果**的（感受野只向左看），因此可以做到**流式编码**：边录边编码、边传边解码。这对实时语音通信至关重要。EnCodec 在 v3 版本里也加入了流式模式，但默认实现是非因果的（双向感受野）。

## 四、EnCodec：开源版的工业部署

### 4.1 Meta 的开源方案

EnCodec（Défossez et al., 2022, Meta AI）是 SoundStream 思路的开源实现，核心改进：

1. **LSTM 中间层**：在编码器与解码器之间加入 2 层 LSTM，捕捉更长上下文；
2. **更深的卷积栈**：编码器 4 层 + 解码器 4 层 + 中间 4 个残差块；
3. **多尺度 STFT 判别器**：替代 HiFi-GAN 风格判别器，频谱质量更好；
4. **公开训练代码与多语料预训练权重**：24 kHz 单声道模型权重直接可用。

EnCodec 默认配置：50 Hz token rate，$K=32$ 码本，目标比特率 12-24 kbps。Meta 同期还发布了 **HiFi-GAN 风格的神经声码器**作为对比基线。

### 4.2 与 SoundStream 的对比

| 维度 | SoundStream | EnCodec |
| --- | --- | --- |
| 架构 | 全卷积 | 卷积 + LSTM |
| 流式 | 因果 | 非因果（v3 加入因果） |
| 判别器 | HiFi-GAN | 多尺度 STFT |
| 开源权重 | 否（Google 内部） | 是（Meta 开源） |
| 24 kHz 语音 MOS | 4.3+ | 4.2+ |

两者质量接近，EnCodec 因开源成为业界事实标准——AudioCraft、VALL-E、MusicGen 都用它。

## 五、DAC：更高重建质量的下一站

DAC（Descript Audio Codec, Kumar et al., 2023）在 EnCodec 基础上做了三项工程改进：

1. **更深的编码器**：12 层卷积，压缩率最高 320×，但每层步幅更小、信息保留更多；
2. **SNR 感知的码本选择**：训练时随机丢弃部分码本，让模型在低比特率下也能稳健重建；
3. **更好的激活函数**：用 Snake 激活函数替代 ReLU，捕捉音频的周期性。

DAC 在 24 kHz 语音上把 ViSQOL 分数从 EnCodec 的 4.0 提升到 4.5，更接近原始波形。**代价是模型体积更大、推理更慢**，对实时系统不太友好。

## 六、语义 vs 声学 Tokenizer：两条路线的权衡

### 6.1 声学码本

EnCodec、SoundStream 的 token 是**纯声学**的：每个码本索引对应"一段固定频率的波形细节"。优点是重建质量极高，缺点是**语义不可解释**——同一个"鼓点"在声学码本里可能是 (3, 17, 89, 4, ...) 这样一串无规律数字。

### 6.2 语义码本

**HuBERT**（Hsu et al., 2021）的 token 是**自监督聚类**得到的语义类别：用 k-means 把 HuBERT 特征聚成 100-500 类，每帧分配一个语义标签。优点：token 序列本身有"词义"（连续相同的 token 意味着同一个音持续），适合作为 LLM 的训练目标。缺点：重建质量差——从 HuBERT token 恢复的波形听起来像"模糊的同音"，因为聚类丢掉了大量细节。

### 6.3 混合路线

当前最先进系统（VALL-E 2, AudioPaLM）采用**两阶段 token**：

$$
\mathbf{c} = (\mathbf{c}_{\text{sem}}, \mathbf{c}_{\text{acoustic}})
$$

1. 先用 HuBERT 风格的语义 token 训练 LLM（负责"说什么"）；
2. 再用 EnCodec 的声学 token 训练声码器（负责"怎么说"）。

这种分工让 LLM 学习**高层语义结构**，声码器学习**底层波形细节**。VALL-E 2（Chen et al., 2024）在这种范式下首次实现"3 秒样本克隆 30 秒语音"。

## 七、EnCodec 使用示例代码

下面是 EnCodec 的最小调用示例，展示编码、解码、24 kHz 重建全流程：

```python
from encodec import EncodecModel
from encodec.utils import save_audio, convert_audio
import torch
import torchaudio

# 1. 加载预训练 24 kHz 模型（自动下载约 300 MB）
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)   # 6 kbps，目标带宽
model.eval()

# 2. 加载音频并重采样到 24 kHz
wav, sr = torchaudio.load("speech.wav")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
# wav shape: (1, 1, T)，T = 时长(秒) × 24000

# 3. 编码：waveform → discrete codes
with torch.no_grad():
    encoded = model.encode(wav)
codes = encoded.codes              # (B, K, L) — K 个码本 × L 帧
print("codes shape:", codes.shape)  # e.g. (1, 32, 1500) for 30s

# 4. 查看每个码本的使用情况
for k in range(codes.shape[1]):
    n_unique = codes[0, k].unique().numel()
    print(f"codebook {k}: {n_unique}/1024 active")

# 5. 解码：codes → waveform
with torch.no_grad():
    decoded = model.decode(encoded)
recon_wav = decoded.audio           # (B, 1, T)

# 6. 保存重建音频
save_audio(recon_wav, "recon.wav", model.sample_rate)

# 7. 计算重建质量
import torchaudio.functional as F
mse = ((wav - recon_wav) ** 2).mean().item()
print(f"reconstruction MSE: {mse:.6f}")
```

如果要把 EnCodec 集成到 LLM 训练流程中，关键是 `codes` 张量可以**直接当作整数 token** 喂给 Embedding 层：

```python
# 把 codes 压平 (B, K*L) 作为 LLM 的输入 token
flat_codes = codes.transpose(1, 2).reshape(codes.shape[0], -1)
# flat_codes shape: (B, L*K)，可用 nn.Embedding(V, D) 查表

# LLM 输出后，reshape 回 (B, K, L) 再用 model.decode 还原波形
predicted_codes = llm_output.argmax(dim=-1).reshape(B, L, K).transpose(1, 2)
recon = model.decode([predicted_codes])
```

## 八、未来方向：多带宽、低比特率、长上下文

音频 Tokenizer 的下一个突破点在三个方向：

1. **多带宽自适应**：同一模型支持 6 / 12 / 24 kbps 三档比特率，根据场景自动选择。DAC 已部分实现。
2. **超低比特率语义保留**：1 kbps 级别的码本仍能保留可懂的语音内容，适合 LLM 训练。SoundStream 的扩展工作 SoundStorm（Borsos et al., 2023）正是为此设计。
3. **长上下文建模**：当前码本 50 Hz，对应 20 ms 一帧。下一代 tokenizer 可能降到 10 Hz（100 ms 一帧），让 LLM 处理 10 分钟级音频只需 6000 个 token。

## 小结

| Tokenizer | 量化方式 | 比特率 | 开源 | 重建质量 | 典型场景 |
| --- | --- | --- | --- | --- | --- |
| SoundStream | RVQ + 判别器 | 3-18 kbps | 否 | 高 | Google 内部 |
| EnCodec | RVQ + LSTM + STFT 判别器 | 6-24 kbps | 是 | 高 | MusicGen / VALL-E |
| DAC | 深度卷积 + Snake + SNR 感知 | 6-24 kbps | 是 | 很高 | 高保真语音 |
| HuBERT | k-means 聚类 | < 1 kbps | 是 | 低（语义） | 语义 LLM |
| 混合（语义 + 声学） | HuBERT + EnCodec | 6-12 kbps | 部分 | 高 | VALL-E 2 |

音频 Tokenizer 是**音频世界通往 LLM 的翻译器**。没有它，再强大的语言模型也无法"听见"声音。下一篇我们将讨论如何评价这些模型——**客观指标与主观听测**的组合评估协议。
