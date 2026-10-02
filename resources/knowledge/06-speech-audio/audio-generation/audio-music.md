# 音乐生成：Jukebox 到 MusicGen

音乐是音频生成皇冠上最难摘的明珠。一段 4 分钟、44.1 kHz 立体声的钢琴曲展开成序列是 4 × 60 × 44100 ≈ 1000 万个采样点，比 WaveNet 处理的句子长 4 个数量级。音乐还要同时满足**长程结构**（曲式、调性、段落）、**多轨并行**（鼓、贝斯、和声、人声）和**复调关系**（多个声部同时发声且彼此呼应）。这一篇沿着 OpenAI 的 Jukebox（2020）→ Google 的 MusicLM（2023）→ Meta 的 MusicGen（2023）→ 工业级 Stable Audio / Suno 的演化路径，剖析音乐生成的工程难题与代表方案。

## 一、音乐生成的特殊挑战

### 1.1 长序列依赖

语音的句子级时长通常 < 30 秒，音乐的"句子"是乐句，长度可达数分钟。Transformer 的自注意力是 $O(L^2)$，10M 长度的注意力矩阵需要 4 TB 显存（FP32）——**直接全注意力完全不可行**。这迫使 Jukebox 用稀疏注意力、MusicGen 用码本交错把序列长度压缩到数百量级。

### 1.2 多轨与复调

音乐天然是多通道信号。钢琴曲同时有 88 个键可能发声，乐队混音有十几个声部。MIDI 表示擅长处理复调（每个音是独立事件），但波形表示就要求模型自己学会"多个源叠加"的解混。MusicGen 的解法是把多轨压平成单条 token 流，让 Transformer 学习**隐式的多轨结构**。

### 1.3 风格控制与可控性

文本控制音乐生成比控制语音难得多。音乐提示词包含**风格**（jazz、lo-fi）、**情绪**（sad、energetic）、**乐器**（piano、guitar）、**节奏**（90 BPM、4/4 拍）、**人声歌词**等多个维度，且彼此耦合。MusicLM 的 MuLan 嵌入把这种耦合统一到 128 维向量；MusicGen 用 T5 + 码本交错做解耦控制。

## 二、Jukebox：分层 VQ-VAE + 稀疏 Transformer

### 2.1 三层 VQ-VAE

Jukebox（Dhariwal et al., 2020）的核心是**把分钟级音频压缩成百级长度的离散码**。它用三个独立的 VQ-VAE 分层压缩：

| 层级 | 输入 | 压缩率 | 输出长度（4 分钟） | 码本大小 |
| --- | --- | --- | --- | --- |
| Top | 44.1 kHz 立体声 | 128× | ~13k tokens | 2048 |
| Middle | Top 重建 | 32× | ~52k tokens | 2048 |
| Bottom | 原始波形 | 8× | ~2.6M tokens | 2048 |

最底层负责细节，中间层负责纹理，最高层负责曲式结构。每个 VQ-VAE 用类似 VQ-VAE-2（Razavi et al., 2019）的"自编码 + 离散潜码"思路训练。

### 2.2 稀疏 Transformer

Jukebox 在三个码本层上分别训练 Transformer。Top 层用**全注意力**（因为序列短），Middle 与 Bottom 层用**稀疏注意力**：

$$
\text{Attn}_{\text{sparse}}(Q, K, V) = \sum_{s \in S} \text{softmax}\left(\frac{Q K_s^\top}{\sqrt{d}}\right) V_s
$$

其中 $S$ 是**局部窗口** + **跨码本连接**的稀疏模式。每个查询只关注最近 $W$ 个键（局部时间窗口）和顶层对应的位置（跨层桥接）。这种设计把 $O(L^2)$ 降到 $O(L \cdot W)$，使得 13k 长度的顶层可以在 32GB 显存内训练。

### 2.3 局限与遗产

Jukebox 的音质与人类的 5 分 MOS 评分差距显著（约 2.8 vs 4.5），主要问题是：（1）VQ-VAE 的码本容量限制了高频细节；（2）歌词对齐差，长序列里"唱歌跑调"常见；（3）训练成本极高，单次训练需数百张 V100 跑数周。

但 Jukebox 留下了三大遗产：**分层压缩范式**、**稀疏注意力长序列建模**、以及作为音乐理解模型被广泛复用的预训练权重（JuKeM、MusicLM 都用了 Jukebox 的特征）。

## 三、MusicLM：MuLan 联合嵌入 + 多阶段 Transformer

### 3.1 MuLan：音乐-文本联合嵌入

MusicLM（Agostinelli et al., 2023）的第一步是训练 **MuLan**——一个类似 CLIP 但针对音乐的对比嵌入模型。MuLan 把 30 秒音频片段和对应文本描述投影到同一个 128 维球面：

$$
\mathcal{L}_{\text{MuLan}} = -\log \frac{\exp(\text{sim}(a_i, t_i)/\tau)}{\sum_j \exp(\text{sim}(a_i, t_j)/\tau)}
$$

训练数据来自公开音乐数据集 + 弱标注（音频描述由 GPT-3 风格模型生成）。MuLan 让"用文本控制音乐生成"成为可能，无需任何配对的音乐-文本标注。

### 3.2 多阶段 Transformer

MusicLM 不试图直接生成原始波形，而是三级串联：

1. **语义阶段**：用 MuLan 嵌入条件，生成 **SoundStream 令牌**（约 600 token / 30 秒）。SoundStream（Zeghidour et al., 2021）是 Google 的音频神经编解码器，详见下一篇。
2. **声学细化**：用对比预测编码（CPC）风格的解码器把语义令牌细化成更细粒度的声学令牌。
3. **波形合成**：SoundStream 解码器把令牌恢复成 24 kHz 波形。

这种"语义先验 + 声学细化"的范式后来被 AudioLDM 2 和 MusicGen 部分继承。

### 3.3 局限

MusicLM 是闭源模型，权重未发布，工程界对其细节了解有限；同时 MusicLM 在人声生成上有版权争议，Google 在 2023 年发布但**拒绝公开发布**权重，转而通过 AI Test Kitchen 提供有限访问。

## 四、MusicGen：单阶段 Transformer + 码本交错

### 4.1 核心思想

MusicGen（Copet et al., 2023, Meta）用一个**单阶段 Transformer** 同时处理文本条件、旋律条件、声学令牌，避免了 MusicLM 的多阶段复杂性。关键创新是**码本交错（Codebook Interleaving）**：

EnCodec 把音频压成 $K$ 个并行码本流（每条 50 Hz），MusicGen 不用 flatten 把它们堆叠，而是按时间步"交错"成一条 50 × K Hz 的 token 流：

$$
\mathbf{c}_t = (c_t^{(1)}, c_t^{(2)}, \dots, c_t^{(K)}), \quad K = 4
$$

这样每步预测 4 个码本的联合分布，用一个**延迟预测（Delay Pattern）** 让第 $k$ 个码本只看到前 $k-1$ 个码本的信息：

$$
p(\mathbf{c}_t) = \prod_{k=1}^{K} p(c_t^{(k)} \mid c_{<t}, c_t^{(<k)})
$$

交错模式让 Transformer 一次只预测一个码本，但训练时各码本位置共享上下文，推理并行展开。

### 4.2 架构

MusicGen 是一个标准的解码器 Transformer：

- **文本条件**：T5 编码器输出的文本嵌入，与音频 token 嵌入相加；
- **旋律条件**：可选输入一段参考旋律（梅尔谱），经 Chroma 特征提取后通过线性层投影；
- **位置编码**：旋转位置编码（RoPE）；
- **规模**：300M / 1.5B / 3.3B 三个变体，3.3B 已超过 MusicLM 的容量但训练数据更小。

### 4.3 训练数据与许可

MusicGen 是工业界少有的**使用合法授权数据**训练的大模型——20k 小时来自 Shutterstock 与 Pond5 的音乐库，每条都有版权方授权。这与 Jukebox（用网络爬取的 120 万首歌）和 MusicLM（用私有数据集）形成鲜明对比，也使 MusicGen 成为**可商用**的开源代表。

## 五、工业系统：Stable Audio / Suno / Udio

### 5.1 Stable Audio（Stability AI, 2023）

Stable Audio 用**潜在扩散**（类似 AudioLDM）做音乐生成，编码器是 Stable Audio 自己训练的 DAC（Descript Audio Codec），条件是 CLAP 文本嵌入 + 时长 + 起止时间。最大变体支持 90 秒生成，可商用。

### 5.2 Suno / Udio

Suno（2024）和 Udio（2024）代表了 **diffusion + Transformer 混合** 的最新工业范式：Suno 用 Bark 做歌词+人声，BLOOM 做歌词理解；Udio 则类似 Stable Audio 但用更大的数据集与更精细的条件控制。两者都已支持 4 分钟级别的歌曲生成，并引发了广泛的版权争议——大量训练数据被怀疑包含未授权的商业歌曲。

### 5.3 路线对比

| 系统 | 范式 | 数据来源 | 可商用 | 时长上限 |
| --- | --- | --- | --- | --- |
| Jukebox | VQ-VAE + 稀疏 Transformer | 爬虫 | 否 | 4 min |
| MusicLM | 多阶段 Transformer | 私有 | 否（受限访问） | 5 min |
| MusicGen | 单阶段 Transformer + 交错 | 授权 | 是（CC-BY-NC） | 30 s |
| Stable Audio | 潜在扩散 | 自有+授权 | 是 | 90 s |
| Suno / Udio | 混合闭源 | 大量 | 是（订阅） | 4 min |

## 六、HuggingFace MusicGen 推理示例

下面展示如何用 `transformers` 一行代码加载 MusicGen 并生成 30 秒的 lo-fi 钢琴曲：

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile
import torch

# 1. 加载预训练模型（3.3B 参数量约 10 GB 显存）
processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")

# 2. 文本条件 → token
prompt = "lo-fi study beat with mellow piano, soft drums, 90 BPM"
inputs = processor(text=[prompt], padding=True, return_tensors="pt")

# 3. 自回归生成 audio tokens
#    MusicGen 在 EnCodec 空间生成 4 个码本 × 1500 步（30 s × 50 Hz）
audio_tokens = model.generate(
    **inputs,
    max_new_tokens=1500,          # 50 Hz × 30 s
    do_sample=True,
    temperature=1.0,
    top_k=250,
    guidance_scale=3.0,           # classifier-free guidance
)

# 4. EnCodec 解码：tokens → 32 kHz 波形
sampling_rate = model.config.audio_encoder.sampling_rate  # 32000
waveform = model.audio_encoder.decode(audio_tokens)

# 5. 保存为 wav
wavfile.write("lofi.wav", sampling_rate, waveform[0, 0].cpu().numpy())
```

如果想加入**旋律条件**（让生成的音乐参考某段已有旋律）：

```python
import torchaudio

# 加载参考音频并提取 Chroma 特征
ref_audio, sr = torchaudio.load("reference_melody.wav")
if sr != 32000:
    ref_audio = torchaudio.functional.resample(ref_audio, sr, 32000)

inputs = processor(
    audio=ref_audio.squeeze().numpy(),
    sampling_rate=32000,
    text=["acoustic guitar fingerstyle"],
    padding=True,
    return_tensors="pt",
)
audio_tokens = model.generate(**inputs, max_new_tokens=1500, guidance_scale=3.5)
```

工程上几个关键超参：

- **guidance_scale**：classifier-free guidance 强度，3.0-4.0 是常用区间，过大会过拟合提示词；
- **top_k**：截断采样温度，避免稀有码本被采样；
- **max_new_tokens**：每秒钟 50 Hz × $K$ 码本交错；30 秒对应 1500 步。

## 七、版权、数据与可控性：三条路线的权衡

音乐生成面临比语音 TTS 更复杂的伦理与法律问题：

- **版权**：训练数据是否合法？Suno / Udio 被多家唱片公司起诉；
- **艺人模仿**：能否生成"模仿 Taylor Swift 嗓音"的歌曲？欧盟 AI Act 与中国《生成式人工智能服务管理暂行办法》都明确限制；
- **训练数据透明**：Jukebox、MusicLM 都不公开数据来源，MusicGen 是少有的"白盒"代表。

可控性上，MusicGen 的**旋律条件 + 文本条件**双输入是当前最实用的折中；Jukebox 用"艺术家风格 + 歌词"做控制但精度差；MusicLM 只能纯文本控制。下一代系统（2025+）正在探索**结构化提示词**（如"前奏-主歌-副歌"的乐段标签）和**MIDI 条件**（让用户指定和声与节奏）。

## 小结

| 系统 | 核心创新 | 数据合规 | 时长 | 可控维度 |
| --- | --- | --- | --- | --- |
| Jukebox | 分层 VQ-VAE + 稀疏注意力 | 爬虫 | 4 min | 风格 + 歌词 |
| MusicLM | MuLan + 多阶段 | 私有 | 5 min | 文本 |
| MusicGen | 码本交错 + T5 条件 | 授权 | 30 s | 文本 + 旋律 |
| Stable Audio | 潜在扩散 + CLAP | 授权 | 90 s | 文本 + 时长 |
| Suno / Udio | 闭源混合 | 部分 | 4 min | 文本 + 歌词 |

音乐生成已经走出实验室，但**长序列 × 高质量 × 合法数据 × 精细控制**的四元约束仍未被完美解决。下一步是"用 LLM 的方式生成音乐"——把音频 token 接入大语言模型，让音乐生成继承 LLM 的推理能力。这要求音频世界也有自己的 **Tokenizer**，下一篇将讨论 EnCodec 与 SoundStream 这一基础组件。
