# 大模型时代 TTS：从 VALL-E 到 SpeechGPT

2023 年起，大语言模型（LLM）的范式开始席卷语音合成。核心思路是：**把语音先压缩成一串离散 token（discrete speech tokens），再让一个自回归语言模型直接生成这些 token**。这条路线把 TTS 变成了一个**条件语言建模问题**——给定文本 / 说话人 prompt，预测下一段语音 token。它带来了三大变化：（1）TTS 与语音识别（ASR）、语音翻译、对话第一次共用同一个模型架构；（2）zero-shot 语音克隆能力大幅提升；（3）大模型的多语言、上下文、指令遵循能力可直接迁移到语音。本篇梳理 SoundStream/EnCodec 的离散化基础、VALL-E 的条件语言建模、跨语言 VALL-E X，以及 SpeechGPT / Qwen2-Audio 等语音 LLM 一体化模型。

## 一、为什么需要离散语音 Token？

LLM 之所以强大，是因为它能在统一的离散 token 空间上做自回归。文本有现成的 token 化方案（SentencePiece、BPE），但语音是连续波形，要喂给 LLM 必须先离散化。理想中的语音 tokenizer 应该满足：

1. **重建质量高**：从 token 还原波形时不能丢失关键信息。
2. **压缩率高**：1 秒音频应该对应较少的 token（否则序列太长、训练成本爆炸）。
3. **语义可分**：token 序列应该与音素、韵律有一定对应关系，方便 LLM 学到语音规律。

## 二、神经编解码器：SoundStream 与 EnCodec

### 2.1 SoundStream

2021 年 Google 提出的 **SoundStream**（Zeghidour et al., 2021）是首个端到端的语音/音频编解码器。它由**编码器、量化器、解码器**三部分组成，用重建损失 + 对抗损失联合训练。关键创新是 **RVQ（Residual Vector Quantization）**：把编码器的连续表示 $\mathbf{z}$ 用 $N$ 层量化器逐层细化：

$$
\hat{\mathbf{z}} = \sum_{i=1}^{N} \mathbf{e}_{i, k_i}, \quad k_i = \arg\min_j \| \mathbf{r}_{i-1} - \mathbf{e}_{i, j} \|_2
$$

其中 $\mathbf{r}_i = \hat{\mathbf{z}} - \sum_{j \le i} \mathbf{e}_{j, k_j}$ 是残差。RVQ 可以在码本大小固定的情况下**灵活调节码率**——少用几层就得到低码率粗表示，多用几层就得到高码率细表示。

### 2.2 EnCodec

Meta 的 **EnCodec**（Défossez et al., 2022）思路类似，但在判别器上做了更多工程优化：除了 MPD + MSD，还加了**波形距离判别器 + 梅尔距离判别器**，并使用多尺度 STFT 损失（多组窗长 / hop）。EnCodec 默认输出 24 kHz 音频、码率 6 kbps，1 秒音频压成约 75 个离散 token。

下面是用 EnCodec / SoundStream 做语音编解码的简化 PyTorch 示例（基于 `encodec` 官方 API 风格）：

```python
import torch
from encodec import EncodecModel
from encodec.utils import audio_to_torch


# 加载 24 kHz 单声道预训练模型（facebook/encodec_24khz）
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)   # 6 kbps → 75 tokens/sec

# 假设 audio: (B, 1, T) float tensor, 范围 [-1, 1]
encoded = model.encode(audio)
# encoded.codes: list[Tensor]，长度 = n_q (RVQ 层数)，
#   每个 shape = (B, 1, n_frames)
codes = torch.cat(encoded.codes, dim=0)        # (n_q, B, 1, n_frames)
flat_codes = codes.permute(1, 2, 0, 3).reshape(codes.size(1), 1, -1)
# flat_codes: (B, 1, n_q * n_frames) 展平所有 RVQ 层的 token

decoded = model.decode(encoded)
waveform = decoded.audio.squeeze(0)            # (1, T)
print(f"recovered {waveform.shape[-1]} samples from {flat_codes.size(-1)} tokens")
```

类似的，**SoundStream** 的推理流程也是先 `encode → RVQ → discrete codes`，再 `codes → RVQ → decode`。

### 2.3 语义 token 与声学 token

RVQ 给的是**声学 token**（acoustic tokens）——它们能高保真重建波形，但和音素没有直接对应。学界还提出过**语义 token**（semantic tokens），如 HuBERT、wav2vec 2.0 的聚类 ID。它们语义信息密集但重建质量差。因此当前主流 LLM-TTS 通常采用**两路 token 策略**：

- **语义路**：HuBERT 风格聚类 token，提供音素级别的内容信息。
- **声学路**：SoundStream / EnCodec 风格 RVQ token，提供音色、韵律的细节。

如 **VALL-E** 用的"SoundStorm + Encodec"组合，就属于声学 token 路线。

## 三、VALL-E：把 TTS 当作条件语言建模

2023 年 1 月 Microsoft 提出的 **VALL-E**（Wang et al., 2023）首次把 TTS 重新定义为"**神经编解码语言模型**"。它的核心公式与 GPT 自回归训练一致：

$$
p(\mathbf{y} \mid \mathbf{x}, \mathbf{p}) = \prod_{t=1}^{T} p(y_t \mid y_{<t}, \mathbf{x}, \mathbf{p})
$$

其中 $\mathbf{x}$ 是文本 token 序列，$\mathbf{p}$ 是说话人 prompt 的语音 token，$\mathbf{y}$ 是目标语音的离散 token。

### 3.1 训练阶段

把训练数据里的所有语音都用 EnCodec 编成 token，然后让语言模型学习：

- 给定一段文本 + 同一说话人 3-10 秒的 prompt token，预测该说话人说出该文本时的全部语音 token。

这就把 TTS 变成了**多任务条件语言建模**：文本条件 + 说话人条件 + 时间序列建模。

### 3.2 推理阶段（zero-shot 克隆）

推理时，给定一个从未见过的说话人 3 秒录音 + 目标文本，模型可以直接生成该说话人说出目标文本的语音 token，再交给 EnCodec 解码器恢复波形。**关键点是不需要 fine-tuning**——只要 prompt 足够短（3-10 秒），模型在 zero-shot 设置下就能克隆出"像不像"的音色。

### 3.3 局限

VALL-E 原版有几个明显问题：

- 训练数据未公开，仅有论文 demo，论文结论难以复现。
- 对长 prompt 和情感多样性鲁棒性差。
- 解码阶段需要 EnCodec 模型，可能引入声学失真。

但它打开了"用 LM 解决 TTS"的范式，后续的 SPEAR-TTS、Bark、Suno-Bark、SoundStorm 都在此基础上演化。

## 四、VALL-E X：跨语言语音克隆与翻译

VALL-E X 在 VALL-E 基础上扩展到**跨语言场景**：

- 同一说话人讲英语和中文两种语言，把源语言语音编码成内容 token + 说话人 embedding。
- 用翻译模型把内容 token 翻成目标语言，再让解码器生成目标语言的语音。

$$
p(\mathbf{y}_{\text{tgt}} \mid \mathbf{x}_{\text{tgt}}, \mathbf{p}_{\text{src}}, \mathbf{y}_{\text{src}}) = \prod_{t} p(y_{\text{tgt},t} \mid y_{\text{tgt},<t}, \mathbf{x}_{\text{tgt}}, \mathbf{f}(\mathbf{p}_{\text{src}}, \mathbf{y}_{\text{src}}))
$$

$\mathbf{f}$ 是说话人 / 内容编码器，$\mathbf{x}_{\text{tgt}}$ 是翻译后的目标文本。结果是：**保留源说话人音色 + 说出目标语言内容**，实现"用你的声音说外语"。

## 五、语音 LLM 一体化模型

把 TTS 推进到和 GPT 同等地位的真正突破，是**语音 LLM 一体化**——一个模型同时支持语音识别、语音合成、语音翻译、对话。

### 5.1 SpeechGPT

2023 年的 **SpeechGPT**（Zhang et al., 2023）最早做了"语音 token + LLM"的尝试：

- 把语音先用 HuBERT 离散化为单元序列，再用 SoundStream 编码细节。
- 把"语音单元 → 文本"和"文本 → 语音单元"都建模为 LLM 的下一 token 预测任务。
- 通过统一的 prompt 模板，让同一个模型既能 ASR 又能 TTS 还能对话。

### 5.2 Qwen2-Audio

阿里 **Qwen2-Audio**（Chu et al., 2024）走的是"**统一 ASR + TTS + 音频理解**"路线：

- 输入侧支持任意音频（语音、音乐、环境声）+ 文本指令。
- 输出侧既可以输出文本，也可以输出语音 token。
- 用一个 Qwen2 LLM 作为统一 backbone，语音编码器 + 语音解码器分别挂在输入端和输出端。

$$
p(\mathbf{o} \mid \mathbf{x}_{\text{text}}, \mathbf{x}_{\text{audio}}) = \prod_{t=1}^{T} p(o_t \mid o_{<t}, \mathbf{x}_{\text{text}}, \mathbf{x}_{\text{audio}})
$$

其中 $\mathbf{o}$ 可以是文本 token 也可以是语音 token。

### 5.3 Step-Audio

阶跃星辰的 **Step-Audio** 是国内首个工业级语音 LLM，把 TTS、ASR、语音翻译、语音对话全部塞进一个 130B 参数的端到端模型，支持中英文混合、情感控制、低延迟实时对话。

### 5.4 GLM-4-Voice / Mini-Omni / Moshi

这一波"全双工语音 LLM"的共同特点是：

- **流式（streaming）**：边听边说，无需等待完整输入。
- **全双工（full-duplex）**：能同时听和说，自然地打断、被打断。
- **多模态输出**：语音 token 直接驱动神经声码器，绕过传统 TTS 流水线。

**Moshi**（Kyutai Labs, 2024）甚至把"语义 token + 声学 token"双流建模，让模型可以一边生成语音一边生成对应的文本（用于字幕和监控）。

## 六、基于 LLM 的 TTS 推理与解码

LLM-TTS 的推理有两个关键工程问题：

### 6.1 解码策略

朴素的自回归解码（top-k / nucleus sampling）会让合成语音出现"杂音、卡顿、重复"。实践中的优化：

- **重复惩罚（repetition penalty）**：降低已经生成过的 token 的采样概率。
- **说话人一致性约束**：在采样时用分类器引导，让生成的 token 更像 prompt 的说话人。
- **lookahead 解码 / speculative decoding**：用一个小模型草拟几个 token，再用大模型批量验证，加速 2-3 倍。

### 6.2 声码器后处理

LLM 输出的离散 token 必须再经过声码器重建波形。两种主流选择：

- **EnCodec / SoundStream**（Meta / Google）：RVQ token → 波形。
- **SoundStorm**（Google, 2023）：基于 MaskGIT 的并行解码，比自回归快 100x，但训练复杂度高。

$$
\text{EnCodec decode}: \mathbf{y}_{\text{wave}} = \text{Decoder}\Big( \sum_{i=1}^{N} \text{Embed}_i(\text{RVQ}_i(\text{tokens})) \Big)
$$

## 七、SoundStream / EnCodec 使用示例

下面是一个完整的端到端示例：先把一段音频压成 token → 用一个简单的 LSTM 语言模型做"无意义预测" → 再解码回波形，演示 LLM-TTS 的最小骨架：

```python
import torch
import torch.nn as nn
from encodec import EncodecModel


class ToyLM(nn.Module):
    """极简语音 LLM：给定 prompt tokens，自回归预测后续 tokens。"""

    def __init__(self, vocab_size: int, n_q: int, d_model: int = 256):
        super().__init__()
        # 把 (n_q, ) 的 RVQ 索引嵌入到 d_model
        self.emb = nn.Embedding(vocab_size * n_q, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size * n_q)

    def forward(self, tokens):
        # tokens: (B, T, n_q)  RVQ 索引
        B, T, n_q = tokens.shape
        flat = tokens.reshape(B, T * n_q)         # 把 n_q 维展平到序列维度
        h = self.emb(flat)
        h, _ = self.lstm(h)
        return self.head(h)[:, -(n_q):]            # 只预测最后一帧的 n_q 个 token


# 1. 加载 codec
codec = EncodecModel.encodec_model_24khz()
codec.set_target_bandwidth(6.0)

# 2. 编码 prompt（3 秒）和 ground-truth 语音
prompt_audio = torch.randn(1, 1, 24000 * 3)
target_audio = torch.randn(1, 1, 24000 * 5)
prompt_codes = torch.cat(codec.encode(prompt_audio).codes, dim=0).permute(1, 2, 0)
target_codes = torch.cat(codec.encode(target_audio).codes, dim=0).permute(1, 2, 0)

# 3. 训练 ToyLM 预测 target_codes（接在 prompt 后）
lm = ToyLM(vocab_size=codec.quantizer.vq.layers[0].codebook_size, n_q=prompt_codes.size(-1))
# ... 训练循环省略 ...

# 4. 推理：生成 token → codec 解码
with torch.no_grad():
    gen = lm.generate(torch.cat([prompt_codes, target_codes[:, :1]], dim=1))
    recovered = codec.decode([(gen.transpose(0, 1).unsqueeze(0), None)])[0]
```

这个例子非常粗糙，但展示了 LLM-TTS 的核心流程：**RVQ token 化 → 条件语言建模 → codec 解码**。工业系统只是把 LSTM 换成 Transformer / Mamba，把 toy 数据换成百万小时多说话人语料，再挂上说话人条件、情感条件、上下文条件。

## 八、未来趋势：低延迟、全双工、个性化、情感可控

LLM-TTS 的下一步演化集中在四个方向：

1. **低延迟流式合成**：把第一段音频的延迟从 300 ms 降到 100 ms 以下，让对话体验逼近真人。
2. **全双工对话**：模型同时听和说，处理打断、重叠、轮次切换（如 Moshi、GPT-4o Voice Mode）。
3. **个性化与情感控制**：在 prompt 中加入"用更兴奋的语气说"、"加入喘气"、"降低语速"等指令，实现细粒度韵律调节。
4. **多模态融合**：与视觉、文本、记忆系统结合，让 TTS 输出能匹配虚拟人面部表情、上下文情感、长期偏好。

可以预见，未来 3-5 年的 TTS 系统将不再是独立的"文本→语音"模块，而是**一个统一的多模态大模型**的一个能力——文本理解、对话、记忆、语音生成全部在同一空间内联合优化。

## 小结

| 模型 / 技术 | 关键创新 | 应用场景 |
| --- | --- | --- |
| SoundStream / EnCodec | RVQ 神经编解码 | 语音离散化基础 |
| VALL-E | TTS 当作条件 LM | zero-shot 克隆 |
| VALL-E X | 跨语言 + 翻译 | 跨语言语音克隆 |
| SpeechGPT | 语音 + 文本统一 token | 多任务语音 LLM |
| Qwen2-Audio | 统一 ASR + TTS + 音频理解 | 多模态对话 |
| Step-Audio / Moshi | 全双工 + 流式 | 实时语音交互 |
| SoundStorm | 并行解码 | 加速 LLM-TTS |

LLM-TTS 把 TTS 从"声学模型 + 声码器"的传统范式推进到了"统一多模态生成"，未来这条路线将继续向更低延迟、更强可控、更深融合的方向演化。本系列从传统 TTS、神经 TTS、端到端 TTS 一路梳理到大模型 TTS，希望能让读者看清这条 30 年技术演化的核心脉络。
