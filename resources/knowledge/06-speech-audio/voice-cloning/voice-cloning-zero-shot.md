# 零样本声音克隆：YourTTS 到 VALL-E X

零样本声音克隆（Zero-Shot Voice Cloning）的工程目标是：**给定一段从未在训练集中出现过的目标说话人参考音频（3-30 秒），无需任何梯度更新，立刻合成该说话人的任意新文本**。它与传统微调范式最大的区别是"数据冷启动"——不需要为每个新说话人启动训练任务。实现这一目标需要在三个层面做范式转变：（1）**说话人条件从微调权重转为上下文编码**；（2）**声学表征从 mel 谱转向神经编解码 token**；（3）**生成模型从声学解码器转向自回归语言模型**。本文以 YourTTS（2021）→ VALL-E（2023）→ VALL-E X（2023）→ NaturalSpeech 3（2024）的演进为主线，梳理零样本声音克隆的核心方法与挑战。

## 一、零样本与少样本的本质区别

少样本克隆（上一讲）需要一个 Speaker Encoder 在大量说话人上学到的判别度量空间，推理时为每个目标说话人计算嵌入均值。零样本克隆更进一步：**Speaker Encoder 与声学模型是同一个端到端网络**，参考音频被 token 化后直接进入声学解码器。形式上：

$$
\hat{X}_{\text{tokens}} = \text{LM}_{\theta}\big( Y_{\text{tokens}}, \text{Tokenize}(X_{\text{ref}}) \big)
$$

解码出来的 token 可以是 mel 谱、神经编解码码本、甚至是离散声学特征。YourTTS 用前者做奠基，VALL-E 系列用后者做质变。

## 二、YourTTS：多说话人多语言零样本（Casanova et al., 2021）

### 2.1 模型结构

YourTTS（Cisco, Interspeech 2021）是第一个**真正零样本、且同时多说话人 + 多语言**的 TTS。其结构由 VITS（Kim et al., 2021）演化而来，核心是 VITS 的三件套：

- **Posterior Encoder**：把波形编码为 latent $z$。
- **Decoder（HiFi-GAN）**：从 $z$ 重建波形。
- **Normalizing Flow + Duration Predictor**：把 $z$ 与文本隐藏状态对齐。

YourTTS 在 VITS 之上引入两条新通路：

1. **Speaker Consistency Loss**：让合成的梅尔谱通过预训练的 Speaker Encoder（同样是 ECAPA-TDNN），强制其说话人嵌入与参考说话人嵌入的余弦相似度 > 阈值 $\tau$（通常 0.85）。
2. **Language Embedding**：用语言 one-hot 喂给 Decoder，让单一模型支持英语、葡萄牙语和法语。

$$
\mathcal{L} = \mathcal{L}_{\text{VITS}} + \lambda_{\text{spk}} \cdot \max(0, \tau - \cos(\mathbf{e}_{\text{syn}}, \mathbf{e}_{\text{ref}}))
$$

### 2.2 零样本能力来源

YourTTS 训练时**没有见过任何目标说话人的语音**，推理时只需一段参考音频做说话人条件注入。它证明了一件事：**只要 Speaker Encoder 在足够多说话人上学过，并被作为条件注入到一个足够强的声学模型上，零样本克隆就是"自然涌现"的能力**，而不需要专门设计。

## 三、VALL-E：神经编解码语言模型范式（Wang et al., 2023）

### 3.1 范式转变：从 mel 到 codec token

Microsoft 的 VALL-E（arXiv 2023）把声音克隆重新定义为**神经语音编解码语言建模问题**。核心思想是：

- 用预训练的神经编解码器（**EnCodec（Défossez et al., 2022）**）把语音切成 8 个码本（codebook）的离散 token 序列，每个码本包含 $\sim$100 个 entries。
- 把**参考语音的 codec token**当作"提示词"，把**目标音的 codec token** 当作"生成序列"，训练一个自回归语言模型（类似 GPT）按文本条件预测下一 token。

形式化：

$$
P(\hat{x} \mid X_{\text{ref}}, Y) = \prod_{t} P\big( \hat{x}_t \mid \hat{x}_{<t}, X_{\text{ref}}, Y; \theta \big)
$$

其中 $\hat{x}_t$ 是 8 个码本并行采样的复合 token。

### 3.2 两阶段条件化

VALL-E 把 codec token 分成**内容（Content）**和**声学（Acoustic）**两路：

- **Content token**：预测主要受**文本 phoneme** 控制，决定说什么。
- **Acoustic token**：预测主要受**参考音频 prompt** 控制，决定音色与韵律。

训练时用提示学习（Prompt-based Pre-training）：每段语音的前缀 $\mathbf{X}_\text{ref}$ 作为 prompt，后续目标序列 $\hat{x}$ 作为生成。推理时给定任意参考音频 $\hat X_\text{ref}$，模型按相同模式续写。

### 3.3 训练数据规模

VALL-E 用了 **LibriLight 60K 小时**英语语音做预训练。这比 YourTTS 的几百小时多两个数量级——它是"大模型 + 大数据"在 TTS 领域的首次亮相。结果显示 VALL-E 在说话人相似度（SVS）上接近真人录音，自然度 MOS 达到 4.21，但**仅用 3 秒参考音频**，是真正的零样本。

## 四、VALL-E X：跨语言克隆与翻译保留（Chen et al., 2023）

VALL-E X 把 VALL-E 扩展到**跨语言声音克隆**：参考音频是英语、合成语言是中文（或反之）。关键设计：

1. **双语 phoneme 词典**：训练时中英语料混在同一 batch，phoneme ID 用 BPE 编码。
2. **语言无关的说话人 prompt**：让 codec token 中的说话人信息**不被语言切换带走**。实现方法是引入**语种判别器（Language ID Classifier）作为对抗损失**，迫使说话人嵌入与语种正交。
3. **韵律保留**：跨语言克隆最怕"音色对但口音怪"，VALL-E X 引入**韵律蒸馏损失**，从参考音频的 prosody embedding（用 FastSpeech2 的 variance adaptor 编码）继承语调模板。

$$
\mathcal{L}_{\text{prosody}} = \left\| f_{\text{prosody}}(\hat{X}) - f_{\text{prosody}}(X_{\text{ref}}) \right\|_2^2
$$

实验上 VALL-E X 在从未在 FID 中配对的英→中任务上达到 SVS 0.89、CER 4.3%，明显优于简单拼接英语 TTS 和中文 TTS 的两阶段方案。

## 五、NaturalSpeech 3：因子化分解 + 全自回归（Le et al., 2024）

### 5.1 动机

VALL-E 用 codec token 的好处是"无需声码器"，但代价是 codec token **纠缠了内容、音色、韵律、情感**——参考音频一旦含笑声或气口，模型就会被噪声带跑。NaturalSpeech 3 提出**因子化神经编解码（Factorized Neural Codec, FACodec）**，把 codec token 分解成四个独立子空间：

| 子空间 | 维度 | 控制因素 |
|---------|------|----------|
| Content | 8192 | 文本 phoneme |
| Timbre | 1024 | 说话人身份 |
| Prosody | 1024 | 韵律/语速 |
| Acoustic Detail | 1024 | 噪声/混响 |

每个子空间用独立的 VQ-VAE 训练得到，四个子空间的 token 拼接后输入一个**全自回归语言模型**。推理时只需替换 Timbre 子空间为目标说话人，其他子空间可独立调节。

### 5.2 全自回归 + 非自回归混合

NaturalSpeech 3 还提出 **Full-Autoregressive（FAR）/ Non-Autoregressive（NAR）** 双模推理：FAR 模式按 token 逐个采样，质量更高；NAR 模式用 mask-predict 一次性解码，延迟更低（首个音频块延迟 < 200 ms）。

## 六、关键挑战：韵律迁移、情感、长文本一致性

### 6.1 韵律迁移（Prosody Transfer）

参考音频只有 3 秒，但文本可能长达 30 秒。模型必须把参考的"短韵律"外推到"长韵律"。YourTTS 用 duration predictor + VITS 后端实现弱韵律外推；VALL-E 系列靠语言模型的 in-context learning 涌现这一能力。NaturalSpeech 3 通过**显式 prosody 子空间**让外推可控。

### 6.2 情感保留（Emotion Preservation）

零样本克隆在情感上常常"中性化"——模型倾向于合成中性情感，因为训练数据的均值。VITS-emotional 和 NaturalSpeech 3 通过**情感 token**显式控制。推理时若不指定情感，模型就回落到中性。

### 6.3 长文本一致性：长音频解码漂移

VALL-E 类模型在合成 > 30 秒音频时容易"音色漂移"——前后音色不统一。原因是 codec token 序列长，自回归 attention 难以保持全局一致性。常用缓解：

1. **重置 prompt**：每隔 30 秒用参考音频重新注入说话人条件。
2. **滑动窗口 + 平滑拼接**：合成 30 秒一段，相邻段用 overlap-add 拼接。
3. **CFG（Classifier-Free Guidance）**：训练时 10% 概率丢弃说话人条件，推理时同时做有条件 / 无条件生成，按 $\hat{x} = (1+\gamma) x_\text{cond} - \gamma x_\text{uncond}$ 强化条件。

## 七、VALL-E 风格推理代码示例

```python
import torch
import torch.nn.functional as F
from encodec import EncodecModel
from transformers import AutoModelForCausalLM


class VALLEStyleInference:
    """VALL-E 风格的零样本声音克隆推理。"""

    def __init__(self, lm_path: str, codec_path: str = "facebook/encodec_24khz"):
        # 加载神经编解码器（24 kHz, 8 个码本）
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_sample_rate(24_000)
        # 加载 codec 语言模型（自回归 transformer）
        self.lm = AutoModelForCausalLM.from_pretrained(lm_path).cuda().eval()
        # 文本 tokenizer + phoneme 编码
        from text2phoneme import G2P
        self.g2p = G2P()

    @torch.no_grad()
    def encode_codec(self, wav_path: str) -> torch.Tensor:
        """把参考音频切成 8 个码本的 token 序列。"""
        wav = self.codec.preprocess(wav_path).cuda()
        codes = self.codec.encode(wav)  # (B, K, T)  K=8 个码本
        return codes.squeeze(0).long()  # (K, T)

    @torch.no_grad()
    def synthesize(self, text: str, ref_wav: str, max_len: int = 1000, cfg_scale: float = 1.5):
        """主推理：text + 参考音频 -> 目标 codec token -> 波形。"""
        # 1. phoneme 化
        phonemes = self.g2p(text)
        # 2. 编码参考音频
        ref_codes = self.encode_codec(ref_wav)        # (K_ref, T_ref)
        # 3. 构造 prompt：phoneme + ref_codes
        prompt = torch.cat([phonemes, ref_codes.T.flatten()], dim=-1)  # 展平 K 个码本
        # 4. 自回归生成目标 codec token（带说话人 CFG）
        gen = self.lm.generate(
            input_ids=prompt[None, :],
            max_new_tokens=max_len,
            do_sample=True, top_k=20, temperature=0.8,
            # CFG 简化做法：把 CFG scale 作为 logit 调整
        )
        target_codes = gen.squeeze(0)[len(prompt):].reshape(-1, 8).T  # (K, T_target)
        # 5. 编解码器解码为波形
        wav = self.codec.decode([(target_codes, None)])
        return wav.squeeze().cpu().numpy()


# 烟测：合成一段 10 秒的中文克隆
if __name__ == "__main__":
    infer = VALLEStyleInference(lm_path="path/to/vall_e_lm")
    wav = infer.synthesize(
        text="今天天气真不错，我们去公园散步吧。",
        ref_wav="reference_3sec.wav",
    )
    import soundfile as sf
    sf.write("cloned.wav", wav, samplerate=24000)
```

## 八、零样本 vs 微调的工程取舍

| 维度 | 零样本（YourTTS / VALL-E） | 微调（COSYVOICE / VITS） |
|------|-----------------------------|---------------------------|
| 数据需求 | 仅 3-30 秒参考音频 | 30 分钟-数小时 |
| 推理延迟 | 100-300 ms 首个块 | 实时（声码器瓶颈） |
| 显存占用 | 单卡 A100 可批量 | 微调时 24GB+ |
| 自然度 MOS | 4.0-4.4 | 4.4-4.7 |
| 相似度 SVS | 0.85-0.93 | 0.93-0.97 |
| 可控性 | 中（受参考音频主导） | 高（可调韵律 / 情感） |
| 部署成本 | 一份模型 + 任意参考 | 每说话人一份权重 |

选择规则：**大规模、个性化、可冷启动**选零样本；**专业虚拟主播、对音色还原极致**选微调。

## 小结

| 方法 | 范式 | 关键创新 | 数据规模 |
|------|------|----------|----------|
| YourTTS | VITS + Speaker Encoder | 多语言 + 说话人一致性损失 | 几百小时 |
| VALL-E | Codec LM（GPT 风格） | 神经编解码语言模型 | LibriLight 60K 小时 |
| VALL-E X | 跨语言 Codec LM | 双语 phoneme + 韵律蒸馏 | 双语混合 |
| NaturalSpeech 3 | 因子化 Codec FAR/NAR | 四子空间解耦 | 10K+ 小时 |
| 工程核心 | Codec token 取代 mel 谱 | 让音色 / 内容 / 韵律可独立调控 | 100+ 小时 |

零样本声音克隆用"神经编解码 + 语言模型"重新定义了 TTS 的能力边界，下一步的微调策略将在下一篇讨论。