# Whisper 与多任务统一 ASR

传统 ASR 系统需要分别训练声学模型、发音字典、语言模型；端到端系统把多模块压成一个网络，但仍要针对特定语言、特定数据集微调。**Whisper（Radford et al., 2022, OpenAI）** 走了一条反路：直接训练一个**多任务、多语言的 seq2seq Transformer**，输入 30 秒 Log-Mel，输出**带特殊 token 的文本序列**——同一个模型既能识别英语，也能识别中文，还能翻译，还能吐时间戳。这种"以大为美"的统一范式，背后是 **68 万小时弱监督音频**的工程奇迹，也是 2023 年以后 LLM 与 ASR 融合的引子。

## 一、Whisper 的核心思想：把 ASR 看作 seq2seq 多任务

### 1.1 任务统一化

Whisper 的设计哲学：**不要为每个任务设计不同模型，而是把任务编码进 decoder 的输出 token 序列**。具体而言，每条训练样本都被打上一个"任务标识符"：

- `<|transcribe|>`：把语音转写为同语言文字。
- `<|translate|>`：把语音翻译成英语。
- `<|en|>`、`<|zh|>`、`<|ja|>` 等：源语言标签。
- `<|notimestamps|>` / `<|0.0|>` ... `<|30.0|>`：时间戳模式或具体时间。

最终 decoder 的目标序列形如：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|>Hello world<|endoftext|>
```

这意味着 **同一个 Transformer 解码器对所有任务是同一套参数**，只靠 prompt 控制。

### 1.2 多任务目标的概率视角

设任务指令 $T$（由语言标签、转写/翻译标签、时间戳标签组成）。Whisper 最大化：

$$
\log P(Y \mid X, T) = \sum_{t=1}^{U} \log P(y_t \mid y_{<t}, X, T)
$$

训练时，每条样本的 $T$ 是 ground truth；推理时，**用户用 prompt 控制 $T$**——这就是 Whisper 所谓"zero-shot"的来源。

### 1.3 Log-Mel 输入

Whisper 不再用 Fbank，而是直接用 **80 维 Log-Mel 频谱**，窗口长 25 ms，步长 10 ms。30 秒音频被切成 3000 帧（最长序列），强制不够的补零、过长的截断。Log-Mel 比 Fbank 多了一步 log 压缩，更接近人耳感知响度：

$$
\text{Log-Mel}_t = \log\left( \sum_{k} |X_T[k]|^2 \, W_k^{(\text{mel})} + \epsilon \right)
$$

## 二、弱监督大规模数据：68 万小时的暴力美学

### 2.1 数据来源

Whisper 训练数据来自：

- **Common Voice**：Mozilla 的众包多语言朗读语料。
- **LibriSpeech / LibriLight**：英文有声书。
- **TED-LIUM、CoVoST、FLEURS** 等公开数据集。
- **YouTube 字幕 + 视频自动转写**——这是 68 万小时中**最大的一部分**，也是"弱监督"的来源。

### 2.2 弱监督的代价

YouTube 自动字幕噪声巨大：标点缺失、错字、漏字、混语言、说话人重叠。Whisper 团队用了一组**启发式过滤器**清洗：

1. 字幕语言必须与音频语言一致（用 langdetect）。
2. 字幕长度与音频时长比例必须合理（避免"30 秒视频配 1 分钟字幕"）。
3. 字幕中混杂标签、URL、表情符号的样本剔除。

但即便是清洗后，仍然有不少噪声。**这是 Whisper 的根本 trade-off**：用数据规模换质量。结果是——Whisper 在学术集（AISHELL-1、CER）上未必比 Conformer 微调模型强，但在**真实场景噪声、口音、多语言**上泛化惊人。

### 2.3 数据规模 vs 模型规模的 Scaling

Whisper 论文给出了清晰的 scaling 曲线：

- Whisper tiny：39M 参数，680k 小时训练。
- Whisper small：244M。
- Whisper medium：769M。
- Whisper large-v3：1550M。

WER 随规模下降；**多语言 zero-shot 性能也在 scaling**。这印证了"more data + more compute + more tasks"的统一范式潜力。

## 三、Transformer 编码-解码架构

### 3.1 Encoder：只关注声学

Whisper 的 encoder 是一个标准的 Transformer encoder：$\log\text{-Mel} \in \mathbb{R}^{3000 \times 80}$ 经过 $2\times$ 卷积下采样（stride=2）后变成 1500 帧，再过 32 层 Transformer block。每层包含 Multi-Head Self-Attention + FFN + LayerNorm。**没有相对位置编码**——固定 1500 帧位置，用正弦位置编码嵌入。

### 3.2 Decoder：cross-attend encoder 输出

Decoder 与 encoder 对称：每层包含 Masked Self-Attention（保证自回归）+ **Cross-Attention**（让 decoder 关注 encoder 输出）+ FFN。Cross-Attention 的 Q 来自 decoder，K/V 来自 encoder——这是 seq2seq 的标准结构。

### 3.3 简化 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperLoRA(nn.Module):
    """Whisper + LoRA 微调封装。仅在 decoder 的 cross-attn 注入 LoRA。"""

    def __init__(self, model_name: str = "openai/whisper-small", lora_r: int = 16):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.base = WhisperForConditionalGeneration.from_pretrained(model_name)

        # 冻结所有参数
        for p in self.base.parameters():
            p.requires_grad = False

        # 在 cross-attn 注入 LoRA（仅示意；实际需修改 attention 计算）
        self._inject_lora(lora_r)

    def _inject_lora(self, r: int):
        """伪代码：对每个 cross-attn 的 q_proj、v_proj 注入 LoRA 旁路。"""
        for layer in self.base.model.decoder.layers:
            attn = layer.encoder_attn
            in_dim, out_dim = attn.k_proj.in_features, attn.k_proj.out_features

            attn.lora_A = nn.Linear(in_dim, r, bias=False)
            attn.lora_B = nn.Linear(r, out_dim, bias=False)
            nn.init.kaiming_uniform_(attn.lora_A.weight, a=5 ** 0.5)
            nn.init.zeros_(attn.lora_B.weight)
            attn.lora_A.weight.requires_grad = True
            attn.lora_B.weight.requires_grad = True

    def forward(self, audio, text):
        """audio: (B, T) 波形；text: (B, U) 目标 id。"""
        # 1) 特征抽取：log-mel
        feats = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.base.device)

        # 2) tokenizer
        labels = self.processor.tokenizer(
            text, return_tensors="pt", padding=True
        ).input_ids.to(self.base.device)

        out = self.base(input_features=feats, labels=labels)
        return out.loss

    @torch.no_grad()
    def transcribe(self, audio_paths, language: str = "zh", task: str = "transcribe"):
        """推理：调用 generate() 接口。"""
        from transformers import pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.base,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=self.base.device.index,
        )
        return pipe(
            audio_paths,
            generate_kwargs={"language": language, "task": task},
            return_timestamps=True,
        )
```

工程要点：

- **`return_timestamps=True`** 让 Whisper 输出 `<|0.0|>...<|3.0|>` 时间戳——这需要模型支持时间戳模式（large 模型支持），是字幕生成的核心。
- **LoRA 仅注入 cross-attn**：冻结 encoder、decoder 的 self-attn/FFN，只微调 decoder 的 cross-attn。这能让模型学到"如何把中文/方言的声学模式映射到对应文字"，同时保留 Whisper 原始的多语言对齐能力。

## 四、与 LLM 融合的新趋势

### 4.1 为什么 ASR 需要 LLM

Whisper 在**真实场景**的鲁棒性来自数据规模，但**语言建模**仍弱：拼写错误、漏字、长尾词、口头禅处理不佳。直接外挂 LLM（如 GPT-4）做纠错——效果好但延迟高、隐私差。**LLM-based ASR** 的新范式是：**把 LLM 作为 ASR 解码器**。

### 4.2 代表工作

- **USM（Google, 2023）**：用 12M 小时多语言数据预训练 2B 参数 Conformer encoder，再外挂 100B 参数 PaLM-2 做解码与文本规范化。
- **SALMONN（Yu et al., 2024, Tsinghua）**：双 encoder（Whisper + BEATs 音频事件） + Vicuna LLM，**一个模型既能 ASR 又能音频问答**。
- **Qwen2-Audio（Alibaba, 2024）**：Qwen2 LLM 接收 Whisper 编码器输出，支持语音 + 音频 + 文字混合输入。
- **GRASS（2024）**：纯 LLM-based ASR，从零训练一个 LLM 直接消费 log-mel，挑战 Conformer 的传统地位。

### 4.3 共同架构：Encoder + LoRA + LLM

主流 LLM-based ASR 流水线：

```
Log-Mel → Whisper Encoder → Adapter (LoRA) → LLM Prompt → LLM → 文字
```

**Adapter 把声学特征投影到 LLM 的 embedding 空间**，让 LLM 直接做语言推理。这一架构有两个**根本优势**：（1）LLM 已有强大的拼写、句法、世界知识；（2）多任务统一——加标点、改语段、抽取实体、问答都能在同一 LLM 上完成。

## 五、Whisper 推理 + LoRA 微调示例

```python
# 推理
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device="cuda:0",
)
result = asr(
    "audio.wav",
    generate_kwargs={"language": "chinese", "task": "transcribe"},
    return_timestamps=True,
)
print(result["text"], result["chunks"])
```

```python
# LoRA 微调：使用 peft + transformers
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

lora_cfg = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],  # 同时注入 encoder 和 decoder
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()  # 应该 < 5% 总参数

# 然后按常规 HF Trainer 训练
```

工程要点：

- **target_modules**：建议同时覆盖 encoder 和 decoder 的 `q_proj`、`v_proj`。仅注入 decoder cross-attn 效果有限。
- **学习率**：建议 `2e-4` 到 `5e-4`（LoRA 通常比全量微调学习率高一个量级）。
- **推理合并**：训练完后用 `model.merge_and_unload()` 把 LoRA 权重合并回基础模型，避免推理时框架不支持 LoRA。

## 六、Whisper 的优势与局限

### 6.1 优势

- **零样本鲁棒**：跨语言、跨口音、跨噪声几乎不需要微调。
- **多任务统一**：转写、翻译、时间戳、说话人分段在同一模型。
- **生态成熟**：HuggingFace / faster-whisper / Whisper.cpp / Whisper streaming 都有完善支持。

### 6.2 局限

- **幻觉（hallucination）**：在静音、噪音、音乐片段，Whisper 可能输出完全无关文字。这在会议记录、安防场景是严重缺陷。
- **长尾词**：罕见人名、专业术语仍识别差，需要热词偏置或 LLM 后处理。
- **延迟**：30 秒一段的窗口设计决定了最低延迟 ~1 秒级；流式 Whisper 需要做 chunk 切片。
- **标点符号**：Whisper 自带的标点质量一般，需要专门的标点模型（如 PunctuationModel）后处理。
- **无显式语言模型接口**：相比 CTC 可以 shallow fusion KenLM，Whisper 的"语言模型"是烧死在参数里的，难以做领域自适应。

## 小结

| 维度 | 传统 ASR | Whisper | LLM-based ASR |
| --- | --- | --- | --- |
| 数据规模 | 千小时级 | 68 万小时 | 数百万小时 |
| 多任务 | 各系统独立 | 单一 seq2seq | Encoder + LLM 联合 |
| 流式 | 强 | 弱（30 s 窗口） | 取决于 LLM |
| 长尾词 | 热词表 | 弱 | LLM 后处理可改善 |
| 部署 | 中等 | 高（whisper.cpp） | 高（需要 LLM） |

Whisper 把"暴力美学"推到了极致——**用规模换泛化、用多任务换通用**。但其幻觉、延迟、长尾词等问题，又反向推动了"LLM + ASR Encoder"的下一代架构——USM、SALMONN、Qwen2-Audio 让 ASR 不再是孤立的语音识别模块，而是 LLM 的"耳朵"。可以预见，**未来 2-3 年的 ASR 主线是 LLM-native 化**：声学编码器退化为 LLM 的输入适配器，文字生成完全交给 LLM。这对工程、对研究都是新范式——下一篇我们将讨论 ASR 系统如何**评估、解码、部署**，让研究原型变成生产级服务。