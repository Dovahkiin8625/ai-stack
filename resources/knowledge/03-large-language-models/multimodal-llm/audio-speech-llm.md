# 音频/语音大模型：Whisper、SALMONN、Qwen2-Audio

文本和图像之外，**音频**是第三种被广泛部署的多模态信号。它涵盖自动语音识别（ASR）、语音合成（TTS）、音频问答、声音事件检测等任务。本文从 Whisper 出发，沿 SALMONN → Qwen2-Audio 梳理语音大模型的主流路线，并给出 HuggingFace Transformers 上跑 Whisper 的最小代码。

## 一、音频任务谱系

音频是一个总称，按任务可分为：

| 任务 | 输入 | 输出 | 代表模型 |
| --- | --- | --- | --- |
| **ASR** | 语音波形 | 文本 | Whisper、wav2vec 2.0、Paraformer |
| **AST** | 音频波形 | 文本（描述、分类标签） | CLAP、AudioLDM |
| **TTS** | 文本 | 语音波形 | VALL-E、Bark、Suno |
| **Speech-LLM** | 语音 + 文本 prompt | 文本回答 | SALMONN、Qwen2-Audio |
| **Voice Cloning** | 参考音频 + 文本 | 与参考音色一致的语音 | VALL-E 2、NaturalSpeech 3 |

下面聚焦 ASR 与 Speech-LLM 两个最活跃的方向。

## 二、ASR 的演进：从 wav2vec 到 Whisper

### 2.1 wav2vec 2.0（2020）

Meta 提出的 wav2vec 2.0 是自监督语音预训练的里程碑。它先用 CNN 提取 50Hz 的语音特征，再用 Transformer 编码，最后用对比学习在 mask 后的 latent 上预测原 latent。

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 输入: 16kHz 采样率音频
input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values
logits = model(input_values).logits   # (B, T, vocab_size)
pred_ids = torch.argmax(logits, dim=-1)
transcript = processor.batch_decode(pred_ids)[0]
```

### 2.2 Whisper（2022）

OpenAI 的 Whisper 是一次彻底重构：把 ASR 当作"语音到文本的 seq2seq"，用 68 万小时多语种数据训练一个大编码器-解码器 Transformer。

**核心创新**：

1. **多任务统一**：在同一个模型上同时训练 ASR、语音翻译、语种识别、语音活动检测。
2. **大模型 + 大数据**：small/base/small.en/medium/large 共 5 个规模，最大 1550M 参数。
3. **直接预测文本**：用 BPE tokenizer，把解码器当 LLM 用。

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("cuda")
model.eval()

# 1. 加载音频（demo 用 librosa 或 soundfile）
import librosa
audio, sr = librosa.load("speech.wav", sr=16000)

# 2. 特征提取
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to("cuda")

# 3. 强制中文转写（可选关键参数）
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="chinese", task="transcribe"
)

# 4. 生成
with torch.inference_mode():
    pred_ids = model.generate(
        **inputs,
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=200,
        num_beams=1,                  # 1 = greedy, 5 = beam search 更准但慢
        return_timestamps=True,        # 长音频需要 chunking
    )

print(processor.batch_decode(pred_ids, skip_special_tokens=True)[0])
```

**关键参数说明**：

- `language`：锁定转写语种，减少语种识别错误（如 zh、en、ja）。
- `task`：`transcribe` 同语种、`translate` 翻成英文。
- `return_timestamps`：返回单词级时间戳，配合 long-form 推理。
- `chunk_length_s`：长音频切片（默认 30 秒），由 [`pipeline`] 自动处理。
- `condition_on_prev_tokens`：相邻 chunk 是否以前文为条件（默认 True）。

### 2.3 Whisper 蒸馏与加速

Whisper-large 在 CPU 上实时性差，工业部署常用三种加速手段：

| 方法 | 加速比 | 质量损失 |
| --- | --- | --- |
| **Whisper.cpp**（量化 + C++） | 3-10x | 极小 |
| **Faster-Whisper**（CTranslate2） | 4x | 几乎无 |
| **Distil-Whisper**（蒸馏） | 6x | WER +1-2% |

```bash
# Faster-Whisper 一行启动
pip install faster-whisper
python -c "
from faster_whisper import WhisperModel
model = WhisperModel('large-v3', device='cuda', compute_type='float16')
segments, info = model.transcribe('speech.wav', language='zh')
for s in segments:
    print(f'[{s.start:.1f}-{s.end:.1f}] {s.text}')
"
```

## 三、Speech-LLM：让 LLM "听懂"声音

把音频能力装进 LLM 是 2023-2024 年的研究热点。核心挑战：**音频信号是连续的、长序列的，而 LLM 习惯离散的文本 token**。

### 3.1 SALMONN（2023）

清华 + 上海 AI Lab 的 SALMONN 用**双 encoder + LLM** 的结构：

```
音频 ──→ Whisper encoder (语音内容)
       ↘
        Concat → LLaMA-13B → 文本输出
       ↗
音频 ──→ BEATs encoder      (声音事件 / 音乐)
```

**关键设计**：

- 训练时**两个 encoder 都冻结**，只训中间一个"音频 bridge"。
- 用 LoRA 微调 LLaMA，保证不破坏语言能力。
- 支持语音问答、音乐评论、音视频融合等多种任务。

### 3.2 Qwen2-Audio（2024）

阿里发布的 Qwen2-Audio 把音频输入统一为"自然语言 prompt"：

- 用 Whisper-large-v3 风格的 encoder 抽取特征。
- 过一个 MLP projector 进入 Qwen-7B 的 embedding 空间。
- 支持两种模式：
  - **Audio Analysis**（音频 → 文本描述）。
  - **Audio Chat**（音频 + 文本指令 → 文本回答）。

```python
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

# 多轮对话
conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://example.com/cat.wav"},
        {"type": "text", "text": "这是什么声音？"},
    ]},
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, audios=audio_array, sampling_rate=16000,
                   return_tensors="pt", padding=True)
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
```

### 3.3 其他重要模型

| 模型 | 特点 |
| --- | --- |
| **AudioPaLM**（Google） | 把 ASR/TTS/翻译统一在 PaLM-2 框架 |
| **SpeechGPT**（开源） | 早期 speech-instruct 模型 |
| **VALL-E 2**（Microsoft） | TTS + 声音克隆，SOTA 真实度 |
| **SoundStorm**（Google） | 高效并行 TTS |
| **MusicGen / Suno** | 音乐生成大模型 |

## 四、音频理解的任务分类

随着 Speech-LLM 兴起，"音频理解"成为一个统称，可拆解为：

1. **语音内容层**：识别说了什么（ASR）、翻译（Speech Translation）。
2. **声学特征层**：识别谁在说（说话人识别）、情绪、语气、年龄。
3. **语义层**：理解语义、回答问题、生成摘要。
4. **非语音层**：音乐、动物叫声、环境声、警报声等声音事件。

新一代模型往往把 1-4 全部纳入训练，对应一套**统一的 prompt 模板**，例如：

```text
[Audio] <audio features>
[Instruction] 请回答：这段语音里的人在讨论什么话题？情绪如何？
[Output] 这是一段关于……的对话，说话者语气较为……"
```

## 五、训练数据与评测

| 类别 | 数据集 |
| --- | --- |
| **ASR 训练数据** | Common Voice、Librispeech、AISHELL、WenetSpeech |
| **音频理解** | AudioCaps、Clotho（音频描述）、MMAU（音频问答） |
| **语音翻译** | CoVoST、MuST-C |
| **评测** | WER（词错误率）、BLEU、AudioBench、AudioBench-AQA |

## 六、工程要点

1. **采样率统一**：所有模型要求 16kHz；高采样率音频必须重采样。
2. **长音频切片**：Whisper 默认 30 秒一片，多片拼接时用 `condition_on_prev_tokens`。
3. **VAD 预处理**：先做语音活动检测（silero-vad），剔除静音段，提升 WER。
4. **多语种路由**：先识别语种再分发到对应 ASR，可减少混淆。
5. **流式推理**：Whisper streaming、Voski、WeNet Streaming 是工业首选。

## 小结

音频大模型的演进可以分成两条主线：**纯语音（ASR/TTS）** 和 **Speech-LLM（通用音频理解）**。前者以 Whisper 为代表已经高度成熟，后者以 SALMONN、Qwen2-Audio 为代表正在快速追赶文本与视觉多模态的能力。下一篇我们将讨论一个贯穿所有多模态任务的关键议题——如何系统地**评测**这些模型。
