# ASR 评估、解码与部署优化

模型训出来只是开始——**如何衡量它好不好**、**如何把概率分布变成最终文字**、**如何把 1.5B 参数量压缩到能在浏览器、车载、低端手机实时跑**——这些才是 ASR 从论文走向生产的三大核心问题。本篇围绕**评估指标**、**解码算法**、**部署优化**三个主题展开，是前 3 篇 ASR 文章的"收尾工程篇"。

## 一、词错误率（WER）与字错误率（CER）

### 1.1 WER 定义

**WER（Word Error Rate）** 是 ASR 最经典的指标，定义为**编辑距离**（Levenshtein distance）与参考文本词数的比值：

$$
\text{WER} = \frac{S + D + I}{N} \times 100\%
$$

其中 $S$ 是**替换（substitution）** 数，$D$ 是**删除（deletion）** 数，$I$ 是**插入（insertion）** 数，$N$ 是参考文本的词数。

### 1.2 编辑距离 = 动态规划

词对齐通过**编辑距离算法**求出：

$$
d(i, j) = \min \begin{cases}
d(i-1, j) + 1 & \text{删除参考词 } w_i \\
d(i, j-1) + 1 & \text{插入候选词 } \hat{w}_j \\
d(i-1, j-1) + [w_i \neq \hat{w}_j] & \text{替换或正确}
\end{cases}
$$

PyTorch 风格的实现并不复杂，但工程中我们都用现成工具：`jiwer`（Python）、`sclite`（NIST）。

```python
import jiwer

ref = ["the quick brown fox jumps over the lazy dog"]
hyp = ["the quick brown fox jumps over lazy dog"]  # 漏 "the"

# 注意标准化参数，中文场景不要 lower-case 和 ignore-case
out = jiwer.process_words(ref[0], hyp[0])
print(out.wer)        # 1/11 ≈ 9.09%
print(out.substitutions, out.deletions, out.insertions)  # 0 1 0
```

### 1.3 CER 与中英文差异

中文没有"词"的概念，**字符错误率（CER, Character Error Rate）** 直接在字符级别计算：

$$
\text{CER} = \frac{S + D + I}{N_{\text{char}}} \times 100\%
$$

中英文差异的几个工程要点：

| 维度 | 英文 | 中文 |
| --- | --- | --- |
| 切分单位 | 词（空格分隔） | 字（无空格） |
| 同音字 | 罕见 | 极常见 |
| 标点 | 标准 | 中英文标点混用 |
| 数字/单位 | "12 thousand" | "1.2 万" |

**中文 CER 通常比英文 WER 难一倍以上**：同音字多、方言多、繁简混用、专业术语没规范。因此 ASR 系统通常会**对中文单独训练或微调**。

### 1.4 文本归一化（Text Normalization）

直接对比 WER 有个陷阱："$100" 与 "一百美元" 是否算错？ASR 论文常用两种：

- **Standard WER**：不做归一化，最严格。
- **Normalized WER**：先转小写、去标点、数字转口语等，更宽松，反映**语义正确率**。

中文常需要做"简繁转换"、"全半角转换"、"数字标准化"等。**这一规范化选择常被新手忽视，导致报告 WER 看起来很好，实际产品体验差**。

## 二、解码：从概率到文字

### 2.1 贪心 vs Beam Search

最朴素的解码是每帧 argmax：

$$
y_t^* = \arg\max_k p_t(k)
$$

但 ASR 中**贪心搜索几乎总是次优**——每帧独立最优 ≠ 整句最优。**Beam Search** 维护 $k$ 个最优候选，扩展时只保留 top-$k$ 累积概率：

$$
\text{score}(y_{1:t}) = \sum_{s=1}^{t} \log p(y_s \mid y_{<s}, X)
$$

Beam size 通常 5-50，再大边际收益迅速递减。

### 2.2 CTC Prefix Beam Search

CTC 解码常用**prefix beam search**——状态定义为前缀 $p$，维护两个概率：

- $p_b(p)$：以 blank 结尾的概率。
- $p_n(p)$：以非 blank 结尾的概率。

$$
p(p) = p_b(p) + p_n(p)
$$

每步扩展时，新前缀 $p'$ 来自三种操作：

| 操作 | 概率 |
|---|---|
| $p$ 末尾接 blank | $p(p) \cdot p_t(\epsilon)$ |
| $p$ 末尾接已存在的字符 $c$ | $p_n(p) \cdot p_t(c)$ |
| $p$ 末尾接新字符 $c$ | $p(p) \cdot p_t(c)$ |

在扩展过程中，**维护每个前缀对应的 $p_b$ 和 $p_n$**，这是 CTC 比 vanilla beam search 复杂度更高的根本原因——但也是支持空白符的代价。

### 2.3 外部语言模型融合

CTC / RNN-T 自身不含强语言信息，**外挂 n-gram KenLM** 能显著降低 WER（5-15% 相对降幅）。常见三种融合：

#### 2.3.1 Shallow Fusion（最常用）

解码时按加权插值：

$$
\log P(y_t \mid y_{<t}, X) = \log P_{\text{AM}}(y_t \mid y_{<t}, X) + \lambda_{\text{lm}} \log P_{\text{LM}}(y_t \mid y_{<t})
$$

其中 $\lambda_{\text{lm}}$ 通过开发集搜索最优。**优点**：训练无需改动，部署灵活。**缺点**：声学和语言模型独立优化，融合系数靠调。

#### 2.3.2 Cold Fusion

训练阶段就把语言模型**嵌入到解码器**。预测器网络 $L$ 的输出 $p_{\text{lm}}(y_t \mid y_{<t})$ 通过一个 gate 与声学分布融合：

$$
p_{\text{fusion}}(y_t \mid y_{<t}, X) = g_t \cdot p_{\text{lm}}(y_t \mid y_{<t}) + (1 - g_t) \cdot p_{\text{am}}(y_t \mid y_{<t}, X)
$$

$g_t$ 是一个 sigmoid 网络，训练时学习。**优点**：训练目标统一，融合系数自适应。**缺点**：训练复杂，需要 LM 与 AM 联合训练。

#### 2.3.3 Deep Fusion

把 LM embedding 直接拼接到 AM hidden state，再过一个融合网络。介于 shallow 和 cold fusion 之间。

### 2.4 热词偏置（Hotword Boosting）

会议、产品名、人名识别是工业 ASR 的核心痛点。**热词偏置**在解码时对特定词加分：

$$
\text{score}'(y_{1:t}) = \text{score}(y_{1:t}) + \sum_{w \in \text{hotwords}} \mathbb{1}[w \in y_{1:t}] \cdot \beta_w
$$

或者在 CTC prefix beam search 里，对包含热词的前缀直接给一个 $\beta$ 增量。工程上热词权重 $\beta$ 通常 2-5，相对误差 10%。

## 四、流式 ASR：低延迟的关键

### 4.1 流式 vs 离线

- **离线 ASR**：必须等完整音频（如 30 秒窗口）才能解码。延迟高，但准确。
- **流式 ASR**：边输入边解码。延迟 < 300ms 即可支持实时字幕、同声传译。

### 4.2 Chunk-based 流式

最朴素的方案：**每 200-500 ms 切一段**，对每段独立跑 Conformer encoder，再增量更新 beam search：

$$
H_t = \text{Encoder}(X_{t-L:t})
$$

其中 $L$ 是**左上下文窗口**。$L$ 越大准确率越高（更多声学），但延迟也越高（必须等 $L$ 帧才能开始解码）。

### 4.3 Emformer 与缓存机制

**Emformer（Shi et al., Microsoft, 2021）** 是流式 Transformer 的工业代表。它的关键设计是**双端注意力掩码 + 内存缓存**：

- **左内存** = 历史 chunk 的 K/V 缓存。
- **右上下文** = 当前 chunk 后 $R$ 帧（用于感受野）。
- **总结向量**：每隔 $M$ 帧压缩一个 summary token，丢弃不重要的历史 K/V。

数学上，第 $i$ 块对第 $j$ 块的注意力权重为：

$$
\alpha_{i, j} = \begin{cases}
\text{full} & j = i \text{（当前块）} \\
\text{summary} & j \le i-1 \text{（历史块，仅看 summary）} \\
\text{limited} & j = i+1 \text{（未来块，右上下文）} \\
0 & \text{otherwise}
\end{cases}
$$

这让 Emformer 在流式场景下保持接近非流式的准确率。

### 4.4 Whisper 流式化（whisper-streaming）

Whisper 本身不是流式。社区方案 **whisper-streaming（ufal, 2023）** 用 **LocalAgreement** 策略：

1. 增量跑 Whisper，每接收 1 秒音频得到当前假设。
3. 对每个 token，检查后续 2 个 chunk 是否仍包含相同 token——若 3 个 chunk 连续一致，**emit**；否则等待。

这能把 Whisper 部署为 0.3-0.5 秒延迟的实时 ASR，但准确率下降 3-5%。

## 五、部署优化：从论文到生产

### 5.1 量化（Quantization）

模型量化是把 FP32 权重压缩到 INT8/INT4：

$$
q = \text{round}\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}, \quad x \approx (q - \text{zero\_point}) \cdot \text{scale}
$$

| 精度 | 模型大小 | 推理速度 | WER 损失 |
| --- | --- | --- | --- |
| FP32 | 1.5 GB | 1x | 0% |
| FP16 | 0.75 GB | 2-3x | < 0.1% |
| INT8 | 0.4 GB | 4-5x | 0.5-1% |
| INT4 | 0.2 GB | 6-10x | 2-5% |

INT8 几乎无损（**PTQ（训练后量化）**）。INT4 需要 **QAT（量化感知训练）** 或 GPTQ 等高级方法。

### 5.2 ONNX / TensorRT 导出

ONNX 是模型与模型交换的中间格式：

```python
import torch, onnx, onnxruntime as ort

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").eval()
dummy_input = torch.randn(1, 80, 3000)
torch.onnx.export(
    model, dummy_input, "whisper.onnx",
    input_names=["input_features"],
    output_names=["logits"],
    dynamic_axes={"input_features": {1: "mel"}, "logits": {1: "tokens"}},
)
```

**TensorRT** 在 ONNX 之上做算子融合、kernel auto-tuning、低精度推理，延迟再降 30%-50%。

### 5.3 KV Cache：流式推理的关键

Decoder 自回归每步需要重新计算 K/V，**KV Cache** 把历史 K/V 缓存：

$$
\text{memory} = \sum_{\text{layer}} 2 \times \text{seq\_len} \times d_{\text{head}} \times \text{num\_heads} \times 2 \text{ bytes (FP16)}
$$

对 Whisper-large，每 token KV cache 约 12 MB；推理 500 token 序列占用 6 GB——**这是 LLM 推理内存的主要来源**，但**对 ASR encoder 几乎无开销**（因为 encoder 不自回归）。

### 5.4 模型剪枝与蒸馏

- **结构化剪枝**：剪掉 attention head 或 FFN 维度（如 30%）。
- **蒸馏**：用大模型（Whisper-large）输出作为监督，训练小模型（Whisper-tiny + 蒸馏损失）。
- **SparseGPT / Wanda**：单 GPU 剪枝百亿模型无需重训，几小时内完成。

### 5.5 端侧部署：whisper.cpp / sherpa-onnx

- **whisper.cpp**：GGML 框架，Whisper 在树莓派上跑 ~3x 实时。
- **sherpa-onnx**：k2-fsa 出品，支持 Conformer、RNN-T、Whisper 全系列，跨平台。

```bash
# whisper.cpp 编译 + 量化
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && make -j
./main -m models/ggml-base.en.bin -f audio.wav
```

```bash
# sherpa-onnx 流式 ASR
sherpa-onnx \
  --model=/path/to/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
  --audio=capture.wav \
  --debug=1
```

## 六、生产级 ASR 的工程全景

最后给一个生产级 ASR 服务的典型 pipeline：

```
音频输入 (16 kHz) →
  VAD (silero) →
  分块 (chunk 200 ms) →
  Conformer Encoder (ONNX Runtime + INT8) →
  RNN-T Decoder (带 KV cache) →
  CTC prefix beam search + KenLM shallow fusion →
  热词偏置 →
  ITN 文本规范化（数字、日期、标点）→
  输出文字
```

每一步都对应一个独立模块，可单独替换、单独调优、单独监控。这是工业 ASR 系统的标准形态——**单一模型无法解决所有问题，组合 + 调优才是工程之道**。

## 小结

| 主题 | 关键点 | 工业实践 |
| --- | --- | --- |
| 评估 | WER/CER + 文本归一化 | jiwer、标准化 |
| 优势 | 简单、可分项 | sclite 0.05% |
| 局限 | 不能反映语义 | CER + 人工 spot check |
| 解码 | Beam search + LM fusion | Shallow fusion 是首选 |
| CTC prefix | 处理 blank | Top-$k$ |
| 热词 | 工业必备 | 权重 2-5 |
| 流式 | Chunk + Emformer | 200 ms chunk |
| 部署 | 量化 + ONNX/TensorRT | INT8 几乎无损 |
| KV cache | 流式推理核心 | FP16 |

评估让你知道模型**好不好**；解码让你把概率变成**用户能用的文字**；部署让你把研究原型变成**几亿人能流畅使用的应用**。这三件事在 ASR 工程中的重要性，丝毫不亚于模型本身——许多团队训练出 SOTA WER 的网络，却因为解码策略差、热词缺失、ITN 不规范、量化损失大，最终落地效果远不如一个工程扎实的 base 模型。这是 ASR 这门学问"细节决定成败"的真实写照。