# GPT 与 BERT 的演化：三条预训练路线的兴衰

前两篇 [attention-mechanism-explained.md](attention-mechanism-explained.md) 与 [transformer-architecture.md](transformer-architecture.md) 解决了"如何搭建 Transformer"。现在剩下的问题是：**用什么目标在大规模语料上预训练它？** 这个看似简单的选择，把 2018 年之后的 NLP 分成了三条路线——BERT（encoder-only）、GPT（decoder-only）、T5/BART（encoder-decoder）。本文梳理三条路线的目标函数、代表工作、各自的优势，以及为什么到 2022 年底 decoder-only 几乎统一江湖。下一篇 [gpt-vs-llama-architecture.md](gpt-vs-llama-architecture.md) 会聚焦于"decoder-only 内部"——GPT-3 与 LLaMA 之间到底有哪些工程细节差异。

## 一、单一问题的三个答案

2018-2023 年 NLP 有一个几乎统治一切的核心问题：**在无标签文本上，模型应该预测什么？**

```text
                ┌─────────────────────┐
                │   大量无标签文本      │
                └──────────┬──────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       BERT: 填空      GPT: 接龙       T5: 拼图
   (Masked LM)     (Causal LM)     (Span Corruption)
            │              │              │
       encoder-only    decoder-only   encoder-decoder
            │              │              │
       双向 / 填空     单向 / 生成     输入→输出
```

三种目标都"自监督"，都不需要人工标注，但学到的模型能力与下游用途截然不同。

## 二、BERT：双向 Encoder + 填空

### 2.1 Masked Language Modeling（MLM）

BERT（Devlin et al., 2018）随机遮住 15% 的 token，让模型根据**左右上下文**预测被遮住的词：

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i \mid x_{\setminus M})
$$

注意**双向**：位置 $i$ 同时看到左边和右边的 token。这是与 GPT 最大的区别——也是 BERT 擅长理解、不擅长生成的根本原因。

**15% mask 的"80/10/10"策略**：

```text
选中 15% 的位置，对每个位置：
  80% 替换为 [MASK]            ──► 模型学会填空
  10% 替换为随机 token         ──► 模型不能"只看见 [MASK] 就偷懒"
  10% 保持原 token             ──► 模型需要判断"是否被改"
```

**为什么不全用 `[MASK]`**：下游 fine-tuning 时不会出现 `[MASK]`，预训练与微调不一致会让模型对 `[MASK]` 这个特殊 token 过拟合。引入 10%/10% 的"噪声"让模型学会"看上下文猜原词"，而不是"看见 [MASK] 才反应"。

### 2.2 Next Sentence Prediction（NSP）

50% 概率把文档中的第二句换成随机句子，让模型判断"是否是连续两句"：

```text
[CLS] 句子 A  [SEP] 句子 B  [SEP]   →  IsNext: True
[CLS] 句子 A  [SEP] 随机句子 [SEP]  →  IsNext: False
```

目标：让 [CLS] 位置的表示学到"句子关系"，用于 NLI、QA 等下游任务。

**后来的发现**：RoBERTa（Liu et al., 2019）实验显示，去掉 NSP 反而**显著更好**——NSP 太简单，模型很快就能完成，对表示学习帮助有限。后续 BERT 变体几乎都放弃了 NSP。

### 2.3 BERT 家族

| 模型 | 关键改动 | 年份 |
|---|---|---|
| BERT-base / -large | MLM + NSP | 2018 |
| RoBERTa | 去 NSP，更多数据，更多步数，更大 batch | 2019 |
| ALBERT | 跨层参数共享，Embedding 分解 | 2019 |
| DeBERTa | 解耦相对位置与内容注意力 | 2020 |
| ELECTRA | 用判别式"替换 token 检测"替代 MLM | 2020 |
| SpanBERT | 预测连续 span 而非单 token | 2019 |

**BERT 擅长的任务**：分类（情感、topic）、序列标注（NER、POS）、检索 embedding（sentence-transformers、ColBERT 都基于 BERT 系）、问答抽取。

## 三、GPT：单向 Decoder + 接龙

### 3.1 Causal Language Modeling（CLM）

GPT 系列只做一件事——给定前 $t-1$ 个 token，预测第 $t$ 个：

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

```text
输入:   我  爱  NLP
目标:   爱  NLP [EOS]
每个位置的预测只能看到它左边的 token（causal mask）
```

形式简单到几乎不需要解释，但这个"接龙"目标有一个惊人的性质——**几乎所有 NLP 任务都可以被改写成"文本接龙"**：

```text
分类:        "这段评论的情感是 [正面]"  → 让模型预测下一个 token
翻译:        "Translate to French: I love you →" → "j'aime"
问答:        "Q: 什么是 RoPE? A:" → 模型生成答案
摘要:        "文章: ... TL;DR:" → 模型生成摘要
```

这就是 GPT 走向通用模型的逻辑起点。

### 3.2 GPT 家族

| 模型 | 参数量 | 关键能力 | 年份 |
|---|---|---|---|
| GPT-1 | 117M | 首次证明生成式预训练可行 | 2018 |
| GPT-2 | 1.5B | Zero-shot 任务迁移（无下游训练） | 2019 |
| GPT-3 | 175B | Few-shot（in-context learning） | 2020 |
| InstructGPT | - | 指令微调（RLHF） | 2022 |
| ChatGPT | - | 对话 + RLHF | 2022 |
| GPT-4 | 估计 1T+ | 多模态、长上下文 | 2023 |

**Scale-driven emergence**：GPT-2 展示了"任务可以零样本完成"，GPT-3 展示了"模型可以在 prompt 里看几个例子就学会新任务"——后者就是 **in-context learning**，是大模型能力的核心现象之一。

### 3.3 为什么 CLM "够用"

直觉：CLM 强迫模型在每一个位置学到"基于上文的最优预测"——这要求模型**真正理解上下文**才能做好。一个模型如果能完美做 CLM，它就能完美做任何文本任务（因为都可以 cast 成 CLM）。这给 CLM 一个"通用目标"的位置。

代价是：CLM 只学了"因果方向"的能力，无法直接得到双向的"理解"表示——但实践中通过超大模型 + 多任务 prompt，这一短板被大幅弥补。

## 四、T5 / BART：Encoder-Decoder + 拼图

### 4.1 Span Corruption

T5（Raffel et al., 2019）把 BERT 的"单 token mask"扩展为"span mask"——随机遮住连续片段，用 **sentinel token** 替换，要求 decoder 输出原始文本：

```text
原文:    "I love natural language processing"
随机 mask:  "I love X1 processing"  (X1 替换 "natural language")
目标:      "<X1> natural language"

再 mask:   "I X2 X1 X3"  (X2 替换 "love", X3 替换 "processing")
目标:      "<X2> love <X3> processing"
```

为什么用 sentinel：`X1, X2, X3` 是模型可控的特殊 token，decoder 必须"按顺序"还原——比单纯填空难。

### 4.2 T5 / BART 适用场景

- **机器翻译**：天然 encoder-decoder，源文与目标文严格分离。
- **摘要**：源文 → 摘要。
- **结构化预测**：输入 → 输出形式严格不同。
- **任何显式 seq2seq 任务**：分类、QA 生成版本、对话生成。

BART（Lewis et al., 2020）用相似的"文本破坏 + 生成"思路，是 T5 的近亲。

## 五、对比：三种范式一览

| 维度 | BERT (encoder) | GPT (decoder) | T5/BART (enc-dec) |
|---|---|---|---|
| 目标 | MLM（双向填空） | CLM（单向接龙） | Span Corruption（拼图） |
| Attention 模式 | 双向 self-attention | causal self-attention | enc 双向 + dec causal + cross |
| 训练样本量 | 3.3B 词 (BERT) → 160B (RoBERTa) | 300B → 2T → 15T | ~1T tokens (T5) |
| 优势任务 | 分类、检索 embedding、NER | 生成、对话、in-context learning | 翻译、摘要、seq2seq |
| 代表模型 | BERT、RoBERTa、ELECTRA、DeBERTa | GPT-1/2/3、LLaMA、Qwen、Mistral | T5、BART、mBART、FLAN-T5 |
| 是否仍在主流 | 是（检索/小模型） | **主流**（所有大语言模型） | 局部（翻译、专用） |
| 推理模式 | 单次 forward | 自回归生成 | 自回归生成（仅 decoder） |
| KV cache | 不需要 | 必需 | 必需 |

## 六、为什么 Decoder-Only 赢了

到 2022-2023 年，几乎所有"通用大语言模型"都是 decoder-only。原因是多个因素的合流：

### 6.1 目标统一性

```text
BERT 的 MLM：   "填空"                       ← 单任务
GPT 的 CLM：    "预测下一个 token"             ← 通用任务
                → 分类、改写、翻译、问答……都能 cast
```

CLM 是**最通用的自监督目标**——学完 CLM，等于学了"所有"以文本形式表达的任务。这让 decoder-only 模型天然具备"通用助手"的潜质。

### 6.2 规模可预测性

Decoder-only 的 scaling laws 已被反复验证（GPT-3、Chinchilla、PaLM）。增加数据 / 参数 / 算力，loss 几乎单调下降。BERT 类模型的 scaling 收益更早饱和。

### 6.3 推理效率：KV Cache + 自回归

Decoder-only + 自回归生成 → KV cache 工程非常清晰：

```text
第 t 步：只需算 (Q_t, K_≤t, V_≤t) → 节省 (t-1) 步的重算
所有 LLM serving 系统（vLLM、TGI、TensorRT-LLM）都围绕这个假设设计
```

BERT 类模型没有 KV cache 优势——它们做分类/检索都是一次 forward，本来就不需要生成。

### 6.4 开源生态：LLaMA 引爆

2023 年 Meta 发布 LLaMA，**完全开源 + decoder-only**。整个开源生态（LLaMA → Alpaca → Vicuna → Qwen → Mistral → Yi → DeepSeek）几乎全是 decoder-only。工具链（vLLM、HF Transformers、TGI、llama.cpp）也都是先为 decoder-only 设计。

### 6.5 BERT 仍然活着

虽然 decoder-only 主导，**encoder-only 并没有消亡**：

- **检索 embedding**：sentence-transformers（基于 BERT/MiniLM）、ColBERT、BGE、E5——这些至今仍在用 encoder，因为"双塔检索"需要一次 forward 得到 embedding。
- **小模型分类 / NER**：100M 参数的 BERT 在工业界仍广泛部署——小模型不需要 in-context learning。
- **重排序 / rerank**：cross-encoder 用 encoder 做精排。

## 七、时间线

```text
2017  Transformer        (Vaswani et al., "Attention Is All You Need")
       │
2018  BERT      GPT-1    (双向填空 vs 单向接龙，两条路线同时诞生)
       │
2019  RoBERTa   GPT-2    T5     (BERT 改良 / GPT zero-shot / 统一框架)
       │
2020  ELECTRA   GPT-3    BART         (GPT-3 175B 证明规模奇迹)
       │
2021                       InstructGPT (RLHF 起步)
       │
2022  ChatGPT / GPT-3.5                    (对话爆火，decoder-only 主流)
       │
2023  LLaMA-1/2, Qwen, Mistral, Yi        (开源 decoder-only 全面铺开)
       │
2024  LLaMA-3, DeepSeek-V3, GPT-4o        (更大、更长、多模态)
```

观察：从 2022 年起，主流几乎只剩 decoder-only；BERT 系退守"检索 / 小模型分类"。

## 八、GPT 与 BERT 的本质区别再总结

```text
              BERT                       GPT
              ────                       ───
         双向 attention              单向 causal attention
         看到"上下文"                看到"历史"
         预测"被遮的词"              预测"下一个词"
         输出：token 级表示          输出：每个位置的下一个 token
         用于：理解、检索、分类       用于：生成、对话、推理
         现代地位：工具型模型          现代地位：通用助手 / 主力路线
```

一个简化的口诀：

- **BERT = 双向理解**（"看懂"）
- **GPT = 单向生成**（"接龙"）
- **T5 = 输入→输出**（"翻译"）

到 2024 年，"懂"和"译"在 decoder-only 上**被一起学会**了——这就是 LLaMA-3 等大模型既能聊、能答、又能翻译的原因。

## 小结

BERT（encoder-only + MLM）、GPT（decoder-only + CLM）、T5/BART（encoder-decoder + span corruption）三大预训练范式分别抓住了 NLP 的三种"自监督信号"——填空、接龙、拼图。BERT 在分类、检索、NER 上至今重要，encoder 的双向表示难以替代；T5 系在显式 seq2seq 任务（翻译、摘要）仍有结构优势；而 GPT 系的 CLM 因其"任何任务都能 cast 成接龙"的通用性，加上 KV cache 工程友好、scaling laws 平滑、开源生态聚焦，最终成为 2022 年以来的绝对主流。这并不意味着其他范式消亡——encoder-only 在检索和小模型场景仍是主力，而 encoder-decoder 在结构化生成任务上仍有不可替代性。本文确立"decoder-only 是大模型主力"的事实——下一篇 [gpt-vs-llama-architecture.md](gpt-vs-llama-architecture.md) 会进一步深入到 decoder-only 内部，看 GPT-3 与 LLaMA 这两个里程碑之间到底在 RoPE、RMSNorm、SwiGLU、GQA 这些工程选择上做了哪些关键权衡。