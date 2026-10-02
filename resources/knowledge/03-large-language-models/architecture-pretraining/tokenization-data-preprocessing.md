# Tokenization 与数据预处理

LLM 不直接读字符串——它们读 **token**。Tokenizer 决定了模型的"基本字表"：一段文本被切成什么样的子词、哪些特殊 token 标识对话边界、词表多大才合理。本文系统讲清 BPE / WordPiece / SentencePiece / Unigram 四种主流分词算法的核心思想，再用一个可运行的 SentencePiece 训练示例与 ChatML 格式化代码，把"从原始文本到训练样本"的整条链路串起来。

## 一、为什么需要 Subword Tokenization

早期 NLP 用词级（word-level）词表：OOV（Out-Of-Vocabulary）问题严重、生僻词学不到、词表动辄百万级。字符级（char-level）词表太小，但序列长度暴涨、训练效率差。

**Subword** 的折中：把高频词保留为整体，把低频词按字符片段切分。一个直观例子：

```text
"unhappiness"  →  ["un", "happiness"]   (常见词根合并)
"语言模型"      →  ["语", "言", "模型"]   (按字节 / 字符频切)
```

这样既控制了词表大小（通常 32k~256k），又能表示任意新词（任何字符串总能切成 subword 序列）。

## 二、BPE：Byte Pair Encoding

BPE（最初是数据压缩算法）被 Sennrich et al. (2016) 引入 NLP 子词切分。流程：

1. 初始化词表为所有单字节（或单字符）。
2. **统计语料中相邻 token 对的出现频率**，把频率最高的 pair 合并成新 token。
3. 把新 token 加入词表，重复步骤 2 直到词表达到目标大小。
4. 切分时，对输入文本做同样的 merge 序列，得到 subword 序列。

直觉上：**最频繁出现的字符对先合并成"词"**，所以高频词最终保留为整体，低频词被切细。

数学地看，merge 的优先级 $f(a,b)$ 是该 pair 在语料中的总出现次数。贪心 BPE 是局部最优，不能保证全局最优——但实现简单、效果好，工业界主流。

## 三、WordPiece vs BPE

WordPiece（Schuster & Nakajima, 2012，Google BERT 用）几乎与 BPE 一样，但**合并优先级不是频率，而是互信息**：

$$
\text{score}(a, b) = \frac{f(a, b)}{f(a) \cdot f(b)}
$$

分子是 pair 出现次数，分母是各自单独出现次数的乘积——衡量"这俩 token 一起出现比偶然更频繁多少"。**含义**：低频 pair 若其组成 token 也很低频，score 仍可能很高；这让 WordPiece 更倾向合并"语义相关"的子词。

实际切分上，WordPiece 用**最大前缀匹配**（贪心最长），而 BPE 用 merge 顺序应用。

## 四、Unigram LM（Kudo, 2018）

BPE / WordPiece 是 bottom-up 的合并方法。Unigram 反过来：**先假设一个大词表，通过 EM 算法逐步剪枝**。

1. 初始化一个大词表（比如所有字符 + 常见 subword）。
2. 对每种可能的切分方式计算其概率（基于 unigram LM）。
3. 对每个 token，统计"它在语料中作为切分单元出现的边际概率"，剪掉低贡献的 token。
4. 重复直到词表达到目标大小。

切分时，用 **Viterbi 算法**找到概率最大的切分方式（不是贪心）。优点：能给出多种合理切分的概率分布；SentencePiece 同时支持 BPE 和 Unigram 两种训练模式。

## 五、SentencePiece：与语言无关的分词框架

SentencePiece（Kudo & Richardson, 2018）由 Google 提出，关键创新：

- **把输入视为字节流**（或 Unicode 码位流），不依赖任何"预切词"。这意味着中日韩等无空格语言不需要先分词。
- **提供 Normalization**（NFKC、大小写归一化等）作为 pipeline 一部分。
- **同时支持 BPE 和 Unigram 训练**，输出统一的 `model` / `vocab` 文件。

`--model_type=unigram` 训练得到 unigram，`--model_type=bpe` 得到 BPE。训练产物（`.model`）跨语言、跨平台一致。

## 六、用 SentencePiece 训练一个 BPE 模型

下面给出一个可运行的最小训练脚本。准备一个纯文本文件 `corpus.txt`，每行一段文本：

```python
"""train_sp.py: 训练一个 SentencePiece BPE 模型。"""
import sentencepiece as spm

# 1) 训练：输入 corpus.txt，输出 spm.model / spm.vocab
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_sp",
    vocab_size=8000,
    model_type="bpe",            # 也可改成 "unigram"
    character_coverage=0.9995,    # 中文/英文一般 0.9995
    byte_fallback=True,           # 未知字符回退到字节，保证可逆
    normalization_rule_name="nfkc",
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    input_sentence_size=200000,   # 采样前 N 行训练
    shuffle_input_sentence=True,
    num_threads=8,
)

# 2) 加载并测试
sp = spm.SentencePieceProcessor(model_file="my_sp.model")
for text in [
    "Hello, world!",
    "Transformer 是现代 LLM 的核心架构。",
    "unhappiness",                # 测试 BPE 切词
]:
    ids = sp.encode(text, out_type=int)
    pieces = sp.encode(text, out_type=str)
    print(f"text={text!r}")
    print(f"  pieces={pieces}")
    print(f"  ids   ={ids}\n")

# 3) 解码验证可逆性
print("decoded:", sp.decode(sp.encode("Hello, world!")))
```

几个关键参数：

- **`vocab_size=8000`**：小语料 / demo 用 8k 即可，生产 LLM 一般 32k~128k。
- **`character_coverage=0.9995`**：中文建议 0.9995，英文可降到 0.9995~1.0。低频字符用 `<unk>` 替代。
- **`byte_fallback=True`**：把每个 byte 也加入词表，任何 UTF-8 字符都能编码；代价是平均序列长度增加。
- **`normalization_rule_name="nfkc"`**：Unicode 正规化，把"全角数字"和"半角数字"统一，避免同一字符多种编码。

## 七、特殊 Token 与 Chat Template

现代 LLM 用一套**特殊 token** 标识对话边界、角色、系统提示等：

| Token | 含义 | 典型例子 |
|---|---|---|
| `<bos>` | 序列开始 | LLaMA: `<s>` |
| `<eos>` | 序列结束 | LLaMA: `</s>` |
| `<pad>` | 填充 | 多用于 batch 训练 |
| `<unk>` | 未知字符 | byte_fallback 后基本不用 |
| `<|im_start|>` | 消息开始 | ChatML (Qwen) |
| `<|im_end|>` | 消息结束 | ChatML (Qwen) |
| `<|user|>` / `<|assistant|>` | 角色标识 | Llama-3 / Mistral |

不同模型有不同的 **chat template**，把多轮对话格式化成单字符串。HuggingFace 把它存进 `tokenizer_config.json` 的 `chat_template` 字段，`tokenizer.apply_chat_template(messages)` 一键调用。

## 八、ChatML / Llama-3 / Mistral 模板对比

### ChatML（Qwen 等）

```text
<|im_start|>system
你是一名严谨的助手。<|im_end|>
<|im_start|>user
什么是 RoPE？<|im_end|>
<|im_start|>assistant
旋转位置编码……<|im_end|>
```

### Llama-3

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is RoPE?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### Mistral（v0.3+）

```text
<s>[INST] You are a helpful assistant.

What is RoPE? [/INST]
```

### 编程调用

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
messages = [
    {"role": "system", "content": "你是一名严谨的助手。"},
    {"role": "user",   "content": "什么是 RoPE？"},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
# <|im_start|>system\n你是一名严谨的助手。<|im_end|>\n
# <|im_start|>user\n什么是 RoPE？<|im_end|>\n
# <|im_start|>assistant\n
```

`add_generation_prompt=True` 会在末尾追加 assistant 开头标记，方便模型直接接续生成。

## 九、训练数据预处理 Pipeline

一段文本到"训练样本"的完整链路：

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 1) 加载原始文本
ds = load_dataset("json", data_files="data/pretrain.jsonl", split="train")

# 2) 用 chat template 拼装（若没有 messages 字段，可当作单轮 user）
def format_example(ex):
    messages = [
        {"role": "user",      "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    text = tok.apply_chat_template(messages, tokenize=False)
    return {"text": text}

ds = ds.map(format_example, remove_columns=ds.column_names)

# 3) tokenize + 打包到固定 seq_len
block_size = 4096
def tokenize_and_pack(batch):
    ids = tok(batch["text"], add_special_tokens=False)["input_ids"]
    # 简单拼接（生产可加多文档分隔符）
    concatenated = sum(ids, [])
    total_len = (len(concatenated) // block_size) * block_size
    chunks = [
        concatenated[i: i + block_size]
        for i in range(0, total_len, block_size)
    ]
    return {"input_ids": chunks, "labels": [c.copy() for c in chunks]}

ds = ds.map(tokenize_and_pack, batched=True, remove_columns=["text"])
print(ds)
```

几个常见细节：

- **多文档分隔**：不同文档直接拼接会让模型学到"上段结束就是下段开始"的虚假模式。生产里会在每段间插入 `<eos>` 或自定义 `<|endoftext|>`，让 attention mask 也能隔开。
- **长度过滤**：丢掉 < N token 的样本，避免 packing 后大量 padding。
- **特殊 token 计数**：`<|im_start|>` 等也会被算进 `seq_len`，别忘了算 budget。

## 十、选哪种分词器？

| 算法 | 代表模型 | 优点 | 缺点 |
|---|---|---|---|
| BPE | GPT-2/3, LLaMA, Mistral | 简单、训练快 | 切分可解释性一般 |
| WordPiece | BERT, DistilBERT | 互信息合并，语义更稳 | 不能直接训 SentencePiece |
| Unigram LM | SentencePiece, T5, XLNet | 给出切分概率，可多种切分 | 训练慢（EM） |
| tiktoken (cl100k) | GPT-4, GPT-3.5 | 极致压缩比，多语言好 | 不开源训练代码 |

**经验法则**：今天做新模型选 SentencePiece + Unigram 几乎不会错——多语言好、可控词表大小、能 byte fallback。训练时把 chat template 锁死，推理时用 `apply_chat_template`，就能避免"训练与推理模板不一致"的经典坑。

## 小结

Tokenizer 决定了模型"看到什么"——切得太粗，词表爆炸；切得太细，序列变长、训练变慢。BPE 用贪心合并实现简单、效果稳；WordPiece 用互信息更"语义化"；SentencePiece 提供语言无关的统一框架；Unigram LM 用概率视角给出多切分。配合 chat template 与多文档分隔，文本到训练样本的整条链路才算真正跑通。下一篇我们看更大尺度的问题：当模型从几 B 涨到几百 B，单机已经装不下——**分布式训练**怎么做。
