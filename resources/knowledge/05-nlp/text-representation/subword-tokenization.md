# 子词切分：BPE、WordPiece、Unigram 与 SentencePiece

Word2Vec 把每个词当成原子，但词表外的词（OOV）和罕见词会被丢弃或映射到 `<unk>`。**子词切分（Subword Tokenization）**通过把词拆成更小的单元，让模型在词表大小可控的前提下覆盖几乎所有文本。从 GPT-2 的 BPE、到 BERT 的 WordPiece、再到 T5/SentencePiece 的 Unigram，三种主流算法各有所长。本文从信息论的角度统一理解它们，并展示如何在工程上训练和使用一个子词切分器。

## 一、为什么需要子词

直觉上，"完全切到字符"（char-level）能解决 OOV，但模型要花大量容量学拼写规则；"完全按词切"（word-level）会遇到海量 OOV 和庞大的词表。子词切分走中间路线——**高频词保留完整，低频词按规则拆分**。它有三个直接好处：

1. **OOV 几乎消失**：任何文本都能拆成有限子词的组合。
2. **词表可控**：典型设置 8K-128K，远小于百万级 word-level。
3. **形态学共享**：英语的 `-ing`、德语的复合词、汉字的偏旁都能被多个词共享。

子词切分也是现代 LLM 的必备——LLaMA-3 的词表 128K（基于 BPE）、Qwen 的词表 152K（基于 BPE）、GPT-4 的 tiktoken（基于 BPE）无一例外。

## 二、BPE：Byte Pair Encoding

BPE 最早是文本压缩算法（Gage, 1994），Sennrich 等人在 2016 年把它引入 NLP。

### 2.1 算法流程

输入是语料文本，输出是固定大小的子词词表。

1. **初始化词表**：所有字符（或 Unicode 字节）+ 词尾标记 `</w>`，例如 `low` → `l o w </w>`。
2. **统计双字节频率**：统计相邻子词对的共现次数。
3. **合并频率最高的对**：把它们合并成新子词，加入词表。
4. **重复 2-3**，直到词表达到目标大小 $V$。

举例：假设语料是 `low low low lower lowest`，字符级初始化后：

```
l o w </w>: 3 次   l o w e r </w>: 1 次   l o w e s t </w>: 1 次
```

最高频对是 `l o`（5 次），合并为 `lo`，词表新增 `lo`。再迭代一次，`lo w </w>` 出现 3 次，再合并为 `low`... 直到词表达到 $V$。

### 2.2 BPE 的编码与解码

编码时对新文本**贪心应用合并规则**：从最长到最短（或按规则优先级），尝试合并相邻子词对。HuggingFace 的 `tokenizers` 库用 Rust 实现，速度极快（每秒 GB 级）。

字节级 BPE（Byte-level BPE, used by GPT-2）：直接以字节（256）为初始词表，覆盖任意 UTF-8 文本而无需额外的 Unicode normalization。

### 2.3 BPE 的特点

- **确定性**：同样的训练语料和词表大小，得到的合并规则完全相同。
- **高频词保留完整**：像 `the`、`is` 这种几乎总是作为整体出现。
- **问题**：合并顺序是**贪心局部最优**，不保证全局最优；对噪声敏感。

## 三、WordPiece：BERT 的选择

WordPiece（Schuster & Nakajima, 2012）由 Google 提出，BERT 用的就是它。算法流程与 BPE 几乎一样，唯一的区别是**选择合并对的标准**。

BPE 选择**频率最高的对**；WordPiece 选择**能最大化语言模型似然的对**，等价于选择**互信息最大**的对：

$$
\text{score}(x, y) = \frac{P(xy)}{P(x) \cdot P(y)} = \frac{\text{count}(xy)}{\text{count}(x) \cdot \text{count}(y)}
$$

直觉：频率高不一定"语义相关"——`"th"` 在英语里频率极高，但合并它对语义帮助小；`"ing"` 合并后能显著提升语言模型对动词形态的建模。互信息把"频率"和"独立性"结合，倾向于合并**语义上有意义的**对。

实际实现上，WordPiece 的**编码不是贪心**——它对每个位置尝试所有可能子词组合，用最大化似然的方式选最优（动态规划）。这让 WordPiece 在低频词上比 BPE 更"合理"，但训练更慢。

## 四、Unigram Language Model：T5 和 SentencePiece

Kudo（2018）提出的 Unigram 反过来思考：**从大词表开始，逐步裁剪**。它假设每个子词在句中独立出现，整个句子的概率是各子词概率的乘积：

$$
P(x) = \prod_{i=1}^{n} p(x_i), \quad \sum_{x \in V} p(x) = 1
$$

### 4.1 算法流程

1. **初始化大词表**：例如所有字符 + 高频子串 + 高频词，约 100K-300K。
2. **EM 估计概率**：用当前词表，对语料做最优切分（维特比），统计每个子词的出现频次，最大化似然。
3. **裁剪**：删除一个子词会使**总似然下降最少**，删除 $k$ 个（典型为当前词表的 10-20%）。
4. **重复 2-3**，直到词表达到目标大小。

### 4.2 关键优势：概率化切分

Unigram 的最大亮点是**多种切分方式**——同一个句子可以有多个合理的子词序列，每个序列有自己的概率。训练时通过 EM 把所有切分方式都纳入考虑；推理时可以采样（temperature 控制），这让数据增强和集成变得容易。

LLaMA-2 之后的许多 LLM 选择 Unigram + SentencePiece 实现，原因是它对**多语言**（尤其是日语、中文这种词边界模糊的语言）更友好。

## 五、SentencePiece：语言无关的端到端工具

Kudo & Richardson（2018）的 SentencePiece 是一个**完整工具包**，把上述算法封装成一个统一的 CLI。它的关键设计：

1. **把输入当作字节流**：不需要预先做分词或 Unicode normalization，直接处理原始文本。这意味着中日韩（CJK）语言不需要先分词——把整段汉字当作字符流，算法自己学最优切分。
2. **支持 BPE 和 Unigram**：训练时指定 `--model_type=bpe|unigram`。
3. **可逆性**：`encode` → `decode` 严格无损（保留空格信息）。

```bash
# 训练一个 16K 词表的 SentencePiece BPE
spm_train --input=corpus.txt --model_prefix=mypiece \
          --vocab_size=16000 --model_type=bpe \
          --character_coverage=0.9995

# 使用
spm_encode --model=mypiece.model < input.txt > output.txt
```

`--character_coverage=0.9995` 是中文场景的关键参数——CJK 字符极多，把覆盖率从默认 0.9995 提到 0.9999 才能完整覆盖生僻字。

## 六、常见词表与归一化问题

### 6.1 特殊 token

几乎所有切分器都会预留特殊 token：

- `<unk>`：未登录子词（Unigram 仍需要它，BPE 字节级版本则几乎用不到）。
- `<pad>`：padding。
- `<bos>` / `<eos>`：句首/句尾。
- `<mask>`：BERT 的掩码 token。

LLaMA-3 等还把 `<|begin_of_text|>`、`<|end_of_text|>`、`<|eot_id|>` 等功能性 token 加入词表，让模型学会"什么时候停止"或"切到对话新角色"。

### 6.2 数字与特殊符号

数字切分是个微妙的工程问题。BPE 通常把 `1234567` 切成 `123 4567` 或 `12 34 56 7`，不同合并规则导致数字推理结果不稳定。GPT-4 的 tiktoken 把每个数字单独切，BERT 的 WordPiece 则倾向于把整段数字当作一个 token。

### 6.3 归一化与大小写

- BERT WordPiece 默认 **Lower-Cased**：所有字母转小写。中文、阿拉伯数字保留。
- GPT-2 BPE 保留大小写：词表里同时有 `the` 和 `The`。
- LLaMA 词表里有 `▁`（U+2581, LOWER ONE EIGHTH BLOCK）作为"词首空格"的标记，避免空格信息丢失。

## 七、PyTorch：用 HuggingFace tokenizers 训练 BPE

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# 1. 初始化 BPE 模型
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 2. 预切分：按空白拆词（也可用 ByteLevel/Metaspace）
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 3. 训练器：定义特殊 token、词表大小、字符集
trainer = trainers.BpeTrainer(
    vocab_size=16000,
    special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

# 4. 训练
files = ["corpus.txt"]
tokenizer.train(files, trainer)

# 5. 保存与使用
tokenizer.save("mybpe.json")
encoding = tokenizer.encode("Hello, world!")
print(encoding.tokens)     # ['Hello', ',', ' world', '!']
print(encoding.ids)        # [15496, 11, 616, 0]
```

`pre_tokenizers.ByteLevel` 是 GPT-2 风格的字节级预切分，编码出来的 token 形如 `Hello,` 而非 `Hello ,`。换 LLaMA 风格用 `pre_tokenizers.Metaspace(replacement="▁")` 即可。

## 八、选型决策树

面对一个新任务，怎么选切分器？

```
多语言（含中日韩、阿拉伯文）？
├── 是 → Unigram + SentencePiece（character_coverage=0.9995）
└── 否 → 主要是英语？
        ├── 是 → Byte-level BPE（GPT-2/LLaMA 风格）
        └── 否 → WordPiece（BERT）或 BPE 都可
```

词表大小的经验值：

- 单语小模型：8K-32K
- 多语言中等模型：32K-64K
- 现代 LLM：100K-200K（越大推理越快，但预训练 embedding 矩阵越大）

## 小结

| 算法 | 合并/裁剪标准 | 代表 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| BPE | 频率最高 | GPT-2, LLaMA, RoBERTa | 简单、训练快 | 贪心局部最优 |
| WordPiece | 互信息最大 | BERT | 语义相关合并 | 训练慢、需 DP |
| Unigram | 总似然最大 | T5, XLNet, SentencePiece | 概率化、可采样 | 训练最慢 |

子词切分看似"工程细节"，但它直接影响 LLM 的**数值稳定性、推理速度、多语言能力**。一个好的 tokenization 让模型的"每秒 token 数"翻倍的同时还能减少 OOV——这是预训练成本动辄百万美元时必须死磕的细节。下一篇我们将看到子词切分之上的**上下文相关词向量**：ELMo 与 BERT 如何让同一个词在不同句子里拥有不同的向量。
