# 统计机器翻译：从 IBM Models 到 Moses 的概率框架

神经机器翻译（NMT）在 2014-2016 年间全面取代统计机器翻译（SMT），但 SMT 留下的概念框架——对齐、语言模型、解码算法——仍是理解现代 MT 的基础。本文沿着 IBM Models（1990s）→ 短语翻译模型（2003）→ Moses（2007）→ 层次短语（2010s）的时间线，剖析 SMT 的核心思想、算法实现及其对 NMT 的影响。

## 一、机器翻译的两条主线

机器翻译的两大范式：

| 范式 | 时间 | 核心思想 | 代表 |
| --- | --- | --- | --- |
| 理性主义（规则） | 1950s-1990s | 词典 + 句法规则 + 转换 | Systran, Eurotra |
| 经验主义（统计） | 1990s-2016 | 概率模型 + 大规模平行语料 | IBM Models, Moses |
| 神经 | 2014- | 端到端神经网络 | Seq2Seq, Transformer, LLaMA |

理性主义派在 80 年代占主流，依赖语言学家手工编写规则，跨语言迁移代价巨大。Brown et al.（1990）在 IBM 提出的 IBM Models 开创了**经验主义范式**——只要有足够多的双语平行语料，机器能自动学出翻译模型。这一思想后来被深度学习继承：今天的 NMT 也是从平行语料中"学"翻译，只是模型更深、更强大。

## 二、噪声信道模型（Noisy Channel）

SMT 的数学基础是**噪声信道模型**。把源语言句子 $\mathbf{f}$ 看作"经过噪声信道"后得到的"破损"版本，原始信号是目标语言句子 $\mathbf{e}$：

$$
\mathbf{e}^* = \arg\max_{\mathbf{e}} P(\mathbf{e} \mid \mathbf{f}) = \arg\max_{\mathbf{e}} P(\mathbf{f} \mid \mathbf{e}) P(\mathbf{e})
$$

三项分别是：

- **翻译模型** $P(\mathbf{f} \mid \mathbf{e})$：给定目标句子 $\mathbf{e}$，生成源句子的概率。
- **语言模型** $P(\mathbf{e})$：目标句子本身的"流利度"。
- **解码**：找 $\arg\max$ 的目标句子。

直觉：好翻译 = "翻译模型分高" + "语言模型分高"。翻译模型保证"意思对"，语言模型保证"句子通顺"。

## 三、语言模型：n-gram 平滑

语言模型 $P(\mathbf{e}) = P(e_1) P(e_2 \mid e_1) \dots P(e_n \mid e_{n-1}, \dots, e_1)$ 在长序列下条件依赖爆炸。n-gram 假设近似：

$$
P(\mathbf{e}) \approx \prod_{i=1}^{n} P(e_i \mid e_{i-n+1}, \dots, e_{i-1})
$$

典型 5-gram + Kneser-Ney 平滑（Kneser & Ney, 1995）。Kneser-Ney 是公认最强的 n-gram 平滑方法，关键思想是**回退概率基于"前驱词的多样性"**，而非"前驱词的频率"。

## 四、IBM Models：对齐与翻译概率

IBM 在 1990 年代提出的 5 个模型（IBM Model 1-5）层层递进，每一层放松一个假设。

### 4.1 IBM Model 1：最朴素的词对齐

假设：每个源词 $\mathbf{f}_j$ 由**任一**目标词 $\mathbf{e}_a$ 生成，**对齐概率均匀**：

$$
P(\mathbf{f} \mid \mathbf{e}, a) = \prod_{j=1}^{m} P(f_j \mid e_{a_j})
$$

参数是 $P(f \mid e)$：给定目标词 $e$，生成源词 $f$ 的概率。所有对齐方式 $(m!)^m$ 等可能。

EM 算法估计参数：
- **E 步**：固定 $P(f \mid e)$，计算每个 $(f_j, e_a)$ 对应的后验对齐概率。
- **M 步**：用后验概率更新 $P(f \mid e)$。

EM 在 Model 1 上收敛很快（因为对齐空间虽然大，但梯度信号清晰）。

### 4.2 IBM Model 2：加入对齐位置偏好

Model 1 假设对齐完全随机，但实际中"目标第 $a$ 个词倾向于对齐到源第 $j$ 个位置"是有规律的。Model 2 加入对齐概率：

$$
P(a_j = i \mid j, m, n) \sim \text{某种对齐分布}
$$

直觉：句首的词倾向于对齐到句首，距离偏差 $\mid i - a \cdot m/n \mid$ 越小概率越高。

### 4.3 IBM Model 3：fertility（繁衍数）

引入 **fertility** $\phi(e \mid a)$：每个目标词 $e$ 生成多少个源词。

例如英文 "the" 在法译中可能对应 "le"、"la"、"les" 等不同零碎，fertility 模型能学出"the → 1 个法语词"是常态。

### 4.4 IBM Model 4 & 5：相对位置对齐

Model 4 引入**相对位置**——一个新源词对齐的位置取决于上一个对齐词的位置。Model 5 在此基础上加入"缺陷"（deficiency）概念。

实际工程中，IBM Models 通常按 1 → 2 → 3 → 4 顺序训练，每个模型的参数初始化为上一个的结果。这是 GIZA++（Och & Ney, 2003）的标准流程。

### 4.5 对齐评估

- **AER（Alignment Error Rate）**：人工标注对齐 vs 模型预测对齐的差异率。
- **IBM Model 4** 在英法对上 AER 约 8-12%，是当时最好的非神经方法。

## 五、对称化：生长-合并（Grow-Diagonal-Final）

IBM Models 给出 $P(\mathbf{f} \mid \mathbf{e})$ 的对齐，但实际中我们既想要 $\mathbf{f} \to \mathbf{e}$ 的对齐，也想要 $\mathbf{e} \to \mathbf{f}$ 的对齐。两个方向的对齐往往不一致。**生长-合并**算法（GDF, Och & Ney, 2003）：

1. 训练两个方向：$P(\mathbf{f} \mid \mathbf{e})$ 与 $P(\mathbf{e} \mid \mathbf{f})$。
2. 取两个方向对齐的**交集**作为种子。
3. 反复扩展：把与种子相邻的对齐点加入，直到收敛。
4. 最终用启发式合并相邻对齐点。

这一步对最终翻译质量至关重要——好的对齐才能学到好的短语翻译表。

## 六、短语翻译模型（Phrase-Based MT）

IBM Models 的局限：只建模**词级**翻译，但许多翻译是短语级的（如"中华人民共和国" ↔ "People's Republic of China"）。**短语翻译模型**（Koehn et al., 2003, Och & Ney, 2004）解决了这个问题。

### 6.1 短语对的提取

给定对齐矩阵 $\mathcal{A}$（哪些源词与哪些目标词互相对齐），提取所有**短语对**：

- 若短语 $\bar{f} = (f_i, \dots, f_{i+k})$ 与 $\bar{e} = (e_j, \dots, e_{j+l})$ 的所有词都在 $\mathcal{A}$ 内对齐，则它们是一个合法短语对。
- 提取所有可能的短语对（通常长度 ≤ 7）。

### 6.2 短语翻译概率

给定训练语料，统计每个短语对 $(\bar{f}, \bar{e})$ 的出现次数，估计两个方向的条件概率：

$$
P(\bar{f} \mid \bar{e}) = \frac{\text{count}(\bar{f}, \bar{e})}{\sum_{\bar{f}'} \text{count}(\bar{f}', \bar{e})}
$$
$$
P(\bar{e} \mid \bar{f}) = \frac{\text{count}(\bar{f}, \bar{e})}{\sum_{\bar{e}'} \text{count}(\bar{f}, \bar{e}')}
$$

以及词汇权重（lexical weighting）：

$$
\text{lex}(\bar{f} \mid \bar{e}, a) = \prod_{j=i}^{i+k} \frac{1}{\mid \{i' \mid (i', j) \in \mathcal{A}\} \mid} \sum_{i'} P(f_j \mid e_{i'})
$$

直觉：短语翻译概率反映"哪个翻译最常见"，词汇权重反映"对齐的合理性"。

### 6.3 对数线性模型与特征工程

Och & Ney（2002）提出**对数线性模型**统一多个特征：

$$
P(\mathbf{e} \mid \mathbf{f}) \propto \exp \sum_{m=1}^{M} \lambda_m h_m(\mathbf{e}, \mathbf{f})
$$

$M$ 个特征 $h_m$ 各自承担一个翻译能力：

1. **短语翻译概率** $P(\bar{f} \mid \bar{e})$ 与 $P(\bar{e} \mid \bar{f})$。
2. **词汇权重**（两个方向）。
3. **语言模型** $\log P(\mathbf{e})$。
4. **重排模型**（reordering model）：惩罚调序跨度。
5. **句子长度惩罚**：避免过短或过长翻译。

$\lambda_m$ 在开发集上用 **Minimum Error Rate Training**（MERT, Och 2003）调优——直接以 BLEU 分数为目标。

### 6.4 解码：栈解码（Stack Decoding）

短语翻译的解码是**NP 难问题**——必须搜索整个翻译空间。实用算法：

- **栈解码**（stack decoding）：维护一个候选翻译堆，按"成本"扩展。
- **Cube pruning**（Huang & Chiang, 2007）：剪枝低概率分支，加速解码。
- **Beam search**：维护 top-K 个候选，每步扩展。

典型解码速度：Moses 在新闻测试集上每秒解码 5-20 句（CPU 模式）。

## 七、层次短语模型（Hierarchical Phrase-Based MT）

短语翻译模型只能处理**局部调序**。处理远程调序（如英德 SOV 语序转换）需要**层次短语**（Chiang, 2005, Hiero）。

核心思想：在普通短语规则之外，加入**带变量的规则**：

```
X → 〈X1〉 的 〈X2〉    [de 的]
X → 〈X1〉 〈X2〉 | 〈X2〉 〈X1〉  [调序]
```

规则以同步上下文无关文法（SCFG）形式表达，变量 $X$ 可以递归展开。解码用 CYK + cube pruning。

层次短语让翻译模型能处理长距离调序，是 Moses 后期默认支持的模型。

## 八、Moses：开源统计机器翻译工具包

Moses（Koehn et al., 2007）是 SMT 时代最具影响力的开源工具包，标准化了训练 + 解码流程：

```
parallel corpus → GIZA++（IBM Models） → 短语抽取 → MERT（调参） → 解码器
```

典型工作流：
1. **数据预处理**：tokenization、lower-casing、长句切分（>80 token 切句）。
2. **词对齐**：GIZA++ 双向训练 + 合并。
3. **短语抽取**：从对齐矩阵中提取短语对。
4. **调参**：MERT / MIRA / PRO 在 dev 集上优化权重。
5. **解码**：用训练好的模型对测试句翻译。

Moses 在工业界用了 10 多年（2007-2017），至今部分低资源语言对仍在使用。

## 九、评估：BLEU 与人工评测

### 9.1 BLEU（Papineni et al., 2002）

最经典的机器翻译自动评估指标。计算 n-gram（n=1,2,3,4）的精确率，再乘以**短句惩罚**（brevity penalty, BP）：

$$
\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^{4} \frac{1}{4} \log p_n \right)
$$

其中 $p_n$ 是 n-gram 精确率——统计候选翻译中每个 n-gram 在参考翻译中出现的次数 / 候选中总 n-gram 数。

直觉：好翻译的 n-gram 都能在参考翻译中找到。

### 9.2 BLEU 的局限性

- **不区分语义**：同义替换会被罚。
- **不评估流畅度**：纯靠 n-gram 重合。
- **不评估语法错误**：句法错误有时不影响 BLEU。

后续工作：

- **METEOR**（Banerjee & Lavie, 2005）：引入 WordNet 同义词和词干匹配。
- **chrF**（Popović, 2015）：基于字符 n-gram，对形态学丰富语言更友好。
- **BERTScore**（Zhang et al., 2020）：用 BERT 表示计算余弦相似度，语义层面评估。
- **BLEURT**（Sellam et al., 2020）：用监督学习训练的 BERT 评估模型。

### 9.3 人工评估

- **Direct Assessment (DA)**：0-100 分的人工评分，取多人平均。
- **Pairwise Comparison**：A/B 测试，比较两个系统的翻译质量。

DA 与自动指标的相关性约 0.6-0.8，但人工成本高、主观性强。

## 十、SMT 的遗产与局限

### 10.1 遗产

- **语言模型的独立性**：把"流利度"与"翻译"分离建模的思想被 NMT 继承（如 dual encoder + LM 融合）。
- **对齐的概念**：NMT 里的 attention 机制本质是"软对齐"。
- **对数线性模型**：多个特征加权融合的方式被 BERT 微调的多任务学习借鉴。
- **BLEU 评估**：至今仍是 MT 评测的事实标准。

### 10.2 局限

- **特征工程**：每个特征都需手工设计，难以扩展。
- **局部最优**：短语翻译模型不能处理长距离调序（虽然层次短语部分缓解）。
- **数据效率低**：需要百万级平行句对才能训好，远超神经模型的样本效率。
- **未见词问题**：词表外（OOV）的词完全无法翻译。

## 十一、SMT → NMT 的过渡

2014 年，Sutskever et al. 用 Seq2Seq（LSTM encoder-decoder）做翻译，在 WMT'14 英法上 BLEU 达 34.8，首次超越 Moses 的 33.3。这是 NLP 翻译史的转折点。2016 年 Google Translate 全面切换到 NMT（GNMT, Wu et al., 2016）。

但 SMT 的概念框架——**对齐、语言模型、解码、评估**——从未过时。它们换了一种形式出现在 NMT 里：attention 替代了 IBM Models，神经 LM 替代了 n-gram，beam search 替代了 stack decoding，BLEU 仍是评估标配。下一篇我们将详细剖析神经机器翻译的 Seq2Seq 与 Attention 框架。
