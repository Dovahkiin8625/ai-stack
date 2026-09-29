// 在 resources/knowledge/ 下生成 10 篇真实内容的 AI 学习笔记
import { mkdir, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';

const ROOT = 'resources/knowledge';

// ========== Category 1: Mathematics ==========
const linearAlgebra = `# 线性代数基础

线性代数是机器学习的"语言"。几乎所有现代 AI 模型——从线性回归到 Transformer——都可以被表达为向量、矩阵和张量上的运算。本文梳理四个最重要的概念：向量、矩阵、特征值与奇异值分解（SVD），并通过小例子说明它们为何重要。

## 一、数据即矩阵

在机器学习中，我们通常把每一行当作一个样本，每一列当作一个特征。给定 $n$ 个样本、$d$ 个特征的数据集，可以用一个 $n \\times d$ 的矩阵 $X$ 表示。预测目标也常常写作向量 $y \\in \\mathbb{R}^n$。这种"表格思维"让线性代数天然契合 ML：模型参数是一组权重向量，前向传播就是矩阵乘法。

## 二、向量与矩阵运算

两个向量的点积 $\\mathbf{a} \\cdot \\mathbf{b} = \\sum_i a_i b_i$ 度量了它们的方向一致性。点积越大，方向越接近。Cosine 相似度、注意力分数，本质上都是点积的归一化形式。

矩阵乘法是把线性变换"串起来"的工具。例如，给定：

\`\`\`math
A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix},\\quad
B = \\begin{pmatrix} 5 & 6 \\\\ 7 & 8 \\end{pmatrix}
\`\`\`

那么 $AB$ 的计算方式为：

\`\`\`math
AB = \\begin{pmatrix} 1\\cdot5+2\\cdot7 & 1\\cdot6+2\\cdot8 \\\\ 3\\cdot5+4\\cdot7 & 3\\cdot6+4\\cdot8 \\end{pmatrix}
= \\begin{pmatrix} 19 & 22 \\\\ 43 & 50 \\end{pmatrix}
\`\`\`

注意矩阵乘法不满足交换律，即 $AB \\neq BA$。理解这一点对理解神经网络中"权重矩阵左乘输入"很重要。

## 三、特征值与特征向量

对于方阵 $A$，若存在非零向量 $v$ 和标量 $\\lambda$ 使得 $Av = \\lambda v$，则 $v$ 是特征向量，$\\lambda$ 是特征值。直观上，特征向量是矩阵作用后"方向不变"的向量；特征值衡量被拉伸的倍数。

以 $A = \\begin{pmatrix} 2 & 0 \\\\ 0 & 3 \\end{pmatrix}$ 为例，沿 $x$ 轴的向量被拉伸 2 倍，沿 $y$ 轴的向量被拉伸 3 倍，因此特征值为 2 和 3，对应特征向量分别是 $(1, 0)^\\top$ 和 $(0, 1)^\\top$。

在 ML 中，特征值分析用于 PCA、主成分分析、共轭梯度下降收敛性分析等场景。

## 四、SVD 与降维

奇异值分解把任意矩阵 $M \\in \\mathbb{R}^{m \\times n}$ 写成 $M = U \\Sigma V^\\top$，其中 $\\Sigma$ 是对角矩阵，对角线上是按大小排列的奇异值。SVD 的一个核心用途是低秩近似：只保留前 $k$ 个奇异值，可以用最小的秩-$k$ 矩阵逼近 $M$。

\`\`\`python
import numpy as np
X = np.random.randn(100, 20)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
X_approx = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]
\`\`\`

这正是推荐系统、主题模型、以及 LLM 权重压缩背后的数学工具。

## 小结

向量和点积让我们能度量相似度；矩阵乘法串联线性变换；特征值揭示矩阵结构；SVD 提供降维与压缩。掌握这四块，你就具备了阅读 90% ML 论文底层的数学直觉。
`;

const calculus = `# 机器学习中的微积分

微积分是"学习"的数学基础。模型训练本质是寻找让损失函数最小的参数，而"怎么走"由梯度决定。本文覆盖导数、偏导、梯度、链式法则，并通过小例子演示梯度下降如何工作。

## 一、导数与偏导数

单变量函数 $f(x)$ 的导数 $f'(x)$ 描述 $f$ 在 $x$ 处的瞬时变化率：

\`\`\`math
f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}
\`\`\`

当函数依赖多个变量时，我们用偏导数 $\\partial f / \\partial x_i$ 描述在某个坐标方向上的变化率。把所有偏导数拼起来，就是

\`\`\`math
\\nabla f = \\left( \\frac{\\partial f}{\\partial x_1}, \\frac{\\partial f}{\\partial x_2}, \\dots, \\frac{\\partial f}{\\partial x_n} \\right)
\`\`\`

这是 ML 中最常见的梯度形式。

## 二、链式法则

深度学习的关键在于：模型是层层嵌套的复合函数。链式法则告诉我们如何"把梯度传回去"。

若 $y = f(g(x))$，则

\`\`\`math
\\frac{dy}{dx} = \\frac{df}{dg} \\cdot \\frac{dg}{dx}
\`\`\`

多变量版本：若 $\\mathbf{y} = f(\\mathbf{u}), \\mathbf{u} = g(\\mathbf{x})$，则

\`\`\`math
\\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{x}} = \\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{u}} \\cdot \\frac{\\partial \\mathbf{u}}{\\partial \\mathbf{x}}
\`\`\`

这就是反向传播（Backpropagation）的数学原理。

## 三、梯度下降

梯度下降通过迭代更新参数来最小化损失 $L(\\theta)$：

\`\`\`math
\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta L(\\theta_t)
\`\`\`

其中 $\\eta$ 是学习率。直观上，梯度指向函数上升最快的方向，所以减去梯度就走到了下降最快的方向。

以 $L(\\theta) = (\\theta - 3)^2$ 为例，$\\nabla L = 2(\\theta - 3)$。从 $\\theta_0 = 0$、$\\eta = 0.1$ 开始：

\`\`\`
step 0: theta = 0.00,  loss = 9.00
step 1: theta = 0.60,  loss = 5.76
step 2: theta = 1.08,  loss = 3.69
...
step 20: theta ≈ 2.94, loss ≈ 0.0036
\`\`\`

可以看到 $\\theta$ 不断逼近 3，损失不断下降。

## 四、常见陷阱与改进

- **学习率过大**：参数会在最优值附近震荡甚至发散。
- **学习率过小**：收敛极慢。
- **局部极小 / 鞍点**：高维非凸函数中常见；带动量（Momentum）、Adam 等优化器能缓解。
- **梯度消失 / 爆炸**：深层网络会出现，使用残差连接、归一化、合适的激活函数可以改善。

## 五、激活函数与梯度流

激活函数 $\\sigma(x)$ 的导数直接决定反向传播时梯度的"乘子"。一个经典的例子是 sigmoid：

\`\`\`math
\\sigma(x) = \\frac{1}{1 + e^{-x}}, \\quad \\sigma'(x) = \\sigma(x)(1 - \\sigma(x))
\`\`\`

由于 $\\sigma'(x) \\le 0.25$，梯度经过多层 sigmoid 后会指数级缩小——这就是 sigmoid 深层网络中"梯度消失"的根源之一。ReLU 在正区间导数为常数 1，因此在现代网络中很大程度上缓解了这个问题。GeLU、SwiGLU 等更平滑的替代品则在保持梯度通畅的同时引入轻微非线性。理解这一点，是从"模型能跑"过渡到"模型训得稳"的关键一步。

## 小结

偏导给出方向，梯度把所有方向整合起来，链式法则让深层网络可训练，梯度下降负责实际更新，激活函数的导数决定了梯度能否"流到"底层参数。理解了这五件事，你就具备了"读懂训练循环"的数学基础。
`;

const probability = `# 概率与统计：机器学习的语言

概率论是 AI 不确定性的来源，也是我们量化"模型多自信"的工具。本文从随机变量讲起，介绍常见分布、贝叶斯定理、期望与方差，再引入信息论中的熵和 KL 散度，最后说明这些概念如何出现在现代 ML 中。

## 一、随机变量与分布

随机变量 $X$ 取值具有不确定性，由一个概率分布描述。常见分布：

- **伯努利分布** $\\text{Bern}(p)$：二值结果（成功/失败），如点击预测、token 是否被掩码。
- **类别分布** $\\text{Cat}(p_1, \\dots, p_K)$：多类结果的推广，是分类任务输出层 softmax 的目标分布。
- **高斯分布** $\\mathcal{N}(\\mu, \\sigma^2)$：连续变量最常见，权重初始化、VAE 中的隐变量都假设近似高斯。
- **指数族分布**：包括伯努利、高斯、泊松等，GLM 与极大似然估计都建立在它之上。

## 二、贝叶斯定理

\`\`\`math
P(A \\mid B) = \\frac{P(B \\mid A) P(A)}{P(B)}
\`\`\`

它把"已知结果反推原因"这件事形式化。在 ML 中的应用：

- **朴素贝叶斯分类器**：直接应用贝叶斯定理 + 特征条件独立假设。
- **后验采样**：贝叶斯神经网络、贝叶斯优化都关心参数的后验 $P(\\theta \\mid D)$。
- **语言模型中的 smoothing**：n-gram 模型用贝叶斯思路避免零概率。

## 三、期望、方差与协方差

期望 $\\mathbb{E}[X]$ 是分布的"中心"，方差 $\\text{Var}(X)$ 是"分散程度"。

\`\`\`math
\\mathbb{E}[X] = \\sum_x x P(x),\\quad
\\text{Var}(X) = \\mathbb{E}[(X - \\mathbb{E}[X])^2]
\`\`\`

两个变量的协方差 $\\text{Cov}(X,Y)$ 衡量它们线性相关的强度和方向。在 PCA 中，我们对协方差矩阵做特征分解来寻找主方向。

## 四、信息论基础

**熵** 衡量分布的不确定性：

\`\`\`math
H(P) = -\\sum_x P(x) \\log P(x)
\`\`\`

**交叉熵** 是分类任务最常用的损失：

\`\`\`math
H(P, Q) = -\\sum_x P(x) \\log Q(x)
\`\`\`

训练时 $P$ 是真实 one-hot 标签，$Q$ 是模型预测分布，最小化交叉熵等价于极大似然估计。

**KL 散度** $D_{KL}(P \\Vert Q) = H(P, Q) - H(P)$ 衡量两个分布的差异。变分自编码器（VAE）、RLHF 中的偏好模型、知识蒸馏都用到了 KL。

## 小结

随机变量 + 分布描述数据；贝叶斯定理连接先验与后验；期望方差刻画统计量；熵与 KL 散度为训练目标提供数学支撑。这些工具组合在一起，构成了几乎所有生成式模型的概率骨架。
`;

// ========== Category 2: Transformers ==========
const attentionExplained = `# 注意力机制详解

Transformer 是过去十年最具影响力的神经网络架构，而注意力机制（Attention）是它的核心。本文从直觉出发，推导缩放点积注意力公式，并通过一个 3-token 的小例子演示注意力权重是如何计算出来的。

## 一、Query、Key、Value 的直觉

把注意力想象成"信息检索"：

- **Query（查询，Q）**：当前 token 想问的问题。
- **Key（键，K）**：每个 token 能提供的"标签"。
- **Value（值，V）**：每个 token 真正承载的信息。

计算过程：先用 $Q$ 去和每个 $K$ 比相似度，得到一组权重，再用这些权重对 $V$ 做加权平均。整个过程可以并行化，是 Transformer 相对 RNN 的最大优势。

## 二、缩放点积注意力公式

\`\`\`math
\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left( \\frac{Q K^\\top}{\\sqrt{d_k}} \\right) V
\`\`\`

步骤拆解：

1. 计算 $Q K^\\top$，得到一个形状为 $(\\text{seq\\_len}, \\text{seq\\_len})$ 的相似度矩阵。
2. 除以 $\\sqrt{d_k}$，避免点积过大导致 softmax 饱和。
3. 按行做 softmax，得到归一化权重。
4. 用权重乘 $V$，得到每个位置的加权和。

## 三、为什么需要除以 $\\sqrt{d_k}$

假设 $Q$、$K$ 的每个分量是独立零均值、方差为 1 的随机变量，则 $q \\cdot k$ 的方差约为 $d_k$。当 $d_k$ 较大（如 64、128）时，点积会落在很大的数值范围，让 softmax 输出接近 one-hot，反向传播梯度也会变得极小。除以 $\\sqrt{d_k}$ 后方差回到 $\\sim 1$，训练更稳定。

## 四、多头注意力

单一注意力头只能捕捉一种关系模式；多头注意力把 $Q, K, V$ 切成 $h$ 份独立投影，让每份各自做注意力，再拼接回原维度：

\`\`\`math
\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O
\`\`\`

其中 $\\text{head}_i = \\text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$。

直觉上不同头可以学习不同语法或语义关系，例如一个头关注局部依赖，另一个头关注长距离指代。

## 五、3-token 小例子

设三个 token，每个 $d_k = 2$，初始 $Q = K = V$：

$$Q = K = V = \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 1 & 1 \\end{pmatrix}$$

1. $QK^\\top = \\begin{pmatrix} 1 & 0 & 1 \\\\ 0 & 1 & 1 \\\\ 1 & 1 & 2 \\end{pmatrix}$
2. 除以 $\\sqrt{2}$：数值约为 $\\begin{pmatrix} 0.71 & 0 & 0.71 \\\\ 0 & 0.71 & 0.71 \\\\ 0.71 & 0.71 & 1.41 \\end{pmatrix}$
3. 行 softmax（以第一行为例）：$\\text{exp}(0.71)\\approx 2.03$，$\\text{exp}(0) = 1$，$\\text{exp}(0.71)\\approx 2.03$，归一化得 $(0.40,\\ 0.20,\\ 0.40)$。
4. 用这组权重乘 $V$，得到 token 1 的新表示 $(0.40 \\cdot 1 + 0.20 \\cdot 0 + 0.40 \\cdot 1,\\ 0.40 \\cdot 0 + 0.20 \\cdot 1 + 0.40 \\cdot 1) = (0.80,\\ 0.60)$。

可以看到，token 1 的新表示融合了自身（权重 0.40）和 token 3（权重 0.40）的内容，仅少量受 token 2 影响。

## 小结

Q/K/V 让模型学会"按需查找"；缩放让训练稳定；多头让模型并行学习多种关系。掌握这些，你就已经理解了 Transformer 的"心脏"。
`;

const transformerArch = `# Transformer 架构详解

上一篇文章介绍了注意力机制，本文把它"装进"完整的 Transformer 中。我们会逐层拆解 encoder、decoder、关键子模块（位置编码、层归一化、残差连接、前馈网络），最后走一遍端到端的前向传播。

## 一、整体结构

\`\`\`
                ┌──────────────────────┐
                │       Encoder        │
                │  ┌────────────────┐  │
input tokens → │ Embedding + Pos │  │
                │  └─────┬────────┘  │
                │        ▼            │
                │  ┌─────────────┐    │
                │  │ Multi-Head  │    │ × N
                │  │ Self-Attn   │    │
                │  └─────┬───────┘    │
                │  ┌─────▼───────┐    │
                │  │ Feed-Forward│    │
                │  └─────────────┘    │
                └─────────┬───────────┘
                          │  context
                          ▼
                ┌──────────────────────┐
                │       Decoder        │
                │  Masked Self-Attn    │
                │  Cross-Attn (Q←K,V)  │
                │  Feed-Forward        │
                └──────────────────────┘
                          ▼
                       Output
\`\`\`

编码器把输入序列压成富含上下文的表示；解码器一边自回归生成，一边通过 cross-attention "回头看"编码器输出。

## 二、位置编码

注意力是置换不变的——它天然忽略 token 顺序。我们必须显式注入位置信息。

- **正弦位置编码（Sinusoidal）**：用不同频率的正弦/余弦函数生成位置向量，公式：

\`\`\`math
PE_{(pos, 2i)}   = \\sin(pos / 10000^{2i/d})
PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d})
\`\`\`

- **RoPE（旋转位置编码）**：现代 LLM 常用，通过把 Q/K 视作复数并按位置旋转，把相对位置信息编码进点积。

## 三、子模块组合

每个 encoder/decoder block 都包含以下关键子层（顺序以原始论文为参考）：

1. **多头自注意力**：见上一篇文章。
2. **残差连接 + 层归一化**：$\\text{LayerNorm}(x + \\text{Sublayer}(x))$，缓解梯度消失、加速收敛。
3. **逐位置前馈网络（FFN）**：两层全连接 $+\\ \\text{ReLU}$（或 GeLU/SwiGLU）。它提供非线性，是模型记忆事实的关键。

## 四、Masked Self-Attention

解码器在生成第 $t$ 个 token 时，不能"看到"未来的 token。做法是在 softmax 之前对未来位置加上一个很大的负数：

\`\`\`math
\\text{mask}(M_{ij}) = \\begin{cases} 0, & j \\le i \\\\ -\\infty, & j > i \\end{cases}
\`\`\`

这样训练和推理都自回归地逐位预测。

## 五、端到端前向传播（以翻译为例）

1. **源语言 token 化**：例如 "I love you" → $[I, \\text{love}, you]$。
2. **embedding + 位置编码**：得到 $X \\in \\mathbb{R}^{3 \\times d}$。
3. **编码器**：经 N 个 block 处理，得到上下文表示 $C$。
4. **解码器输入**：已生成的目标 token（推理时为当前序列；训练时用 teacher forcing）。
5. **Masked self-attention**：目标序列内部依赖建模。
6. **Cross-attention**：以解码器状态为 Q，$C$ 提供 K/V。
7. **FFN + 残差 + LayerNorm**。
8. **线性 + Softmax**：输出下一个 token 的概率分布。
9. **重复 5-8** 直到生成 EOS 标记。

## 小结

Transformer = 位置编码 + 多头注意力 + FFN + 残差 + LayerNorm。这套组合既可并行训练，又能捕捉长距离依赖，是它统治 NLP 领域十年的关键。下一篇我们会沿着这条主线，比较 GPT、BERT 与 LLaMA 的演进。
`;

const gptBert = `# GPT 与 BERT 的演化

自 2017 年 Transformer 诞生起，整个 NLP 领域围绕一个问题展开：**预训练时应该让模型做什么任务？** 这个问题催生了三大范式：仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5/BART）。本文梳理它们的设计差异、训练目标，以及为何今天的大模型几乎都走上了"decoder-only"的路。

## 一、BERT：双向编码器

Google 2018 年发布的 BERT 采用**仅编码器**结构。它的核心创新是**掩码语言模型（Masked Language Modeling, MLM）**：随机遮住输入中 15% 的 token，让模型根据左右两侧上下文预测被遮住的词。

辅助任务还包括**下一句预测（NSP）**：判断两段文本是否相邻，强化句间关系建模。

由于 encoder 天然允许每个位置看到完整上下文，BERT 特别适合理解类任务：文本分类、问答、命名实体识别。Google 后续也推出了 RoBERTa、ALBERT、DeBERTa 等改进。

## 二、GPT：自回归解码器

OpenAI 的 GPT 系列采用**仅解码器**结构。预训练目标是经典的**因果语言模型（Causal Language Modeling, CLM）**：给定前文 $x_1, \\dots, x_{t-1}$，预测下一个 token $x_t$：

\`\`\`math
\\mathcal{L}_{\\text{CLM}} = -\\sum_t \\log P_\\theta(x_t \\mid x_{<t})
\`\`\`

这种目标天然契合生成任务——只要逐 token 自回归采样即可。GPT-2 / GPT-3 通过把规模放大到 175B 参数，展示了**少样本学习**的涌现能力。

## 三、T5 / BART：编码器-解码器

Google 的 T5 与 Facebook 的 BART 采用完整的 encoder-decoder。训练目标多样：

- 文本 span corruption（类似 MLM 但生成式）
- 翻译、摘要等监督任务
- 去噪自编码

它们擅长"输入 → 输出"的转换类任务，例如翻译、摘要、问答生成。

## 四、为什么 decoder-only 赢了

到 2020 年后，社区几乎一致转向 decoder-only。原因有几条：

1. **统一性**：CLM 一个目标既能做理解又能做生成，而 MLM 只能填空。
2. **规模效率**：decoder-only 的 in-context learning 随规模呈涌现式增长。
3. **推理效率**：decoder 自回归生成可直接复用 KV cache，无需额外处理 encoder 输出。
4. **生态惯性**：开源社区（Llama、Mistral、Qwen）几乎全是 decoder-only，下游工具链（vLLM、TGI）也围绕它优化。

但 encoder-only 也有自己的位置：BERT 类小模型在分类、检索 embedding 上至今仍是性价比最高的选择。

## 五、范式对比表

| 范式 | 代表模型 | 预训练目标 | 擅长任务 |
| --- | --- | --- | --- |
| Encoder-only | BERT, RoBERTa | MLM + NSP | 分类、检索、NER |
| Decoder-only | GPT, LLaMA | CLM | 生成、对话、in-context learning |
| Encoder-Decoder | T5, BART | Span corruption | 翻译、摘要、seq2seq |

## 小结

MLM 与 CLM 的差异看似微小，却塑造了两种截然不同的模型行为。今天的大模型主流选择是 decoder-only + CLM，但理解 BERT 这条线，仍然是理解"模型如何表征语言"的关键基础。
`;

const gptVsLlama = `# GPT vs LLaMA：现代开源 LLM 的架构差异

GPT-3 之后，开源社区以 Meta 的 LLaMA 系列为代表，迅速把"decoder-only + 自回归"的范式推向工业级。本文聚焦四组核心架构差异：位置编码、归一化、激活函数、注意力优化，并附上 GPT-3 / LLaMA-2 / LLaMA-3 的简要对比表。

## 一、位置编码：绝对 vs RoPE

- **GPT-3 使用绝对位置编码**：每个位置对应一段可学习的 embedding，与 token embedding 相加。问题是对训练时没见过的长度泛化能力弱。
- **LLaMA 使用 RoPE（Rotary Position Embedding）**：把 Q、K 视作复数并按位置旋转，使得两点积天然包含相对位置信息。RoPE 对长度外推更友好，也是今天几乎所有主流开源 LLM 的默认选择。

## 二、归一化：LayerNorm vs RMSNorm

GPT 系列使用经典 LayerNorm：

\`\`\`math
\\text{LayerNorm}(x) = \\gamma \\cdot \\frac{x - \\mu}{\\sigma} + \\beta
\`\`\`

LLaMA 使用 **RMSNorm**：

\`\`\`math
\\text{RMSNorm}(x) = \\gamma \\cdot \\frac{x}{\\sqrt{\\text{mean}(x^2) + \\epsilon}}
\`\`\`

RMSNorm 去掉了均值中心化步骤，只缩放。在数学上仍能稳定训练，且计算更快，对 LLM 这种"小算力大参数"的场景非常合适。

## 三、激活函数：GeLU vs SwiGLU

- **GPT-3 FFN**：GeLU 激活，结构为 $\\text{FFN}(x) = W_2 \\cdot \\text{GeLU}(W_1 x + b_1) + b_2$。
- **LLaMA FFN**：使用 **SwiGLU**，三组权重：

\`\`\`math
\\text{SwiGLU}(x) = \\text{Swish}_\\beta(xW_1) \\otimes (xW_2) \\cdot W_3
\`\`\`

其中 $\\text{Swish}_\\beta(x) = x \\cdot \\sigma(\\beta x)$，$\\otimes$ 是逐元素乘。SwiGLU 引入一个"门控"分支，让信息流更可控，论文与开源实现都报告了稳定的质量提升。

## 四、注意力：MHA vs GQA

- **GPT-3** 使用标准多头注意力（MHA），每个 head 都有独立的 Q/K/V 投影。
- **LLaMA-2** 在 70B 规模仍使用 MHA。
- **LLaMA-3 与更新版本** 采用 **GQA（Grouped-Query Attention）**：Q 头数 $h_q$ 较大，K/V 头数 $h_{kv}$ 较小（通常 $h_{kv} = h_q / 4$ 或 $h_q / 8$）。

直观上，K/V 投影是显存和 KV cache 的主要开销；GQA 用更少的 K/V 头数换来几乎不损失的质量。推理时 KV cache 大小可减少 4-8 倍，长上下文推理吞吐显著提升。

## 五、模型对比

| 项 | GPT-3 (2020) | LLaMA-2 (2023) | LLaMA-3 (2024) |
| --- | --- | --- | --- |
| 位置编码 | 绝对 | RoPE | RoPE |
| 归一化 | LayerNorm | RMSNorm | RMSNorm |
| FFN 激活 | GeLU | SwiGLU | SwiGLU |
| 注意力 | MHA | MHA (70B) / GQA | GQA |
| 上下文长度 | 2K | 4K | 8K+ |
| 训练 token 数 | ~300B | ~2T | ~15T |

## 六、KV Cache 与推理

GQA 与上述其他优化共同支撑了一个事实：**推理时 KV cache 的体积决定吞吐**。当上下文从 4K 扩到 128K，KV cache 是显存的主要消耗项。GQA + 8-bit KV 量化 + PagedAttention（vLLM）这类组合，已成为生产级 LLM 服务的事实标准。

## 小结

RoPE、RMSNorm、SwiGLU、GQA 共同构成了"现代开源 LLM"的标志特征。它们的设计哲学都是：**用更便宜的结构换可衡量的质量或吞吐改进**。这套组合拳也是 Mistral、Qwen、DeepSeek 等后续开源模型共同采纳的路线。
`;

// ========== Category 3: RAG ==========
const ragFundamentals = `# RAG 基础：检索增强生成

大模型知识陈旧、容易"幻觉"、又接触不到私有数据。检索增强生成（Retrieval-Augmented Generation, RAG）通过在生成前先"翻资料"，缓解了这些问题。本文介绍 RAG 的核心动机、典型 pipeline，以及工程上最常见的几个坑。

## 一、为什么需要 RAG

LLM 有三大局限：

1. **知识截止**：训练数据停留在某个时间点，无法回答之后发生的事件。
2. **幻觉**：在没有确切答案时仍会"自信地胡说"。
3. **私有数据**：模型无法直接访问企业内部文档、数据库等专有信息。

RAG 的思路很简单：**让模型在回答前先查一份可控的知识库**。这样既能"喂"给它最新或私有的事实，又能在一定程度上用原文约束生成，减少幻觉。

## 二、典型 pipeline

\`\`\`
文档 → 切块 → embedding → 向量库
                              ↑
用户问题 → 检索（top-k） → 重排（可选）→ Prompt 组装 → LLM → 答案
\`\`\`

下面逐阶段说明。

### 1. 切块（Chunking）

文档太长了，模型上下文窗口和 embedding 模型都有长度限制，所以必须切块。常见策略：

- **固定长度切块**：每 N 个字符或 token 切一刀。
- **按段落 / 标题切**：保留语义结构。
- **滑动窗口**：相邻块之间保留 overlap，避免句子被截断。

### 2. Embedding

用 embedding 模型（如 BGE、OpenAI text-embedding-3、bge-m3）把每个 chunk 映射成一个向量。语义相近的文本向量距离更近。

### 3. 存储与检索

把向量连同原文一起存入向量数据库（FAISS、Milvus、pgvector、Qdrant 等）。检索时计算查询向量与库中向量的余弦相似度或点积，取 top-k。

### 4. Prompt 组装

把检索到的若干 chunk 拼成上下文，连同用户问题一起送给 LLM。常用模板：

\`\`\`
你是一名知识助手。请仅根据以下参考资料回答用户问题，
如果参考资料中找不到答案，请说"我不知道"。

参考资料：
[1] {{chunk_1}}
[2] {{chunk_2}}
...

问题：{{user_question}}
\`\`\`

### 5. 生成

LLM 在上下文约束下生成最终答案。可在 prompt 中要求它**给出引用编号**，便于溯源。

## 三、常见坑

1. **块边界切断语义**：例如表格被劈成两半，正则表达式被切断。解决办法：按语义单元切，或对表格 / 代码单独处理。
2. **embedding 模型选错**：通用多语 embedding 不一定适合特定领域。垂直领域最好微调或在语料上评估。
3. **检索 top-k 过小或过大**：过小易漏信息，过大引入噪声。一般 4-10 起步，再按延迟与质量权衡。
5. **不评估就上线**：没有离线评估的 RAG 系统很容易"看上去工作"，实则在边界场景一塌糊涂。
6. **元数据缺失**：chunk 应保留 source、section、page 等元数据，方便引用和后续过滤。

## 小结

RAG 的核心是把"记忆"外置到向量库，让 LLM 退化成"理解 + 引用 + 总结"的组件。下一篇我们将介绍更复杂的进阶模式：重排序、混合检索、查询改写与多跳推理。
`;

const advancedRag = `# 进阶 RAG 模式

朴素 RAG 在很多业务上"够用"，但要追求更高的召回率、答案准确率或复杂推理能力，就需要更聪明的检索与编排策略。本文介绍五种被广泛验证有效的进阶模式，并简述它们何时值得使用。

## 一、Re-ranking（重排序）

朴素检索只用单一相似度（cosine 或点积）排序，对短查询与长文档之间的语义鸿沟常常束手无策。Re-ranking 在初步召回的 top-k 候选上，再用一个**更强的 cross-encoder** 给每对 (query, chunk) 打一个更精细的相关性分。

\`\`\`python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
hits = retriever.search(query, top_k=20)
scores = reranker.predict([(query, h.text) for h in hits])
hits = sorted(zip(hits, scores), key=lambda x: -x[1])[:5]
\`\`\`

**何时用**：检索质量是瓶颈、延迟预算允许（cross-encoder 比 bi-encoder 慢 10-100 倍）。

## 二、混合检索（BM25 + Dense）

BM25 是经典的稀疏检索算法，基于词频与文档长度，对精确术语、人名、产品型号特别敏感；dense retrieval 擅长语义匹配但对稀有词不友好。混合检索对两者结果做加权融合：

\`\`\`math
\\text{score}(q, d) = \\alpha \\cdot \\text{BM25}(q, d) + (1 - \\alpha) \\cdot \\text{cosine}(q, d)
\`\`\`

**何时用**：领域含有大量专业术语或代码标识符。

## 三、查询改写 / HyDE

用户的查询常常很口语化甚至有错别字，导致检索不到。常见补救：

- **Query rewriting**：用 LLM 把用户原始问题改写成更适合检索的形式，例如补充同义词、拆分复合问题。
- **HyDE（Hypothetical Document Embeddings）**：让 LLM 先凭空生成一篇"假想的"答案文档，再把这段文本送进 embedding 检索。它的直觉是：答案空间里的文档比问题空间里的查询更接近目标文档。

\`\`\`python
hyde_doc = llm(f"请根据问题写一段可能包含答案的短文：{query}")
hits = retriever.search(hyde_doc, top_k=5)
\`\`\`

**何时用**：用户查询很短或噪声大。

## 四、多跳检索（Multi-hop Retrieval）

有些问题需要串联多个文档才能回答。例如"2018 年那次收购后，被收购公司的 CEO 后来加入了哪家初创公司？"——必须先找到收购记录，再找到该 CEO 的去向。

实现方式：

1. 让 LLM 在每一步判断"还需要什么信息"，再针对性检索。
2. 把前几步检索到的中间结论拼进下一轮查询。

**何时用**：复杂分析任务、研究类问答。

## 五、Agentic RAG（ReWOO / ReAct）

当问题需要组合多种工具（检索 + 计算 + API 调用）时，可以让 LLM 担任"调度者"，按 ReAct 或 ReWOO 范式决定下一步动作。

- **ReAct**：让模型在"思考 → 行动 → 观察"循环中推进。
- **ReWOO**：把所有动作先"规划"出来，再批量执行，最后由 LLM 综合。ReWOO 节省中间轮推理成本。

**何时用**：开放域问答、研究助手、企业知识工作流。

## 六、模式对比

| 模式 | 主要收益 | 主要代价 | 典型场景 |
| --- | --- | --- | --- |
| Re-ranking | 提升 top-k 精度 | 增加延迟 | 高质量问答 |
| 混合检索 | 兼顾术语与语义 | 需维护两套索引 | 法律 / 代码 / 医疗 |
| Query rewriting | 提升召回 | 一次额外 LLM 调用 | 短查询 / 口语化 |
| HyDE | 解决查询-文档 gap | LLM 调用 + 噪声 | 文档库小且杂 |
| Multi-hop | 支持复杂推理 | 多轮检索成本 | 研究类问题 |
| Agentic RAG | 多工具协同 | 延迟与可控性 | 复杂工作流 |

## 小结

从朴素 RAG 到 agentic RAG，本质是在"让模型多花一点算力，换更高的答案质量"和"延迟预算"之间权衡。下一篇我们会讨论如何**评估**这些系统的输出，避免"看起来好、实际上没改进"的陷阱。
`;

const ragEval = `# RAG 评估指标：如何知道你的系统真的变好了

没有评估就没有迭代。但 RAG 的答案不像机器翻译有标准答案，无法直接套用 BLEU/ROUGE。本文介绍 RAGAS 框架的核心维度、为什么传统指标会失灵，并展示一个"LLM-as-judge"的最小可行评估代码。

## 一、为什么 BLEU/ROUGE 不够

BLEU/ROUGE 是 n-gram 重叠率，对**生成文本与参考答案的字面相似度**敏感。但在 RAG 中：

- 同一事实可以有多种正确表述；
- 真正该评估的是"答案是否忠于上下文"、"是否答到点子上"，而不是与参考表述多像。

所以我们需要的是**对单个维度的细粒度评估**，而不是全文相似度。

## 二、RAGAS 的三大维度

RAGAS（Retrieval-Augmented Generation Assessment）是当下最广泛使用的 RAG 评估框架之一，把质量拆成三个独立维度：

1. **Context Relevance（上下文相关性）**：检索回来的 chunks 与用户问题的相关程度。分数高 = 检索器精准。
2. **Answer Faithfulness（答案忠实度）**：最终答案中陈述的事实是否能从给定上下文中找到依据。分数高 = 答案不幻觉。
3. **Answer Relevance（答案相关性）**：答案是否真的回答了用户问题，而不是答非所问。

每个维度都由一个 LLM 担任 judge，按 0-1 打分或 0-5 打分。

## 三、数学定义（简化）

设 $Q$ 为用户问题，$C$ 为检索上下文，$A$ 为生成答案：

\`\`\`math
\\text{ContextRelevance} = \\frac{|\\text{claims}(Q) \\cap \\text{claims}(C)|}{|\\text{claims}(Q)|}
\`\`\`

\`\`\`math
\\text{Faithfulness} = \\frac{|\\text{claims}(A) \\cap \\text{claims}(C)|}{|\\text{claims}(A)|}
\`\`\`

\`\`\`math
\\text{AnswerRelevance} = 1 - \\text{distance}(\\text{embed}(Q), \\text{embed}(A\\text{ from }Q'))
\`\`\`

其中 $Q'$ 是从 $A$ 反推出的"如果我问这个问题，答案会是什么"。直觉上：好的答案应该既忠于证据、又切题。

## 四、LLM-as-judge 代码示例

\`\`\`python
import json
from openai import OpenAI

client = OpenAI()

JUDGE_PROMPT = """你是一名严格的 RAG 评估员。
请根据参考资料评估生成答案的忠实度。

问题：{question}
参考资料：{contexts}
答案：{answer}

只输出 JSON：{{"faithfulness": 0.0~1.0, "reason": "<简短理由>"}}
"""

def judge_faithfulness(question, contexts, answer):
    msg = JUDGE_PROMPT.format(question=question, contexts="\\n".join(contexts), answer=answer)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": msg}],
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)

# 在一个评估集上批量打分
results = [judge_faithfulness(ex.q, ex.contexts, ex.answer) for ex in eval_set]
print("mean faithfulness:", sum(r["faithfulness"] for r in results) / len(results))
\`\`\`

为了让 judge 更稳定，可以：

- 多采样取均值；
- 与人工标注的子集做相关性检查；
- 用更强的模型定期抽样验证。

## 五、其它常用指标

- **Context Recall**：真实答案涉及的事实是否被检索到。需要有人工标注的 ground-truth chunks。
- **Context Precision**：检索结果中相关 chunk 占的比例。
- **Answer Correctness**：与参考答案事实一致的程度（适合有标准答案的场景，如 HotpotQA）。

## 六、评估集的构造

光有指标不够，还需要**评估集**。建议：

1. **采样真实线上问题**，再人工补充 ground-truth。
2. **合成难题**：用 LLM 基于文档生成多跳问题、矛盾问题、超出知识库问题。
3. **定期更新**：每次系统改动后，跑全套评估，看到底是哪个维度掉了。

## 小结

RAG 的评估是一个"评估维度 + LLM-as-judge + 高质量评估集"三位一体的工作。RAGAS 是起点而非终点——把它当作一个参考，结合业务场景定义自己的指标，才能真正驱动系统持续改进。
`;

// ========== Write all ==========
const articles = [
  { path: '01-foundations/01-mathematics/linear-algebra-foundations.md', content: linearAlgebra },
  { path: '01-foundations/01-mathematics/calculus-for-ml.md', content: calculus },
  { path: '01-foundations/01-mathematics/probability-and-statistics.md', content: probability },
  { path: '02-deep-learning/transformers/attention-mechanism-explained.md', content: attentionExplained },
  { path: '02-deep-learning/transformers/transformer-architecture.md', content: transformerArch },
  { path: '02-deep-learning/transformers/gpt-and-bert-evolution.md', content: gptBert },
  { path: '02-deep-learning/transformers/gpt-vs-llama-architecture.md', content: gptVsLlama },
  { path: '03-large-language-models/rag/rag-fundamentals.md', content: ragFundamentals },
  { path: '03-large-language-models/rag/advanced-rag-patterns.md', content: advancedRag },
  { path: '03-large-language-models/rag/rag-evaluation-metrics.md', content: ragEval },
];

function approxWordCount(text) {
  // For Chinese mixed with English: count CJK characters + whitespace-separated words.
  const cjk = (text.match(/[一-鿿]/g) || []).length;
  const ascii = (text.match(/[A-Za-z]+/g) || []).length;
  return cjk + ascii;
}

async function main() {
  const summary = [];
  for (const { path, content } of articles) {
    const full = join(ROOT, path);
    await mkdir(dirname(full), { recursive: true });
    await writeFile(full, content, 'utf8');
    const wc = approxWordCount(content);
    summary.push(`${path.split('/').pop()} (${wc} words)`);
  }
  console.log(`Wrote ${articles.length} articles: ${summary.join(', ')}.`);
}

main().catch((e) => { console.error(e); process.exit(1); });