# 传统 ASR 的演进：从 GMM-HMM 到 CTC

把一段 16 kHz 的波形变成可读文字，看似只是"听写"，实则要跨越三个本质障碍：**可变长度**（一句话可能 1 秒也可能 10 秒）、**时间-文字的弱对齐**（一个音素约 25 ms，一个汉字约 200 ms，比率并不固定）、**声学-语言的双重不确定性**（同音字、噪声、口音）。过去 30 年 ASR 的演进史，就是一部围绕这三个障碍的求解史——从概率图模型的 **GMM-HMM**，到深度学习的 **CTC**，再到端到端的 Transformer-Transducer。本篇聚焦前两代：声学特征、GMM-HMM、CTC。

## 一、声学特征：从波形到频谱包络

### 1.1 为什么不能直接用原始波形

原始波形每秒 16000 个采样点，相邻采样高度相关，信息密度低。直接把它送进模型，等于让网络从零学习"频率"概念。**声学特征（acoustic feature）** 的目标就是把 25 ms 左右的短时帧压缩到一个低维向量，让"频率"、"音色"这种物理量直接出现在特征里。

### 1.2 预加重

语音高频能量衰减快，先用一个一阶高通滤波器补偿：

$$
y[n] = x[n] - \alpha \cdot x[n-1], \quad \alpha \approx 0.97
$$

### 1.3 分帧与加窗

将连续波形切成 25 ms 一帧、10 ms 步长的短时帧，叠加汉明窗（Hamming）抑制频谱泄漏：

$$
w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right), \quad 0 \le n < N
$$

每帧做 FFT 得到频谱 $X[k]$，再求模平方得到**能量谱** $|X[k]|^2$。

### 1.4 Mel 滤波器组与 Fbank

人耳对频率的感知是非线性的——对 1 kHz 以下近似线性，对 1 kHz 以上近似对数。**Mel 尺度** 就是这种感知的工程化：

$$
\text{mel}(f) = 1127 \ln\left(1 + \frac{f}{700}\right)
$$

把能量谱通过 40 个三角 Mel 滤波器组（中心频率在 Mel 尺度均匀分布），取对数后得到 **Fbank**（40 维）。若再对 Fbank 做离散余弦变换（DCT，取前 13 维）就是 **MFCC**。

**MFCC vs Fbank**：MFCC 去相关，利于 GMM；Fbank 保留更多谐波信息，深度模型几乎都用 Fbank。下表给出两者的物理意义对比：

| 特征 | 维度 | 去相关 | 物理含义 | 主流使用 |
| --- | --- | --- | --- | --- |
| Fbank | 40 | 否 | 频带能量对数 | DNN 时代标配 |
| MFCC | 13 | 是（DCT） | 频谱包络的低阶系数 | GMM-HMM |

### 1.5 CMVN 与能量归一化

口音和音量差异会带来逐句均值偏移。对每条 utterance 做 **倒谱均值方差归一化（CMVN）**：

$$
\hat{x}_t = \frac{x_t - \mu}{\sigma}
$$

其中 $\mu,\sigma$ 是该句所有帧的均值与标准差。这一步对跨说话人、跨麦克风的鲁棒性至关重要。

## 二、GMM-HMM：概率图模型时代的巅峰

### 2.1 隐马尔可夫模型（HMM）的形式化

语音被认为是一个**双重随机过程**：可观测的声学特征序列 $O = (o_1, \dots, o_T)$ 由不可观测的**隐状态序列** $Q = (q_1, \dots, q_T)$ 生成。HMM 由三要素定义：

$$
\lambda = (\pi, A, B)
$$

- **初始概率** $\pi_i = P(q_1 = s_i)$
- **转移概率** $A_{ij} = P(q_{t+1}=s_j \mid q_t = s_i)$
- **发射概率** $B_i(o) = P(o \mid q=s_i)$

HMM 解决三个核心问题：（1）似然 $P(O \mid \lambda)$（前向算法）；（2）解码最优状态序列 $\arg\max_Q P(Q \mid O, \lambda)$（**维特比算法**）；（3）参数估计（**Baum-Welch / EM 算法**）。

### 2.2 GMM 拟合发射概率

传统做法里，$B_i(o)$ 用一个**高斯混合模型（GMM）** 拟合：

$$
B_i(o) = \sum_{k=1}^{K} w_{ik} \, \mathcal{N}(o; \mu_{ik}, \Sigma_{ik})
$$

直觉：每个音素（phone）的不同发音（如元音的清/浊、不同口音）对应高斯空间中的一个聚类。典型 $K=8 \sim 16$ 个分量，对角协方差 $\Sigma_{ik}$ 假设各维独立——计算上便宜，但对频谱相关性建模不足。

### 2.3 三状态 HMM 与上下文相关音素

一个音素内部，**起始（begin）、中间（middle）、结束（end）** 的声学特征差异巨大。所以 Kaldi 之前的系统使用**三状态 left-to-right HMM**：

$$
s_1 \xrightarrow{a_{12}} s_2 \xrightarrow{a_{23}} s_3, \quad a_{ii} \neq 0
$$

转移只能从 $s_1$ 到 $s_2$，$s_2$ 到 $s_3$，且 $s_2$ 可以自环（持续任意帧）。再进一步，**协同发音（co-articulation）** 让同一个音素在不同左右上下文里表现不同，于是把单音素扩展为**三音素（triphone）**——例如 `ah` 在 `(b-ah+l)` 和 `(g-ah+n)` 中是不同状态。引入**决策树聚类（decision tree clustering）** 把上万种 triphone 状态合并到约 4000 个**绑定状态（tied state）**，这就是 **HMM 状态的高斯化**。

### 2.4 维特比（Viterbi）解码

给定声学模型 $\lambda^{\text{am}}$ 和发音字典 $\mathcal{L}$，解码就是在 HMM 状态空间里找最优路径：

$$
Q^* = \arg\max_Q \, P(O \mid Q, \lambda^{\text{am}}) \, P(Q \mid \mathcal{L})
$$

定义动态规划变量：

$$
\delta_t(j) = \max_{q_1, \dots, q_{t-1}} P(o_1, \dots, o_t, q_t=s_j \mid \lambda)
$$

递推：

$$
\delta_t(j) = \max_i \left[ \delta_{t-1}(i) \, A_{ij} \right] \cdot B_j(o_t)
$$

辅以**回溯指针** $\psi_t(j)$ 即得最优状态序列，再映射到词序列。在大词表连续语音识别（LVCSR）里，常用**加权有限状态转换器（WFST）** 把声学、发音、语言三本知识编译为一张巨大的状态图，搜索就变成在该图上的最短路径问题——这是 Kaldi（2011，Povey 等）的核心抽象。

### 2.5 GMM-HMM 的局限

GMM-HMM 在 1990-2010 年统治了 ASR，但有四个结构性痛点：

1. **各模块独立优化**：声学、字典、语言模型（n-gram SRILM/KenLM）分别最大似然，没有一个全局 loss 在调度。
3. **对齐标注稀缺**：训练需要帧级别的状态序列，传统做法要么用强制对齐（forced alignment）从声学模型自举，要么用 Viterbi 硬对齐——一旦对齐错误，会被自举循环放大。
4. **GMM 表达力天花板**：对角协方差的高斯混合难以捕捉语音频谱的长时相关与非线性。

## 三、CTC：让神经网络摆脱对齐依赖

### 3.1 核心思想

**CTC**（Connectionist Temporal Classification，Graves et al., 2006）的核心观察是：不用显式对齐，只要让网络输出**所有可能的对齐路径**的概率之和即可。引入一个**空白符（blank，$\epsilon$）**——一种特殊"无输出"符号，网络在每个时间步可以输出一个真实标签或 blank。

定义一条对齐路径 $\pi = (\pi_1, \dots, \pi_T)$，其中 $\pi_t \in \mathcal{V} \cup \{\epsilon\}$。CTC 定义一个**多对一的映射** $\mathcal{B}: (\mathcal{V} \cup \{\epsilon\})^* \to \mathcal{V}^*$：

1. 合并连续重复的标签：`aaab\epsilon \to ab`
2. 删除所有 blank：`a\epsilon b\epsilon \to ab`

例如 $\mathcal{B}(a\epsilon bb\epsilon) = (a, b)$，$\mathcal{B}(aa\epsilon b) = (a, b)$。

### 3.2 CTC 损失（Loss）= 前向 + 后向

给定输入 $O$，网络输出 softmax 后得到每帧的后验 $p_t(k) = P(\pi_t = k \mid O)$。CTC 损失就是**目标标签序列 $y$ 的所有合法对齐的负对数似然**：

$$
\mathcal{L}_{\text{CTC}} = -\log P(y \mid O) = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} p_t(\pi_t)
$$

直接枚举 $\pi$ 不可行。CTC 用**动态规划**把 $\sum_{\pi}$ 拆成前缀求和。定义前向变量：

$$
\alpha_t(s) = \sum_{\substack{\pi_{1:t} \\ \mathcal{B}(\pi_{1:t}) = y_{1:s}}} \prod_{k=1}^{t} p_k(\pi_k)
$$

递推分**两种情形**：

$$
\alpha_t(u) = \begin{cases}
(\alpha_{t-1}(u) + \alpha_{t-2}(u)) \, p_t(y_u) & \text{if } y_u \neq y_{u-1}, y_u \neq \epsilon \\
(\alpha_{t-1}(u) + \alpha_{t-1}(u-1)) \, p_t(y_u) & \text{otherwise}
\end{cases}
$$

对称地定义后向变量 $\beta_t(s)$，梯度可解析写出。PyTorch 的 `nn.CTCLoss` 已实现这一切；下面给出**最小 PyTorch 实现**。

```python
import torch
import torch.nn as nn


class CTCHead(nn.Module):
    """仅演示 CTC 的 forward-backward 与 loss 计算；声学编码器省略。"""

    def __init__(self, encoder_dim: int, vocab_size: int):
        super().__init__()
        # vocab_size 含 blank；按惯例 blank 放在第 0 位
        self.proj = nn.Linear(encoder_dim, vocab_size)
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(
        self,
        encoder_out: torch.Tensor,      # (B, T, D)  声学编码器输出
        targets: torch.Tensor,           # (B, S)    目标标签 id，blank 不出现
        input_lens: torch.Tensor,        # (B,)      每条 utterance 的有效帧数 T
        target_lens: torch.Tensor,       # (B,)      每条 utterance 的标签长度 S
    ) -> torch.Tensor:
        logits = self.proj(encoder_out)             # (B, T, V)
        log_probs = logits.log_softmax(dim=-1)     # CTC 要求 log-softmax
        # CTCLoss 期望 (T, B, V)
        loss = self.ctc(
            log_probs=log_probs.transpose(0, 1),
            targets=targets,
            input_lengths=input_lens,
            target_lengths=target_lens,
        )
        return loss
```

代码里几个**容易踩坑的点**：

- `blank=0` 是约定，把 blank 放词表第 0 位能让解码时直接 `argmax` 后跳过第 0 个类。
- `log_softmax(dim=-1)`：**CTC 要求对数概率**而不是 softmax。许多新手直接喂 softmax，导致 NaN。
- `zero_infinity=True`：当目标长度大于输入帧数时，CTC 损失可能是 $+\infty$，开启后会把这些样本的 loss 置 0 并跳过梯度。

### 3.3 空白符的物理意义

为什么需要 blank？因为没有 blank，CTC 会强制每帧输出一个标签，模型为了对齐就得"挤"出重复字符。blank 让网络在无声段（静音、过渡帧）"闭嘴"，只在真正的音素边界附近输出标签。这也是为什么 CTC 输出长度通常 $\le$ 输入帧数——空白帧不贡献标签。

### 3.4 CTC prefix beam search

训练结束后需要解码。**贪心解码**（每帧取 argmax 后 $\mathcal{B}$ 映射）快但不准。**Beam search** 维护 $k$ 条前缀，每步扩展时考虑（a) 重复上一标签（不加分隔符）、（b) 输出新标签、（c) 输出 blank。状态定义为：

$$
\text{key} = (p, l) : p \in \text{prefix}, l = \text{最后字符}
$$

并维护两个概率：**以 blank 结尾**（$p_b$）和**以非 blank 结尾**（$p_n$）。同一前缀合并：

$$
p(p) = p_b(p) + p_n(p)
$$

这是 CTC 与语言模型融合的入口——在扩展时叠加 n-gram 概率。

### 3.5 外接语言模型：KenLM shallow fusion

CTC 本身不含语言信息。训练一个 n-gram KenLM（Heafield, 2011），解码时按**shallow fusion** 插值：

$$
\log P_{\text{decode}}(w \mid h) = \log P_{\text{CTC}}(w \mid h) + \lambda_{\text{lm}} \log P_{\text{LM}}(w \mid h) + \lambda_{\text{wb}} |w|
$$

$\lambda_{\text{wb}}|w|$ 是**词奖励（word bonus）**，平衡 CTC 对字符的偏置；字符级 CTC 倾向短词，词奖励能矫正输出过短问题。

## 四、传统方法的工程反思

| 模块 | 代表技术 | 优势 | 痛点 |
| --- | --- | --- | --- |
| 特征 | MFCC、Fbank | 物理可解释 | 信息有损 |
| 声学模型 | GMM-HMM | 概率图清晰 | 各模块独立优化 |
| 解码 | Viterbi + WFST | 数学优雅 | 工程复杂 |
| 对齐 | 强制对齐 / 硬对齐 | 标签易得 | 误差自举放大 |
| 语言模型 | KenLM n-gram | 训练快 | 不能端到端 |

最致命的**多模块独立训练**：声学、发音、语言模型各优化自己的似然，**没有一个全局 loss 在调度**。任何一模块的偏差都会向下游传播。CTC 的伟大之处在于：第一次让神经网络可以**端到端地**从声学到文字，但仍缺乏注意力机制带来的"软对齐"灵活性——这就为下一篇要讲的 **LAS、RNN-T、Transformer-Transducer** 埋下了引线。

## 小结

| 模块 | GMM-HMM 时代 | CTC 时代 |
| --- | --- | --- |
| 声学特征 | MFCC（13 维） | Fbank（40-80 维） |
| 声学模型 | GMM 发射概率 + 三状态 HMM | 深度网络 + CTC 损失 |
| 对齐 | 强制对齐（硬对齐） | 自动学习（软对齐） |
| 解码 | WFST + 维特比 | Prefix beam + 语言模型 |
| 优势 | 可解释、数据高效 | 端到端、无对齐标注 |
| 痛点 | 多模块独立、协方差弱 | 无注意力、对长序列建模弱 |

GMM-HMM 把 ASR 推到了语音识别商品化的门槛；CTC 则第一次让深度网络摆脱了对帧级标注的依赖。但 CTC 强大的能力是把"对齐"压成单调的——这对长句、远场、口音仍是瓶颈。下一篇我们将看到 **LAS（Listen-Attend-Spell）** 如何用注意力机制打破 CTC 的单调性，以及 **RNN-T、Conformer** 如何在工业级流式场景中登顶。