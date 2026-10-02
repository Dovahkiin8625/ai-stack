# 神经机器翻译：从 Seq2Seq 到 Attention 的范式跃迁

2014 年是机器翻译史的转折点。Sutskever et al. 用 Seq2Seq（LSTM encoder-decoder）在 WMT'14 英法任务上首次超越 Moses；Bahdanau et al. 紧接着提出 Attention 机制，解决了 Seq2Seq 的长序列瓶颈；Google Translate 在 2016 年全面切换到神经模型。本文从 Seq2Seq 出发，深入剖析 Attention、Teacher Forcing、暴露偏差等核心概念，并讨论它们如何为 Transformer 时代铺路。

## 一、Seq2Seq：把翻译建模为序列到序列

### 1.1 基本架构

Seq2Seq（Sutskever et al., 2014）把机器翻译视为"把一个序列映射到另一个序列"的问题：

```
[Encoder]   →   [固定维度向量 c]   →   [Decoder]   →   输出序列
源序列 x_1...x_n                    目标序列 y_1...y_m
```

**Encoder**：LSTM 逐 token 读取源序列，最后一个隐藏状态 $\mathbf{h}_n$ 是"整句的压缩表示"。

$$
\mathbf{h}_t = \text{LSTM}_{\text{enc}}(\mathbf{h}_{t-1}, \mathbf{E}[x_t])
$$

**Decoder**：以 $\mathbf{h}_n$ 为初始状态，逐 token 生成目标序列：

$$
\mathbf{s}_t = \text{LSTM}_{\text{dec}}(\mathbf{s}_{t-1}, \mathbf{E}[y_{t-1}])
$$
$$
P(y_t \mid y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t)
$$

直觉：Encoder 把源语言"压缩"成一个向量 c，Decoder 从 c 中"解压"出目标语言。

### 1.2 关键设计

- **深层 LSTM**：4 层 LSTM（典型）效果优于 1 层。
- **反向 Encoder**：源序列反序输入，让源句末词更靠近 encoder 的最终状态（因为解码时通常先生成源句首词的内容）。
- **Decoder 初始化**：用 encoder 末态 $\mathbf{h}_n$ 初始化 decoder 初始状态 $\mathbf{s}_0$。
- **大词表处理**：softmax 计算量是 $O(V)$，百万词表很慢。解决方案：
  - **采样 softmax**：每步只对部分负样本计算。
  - **Noise Contrastive Estimation**：把多分类变成二分类。
  - **Class-based softmax**：词聚类成大类，先大类后小类。

### 1.3 Seq2Seq 的根本局限

Seq2Seq 把整句压缩成**单一向量 $\mathbf{h}_n$**。这个设计有两个问题：

1. **信息瓶颈**：无论源句多长，所有信息必须经过这一个向量。Sutskever 2014 的实验显示 35-50 词后性能急剧下降。
2. **梯度路径长**：Decoder 每步的梯度必须穿过整个 encoder（LSTM 链），长句训练极慢。

解决思路：**让 decoder 每一步都能"看到" encoder 的所有位置**——这就是 Attention 机制的核心思想。

## 二、Bahdanau Attention：第一个 Attention 机制

### 2.1 核心思想

Bahdanau et al.（2015）的 Attention 让 decoder 每步从 encoder 的所有位置"选"信息：

$$
\mathbf{c}_t = \sum_{i=1}^{n} \alpha_{t,i} \mathbf{h}_i^{\text{enc}}
$$

其中 $\alpha_{t,i}$ 是 decoder 步 $t$ 对 encoder 位置 $i$ 的"关注度"：

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{n} \exp(e_{t,j})}
$$
$$
e_{t,i} = \mathbf{v}^\top \tanh(\mathbf{W}_a [\mathbf{s}_{t-1}; \mathbf{h}_i^{\text{enc}}])
$$

直觉：

- $e_{t,i}$ 是 decoder 上一步状态 $\mathbf{s}_{t-1}$ 与 encoder 位置 $i$ 的"匹配分数"。
- $\alpha_{t,i}$ 是 softmax 后的关注权重——和为 1。
- $\mathbf{c}_t$ 是 encoder 的加权平均，作为 decoder 当前步的"上下文向量"。

Decoder 状态更新变为：

$$
\mathbf{s}_t = \text{LSTM}_{\text{dec}}(\mathbf{s}_{t-1}, [y_{t-1}; \mathbf{c}_t])
$$

### 2.2 Attention 解决了什么

- **信息瓶颈消失**：decoder 每步直接看到 encoder 的所有位置，不再依赖单一压缩向量。
- **梯度路径短**：attention 是"软指针"，可以跳过中间 token。
- **可解释**：可视化 attention 权重，能看到"对齐"——如翻译 `中国` 时注意力集中在 `China`。

### 2.3 Additive vs Multiplicative Attention

Bahdanau 的 attention 是 **additive**（又称 `concat`）——把 $\mathbf{s}_{t-1}$ 与 $\mathbf{h}_i$ 拼接后过 tanh 与线性层。

后续 Luong et al.（2015）提出 **multiplicative**（又称 `dot` / `general`）：

$$
e_{t,i} = \mathbf{s}_{t-1}^\top \mathbf{W} \mathbf{h}_i^{\text{enc}}
$$

更高效（一个矩阵乘法），效果相当。Transformer 用的是 dot-product attention 的扩展（详见下文）。

## 三、训练机制：Teacher Forcing 与暴露偏差

### 3.1 Teacher Forcing

Seq2Seq 训练时，decoder 每步输入是**真实的前一个 token**（而不是模型自己预测的）：

```
目标序列：<BOS> I love China <EOS>
decoder 输入: <BOS>, I, love, China
decoder 预测: I, love, China, <EOS>
```

这种"用参考答案作为下一步输入"的方式叫 **Teacher Forcing**——大大加速训练收敛。

### 3.2 暴露偏差（Exposure Bias）

推理时 decoder 输入的是**自己预测的 token**，可能出错：

```
真实情况: 输入"I" → 预测"love" → 输入"love" → 预测"China" ...
错误情况: 输入"I" → 预测"like" → 输入"like" → 预测"..." → 错误传播
```

训练时 decoder 从未见过自己的错误，推理时遇到错误就可能"雪崩"。这是**暴露偏差**。

### 3.3 缓解方案

- **Scheduled Sampling**（Bengio et al., 2015）：训练时以一定概率用模型预测替代真实 token，逐步增加比例。
- **Prof-Forcing**（Lamb et al., 2016）：让训练过程模拟推理路径。
- **Sequence-Level Loss**：直接优化 BLEU 等序列级指标（但不可微，需 REINFORCE 等强化学习技巧）。
- **Label Smoothing**：把 one-hot 标签改成软标签（如 0.9/0.1），减少过自信。

## 四、Luong Attention 与全局/局部 Attention

### 4.1 Luong 的 Attention 变种

Luong et al.（2015）提出两种 attention 形式：

- **Global Attention**：decoder 每步看 encoder 的所有位置（同 Bahdanau）。
- **Local Attention**：decoder 每步只看 encoder 的一个窗口（中心在预测的对齐位置附近），类似 hard attention。

Local Attention 在长句上更高效，且 attention 权重可视化更聚焦。

### 4.2 Input Feeding

Luong 还提出**输入馈送**（input feeding）——把前一步的 attention 输出 $\tilde{\mathbf{s}}_{t-1}$ 拼接到当前步输入：

$$
\mathbf{s}_t = \text{LSTM}_{\text{dec}}(\mathbf{s}_{t-1}, [y_{t-1}; \mathbf{c}_t; \tilde{\mathbf{s}}_{t-1}])
$$

直觉：让 decoder 知道"之前关注了哪里"，避免重复关注同一区域。

## 五、PyTorch 实现：Seq2Seq + Bahdanau Attention

下面是一个端到端可训练的中英翻译模型（约 100 行）：

```python
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)   # encoder
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)   # decoder
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask=None):
        # decoder_state: (B, H)
        # encoder_outputs: (B, T, H)
        # mask: (B, T) 1=有效，0=pad
        scores = self.v(torch.tanh(
            self.W_h(encoder_outputs) + self.W_s(decoder_state).unsqueeze(1)
        )).squeeze(-1)                       # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, T)
        ctx = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, H)
        return ctx, attn


class Seq2Seq(nn.Module):
    """Encoder-Decoder with Bahdanau Attention."""

    def __init__(self, src_vocab: int, tgt_vocab: int, emb_dim: int = 256,
                 hid_dim: int = 512, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, emb_dim, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                               bidirectional=True, batch_first=True, dropout=dropout)
        # decoder 用单向 LSTM（attention 接管双向信息）
        self.decoder = nn.LSTM(emb_dim + hid_dim * 2, hid_dim,
                               num_layers=n_layers, batch_first=True, dropout=dropout)
        self.attention = BahdanauAttention(hid_dim)
        self.out_proj = nn.Linear(hid_dim, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask):
        emb = self.dropout(self.src_embed(src))
        lengths = src_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out, (h, c) = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # 把双向拼接为 2H 维：decoder 单向输入 2H 维
        return out  # (B, T_src, 2H)

    def decode_step(self, prev_token, last_state, enc_out, src_mask):
        emb = self.dropout(self.tgt_embed(prev_token)).unsqueeze(1)  # (B, 1, E)
        ctx, _ = self.attention(last_state[0][-1], enc_out, src_mask)  # (B, 2H)
        ctx = ctx.unsqueeze(1)
        out, state = self.decoder(torch.cat([emb, ctx], dim=-1), last_state)
        logits = self.out_proj(out.squeeze(1))  # (B, V)
        return logits, state

    def forward(self, src, tgt, src_mask):
        enc_out = self.encode(src, src_mask)
        B, T = tgt.shape
        # 用 encoder 最终状态初始化 decoder
        h = enc_out.new_zeros(self.decoder.num_layers, B, self.decoder.hidden_size)
        c = h.clone()
        h[-1] = enc_out[:, -1, :self.decoder.hidden_size]
        c[-1] = enc_out[:, -1, self.decoder.hidden_size:]
        state = (h, c)
        all_logits = []
        for t in range(T - 1):
            logits, state = self.decode_step(tgt[:, t], state, enc_out, src_mask)
            all_logits.append(logits)
        return torch.stack(all_logits, dim=1)  # (B, T-1, V)


# 训练循环（伪代码）
model = Seq2Seq(src_vocab=30000, tgt_vocab=30000).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(20):
    for src, tgt in train_loader:
        src, tgt = src.cuda(), tgt.cuda()
        src_mask = (src != 0).float()
        logits = model(src, tgt[:, :-1], src_mask)
        loss = criterion(logits.reshape(-1, 30000), tgt[:, 1:].reshape(-1))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

实现要点：

- **双向 encoder + 单向 decoder**：encoder 用 BiLSTM 编码双向信息，decoder 单向（LSTM 本就适合单向生成）。
- **状态投影**：encoder 末态（2H）投影到 decoder 初始状态（H）。
- **Teacher Forcing**：训练时 `tgt[:, t]` 是真实 token。
- **梯度裁剪**：LSTM 训练必备。

## 六、暴露偏差的现代视角

虽然 Seq2Seq 的"暴露偏差"被广泛讨论，但 Transformer 时代这一担忧减弱，原因有三：

1. **并行训练**：Teacher Forcing 让整个目标序列并行计算，效率极高。
2. **大模型容错强**：GPT-3 / LLaMA 等大模型在推理时即使前几步预测错，也能"自我修正"。
3. **RLHF 微调**：现代 LLM 用 RLHF 训练，模型学到了从错误中恢复的能力。

但在**低资源 NMT** 中，暴露偏差仍是问题——BLEU 在 dev 上可能远高于 test。

## 七、解码策略：Beam Search 的复兴

推理时 Seq2Seq 用 Beam Search（详见 [[text-generation/decoding-strategies]]）：

```
每步保留 top-K 个候选：
  扩展：每个候选生成 V 个下一步 token
  评分：累计对数概率 + 长度惩罚
  剪枝：保留 top-K
终止：所有候选都生成 <EOS> 或达到最大长度
```

典型 K=5-10。K 越大 BLEU 越高（但收益递减），推理越慢。

## 八、评估：BLEU、chrF 与 TER

### 8.1 BLEU

详见 [[statistical-mt]] 一文。WMT 评测的标准是 case-sensitive BLEU + 4 种参考翻译平均。

### 8.2 chrF

Popović（2015）提出的字符级 F1，对形态学丰富语言（捷克语、芬兰语、土耳其语）更友好：

$$
\text{chrF}_\beta = (1 + \beta^2) \frac{P_{\text{chr}} \cdot R_{\text{chr}}}{\beta^2 P_{\text{chr}} + R_{\text{chr}}}
$$

$P_{\text{chr}}, R_{\text{chr}}$ 是字符 n-gram（n=2-6）的精确率与召回率。

### 8.3 TER（Translation Error Rate）

编辑距离的归一化版本，反映"修改多少词能让翻译正确"。

### 8.4 COMET 与 BLEURT

现代神经评估模型：

- **COMET**（Rei et al., 2020）：用 XLM-R 编码源 + 候选 + 参考，回归到人工评分。在 WMT 与人类评分的相关性达 0.85+，远超 BLEU 的 0.4-0.5。
- **BLEURT**（Sellam et al., 2020）：BERT 预训练 + 人工评分微调。
- **GEMBA**（Kocmi & Federmann, 2023）：用 GPT-4 做 MT 评估，相关性可达 0.9。

工业 MT 系统现在更常用 COMET / BLEURT 评估，而非 BLEU。

## 九、Seq2Seq 的遗产

### 9.1 为 Transformer 铺路

Seq2Seq + Attention 是 Transformer 的**直接前身**：

- 抛弃 LSTM / GRU encoder-decoder → Transformer encoder-decoder。
- 加性 attention → 缩放点积 attention（scaled dot-product）。
- 双向 RNN → 自注意力 + 位置编码。
- 单层 attention → 多头 + 多层。

Transformer 的成功证明：Attention 机制本身就是足够的特征提取器，不需要 RNN 的循环结构。

### 9.2 训练技巧

- **Teacher Forcing + Label Smoothing** 仍是标配。
- **Beam Search + 长度惩罚** 仍是最佳解码策略。
- **BPE / WordPiece 子词切分** 大幅降低 OOV 率。
- **Back-translation**（Sennrich et al., 2016）：用目标语言单语语料反向翻译，扩充训练集。

### 9.3 工业部署

Seq2Seq 时代 GNMT（Google Neural Machine Translation, Wu et al., 2016）是 Google Translate 的基础。DeepL、Linguee、Yandex 等也都基于 Seq2Seq / Transformer。这些系统在 2016-2022 年间把"机器翻译"从"勉强能读"提升到"接近人工水平"。

## 十、未来方向

1. **多语言统一模型**：mBART、mT5、NLLB 一个模型翻译 100+ 语言。
2. **大模型时代**：GPT-4、Claude 等用 in-context learning 做翻译，零样本就能接近微调模型。
3. **同声传译**：流式翻译 + 上下文等待策略。
4. **多模态翻译**：从图像 + 文本联合翻译（看图说话 + 翻译）。

## 小结

| 时代 | 模型 | 关键创新 | WMT'14 EN-FR BLEU |
| --- | --- | --- | --- |
| 2014 (SMT) | Moses 短语翻译 | 概率框架 | 33.3 |
| 2014 (Seq2Seq) | LSTM encoder-decoder | 端到端 | 34.8 |
| 2015 (Bahdanau) | + Attention | 信息瓶颈突破 | 36.1 |
| 2016 (GNMT) | 8 层 LSTM + Attention | 深度 + 残差 | 38.9 |
| 2017 (Transformer) | Self-attention | 并行训练 | 41.0+ |

Seq2Seq + Attention 是神经机器翻译的"起点"——它证明了"端到端深度学习可以超越手工特征工程"，也暴露了"循环结构"的天花板。这些发现直接催生了 Transformer。下一篇我们将看到 Transformer 如何用 self-attention 彻底取代 RNN，把 NMT 推向新的高度。
