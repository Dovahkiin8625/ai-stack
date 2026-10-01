# Seq2Seq 与注意力机制

## 一、Seq2Seq：用 RNN 解决"输入输出都是序列"的任务

机器翻译、摘要、对话、代码补全——这些任务都有一个共同特征：**输入是变长序列，输出也是变长序列**，且长度通常不等。

**Sutskever et al. (2014)** 提出的 Seq2Seq 用两个 RNN 接力：编码器把输入序列压成一个向量，解码器从该向量出发逐 token 生成输出。

```text
        编码器 (Encoder)                    解码器 (Decoder)
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │                          │         │                          │
   "I" ─► [RNN] ─► h₁        │         │                          │
           │                  │         │                          │
   "love" ─► [RNN] ─► h₂     │         │                          │
           │                  │         │                          │
   "NLP" ─► [RNN] ─► h₃      │         │                          │
           │                  │         │                          │
   "." ─► [RNN] ─► h₄         │         │                          │
                          ╲   │         │                          │
                           ╲  ▼         │                          │
                       c = h₄ ──────────►│ ──► [RNN] ──► "我"       │
                                           │       │                │
                                           │       ▼                │
                                           │  ──► [RNN] ──► "爱"     │
                                           │       │                │
                                           │       ▼                │
                                           │  ──► [RNN] ──► "NLP"   │
                                           │       │                │
                                           │       ▼                │
                                           │  ──► [RNN] ──► "。"     │
                                           │       │                │
                                           │       ▼                │
                                           │     <eos> 停止         │
   └──────────────────────────┘         └──────────────────────────┘
```

$c$ 是**上下文向量**（通常取编码器最后隐状态），是连接两个 RNN 的唯一信息通道。

### 1.1 推理与训练

- **推理**：解码器从 `<bos>` 起步，每步采样一个 token 喂给下一步，直到生成 `<eos>`。
- **训练**：用**教师强制**（teacher forcing）——每步喂**真实上一 token** 而非模型自己的预测：

```python
# 训练时
decoder_input = target[:-1]            # 整体右移一格
decoder_target = target[1:]
logits = decoder(decoder_input, encoder_hidden)
loss = F.cross_entropy(logits.reshape(-1, V), decoder_target.reshape(-1))
```

推理时换成"自回归"：上一步预测的 token 当下步输入，**暴露偏差**（exposure bias）会随步数累积。

## 二、瓶颈：固定大小的上下文向量

把整段源序列压成一个 $d_h$ 维向量，看似简洁，问题立刻暴露：

- 一句 5 词的句子和一句 50 词的句子编码成**同样大小**的向量——后者信息密度被严重压缩。
- 距离编码器末端越远的 token，在反复 BPTT 后梯度信号越弱（我们已在 RNN/LSTM 那两篇见过）。
- 解码器每一步都看**同一个** $c$，但不同位置生成的内容需要**不同**的源端信息（如翻译到中文动词时想看英文谓语）。

直觉上这是一个**信息瓶颈**——下游模型的好坏被这个向量"卡死"。**Bahdanau 注意力** (2014) 给出了优雅的解法：让解码器在每一步**回头看所有编码器隐状态**，自己学"应该关注哪儿"。

## 三、Bahdanau 注意力（加性）

### 3.1 对齐分数

给定解码器上一时刻隐状态 $s_{i-1}$ 和编码器第 $j$ 个隐状态 $h_j$，计算它们的相关性：

$$
e_{ij} \;=\; v_a^\top \,\tanh\!\left( W_a\, s_{i-1} + U_a\, h_j \right)
$$

$W_a \in \mathbb{R}^{d_a \times d_h}, U_a \in \mathbb{R}^{d_a \times d_h}, v_a \in \mathbb{R}^{d_a}$ 是可学习参数。这是一种**加性**（additive）注意力——把 $s$ 和 $h$ 投影到同一空间再相加。

### 3.2 注意力权重（softmax 归一化）

$$
\alpha_{ij} \;=\; \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

$\alpha_{ij}$ 表示"生成第 $i$ 个目标 token 时，第 $j$ 个源 token 的重要性"。

### 3.3 上下文向量

$$
c_i \;=\; \sum_{j=1}^{T_x} \alpha_{ij}\, h_j
$$

$c_i$ 是编码器所有隐状态的**加权平均**，权重由对齐分数决定。解码器在第 $i$ 步用 $c_i$（而非固定的 $c$）配合 $s_{i-1}$ 算出 $s_i$ 并生成 $y_i$。

```text
   编码器隐状态:     h_1    h_2    h_3    h_4    h_5
                       ╲      │      │      │     ╱
                        ╲     ▼      ▼      ▼    ╱
   对齐分数:            e_{i1} e_{i2} e_{i3} e_{i4} e_{i5}
                        ╲     │      │      │     ╱
                         softmax 归一化
                            ╲   │   │   │   ╱
                             ▼  ▼   ▼   ▼  ▼
                          α_{i1} α_{i2} α_{i3} α_{i4} α_{i5}
                             │   │     │    │    │
                             ▼   ▼     ▼    ▼    ▼
                       c_i = Σ α_{ij} h_j   ← 加权平均
                                 │
                                 ▼
                              解码器 s_{i-1}
                                 │
                                 ▼
                              输出 y_i
```

## 四、Luong 注意力（乘性）

**Luong et al. (2015)** 提出更简洁的**乘性**形式：

$$
e_{ij} \;=\; s_{i-1}^\top \, W_a\, h_j \quad \text{(general)}
$$

当 $W_a = I$ 时退化为最简单的**点积**：

$$
e_{ij} \;=\; s_{i-1}^\top h_j \quad \text{(dot / Luong dot)}
$$

点积版无额外参数、计算更快，但要求 $s, h$ 同维度；加性版（Bahdanau）更灵活、对维度不匹配更鲁棒——这也正是后来 Transformer 选择加性"扩展形式"再让 $d_k$ 足够大的原因（详见注意力机制那篇）。

| 类型 | 公式 | 参数 | 速度 | 适用 |
|---|---|---|---|---|
| 加性 (Bahdanau) | $v^\top \tanh(W_a s + U_a h)$ | 多 | 慢 | 维度不等、需高表达 |
| 乘性 (Luong general) | $s^\top W_a h$ | 中 | 中 | 一般首选 |
| 点积 (Luong dot) | $s^\top h$ | 0 | 最快 | 维度一致 |
| 缩放点积 (Transformer) | $s^\top h / \sqrt{d_k}$ | 0 | 最快 | 维度大时数值稳 |

## 五、注意力如何打破瓶颈

把 $c_i$ 替换成"动态"的上下文后：

- **每步都有完整源端视野**：不再被某个固定向量卡住。
- **长距离不衰减**：$\alpha_{ij}$ 由 $s_{i-1}, h_j$ 直接决定，不需要把信息压成一条链。
- **可解释性**：可视化 $\alpha$ 矩阵（如翻译的"对齐矩阵"）能看到模型"看哪儿"——这就是 Transformer 时代 attention map 的雏形。

```text
   源:    "I"   "love"   "NLP"   "."
目标 "我": α=0.7  α=0.1   α=0.1  α=0.1
目标 "爱": α=0.1  α=0.8   α=0.05 α=0.05
目标 "NLP": α=0.1 α=0.05  α=0.8  α=0.05
目标 "。": α=0.2 α=0.1   α=0.1  α=0.6
```

每行的概率分布就是"模型在生成这个词时对源端的关注图"。

## 六、用 PyTorch 搭一个数字序列反转的 Seq2Seq + Attention

下面给一个**易于理解**的例子：输入 $[3, 1, 4, 1, 5]$，输出 $[5, 1, 4, 1, 3]$（反转）。它小到能在 CPU 上几分钟跑完，又足以展示 Seq2Seq + 注意力。

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab, d_emb, d_hid):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_emb)
        self.rnn = nn.GRU(d_emb, d_hid, batch_first=True)
    def forward(self, x):                          # x: (B, T)
        emb = self.emb(x)
        out, h = self.rnn(emb)                     # out: (B, T, d_hid)
        return out, h                             # out 给 attention 用

class BahdanauAttention(nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.W_s = nn.Linear(d_hid, d_hid, bias=False)
        self.U_h = nn.Linear(d_hid, d_hid, bias=False)
        self.v   = nn.Linear(d_hid, 1, bias=False)
    def forward(self, s_prev, enc_out):            # s_prev: (B, d_hid), enc_out: (B, T, d_hid)
        T = enc_out.size(1)
        s_prev = s_prev.unsqueeze(1).expand(-1, T, -1)
        e = self.v(torch.tanh(self.W_s(s_prev) + self.U_h(enc_out))).squeeze(-1)
        alpha = F.softmax(e, dim=1)                # (B, T)
        ctx = (alpha.unsqueeze(-1) * enc_out).sum(dim=1)   # (B, d_hid)
        return ctx, alpha

class Decoder(nn.Module):
    def __init__(self, vocab, d_emb, d_hid):
        super().__init__()
        self.emb  = nn.Embedding(vocab, d_emb)
        self.rnn  = nn.GRU(d_emb + d_hid, d_hid, batch_first=True)
        self.attn = BahdanauAttention(d_hid)
        self.fc   = nn.Linear(d_hid, vocab)
    def forward(self, y_prev, s_prev, enc_out):    # 单步解码
        emb = self.emb(y_prev).unsqueeze(1)       # (B, 1, d_emb)
        ctx, alpha = self.attn(s_prev.squeeze(0), enc_out)
        out, s = self.rnn(torch.cat([emb, ctx.unsqueeze(1)], dim=-1), s_prev)
        logits = self.fc(out.squeeze(1))          # (B, vocab)
        return logits, s, alpha

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
    def forward(self, src, tgt):
        enc_out, h = self.encoder(src)
        s, logits = h, []
        for t in range(tgt.size(1)):
            step, s, _ = self.decoder(tgt[:, t], s, enc_out)
            logits.append(step)
        return torch.stack(logits, dim=1)          # (B, T_tgt, vocab)
```

训练时用教师强制喂 `tgt[:, :-1]`，预测 `tgt[:, 1:]`；推理时自回归采样。

## 七、从 RNN Seq2Seq 到"Attention is All You Need"

到这里我们发现：注意力已经能完成"对齐 + 信息汇聚"的大部分工作。剩下的问题是——**还需要 RNN 吗？**

Vaswani et al. (2017) 给出了颠覆性的答案：**完全不需要**。Transformer 把序列里的每一对位置都直接用注意力相连，所有步可以**并行**计算，序列长度 $T$ 也只引入 $O(T^2)$ 而非 RNN 的 $O(T)$ 串行依赖。从此：

- **并行性**：训练速度提升一个数量级，催生 GPT / BERT 等大模型。
- **长程依赖**：任意两位置距离恒为 1，消失问题彻底不存在。
- **规模扩展**：参数可以堆到百亿、千亿而仍可训练。

这是 RNN 在主流任务上的退场时刻——不是因为它们错了，而是因为它们"不够并行、不够长、不够大"。

## 八、小结

Seq2Seq 用编码器–解码器把变长序列映射为变长序列，但固定大小的上下文向量是显著瓶颈。Bahdanau 注意力通过"每步重新算对齐分数 + 加权汇聚"让解码器动态访问全部编码器状态，从根本上打破瓶颈；Luong 用乘性 / 点积形式进一步简化。这套机制奠定了现代 Transformer 的对齐直觉——"哪部分输入对应哪部分输出"。最终，Vaswani 等人把所有 RNN 结构剥离，只保留注意力 + 前馈网络 + 残差 + 位置编码，证明了"**注意力就够了**"。理解 Seq2Seq → 注意力 → Transformer 的演化链，比单独看 Transformer 架构本身更能体会"为什么是这个设计"。
