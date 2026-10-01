# LSTM 与 GRU：门控记忆的循环网络

## 一、LSTM 的动机：为什么 Vanilla RNN 不够

上篇文章看到，Vanilla RNN 在超过约 10 步的长程依赖上就会梯度消失。**Hochreiter & Schmidhuber (1997)** 的关键洞察是：消失的根源是每步都要做**乘性**更新 $h_t = W \cdot h_{t-1} + \dots$，导致信息只能按 $\|W\|^k$ 缩放地传播。

LSTM（Long Short-Term Memory）的解法是引入一个**额外的"细胞状态"通道 $c_t$**，让信息流主要走**加性**路径（$c_t \approx c_{t-1} + \text{小量}$），只有写入/读出时才被门控做乘性调制。

## 二、LSTM 单元：四组公式

LSTM 在每个时间步 $t$ 同时维护两个状态：隐状态 $h_t$（对外输出）和细胞状态 $c_t$（长期记忆）。三个**门**控制信息的写入、保留和读出：

$$
f_t \;=\; \sigma\!\left( W_f\, [h_{t-1}, x_t] + b_f \right) \quad \text{(遗忘门)}
$$

$$
i_t \;=\; \sigma\!\left( W_i\, [h_{t-1}, x_t] + b_i \right) \quad \text{(输入门)}
$$

$$
\tilde{c}_t \;=\; \tanh\!\left( W_c\, [h_{t-1}, x_t] + b_c \right) \quad \text{(候选记忆)}
$$

$$
o_t \;=\; \sigma\!\left( W_o\, [h_{t-1}, x_t] + b_o \right) \quad \text{(输出门)}
$$

更新细胞状态和隐状态：

$$
c_t \;=\; f_t \odot c_{t-1} \;+\; i_t \odot \tilde{c}_t
$$

$$
h_t \;=\; o_t \odot \tanh(c_t)
$$

其中 $\sigma$ 是 sigmoid（输出 $[0,1]$，作为"比例系数"），$\odot$ 是逐元素乘。

### 2.1 数据流图

```text
                        x_t
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
            [Linear]   [Linear]   [Linear]   ... (f, i, o 各一组)
              │          │          │
              σ          σ          σ
              │          │          │
              f_t        i_t        o_t
              │          │          │
              ×          ×          ×        ← 逐元素乘
              │          │          │
   c_{t-1} ──►│          │          │
              ×          │          │
              │  ┌────► tanh ──►×    │
              │  │  (c̃_t)        │    │
              │  │               │    │
              └──┴──────(+)──────┴────┘
                       │
                      c_t ──► tanh ──►×──► h_t
                                      │
                                      o_t
```

### 2.2 三个门的直觉

- **遗忘门 $f_t$**：决定从 $c_{t-1}$ 中**丢掉**什么。$f_t = 1$ 全保留，$f_t = 0$ 全遗忘。语言模型中遇到句号 $f_t \to 1$ 保留主语，遇到新主语时旧的 $f_t \to 0$ 释放。
- **输入门 $i_t$**：决定把**新候选**写入 $c_t$ 的多少。$i_t$ 接近 0 时，"我不打算更新这个槽位"。
- **输出门 $o_t$**：决定从 $c_t$ 中**读出**多少作为 $h_t$。即使 $c_t$ 里有信息，$o_t = 0$ 也对外"保密"。

直觉总结：$c_t$ 走一条**高速公路**，只在三个闸口被门控按比例**保留**信息，避免了 RNN 那种"每次都被矩阵乘"的信息损耗。

## 三、GRU：更精简的版本

**Gated Recurrent Unit (Cho et al. 2014)** 把 LSTM 的三个门合并成两个，移除独立的细胞状态：

$$
z_t \;=\; \sigma\!\left( W_z\, [h_{t-1}, x_t] \right) \quad \text{(更新门)}
$$

$$
r_t \;=\; \sigma\!\left( W_r\, [h_{t-1}, x_t] \right) \quad \text{(重置门)}
$$

$$
\tilde{h}_t \;=\; \tanh\!\left( W_h\, [r_t \odot h_{t-1},\, x_t] \right) \quad \text{(候选)}
$$

$$
h_t \;=\; (1 - z_t) \odot h_{t-1} \;+\; z_t \odot \tilde{h}_t
$$

直觉：

- **更新门 $z_t$**：在"保留旧状态"和"采用新候选"之间插值，相当于 LSTM 的遗忘+输入二合一。
- **重置门 $r_t$**：控制写入候选时**忽略多少旧状态**——$r_t=0$ 时候选只依赖当前输入。

## 四、LSTM vs GRU 怎么选

| 维度 | LSTM | GRU |
|---|---|---|
| 门数量 | 3 ($f, i, o$) | 2 ($z, r$) |
| 状态向量 | $c_t, h_t$ 两个 | 仅 $h_t$ |
| 每步参数 | $4 \times (d_h^2 + d_h \cdot d_x)$ | $3 \times (d_h^2 + d_h \cdot d_x)$ |
| 速度 | 较慢 | 快约 20–30% |
| 长序列性能 | 通常更强 | 短序列持平、长序列略逊 |
| 参数量 | 更大 | 更小（数据少时更不易过拟合） |
| 何时选 | 数据多、依赖长、需要细粒度控制 | 数据少、追求速度 / 资源受限 |

经验法则：**默认从 LSTM 起手**；如果训练慢或过拟合明显再换 GRU。两者在大多数任务上差距远小于"模型 vs 数据量"的影响。

### 4.1 变体（仅作了解）

- **Peephole LSTM**（Gers & Schmidhuber 2000）：让门看到 $c_{t-1}$，$f_t = \sigma(W_f \cdot [h_{t-1}, x_t, c_{t-1}] + b_f)$。某些计时任务上更准，但参数增加、PyTorch 不内置。
- **Coupled 门**（Greff et al. 2016 的搜索结论）：设 $f_t = 1 - i_t$（遗忘和输入互补），参数更少但效果接近。
- **QRNN**（Bradbury et al. 2016）：用跨步卷积替代 $h_{t-1}$ 递归，可大幅并行。

## 五、双向 RNN：用上下文双向看序列

普通 RNN 只看**历史**。但很多任务（NER、情感、机器翻译）当前 token 的解释也依赖**未来**——`"the cell phone ..."` 中 `cell` 的歧义需要后面的 `phone` 才解开。

**Bidirectional RNN** 同时维护前向和反向两条链：

```text
   x_1 ─► [RNN] ──► h_1^→
                       │
   x_2 ─► [RNN] ──► h_2^→   h_2^← ─◄── [RNN] ◄── x_2
                       │           │
   x_3 ─► [RNN] ──► h_3^→   h_3^← ─◄── [RNN] ◄── x_3
                       │           │
                       ▼           ▼
                    拼接 h_t = [h_t^→, h_t^←]
```

每个时间步的输出是 $\vec{h}_t$ 与 $\reflectbox{$\vec{h}$}_t$ 的拼接。代价：必须**等整条序列读完**才能开始计算，无法做在线流式推理。

```python
self.rnn = nn.RNN(d_emb, d_hid, num_layers=2,
                  batch_first=True, bidirectional=True)
# h: (num_layers * 2, B, d_hid)，最后一层双向拼接：torch.cat((h[-2], h[-1]), dim=-1)
```

## 六、深度（堆叠）RNN

把多层 RNN 堆叠：第 $l$ 层的输入是第 $l-1$ 层在所有时间步的隐状态：

```text
   x ─► [RNN¹] ─► [RNN²] ─► [RNN³] ─► ... ─► 输出层
          │           │           │
        h¹_t        h²_t        h³_t
```

每一层学不同抽象：底层偏词法/形态，中层偏短语结构，高层偏语义。PyTorch 用 `num_layers=N` 自动堆叠。深度 RNN 容易过拟合，几乎必加 `dropout`（注意：`dropout` 加在**层之间**，不是时间步之间）。

## 七、用 PyTorch 做 IMDB 情感分类

下面给一个**端到端的最小例子**：嵌入 → LSTM → 取末态 → 分类头。

```python
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, d_emb=128, d_hid=128,
                 n_layers=2, n_classes=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_emb, padding_idx=0)
        self.lstm = nn.LSTM(d_emb, d_hid, num_layers=n_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        d_out = d_hid * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_out, n_classes),
        )
    def forward(self, ids, lengths):
        emb = self.emb(ids)                                   # (B, T, d_emb)
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)                         # h: (n_layers*dirs, B, d_hid)
        if self.lstm.bidirectional:
            last = torch.cat((h[-2], h[-1]), dim=-1)           # 拼接最后层前向/反向末态
        else:
            last = h[-1]                                      # 最后一层末态
        return self.fc(last)

# 训练循环（伪代码）
# model = LSTMClassifier(vocab_size=20000).to('cuda')
# opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# for ids, lengths, labels in train_loader:
#     logits = model(ids, lengths)
#     loss = F.cross_entropy(logits, labels)
#     opt.zero_grad(); loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#     opt.step()
```

GRU 版本只需把 `nn.LSTM(...)` 换成 `nn.GRU(...)`——它没有 cell state，返回值从 `(h, c)` 变成只有 `h`。

## 八、小结

LSTM 与 GRU 通过**门控加性记忆**绕过了 Vanilla RNN 的梯度消失问题：细胞状态（或 GRU 的隐状态）走加性路径，三个/两个门用 sigmoid 在 $[0,1]$ 内做按比例调制，既保留长程信息又不让网络"失控更新"。实战默认从 LSTM 起手，资源紧张或短序列换 GRU；想用上下文双向信息加双向；想抽更高层特征就堆叠多层并加 `dropout`。这些门控 RNN 在 2015 年前后是文本、语音的主流方案，直到一个更大的架构创新——**注意力机制 + Transformer**——彻底改变了格局。下一篇文章介绍 Seq2Seq 与注意力，看到 RNN 是如何被"嫁接"出新的能力，又是如何暴露出**固定大小上下文瓶颈**的。
