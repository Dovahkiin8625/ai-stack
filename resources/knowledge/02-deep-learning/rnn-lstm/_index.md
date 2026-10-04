# 循环神经网络（RNN / LSTM）

> 分类：**深度学习** → **循环神经网络**
> 路径：`resources/knowledge/02-deep-learning/rnn-lstm`

本目录聚焦"序列建模"这条与 CNN 并列的主线：从 Vanilla RNN 出发，理解循环结构与 BPTT，再通过 LSTM/GRU 的门控机制解决梯度消失，最后到 Seq2Seq 框架与早期注意力机制——它正是 Transformer 诞生的直接前身。

## 文章目录

### 基础与门控

- [RNN 基础：从 Vanilla RNN 到 BPTT](./rnn-fundamentals.md) — 序列建模动机、RNN 单元方程、BPTT、梯度消失/爆炸（$\|W\|^k$ 直觉）、梯度裁剪、PyTorch `nn.RNN` + `pack_padded_sequence`。
- [LSTM 与 GRU：门控记忆单元](./lstm-and-gru.md) — LSTM 三大门方程、GRU 简化版、双向 RNN、深度堆叠、PyTorch IMDB 情感分类完整示例。

### 序列到序列

- [Seq2Seq 与注意力机制](./seq2seq-and-attention.md) — 编码器-解码器、上下文瓶颈、Bahdanau 加性注意力、Luong 乘性注意力、PyTorch 数字反转完整示例，"Attention Is All You Need" 的铺垫。

## 三方资料

- [GRU 论文](./三方资料/gru.pdf) — Cho et al. 2014 提出的简化门控循环单元。
- [LSTM 原始论文](./三方资料/lstm-1997.pdf) — Hochreiter & Schmidhuber 1997 提出长短期记忆网络。
- [Seq2Seq 论文](./三方资料/seq2seq.pdf) — Sutskever et al. 2014 端到端序列到序列学习。
