# 开源仓库：RNN / LSTM

## PyTorch 序列建模示例

- 仓库：https://github.com/pytorch/examples/tree/main/word_language_model
- 简介：PyTorch 官方 LSTM/GRU 词语言建模示例，覆盖 packed sequence、tie weights 等工程细节。
- 适用：复现经典 RNN 基线、学习 PyTorch 序列 API。

## fairseq (Meta)

- 仓库：https://github.com/facebookresearch/fairseq
- 简介：Meta 出品的序列建模工具包，原始实现涵盖 LSTM seq2seq、Transformer、wav2vec、BART 等。
- 适用：研究级 NLP / 语音 seq2seq 实验、多语言翻译。

## huggingface/transformers

- 仓库：https://github.com/huggingface/transformers
- 简介：包含 RNN/LSTM 在内的完整 seq2seq 实现与预训练权重（虽然主流以 Transformer 为主，但仓库保留 LSTM 路径用于对比）。
- 适用：现代统一 API 下做 RNN vs Transformer 消融。
