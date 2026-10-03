# 开源仓库：序列标注

## Hugging Face Transformers + Token Classification

- 仓库：https://github.com/huggingface/transformers
- 简介：内置 NER / POS / Chunking 等 token classification pipeline；`AutoModelForTokenClassification` 一行接入 BERT/RoBERTa 标注任务。

## flair

- 仓库：https://github.com/flairNLP/flair
- 简介：Zalando 出品的 NLP 框架，提供 BiLSTM-CRF、Transformer + CRF、上下文字符串嵌入（contextual string embeddings），NER/POG/Chunking 评测领先。
- 适用：高质量 NER、PoS、关系抽取。

## spaCy

- 仓库：https://github.com/explosion/spaCy
- 简介：工业级 NLP 库，提供高速 tokenizer / POS / NER / 依存解析；与 HuggingFace / Transformers 互操作。
- 适用：生产级 NLP 流水线、多语种支持。
