# Inference Optimization

> 分类：**大语言模型** → **Inference Optimization**
> 路径：`resources/knowledge/03-large-language-models/inference-optimization`

本目录用于存放与「Inference Optimization」相关的学习资料。

## 收录范围

- 教材与讲义（Markdown / PDF）
- 论文（PDF）
- 讲稿与笔记（Markdown / Word / PPTX）

## 命名约定

- 每个子主题一个文件夹，文件夹命名用 kebab-case。
- 每个文件夹下放一个 `_index.md` 作为目录索引（阶段 2 由索引生成器读取）。
- 文件名建议：`YYYY-MM-DD-<title>.md` 或原文件名。

## 文章列表

- [LLM 推理基础：从 latency 到 throughput](./inference-fundamentals.md)
- [KV Cache 与 PagedAttention](./kv-cache-paged-attention.md)
- [量化技术：INT8 / INT4 / GGUF / AWQ / GPTQ](./quantization-techniques.md)
- [Speculative Decoding 与推理加速](./speculative-decoding.md)
- [推理服务化：vLLM、TGI 与生产部署](./inference-serving.md)
