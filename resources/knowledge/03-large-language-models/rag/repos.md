# 开源仓库：RAG 工具链

## LangChain

- 仓库：https://github.com/langchain-ai/langchain
- 简介：提供完整的 RAG 工具链：DocumentLoader / Splitter / Embeddings / Retriever / ReRanker 等可组合模块。

## LlamaIndex

- 仓库：https://github.com/run-llama/llama_index
- 简介：以数据为中心的 RAG 框架，内置多种 Index 类型（Vector/List/Tree/Keyword）与高级 RAG 模式（SubQuestion、Router、Agentic RAG）。

## DSPy

- 仓库：https://github.com/stanfordnlp/dspy
- 简介：把 prompt 和检索策略当作可编译、可优化的模块，通过 BootstrapFewShot / MIPRO 等优化器自动寻找最优 pipeline。
- 适用：RAG/Agent pipeline 的自动 prompt 与权重优化。

## rerankers

- 仓库：https://github.com/AnswerDotAI/rerankers
- 简介：统一封装 Cohere / ColBERT / cross-encoder 等重排序模型，轻量集成到 RAG 检索后步骤。
- 适用：提升 RAG 召回质量。
