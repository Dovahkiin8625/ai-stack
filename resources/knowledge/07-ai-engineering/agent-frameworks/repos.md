# 开源仓库：Agent / LLM 应用框架

> 同时参考 03-large-language-models/llm-applications 与 08-ai-agents 子目录中的索引。

## LangChain

- 仓库：https://github.com/langchain-ai/langchain
- 简介：LLM 应用编排事实标准。提供 LLM / 工具 / 向量库 / 记忆 / 链的统一抽象，含 LangSmith 观测、LangServe 部署、LangGraph 工作流引擎。

## LlamaIndex

- 仓库：https://github.com/run-llama/llama_index
- 简介：以「数据 → 索引 → 查询」为主线的 RAG / Agent 框架，对复杂文档 ingestion 与检索优化做得深入。

## Haystack (deepset)

- 仓库：https://github.com/deepset-ai/haystack
- 简介：端到端 NLP / RAG 流水线，工业部署能力（REST/gRPC）成熟。

## LangGraph

- 仓库：https://github.com/langchain-ai/langgraph
- 简介：用图状态机表达 agent 工作流，比传统 chain 更灵活；支持循环、人工介入、持久化。
- 适用：复杂 multi-step agent。
