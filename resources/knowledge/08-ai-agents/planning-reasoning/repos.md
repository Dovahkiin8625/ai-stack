# 开源仓库：规划与推理

> 本目录以论文为主（CoT / ToT / ReAct / Reflexion），仓库参考如下。

## LangChain / LangGraph

- 仓库：https://github.com/langchain-ai/langgraph
- 简介：用图实现 CoT、ToT、ReAct 等规划与循环工作流；支持状态持久化和可视化。

## DSPy

- 仓库：https://github.com/stanfordnlp/dspy
- 简介：把 prompt / 模块组合当作可优化对象；其 ChainOfThought / ReAct 模块可与任意 LM 一起工作，并由优化器自动搜索最佳指令。
- 适用：把 CoT 思路落地到生产 pipeline 并自动优化。

## guidance (Microsoft)

- 仓库：https://github.com/microsoft/guidance
- 简介：通过模板约束 token 生成，把 CoT 等推理结构化插入 LLM 输出；可与 HF Transformers 配合。

## outlines (Normal Computing)

- 仓库：https://github.com/outlines-dev/outlines
- 简介：基于正则 / JSON Schema / CFG 约束 LLM 输出，确保推理链条结构合法。
