# 开源仓库：LLM 推理优化

## vLLM

- 仓库：https://github.com/vllm-project/vllm
- 简介：实现 PagedAttention 与连续批处理（continuous batching），是当前吞吐量最高的主流开源推理引擎之一。
- 适用：高 QPS 在线服务、OpenAI-compatible API。

## llama.cpp

- 仓库：https://github.com/ggerganov/llama.cpp
- 简介：纯 C/C++ 实现 LLaMA 系列推理，支持 GGUF 量化（Q4/Q5/Q8），CPU/GPU/Apple Silicon 全平台运行。
- 适用：本地部署、低显存推理、端侧推理。

## TensorRT-LLM (NVIDIA)

- 仓库：https://github.com/NVIDIA/TensorRT-LLM
- 简介：基于 TensorRT 的 LLM 推理加速库，支持 in-flight batching、KV cache 优化、INT8/INT4 量化。
- 适用：NVIDIA GPU 上的生产级 LLM 服务。

## sglang

- 仓库：https://github.com/sgl-project/sglang
- 简介：通过 RadixAttention 与结构化编程原语优化 LLM 程序执行，对 agent / 多调用场景吞吐极佳。
- 适用：复杂 LLM 工作流（多轮、agent、tool use）。
