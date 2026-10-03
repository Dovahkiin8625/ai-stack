# 开源仓库：模型服务化框架

## BentoML

- 仓库：https://github.com/bentoml/BentoML
- 简介：统一多框架（PyTorch / TF / sklearn / LLM）模型打包、服务化、部署工具；Bento 格式 + 适配 REST/gRPC/队列多种运行时。

## Ray Serve

- 仓库：https://github.com/ray-project/ray
- 简介：基于 Ray 的模型服务平台，原生支持自动扩缩容、流量切分、组合 DAG；与 Ray Tune / Train 生态打通。

## KServe

- 仓库：https://github.com/kserve/kserve
- 简介：Kubernetes 原生模型服务平台，兼容 Triton / TF Serving / TorchServe，支持 transformer / 自定义 predictor。

## Seldon Core

- 仓库：https://github.com/SeldonIO/seldon-core
- 简介：Kubernetes 上的 ML 部署框架，支持 A/B 测试、金丝雀、复杂 inference graph。

## Hugging Face Text Generation Inference (TGI)

- 仓库：https://github.com/huggingface/text-generation-inference
- 简介：Rust + Python 的生产级 LLM 推理服务，支持 continuous batching、tensor parallelism、量化（bitsandbytes / GPT-Q）。
