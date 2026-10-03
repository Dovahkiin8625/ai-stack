# 开源仓库：模型部署 / 推理优化

## NVIDIA Triton Inference Server

- 仓库：https://github.com/triton-inference-server/server
- 简介：支持 TensorFlow / PyTorch / ONNX / TensorRT 多后端，dynamic batching、模型编排、HTTP/gRPC 接口；生产级 GPU 推理服务。

## TensorRT

- 仓库：https://github.com/NVIDIA/TensorRT
- 简介：NVIDIA GPU 上的高性能推理优化器（层融合、量化、内核调优），吞吐量可达原生框架数倍。

## ONNX Runtime

- 仓库：https://github.com/microsoft/onnxruntime
- 简介：跨平台跨框架推理引擎（CPU/GPU/NPU），支持 ONNX / TVM / TensorRT EP；性能优于原生框架。

## OpenVINO (Intel)

- 仓库：https://github.com/openvinotoolkit/openvino
- 简介：Intel CPU/GPU/NPU 推理加速，模型转换 + 量化 + 部署一站式。

## TorchServe (AWS + Meta)

- 仓库：https://github.com/pytorch/serve
- 简介：PyTorch 官方模型服务框架，支持自定义 handler / 多模型加载 / metrics。
