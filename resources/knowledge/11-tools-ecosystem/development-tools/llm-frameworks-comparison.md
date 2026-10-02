# LLM 推理框架对比：vLLM、TGI、TensorRT-LLM、LMDeploy

生产环境部署 LLM 需要高吞吐量、低延迟、低显存——这催生了一批专用推理框架。本文系统对比主流 LLM 推理框架（vLLM、TGI、TensorRT-LLM、LMDeploy、SGLang、MLX）的性能特性、适用场景、部署方式。

## 一、为什么需要专用推理框架

```python
# 通用 PyTorch 推理 LLM 的挑战
challenges = {
    "显存爆炸": "KV Cache 占用大量显存",
    "GPU 利用率低": "生成是 sequential，GPU 经常空闲",
    "吞吐量低": "无法处理高并发请求",
    "批处理弱": "静态 padding 浪费算力",
    "prefix memory": "静态预分配，无法动态调度",
}

# 专用推理框架的核心优化
optimizations = [
    "1. PagedAttention（vLLM）——分页式 KV Cache 管理",
    "2. Continuous Batching —— 动态批处理",
    "3. Speculative Decoding —— 投机解码",
    "4. Quantization —— 量化（INT4/INT8）",
    "5. Tensor Parallelism —— 张量并行",
    "6. Prefix Sharing —— 前缀共享",
    "7. Kernel Fusion —— 内核融合",
    "8. 推测执行 + 提前终止",
]
```

## 二、vLLM：当下最流行的推理框架

### 核心特性

```python
# vLLM（UC Berkeley，2023）
# 核心：PagedAttention —— 分页式管理 KV Cache

vllm_features = {
    "核心创新": "PagedAttention（KV Cache 虚拟内存分页）",
    "显存效率": "比传统方案高 4-24 倍",
    "吞吐量": "vs HuggingFace Transformers，14-24x 提升",
    "Continuous Batching": "支持",
    "Speculative Decoding": "支持",
    "Tensor Parallel": "支持",
    "模型支持": "几乎所有主流开源 LLM",
    "部署": "OpenAI 兼容 API",
}
```

### 用法示例

```python
# 离线推理（batch）
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,         # 4 卡并行
    gpu_memory_utilization=0.9,     # 显存利用率
    dtype="float16",                # 或 bfloat16
    quantization="awq",             # 量化方案
    max_model_len=8192,
)

prompts = ["什么是 LLM?", "解释量子计算"]
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

```python
# 在线服务（OpenAI 兼容 API）
# 启动：vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

import openai
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
)
print(response.choices[0].message.content)

# 性能监控
# curl http://localhost:8000/metrics
```

### 进阶特性

```python
# Speculative Decoding（小模型推测 + 大模型验证）
vllm.SpeculativeConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",  # 推测模型
    num_speculative_tokens=5,                  # 推测 token 数
)

# Prefix Caching（前缀共享）
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,   # 相同前缀缓存
)

# Chunked Prefill（分块预填充）
# 长 prompt 分块处理，提升交互性
```

## 四、TGI (Text Generation Inference)

```python
# HuggingFace TGI
# 生态完善，HuggingFace 一等公民

tgi_features = {
    "开发者": "HuggingFace",
    "后端": "Rust + Python",
    "特性": [
        "Continuous Batching",
        "Paged Attention",
        "Quantization (GPTQ/AWQ/BitsAndBytes)",
        "Tensor Parallelism",
        "Safetensors 权重",
    ],
    "部署": "Docker / Kubernetes",
    "API": "REST",
}
```

```python
# 部署
# docker run -p 8080:80 \
#   -v $PWD/data:/data \
#   ghcr.io/huggingface/text-generation-inference:latest \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --quantize awq \
#   --num-shard 2

# 调用
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")
response = client.text_generation(
    "什么是 vLLM?",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)
```

## 五、TensorRT-LLM：NVIDIA 极致优化

```python
# TensorRT-LLM（NVIDIA）
# 针对 NVIDIA GPU 极致优化，性能最强

tensorrt_llm_features = {
    "开发者": "NVIDIA",
    "后端": "TensorRT",
    "性能": "vs vLLM，1.5-3x 提升（NVIDIA GPU 上）",
    "优化": [
        "Kernel Fusion（层融合）",
        "In-flight Batching",
        "INT4/INT8/FP8 量化",
        "Tensor Parallelism",
        "Pipeline Parallelism",
        "Custom CUDA Kernels",
    ],
    "缺点": [
        "编译时间长（首次部署）",
        "模型转换复杂",
        "生态较封闭",
        "调试困难",
    ],
}
```

```python
# TensorRT-LLM 部署流程
# 1. 转换模型
# trtllm-build --checkpoint_dir ./llama_ckpt \
#               --output_dir ./llama_engine \
#               --gemm_plugin float16 \
#               --max_batch_size 64 \
#               --max_input_len 4096 \
#               --max_output_len 1024

# 2. 启动服务
# trtllm-serve ./llama_engine

# 3. Python 调用
from tensorrt_llm import LLM

llm = LLM(model_dir="./llama_engine")
output = llm.generate(["你好"])
```

## 六、LMDeploy：清华 LLM 部署框架

```python
# LMDeploy（上海 AI Lab）
# 国产 LLM 部署引擎，针对中文场景

lmdeploy_features = {
    "开发者": "上海 AI Lab",
    "后端": "C++/CUDA + Python",
    "优势": [
        "针对 InternLM 等中文模型深度优化",
        "Turbomind 推理引擎（自研）",
        "支持量化（AWQ/GPTQ）",
        "Continuous Batching",
        "高性能 KV Cache 管理",
    ],
    "工具链": [
        "模型转换工具",
        "量化工具",
        "服务部署",
        "可视化界面（Gradio）",
    ],
}
```

```python
# 用法
from lmdeploy import pipeline, TurbomindEngineConfig

pipe = pipeline(
    "internlm/internlm2_5-7b-chat",
    backend_config=TurbomindEngineConfig(
        model_format="awq",     # AWQ 量化
        tp=5,             # 4 卡张量并行
        cache_max_entry_count=0.7,
    ),
)

response = pipe(["解释深度学习"])
print(response[0].text)

# 服务化部署
# lmdeploy serve api_server internlm/internlm2_5-7b-chat --backend turbomind --tp 5
```

## 七、SGLang：面向 Agent 与结构化输出

```python
# SGLang（Stanford / LMSYS 团队）
# 面向 LLM 程序与结构化输出

sglang_features = {
    "开发者": "Stanford / LMSYS",
    "核心理念": "把 LLM 调用看作"程序"",
    "特性": [
        "CoT/分支/循环等 DSL",
        "RadixAttention（前缀共享）",
        "结构化输出（JSON 模式）",
        "Speculative Decoding",
        "与 Chatbot Arena 同源团队",
    ],
    "优势": [
        "复杂 LLM 程序（Agent）友好",
        "结构化输出高效",
        "高并发场景性能优",
    ],
}
```

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, question1, question2):
    s += sgl.user(question1)
    s += sgl.assistant(sgl.gen("answer1"))
    s += sgl.user(question2)
    s += sgl.assistant(sgl.gen("answer2"))

# 执行
state = multi_turn_qa.run(
    question1="什么是 vLL?",
    question2="它与 TGI 的区别?",
)

print(state["answer1"], state["answer2"])
```

## 八、MLX：Apple Silicon 推理

```python
# MLX（Apple）
# 专为 Apple Silicon（M1/M2/M3）优化

mlx_features = {
    "开发者": "Apple",
    "支持设备": "Mac M1+",
    "特性": [
        "Unified Memory",
        "量化（4-bit/8-bit）",
        "LoRA 微调",
        "本地推理",
    ],
    "适合": "Mac 本地开发、原型验证",
}
```

```python
# 用法
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
response = generate(
    model,
    tokenizer,
    prompt="解释 MLX 框架",
    max_tokens=256,
)
print(response)
```

## 九、性能基准对比

```python
# Llama-3.1-70B 在 4 卡 H100 上的对比（DD 公开数据）
# 注：实际数据以官方 benchmark 为准

benchmark = {
    "framework": {
        "vLLM 0.6": {
            "throughput": "~3500 tokens/s",
            "latency_p50": "85ms",
            "gpu_memory": "~75GB",
        },
        "TensorRT-LLM 0.10": {
            "throughput": "~5200 tokens/s",
            "latency_p50": "62ms",
            "gpu_memory": "~70GB",
        },
        "TGI 2.3": {
            "throughput": "~2800 tokens/s",
            "latency_p50": "92ms",
            "gpu_memory": "~78GB",
        },
        "LMDeploy Turbomind": {
            "throughput": "~3100 tokens/s",
            "latency_p50": "78ms",
            "gpu_memory": "~76GB",
        },
    },
    "注意": [
        "性能随 batch size、prompt 长度、模型版本变化",
        "NVIDIA TensorRT-LLM 通常最快",
        "vLLM 在生态/易用性最优",
    ],
}
```

## 十、量化方案

```python
# LLM 推理常用量化方案

quantization_methods = {
    "GPTQ": {
        "精度": "INT4 / INT8",
        "速度": "快",
        "质量": "好",
        "适用": "通用 LLM",
        "工具": "AutoGPTQ",
    },
    "AWQ": {
        "精度": "INT4",
        "速度": "快",
        "质量": "好（保护重要权重）",
        "适用": "通用 LLM",
        "工具": "AutoAWQ",
    },
    "BitsAndBytes": {
        "精度": "INT4 / INT8",
        "速度": "中",
        "质量": "好",
        "适用": "HuggingFace 集成",
        "工具": "transformers 集成",
    },
    "SmoothQuant": {
        "精度": "INT8",
        "速度": "快",
        "质量": "好",
        "适用": "NVIDIA GPU",
    },
    "FP8 (E4M3 / E5M2)": {
        "精度": "FP8",
        "速度": "快",
        "质量": "好",
        "适用": "H100/Ada",
    },
    "GGUF (llama.cpp)": {
        "精度": "INT4/INT5/INT6/INT8",
        "速度": "中",
        "质量": "好",
        "适用": "CPU/Mac/边缘",
    },
}
```

## 十一、框架选型决策

```python
# 推理框架选型决策

decision_framework = {
    "选 vLLM 当": [
        "1. 通用 LLM 部署（最稳选择）",
        "2. 需要 OpenAI 兼容 API",
        "3. 高吞吐量场景",
        "4. 团队不熟悉 CUDA 内核优化",
    ],
    "选 TensorRT-LLM 当": [
        "1. NVIDIA GPU + 性能极致要求",
        "2. 大批量生产环境",
        "3. 团队熟悉 TensorRT",
        "4. 可以接受编译时间",
    ],
    "选 TGI 当": [
        "1. 已用 HuggingFace 生态",
        "2. Kubernetes 部署",
        "3. 需要完善的监控",
    ],
    "选 LMDeploy 当": [
        "1. 部署中文模型（InternLM/ChatGLM）",
        "2. 国产化需求",
        "3. 需要可视化 Web 界面",
    ],
    "选 SGLang 当": [
        "1. 复杂 LLM 程序（Agent）",
        "2. 结构化输出需求",
        "3. 高并发对话",
    ],
    "选 MLX 当": [
        "1. Mac 本地开发",
        "2. 原型验证",
        "3. 边缘部署",
    ],
}
```

## 十二、生产部署建议

```python
# 生产部署的关键考虑

production_checklist = {
    "1. 模型选型": "蒸馏 vs 全参数 vs MoE",
    "2. 量化方案": "AWQ/FP8/GPTQ，权衡质量与性能",
    "3. 并行策略": "TP/PP，根据模型大小",
    "4. 推理引擎": "vLLM/TGI/TensorRT-LLM",
    "5. 监控指标": [
        "吞吐量（tokens/s）",
        "延迟（TTFT, TPOT）",
        "显存使用率",
        "错误率",
        "GPU 利用率",
    ],
    "6. 弹性伸缩": "K8s HPA / queue",
    "7. 缓存策略": "Prefix Cache / Semantic Cache",
    "8. 限流熔断": "保护后端不被突破",
    "9. A/B 测试": "新旧模型对比",
    "10. 成本优化": "GPU 选型、Spot Instance",
}
```

## 小结

LLM 推理框架的核心是 **KV Cache 管理 + Continuous Batching + 量化**。**通用场景首选 vLLM，性能极致选 TensorRT-LLM，中文场景选 LMDeploy，复杂 LLM 程序选 SGLang**。生产环境需关注**延迟、吞吐量、显存、成本**四要素的权衡。下一篇我们讨论 **实验追踪**——Weights & Biases、MLflow、TensorBoard。