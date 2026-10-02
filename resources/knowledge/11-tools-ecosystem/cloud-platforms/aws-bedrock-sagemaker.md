# AWS AI/ML 云平台：Bedrock、SageMaker 与自研芯片

AWS 在 AI/ML 云服务上提供全栈能力：从底层自研芯片（Trainium/Inferentia），到托管训练平台（SageMaker），到模型即服务（Bedrock）。本文梳理 AWS AI 服务全景、定价模型、与 OpenAI/Anthropic 直连 API 的取舍，帮助判断什么时候该用 AWS、什么时候该用裸 API。

## 一、AWS AI/ML 服务全景

```python
# AWS AI 服务分类
aws_ai_stack = {
    "compute": [
        "EC2 (P4/P5 instances with H100/A100)",   # GPU 实例
        "Trainium2 / Trainium",                   # 自研训练芯片
        "Inferentia2 / Inferentia",               # 自研推理芯片
        "Neuron SDK",                              # 自研芯片 SDK
    ],
    "managed_ml": [
        "SageMaker",          # 端到端 ML 平台
        "SageMaker JumpStart",  # 预训练模型市场
    ],
    "model_as_service": [
        "Bedrock",             # 多家模型托管
        "Bedrock Marketplace",  # 模型市场
    ],
    "applications": [
        "Rekognition",          # 视觉
        "Comprehend",           # NLP
        "Translate",            # 翻译
        "Polly / Lex",          # 语音
        "Forecast",             # 时序预测
        "Fraud Detector",       # 异常检测
    ],
}
```

## 二、Amazon Bedrock：模型即服务

### 核心定位

```python
# Bedrock 的核心定位
# 1. 多家基础模型 API 托管（统一接口）
# 2. 微调（RAG/微调/Fine-tuning）
# 3. 知识库 + Agent 能力
# 4. 安全合规（VPC 隔离、IAM、审计）
# 5. 多模型路由（按场景路由到不同模型）

bedrock_features = {
    "models_supported": [
        "Anthropic Claude 4.5 (Opus/Sonnet/Haiku)",
        "Meta Llama 4 (90B/70B/8B)",
        "Mistral Large/Mixtral",
        "Amazon Titan (自家)",
        "Cohere Command R+",
        "AI21 Jamba",
        "Stability AI (SDXL)",
    ],
    "capabilities": [
        "模型调用",
        "Fine-tuning",
        "Continued Pre-training",
        "Knowledge Base (RAG)",
        "Agents",
        "Guardrails",
        "Model Evaluation",
    ],
}
```

### 用法示例

```python
import boto3
import json

# Bedrock 调用示例
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

def invoke_claude(prompt: str, max_tokens: int = 1024) -> str:
    """通过 Bedrock 调用 Claude 4.5 Sonnet"""
    body = {
        "anthropic_version": "bedrock-2024-10-22",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    response = bedrock.invoke_model(
        modelId="anthropic.claude-sonnet-4-5-20250929",
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

# 调用 Llama 4
def invoke_llama(prompt: str) -> str:
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.7,
    }
    response = bedrock.invoke_model(
        modelId="meta.llama4-90b-instruct-v1:0",
        body=json.dumps(body),
    )
    return json.loads(response["body"].read())["generation"]
```

### Knowledge Base（RAG）

```python
# Bedrock Knowledge Base 配置
knowledge_base_config = {
    "name": "company-docs-kb",
    "data_source": "s3://my-bucket/docs/",  # S3 数据源
    "embedding_model": "amazon.titan-embed-text-v2:0",
    "vector_store": "opensearch",  # 或 Pinecone/RDS pgvector
    "chunk_strategy": {
        "type": "hierarchical",
        "max_tokens": 300,
        "overlap": 50,
    },
    "metadata": ["source", "date", "author"],
}

# RAG 调用
response = bedrock.retrieve_and_generate(
    input={"text": "公司年假政策是什么?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "KB12345",
            "modelArn": "anthropic.claude-sonnet-4-5-20250929",
        },
    },
)
print(response["output"]["text"])
```

## 三、Amazon SageMaker：端到端 ML 平台

### 核心定位

```python
# SageMaker 提供 ML 全流程托管
sagemaker_components = {
    "ground_truth": "数据标注服务",
    "data_wrangler": "数据准备可视化",
    "feature_store": "特征存储",
    "training": "分布式训练",
    "autopilot": "AutoML",
    "hyperparameter_tuning": "超参搜索",
    "endpoint": "模型部署推理",
    "pipeline": "MLOps 流水线",
    "clarify": "公平性与可解释性",
    "model_monitor": "模型监控",
}
```

### 训练任务示例

```python
# SageMaker 启动训练任务
import sagemaker
from sagemaker.huggingface import HuggingFace

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# HuggingFace 训练
estimator = HuggingFace(
    entry_point="train.py",          # 训练脚本
    source_dir="./src",
    instance_type="ml.p5.48xlarge",   # 8×H100 GPU
    instance_count=2,
    transformers_version="4.46",
    pytorch_version="2.3",
    py_version="py311",
    role=role,
    hyperparameters={
        "model_name": "meta-llama/Llama-3.1-8B",
        "epochs": 3,
        "batch_size": 4,
        "lr": 2e-5,
    },
)

estimator.fit({"train": "s3://my-bucket/train/"})
```

### 部署端点

```python
# 部署到实时推理端点
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",  # A10G GPU
    endpoint_name="llama-8b-endpoint",
)

# 调用
result = predictor.predict({
    "inputs": "解释量子计算",
    "parameters": {"max_new_tokens": 256},
})
```

## 四、Trainium2 / Inferentia2：自研芯片

### 芯片定位

```python
# AWS 自研芯片
aws_chips = {
    "Trainium2": {
        "type": "训练",
        "vs_gpu": "vs H100，性价比 ~40% 提升",
        "use_case": "大模型分布式训练",
        "sdk": "Neuron SDK",
    },
    "Inferentia2": {
        "type": "推理",
        "vs_gpu": "vs A10G，性价比 ~70% 提升",
        "use_case": "大模型推理服务",
        "sdk": "Neuron SDK",
    },
}
```

### Neuron SDK 用法

```python
# Neuron SDK 训练示例（类似 PyTorch）
import torch
import torch_neuronx

# 编译模型到 Neuron 格式
model = MyModel()
model_neuron = torch_neuronx.trace(
    model,
    example_inputs=(torch.zeros(1, 3, 224, 224),),
)

# 在 Trainium2 上训练
from neuronx_distributed import NeuronxDistributedConfig
config = NeuronxDistributedConfig(
    tensor_parallel_size=8,
    pipeline_parallel_size=1,
)
# ... 启动分布式训练
```

## 五、定价模型

```python
# AWS AI 服务的定价模式
pricing_models = {
    "bedrock_on_demand": {
        "Claude Opus 4.5": "$15 / 1M input tokens, $75 / 1M output tokens",
        "Llama 4 90B": "$2.7 / 1M input, $2.7 / 1M output",
        "Mistral Large": "$4 / 1M input, $12 / 1M output",
    },
    "bedrock_provisioned": {
        "model_unit": "$50-200 / hour (预置吞吐量)",
        "use_case": "稳定吞吐量需求",
    },
    "sagemaker_training": {
        "ml.p5.48xlarge": "$98.32 / hour",
        "ml.p4d.24xlarge": "$32.77 / hour",
    },
    "sagemaker_endpoint": {
        "ml.g5.2xlarge": "$1.21 / hour",
        "ml.inf2.xlarge": "$0.76 / hour (Inferentia2)",
    },
}
```

## 六、什么时候用 AWS vs 直连 API

```python
# 选择 AWS Bedrock 还是直连 OpenAI/Anthropic API

decision_framework = {
    "use_bedrock_when": [
        "1. 数据合规要求（HIPAA、GDPR）—— AWS VPC 隔离",
        "2. 需要多家模型灵活切换",
        "3. 大规模推理需要 Inferentia2 降本",
        "4. 需要 RAG/Agent 整套能力",
        "5. 已用 AWS 生态（S3、IAM、VPC）",
    ],
    "use_direct_api_when": [
        "1. 简单调用、无合规需求",
        "2. 需要最新的 OpenAI 模型（o1/o3）",
        "3. 不想绑定到 AWS",
        "4. 团队小、不需要复杂的企业级控制",
        "5. 快速原型阶段",
    ],
}
```

## 七、Bedrock vs SageMaker 的取舍

```python
# Bedrock vs SageMaker 的取舍
bedrock_vs_sagemaker = {
    "Bedrock": {
        "适合": "调用预训练模型（零启动）",
        "优势": "快、便宜、模型多样",
        "劣势": "模型微调受限（只能支持部分模型）",
        "定价": "按 token",
    },
    "SageMaker": {
        "适合": "自训练/微调/部署自定义模型",
        "优势": "完全控制、灵活",
        "劣势": "需自己运维、成本较高",
        "定价": "按实例小时",
    },
}
```

## 八、AWS AI 安全与合规

```python
# AWS Bedrock 的安全能力
bedrock_security = {
    "data_privacy": [
        "数据不用于训练 AWS 模型",
        "支持 VPC 私有链接（PrivateLink）",
        "支持 KMS 加密",
    ],
    "compliance": [
        "HIPAA",
        "GDPR",
        "ISO 27001",
        "SOC 1/2/3",
        "FedRAMP",
    ],
    "guardrails": {
        "功能": "内容过滤、主题限制、PII 屏蔽",
        "用法": "Bedrock Guardrails API",
    },
    "iam": "细粒度 IAM 权限控制模型访问",
}
```

## 九、AWS AI 服务实战建议

```python
# 实战建议
best_practices = {
    "prototype": [
        "用 Bedrock Playground 体验不同模型",
        "用 On-Demand 模式小规模测试",
    ],
    "production": [
        "Provisioned Throughput 保证 SLA",
        "跨区域部署提升可用性",
        "CloudWatch + Bedrock Model Evaluation 监控",
    ],
    "cost_optimization": [
        "小流量：On-Demand",
        "中等流量：Provisioned + Auto Scaling",
        "大流量：Inferentia2 + Neuron 优化",
        "模型路由：简单任务用 Haiku/小模型，复杂任务用 Opus",
    ],
    "compliance": [
        "用 Bedrock + PrivateLink + KMS",
        "打开 CloudTrail 审计日志",
        "配置 Guardrails 内容过滤",
    ],
}
```

## 小结

AWS 提供从自研芯片（Trainium2/Inferentia2）到 SageMaker（端到端 ML 平台）再到 Bedrock（模型即服务）的完整 AI 栈。**Bedrock 适合快速接入多家基础模型，SageMaker 适合自定义训练/部署，Trainium2/Inferentia2 适合大规模成本优化**。下一篇我们将讨论 **Azure AI Foundry**——微软的 AI 云服务平台。