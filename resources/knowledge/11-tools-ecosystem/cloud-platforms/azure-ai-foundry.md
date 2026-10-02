# Azure AI Foundry：从 Azure OpenAI 到模型目录与 Agent 服务

微软 Azure 在 AI 云服务上的布局围绕 Azure AI Foundry（前身为 Azure AI Studio）：统一接入 OpenAI/Anthropic/Meta 等多家模型，提供 RAG、Fine-tuning、Agent 编排、评测、安全合规等企业级能力。本文梳理 Azure AI 服务全景、与 AWS Bedrock 的对比，以及何时该选 Azure。

## 一、Azure AI 服务全景

```python
# Azure AI 服务分类
azure_ai_stack = {
    "compute": [
        "Azure VMs (ND H100 v5 series)",   # H100 GPU 实例
        "Azure ND MI300X v5",              # AMD MI300X GPU
        "Azure Kubernetes Service (AKS)",  # K8s 容器编排
    ],
    "managed_ml": [
        "Azure Machine Learning",          # 端到端 ML 平台
        "Azure ML Studio",                  # 可视化建模
    ],
    "model_as_service": [
        "Azure OpenAI",                  # OpenAI 模型独占
        "Azure AI Foundry",              # 多家模型 + Agent
        "Phi-3 / Phi-4 (微软自研)",       # 小模型
    ],
    "applications": [
        "Azure AI Vision",                # 视觉
        "Azure AI Language",              # NLP
        "Azure AI Speech",                # 语音
        "Azure AI Document Intelligence", # 文档
        "Azure AI Content Safety",        # 内容安全",
    ],
}
```

## 二、Azure OpenAI：OpenAI 模型的"独占通道"

```python
# Azure OpenAI 与 OpenAI 直连 API 的对比
azure_openai_vs_openai = {
    "azure_openai_advantages": [
        "1. 企业级合规（HIPAA、GDPR、ISO、SOC）",
        "2. Azure AD / Entra ID 身份认证",
        "3. VNet 私有部署 + Private Link",
        "4. 数据不用于训练（合同保障）",
        "5. 与 Azure 生态深度集成（Key Vault、Monitor）",
        "6. PTU（Provisioned Throughput Units）保证 SLA",
    ],
    "azure_openai_disadvantages": [
        "1. 模型版本通常滞后 OpenAI 直连 1-2 周",
        "2. 价格略高于 OpenAI 直连",
        "3. 配额管控更严",
        "4. 部分功能（如 Realtime API）限制多",
    ],
    "openai_direct_advantages": [
        "1. 最新模型即时可用（o3、o4-mini）",
        "2. 灵活度高、配额松",
        "3. 价格略低",
    ],
}
```

### 用法示例

```python
from openai import AzureOpenAI

# Azure OpenAI 客户端
client = AzureOpenAI(
    azure_endpoint="https://YOUR-RESOURCE.openai.azure.com/",
    api_key="YOUR-API-KEY",  # 生产环境用 Key Vault
    api_version="2024-10-21",
)

# 调用 GPT-4o
response = client.chat.completions.create(
    model="gpt-4o",  # Azure 部署名
    messages=[
        {"role": "system", "content": "你是一名资深 Python 工程师"},
        {"role": "user", "content": "解释 asyncio 的核心原理"},
)
print(response.choices[0].message.content)
```

## 三、Azure AI Foundry：模型目录与 Agent 服务

### 核心定位

```python
# Azure AI Foundry（前身 Azure AI Studio）
# 整合多家模型 + Agent 能力

foundry_features = {
    "model_catalog": [
        "Azure OpenAI (GPT-4o, o1, o3)",
        "Anthropic Claude 4.5 (Opus/Sonnet/Haiku)",
        "Meta Llama 3.2/3.3/4",
        "Mistral Large/Mixtral",
        "Cohere Command R+",
        "Microsoft Phi-3/Phi-4 (自家小模型)",
        "DeepSeek R1/V3",
        "Hugging Face 模型",
    ],
    "capabilities": [
        "模型调用（统一 SDK）",
        "Fine-tuning",
        "RAG（带 Vector Index）",
        "Prompt Flow（可视化 Prompt 工程）",
        "Agent Service（多 Agent 编排）",
        "AI Evaluation（自动评测）",
        "Content Safety（内容安全）",
    ],
}
```

### 模型调用（统一 SDK）

```python
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# 统一的 Inference SDK 调用不同模型
client = ChatCompletionsClient(
    endpoint="https://YOUR-RESOURCE.services.ai.azure.com/models",
    credential=AzureKeyCredential("YOUR-KEY"),
)

# 调用 Llama 4
response = client.complete(
    model="Llama-4-90B",
    messages=[
        SystemMessage("你是一名数学老师"),
        UserMessage("解释勾股定理"),
    ],
    max_tokens=512,
)

# 调用 Phi-4
response = client.complete(
    model="Phi-4",
    messages=[UserMessage("翻译成英文：你好世界")],
    max_tokens=128,
)

# 调用 DeepSeek R1
response = client.complete(
    model="DeepSeek-R1",
    messages=[UserMessage("解这道数学题...")],
    max_tokens=2048,
)
```

### Prompt Flow：可视化 Prompt 工程

```python
# Prompt Flow 是 Azure 的可视化 Prompt 编排工具
# 核心概念
prompt_flow_concepts = [
    "Flow: 一个完整的 Prompt 流程",
    "Node: 流程中的一个节点（LLM/Python/工具）",
    "Variant: 参数变体（实验不同模型/超参）",
    "Evaluation: 评估流",
    "Deployment: 部署到托管端点",
]

# 示例流程（YAML 格式）
flow_yaml = """
name: rag-flow
inputs:
  question:
    type: string
nodes:
  - name: retrieve
    type: python
    source:
      type: code
      path: retrieve.py
    inputs:
      query: ${inputs.question}
  - name: generate
    type: llm
    source:
      type: code
      path: generate.jinja2
    connection: azure_openai
    inputs:
      deployment_name: gpt-4o
      temperature: 0.7
      prompt: ${retrieve.output.documents_text}
outputs:
  answer: ${generate.output}
"""
```

## 四、Azure AI Agent Service

```python
# Azure AI Agent Service
# 2025 年 GA，支持多 Agent 编排

agent_service_features = {
    "agent_types": [
        "Single Agent（单一 Agent）",
        "Multi-Agent（多 Agent 协作）",
        "Connected Agent（互联 Agent）",
    ],
    "tools": [
        "Bing Search",
        "Code Interpreter",
        "Function Calling",
        "Azure AI Search",
        "OpenAPI",
        "Fabric Data Agent",
        "SharePoint",
    ],
    "orchestration": "Azure 自研编排框架",
    "monitoring": "Application Insights + Tracing",
}

# 简单 Agent 示例
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import Agent, AgentThread, Tool

# 工具定义
code_interpreter = CodeInterpreterTool()
bing_search = BingGroundingTool(connection=bing_connection)

# 创建 Agent
agent = project.agents.create_agent(
    model="gpt-4o",
    name="research-agent",
    instructions="你是一名研究助手，擅长搜集资料并撰写报告",
    tools=[code_interpreter.definitions[0], bing_search.definitions[0]],
)

# 与 Agent 交互
thread = project.agents.create_thread()
project.agents.create_message(
    thread_id=thread.id,
    role="user",
    content="调研 2026 年最火的 AI 趋势",
)
run = project.agents.create_and_process_run(
    thread_id=thread.id,
    agent_id=agent.id,
)
messages = project.agents.list_messages(thread_id=thread.id)
```

## 五、Azure AI Content Safety

```python
# Azure AI Content Safety 提供多层内容安全检测
content_safety_categories = [
    "Hate (仇恨)",
    "SelfHarm (自残)",
    "Sexual (色情)",
    "Violence (暴力)",
]

# 用法
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions

client = ContentSafetyClient(endpoint=endpoint, credential=credential)

request = AnalyzeTextOptions(
    text="要检测的文本",
    categories=["Hate", "SelfHarm", "Sexual", "Violence"],
    output_type="FourSeverityLevels",  # Safe/Low/Medium/High
)
response = client.analyze_text(request)

# 响应
{
    "hate_result": {"severity": 0},
    "self_harm_result": {"severity": 0},
    "sexual_result": {"severity": 0},
    "violence_result": {"severity": 2},  # Medium
}
```

## 六、Azure AI Search：托管 RAG 检索

```python
# Azure AI Search（前身 Cognitive Search）
# 完整的搜索 + RAG 平台

azure_search_features = {
    "indexing": [
        "自动文档抽取（PDF/Word/Excel）",
        "AI 扩充（OCR/实体抽取/关键短语）",
        "技能集（Skillsets）",
    ],
    "query": [
        "向量检索（kNN + HNSW）",
        "混合检索（BM25 + 向量）",
        "语义排序（Semantic Ranker）",
        "过滤 + Facet",
    ],
    "integrations": [
        "OpenAI / Foundry",
        "Azure AI Foundry Knowledge",
        "Copilot Studio",
    ],
}

# 创建向量索引
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration,
)

index = SearchIndex(
    name="docs-index",
    fields=[
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile="default",
        ),
    ],
    vector_search=VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default")],
    ),
)
index_client.create_or_update_index(index)
```

## 七、Azure ML：训练与微调

```python
# Azure ML Studio 提供模型训练与微调能力
# 与 SageMaker 类似

azure_ml_features = {
    "compute": [
        "GPU 实例（ND H100 v5）",
        "Kubernetes 计算",
        "低优先级 VM（折扣）",
    ],
    "training": [
        "SDK / CLI / Studio 三种方式",
        "分布式训练（PyTorch/Hugging Face）",
        "AutoML",
        "Sweeps（超参搜索）",
    ],
    "deployment": [
        "Managed Endpoints",
        "Kubernetes Endpoints",
        "Batch Endpoints",
        "Serverless Endpoints",
    ],
}

# 微调示例
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import Model, Environment

# 微调命令
job = command(
    code="./src",
    command="python train.py --lr ${{inputs.lr}}",
    inputs={"lr": Input(type="number", default=0.001)},
    environment=environment,
    compute="gpu-cluster",
)
ml_client.jobs.create_or_update(job)
```

## 八、Phi-3/Phi-4：微软自研小模型

```python
# 微软自研小模型——性价比突出
phi_models = {
    "Phi-3-mini": {"params": "3.8B", "context": "4K", "use": "边缘/移动端"},
    "Phi-3-small": {"params": "7B", "context": "8K", "use": "中等任务"},
    "Phi-3-medium": {"params": "14B", "context": "4K", "use": "复杂任务"},
    "Phi-4": {"params": "14B", "context": "16K", "use": "推理/数学/科学"},
    "Phi-4-multimodal": {"params": "5.6B", "context": "16K", "use": "多模态"},
}

# Phi-4 跑分（vs 同档）
phi4_benchmarks = {
    "MMLU": "84.6%",  # 接近 GPT-4o
    "GSM8K": "92.5%",
    "HumanEval": "82.8%",
    "MT-Bench": "8.7",
}
```

## 九、Azure vs AWS Bedrock vs GCP Vertex AI

```python
# 三家对比
cloud_ai_comparison = {
    "Azure": {
        "优势": [
            "OpenAI 模型独占（Azure OpenAI）",
            "企业生态（Office 365、Teams 集成）",
            "Prompt Flow 可视化",
            "Phi 系列小模型性价比",
        ],
        "劣势": [
            "非 OpenAI 模型支持较 AWS 略少",
            "国际版与中国版隔离",
        ],
    },
    "AWS Bedrock": {
        "优势": [
            "模型最齐全（Anthropic/Llama/Mistral 等）",
            "自研芯片性价比高（Trainium2）",
            "SageMaker 端到端 ML",
        ],
        "劣势": [
            "没有 OpenAI 模型",
            "无 OpenAI 等价的企业级 Prompt 工具",
        ],
    },
    "GCP Vertex AI": {
        "优势": [
            "Gemini 多模态能力强",
            "TPU 自研芯片",
            "与 Google 生态集成（Gmail/Docs）",
            "BigQuery 集成",
        ],
        "劣势": [
            "Gemini 之外模型较少",
        ],
    },
}
```

## 十、Azure AI 实战建议

```python
# Azure AI 选型建议
selection_guidance = {
    "use_azure_openai_when": [
        "需要 OpenAI 模型 + 企业合规",
        "已在 Microsoft 生态（Office/Teams）",
        "需要 PTU 保证吞吐量 SLA",
    ],
    "use_azure_ai_foundry_when": [
        "需要多家模型灵活切换",
        "需要 RAG/Agent/可视化 Prompt 工具",
        "需要多 Agent 编排",
    ],
    "use_phi_models_when": [
        "成本敏感场景",
        "边缘/移动端部署",
        "中小型任务（不需要 GPT-4 等级）",
    ],
    "use_azure_ml_when": [
        "需要自定义训练/微调",
        "需要 Kubernetes 部署",
        "已有 Azure ML 投入",
    ],
}
```

## 小结

Azure AI Foundry 是微软的统一 AI 服务平台，整合 Azure OpenAI + 多家模型 + Prompt Flow + Agent Service + 内容安全。**OpenAI 模型首选 Azure OpenAI，多模型/RAG/Agent 首选 Foundry，成本敏感首选 Phi 系列**。下一篇我们将讨论 **GCP Vertex AI**——谷歌的 AI 云服务平台及其 Gemini 多模态优势。