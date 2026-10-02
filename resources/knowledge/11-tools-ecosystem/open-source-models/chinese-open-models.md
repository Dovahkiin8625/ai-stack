# 中文开源模型：Qwen、DeepSeek、ChatGLM、Yi 与 Baichuan

中文开源 LLM 生态在过去两年实现弯道超车——Qwen、DeepSeek、ChatGLM 等模型在多项评测上达到甚至超越 GPT-4 水平。本文梳理中文开源模型版图、技术特点、适用场景与选型建议。

## 一、中文开源模型版图

```python
# 中文开源模型主要玩家
chinese_models = {
    "阿里 Qwen": {
        "团队": "阿里通义实验室",
        "代表": "Qwen 3 系列 (2025)",
        "特点": "全谱系（0.5B-72B）、多模态、中文顶配",
    },
    "DeepSeek": {
        "团队": "深度求索（杭州）",
        "代表": "DeepSeek V3 / R1 (2024-2025)",
        "特点": "MoE 架构 + 推理强 + 极致成本",
    },
    "智谱 ChatGLM": {
        "团队": "智谱 AI（北京）",
        "代表": "GLM-4 系列",
        "特点": "中文 + 多模态 + 工具调用",
    },
    "百川 Baichuan": {
        "团队": "百川智能",
        "代表": "Baichuan 4 系列",
        "特点": "中文 + 长上下文",
    },
    "零一万物 Yi": {
        "团队": "零一万物（李开复）",
        "代表": "Yi-Lightning",
        "特点": "高性能 + 开源",
    },
    "上海 AI Lab InternLM": {
        "团队": "上海人工智能实验室",
        "代表": "InternLM 3 系列",
        "特点": "书生·浦语，工具链完整",
    },
    "MiniMax": {
        "团队": "MiniMax",
        "代表": "MiniMax-01 系列",
        "特点": "长上下文 + 多模态",
    },
    "腾讯混元": {
        "团队": "腾讯",
        "代表": "Hunyuan 系列",
        "特点": "工业级、多模态",
    },
    "字节豆包": {
        "团队": "字节跳动",
        "代表": "豆包（部分开源）",
        "特点": "端到端优化",
    },
}
```

## 二、Qwen（阿里通义）

```python
# Qwen 系列详解
qwen_family = {
    "Qwen 2.5": {
        "年份": 2024,
        "规格": "0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B",
        "context": "128K tokens",
        "语言": "29+ 语言（中文最强）",
        "性能": "MMLU 70%+ (7B), 80%+ (72B)",
    },
    "Qwen 3": {
        "年份": 2025,
        "规格": "0.6B / 1.7B / 4B / 8B / 14B / 32B / 235B (MoE)",
        "context": "128K tokens",
        "架构": "Dense + MoE",
        "性能": "对标 Llama 4 + DeepSeek V3",
        "特点": "原生多语言、推理增强",
    },
    "Qwen-VL": {
        "类型": "多模态",
        "规格": "2B / 7B / 72B",
        "能力": "图像理解 + OCR + 中文识别",
    },
    "Qwen-Coder": {
        "类型": "代码",
        "规格": "1.5B / 7B / 32B",
        "性能": "对标 GPT-4o 在代码任务",
    },
    "QwQ": {
        "类型": "推理（Reasoning）",
        "规格": "32B",
        "特点": "强化推理（对标 o1）",
    },
}
```

### Qwen 特色能力

```python
# Qwen 的核心特色
qwen_features = {
    "中文顶配": "中文 NLP 任务 SOTA",
    "全尺寸谱系": "从 0.5B 到 235B 完整覆盖",
    "MoE 创新": "Qwen 3 235B-A22B（22B 激活）",
    "工具调用": "原生 Tool Use + Function Calling",
    "多语言": "29+ 语言，中英最强",
    "代码": "Qwen-Coder 对标 GPT-4o",
    "推理": "QwQ 32B 对标 o1-mini",
    "Agent": "原生 Agent 优化",
}
```

```python
# Qwen 部署示例
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

messages = [{"role": "user", "content": "你好"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)
response = tokenizer.decode(
    generated_ids[0][len(model_inputs.input_ids[0]):],
    skip_special_tokens=True,
)
```

## 三、DeepSeek

```python
# DeepSeek 系列（深度求索）
deepseek_family = {
    "DeepSeek V2": {
        "年份": 2024,
        "规格": "236B 总参，21B 激活",
        "架构": "MoE + MLA (Multi-head Latent Attention)",
        "性能": "GPT-4 级别",
        "训练成本": "$5.6M（极低）",
    },
    "DeepSeek V3": {
        "年份": 2024-12,
        "规格": "671B 总参，37B 激活",
        "架构": "MoE + MLA",
        "context": "64K tokens",
        "性能": "对标 GPT-4o/Claude 3.5",
        "训练成本": "$5.5M（极致低成本）",
    },
    "DeepSeek R1": {
        "年份": 2025-01",
        "类型": "推理（Reasoning）",
        "规格": "671B MoE",
        "性能": "对标 o1",
        "特点": "纯 RL 训练的推理模型",
        "开源": "完整推理链 + 训练数据",
    },
    "DeepSeek R1 Distill": {
        "类型": "蒸馏版",
        "规格": "1.5B / 7B / 8B / 14B / 32B / 70B",
        "特点": "把 R1 能力蒸馏到 Llama/Qwen 小模型",
        "性能": "对标 o1-mini (70B 版)",
    },
}
```

### DeepSeek 技术亮点

```python
# DeepSeek 创新
deepseek_innovations = {
    "MoE 优化": "细粒度专家 + 共享专家",
    "MLA (Multi-head Latent Attention)": "降低 KV Cache 显存",
    "FP8 训练": "首个大规模 FP8 训练",
    "极致低成本": "V3 训练 $5.5M（vs GPT-4 估 $100M+）",
    "R1 纯 RL": "GRPO 算法，无需监督微调",
    "开放权重 + 论文": "完全开源技术报告",
}
```

### DeepSeek 部署

```python
# DeepSeek V3 部署（671B-MoE）
# 需要 8×H100 或更多

# vLLM 部署
# vllm serve deepseek-ai/DeepSeek-V3 \
#   --tensor-parallel-size 8 \
#   --max-model-len 65536 \
#   --gpu-memory-utilization 0.95

# DeepSeek R1 蒸馏版（70B，单卡/双卡）
# vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B

# 调用
from openai import OpenAI
client = OpenAI(
    base_url="https://api.deepseek.com",  # 或本地 vLLM
    api_key="your-key",
)
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "解释 MLA"}],
)
```

## 四、ChatGLM（智谱）

```python
# ChatGLM 系列（智谱 AI）
chatglm_family = {
    "ChatGLM 3 / GLM-4": {
        "年份": 2023-2024,
        "规格": "6B / 9B / 13B / GLM-4-Air (106B)",
        "context": "128K",
        "语言": "中英双语",
        "性能": "中文 SOTA，多任务强",
    },
    "GLM-4.5 / 4.6": {
        "年份": 2025,
        "规格": "9B / 32B / 106B",
        "context": "128K",
        "特点": "原生 Agent + 工具调用",
        "性能": "对标 Claude Sonnet 4.5",
    },
    "CogVLM / CogAgent": {
        "类型": "多模态",
        "特点": "视觉 + Agent（GUI 操作）",
    },
    "CodeGeeX": {
        "类型": "代码",
        "规格": "7B / 13B",
        "语言": "20+ 编程语言",
    },
}
```

```python
# GLM-4 Agent 能力
glm4_agent_features = {
    "Function Calling": "原生工具调用",
    "代码解释器": "内置",
    "Web Browsing": "内置浏览器工具",
    "All Tools": "一次调用多个工具",
    "Long Chain-of-Thought": "长推理链",
}
```

## 五、其他重要中文模型

```python
# Yi（零一万物）
yi_models = {
    "Yi-Lightning": {
        "规格": "未知（闭源）",
        "性能": "中文 SOTA（SuperCLUE 第一）",
    },
    "Yi-1.5": {
        "年份": 2024,
        "规格": "9B / 34B",
        "context": "32K",
        "开源": "完全开源",
    },
}

# 百川 Baichuan
baichuan_models = {
    "Baichuan 4": {
        "规格": "未知",
        "性能": "中文领先",
    },
    "Baichuan 3": {
        "规格": "7B",
        "context": "192K（最早超长上下文）",
    },
}

# InternLM（上海 AI Lab）
internlm_family = {
    "InternLM 3": {
        "规格": "8B / 20B",
        "context": "32K",
        "工具链": "XTuner / LMDeploy / Lagent 完整",
    },
    "InternLM-XComposer": {
        "类型": "多模态",
        "特点": "图像理解 + 长文",
    },
}
```

## 六、性能对比

```python
# 中文 SOTA 模型对比（2025-2026 数据）

benchmark = {
    "MMLU (中文)": {
        "Qwen 3 235B": "88.0",
        "DeepSeek V3": "88.5",
        "GLM-4.5 106B": "85.6",
        "GPT-4o": "88.7",
    },
    "C-Eval": {
        "Qwen 3 235B": "91.2",
        "DeepSeek V3": "90.5",
        "GLM-4.5 106B": "87.8",
    },
    "CMMLU": {
        "Qwen 3 235B": "89.0",
        "DeepSeek V3": "88.0",
        "GLM-4.5 106B": "85.0",
    },
    "GSM8K (中文数学)": {
        "Qwen 3 235B": "94.5",
        "DeepSeek V3": "94.0",
        "GLM-4.5 106B": "92.0",
    },
    "HumanEval (中文版)": {
        "Qwen 3 235B": "85.0",
        "DeepSeek V3": "82.5",
    },
    "HumanEval (英文)": {
        "Qwen 3 235B": "92.0",
        "DeepSeek V3": "92.5",
        "GLM-4.5 106B": "90.0",
        "GPT-4o": "90.2",
    },
    "结论": "中文 SOTA 已经与 GPT-4o 持平或超越",
}
```

## 七、选型决策

```python
# 中文模型选型

selection_guide = {
    "通用主力": {
        "推荐": "Qwen 3 32B / 72B",
        "理由": "中文最强 + 完整生态 + 多尺寸",
    },
    "极致推理": {
        "推荐": "DeepSeek R1 / QwQ-32B",
        "理由": "对标 o1，纯 RL 训练",
    },
    "极致低成本": {
        "推荐": "DeepSeek V3 (API) / DeepSeek R1-Distill 32B",
        "理由": "训练成本最低，蒸馏版性价比高",
    },
    "Agent 应用": {
        "推荐": "GLM-4.5 / Qwen 3",
        "理由": "原生 Function Calling + 工具使用",
    },
    "代码任务": {
        "推荐": "Qwen-Coder 32B / DeepSeek-Coder",
        "理由": "对标 GPT-4o 代码任务",
    },
    "多模态": {
        "推荐": "Qwen-VL / GLM-4V",
        "理由": "中文 OCR + 视觉强",
    },
    "边缘部署": {
        "推荐": "Qwen 3 0.6B / 1.7B / GLM-Edge",
        "理由": "极致小尺寸",
    },
    "数据安全": {
        "推荐": "本地部署 + Qwen / DeepSeek",
        "理由": "完全本地，无数据外传",
    },
}
```

## 八、部署成本对比

```python
# 各模型部署资源需求

deployment_cost = {
    "Qwen 3 0.6B": {
        "VRAM": "1-2GB",
        "适用": "CPU / 移动端",
        "硬件成本": "< $500",
    },
    "Qwen 3 8B": {
        "VRAM": "16GB (4-bit) / 24GB (FP16)",
        "适用": "单卡 RTX 4090/3090",
        "硬件成本": "$1500-3000",
    },
    "Qwen 3 32B": {
        "VRAM": "40GB (4-bit) / 64GB (FP16)",
        "适用": "单卡 A100 80GB",
        "硬件成本": "$5000-10000",
    },
    "Qwen 3 72B": {
        "VRAM": "80GB (4-bit) / 150GB (FP16)",
        "适用": "2-4 卡 H100",
        "硬件成本": "$15000-40000",
    },
    "Qwen 3 235B (MoE)": {
        "VRAM": "激活 22B → 40-80GB",
        "适用": "2-4 卡 A100",
        "硬件成本": "$15000-30000",
    },
    "DeepSeek V3 (671B MoE)": {
        "VRAM": "激活 37B → 80-160GB",
        "适用": "8 卡 H100",
        "硬件成本": "$100000-300000",
    },
}
```

## 九、中文模型的发展趋势

```python
# 中文模型未来方向

trends = {
    "1. MoE 普及": "Qwen 3 235B、DeepSeek V3 都用 MoE",
    "2. 极致成本": "DeepSeek V3 训练 $5.5M，引领低成本",
    "3. 推理增强": "QwQ、R1 引领 o1 级别推理",
    "4. Agent 原生": "GLM-4.5、Qwen-Agent 优化 Agent",
    "5. 长上下文": "1M+ tokens",
    "6. 多模态": "VL、Vision、CogVLM",
    "7. 端云协同": "端侧 1B + 云侧 200B+",
    "8. 全栈开源": "训练代码、训练数据、模型权重",
}
```

## 小结

中文开源模型已经实现"国际一流"水平。**Qwen 是中文通用首选，DeepSeek 适合极致成本与推理，其他模型各有特色**。生态完善度上，Qwen > DeepSeek > GLM > 其他。下一阶段的发展是 **MoE 架构 + 推理增强 + Agent 原生 + 多模态融合**。