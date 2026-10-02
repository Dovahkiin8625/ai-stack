# Llama 系列：从 Llama 1 到 Llama 4 的开源旗舰

Llama 是当下最重要的开源 LLM 系列——从 Meta（Facebook）2023 年发布 Llama 1，到 2024 年 Llama 3，再到 2025 年 Llama 4，开源生态围绕 Llama 形成了完整的工具链、社区、衍生模型。本文梳理 Llama 系列的发展、模型规格、性能对比，以及围绕 Llama 形成的生态。

## 一、Llama 系列发展史

```python
# Llama 时间线
timeline = {
    "2023-02": "Llama 1（论文）— 65B/33B/13B/7B",
    "2023-07": "Llama 2 — 70B/13B/7B（可商用）",
    "2023-11": "Code Llama — 代码专用",
    "2024-04": "Llama 3 — 8B/70B，128K 上下文",
    "2024-07": "Llama 3.1 — 405B! 128K，多语言",
    "2024-09": "Llama 3.2 — 1B/3B 端侧 + 11B/90B 视觉",
    "2024-12": "Llama 3.3 — 70B 升级版（替代 3.1-70B）",
    "2025-04": "Llama 4 — 8B/70B/400B（MoE 架构）",
    "2025-10": "Llama 4.1 / 4.2 — 持续迭代",
}
```

## 二、Llama 3.x 系列详解

```python
# Llama 3 系列规格对比
llama_3_family = {
    "Llama 3 8B": {
        "params": "8B",
        "context": "8K",
        "use": "轻量级、边缘部署",
        "languages": "英文为主",
    },
    "Llama 3.1 8B": {
        "params": "8B",
        "context": "128K",
        "languages": "8 种",
        "use": "通用主力",
    },
    "Llama 3.1 70B": {
        "params": "70B",
        "context": "128K",
        "languages": "8 种",
        "use": "高难度任务",
    },
    "Llama 3.1 405B": {
        "params": "405B",
        "context": "128K",
        "languages": "8 种",
        "use": "旗舰（vs GPT-4）",
    },
    "Llama 3.2 1B/3B": {
        "params": "1B / 3B",
        "context": "128K",
        "use": "端侧/移动/嵌入式",
        "specialty": "超小尺寸",
    },
    "Llama 3.2 11B/90B Vision": {
        "params": "11B / 90B",
        "context": "128K",
        "use": "多模态（图+文）",
        "specialty": "视觉理解",
    },
    "Llama 3.3 70B": {
        "params": "70B",
        "context": "128K",
        "use": "替代 3.1-70B，性能更优",
        "specialty": "推理增强",
    },
}
```

## 三、Llama 4：MoE 架构革命

```python
# Llama 4（2025 年发布）
# 首次采用 MoE（Mixture of Experts）架构

llama_4_features = {
    "架构创新": "MoE（混合专家）",
    "激活参数": "每次推理激活部分专家",
    "总参数 vs 激活": {
        "Llama 4 8B": "8B 总参，激活 ~3B",
        "Llama 4 70B": "70B 总参，激活 ~17B",
        "Llama 4 400B": "400B 总参，激活 ~120B",
    },
    "优势": [
        "更大总参数（更强能力）",
        "更少激活参数（更快推理）",
        "训练成本与传统 dense 模型接近",
    ],
    "挑战": [
        "推理调度复杂",
        "显存需求高",
    ],
}

# Llama 4 性能
llama_4_benchmarks = {
    "Llama 4 70B": {
        "MMLU": "86.5%",
        "HumanEval": "88.0%",
        "GSM8K": "94.2%",
        "vs_GPT-4o": "接近或超越",
    },
    "Llama 4 400B": {
        "MMLU": "88.7%",
        "HumanEval": "92.5%",
        "vs_GPT-4.5": "接近",
        "vs_Claude_4.5_Opus": "接近",
    },
}
```

## 四、模型架构演进

```python
# Llama 架构演进

architectures = {
    "Llama 1": {
        "norm": "RMSNorm",
        "activation": "SwiGLU",
        "attention": "MHA (Multi-Head Attention)",
        "rope": "RoPE",
        "ffn": "Standard FFN",
    },
    "Llama 2": {
        "norm": "RMSNorm",
        "activation": "SwiGLU",
        "attention": "MHA + GQA (Grouped-Query)",
        "rope": "RoPE",
        "ffn": "Standard FFN",
    },
    "Llama 3": {
        "norm": "RMSNorm",
        "activation": "SwiGLU",
        "attention": "GQA",
        "rope": "RoPE（扩展到 128K）",
        "ffn": "Standard FFN",
        "tokenizer": "128K BPE vocab",
    },
    "Llama 4": {
        "norm": "RMSNorm",
        "activation": "SwiGLU",
        "attention": "GQA + MoE",
        "rope": "RoPE（更长上下文）",
        "ffn": "MoE Experts",
        "tokenizer": "200K BPE vocab（更大）",
    },
}
```

## 五、训练数据与流程

```python
# Llama 3 训练细节（Meta 2024 论文公开）

training_details = {
    "data_size": "15T tokens（Llama 3）",
    "data_sources": [
        "CommonCrawl（多语言清洗）",
        "GitHub",
        "Wikipedia",
        "Books (多领域)",
        "ArXiv",
        "StackExchange",
        "问答网站",
    ],
    "训练算力": "Llama 3.1-405B：30M H100 GPU-hours",
    "训练时长": "几个月",
    "数据清洗": [
        "去重",
        "质量过滤（fastText + LLM）",
        "PII 移除",
        "多语言平衡",
    ],
    "RLHF 流程": "拒绝采样 + SFT + DPO/PPO",
}
```

## 六、Llama 生态

```python
# Llama 形成的生态

ecosystem = {
    "推理框架": {
        "vLLM": "高支持 Llama 系列",
        "TGI": "首类公民支持",
        "llama.cpp": "GGUF 量化 + CPU/Mac 推理",
        "Ollama": "本地运行（一行命令）",
        "LM Studio": "桌面 GUI",
    },
    "微调工具": {
        "HuggingFace TRL (SFTTrainer, DPOTrainer)",
        "Axolotl",
        "LLaMA-Factory",
        "Unsloth（2-5x 加速）",
        "PEFT (LoRA/QLoRA)",
    },
    "衍生模型": {
        "中文": "Llama-Chinese, Chinese-LLaMA, Llama3-Chinese",
        "代码": "CodeLlama, WizardCoder",
        "对话": "Llama-2-Chat, Vicuna",
        "推理": "OpenLlama, NousResearch",
        "量化": "TheBloke (GGUF/GPTQ)",
        "多模态": "LLaVA, Llama-3.2-Vision",
    },
    "数据集": {
        "Open-Orca（Llama 3 训练风格）",
        "Hermes（NousResearch）",
        "Open-Hermes-2.5",
        "Tulu-3-SFT (Allen AI)",
    },
}
```

## 七、Llama 部署

```python
# Llama 本地部署示例

# 方法 1：Ollama（一键启动）
# ollama run llama3.1:8b

# 方法 2：llama.cpp（量化 + CPU）
# ./main -m llama-3.1-8b-instruct.Q4_K_M.gguf \
#        -n 512 --ctx-size 8192 \
#        --interactive-first

# 方法 3：vLLM（GPU 高吞吐）
# vllm serve meta-llama/Llama-3.1-8B-Instruct \
#   --port 8000 --gpu-memory-utilization 0.9

# 方法 4：HuggingFace Transformers（最基础）
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

## 八、Llama 与商用 API 对比

```python
# Llama 开源 vs 闭源 API

comparison = {
    "数据隐私": {
        "Llama": "✅ 完全本地",
        "API": "❌ 数据传给 OpenAI/Anthropic",
    },
    "成本": {
        "Llama": "一次性部署 + GPU 成本",
        "API": "按 token 计费",
    },
    "最新能力": {
        "Llama": "❌ 滞后 6-12 个月",
        "API": "✅ 永远最新",
    },
    "可定制": {
        "Llama": "✅ 微调、RAG、Agent 全可控",
        "API": "有限（仅 prompt + 微调）",
    },
    "运维成本": {
        "Llama": "高（GPU 运维）",
        "API": "零（云端）",
    },
    "延迟": {
        "Llama": "本地，~10-50ms",
        "API": "网络，~200-1000ms",
    },
}
```

## 九、Llama 微调实践

```python
# LoRA 微调示例
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
    ),
    peft_config=lora_config,
)
```

## 十一、Llama 选择建议

```python
# Llama 模型选择决策

selection_guide = {
    "8B (Llama 3.1)": {
        "适用": [
            "本地个人电脑部署",
            "中低复杂度任务",
            "预算敏感",
        ],
        "VRAM": "16-24GB（4-bit）",
    },
    "70B (Llama 3.3)": {
        "适用": [
            "高性能单卡/双卡服务器",
            "复杂业务",
            "性价比选择",
        ],
        "VRAM": "40-80GB（4-bit）",
    },
    "405B (Llama 3.1)": {
        "适用": [
            "数据中心部署",
            "最高质量",
            "需要多个 GPU",
        ],
        "VRAM": "需要 8×H100 或更多",
    },
    "1B/3B (Llama 3.2)": {
        "适用": [
            "移动端/嵌入式",
            "极端低延迟",
            "边缘计算",
        ],
        "VRAM": "1-4GB",
    },
    "Vision 11B/90B (Llama 3.2)": {
        "适用": [
            "图像理解",
            "多模态应用",
        ],
        "VRAM": "24-90GB",
    },
    "Llama 4 (MoE)": {
        "适用": [
            "最强性能需求",
            "愿意处理 MoE 复杂性",
        ],
        "VRAM": "激活 17B，约 70GB",
    },
}
```

## 十二、社区与未来

```python
# Llama 社区与生态影响力

impact = {
    "GitHub Stars": "100K+（huggingface/transformers 衍生项目）",
    "HuggingFace 下载量": "Llama 系列累计 2 亿+ 次",
    "衍生模型": "10 万+ 衍生模型在 Hub",
    "学术引用": "Llama 论文 1 万+ 引用",
    "商业用户": "Meta、Microsoft、Shopify 等",
    "影响": [
            "证明开源 LLM 可与商用前沿抗衡",
            "带动 Mistral、Qwen、DeepSeek 等开源发展",
            "促成模型许可证生态（商业 vs 研究）",
        ],
}

# 未来方向
future_directions = {
    "1. MoE 普及": "Llama 4 引领 MoE 架构",
    "2. 多模态原生": "不依赖 LLaVA 改装",
    "3. 更长上下文": "1M+ tokens",
    "4. Agent 优化": "原生工具调用能力",
    "5. 端云协同": "1B/3B 端侧 + 大模型云侧",
    "6. 硬件协同": "与 Apple、NVIDIA 深度优化",
    "7. 合规与可商用": "解决部分商用限制",
}
```

## 小结

Llama 系列是开源 LLM 的"标杆"——**从 Llama 1 到 Llama 4 完成了 dense → MoE 的架构进化**。**Llama 3.1 8B/64B 是当下主力，405B 是开源旗舰，Llama 3.2 适合端侧，Llama 4 是 MoE 新一代**。生态完善（推理、微调、衍生、社区）使其成为开源应用的事实标准。下一篇我们讨论 **Mistral、Qwen、DeepSeek 等其他开源模型**。