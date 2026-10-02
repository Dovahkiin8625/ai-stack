# 开源模型选型：如何从 Llama/Qwen/DeepSeek/Mistral 中选择

面对上百个开源 LLM，如何找到最适合自己的那个？本文给出一个**任务驱动 + 性能评测 + 部署成本 + 生态兼容**四维度的选型框架，并讨论 2025-2026 年的主流开源模型对比。

## 一、选型框架

```python
# 开源模型选型的四维度框架

selection_framework = {
    "1. 任务适配": "任务类型（对话/分类/代码/推理）与模型擅长是否对齐",
    "2. 性能评测": "在关键 benchmark 上表现",
    "3. 部署成本": "显存、算力、运维成本",
    "4. 生态兼容": "框架支持、社区活跃、文档质量",
}
```

## 二、按任务类型选模型

```python
# 任务-模型映射
task_model_map = {
    "中文对话（通用）": {
        "首选": "Qwen 3 32B / 72B",
        "备选": "GLM-4.5 32B / DeepSeek V3",
    },
    "中文推理": {
        "首选": "DeepSeek R1 / QwQ-32B",
        "备选": "Qwen 3 32B",
    },
    "英文对话": {
        "首选": "Llama 3.3 70B / Llama 4 70B",
        "备选": "Qwen 3 72B / Mistral Large",
    },
    "英文推理": {
        "首选": "DeepSeek R1",
        "备选": "QwQ-32B / OpenAI o1 (API)",
    },
    "代码生成": {
        "首选": "Qwen-Coder 32B / DeepSeek-Coder-V2",
        "备选": "CodeLlama 70B / Qwen 3 72B",
    },
    "数学解题": {
        "首选": "DeepSeek R1 / QwQ-32B",
        "备选": "Qwen 3 32B",
    },
    "多模态（图+文）": {
        "首选": "Qwen-VL 72B / InternVL 2.5",
        "备选": "Llama 3.2 Vision 90B / GLM-4V",
    },
    "多模态（视频）": {
        "首选": "Qwen 2.5-VL / MiniMax-VL",
        "备选": "Video-LLaVA / LongVU",
    },
    "长上下文（>128K）": {
        "首选": "Qwen 3 235B / GLM-4.5",
        "备选": "Llama 3.1 405B / DeepSeek V3",
    },
    "本地隐私部署": {
        "首选": "Qwen 3 / Llama 3.3",
        "备选": "Mistral / Yi",
    },
    "端侧部署（<2GB）": {
        "首选": "Qwen 3 0.6B / Llama 3.2 1B",
        "备选": "Gemma 3 1B / Phi-3",
    },
}
```

## 三、按部署资源选模型

```python
# 部署资源-模型映射

resource_model_map = {
    "CPU only / 边缘设备": {
        "VRAM": "N/A",
        "模型": "Qwen 3 0.6B, Llama 3.2 1B, Phi-3 mini, Gemma 3 1B",
        "量化": "GGUF Q4_K_M",
        "工具": "llama.cpp, Ollama",
    },
    "单卡 RTX 4090/3090 (24GB)": {
        "VRAM": "24GB",
        "模型": "Qwen 3 14B, Llama 3.1 8B, Mistral 7B",
        "量化": "AWQ INT4 / FP16 (8B)",
        "工具": "vLLM, TGI",
    },
    "单卡 A100 80GB": {
        "VRAM": "80GB",
        "模型": "Qwen 3 32B, Llama 3.3 70B, DeepSeek R1-Distill 70B",
        "量化": "AWQ INT4",
        "工具": "vLLM, SGLang",
    },
    "多卡 A100/H100": {
        "VRAM": "160GB+",
        "模型": "Qwen 3 72B, DeepSeek V3, Llama 3.1 405B",
        "量化": "FP16 / AWQ",
        "工具": "vLLM + TP/PP",
    },
    "集群 8×H100+": {
        "VRAM": "640GB+",
        "模型": "Qwen 3 235B, DeepSeek V3 (全量)",
        "量化": "FP16",
        "工具": "TensorRT-LLM, SGLang",
    },
}
```

## 四、性能评测对比

```python
# 主流开源模型性能对比（2025-2026 数据，注：实际以官方为准）

benchmark_comparison = {
    "MMLU (知识)": {
        "Llama 3.1 405B": "88.6",
        "Qwen 3 235B": "88.0",
        "DeepSeek V3": "88.5",
        "GLM-4.5 106B": "85.6",
        "GPT-4o": "88.7",
        "Claude Sonnet 4.5": "89.0",
    },
    "HumanEval (代码)": {
        "Llama 3.1 405B": "89.0",
        "Qwen 3 235B": "92.0",
        "DeepSeek V3": "92.5",
        "Qwen-Coder 32B": "92.0",
        "GPT-4o": "90.2",
    },
    "GSM8K (数学)": {
        "Llama 3.1 405B": "96.0",
        "Qwen 3 235B": "94.5",
        "DeepSeek V3": "94.0",
        "DeepSeek R1": "97.0",
        "GPT-4o": "92.0",
    },
    "MATH (竞赛数学)": {
        "DeepSeek R1": "95.0",
        "QwQ-32B": "90.0",
        "OpenAI o1": "96.0",
    },
    "MT-Bench (对话)": {
        "Llama 3.1 405B": "8.96",
        "Qwen 3 235B": "9.0",
        "DeepSeek V3": "9.1",
        "GPT-4o": "9.2",
    },
    "Chatbot Arena (Elo)": {
        "Llama 3.1 405B": 1268,
        "Qwen 3 235B": 1285,
        "DeepSeek V3": 1310,
        "Claude Sonnet 4.5": 1290,
        "GPT-4o": 1285,
    },
    "中文 C-Eval": {
        "Qwen 3 235B": "91.2",
        "DeepSeek V3": "90.5",
        "GLM-4.5 106B": "87.8",
    },
    "结论": "开源已经追平或接近闭源前沿",
}
```

## 六、模型许可证对比

```python
# 主流模型许可证

model_licenses = {
    "Llama 3.x / 4.x": {
        "许可证": "Llama 3 Community License",
        "商用": "✅ 月活 < 7 亿用户可商用",
        "限制": "大型科技公司有限制",
        "特殊条款": "禁止用于改进其他 LLM",
    },
    "Qwen 2.5 / 3": {
        "许可证": "Apache 2.0（多数）/ Qwen License",
        "商用": "✅ 完全可商用",
        "限制": "Qwen License 对大公司有限制",
    },
    "DeepSeek V3 / R1": {
        "许可证": "MIT（DeepSeek V3）/ DeepSeek License",
        "商用": "✅ 完全可商用",
        "限制": "几乎无",
    },
    "Mistral (Apache 2)": {
        "许可证": "Apache 2.0",
        "商用": "✅ 完全可商用",
    },
    "GLM-4 (智谱)": {
        "许可证": "GLM License",
        "商用": "✅",
        "限制": "大型企业单独洽谈",
    },
    "Gemma (Google)": {
        "许可证": "Gemma License",
        "商用": "✅",
        "限制": "禁止用于训练其他 LLM",
    },
    "Phi (Microsoft)": {
        "许可证": "MIT",
        "商用": "✅",
    },
}
```

## 七、模型生态成熟度

```python
# 生态成熟度评估

ecosystem_maturity = {
    "Llama": {
        "推理": "★★★★★ (vllm, TGI, llama.cpp, ollama 全支持)",
        "微调": "★★★★★ (TRL, Axolotl, Unsloth, LLaMA-Factory)",
        "衍生": "★★★★★ (10 万+ 衍生模型)",
        "文档": "★★★★★",
        "社区": "★★★★★",
    },
    "Qwen": {
        "推理": "★★★★★",
        "微调": "★★★★★ (含中文微调工具)",
        "衍生": "★★★★ (3 万+ 衍生)",
        "文档": "★★★★ (中文文档最全)",
        "社区": "★★★★★ (中文社区活跃)",
    },
    "DeepSeek": {
        "推理": "★★★★★",
        "微调": "★★★★ (增长中)",
        "衍生": "★★★ (蒸馏版流行)",
        "文档": "★★★★ (技术报告详尽)",
        "社区": "★★★★ (增长快)",
    },
    "GLM": {
        "推理": "★★★★★ (LMDeploy 优化)",
        "微调": "★★★★",
        "衍生": "★★★",
        "文档": "★★★★",
        "社区": "★★★ (中文活跃)",
    },
}
```

## 八、选型决策树

```python
# 开源模型选型决策树

decision_tree = {
    "Q1: 你的应用以中文为主还是英文为主?": {
        "中文为主": {
            "Q2: 需要推理/数学能力?": {
                "是": "DeepSeek R1 / QwQ-32B",
                "否": {
                    "Q3: 需要多模态?": {
                        "是": "Qwen-VL / GLM-4V",
                        "否": {
                            "Q4: 显存预算?": {
                                "16GB": "Qwen 3 8B / GLM-4.5 9B",
                                "40GB": "Qwen 3 32B / GLM-4.5 32B",
                                "80GB+": "Qwen 3 72B / GLM-4.5 106B",
                            },
                        },
                    },
                },
            },
        },
        "英文为主": {
            "Q2: 需要 SOTA 推理?": {
                "是": "DeepSeek R1 / OpenAI o1 API",
                "否": {
                    "Q3: 显存预算?": {
                        "16GB": "Llama 3.1 8B / Mistral 7B",
                        "40GB": "Llama 3.3 70B / Qwen 3 32B",
                        "80GB+": "Llama 3.1 405B / Qwen 3 72B",
                    },
                },
            },
        },
    },
}
```

## 九、实测建议

```python
# 在真实业务中如何测试和评估

evaluation_plan = {
    "1. 准备数据集": [
        "内部业务数据（100-1000 真实样本）",
        "领域 benchmark",
        "对抗样本（边界情况）",
    ],
    "2. 设定指标": [
        "主任务准确率 / pass@1",
        "延迟（TTFT, TPOT）",
        "吞吐量（tokens/s）",
        "成本（每千次调用）",
        "失败率 / 鲁棒性",
    ],
    "3. 多模型对比": [
        "选 3-5 个候选模型",
        "统一 prompt 模板",
        "同一评测脚本",
        "统计显著性检验",
    ],
    "4. 灰度上线": [
        "10% 流量给候选模型",
        "对比业务指标",
        "用户反馈收集",
        "逐步放大",
    ],
}
```

## 十、未来趋势

```python
# 开源模型的发展趋势

future_trends = {
    "1. MoE 普及": "Qwen 3 235B、DeepSeek V3、Llama 4 都用 MoE",
    "2. 推理增强": "R1、QwQ 引领，纯 RL 训练",
    "3. Agent 原生": "GLM-4.5、Qwen-Agent 优化 Agent 工具调用",
    "4. 长上下文": "Qwen 3 1M+ tokens",
    "5. 多模态融合": "原生 Vision + Audio + Text（Llama 4、Gemini）",
    "6. 极致低成本": "DeepSeek V3 训练 $5.5M，引领成本革命",
    "7. 端侧小模型": "Qwen 3 0.6B、Llama 3.2 1B",
    "8. 全栈开源": "权重 + 训练代码 + 训练数据 + 技术报告",
    "9. AI 安全": "完善 RLHF / Constitutional AI",
    "10. 合规与可商用": "更宽松的许可证",
}
```

## 小结

开源模型选型没有"银弹"——**Qwen 是中文通用首选，DeepSeek 适合推理与低成本，Llama 适合英文与端侧，GLM 适合 Agent**。**最终选哪个，要在真实业务数据上测试，而不是 benchmark**。未来开源会继续追赶甚至超越闭源 API，关键是关注 **MoE 架构 + 推理增强 + Agent 原生 + 极致低成本** 四大方向。