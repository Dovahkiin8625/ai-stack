# Fine-tuning 评估与避坑指南

微调结束不是终点。**"loss 降了"≠"能力变强"**：模型可能学会了某种被偏好的措辞、可能背下了训练样本、可能把通用能力搞丢。本文介绍 fine-tuning 评估的核心维度、通用 benchmark、领域评测与回归测试，并给出一个能跑的评测 pipeline。

## 一、为什么要做评测

微调后的常见"幻觉"：

- **灾难性遗忘（catastrophic forgetting）**：模型在领域任务上变强，但通用能力（MMLU、推理、代码）骤降。
- **格式过拟合**：模型只会用训练时的措辞回答，遇到新问法就崩。
- **过训（overtraining）**：val loss 已经上升，train loss 还在降，但没人看 val。
- **对话模板漂移**：训练用 ChatML，推理用 Llama-3 prompt，输出格式全乱。

没有评测，这些问题只能等上线后被用户发现。

## 二、评估集的两类来源

### 1. 通用 benchmark（开箱即用）

| 评测 | 维度 | 用途 |
| --- | --- | --- |
| **MMLU** | 57 个学科的多选题 | 通用知识广度 |
| **GSM8K / MATH** | 数学推理 | 推理能力 |
| **HumanEval / MBPP** | 代码补全 | 编程能力 |
| **IFEval** | 指令遵循（格式约束） | 指令对齐 |
| **MT-Bench / AlpacaEval** | 多轮对话质量 | 对话能力 |
| **TruthfulQA** | 真实性 | 防幻觉 |
| **BBH** | 23 个推理任务 | 综合推理 |

每个 benchmark 通常以 few-shot 或 zero-shot 方式跑，归一化后比较分数。**微调后这些分数不应大幅下降**，否则就是灾难性遗忘。

### 2. 领域评测集（自家造）

通用 benchmark 不能替代业务评测。建议：

- 收集 **50–200 条真实业务问题**，人工给出标准答案。
- 用 LLM-as-judge 或规则评分，跑回归。
- 每次微调后都跑一遍这张表，**分数不能掉**。

## 三、评测的三大维度

参考 RAGAS 的思路，可以把 fine-tuning 评估拆成独立维度：

1. **任务能力**：模型在目标任务上的正确率 / 通过率。
2. **指令遵循**：能否按指定格式（如 JSON、特定模板）输出。
3. **通用能力**：MMLU / GSM8K 等分数不大幅下降。

每个维度独立打报告，能更精准定位问题。

## 四、LLM-as-judge：自动评分

人工标注贵且慢。**LLM-as-judge** 用一个更强的模型（如 GPT-4o、Claude Sonnet）对生成结果打分：

```python
import json
from openai import OpenAI

client = OpenAI()

JUDGE_PROMPT = """你是严格的模型评测员，请对模型的回答按 1-5 打分，并给出理由。

评分维度：
- 准确性 (1-5)：事实是否正确
- 完整性 (1-5): 是否答到点子上
- 格式 (1-5)：是否严格遵循指令格式

问题：{question}
参考答案：{reference}
模型回答：{response}

只输出 JSON：{{"accuracy": <int>, "completeness": <int>, "format": <int>, "reason": "<string>"}}
"""

def judge(question, reference, response, model="gpt-4o-mini"):
    msg = JUDGE_PROMPT.format(question=question, reference=reference, response=response)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": msg}],
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def eval_pipeline(eval_set, model_fn):
    """对一个评测集跑模型并打分"""
    results = []
    for ex in eval_set:
        pred = model_fn(ex["question"])
        score = judge(ex["question"], ex["reference"], pred)
        results.append({"question": ex["question"], "pred": pred, "score": score})
    # 汇总
    avg = {
        dim: sum(r["score"][dim] for r in results) / len(results)
        for dim in ["accuracy", "completeness", "format"]
    }
    return {"average": avg, "details": results}


# 示例：调用本地微调后的模型
def my_model(prompt):
    from transformers import pipeline
    pipe = pipeline("text-generation", model="ckpts/qwen7b-merged")
    out = pipe(prompt, max_new_tokens=512, do_sample=False)
    return out[0]["generated_text"]

report = eval_pipeline(eval_set, my_model)
print(report["average"])
```

## 五、回归测试与灾难性遗忘检测

**回归测试** 是 fine-tuning 评估的"安全网"。标准做法：

1. **每次微调前**：跑一遍通用 benchmark + 领域评测，记录 baseline。
2. **每次微调后**：跑同样测试，对比分数变化。
3. **任何维度掉点 > 阈值**（例如 MMLU -2，领域 -5）：拒绝合并。

```python
def regression_check(new_scores, baseline, thresholds):
    """新分数 vs 基线，超过阈值则报警"""
    alerts = []
    for metric, delta_th in thresholds.items():
        delta = new_scores[metric] - baseline[metric]
        if delta < -delta_th:
            alerts.append(f"{metric}: {baseline[metric]:.3f} -> {new_scores[metric]:.3f} (Δ={delta:+.3f})")
    return alerts

thresholds = {
    "MMLU":       0.02,   # 通用知识允许掉 2%
    "GSM8K":      0.03,   # 数学允许掉 3%
    "IFEval":     0.03,
    "domain_acc": 0.05,   # 领域任务允许掉 5%
}
alerts = regression_check(new_scores, baseline, thresholds)
if alerts:
    print("!!! Regression detected:")
    for a in alerts: print("  -", a)
```

## 六、最常见的"坑"清单

| 坑 | 表现 | 解决 |
| --- | --- | --- |
| **学习率过大** | loss 抖 / 训练前期 val loss 飙升 | SFT 用 `1e-5 ~ 5e-5`，LoRA 用 `1e-4 ~ 3e-4` |
| **epoch 过多** | 训练后期 val loss 升 / 风格僵化 | 加 early stop，3 epoch 内观察 |
| **对话模板不一致** | 训练用 ChatML，推理换 Llama-3 prompt | 强制用同一套 chat template |
| **数据被评测集污染** | 评测分数虚高 | 用 n-gram / embedding 做去污染 |
| **保存缺 tokenizer / generation_config** | 换环境后特殊 token 失效 | `save_pretrained` 同时存 tokenizer |
| **过拟合到格式而非内容** | 输出格式漂亮但内容空洞 | 在评测中加"内容维度"，区分格式分和能力分 |
| **灾难性遗忘** | 领域任务涨、通用任务大跌 | 混入通用数据（如 OpenHermes 5%） |

## 小结

fine-tuning 评估是一个"**任务维度 + 通用维度 + LLM-as-judge + 回归测试**"四位一体的工作。微调之后**只盯 train loss 是危险信号**——必须同时监控 val loss、领域评测、通用 benchmark，并设置自动化的回归报警。这样才能让微调真正"提能力"而不是"换能力"。整个 fine-tuning 系列到这里就告一段落：从决策到数据、对齐到评估，构成了一条完整的工程闭环。
