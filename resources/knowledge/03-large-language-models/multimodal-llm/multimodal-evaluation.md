# 多模态评测：MMBench、MMMU、MathVista

多模态大模型的"高分"在 2023-2024 年被反复打假——VLM 看见名字答对题、OCR 抄错数字蒙对答案、benchmark 数据泄漏比比皆是。本文系统梳理多模态评测的几个核心维度、代表性 benchmark、评测流程与常见陷阱，并给出用 VLMEvalKit 跑 MMBench 的最小命令。

## 一、为什么多模态评测特别难

多模态评测比纯文本 LLM 评测多两层困难：

1. **视觉信息多义性**：一张图可以回答多个不同问题，模型答对一个不能算"真正懂"。
2. **视觉捷径（visual shortcut）**：模型可能忽略图像，只看文本选项就能猜对。比如"图中有猫"这种题，猜"是"有 50% 正确率。
3. **数据污染**：训练集和评测集来自同一批网络爬取的图片，模型可能"见过"答案。
4. **评分主观性**：开放回答（图描述、VQA）没有唯一正确答案，需要 GPT-4 或人工打分。

这要求 benchmark 设计必须**多维度、可复现、可校验**。

## 二、评测的五个维度

### 1. 感知（Perception）

考察模型"看见"图像细节的能力：物体识别、OCR、图表读取、空间关系。

- **MMBench / MMBench-V11**：英文单选/多选，覆盖 20+ 视觉能力维度。
- **MMStar**：更"纯视觉"的精选子集，去掉可由文本捷径答对的题目。

### 2. 推理（Reasoning）

考察模型在视觉之上做多步推理：数学、逻辑、常识。

- **MMMU**（Massive Multi-discipline Multimodal Understanding）：11.5K 道大学级别问题，覆盖艺术、商业、科学、医学等 30 个学科，含图表、电路图、化学结构式等多种图像。
- **MathVista**：数学视觉推理，含几何题、函数图像、统计图。
- **MathVision**：更具挑战性的数学题。

### 3. OCR / 文档理解

考察模型"读字"能力：扫描文档、表格、发票、屏幕截图。

- **DocVQA**：文档问答。
- **ChartQA**：图表问答（柱状图、折线图、饼图）。
- **TextVQA**：自然场景中的文字识别与问答。
- **OCRBench**：纯 OCR 能力评测，覆盖 29 个子任务。

### 4. 幻觉（Hallucination）

考察模型是否会"看见不存在的东西"——这是 VLM 的高发问题。

- **POPE**（Polling-based Object Probing）：用"图中是否有 X"二选题，统计 hallucination 率。
- **HallusionBench**：设计成对的"看似存在 vs 真实存在"图像，检测幻觉与错觉。
- **AMBER**（Aerial Multi-modal Benchmark）：同时测 hallucination 类别与存在性。

### 5. 安全 / 偏见

- **MM-SafetyBench**：越狱攻击、隐私、有害内容检测。
- **VLM-Bias**：考察职业、性别、种族等社会偏见。

## 三、代表性 Benchmark 速览

| Benchmark | 样本量 | 题型 | 主要考察 | 难度 |
| --- | --- | --- | --- | --- |
| **MMBench-V11** | ~1200 | 单选 | 感知 + OCR + 推理 | 中 |
| **MMMU** | 11.5K | 单选+开放 | 跨学科推理 | 高 |
| **MathVista** | 6.1K | 多选+开放 | 数学视觉 | 高 |
| **ChartQA** | 18.3K | 开放 | 图表问答 | 中 |
| **DocVQA** | 39.5K | 开放 | 文档理解 | 中 |
| **POPE** | 9K | 二选 | 物体幻觉 | 低-中 |
| **HallusionBench** | 1.1K | 是/否 | 幻觉与错觉 | 中 |
| **OCRBench** | 1K | 开放 | OCR | 中 |
| **MMStar** | 1.5K | 单选 | 纯视觉感知 | 中 |

## 四、自动评测方法

### 4.1 客观题：精确匹配

对单选、多选、二选题，主流做法是：

1. 让模型输出**完整答案文本**（不是只输出 A/B/C/D），例如 "The answer is A."
2. 用正则 `r'\b([A-Z])\b'` 提取选项。
3. 与 ground-truth 严格匹配。

```python
import re

def extract_choice(model_output: str, n_choices=4) -> str:
    """从模型输出中提取 A/B/C/D 选项。"""
    # 优先匹配 "answer is X"
    m = re.search(r'(?:answer|选项)\s*(?:is|为|:)?\s*\(?([A-Z])\)?', model_output, re.I)
    if m:
        return m.group(1).upper()
    # 退化匹配第一个出现的独立大写字母
    m = re.search(r'\b([A-Z])\b', model_output)
    return m.group(1).upper() if m else ""
```

### 4.2 开放题：LLM-as-Judge

对图表问答、文档问答等开放题，传统 BLEU/ROUGE 完全失灵（正确答案可能用不同表述）。主流做法是用一个更强的 LLM（如 GPT-4o、Claude）担任 judge，按 0-10 打分。

```python
JUDGE_PROMPT = """你是一名严格的视觉问答评分员。

问题：{question}
参考答案：{reference}
模型答案：{prediction}

请按以下标准打分：
- 5 分：完全正确，事实一致。
- 3 分：部分正确，遗漏关键信息。
- 1 分：错误或幻觉。

只输出一个数字 1-5，不要解释。
"""

def judge(question, reference, prediction, client):
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, prediction=prediction)
    resp = client.chat.completions.create(
        model="gpt-4o",            # 评委必须比选手更强
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return int(resp.choices[0].message.content.strip())
```

> **注意**：judge 模型必须能"看到"图像（GPT-4o、Claude 3.5V），否则只评文本可能误判。

### 4.3 人工评测

LLM-as-judge 仍然存在偏见（如偏爱长答案、偏爱某种风格）。在关键决策（论文 SOTA、产品上线前）必须辅以人工评测：

- 至少 **3 名标注员**取众数。
- 设计明确的**评分 rubric**（每个分数对应的具体行为）。
- 计算 **Krippendorff's α** 评估标注一致性，> 0.6 算可信。

## 五、训练数据去污染（Decontamination）

数据污染是 MLLM 评测最大的隐患。OpenAI、Anthropic 等都报告过 MMMU 等 benchmark 在 GPT 训练集中**直接出现过**。

**常用去污染手段**：

1. **n-gram 重叠检测**：训练样本与评测样本的 13-gram 重叠率 > 0.5 直接剔除。
2. **图片哈希**：对图像计算 perceptual hash（pHash），去除与评测集图片近似的样本。
3. **题目级去重**：训练集中的问题文本不能与评测集题目高度相似。
4. **污染后处理**：训练结束后用 1-2 epoch 在评测集上验证，若某个子集准确率异常高（例如 95%+），就要怀疑泄漏。

VLMEvalKit 在评测时会输出 `--reuse` 开关来跳过已评测的子集，**应主动清空缓存重跑**避免数据被复用。

## 六、用 VLMEvalKit 跑 MMBench

VLMEvalKit（OpenGVLab 维护）是当下最方便的多模态 benchmark 评测框架，支持 80+ 模型和 20+ benchmark。

### 6.1 安装

```bash
git clone https://github.com/open-compass/VLMEvalKit
cd VLMEvalKit
pip install -e .
```

### 6.2 配置模型与 API Key

```bash
# 方式 A：本地模型路径
export LMUDATA="./lmu_data"
# 方式 B：API 模型
export OPENAI_API_KEY="sk-..."
```

### 6.3 跑 MMBench 子集

```bash
python run.py \
  --data MMBench_DEV_EN \
  --model InternVL2-8B \
  --reuse False
```

参数说明：

- `--data`：benchmark 名称。常用值：`MMBench_DEV_EN`、`MMBench_TEST_EN_V11`、`MMMU_DEV_VAL`、`MathVista_MINI`。
- `--model`：VLMEvalKit 支持的模型名（见 `vlmeval/config.py`）。
- `--reuse`：是否复用上一次的推理结果，生产化评测一定要 `False`。
- `--nproc`：并行进程数（默认 4，按显存调）。

输出示例：

```
[MMBench_DEV_EN] Overall Accuracy: 0.742
  - perception: 0.781
  - reasoning: 0.692
  - OCR: 0.654
```

每条样本的预测、回答时间、判分都保存在 `outputs/MMBench_DEV_INTERNVL2-8B/` 下，方便出错排查。

### 6.4 跑多个 benchmark

```bash
for bench in MMBench_DEV_EN MathVista_MINI MMMU_DEV_VAL; do
  python run.py --data $bench --model InternVL2-8B --reuse False
done
python scripts/collect_results.py   # 汇总成 table
```

## 七、常见评测坑

1. **Prompt 不一致**：不同 benchmark 有自己的 prompt 模板，不修改直接套用会让模型掉 5-10 分。
2. **图像分辨率不匹配**：把 1080p 强压到 224 会损失关键细节，必须按 ViT 训练分辨率（通常 336/448）传入。
3. **judge 模型偏置**：GPT-4o judge 偏爱自家模型风格，导致跨模型对比失真。
4. **少样本泄漏**：用 in-context 示例评测时，示例可能来自评测集本身。
5. **温度不为 0**：评测必须 `temperature=0`，否则同一问题答案会变。
6. **多语种 MMBench 错版**：MMBench 有 DEV/TEST/EN/CN 多个版本，要按报告需求选对版本。

## 小结

多模态评测是"维度 + benchmark + 流程 + 去污染"的系统工程。感知、推理、OCR、幻觉、安全五大维度需要分开评测，不能用单一数字掩盖能力短板。VLMEvalKit 是工业事实标准，但**评判模型强不强，最终仍要看真实业务场景的人工评测**——benchmark 上的分数只是入场券，不是终点。
