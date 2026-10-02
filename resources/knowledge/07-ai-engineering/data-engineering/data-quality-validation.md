# 数据质量评估：从启发式到模型打分

"垃圾进，垃圾出"在 AI 训练里格外真实。数据质量评估的目标是给每条样本打分，让流水线能**自动过滤低质量数据、按质量加权训练、定位数据分布漂移**。本文梳理三类主流方法：基于规则、基于 perplexity、基于模型打分，以及工业级的质量过滤 pipeline。

## 一、质量评估的三类方法

| 方法 | 代表 | 优点 | 缺点 |
|---|---|---|---|
| **规则 / 启发式** | KenLM 困惑度、长度分布、词比例 | 极快、可解释 | 难捕捉语义质量 |
| **Perplexity / LM 评分** | KenLM、fastText 分类器 | 中等成本、能评估"自然度" | 偏向高频模式 |
| **模型打分** | GPT-4-as-judge、BERT classifier | 接近人类判断 | 贵、有偏差 |

实务上通常三段串联：**启发式粗筛 → perplexity 精筛 → 模型打分终判**。

## 二、启发式过滤：便宜有效

最便宜的过滤方法，跑得飞快，适合亿级数据。

```python
from langdetect import detect
import fasttext

# 1. 语言检测：只保留中文
lang_model = fasttext.load_model("lid.218e.bin")

def is_chinese(text: str, threshold: float = 0.8) -> bool:
    labels, probs = lang_model.predict(text, k=1)
    return labels[0] == "__label__zh" and probs[0] > threshold

# 2. 基础统计
def quality_score(text: str) -> dict:
    n_chars = len(text)
    n_words = len(text.split())
    return {
        "length_ok": 50 < n_chars < 50_000,
        "word_ratio": n_words / max(n_chars, 1),     # 中英文比例
        "has_punct": any(c in "。！？；，、,.!?;" for c in text),
    }
```

进阶指标：

- **符号-字符比**：广告、爬虫残页通常符号占比异常高。
- **平均词长**：太短可能是"标题党"，太长可能是机器拼接。
- **n-gram 重复率**：重复 n-gram 多半是模板或 SEO 内容。

## 三、Perplexity / KenLM 过滤

KenLM 训练一个轻量 n-gram 语言模型，给每条文本打分。**困惑度（perplexity）越低，文本越"像训练分布"**。

```bash
# 训练
bin/lmplz -o 5 < clean_text.txt > model.arpa
bin/build_binary model.arpa model.bin
```

```python
import kenlm

model = kenlm.Model("model.bin")

def perplexity(text: str) -> float:
    return model.perplexity(text)

# 过滤：保留 perplexity 在合理区间的样本
filtered = [t for t in texts if 50 < perplexity(t) < 1000]
```

经验：困惑度过低的可能是"广告、重复短语"，过高的可能是"乱码、非自然语言"。一般取分布的 5%-95% 分位。

## 四、模型打分：GPT-4-as-Judge

当需要"判断这段文本是否教育性、是否有害、是否事实正确"等复杂语义属性时，用大模型打分。

```python
from openai import OpenAI
client = OpenAI()

SCORE_PROMPT = """你是一名数据质量评审。请给下面这段文本打 1-5 分，
1 分：完全无用（乱码、广告、机器生成）
3 分：可用但平庸
5 分：高质量（信息密度高、表达清晰、有教育价值）

只输出一个整数。

文本：
{text}
"""

def llm_score(text: str) -> int:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": SCORE_PROMPT.format(text=text[:2000])}],
        temperature=0,
    )
    try:
        return int(resp.choices[0].message.content.strip())
    except ValueError:
        return 3  # 兜底

# 过滤：保留 ≥ 4 分
high_quality = [t for t in texts if llm_score(t) >= 4]
```

**注意偏差**：

- LLM 倾向于给"看起来规整"的文本高分（格式偏见）。
- 评分受 prompt 措辞影响大，要做 A/B 校准。
- 对低资源语言、长尾领域能力有限。

## 五、BERT 二分类：成本-精度折中

如果需要上百万样本打分，用 GPT-4 太贵。可以训练一个 BERT 二分类器作为 proxy：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./quality_bert")
tok = AutoTokenizer.from_pretrained("./quality_bert")

@torch.inference_mode()
def bert_score(text: str) -> float:
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    logits = model(**inputs).logits
    prob = torch.softmax(logits, dim=-1)[0, 1].item()
    return prob  # 1: 高质量

# 用 GPT-4 给小批量数据打标 → 训 BERT → 大规模推理
```

训练数据的标注来源可以是人工标注 + LLM 标注混合。Llama 3 用了类似策略训练它们的 quality classifier。

## 六、工业级 Pipeline 示例

```python
def filter_pipeline(text: str) -> bool:
    # 1. 启发式粗筛（~99% 数据通过）
    if not basic_stats_ok(text):
        return False
    if not is_chinese(text):
        return False

    # 2. Perplexity 精筛（~80% 通过）
    ppl = perplexity(text)
    if not 50 < ppl < 1000:
        return False

    # 3. BERT 打分（~70% 通过）
    if bert_score(text) < 0.6:
        return False

    # 4. 可选：GPT-4 抽检关键样本
    return True

# 处理 1 亿条语料
filtered = [t for t in tqdm(corpus) if filter_pipeline(t)]
```

## 七、质量评估的元问题

1. **谁来定义"质量"？** 教育性？事实性？多样性？不同目标函数对应不同 filter。
2. **过滤后分布偏移**：极端过滤会让训练分布偏窄，反而损害泛化。Llama 论文里强调"不要过度过滤"。
3. **评估闭环**：定期抽 1000 条让人类标注，看自动打分与人类判断的相关性，发现漂移。
4. **可重现性**：filter 规则 + 阈值 + 模型版本要全部记录到 data card。

## 小结

数据质量评估是 AI 数据流水线的"质量门"。启发式粗筛 + perplexity 精筛 + 模型打分终判的三段式，是当前最经济有效的组合。LLM 时代下，"高质量数据"比"海量数据"更稀缺——10M 条精挑细选的样本，往往胜过 1B 条粗制滥造。下一篇我们将进入 **Feature Store**——如何把工程化的特征管理引入 AI 系统。
