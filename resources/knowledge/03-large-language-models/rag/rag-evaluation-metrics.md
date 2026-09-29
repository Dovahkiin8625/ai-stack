# RAG 评估指标：如何知道你的系统真的变好了

没有评估就没有迭代。但 RAG 的答案不像机器翻译有标准答案，无法直接套用 BLEU/ROUGE。本文介绍 RAGAS 框架的核心维度、为什么传统指标会失灵，并展示一个"LLM-as-judge"的最小可行评估代码。

## 一、为什么 BLEU/ROUGE 不够

BLEU/ROUGE 是 n-gram 重叠率，对**生成文本与参考答案的字面相似度**敏感。但在 RAG 中：

- 同一事实可以有多种正确表述；
- 真正该评估的是"答案是否忠于上下文"、"是否答到点子上"，而不是与参考表述多像。

所以我们需要的是**对单个维度的细粒度评估**，而不是全文相似度。

## 二、RAGAS 的三大维度

RAGAS（Retrieval-Augmented Generation Assessment）是当下最广泛使用的 RAG 评估框架之一，把质量拆成三个独立维度：

1. **Context Relevance（上下文相关性）**：检索回来的 chunks 与用户问题的相关程度。分数高 = 检索器精准。
2. **Answer Faithfulness（答案忠实度）**：最终答案中陈述的事实是否能从给定上下文中找到依据。分数高 = 答案不幻觉。
3. **Answer Relevance（答案相关性）**：答案是否真的回答了用户问题，而不是答非所问。

每个维度都由一个 LLM 担任 judge，按 0-1 打分或 0-5 打分。

## 三、数学定义（简化）

设 $Q$ 为用户问题，$C$ 为检索上下文，$A$ 为生成答案：

```math
\text{ContextRelevance} = \frac{|\text{claims}(Q) \cap \text{claims}(C)|}{|\text{claims}(Q)|}
```

```math
\text{Faithfulness} = \frac{|\text{claims}(A) \cap \text{claims}(C)|}{|\text{claims}(A)|}
```

```math
\text{AnswerRelevance} = 1 - \text{distance}(\text{embed}(Q), \text{embed}(A\text{ from }Q'))
```

其中 $Q'$ 是从 $A$ 反推出的"如果我问这个问题，答案会是什么"。直觉上：好的答案应该既忠于证据、又切题。

## 四、LLM-as-judge 代码示例

```python
import json
from openai import OpenAI

client = OpenAI()

JUDGE_PROMPT = """你是一名严格的 RAG 评估员。
请根据参考资料评估生成答案的忠实度。

问题：{question}
参考资料：{contexts}
答案：{answer}

只输出 JSON：{{"faithfulness": 0.0~1.0, "reason": "<简短理由>"}}
"""

def judge_faithfulness(question, contexts, answer):
    msg = JUDGE_PROMPT.format(question=question, contexts="\n".join(contexts), answer=answer)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": msg}],
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)

# 在一个评估集上批量打分
results = [judge_faithfulness(ex.q, ex.contexts, ex.answer) for ex in eval_set]
print("mean faithfulness:", sum(r["faithfulness"] for r in results) / len(results))
```

为了让 judge 更稳定，可以：

- 多采样取均值；
- 与人工标注的子集做相关性检查；
- 用更强的模型定期抽样验证。

## 五、其它常用指标

- **Context Recall**：真实答案涉及的事实是否被检索到。需要有人工标注的 ground-truth chunks。
- **Context Precision**：检索结果中相关 chunk 占的比例。
- **Answer Correctness**：与参考答案事实一致的程度（适合有标准答案的场景，如 HotpotQA）。

## 六、评估集的构造

光有指标不够，还需要**评估集**。建议：

1. **采样真实线上问题**，再人工补充 ground-truth。
2. **合成难题**：用 LLM 基于文档生成多跳问题、矛盾问题、超出知识库问题。
3. **定期更新**：每次系统改动后，跑全套评估，看到底是哪个维度掉了。

## 小结

RAG 的评估是一个"评估维度 + LLM-as-judge + 高质量评估集"三位一体的工作。RAGAS 是起点而非终点——把它当作一个参考，结合业务场景定义自己的指标，才能真正驱动系统持续改进。
