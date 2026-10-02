# LLM 成本优化：从 Prompt 到架构

LLM 调用费高，且“有点”隐藏成本：输入 token、输出 token、超时补调、缓存未命中、用户可能有难以忍受的额外 token 损耗。本文拆解 LLM 成本的各个驱动，并给出模型路由、prompt 压缩、KV cache 等工程手段。

## 一、成本拆解

一个 LLM 请求的单次调用成本可以表示为：

$$
C_{\text{req}} = C_{\text{in}} \cdot t_{\text{in}} + C_{\text{out}} \cdot t_{\text{out}} + C_{\text{infra}} \cdot T_{\text{latency}}
$$

其中：

- $t_{\text{in}}, t_{\text{out}}$：输入、输出 token 数。
- $C_{\text{in}}, C_{\text{out}}$：每 token 单价（一般输出 > 输入 3-5 倍）。
- $T_{\text{latency}}$：服务耗时（GPU·小时 或 后台增量）。

公式提示三个优化路径：减少 token、选择便宜模型、缩短服务耗时。

## 二、常见模型价价

| 模型 | 输入单价 / 1M | 输出单价 / 1M | 场景 |
| --- | --- | --- | --- |
| GPT-4o | $2.5 | $10 | 复杂推理、多模态 |
| GPT-4o-mini | $0.15 | $0.60 | 高并发、轻量任务 |
| Claude 3.5 Sonnet | $3 | $15 | 强推理、长上下文 |
| Claude 3.5 Haiku | $0.80 | $4 | 低延迟任务 |
| DeepSeek-V3 | $0.14 (缓存 $0.014) | $0.28 | 性价比高 |
| Qwen2.5-72B | $0.4 | $1.2 | 中文、多语言 |
| Llama-3.1-70B (自托管) | ~ GPU 成本 | 同上 | 高 QPS、可控 |

## 三、自托管 vs API 的选择

不是“产品负担越低越选自托管”。决策树：

```
预计 QPS ≥ 30 且平均推理 ≥ 1k token？
├─ 否 → API：是唯一可行性 + 最低负载。
└─ 是
    ├─ 有现成 GPU 资源（空闲）？且折旧 3 年内可接受？
    │   └─ 是 → 评估自托管（如 vLLM / SGLang）。GPU 成本 / 月可估算。
    └─ 否 → API。考虑商业供应商 + 折扣套餐。
```

补充变量：

- **延迟预算**：用户-系统 延迟要求 < 100ms 首选 API。
- **数据隐私**：保密领域首选自托管。
- **模型选择**：自托管仅能选开源 / 商用许可允许的。

## 四、Prompt 压缩

模型调用费主要是 token。Prompt 压缩能轻松减掉 20-50%。思路：

- **XML/YAML 代替 JSON**：原本多个 key-value 拿了多话、文本用 XML 可少一连串。
- **压缩表达**：“同能”代替 “super helpful”，节省 0。
- **重建 prompt 模板**：拼装 prompt 上去后由系统去重。
- **总结上下文**：会话中多余上下文以 LLM 总结。

例：从 system prompt 中删除“我能帮助你的话题” 限定词，“你是一个助理”，平均 prompt 从 700 字下降到 220 字。

## 五、模型路由

最有效的优化是“不是所有请求都需要 GPT-4o”。任务路由代码：

```python
from enum import Enum
from openai import OpenAI

client = OpenAI()

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

def route(prompt: str, history_len: int) -> str:
    # 规则路由：几种简单分类
    if len(prompt) < 80 and history_len < 3:
        return "gpt-4o-mini"
    if any(k in prompt for k in ["证明", "推导", "分析", "compare"]):
        return "gpt-4o"
    return "gpt-4o-mini"

def smart_chat(prompt: str, history_len: int):
    model = route(prompt, history_len)
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content
```

进阶路由：

- **LLM-as-router**：轻量 LLM 先判断问题难度，再选模型。
- **Cascade**：先以便宜模型尝试，置信度不够才升级。
- **领域路由**：仅限“客服 / 代码 / 代码生成"类场景分别用一个 모델。

## 七、KV/Prompt Cache

重复请求提示中有 KV cache（推理时）或 prompt cache（API 供应商） 上是策略。

- **Anthropic prompt cache**：≥ 1024 token 长 system → 可起明，手动 +80% 节省。
- **OpenAI prompt cache**：未公开。依赖供应商变化。
- **DeepSeek cache**：明面 0.014/1M token。

KV cache 在 vLLM 中是默认。可进一步使用 prefix caching：

- **prefix caching**：服务器中共享同一 prefix 的多个请求。适合“长 system + 多种 query”。

## 九、批处理与连续 batching

不一定总是需要顺序处理：

- **动态批处理**：连续 batching（Continuous Batching）可同时合并几十个请求，最大化 GPU 占用率。
- **吞吐导向**：不需要“象人一样快”的场景下“绝批”可以提近 20x throughput。

vLLM / TGI / SGLang 默认开启 continuous batching。vLLM 实例是“前端代理 + 性能可控”设计。

## 十、长上下文剪枝

长上下文不仅费，还“噪声多”。常见策略：

- **检索代替上下文**：RAG，“仅送最相关 chunk”代替“送上下文”。
- **过滤：不送相关文本 + 折叠重复内容。
- **过滤与优先序**：优先选取“与 query 最相关”的 top-k。
- **项目级上下文（代码 / 文档）**：仅 send 当前文件、几个相关函数。

## 十一、Token 计费陷阱

以下是几个常见“预算超控”的原因：

1. **输出 token 隐藏成本**：输出费 > 输入费。品验中“模型啰嗦”会右“额外多亿”。
3. **多轮上下文爆炸**：会话越长，每次调用都附上越多上下文。
4. **模型拒答却输出多文本**：某些提供商返回“不可用文本”还会计费。
5. **Function calling 额外轮次**：工具调用多轮中每次都带全文本 + 高价轮。
6. **多模态 token**：PDF 图片 token 费高，一个 page 可费 1k+ 上游。

## 十二、预算与监控

需重点实现：

- **按项目 / 团队 / 用户 / 任务限预算**：超限报警 + 限流。
- **每分钟 token 使用总量**：作为预算与采购依据。
- **P99 单次调用价格**：限制高价单次调用。

```python
def cost_limit_guard(model: str, prompt_tokens: int, user_id: str):
    cap = {"gpt-4o": 20000, "gpt-4o-mini": 200000}
    used = get_user_usage(user_id, "daily")
    if used + prompt_tokens > cap.get(model, 1000000):
        raise RateLimitError("今日配额超限")
```

## 小结

LLM 成本优化的顺序是：1）**减少 token**（prompt 压缩、RAG、上下文剪枝）；2）**选择便宜模型**（路由、批处理）；3）**使用缓存**（prompt cache、prefix caching）；5）**选择部署形式**（API vs 自托管）。每一次优化都不只是“跱 function”，都是“包颧-用跱”之间的权衡。下一步会讨论 LLM 应用的架构与可观测性。