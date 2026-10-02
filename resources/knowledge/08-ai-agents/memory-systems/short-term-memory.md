# 短期记忆：上下文窗口与 Working Memory

短期记忆是 Agent 最基础也最关键的认知组件——它决定了 Agent 在一个会话内能"记住"多少信息。从 LLM 的 context window 到认知科学中的 working memory，短期记忆的设计直接决定了 Agent 能处理多复杂的任务。本文深入短期记忆的容量限制、信息编码、压缩策略，以及 LLM 时代下的工程实现。

## 一、什么是短期记忆

认知科学中，**短期记忆（Short-Term Memory, STM）**或**工作记忆（Working Memory）**指个体在短时间内保持并主动操纵信息的能力。

```text
长期记忆（硬盘）  →  短期记忆（内存）  →  处理单元（CPU）
   长期事实             当前会话上下文          LLM 推理
```

### 1.1 经典模型：Atkinson-Shiffrin

```text
环境 ──→ 感官寄存器 ──→ 短期记忆 ──→ 长期记忆
        (1-2 秒)       (15-30 秒)     (永久)
                          ↑
                       rehearsal
                       复述
```

关键数字：

- **Miller's Law**：短期记忆容量约 7±2 个块（chunk）。
- **持续时间**：12-30 秒（无复述）。
- **容量换算**：约 4 个 chunk 在 5 秒内可保留。

### 1.2 LLM 时代映射

| 认知概念 | LLM 对应 |
|---|---|
| 短期记忆 | Context window（4K - 1M tokens） |
| 工作记忆 | KV cache（推理时活跃状态） |
| 注意 | Attention 权重 |
| 复述 | prompt 重述 / 重新生成 |
| 容量限制 | Max context length |
| 衰退 | 无（一旦进入 context 就保留） |

## 二、Context Window 的物理限制

### 2.1 主流模型的 context 长度

```text
GPT-3.5         4K → 16K
GPT-4           8K → 128K
GPT-4o          128K
Claude 3.5      200K
Gemini 1.5 Pro  1M - 2M
Llama 3.1       128K
Qwen 2          128K
```

### 2.2 Context 与成本的关系

```text
输入成本 = 输入 tokens × 单价

GPT-4o:    $2.5 / 1M input tokens
128K context 输入 = $0.32 per request

如果每请求都用满 128K，1 万请求 = $3,200
```

### 2.3 Context 与延迟的关系

LLM 推理时间大致 $O(n^2)$ 于 context 长度（attention）：

```python
# 量级估算
context_len = 128_000
attention_time = context_len ** 2 * 1e-9     # ≈ 16 s
# 实际由于 Flash Attention、PagedAttention，O(n) 也可能
```

## 三、短期记忆的工程实现

### 3.1 直接 prompt 注入

```python
class SimpleShortTermMemory:
    def __init__(self, max_tokens: int = 8000):
        self.messages: list[dict] = []
        self.max_tokens = max_tokens

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._truncate_if_needed()

    def _truncate_if_needed(self):
        while self.token_count() > self.max_tokens:
            # 保留 system + 最近 N 轮
            self.messages.pop(1)        # 移除最早的用户消息

    def token_count(self) -> int:
        return sum(len(m["content"]) // 4 for m in self.messages)

    def get_context(self) -> list[dict]:
        return self.messages
```

最简单，但容易丢失关键信息。

### 3.2 滑动窗口

```python
class SlidingWindowMemory:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.messages = []

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.window_size:
            self.messages.pop(0)
```

固定保留最近 N 条消息，简单但上下文切换时丢失旧信息。

### 3.3 Token-aware 截断

```python
import tiktoken

class TokenAwareMemory:
    def __init__(self, max_tokens: int = 8000, model: str = "gpt-4o"):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
        self.messages = []

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._compact()

    def _compact(self):
        while self.total_tokens() > self.max_tokens:
            # 优先保留 system message
            system_msgs = [m for m in self.messages if m["role"] == "system"]
            other_msgs = [m for m in self.messages if m["role"] != "system"]

            if len(other_msgs) > 2:
                other_msgs.pop(0)      # 移除最早的非 system 消息
            else:
                break

            self.messages = system_msgs + other_msgs

    def total_tokens(self) -> int:
        return sum(len(self.encoding.encode(m["content"])) + 4 for m in self.messages)
```

### 3.4 Summary-based 压缩

```python
class SummaryMemory:
    """把早期消息压缩为摘要，保留近期消息完整"""
    def __init__(self, llm, max_recent: int = 10, summary_max_tokens: int = 500):
        self.llm = llm
        self.summary = ""
        self.recent: list[dict] = []
        self.max_recent = max_recent
        self.summary_max_tokens = summary_max_tokens

    def add(self, role, content):
        self.recent.append({"role": role, "content": content})

        if len(self.recent) > self.max_recent:
            # 把最早的一条"消化"到 summary
            evicted = self.recent.pop(0)
            self.summary = self.llm.invoke(
                f"现有摘要：{self.summary}\n新事件：{evicted}\n更新摘要（<{self.summary_max_tokens} 字）："
            )

    def get_context(self) -> list[dict]:
        system = {"role": "system", "content": f"对话摘要：{self.summary}"}
        return [system] + self.recent
```

这种"摘要 + 近期窗口"是 ChatGPT / Claude 实际用的方案。

## 四、Memory 管理策略

### 4.1 FIFO（First In First Out）

最旧消息先丢。简单但可能丢关键早期信息。

### 4.2 Importance-based（重要性）

用规则或模型评估每条消息的"重要性"，保留重要的：

```python
def importance_score(msg: dict) -> float:
    score = 0
    score += len(msg["content"]) * 0.01        # 长消息略重要
    score += msg["content"].count("?") * 0.5    # 问题更重要
    if msg["role"] == "user":
        score += 1                              # 用户消息更重要
    if "记住" in msg["content"]:
        score += 10                             # 显式要求记忆
    return score
```

### 4.3 Recency × Importance

```python
def retention_score(msg, current_step):
    recency = 1 / (current_step - msg.step + 1)
    importance = importance_score(msg)
    return 0.7 * recency + 0.3 * importance
```

MemGPT 等论文用类似公式做 eviction。

### 4.4 Token Budget 分配

```python
TOTAL_BUDGET = 8000
SYSTEM_TOKENS = 500
SUMMARY_TOKENS = 500
RECENT_TOKENS = 4000
RETRIEVED_TOKENS = 3000     # 从长期记忆检索回来的

assert TOTAL_BUDGET >= SYSTEM_TOKENS + SUMMARY_TOKENS + RECENT_TOKENS + RETRIEVED_TOKENS
```

明确划分预算比"塞满为止"更可控。

## 五、Working Memory 与 Attention

认知科学的 Baddeley 模型：

```text
                  Phonological Loop（语音）
                          ↕
Central Executive（中央执行） ──→ Episodic Buffer（情节缓冲）
                          ↕
                  Visuospatial Sketchpad（视觉空间）
```

对应到 LLM：

| Baddeley 组件 | LLM 对应 |
|---|---|
| Central Executive | 主注意力机制 |
| Phonological Loop | 当前对话文本 |
| Visuospatial Sketchpad | 多模态输入（图像、音频） |
| Episodic Buffer | 当前任务的中间状态 |

**工程启示**：LLM Agent 应该设计"中央调度器"来决定何时关注哪种信息，避免注意力被无关上下文稀释。

## 六、KV Cache 作为短期记忆

LLM 推理时 KV cache 才是真正的"工作记忆"：

```python
# vLLM 中每个请求有自己的 KV cache
# Block 数量 = (prompt_len + max_output_len) / block_size
# 当所有 block 用完，新 token 进不来 → 必须 evict
```

```text
prompt: [BLOCK 0][BLOCK 1][BLOCK 2]
generate: 不断占新 BLOCK...
max_len=4096, block_size=16 → 256 blocks 上限
```

**PagedAttention** 通过分页管理，把 KV cache 利用率推到接近 100%。

## 七、实战陷阱

### 1. "Lost in the Middle" 现象

Liu et al. 2023 发现：LLM 对 context **首尾**的信息回忆准确率最高，**中间**容易丢失。

```text
准确率：
  首部: 70%
  中部: 40%
  尾部: 75%
```

工程对策：把关键信息（系统提示、关键事实）放在 prompt 的开头或结尾。

### 2. 系统提示被稀释

```python
# 不好：长 history 让 system prompt 权重被冲淡
messages = [
    {"role": "system", "content": "你是一名医生"},       # ← 容易忽略
    {"role": "user", "content": "..."},                   # × 50 轮对话
    {"role": "user", "content": "感冒吃什么药？"}
]

# 好：定期重新强调关键约束
messages.append({"role": "system", "content": "[再次提醒] 你是一名医生"})
```

### 3. 工具结果爆炸

一次 tool 调用可能返回 10K tokens。短期记忆瞬间被占满：

```python
# 解决：truncate tool result
result = tool_call()
if len(result) > 2000:
    result = result[:2000] + "...[truncated]"
```

## 小结

短期记忆是 Agent 设计的"内存管理"问题。Context window 是硬约束，token 成本与延迟是软约束。工程上要设计**摘要 + 近期窗口 + 重要度**的混合策略，避免"塞满"和"丢失关键信息"两个极端。LLM 的"lost in the middle"现象提醒我们：**信息位置** 和 **token 预算**同样重要。下一篇我们将进入 **长期记忆**——如何让 Agent 跨会话保留知识。
