# 聊天机器人与对话 AI

"Chatbot"是 LLM 时代最被低估、也最被高估的应用形态。说低估，是因为一模型能力之上，对话系统的工程深度远超几个 prompt；说高估，是因为脱离上下文、记忆、约束的 chatbot 几乎一文不值。本文从模型可解释的几个核心维度展开：词、句子、篇章，并匹配最强可能的实现代码。

## 一、单轮 vs 多轮对话

单轮对话只是一问一答：

$$
\text{output} = f(\text{Prompt})
$$

多轮对话则需考虑历史：

$$
\text{output}_t = f(\text{System}, \text{History}_{1..t-1}, \text{Query}_t)
$$

历史信息非常重要，但也有代价：每次调用都需拼接所有上文 → token 越多、单价越高、且一旦超过 context window，模型“失忆”。这也是后面所有设计的根源。

## 三、Memory 设计：短期 + 长期

推荐“**短期原文+长期原文**”的混合设计：

- **短期 Memory（Working Memory）**：在一次会话中保留近 N 轮原文，供上下文连贯。
- **长期 Memory**：跨会话的个性化事实（姓名、偏好、过去聊过的重要话题）。通常是向量检索。

设近 $N_{轮原文}$ 轮进入 LLM context，往期写“用户喜欢简洁表达”等事实进入 long-term store，则混合检索公式为：

$$
\text{Response} = LLM(\text{System}, \text{Facts}_{LT}, \text{History}_{ST}, \text{Query})
$$

下面是一个完整可运行的 chatbot demo。

```python
import time, uuid
from typing import List, Dict
from openai import OpenAI

client = OpenAI()
SESSION = {1: "你是一名克制、专业的助理，回复不超过 80 字。"}
LT_STORE: Dict[str, List[str]] = {}     # 长期记忆：user_id -> facts

# ---------- 记忆辅助 ----------
class WorkingMemory:
    def __init__(self, max_turns: int = 8, max_chars: int = 4000):
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.msgs: List[dict] = []

    def add(self, role: str, content: str):
        self.msgs.append({"role": role, "content": content})
        # 滑动窗口：超过轮数或字数都丟最老
        while len(self.msgs) > self.max_turns \
              or sum(len(m["content"]) for m in self.msgs) > self.max_chars:
            self.msgs.pop(0)

def retrieve_facts(user_id: str, query: str, k: int = 3) -> List[str]:
    # 这里伪实现：真实场景接 Chroma / Qdrant
    return LT_STORE.get(user_id, [])[:k]

def memorize(user_id: str, fact: str):
    LT_STORE.setdefault(user_id, []).append(fact)

# ---------- 主循环 ----------
def chat(user_id: str, user_msg: str, wm: WorkingMemory) -> str:
    wm.add("user", user_msg)
    facts = retrieve_facts(user_id, user_msg)

    messages = [
        {"role": "system",
         "content": SESSION[1] + "\n已知用户事实：" + "; ".join(facts)},
        *wm.msgs,
    ]
    reply = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.4,
    ).choices[0].message.content
    wm.add("assistant", reply)
    return reply

# ---------- 使用 ----------
uid = "u_42"
wm = WorkingMemory()
print(chat(uid, "我叫 Dovah，最近在调 LLM 服务。", wm))
memorize(uid, "用户名叫 Dovah，是后端工程师")
print(chat(uid, "你还记得我是谁吗？", wm))
```

代码里有 3 个工程点值得反复看：

1. **滑动窗口同时控轮数 + 控字数**，避免“轮数不多但 token 爆炸”。
2. **长期记忆伪实现是 vector DB**，生产中需加上嵌入、去重、过期清理。
3. **System prompt 可以动态拼装**，但不要超出模型默认设定之外的训练分布。

## 四、上下文管理的其他选择

除了上述记忆，还有 5 种常见上下文压缩方法：

- **滑动窗口**：如上所示，最简、但会丢失远程上下文。
- **摘要压缩**：AIn 的另一种常见方式是“老 context + 新 query → 摘要”以准会话。
- **Summarization**：一次性会话超出时，把较早上下文压缩为总结。
- **Token-aware truncation**：不是按段落，而是按 token 计数。
- **RAG over dialogue**：别再把整个会话压进 prompt，而是把历史的文本嵌入检索。

## 五、角色一致性（System Prompt 设计）

System prompt 是 chatbot “人格”的唯一可信赖抓手。设计原则：

1. **明确身份、语气、限制**。不要一上来堆 5 页。
2. **包含 “不知道怎么办”**。明确告知，遇到不确定时可以说“我不知道”。
3. **给出示例对话**。模型对 few-shot 的道循准确度高于自然语言描述。
4. **动态追加但不动静态部分**。运行时拼装动态事实，避免改写 persona。

反面示例：“你是一个智能助手，能够回答各种问题。”这与没说一样。
正面示例：“你是『茶茶』，中国茶馆出品助手。默认不用 30 量，用 80-150 字。说话不夸大商品功效。若不确定，请说"我查询后告知你"。”

## 六、Guardrails：过滤输入 + 校验输出

即使能力逆天的模型，也需额外一层保护：

```python
import re

DISALLOWED = ["phishing", "恶意代码", "赌博"]

def input_guard(text: str) -> bool:
    return not any(w in text for w in DISALLOWED)

def output_guard(text: str) -> str:
    # 去掉明显不符合要求的输出模式
    return re.sub(r"ignore previous instructions", "[已脱敏]", text, flags=re.I)

def safe_chat(user_msg: str) -> str:
    if not input_guard(user_msg):
        return "抱歉，我不能处理这类请求。"
    raw = chat(uid, user_msg, wm)
    return output_guard(raw)
```

生产中常使用 Guardrails AI、LlamaGuard 这类专用工具。同样输出前哨能拦截明显的高风险输出。

## 七、情感与拟人

智能体情感识别不是“人造同理心”（ps：）本身，而是让对话避免“通用抗顺”。三个反作用场景：

- **检测到用户不满**：道歉 + 主动升级人类。
- **检测到用户困惑**：优化完关键问题后再问。
- **避免滥用情感词**：尽量“不手软”、“‏备別”等。

```python
from anthropic import Anthropic

claude = Anthropic()
SENTIMENT_PROMPT = """判断以下文本的情感，只输出 positive/neutral/negative。
文本：{msg}
"""
def sentiment(msg: str) -> str:
    r = claude.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": SENTIMENT_PROMPT.format(msg=msg)}],
    )
    return r.content[0].text.strip()

if sent in ("negative",):
    # 升级到人类客服
    return "我转人工处理了。”"
```

## 八、Chatbot 评估

chatbot 是一个“词、句子、上下文、质量”都可被检测的系统。常见指标：

- **任务完成率**：任务成功率。
- **Context Retention**：上下文的连贯性。
- **Persona Consistency**：人设一致性。
- **Safety**：安全 / 合规 / 反歧视。

评估场景：集成后静态评估集 + 动态评估（LLM-as-judge）。

## 小结

chatbot 是 LLM 工程的“入门题”，但其工程深度超过大多数 prompt 设计：从 Memory、角色一致性、Guardrails 到情感识别与评估，都需独立实现。本文中的代码可在本地直接运行，下一步可以接入实际 vector DB（Chroma / Qdrant）以实现长期记忆。