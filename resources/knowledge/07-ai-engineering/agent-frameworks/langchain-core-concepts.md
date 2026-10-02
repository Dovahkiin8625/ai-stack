# LangChain 核心概念：Chain、Runnable 与 LCEL

LangChain 是当前 LLM 应用层最普及的开发框架之一。它的设计思想经历了从「Chain 类硬编码」到「LCEL（LangChain Expression Language）」的重构——把所有组件抽象成 `Runnable` 接口，使组合、并行、异步、流式都变成统一的一等公民。本文从核心抽象出发，串起 Chain、PromptTemplate、OutputParser、Memory 与 Runnable 的关系，并给出一个端到端最小可运行示例。

## 一、核心抽象一览

```text
Component      │ 作用                       │ 典型实现
───────────────┼────────────────────────────┼─────────────────────────────
ChatModel      │ 调用大模型                 │ ChatOpenAI, ChatAnthropic
PromptTemplate │ 拼 prompt                  │ ChatPromptTemplate
OutputParser   │ 解析模型输出               │ StrOutputParser, PydanticOutputParser
Retriever      │ 检索外部知识               │ VectorStoreRetriever
Tool           │ 让模型调用外部函数         │ @tool decorator
Memory         │ 跨轮保存上下文             │ ConversationBufferMemory
Runnable       │ 统一接口：invoke/stream/batch│ 所有上述组件的基类
```

`Runnable` 是关键。它有四个统一方法：`invoke` / `stream` / `batch` / `ainvoke`。任何组件只要实现了 `Runnable`，就可以用同一个表达式语言（LCEL）组合。

## 二、LCEL：组合即管道

LCEL 用 Python 的 `|` 运算符把多个 `Runnable` 串成管道：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名严谨的技术文档翻译，把用户输入翻译成英文。"),
    ("user", "{text}"),
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

chain = prompt | model | parser   # ← LCEL 管道

# 等价于：parser.invoke(model.invoke(prompt.invoke({"text": "你好"})))
print(chain.invoke({"text": "你好，世界"}))
```

`|` 不是普通语法糖，它返回的是一个 `RunnableSequence`，自动获得 invoke / stream / batch / async 全套能力。

## 三、并行与分支

LCEL 用 `RunnableParallel` 拼多个分支，用 `RunnableBranch` 做条件路由：

```python
from langchain_core.runnables import RunnableParallel, RunnableBranch

# 并行：同一个输入同时送给多个 chain
parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
    sentiment=sentiment_chain,
)

result = parallel.invoke({"text": long_article})
# {"summary": "...", "keywords": [...], "sentiment": "positive"}
```

```python
# 条件分支：根据输入选择不同 chain
branch = RunnableBranch(
    (lambda x: "代码" in x["topic"], code_chain),
    (lambda x: "数学" in x["topic"], math_chain),
    default_chain,
)
```

`RunnablePassthrough.assign()` 用于在管道中"注入"额外字段，是构建 RAG / agent 链的常用工具。

## 四、OutputParser：从字符串到结构化对象

LLM 输出通常是字符串，但下游经常需要 Pydantic 对象、列表、字典：

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="电影名")
    score: float = Field(description="评分，0-10")
    pros: list[str] = Field(description="优点列表")

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是影评人。请按格式输出。\n{format_instructions}"),
    ("user", "评价电影：{title}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

review = chain.invoke({"title": "Inception"})
# MovieReview(title='Inception', score=9.2, pros=['叙事精巧', '视觉震撼'])
```

把 LLM 输出从「不可控的字符串」变成「强类型对象」，是工程化的关键一步。

## 五、Memory：跨轮上下文

LLM 本身是无状态的，多轮对话要靠 Memory 显式管理。最简形式——用 `MessagesPlaceholder` 把历史塞进 prompt：

```python
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名耐心的助手"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
])

# 自维护 history 列表
from langchain_core.messages import HumanMessage, AIMessage

history = []
def chat(user_input: str) -> str:
    answer = (prompt | model | parser).invoke(
        {"history": history, "input": user_input}
    )
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))
    return answer
```

生产环境通常用 `RedisChatMessageHistory`、`PostgresChatMessageHistory` 持久化，并配合 `RunnableWithMessageHistory` 自动注入。

## 六、流式输出（SSE）

LLM 应用几乎都要流式响应用户体验。LCEL 的 `stream()` 方法直接吐 token：

```python
for chunk in chain.stream({"text": "写一首关于秋天的诗"}):
    print(chunk, end="", flush=True)
```

在 FastAPI 里包一层 SSE：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    def gen():
        for chunk in chain.stream({"text": req.text}):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

`chain.astream_events()` 还能产出结构化事件流（on_llm_start、on_tool_end 等），是 LangSmith 追踪和 Agent 调试的底层支撑。

## 七、LangSmith：可观测性

LCEL 链的所有运行都会自动上报到 LangSmith（如果设置了 `LANGCHAIN_TRACING_V2=true`），可以看到：

- 每一步的输入 / 输出 / 延迟
- LLM 调用的 prompt、completion、token 用量
- Tool 调用的参数和返回值
- 失败重试与 fallback 路径

这相当于给 LLM 应用装上 APM（Application Performance Monitoring），定位「为什么这个回答慢」、「为什么模型没按预期调用 Tool」等问题。

## 小结

LangChain 的核心是用 `Runnable` 把所有组件统一成一个组合式语言（LCEL）。一旦掌握了 `prompt | model | parser` 的基本管道，以及 `RunnableParallel` / `RunnableBranch` 的组合原语，就能写出可读、可测、可流式、可观测的 LLM 应用。下一篇我们将进入 **Tool Use 与 Agent Loop**——让模型自己决定"什么时候调什么工具"，是 Agent 框架的灵魂。
