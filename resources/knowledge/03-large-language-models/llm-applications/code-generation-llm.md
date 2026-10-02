# 代码生成 LLM：从 Copilot 到 Agent

代码生成是 LLM 商业化最成功的场景之一——GitHub Copilot、Cursor、Continue、Claude Code 背后的核心都是"代码 LLM"。本文介绍其模型谱系、IDE 集成架构、FIM 训练目标、Agent 式编程，并给出 OpenAI FIM API 的最小补全客户端代码。

## 一、代码 LLM 的谱系

代码 LLM 大致经历三代：

| 代际 | 代表模型 | 主要特点 |
| --- | --- | --- |
| **第一代（2021-2022）** | Codex、Codex-Java、CodeT5 | GPT-3 微调，API 调用补全，能写小函数但常"扯" |
| **第二代（2023-2024）** | Code Llama、DeepSeek-Coder、Qwen-Coder、StarCoder2 | 开源、长上下文、FIM 训练、多语言 |
| **第三代（2024-）** | Claude 3.5 Sonnet、GPT-4o、Cursor Composer、Devin | Agent 式：理解 repo、调工具、多文件修改 |

闭源关系（封装能力）vs 开源（本地可用）。后者在 IDE、本地部署上越来越受欢迎。

## 二、典型 IDE 集成架构

以 Cursor / Copilot / Continue 为代表：

```
[IDE 编辑器]
   ├─ 当前光标 + 上下文窗口（前后 N 行）
   ├─ 打开的其它文件（可选）
   └─ Repo-level RAG（chunk of related code, docs）
            │
            ▼
     [Prompt 组装] ──► [代码 LLM] ──► [流式补全]
                                  │
                                  └─► [Inline suggestion / Diff]
```

IDE 集成的几个工程要点：

1. **延迟预算极紧**：Tab 键按下到首字出现应 < 300ms，否则“Tab 键”感会崩坏。
2. **上下文需要“上下文打包”**：从光标周围 + 同文件中抽取出可能相关变量、函数、项目“约定”。
3. **多文件 RAG**：粗粝的向量化对比文件 chunk + top-k 检索。
4. **补全 vs 对话 vs Agent**：IDE 中嵌入三类不同模型 + 不同提示词都是。

## 三、Fill-in-Middle（FIM）

普通语言模型是“左边变右边”。FIM 则同时考虑“左+右 → 中间”，适合补全中间的代码。训练样本拼接为：

$$
\text{[前缀]} \, \text{[sep\_prefix]} \, \text{[后缀]} \, \text{[sep\_suffix]} \, \text{[中间]}
$$

推理时给定前缀 + 后缀，模型补全中间部分。OpenAI 的 `completions` 接口通过 `prompt` + `suffix` 两个字段直接暴露了 FIM。

```python
from openai import OpenAI
client = OpenAI()

prefix = "def add(a, b):\n    "
# 后缀是某种注释中提示返回类型
suffix = "\n    # end"

resp = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prefix,
    max_tokens=64,
    temperature=0,
    stop=["\n\n"],
)
print(prefix + resp.choices[0].text)
# 期望输出:
#     return a + b
#     # end
```

> 提示：FIM 对“补全中间”很有效，但对"跨函数补全" 依然依赖上下文。提高补全质量的关键是提供**完整函数签名 / 注释**作为前置上下文。

## 四、Repo-level 上下文与代码 RAG

FIM 本质上还是“左右内”。当补全跨文件、跨函数、跨库时，必须检索。例如“在文件 A 中调用文件 B 的某个函数”：

1. **静态解析**：Tree-sitter / LSP 获取项目符号表。
2. **Embedding + 向量检索**：从光标处召出相关 chunk。
3. **拼接为 prompt**：以“以下是项目中相关定义”的形式拼到 FIM 提示。

伪代码：

```python
import chromadb
from openai import OpenAI
chroma = chromadb.PersistentClient(path=".idx")
col = chroma.get_or_create_collection("code")

def embed_code(text: str):
    return OpenAI().embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

def repo_context(file_path: str, symbol: str, k: int = 5):
    qv = embed_code(f"file={file_path} symbol={symbol}")
    return col.query(query_embeddings=[qv], n_results=k)

# 拼装 FIM 提示
prefix = "def render_report():\n    # TODO: 调用 utils.chart.tables()\n    "
ctx = "\n\n".join(d for d in repo_context("report.py", "render_report")["documents"][0])
suffix = "\n    return fig"
full_prompt = f"{ctx}\n\n{prefix}"
```

## 六、Agent 式代码 LLM（Devin / SWE-Agent）

Agent 与“补全”不同，它是多步决策+多工具调用。典型动作空间：

- 读文件 / 写文件。
- 运行 shell 命令。
- 调用 LSP / grep / 测试。

流程是：

```
Repo ──► [Plan] ──► [Edit/Test 循环] ──► [Submit]
                  ↑           │
                  └───────────┘
```

代表性 benchmark 是 SWE-bench：一个包含真实 GitHub issue 的集合，要求 agent 修改对应 repo 中的多个文件并使现有测试通过。2024 年下半年顶级模型在 SWE-bench 上从 5% 以下提升到 50%+。

## 七、代码评测基准

| 基准 | 主要场景 | 输入长度 | 评测方式 |
| --- | --- | --- | --- |
| **HumanEval** | 单函数补全 | ~50 行 | pass@k |
| **MBPP** | 极简编程题 | ~5 行 | pass@k |
| **LiveCodeBench** | 定期更新的竞赛题 | 中等 | pass@1 + 隐藏测试 |
| **SWE-bench** | 真实 GitHub issue | 全 repo | 仓库级测试通过率 |
| **RepoBench** | 跨文件补全 | 长 | pass@1 |
| **BigCodeBench** | 复杂 API 调用 | 中等 | pass@k |

pass@k 定义为：

$$
\text{pass@k} = \mathbb{E}_{\text{problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

其中 $n$ 为采样数，$c$ 为通过数。

## 八、一个最小补全客户端 + 评测示例

```python
import json, time
from openai import OpenAI
client = OpenAI()

# HumanEval-style 样例
problems = [
    {"id": 1,
      "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Return a+b.\"\"\"\n    ",
      "tests": "assert add(2, 3) == 5"},
    {"id": 2,
      "prompt": "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is prime.\"\"\"\n    ",
      "tests": "assert is_prime(7) and not is_prime(9)"},
]

def complete(prompt: str) -> str:
    r = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt, max_tokens=128, temperature=0,
    )
    return prompt + r.choices[0].text

passed = 0
for p in problems:
    code = complete(p["prompt"])
    try:
        exec(code + "\n" + p["tests"], {})
        passed += 1
        print(f"[{p['id']}] PASS")
    except Exception as e:
        print(f"[{p['id']}] FAIL: {e}")

print(f"\npass@1 = {passed}/{len(problems)}")
```

> 提示：本地跑 HumanEval/MBPP 时实际是生成补全 + 跑测试。生产中请用 evalplus / bigcode-evaluation-harness。

## 九、IDE 集成的隐含成本

很多人低估了以下几个点：

- **Tab 接受率**：Copilot 公开数据约 30%。开发者实际生产力提升与 Tab 接受率强相关。
- **代码安全**：LLM 可能补全含漏洞代码、硬编码密钥、过期 API。
- **License 风险**：严格遵循训练集 ≠ 严格遵循该代码原文 license。

## 小结

代码 LLM 是"领域模型 × IDE 体验"的联合产物。模型本身（FIM、Agent）只是一部分，IDE 集成、RAG 检索、上下文装配、补全延迟、安全过滤、benchmark 评测共同决定了产品体验。下一篇会聊“如何评估这些应用”：从 MMLU 到 SWE-bench。