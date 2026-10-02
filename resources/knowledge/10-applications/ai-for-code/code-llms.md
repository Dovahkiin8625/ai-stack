# 代码大模型：从补全到 Copilot

代码 LLM 是过去三年 AI 应用最成功的赛道——GitHub Copilot 累计百万付费用户，Cursor、Codeium、Cody 等产品百花齐放，Cursor 的 ARR 已突破 $100M。本文梳理代码 LLM 的训练数据、评测基准、代表模型（Codex、Code Llama、DeepSeek-Coder、Qwen-Coder），并讨论工程实践中的关键问题——补全延迟、长上下文、安全护栏。

## 一、为什么代码 LLM 率先突破

代码 LLM 率先商业化有几个特殊原因：

1. **数据丰富且结构化**：GitHub 公开了 PB 级代码，包含 commit 历史、PR、issue、文档——天然结构化数据。
2. **可自动验证**：代码有单元测试、编译器、Linter——可以**自动判断对错**（这是 RLVR 的理想场景）。
3. **任务封闭**：补全、翻译、重构、debug 等都是**目标明确**的任务。
4. **企业付费意愿强**：开发效率提升可量化——Copilot 数据报告 55% 速度提升，付费 ROI 明确。

## 二、训练数据：The Stack 与 CodeNet

### The Stack（BigCode, 2022）

HuggingFace BigCode 项目发布的 6.4 TB 代码数据，覆盖 **358 种编程语言**：

```python
# The Stack 结构示例
{
    "repo_name": "django/django",
    "path": "django/db/models/base.py",
    "language": "Python",
    "license": "BSD-3-Clause",
    "size": 12345,
    "content": "..."
}
```

关键决策：**仅保留许可证合规的代码**（MIT、Apache、BSD 等），过滤掉 GPL/LGPL——这是为避免代码"传染"开源义务。

### CodeNet（IBM, 2021）

侧重**算法题解**：1300 万+ 提交，覆盖 50+ 语言的代码竞赛题目：

```
problem_id: p00000
language: Python
status: Accepted
runtime: 0.05
code: "a, b = map(int, input().split()); print(a+b)"
```

适合训练"算法题"解题能力。

### 常见预处理

```python
def preprocess_code(code: str, language: str) -> str:
    """代码预处理：保留结构、去掉噪声。"""
    # 1) 规范化空白
    code = re.sub(r'\s+\n', '\n', code)
    code = re.sub(r'\n{3,}', '\n\n', code)
    
    # 2) 提取函数 / 类签名作为元信息
    if language == "python":
        tree = ast.parse(code)
        # ... 提取 docstring、类型注解
    
    # 3) 去掉调试信息
    code = re.sub(r'# TODO.*\n', '', code)
    
    # 4) 检测并删除硬编码密钥
    code = re.sub(r'(api_key|password|token)\s*=\s*[\'"][^\'"]+[\'"]', 
                  r'\1 = "<REDACTED>"', code, flags=re.IGNORECASE)
    
    return code
```

## 三、Tokenization 的特殊性

代码的 tokenizer 与自然语言**差异显著**：

```python
# 自然语言 vs 代码 tokenization 对比
from transformers import AutoTokenizer

samples = {
    "natural": "The quick brown fox jumps over the lazy dog.",
    "code_python": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "code_sql": "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id;",
}

for model in ["gpt2", "codellama/CodeLlama-7b", "deepseek-ai/deepseek-coder-6.7b"]:
    tok = AutoTokenizer.from_pretrained(model)
    print(f"\n=== {model} ===")
    for kind, text in samples.items():
        n = len(tok.encode(text))
        compression = len(text) / n
        print(f"  {kind:15s}: {n:4d} tokens, compression={compression:.2f}")
```

**结果**：
- 自然语言：GPT-2 约 4 char/token。
- Python 代码：CodeLlama 约 6 char/token（比自然语言高 50%）。
- 缩进敏感：Python 缩进常被切成多个 token。

**优化方向**：训练**专用 tokenizer**，识别缩进、关键字、操作符的特殊模式。

## 四、代表模型

### Codex（OpenAI, 2021）

第一个引发广泛关注的代码 LLM——GitHub Copilot 的核心：

| 模型 | 规模 | 训练数据 |
|---|---|---|
| code-cushman-001 | 12B | The Stack 早期版 + 微调 |
| code-davinci-002 | 175B（推测） | 大量 GitHub 代码 |

### Code Llama（Meta, 2023）

基于 LLaMA-2 微调的代码专用模型：

```text
Code Llama (基础)        → Python / 多种语言
Code Llama Python        → 仅 Python 微调
Code Llama Instruct      → 对话版（受指令微调）
Code Llama 34B           → 最大版本
```

### DeepSeek-Coder（DeepSeek, 2024）

**在 HumanEval 上与 GPT-3.5 相当**，开源：

| 模型 | HumanEval pass@1 |
|---|---|
| DeepSeek-Coder 33B | 56.1% |
| DeepSeek-Coder 6.7B | 49.4% |
| CodeLlama-34B | 48.2% |
| GPT-3.5 | 48.1% |

关键训练策略：
- **2T token 训练**（87% 代码 + 13% 自然语言）。
- **Fill-in-the-Middle（FIM）训练**：让模型能从前后文补全中间。
- **Repo-level 训练**：把整个仓库作为输入，理解跨文件依赖。

### Qwen-Coder（阿里, 2024）

通义千问代码版，支持 92 种语言，长上下文（128K）。

## 五、FIM：Fill-in-the-Middle

代码 LLM 的特殊训练目标——**填中间**：

```python
# 普通 next-token 训练
input: "def add(a, b):\n    return"
output: " a + b"

# FIM 训练
prefix: "def add(a, b):"
suffix: "    # 求和函数\nprint(add(1, 2))"
output: "\n    return a + b\n"
```

实现：

```python
def fim_loss(prefix, middle, suffix, model):
    """
    FIM 损失：给定 prefix + suffix，预测 middle。
    """
    # 用特殊 token 标记边界
    fim_input = f"<PRE>{prefix}<SUF>{suffix}<MID>"
    target = f"{middle}<EOM>"
    
    # ... (类似 CLM 损失)
```

DeepSeek-Coder 的实验显示 FIM 让 HumanEval 提升 **3~5%**。

## 六、评测基准

### HumanEval（OpenAI, 2021）

164 个手写编程题，每题有单元测试：

```python
def humaneval_prompt():
    return '''def add(a: int, b: int) -> int:
    """Add two integers.
    
    >>> add(1, 2)
    3
    >>> add(-1, 1)
    0
    """
'''

# 模型需要补全函数体
# 然后跑 assert add(1, 2) == 3 等测试
```

**指标**：
- **pass@1**：一次生成就通过的概率。
- **pass@K**：K 次生成中至少一次通过。

### MBPP（Mostly Basic Python Problems）

974 个 Python 题，更基础、覆盖更广。

### LiveCodeBench

**持续更新的评测**——避免训练集污染：

```python
# LiveCodeBench 收集 LeetCode / Codeforces 比赛题目
# 数据带时间戳，确保模型未见过
```

### SWE-bench（更难的评测）

**真实 GitHub issue**：给模型 issue 描述 + 仓库，模型修改代码修复：

```python
{
    "issue": "TypeError when calling save() with a custom encoder",
    "repo": "django/django",
    "commit_before": "abc123",
    "commit_after": "def456",
    "test_patch": "...",   # 用于验证
}
```

GPT-4 在 SWE-bench 上仅 **1.7% pass@1**——这是个**极难**的任务。

### RepoBench

评测模型在**长上下文、跨文件**场景下的代码能力。

## 七、推理优化：补全延迟

代码补全对延迟**极敏感**——用户期望 < 200ms。

### 关键技术

#### 1. Speculative Decoding

小模型先生成 K 个 token，大模型并行验证：

```python
def speculative_decoding(draft_model, target_model, prompt, k=4):
    """小模型生成 k 个 token，大模型并行验证。"""
    draft_tokens = draft_model.generate(prompt, max_new_tokens=k)
    target_logits = target_model(prompt + draft_tokens)
    
    # 接受 draft 中"前 N 个"与 target 一致的 token
    accepted = 0
    for i in range(k):
        if accept_token(target_logits[i], draft_tokens[i]):
            accepted += 1
        else:
            break
    
    return draft_tokens[:accepted]  # + 从 i 重新采样
```

**加速**：通常 **2~3x**。

#### 2. KV-Cache 复用

代码补全是**连续上下文**——用户每输入一个字符，前缀都没变。复用 KV cache：

```python
# 维护一个 KV cache，按用户输入动态追加
class CodeCompletionService:
    def __init__(self, model):
        self.kv_cache = None
        self.last_prompt = ""
    
    def complete(self, prompt):
        # 找出 prompt 与 last_prompt 的公共前缀
        common = common_prefix(prompt, self.last_prompt)
        
        # 复用 KV cache 的前缀部分
        self.kv_cache = self.kv_cache[:common_len]
        
        # 仅计算新增部分的 attention
        new_logits = model.forward(prompt[common_len:], kv_cache=self.kv_cache)
        self.kv_cache = new_logits.kv_cache
        self.last_prompt = prompt
        return new_logits.next_token
```

**加速**：连续补全时 **10~50x**。

#### 3. 增量编译

把模型编译为 **TensorRT / ONNX**：

```bash
# 用 HuggingFace Optimum 编译
optimum-cli export onnx --model deepseek-ai/deepseek-coder-6.7b ./model_onnx/
```

#### 4. 量化

INT4 / INT8 量化减少推理时间和显存：

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b",
    quantization_config=quant_config,
)
```

## 八、安全与许可问题

代码 LLM 面临独特的法律风险：

### 1. 代码许可证"传染"

如果模型生成 GPL 代码用于商业产品，整段代码可能"传染" GPL——这是法律灰区。

**缓解**：
- 训练数据过滤（BigCode 的 The Stack 仅含许可代码）。
- 输出过滤器检测明显复用。
- 用户声明"AI 生成的代码"以澄清来源。

### 2. 训练数据中的个人数据

GitHub commit 包含开发者邮箱、SSH key——这些都是**个人数据**。

**缓解**：
- 训练前去 PII。
- 提供 "do not train" 工具（GitHub 的 `copilot_exclusion_file`）。

### 3. 不安全代码生成

模型可能生成有 SQL 注入、XSS、路径遍历漏洞的代码——攻击者主动 prompt 也能诱导。

**缓解**：
- 安全分类器扫描输出。
- 提示用户"review the generated code carefully"。

## 九、应用模式

### 1. 行内补全（Inline Completion）

Copilot 风格——光标后自动补全一行或一段：

```python
# 用户输入
def fibonacci(n):
    # 光标在这里
# 模型补全
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**要求**：低延迟（< 200ms），与 IDE 深度集成。

### 2. Chat 模式（Cursor、Claude Code）

多轮对话修改整个文件/项目：

```text
User: "重构这个类用 dataclass"
[选中代码]
Assistant: [生成新版本]
User: "加上 type hints"
Assistant: [再次修改]
```

### 3. Agent 模式（Code agents）

模型自主完成多步任务——读代码、修改、运行测试、修复 bug：

```text
User: "修复 issue #1234"
Agent:
  1. 阅读 issue 描述
  2. 找到相关文件
  3. 分析 bug
  4. 修改代码
  5. 运行测试
  6. 如失败则迭代
  7. 创建 PR
```

代表工作：Devin、SWE-Agent、OpenHands——但**成功率仍很低**（SWE-bench 20%）。

## 十、未来方向

1. **更强的 Agent**：多步推理 + 工具使用 + 长上下文。
2. **个性化补全**：根据用户历史代码风格、命名习惯定制。
3. **测试驱动**：模型自动生成测试用例验证代码。
4. **多模态**：截图 → 代码、UI 设计 → 前端代码。
5. **形式化验证**：模型生成的代码自动用 SMT 验证器检查正确性。

## 小结

代码 LLM 是 LLM 应用最成熟的赛道——**训练数据丰富、目标明确、可自动验证**。代表模型从 Codex 到 DeepSeek-Coder 在 HumanEval 上接近 60%，SWE-bench 等更难任务仍是个挑战。**工程难点在补全延迟**（speculative decoding + KV cache 复用）和**安全合规**（许可证、PII、不安全代码）。下一篇我们将看到代码 LLM 的"放大版"——**代码 Agent**：让 LLM 不只是补全，而是自主完成多步编程任务。
