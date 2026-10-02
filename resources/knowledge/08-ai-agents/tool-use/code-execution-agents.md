# Code Execution Agents：让 LLM 自己写代码并执行

把 "执行任意 Python 代码" 这一项暴露给 LLM，是最强大、也是最危险的 Tool Use。Code Interpreter（OpenAI 2023）让模型可以自己写代码、自己运行、自己看 traceback、自己修改——把"写-测-改"循环自动化。本文深入 Code Interpreter 的沙箱设计、典型应用、安全边界，以及生产环境的工程实现。

## 一、为什么 Code Execution 是 Agent 的"核武器"

```text
普通 Tool: 加法、调 API、查询数据库
Code Execution: 任何可计算的事

举例：
  - 复杂数学：算微积分、解微分方程、模拟蒙特卡洛
  - 数据分析：pandas 读 CSV、画图、统计
  - 文件操作：解析 PDF、转码、批量重命名
  - 算法实现：动态规划、图算法
  - 跨工具组合：先爬数据 → 再清洗 → 再可视化
```

OpenAI 的 Code Interpreter 论文报告：

- 准确率比纯 CoT 高 20-40%（数学任务）
- 模型能"自调试"——第一次写错，看到 traceback 后修正

## 二、Code Interpreter 的工作模式

### 2.1 循环结构

```text
For each step:
    1. Thought:   模型分析要做什么
    2. Code:      模型生成 Python 代码
    3. Execute:   沙箱执行代码
    4. Output:    捕获 stdout / stderr / 图
    5. 模型基于 output 决定下一步
```

### 2.2 模型输出格式

```python
# ChatGPT 的 Code Interpreter 内部 prompt
EXEC_SYSTEM_PROMPT = """
You have access to a Python code interpreter.

When you want to execute Python code:
1. Wrap it in ```python ... ``` blocks
2. Wait for the output
3. Use the output to inform your next steps

Available libraries: numpy, pandas, matplotlib, scikit-learn, sympy, ...
"""

# 模型输出
"""
Let me analyze the data first.

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
```

"""
# 沙箱执行后返回
"""
       sales  profit
count  100.0   100.0
mean    50.3    12.1
std     20.1     5.4
min     10.0     1.2
max     95.0    24.5
"""

# 模型继续
"""
The data has 100 rows. Let me visualize the distribution.

```python
import matplotlib.pyplot as plt
df['sales'].hist(bins=20)
plt.title('Sales Distribution')
plt.savefig('hist.png')
```
"""
```

### 2.3 自调试循环

```text
Code: df.head()
Output: AttributeError: 'DataFrame' object has no attribute 'head'
       Did you mean: 'tail', 'head'?
Code (修正): df.head(5)
Output:    sales  profit
       0    52     12
       1    48     11
       ...
```

模型把 traceback 当作反馈信号，自动修正。

## 三、沙箱设计

### 3.1 安全沙箱的需求

```text
- 隔离：不能访问生产数据库、生产文件系统
- 受限资源：CPU、内存、磁盘、网络
- 时间限制：单次执行不能超过 N 秒
- 审计：所有代码和输出要记录
- 状态持久：会话内的变量可以跨多次执行
```

### 3.2 实现方案

| 方案 | 隔离强度 | 性能 | 复杂度 |
|---|---|---|---|
| **subprocess + virtualenv** | 中 | 高 | 低 |
| **Docker 容器** | 高 | 中 | 中 |
| **gVisor / Firecracker** | 极高 | 中 | 高 |
| **Jupyter Kernel** | 中 | 高 | 中 |
| **Pyodide (WebAssembly)** | 极高 | 低 | 中 |
| **云端执行（E2B / Modal）** | 高 | 中 | 低 |

### 3.3 Docker 沙箱示例

```python
import docker
import tempfile

client = docker.from_env()

def execute_code(code: str, timeout: int = 30) -> dict:
    """在 Docker 容器中执行 Python 代码"""
    container = client.containers.run(
        image="python:3.11-slim",
        command=["python", "-c", code],
        mem_limit="512m",
        cpu_quota=50000,            # 50% CPU
        network_mode="none",        # 禁止网络
        remove=True,
        stdout=True,
        stderr=True,
        detach=True,
    )

    try:
        result = container.wait(timeout=timeout)
        stdout = container.logs(stdout=True, stderr=False).decode()
        stderr = container.logs(stdout=False, stderr=True).decode()
        return {"stdout": stdout, "stderr": stderr, "exit_code": result["StatusCode"]}
    except Exception as e:
        container.kill()
        return {"error": f"timeout or error: {e}"}
    finally:
        try:
            container.remove(force=True)
        except:
            pass
```

### 3.4 持久化会话

```python
class PersistentCodeSession:
    """保持会话状态的代码执行（变量跨多次调用）"""
    def __init__(self):
        self.container = client.containers.run(
            image="python:3.11-slim",
            command=["python", "-i", "-u"],      # interactive, unbuffered
            stdin_open=True,
            tty=True,
            mem_limit="1g",
            network_mode="none",
            detach=True,
        )

    def execute(self, code: str) -> str:
        # 通过 stdin 喂代码
        sock = self.container.attach_socket(params={"stdin": 1, "stream": 1})
        sock._sock.send(code.encode() + b"\n")
        return sock._sock.recv(4096).decode()

    def cleanup(self):
        self.container.kill()
        self.container.remove()
```

### 3.5 E2B / Modal 云端沙箱

```python
# E2B (e2b.dev) - 专为 Code Interpreter 设计
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()
execution = sandbox.notebook.exec_cell("x = 5; print(x ** 2)")
print(execution.logs)        # 25

# 状态持久
execution = sandbox.notebook.exec_cell("print(x + 1)")
print(execution.logs)        # 6

sandbox.close()
```

## 四、典型应用模式

### 4.1 数据分析

```python
def analyze_data(user_query: str, file_path: str) -> str:
    prompt = f"""用 Python 分析数据并回答：{user_query}
    数据文件路径：{file_path}
    """
    code = llm.generate_code(prompt)
    result = sandbox.execute(code)
    interpretation = llm.interpret(result, user_query)
    return interpretation
```

### 4.2 数学求解

```python
def solve_math(problem: str) -> str:
    code = llm.generate_code(f"用 sympy 解：{problem}")
    result = sandbox.execute(code)
    return result.stdout
```

### 4.3 算法实现

```python
def implement_algorithm(requirement: str) -> str:
    code = llm.generate_code(f"实现以下算法：{requirement}")
    result = sandbox.execute(code)

    # 让模型看 traceback 自动调试
    while "Error" in result.stderr:
        code = llm.generate_code(f"修复这段代码：\n{code}\n错误：{result.stderr}")
        result = sandbox.execute(code)

    return result.stdout
```

### 4.4 文件处理流水线

```python
def process_files(input_dir: str) -> None:
    """LLM 自动生成文件处理脚本"""
    code = llm.generate_code(f"""
        任务：把 {input_dir} 下所有 PDF 转成 Markdown
        用 PyPDF2 读，pandoc 转
        输出到 {input_dir}/md/
    """)
    sandbox.execute(code)
```

## 五、安全边界

### 5.1 危险 API 列表

```python
BLOCKED_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "ctypes"}
BLOCKED_FUNCTIONS = {"exec", "eval", "compile", "__import__", "open"}

def execute_safe(code: str) -> dict:
    # 1. AST 检查：禁用危险调用
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in BLOCKED_MODULES:
                    return {"error": f"import '{alias.name}' 禁用"}
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_FUNCTIONS:
                return {"error": f"函数 '{node.func.id}' 禁用"}

    # 2. Docker 沙箱执行
    return execute_code(code)
```

### 5.2 网络限制

```python
# 在沙箱内运行一个白名单 HTTP proxy
ALLOWED_DOMAINS = {"api.weather.com", "www.googleapis.com"}

class WhitelistHTTPProxy:
    def handle(self, request):
        host = urlparse(request.url).hostname
        if host not in ALLOWED_DOMAINS:
            return Response(status=403, body="Domain not allowed")
        return self.proxy(request)
```

### 5.3 文件系统隔离

```python
# 沙箱内的文件系统是 ephemeral 的
# 重要数据通过 mount 注入
sandbox = Sandbox(
    mounts={"/data": user_uploaded_files},   # 只读挂载
    tmp_size_limit="100m",
)
```

### 5.4 资源限制

```python
SANDBOX_LIMITS = {
    "cpu_time_seconds": 30,
    "memory_mb": 512,
    "disk_mb": 100,
    "open_files": 64,
    "subprocesses": 0,        # 禁止子进程
}
```

## 六、调试与可观测性

### 6.1 完整执行轨迹

```python
@dataclass
class CodeExecution:
    session_id: str
    step: int
    code: str
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timestamp: datetime
    cost_estimate_usd: float
```

### 6.2 关键指标

```text
- 执行成功率：多少代码一次跑通？
- 平均调试次数：需要几次才能成功？
- 单次执行耗时
- 代码 token 数（成本）
- 危险 API 拦截次数
- 沙箱 OOM 次数
```

### 6.3 失败的常见原因

```text
- 模型生成语法错误 → 自动用 try-except 包
- 调用未安装的库 → 在 prompt 中列出可用库
- 算术错误 → 用 sympy 精确计算
- 死循环 → 加 timeout + 强制 kill
- 内存爆炸 → 用 chunked 处理
```

## 七、生产框架对比

| 框架 | 沙箱 | 易用性 | 适用 |
|---|---|---|---|
| **OpenAI Code Interpreter** | 闭源，OpenAI 沙箱 | 极简 | ChatGPT Plus 用户 |
| **Anthropic Computer Use** | OS 级沙箱 | 中 | Claude 高级用户 |
| **E2B** | 云端 Docker | 高 | 自建生产 |
| **Modal** | 云端函数 + 沙箱 | 高 | 重计算任务 |
| **Jupyter Kernel + Docker** | 自建 | 中 | 内部工具 |
| **LangChain PythonREPL** | 进程内（不安全） | 极简 | PoC |

## 八、LangChain 的实现

```python
from langchain_experimental.tools import PythonREPLTool

python_repl = PythonREPLTool()

# 直接用
result = python_repl.run("print(2 + 2)")

# 与 Agent 集成
from langchain.agents import create_openai_functions_agent

tools = [python_repl, search_tool, calculator_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

executor.invoke({"input": "统计 data.csv 的销售额均值"})
```

注意：`PythonREPLTool` 是进程内执行，**仅用于本地开发**。生产必须用沙箱。

## 九、未来方向

1. **LLM 自主写测试**：先写 unit test，再写实现，最后跑测试。
2. **多语言执行**：除了 Python，还支持 JavaScript、SQL、Bash。
3. **持久状态 + 版本管理**：类似 git 的代码版本化，Agent 可以回滚。
4. **协作沙箱**：多 Agent 共享一个代码空间。
5. **离线代码执行**：本地 Docker 沙箱，避免数据出域。

## 小结

Code Execution 是 Agent 工具箱里最强大、最危险的工具。它把"任何可计算的事"都暴露给 LLM，把"写-跑-改"循环自动化。生产环境必须用**严格沙箱**（Docker / gVisor / E2B），禁止危险 API，限制资源，记录所有执行。OpenAI Code Interpreter 证明了这条路可行，但把沙箱交给第三方也意味着数据出境——内部部署 + 严格隔离是企业的首选。
