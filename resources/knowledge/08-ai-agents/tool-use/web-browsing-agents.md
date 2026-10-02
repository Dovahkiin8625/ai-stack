# Web Browsing Agents：让 LLM 自主浏览与操作网页

把"浏览网页"作为工具暴露给 LLM，Agent 就能自主订机票、查资料、填表单、抓数据——但也面临反爬、JavaScript 渲染、CAPTCHA、登录态等真实世界挑战。本文深入 Web Agent 的能力分层、主流实现（WebArena / Mind2Web / Anthropic Computer Use）、浏览器自动化底层（Playwright / Selenium）、RAG 与 Web Agent 的差异，以及生产环境的工程实践。

## 一、Web Agent 能力分层

```text
Level 1: 阅读    - GET 页面 → 提取文本 → LLM 读懂
Level 2: 搜索    - 输入关键词 → 看结果 → 点击链接
Level 3: 表单    - 填字段 → 选下拉 → 提交
Level 4: 登录    - 输入凭证 → 处理 2FA
Level 5: 多步流程 - 注册账号 → 验证邮箱 → 设置偏好
Level 6: 反爬应对 - 切换 IP → 解 CAPTCHA → 模拟人类
```

Web Agent 的"智能"主要体现在 Level 3 以上。

## 二、RAG vs Web Agent

```text
RAG：知识检索增强
  - 静态知识库（预先抓取）
  - 速度快、成本低
  - 信息可能过时
  
Web Agent：动态信息获取
  - 实时抓取（每次都跑）
  - 速度慢、成本高
  - 信息永远最新
```

何时用 Web Agent：

- 实时数据（股票、天气、新闻）
- 复杂交互（订机票、改行程）
- 用户特定操作（登录、查私人邮件）

## 三、浏览器自动化底层

### 3.1 Playwright（最推荐）

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # 导航
    page.goto("https://example.com")

    # 提取内容
    title = page.title()
    body = page.locator("body").inner_text()

    # 交互
    page.fill("#search-box", "machine learning")
    page.click(".search-button")
    page.wait_for_selector(".results")

    results = page.locator(".result-item").all_text_contents()

    browser.close()
```

### 3.2 关键操作

```python
class WebActions:
    """Web Agent 的原子操作集"""

    def goto(self, url: str):
        self.page.goto(url, wait_until="networkidle")

    def click(self, selector: str):
        self.page.click(selector)

    def type_text(self, selector: str, text: str):
        self.page.fill(selector, text)

    def extract_text(self, selector: str) -> str:
        return self.page.locator(selector).inner_text()

    def extract_html(self, selector: str) -> str:
        return self.page.locator(selector).inner_html()

    def screenshot(self) -> bytes:
        return self.page.screenshot(full_page=True)

    def get_url(self) -> str:
        return self.page.url

    def scroll(self, direction: str = "down", amount: int = 500):
        delta = amount if direction == "down" else -amount
        self.page.mouse.wheel(0, delta)

    def wait(self, ms: int = 1000):
        self.page.wait_for_timeout(ms)
```

### 3.3 LLM 决定下一步

```python
class WebAgent:
    def __init__(self, llm, browser: WebActions):
        self.llm = llm
        self.browser = browser
        self.history = []

    def step(self, task: str):
        # 1. 截屏 + DOM 摘要
        screenshot = self.browser.screenshot()
        dom_summary = self.extract_accessible_dom()

        # 2. 让 LLM 决定下一步
        response = self.llm.invoke(
            SYSTEM_PROMPT + f"""
            任务：{task}
            历史：{self.history[-5:]}
            当前页面 URL：{self.browser.get_url()}
            当前页面摘要：{dom_summary}
            
            决定下一步操作。返回 JSON：
            {{"action": "click"|"type"|"goto"|"extract"|"screenshot"|"done", "target": ..., "value": ...}}
            """
        )

        action = json.loads(response)

        # 3. 执行
        if action["action"] == "click":
            self.browser.click(action["target"])
        elif action["action"] == "type":
            self.browser.type_text(action["target"], action["value"])
        # ...

        self.history.append(action)
        return action
```

## 四、多模态 Web Agent

### 4.1 截图 + 视觉理解

现代 Agent 用 GPT-4V / Claude with vision 看截图：

```python
def decide_with_vision(task: str, page) -> dict:
    screenshot = page.screenshot()
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"任务：{task}\n请基于截图决定下一步操作。返回 JSON ..."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(screenshot).decode()}"}},
                ],
            }
        ],
    )
    return json.loads(response.choices[0].message.content)
```

Anthropic 的 Computer Use 直接基于截图 + 鼠标键盘坐标操作：

```python
# Claude Computer Use 示例
response = client.beta.messages.create(
    model="claude-3-5-sonnet-20241022",
    tools=[{"type": "computer_20241022", "name": "computer"}],
    messages=[{"role": "user", "content": "请帮我订明天去北京的机票"}],
)
# 返回: {"action": "left_click", "coordinate": [123, 456]}
```

### 4.2 Accessibility Tree

DOM 的 accessibility tree 是更结构化的页面表示：

```python
def get_accessibility_tree(page) -> str:
    """获取页面的 accessibility snapshot"""
    return page.accessibility.snapshot()
```

输出示例：

```json
{
  "role": "WebArea",
  "name": "Example Page",
  "children": [
    {
      "role": "textbox",
      "name": "Search",
      "selector": "#search",
      "value": ""
    },
    {
      "role": "button",
      "name": "Submit",
      "selector": "#submit-btn"
    }
  ]
}
```

LLM 基于这个结构化树决定下一步，比直接看 HTML 干净得多。

## 五、典型应用：订机票 Agent

```python
class FlightBookingAgent:
    def __init__(self, llm):
        self.llm = llm
        self.browser = WebActions()
        self.task = ""

    def run(self, task: str):
        self.task = task

        # 1. 启动浏览器
        self.browser.goto("https://flights.example.com")

        for step in range(20):
            # 2. 看到当前状态，决定动作
            action = self.decide_action()
            self.browser.execute(action)

            # 3. 检查是否完成
            if action["action"] == "done":
                return action["result"]

    def decide_action(self):
        snapshot = self.browser.get_accessibility_tree()
        prompt = f"""
        任务：{self.task}
        当前页面：
        {snapshot}
        
        请输出下一步操作（JSON）。
        """
        response = self.llm.invoke(prompt)
        return json.loads(response)
```

执行流程：

```text
Step 1: 看到首页 → click "搜索航班"
Step 2: 看到表单 → type "PEK" 到出发地
Step 3: type "SHA" 到目的地
Step 4: type "2026-10-03" 到日期
Step 5: click "搜索"
Step 6: 看到结果列表 → extract 航班信息
Step 7: click "选择"
Step 8: 看到乘客表单 → type "Alice"
Step 9: click "确认"
Step 10: 完成 → return booking confirmation
```

## 六、关键挑战

### 6.1 反爬虫机制

```text
- IP 速率限制
- User-Agent 检测
- Cookie / Session 跟踪
- CAPTCHA（reCAPTCHA、Cloudflare）
- 行为分析（鼠标轨迹、停留时间）
```

对策：

```python
class StealthBrowser:
    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        self.context = self.browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ...",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )

    def human_like_click(self, selector: str):
        """模拟人类点击（带随机延迟、贝塞尔曲线鼠标移动）"""
        box = self.page.locator(selector).bounding_box()
        start = self.page.mouse.position
        end = (box["x"] + box["width"]/2, box["y"] + box["height"]/2)
        path = bezier_curve(start, end)
        for x, y in path:
            self.page.mouse.move(x, y)
            time.sleep(random.uniform(0.005, 0.015))
        self.page.mouse.click(*end)
```

### 6.2 登录与 Session

```python
class AuthenticatedBrowser:
    def __init__(self, credentials: dict):
        self.browser = launch_browser()
        self.context = self.browser.new_context(storage_state="cookies.json")
        self.page = self.context.new_page()

    def login_if_needed(self):
        if "login" in self.page.url:
            self.page.fill("#email", self.credentials["email"])
            self.page.fill("#password", self.credentials["password"])
            self.page.click("#submit")
            self.context.storage_state(path="cookies.json")
```

### 6.3 JavaScript 渲染

许多网站内容是 JS 动态加载的：

```python
def wait_for_content(page, selector: str, timeout: int = 10000):
    """等待动态内容加载"""
    page.wait_for_selector(selector, timeout=timeout)
    page.wait_for_load_state("networkidle")
```

### 6.4 CAPTCHAs

```text
- 简单 CAPTCHA：调用打码平台（2Captcha、Anti-Captcha）
- reCAPTCHA：极度困难，几乎只能靠人工
- Cloudflare Turnstile：用 undetected-browser 库
```

## 七、生产框架对比

| 框架 | 浏览器引擎 | 多模态 | 适用 |
|---|---|---|---|
| **Anthropic Computer Use** | 远程 OS 桌面 | ✓ | 通用桌面操作 |
| **OpenAI Operator** | 远程浏览器 | ✓ | 网页任务（2025） |
| **WebArena** | 自建网站 | ✗ | 研究基准 |
| **Mind2Web** | 任意网站 | 部分 | 学术研究 |
| **Selenium** | 真实浏览器 | ✗ | 传统自动化 |
| **Playwright + 自研 Agent** | 真实浏览器 | 自接 | 生产自建 |
| **Skyvern** | 真实浏览器 + LLM | ✓ | 企业 RPA |

## 八、WebArena 基准

CMU 2023 发布的 Web Agent 综合基准：

```text
真实网站：
  - 购物：Amazon / eBay
  - 论坛：Reddit
  - 软件开发：GitLab
  - 地图：OpenStreetMap
  - 内容管理：CMS

任务：912 个，覆盖搜索、购物、社交、地图等
评估：任务完成率
SOTA (2024)：~35%
```

Anthropic 的 Computer Use 在 WebArena 公开评估上达到 ~35%，与人手操作仍有显著差距。

## 九、生产实践要点

### 9.1 任务白名单

```python
ALLOWED_DOMAINS = ["company-internal-tools.com"]
BLOCKED_ACTIONS = ["submit_payment", "send_email", "delete_account"]

def step(self, action):
    if action["url"] not in ALLOWED_DOMAINS:
        raise SecurityError("Out of scope domain")
    if action["type"] in BLOCKED_ACTIONS:
        raise SecurityError("Dangerous action")
```

### 9.2 重试与恢复

```python
def robust_step(self, action, max_retries=3):
    for attempt in range(max_retries):
        try:
            self.execute(action)
            return
        except TimeoutError:
            self.page.reload()                       # 简单恢复
        except ElementNotFound:
            self.wait_for_element_or_give_up()

    self.escalate_to_human(f"Action failed after {max_retries} tries")
```

### 9.3 审计与可观测

```python
# 每个 Web Agent 步骤记录到审计日志
audit_log.info({
    "agent_id": self.id,
    "task": self.task,
    "step": self.step_count,
    "url": self.browser.get_url(),
    "action": action,
    "screenshot_hash": hash(self.browser.screenshot()),
    "duration_ms": elapsed,
    "success": True,
})
```

### 9.4 人类兜底

```python
def step_with_human_fallback(self, action):
    if action["type"] in ["submit_payment", "confirm_booking"]:
        # 关键动作：让用户确认
        screenshot_b64 = self.browser.screenshot_base64()
        approved = ask_user(
            f"Agent 想执行：{action}\n请确认（截图已显示）",
            screenshot=screenshot_b64,
        )
        if not approved:
            return self.replan()
    self.execute(action)
```

## 十、未来方向

1. **Computer Use 普及**：Anthropic、OpenAI 都把"操作系统级 Agent" 作为前沿方向。
2. **专用模型**：训练专门做 Web 操作的小模型（更便宜、更快）。
3. **企业 RPA 替代**：传统 RPA（UiPath、Blue Prism）正被 LLM Agent 重塑。
4. **反爬与爬的军备竞赛**：Web Agent 与反爬系统的对抗将持续升级。
5. **多模态融合**：屏幕截图 + DOM + 用户意图理解。

## 小结

Web Browsing Agent 把 LLM 从"对话系统"升级为"在线操作者"。Playwright / Selenium 提供浏览器自动化基础，多模态 LLM（GPT-4V、Claude）让 Agent 能"看懂"页面，ReAct 模式让 Agent 能"决策"下一步动作。生产环境的核心挑战是**反爬应对、登录态管理、安全边界、人类兜底**。Anthropic 的 Computer Use 与 OpenAI 的 Operator 标志着 Web Agent 走向主流，但距离"完全自主"仍有 30-50 个百分点的任务成功率差距。
