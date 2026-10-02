# Tree of Thoughts：让 LLM 探索多条推理路径

Chain-of-Thought 把推理展开为"单链"。但很多问题（如数独、24 点、战略游戏）的解空间是**树形或图形**——一次错误选择会让后续推理全部失败。Tree of Thoughts（ToT, Yao et al. 2023）让 LLM 在多条推理路径上**分支探索 + 评估 + 回溯**，把推理从"线性"升级为"树形搜索"。本文深入 ToT 的算法、BFS/DFS 实现、与经典 AI 搜索的对比，以及工程实践。

## 一、为什么需要 Tree of Thoughts

### 1.1 CoT 的局限

```text
CoT:  Q → step1 → step2 → step3 → A
                            ↑
                       一步错，步步错

类比：CoT 像下棋只看一条变化；
      ToT 像下棋算多步变化 + 评估每条线。
```

### 1.2 需要树搜索的问题

```text
1. 24 点游戏：给定 4 个数字，用 + - * / 凑出 24
   - 每步选择：选哪两个数、用什么运算符
   - 单链推理很难，回溯是关键

2. 数独：从初始状态找合法解
   - 每步选一个空格填什么
   - 错误选择立即终结路径

3. 创意写作：写一个侦探故事
   - 关键情节分支（凶手是谁？）
   - 中段推理不同分支方向

4. 战略规划：怎么分配 Q1 预算？
   - 多种策略，每种有不同的优劣
```

## 二、ToT 的形式化

ToT 把推理问题形式化为 **搜索树**：

```text
状态（State）   : 当前推理中间结果
动作（Action）  : 从当前状态生成下一步
评估（Eval）    : 当前状态离目标有多远
搜索（Search）  : BFS / DFS / Beam Search
```

```python
class TreeNode:
    def __init__(self, state: str, parent=None):
        self.state = state               # 当前推理内容
        self.parent = parent
        self.children: list[TreeNode] = []
        self.score: float = 0.0          # 评估分数
        self.depth: int = 0
```

### 2.1 ToT 的四步流程

```text
1. Thought Decomposition（思维分解）
   把问题分成几个中间步骤

2. Thought Generation（思维生成）
   对每个状态，生成 k 个候选下一步

3. State Evaluation（状态评估）
   给每个候选打分（用 LLM 自身或专门 verifier）

4. Search Algorithm（搜索算法）
   BFS / DFS / Beam Search 选择下一个状态
```

## 三、BFS 版 ToT：广度优先

### 3.1 算法

```python
def tot_bfs(initial_state: str, llm, max_depth: int = 5, beam_size: int = 3) -> str:
    current_states = [(initial_state, 0)]

    for depth in range(max_depth):
        candidates = []

        for state, _ in current_states:
            # 1. 生成 k 个候选
            thoughts = llm.invoke(
                f"基于以下推理，生成 {beam_size} 个不同下一步：\n{state}",
                n=beam_size,                                 # 采样多个
                temperature=0.7,
            )

            for thought in thoughts:
                new_state = state + "\n" + thought
                # 2. 评估
                score = llm.evaluate(
                    f"评估以下推理路径到目标的距离（0-10，越大越好）：\n{new_state}"
                )
                candidates.append((new_state, score))

        # 3. 取 top-beam_size
        candidates.sort(key=lambda x: -x[1])
        current_states = candidates[:beam_size]

    return current_states[0][0]
```

### 3.2 24 点游戏实例

```python
def game_of_24_tot(numbers: list[int]) -> list[str]:
    """用 ToT 解决 24 点游戏"""
    state = f"数字：{numbers}"

    def evaluate(state: str) -> float:
        """评估当前状态离 24 有多远"""
        # 让 LLM 评估"剩余数字能否凑出 24"
        return float(llm.invoke(
            f"基于：{state}，评估能否用剩余数字凑出 24（0-10）"
        ))

    def generate(state: str, n: int = 3) -> list[str]:
        """生成 n 个下一步"""
        return llm.invoke(
            f"基于：{state}\n生成 {n} 个不同的下一步操作（选两个数、用运算符）：",
            n=n, temperature=0.8,
        )

    # BFS
    current = [(state, 0)]
    for depth in range(3):
        candidates = []
        for s, _ in current:
            for thought in generate(s, 3):
                new_s = s + f"\n{thought}"
                candidates.append((new_s, evaluate(new_s)))
        candidates.sort(key=lambda x: -x[1])
        current = candidates[:3]

    return [s for s, _ in current]
```

Yao et al. 论文报告 ToT 把 24 点的成功率从 CoT 的 4% 提升到 **74%**。

## 四、DFS 版 ToT：深度优先 + 回溯

DFS 更适合需要"一条路走到底"的问题：

```python
def tot_dfs(state: str, llm, depth: int = 0, max_depth: int = 5, threshold: float = 5.0) -> str | None:
    if is_goal(state):
        return state

    if depth >= max_depth:
        return None

    thoughts = generate_thoughts(state, llm)

    for thought in thoughts:
        new_state = state + "\n" + thought
        score = evaluate(new_state, llm)

        if score < threshold:    # 剪枝：差的路径不再深入
            continue

        result = tot_dfs(new_state, llm, depth + 1, max_depth, threshold)
        if result is not None:
            return result

    return None   # 整条路径失败
```

## 五、Beam Search 版 ToT

BFS 内存爆炸，DFS 容易陷入错误分支。**Beam Search** 是平衡：

```python
def tot_beam_search(initial_state, llm, beam_width=5, max_steps=10):
    beams = [(initial_state, 0.0)]   # (state, cumulative_score)

    for step in range(max_steps):
        all_next = []

        for state, score in beams:
            # 生成 k 个候选
            candidates = llm.invoke(
                f"基于当前推理，生成下一步候选：\n{state}",
                n=3, temperature=0.7,
            )
            for c in candidates:
                new_state = state + "\n" + c
                eval_score = llm.invoke(f"评估这条路径（0-10）：\n{new_state}")
                all_next.append((new_state, score + eval_score))

        # 排序 + 截断
        all_next.sort(key=lambda x: -x[1])
        beams = all_next[:beam_width]

    return beams[0][0]
```

## 六、ToT 与经典 AI 搜索的对比

```text
经典搜索（如 A*）：
  状态空间：显式定义的（棋盘状态、地图节点）
  评估函数：手写（曼哈顿距离、剩余目标数）
  动作空间：预定义（上下左右）
  
ToT：
  状态空间：自然语言描述（隐式）
  评估函数：LLM 自己打分（语义级）
  动作空间：LLM 自由生成
```

| 维度 | 经典搜索 | ToT |
|---|---|---|
| 状态表示 | 结构化 | 自然语言 |
| 评估函数 | 启发式 | LLM |
| 适用 | 小状态空间 | 大且模糊 |
| 成本 | 极低 | 高（多次 LLM 调用） |

## 七、ToT 的工程挑战

### 7.1 成本爆炸

```python
# ToT 调用次数：state 数 × 每个 state 的候选数 × 评估次数
# 24 点：depth=3, beam=3 → 27 次生成 + 27 次评估 ≈ 54 次 LLM 调用
```

缓解策略：

```python
def cost_aware_tot(question, max_cost_usd=1.0):
    """按成本限制 ToT"""
    cost_per_call = 0.01
    budget_calls = max_cost_usd / cost_per_call

    # 自适应 beam width
    beam = max(1, int(budget_calls / (depth * 4)))
    ...
```

### 7.2 评估可靠性

LLM 给自己的推理打分往往**过度乐观**：

```python
# 不靠谱
score = llm.invoke(f"这条路径好不好？0-10")
# → 大多给 7-9 分

# 更靠谱：用专门 verifier 或对比 ground truth
score = verifier_model.predict(state, goal)
```

### 7.3 收敛问题

LLM 可能在同一状态生成相同候选（缺乏多样性）：

```python
def diverse_sampling(llm, prompt, n=3):
    """强制多样性采样"""
    candidates = []
    seen = set()
    for attempt in range(n * 3):
        c = llm.invoke(prompt, temperature=1.2)
        if c not in seen:
            candidates.append(c)
            seen.add(c)
        if len(candidates) == n:
            break
    return candidates
```

## 八、ToT 的典型应用

### 8.1 创意写作

```python
tot_story_writing(prompt):
    candidates = []
    for style in ["悬疑", "温情", "黑色幽默"]:
        outline = llm.generate(f"用{style}风格写大纲：{prompt}")
        scored = llm.evaluate(f"评估大纲质量：\n{outline}")
        candidates.append((style, outline, scored))
    return max(candidates)
```

### 8.2 战略规划

```python
def strategic_planning(question):
    """商业战略 ToT"""
    initial = f"问题：{question}"

    # 第一层：3 种可能策略方向
    strategies = llm.generate(f"对 '{question}' 给出 3 种不同战略方向", n=3)

    # 第二层：每种策略的执行细节
    best_strategy = None
    best_score = -1
    for strategy in strategies:
        details = llm.generate(f"为这个战略生成执行细节：{strategy}")
        score = llm.evaluate(f"评估可行性：{details}")
        if score > best_score:
            best_score = score
            best_strategy = (strategy, details)

    return best_strategy
```

### 8.3 代码问题求解

```python
def tot_coding(problem):
    """算法问题求解 ToT"""
    state = problem

    # 探索不同算法思路
    approaches = llm.generate(
        f"问题：{problem}\n给出 3 种不同解法思路",
        n=3,
    )

    # 评估每种思路
    best = max(approaches, key=lambda a: llm.evaluate(f"评估：\n{a}"))

    # 生成完整代码
    return llm.generate(f"基于以下思路写代码：\n{best}")
```

## 九、ToT 的衍生方法

### 9.1 Graph of Thoughts (GoT)

把"树"扩展为"图"，允许子节点共享：

```text
    ┌─→ A1 ─┐
Root ─┤       ├─→ Combined
    └─→ A2 ─┘
```

### 9.2 Reason + Act (ReAct)

CoT + Tool Use 交替，详见 react-pattern.md。

### 9.3 Self-Refine

不靠搜索，靠 LLM 自我批判 + 修订：

```python
def self_refine(llm, draft):
    for round in range(3):
        critique = llm.invoke(f"批评以下回答：\n{draft}")
        refined = llm.invoke(f"基于批评改进：\n{critique}\n原回答：\n{draft}")
        if is_good_enough(refined):
            break
        draft = refined
    return draft
```

### 9.4 RAP (Reasoning + Acting Plan)

ToT + ReAct + Environment 反馈：

```python
def rap_step(llm, env, state):
    """每步：基于状态推理 + 执行 + 观察新状态"""
    action = llm.decide(state)              # ToT 推理
    new_state = env.step(action)             # 执行
    reward = env.evaluate(new_state)         # 反馈
    return new_state, reward
```

## 十、ToT 的局限与适用边界

### 10.1 何时不用 ToT

```text
- 答案唯一且容易（多数问答）→ 直接答
- 强算术任务 → CoT + calculator
- 单步推理（分类）→ Zero-Shot
- 高成本敏感场景 → 不用
```

### 10.2 何时用 ToT

```text
- 解空间巨大、需要回溯
- 多个可能路径，需要比较
- 创意 / 战略 / 开放式问题
- 单次 CoT 准确率低
```

### 10.3 计算开销

```text
任务：算 24 点
CoT：1 次 LLM 调用
ToT (depth=3, beam=3)：54 次调用
Self-Consistency：5 次调用

ToT 比 CoT 贵 50×，但准确率从 4% → 74%
```

## 小结

ToT 把 LLM 推理从"单链"扩展到"搜索树"，用 LLM 自身做候选生成 + 状态评估，配合 BFS/DFS/Beam Search 实现回溯与剪枝。ToT 在 24 点、创意写作、战略规划等需要回溯的任务上效果显著，但成本是 CoT 的几十倍。生产中要按任务特性权衡：简单任务用 CoT，回溯任务用 ToT，组合任务用 Graph of Thoughts。下一篇我们将深入 **react-pattern**——CoT 与 Tool Use 的融合。
