# LLM 教学法：让 AI 扮演教师与学生

LLM 不只是"答题工具"——它可以**扮演教师**（苏格拉底式引导、个性化反馈）、**扮演学生**（模拟错误、让真人教师练习）、**扮演助教**（批改作业、出题）。**LLM Pedagogy** 是研究"LLM 如何与教学法结合"的新兴领域——把 AI 与**认知科学、学习科学、教育心理学**的核心理论对齐，让 AI 不只是"聪明"而是"善于教学"。本文梳理 LLM 在教学中的多种角色、关键设计原则、代表案例，以及与学习科学理论的结合。

## 一、LLM 在教育中的角色矩阵

LLM 可以扮演的角色远不止"辅导老师"：

| 角色 | 描述 | 典型场景 |
|---|---|---|
| **教师（Tutor）** | 一对一辅导 | Khanmigo |
| **助教（TA）** | 答疑、批改 | MagicSchool |
| **模拟学生（SimStudent）** | 模拟错误让真人老师练习 | TeachBook |
| **同伴（Peer）** | 与学生一起讨论 | CS50 Bot |
| **辩论对手** | 训练批判性思维 | ArgumentBot |
| **作业批改员** | 自动评分、反馈 | Gradescope AI |
| **出题者** | 生成练习题 | QuestGen |
| **学习伙伴** | 长期学习陪伴 | Replika Edu |

每种角色对应不同的 prompt 设计、知识库、交互模式。

## 二、LLM 扮演教师

### 1. 苏格拉底式教学（再探）

苏格拉底式教学的核心是**"不直接给答案"**——通过问题引导学生思考：

```python
SOCRATIC_TUTOR_PROMPT = """你扮演苏格拉底式辅导老师。

角色设定：
- 永远不直接给答案
- 通过问题引导思考
- 把大问题拆成小问题
- 学生卡壳时给最小提示
- 鼓励学生表达自己的想法
- 评价时不评判对错，而是分析思路

教学步骤：
1. 听学生的问题
2. 让学生先说自己的想法
3. 顺着学生思路追问
4. 让学生意识到自己思路的漏洞
5. 学生自己修正

学生问题: {question}

你的回复（不超过 50 字）:"""
```

实际案例：

```text
学生: "为什么 π 是无理数？"

❌ 错例 (直接给答案):
"π 是无理数因为它不能表示为两个整数的比..."

✅ 正例 (苏格拉底式):
"好问题。我们先想想：什么样的数叫'有理数'？"
学生: "能写成分数的"
"对。那 π 能写成什么分数吗？为什么难？"
```

### 2. 元认知提示

元认知（metacognition）——"对思考的思考"——是高效学习的关键：

```python
METACOGNITIVE_PROMPTS = [
    "在开始前，你能先用自己的话解释这道题要做什么吗？",
    "你打算用什么方法？为什么选这个方法？",
    "做完之后，你检查了吗？怎么检查的？",
    "如果让你重新做一遍，你会做哪些不同的？",
    "这道题和上次的某道题有什么相似？",
]
```

研究表明：明确**教元认知**的学生比单纯做题的学生学习效果好 30%+。

### 3. 错误作为学习机会

优秀教师不是惩罚错误，而是**把错误变成学习机会**——LLM 也可以这样 prompt：

```python
def handle_student_error(question, wrong_answer, correct_answer):
    """让学生从错误中学习。"""
    return f"""学生答错了：{wrong_answer}
正确答案: {correct_answer}

请按以下方式回应：
1. 不直接说"错"
2. 让学生解释自己的思路
3. 在学生思路中找到合理部分
4. 引导发现思路中的小漏洞
5. 学生自己修正
"""
```

### 4. 自适应难度

ITS 必须根据学生表现**动态调整难度**——太简单会无聊，太难会挫败：

```python
def adjust_difficulty(student_state):
    """根据学生最近表现调整下一题难度。"""
    recent_correct_rate = student_state["recent_correct_rate"]
    
    if recent_correct_rate > 0.85:
        return "更难的题目"
    elif recent_correct_rate > 0.6:
        return "难度相当的题目"
    elif recent_correct_rate > 0.4:
        return "稍简单的题目（巩固基础）"
    else:
        return "回到基础概念，先补这块"
```

**原理**：维持**"心流区间"**（Csikszentmihalyi 的 Flow Theory）——挑战略高于能力 5~10%。

## 三、LLM 扮演学生

### 1. 模拟学生（SimStudent）

让 LLM 扮演有特定知识状态的学生，**让真人老师练习**——这是教师培训的重要应用：

```python
SIMSTUDENT_PROMPT = """你现在扮演一个 7 年级学生，名叫小明。
你的状态：
- 已掌握：分数加减法、整数乘除
- 未掌握：一元一次方程、长方形面积
- 性格：内向、缺乏自信、遇到难题会沉默

规则：
- 回答问题时，按你的知识状态给答案
- 不知道时，说"我不知道"或沉默
- 老师引导时，如果想通了就说"哦！我明白了"
- 如果老师讲得太快，说"能再说一遍吗？"
- 不要主动跳出角色

老师现在说: {teacher_input}

小明的回复:"""
```

实际案例：

```text
教师: "小明，一元一次方程怎么解？"
小明: "...（沉默）"
教师: "比如 x + 3 = 7，x 是多少？"
小明: "... 4？"  ← 不确定
教师: "对！那 2x + 5 = 11 呢？"
小明: "先把 5 移到右边... 2x = 6... x = 3？"
教师: "很好！你是怎么想到的？"
小明: "哦，我看出来的... 把 5 移过去，x 前面那个数变 1"
教师: "嗯，这叫'移项'。如果我们有 3x + 1 = 10 呢？"
小明: "... 不知道"
```

真人教师通过这种模拟获得**低风险练习机会**——可在不影响真学生的情况下反复练习。

### 2. 错误模式生成

LLM 模拟**有特定错误模式**的学生：

```python
class CommonErrors:
    """学生常见错误模式。"""
    NEGATIVE_SIGN = "忽略负号"
    CARRIED_DIGIT = "进位错误"
    WORD_PROBLEM_DECODE = "读不懂应用题"
    CONCEPT_CONFUSION = "混淆相似概念（如'面积'和'周长'）"
    PLACE_VALUE = "位值理解错"


def simulate_student_with_error(question, error_type: str):
    """模拟有特定错误的学生回答。"""
    error_prompt = f"""扮演一个会犯 '{error_type}' 错误的 8 年级学生。
回答以下数学题，故意体现这个错误：

题目: {question}

学生答案（不要解释，体现错误）:"""
    return llm.generate(error_prompt)
```

用途：教师**练习识别**学生错误模式。

### 3. 多学生协作模拟

模拟**多个学生同时讨论**：

```text
Teacher: "今天我们讨论什么是公平"

Student_A: "公平就是每个人都一样"
Student_B: "我觉得公平是按需分配，比如残疾人需要更多"
Student_C: "但如果每个人都一样，会不会对能力强的人不公平？"
Student_D: "同意 B，公平不等于平等"

Teacher: "很好，你们提到了'公平'和'平等'的差别。让我再追问..."
```

LLM 模拟多个不同立场的学生，让真人教师**练习管理课堂讨论**。

## 四、LLM 扮演助教

### 1. 自动批改

```python
def grade_essay(essay: str, rubric: dict) -> dict:
    """用 LLM 评分作文。"""
    prompt = f"""根据以下 rubric 评分作文。

Rubric:
{rubric}

作文:
{essay}

评分维度:
1. 论点 (1-5)
2. 论据 (1-5)
3. 结构 (1-5)
4. 文采 (1-5)
5. 总评 (1-5)

输出 JSON 格式评分，并给出每个维度的具体反馈。"""
    return llm.generate(prompt, response_format={"type": "json_object"})
```

**关键**：rubric 越具体，评分越一致。

### 2. 自动反馈

```python
def give_constructive_feedback(assignment, submission, rubric):
    """给学生建设性反馈。"""
    return llm.generate(f"""你是一位细心、有建设性的老师。给学生以下作业反馈。

作业要求: {assignment}
学生提交: {submission}
评分标准: {rubric}

反馈应该：
1. 先肯定学生做得好的部分
2. 具体指出可改进的地方（引用具体段落）
3. 给出可操作的改进建议
4. 鼓励学生继续努力

反馈:""")
```

研究显示：**反馈越具体、越早**，学习效果越好。

### 3. 自动出题

```python
def generate_practice_questions(topic, difficulty, n=5):
    """基于知识点和难度出题。"""
    return llm.generate(f"""为主题 '{topic}' 出 {n} 道题，难度 '{difficulty}'。
每题包括：
- 题干
- 4 个选项（如果是选择题）
- 正确答案
- 详细解析

输出 JSON。""")
```

**优点**：随时出题，难度可控，覆盖面广。

## 五、与学习科学理论的结合

### 1. Bloom 分类学

Bloom 把认知目标从低到高分为：

```
记忆 → 理解 → 应用 → 分析 → 评价 → 创造
```

好的 ITS 应该**覆盖所有层级**：

```python
BLOOM_PROMPTS = {
    "记忆": "请列出 X 的三个特征",
    "理解": "请用自己的话解释 X",
    "应用": "用 X 解决这个具体问题",
    "分析": "X 和 Y 有什么异同？",
    "评价": "你认为 X 方案是否合理？为什么？",
    "创造": "如果让你重新设计 X，你会怎么做？",
}
```

LLM Tutor 应该引导学生在不同层级思考——**不能只停留在记忆和应用**。

### 2. Spaced Repetition（间隔重复）

**Ebbinghaus 遗忘曲线**——已学知识如果不复习，会指数遗忘：

```
记忆强度
  ^
  |\
  | \
  |  \_____
  |        \____
  |             \____
  +--------------------> 时间
     1天   7天   30天
```

Spaced repetition 算法（如 Anki 用的 SM-2）：

```python
def update_next_review(card, quality):
    """
    quality: 0=完全忘记, 5=完美回答
    """
    if quality < 3:
        card.interval = 1
    else:
        if card.repetitions == 0:
            card.interval = 1
        elif card.repetitions == 1:
            card.interval = 6
        else:
            card.interval = round(card.interval * card.ease_factor)
    
    # 更新 ease factor
    card.ease_factor = max(
        1.3,
        card.ease_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    )
    
    card.repetitions += 1
    return card
```

LLM + spaced repetition = **自适应复习系统**。

### 3. Cognitive Load Theory（认知负荷理论）

Sweller 提出：工作记忆容量有限，**教学应减少不必要的负荷**：

| 负荷类型 | 来源 | 缓解 |
|---|---|---|
| **内在负荷** | 任务本身的复杂度 | 不可消除，但可分块 |
| **外在负荷** | 呈现方式不当 | 优化 UI、清晰表达 |
| **相关负荷** | 学习投入 | 引导思考 |

LLM Tutor 应该：

```python
REDUCE_COGNITIVE_LOAD = """
- 一次只展示一个概念
- 用类比解释抽象概念
- 提供可视化（生成图表）
- 拆分多步骤问题
- 避免术语堆砌
"""
```

### 4. ZPD 与最近发展区（Vygotsky）

ZPD（Zone of Proximal Development）：学生**自己能做到**和**在帮助下能做到**之间的区域——好的辅导应该瞄准这里：

```python
def select_zpd_question(student_ability, all_questions):
    """选 ZPD 内的题目——略高于学生能力。"""
    # ZPD 通常是 ability ± 0.5σ
    zpd_questions = [
        q for q in all_questions
        if abs(q.difficulty - student_ability) < 0.5
    ]
    return zpd_questions
```

LLM Tutor 用 ZPD 选择题目，最大化学习效率。

## 六、代表案例

### 1. Khanmigo（Khan Academy）

**AI 学生 + AI 教师** 双向角色：

- 学生可以**让 Khanmigo 犯错误**，练习当老师。
- 教师可以**让 Khanmigo 模拟学生**，练习识别错误。

### 2. CS50 Bot（Harvard）

哈佛 CS50 课程的 AI 助教：

- 答疑（基于课程讲义 RAG）。
- 鼓励学生，但不直接给代码。
- 数千学生同时在线使用，扩展了 CS50 的覆盖。

### 3. Speak（语言学习）

- 让 LLM 与学生对话。
- 实时纠正发音、语法。
- 模拟不同场景（点餐、面试、闲谈）。

### 4. Photomath / Mathos AI

拍照解数学题——但**不是辅导**，是查答案。

争议：是否削弱学生独立思考？研究显示**不当使用降低学习效果**。

### 5. Quill.org

**写作辅导 AI**——不直接改学生作文，而是通过提问让学生自己修改：

```text
Student: "I think school is good"
Quill: "你为什么这么认为？能给三个理由吗？"
Student: "1. ... 2. ... 3. ..."
Quill: "好！每个理由可以加例子支持吗？"
```

## 七、教学法的核心原则

LLM Tutor 必须遵守的教育心理学原则：

### 1. 主动学习（Active Learning）

```text
❌ 被动: 学生听 LLM 讲
✅ 主动: 学生回答 LLM 的问题，做练习
```

### 2. 即时反馈

```text
❌ 延迟反馈: 作业交一周后反馈
✅ 即时反馈: 学生答完立即得到反馈
```

### 3. 难度梯度

```text
❌ 任意难度
✅ 按 ZPD 调整
```

### 4. 元认知培养

```text
❌ 只关注对错
✅ 引导学生思考"我为什么这么想"
```

### 5. 错误友好

```text
❌ 错误 = 失败
✅ 错误 = 学习机会
```

## 八、伦理与风险

### 1. 学术诚信

学生用 LLM 完成作业——是作弊还是合理使用？

**新规范**：
- **透明声明**：学生必须声明 AI 使用。
- **AI 辅助级别**：哪些允许、哪些不允许。
- **AI 素养**：教学生**如何正确使用**而非禁止。

### 2. 数据隐私

学生对话涉及**未成年人**——必须符合 COPPA / GDPR：

- 端到端加密。
- 数据不用于训练。
- 家长可查看与删除。

### 3. 公平性

- 富裕学校可能有更先进的 AI 教育，**加剧不平等**。
- 不同方言 / 语言的支持质量差异。

### 4. 过度依赖

- 学生可能**丧失独立思考**能力。
- 教师可能**丧失判断力**。

**应对**：
- 教学生**与 AI 协作**而非替代。
- 教师最终审核 AI 反馈。
- 定期做"无 AI" 练习。

## 九、给教育从业者的清单

1. **明确 AI 角色**：是辅导、批改还是模拟学生？
2. **设计 prompt**：用苏格拉底式、错误友好、元认知引导。
3. **学生模型**：追踪每个学生的状态。
4. **教师参与**：教师最终审核 AI 输出。
5. **公平性测试**：用不同方言、性别、背景测试 AI 行为。
6. **透明**：学生必须知道与 AI 互动。
7. **隐私优先**：端到端加密，数据本地化。
8. **持续评估**：A/B 测试不同 prompt / 策略的效果。
9. **AI 素养**：教学生如何正确使用。

## 小结

LLM Pedagogy 把 LLM 角色从"答题工具"扩展为**教师、学生、助教、辩论对手**等多种角色，核心是**用 AI 实现教育心理学的核心理论**——苏格拉底式教学、ZPD、间隔重复、认知负荷理论。元认知培养与错误友好的反馈是 LLM Tutor 设计的核心。**挑战**在于学术诚信、隐私、公平性、过度依赖。**真正成功的 LLM Pedagogy 不只是"用 LLM 替代老师"，而是"用 LLM 让真人老师更强"**。下一篇我们将看到**自适应学习**——把 LLM 与知识追踪、推荐算法结合，实现真正的个性化教育。
