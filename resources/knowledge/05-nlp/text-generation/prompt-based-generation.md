# 提示驱动生成：In-Context Learning、Chain-of-Thought 与 ReAct

大语言模型时代的 NLP 应用范式从"训练专用模型"转向"提示驱动生成"（Prompt-based Generation）。本文系统梳理 In-Context Learning、Chain-of-Thought、ReAct、Self-Consistency 等核心技术，深入剖析它们的数学原理、实践经验与局限。

## 一、提示驱动范式的崛起

### 1.1 范式转变

NLP 任务的解决方式在过去 8 年里发生了三次范式跃迁：

| 范式 | 时间 | 核心 | 数据需求 | 代表 |
| --- | --- | --- | --- | --- |
| 监督学习 | 2014-2018 | 任务专用模型 + 标注数据 | 10K-1M 标注 | BiLSTM-CRF、BERT 微调 |
| 预训练 + 微调 | 2018-2022 | 通用预训练 + 任务微调 | 1K-100K 标注 | BERT、GPT-2、T5 |
| 提示驱动 | 2020- | 通用大模型 + Prompt 工程 | 0-100 示例 | GPT-3、ChatGPT、Claude |

提示驱动的核心洞察：**大规模预训练让 LLM 学到了"按指令执行任务"的能力**，下游任务不再需要训练专用模型，只需构造恰当的 prompt。

### 1.2 优势

- **零样本 / 少样本能力**：用 prompt 即可，无需训练。
- **灵活性**：同一个模型应对无数任务。
- **迭代快**：改 prompt 即可调整行为，无需重训。

### 1.3 局限

- **不稳定**：同样的 prompt 不同次生成结果可能差异大。
- **成本高**：每次推理都调用大模型，比专用模型贵。
- **可控性**：受限于模型的"理解能力"。

## 二、In-Context Learning（ICL）

### 2.1 核心思想

Brown et al.（2020）的 GPT-3 论文首次系统提出 ICL：在 prompt 中给出几个"输入-输出"示例，让模型"看例题"完成任务：

```
提示：
英文: cat → 中文: 猫
英文: dog → 中文: 狗
英文: bird → 中文: 鸟
英文: horse → 中文: ?
```

模型基于"看到模式"自动补全答案，无需任何参数更新。

### 2.2 三种设定

| 设定 | 示例数 | 性能 |
| --- | --- | --- |
| Zero-Shot | 0 | 弱 |
| One-Shot | 1 | 中 |
| Few-Shot | 5-100 | 强 |

GPT-3 在 SuperGLUE 上：

- Zero-shot：~50% 准确率。
- Few-shot (32)：~65% 准确率。

### 2.3 ICL 的理论解释

ICL 的工作机制至今是研究热点。主流理论：

#### (a) 隐式贝叶斯推断

ICL 是模型在"推断 latent concept"——给定几个 $(x_i, y_i)$ 对，模型推断"什么样的函数 $f$ 满足 $f(x_i) = y_i$"。

#### (b) Meta-Learning 视角

预训练阶段模型就在"学习如何学习"——见到很多 (input, output) 对的格式。ICL 是这种能力的激活。

#### (c) 回归视角

ICL 等价于模型在"做 regression"：在 prompt 中找到映射 $f$，应用到 query。

实验支持：

- 训练数据分布对 ICL 影响巨大（与人类元学习相似）。
- 增大模型规模让 ICL 性能"相变"涌现。

### 2.4 提示选择策略

ICL 的性能高度依赖**示例选择**：

#### (a) 随机选择

最简单但效果不稳定。

#### (b) K-NN 选择

用 sentence embedding 找与 query 最相似的 K 个训练样本：

```python
embeddings = embed_model.encode(train_texts)
query_emb = embed_model.encode(query)
top_k_indices = cosine_similarity(query_emb, embeddings).argsort()[-k:]
```

直觉：相似的输入应该有相似的输出。

#### (c) Diverse 选择

让 K 个示例覆盖多样模式：

```python
selected = []
remaining = list(range(len(train)))
for _ in range(k):
    scores = diversity_score(selected, remaining)
    selected.append(argmax(scores))
    remaining.remove(selected[-1])
```

代表：Su et al．（2022）的 Vote-K。

#### (d) Self-Generated Selection

让模型自己生成示例，再用模型筛选：

```
模型生成的 10 个示例
→ 模型评分（哪个示例最有信息量）
→ 选 top-K
```

代表：Self-Generated In-Context Examples（SG-ICE）。

### 2.5 示例顺序

ICL 对示例顺序敏感。同一组示例，不同顺序可能让性能差 10+ 个点。

缓解：

- 多次随机顺序评估取平均。
- 用 Random Search 找到最优顺序。
- 用多样性 + 相似性排序（diversity-similarity ordering）。

## 三、Chain-of-Thought（CoT）

### 3.1 核心思想

Wei et al.（2022）发现，让模型"显式推理"能大幅提升复杂任务的性能：

**标准提示**：

```
问题：一个水果店有 15 个苹果，卖出 8 个，又进了 12 个，现在有几个？
答案：
```

模型可能直接答错。

**CoT 提示**：

```
问题：一个水果店有 15 个苹果，卖出 8 个，又进了 12 个，现在有几个？
推理：原有 15 个苹果，卖出 8 个剩 7 个，又进 12 个，总共 19 个。
答案：19
```

模型会"模仿"推理过程，生成中间步骤。GSM8K 数学推理从 10-20% 提升到 60-80%。

### 3.2 CoT 的工作机制

为什么 CoT 有效？三个理论：

#### (a) 序列化推理

把多步推理"摊开"到 token 序列中，每步只需局部推理，降低错误传播。

#### (b) 自我解释

模型在生成推理时"激活"了与该任务相关的知识，类似于"出声思考"。

#### (c) 测试时计算扩展

CoT 让模型用更多 token 完成更多"思考步骤"，本质是 test-time compute 的扩展。

### 3.3 Zero-Shot CoT

Kojima et al.（2022）发现，只需在 prompt 末尾加一句"让我们一步一步思考"（Let's think step by step），模型就能自动生成推理：

```
问题：15 个苹果卖出 8 个，又进 12 个，现在几个？
让我们一步一步思考。
```

模型会自动推理出答案。Zero-Shot CoT 在 GSM8K 上达到 40-50%，远高于 Greedy 的 10-20%。

### 3.4 Self-Consistency

Wang et al．（2023）提出多次采样 + 投票：

```
CoT 推理 1: ... → 答案 19
CoT 推理 2: ... → 答案 19
CoT 推理 3: ... → 答案 18  ← 异常
CoT 推理 4: ... → 答案 19
投票: 答案 19（3 票）
```

直觉：错误的推理路径通常不一致，正确答案的路径更"收敛"。Self-Consistency 在 GSM8K 上从 60% 提升到 80%+。

### 3.5 CoT 的局限

- **依赖模型能力**：弱模型（< 10B）CoT 收益有限。
- **推理路径错误**：CoT 可能生成看似合理但错误的推理。
- **计算成本**：每步都要生成 token，token 数 × 3-5 倍。

## 四、ReAct：推理 + 行动

### 4.1 核心思想

Yao et al．（2023）的 ReAct 把推理（Reasoning）与行动（Acting）结合，让 LLM 在思考的同时调用外部工具：

```
Thought 1: 我需要查 2024 年奥运会的主办城市。
Action 1: Search[2024 年奥运会主办城市]
Observation 1: 2024 年夏季奥运会在巴黎举办。
Thought 2: 现在我知道主办城市是巴黎。
Action 2: Finish[巴黎]
```

### 4.2 ReAct 的轨迹格式

每个轨迹包含 Thought-Action-Observation 三元组：

```
Thought: 当前的思考、计划、推理。
Action: 调用的工具（如 Search、Calculator）。
Observation: 工具返回的结果。
... (循环)
Action: Finish[最终答案]
```

### 4.3 ReAct vs CoT

| 维度 | CoT | ReAct |
| --- | --- | --- |
| 外部信息 | 仅依赖模型内部知识 | 调用工具获取真实信息 |
| 可信度 | 可能幻觉 | 基于真实数据 |
| 适用范围 | 封闭任务 | 开放任务 |
| 速度 | 快 | 慢（多次工具调用） |

### 4.4 工具调用

ReAct 依赖于工具的可靠性。典型工具集：

- **Search**：Web 搜索、Wikipedia、数据库查询。
- **Calculator**：精确计算（LLM 数学弱）。
- **Code Interpreter**：执行 Python 代码。
- **Custom API**：业务系统接口。

OpenAI 的 Function Calling、Claude 的 Tool Use 是 ReAct 的工业化实现。

## 五、Tree-of-Thoughts（ToT）

### 5.1 核心思想

Yao et al．（2023）的 Tree-of-Thoughts 把推理组织成**搜索树**：

- 每个节点是一个"思考状态"。
- 边表示"扩展该状态的可能路径"。
- 用 BFS / DFS 搜索最优路径。

```
       [初始问题]
       /     |      \
  [推理 a] [推理 b] [推理 c]
   /   \     |      /   \
 [a1] [a2] [b1] [c1] [c2]
```

每步评估每个节点的"价值"（如 LLM 自评、reward model），剪枝低价值分支。

### 5.2 ToT 的应用

- **24 点游戏**：搜索所有可能的数字组合。
- **创意写作**：探索多个故事分支。
- **数学证明**：尝试多种证明路径。

ToT 在 24 点游戏上达到 74% 成功率（Greedy 仅 4%），在创意写作上人类评估显著优于 CoT。

### 5.3 ToT 的局限

- **计算昂贵**：树搜索比线性 CoT 多 10-100 倍 token。
- **价值评估难**：如何定义"思考状态的优劣"。
- **实施复杂**：需要定制搜索算法。

## 六、Graph-of-Thoughts（GoT）

### 6.1 核心思想

Yao et al．（2023）的 Graph-of-Thoughts 把推理组织成**图**而非树：

- 节点是思考状态。
- 边表示依赖关系，可以多输入聚合。
- 支持"合并多个推理路径"。

代表工作：

- **GoT**：把多个推理片段聚合成更高级的结论。
- **Program-of-Thoughts**：用代码表达推理。

### 6.2 与 ToT 的区别

| | ToT | GoT |
| --- | --- | --- |
| 结构 | 树 | 图 |
| 合并 | 无 | 任意节点可合并 |
| 表达力 | 中 | 高 |
| 复杂度 | 中 | 高 |

GoT 更灵活，但工程实现更复杂。

## 七、Self-Reflection 与 Self-Correction

### 7.1 Self-Refine

Madaan et al．（2023）的 Self-Refine 让模型**迭代优化自己的输出**：

```
初始输出 → 反馈 → 改进 → 反馈 → 改进 → ...
```

每步模型生成反馈（"这段太啰嗦"、"这个推理有错"），然后基于反馈改进。

### 7.2 Reflexion

Shinn et al.（2023）的 Reflexion 在多轮任务中反思：

```
试错 1: 失败 → 反思"为什么失败"
试错 2: 基于反思调整策略 → 失败 → 反思
试错 3: → 成功
```

Reflexion 让 LLM 在长期任务（如 ALFWorld、HotPotQA）上从 30% 提升到 70%+。

### 7.3 Self-Correcting RAG

RAG 系统中加入自我纠错：

```
生成 → 自我检查（"这个答案有引用支持吗"）
       ↓ 引用不足
       重新检索 → 重新生成
```

## 八、System Prompt 与角色扮演

### 8.1 System Prompt 设计

System Prompt 是模型的"人格设定"，强烈影响输出风格：

```python
system = """你是一个专业的数据科学家，专门做 NLP 任务的代码实现。
你的回答应该：
1. 包含完整可运行的代码。
2. 解释关键算法步骤。
3. 给出可能的改进方向。

回答风格：简洁但严谨。"""
```

常见 system prompt 模式：

- **角色设定**："你是一个 X 专家"
- **输出格式**："请用 JSON 格式输出"
- **约束条件**："回答不超过 200 字"
- **示例参考**："参考以下例子的风格"

### 8.2 角色扮演（Role-Playing）

让模型扮演特定角色：

```
"假设你是苏格拉底。请用问答法引导我思考'什么是美'。"
```

模型会"模仿"苏格拉底的对话风格，提高生成质量与风格一致性。

## 九、Prompt Engineering 经验总结

### 9.1 高质量 Prompt 的特征

1. **明确的任务定义**：让模型知道做什么。
2. **清晰的输入格式**：避免歧义。
3. **详细的输出规范**：JSON / Markdown / 长度。
4. **Few-Shot 示例**：3-5 个典型例子。
5. **思维链 / 推理步骤**：复杂任务必备。

### 9.2 常见错误

- **过于含糊**："写一段关于 AI 的内容" → 太宽泛。
- **缺少示例**：仅靠描述难以让模型理解期望。
- **指令冲突**：多条规则互相矛盾。
- **超出能力**：要求模型做不擅长的事（如精确计算）。

### 9.3 调试技巧

- **A/B 测试**：对比不同 prompt 的输出。
- **分步验证**：把任务拆解，逐步验证。
- **LLM-as-Judge**：用 LLM 评估生成质量。
- **失败案例分析**：找出 prompt 的边界与失败模式。

## 十、实际应用模式

### 10.1 文档问答

```python
prompt = f"""基于以下文档回答问题。如果文档没有提到答案，请回答"我不知道"。

文档：
{document}

问题：{question}

回答：
"""
```

加入"我不知道"指令减少幻觉。

### 10.2 摘要

```python
prompt = f"""请用 100 字以内总结以下文本的核心观点。保留关键数据。

文本：
{text}

摘要：
"""
```

明确长度、风格、信息保留要求。

### 10.3 信息抽取

```python
prompt = f"""从以下文本中抽取命名实体，返回 JSON 列表。

实体类型：PER、LOC、ORG、DATE

文本：{text}

JSON 列表：
"""
```

明确类型、格式、边界。

### 10.4 代码生成

```python
prompt = f"""实现以下函数。要求：
1. 包含完整 docstring。
2. 处理边界情况（空列表、负数等）。
3. 包含测试用例。

函数描述：{description}

```python
def {func_name}({params}):
    \"\"\"...\"\"\"
    ...
```
"""
```

明确要求 + 格式约束 + 边界处理。

## 十一、未来方向

1. **自动 Prompt 优化**：用模型自动生成 / 优化 prompt（APE, OPRO）。
2. **Long-Horizon Agents**：多步骤、长期任务的 prompt 架构。
3. **Multi-Modal Prompting**：文本 + 图像 + 视频的联合提示。
4. **Self-Improving Systems**：模型从错误中学习并自动调整 prompt。

## 小结

| 技术 | 核心 | 适用 |
| --- | --- | --- |
| Zero-Shot | 仅指令 | 简单任务 |
| Few-Shot ICL | K 个示例 | 通用任务 |
| CoT | 推理步骤 | 推理任务 |
| Self-Consistency | 多采样投票 | 推理任务 |
| ReAct | 推理 + 工具 | 开放任务 |
| ToT / GoT | 树 / 图搜索 | 复杂规划 |
| Self-Refine | 迭代优化 | 质量提升 |

提示驱动生成是 LLM 时代的"编程范式"——不再写代码训练模型，而是写 prompt 引导模型。从 GPT-3 的 ICL 到 ReAct 的工具调用，再到 ToT 的搜索式推理，每一步都让 LLM 离"通用智能体"更近。掌握这些技术，你就能用 LLM 构建强大而灵活的应用。
