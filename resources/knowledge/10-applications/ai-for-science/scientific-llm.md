# 科学 LLM：从文献综述到自动化研究

科学研究是 LLM 应用**最具潜力也最需要严谨**的领域——写论文、做综述、生成假设、设计实验、分析数据，每一步都有 AI 的空间。从 GPT-4 通过材料学考试，到 Coscientist 自动做化学实验，从 SciGLM 助力中文科研，到 ChemCrow 与机器人结合，"AI for Science" 正在改变科研的工作方式。本文系统介绍科学 LLM 的核心任务、代表系统、评测基准，以及与传统科研工作流的关系。

## 一、科学 LLM 的应用层级

```text
Level 1: 信息检索
  文献综述 / 知识问答 / 数据查询
  案例: Consensus, Scite, Elicit

Level 2: 内容生成
  论文写作 / 摘要 / 翻译 / 改写
  案例: Jenni AI, Scispace

Level 3: 推理与假设生成
  跨文献推理 / 提出假设 / 推理新结论
  案例: Coscientist, AI Scientist

Level 4: 实验自动化
  设计实验 + 机器人执行 + 数据分析
  案例: Coscientist + 机器人, ChemCrow

Level 5: 端到端自动化研究
  从问题到论文的完整流程
  案例: 仍处于探索阶段（AI Scientist v2 等）
```

每一级都比上一级**能力更强、风险更高**。

## 二、训练科学 LLM

### 1. 数据来源

```python
SCIENCE_DATA_SOURCES = {
    "papers": ["arXiv", "PubMed", "bioRxiv", "OpenReview", "CNKI"],
    "books": ["Springer", "Elsevier", "Cambridge", "Wiley"],
    "code": ["GitHub", "Papers with Code", "Kaggle"],
    "data": ["Kaggle", "UCI", "政府公开数据", "PDB", "GenBank"],
    "tools": ["Wolfram Alpha", "PubChem", "Materials Project"],
}
```

### 2. 代表模型

#### SciGPT / SciGLM（智源）

中文科学 LLM：

```python
class SciGLMConfig:
    model_type = "decoder-only transformer"
    n_params = "6B / 32B"
    training_data = "中英文科学论文 + 教科书 + 百科"
    capabilities = ["科学问答", "文献综述", "论文摘要", "代码生成"]
```

#### ChemLLM / ChemDFM

化学专用 LLM。

#### BioMedLM / Meditron / OpenBioLLM

生物医学 LLM（见医疗 LLM 章）。

#### Galactica（Meta）

科学专用 LLM（已下架），但其训练方法影响后续：

```python
class GalacticaSpecializedTokens:
    """科学专用 token。"""
    SPECIAL_TOKENS = [
        "[START_REF]", "[END_REF]",     # 引用
        "[START_EQ]", "[END_EQ]",        # 公式
        "[START_SMILES]", "[END_SMILES]", # 分子式
        "[START_AMINO]", "[END_AMINO]",   # 蛋白质序列
        "<citation>", "</citation>",
        "<formula>", "</formula>",
    ]
```

### 3. 微调与指令对齐

```python
def scientific_instruction_tuning(model, dataset):
    """科学指令微调。"""
    # 数据集：科研问题 + 高质量回答
    examples = [
        {
            "instruction": "总结这篇论文的创新点",
            "input": "<论文全文>",
            "output": "1. ...\n2. ...\n3. ...",
        },
        {
            "instruction": "为以下分子设计合成路线",
            "input": "目标分子 SMILES: ...",
            "output": "Step 1: ...\nStep 2: ...",
        },
        ...
    ]
    
    # 用 SFT 训练
    trainer = SFTTrainer(model=model, dataset=examples)
    trainer.train()
```

## 三、核心应用任务

### 1. 文献综述与检索

```python
def literature_synthesis(research_question: str, n_papers=50) -> str:
    """综合多篇论文回答研究问题。"""
    # 1) 检索相关论文
    papers = retrieve_papers(research_question, n=n_papers)
    
    # 2) 让 LLM 提取每篇论文关键信息
    summaries = []
    for p in papers:
        s = llm.generate(f"""提取以下论文的关键信息：

{p['title']}
{p['abstract']}
{p['full_text']}

输出：
- 研究问题
- 方法
- 主要结果
- 局限性
- 对你的研究问题 '{research_question}' 的相关性""")
        summaries.append({"paper": p, "summary": s})
    
    # 3) 综合多篇
    return llm.generate(f"""综合以下论文摘要，回答：{research_question}

{chr(10).join(s['summary'] for s in summaries)}

要求：
- 标注每个论断来自哪篇论文
- 指出共识与分歧
- 指出尚不明确的地方""")
```

**代表工具**：Elicit、Consensus、Scite——它们都是 LLM + 文献检索的组合。

### 2. 论文写作辅助

```python
PAPER_WRITING_PROMPTS = {
    "abstract": """基于以下论文内容写一个结构化摘要：

方法: {methods}
结果: {results}
结论: {conclusion}

要求：
- 150~250 字
- 含背景、方法、结果、结论
- 不含引用、缩写解释""",
    
    "introduction": """为论文 '{title}' 写 introduction：

核心贡献: {contribution}

要求：
- 3~4 段
- 从背景 → 现状 → 空白 → 本文贡献
- 引用领域经典文献""",
    
    "related_work": """对比本文与现有方法：

本文: {our_method}
相关工作: {related_papers}

要求：
- 按主题分类
- 突出差异
- 客观评价""",
    
    "discussion": """基于以下结果写 discussion：

结果: {results}
现有文献: {literature}

要求：
- 解释结果
- 与文献对比
- 局限性
- 未来方向""",
}
```

**代表工具**：Jenni AI、Scispace、Wordtune、Paper Digest。

### 3. 假设生成

```python
def generate_hypotheses(background: str, n_hypotheses=10) -> list:
    """用 LLM 生成研究假设。"""
    prompt = f"""基于以下领域背景，提出 {n_hypotheses} 个可验证的科研假设：

领域: {background}

每个假设应该：
1. 基于现有知识但有新颖性
2. 可被实验或观察验证
3. 有明确变量与因果
4. 列出潜在实验设计

输出 JSON 数组。"""
    return json.loads(llm.generate(prompt))
```

**代表工作**：GPT-4 在材料学发现新电池电解质（2024 Nature 论文）。

### 4. 实验设计

```python
def design_experiment(hypothesis: str, available_resources: dict) -> dict:
    """设计实验验证假设。"""
    prompt = f"""为以下假设设计实验：

假设: {hypothesis}

可用资源: {available_resources}

输出：
{{
    "objective": "实验目标",
    "variables": {{
        "independent": [...],
        "dependent": [...],
        "controlled": [...]
    }},
    "methodology": "实验步骤详细描述",
    "sample_size": "...",
    "controls": [...],
    "expected_results": "...",
    "potential_issues": [...],
    "timeline": "...",
    "budget_estimate": "..."
}}"""
    return json.loads(llm.generate(prompt))
```

### 5. 数据分析辅助

```python
def analyze_data_with_llm(data_summary: dict, research_question: str) -> dict:
    """让 LLM 辅助解读数据分析结果。"""
    prompt = f"""解读以下数据分析结果：

研究问题: {research_question}

数据摘要:
- 样本量: {data_summary['n']}
- 主要统计: {data_summary['stats']}
- 关键图表: {data_summary['key_charts']}

请提供：
1. 主要发现
2. 统计显著性
3. 效应大小
4. 可能的混杂因素
5. 下一步分析建议
6. 适合的报告格式"""
    return llm.generate(prompt)
```

### 6. 同行评审

```python
def review_paper(paper_text: str) -> dict:
    """LLM 作为辅助评审。"""
    return llm.generate(f"""请作为领域专家评审以下论文：

{paper_text}

输出结构化评估：
1. 原创性 (1-5)
2. 技术严谨性 (1-5)
3. 实验设计合理性 (1-5)
4. 结果解释充分性 (1-5)
5. 写作质量 (1-5)
6. 总体评分 (1-5, accept / revise / reject)
7. 详细反馈（strengths / weaknesses）
8. 给作者的具体建议""")
```

**注意**：LLM 评审**不能替代人类**——但可以加速初审。

## 四、代表系统

### 1. Coscientist（Google DeepMind + PSC, 2024）

**AI 自动化化学家**：

```python
class Coscientist:
    def plan_synthesis(self, target_molecule):
        # 1) 规划合成路线
        route = self.llm.plan(target_molecule)
        # ["Step 1: 加入 X", "Step 2: 加热到 80°C", ...]
        
    def execute_synthesis(self, route):
        # 2) 让机器人执行
        for step in route:
            self.robot.execute(step)
            
    def analyze_results(self, samples):
        # 3) 分析实验结果
        return self.llm.analyze(samples)
```

**成就**：用 GPT-4 自主规划并合成了**多个药物分子**。

### 2. AI Scientist（Sakana AI, 2024）

**端到端自动化研究**：

```text
流程:
1. 读领域论文，提出新颖研究方向
2. 设计实验并写代码
3. 跑实验、生成图表
4. 写完整论文（ICML 格式）
5. 模拟同行评审
```

**争议**：是否能产出"真贡献"？目前是初步证明概念。

### 3. ChemCrow / ChemLLM / 各种化学 Agent

化学专用 Agent——结合化学工具（PubChem、RDKit、机器人）：

```python
class ChemAgent:
    def __init__(self):
        self.tools = [
            "molecule_similarity",
            "synthesize_route_planning",
            "predict_reaction_yield",
            "safety_check",
        ]
    
    def answer_chem_query(self, query):
        # 多步推理 + 工具调用
        ...
```

### 4. 代码生成：Code4Science

```python
def generate_analysis_code(research_question: str, data_description: str):
    """生成科研数据分析代码。"""
    return llm.generate(f"""基于研究问题 '{research_question}' 和数据描述：

{data_description}

写出完整的 Python 分析代码，包括：
- 数据加载与清洗
- 探索性分析
- 统计检验
- 可视化
- 结果保存""")
```

**代表**：Anthropic Artifacts for Science、GitHub Copilot for Science。

### 5. 国内代表

- **智源 SciGLM**：中英文科学问答。
- **DeepSeek 学术版**：长上下文代码辅助。
- **腾讯 MatChat**（材料科学）。
- **阿里云通义晓医**（医学科研）。
- **百度学术 AI**。

## 五、评测基准

### 1. SciQA

```python
SCIENCE_QA_DATASETS = {
    "SciQ": "中学科学问答",
    "ARC-Challenge": "科学推理挑战",
    "OpenBookQA": "开放书本问答",
    "MMLU-STEM": "STEM 子集",
    "GPQA": "研究生级问答（高难度）",
    "TheoremQA": "定理问答",
    "SciEval": "中文科学评测",
}
```

### 2. GPQA（Graduate-Level QA）

**极难的科学评测**——博士级问题：

```python
# GPQA 例子
QUESTION = """A new compound is found to have a half-life of 3.5 hours 
in a first-order decomposition reaction. Starting with 0.1 mol of the 
compound, how many moles remain after 12 hours?

A) 0.0125
B) 0.025
C) 0.05
D) 0.075

Answer: A"""
```

**结果**：
- GPT-4：~50%（虽然随机是 25%）。
- 人类专家（博士）：~65%。
- GPT-4 + 工具 + 多步：~70%+。

### 3. SciEval / C-Eval（中文）

```python
CHINESE_SCIENCE_BENCHMARKS = {
    "C-Eval": "中文综合评测（含 STEM）",
    "SciEval": "中文科学评测",
    "MMCU": "中文多学科评测",
    "AGIEval": "中文大学考试题",
}
```

### 4. 任务特定评测

```python
TASK_SPECIFIC_BENCHMARKS = {
    "literature_synthesis": "QASPER, Scientific Document QA",
    "hypothesis_generation": "human evaluation",
    "experiment_design": "domain-specific (e.g., chemistry synthesis)",
    "code_generation": "HumanEval-X, MMLU-Code",
    "paper_review": "human evaluation vs real reviews",
}
```

## 六、跨学科应用

### 1. 数学

```python
class MathProofAssistant:
    """AI 数学证明助手。"""
    def prove_theorem(self, statement):
        # Lean / Coq / Isabelle 集成
        proof_attempt = self.llm.generate_proof(statement)
        
        # 自动验证
        if verify_in_lean(proof_attempt):
            return {"success": True, "proof": proof_attempt}
        else:
            return self.iterate(statement, proof_attempt)
```

代表：**LeanDojo**、**Lean-GPT**、DeepMind **AlphaProof**（IMO 银牌）。

### 2. 物理

```python
class PhysicsHypothesisGenerator:
    """AI 物理假设生成。"""
    def generate_research_questions(self, field):
        # 1) 读取该领域近期论文
        papers = retrieve_arxiv(field, days=180)
        
        # 2) 让 LLM 总结未解决的问题
        questions = self.llm.generate(f"""基于以下近期论文，找出未解决的关键问题：

{papers}

输出：
1. 3~5 个未解决问题
2. 每个问题的潜在突破方向
3. 可能的实验/理论验证方法""")
        
        return questions
```

### 3. 化学

```python
class ChemistryAgent:
    """AI 化学 Agent。"""
    def run_synthesis(self, target_smiles):
        # 1) 规划合成路线
        route = self.plan_synthesis(target_smiles)
        
        # 2) 预测每步反应
        for step in route:
            yield_pred = self.predict_yield(step)
            if yield_pred < 0.3:
                route = self.revise_route(route, step)
        
        # 3) 机器人执行
        return self.execute_with_robot(route)
```

### 4. 生物医学

```python
class BiomedicalResearch:
    """生物医学研究 AI。"""
    def find_drug_target(self, disease_genes: list) -> list:
        # 1) 找关键蛋白
        proteins = self.identify_proteins(disease_genes)
        
        # 2) AlphaFold 预测结构
        structures = [alphafold.predict(p) for p in proteins]
        
        # 3) 找药物口袋
        pockets = [find_pocket(s) for s in structures]
        
        # 4) 虚拟筛选
        candidates = self.virtual_screen(pockets)
        
        return candidates
```

### 5. 材料科学

```python
class MaterialsDiscovery:
    """AI 材料发现。"""
    def find_new_battery_material(self, requirements):
        # 1) 用 LLM 分析文献
        candidates = self.llm.generate(f"基于这些要求，推荐 5 种新型电池材料：{requirements}")
        
        # 2) 性质预测（用图神经网络）
        for c in candidates:
            c.properties = self.gnn.predict(c.structure)
        
        # 3) 排序推荐
        return sorted(candidates, key=lambda x: x.properties.match_score)
```

## 七、关键风险与挑战

### 1. 幻觉

科学幻觉特别危险：

```text
❌ 一般幻觉: "巴黎是德国首都" → 用户识破
❌ 科学幻觉: "F = ma 是爱因斯坦发现的" → 用户可能传播
```

**缓解**：
- 强制引用（RAG）。
- 多模型交叉验证。
- 人类专家最终审核。

### 2. 错误推理

LLM 在**多步推理**上仍弱：

```text
问题: 2n + 3 = 11，求 n
GPT-4: 答案 n = 4  ← 正确（直答）
但 LLM 推理路径可能错（多步时）
```

**缓解**：
- 思维链（CoT）+ 程序辅助（PAL）。
- 与符号系统结合。

### 3. 知识时效

```text
训练截止: 2024 年 6 月
新论文: 2024 年 10 月（LLM 不知道）
```

**缓解**：
- RAG 接入最新文献。
- 定期更新。

### 4. 科研诚信

LLM 在论文写作中的使用引发争议：

```text
过度依赖 LLM:
- 写论文 → 失去个人思考
- 生成数据 → 学术不端
- 重复发表 → 学术不端

合理使用 LLM:
- 语言润色
- 思路启发
- 文献整理
```

**建议**：所有 AI 辅助必须**透明声明**。

### 5. 公平性

顶级期刊倾向英美作者——LLM 可能放大这种偏见。

## 八、科研工作流的演化

### 传统科研流程

```text
问题 → 读文献 → 提假设 → 设计实验 → 跑实验 → 数据分析 → 写论文 → 同行评审 → 修改 → 发表
  ↓ 数年时间
```

### AI 增强流程

```text
问题 → AI 文献综述 → AI 假设建议 → 人机协作设计 → 部分 AI 自动化 → AI 数据分析 → AI 辅助写作 → AI 辅助评审 → 发表
  ↓ 可能缩短到数月
```

### 关键：人在哪里？

```python
HUMAN_IRREPLACEABLE = {
    "原创想法": "真正新颖的思路仍需人",
    "实验设计判断": "哪些实验最有价值",
    "数据质量把控": "哪些结果可信",
    "伦理决策": "研究是否符合伦理",
    "团队合作": "沟通、协调、领导",
    "教学传承": "培养下一代科学家",
}
```

## 九、未来方向

### 1. AI Scientist 的真正端到端

```text
未来可能:
- AI 自己提问题
- 设计实验 + 写代码 + 跑实验
- 分析数据 + 写论文
- 模拟评审
- 投稿到真实期刊（可能被识别为 AI）
```

### 2. 数字孪生实验室

```python
class DigitalTwinLab:
    """数字孪生实验室——先在虚拟中跑实验。"""
    def simulate_experiment(self, design):
        # 用 LLM + 模拟器预测实验结果
        predicted = self.simulator(design)
        return predicted
    
    def optimize_design(self, objective):
        # 多次模拟找最优实验设计
        return self.bayesian_optimization(objective)
```

### 3. 跨学科 AI

打破学科壁垒——生物 + AI + 材料：

```python
class CrossDisciplinaryAI:
    """跨学科 AI——连接多领域知识。"""
    def __init__(self):
        self.domains = ["biology", "chemistry", "physics", "math", "cs"]
        self.knowledge_graphs = {
            d: load_kg(d) for d in self.domains
        }
    
    def cross_insight(self, question):
        # 在多个知识图谱中推理
        ...
```

### 4. 开放科学与 AI

```text
未来:
- 论文 + 实验 + 数据 + AI 全开放
- 任何人可复现
- AI 辅助审稿
- 共同推动知识
```

## 十、给科研团队的清单

1. **保持怀疑**：LLM 输出必须经过验证。
2. **学习使用工具**：Elicit、Consensus、Copilot、AlphaFold 等。
3. **记录 AI 辅助**：投稿时透明声明。
4. **保护原创性**：AI 帮你写，但不能替代你思考。
5. **多工具交叉**：用多个 LLM 验证关键结论。
6. **结合专业知识**：AI 加速但不能替代深度专业知识。
7. **关注伦理**：AI 生成内容、伪造数据等问题。
8. **持续学习**：AI 工具在快速演进。
9. **协作文化**：让 AI 增强团队协作。
10. **人机共生**：找到你的科研中 AI 不可替代的部分。

## 小结

科学 LLM 已经从"辅助写作"走到"辅助推理"再到"自动化实验"——Coscientist、AI Scientist、ChemCrow 等系统展示了"AI 参与科研"的多种可能。**核心任务**包括文献综述、论文写作、假设生成、实验设计、数据分析、同行评审。**核心挑战**是幻觉、错误推理、知识时效、科研诚信。**未来方向**是端到端自动化、数字孪生实验室、跨学科 AI、开放科学。**真正成功的科学 LLM** 是"加速人而不是替代人"——让科学家从繁琐工作中解脱，专注于真正需要创造力的部分。下一篇我们将看到 AI for Science 的另一个具体应用——**AI 天气与气候建模**。
