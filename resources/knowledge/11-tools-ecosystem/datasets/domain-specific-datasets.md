# 领域特定数据集：医疗、法律、金融、教育

通用 LLM 在垂直领域表现往往不够好——因为专业术语、行业规范、领域推理都需要专门训练。本文讨论医疗、法律、金融、教育四个典型垂直领域的数据集构建、来源、挑战与最佳实践。

## 一、为什么需要领域特定数据集

```python
# 通用模型在垂直领域的局限
limitations = [
    "1. 专业术语理解不准确（如 ICD 编码、法律条款引用）",
    "2. 行业规范与标准不熟（如医疗的诊疗指南）",
    "3. 推理模式不同（如法律三段论、医学鉴别诊断）",
    "4. 数据陈旧（专业领域需要最新指南）",
    "5. 合规风险（医疗、法律错误代价高）",
    "6. 输出格式专业（病历 JSON / 法律意见书）",
]

# 垂直领域模型分类
vertical_ai_models = {
    "医疗": "Med-PaLM 2, Meditron, BioMistral, Huatuo",
    "法律": "LawGPT, ChatLaw, SaulLM, LegalBERT",
    "金融": "FinGPT, BloombergGPT, FinTral",
    "教育": "EduChat, MathGPT, CodeTutor",
    "科学": "SciGPT, ChemLLM, Galaxy",
}
```

## 二、医疗领域数据集

### 数据来源

```python
# 医疗数据来源
medical_data_sources = {
    "公开文献": [
        "PubMed (~36M 篇摘要)",
        "PubMed Central (PMC, ~9M 全文)",
        "Cochrane Library (系统评价)",
        "UpToDate (临床决策)",
    ],
    "电子病历 (EHR)": [
        "MIMIC-III/IV (重症监护)",
        "eICU (多中心 ICU)",
        "PhysioNet (生理信号)",
    ],
    "问答对": [
        "MedQA (USMLE 风格)",
        "PubMedQA",
        "MedMCQA",
        "LiveQA (真实医患问答)",
    ],
    "知识库": [
        "UMLS (Unified Medical Language System)",
        "ICD-10/11 (疾病分类)",
        "SNOMED CT (临床概念)",
        "DrugBank (药物)",
        "Gene Ontology",
    ],
}
```

### 主流医疗数据集

```python
# 医疗 LLM 训练数据集

medical_datasets = {
    "PubMed": {
        "size": "36M abstracts",
        "use": "医学知识预训练",
        "license": "开放",
    },
    "MedQA": {
        "size": "61K 多选题 (USMLE 风格)",
        "use": "医学考试评测",
        "language": "英文 + 中文",
    },
    "PubMedQA": {
        "size": "1K expert + 211K artificial",
        "use": "医学问答评测",
        "task": "yes/no/maybe 分类",
    },
    "MedMCQA": {
        "size": "194K 多选题 (印度医学入学)",
        "use": "医学考试评测",
        "subjects": "2.4K 主题",
    },
    "MIMIC-III/IV": {
        "size": "40K+ ICU 患者",
        "use": "电子病历研究",
        "access": "PhysioNet 申请",
    },
    "ChatDoctor / Huatuo": {
        "size": "100K+ 医患对话",
        "use": "中文医疗指令微调",
    },
}
```

### 数据处理特殊考虑

```python
# 医疗数据处理的特殊挑战
medical_data_challenges = {
    "1. 隐私保护": "HIPAA 合规，去标识化",
    "2. 数据稀缺": "高质量标注数据少",
    "3. 多模态": "CT/MRI/X-ray + 文本",
    "4. 标注昂贵": "需医生专家标注",
    "5. 术语标准化": "ICD / SNOMED 编码",
    "6. 时效性": "新药物/新诊疗指南更新快",
}

# 去标识化示例（HIPAA 18 项标识符）
def deidentify_medical_text(text: str) -> str:
    """去除 PHI (Protected Health Information)"""
    # 姓名
    text = re.sub(r"\bDr\.\s+[A-Z][a-z]+\b", "[DOCTOR]", text)
    # 日期（保留时间关系，但去具体）
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]", text)
    # 医院名（用占位符）
    text = re.sub(r"\b[A-Z][a-z]+ Hospital\b", "[HOSPITAL]", text)
    # 身份证 / SSN
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ID]", text)
    return text
```

### 医疗 LLM 评测

```python
# 医疗 LLM 评测基准
medical_benchmarks = {
    "USMLE / MedQA": "美国医师执照考试风格",
    "PubMedQA": "生物医学文献问答",
    "MedMCQA": "AIIMS / NEET PG 风格",
    "MultiMedQA (Google)": "MedQA + PubMedQA + MedMCQA + ... + LiveQA",
    "CliBench (中医)": "中医临床考试",
    "CMB (Chinese Medical Benchmark)": "中文医师考试",
    "Huatuo-26M (中文)": "中文医学问答",
}
```

## 三、法律领域数据集

### 数据来源

```python
# 法律数据来源
legal_data_sources = {
    "法规文本": [
        "国家法律法规数据库",
        "美国法典 (US Code)",
        "欧盟 EUR-Lex",
        "中国裁判文书网 (部分)",
    ],
    "判例": [
        "CourtListener (美国)",
        "Westlaw / LexisNexis (商业)",
        "中国裁判文书网",
        "中国裁判文书网",
        "OpenJurist",
    ],
    "学术": [
        "SSRN",
        "LawArchive",
        "HeinOnline",
    ],
    "合同模板": [
        "SEC EDGAR (合同)",
        "Kaggle CUAD (合同理解)",
        "ATT-CK Dataset",
    ],
    "问答/咨询": [
        "LegalBench",
        "Law Stack Exchange",
        "中国法律咨询网站",
    ],
}
```

### 主流法律数据集

```python
# 法律 LLM 数据集
legal_datasets = {
    "LegalBench": {
        "size": "162 任务",
        "use": "法律推理评测",
        "creator": "Stanford / ICML 2023",
    },
    "CUAD": {
        "size": "510 合同 + 13K 标注",
        "use": "合同理解评测",
        "task": "14 类合同条款抽取",
    },
    "CaseHOLD": {
        "size": "53K 多选题",
        "use": "判例引用预测",
    },
    "CAIL (中国)": {
        "size": "1.6M 案件",
        "use": "中文法律判决预测",
        "year": 2018-2022",
    },
    "LawGPT": {
        "size": "20K 法律指令",
        "use": "中文法律对话",
    },
    "ChatLaw": {
        "size": "100K 法律指令",
        "use": "中文法律咨询",
    },
}
```

### 法律任务分类

```python
# 法律 LLM 任务
legal_tasks = {
    "1. 法律检索 (Legal Retrieval)": "从法规/判例中找到相关条款",
    "2. 条款分类 (Clause Classification)": "合同条款类型识别",
    "3. 法律命名实体 (Legal NER)": "当事人、条款、金额抽取",
    "4. 判决预测 (Judgment Prediction)": "给定事实预测判决",
    "5. 法律问答 (Legal QA)": "咨询类问题回答",
    "6. 合同审查 (Contract Review)": "风险条款识别",
    "7. 案例摘要 (Case Summarization)": "长判决书摘要",
    "8. 法律推理 (Legal Reasoning)": "复杂逻辑推理",
}
```

## 四、金融领域数据集

### 数据来源

```python
# 金融数据来源
finance_data_sources = {
    "市场数据": [
        "Yahoo Finance (历史价格)",
        "Alpha Vantage (技术指标)",
        "Wind (中国)",
        "Tushare (中国)",
        "Bloomberg Terminal",
    ],
    "新闻/公告": [
        "Reuters / Bloomberg News",
        "SEC EDGAR (美股公告)",
        "巨潮资讯网 (中国 A 股)",
        "新浪财经",
    ],
    "研报": [
        "券商研报",
        "Wind 研报库",
        "Bloomberg Research",
    ],
    "财报": [
        "10-K / 10-Q (美国)",
        "上市公司年报",
        "财务指标数据库",
    ],
    "另类数据": [
        "社交媒体情绪",
        "信用卡数据",
        "卫星图像",
    ],
    "监管文件": [
        "Fed 公告",
        "央行政策",
        "招股说明书",
    ],
}
```

### 主流金融数据集

```python
# 金融 LLM 数据集
finance_datasets = {
    "FinGPT": {
        "size": "100K+ 金融新闻/公告",
        "use": "金融情绪分析",
    },
    "FLARE (Financial Language Understanding)": {
        "size": "多任务金融 NLP",
        "use": "金融评测",
    },
    "FinQA": {
        "size": "8K 财务问答",
        "use": "财报推理",
    },
    "TAT-QA": {
        "size": "财报 + 表格问答",
    },
    "FinBen": {
        "size": "23 任务 + 35 数据集",
        "use": "综合金融评测",
    },
    "BloombergGPT-Data": {
        "size": "363B tokens (未公开)",
        "use": "金融预训练",
    },
}
```

### 金融任务分类

```python
# 金融 LLM 任务
finance_tasks = {
    "1. 情绪分析": "新闻/公告的情绪判断",
    "2. 命名实体": "股票代码、公司名、人物识别",
    "3. 关系抽取": "公司-人物-事件关系",
    "4. 时间序列预测": "价格预测（注意：不是 LLM 强项）",
    "5. 财报分析": "10-K/年报问答",
    "7. 研报摘要": "长研报浓缩",
    "8. 风险识别": "合同/贷款风险",
}
```

## 五、教育领域数据集

### 数据来源

```python
# 教育数据来源
education_data_sources = {
    "教材": [
        "K12 教材",
        "大学教材",
        "国际课程 (IB/AP/A-Level)",
        "中文教材（人教/沪教/北师）",
    ],
    "题库": [
        "Khan Academy (公开)",
        "可汗学院中文化",
        "国家中小学智慧教育平台",
        "各学科题库网站",
    ],
    "习题与解答": [
        "Stack Exchange (各学科)",
        "Math Stack Exchange",
        "Physics Stack Exchange",
        "LeetCode (编程)",
    ],
    "公开课": [
        "MIT OCW",
        "Coursera",
        "edX",
        "中国大学 MOOC",
    ],
    "学习行为": [
        "MOOC 学习日志",
        "智能辅导系统对话",
    ],
}
```

### 主流教育数据集

```python
# 教育 LLM 数据集
education_datasets = {
    "MathInstruct": {
        "size": "262K 数学指令",
        "use": "数学解题",
    },
    "OpenMathInstruct-2": {
        "size": "14M 数学题 (Llama-3.1-405B 生成)",
        "use": "数学指令",
    },
    "MetaMathQA": {
        "size": "395K 数学题（GSM8K/MATH 增强）",
        "use": "数学微调",
    },
    "Code-Alpaca / Magicoder": {
        "size": "100K+ 代码指令",
    },
    "Open-Orca": {
        "size": "1M 通用指令",
    },
    "GAIR-MathPile": {
        "size": "9B tokens 数学数据",
        "use": "数学预训练",
    },
    "CMRC (中文阅读理解)": {
        "size": "20K 中文阅读理解",
    },
}
```

### 教育任务分类

```python
# 教育 LLM 任务
education_tasks = {
    "1. 解题": "数学/物理/化学题目",
    "3. 讲解": "步骤化讲解知识点",
    "4. 出题": "根据知识点生成题目",
    "5. 批改": "学生答案评分",
    "6. 苏格拉底式辅导": "引导式提问",
    "7. 个性化": "针对学生水平调整",
}
```

## 六、领域数据构建流程

```python
# 垂直领域数据集构建的标准流程

pipeline = {
    "1. 需求分析": "明确模型能力目标（如医疗问答 + 病历摘要）",
    "2. 数据源调研": "公开数据 + 内部数据 + 合作伙伴数据",
    "3. 数据采集": "网络爬虫、API、商业数据库",
    "4. 数据清洗": "去重、过滤、PII 移除",
    "5. 专家标注": "医生/律师/教师等专家标注",
    "6. 数据增强": "LLM 合成 + 回译",
    "7. 质量控制": "IAA 一致性 + 抽样审核",
    "8. 评测验证": "领域基准 + 内部测试",
    "9. 合规审查": "HIPAA / GDPR / 行业监管",
    "10. 持续更新": "领域知识更新（每年/季度）",
}
```

## 七、领域数据挑战

```python
# 领域数据的核心挑战

challenges = {
    "数据获取": {
        "医疗": "病历隐私 + 医院壁垒",
        "法律": "判例版权 + 律所壁垒",
        "金融": "数据商业化 + 时效性",
        "教育": "题库版权 + 答案质量",
    },
    "标注成本": {
        "医疗": "医生时间贵（$100-500/小时）",
        "法律": "律师费用高",
        "金融": "分析师稀缺",
        "教育": "教师时间有限",
    },
    "质量控制": {
        "医疗": "诊断分歧（不同医生看法不同）",
        "法律": "法律解读争议",
        "金融": "市场判断主观",
    },
    "合规风险": {
        "医疗": "HIPAA / GDPR / 医疗广告法",
        "法律": "未授权法律建议",
        "金融": "金融建议监管（SEC / 证监会）",
        "教育": "未成年人保护",
    },
}
```

## 八、领域微调 vs RAG

```python
# 垂直领域的两种主流方案

# 方案 A：领域微调（Fine-tuning）
finetune_pros_cons = {
    "优势": [
        "模型深度理解领域术语",
        "输出风格稳定",
        "推理时可离线使用",
        "可以学到领域推理模式",
    ],
    "劣势": [
        "训练成本高",
        "知识更新需重新训练",
        "数据需求大",
        "灾难性遗忘风险",
    ],
    "适用": "稳定领域的核心能力（如医学推理）",
}

# 方案 B：RAG（检索增强生成）
rag_pros_cons = {
    "优势": [
        "知识即时更新",
        "无需训练成本",
        "可溯源（提供引用）",
        "私有数据隔离",
    ],
    "劣势": [
        "依赖检索质量",
        "上下文长度受限",
        "延迟较高",
        "风格不统一",
    ],
    "适用": "需要最新知识（黑盒化信息检索）",
}

# 最佳实践：两者结合
best_practice = {
    "基础模型": "通用 LLM + 领域微调",
    "知识更新": "RAG 提供最新信息",
    "事实查询": "RAG 优先",
    "专业推理": "微调模型优先",
}
```

## 小结

垂直领域 LLM 的关键是**高质量领域数据 + 领域专家标注 + 合规审查**。**通用 LLM + 领域微调 + RAG** 是当下主流方案。下一步发展方向：**多模态领域数据**（医学影像 + 病历、合同图像 + 文本）、**领域 Agent**（医疗 Agent 自主调用工具）、**联邦学习**（隐私合规的数据协作）。