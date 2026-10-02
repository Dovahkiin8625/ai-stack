# 医疗 LLM：从临床决策到患者教育

医疗是 LLM 应用**最有价值也最危险**的领域——错误的诊断建议可能直接危及生命。从 Med-PaLM 到 GPT-4 在 USMLE 上的突破，从 HCA、Mayo Clinic 的实际部署到 FDA 监管框架，医疗 LLM 走过一条**谨慎而坚定**的道路。本文系统梳理医疗 LLM 的核心应用（临床决策支持、病历生成、患者教育、药物咨询）、代表模型（Med-PaLM 2、GPT-4、ClinicalBERT）、评测基准（MedQA、USMLE）、监管要求（FDA、HIPAA），以及行业级挑战。

## 一、医疗 LLM 的独特挑战

医疗 LLM 与一般应用相比有**显著差异**：

| 维度 | 一般 LLM | 医疗 LLM |
|---|---|---|
| 容错率 | 80%+ 可接受 | **99.9%+ 是底线** |
| 监管 | 宽松 | **FDA 严格审批** |
| 隐私 | 一般 | **HIPAA / GDPR 严格** |
| 可解释 | 一般 | **必须**（医生要理解推理） |
| 责任 | 模糊 | **明确**（厂商要承担） |
| 知识 | 通用 | 需医学专业训练 |

这导致医疗 LLM 必须**格外严谨**——RAG + 多模型验证 + 医生最终决策是标配。

## 二、训练医疗 LLM：Med-PaLM 系列

### 1. Med-PaLM（Google, 2022）

**首个在 USMLE 上达到及格水平的 LLM**：

```python
# Med-PaLM 训练流程（简化）
class MedPaLMTrainer:
    def __init__(self):
        # 数据来源
        self.sources = {
            "medical_qa": "MedQA, USMLE, MedMCQA",       # 考试题
            "clinical_notes": "MIMIC-III, MIMIC-IV",      # 临床记录
            "pubmed": "PubMed 摘要 + 部分全文",           # 论文
            "drug_db": "DrugBank, RxNorm",                # 药物
            "guidelines": "UpToDate, NICE, 临床指南",    # 指南
        }
    
    def train(self):
        # 1) 基础：PaLM 540B 微调
        # 2) Instruction tuning：医学指令
        # 3) RLHF：医生反馈（"helpful, honest, harmless"）
```

**USMLE 结果**：Med-PaLM 1 达到 **60.3%**，接近及格线。

### 2. Med-PaLM 2（Google, 2023）

**在 MedQA 上达到 86.5%**——超过人类医生平均水平：

| 基准 | Med-PaLM 1 | Med-PaLM 2 | 人类专家 |
|---|---|---|---|
| MedQA (USMLE) | 60.3% | **86.5%** | ~87% |
| PubMedQA | 79.0% | **81.8%** | ~78% |
| MedMCQA | 57.6% | **72.3%** | ~70% |

关键改进：

- **MedPrompt**：用精心设计的 prompt 引导推理。
- **Self-consistency**：多次推理取众数。
- **Ensemble**：多个 prompt 模板组合。

### 3. 开源医疗 LLM

```python
# 代表开源模型
MEDICAL_LLMS = {
    "BioMedLM": "Stanford 2.7B",                  # 早期
    "Meditron": "EPFL 7B/70B",                    # 开源 SOTA
    "ClinicalBERT": "医学临床记录 BERT",
    "BioGPT": "Microsoft 医学生成",               # 1.5B
    "Llama-3-Med": "LLaMA-3 微调医学版",
    "OpenBioLLM": "Llama-3 8B/70B 医学版",
    "HuatuoGPT": "华佗 GPT（中文）",
    "MedicalGPT-zh": "中文医疗 GPT",
}
```

## 三、核心应用场景

### 1. 临床决策支持（CDS）

```python
def clinical_decision_support(patient_data: dict, query: str) -> str:
    """临床决策支持。"""
    prompt = f"""你是一名医疗 AI 助手。基于以下患者信息提供临床建议。

[患者信息]
- 年龄/性别: {patient_data['age']}/{patient_data['gender']}
- 主诉: {patient_data['chief_complaint']}
- 病史: {patient_data['history']}
- 生命体征: {patient_data['vitals']}
- 化验结果: {patient_data['labs']}
- 影像: {patient_data['imaging']}

[问题]
{query}

[回答规则]
1. 必须严格基于患者实际信息
2. 列出可能的鉴别诊断（differential diagnosis）
3. 推荐进一步检查
4. 治疗建议必须含明确证据等级
5. 标记任何红旗症状（red flags）
6. 明确说"最终决策需主治医生确认"

回答:"""
    return llm.generate(prompt)
```

**输出示例**：

```text
[可能的鉴别诊断]
1. 急性冠脉综合征（高 - 基于胸痛 + 心电图 ST 段变化）
2. 主动脉夹层（中 - 需排除）
3. 肺栓塞（中 - D-dimer 升高）

[推荐检查]
- 立即：肌钙蛋白 I（动态监测）
- 紧急：胸部 CTA（排除 PE / 夹层）

[红旗症状]
- 胸痛伴放射至左臂 + 出汗 = 高度怀疑 ACS
- 任何晕厥需立即处理

[治疗建议（证据等级 B）]
- 阿司匹林 300mg 即刻
- 准备 PCI

⚠️ 此为 AI 辅助建议，最终决策需主治医生确认。
```

### 2. 病历生成与结构化

#### SOAP 病历生成

```python
SOAP_PROMPT = """基于以下医患对话，生成 SOAP 病历：

S (Subjective - 主诉):
  患者自述的症状、病史

O (Objective - 客观):
  体格检查、生命体征、化验、影像

A (Assessment - 评估):
  鉴别诊断、当前主要诊断

P (Plan - 计划):
  检查、治疗、随访

对话:
{dialogue}

SOAP 病历:"""
```

#### 病历结构化抽取

```python
STRUCTURED_NOTE_PROMPT = """从以下非结构化病历中提取结构化字段：

{clinical_note}

输出 JSON:
{{
    "diagnoses": ["ICD-10 编码及名称"],
    "medications": [{{"name": "...", "dose": "...", "frequency": "..."}}],
    "allergies": [...],
    "vitals": {{"bp": "...", "hr": "...", "temp": "...", ...}},
    "lab_results": [{{"name": "...", "value": "...", "unit": "...", "flag": "..."}}],
    "follow_up": "..."
}}"""
```

### 3. 患者教育与问答

```python
PATIENT_EDUCATION_PROMPT = """用通俗易懂的语言（5-8 年级水平）回答患者问题。

问题: {patient_question}

规则：
1. 不使用专业术语（如必须用，括号解释）
2. 不替代医生诊断
3. 提供可执行建议
4. 提示何时该看医生
5. 包含本地化（中国患者视角）"""
```

**示例**：

```text
患者: "医生说我有高血压，我需要注意什么？"

回答（通俗版）:
"高血压就像水管压力太大，时间久了水管会爆。你的血压高了，心脏和血管也类似。

日常注意：
1. **少吃盐**（每天 < 5g），多用蒸煮，少吃咸菜腌肉
2. **控制体重**：每减 10 公斤，血压可能降 5-10 mmHg
3. **规律运动**：每天 30 分钟快走
4. **少喝酒**：男性每天 < 25g 酒精
5. **定期量血压**：建议家庭血压计，早晚各一次

什么时候立即就医？
- 血压 > 180/120
- 胸痛、头痛、视力模糊
- 一侧肢体无力

⚠️ 本建议不能替代医生的诊疗方案。具体用药请遵医嘱。
"""
```

### 4. 药物咨询与相互作用检查

```python
def drug_interaction_check(medications: list) -> dict:
    """检查药物相互作用。"""
    prompt = f"""检查以下药物之间的相互作用：

药物列表: {medications}

输出 JSON:
{{
    "interactions": [
        {{
            "drug_a": "...",
            "drug_b": "...",
            "severity": "严重/中度/轻微",
            "description": "...",
            "recommendation": "..."
        }}
    ],
    "contraindications": [...],
    "warnings": [...]
}}"""
    return llm.generate(prompt)
```

### 5. 医学文献总结

```python
def summarize_paper(paper_text: str) -> str:
    """用临床医生视角总结论文。"""
    prompt = f"""用结构化方式总结以下医学论文：

{paper_text}

输出：
1. 研究问题
2. 研究设计（RCT / 队列 / 病例对照 / 横断面）
3. 样本量、随访期
4. 主要结果（效应大小、95% CI、p 值）
5. 临床意义（NNT / NNH）
6. 局限性
7. 对临床实践的影响
8. 证据等级（1a / 1b / 2a / 2b / 3 / 4 / 5）"""
    return llm.generate(prompt)
```

### 6. 多模态医学（图像 + 文本）

```python
class MultimodalMedicalAI:
    def __init__(self, vlm, llm):
        self.vlm = vlm  # 视觉语言模型（如 LLaVA-Med）
        self.llm = llm
    
    def analyze(self, image: bytes, text: str) -> str:
        """多模态分析。"""
        # 1) 视觉模型描述图像
        image_desc = self.vlm.describe(image)
        
        # 2) 整合文本与图像描述
        return self.llm.generate(f"""基于以下信息给出临床判断：

[医学影像描述]
{image_desc}

[临床信息]
{text}

请给出综合分析。""")
```

## 四、评测基准

### 1. MedQA（USMLE）

```python
# MedQA 例子
QUESTION = """A 55-year-old man comes to the emergency department because of 
sudden onset chest pain and shortness of breath for 2 hours. He has a history 
of hypertension and smoking. Blood pressure is 90/60, heart rate is 110, 
respiratory rate is 24. ECG shows ST-segment elevation in leads II, III, aVF. 
What is the most likely diagnosis?

A) Anterior STEMI
B) Inferior STEMI
C) Pulmonary embolism
D) Aortic dissection

Answer: B
"""
```

### 2. PubMedQA

生物医学文献的问答——测试模型对论文的理解。

### 3. MedMCQA

印度医学院入学考试——多语种覆盖。

### 4. MultiMedQA

Google 综合评测集：MedQA + PubMedQA + MedMCQA + 其他。

### 5. 临床推理评测

```python
def clinical_reasoning_eval(model, case):
    """评估临床推理能力。"""
    # 给出完整病历，让模型推理
    response = model.generate(case["prompt"])
    
    # 评估
    score = {
        "diagnosis_accuracy": match_diagnosis(response, case["correct_diagnosis"]),
        "differential_complete": check_differential(response, case["differentials"]),
        "reasoning_quality": llm_judge(response, case["expert_response"]),
        "safety": check_red_flags(response),
    }
    return score
```

## 五、监管合规

### 1. FDA 监管路径

美国 FDA 把 AI/ML 医疗设备按风险分级：

| 类别 | 例子 | FDA 路径 |
|---|---|---|
| **Class I**（低风险） | 一般健康追踪 | 大部分豁免 |
| **Class II**（中风险） | 影像辅助诊断 | 510(k) 上市前通知 |
| **Class III**（高风险） | 自动诊断、独立治疗决策 | PMA（上市前批准） |

**关键概念**：

- **SaMD**（Software as Medical Device）：软件即医疗器械。
- **PCCP**（Predetermined Change Control Plan）：预定变更计划（AI 模型持续更新）。

### 2. 代表审批案例

- **IDx-DR**（2018）：第一个 FDA 批准的 AI 诊断设备——糖尿病视网膜病变筛查。
- **Viz.ai**（2018）：卒中 LVO 检测，缩短治疗时间。
- **Paige.AI**（2021）：前列腺癌检测。
- **Apple Watch ECG**（2018）：房颤检测。

### 3. HIPAA 合规

```python
class HIPAAGuard:
    """HIPAA 合规检查。"""
    
    PHI_FIELDS = [
        "name", "address", "ssn", "phone", "email",
        "dob", "mrn", "ip_address", "biometric",
    ]
    
    def sanitize(self, data: dict) -> dict:
        """去除 PHI 字段。"""
        sanitized = {}
        for key, value in data.items():
            if key.lower() in self.PHI_FIELDS:
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize(value)
            else:
                sanitized[key] = value
        return sanitized
    
    def audit_log(self, action: str, user_id: str, patient_id: str):
        """HIPAA 要求的所有 PHI 访问必须记录。"""
        log_entry = {
            "timestamp": datetime.now(),
            "user": user_id,
            "patient": patient_id,
            "action": action,
            "ip": request.ip,
        }
        audit_db.insert(log_entry)
```

## 六、医院部署案例

### 1. HCA Healthcare

美国最大医院集团之一，与 Google Cloud + Augmedix 合作：
- 病历自动生成（医生口述 → 结构化病历）。
- 减少 30~50% 的病历时间。

### 2. Mayo Clinic

与 Google Health 合作：
- 放射学 AI 辅助。
- 病理学 AI 辅助。
- 心血管风险预测。

### 3. Epic + Microsoft Nuance（DAX）

```python
# DAX Copilot 风格
# 医生问诊 → AI 自动生成病历草稿
def dax_copilot(doctor_patient_conversation: str) -> str:
    return llm.generate(f"""从以下医患对话生成临床病历：

{dialogue}

要求：
- 使用 SOAP 格式
- 用标准医学术语（SNOMED CT / ICD-10）
- 提取关键临床决策
- 包含必要的客观数据""")
```

### 4. Johns Hopkins

与微软合作部署 AI 文档助手，**减少医生文档时间 50%+**。

## 七、安全与伦理

### 1. 幻觉风险

医疗幻觉是**最危险的 LLM 错误**：

```text
患者: "我怀孕 4 周，吃了布洛芬"
LLM: "继续吃就好"  ← 完全错误！
正确: "布洛芬在孕中晚期可能导致胎儿心脏问题，请立即停药并咨询医生"
```

**缓解**：
- **RAG**：所有药物建议基于药品说明书。
- **强制引用**：每个建议都有文献支持。
- **人类最终决策**：医生 review 所有 AI 输出。
- **领域训练**：医疗专用模型减少幻觉。

### 2. 公平性

研究显示医疗 AI 可能对**少数族裔、低收入**群体表现差：
- 皮肤科 AI：在深色皮肤上准确率显著下降。
- 心血管 AI：在女性 / 黑人患者中漏诊率高。

### 3. 责任归属

AI 出错时谁负责？

| 角色 | 责任 |
|---|---|
| 医生 | 最终诊断与治疗决策 |
| 医院 | 系统部署与监管 |
| AI 厂商 | 模型准确性与安全 |
| 监管 | 标准制定与审批 |

### 4. 数据隐私

医疗数据是**最敏感**的个人数据——必须严格保护：
- 端到端加密。
- 本地部署 / 私有云。
- 联邦学习 / 差分隐私。

## 八、未来方向

### 1. 多模态医疗 AI

```text
文本 + 影像 + 基因组 + 临床数据 + 时序信号
        ↓
   综合诊断与个性化治疗
```

### 2. 自主诊断（高水平监管）

当前 AI 多为**辅助**，未来可能有 FDA 批准的**自主诊断**——但监管极其严格。

### 3. 患者端 AI

- **症状自查**：引导患者判断是否需要就医。
- **慢病管理**：糖尿病、高血压的日常管理。
- **心理健康**：CBT 风格的 AI 心理治疗。

### 4. 药物研发加速

医疗 LLM + 蛋白质预测（AlphaFold）+ 临床数据 → 加速新药发现。

## 九、给医疗 AI 团队的清单

1. **领域专用模型**：用医疗专用模型而非通用 LLM。
2. **RAG 优先**：所有事实性回答基于权威医学文献。
3. **强制引用**：每个建议都有 ICD-10 / 文献支持。
4. **人类最终决策**：医生必须 review AI 输出。
5. **FDA 路径**：明确产品分类与监管路径。
6. **HIPAA 合规**：PHI 保护、审计日志、加密。
7. **公平性审计**：测试不同人群的 AI 表现。
8. **持续监控**：模型漂移、效果衰减。
9. **灾备方案**：AI 故障时人工接管流程。
10. **透明度**：明确告诉医生和患者"这是 AI 建议"。

## 小结

医疗 LLM 是 LLM 应用**最有价值也最危险**的领域——Med-PaLM 2 在 MedQA 上已达人类专家水平，GPT-4 通过 USMLE 等考试。**六大应用场景**——临床决策支持、病历生成、患者教育、药物咨询、文献总结、多模态诊断——各有侧重。**核心挑战是幻觉、公平性、责任归属、监管合规**。**真正成功的医疗 AI** 不是"AI 替代医生"，而是"AI 辅助医生，让医生专注于真正需要人的判断"。**FDA 严格监管 + HIPAA 数据保护 + 医生最终决策**是铁三角。下一篇我们将看到医疗 AI 的另一关键方向——**医学影像 AI**。
