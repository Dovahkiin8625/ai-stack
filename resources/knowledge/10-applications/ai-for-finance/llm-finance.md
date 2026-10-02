# LLM 在金融中的应用：从研究到风控

LLM 在金融领域的应用**范围广、价值高、风险也高**。从卖方研究的财报分析、买方投资的尽调、风控的合同审查，到客服、合规、欺诈检测——LLM 已经渗透金融的全链路。本文系统梳理 LLM 在金融中的六大应用场景、典型代表（BloombergGPT、FinGPT、Stripe、Klarna）、关键风险（幻觉、监管、内幕信息），以及行业级落地实践。

## 一、金融 LLM 的特殊性

金融对 LLM 的要求与一般应用**显著不同**：

| 维度 | 一般应用 | 金融应用 |
|---|---|---|
| 准确率要求 | 80%+ 可接受 | **99%+ 是底线**（幻觉 = 金钱损失） |
| 时效性 | 可日级 | **分钟级**（市场瞬息万变） |
| 数据来源 | 公开 | 大量私有 + 内幕信息边界 |
| 监管 | 宽松 | **严格**（SEC、FINRA、央行） |
| 责任归属 | 模糊 | **明确**（银行要承担损失） |
| 可解释性 | 一般 | **必须**（监管要求） |

这导致金融 LLM 必须**格外严谨**——RAG + 多模型验证 + 人类审核是标配。

## 二、训练金融 LLM：BloombergGPT 与 FinGPT

### 1. BloombergGPT（Bloomberg, 2023）

**首个金融领域专用 LLM**：

| 指标 | BloombergGPT | 同规模通用 LLM |
|---|---|---|
| 训练数据 | 363B token 金融文本 + 345B token 通用 | 全部通用 |
| 参数量 | 50B | ~50B |
| 金融任务 | **显著优于通用** | 基线 |
| 通用任务 | 接近专用 | 基线 |

关键数据来源：

```python
FIN_DATA_SOURCES = {
    "news": ["Bloomberg News", "Reuters", "Dow Jones"],
    "filings": ["SEC EDGAR 10-K/10-Q", "财报", "招股书"],
    "press": ["公司新闻稿", "投资者关系材料"],
    "research": ["卖方研究报告", "学术论文"],
}
```

**教训**：专用领域训练在**领域任务**上效果好，但**通用能力**会下降——需要混合训练。

### 2. FinGPT（AI4Finance, 2023）

**开源的金融 LLM**——支持低资源微调：

```python
from peft import LoraConfig, get_peft_model

# 用 LoRA 在金融数据上微调 LLaMA
lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4.2M || all params: 6.7B || trainable%: 0.06%
```

**优势**：LoRA 微调成本低，普通 GPU 也能训练。

### 3. 中文金融 LLM

- **DISC-FinLLM**（复旦）：中文金融 LLM。
- **Tongyi-Finance**（阿里）：通义金融版。
- **Xuanyuan**（中科院）：轩辕系列。

中文金融数据挑战：
- 繁体 vs 简体。
- A 股 vs 港股 vs 美股差异。
- 中国会计准则 vs 国际准则差异。

## 三、六大核心应用场景

### 1. 财报分析与摘要

```python
def analyze_10k_filing(filing_text: str) -> dict:
    """用 LLM 分析 10-K 年报。"""
    prompt = f"""分析以下 10-K 年报，提取关键信息：

{filing_text}

输出 JSON：
{{
    "revenue_growth_yoy": <float, 同比营收增长>,
    "key_risks": ["..."],
    "business_segments": {{"segment": "revenue_share"}},
    "management_outlook": "...",
    "red_flags": ["..."]  # 任何可疑指标
}}
"""
    return llm.generate(prompt, response_format={"type": "json_object"})
```

**典型任务**：
- 提取关键财务指标。
- 识别风险因素（Risk Factors）。
- 总结管理层展望（MD&A）。
- 对比同期 / 同业。

### 2. 投资研究与尽调

```python
DUE_DILIGENCE_PROMPT = """为以下公司做投资尽调：

公司：{company_name}
行业：{industry}

请基于公开信息分析：
1. 商业模式（核心收入来源、客户类型）
2. 竞争格局（主要竞品、护城河）
3. 财务健康（最近 3 年关键指标）
4. 管理层背景（核心高管经验）
5. 主要风险（监管、市场、技术）
6. 估值合理性（与同业对比）

输出结构化报告。"""
```

**优势**：从 PB 级的财报、新闻、研报中**自动抽取关键信息**。

### 3. 风控与合规审查

#### 反洗钱（AML）

```python
def aml_transaction_review(transaction: dict) -> dict:
    """审查可疑交易。"""
    prompt = f"""分析以下交易是否可疑（洗钱/恐怖融资风险）：

交易: {json.dumps(transaction)}

请检查：
1. 金额异常（大额、整数、刚好低于报告阈值）
2. 频率异常（短时间内多次）
3. 模式异常（拆分、结构化）
4. 客户背景（与 PEP 关联？）
5. 资金来源/用途（与客户画像一致？）

输出风险评分 (0-100) 和理由。"""
    return llm.generate(prompt)
```

#### 合规审查

```python
def review_contract(contract: str) -> dict:
    """审查合同条款。"""
    prompt = f"""审查以下合同：

{contract}

提取：
1. 关键条款（金额、期限、违约责任）
2. 风险条款（对己方不利的）
3. 不合规条款（违反监管要求）
4. 建议修改"""
    return llm.generate(prompt)
```

### 4. 欺诈检测

```python
def detect_fraud_pattern(transaction_history: list[dict], profile: dict) -> dict:
    """检测欺诈模式。"""
    prompt = f"""分析用户交易历史，识别欺诈模式：

用户画像: {profile}
交易历史: {transaction_history}

请识别：
1. 异常消费（金额、地点、商家类别）
2. 账户接管迹象（IP 异常、设备变化）
3. 套现模式（短时间内大额转入转出）
4. 合成身份迹象

输出欺诈概率 (0-1) 和理由。"""
    return llm.generate(prompt)
```

### 5. 客户服务

金融客服是 LLM 最早大规模应用的场景：

- **Klarna**：AI 客服处理 2/3 客服对话，等效 700 人工客服工作量。
- **Stripe**：用 LLM 帮商户理解交易问题。
- **Bank of America**：Erica 助手服务超 2000 万客户。

```python
def banking_support(user_query: str, user_context: dict) -> str:
    """银行客服 AI。"""
    prompt = f"""你是银行客服，名为 Olivia。

用户身份: {user_context['profile']}  # 不含敏感字段
账户摘要: {user_context['account_summary']}

规则：
- 不透露账户具体余额给"他"
- 涉及交易争议转人工
- 涉及投资建议转人工
- 语气专业、亲切

用户: {user_query}"""
    return llm.generate(prompt)
```

### 6. 算法交易与量化研究

详见后续文章。

## 四、金融 LLM 的关键风险

### 1. 幻觉（Hallucination）

**金融幻觉特别危险**：

```text
❌ 一般 LLM 幻觉: "巴黎是德国首都" → 用户可能识破
❌ 金融 LLM 幻觉: "AAPL 上季度营收 $200B" → 用户可能据此决策
```

**缓解**：
- **强制 RAG**：所有事实性回答必须基于检索。
- **数值交叉验证**：用结构化数据核对 LLM 输出。
- **置信度标注**：每个数字标"来源 + 可信度"。

```python
def finance_answer_with_citations(question: str, source_docs: list) -> str:
    """带引用的金融回答。"""
    prompt = f"""回答以下金融问题，必须严格基于提供的资料。

问题: {question}

资料:
{chr(10).join([f"[{i}] {d['text']}" for i, d in enumerate(source_docs)])}

规则：
- 每个事实必须引用 [数字]
- 不能编造数据
- 如资料不足，明确说"资料未提及"
- 用中文回答"""
    return llm.generate(prompt)
```

### 2. 内幕信息边界（MNPI）

**LLM 可能无意中暴露内幕信息**——这是监管红线：

```python
class MNPIFilter:
    """Material Non-Public Information 过滤器。"""
    
    def __init__(self, user_role: str):
        self.user_role = user_role  # "internal", "external", "compliance"
        self.restricted_topics = []
    
    def filter_response(self, response: str, context: dict) -> str:
        """根据用户角色过滤敏感信息。"""
        if self.user_role == "external":
            # 检查是否含未发布信息
            for topic in self.restricted_topics:
                if topic in response:
                    return "[信息受限]"
        return response
```

### 3. 监管合规

各地区都有金融 AI 法规：

| 地区 | 法规 | 关键要求 |
|---|---|---|
| **美国** | SEC AI Guidance | 解释性、公平性 |
| **欧盟** | AI Act + DORA | 高风险 AI 严格审查 |
| **中国** | 金融 AI 管理办法 | 算法备案、可解释 |
| **英国** | FCA AI Update | 治理框架、风险管理 |

### 4. 模型偏差

金融模型可能对**特定人群**不公：

- 信贷模型拒绝率：少数族裔更高（ProPublica 报道）。
- 保险定价：女性 vs 男性。
- 投资建议：富人有更好建议。

**缓解**：公平性审计、监管审查、人类监督。

## 五、BloombergGPT 的架构与训练

```python
# BloombergGPT 训练 pipeline（简化）
class BloombergGPTTrainer:
    def __init__(self):
        # 数据：50% Bloomberg 金融数据 + 50% 通用
        self.mix = {
            "bloomberg_news": 0.20,    # Bloomberg 新闻
            "filings": 0.10,           # 财报
            "press": 0.05,             # 公司公告
            "web_general": 0.50,       # 通用网络
            "books": 0.15,             # 书籍
        }
    
    def train(self):
        # 1) Tokenizer 训练（金融专用）
        self.train_tokenizer()
        
        # 2) 标准 CLM 预训练
        for step in range(total_steps):
            batch = self.sample_batch(self.mix)
            loss = self.clm_loss(batch)
            loss.backward()
            self.optimizer.step()
        
        # 3) 金融任务微调
        for task in ["sentiment", "ner", "summarization", "qa"]:
            self.finetune(task)
```

**训练成本**：约 **$2.6M**（Bloomberg 自有数据 + 算力）。

## 六、FinGPT 的开源实践

```python
# FinGPT 用 LoRA 微调 LLaMA 的代码骨架
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# 1) 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2) LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3) 加 LoRA
model = get_peft_model(base_model, lora_config)

# 4) 加载金融数据
dataset = load_dataset("FinGPT/fingpt-sentiment-train")

# 5) 训练
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    args=TrainingArguments(
        output_dir="./fingpt-output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
    ),
)
trainer.train()
```

**FinGPT-Fineval 评测基准**：

```python
FINEVAL_TASKS = {
    "sentiment_analysis": "金融文本情感分类",
    "headline_classification": "新闻标题分类",
    "named_entity_recognition": "金融实体识别",
    "relation_extraction": "实体关系抽取",
    "summarization": "金融摘要",
    "qa": "金融问答",
}
```

## 七、Stripe 的 LLM 应用

Stripe 在多个层面用 LLM：

### 1. 文档问答（支持代理）

```python
def support_doc_qa(query: str, doc_corpus: list) -> str:
    """Stripe 风格：从文档中回答商户问题。"""
    # 1) 检索相关文档
    relevant_docs = retrieve(query, doc_corpus, k=5)
    
    # 2) LLM 合成答案
    return llm.generate(f"""基于以下 Stripe 文档回答问题。
如资料不足，明确说"请咨询 Stripe 支持"。

问题: {query}
资料: {relevant_docs}

回答:""")
```

### 2. 风险评估

```python
def merchant_risk_assessment(merchant_data: dict) -> dict:
    """评估商户欺诈风险。"""
    return llm.generate(f"""基于以下信息评估该商户的欺诈/退单风险：

商户: {merchant_data}

请分析：
1. 行业风险水平
2. 经营时间长度
3. 交易模式异常
4. 地域风险
5. 历史退单率

输出风险评分 (0-100) 和建议。""")
```

### 3. Radar 风控（早期 ML 模型）

Stripe Radar 用深度学习做交易欺诈检测，**~30B 美元/年的支付**——AI 必须零差错。

## 八、Klarna 的客服 AI

Klarna 2024 年用 LLM 替代 700 名客服：

```python
def klarna_support_agent(customer_query: str, customer_history: dict) -> str:
    """Klarna 风格客服。"""
    prompt = f"""你是 Klarna 的 AI 客服 Olivia。

客户查询: {customer_query}
客户背景: {customer_history}

可执行的操作：
- 查询订单状态
- 处理退款
- 解释账单
- 处理分期付款问题
- 转人工客服

规则：
- 礼貌、专业
- 不承诺超出权限的内容
- 涉及投诉或复杂情况 → 转人工
- 涉及支付信息 → 不直接处理，引导到安全通道"""
    return llm.generate(prompt)
```

**效果（Klarna 自报）**：
- 2/3 客服对话由 AI 处理。
- 平均处理时间从 11 分钟 → 2 分钟。
- 客户满意度与人工相当。

**争议**：可能影响 700 名客服就业——**AI 替代 vs AI 增强**是社会议题。

## 九、合规与审计

金融 AI 必须**可审计**：

### 1. 决策日志

```python
def make_decision_with_audit(
    customer_id: str,
    decision_type: str,    # "loan", "trade", "fraud_alert"
    inputs: dict,
    decision: str,
    reasoning: str,
    model_version: str,
):
    """记录所有 AI 决策便于审计。"""
    audit_log = {
        "timestamp": datetime.now().isoformat(),
        "customer_id": customer_id,
        "decision_type": decision_type,
        "inputs": inputs,
        "decision": decision,
        "reasoning": reasoning,
        "model_version": model_version,
        "human_reviewed": False,
    }
    audit_db.insert(audit_log)
```

### 2. 可解释性

```python
def explain_decision(decision: dict) -> str:
    """生成决策解释。"""
    return llm.generate(f"""用通俗语言解释以下决策：

决策: {decision['decision']}
输入: {decision['inputs']}
推理: {decision['reasoning']}

要求：
- 不超过 100 字
- 突出关键因素
- 客户能理解""")
```

### 3. 偏见检测

```python
def check_decision_bias(decisions: pd.DataFrame, protected_attrs: list) -> dict:
    """检查决策中是否有群体偏差。"""
    results = {}
    for attr in protected_attrs:
        groups = decisions.groupby(attr)
        approval_rates = groups["decision"].apply(
            lambda x: (x == "approved").mean()
        )
        # 群体间批准率差距
        gap = approval_rates.max() - approval_rates.min()
        results[attr] = {
            "rates": approval_rates.to_dict(),
            "gap": gap,
            "flagged": gap > 0.1,  # 10% 差距触发警报
        }
    return results
```

## 十、未来方向

### 1. 多模态金融 AI

- 财报图片 → 结构化数据（OCR + LLM）。
- 视频财报会议 → 摘要 + Q&A。
- K 线图 → 趋势分析。

### 2. 实时市场分析

- 实时新闻流 → 自动交易信号。
- 社交媒体情绪 → 短期价格预测。
- 中央银行讲话 → 政策影响分析。

### 3. 个人化金融助手

- 财富管理 AI（wealth tech）。
- 税务规划助手。
- 退休规划。

### 4. 跨机构协作

- 多机构联合风控（合规 + 反洗钱）。
- 跨境支付 AI。

## 十一、给金融团队的清单

1. **建立金融 RAG pipeline**：所有事实性回答必须基于检索。
2. **多层验证**：LLM 输出必须经结构化数据校验。
3. **审计日志**：所有决策记录可追溯。
4. **可解释性**：监管要求的解释必须可生成。
5. **偏见审计**：定期检查不同人群的决策差异。
6. **MNPI 隔离**：敏感信息不能用于训练。
7. **人类最终决策**：高 stakes 决策必须有人 review。
8. **监管合规**：跟踪 SEC、央行、AI Act 等法规。
9. **灾备**：AI 故障时人工接管流程。
10. **持续监控**：模型漂移、数据漂移。

## 小结

LLM 在金融中已经从研究 → 落地，从单点应用 → 全链路集成。BloombergGPT、FinGPT 等专用模型在金融任务上**显著优于通用模型**，但幻觉风险更高。**六大应用场景**——财报分析、研究尽调、风控合规、欺诈检测、客服、交易——各有侧重。**核心挑战是幻觉、监管、内幕信息边界与公平性**。**真正成功的金融 LLM** 是"AI 决策 + 人类监督 + 严格审计"的组合。下一篇我们将看到金融 RAG 的具体实现——如何让 LLM 基于实时市场数据给出准确回答。
