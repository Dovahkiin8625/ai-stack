# 金融 RAG：让 LLM 基于实时市场数据回答

金融数据的**实时性、准确性、可追溯性**是 LLM 在金融应用的核心挑战——模型可能用过期信息、编造数字、忽略监管披露。**金融 RAG（Financial Retrieval-Augmented Generation）** 把 LLM 与实时数据源结合：市场行情、财报披露、新闻、研报。本文系统介绍金融 RAG 的数据源、embedding、检索策略、合规要求，以及工业级实现（Bloomberg、AlphaSense、Ramp）。

## 一、金融 RAG 的特殊挑战

金融 RAG 与一般 RAG 相比有几大差异：

### 1. 数据时效性

```text
普通 RAG: 数据可日级更新
金融 RAG: 分钟级实时（股价、新闻）
       : 事件级实时（财报、公告）
       : 历史数据回溯（10 年财报）
```

### 2. 数字精度

```text
普通 RAG: "Python 是一种编程语言"  ← 模糊表述可接受
金融 RAG: "AAPL 当前价 $195.34"   ← 必须精确到分
```

### 3. 数据源多样性

金融数据包括：

- **结构化**：股价、财务指标、宏观经济数据。
- **半结构化**：财报 HTML、监管 XML。
- **非结构化**：新闻、研报、社交媒体。

### 4. 强可追溯性

每个数字必须能追溯到**具体数据源与时间**——监管要求。

### 5. 多语言与多市场

A 股、港股、美股各用不同会计准则；中文、英文研报并存。

## 二、金融数据源全景

### 1. 结构化数据

```python
FINANCIAL_DATA_SOURCES = {
    # 市场数据
    "real_time_quotes": {
        "provider": ["Bloomberg", "Refinitiv", "Wind", "Tushare"],
        "latency": "ms 级",
        "cost": "$$$ (Bloomberg) ~ $ (Tushare)",
    },
    
    # 财务数据
    "fundamentals": {
        "provider": ["Bloomberg", "FactSet", "Capital IQ", "SEC EDGAR"],
        "update": "季度 / 年报发布后",
        "data": ["营收", "净利润", "ROE", "现金流"],
    },
    
    # 宏观经济
    "macro": {
        "provider": ["FRED", "World Bank", "国家统计局"],
        "data": ["GDP", "CPI", "失业率", "利率"],
    },
    
    # 另类数据
    "alternative": {
        "satellite": "卫星图像（停车场车辆）",
        "credit_card": "信用卡消费数据",
        "social_media": "Twitter/Reddit 情绪",
        "geolocation": "人流数据",
    },
}
```

### 2. 非结构化数据

```python
UNSTRUCTURED_SOURCES = {
    # 监管披露
    "SEC_filings": ["10-K", "10-Q", "8-K", "DEF 14A", "招股书"],
    "HKEX_filings": ["年报", "中期报告", "公告"],
    "CSRC_filings": ["A股年报", "季报", "重大事项公告"],
    
    # 新闻
    "news": ["Bloomberg News", "Reuters", "WSJ", "财新", "路透中文"],
    
    # 研究报告
    "research": ["卖方研报", "独立研究机构", "学术论文"],
    
    # 公司自有材料
    "company_materials": ["投资者电话会议", "业绩发布会", "IR 网站"],
}
```

## 三、金融 RAG 架构

```text
┌─────────────────────────────────────────────────────────┐
│  1. Ingestion（数据接入）                                  │
│     - 实时：行情、新闻流                                    │
│     - 离线：财报、研报、监管披露                              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  2. Processing（数据处理）                                  │
│     - 解析 PDF / HTML / XBRL                              │
│     - 切分 + 元数据标注                                     │
│     - 标准化（公司名、日期、币种）                            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  3. Indexing（索引）                                       │
│     - 文本 embedding（语义检索）                            │
│     - 关键词索引（BM25 / Elasticsearch）                   │
│     - 结构化数据 SQL（财务指标查询）                          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  4. Retrieval（检索）                                      │
│     - 多路召回（语义 + 关键词 + 结构化）                     │
│     - 重排序                                              │
│     - 时间过滤 / 来源过滤                                   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  5. Generation（生成）                                     │
│     - 拼装 prompt                                          │
│     - 强制引用                                             │
│     - 输出验证                                             │
└─────────────────────────────────────────────────────────┘
```

## 四、数据接入与处理

### 1. PDF 财报解析

财报是金融 RAG 的核心数据源——多为 PDF：

```python
import pdfplumber
import re

def parse_10k(pdf_path: str) -> list[dict]:
    """解析 10-K PDF。"""
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join([page.extract_text() for page in pdf.pages])
    
    # 提取关键 sections（按 SEC 规定）
    section_patterns = {
        "business": r"ITEM 1\.\s+BUSINESS(.+?)ITEM 1A",
        "risk_factors": r"ITEM 1A\.\s+RISK FACTORS(.+?)ITEM 1B",
        "financials": r"ITEM 8\.\s+FINANCIAL STATEMENTS(.+?)ITEM 9",
        "mda": r"ITEM 7\.\s+MANAGEMENT'S DISCUSSION(.+?)ITEM 7A",
    }
    
    for section, pattern in section_patterns.items():
        match = re.search(pattern, full_text, re.DOTALL)
        if match:
            sections.append({
                "section": section,
                "text": match.group(1).strip()[:50000],  # 限制长度
                "source": pdf_path,
                "filing_type": "10-K",
            })
    
    return sections
```

### 2. XBRL 解析（结构化财务数据）

XBRL 是 SEC 财报的标准格式：

```python
from lxml import etree

def parse_xbrl_facts(xbrl_path: str) -> dict:
    """从 XBRL 提取结构化财务事实。"""
    tree = etree.parse(xbrl_path)
    root = tree.getroot()
    
    facts = {}
    for fact in root.iter():
        if 'us-gaap:' in fact.tag or 'dei:' in fact.tag:
            tag = fact.tag.split('}')[1]
            value = fact.text
            
            # 单位与上下文
            context_ref = fact.get('contextRef', '')
            unit_ref = fact.get('unitRef', 'USD')
            
            facts[tag] = {
                "value": value,
                "context": context_ref,
                "unit": unit_ref,
            }
    
    return facts


# 使用
facts = parse_xbrl_facts("aapl_10k.xml")
revenue = facts.get("us-gaap:Revenues", {})
print(f"AAPL Revenue: ${revenue['value']} {revenue['unit']}")
```

### 3. 实时新闻流接入

```python
import asyncio
import aiohttp
from datetime import datetime

class NewsStream:
    """实时新闻流订阅。"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
    
    async def connect(self):
        self.session = aiohttp.ClientSession()
        url = f"https://newsapi.example.com/stream?key={self.api_key}"
        async with self.session.get(url) as resp:
            async for line in resp.content:
                article = parse_news(line)
                yield article
    
    async def index_article(self, article):
        """索引新文章。"""
        embedding = embed_text(article["title"] + " " + article["content"])
        self.vector_db.upsert(
            id=article["id"],
            vector=embedding,
            metadata={
                "source": article["source"],
                "timestamp": article["published_at"],
                "tickers": extract_tickers(article),
                "title": article["title"],
            }
        )
```

## 五、Embedding 与索引

### 1. 金融专用 Embedding

通用 embedding 对金融术语理解弱——用金融专用模型：

```python
# FinE5 / FinBERT / BloombergGPT 的 embedding
from sentence_transformers import SentenceTransformer

# 推荐模型
models = {
    "fin-e5": "FinGPT/fin-e5",                  # 中文金融
    "fin-bert": "ProsusAI/finbert",             # 英文金融情感
    "voyage-finance": "voyage-finance-2",       # 商业
    "bge-m3": "BAAI/bge-m3",                   # 通用但中文强
}

embedder = SentenceTransformer("FinGPT/fin-e5")
```

### 2. 混合检索

金融 RAG 需要同时支持**语义检索**（"增长稳健的科技股"）和**精确查询**（"AAPL 2023Q4 营收"）：

```python
class HybridFinanceRetriever:
    def __init__(self, vector_db, elastic, sql_db):
        self.vector_db = vector_db    # 向量检索
        self.elastic = elastic        # 关键词检索
        self.sql_db = sql_db          # 结构化查询
    
    def retrieve(self, query: str, k=10) -> list[dict]:
        results = []
        
        # 1) 结构化查询（识别 query 中的数字、公司、时间）
        if has_structured_query(query):
            sql_result = self.parse_and_query_sql(query)
            results.extend(sql_result)
        
        # 2) 关键词检索（精确匹配）
        bm25_results = self.elastic.search(query, top_k=k)
        results.extend(bm25_results)
        
        # 3) 语义检索（理解意图）
        q_emb = embedder.encode(query)
        vec_results = self.vector_db.search(q_emb, k=k)
        results.extend(vec_results)
        
        # 4) 重排序 + 去重
        results = self.rerank(query, results)[:k]
        return results
    
    def parse_and_query_sql(self, query: str) -> list:
        """把自然语言转为 SQL 查询结构化数据。"""
        sql = llm.generate(f"""把以下查询转为 SQL：

查询: {query}

数据库 schema:
- companies(id, name, ticker, sector)
- financials(company_id, period, revenue, net_income, ...)

SQL:""")
        return self.sql_db.execute(sql)
```

### 3. 时间过滤

金融数据**时间敏感**——必须支持时间过滤：

```python
def retrieve_with_time_filter(query, start_date=None, end_date=None, k=10):
    """带时间过滤的检索。"""
    filters = {}
    if start_date:
        filters["timestamp"] = {"$gte": start_date}
    if end_date:
        filters["timestamp"] = {"$lte": end_date}
    
    return vector_db.search(
        query_embedding=q_emb,
        top_k=k,
        filter=filters,
    )
```

## 六、Prompt 构造与生成

### 1. 强制引用

```python
FINANCE_RAG_PROMPT = """你是严谨的金融分析师。基于以下资料回答问题。

严格规则：
1. 所有数字必须有资料引用 [编号]
2. 不能编造数据
3. 如资料不足，明确说"资料未提及"
4. 引用时同时给出数据时间

资料：
{context}

问题: {query}

回答（带引用）:"""
```

### 2. 多模态上下文

```python
def build_finance_prompt(query, retrieved_chunks, charts=None, tables=None):
    """构造多模态金融 prompt。"""
    parts = []
    
    # 1) 文本资料
    for i, chunk in enumerate(retrieved_chunks):
        parts.append(f"[{i+1}] 来源: {chunk['source']}, "
                    f"时间: {chunk['timestamp']}\n{chunk['text']}\n")
    
    # 2) 表格（如有）
    if tables:
        for tbl in tables:
            parts.append(f"[表格] {tbl['title']}\n{tbl['markdown']}\n")
    
    # 3) 图表（如有）
    if charts:
        for chart in charts:
            parts.append(f"[图表] {chart['title']}: {chart['url']}")
    
    return "\n".join(parts)
```

### 3. 输出验证

```python
def validate_finance_output(answer: str, context: list) -> dict:
    """验证 LLM 输出是否有幻觉。"""
    # 1) 提取答案中的数字
    numbers = re.findall(r"\$?[\d,]+\.?\d*", answer)
    
    # 2) 每个数字检查是否在 context 中
    flagged = []
    for num in numbers:
        if num not in str(context):
            flagged.append(num)
    
    # 3) 让 LLM 自我审查
    self_review = llm.generate(f"""审查以下回答是否有编造或错误：

回答: {answer}

如果有，标出具体位置并修正。无误则说 "OK"。""")
    
    return {
        "numbers_flagged": flagged,
        "self_review": self_review,
        "is_valid": len(flagged) == 0 and "OK" in self_review,
    }
```

## 七、典型场景实现

### 1. 财报问答

```python
def answer_earnings_question(question: str, ticker: str) -> str:
    """回答财报问题。"""
    # 1) 检索相关财报
    chunks = retriever.retrieve(
        query=question,
        filter={"ticker": ticker, "filing_type": {"$in": ["10-K", "10-Q"]}},
        k=10,
    )
    
    # 2) 检索相关结构化数据
    metrics = sql_db.query(
        f"SELECT * FROM financials WHERE ticker='{ticker}' "
        f"ORDER BY period DESC LIMIT 8"
    )
    
    # 3) 构造 prompt
    context = build_finance_prompt(question, chunks, tables=[metrics])
    
    # 4) 生成 + 验证
    answer = llm.generate(FINANCE_RAG_PROMPT.format(query=question, context=context))
    validation = validate_finance_output(answer, chunks)
    
    if not validation["is_valid"]:
        answer += "\n\n⚠️ 此回答包含可能需要核实的数据点。"
    
    return answer
```

### 2. 实时市场分析

```python
def realtime_market_analysis(query: str) -> str:
    """实时市场分析。"""
    # 1) 拉取最新行情
    quotes = market_data_api.get_quotes(["AAPL", "MSFT", "GOOGL"])
    
    # 2) 拉取最新新闻
    recent_news = news_api.search(query, last_hours=24)
    
    # 3) 检索历史研报
    historical = retriever.retrieve(query, k=5)
    
    # 4) 合成
    context = f"""
[当前行情]
{quotes.to_markdown()}

[最近 24 小时新闻]
{recent_news}

[相关历史研报]
{historical}
"""
    
    return llm.generate(f"基于以下数据回答：{query}\n\n{context}")
```

### 3. 投资尽调

```python
def due_diligence_report(company: str) -> str:
    """为一家公司生成尽调报告。"""
    # 1) 公司基本信息
    profile = sql_db.query(f"SELECT * FROM companies WHERE name='{company}'")
    
    # 2) 财务数据（多年）
    financials = sql_db.query(
        f"SELECT * FROM financials WHERE company='{company}' "
        f"ORDER BY period DESC LIMIT 12"
    )
    
    # 3) 风险因素
    risk_chunks = retriever.retrieve(
        "risk factors regulatory competition",
        filter={"company": company, "section": "risk_factors"},
        k=5,
    )
    
    # 4) 最近新闻
    news = news_api.search(company, last_days=30)
    
    # 5) 行业研报
    industry_research = retriever.retrieve(
        "industry outlook trends",
        filter={"sector": profile["sector"]},
        k=5,
    )
    
    # 6) 构造完整报告 prompt
    prompt = f"""为 {company} 生成投资尽调报告。

[公司画像]
{profile}

[财务表现]
{financials}

[风险因素]
{risk_chunks}

[近期新闻]
{news}

[行业研究]
{industry_research}

请生成结构化报告：
1. 公司概况
2. 财务分析
3. 主要风险
4. 投资亮点
5. 估值参考"""
    
    return llm.generate(prompt)
```

## 八、工业实践：AlphaSense、Bloomberg、Sentieo

### AlphaSense

**企业级金融搜索 + AI**：
- 索引 10000+ 数据源（SEC filings, broker research, news）。
- Smart Synonyms：金融术语标准化。
- AI 摘要：自动生成关键洞察。
- 主要客户：投行、PE、咨询、企业战略。

### Bloomberg Terminal AI

- BloombergGPT 内嵌（私有）。
- 实时报价、新闻、研究一键访问。
- AI 助手：自然语言查询。

### Sentieo (now AlphaSense)

- Excel + AI 集成。
- 文档协同 + 智能搜索。

### Ramp / Brex AI

- 自动从发票、收据提取信息。
- AI 财务建议。

## 九、合规要求

### 1. 数据来源合规

```python
class DataSourceCompliance:
    """检查数据源使用合规。"""
    def __init__(self):
        self.allowed_sources = {
            "public": True,   # SEC 公开
            "licensed": True, # 有授权
            "mnpi": False,    # 内幕信息禁止
            "personal": False,# 个人信息
        }
    
    def can_use(self, source_type: str) -> bool:
        return self.allowed_sources.get(source_type, False)
```

### 2. 输出监管

```python
def compliance_check_output(answer: str, user_role: str) -> bool:
    """输出合规检查。"""
    # 1) 不含投资建议（除非用户是合规客户）
    if user_role != "professional" and "建议" in answer:
        return False
    
    # 2) 不含未来价格预测
    if any(kw in answer for kw in ["目标价", "预测涨幅", "建议买入"]):
        return False
    
    # 3) 含明确免责声明
    if "免责声明" not in answer:
        return False
    
    return True
```

### 3. 审计日志

```python
def log_rag_query(query, retrieved_chunks, answer, user_id):
    """记录所有 RAG 查询用于审计。"""
    audit_db.insert({
        "timestamp": datetime.now(),
        "user_id": user_id,
        "query": query,
        "retrieved_sources": [c["source"] for c in retrieved_chunks],
        "answer": answer,
        "model_version": "fingpt-v3.2",
    })
```

## 十、未来方向

### 1. 实时流式 RAG

```text
传统 RAG: 用户问 → 检索 → 回答
流式 RAG: 持续索引新数据 → 用户问 → 检索最新 → 回答
```

实现：长连接 WebSocket + 增量索引。

### 2. 多源数据融合

- 财报 + 新闻 + 行情 + 社交媒体 + 卫星图像。
- 跨数据源推理（"销售旺季 + 库存数据 + 仓储公司提价" → 推断行业景气）。

### 3. 预测性 RAG

不只是回顾——结合预测模型：

```python
def predict_with_rag(query):
    # 1) 检索历史
    history = retriever.retrieve(query)
    
    # 2) 时序模型预测
    forecast = time_series_model.predict(...)
    
    # 3) 综合输出
    return llm.generate(f"历史: {history}\n预测: {forecast}\n\n回答: {query}")
```

## 十一、给金融 RAG 团队的清单

1. **数据源合规**：只用合规数据源，MNPI 严格隔离。
2. **混合检索**：语义 + 关键词 + 结构化。
3. **强制引用**：每个数字必须可追溯。
4. **幻觉验证**：输出必须经结构化数据校验。
5. **审计日志**：所有查询记录可追溯。
6. **实时性**：分钟级更新行情、新闻。
7. **多语言支持**：中英文研报统一处理。
8. **人类最终决策**：高 stakes 决策必须有人 review。
9. **持续监控**：检索质量、回答准确率、用户反馈。
10. **灾备**：AI 故障时人工接管流程。

## 小结

金融 RAG 是金融 LLM 落地的**关键技术**——把 LLM 与实时市场数据、财报披露、新闻研报结合，让回答**准确、可追溯、时效性强**。核心组件是**混合检索**（语义 + 关键词 + 结构化）+ **强制引用** + **幻觉验证**。**工业实践**（AlphaSense、Bloomberg、Stripe）已经证明商业价值，但**幻觉、监管、内幕信息边界**仍是核心风险。**真正成功的金融 RAG** 是"AI 决策 + 人类监督 + 严格审计 + 合规闭环"的组合。下一篇我们将看到金融 AI 最具争议的应用——**算法交易**：用 AI 自动化交易决策。
