# 预训练数据策展：从 Common Crawl 到 FineWeb

预训练数据的质量直接决定模型能力的上限。GPT-4、Claude、Gemini 等顶级模型的核心竞争力之一就是数据。本文系统讨论预训练数据的来源（Common Crawl、Wikipedia、代码、书籍）、清洗流程（去重、过滤、质量评分）、以及前沿项目（FineWeb-Edu、SlimPajama、RedPajama）的设计哲学。

## 一、为什么数据是 LLM 的"第一性原理"

```python
# 数据决定模型能力上限的几个事实
data_facts = {
    "scaling_law": "数据量、参数量、训练算力三者协同 scaling，缺一不可",
    "data_quality": "高质量数据 1T tokens > 低质量数据 10T tokens（GPT-4 论文）",
    "data_diversity": "多领域数据（代码、数学、对话）让模型能力更均衡",
    "data_contamination": "测试集数据出现在训练集会大幅高估真实能力",
    "data_tokenizer": "BPE/SentencePiece tokenizer 在数据上训练，影响 token 效率",
}
```

**核心结论**：数据 > 架构 > 算力（在算力够的前提下）。

## 二、预训练数据来源

```python
# 主要数据源
data_sources = {
    "web": [
        "Common Crawl (CC-MAIN)",          # ~250B 网页，原始 HTML
        "CCNet",                            # Common Crawl 清洗版
        "FineWeb / FineWeb-Edu",            # HuggingFace 2024 高质量
        "RedPajama-Data-V2",                # Together AI 大规模清洗
        "The Pile",                          # EleutherAI 825GB 多源
        "WuDao-Corpora",                    # 中文 200GB",
    ],
    "knowledge": [
        "Wikipedia (多语言)",                # 高质量百科
        "Wikidata",                         # 结构化知识
        "PubMed / ArXiv",                  # 学术文献
    ],
    "code": [
        "The Stack (BigCode)",              # 6TB 多语言代码
        "GitHub Archive",                   # 公开仓库
        "StackOverflow",                    # 问答
    ],
    "books": [
        "Books3 / Books Corpus",            # 公开书籍
        "Project Gutenberg",                # 公版书籍
    ],
    "dialogue": [
        "公开对话数据集",                    # Reddit/Twitter/论坛
    ],
    "synthetic": [
        "Self-Instruct / Magpie",          # LLM 自生成
        "蒸馏数据",                          # 用 GPT-4 生成",
    ],
}
```

### Common Crawl

```python
# Common Crawl 是最大公开网页语料
# 每月 ~250B 网页，几 TB 压缩
# 格式：WARC（网页原始）/ WAT（文本提取）/ WET（纯文本）

common_crawl = {
    "size_per_crawl": "~250B pages, ~3-5 TB compressed",
    "frequency": "monthly",
    "format": "WARC, WAT, WET",
    "languages": "多语言，英文为主",
    "license": "每页独立 license（多为 CC 或版权）",
    "challenge": "噪声极大、广告、垃圾、NSFW",
}

# 处理流程
from warcio.archiveiterator import ArchiveIterator

def extract_text_from_warc(warc_file):
    with open(warc_file, "rb") as f:
        for record in ArchiveIterator(f):
            if record.rec_type == "response":
                url = record.rec_headers.get_header("WARC-Target-URI")
                content = record.content_stream().read()
                yield {"url": url, "content": content.decode("utf-8", errors="ignore")}
```

## 三、数据清洗流水线

### 整体流程

```python
# 典型预训练数据清洗流水线
pipeline = [
    "1. HTML 提取（trafilatura / jusText / BeautifulSoup）",
    "2. 语言识别（fastText langdetect）",
    "3. 启发式过滤（长度、符号比例、NSFW）",
    "4. 去重（MinHash / 精确哈希）",
    "5. 质量评分（fastText quality classifier）",
    "6. PII 移除（正则 + NER）",
    "7. 分词 & Token 训练",
    "8. 数据混合配比",
]
```

### 1. HTML 提取

```python
import trafilatura

def extract_clean_text(html: str) -> str:
    """从 HTML 提取干净正文"""
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        include_images=False,
        favor_precision=True,
    )
    return text or ""

# 批量处理
from multiprocessing import reduce

with reduce(processes=50) as pool:
    clean_texts = pool.map(extract_clean_text, raw_htmls)
```

### 2. 语言识别

```python
import fasttext

# 下载语言识别模型
lang_model = fasttext.load_model("lid.176.bin")

def detect_language(text: str) -> str:
    """返回 ISO 639-1 代码"""
    predictions = lang_model.predict(text, k=1)
    return predictions[0][0].replace("__label__", "")

# 批量过滤非英语
def is_english(text: str, threshold: float = 0.7) -> bool:
    lang, prob = lang_model.predict(text, k=1)
    return lang[0] == "__label__en" and prob[0] > threshold
```

### 3. 启发式过滤

```python
import re

def heuristic_quality_filter(text: str) -> bool:
    """基于简单规则的过滤器"""
    if len(text) < 100:
        return False
    if len(text) > 500_000:
        return False
    # 单词平均长度
    words = text.split()
    if not words:
        return False
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 3 or avg_word_len > 10:
        return False
    # 符号比例
    symbol_ratio = len(re.findall(r"[^\w\s]", text)) / len(text)
    if symbol_ratio > 0.1:
        return False
    # 重复行
    lines = text.split("\n")
    if len(lines) > 10:
        max_repeat = max(Counter(lines).values())
        if max_repeat / len(lines) > 0.3:
            return False
    # 关键词黑名单
    blacklist = ["lorem ipsum", "cookie policy", "javascript required"]
    if any(kw in text.lower() for kw in blacklist):
        return False
    return True
```

### 4. 去重

```python
# MinHash 去重（大规模近似去重）
from datasketch import MinHash, MinHashLSH

def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    words = text.split()
    for i in range(0, len(words) - 5):
        shingle = " ".join(words[i:i+5])
        m.update(shingle.encode("utf-8"))
    return m

# LSH 索引
lsh = MinHashLSH(threshold=0.8, num_perm=128)
minhashes = {}
for doc_id, text in enumerate(documents):
    m = create_minhash(text)
    lsh.insert(doc_id, m)
    minhashes[doc_id] = m

# 查询重复
def find_duplicates(target_text: str):
    m = create_minhash(target_text)
    candidates = lsh.query(m)
    return candidates

# 精确去重（文档级）
seen_hashes = set()
def exact_dedup(text: str) -> bool:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    return True
```

### 5. 质量评分

```python
# 用 fastText / BERT 训练质量分类器
import fasttext

# 训练质量分类器
def train_quality_classifier(positive_texts, negative_texts, model_path="quality.bin"):
    with open("pos.txt", "w") as f:
        for t in positive_texts:
            f.write(f"__label__high {t}\n")
    with open("neg.txt", "w") as f:
        for t in negative_texts:
            f.write(f"__label__low {t}\n")

    model = fasttext.train_supervised(
        input="training_data.txt",
        lr=1.0,
        epoch=5,
        wordNgrams=2,
        bucket=200000,
        dim=50,
        loss="softmax",
    )
    model.save_model(model_path)
    return model

# 用 LLM 做质量评分（FineWeb-Edu 风格）
from openai import OpenAI

def llm_quality_score(text: str) -> float:
    """用 GPT-4 评估教育价值（0-5 分）"""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""请评估以下网页内容的教育价值（0-5 分，5 最高）。
考虑因素：信息密度、事实准确性、教学价值、写作质量。
输出格式：仅返回一个数字。

内容：{text[:3000]}"""
        }],
        temperature=0,
        max_tokens=5,
    )
    try:
        score = float(response.choices[0].message.content.strip())
        return score
    except ValueError:
        return 3.0
```

### 6. PII 移除

```python
import re

def remove_pii(text: str) -> str:
    """移除个人隐私信息"""
    # Email
    text = re.sub(r"\b[\w.-]+@[\w-]+\.[\w.-]+\b", "[EMAIL]", text)
    # 电话
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
    # SSN
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
    # IP
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]", text)
    # 信用卡
    text = re.sub(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[CC]", text)
    return text

# 用 Presidio / NER 更智能
from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()
results = analyzer.analyze(text=text, language="en")
for result in results:
    text = text.replace(
        text[result.start:result.end],
        f"[{result.entity_type}]",
    )
```

## 四、前沿数据集项目

### FineWeb-Edu（HuggingFace 2024）

```python
# FineWeb-Edu
# - 1.3T tokens 高质量网页
# - 用 Llama-3-70B 做质量评分
# - 仅保留教育价值 >= 3 的网页
# - 开源（huggingface.co/datasets/HuggingFaceFW/fineweb-edu）

fineweb_edu_features = {
    "size": "1.3T tokens",
    "source": "Common Crawl 2013-2024",
    "filter": "Llama-3-70B 教育评分 >= 3",
    "language": "英文",
    "performance": "同等 token 下，Llama-2 性能超过用其他数据集",
    "license": "ODC-BY-1.0",
}
```

### RedPajama-Data-V2

```python
# RedPajama-Data-V2（Together AI）
# - 30T tokens 原始 + 标注
# - 完整质量标签（30+ 维度）
# - 支持用户自定义过滤策略

redpajama_v2 = {
    "size": "30T tokens raw, 5T tokens annotated",
    "sources": ["CommonCrawl", "C4", "GitHub", "ArXiv", "Books", "Wikipedia", "StackExchange"],
    "annotations": ["URL 分类", "NSFW 分数", "毒性与危害", "语言", "质量"],
    "advantage": "用户可按需构建自己的训练集",
}
```

### SlimPajama

```python
# SlimPajama（CerebrasAI）
# - The Pile 清洗版
# - 仅 627B tokens，但质量更高
# - 训练效率提升 1.7x

slimpajama = {
    "size": "627B tokens",
    "vs_the_pile": "仅 35% 数据，性能更好",
}
```

### The Stack v2（BigCode）

```python
# The Stack v2（600+ 语言代码数据）
# - 67.5TB 原始代码
# - 4.5TB 清洗后
# - GitHub commit 粒度

the_stack_v2 = {
    "languages": "600+",
    "size_raw": "67.5 TB",
    "size_clean": "4.5 TB",
    "filter": "许可证过滤 + 启发式质量",
    "format": "GitHub commits + 单独文件",
}
```

## 五、数据混合策略

```python
# 不同训练阶段的数据混合

# 阶段 1：通用预训练（90%+ 网页/百科）
stage_1_mix = {
    "web": 0.60,          # Common Crawl 清洗
    "wiki": 0.05,         # 百科
    "code": 0.15,         # 代码
    "books": 0.10,        # 书籍
    "academic": 0.05,     # ArXiv / PubMed
    "dialogue": 0.05,     # 对话数据
}

# 阶段 2：增强推理（数学/代码比例提升）
stage_2_mix = {
    "web": 0.40,
    "wiki": 0.05,
    "code": 0.25,         # 提升
    "math": 0.10,         # 新增
    "books": 0.10,
    "academic": 0.10,
}

# 阶段 3：指令微调（指令/对话为主）
stage_3_mix = {
    "instruction": 0.70,  # 指令数据
    "dialogue": 0.20,
    "general_web": 0.10,
}

# DoReMi（Google 2023）
# 自动学习最优数据混合权重
```

## 六、数据 Token 化与训练

```python
# SentencePiece tokenizer 训练
import sentencepiece as spm

# 训练 tokenizer
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=128_000,
    model_type="bpe",
    character_coverage=0.9995,
    num_threads=64,
    train_extremely_large_corpus=True,
)

# 用 tokenizer 处理数据
sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
token_ids = sp.encode("你好世界", out_type=int)

# 预训练分词（大量数据）
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )

tokenized_ds = ds.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=64,
    remove_columns=ds.column_names,
)
```

## 七、数据质量评估

```python
# 如何评估清洗后数据的质量

data_quality_metrics = {
    "语言覆盖率": "各语言 token 数（确保目标语言足够）",
    "领域分布": "web/code/math/dialogue 比例",
    "重复率": "文档级重复率应<5%",
    "毒害率": "NSFW/毒害内容比例<0.1%",
    "PII 率": "个人信息残留率",
    "训练/测试重叠率": "与常见 benchmark 重叠率",
    "下游任务表现": "在 MMLU/HumanEval 等评测上的表现",
}

# 计算下游任务表现
def evaluate_on_benchmarks(model, tokenizer, datasets):
    results = {}
    for name, eval_data in datasets.items():
        metric = run_evaluation(model, tokenizer, eval_data)
        results[name] = metric
    return results
```

## 八、数据相关挑战

```python
# 预训练数据的核心挑战

challenges = {
    "1. 数据污染": "测试集出现在训练集 → 评测虚高",
    "2. 偏置放大": "训练数据的偏置在模型中放大",
    "3. 版权问题": "网页内容版权不清，模型可能输出原文",
    "4. 隐私风险": "PII 数据可能泄露",
    "5. 静态数据陈旧": "训练截止后新知识缺失",
    "6. 质量/数量权衡": "高质量但量少 vs 大量但低质量",
    "7. 多语言平衡": "英文数据远多于其他语种",
    "8. 评估困难": "数据好坏难直接评估",
}
```

## 小结

预训练数据是 LLM 能力的根基。**FineWeb-Edu、SlimPajama、RedPajama-V2 等项目代表了当下数据工程的最高水平**——大规模清洗 + LLM 评分 + 多维度标注。**网页（Common Crawl）+ 代码（GitHub）+ 百科 + 书籍**构成主流数据源，**去重、质量过滤、PII 移除**是关键清洗步骤。下一篇我们讨论**领域特定数据集**——医疗、法律、金融等垂直场景如何构建数据集。