# AI 数据流水线：从原始数据到训练就绪

"模型的天花板由数据决定"。AI 数据流水线负责把原始日志、文档、图片、对话流，转换成模型可直接消费的 `(input_ids, labels)` 或 `(pixel_values, labels)`。本文梳理 AI 数据流水线的典型架构、核心组件（采集、清洗、切分、配比、去重/去污染），以及在 LLM 时代下的特殊挑战（数据配比、质量过滤、PII 去除）。

## 一、典型流水线架构

```text
原始源          采集            存储           处理           训练
─────────    ─────────    ─────────    ─────────    ─────────
业务日志  ─→  Kafka/PubSub  ─→  Data Lake  ─→  Spark/Ray  ─→  Arrow/MDS
爬虫数据  ─→  Airbyte/Fivetran  (S3/GCS)    (清洗/去重)    (TFDS/DS)
用户反馈  ─→  CDC/Debezium    (Delta/Iceberg)
```

每个环节都有自己的工具栈，但**端到端的可观测性**和**幂等性**是共通要求——任何一步失败，重新跑时要能恢复，不能污染下游。

## 二、采集：批 vs 流

**批处理**（Batch）：用 Airflow / Dagster 周期性跑，适合离线训练数据。

```python
# Airflow DAG：每天抓取新闻
@dag(schedule="@daily", start_date=datetime(2024, 1, 1))
def news_pipeline():
    @task
    def crawl():
        return fetch_news()
    @task
    def clean(raw):
        return clean_text(raw)
    @task
    def store(records):
        write_to_s3(records, "s3://bucket/news/{{ ds }}/")
    store(clean(crawl()))
```

**流处理**（Streaming）：用 Kafka + Flink/Spark Streaming，适合实时数据（用户对话、推荐反馈）。

LLM 时代越来越倾向**Lambda 架构**——离线全量 + 在线增量，既保证训练数据完整性，又能反映最新分布。

## 三、清洗与质量过滤

文本清洗常见步骤：

```python
import re

def clean(text: str) -> str | None:
    text = text.strip()
    if len(text) < 50:
        return None                  # 太短，丢弃
    if len(text) > 100_000:
        return None                  # 太长，截断或丢弃
    text = re.sub(r"<[^>]+>", "", text)            # 去 HTML 标签
    text = re.sub(r"http\S+", "", text)             # 去 URL
    text = re.sub(r"\s+", " ", text)                # 合并空白
    # 启发式质量过滤
    if ratio_of_garbled_chars(text) > 0.3:
        return None
    return text
```

更高级的质量评估用模型本身——例如用一个小模型给每篇文档打分（"教育性"、"有害性"），保留高质量子集。GPT-3 论文里就用 classifier 把数据分成"高质量/低质量"两个 bucket，单独配比训练。

## 四、去重与去污染

**去重**对训练至关重要——重复样本会让模型过拟合、降低多样性。常用工具：

- **MinHash / LSH**：近似文档相似度，可处理亿级语料。
- **SimHash**：用汉明距离判定近重复，Google 用在爬虫去重。
- **Embedding 聚类**：用 sentence embedding + 层次聚类找近似簇。

```python
from datasketch import MinHash, MinHashLSH

lsh = MinHashLSH(threshold=0.8, num_perm=128)
for doc in corpus:
    mh = MinHash(num_perm=128)
    for word in set(doc.split()):
        mh.update(word.encode("utf-8"))
    if not lsh.query(mh):
        lsh.insert(doc.id, mh)
```

**去污染（Decontamination）**是另一类问题：评估集（benchmark）可能泄漏到训练集里。要把评估集的 n-gram 与训练集比对，剔除高重合样本。Llama 2 / 3 论文都把这一步作为发布前的强制检查。

## 五、数据配比（Mixture）

LLM 训练数据由多个领域（网页、代码、对话、百科、书籍）按比例混合：

```yaml
mixture:
  common_crawl: 0.50      # 网页
  github_code:  0.15      # 代码
  wikipedia:    0.05      # 百科
  books:        0.10      # 书籍
  arxiv:        0.05      # 论文
  qa_forums:    0.10      # 问答
  math:         0.05      # 数学
```

配比对模型能力影响巨大。Llama 3 报告里详细列出了它们的配比，并通过消融实验确定每类数据的最佳 sampling temperature。

工程上常用 **DoReMi**（NVIDIA）自动学习最优配比：用一个小模型 + 域权重代理，搜索让 loss 最小的混合比例。

## 六、PII 与合规

训练数据往往含个人隐私信息（手机号、邮箱、地址）。生产流水线必须包含 PII 检测与脱敏：

```python
import re

PII_PATTERNS = {
    "phone_cn": r"1[3-9]\d{9}",
    "email":    r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}",
    "id_card":  r"\d{17}[\dXx]",
}

def mask_pii(text: str) -> str:
    for tag, pat in PII_PATTERNS.items():
        text = re.sub(pat, f"[{tag.upper()}_REDACTED]", text)
    return text
```

更严谨的做法是用 NER 模型（Presidio、Flair）识别 PII 实体。对合规要求高的场景（医疗、金融），还要支持"right to be forgotten"——按用户 ID 从训练数据里删除其全部记录。

## 七、格式与打包

最后一步是把处理好的数据打成训练框架能直接读的格式：

- **TFDS / HuggingFace Datasets**：内存映射（`mmap`）的 Arrow 格式，支持流式读取。
- **MosaicML MosaicML MDS / WebDataset**：二进制分片，适合大规模分布式训练。
- **Parquet + DuckDB**：轻量分析友好。

```python
import datasets

ds = datasets.Dataset.from_list(records)
ds.save_to_disk("s3://bucket/processed/v1/")
# 训练时直接 datasets.load_from_disk(...) 即可
```

## 小结

AI 数据流水线的核心是把**原始多源异构数据**，经过**采集 → 清洗 → 去重 → 去污染 → 配比 → 脱敏 → 打包**，变成可复现、可监控、合规的训练语料。LLM 时代下，数据质量、配比策略、PII 处理的重要性大幅提升——同一份模型架构，数据的差别可以造成 5-10 个点的能力差异。下一篇我们将聚焦 **数据质量评估**——如何度量"这份数据好不好"。
