# HuggingFace Datasets Hub：AI 训练数据的集散地

HuggingFace Datasets Hub 是当下最重要的开源数据集聚合平台——超过 20 万个公开数据集，覆盖 NLP、语音、视觉、多模态。本文讨论 Hub 的核心能力（datasets 库、Dataset Card、streaming、版本控制），以及如何利用 Hub 高效收集训练数据。

## 一、HuggingFace Datasets Hub 概况

```python
# HuggingFace Hub 规模（2026 数据）
hub_stats = {
    "datasets": 200_000,
    "models": 1_500_000,
    "spaces": 400_000,
    "monthly_visitors": "30M+",
    "users": "10M+",
    "languages": "180+",
    "domain_leaders": "HuggingFace 是 AI 界的 GitHub",
}
```

**核心价值**：一个平台聚合 + 标准化加载 + 版本控制 + 数据集卡片。

## 二、datasets 库：标准化加载

### 基本加载

```python
from datasets import load_dataset

# 加载内置数据集
dataset = load_dataset("imdb")  # 25k 影评 + 标签
print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 25000
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 25000
#     })
# })

# 加载指定 split
train_data = load_dataset("imdb", split="train")

# 加载指定配置/语言
dataset = load_dataset("wikipedia", "20231101.zh", split="train")  # 中文维基

# 流式加载（不下载整个数据集）
dataset = load_dataset("wikipedia", "20231101.zh", split="train", streaming=True)
for batch in dataset:
    print(batch["title"])
    break
```

### 数据集操作

```python
from datasets import load_dataset

ds = load_dataset("squad", split="train")

# 过滤
ds_filtered = ds.filter(lambda x: len(x["question"]) > 50)

# 映射（transform）
def add_length(example):
    example["question_length"] = len(example["question"].split())
    return example
ds_with_len = ds.map(add_length)

# 排序
ds_sorted = ds.sort("question_length", reverse=True)

# 选择列
ds_selected = ds.select_columns(["question", "answers"])

# 切分
train, val = ds.train_test_split(test_size=0.1, seed=42).values()

# 批处理
ds_batched = ds.map(process_fn, batched=True, batch_size=1000)

# Arrow 高效列存
print(ds.cache_files)  # 缓存文件位置
```

### 多模态数据集

```python
from datasets import load_dataset

# 加载图文数据集
dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")
# 包含图像 + 对话

# 加载视频数据集
dataset = load_dataset("lmms-lab/LLaVA-Video-178K")
# 视频 + 对话（注意视频较大）

# 加载音频数据集
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "zh-CN", split="train")
# 音频 + 转录文本
```

## 三、流式加载：避免下载整个数据集

```python
# 大数据集（如 LAION 5B）无法下载，需要 streaming
from datasets import load_dataset

# 流式加载
dataset = load_dataset(
    "laion/laion2B-en-aesthetic",
    split="train",
    streaming=True,
)

# 流式迭代
for i, example in enumerate(dataset):
    if i >= 10:
        break
    print(example["TEXT"], example["URL"])
    # {'TEXT': 'a cute cat', 'URL': 'https://...', ...}

# 流式 + 过滤 + 采样
dataset = load_dataset("oscar-corpus/OSCAR-2301", "zh", split="train", streaming=True)
dataset = dataset.filter(lambda x: len(x["text"]) > 1000)
dataset = dataset.shuffle(buffer_size=10000, seed=42)
for example in dataset:
    print(example["text"][:200])
    break

# 流式 + map
def tokenize(example):
    example["tokens"] = tokenizer.encode(example["text"])[:512]
    return example
dataset = dataset.map(tokenize)
```

## 四、数据集卡片（Dataset Card）

```python
# 数据集卡片是 HuggingFace 数据集的元数据规范
# 字段示例：

dataset_card_yaml = """
---
annotations_creators:
  - crowdsourced
language:
  - en
license:
  - apache-2.0
multilinguality:
  - monolingual
size_categories:
  - 100K<n<1M
task_categories:
  - text-classification
  - question-answering
tags:
  - sentiment
  - movie-reviews
pretty_name: IMDB
configs:
  - config_name: plain_text
    data_files:
      - split: train
        path: {path}
      - split: test
        path: {path}
---

# Dataset Card for IMDB

## Dataset Description
- **Homepage:** https://ai.stanford.edu/~amaas/data/sentiment/
- **Repository:** N/A
- **Paper:** Maas et al. 2011
- **Point of Contact:** Andrew Maas

## Dataset Structure
### Data Instances
[Example instance description]

### Data Fields
- `text`: The movie review text
- `label`: 0 (negative) or 1 (positive)

### Data Splits
| name | train | test |
|---|---|---|
| plain_text | 25000 | 25000 |

## Considerations for Using the Data
### Social Impact of Dataset
[Discussion of biases and ethical concerns]
"""
```

## 五、数据集版本控制

```python
# HuggingFace 数据集版本管理
# 类似 Git，每个 commit 是一个版本

from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

# 查看可用版本
configs = get_dataset_config_names("wikipedia")
# ['20231101.en', '20231101.zh', ...]

# 加载特定版本
ds = load_dataset("wikipedia", "20231101.zh", split="train")

# 多个版本并存
ds_old = load_dataset("wikipedia", "20220301.zh", split="train")
ds_new = load_dataset("wikipedia", "20231101.zh", split="train")
```

## 六、Hub API：编程化管理数据集

```python
from huggingface_hub import HfApi, list_datasets

api = HfApi()

# 搜索数据集
datasets = list(
    list_datasets(
        search="instruction",
        filter="task_categories:text-generation",
        sort="downloads",
        limit=10,
    )
)
for d in datasets:
    print(d.id, d.downloads)

# 下载数据集文件
from huggingface_hub import hf_hub_download
file_path = hf_hub_download(
    repo_id="anthropic/hh-rlhf",
    filename="harmless-base/train.jsonl",
    repo_type="dataset",
)

# 上传自己的数据集
api.upload_folder(
    folder_path="./my_dataset",
    repo_id="username/my-dataset",
    repo_type="dataset",
)

# 创建数据集卡片
api.upload_file(
    path_or_fileobj="./README.md",
    path_in_repo="README.md",
    repo_id="username/my-dataset",
    repo_type="dataset",
)
```

## 七、常用数据集推荐

```python
# 各任务常用数据集

recommended_datasets = {
    "pre_training": {
        "FineWeb-Edu": "1.3T tokens，高质量网页（HuggingFace 2024）",
        "The Pile": "825GB，多领域文本（EleutherAI 2020）",
        "RedPajama": "1.2T tokens（LLaMA 复现）",
        "SlimPajama": "627B tokens，The Pile 清洗版",
        "Common Crawl": "原始网页",
        "Wikipedia": "多语言百科",
    },
    "instruction_tuning": {
        "Alpaca": "52K 指令（GPT-3.5 生成）",
        "ShareGPT": "用户真实对话",
        "OpenHermes-2.5": "100 万指令（多源聚合）",
        "UltraChat": "140 万高质量多轮对话",
        "Magpie": "LLM 自生成指令",
    },
    "preference_rlhf": {
        "UltraFeedback": "64K 偏好对",
        "HH-RLHF": "Anthropic 人类偏好",
        "HelpSteer": "NVIDIA 偏好数据",
        "Magpie-Preference": "LLM 偏好",
    },
    "vision": {
        "COCO": "检测分割",
        "ImageNet": "分类",
        "LAION-5B": "图文对",
        "SAM-1B": "分割",
        "LVIS": "长尾检测",
    },
    "multimodal": {
        "LLaVA-Instruct": "图文指令",
        "ShareGPT4V": "细粒度图文",
        "Video-LLaVA": "视频指令",
    },
    "code": {
        "HumanEval": "代码生成评测",
        "The Stack": "6TB 代码",
        "Magicoder": "代码指令",
        "CodeContests": "编程竞赛",
    },
    "math": {
        "GSM8K": "小学数学",
        "MATH": "竞赛数学",
        "MetaMath": "数学指令",
    },
}
```

## 八、数据集工程技巧

### 数据清洗

```python
from datasets import load_dataset
import re

ds = load_dataset("wikipedia", "20231101.zh", split="train")

# 去重
def is_unique(example, idx, seen_hashes):
    h = hash(example["text"][:1000])
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    return True

seen = set()
ds = ds.filter(lambda x, idx: is_unique(x, idx, seen), with_indices=True)

# 质量过滤
def quality_filter(example):
    text = example["text"]
    if len(text) < 100:
        return False
    if re.search(r"[a-z]{30,}", text):  # 长串英文可能是错误
        return False
    if text.count("\n") / max(len(text), 1) > 0.3:  # 换行过多
        return False
    return True

ds_clean = ds.filter(quality_filter)
```

### 数据集统计

```python
import numpy as np

ds = load_dataset("imdb", split="train")

# 长度统计
lengths = [len(x["text"]) for x in ds]
print(f"Mean: {np.mean(lengths):.0f}")
print(f"Median: {np.median(lengths):.0f}")
print(f"P95: {np.percentile(lengths, 95):.0f}")

# 类别分布
from collections import Counter
labels = [x["label"] for x in ds]
print(Counter(labels))

# Token 长度分布
token_lengths = [len(tokenizer.encode(x["text"])) for x in ds.select(range(1000))]
print(f"Token mean: {np.mean(token_lengths):.0f}")
print(f"Token P99: {np.percentile(token_lengths, 99):.0f}")
```

## 九、DuckDB + Parquet：大规模数据分析

```python
# HuggingFace 数据集默认存为 Arrow / Parquet
# 可用 DuckDB 高效分析

import duckdb

# 直接查询 Parquet
result = duckdb.query("""
    SELECT label, COUNT(*) as count
    FROM 'imdb-train.parquet'
    GROUP BY text
""").df()

# 与 datasets 配合
from datasets import Dataset
ds = load_dataset("imdb", split="train")
df = ds.to_pandas()
result = duckdb.query("""
    SELECT label, AVG(LEN(text)) as avg_len
    FROM df
    GROUP BY label
""").df()
```

## 十、私有数据集与商用

```python
# 企业使用私有数据集的方式
private_dataset_options = {
    "1. HuggingFace Enterprise": "私有 Hub，SOC2/ISO27001",
    "2. Self-hosted Hub": "自己部署 HuggingFace Hub",
    "3. 内部 S3/OSS": "用 datasets 库直接加载本地 Parquet",
    "4. 直接 Arrow": "用 PyArrow 操作大规模数据",
}

# 直接加载本地数据
from datasets import load_dataset, Dataset

# 从 Parquet 加载
ds = load_dataset("parquet", data_files="data/*.parquet")

# 从 CSV 加载
ds = load_dataset("csv", data_files="data.csv", split="train")

# 从 JSON 加载
ds = load_dataset("json", data_files="data.jsonl")

# 从 Pandas DataFrame
import pandas as pd
df = pd.DataFrame({"text": [...], "label": [...]})
ds = Dataset.from_pandas(df)
```

## 小结

HuggingFace Datasets Hub 是 AI 训练数据的"基础设施"——**标准化加载、流式处理、版本控制、丰富的元数据**。**小数据集全量加载，大数据集 streaming，配合 DuckDB 做分析**。下一篇我们讨论 **预训练数据清洗**——如何把原始网页变成高质量训练集。