# 实验可重现性：代码、数据、环境三位一体

"上次我跑出 92% 的准确率，现在只能跑到 89%——是数据变了，还是改了什么参数？" 这几乎是每个 ML 团队都会遇到的噩梦。实验可重现性的目标是：**给定一个 run id / tag，任何工程师都能在任何机器上复现那次实验的全部结果**。本文从代码、数据、环境、随机性四个维度，梳理工程化实践。

## 一、为什么 ML 实验难重现

ML 实验的"输入空间"比传统软件大得多：

| 维度 | 传统软件 | ML 实验 |
|---|---|---|
| 代码版本 | git commit | + notebook / config |
| 数据 | 静态文件 | + 动态生成 / 标注 |
| 环境 | Python 版本 + 系统库 | + CUDA / cuDNN / 框架版本 |
| 随机性 | 通常无 | 初始化 / shuffle / dropout |
| 计算 | 确定的 | 浮点非结合律 |

任何一个维度漂移，结果都可能不同。

## 二、代码可重现：Git + 配置管理

### 2.1 强制 git commit 关联

```python
import subprocess
import mlflow

def get_git_info() -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    ).decode().strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"]
    ).decode().strip())
    return {"commit": commit, "branch": branch, "dirty": dirty}

with mlflow.start_run():
    mlflow.log_params(get_git_info())
```

**强制策略**：CI 上拒绝运行有未提交修改的实验。

### 2.2 配置外部化

不要把超参写在代码里，用 YAML / Hydra / OmegaConf：

```yaml
# config.yaml
model:
  name: resnet50
  pretrained: true
training:
  lr: 0.001
  batch_size: 64
  epochs: 30
  optimizer: adamw
data:
  version: 2024-q1
  train_path: s3://bucket/train/v3/
```

每次 `start_run()` 都 `mlflow.log_artifact("config.yaml")`，并把整个 config dict 也 `log_params`——双保险。

### 2.3 Hydra 的命令式覆盖

```bash
python train.py \
    model.name=vit_large \
    training.lr=0.0005 \
    training.epochs=50
```

每次运行 Hydra 自动生成 `outputs/2024-01-15/12-30-45/.hydra/config.yaml`，直接 log 到 MLflow。

## 三、数据可重现：版本化 + 校验和

### 3.1 数据版本化工具

| 工具 | 思路 |
|---|---|
| **DVC** | Git-like 数据版本控制，文件指针存 git、大文件存 S3 |
| **Pachyderm** | 数据 pipeline + 版本化 |
| **LakeFS** | 对象存储上做 git-like 分支 |
| **Delta Lake / Iceberg** | 表格式版本（适合结构化数据） |

```bash
dvc add data/train.parquet
git add data/train.parquet.dvc data/.gitignore
git commit -m "data v3"
```

`train.parquet.dvc` 里存的是 md5 hash + 文件大小 + S3 路径，git commit 记录的是数据的"指针"。

### 3.2 数据校验和

```python
import hashlib

def file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

# 训练开始时校验
expected = "d41d8cd98f00b204e9800998ecf8427e"
actual = file_md5("data/train.parquet")
assert actual == expected, f"data drift! expected={expected} got={actual}"
```

### 3.3 Feature Store + 数据契约

更工业级的做法是用 Feature Store（见 feature-stores.md）——特征定义即版本，"上次训练用的特征版本"和"这次推理用的特征版本"都能追溯。

## 四、环境可重现：容器化

### 4.1 Docker 锁定环境

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN pip install \
    transformers==4.36.0 \
    datasets==2.14.0 \
    accelerate==0.24.0 \
    mlflow==2.9.0

WORKDIR /workspace
COPY . /workspace
CMD ["python", "train.py"]
```

把 **Python 版本、CUDA 版本、cuDNN 版本、pip 包版本**全部钉死。

### 4.2 完全锁定：Pip freeze + hash

```bash
pip install --require-hashes -r requirements.txt
```

`requirements.txt` 里每行带 `--hash=sha256:...`，pip 会校验每个 wheel 的 hash，缺一个都装不上。

### 4.3 系统级：Nix / Apptainer

对极端严格的场景，用 Nix 描述完整系统级依赖（编译器、BLAS、CUDA toolkit），或者用 Apptainer / Singularity 镜像把整个 OS 打包。

## 五、随机性可重现：种子管理

```python
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDA 卷积确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
```

注意几个隐藏随机源：

- **DataLoader 的 worker**：每个 worker 都要 seed。
- **dropout**：默认就有随机。
- **Python `dict` / `set` 迭代顺序**：Py3.7+ 是插入序，但 hash 仍随机。
- **CUDA 非确定性算子**：atomicAdd、scatter 等，要走 deterministic 版本。

### 5.1 DataLoader 关键

```python
def worker_init_fn(worker_id):
    import torch
    base_seed = torch.initial_seed() % 2**32
    import numpy as np
    np.random.seed(base_seed + worker_id)
    import random
    random.seed(base_seed + worker_id)

loader = DataLoader(ds, num_workers=4, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(42))
```

## 六、可重现性检查清单

一个真正可重现的实验，应该回答这些问题：

| 问题 | 回答来源 |
|---|---|
| 用了什么代码？ | git commit hash |
| 代码是否干净？ | git status dirty 标志 |
| 用了什么配置？ | config.yaml + MLflow params |
| 用了什么数据？ | DVC pointer / 数据 hash |
| 数据是否完整？ | 校验和 |
| 跑在什么环境？ | Docker image digest |
| 用了什么随机种子？ | MLflow params |
| 用了哪些 GPU？ | nvidia-smi log artifact |

## 七、组织级保障

1. **禁止 notebook 直接出模型**：notebook 适合探索，不适合上线。原型验证后必须重构成模块化脚本。
2. **每次实验生成 Run ID card**：commit + data version + image digest + seed + config，钉死在 MR 里。
3. **定期复现**：每月随机抽 5 个 historical run 重新跑一遍，看结果偏差。偏差 > 1% 要排查。
4. **离线包管理**：用 `pip download` + 本地 pypi 镜像，避免公网依赖变化导致环境漂移。

## 小结

实验可重现性是 ML 工程的"工程质量底线"。代码版本（git + config）、数据版本（DVC / hash）、环境版本（Docker）、随机性（seed）四个维度同时钉死，才能让实验从"玄学"变成"工程"。代价是前期投入更多时间配置和工具链——但一次混乱排查省下的时间，往往是几十倍回报。下一篇我们将进入 **超参数调优**——如何在大规模搜索中高效找到最佳配置。
