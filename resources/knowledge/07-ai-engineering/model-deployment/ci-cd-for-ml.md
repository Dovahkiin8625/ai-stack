# CI/CD for ML：训练、评估、部署的全链路流水线

传统 CI/CD 围绕"代码变更 → 构建 → 测试 → 部署"。ML 系统的 CI/CD 在此基础上还要管**数据变更、模型再训练、指标对比、A/B 发布**——多了一个不断循环的"训练-评估-发布"环节。本文梳理 ML CI/CD 的四个阶段、典型工具链，以及用 GitHub Actions / GitLab CI 搭建端到端流水线的实战。

## 一、传统 CI/CD vs ML CI/CD

```text
传统 CI/CD                       ML CI/CD
──────────                       ────────
code change ─→ build → test → deploy
                                        ↓
                                  data change / drift
                                        ↓
                                  trigger retrain
                                        ↓
                                  evaluate vs prod model
                                        ↓
                                  (better?) ─→ deploy
```

ML 流水线有两个**触发源**：

1. **代码变更**：新算法、新超参 → 重新训练 → 评估 → 部署。
2. **数据变更**：新数据 / 数据漂移 → 触发再训练 → 自动评估。

## 二、四阶段流水线

```text
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│ CI     │   │ Train  │   │ CD     │   │ Monitor│
│        │   │        │   │        │   │        │
│ 单元测试│ → │ 数据   │ → │ 评估   │ → │ 线上   │
│ lint   │   │ 训练   │   │ 部署   │   │ 监控   │
│ schema │   │ 验证   │   │ 发布   │   │ 反馈   │
└────────┘   └────────┘   └────────┘   └────────┘
   5 min        数小时        1 小时        持续
```

## 三、CI 阶段：代码与数据契约

### 3.1 代码质量

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: ruff check src/                      # lint
      - run: mypy src/                            # type check
      - run: pytest tests/ -v                     # unit tests
```

### 3.2 数据契约校验

```python
# tests/test_data_contract.py
from pandera import DataFrameSchema, Column, Check

schema = DataFrameSchema({
    "user_id": Column(str, Check.str_length(1, 64)),
    "age": Column(int, Check.in_range(0, 150)),
    "label": Column(int, Check.isin([0, 1])),
})

def test_train_data_conforms():
    df = load_train_data()
    schema.validate(df)   # 任何列不符就抛 ValidationError
```

### 3.3 Schema 兼容性

新模型输出的字段不能少，否则下游服务会崩。用 Pandera / Great Expectations 做严格校验。

## 四、Train 阶段：自动训练流水线

### 4.1 触发条件

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'configs/**'
  schedule:
    - cron: '0 2 * * 0'        # 每周日凌晨 2 点
  workflow_dispatch:           # 手动触发
```

也可以基于**数据漂移**自动触发（见 monitoring/drift-detection.md）。

### 4.2 训练任务

```yaml
train:
  runs-on: [self-hosted, gpu]    # 自托管 GPU runner
  steps:
    - uses: actions/checkout@v4
    - name: Setup Python
      uses: actions/setup-python@v5
    - name: Install
      run: pip install -r requirements.txt
    - name: Train
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
      run: |
        python src/train.py \
          --config configs/prod.yaml \
          --output-dir s3://bucket/runs/${{ github.sha }}/
    - name: Upload artifacts
      uses: actions/upload-artifact@v4
      with:
        name: model
        path: output/model/
```

自托管 runner（GitHub Actions / GitLab Runner + GPU 节点）让 CI 直接用上 GPU。Kubernetes 上跑 GitHub Actions 推荐使用 [actions-runner-controller](https://github.com/actions/actions-runner-controller)。

## 五、CD 阶段：评估 + 部署

### 5.1 自动评估

训练完不能直接部署，要先在**保留的测试集**上评估：

```python
# eval.py
import json, mlflow

prod_metrics = load_production_metrics()    # 上线版本的指标
new_metrics = evaluate_new_model()

delta = {
    "accuracy":  new_metrics["accuracy"] - prod_metrics["accuracy"],
    "f1":        new_metrics["f1"] - prod_metrics["f1"],
    "p99_latency": new_metrics["p99_latency"] - prod_metrics["p99_latency"],
}
print("delta:", json.dumps(delta, indent=2))

# 决策阈值
if delta["accuracy"] < -0.01:
    raise SystemExit("新模型 accuracy 下降 >1%，拒绝部署")

mlflow.log_metrics(delta)
```

### 5.2 多环境部署

```text
Dev (1 节点) ─→ Staging (3 节点, 镜像生产) ─→ Canary (5%) ─→ Production (100%)
```

```yaml
deploy-staging:
  needs: train
  runs-on: ubuntu-latest
  steps:
    - name: Deploy to staging
      run: |
        kubectl --context=staging set image deployment/ml-model \
          ml-model=myregistry/myimage:${{ github.sha }}

deploy-prod:
  needs: [deploy-staging, eval]
  if: github.ref == 'refs/heads/main'
  environment:
    name: production
    url: https://ml.example.com
  steps:
    - name: Canary 10%
      run: ./scripts/canary.sh ${{ github.sha }} 10
    - name: Wait & verify
      run: sleep 600 && ./scripts/check_health.sh
    - name: Promote to 100%
      run: ./scripts/canary.sh ${{ github.sha }} 100
```

`canary.sh` 调 Kubernetes / Istio / 负载均衡器 API，逐步放量。

## 六、CD 工具生态

| 工具 | 定位 | 特点 |
|---|---|---|
| **GitHub Actions** | 代码托管内置 | 易上手、Marketplace 丰富 |
| **GitLab CI** | 一体化 | 自托管 runner 简单 |
| **Jenkins** | 老牌 | 灵活、插件多、运维重 |
| **Argo Workflows** | K8s 原生 | 工作流 + DAG 编排 |
| **Kubeflow Pipelines** | K8s + ML | 专为 ML 设计 |
| **Airflow** | 通用编排 | 适合数据 pipeline |
| **Dagster** | 数据 + ML | 资产（asset）视角 |
| **Metaflow** | Netflix 出品 | 数据科学友好 |

Kubeflow Pipelines（KFP） 是 ML 专用编排系统的代表：

```python
import kfp
from kfp import dsl

@dsl.component
def train_op(data_path: str) -> str:
    # 训练
    return "s3://bucket/model/"

@dsl.component
def evaluate_op(model_path: str) -> float:
    # 评估
    return 0.91

@dsl.pipeline(name="train-deploy")
def pipeline(data_path: str = "s3://data/train/"):
    train_task = train_op(data_path=data_path)
    eval_task = evaluate_op(model_path=train_task.output)
```

## 七、CD 的反模式

1. **手动 cp 权重上线**：不可追溯、不可回滚。
2. **训练和部署在同一个分支**：评估不通过无法阻断。
3. **没有评估门槛**：新模型"看上去差不多"就上线，实则在边角 case 翻车。
4. **训练数据版本与代码版本脱钩**：代码 v10 + 数据 v3 实际跑的是哪个组合说不清。

## 八、组织级保障

1. **模型注册即服务**：所有上线版本必须经过 MLflow Registry，绕过即不合规。
2. **评估门槛卡死**：CI 中 `eval` 步骤失败 → 不允许 merge / deploy。
3. **回滚一键化**：每次 deploy 后保留前 2 个版本，30 秒内可回滚。
4. **审计日志**：每次部署记录 git commit + data version + image digest + 评估指标，留 1 年。

## 小结

ML CI/CD 把"训练"和"部署"变成可重复、可审计、可回滚的工程流水线。核心是**评估门槛 + 多环境渐进 + 自动回滚**。工具选型看团队规模——小团队 GitHub Actions 自托管 runner 足够，大团队 Kubeflow / Argo + K8s 更可控。下一篇我们将进入 **模型服务化**——上线之后，如何用 vLLM / Triton / TGI 等框架扛住生产流量。
