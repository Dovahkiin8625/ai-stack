# MLflow 基础：实验追踪、模型注册与项目复现

ML 训练是**高度实验性**的工作：调一个学习率、换一种数据增强、试试不同 optimizer——都可能让结果发生戏剧性变化。MLflow 把"实验"这件事做了标准化：参数、指标、artifact、模型、代码版本都记录下来，让"为什么这次跑得好"可追溯。本文从核心组件出发，演示一个端到端实验记录 + 模型注册流程。

## 一、MLflow 的四大组件

```text
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Tracking    │  │ Projects    │  │ Models      │  │ Registry    │
│             │  │             │  │             │  │             │
│ 记录参数/   │  │ 打包代码    │  │ 标准化模型  │  │ 版本管理/   │
│ 指标/artifact│ │ 一键复现    │  │ 多框架导出  │  │ 阶段晋升    │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

最常用的是 **Tracking** 和 **Registry**——前者记录实验，后者管理模型生命周期。

## 二、Tracking：每次跑都留痕

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("churn-prediction")

with mlflow.start_run(run_name="rf-v1"):
    # 1. 记录超参数
    n_estimators = 200
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # 2. 训练
    X_train, X_test, y_train, y_test = train_test_split(...)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train, y_train)

    # 3. 记录指标
    y_pred = clf.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1", f1_score(y_test, y_pred, average="macro"))

    # 4. 记录 artifact（图、报告、特征重要性）
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
    mlflow.log_figure(fig, "feature_importance.png")

    # 5. 记录模型
    mlflow.sklearn.log_model(clf, "model",
                              registered_model_name="churn-rf")

print("Run 完成，UI: http://mlflow-server:5000")
```

每次 `start_run()` 都生成一条记录，包含完整上下文：

- **Parameters**：超参（n_estimators、lr、batch_size）
- **Metrics**：指标（accuracy、loss、P99 latency）
- **Artifacts**：文件（图表、混淆矩阵、HTML 报告、ONNX 模型）
- **Tags**：自定义标签（"baseline"、"production-candidate"）
- **Source**：代码 git commit、运行命令

## 三、自动记录：autolog

很多框架 MLflow 支持 autolog——一行开启自动捕获：

```python
mlflow.sklearn.autolog()           # sklearn
mlflow.pytorch.autolog()           # PyTorch
mlflow.tensorflow.autolog()        # TF/Keras
mlflow.transformers.autolog()      # HuggingFace
```

之后所有 `fit()` 自动记录 n_estimators、max_depth、training_loss、learning_rate 调度曲线等。

## 四、UI 与对比

MLflow UI 提供实验对比视图：

```text
Run Name       │ accuracy │ f1    │ n_estimators │ max_depth │ duration
───────────────┼──────────┼───────┼──────────────┼───────────┼─────────
rf-v1          │ 0.872    │ 0.851 │ 200          │ 10        │ 12.3s
rf-v2          │ 0.881    │ 0.864 │ 500          │ 15        │ 28.1s
xgb-v1         │ 0.893    │ 0.879 │ -            │ 6         │  8.7s
```

可以选多个 run 做 **Parallel Coordinates Plot**——一眼看出哪个超参维度对指标影响最大。

## 五、Model Registry：版本化模型

光记录模型不够，还要管理**生命周期**：

```text
None → Staging → Production → Archived
```

```bash
# CLI 方式
mlflow models transition-model-version-stage \
    --model-name "churn-rf" \
    --version 3 \
    --stage "Production"
```

或通过 Python SDK：

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.transition_model_version_stage(
    name="churn-rf",
    version=3,
    stage="Production",
)

# 加描述 / 注释
client.update_model_version(
    name="churn-rf",
    version=3,
    description="2024-Q1 上线版本，AUC=0.91"
)
```

Registry 还支持**模型血缘**——自动关联到生成它的 run、run 关联到代码 commit、commit 关联到数据版本。

## 六、Projects：用 MLproject 打包代码

`MLproject` 文件定义可复现的运行环境：

```yaml
# MLproject
name: churn-prediction
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 200}
      max_depth:    {type: int, default: 10}
    command: "python train.py {n_estimators} {max_depth}"
```

```bash
mlflow run . -P n_estimators=500 -P max_depth=15
```

MLflow 自动 git checkout 到指定 commit、用 conda / docker 重建环境、记录所有参数和输出。这解决了"在我机器上能跑"的经典噩梦。

## 七、生产实践要点

1. **Tracking server 选型**：
   - 单机 SQLite：本地开发
   - 远程 + PostgreSQL + S3：团队共享
   - Databricks Managed MLflow：云上全家桶

2. **artifact 存储**：用 S3 / GCS / Azure Blob + 对象存储生命周期（30 天清理中间 run，永久保留 production run）。

3. **metric 记录频率**：训练中每 N step 调一次 `log_metric`，避免每 step 写入造成 DB 压力。

4. **Tag 规范**：约定 `team` / `project` / `dataset_version` 等公共 tag，方便后续过滤。

5. **敏感数据**：artifact 里不要直接 log 原始训练数据（太大 + 合规风险），只 log 摘要统计。

## 小结

MLflow 是 ML 实验管理的"事实标准"。它用统一的 API 把参数、指标、artifact、模型全部记录下来，配合 Model Registry 做版本与生命周期管理，配合 Projects 保证可复现。一套 MLflow 跑起来后，"这个模型效果为什么好"、"上次的最佳参数是什么"、"哪个版本在线上跑"这些问题都可以秒答。下一篇我们将进入 **实验可重现性**——除了记录，还要从工程上保证"明天还能跑出同样的结果"。
