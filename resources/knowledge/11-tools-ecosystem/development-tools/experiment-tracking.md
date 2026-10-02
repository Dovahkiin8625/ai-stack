# 实验追踪：Weights & Biases、MLflow、TensorBoard

ML 实验管理是工程化的核心——上百次实验、超参数组合、模型版本、数据集版本，混元管理会迅速失控。本文对比主流实验追踪工具（Weights & Biases、MLflow、TensorBoard、ClearML），并讨论最佳实践。

## 一、为什么需要实验追踪

```python
# ML 实验的典型混乱状态
ml_chaos = {
    "1. 实验太多": "微调、prompt、数据组合指数级增长",
    "2. 参数分散": "超参散落在代码、命令行、配置文件中",
    "3. 结果丢失": "训练日志、指标散落各目录",
    "4. 复现不一致": "环境、依赖、数据版本不可复现",
    "5. 对比困难": "多次实验结果难以横向比较",
    "6. 团队协作": "实验结果无法在成员间共享",
    "7. 选型决策": "没有历史数据，难以决定下一步方向",
}

# 实验追踪的核心价值
value = [
    "记录每次运行：参数、指标、artifact",
    "对比多次实验：曲线、表格、散点图",
    "追踪资源：GPU/CPU/内存使用",
    "复现：代码、依赖、环境快照",
    "团队：共享 dashboard",
    "决策：用数据驱动的迭代",
]
```

## 二、Weights & Biases（wandb）

### 核心特性

```python
# W&B（业界最流行的 ML 追踪平台）
wandb_features = {
    "开发者": "Weights & Biases",
    "类型": "SaaS（云端 + 自托管）",
    "核心": [
        "实验追踪（Experiments）",
        "可视化（Charts, Media）",
        "Artifact（数据/模型版本）",
        "Sweep（超参搜索）",
        "Reports（报告）",
        "Tables（表格）",
        "Launches（远程训练任务）",
    ],
    "集成": "PyTorch / HuggingFace / JAX / TensorFlow",
}
```

### 实验追踪示例

```python
import wandb

# 初始化 experiment
wandb.init(
    project="llm-finetune",
    name="llama-3.1-8b-alpaca-lr2e5",
    config={
        "model": "meta-llama/Llama-3.1-8B",
        "dataset": "alpaca",
        "lr": 2e-5,
        "batch_size": 4,
        "epochs": 3,
    },
)

# 训练循环中记录
for epoch in range(3):
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)
        wandb.log({
            "train/loss": loss,
            "train/lr": scheduler.get_lr(),
            "train/epoch": epoch,
            "train/step": step,
        })
    eval_loss = evaluate(model, val_loader)
    wandb.log({"eval/loss": eval_loss, "eval/epoch": epoch})

# 记录 artifact
wandb.log({"sample_outputs": wandb.Table(
    columns=["prompt", "output"],
    data=[["什么是 LLM?", "Large Language Model 是..."]],
)})

# 保存模型 artifact
artifact = wandb.Artifact("llama-8b-alpaca", type="model")
artifact.add_file("model.bin")
wandb.log_artifact(artifact)

wandb.finish()
```

### Sweep：自动超参搜索

```python
# wandb sweep 配置
sweep_config = {
    "method": "bayes",  # 或 "grid", "random"
    "metric": {"name": "eval/loss", "goal": "minimize"},
    "parameters": {
        "lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-3},
        "batch_size": {"values": [4, 8, 16, 32]},
        "warmup_ratio": {"values": [0.0, 0.05, 0.1]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="llm-finetune")

# 运行 sweep agent
wandb.agent(sweep_id, function=train_function, count=20)
```

## 三、MLflow

### 核心特性

```python
# MLflow（开源，自我托管）
mlflow_features = {
    "开发者": "Databricks",
    "类型": "开源（Apache 2.0）",
    "组件": [
        "MLflow Tracking —— 追踪",
        "MLflow Projects —— 项目打包",
        "MLflow Models —— 模型格式",
        "MLflow Registry —— 模型注册中心",
    ],
    "优点": "完全开源，可本地部署，无数据外传",
    "缺点": "UI 较简陋，协作功能弱于 W&B",
}
```

### Tracking 用法

```python
import mlflow
import mlflow.pytorch

# 设置 tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("llm-finetune")

# 开始 run
with mlflow.start_run(run_name="llama-3.1-8b-alpaca"):
    # 记录参数
    mlflow.log_param("model", "meta-llama/Llama-3.1-8B")
    mlflow.log_param("lr", 2e-5)
    mlflow.log_param("batch_size", 4)
    mlflow.log_params({"epochs": 3, "warmup_ratio": 0.1})

    # 训练
    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            mlflow.log_metric("train_loss", loss, step=step)
        mlflow.log_metric("eval_loss", eval_loss, step=epoch)

    # 记录模型
    mlflow.pytorch.log_model(model, "model")

    # 记录 artifact
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("tokenizer")

# 查询历史实验
runs = mlflow.search_runs(experiment_ids=["1"])
best_run = runs.sort_values("metrics.eval_loss").iloc[0]
print(best_run[["params.lr", "metrics.eval_loss"]])
```

### Model Registry

```python
# MLflow Registry 模型生命周期管理
import mlflow

# 注册模型
mlflow.register_model(
    "runs:/abc123/model",
    "llama-8b-production",
)

# 模型阶段转换
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="llama-8b-production",
    version=3,
    stage="Production",   # Staging -> Production -> Archived
)

# 加载模型
model = mlflow.pytorch.load_model("models:/llama-8b-production/3")
```

## 四、TensorBoard

### 核心特性

```python
# TensorBoard（TensorFlow 官方，PyTorch 也支持）
tensorboard_features = {
    "开发者": "TensorFlow 团队",
    "类型": "开源",
    "特点": [
        "轻量、本地使用",
        "实时监控（Web 界面）",
        "可视化：loss curve、histogram、image、text",
        "免费",
        "适合单机训练",
    ],
    "缺点": "协作、模型注册、artifact 管理弱",
}
```

### PyTorch 集成

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/llama-3.1-8b-experiment-1")

# 记录标量
for epoch in range(3):
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/eval", eval_loss, epoch)
    writer.add_scalar("LR", scheduler.get_lr(), epoch)

# 记录 histogram
for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)

# 记录 image
writer.add_image("generated_image", img_tensor, epoch)

# 记录 text
writer.add_text("sample_output", "Model output: ...", epoch)

# 记录 graph
writer.add_graph(model, input_to_model)

# 记录 PR curve / confusion matrix
writer.add_pr_curve("pr_curve", labels, predictions)

writer.close()

# 启动 tensorboard
# tensorboard --logdir=runs
```

## 五、ClearML

```python
# ClearML（前身 allegro.ai）
# 自动追踪 + 完整 MLOps

clearml_features = {
    "类型": "开源 + 商业",
    "特点": [
        "自动代码快照（无需手动 log）",
        "Pipeline 编排",
        "模型注册",
        "远程执行",
        "Web dashboard 完善",
    ],
    "适合": "需要完整 MLOps 平台",
}

# 自动追踪
import clearml
from clearml import Task

task = Task.init(
    project_name="llm-finetune",
    task_name="llama-8b-experiment-1",
)

# 自动记录：
# - 代码（git commit）
# - 参数（argparse / config）
# - 指标（pytorch tensorboard）
# - artifact（pytorch model）

# 远程执行
task.execute_remotely(queue_name="gpu-queue", clone=False)
```

## 六、对比表格

```python
# 实验追踪工具对比
comparison = {
    "feature": {
        "W&B": {
            "易用性": "★★★★★",
            "UI": "★★★★★",
            "开源": "部分开源",
            "价格": "$/用户/月",
            "协作": "★★★★★",
            "Artifact 管理": "★★★★★",
            "超参搜索": "★★★★★ (内置)",
            "Pipeline 编排": "★★★★ (Launches)",
            "私有部署": "支持 (Enterprise)",
        },
        "MLflow": {
            "易用性": "★★★★",
            "UI": "★★★",
            "开源": "完全开源",
            "价格": "免费",
            "协作": "★★★",
            "Artifact 管理": "★★★★",
            "超参搜索": "需集成 Optuna",
            "Pipeline 编排": "需 MLflow Projects",
            "私有部署": "完全自托管",
        },
        "TensorBoard": {
            "易用性": "★★★★★",
            "UI": "★★★",
            "开源": "完全开源",
            "价格": "免费",
            "协作": "★★",
            "Artifact 管理": "★★",
            "超参搜索": "无",
            "Pipeline 编排": "无",
            "私有部署": "本地",
        },
        "ClearML": {
            "易用性": "★★★★",
            "UI": "★★★★",
            "开源": "部分开源",
            "价格": "免费/付费",
            "协作": "★★★★",
            "Artifact 管理": "★★★★",
            "超参搜索": "★★★★ (内置)",
            "Pipeline 编排": "★★★★★",
            "私有部署": "支持",
        },
    },
}
```

## 七、Prompt 实验追踪

```python
# LLM 时代还需要追踪 prompt 实验
# 因为 prompt 是新的"超参"

# 用 W&B 追踪 prompt
import wandb

wandb.init(project="prompt-experiment")

# 记录 prompt 版本
prompts = {
    "v1": "Translate to French: {text}",
    "v2": "Please translate the following English text to French, keeping the meaning and tone:\n{text}",
    "v3": "你是一名专业翻译，请将以下英文翻译成法语：\n{text}",
}

results = {}
for version, prompt in prompts.items():
    outputs = [call_llm(prompt.format(text=t)) for t in test_set]
    quality = evaluate_translations(outputs)
    wandb.log({
        f"prompt_{version}/quality": quality,
        f"prompt_{version}/length": np.mean([len(o) for o in outputs]),
    })
    results[version] = quality

# Trace：单次推理的完整链路
wandb.log({"llm_trace": wandb.Table(
    columns=["prompt", "response", "latency", "tokens"],
    data=[(p, q.r, latency, n_tokens)],
)})
```

## 八、最佳实践

```python
# 实验追踪的最佳实践

best_practices = {
    "命名规范": [
        "1. 实验名包含关键参数（model-lr-bs-epochs）",
        "2. 每次实验一个 run，不混淆",
        "3. 用 group 聚合相关实验",
    ],
    "记录什么": [
        "1. 所有超参（即使默认）",
        "2. 训练/测试指标",
        "3. GPU/CPU/内存使用",
        "4. 代码版本（git commit）",
        "5. 数据集版本（hash 或版本号）",
        "6. 环境（python、torch 版本）",
        "7. 模型 checkpoint",
        "8. 失败案例 + 异常",
    ],
    "不要记录": [
        "1. 敏感数据（API key、PII）",
        "2. 大型原始数据集（用 hash 代替）",
        "3. 冗余信息（每次 step 记录）",
    ],
    "工作流": [
        "1. 实验前：检查目录命名、跟 project",
        "2. 实验中：每个 epoch 记录、监控异常",
        "3. 实验后：写实验结论、写报告",
        "4. 复盘：定期回顾、清理无效实验",
    ],
}
```

## 九、与 CI/CD 集成

```python
# 实验追踪 + CI/CD 流程

mlops_pipeline = {
    "1. 触发": "代码 push / 数据更新",
    "2. 实验运行": "自动启动训练（ClearML/MLflow）",
    "3. 评估": "自动评测（MMLU/HumanEval）",
    "4. 决策": [
        "指标 > 阈值 → 提升到 Staging",
        "Staging 测试通过 → 提升到 Production",
        "失败 → 通知",
    ],
    "5. 部署": "Kubernetes / Serverless",
    "6. 监控": "线上指标回流",
    "7. 触发再训练": "数据漂移/性能下降",
}
```

## 小结

实验追踪是 ML 工程化的"基础设施"——**W&B 适合云端协作、MLflow 适合自托管、TensorBoard 适合轻量单机、ClearML 适合完整 MLOps**。生产环境建议用 W&B 或 MLflow，配合 Git + CI/CD 实现完整闭环。下一篇我们讨论 **Prompt IDE 工具**——LangSmith、PromptLayer、PromptTools。