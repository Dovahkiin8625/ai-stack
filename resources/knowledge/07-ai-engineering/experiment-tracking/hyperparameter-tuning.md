# 超参数调优：从网格搜索到贝叶斯优化

超参数（learning rate、batch size、正则化系数、网络深度）往往决定了模型的"上限"。穷举所有组合既贵又不现实。本文梳理主流调优方法——Grid / Random / Bayesian / Hyperband / Population Based，并给出 Optuna + Ray Tune 的工程实现。

## 一、调优方法谱系

```text
效率 ↑
  │
  │     ┌──────────────┐
  │     │ Bayesian Opt │
  │     │ (Optuna, BOHB)│
  │     └──────────────┘
  │   ┌──────────────┐
  │   │ Hyperband/ASHA│
  │   └──────────────┘
  │ ┌──────────────┐
  │ │ Random Search│
  │ └──────────────┘
  │ ┌──────────────┐
  │ │ Grid Search  │
  │ └──────────────┘
  │
  ↓ 暴力程度 ↑
```

越往下越暴力，越往上越智能。下面逐个拆解。

## 二、Grid Search：朴素基线

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
}

search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

**优点**：实现简单、并行友好。**缺点**：维度爆炸（10 个超参 × 5 个值 = 977 万次实验）；对**不重要**的维度也穷举，浪费算力。

## 三、Random Search：横扫高维

Bergstra & Bengio 2012 的经典论文证明：**随机搜索在高维空间下远胜网格搜索**。

直觉：网格搜索对每个维度分配相同预算，如果某个维度其实无关紧要，这部分预算浪费了。随机搜索把同等预算洒到所有维度上，每维都被探索。

```python
import random

def sample_params():
    return {
        "lr": 10 ** random.uniform(-5, -1),       # log-uniform
        "batch_size": random.choice([16, 32, 64, 128]),
        "dropout": random.uniform(0.0, 0.5),
        "weight_decay": 10 ** random.uniform(-6, -2),
    }

for trial in range(100):
    params = sample_params()
    score = train_and_evaluate(params)
    log(trial, params, score)
```

**log-uniform** 是关键技巧——学习率/正则化这种"对数尺度敏感"的参数，要在 $\log$ 空间均匀采样。

## 四、Bayesian Optimization：用过去推断未来

把超参数 → 指标看作一个**黑盒函数** $f(x)$，贝叶斯优化用高斯过程（GP）或 Tree-structured Parzen Estimator（TPE）建模 $f$ 的后验分布，然后选**期望改进最大**的点。

```text
已评估点 ──→ GP 后验 ──→ 采集函数 (EI/UCB) ──→ 下一个点
    ↑                                                    │
    └─────────────── 真实评估 ←──────────────────────────┘
```

### Optuna 实现

```python
import optuna

def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "n_layers": trial.suggest_int("n_layers", 2, 8),
    }
    score = train_and_evaluate(params)
    return score

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=100, n_jobs=4)

print("best params:", study.best_params)
print("best score:", study.best_value)
```

**优点**：样本高效——几十次 trial 就能找到接近最优。**缺点**：GP 在高维（>20 维）变慢，TPE 更稳健。

### 多目标优化

业务常常要兼顾"准确率"和"延迟"：

```python
def objective(trial):
    score, latency = train_and_evaluate(...)
    return score, latency                       # 多目标

study = optuna.create_study(directions=["maximize", "minimize"])
study.optimize(objective, n_trials=200)

# 帕累托前沿
for trial in study.best_trials:
    print(trial.values, trial.params)
```

## 五、Hyperband / ASHA：早停策略

很多 trial 在前 10 epoch 就已经能看出"会失败"。早停能省 5-10× 算力。

### Hyperband

把预算分成多个 bracket，每个 bracket 内做 Successive Halving：

```text
Bracket 1: 81 configs × 1 epoch ─→ 砍半 ─→ 27 × 3 ─→ 砍半 ─→ 9 × 9 ─→ 砍半 ─→ 3 × 27 ─→ 砍半 ─→ 1 × 81
Bracket 2: 27 configs × 3 epoch ─→ ...
Bracket 3: 9 configs × 9 epoch ─→ ...
```

### ASHA（Asynchronous Successive Halving）

异步版 Hyperband——资源一空闲就拉下一个 trial，不用等所有 trial 都到同一阶段：

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

asha = ASHAScheduler(
    time_attr="training_iteration",
    max_t=100,
    grace_period=10,
    reduction_factor=3,
)

tuner = tune.Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        num_samples=100,
        scheduler=asha,
        search_alg=OptunaSearch(),         # 嵌入 Bayesian
    ),
)
tuner.fit()
```

**ASHA + TPE** 是当前深度学习调优的事实标准组合。

## 六、Population Based Training (PBT)

PBT 把超参数搜索与训练过程**并行**：维护一组（参数, 权重）二元组，每隔一段时间：

1. **Exploit**：差的复制好的参数和权重。
2. **Explore**：在好参数基础上扰动（lr × 0.8 或 × 1.2）。

```python
# 概念示意
for generation in range(10):
    population = train_one_step(population)
    population = exploit_and_explore(population)
```

PBT 能"在线"调整学习率 schedule，比固定 schedule 更灵活。DeepMind 在星际争霸 / 围棋 RL 里大量使用。

## 七、LLM 时代的特殊调优

1. **不要调 LLM 全量超参**：训练 70B 模型一次几百万美元，调 100 次不可能。
2. **用 proxy 模型调优**：在小模型（125M、350M）上做大规模搜索，把最佳超参外推到目标尺寸。
3. **聚焦数据配比 + 训练 schedule**：现代 LLM 调优主要在数据配比、learning rate schedule、warmup 比例。
4. **不要忘记录制每个 trial 的细节**：token 数、GPU 时长、能耗——这些是评估"性价比"的依据。

## 八、工具选型

| 工具 | 优势 | 适用 |
|---|---|---|
| **Optuna** | API 简洁、TPE / GP 可选、剪枝内置 | 中小规模实验 |
| **Ray Tune** | 分布式、任意搜索算法、可接 K8s | 大规模分布式调优 |
| **Weights & Biases Sweeps** | 协作 + 可视化 | 团队多人协作 |
| **Hyperopt** | 历史悠久、Bayesian 强 | 传统 ML |
| **Determined AI** | 平台化、调优 + 训练 + 服务 | 企业级 |

## 小结

超参数调优从 Grid → Random → Bayesian → Hyperband → PBT 一路演进，核心思想是**用更聪明的方式分配算力**。Optuna + ASHA + 多目标优化是当前深度学习调优的"标配组合"。LLM 时代下，调优策略要转向**proxy 模型 + 数据配比 + schedule**，把单次 trial 成本压到可承受。下一篇我们将进入 **模型部署**——如何把训练好的模型安全、稳定地推到生产环境。
