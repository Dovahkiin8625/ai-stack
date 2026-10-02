# 漂移检测：Data Drift、Concept Drift 与 LLM 时代的 Embedding Drift

ML 模型上线后效果慢慢变差，90% 的根因是**漂移（drift）**。漂移分三类：输入分布变了（Data Drift）、输入与输出的关系变了（Concept Drift）、输出分布本身异常（Prediction Drift）。本文梳理这三类漂移的数学检测方法、工具实现，以及 LLM 应用特有的 embedding 漂移与 prompt 漂移。

## 一、三类漂移定义

```text
Data Drift (X drift):
  P(X) 变了, P(Y|X) 没变
  例：用户年龄分布向年轻化偏移，但同样年龄的购买习惯没变

Concept Drift (X→Y 关系 drift):
  P(Y|X) 变了, P(X) 没变
  例：同样搜索词，现在用户期望的商品类型变了（季节性 / 政策影响）

Prediction Drift (Y drift):
  P(Y) 分布变了
  例：原本 90% 预测为类别 A，现在 70% 是类别 B
```

实际生产中三者经常同时发生。

## 二、Data Drift 检测

### 2.1 单变量检测

#### KS 检验（连续变量）

```python
from scipy.stats import ks_2samp

def ks_drift(reference: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> dict:
    stat, p = ks_2samp(reference, current)
    return {
        "statistic": stat,
        "p_value": p,
        "drift": p < alpha,             # p < 0.05 拒绝"分布一致"假设
        "severity": "high" if stat > 0.2 else ("medium" if stat > 0.1 else "low"),
    }
```

#### Chi-Square 检验（类别变量）

```python
from scipy.stats import chisquare

def chi2_drift(ref_counts: np.ndarray, cur_counts: np.ndarray) -> dict:
    # 把 ref 的分布作为期望
    expected = ref_counts / ref_counts.sum() * cur_counts.sum()
    stat, p = chisquare(cur_counts, expected)
    return {"statistic": stat, "p_value": p, "drift": p < 0.05}
```

#### PSI（Population Stability Index）

适合生产监控的"分位数法"，不依赖统计假设：

```python
import numpy as np

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    expected_counts = np.histogram(expected, breakpoints)[0] + 1e-6
    actual_counts   = np.histogram(actual, breakpoints)[0] + 1e-6
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct   = actual_counts / actual_counts.sum()
    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

# PSI 阈值（业界经验）
# PSI < 0.1:  无显著漂移
# 0.1 - 0.25: 轻微漂移，警惕
# > 0.25:     显著漂移，需要重训
```

### 2.2 多变量检测

单变量"都 OK"≠ 整体 OK。变量之间的相关性可能变了。

#### Hotelling's T² 检验（高维均值）

```python
from scipy.stats import hotellings_t2

def hotelling_drift(X_ref: np.ndarray, X_cur: np.ndarray):
    stat, p = hotellings_t2(X_ref, X_cur)
    return {"statistic": stat, "p_value": p}
```

适用特征数 ≤ 几十的场景。

#### 最大均值差异（MMD）

用核方法度量两个分布的距离：

```python
import torch

def mmd_rbf(X: torch.Tensor, Y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """RBF 核的 MMD 距离"""
    XX = torch.cdist(X, X) ** 2
    YY = torch.cdist(Y, Y) ** 2
    XY = torch.cdist(X, Y) ** 2
    return torch.exp(-gamma * XX).mean() + torch.exp(-gamma * YY).mean() - 2 * torch.exp(-gamma * XY).mean()

# MMD 越大，分布差异越大；阈值靠经验或 bootstrap
```

#### 分类器法（Unsupervised Domain Adaptation）

训练一个二分类器区分"参考 vs 当前"，看 AUC：

```python
from sklearn.ensemble import GradientBoostingClassifier

def classifier_drift(X_ref, X_cur):
    X = np.vstack([X_ref, X_cur])
    y = np.array([0] * len(X_ref) + [1] * len(X_cur))

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    clf.fit(X, y)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])

    # AUC ≈ 0.5 → 两分布难区分 → 无漂移
    # AUC → 1.0 → 完美区分 → 严重漂移
    return {"auc": auc, "drift_score": 2 * abs(auc - 0.5)}
```

经验：AUC > 0.7 表示显著漂移。

## 三、Concept Drift 检测

Concept Drift 检测比 Data Drift 难——它需要**真实标签**，但生产中标签往往是延迟的（用户次日才反馈、退货要等 30 天）。

### 3.1 ADWIN（自适应窗口）

```python
# River: 在线 ML 库
from river import drift

detector = drift.ADWIN()
for i, (x, y) in enumerate(stream):
    y_pred = model.predict(x)
    err = abs(y_pred - y)
    detector.update(err)
    if detector.drift_detected:
        print(f"漂移在样本 {i} 检测到！")
        model = train_new_model()    # 触发再训练
```

### 3.2 Page-Hinkley

```python
from river import drift

detector = drift.PageHinkley()
for x, y in stream:
    err = abs(model.predict(x) - y)
    detector.update(err)
    if detector.drift_detected:
        ...
```

适合**突变式**漂移。

### 3.3 DDM（Drift Detection Method）

```python
# 监控错误率 + 标准差
p_i = errors_so_far / i
s_i = sqrt(p_i * (1 - p_i) / i)

if p_i + s_i > p_min + 2 * s_min:    # 警告
    warn()
if p_i + s_i > p_min + 3 * s_min:    # 漂移
    drift()
    p_min, s_min = p_i, s_i          # 重置基准
```

## 四、Prediction Drift

监控**模型输出分布**比监控输入更直接——它直接反映了"现在模型怎么想"。

```python
@app.post("/v1/predict")
async def predict(req: PredictRequest):
    pred = await model.predict(req)
    PRED_SCORE_HISTOGRAM.observe(pred.score)
    PRED_LABEL.labels(label=pred.label).inc()
    return pred
```

如果一个二分类模型上线时预测为 1 的比例是 10%，某天突然变成 50%——强烈暗示输入分布或模型行为异常。

## 五、LLM 特有的漂移

### 5.1 Embedding Drift

LLM 应用中，用户 query 的 embedding 分布变化往往先于业务指标变化。

```python
from prometheus_client import Histogram

EMB_MEAN = Histogram("emb_dim0", "Embedding dim0 mean", buckets=[-2, -1, 0, 1, 2])
EMB_STD  = Histogram("emb_dim1", "Embedding dim1 std",  buckets=[0, 0.5, 1, 1.5, 2])

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    emb = embedding_model.encode(req.prompt)
    EMB_MEAN.observe(emb[0])
    EMB_STD.observe(emb[1])
    return await llm.generate(req)
```

更严谨：

```python
# 主成分漂移
from sklearn.decomposition import PCA

pca = PCA(n_components=5).fit(ref_embeddings)

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    emb = embedding_model.encode(req.prompt)
    proj = pca.transform([emb])[0]
    for i, val in enumerate(proj):
        EMB_PC.labels(pc=f"pc{i}").observe(val)
    ...
```

主成分的偏移往往反映出"用户开始问另一种类型的问题"。

### 5.2 Prompt Template 漂移

新接入的渠道（Slack、小程序）可能让 prompt 格式变化：

```python
@app.post("/v1/chat")
async def chat(req: ChatRequest):
    # 监控 prompt 长度分布
    PROMPT_LEN.observe(len(req.prompt))
    # 监控 system prompt 模板版本
    TEMPLATE_VERSION.labels(version=req.template_version).inc()
    return await llm.generate(req)
```

### 5.3 输出质量漂移

LLM 输出的"质量"难以量化，但可以用以下代理指标：

- **回复长度分布**：突然变长 / 变短往往不正常。
- **JSON 解析失败率**：如果输出应该 JSON，parse 失败 = 模型行为异常。
- **幻觉率**：用 LLM-as-judge 抽检 1% 输出看是否有幻觉。
- **重复率**：n-gram 重复率飙升说明模型退化。
- **用户反馈率**：点赞率下降是最直接信号。

## 六、检测频率与告警

```text
实时（每条请求）:
  - 输入 schema 校验
  - 异常值检测（如年龄 > 150）
  - 业务硬约束（必须有 user_id）

分钟级（每 5 分钟聚合）:
  - QPS / 错误率 / 延迟
  - 输出分布

小时级:
  - PSI 单变量漂移
  - 用户反馈聚合

天级:
  - 多变量 MMD / Hotelling
  - Embedding 主成分漂移
  - 在线指标（需要标签回流）
```

## 七、自动响应

检测到漂移后的动作分级：

```python
def on_drift(drift_info):
    if drift_info["severity"] == "low":
        log_warning(drift_info)
    elif drift_info["severity"] == "medium":
        alert_oncall(drift_info)
        # 增大采样率，准备更多数据
    elif drift_info["severity"] == "high":
        alert_oncall(drift_info, severity="page")
        # 自动触发重训 pipeline
        trigger_retrain()
        # 或回滚到上一个稳定版本
        rollback_to_last_stable()
```

## 小结

漂移检测是 ML 监控的核心难点。Data Drift 靠单变量（KS/Chi-Square/PSI）+ 多变量（MMD/Hotelling）方法；Concept Drift 靠在线流式检测（ADWIN/DDM）；Prediction Drift 看输出分布。LLM 时代还要额外关注 embedding 漂移、prompt 漂移、输出质量代理指标。把漂移检测做成"分级自动响应"——告警 → 增大采样 → 重训 → 回滚——才能在漂移初期把损失降到最低。下一篇我们将进入 **可观测性栈**——把指标、日志、trace 整合成一套完整观测体系。
