# 模型评估与选择

## 一、分类任务的核心指标

### 准确率与它的局限

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**问题**：在 99% 负样本的数据上，预测全负类也能拿到 99% 准确率——完全没用。

### 精确率、召回率、F1

把正类视为"关注类"：

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}
$$

F1 是精确率与召回率的调和平均：

$$
F_1 = \frac{2 \cdot P \cdot R}{P + R}
$$

F1 适用于"正负类不平衡、两者都要关注"（如信息检索、欺诈检测）。

### ROC 与 PR 曲线

**ROC 曲线**：横轴 FPR（假正率 = $FP / (FP + TN)$），纵轴 TPR（真正率 = Recall）。AUC-ROC 衡量"随机正样本排在随机负样本前的概率"。

**PR 曲线**：横轴 Recall，纵轴 Precision。**类别极不平衡时，PR-AUC 比 ROC-AUC 更能反映模型好坏**（ROC 的 FPR 在大量负样本下"虚高"）。

```python
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_prob))
print('PR  AUC:', average_precision_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))
```

### 多分类与多标签

- **宏平均（Macro）**：先算每类指标再平均，对所有类同等对待。
- **微平均（Micro）**：把每类的 TP/FP/FN 累加再算指标，受大类影响。
- **加权平均（Weighted）**：按各类样本数加权。

多标签任务用 **Hamming loss**（预测错的标签比例）、**subset accuracy**（全对才计数）等。

## 三、回归任务的核心指标

$$
\text{MSE}  = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2, \quad
\text{RMSE} = \sqrt{\text{MSE}}
$$

$$
\text{MAE}   = \frac{1}{N}\sum_i |y_i - \hat{y}_i|, \quad
\text{MAPE} = \frac{1}{N}\sum_i \left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

**MAE 对异常值更鲁棒**；**MSE/RMSE 对大误差更敏感**（适合"不要有大错"的场景）；**MAPE 在 $y$ 接近 0 时不稳定**；**$R^2$ 可为负**（模型比均值还差）。

## 四、排序与检索指标**（信息检索 / 推荐）

**NDCG@k**：归一化折损累计增益，对相关度分级排序敏感。

**MRR（Mean Reciprocal Rank）**：$\text{MRR} = \frac{1}{Q}\sum_{q} \frac{1}{\text{rank}_q}$，第一个相关结果位置的倒数。

**MAP（Mean Average Precision）**：对每个 query 算 AP、再平均。

**Hit@k**：前 $k$ 个结果中是否有相关项。Recall@k 类似。

```python
from sklearn.metrics import ndcg_score
# 真实相关度（多个 query 的列表）
true_relevance = [[3, 2, 1, 0, 0]]
scores         = [[0.9, 0.7, 0.5, 0.3, 0.1]]
print('NDCG@5:', ndcg_score(true_relevance, scores, k=5))
```

## 五、概率预测的校准

模型输出的概率应该反映"真实频率"。**校准曲线**（reliability diagram）画预测概率 vs 实际频率。

```python
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
# 对角线 y=x 表示完美校准
```

**Brier Score** = $\frac{1}{N}\sum_i (p_i - y_i)^2$，衡量概率预测的均方误差。

**ECE（Expected Calibration Error）**：把预测概率分桶，桶内 |平均预测 - 实际频率| 按样本数加权平均。

**校准方法**：

- **Platt scaling**：对 logits 训练一个 sigmoid。
- **Isotonic regression**：单调拟合，校准非参数化但易过拟合。
- **Temperature scaling**：除 logits 一个温度 $T$，最简单且常用于深度学习。

```python
# Temperature scaling
T = torch.nn.Parameter(torch.ones(1) * 1.5)
logits_T = logits / T
```

## 六、交叉验证：黄金标准

**K 折交叉验证**：把数据分 $K$ 折，每次用 $K-1$ 折训练、1 折验证，循环 $K$ 次取平均。

```python
from sklearn.model_selection import KFold, cross_val_score
scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
print(f'CV: {scores.mean():.3f} ± {scores.std():.3f}')
```

**变体**：

- **Stratified K-Fold**：每折保持类别比例（分类任务默认）。
- **Group K-Fold**：同组样本不跨折（防数据泄漏）。
- **Time Series Split**：按时间切分，前段训练、后段验证。
- **Leave-One-Out**：极端情况 $K=N$，代价大、方差大。

**嵌套交叉验证**：

- 外层：评估泛化性能。
- 内层：在每折训练集上做超参搜索。

避免"用测试集调超参"导致评估结果过乐观。

## 七、Bootstrap 与置信区间

**Bootstrap** 解决"小样本下难估计方差"的问题：

1. 从原始 $N$ 个样本中有放回抽样 $N$ 个，得到一个 bootstrap 样本。
2. 在 bootstrap 样本上训练模型、记录指标。
3. 重复 $B$（如 1000）次，得到指标分布。

```python
from sklearn.utils import resample
scores = [evaluate(resample(X, y), resample(X_test, y_test)) for _ in range(1000)]
print(f'95% CI: {np.percentile(scores, [2.5, 97.5])}')
```

## 八、超参搜索

### 网格搜索（Grid Search）

```python
from sklearn.model_selection import GridSearchCV
params = {'n_estimators': [100, 500], 'max_depth': [3, 6, 10]}
gs = GridSearchCV(model, params, cv=5, scoring='f1_macro', n_jobs=-1)
gs.fit(X, y)
```

**优点**：全面。**缺点**：组合爆炸、维度灾难。

### 随机搜索（Random Search）

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
params = {'n_estimators': randint(50, 1000), 'max_depth': randint(3, 20)}
rs = RandomizedSearchCV(model, params, n_iter=50, cv=5, random_state=42)
rs.fit(X, y)
```

**经验**：随机搜索在高维超参空间比网格搜索更高效（Bergstra & Bengio 2012）。

### 贝叶斯优化

用高斯过程或 TPE 维护"超参空间里哪个区域可能更好"的概率模型，按 acquisition function（如 EI）选下一个点。

```python
import optuna
def objective(trial):
    n_est = trial.suggest_int('n_estimators', 50, 1000)
    depth = trial.suggest_int('max_depth', 3, 20)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    score = cross_val_score(...)
    return score.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**超参搜索的常见陷阱**：

- 用测试集调超参 → 评估结果过乐观。
- 没设随机种子 → 结果不可复现。
- 搜索空间太大 → 永远搜不完；用先验缩小。
- 训练数据少 → 评估结果方差大，多跑几次取平均。

## 九、数据泄漏的常见形式

数据泄漏（data leakage）会让评估结果偏离真实泛化：

- **Train-test contamination**：标准化、特征选择用到了测试集信息。
- **Target leakage**：用了包含未来信息的特征（如"用户是否流失"使用了流失后的特征）。
- **Duplicate samples**：同一样本同时出现在训练/测试。
- **Group leakage**：同一用户/同一时间段的样本跨折。

```python
# 反例：用全量数据做标准化再 split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)   # ❌ 全量 fit
X_train, X_test = train_test_split(X)  # 信息泄漏！

# 正例：先 split 再 fit
X_train, X_test = train_test_split(X)
scaler = StandardScaler().fit(X_train)  # ✅ 只用训练集
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
```

## 十、A/B 测试与因果推断

机器学习模型的最终评估往往要在**真实业务**中验证：

- **离线指标**（AUC、Recall）与**在线指标**（点击率、转化率）常常不一致。
- **A/B 测试**：把用户随机分到对照组（baseline）和实验组（新模型），比较业务指标。
- **统计显著性**：用 t 检验 / 比例检验判断差异是否显著。提前算好**样本量**避免 underpowered 实验。

```python
from scipy.stats import ttest_ind
t, p = ttest_ind(group_a_metric, group_b_metric)
print(f'p-value = {p:.4f}')  # p < 0.05 才显著
```

## 十一、如何选择"对的"评估策略？

| 任务 | 主要指标 | 注意事项 |
|---|---|---|
| 类别平衡分类 | 准确率 + ROC-AUC | 类别平衡假设 |
| 类别不平衡分类 | PR-AUC + F1 | 关注少数类 |
| 回归 | RMSE/MAE + $R^2$ | 检查离群值 |
| 排序 / 推荐 | NDCG@K + MRR | 用业务相关 K |
| 概率预测 | Brier + ECE | 必要时校准 |
| 生成（文本/图像） | 人工评估 + BLEU/CLIP | 自动指标与人类判断常不一致 |
| 真实业务 | A/B 测试 + 业务指标 | 离线-在线差异 |

## 小结

模型评估是"工程化 ML"的分水岭：选对指标、做好交叉验证、注意数据泄漏、用合适方法做超参搜索、把离线结果用 A/B 测试验证到线上，这五步构成完整的评估闭环。掌握这套方法，你就不会陷入"测试集 99% 准确率但线上完全不能用"的窘境，而是能从一堆模型里选出真正对业务有用的那个。