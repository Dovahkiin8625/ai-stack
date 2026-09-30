# 集成方法：从随机森林到 XGBoost

## 一、为什么集成有效

核心数学：多个独立弱模型的平均，比单模型的误差更低。

假设有 $M$ 个独立模型，每个错误率 $\epsilon = 0.3$。多数投票下：

```math
P_{\text{err}} = \sum_{k > M/2}^{M} \binom{M}{k} \epsilon^k (1-\epsilon)^{M-k}
```

$M=21$ 时多数投票错误率约 0.026——远低于单模型 0.3。

**关键条件**：模型之间要**多样性**（diversity），不能完全相同。否则集成就是"一份预测说一百遍"，没意义。

多样性的来源：

- **样本多样性**：每模型看不同数据子集（Bagging、Boosting）。
- **特征多样性**：每模型看不同特征子集（随机森林）。
- **模型多样性**：不同算法（Stacking）。
- **超参多样性**：相同算法不同超参。

## 二、Bagging：自助采样 + 投票

**Bagging**（Bootstrap Aggregating）：

1. 从训练集有放回采样 $M$ 个 bootstrap 样本。
2. 在每个 bootstrap 上训练一个基模型。
3. 预测时分类用多数投票、回归用平均。

**方差减少原理**：平均 $M$ 个独立同分布模型的预测，方差降为 $1/M$。但 bootstrap 模型并不完全独立（用同一数据集），实际效果略弱。

```python
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    bootstrap_features=False,
    n_jobs=-1
).fit(X_train, y_train)
```

## 三、随机森林：Bagging + 特征随机

**随机森林**在 Bagging 基础上加了**特征随机**——每次分裂只考虑 $\sqrt{d}$ 个随机特征（分类）或 $d/3$（回归）。

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    oob_score=True         # 用 OOB 数据做验证
).fit(X_train, y_train)
print('OOB score:', rf.oob_score_)
```

**OOB（Out-of-Bag）评估**：每个样本约 37% 不在某棵树的 bootstrap 中，可作验证集，省去额外 split。

随机森林的特性：

- ✅ 不易过拟合（树够多后几乎只降不升）。
- ✅ 几乎不需调参（默认参数就能用）。
- ✅ 自带特征重要性（基于平均不纯度下降或 permutation）。
- ❌ 模型大、预测慢、可解释性弱。

**特征重要性**：

```python
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

`shap` 给出比 sklearn 自身更可靠的特征贡献解释。

## 四、Boosting：从错误中学习

**Boosting** 的核心思想：串行训练基模型，**重点关注前序模型的错误样本**。每轮给样本重新分配权重——错的样本权重加大、对的样本权重减小。最终预测是基模型的加权组合。

数学框架（AdaBoost 推导）：

设第 $m$ 轮的样本权重 $w_i^{(m)}$，基模型 $h_m$ 的加权错误率：

```math
\epsilon_m = \frac{\sum_{i: h_m(x_i) \neq y_i} w_i^{(m)}}{\sum_i w_i^{(m)}}
```

基模型权重：

```math
\alpha_m = \frac{1}{2} \ln \frac{1 - \epsilon_m}{\epsilon_m}
```

样本权重更新：

```math
w_i^{(m+1)} = w_i^{(m)} \exp(-\alpha_m y_i h_m(x_i))
```

最后预测：

```math
H(x) = \text{sign}\!\left( \sum_{m=1}^{M} \alpha_m h_m(x) \right)
```

直觉：错得越厉害的样本获得越大权重，下一轮必须"重点攻克"它。

## 五、梯度提升（GBDT）：用梯度代替权重

**Gradient Boosting** 把 boosting 看成函数空间的梯度下降：

1. 初始化 $F_0(x) = \arg\min_\gamma \sum_i L(y_i, \gamma)$。
2. 对 $m = 1, \dots, M$：
   - 计算**残差**（负梯度）$r_i^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$。
   - 用基模型 $h_m$ 拟合这些残差。
   - 用**线搜索**找学习率：$\gamma_m = \arg\min_\gamma \sum_i L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$。
   - 更新：$F_m(x) = F_{m-1}(x) + \eta \gamma_m h_m(x)$。

```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3
).fit(X_train, y_train)
```

**关键参数**：

- `n_estimators`（树的数量）+ `learning_rate`（步长）相互制约——步长小要多棵树。
- `max_depth`：通常 3–8，浅树比深树更稳。
- `subsample`：随机采样部分样本加随机性（"Stochastic GBDT"）。

## 六、XGBoost：GBDT 的工业级实现

XGBoost（eXtreme Gradient Boosting）做了几项关键优化：

**1. 二阶泰勒展开的目标**：

```math
\mathcal{L}^{(t)} = \sum_i\!\left[ L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
```

其中 $g_i, h_i$ 是一阶、二阶梯度。比 GBDT 用一阶更精确。

**2. 正则化目标**：

```math
\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_j w_j^2
```

$T$ 是叶子数、$w_j$ 是叶子权重，$\gamma, \lambda$ 控制复杂度。

**3. 工程优化**：

- 分位数近似（Quantile Sketch）找分裂点。
- 稀疏感知分裂（处理缺失值）。
- 块结构（Column Block）并行。
- Cache-aware 访问、Out-of-core 计算。

```python
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test, label=y_test)
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.1,                 # 学习率
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,              # L2
    'alpha': 0.0,               # L1
    'eval_metric': 'auc',
    'tree_method': 'hist',      # 直方图加速
}
bst = xgb.train(params, dtrain, num_boost_round=500, evals=[(dtest, 'test')],
                early_stopping_rounds=20)
```

## 七、LightGBM：更快更省

LightGBM 在 XGBoost 基础上做了进一步优化：

- **Histogram-based split finding**：把连续特征分桶，在桶内找分裂点。
- **Leaf-wise growth**：每次分裂增益最大的叶子（而非层级生长），更深但更准。
- **GOSS（Gradient-based One-Side Sampling）**：保留大梯度样本，随机采样小梯度样本。
- **EFB（Exclusive Feature Bundling）**：把互斥稀疏特征合并，减少有效特征数。

```python
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l2': 1.0,
}
gbm = lgb.train(params, train_data, num_boost_round=500, valid_sets=[lgb.Dataset(X_test, y_test, reference=train_data)], callbacks=[lgb.early_stopping(20)])
```

**XGBoost vs LightGBM vs CatBoost**：

| 库 | 速度 | 精度 | 类别特征 | 缺失值 | 备注 |
|---|---|---|---|---|---|
| XGBoost | 中 | 高 | 需手动编码 | 原生支持 | 最成熟 |
| LightGBM | 快 | 高 | 有限支持 | 原生支持 | 大数据首选 |
| CatBoost | 中 | 高 | 原生支持 | 原生支持 | 类别特征多时最强 |

## 八、Stacking：跨模型的"元学习"

把多个**不同**模型的预测作为输入，训练一个"元模型"（meta-learner）做最终预测。

```python
from sklearn.ensemble import StackingClassifier
estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('svm', SVC(probability=True)),
    ('rf', RandomForestClassifier(n_estimators=200)),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,                       # 防止元模型看到训练集预测（数据泄漏）
    stack_method='predict_proba',
    n_jobs=-1
).fit(X_train, y_train)
```

**关键细节**：

- **用 CV 生成元特征**（stack_method 的 CV）——否则元模型直接看到训练集预测会过拟合。
- **基模型要多样性**——同质模型集成收益有限。
- **元模型要简单**——通常用线性或浅层模型，否则容易过拟合。

## 九、Boosting vs Bagging 的对比

| 维度 | Bagging | Boosting |
|---|---|---|
| 训练方式 | 并行 | 串行 |
| 主要降低 | 方差 | 偏差 |
| 对过拟合 | 不易过拟合 | 容易过拟合（要早停） |
| 可解释性 | 一般 | 一般 |
| 训练速度 | 快 | 慢 |
| 预测速度 | 慢（要算 $M$ 个模型） | 快（树浅） |

## 十、实战陷阱与最佳实践

**1. 类别不平衡**：用 `scale_pos_weight`（XGBoost）或 `is_unbalance=True`（LightGBM）调整权重，或对少数类上采样。

**2. 类别特征**：CatBoost 能原生处理，XGBoost/LightGBM 建议先做 target encoding / count encoding。

**3. 早停**：几乎所有 GBDT 训练都应配合 `early_stopping_rounds`（如 20–50）。

**4. 超参调优顺序**（影响最大）：

```text
learning_rate × num_rounds  →  max_depth  →  min_data_in_leaf
                          →  subsample / colsample  →  正则项
```

先用较大的 `learning_rate`（如 0.1）找近似最优树数，再降到 0.01–0.05 精细调。

**5. 特征工程仍然重要**：GBDT 不需要标准化、能处理缺失值，但好的特征工程（交叉特征、时间特征、统计特征）仍是性能上限的决定因素。

**6. SHAP 值解释**：

```python
import shap
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)   # 全局特征重要性
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])  # 单样本解释
```

## 小结

集成学习的核心是"**多样性 + 组合**"：Bagging（随机森林）通过自助采样和特征随机降低方差，Boosting（GBDT / XGBoost / LightGBM）通过串行纠错降低偏差，Stacking 把不同算法当作"基学习器"再用元学习器组合。掌握这套方法，你能在表格数据上拿到 SOTA 性能，也能用 SHAP 等工具解释预测——这是工业 ML 工程师的必备技能。