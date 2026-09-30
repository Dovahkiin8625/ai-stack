# 特征工程与预处理

## 一、缺失值处理

### 缺失机制

- **MCAR（Missing Completely At Random）**：缺失与任何变量无关。
- **MAR（Missing At Random）**：缺失与其他观测变量有关。
- **MNAR（Missing Not At Random）**：缺失与缺失值本身有关（如高收入者不愿填收入）。

识别方法：

- 看缺失率分布：哪些列高、是否相关。
- 与目标变量的关系：缺失组目标分布是否异常。

### 处理策略

```math
\text{删除} \;\; | \;\; \text{填充（均值/中位数/众数/0/前后值/KNN/模型预测）} \;\; | \;\; \text{单独建模"缺失"作为特征}
```

```python
import numpy as np, pandas as pd

# 1. 删除
df.dropna(subset=['col_a'])           # 删除 col_a 缺失的行
df.dropna(thresh=len(df.columns)-2)   # 至少保留 N-2 个非空

# 2. 填充
df['col_a'].fillna(df['col_a'].median(), inplace=True)
df['col_a'].fillna(method='ffill')    # 向前填充（时序）

# 3. 缺失作为特征
df['col_a_missing'] = df['col_a'].isna().astype(int)
```

**进阶：模型预测填充**

```python
from sklearn.ensemble import RandomForestRegressor
# 用其他特征预测 col_a 缺失值
mask = df['col_a'].isna()
rf = RandomForestRegressor(n_estimators=200).fit(
    df.loc[~mask].drop(columns='col_a'),
    df.loc[~mask, 'col_a']
)
df.loc[mask, 'col_a'] = rf.predict(df.loc[mask].drop(columns='col_a'))
```

### GBDT 的特殊处理

XGBoost / LightGBM / CatBoost **原生支持缺失值**——缺失自动作为"另一个分支"。这意味着 GBDT 通常不需要预处理缺失。但仍建议把"是否缺失"作为额外特征，可能携带信息。

## 二、异常值处理

### 检测方法

- **统计法**：$|x - \mu| > 3\sigma$、IQR $1.5 \times (Q_3 - Q_1)$ 之外。
- **距离法**：k-NN 距离、LOF（Local Outlier Factor）。
- **模型法**：Isolation Forest、One-Class SVM。
- **重建误差**：自编码器重建误差大的样本视为异常。

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, random_state=42)
mask = iso.fit_predict(X) == 1   # True 表示非离群点
X_clean = X[mask]
```

### 处理策略

- 删除（慎用）。
- Winsorize（截尾到 1%/99% 分位数）。
- 对数变换：$\log(1 + x)$ 压扁长尾。
- Box-Cox / Yeo-Johnson 变换。

```python
from scipy.stats.mstats import winsorize
df['price'] = winsorize(df['price'], limits=[0.01, 0.01])

# 对数变换
df['price_log'] = np.log1p(df['price'])
```

## 三、类别特征编码

### 低基数（< 10 类）

**One-Hot Encoding**：每个类别一个 0/1 列。

```python
df = pd.get_dummies(df, columns=['city'], drop_first=True)
```

**问题**：高基数会让列数爆炸。

### 高基数（成百上千类）

**Target Encoding**（也称 Mean Encoding）：

```math
\text{enc}(c) = \frac{\sum_{i: x_i = c} y_i}{\text{count}(c)}
```

用每个类别的目标均值代替类别 ID。**关键：要做 CV/折叠编码**，否则泄漏训练标签导致过拟合。

```python
# sklearn-contrib 的 category_encoders 库
from category_encoders import TargetEncoder
te = TargetEncoder(smoothing=10).fit(X_train, y_train)
X_train_enc = te.transform(X_train)
```

**Count Encoding** / **Frequency Encoding**：

```math
\text{enc}(c) = \text{count}(\{i: x_i = c\})
```

**Hash Encoding**（Hashing Trick）：把高维稀疏类别哈希到固定维度（如 $2^{10}$），适合超大规模类别。

**CatBoost Encoding**：类似 target encoding，但用有序 TS（Ordered TS）做无泄漏编码，CatBoost 内部实现。

### 有序类别

如"小学 < 中学 < 大学"，直接映射成整数 {0, 1, 2} 即可。如果距离不均匀，可用**有序回归**做编码。

### Embedding

类别作为 NN 输入时，用 embedding 层学低维稠密向量，比 one-hot 更高效：

```python
emb = nn.Embedding(num_categories, embedding_dim)
```

## 四、数值特征缩放

| 方法 | 公式 | 适用 |
|---|---|---|
| **StandardScaler** | $x' = (x - \mu) / \sigma$ | 线性模型、SVM、神经网络、KNN |
| **MinMaxScaler** | $x' = (x - x_{\min}) / (x_{\max} - x_{\min})$ | 有明确边界（图像像素） |
| **RobustScaler** | $x' = (x - Q_2) / (Q_3 - Q_1)$ | 含离群值 |
| **MaxAbsScaler** | $x' = x / \max(\|x\|)$ | 稀疏数据 |
| **QuantileTransformer** | 把值映射到均匀/正态分布 | 任意分布，对离群值鲁棒 |

```python
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
scaler = StandardScaler().fit(X_train)   # ⚠️ 只在训练集 fit
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)        # 用同一组参数
```

**关键**：缩放器只在训练集上 fit，测试集 transform 同一参数。这是数据泄漏的常见源头。

**树模型不需要缩放**：决策树、GBDT 对特征尺度不敏感，所以无需缩放——这是树模型"省心"的地方。

## 五、特征构造

**1. 数值变换**：

```math
\log x,\;\; \sqrt{x},\;\; x^2,\;\; x_1 \cdot x_2,\;\; x_1 / x_2
```

**2. 时间特征**：

```python
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['days_since'] = (df['datetime'] - df['datetime'].min()).dt.days
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)   # 周期性编码
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

**3. 聚合特征**（与 group by 结合）：

```python
# 用户的历史行为统计
user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std', 'min', 'max', 'count'])
df = df.merge(user_stats, on='user_id')
```

**4. 交叉特征**：

```python
df['city_product'] = df['city'].astype(str) + '_' + df['product'].astype(str)
# 然后再做 target encoding
```

**5. NLP / 文本特征**：

- 词袋、TF-IDF。
- N-gram。
- Embedding（预训练 BERT 等）。
- 长度、标点、特殊词计数。

**6. 图像特征**：

- 颜色直方图。
- HOG、SIFT（传统）。
- CNN 特征（预训练 ResNet/ViT）。

## 六、特征选择

### 过滤法（Filter）

按统计指标打分，与模型无关：

```math
\text{相关系数} \;\; | \;\; \chi^2 \;\; | \;\; \text{互信息} \;\; | \;\; \text{方差}
```

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(score_func=mutual_info_classif, k=20).fit(X_train, y_train)
X_train_sel = selector.transform(X_train)
```

### 包装法（Wrapper）

用模型反复训练选子集：

- 前向选择（逐步加特征）。
- 后向消除（逐步减特征）。
- RFE（Recursive Feature Elimination）。

```python
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=20).fit(X_train, y_train)
```

### 嵌入法（Embedded）

模型训练过程中自动选特征：

- L1 正则（Lasso）：让不重要特征权重变 0。
- 树模型特征重要性：基于不纯度下降或分裂次数。

```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5).fit(X_train, y_train)
important = np.abs(lasso.coef_) > 0
```

**SHAP 特征重要性**：比 sklearn 自身更准，能捕捉非线性贡献。

## 七、类别不平衡处理

| 方法 | 说明 |
|---|---|
| **类别权重** | `class_weight='balanced'`、XGBoost 的 `scale_pos_weight` |
| **下采样** | 多数类随机采样少一些 |
| **过采样** | 少数类复制（SMOTE、ADASYN） |
| **阈值调整** | 不用 0.5，按 PR 曲线最优选 |
| **Focal Loss** | 困难样本权重加大 |

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
```

## 八、Pipeline 化：可复现与防泄漏

把所有预处理 + 模型封装成一个 Pipeline，CV 时自动"每折独立 fit 预处理"：

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numeric_features = ['age', 'income']
categorical_features = ['city', 'gender']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
])

pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipe, X, y, cv=5)
```

**这是防数据泄漏的最佳工具**——ColumnTransformer + Pipeline 让"训练集 fit、测试集 transform"自动正确执行。

## 九、特征工程的常见错误

- **数据泄漏**：用全量数据 fit 缩放器、target encoding 用到验证集标签。
- **训练-测试分布不一致**：用未来信息做特征（如次日股价做今日特征）。
- **多重共线性**：高度相关特征同时入模型，让解释变难、回归系数不稳定。
- **盲目 one-hot**：类别过多导致维度爆炸又稀疏。
- **不验证就上线**：新数据的特征分布可能漂移（concept drift）。

## 十、特征工程与深度学习的边界

- **表格数据**：GBDT + 精心设计的特征仍是 SOTA。
- **图像 / 文本 / 语音**：深度学习自动学特征，特征工程重要性下降——但**基础预处理**（归一化、tokenization、数据增强）仍关键。
- **混合场景**：表格 + 文本（推荐系统）通常先用 NN 处理文本得 embedding，再与表格特征合并，过 GBDT 或简单 MLP。

## 小结

特征工程的本质是"**把领域知识编码为模型能用的信号**"：缺失值与异常值处理解决数据质量问题，类别编码与数值缩放统一输入空间，特征构造与选择提高信号密度，Pipeline 化保证可复现。掌握这套工艺后，你会发现在表格数据上，GDBT + 优秀特征工程往往比复杂深度模型更靠谱。深度学习时代，特征工程虽不再是全部，但仍是 ML 工程师的核心竞争力。