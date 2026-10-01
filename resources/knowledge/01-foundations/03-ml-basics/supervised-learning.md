# 监督学习：从线性回归到 SVM

## 一、监督学习的三段式

几乎所有监督学习算法都可以写成三段式：

$$
\text{模型} \; f_\theta(x) \quad + \quad \text{损失} \; L(y, f_\theta(x)) \quad \xrightarrow{\min_\theta} \quad \text{优化器} \; (\text{SGD/Adam/闭式解})
$$

不同算法的差异只在于：

- **模型族** $f_\theta$ 是什么（线性、树、神经网络）。
- **损失函数** $L$ 是什么（MSE、Cross-Entropy、Hinge、0-1 损失）。
- **优化方法**：闭式解、凸优化、贪心分裂、随机梯度。

记住这三段式，你就具备了快速理解任何新算法的能力。

## 二、线性回归：最直白的"拟合"

**模型**：$f_\theta(x) = w^\top x + b$

**损失**：均方误差（MSE）

$$
L(\theta) = \frac{1}{N}\sum_{i=1}^{N} (w^\top x_i + b - y_i)^2
$$

**闭式解**：

$$
\theta^* = (X^\top X)^{-1} X^\top y
$$

实践中很少直接求逆（$O(d^3)$），而是用 QR 分解或 SVD（数值更稳）。

```python
from sklearn.linear_model import LinearRegression
import numpy as np
X = np.random.randn(200, 3)
y = X @ np.array([1.5, -2.0, 0.7]) + 0.1*np.random.randn(200)
model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)
```

**Lasso / Ridge / ElasticNet**：在线性回归上加正则（详见偏差-方差正则化那篇）。

## 三、逻辑回归：分类的起点

**模型**：$f_\theta(x) = \sigma(w^\top x + b)$，其中 $\sigma$ 是 sigmoid。

**损失**：交叉熵

$$
L(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\!\left[ y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]
$$

逻辑回归虽然名字带"回归"，其实是分类——把线性输出过 sigmoid 后看作"正类概率"。多分类版本是 softmax 回归，输出 $K$ 个互斥概率。

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000, multi_class='multinomial').fit(X_train, y_train)
print(clf.predict_proba(X_test[:3]))  # 输出每个类别的概率
```

**训练技巧**：

- 学习率调度：默认即可。
- 正则：L2（默认 `C=1.0`）。
- 类别不平衡：`class_weight='balanced'` 或过采样 SMOTE。

## 四、SVM：最大间隔的几何直觉

支持向量机寻找一个**超平面**，使两类样本到超平面的最小距离（间隔）最大。线性可分时：

$$
\min_{w, b} \; \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i (w^\top x_i + b) \ge 1
$$

KKT 条件指出，多数样本 $\alpha_i = 0$（非支持向量），只有"边缘"样本 $\alpha_i > 0$ 影响决策。这正是 SVM 高效的根源。

**核技巧**：把数据映射到高维空间再做线性分类，避免显式升维：

$$
K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)
$$

常用核：

- **线性核**：$K(x, x') = x^\top x'$
- **RBF 核**：$K(x, x') = \exp(-\gamma \|x - x'\|^2)$
- **多项式核**：$K(x, x') = (x^\top x' + c)^d$

```python
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_train, y_train)
```

SVM 优点：在小样本高维数据（如文本分类）上效果好。缺点：大数据集（$N > 10^5$）训练慢，已被深度学习取代。

## 五、KNN：极简却不可忽视的"懒学习"

K-Nearest Neighbors 没有训练阶段，预测时计算与所有训练样本的距离、选最近的 $K$ 个、多数表决（分类）或平均（回归）。

**距离度量**：

- 欧氏距离（连续特征默认）。
- 曼哈顿距离。
- 余弦相似度（文本 / 高维稀疏）。
- Minkowski $L_p$ 距离。

**关键参数**：

- **$K$**：太小过拟合、太大欠拟合。常用交叉验证选。
- **距离加权**：近邻权重大、远邻权重小。

```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, weights='distance').fit(X_train, y_train)
```

KNN 是**非参数**模型——容量随训练集增长，无显式参数。它是理解"局部方法"的入门，也是图神经网络、RAG 检索的思想源头。

## 六、决策树：可解释的"分而治之"

决策树通过递归分裂特征空间建树，每个内部节点是一个特征 + 阈值的判断。预测时从根走到叶，叶子给出类别/回归值。

**分裂准则**：

- **分类**：基尼系数 $G = 1 - \sum_k p_k^2$、信息增益（决策树详见信息论那篇）。
- **回归**：MSE / MAE。

```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20).fit(X_train, y_train)
```

**剪枝**防止过拟合：

- 预剪枝：`max_depth`、`min_samples_leaf`、`min_samples_split`。
- 后剪枝：cost-complexity `ccp_alpha`。

决策树的优缺点：

- ✅ 易解释、能处理混合类型特征、不需要特征缩放。
- ❌ 单棵树泛化差、对训练数据敏感（一个样本变化可能改整棵树）。
- → 解决方案：**集成**（随机森林、GBDT）。

## 七、朴素贝叶斯：基于条件独立的概率分类

**模型**：$P(y \mid x) \propto P(y) \prod_j P(x_j \mid y)$，假设特征条件独立。

**变体**：

- **高斯 NB**：连续特征。
- **多项式 NB**：文本分类（词频）。
- **伯努利 NB**：二值特征。

朴素贝叶斯训练极快（一次扫描统计频次），在文本分类（尤其小语料）上仍是强 baseline。它是**生成模型**（学习 $P(x,y)$），与判别模型（直接学 $P(y|x)$）视角不同。

```python
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0).fit(X_train_tfidf, y_train)
```

## 八、如何选择"经典算法"？

| 数据规模 | 特征类型 | 推荐 |
|---|---|---|
| 小（<10k） | 表格 | SVM/RF/逻辑回归 |
| 中（10k–1M） | 表格 | 逻辑回归/XGBoost/LightGBM |
| 大（>1M） | 表格 | XGBoost/线性模型/SGD |
| 文本/图像 | 高维稀疏 | 朴素贝叶斯/线性 SVM → 深度学习 |
| 大/小 | 混合 | 随机森林（少调参） |

**经验法则**：表格数据默认从 XGBoost/LightGBM 起手；图像/文本/语音默认从深度学习起手；小数据、无 GPU、想快速出结果时用 SVM 或朴素贝叶斯。

## 九、从"经典"到"现代"的桥梁

- **线性回归 → 神经网络**：线性回归 + 激活函数堆叠 = MLP。
- **逻辑回归 → Softmax → 深度分类器**：神经网络分类头。
- **SVM 核 → 神经网络**：RBF 核 SVM 与小宽度 MLP 等价。
- **KNN → RAG / 检索增强**：检索 + 提示是 KNN 的现代版本。
- **决策树 → 随机森林 → GBDT → XGBoost**：集成树一路演化。

理解这条演化链，你会发现"现代深度学习"并不是凭空出现的——它正是这些经典算法的非线性、可微、可扩展版本。

## 小结

监督学习的核心是"模型 + 损失 + 优化"三段式：线性回归给出闭式解的范式，逻辑回归给出交叉熵与概率输出的范式，SVM 给出几何间隔与核方法的范式，KNN 给出非参数局部方法的范式，决策树给出可解释分治的范式。掌握这些经典算法的核心思想，再看神经网络、GBDT、Transformer 时你会发现它们只是把同样的三段式"换皮"成了非线性、可微、大容量的版本。