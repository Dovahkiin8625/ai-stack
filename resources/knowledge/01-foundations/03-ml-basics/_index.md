# ML Basics

> 分类：**基础理论** → **ML Basics**
> 路径：`resources/knowledge/01-foundations/03-ml-basics`

本目录系统梳理机器学习的核心范式：从监督学习、无监督学习，到偏差-方差权衡与正则化，再到模型评估、集成方法、特征工程，最后到强化学习。这些内容覆盖了 ML 工程师日常工作所需的全部基础知识。

## 文章目录

### 核心范式

- [监督学习：从线性回归到 SVM](./supervised-learning.md) — 模型 + 损失 + 优化三段式，串起线性回归、逻辑回归、SVM、KNN、决策树、朴素贝叶斯。
- [无监督学习：从聚类到流形学习](./unsupervised-learning.md) — K-Means、层次聚类、DBSCAN、PCA、t-SNE/UMAP、GMM、自编码器。
- [强化学习基础：从 MDP 到 PPO](./reinforcement-learning-fundamentals.md) — MDP、价值函数、Q-Learning、Policy Gradient、Actor-Critic、PPO、RLHF/DPO。

### 模型工程

- [偏差-方差与正则化](./bias-variance-and-regularization.md) — 偏差-方差分解、L1/L2、Dropout、数据增强、早停、Label Smoothing。
- [模型评估与选择](./model-evaluation-and-selection.md) — 分类/回归/排序指标、校准、交叉验证、Bootstrap、超参搜索、数据泄漏防御、A/B 测试。
- [集成方法：从随机森林到 XGBoost](./ensemble-methods.md) — Bagging、Boosting、随机森林、GBDT、XGBoost、LightGBM、CatBoost、Stacking。
- [特征工程与预处理](./feature-engineering-and-preprocessing.md) — 缺失值、异常值、类别编码、数值缩放、特征构造、特征选择、Pipeline 化。

## 收录范围

- 教材与讲义（Markdown / PDF）
- 论文（PDF）
- 讲稿与笔记（Markdown / Word / PPTX）

## 命名约定

- 每个子主题一个文件夹，文件夹命名用 kebab-case。
- 每个文件夹下放一个 `_index.md` 作为目录索引（阶段 2 由索引生成器读取）。
- 文件名建议：`YYYY-MM-DD-<title>.md` 或原文件名。