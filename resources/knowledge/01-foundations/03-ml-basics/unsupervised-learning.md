# 无监督学习：从聚类到流形学习

## 一、无监督学习的四大目标

- **聚类**：把样本分到 $K$ 组，使组内相似、组间不同。
- **降维**：把高维 $x \in \mathbb{R}^d$ 映射到低维 $z \in \mathbb{R}^k$，保留关键信息。
- **密度估计**：学习 $P(x)$，可用于异常检测与生成。
- **表示学习**：学习 $f: x \to z$，使 $z$ 对下游任务更有用。

深度学习时代，**对比学习、自监督预训练、生成模型** 都属于无监督学习的现代延伸。

## 二、K-Means：最简单的聚类

**目标**：把 $N$ 个样本分到 $K$ 个簇，最小化簇内平方和：

```math
\mathcal{L} = \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2
```

**算法**（EM 风格的 Lloyd 算法）：

1. 初始化 $K$ 个质心（随机或 K-Means++）。
2. **E 步**：把每个样本分配到最近的质心。
3. **M 步**：用簇内均值更新质心。
4. 重复直到收敛。

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=5, n_init=10, init='k-means++').fit(X)
labels = km.labels_
centers = km.cluster_centers_
```

**K-Means++** 初始化让首批质心尽量分散，显著减少陷入局部最优的概率。复杂度 $O(N K d \cdot T)$，$T$ 是迭代次数。

**确定 $K$**：肘部法则（Elbow）、轮廓系数（Silhouette）、Gap Statistic。

**缺点**：假设簇为凸球形、对离群点和尺度敏感；对非球形簇失败。

## 三、层次聚类与 DBSCAN：弥补 K-Means 的局限

**层次聚类（Agglomerative）**：自底向上，每次合并最近的两簇。

- 簇间距离：单链（min）、全链（max）、平均、Ward（合并后方差最小）。
- 用 **树状图（dendrogram）** 可视化，可在任意高度切分得到不同 $K$。
- 复杂度 $O(N^3)$，大规模需配合 SLINK / BIRCH。

**DBSCAN**（Density-Based Spatial Clustering of Applications with Noise）：

- 把"密度可达"的样本归为同一簇。
- 参数：$\epsilon$（邻域半径）、`min_samples`（形成核心点的最少邻居数）。
- 不需要预设 $K$，能识别**任意形状**的簇、自动识别**离群点**。

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(X)
labels = db.labels_   # -1 表示离群点
```

**HDBSCAN** 是 DBSCAN 的层次版，自动选 $\epsilon$，对多密度簇更鲁棒。

**谱聚类**（Spectral Clustering）：

1. 用样本建相似度图（KNN 或 $\epsilon$-graph）。
2. 算图拉普拉斯 $L = D - W$ 的特征向量。
3. 在低维特征向量上做 K-Means。

能在图结构上找到"切割"——适合社交网络、图像分割。

## 四、PCA：线性降维的"老大哥"

主成分分析寻找正交方向，使数据在这些方向上的方差最大。等价于对协方差矩阵 $\Sigma = \frac{1}{N} X^\top X$ 做特征分解：

```math
\Sigma = U \Lambda U^\top
```

取前 $k$ 个最大特征值对应的特征向量组成 $U_k$，则 $z = U_k^\top (x - \bar{x})$。

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50).fit(X)
Z = pca.transform(X)             # (N, 50)
print(pca.explained_variance_ratio_.sum())  # 保留方差比例
```

**PCA 的局限**：只能捕捉**线性**结构；对非线性流形（瑞士卷、人脸）效果差；对异常值敏感（因为是 L2 重建）。

**Kernel PCA**：把 PCA 升到核空间，$\Sigma$ 换成核矩阵 $K$ 的特征分解，类似 SVM 的核技巧。

## 五、t-SNE / UMAP：高维可视化的双子星

**t-SNE**（t-Distributed Stochastic Neighbor Embedding）：

1. 在高维空间，用高斯核定义样本 $i, j$ 的相似度 $p_{ij}$。
2. 在低维空间，用 $t$ 分布（重尾）定义 $q_{ij}$。
3. 最小化 $p, q$ 的 KL 散度（梯度下降）。

**UMAP**（Uniform Manifold Approximation and Projection）：

- 基于黎曼几何与代数拓扑，假设数据均匀分布在流形上。
- 用吸引/排斥力的弹簧系统优化。
- 比 t-SNE 更快、保留全局结构更好。

```python
from sklearn.manifold import TSNE
import umap
Z_tsne = TSNE(n_components=2, perplexity=30, init='pca').fit_transform(X)
Z_umap = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
```

**重要警告**：

- t-SNE/UMAP 的簇间距离**没有意义**——可视化只反映局部结构。
- 不要用 t-SNE 解释全局几何！
- 不同 `perplexity`（t-SNE）或 `n_neighbors`（UMAP）下形状差异很大。

## 六、概率密度估计

**高斯混合模型（GMM）**：用 $K$ 个高斯分布加权求和拟合数据：

```math
P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)
$$

训练用 EM 算法：E 步算样本对各分量的责任度 $\gamma_{ik}$，M 步更新参数。

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5, covariance_type='full').fit(X)
labels = gmm.predict(X)
log_density = gmm.score_samples(X)   # 用于异常检测
```

**核密度估计（KDE）**：$P(x) = \frac{1}{N h} \sum_i K\!\left(\frac{x - x_i}{h}\right)$，灵活但带宽 $h$ 难调。

**流模型**（Normalizing Flows）通过一系列可逆变换把简单分布变成复杂分布，是现代生成模型三大流派之一（与 VAE、扩散模型并列）。

## 七、自编码器：神经网络的"非线性 PCA"

**自编码器**通过编码器 $f$ 把 $x$ 映射到 $z$，解码器 $g$ 重建 $\hat{x} = g(f(x))$，最小化重建损失：

```math
L = \|x - g(f(x))\|^2
```

**变体**：

- **稀疏 AE**：在 $z$ 上加稀疏惩罚。
- **去噪 AE（DAE）**：输入加噪、目标为干净 $x$。
- **变分 AE（VAE）**：在 $z$ 上加 KL 正则，使其接近先验（高斯）。
- **对比学习（SimCLR/MoCo）**：用 InfoNCE 拉近正样本、推远负样本。

这些是**现代表示学习**的主流——它们把"无监督降维"从线性 PCA 推到了强大得多的非线性版本。

```python
import torch.nn as nn
class AE(nn.Module):
    def __init__(self, d_in, d_h, d_z):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_z))
        self.dec = nn.Sequential(nn.Linear(d_z, d_h), nn.ReLU(), nn.Linear(d_h, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))
```

## 八、关联规则与频繁项挖掘

虽然不算"聚类"主流，但**Apriori、FP-Growth** 等算法在零售推荐中有悠久历史：

- 项集 $I = \{i_1, \dots, i_k\}$ 的支持度 $s(I) = \frac{\text{包含 }I\text{ 的事务数}}{\text{总事务数}}$。
- 置信度 $c(I \to j) = \frac{s(I \cup \{j\})}{s(I)}$。
- 提升度 $\text{lift} = \frac{c(I \to j)}{s(\{j\})}$，衡量关联性而非偶然性。

现在这类问题大多被 **Embedding + 向量检索** 替代——把每个 item 学成向量，最近邻即"关联项"。

## 九、聚类 / 降维的评估

聚类评估（有真标签时）：

- **ARI（Adjusted Rand Index）**：忽略标签置换的相似度。
- **NMI（Normalized Mutual Information）**：基于互信息。
- **V-measure**：同质性 + 完整性的调和平均。

聚类评估（无真标签时）：

- **轮廓系数**：$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$，$a$ 是平均簇内距离、$b$ 是最近邻簇距离。
- **Calinski-Harabasz**：簇间方差 / 簇内方差。
- **Davies-Bouldin**：簇间距离 / 簇内散度。

降维评估：

- **解释方差比**（PCA）：$\sum_{i=1}^k \lambda_i / \sum_{i=1}^d \lambda_i$。
- **重建误差**（AE / 自编码器）。
- **下游任务表现**（推荐）。

## 十、如何选择？

| 目标 | 数据规模 | 推荐 |
|---|---|---|
| 聚类、簇数已知 | 小–中 | K-Means（k-means++） |
| 聚类、簇数未知、含离群点 | 中 | DBSCAN / HDBSCAN |
| 降维到 2D/3D 可视化 | 中 | UMAP（首选）/ t-SNE |
| 线性降维 + 特征压缩 | 中–大 | PCA / 随机 PCA / IncrementalPCA |
| 非线性流形 + 表示学习 | 大 | 自编码器 / SimCLR |
| 概率密度估计 + 异常检测 | 中 | GMM / KDE / 隔离森林 |

## 小结

无监督学习的核心是"**从数据本身发现结构**"：聚类发现分组，降维与表示学习发现紧凑特征，密度估计发现分布。现代深度学习让无监督从"线性 PCA、K-Means"飞跃到"对比学习、扩散模型"，但底层数学（KL 散度、互信息、流形假设、最小重建误差）一脉相承。掌握这些基础，再去读 SimCLR、DDPM、VAE 论文，你会发现它们本质上是同一类问题的更强解法。