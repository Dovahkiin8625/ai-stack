# 图论与离散数学

## 一、图的基本定义

一个图 $G = (V, E)$ 由顶点集 $V$ 和边集 $E \subseteq V \times V$ 组成。按边是否有向、是否多重，可以分为：

- **有向图 / 无向图**：边是否有方向。
- **加权图**：每条边附一个权重（距离、相似度、流量等）。
- **多重图 / 自环**：是否允许多重边或 $v \to v$。
- **二部图**：顶点可以分成两组，组内无边。
- **树**：连通无环的无向图，有 $n$ 个顶点恰好 $n-1$ 条边。

邻接矩阵 $A \in \{0,1\}^{n \times n}$（无向图对称）的元素 $A_{uv} = 1$ 表示存在边 $(u, v)$。加权图里 $A_{uv}$ 存的是权重。**度数矩阵** $D$ 是对角矩阵，$D_{uu} = \sum_v A_{uv}$，表示每个顶点的度。

```python
import numpy as np
A = np.array([
    [0, 1, 1, 0],   # 0 连 1、2
    [1, 0, 1, 1],   # 1 连 0、2、3
    [1, 1, 0, 0],   # 2 连 0、1
    [0, 1, 0, 0],   # 3 连 1
], dtype=float)
D = np.diag(A.sum(axis=1))
print('D =\n', D)
# D = diag([2, 3, 2, 1])
```

## 二、连通性与基本图算法

**深度优先搜索（DFS）** 与**广度优先搜索（BFS）** 是遍历图的两种基本方式，分别用栈和队列实现：

- BFS 可求**无权图**的最短路径、时间复杂度 $O(|V| + |E|)$。
- DFS 可用于拓扑排序（检测环）、连通分量、强连通分量（Kosaraju / Tarjan）。

**Dijkstra 算法**求**非负权图**的单源最短路径，时间复杂度 $O((|V| + |E|) \log |V|)$（用优先队列）。**Bellman-Ford** 可以处理负权边但不能有负权环，时间 $O(|V| \cdot |E|)$。**Floyd-Warshall** 求全对最短路，时间 $O(|V|^3)$，但实现简单，适合小图。

**最小生成树（MST）** 在加权无向连通图中找一棵边权和最小的生成树。Kruskal（按边权排序 + 并查集）和 Prim（类似 Dijkstra）是经典算法，时间都是 $O(|E| \log |V|)$。

```python
import heapq
def dijkstra(A, src, n):
    dist = [float('inf')] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in enumerate(A[u]):
            if w and d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

这些算法是图处理的"积木"——知识图谱查询、地图导航、网络路由都建立在它们之上。

## 三、谱图理论：图的拉普拉斯矩阵

**图拉普拉斯** $L = D - A$。它的特征分解揭示了图的结构：

- $L \mathbf{1} = 0$（常向量是零特征向量，对应"整张图一个分量"）。
- $L$ 半正定，特征值 $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n$。
- 第二个特征值 $\lambda_2$ 称为**代数连通度**，越大图越连通。

**归一化拉普拉斯** $\hat{L} = I - D^{-1/2} A D^{-1/2}$ 在谱卷积中更常用。**谱图卷积**定义为信号 $x \in \mathbb{R}^n$ 乘以 $L$（或 $\hat{L}$）的特征向量矩阵：相当于把图信号投影到"图傅里叶基"上。

这个视角直接催生了 **ChebNet** 和 **GCN**：用 $L$ 的多项式截断近似图卷积核，避免显式做特征分解。

```math
g_\theta \star x = g_\theta(L) x = U g_\theta(\Lambda) U^\top x
```

其中 $U, \Lambda$ 是 $L$ 的特征向量和特征值。GCN 取一阶截断 $g_\theta \approx \theta_0 I + \theta_1 L$，简化后得到著名的传播规则：

```math
H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)}),\quad \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
```

其中 $\tilde{A} = A + I$ 加自环，$\tilde{D}$ 是其度矩阵。

## 四、GNN 的消息传递框架

现代 GNN（GraphSAGE、GAT、消息传递神经网络 MPNN）都可以用**消息传递**框架描述：

```math
\mathbf{h}_v^{(l+1)} = \mathrm{UPDATE}^{(l)}\!\left( \mathbf{h}_v^{(l)},\; \mathrm{AGG}^{(l)}\!\left( \{\mathbf{h}_u^{(l)} : u \in \mathcal{N}(v)\} \right) \right)
```

两步：

1. **聚合（Aggregate）**：把邻居的特征集合 $\{\mathbf{h}_u\}$ 压缩成一个向量（mean/sum/max，或注意力加权）。
2. **更新（Update）**：把当前节点表示与聚合结果合并（拼接 + MLP，或 GRU/Transformer 风格）。

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GCNLayer(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr='add')  # sum 聚合
        self.lin = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x, edge_index):
        # edge_index: (2, E) 形如 [[src...], [dst...]]
        edge_weight = torch.ones(edge_index.size(1), device=x.device)
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j 是源节点特征
        return edge_weight.view(-1, 1) * x_j
```

消息传递的**过平滑（over-smoothing）** 问题：随着层数加深，所有节点的表示趋向相似（收敛到常数向量）。解决方案包括：

- 残差连接（类似 ResNet）。
- PairNorm、DropEdge 等正则化。
- 注意力机制（GAT）放大重要邻居。

## 五、图上的重要任务

- **节点分类**：预测每个节点的标签（如引文网络中论文的领域）。
- **链接预测**：预测两个节点之间是否存在边（推荐系统、知识图谱补全）。
- **图分类**：预测整张图的标签（分子性质预测）。
- **图生成**：生成新图（药物设计、社交网络仿真）。

对应评估指标：

- 节点分类：准确率、F1。
- 链接预测：MRR、Hit@K、AUC。
- 图分类：准确率、ROC-AUC。

## 六、组合数学与离散概率的简短插曲

离散数学里与 ML 强相关的几个点：

- **排列组合**：评估组合搜索空间大小（如特征子集选择有 $2^d$ 种）。
- **鸽巢原理**：解释为什么高维空间的最近邻检索常需要近似算法（局部敏感哈希 LSH）。
- **二项 / 多项分布**：Dropout 层的掩码、多臂老虎机中的奖励分布。
- **图着色**：编译器寄存器分配、MapReduce 调度。

## 小结

图论从"顶点和边的数学"升级为 ML 中描述关系数据的通用语言：邻接矩阵 + 拉普拉斯提供了与线性代数的桥梁，DFS/BFS/最短路/MST 提供了基础算法工具箱，消息传递框架统一了现代 GNN。掌握这些，你既能看懂谱 GCN 的论文推导，也能在 PyG / DGL 里写出生产可用的图模型。