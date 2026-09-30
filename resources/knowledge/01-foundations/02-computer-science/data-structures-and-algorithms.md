# 数据结构与算法

## 一、复杂度记号

我们用**大 O** 描述算法随输入规模 $n$ 的渐近上界：

- $O(1)$ 常数（哈希表查找）
- $O(\log n)$ 对数（二分搜索、堆操作）
- $O(n)$ 线性（遍历）
- $O(n \log n)$ 准线性（快排、归并）
- $O(n^2)$ 平方（朴素矩阵乘）
- $O(2^n)$ 指数（暴力枚举子集）

大 O 隐藏了常数因子，但**常数在工程里很重要**：哈希表虽然 $O(1)$ 但常数很大；Numpy 的向量化循环虽然仍是 $O(n)$ 但常数小到几乎免费。

## 二、数组与链表

**数组**连续内存，$O(1)$ 随机访问，$O(n)$ 中间插入/删除。Python `list` 实际上是"动态数组"——扩缩容时会重新分配内存并拷贝。

**链表**非连续，靠指针串联，$O(n)$ 访问、$O(1)$ 头插/删。在 ML 工程中，链表的典型用途是：

- **LRU 缓存**：双向链表 + 哈希表，$O(1)$ get/put。
- **KV-cache 中的 token 序列**：每个解码步骤追加新节点。
- **邻接表**：稀疏图的 $O(|V| + |E|)$ 表示。

```python
class Node:
    __slots__ = ('key', 'val', 'prev', 'next')
    def __init__(self, k, v):
        self.key, self.val = k, v
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.cap, self.cache = capacity, {}
        self.head = Node(0, 0)  # dummy
        self.tail = Node(0, 0)
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, n):
        n.prev.next = n.next
        n.next.prev = n.prev

    def _add(self, n):
        n.prev, n.next = self.head, self.head.next
        self.head.next.prev = n
        self.head.next = n

    def get(self, k):
        if k in self.cache:
            self._remove(self.cache[k]); self._add(self.cache[k])
            return self.cache[k].val
        return -1

    def put(self, k, v):
        if k in self.cache:
            self._remove(self.cache[k])
        n = Node(k, v); self.cache[k] = n; self._add(n)
        if len(self.cache) > self.cap:
            lru = self.tail.prev
            self._remove(lru); del self.cache[lru.key]
```

## 三、哈希表

哈希表通过哈希函数 $h(k)$ 把 key 映射到桶数组下标。期望 $O(1)$ 增删查，最坏 $O(n)$（全部 hash 到同一桶）。

工程要点：

- **负载因子** 超过阈值（如 0.75）时扩容，避免长链。
- **哈希冲突**：链地址法（Java `HashMap`）或开放寻址（Python `dict`、C++ `unordered_map`）。
- **哈希函数**：MurmurHash、SHA 系列。**不要**用密码学哈希做通用哈希表 key（慢）。

ML 场景：

- **Embedding 表**：key 是 token id 或 user/item id，本质是大哈希表。
- **Python `set/dict`**：去重、统计、计数器。
- **布隆过滤器**：用多个哈希判定元素是否在集合中（允许假阳性，常用于爬虫去重、缓存穿透保护）。

```python
# 手写一个简单哈希表（拉链法）
class HashMap:
    def __init__(self, size=8):
        self.size = size
        self.buckets = [[] for _ in range(size)]

    def _h(self, k):
        return hash(k) % self.size

    def put(self, k, v):
        b = self.buckets[self._h(k)]
        for i, (kk, vv) in enumerate(b):
            if kk == k:
                b[i] = (k, v); return
        b.append((k, v))

    def get(self, k):
        for kk, vv in self.buckets[self._h(k)]:
            if kk == k: return vv
        return None
```

## 四、树与堆

**二叉搜索树（BST）**：每个节点左子树 < 节点 < 右子树，平均 $O(\log n)$，最坏 $O(n)$（退化成链表）。

**平衡 BST**：AVL（严格平衡）、红黑树（近似平衡，Linux 内核 `epoll`、C++ `std::map`）。Java `TreeMap`、Python `bisect` 底层都用平衡 BST。

**堆（优先队列）**：父节点 ≤ 子节点的完全二叉树，$O(\log n)$ 插入/弹出最小元素。

ML 工程实例：

- **Top-K 检索**：维护大小为 $K$ 的最小堆，遍历 $N$ 个元素，总复杂度 $O(N \log K)$。
- **Dijkstra 最短路**：用堆挑最小距离顶点。
- **Huffman 编码**：用堆构造最优前缀码。

```python
import heapq
scores = [0.8, 0.3, 0.9, 0.4, 0.7, 0.1]
top3 = heapq.nlargest(3, scores)   # [0.9, 0.8, 0.7]  —— O(N log 3)
```

## 五、排序

**比较排序**下界是 $O(n \log n)$。常见算法：

| 算法 | 时间 | 空间 | 稳定性 | 特点 |
|---|---|---|---|---|
| 冒泡 | $O(n^2)$ | $O(1)$ | 是 | 教学用 |
| 插入 | $O(n^2)$ | $O(1)$ | 是 | 接近有序时很快 |
| 快排 | $O(n \log n)$ | $O(\log n)$ | 否 | 工业首选，常数小 |
| 归并 | $O(n \log n)$ | $O(n)$ | 是 | 适合链表、外排 |
| 堆排 | $O(n \log n)$ | $O(1)$ | 否 | 不稳定但省内存 |

**非比较排序**（计数、桶、基数）能突破 $O(n \log n)$，但需要输入满足特殊条件。例如**Top-K 频率问题**可以用计数 + 桶排序做到 $O(n)$。

ML 场景：

- **排序训练样本**：bucketing / length-grouping 减少 padding 浪费。
- **多路归并**：从多个 shuffle buffer 合并预排序好的样本流（PyTorch `DataLoader` 用 `sort_key`）。

## 六、字符串与 Trie

**Trie**（前缀树）：每个节点是一个字符到子节点的映射，$O(L)$ 查找（$L$ 是字符串长度），常用于：

- 自动补全
- IP 路由最长前缀匹配
- 词法分析

**后缀数组 / 后缀树**：把字符串的所有后缀排序，$O(n \log n)$ 构造，$O(1)$ LCP（最长公共前缀）查询；用于 DNA 序列比对、文档检索。

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.is_end = True

    def starts_with(self, prefix):
        node = self
        for ch in prefix:
            if ch not in node.children: return []
            node = node.children[ch]
        return self._collect(node, prefix)

    def _collect(self, node, prefix):
        out = [prefix] if node.is_end else []
        for ch, sub in node.children.items():
            out += self._collect(sub, prefix + ch)
        return out
```

## 七、图算法回顾

见姊妹篇"图论与离散数学"。要点回顾：

- BFS/DFS 用于遍历与拓扑排序。
- Dijkstra / Bellman-Ford 用于最短路。
- Kruskal / Prim 用于 MST。
- PageRank 是随机游走 + 幂迭代，本质是图上的特征向量计算——GNN 的简化版思路。

## 八、ML 工程中的算法选择速查

| 场景 | 推荐 |
|---|---|
| 大量增删查 | 哈希表 |
| 需要排序 / Top-K | 堆 |
| 区间最值 | 段树 / 稀疏表 |
| 二维最近邻 | KD-Tree / Ball-Tree（小规模）、HNSW/Annoy（大规模） |
| 字符串前缀 | Trie |
| 大图最短路 | Dijkstra（无负权）/ SPFA（允许负权但高效） |
| Top-K 检索 | 堆或基数选择（线性） |

```python
# KD-Tree 最近邻（小规模向量）
from sklearn.neighbors import KDTree
import numpy as np
X = np.random.randn(1000, 64)
tree = KDTree(X)
dist, idx = tree.query(X[:5], k=3)   # 每个点找 3 个最近邻
```

## 小结

数据结构与算法的核心是"**用空间换时间**"——哈希表用额外数组换 $O(1)$ 查询，堆用树形结构换 $O(\log n)$ 优先级操作，Trie 用共享前缀换 $O(L)$ 字符串查找。ML 工程中很少要"白板写红黑树"，但理解这些结构的**复杂度特征与适用场景**，能让你在写训练循环、推理服务、RAG 检索时本能地选择合适工具，避免在 GPU 训练时用 list 做 O(n²) 查找这种低级错误。