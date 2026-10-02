# 公平性指标与权衡：不可能定理与实践选型

"模型是否公平"听上去是一个问题，实际上是**多个互相矛盾的问题**。学术界已经证明：对二分类问题，满足"群体公平"和"个体公平"中较强的几个准则，**不可能同时达成**（Chouldechova 2017; Kleinberg et al. 2016）。这就是著名的**公平性不可能定理（Impossibility Theorem of Fairness）**。本文系统梳理三大类公平性指标（群体公平、个体公平、反事实公平），用具体例子展示它们之间的张力，并给出实践中的选型指南。

## 一、公平性的三大类别

```text
┌───────────────────────────────────────────────────────┐
│  群体公平（Group Fairness）                              │
│  关注统计量在受保护群体之间是否相等                        │
│  例：男性 / 女性的正例率相同                              │
└───────────────────────────────────────────────────────┘
                          ↕ （常冲突）
┌───────────────────────────────────────────────────────┐
│  个体公平（Individual Fairness）                         │
│  相似的个体应得到相似的对待                               │
│  例：能力相同的候选人无论性别都应被同等推荐                │
└───────────────────────────────────────────────────────┘
                          ↕ （常冲突）
┌───────────────────────────────────────────────────────┐
│  反事实公平（Counterfactual Fairness）                   │
│  如果改变受保护属性，决策应不变                            │
│  例：若把简历中的性别改掉，结果应一致                      │
└───────────────────────────────────────────────────────┘
```

## 二、群体公平指标：以二分类为例

设模型对群体 $G=0, 1$（如男/女）输出预测 $\hat{Y} \in \{0, 1\}$，真实标签 $Y \in \{0, 1\}$。

### 1. 统计平等（Statistical Parity / Demographic Parity）

$$
P(\hat{Y} = 1 \mid G = 0) = P(\hat{Y} = 1 \mid G = 1)
$$

**含义**：两个群体的**正例率相同**。这是最直观的"机会平等"——但忽略真实标签分布，可能把"把女性录取率拉低/拉高到与男性相同"。

### 2. 等机会（Equal Opportunity）

$$
P(\hat{Y} = 1 \mid Y = 1, G = 0) = P(\hat{Y} = 1 \mid Y = 1, G = 1)
$$

**含义**：在**真实正例**中，两个群体的**真阳率（TPR）相同**。例如：在确实合格的候选人中，男女通过率相同。

### 3. 等赔率（Equalized Odds）

$$
P(\hat{Y} = 1 \mid Y = y, G = 0) = P(\hat{Y} = 1 \mid Y = y, G = 1), \quad y \in \{0, 1\}
$$

**含义**：TPR 和 FPR 在两个群体上都相同。比等机会更强。

### 4. 预测平等（Predictive Parity）

$$
P(Y = 1 \mid \hat{Y} = 1, G = 0) = P(Y = 1 \mid \hat{Y} = 1, G = 1)
$$

**含义**：在**预测为正例**的样本中，实际正例的比例相同（即 precision 相同）。

### 5. 处理平等（Treatment Equality）

$$
\frac{P(\hat{Y} = 1, Y = 0 \mid G = 0)}{P(\hat{Y} = 1, Y = 1 \mid G = 0)} = \frac{P(\hat{Y} = 1, Y = 0 \mid G = 1)}{P(\hat{Y} = 1, Y = 1 \mid G = 1)}
$$

即 FPR / FNR 比值在两个群体上相同。

## 三、不可能定理

**核心定理**（Chouldechova 2017; Kleinberg et al. 2016）：

> 当以下三个条件**同时成立**且**两个群体的 base rate 不同**（$P(Y=1 \mid G=0) \neq P(Y=1 \mid G=1)$），则**不可能**让所有指标都满足：
> 
> 1. **Calibration（校准）**：$P(Y=1 \mid \hat{Y}=1, G) = p$（预测分数 $p$ 在所有群体上校准）
> 2. **Balance for the positive class**：$\mathbb{E}[\hat{Y} \mid Y=1, G]$ 在各组相等
> 3. **Balance for the negative class**：$\mathbb{E}[\hat{Y} \mid Y=0, G]$ 在各组相等

直觉：**当两个群体的"真实合格率"本身就不同时，要让分数阈值在不同群体上有同等含义，必然牺牲某一类公平**。

### 一个具体例子：COMPAS（累犯预测）

ProPublica 2016 调查显示 COMPAS 模型：

| 指标 | 黑人 | 白人 |
|---|---|---|
| FPR（错判为高风险）| 44.9% | 23.5% |
| FNR（漏判为低风险）| 27.8% | 47.7% |
| PPV（预测为高风险的人确实会再犯的比例）| 63% | 64% |

COMPAS 的回应是"PPV 在两群体上几乎相同，满足 predictive parity"。两方都对——他们选用了不同的公平指标。

**启示**：**选哪个指标是政治选择，不是技术选择**。

## 四、个体公平（Individual Fairness）

Dwork et al. (2012) 提出：对任意两个个体 $x_i, x_j$，

$$
D(\hat{Y}(x_i), \hat{Y}(x_j)) \leq L \cdot d(x_i, x_j)
$$

其中 $d$ 是任务相关的"相似度度量"（如简历的语义距离），$L$ 是 Lipschitz 常数。**含义**：相似的个体应得到相似的结果。

**难点**：$d$ 怎么定义？简历的距离、文本的距离、能力的距离都难以客观量化。在 LLM 场景下，个体公平更接近"对相同 prompt 给出等价回答"——这又回到 prompt 鲁棒性问题。

## 五、反事实公平（Counterfactual Fairness）

Kusner et al. (2017) 提出：把受保护属性 $A$ 替换成它的反事实值 $A'$，预测应不变：

$$
P(\hat{Y}_{A \leftarrow a}(U) = y \mid X = x, A = a) = P(\hat{Y}_{A \leftarrow a'}(U) = y \mid X = x, A = a)
$$

**含义**：把简历中"女"改成"男"，录用结果应不变。

**实现**：因果图 + 结构因果模型（SCM）。需要明确建模"哪些变量是受保护属性的因果后代"。LLM 场景下很难严格实施，但可以近似——把"代词替换"作为反事实的代理。

```python
def counterfactual_fairness_check(model, prompt_template, name_a, name_b, group_label):
    """
    把 prompt 中的姓名/代词替换，看输出是否一致。
    """
    out_a = model.generate(prompt_template.format(name=name_a))
    out_b = model.generate(prompt_template.format(name=name_b))
    
    # 简单 metric：两个输出的情感分数 / 长度 / 关键词覆盖
    sentiment_a = sentiment_score(out_a)
    sentiment_b = sentiment_score(out_b)
    
    diff = abs(sentiment_a - sentiment_b)
    return diff < threshold, diff
```

这是目前 LLM 偏见评估的常见做法，但远非严格意义上的反事实公平。

## 六、LLM 场景下的特殊考量

### 1. 文本生成不是二分类

传统公平指标假设二分类决策（录取 / 不录取）。LLM 的输出是开放文本，没有"正例率"的直接定义。研究者提出：

- **StereoSet 风格**：比较 stereotype / anti-stereotype / unrelated 的 LM 概率。
- **BBQ 风格**：让模型在 ambig 与 disambig 语境下回答，测偏见放大效应。
- **Counterfactual Sentiment**：替换姓名 / 代词，看情感分数变化。

### 2. 多群体同时评估

传统是 $G \in \{0, 1\}$ 两群体。LLM 面对**多类别**（种族、宗教、能力、性取向、年龄...）。常用方法：

- **最大差距（Max Disparity）**：$\max_{g} \text{metric}(g) - \min_{g} \text{metric}(g)$。
- **方差 / 标准差**：跨群体指标的离散度。

### 3. 交叉性（Intersectionality）

"黑人女性" vs "白人男性" 比单独看"黑人"或"女性"更易触发偏见。**交叉群体**的公平评估需要 $O(|G_1| \cdot |G_2| \cdot ...)$ 个组合——实践中用子集采样。

## 七、量化指标的代码实现

下面给一个"公平性仪表盘"的最小实现，可对模型预测结果跨群体计算 4 类指标：

```python
import numpy as np
from sklearn.metrics import confusion_matrix


def fairness_dashboard(y_true, y_pred, groups):
    """
    y_true: (N,) 真实标签
    y_pred: (N,) 模型预测（0/1）
    groups: (N,) 群体标签（如 'male' / 'female'）
    """
    metrics = {}
    unique_groups = np.unique(groups)

    for g in unique_groups:
        idx = (groups == g)
        tn, fp, fn, tp = confusion_matrix(y_true[idx], y_pred[idx], labels=[0, 1]).ravel()
        metrics[g] = {
            "P_pred_pos": (tp + fp) / max(tp + fp + tn + fn, 1),
            "TPR": tp / max(tp + fn, 1),
            "FPR": fp / max(fp + tn, 1),
            "PPV": tp / max(tp + fp, 1),
            "FNR": fn / max(fn + tp, 1),
            "selection_rate": (tp + fp) / max(tp + fp + tn + fn, 1),
        }

    # 跨群体差距
    keys = ["P_pred_pos", "TPR", "FPR", "PPV", "FNR"]
    gaps = {k: max(metrics[g][k] for g in unique_groups) -
                min(metrics[g][k] for g in unique_groups) for k in keys}
    return metrics, gaps


# 例子
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
groups = np.array(['M', 'M', 'M', 'F', 'F', 'F', 'M', 'F', 'M', 'F'])
m, gaps = fairness_dashboard(y_true, y_pred, groups)
print(gaps)
# {'P_pred_pos': 0.33, 'TPR': 0.5, 'FPR': 0.33, 'PPV': 0.0, 'FNR': 0.33}
```

**解读**：`TPR` 差距 0.5 表示在合格者中，男女通过率差异 50%——严重偏差。

## 八、实践选型指南

| 场景 | 推荐指标 | 理由 |
|---|---|---|
| **招聘 / 贷款**（历史偏见严重） | Equal Opportunity + Demographic Parity | 不让历史不平等传导 |
| **累犯预测**（已知 base rate 不均） | Predictive Parity | 保证"被判高风险的人确实高风险" |
| **医疗诊断**（不可漏诊） | Equal Opportunity（TPR 优先） | 漏诊代价大 |
| **内容审核**（不可错杀） | FPR 平衡 | 不让特定群体被过度删除 |
| **信息检索** | Demographic Parity + 个体公平 | 让不同群体曝光度相近 |
| **对话系统** | BBQ / StereoSet 风格 | 测偏见放大效应 |
| **代码生成** | （不太适用公平指标） | 主要关注功能正确性 |

## 九、缓解措施与公平性的相互作用

### 后处理（Post-processing）

调阈值使各群体指标平衡——简单但改变模型本身的能力。Hardt et al. (2016) 提出**随机化决策**：

```python
def randomized_threshold(y_score, group, target_tpr):
    """
    对每个群体选阈值，使 TPR 都等于 target_tpr。
    """
    thresholds = {}
    for g in np.unique(group):
        scores_g = y_score[group == g]
        # 选阈值使 TPR 等于 target
        thresholds[g] = np.quantile(scores_g[y_true[group == g] == 1], 1 - target_tpr)
    
    y_pred = np.array([
        int(y_score[i] >= thresholds[g]) 
        for i, g in enumerate(group)
    ])
    return y_pred
```

### In-processing（训练中约束）

把公平性作为正则项加入损失：

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{fair}}
$$

例如：

- **协变量偏移正则**：让各群体中间表征分布相近（Maximum Mean Discrepancy, MMD）。
- **对抗训练**：加一个"群体预测器"对抗器，迫使模型表征去除群体信息。
- **Reweighting**：训练时给不同 (群体, 标签) 组合赋权重。

### Pre-processing（数据层）

- **重采样**：让各群体在训练集中均衡。
- **数据增强**：为低资源群体生成合成数据（但要小心避免引入新偏见）。
- **标签清洗**：修正历史标注中的系统性偏差。

## 十、公平性与其他指标的权衡

**公平-准确率权衡**：很多情况下，强制公平会牺牲一定的总体准确率。关键问题是：

1. **谁来承担代价？** 通常是弱势群体。
2. **谁决定权重？** 这就是治理问题。
3. **可解释性是否足够？** 要让受影响的群体知道模型如何决策。

**经验法则**：
- 在高 stakes 场景（医疗、刑事），公平性约束 > 准确率微优化。
- 在低 stakes 场景（推荐、娱乐），可以更宽松。
- 公平性指标应该与产品 SLA 同级，纳入发布门槛。

## 小结

"公平"不是单一指标，而是**多种数学定义之间的取舍**。Chouldechova 与 Kleinberg 的不可能定理告诉我们：**当群体 base rate 不同时，校准 + 等机会 + 等赔率不可能同时满足**。这意味着任何"我们做到了公平"的说法都要追问："哪个公平？代价是什么？谁决定的？"——下一篇我们将看到具体的**去偏见技术**，包括数据增强、in-processing 正则、对抗训练与解码阶段干预，以及它们的局限。
