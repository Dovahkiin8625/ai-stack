# 探针分类器与表征分析

**探针分类器（Probing Classifier）** 是 NLP 可解释性最经典的方法之一：在预训练模型的某一层表征上训练一个简单的分类器，看该层**编码了什么信息**。如果一个浅层分类器就能高准确率地预测某个属性（词性、句法、语义、偏见），说明该层**已经学到该信息**。本文系统介绍 probing 的方法论（线性探针、MLP 探针、selectivity 评估）、典型发现（BERT 的句法树在某一层清晰可分），以及 probing 的根本局限——**信息可解码 ≠ 因果相关**。

## 一、什么是 Probing

Probing 的核心假设：

> **如果模型的某层表征 $h$ 包含某属性 $z$ 的信息，那么存在一个简单分类器 $f$ 能从 $h$ 预测 $z$。**

形式化：给定数据集 $\{(h_i, z_i)\}_{i=1}^N$（h 是层表征，z 是属性标签），训练 $f_\theta$ 最小化：

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \ell\big(f_\theta(h_i), z_i\big)
$$

如果 $f$ 的准确率远高于随机基线，说明该层编码了 $z$。

## 二、Probing 的常见任务

| 任务类型 | 属性 | 数据集 |
|---|---|---|
| 形态 | 词性标注（POS）| UD Treebank |
| 句法 | 依存关系、成分句法 | SST, Penn Treebank |
| 语义 | 语义角色标注、NER | CoNLL-2003, OntoNotes |
| 语用 | 情感、讽刺 | SST, SARC |
| 世界知识 | 实体类型、关系 | FewRel, TACRED |
| 偏见 | 性别 / 种族关联 | StereoSet, CrowS-Pairs |
| 推理 | NLI | SNLI, MultiNLI |

## 三、探针的实现

### 1. 线性探针（Linear Probe）

最简单的形式：单层线性 + softmax/逻辑回归。

```python
import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """最基础的线性探针：单层线性分类器。"""
    def __init__(self, hidden_dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, h):
        return self.classifier(h)  # (B, num_labels)


# 训练循环
def train_probe(representations, labels, num_epochs=10, lr=1e-3):
    """
    representations: (N, D) 某层 token 级别的表征
    labels: (N,) 标签
    """
    probe = LinearProbe(representations.shape[-1], num_labels=int(labels.max() + 1))
    optim = torch.optim.AdamW(probe.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        logits = probe(representations)
        loss = nn.functional.cross_entropy(logits, labels)
        optim.zero_grad(); loss.backward(); optim.step()
    
    return probe


def evaluate_probe(probe, representations, labels):
    with torch.no_grad():
        preds = probe(representations).argmax(-1)
    return (preds == labels).float().mean().item()
```

### 2. MLP 探针

更深但简单的非线性探针：

```python
class MLPProbe(nn.Module):
    def __init__(self, hidden_dim, num_labels, depth=2):
        super().__init__()
        layers = [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, num_labels))
        self.net = nn.Sequential(*layers)
    
    def forward(self, h):
        return self.net(h)
```

### 3. 控制任务（Control Task）

Hewitt & Manning (2019) 提出 **control task**：对**随机打乱的标签**也训练一个探针。

```python
def control_task_accuracy(representations, true_labels):
    """
    把标签随机打乱重新训练探针。
    打乱后的准确率 = baseline（"信息本来就能被学到的难度"）。
    真实标签的准确率 vs 打乱的差距 = 真实可解码性。
    """
    n = len(true_labels)
    perm = torch.randperm(n)
    shuffled_labels = true_labels[perm]
    return train_probe(representations, shuffled_labels)


# 评估
real_acc = evaluate_probe(probe_real, repr, labels)
control_acc = evaluate_probe(probe_control, repr, labels)  # 应该接近随机
selectivity = real_acc - control_acc  # 真实可解码信息
```

**selectivity** 是更严格的指标——如果 control task 也很高，说明标签本身有结构，探针可能在"作弊"。

### 4. Selectivity vs Informativeness 双指标

- **Informativeness（信息量）**：真实任务的探针准确率。反映层编码了多少信息。
- **Selectivity（选择性）**：真实准确率 - control 准确率。反映信息是否**专门为该任务组织**。

理想探针：informativeness 高、selectivity 高——模型"专门为这个任务用了某种组织方式"。

### 5. 极简探针 MDL（Minimum Description Length）

Voita & Titov (2020) 用**信息论视角**评估 probing：

$$
\text{MDL} = \text{bits to encode labels} - \text{bits to encode labels given representation}
$$

结合探针的参数量归一化，更细粒度衡量"信息增益"。

## 四、典型发现

### 1. BERT 的句法信息

Hewitt & Manning (2019) 用 **structural probe**：在 BERT 各层训练探针预测**依存树距离**：

$$
d(\text{token}_i, \text{token}_j) \approx \|h_i - h_j\|
$$

**发现**：
- BERT 中间层（第 8~10 层）几乎**完美预测依存距离**。
- 与人类标注的句法树高度一致。
- 第 1~2 层几乎不编码句法（仅 token-level 信息）。
- 顶层（第 12 层）句法信息减弱（被任务特定信息覆盖）。

### 2. 上下文信息 vs 静态词义

BERT 的同一词在不同上下文有不同表征。探针实验显示：

- 低层捕获**静态词义**（词性、词频）。
- 中层捕获**上下文融合**（实体类型、句法角色）。
- 高层捕获**任务特定**信息。

### 3. 偏见信息

Kurita et al. (2019) 训练探针从 BERT 表征预测职业的性别偏向：

```python
# 给定 [CLS] 表征，预测职业 + 性别关联
# 探针高准确率 → 该层编码了性别偏见
```

发现：BERT 中间层对"男护士 / 女护士"的反应**不对称**——证明偏见确实在内部编码。

### 4. 知识与推理

Petroni et al. (2019) 的 **LAMA 基准**：用 cloze-style prompt 让模型填空，测试**事实性知识**。后续探针工作显示 GPT 系列在**前 1/3 层**就已编码大量事实知识。

### 5. 多语言表征对齐

Libovický et al. (2020) 探针发现：mBERT 在**句法层**有跨语言对齐（探针可跨语言迁移），但在**词汇层**对齐差——这解释了 mBERT 多语言能力的瓶颈。

## 五、Probing 的根本局限

### 1. 相关 ≠ 因果

**核心批评**：探针高准确率只能说明"信息可被解码"，**不能说明模型实际使用了该信息**。

```text
"某层可解码 POS 信息"  ≠  "模型做 POS 决策时用了该层"
```

可能的情况：
- 信息存在但**未被下游使用**（"死信息"）。
- 信息存在且被使用，但**对最终任务贡献极小**。

### 2. 探针能力 ≠ 信息存在

反过来：探针低准确率**不能证明信息不存在**——可能探针没找到正确解码方式。

### 3. 探针 vs 模型能力的混淆

深层探针可能学到了**自己的归纳偏置**，而非模型表征的真实结构。

### 4. 任务特定 vs 通用

Probing 只对**特定任务 z** 有意义。"BERT 编码了句法"是说"BERT 表征可被线性映射到句法标签"——这个映射可能在**其它层**失效，对**其它模型**失效。

### 5. 因果 probing 方法

为解决"相关 ≠ 因果"，研究者提出：

- **Causal Mediation Analysis**：对某层某维度做干预，看下游影响。
- **Activation Patching**：把激活换成不同输入的版本，看输出变化（详见 mechanistic-interpretability）。
- **Edge Probing**：把探针作为"软边"，通过梯度反传到模型。

## 六、Probing 在 AI 安全中的应用

### 1. 偏见审计

```python
def audit_bias(model, layer, tokenizer):
    """
    对每个 (occupation, gender_pronoun) 对，比较层表征的距离。
    """
    templates = [
        "The {gender} works as a {occupation}.",
        "{gender} is a {occupation}.",
    ]
    occupations = ["nurse", "doctor", "engineer", "teacher", ...]
    genders = ["man", "woman", "he", "she", ...]
    
    diffs = []
    for occ in occupations:
        reprs = {g: model.get_repr(templates[0].format(gender=g, occupation=occ))
                 for g in genders}
        # 计算男女对之间的距离
        for g1 in ["man", "he"]:
            for g2 in ["woman", "she"]:
                diff = torch.norm(reprs[g1] - reprs[g2])
                diffs.append((occ, g1, g2, diff.item()))
    
    return diffs
```

如果"doctor" 男女表征距离远大于"nurse"——说明偏见存在。

### 2. 知识探测

测试模型是否编码了**敏感知识**（如"如何制造生物武器"）。如果探针能高准确率地从某层解码——这是个风险信号。

### 3. 早期失败检测

监控训练中各层的探针准确率——**异常变化**可能预示着训练不稳定或数据问题。

### 4. 推理链分析

用探针追踪模型推理时各层的**推理步骤**——例如"假设 A 的探针在 layer X 激活"。

## 七、Probing 的工程实践

### 推荐工具

- **TransformerLens**（Neel Nanda）：完整模型 hook + cache 接口。
- **nnsight**：跨框架的干预工具。
- **Captum**：PyTorch 的可解释性库，含 Integrated Gradients 等。
- **pyvene**：OpenAI 发布的干预库。

### 最佳实践

1. **多层比较**：永远不要只看一层——信息在不同任务的最优层不同。
2. **多种探针**：至少跑线性 + MLP + control task。
3. **统计显著性**：多次随机种子平均 + 标准差。
4. **报告 baseline**：随机基线、词汇基线（仅用 token）、纯频率基线。
5. **可重复性**：固定随机种子、保存数据索引、报告探针超参。

### 常见陷阱

- **过拟合**：探针容量过大 → 记住训练集而非解码信息。用**留出集** + **小探针**。
- **数据泄漏**：训练/测试数据有同源句 → 探针准确率虚高。
- **采样偏差**：标签不平衡 → 用 macro-F1 而非 accuracy。
- **只看顶层**：底层可能编码了任务关键信息。

## 八、Probing vs 其他可解释性方法

| 方法 | 粒度 | 因果性 | 适用场景 |
|---|---|---|---|
| Probing | 单层表征 | **弱**（仅相关） | 知道模型"学到了什么" |
| Activation Patching | 单组件 | **强** | 知道模型"用了什么" |
| Attention Visualization | 单头 | 弱（attention ≠ 重要性） | 直觉理解 |
| SHAP / LIME | 输入特征 | 弱 | 局部解释 |
| Causal Tracing | 单层+token | 强 | 局部知识定位 |
| SAE | 单个特征 | 中 | 找到"概念单元" |
| Logit Lens | 各层 logits | 中 | 看"模型如何一步步想" |

**互补使用**：先用 probing 找"哪层含目标信息"，再用 activation patching 确认"该层是否因果必要"，最后用 SAE 把"信息"分解成可解释特征。

## 九、Probing 的哲学问题

Probing 默认"信息存在于某个可定位的位置"——这是**位置论（localism）**。

**整体论（holism）** 认为：信息是**分布式**的，没有"某个神经元做某件事"这回事，整个网络作为一个整体才"理解"。

BERT 句法探针的成功**支持位置论**——但 SAE 的发现（特征是多神经元组合）更支持**整体论 + 稀疏基底**。这仍是 MI 的开放问题。

## 小结

Probing 是 NLP 可解释性的"老炮"——简单、可解释、能告诉你模型在哪一层编码了什么信息。BERT 句法探针、跨语言对齐探针、偏见审计探针都是经典案例。但**根本局限是"相关 ≠ 因果"**：高探针准确率不能证明模型真的用了该信息。下一篇我们将看到 SAE（稀疏自编码器）如何在这一基础上前进一大步——它直接把模型的混合激活分解成**稀疏单义特征**，让"神经元 = 概念"重新成立。
