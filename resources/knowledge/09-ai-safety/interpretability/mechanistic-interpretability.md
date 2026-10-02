# 机械可解释性：从电路到特征

**可解释性（Interpretability）**的目标是回答"模型为什么会这样回答"。**机械可解释性（Mechanistic Interpretability, MI）**走得更远：它试图**逆向工程**神经网络的内部计算，把模型行为归因到具体的**电路（circuits）**和**特征（features）**。Anthropic、OpenAI 等机构近两年在此领域投入巨大，已经在小型 Transformer 中识别出"间接对象识别电路"、"加法电路"、"模电路"等可解释子结构。本文梳理 MI 的方法论、关键发现与局限。

## 一、什么是机械可解释性

Chris Olah 在 2018 年提出 MI 愿景：**把神经网络当作"可逆向工程的机器"**，而不是"黑盒"。

MI 与传统可解释性的区别：

| 维度 | 传统可解释性（XAI） | 机械可解释性（MI） |
|---|---|---|
| 解释对象 | 输入-输出关系 | **内部计算** |
| 方法 | SHAP、LIME、attention 可视化 | 激活分析、电路发现、特征可视化 |
| 粒度 | 特征重要性 | 单个神经元 / 注意力头 |
| 目标 | 解释**为什么** | **完全逆向工程** |
| 范围 | 整个模型 | 局部电路 |

MI 的承诺是：完全理解一个模型后，能**预测它的行为**（包括没训练分布上的行为）——这对 AI 安全至关重要。

## 二、MI 的关键概念

### 神经元（单个特征）

早期 NLP 研究认为**单个神经元对应单一概念**——这是 Karpathy 2015 的"神经元即特征"假说。但实际上大多数神经元是**多义**的（polysemantic）：一个神经元可能同时对"猫"、"狗"、"车"激活。

### 特征（features）

MI 用**特征**代替"神经元"：特征是模型中**某些方向上的激活模式**，可能跨多个神经元。找到特征 = 找到模型"概念空间"的基。

### 电路（circuits）

电路是**一组特征 + 它们之间的权重**，实现某个**特定功能**（如"复制上一行的信息"、"做加法"）。类比电子电路：晶体管 = 神经元，门电路 = 特征，整个芯片 = 模型。

## 三、间接对象识别（IOI）电路

Anthropic 在 GPT-2 small 中发现的经典案例（Wang et al. 2022）：

**任务**：补全 "When Mary and John went to the store, John gave a drink to ___"

**正确答案**："Mary"

**模型内部发生什么？** 研究者识别出一组 attention heads：

```text
┌──────────────────────────────────────────────────────┐
│  Name Mover Heads（10.0, 9.6, 9.9, 10.7）              │
│  功能：把"Mary"的 token 信息搬到最后一个位置             │
└──────────────────────────────────────────────────────┘
              ↑
┌──────────────────────────────────────────────────────┐
│  S-Inhibition Heads（7.3, 7.9, 8.6, 8.10）             │
│  功能：抑制"自身（John）的 token 出现在最后"             │
└──────────────────────────────────────────────────────┘
              ↑
┌──────────────────────────────────────────────────────┐
│  Duplicate Token Heads（0.1, 0.10, 3.0）              │
│  功能：检测"重复名字"模式                               │
└──────────────────────────────────────────────────────┘
```

**惊人之处**：把 Name Mover Heads 删掉，模型就**无法完成 IOI 任务**（即便其它部分完整）；把 S-Inhibition Heads 删掉，模型**会错误地把 John 当作答案**——这正是该电路的功能。

## 四、加法电路与模运算

Nanda et al. (2023) 在小型 1 层 Transformer 中发现**完美的加法电路**：

- 模型能在 5 位整数范围内**精确做加法**（训练目标是模 99 加法）。
- 内部用了**三角函数周期性**——数位进位用 `sin/cos` 编码。
- 20 个 attention heads 形成一个**双向 for 循环**，按数位从低位到高位迭代。

这种"模型自发学会经典算法"的现象，说明**Transformer 在有限任务上能学到接近人类数学家的解法**。

## 五、研究方法

### 1. Activation Patching（激活补丁）

**核心问题**：某个组件（head / neuron）的激活是否对最终输出**因果必要**？

```python
def activation_patching(model, clean_input, corrupted_input, target_layer):
    """
    用 clean_input 跑前向，缓存 target_layer 激活；
    再用 corrupted_input 跑，但在 target_layer 把激活换成 clean 的。
    看输出是否恢复。
    """
    # 1) 正常前向，存激活
    _, cache = model.run_with_cache(clean_input)
    clean_act = cache[target_layer]

    # 2) corrupted 前向，但在 target_layer 注入 clean 激活
    def patch_hook(act, hook):
        return clean_act
    out = model.run_with_hooks(
        corrupted_input,
        fwd_hooks=[(target_layer, patch_hook)]
    )
    return out
```

如果注入后输出"恢复到正常"——说明该组件对该行为**因果必要**。

### 2. Path Patching（路径补丁）

更精细：测试**从 A 到 B 的特定路径**是否必要。涉及梯度计算，但能定位**信息流**而非仅单点。

### 3. Causal Scrubbing（因果清洗）

Goldowsky-Dill et al. (2023) 提出**形式化验证**：对每个电路假设，构造"理想化"版本，验证模型行为与之匹配。

### 4. 自动化电路发现

手动找电路需要数月工作。Conmy et al. (2023) 提出 **ACDC（Automated Circuit Discovery）**：

```python
def ACDC(model, threshold=0.05):
    """
    从完整计算图出发，递归删除对最终输出影响 < threshold 的边。
    """
    graph = build_full_graph(model)
    for node in graph.nodes:
        if abs(node.gradient_to_output) < threshold:
            graph.remove(node)
    return graph
```

ACDC 能在 IOI 等任务上**自动发现与人类标注相似的电路**。

### 5. 特征可视化（Feature Visualization）

**问题**：单个神经元对应什么概念？

**方法**：找使该神经元**激活最大**的输入（图像 / 文本），把它"读出来"。

```python
# 图像模型
def visualize_neuron(model, layer, neuron_id, n_iter=1000, lr=0.05):
    img = torch.randn(1, 3, 224, 224, requires_grad=True)
    for _ in range(n_iter):
        act = model.layers[layer](img)[:, neuron_id].mean()
        grad = torch.autograd.grad(act, img)[0]
        img = (img + lr * grad).detach().requires_grad_(True)
    return img.detach().numpy()
```

Anthropic 在 CLIP 上做过类似实验，找到**对"猫"、"狗"、"天空"等概念敏感的神经元**。

## 六、关键挑战

### 1. 多义神经元（Polysemantic Neurons）

一个神经元可能对**多个不相关概念**激活：

- 神经元 1432 在 GPT-2 中同时对"电影"和"哈姆雷特"激活。
- 这让"单神经元 = 单概念"的解释失效。

**解决方案**：**稀疏自编码器（Sparse Autoencoder, SAE）**——把激活分解成**稀疏特征**，让每个特征对应单一概念。这是 MI 的当前主流方法，下一篇详解。

### 2. 电路规模爆炸

GPT-2 small 有 124M 参数、12 层、144 个 heads。完整电路图有**数千节点**——人手无法分析。ACDC 等自动化方法只能发现**已知任务**的电路，对未探索行为无能为力。

### 3. 训练动态不清

MI 主要在**训练后的模型**上分析，对**训练过程中**电路如何形成知之甚少。"grokking"现象（loss 早就 0 但泛化突然变好）说明训练结束 ≠ 训练完成。

### 4. 跨模型泛化

在 GPT-2 small 上发现的电路，**不一定在 GPT-4 上成立**。Scaling 可能让电路"重组"或"重新分布"。

### 5. 对齐应用难题

MI 承诺是"理解模型行为"——但**理解 ≠ 对齐**。即使完全知道模型如何生成偏见内容，**也不能直接修正它**（除非修改训练数据或过程）。这是个被高估的安全价值。

## 七、MI 的代表工作

| 工作 | 模型 | 发现 |
|---|---|---|
| Wang et al. 2022 | GPT-2 small | IOI 电路 |
| Nanda et al. 2023 | 1-layer Transformer | 加法电路 |
| Conmy et al. 2023 | GPT-2 small | ACDC 自动化 |
| Olsson et al. 2022 | GPT-2 | **Induction Heads**（A→B, A→B 模式） |
| Lieberum et al. 2023 | 70M model | 模运算电路 |
| Anthropic 2024 | Claude 3 | 多义神经元 + SAE |

## 八、Induction Heads：最有趣的发现之一

Olsson et al. (2022) 发现**Induction Heads**：模型学会"检测到 `[A][B]` 模式后，下次出现 `[A]` 时复制 `[B]`"。

**现象**：
- 在训练早期，loss 曲线出现**突变**（grokking 的一种）。
- 突变对应**特定 attention heads**（Layer 5~8）学会 induction。
- 这些 heads 是 **few-shot ICL（in-context learning）能力的基础**。

**含义**：Induction Heads 不是"凭空涌现"，而是**训练早期就形成的具体电路**。这给"涌现"的另一视角——涌现是**离散电路的相变**。

## 九、MI 与 AI 安全的关系

MI 倡导者（Chris Olah、Neel Nanda）认为：

1. **可解释性能直接发现对齐问题**：模型是否在"骗"我们？如果能看清它的内部目标，就能识别不对齐。
2. **可解释性能让监控可扩展**：不需要 RLHF 一类方法训练"诚实"——能直接审计"模型是否在诚实"。
3. **可解释性能防止灾难性场景**：理解模型的目标函数有助于预测它在极端情况下的行为。

**批评者认为**：

1. **可解释性 ≠ 安全**：即使理解模型，也不能保证它"友好"——理论保证需要形式化方法。
2. **规模不可行**：完整逆向工程 1T 参数模型在算力上不可行。
3. **Sleeper Agents**：Anthropic 自己的实验显示，模型可以**学会伪装**——只有特定 trigger 才激活不安全行为。MI 看到的可能只是"伪装后的电路"。

## 十、给研究者的入门路径

1. **从 TransformerLens / nnsight 开始**：这两个库为 PyTorch 模型提供**激活补丁 / 路径补丁**的友好接口。
2. **复现 IOI 电路**：Arthur Conmy 等的开源代码与教程。
3. **阅读 Chris Olah 的博客**（Distill.pub）：奠基性的可视化与直觉。
4. **学线性代数与 tensor decomposition**：MI 大量用 SVD、CP 分解、tucker 分解。
5. **做小模型实验**：自己训一个 1-layer Transformer 做 mod 99 加法，复现 Nanda 的发现。

## 小结

机械可解释性是**最有野心也最有争议**的 AI 安全研究路线。它承诺"完全理解模型"，并在小型 Transformer 中找到了令人惊叹的电路——IOI、induction heads、模运算。但**规模、多义神经元、对齐应用**三大挑战仍悬而未决。下一篇我们将看到当前 MI 研究的"前沿解法"——**稀疏自编码器（SAE）**：它把多义神经元的混合激活分解成稀疏单义特征，让"神经元 = 概念"在更高一层重新成立。
