# 解码策略：从 Greedy 到 Sampling 的文本生成核心算法

语言模型的训练目标是"预测下一个 token 的概率分布" $P(t_i \mid t_{<i})$，但**如何从这个分布中采样**出一个完整的序列——这是解码策略要回答的问题。从 Greedy、Beam Search 到 Top-K、Top-p，不同策略在不同场景下效果差异巨大。本文深入剖析主流解码算法的数学原理、优缺点与工程实现。

## 一、问题的本质

给定语言模型 $P(\mathbf{y} \mid \mathbf{x}) = \prod_i P(y_i \mid \mathbf{x}, y_{<i})$，我们想找一个"好"的序列 $\mathbf{y}^*$：

$$
\mathbf{y}^* = \arg\max_{\mathbf{y}} \; F(\mathbf{y} \mid \mathbf{x})
$$

$F$ 是评价函数——可能是：

- **对数概率**（Greedy、Beam Search）。
- **质量评分**（如 BERT-based NLI 分数）。
- **人类偏好**（RLHF 微调后的 reward model）。

不同 $F$ 对应不同算法。核心挑战：

1. **搜索空间巨大**：词表 32K，序列长度 100，搜索空间 $32K^{100}$。
2. **训练-推理 mismatch**：训练时 teacher forcing 看真实 token，推理时只能看自己的预测。
3. **质量与多样性**：机器翻译要确定性，创意写作要多样性——没有银弹。

## 二、Greedy Decoding：最简单但最差

### 2.1 算法

每步选概率最大的 token：

$$
y_t^* = \arg\max_{y} P(y \mid \mathbf{x}, y_{<t})
$$

时间复杂度 $O(T \cdot V)$，实现极简单。

### 2.2 问题

- **局部最优**：每步最优不保证全局最优。例如生成"中国"可能让概率排第 2 的"美国"被压制，但后者接续的句子可能更通顺。
- **缺乏多样性**：完全确定性，同样的输入永远生成同样的输出。
- **循环问题**：生成的 token 序列可能陷入循环（如 "the the the"）。

### 2.3 改进：Greedy + No-Repetition Penalty

对已生成 token 的 logits 减去惩罚：

$$
\text{logits}'_i = \text{logits}_i - \alpha \cdot \mathbb{1}[i \in \text{generated}]
$$

$\alpha$ 是超参（典型 0.5-2.0），强制模型不重复。

## 三、Beam Search：经典最优

### 3.1 算法

维护 top-K 个候选，每步扩展：

```
beam = [(<BOS>, log_prob=0)]
for step in range(max_len):
    candidates = []
    for prefix, score in beam:
        if prefix ends with <EOS>:
            candidates.append((prefix, score))
            continue
        top_k = topk(P(·|prefix), k=beam_size)
        for token, lp in top_k:
            candidates.append((prefix + [token], score + lp))
    beam = topk(candidates, k=beam_size, key=lambda x: x[1])
```

### 3.2 长度惩罚

朴素 Beam Search 偏好短句（每步都加负分）。长度惩罚：

- **Wu 公式**：`score / (5 + |y|)^α / (5 + 1)^α`，α=0.6-1.0。
- **Google 公式**：`((5 + |y|) / 6)^α`。
- **简单平均**：`score / |y|`。

### 3.3 Beam Search 的问题

#### (a) 退化（Degradation）

更大的 beam 不一定更好：

- Beam=1（Greedy）：BLEU 30。
- Beam=5：BLEU 35。
- Beam=100：BLEU 33。

原因：beam 越大，越倾向于"高概率但平庸"的句子，反而错过了"中概率但精彩"的候选。

#### (b) 重复

Beam Search 容易生成重复短语（如"the the the"），即使加 no-repeat-ngram penalty 也无法完全消除。

#### (c) 与人类偏好脱节

Welleck et al.（2020）证明 Beam Search 生成的文本在 perplexity 上低，但人类评估反而差——过度优化似然反而生成"无聊"的文本。

### 3.4 改进

#### (a) Diverse Beam Search

把 beam 分成 G 组，组间互斥：

$$
\text{score}_{g} = \text{lm\_score}_{g} - \lambda \cdot \text{cosine}(\text{hidden}_{g}, \text{hidden}_{g'})
$$

强制不同组关注不同子空间，生成多样化候选。

#### (b) Length-Normalized Diverse Beam Search

Vijayakumar et al.（2018）提出结合长度归一化和多样性惩罚，在 Image Caption 上 BLEU 提升 2-3 个点。

#### (c) Constrained Beam Search

加入硬约束（如"必须包含关键词"、"长度范围"）：

$$
\text{score} = \text{lm\_score} + \sum_c \text{constraint}(c, \mathbf{y})
$$

Discourse-Aware Beam Search、Lexically Constrained Decoding 都是这一思路。

## 四、Sampling：随机性回归

### 4.1 朴素 Sampling

每步从 $P(\cdot \mid y_{<t})$ 中随机采样：

```python
logits = model(prefix)
probs = softmax(logits)
next_token = multinomial(probs)
```

问题：

- 长尾噪声：高熵的分布会采到怪词。
- 质量不稳定：同样 prompt 多次生成质量差异巨大。

### 4.2 Temperature

温度 $T$ 控制分布的"尖锐度"：

$$
P_T(y) = \frac{\exp(\text{logits}_y / T)}{\sum_{y'} \exp(\text{logits}_{y'} / T)}
$$

- $T \to 0$：分布尖锐 → Greedy。
- $T = 1$：原始分布。
- $T \to \infty$：分布均匀 → 纯随机。
- $T = 0.7$：创意写作的常用值。

直觉：温度就是"模型的创造性 vs 准确性"旋钮。

## 五、Top-K Sampling

只从前 K 个最高概率 token 中采样：

```
top_k_logits = topk(logits, k=K)
probs = softmax(top_k_logits)
next_token = multinomial(probs)
```

K=50 是 DialoGPT 的默认值，能避免采样到怪词，但 K 是固定值——对不同分布不友好。

### 5.1 问题

- 分布平坦时（高熵）：K=50 可能只占总概率的 30%，遗漏了大量合理 token。
- 分布尖锐时（低熵）：K=50 包含了大量低概率 token，引入噪声。

## 六、Top-p (Nucleus) Sampling

Holtzman et al.（2020）提出 Top-p：选择**累计概率 ≥ p** 的最小 token 集合：

```
sorted_probs = sort(probs, descending=True)
cumsum = cumsum(sorted_probs)
cutoff = argmax(cumsum >= p)
top_p_logits = top_p_logits[:cutoff+1]
```

直觉：

- 分布尖锐时：只保留前 2-3 个 token。
- 分布平坦时：保留前 30-50 个 token。

p=0.9-0.95 是对话生成的常用值。

### 6.1 Top-p 的优势

- **自适应**：K 自动随分布调整。
- **避免怪词**：尾部低概率 token 直接被截断。
- **保持多样性**：在保留的多样性范围内采样。

## 七、Top-K + Top-p 组合

许多现代 LLM 同时使用 Top-K 和 Top-p：

```python
def top_k_top_p_filter(logits, top_k=50, top_p=0.9):
    # Top-K
    if top_k > 0:
        indices_to_remove = logits < topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("inf")
    # Top-p
    sorted_logits, sorted_indices = sort(logits, descending=True)
    cumulative_probs = cumsum(softmax(sorted_logits))
    sorted_indices_to_remove = cumulative_probs > top_p
    # 至少保留一个 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = scatter(sorted_indices_to_remove, sorted_indices)
    logits[indices_to_remove] = -float("inf")
    return logits
```

LLaMA、Qwen 等模型默认 `top_k=50, top_p=0.9, temperature=0.7`。

## 八、Contrastive Search（SimCTG）

Su & Collier（2022）提出的 Contrastive Search 解决"重复 + 多样性"的两难：

$$
x_t = \arg\max_{y \in V^{(k)}} \left\{ (1-\alpha) \cdot P_\theta(y \mid x_{<t}) - \alpha \cdot \max_{x_j \in x_{<t}} \text{sim}(\mathbf{h}_y, \mathbf{h}_{x_j}) \right\}
$$

直觉：在 top-k 候选中，选择"概率高"且"与已生成 token 相似度低"的 token。$\alpha$ 控制多样性与质量的平衡。

实验显示 Contrastive Search 在 LLaMA-7B 上人类评估显著优于 Greedy / Beam Search。

## 九、典型解码器对比

| 算法 | 速度 | 质量 | 多样性 | 适用 |
| --- | --- | --- | --- | --- |
| Greedy | 最快 | 中 | 无 | 简单任务 |
| Beam Search | 中 | 高 | 低 | 机器翻译、摘要 |
| Sampling | 快 | 低 | 高 | 创意写作 |
| Top-K | 快 | 中 | 中 | 对话 |
| Top-p | 快 | 高 | 高 | 通用对话 |
| Contrastive | 慢 | 高 | 高 | 高质量长文本 |

## 十、CFG（Classifier-Free Guidance）

CFG 是扩散模型（Stable Diffusion）的关键技术，但在 LLM 中也开始流行。

核心思路：用两个 prompt（条件 + 无条件）做差：

$$
\tilde{P}(y \mid c, u) = P(y \mid u) + \lambda (P(y \mid c) - P(y \mid u))
$$

$\lambda$ 是 guidance scale：$\lambda=0$ 是无条件，$\lambda=1$ 是标准条件，$\lambda > 1$ 是放大条件的影响。

CFG 让生成更"符合 prompt"，但代价是多样性下降。Stable Diffusion 默认 $\lambda=7.5$。在 LLM 中用于 instruction following 与可控生成。

## 十一、Mixture of Decoders

实际工程中常常**混合使用多种解码策略**：

- **Self-Consistency**：多次采样 + 投票选最一致的答案（Wang et al., 2023）。
- **Best-of-N**：采样 N 个候选，用 reward model 选最好的。
- **ReAct + CoT**：先生成思维链，再用 Greedy 解码。

代表工作：

- **Best-of-N Sampling**：OpenAI 的 RLHF 论文中，生成 4-16 个候选，由人类评估。
- **Self-Consistency**：GSM8K 数学推理从 Greedy 的 50% 提升到 80%+。
- **Tree-of-Thoughts**：搜索思维树，结合 beam search。

## 十二、评估解码质量的指标

### 12.1 自动指标

- **Perplexity**：模型对生成文本的"困惑度"——越低越好。
- **Distinct-N**：生成文本中 unique n-gram 比例——越高越多样。
- **Self-BLEU**：生成文本之间的 BLEU 分数——越低越多样。

### 12.2 人类评估

- **流畅度**（Fluency）：语法是否正确。
- **一致性**（Coherence）：上下文是否连贯。
- **相关性**（Relevance）：是否回答了 prompt。
- **创造性**（Creativity）：是否有新意。

通常用 Likert 5 分制或 pairwise 比较。

## 十三、PyTorch 工具实现

```python
import torch
import torch.nn.functional as F


def top_k_top_p_logits(logits, top_k=0, top_p=1.0):
    """Logits 过滤：保留 top-k 或 top-p。"""
    if top_k > 0:
        # 移除 top-k 以下的 logits
        kth_value = torch.topk(logits, top_k)[0][..., -1, None]
        logits = torch.where(logits < kth_value, torch.full_like(logits, -float("inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits = torch.where(indices_to_remove, torch.full_like(logits, -float("inf")), logits)
    return logits


@torch.no_grad()
def generate(model, input_ids, max_new_tokens=128, temperature=1.0, top_k=0, top_p=1.0):
    """最简单的采样生成。"""
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[:, -1, :] / temperature
        logits = top_k_top_p_logits(logits, top_k, top_p)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return input_ids
```

工程优化：

- **KV-cache**：避免重复编码已生成 token。
- **Batch decoding**：同 batch 内并行生成多个序列。
- **Speculative decoding**：小模型预生成，大模型批量验证。

## 十四、不同任务的最优策略

| 任务 | 推荐解码 | 原因 |
| --- | --- | --- |
| 机器翻译 | Beam Search (K=5) | 准确性优先 |
| 文本摘要 | Beam Search + Length Norm | 准确性 + 长度控制 |
| 命名实体识别 | Greedy | 简单、确定性 |
| 对话 | Top-p (0.9) + Temperature (0.7) | 多样性优先 |
| 创意写作 | Top-p (0.95) + Temperature (1.0) | 最大多样性 |
| 代码生成 | Greedy / Low-temp Sampling | 准确性优先 |
| 数学推理 | Self-Consistency | 推理路径多样性 |
| 故事创作 | Contrastive Search | 避免重复 |

## 十五、未来方向

1. **RLHF-First**：未来 LLM 的主要解码策略由 reward model 决定，而非手动调参。
2. **Constitutional AI**：用 self-critique 而非外部奖励生成。
3. **Speculative Decoding**：小模型预测 + 大模型验证，加速 2-3 倍。
4. **Tree-of-Thoughts**：树搜索 + 自我评估，复杂推理的 SOTA。

## 小结

解码策略是 LLM 应用的"最后一公里"——同一模型在不同解码策略下能产生差异巨大的输出。从 Greedy 到 Beam Search 到 Top-p 到 Contrastive Search，**没有银弹，只有 trade-off**：

- 准确性 → Beam Search / Greedy
- 多样性 → Top-p / Sampling
- 推理 → Self-Consistency
- 创意 → Contrastive / Temperature

工业实践中常见的做法：**多种策略并行生成 + reward model 选最优**。下一篇我们将深入文本生成的**可控性**：如何让模型按指定主题、风格、情感生成。
