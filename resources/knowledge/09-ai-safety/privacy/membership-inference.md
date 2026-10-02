# 成员推断攻击：从模型泄漏训练成员

**成员推断攻击（Membership Inference Attack, MIA）** 是 LLM 时代最实际的隐私威胁之一：给定一条记录（或一段文本），攻击者能判断它**是否在模型的训练集中**？对个人而言，这意味着"我发过的邮件是否被用来训练"；对企业而言，"我的商业文档是否泄漏给了模型"。MIA 在 2017 年由 Shokri et al. 提出，2023 年因 ChatGPT 而广泛进入公众视野——有研究显示 16%+ 的 LLM 输出可被识别为"训练数据"。本文梳理 MIA 的攻击方法、防御手段，以及在 LLM 上的现状。

## 一、什么是成员推断

形式化：给定模型 $\mathcal{M}$ 训练于数据集 $D$，给定查询 $x$，攻击者判断：

$$
\text{MIA}(x) = \mathbb{1}[x \in D]
$$

**三种粒度**：
1. **记录级**：某条具体记录是否在训练集。
2. **序列级**：某段文本是否被完整见过。
3. **信息级**：某事实（如"我的医疗记录"）是否被模型"知道"。

## 二、为什么模型会泄漏成员信息

直觉：模型对**训练样本的拟合度 > 验证样本**。

- 训练样本被模型"反复记忆"，loss 极低。
- 验证样本只在 inference 时见过一次，loss 较高。

这个差距——**loss gap**——就是 MIA 的核心信号。

## 三、MIA 的经典方法

### 1. Loss-Threshold Attack

最简单：训练样本 loss < 阈值 $\tau$，验证样本 loss > $\tau$。

```python
def loss_threshold_attack(model, samples):
    """对每个样本：计算 loss，低于阈值则判为成员。"""
    losses = []
    for x, y in samples:
        with torch.no_grad():
            loss = compute_loss(model(x), y).item()
        losses.append(loss)
    
    # 阈值可以用验证集估计
    threshold = np.percentile(losses, 50)
    return [loss < threshold for loss in losses]
```

**优点**：无需训练攻击模型，简单。
**缺点**：对**预训练数据 vs 微调数据** 区分能力弱（pre-train 时所有数据都"被记忆"过）。

### 2. Likelihood Ratio Attack（Carlini et al. 2022）

更精细：用**两个参考模型**做 likelihood ratio test。

```python
def likelihood_ratio_attack(model, ref_model, sample):
    """
    H0: sample 在训练集 → model 应该 loss 较低
    H1: sample 不在训练集 → model 和 ref_model 应该类似
    """
    loss_model = compute_loss(model, sample)
    loss_ref = compute_loss(ref_model, sample)
    
    # 两者差距
    ratio = loss_ref - loss_model
    return ratio > threshold
```

**优点**：能区分"模型本来就擅长预测"vs"模型记住了该样本"。
**缺点**：需要训练参考模型（成本高）。

### 3. Neighbor Attack（Panda et al. 2024）

针对 LLM 的最新攻击：**比较"原句"与"相似句"的 perplexity**。

```python
def neighbor_attack(model, sample, n_neighbors=10):
    """
    找 sample 的 n 个相似句（mask 一个词）。
    如果 sample 在训练集 → perplexity 显著低于平均邻居。
    """
    p_sample = perplexity(model, sample)
    
    neighbors = [mask_random_word(sample) for _ in range(n_neighbors)]
    p_neighbors = [perplexity(model, n) for n in neighbors]
    
    return p_sample < np.mean(p_neighbors) - threshold
```

**原理**：训练集中"原句"被精确记住，相似但不同的句子 perplexity 更高——差距就是信号。

### 4. Min-K% Prob（Shi et al. 2024）

**核心**：对 sample 中**最不置信的 K% token** 计算平均 log-prob：

```python
def min_k_percent_prob(model, sample, k_percent=20):
    """
    对 sample 的每个 token 计算 log_prob，
    取最低 k% 的平均值。
    """
    tokens = tokenize(sample)
    log_probs = []
    
    for i, token in enumerate(tokens):
        logits = model(sample[:i])
        log_probs.append(F.log_softmax(logits, dim=-1)[token].item())
    
    log_probs.sort()  # 升序
    k = max(1, int(len(log_probs) * k_percent / 100))
    return np.mean(log_probs[:k])  # 取最低 k%
```

**直觉**：训练样本即使是"难"的 token，模型也学到了——所以**最低 log-prob 也较高**；验证样本的难 token 可能完全没学过。

## 四、LLM 上的特殊 MIA：信息泄漏

LLM 的生成能力让"信息泄漏"成为新型 MIA：

### 1. 文本补全攻击

给定 prompt "Alice 的邮箱是 ... "，让模型填空。如果模型补出真实邮箱——说明该邮箱在训练集中。

**经典案例**：
- LLaMA 训练时部分书籍被完整收录，模型能**精确续写**段落。
- ChatGPT 早期能**复述**个人简历内容。
- 一些研究显示，模型能补全用户的**真实 SSN**（社会安全号）。

### 2. PII 提取攻击

Staab et al. (2024) 的系统化研究——让 LLM 自我生成 PII：

```python
def pii_extraction_attack(model, prompt_template, n_samples=1000):
    """
    用"补全式"prompt 让模型生成个人信息。
    """
    extracted = []
    for seed in range(n_samples):
        prompt = prompt_template.format(seed=seed, hint="请补全一个虚构人物的邮箱")
        out = model.generate(prompt, max_length=50, temperature=0.7)
        if is_real_email(out):
            extracted.append(out)
    return extracted
```

**发现**：从 GPT-3.5-turbo 上**成功提取了 1000+ 真实邮箱**。

### 3. 训练数据提取（Training Data Extraction）

Carlini et al. (2021) 开创性工作：从 GPT-2 中**精确提取 600+ 段训练文本**——包括人名、地址、代码片段等。

```text
Prompt: "The company's CEO is"
Output: "John Smith. His phone number is 555-1234. He works at ..."
        ↑ 训练集中某文档的精确复制
```

**结论**：大模型确实会**逐字记忆**部分训练数据，且能被攻击者提取。

## 五、攻击的评测指标

| 指标 | 含义 | 理想值 |
|---|---|---|
| **AUC** | 攻击的 ROC 曲线下面积 | 0.5（随机）|
| **TPR @ low FPR** | 在低假阳率（如 1%）下的真阳率 | 越低越好 |
| **Accuracy** | 攻击分类准确率 | 接近 50%（理想）|
| **Extraction rate** | 能提取的 PII / 真实样本数 | 越少越好 |

"理想模型"应该让 MIA 的 AUC ≈ 0.5（攻击者随机猜）。

## 六、防御 MIA 的方法

### 1. 差分隐私（DP）

**最严格**的防御——训练时加 DP 噪声：

$$
\mathcal{L}_{\text{DP}} = \mathcal{L}_{\text{task}} + \mathcal{N}(0, \sigma^2)
$$

**效果**：DP 训练下 MIA 的 AUC 接近 0.5（基本无效），但**模型精度下降 5~15%**。

### 2. Regularization

不直接 DP，但降低过拟合：

- **Dropout** 增加 → 模型对单样本拟合度降低。
- **Weight decay** → 权重大小受限。
- **Early stopping** → 训练步数受控。
- **Label smoothing** → 避免模型对单样本过拟合。

```python
# 经验配置
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, ...)
for epoch in range(early_stop_at_5):  # 提前停止
    ...
```

### 3. 数据去重

训练前**MinHash 去重**降低模型对单样本的记忆：

```python
import datasketch

def deduplicate(corpus, threshold=0.8):
    """用 MinHash 去重高度相似的文档。"""
    minhashes = []
    for doc in corpus:
        m = datasketch.MinHash()
        for word in doc.split():
            m.update(word.encode())
        minhashes.append(m)
    
    keep = []
    for i, m in minhashes.items():
        if all(m.jaccard(minhashes[j]) < threshold for j in keep):
            keep.append(i)
    return [corpus[i] for i in keep]
```

**效果**：在 Common Crawl 数据集上，deduplication 让 MIA 的 TPR @ 1% FPR 降低 **5~10 倍**。

### 4. Selective Forgetting / Machine Unlearning

训练后发现某条记录被泄漏，**针对性地遗忘**：

```python
def selective_unlearn(model, target_sample, retain_data, alpha=1e-5):
    """
    梯度上升 + 锚定保留数据。
    """
    # 1) 对目标样本做"反向训练"
    target_loss = compute_loss(model, target_sample)
    (-target_loss).backward()  # 最大化 loss
    
    # 2) 锚定保留数据（防止灾难性遗忘）
    for retain_batch in retain_data:
        retain_loss = compute_loss(model, retain_batch)
        retain_loss.backward()
    
    optimizer.step()
```

**挑战**：unlearning 的**理论保证**和**完整性验证**仍是开放问题。

### 5. Output Filtering

训练后用规则或分类器**过滤输出中的 PII**：

```python
def output_filter(text):
    # 1) 正则检测邮箱、电话、SSN
    pii_patterns = [
        r"\b[\w.-]+@[\w.-]+\.\w+\b",                    # email
        r"\b\d{3}-\d{2}-\d{4}\b",                        # SSN
        r"\b1[3-9]\d{9}\b",                              # 中国手机号
    ]
    for pattern in pii_patterns:
        text = re.sub(pattern, "[REDACTED]", text)
    return text
```

**缺点**：攻击者可以**改写**绕过模式匹配（"at 符号 gmail 点 com"）。

### 6. MemFree / Confidentially Redact

Tong et al. (2023) 提出 **MemFree**：训练时把敏感 span mask 掉，让模型**永远不会学**这些信息。

```python
def train_with_masking(model, samples, sensitive_spans):
    """训练时把 sensitive span 替换成 [MASK]，模型学不到这些内容。"""
    for sample in samples:
        for span in sensitive_spans[sample.id]:
            sample.text = sample.text.replace(span, "[MASK]")
    return train(model, samples)
```

## 七、MIA 与其它隐私攻击的对比

| 攻击 | 目标 | 输出 |
|---|---|---|
| **MIA** | 单条记录 | 是/否成员 |
| **属性推断** | 数据集属性 | 数据集偏向哪类 |
| **模型反演** | 训练样本 | 重构图像 / 文本 |
| **梯度泄漏（DLG）** | 联邦学习 | 从梯度恢复样本 |
| **PII 提取** | LLM 输出 | 真实个人数据 |

## 八、LLM 场景的 MIA 难点

### 1. 海量训练数据

LLM 训练集万亿 token，单条记录对整体影响**统计上可忽略**——但**长尾稀有数据**（个人博客、邮件）反而**记忆最强**。

### 2. 数据来源不公开

OpenAI / Anthropic 不公开训练数据组成——攻击者**不知道什么应该被记住**。

### 3. 黑盒访问

仅通过 API 调用——攻击者**看不到模型权重**，只能用 query-response。

### 4. 对话上下文

ChatGPT / Claude 用**对话历史**——攻击者需要把对话完整复现才能精确攻击。

## 九、MIA 的实际案例

### 1. ChatGPT 早期能复述个人简历

用户发现 ChatGPT 能复述**特定人物的精确简历内容**——这些内容很可能被训练过。

### 2. GitHub Copilot 输出原代码

Copilot 有时会**逐字复制**训练集中的 GPL 代码——引发法律纠纷。

### 3. 医疗记录的 MIA

研究显示，在医疗数据上微调的 LLM 对**特定患者的诊断**有显著低 loss——MIA 可识别。

### 4. 学校申请文书

某些大学用 AI 检测学生申请文书——**反向**地，如果模型能识别"这是训练过的文书"，说明模型见过类似样本。

## 十、给 LLM 厂商的建议

1. **训练前去重**：MinHash / SimHash 在 Common Crawl 级别去重。
2. **DP 微调**：对客户数据用 DP-SGD。
3. **PII 检测与过滤**：训练数据预处理。
4. **持续 MIA 评估**：定期用 Neighbor / Min-K% Prob 测 MIA。
5. **输出过滤**：模型输出过 PII 检测器。
6. **透明披露**：model card 中说明已知隐私风险。
7. **用户删除权**：让用户能"删除自己的数据"——配合 selective unlearning。

## 十一、MIA 的根本困境

**记忆与泛化的矛盾**：模型需要"记住"训练数据才能学到模式，但**过度记忆**就会泄漏个体。

- 训练 1 epoch → 记忆弱 → 泛化弱。
- 训练 10 epoch → 记忆强 → 泄漏多。

**唯一的根本解决**：**形式化隐私保证**（DP）。其它方法都是"降低泄漏"而非"消除泄漏"。

## 小结

成员推断攻击是 LLM 时代最现实的隐私威胁——Loss-Threshold、Likelihood Ratio、Neighbor Attack、Min-K% Prob 等方法各有侧重；PII 提取、训练数据提取更是直接威胁个人。**防御的根本路径是差分隐私**，但代价是精度下降。**去重、regularization、输出过滤**是"缓解"而非"解决"。今天 LLM 厂商大多**训练前去重 + 输出过滤 + 持续评估**，但**真正的可证明隐私仍需要 DP**。下一篇我们进入 AI 安全的最前沿话题——**Red Teaming**：主动攻击模型，发现并修补漏洞。
