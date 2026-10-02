# 差分隐私与 DP-SGD：让训练数据"被遗忘"

LLM 在海量文本上预训练，其中不可避免地包含个人隐私信息（姓名、邮箱、电话、医疗记录）。**差分隐私（Differential Privacy, DP）** 是当前学术界与工业界公认的"金标准"隐私定义——它提供**可证明的隐私保证**，无论攻击者有多少背景知识，都无法从模型输出推断特定个体是否在训练集中。本文介绍 DP 的数学定义、DP-SGD 训练算法、隐私预算 $\epsilon$ 的解读，以及 LLM 场景下的实践挑战。

## 一、为什么需要严格的隐私定义

直观的隐私保护方法（脱敏、k-匿名、删除明显 PII）都**不够**：

- **去标识化（De-identification）**：Netflix Prize 数据集把姓名删除，但研究者通过合并 IMDb 评分**重新识别了用户**（Narayanan & Shmatikov 2008）。
- **k-匿名**：$k$ 个不可区分个体，但**准标识符组合**仍能缩小范围（"35 岁男性北京海淀程序员" = 1 人）。
- **删除训练数据**：模型权重中**残留的梯度信息**仍可被攻击者提取（Singer & Mittal 2021）。

差分隐私提供**不依赖攻击者能力的**、**数学可证明的**保证——这是它的核心优势。

## 二、差分隐私的定义

一个随机算法 $\mathcal{M}$ 满足 $(\epsilon, \delta)$-差分隐私，如果对**任意相邻数据集** $D, D'$（$D'$ = $D$ 删去一条记录），对**任意输出集合** $S$：

$$
\Pr[\mathcal{M}(D) \in S] \leq e^{\epsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta
$$

直觉：

- $\epsilon$ 越小 → 隐私保护越强 → 加噪越多 → 模型精度越差。
- $\delta$ 是"小概率失败"项（典型 $\delta \leq 1/N^2$，$N$ 是数据集大小）。

### 高斯机制的 DP

对**实值函数** $f: \mathcal{D} \to \mathbb{R}^k$，加高斯噪声：

$$
\mathcal{M}(D) = f(D) + \mathcal{N}(0, \sigma^2 I)
$$

满足 $(\epsilon, \delta)$-DP 当：

$$
\sigma \geq \frac{\sqrt{2 \ln(1.25 / \delta)} \cdot \text{sensitivity}(f)}{\epsilon}
$$

其中 **sensitivity** 是 $f(D)$ 与 $f(D')$ 的最大差距：

$$
\text{sensitivity}(f) = \max_{D, D'} \|f(D) - f(D')\|_2
$$

### Rényi 差分隐私（RDP）

更精细的分析用 **Rényi divergence**：

$$
D_\alpha(\mathcal{M}(D) \| \mathcal{M}(D')) \leq \epsilon(\alpha)
$$

然后转回标准 $(\epsilon, \delta)$-DP。这是 DP-SGD 的标准分析框架。

## 三、DP-SGD：差分隐私随机梯度下降

Abadi et al. (2016) 的 DP-SGD 是 LLM 训练中事实上的标准：

```python
def dp_sgd_step(model, batch, loss_fn, optimizer, clip_norm=1.0, noise_multiplier=1.1):
    """
    单个 DP-SGD step：
    1) per-sample 梯度
    2) 梯度裁剪到 norm ≤ clip_norm
    3) 加高斯噪声
    4) 平均后反传
    """
    batch_loss = 0
    grads = []
    
    # 1) 计算每个样本的梯度
    for x, y in batch:
        model.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters()})
    
    # 2) 裁剪每个样本的梯度
    for g in grads:
        total_norm = torch.sqrt(sum(t.norm()**2 for t in g.values()))
        clip_coef = min(1.0, clip_norm / (total_norm + 1e-8))
        for t in g.values():
            t.mul_(clip_coef)
    
    # 3) 平均 + 加噪
    for n, p in model.named_parameters():
        avg_grad = sum(g[n] for g in grads) / len(grads)
        noise = torch.randn_like(avg_grad) * noise_multiplier * clip_norm
        p.grad = avg_grad + noise
    
    # 4) 优化器 step
    optimizer.step()
```

**关键三步**：
1. **Per-sample 梯度**：每个样本独立算梯度（不能 mini-batch 平均）。
2. **梯度裁剪**：限制每个样本的影响（sensitivity bound）。
3. **加高斯噪声**：使最终梯度满足 DP。

## 四、隐私预算 $\epsilon$ 的解读

$\epsilon$ 控制**隐私-精度权衡**：

| $\epsilon$ | 隐私强度 | 模型精度 |
|---|---|---|
| 0.1 | 极强 | 几乎随机 |
| 1.0 | 强 | 明显下降 |
| 3.0 | 中等 | 略下降 |
| 10.0 | 弱 | 几乎无影响 |
| $\infty$ | 无保证 | 同非 DP |

**经验法则**：

- $\epsilon \leq 1$：强隐私，**敏感数据**（医疗、政府）。
- $\epsilon \in [1, 10]$：中等隐私，**平衡场景**（推荐系统）。
- $\epsilon \geq 10$：弱隐私，仅作**防御性深度保护**。

**DP 的组合定理**：多次访问数据累加 $\epsilon$。例如 100 epoch 训练，**每 epoch 必须贡献约 $\epsilon / 100$ 的隐私**——这要求极小噪声，精度很差。**隐私会计（Privacy Accounting）** 精确追踪累计 $\epsilon$。

## 五、隐私会计：精确追踪累计 $\epsilon$

Naive 累计 $\epsilon_{\text{total}} = T \cdot \epsilon_{\text{per-step}}$ 太悲观。RDP-based accountant（Wang et al. 2019）能给出**紧致上界**：

```python
from opacus.accountants import RDPAccountant

def train_with_accountant(model, dataloader, target_epsilon=3.0, target_delta=1e-5):
    """
    用 Opacus 的 RDP accountant 自动停止。
    """
    accountant = RDPAccountant()
    
    for epoch in range(max_epochs):
        for batch in dataloader:
            # DP-SGD step
            ...
            
            # 记录这一步消耗的隐私
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
            
            # 检查是否达到目标
            eps = accountant.get_epsilon(delta=target_delta)
            if eps > target_epsilon:
                print(f"Reached target epsilon {eps:.2f} > {target_epsilon}")
                return model
```

**Opacus**（Meta）与 **TensorFlow Privacy**（Google）提供开箱即用的 accountant 与 DP-SGD 实现。

## 六、LLM 场景下的 DP 挑战

### 1. 巨大的隐私预算消耗

GPT-3 训练消耗的隐私预算是 $\epsilon = \infty$（无 DP）。如果加 DP，要达到同等精度，可能需要 $\epsilon \geq 100$——这接近"无保护"。

**现实选择**：**在小数据集上微调**时用 DP——例如 LLaMA-3 + 医疗数据 $\epsilon = 3$，比从零预训练 DP 实用得多。

### 2. 长上下文与变长序列

DP-SGD 需要**逐样本梯度**。长序列（如 8K context）让计算量暴涨。**梯度检查点 + micro-batch** 是常见优化。

### 3. 词表大小的敏感度

Embedding 层对每个 token 都敏感——sensitivity 极高。**Embedding 私有化**：

```python
# 方案 1：只训最后一层 + LoRA，其余冻结
# 方案 2：对 embedding 共享一些"sensitive direction"加更大噪声
```

### 4. 成员推断 vs DP

即使模型用 DP 训练，仍然可能受成员推断攻击——**DP 降低攻击成功率但不归零**。生产中通常**叠加多道防线**：

- DP-SGD 训练
- 输出层加噪（prediction perturbation）
- 后处理过滤（识别并拒答含 PII 的 prompt）

## 七、PATE：私有聚合教师集成

Papernot et al. (2017) 的 **PATE（Private Aggregation of Teacher Ensembles）** 是 DP 的另一思路：

1. 把数据分成 N 份，每份训一个**教师模型**。
2. 学生模型训练时，聚合 N 个教师的预测，但**加噪**：

$$
\text{label} = \arg\max_c \left( \#\{\text{teachers predict } c\} + \text{Laplace noise} \right)
$$

3. 教师模型间的"投票一致性"过滤掉个别教师记忆的特定样本。

**优点**：敏感数据只接触教师；学生模型训练可非 DP。
**缺点**：需要训 N 个教师，成本高。

## 八、差分隐私推断（DP Inference）

训练阶段加 DP 太贵，更轻量的做法是**推理阶段加 DP**：

```python
def dp_text_generation(model, prompt, noise_std=0.1):
    """
    在生成时对 logits 加噪，让输出有 DP 性质。
    """
    logits = model(prompt)
    noisy_logits = logits + torch.randn_like(logits) * noise_std
    return sample(noisy_logits)
```

但 $\epsilon$ 极小（仅对单次推理保护），累积多次推理后 $\epsilon$ 增长——需要**组合定理**分析。

## 九、DP 与其他隐私机制的对比

| 机制 | 强度 | 可证明 | 计算开销 |
|---|---|---|---|
| 差分隐私（DP） | 可调 | **强（数学保证）** | 中-高 |
| k-匿名 | 弱 | 中 | 低 |
| 同态加密 | 强 | 强 | **极高** |
| 联邦学习 | 中 | 弱（仅不传数据） | 中 |
| 安全多方计算 | 强 | 强 | 高 |

DP 的优势在于**数学可证明 + 实用性平衡**——是当前工业部署的主流。

## 十、LLM-DP 实践案例

### 1. Google Gboard

Gboard 在多国用户数据上训**下一词预测模型**，使用 **DP-FedAvg**（联邦 + DP），$\epsilon \approx 8$。

### 2. Apple QuickType

Apple 在用户输入数据上微调，**每用户本地 + 差分隐私**聚合。

### 3. OpenAI 的"私有微调"

OpenAI 提供 DP 微调 API（基于 PATE），声称对金融、医疗数据有强保护。

### 4. 学术工作

- **DP-Forward**：仅在最后一层加 DP。
- **Ghostwriter**：联邦 + DP 的 LLM 微调。
- **PRV（Privacy-Regularized Vicinal）**：用 vicinal risk minimization 减少对单样本的过拟合。

## 十一、DP 的常见误解

### 1. "$\epsilon = 1$ 就够安全"

不对。$\epsilon = 1$ 只保证**单次查询**安全。**多次查询**累加——需要做组合分析。

### 2. "DP 后模型无害"

不对。DP 只保证**单条训练记录不可区分**，但仍可能**生成有害内容**——这是模型能力问题，不是隐私问题。

### 3. "DP 等于删除数据"

不对。**Dwork & Rothblum 提案**（Concentrated DP）：DP 模型仍然记住部分聚合信息。"删除"在 DP 中是**渐近意义**——不是单条记录删除即可。

### 4. "$\delta$ 越小越好"

$\delta$ 必须远小于 $1/N$（$N$ 是数据集大小）才有意义。$\delta = 10^{-5}$ 对 100 万条数据集合理，对 100 条数据集过大。

## 十二、给工程实践者的清单

1. **选好 $(\epsilon, \delta)$**：先确认合规要求（GDPR / HIPAA），定 $\delta \leq 1/N^2$。
2. **用 RDP accountant**：精确追踪累计 $\epsilon$。
3. **梯度裁剪是关键**：clip_norm 选 1.0（标准）或 0.5（更保守）。
4. **噪声乘数自适应**：训练初期噪声大（更 DP），后期可微降。
5. **小数据用 DP**：在微调阶段而非预训练阶段用 DP。
6. **Opacus / TF Privacy**：开源工具开箱即用。
7. **监控成员推断**：即使 DP 训练，仍要测 MIA 攻击成功率。

## 小结

差分隐私提供**数学可证明**的隐私保护——DP-SGD 是 LLM 训练中的事实标准。它的代价是**精度下降 + 计算开销 + 训练不稳定**，但在敏感数据场景下（医疗、金融、个人数据）几乎是唯一选择。**$\epsilon$ 的选择是政治-技术双重决策**：太小保护强但模型差，太大模型好但隐私弱。下一篇我们将看到另一种隐私路线——**联邦学习**：不把数据集中到中心，让模型"上门学习"。
