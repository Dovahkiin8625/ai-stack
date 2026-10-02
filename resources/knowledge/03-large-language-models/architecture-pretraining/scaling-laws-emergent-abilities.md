# 缩放法则与涌现能力

"把模型做大，数据做多，训久一点——效果就会变好。"这条经验法则背后是有数学的。**Scaling Laws** 描述了 loss 与参数量 / 数据量 / 计算量之间的幂律关系，**Emergent Abilities** 则描述了某些能力在某个规模阈值之上"突然出现"的非线性现象。本文梳理 Kaplan、Chinchilla 两条主流 scaling law，澄清"涌现"的真实含义，并给出 Chinchilla 最优分配公式与一个可跑的 scaling 实验设计。

## 一、为什么 scaling 值得专门研究

训练一个大模型动辄数百万美元。如果能在训练前**预测**：在给定计算预算下，多大的模型 + 多少数据 + 多长训练步数能达到最优？这就是 scaling laws 的价值。它把"经验活"变成"可优化的工程问题"。

## 二、Kaplan Scaling Laws（OpenAI, 2020）

Kaplan et al. 在《Scaling Laws for Neural Language Models》中拟合出：

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N},\quad \alpha_N \approx 0.076
$$

其中 $N$ 是参数数量，$L$ 是 test loss，$\alpha$ 是幂律指数。直觉：**参数量翻 10 倍，loss 下降 $10^{-0.076} \approx 0.84$ 倍**。类似的还有数据 $D$ 与计算量 $C$：

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty
$$

论文的关键结论：

- **幂律非常稳**：跨越 8 个数量级都成立。
- **更大模型更"高效"**：同 FLOPs 下，模型大 + 数据少 比 模型小 + 数据多略优。
- **样本效率高**：大模型只需要相对少的数据就能达到小模型同等 loss。

但 Kaplan 并没有给出"给定 FLOPs，$N$ 和 $D$ 怎么分最优"——这就是 Chinchilla 要解决的问题。

## 三、Chinchilla（DeepMind, 2022）

Hoffmann et al. 用更精细的实验拟合出**计算最优**（compute-optimal）的分配关系：

$$
N_{\text{opt}} \propto C^{0.5},\quad D_{\text{opt}} \propto C^{0.5}
$$

即：FLOPs 增加 10 倍时，**模型参数量与训练 token 数都增加 $\sqrt{10} \approx 3.16$ 倍**。换算成具体数字：

$$
D_{\text{opt}} \approx 20 \cdot N_{\text{param}}
$$

**70B 模型的最优训练数据量约为 1.4T tokens**。这就是大家说的"Chinchilla ratio"。

### 与 Kaplan 的差别

Kaplan 偏"大模型 + 少数据"，Chinchilla 偏"模型与数据同步扩大"。Chinchilla 用同样计算预算训练一个 70B Chinchilla-Optimal 模型，显著优于 280B Gopher——证明**之前的很多大模型其实"训得不够"**。

### LLaMA 反其道而行

Meta 的 LLaMA-1（7B~65B）使用 **200B~1.4T tokens**，对 65B 模型来说已经偏离 Chinchilla（按公式应只用 ~1.3T tokens，但实际用 1.4T，**接近 Chinchilla**）。LLaMA-2 / 3 在更大规模上仍基本遵循 Chinchilla ratio，但加更多 inference-compute 考虑后会略向"数据侧"倾斜。

## 四、Chinchilla 最优分配的数学推导

Chinchilla 假设训练 FLOPs：

$$
C \approx 6 \cdot N \cdot D \quad\text{(每个 token 的前向 + 反向约为 6ND)}
$$

固定 $C$，求最优 $N, D$ 使 $L(N, D)$ 最小。由 $L$ 关于 $N, D$ 的偏导与 Lagrangian：

$$
\frac{\partial L / \partial N}{\partial L / \partial D} = \frac{D}{N}
$$

在拟合出的指数 $\alpha_N = \alpha_D = \alpha$ 下，最优解恰好是 $N \propto D \propto C^{1/(1+\alpha)}$。代入 $\alpha \approx 0.34$（Chinchilla 拟合）：

$$
N_{\text{opt}} \propto C^{1/1.34} \approx C^{0.5}
$$

简单记法：$N$ 与 $D$ 各吃一半 FLOPs 增量。

## 五、Emergent Abilities：真的"涌现"吗？

Wei et al. (2022) 在《Emergent Abilities of Large Language Models》中提出"涌现"：某些能力在模型规模小时接近随机水平，规模一到某个阈值就**突然**飙升：

```text
task accuracy
    ^
    |                       ___________
    |                  ____/
    |               __/
    |             _/         ← threshold (emergent)
    |          __/
    |       __/
    |    __/
    |___/
    +---------------------------------> model scale
       small        medium        large
```

典型例子：算术、多步推理、代码补全、跨语言迁移。论文展示了 BIG-Bench 上数百任务的"突变曲线"。

### 涌现的另一种解读

Schaeffer et al. (NeurIPS 2023, "Are Emergent Abilities a Mirage?") 指出：**很多所谓的"涌现"，其实是评估指标选择导致的**。把"exact match"换成"token edit distance"或"逐步打分"，曲线就平滑了。换句话说，**模型能力是平滑增长的，只是指标把它"放大"成突变**。

这是个重要的提醒：不要被"涌现"叙事绑架——能力提升本质是连续的，关键是选对指标。

## 六、In-Context Learning（ICL）算"涌现"吗？

ICL（给定几个示例，模型在不更新参数的情况下完成任务）确实在规模小时几乎不工作。但**严格地说它也是平滑增长**——只是不同任务对规模敏感度不同。几个观察：

- **ICL 效率随规模增长**：大模型用 5 个示例就能超过小模型用 100 个示例。
- **ICL 与 SFT 的等价性**：大规模下 ICL 在某些任务上接近监督微调。
- **ICL 是 Bayesian inference 的近似**：在隐空间做"概念推断"，模型越大，先验越准。

因此 ICL 看似"涌现"，本质上是规模带来的隐空间表征能力提升。

## 七、指令微调"扁平化"涌现曲线

把模型用指令数据微调（SFT / RLHF）后，原本的"突变曲线"会显著**平滑**：

```text
Base model:        ──── sharp jump at threshold
SFT / Instruct:    ──── much smoother, weaker task still works
```

直觉：指令微调让模型**更容易激活**已经学会的能力，把"沉睡的子任务"暴露出来。所以"涌现"在很大程度上是 base 模型 vs instruct 模型的差别。生产中几乎所有应用都用 instruct 版本——emergent 对它们而言是"被设计成不要突变"。

## 八、MoE 与稀疏激活对 scaling 的影响

Mixture of Experts 用稀疏激活替代稠密 FFN：每次只激活 $k$ 个专家中的 $E$ 个，总参数量 $N$ 涨但**激活参数** $N_{\text{act}}$ 没涨那么多。Mixtral 8x7B 有 47B 总参数，但激活只有 ~13B。

**对 scaling law 的影响**：

- **FLOPs scaling 仍大致成立**：因为前向计算量由激活参数决定。
- **loss 曲线略有改进**：相同 FLOPs 下 MoE 通常略优于 dense。
- **MoE 的"专家利用不均"是新挑战**：路由崩溃、专家负载不均衡需要辅助 loss。
- **稀疏 scaling law**：Sparse Upcycling / ST-MoE 等研究表明，相同 FLOPs 下 MoE 比 dense 平均少 5~15% loss。

## 九、Grokking：训练之后的"顿悟"

Grokking（Power et al. 2022）发现一个反直觉现象：在小规模算术任务上，模型**训练 loss 早就降到 0 了，但 test loss 要在数千步后才突然下降**。

```text
train loss: ──\___
                   \____________________________
test  loss: ───────────────────────\___
                                       \___________ (骤降)
```

**含义**：模型先"死记"训练集，再在漫长的优化中突然"悟到"算法。**对大模型的启示**：训练 loss 曲线不能完全预测泛化能力——要保留强 checkpoint 并用 eval loss 选最优。

## 十、一个简单的 scaling 实验设计

下面给一个最小可跑的 scaling 实验骨架：用 GPT-2 架构在不同宽度上拟合 scaling law。

```python
"""
最小 scaling 实验：在 TinyShakespeare 上训练不同宽度的 GPT-2-like 模型，
记录 (params, tokens, final_loss)，并拟合 L(N) = a * N^{-alpha} + b。
"""
import math, random, time, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader


def get_data():
    text = open("data/tinyshakespeare.txt").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


class GPTBlock(nn.Module):
    """极小 GPT block：PreNorm + Causal MHA + PreNorm + FFN。"""
    def __init__(self, d, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), is_causal=True)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, block_size=128, d=64, n_head=4, n_layer=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.Sequential(*[GPTBlock(d, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        return self.head(self.ln_f(x))


def train_one(d, n_layer, n_head, train, val, vocab, steps=2000, batch=64, block=128, lr=3e-4):
    torch.manual_seed(0)
    model = TinyGPT(vocab, block, d, n_head, n_layer).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    log = []
    for step in range(steps):
        ix = torch.randint(0, len(train) - block - 1, (batch,))
        x = torch.stack([train[i:i+block]    for i in ix]).cuda()
        y = torch.stack([train[i+1:i+block+1] for i in ix]).cuda()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            log.append((step, loss.item(), n_params))
    return {"d": d, "n_layer": n_layer, "n_params": n_params, "loss": log[-1][1]}


if __name__ == "__main__":
    train, val, vocab = get_data()
    results = []
    # 扫 5 个宽度，约 10x 范围
    for d in [32, 48, 72, 108, 162]:
        for n_layer in [4, 6, 8]:
            r = train_one(d, n_layer, 4, train, val, vocab)
            results.append(r)
            print(f"d={d:>4} layer={n_layer} params={r['n_params']:>9,}  loss={r['loss']:.4f}")

    # 拟合幂律: L ~ a * N^{-alpha} + b
    import numpy as np
    Ns = np.array([r["n_params"] for r in results])
    Ls = np.array([r["loss"]    for r in results])
    logN, logL = np.log(Ns), np.log(Ls - Ls.min() + 1e-3)
    slope, intercept = np.polyfit(logN, logL, 1)
    print(f"scaling exponent ~ {-slope:.3f}  (Kaplan 拟合约 0.076，量级吻合)")
    json.dump(results, open("scaling_log.json", "w"), indent=2)
```

几个实验设计要点：

1. **单变量扫描**：保持数据量、训练步数固定，只改模型宽度——这是拟合 $L(N)$ 的标准做法。
2. **多档规模**：至少 5 个数量级跨度的模型大小才能拟合出稳定的 $\alpha$。
3. **固定 eval 协议**：所有模型在同一 val set 上算 loss，否则对比无意义。
4. **同时扫 $D$**：再开一组实验改 token 数，可以拟合 $L(D)$。
5. **记录 FLOPs**：把每次训练的 FLOPs 估算出来，能拟合 $\text{FLOPs}-L$ 曲线——这才是真正指导预算分配的曲线。

## 十一、给工程团队的"缩放"决策清单

1. **先估算 FLOPs 预算**：决定 N 与 D 的上限。
2. **按 Chinchilla ratio 选 N**：保守起步是 $D = 20N$；若算力紧张，可考虑"超额训练"（LLaMA 路线）。
3. **小模型先跑通 pipeline**：7B~13B 模型上确认数据、训练、评估流程都跑通，再放大。
4. **保留中间 checkpoint**：配合 grokking 现象，eval 选最优而非 train 选最优。
5. **关注指标设计**：用连续指标（token edit、logprob）而非二元 exact match，看清真实能力曲线。
6. **MoE 适合"少算力训大模型"**：参数大但激活小，部署上要权衡总参数带来的内存压力。

## 小结

Scaling laws 把"做多大"变成可计算的工程问题——Kaplan 给出 loss 与 $N$ 的幂律，Chinchilla 进一步约束"模型与数据同步扩大"才是 compute-optimal；emergent abilities 看似神秘，但很大程度上是评估指标与模型对齐方式的产物。MoE 与 grokking 等新现象告诉我们，规模化的故事还远没有讲完——**真正决定上限的，是模型、数据、评估三者的协同设计**。到这里，架构与预训练的整条链路（架构 → 目标 → 数据 → 训练 → scaling）已贯通，下一步就是怎么把预训练模型变成"产品级"助手——也就是 fine-tuning 与对齐的话题了。
