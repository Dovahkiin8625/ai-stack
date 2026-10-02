# 联邦学习：数据不动模型动

GDPR、HIPAA 等法规让"集中数据训练"变得困难——医疗记录、财务数据、个人对话不能离开本地。**联邦学习（Federated Learning, FL）** 让模型"上门学习"：每个设备/机构本地训练，把**梯度**而非**数据**上传中心聚合。McMahan et al. (2017) 的 FedAvg 是奠基性算法，FedProx、SCAFFOLD 等改进解决客户端漂移问题。本文介绍 FL 的核心算法、隐私分析（FL ≠ DP）、LLM 场景下的挑战，以及与 DP、HE（同态加密）的组合。

## 一、为什么需要联邦学习

集中训练的根本问题：

1. **法规限制**：GDPR（欧盟）、CCPA（加州）、PIPL（中国）都限制个人数据出域。
2. **数据敏感**：医疗记录、银行流水、个人邮件不能集中。
3. **带宽成本**：手机每天产生 GB 级数据，传到中心不现实。
4. **实时性**：输入法、推荐系统需要本地个性化。

联邦学习的核心想法：**数据留在原处，模型参数来去**。

## 二、FedAvg：联邦平均

McMahan et al. (2017) 的 FedAvg 算法：

```text
┌──────────────────────────────────────────────────────────┐
│  Server（中心）                                            │
│  - 初始化全局模型 w_0                                      │
│  - 每轮：                                                  │
│    1. 选 K 个客户端（占总客户端的 C 比例）                   │
│    2. 把当前 w_t 发给选中的客户端                            │
│    3. 等客户端返回更新                                      │
│    4. 加权平均：w_{t+1} = (1/K) Σ w_t^k                   │
└──────────────────────────────────────────────────────────┘
                            ↕  发送模型 / 返回更新
┌──────────────────────────────────────────────────────────┐
│  Client k（本地）                                          │
│  - 接收 w_t                                               │
│  - 用本地数据训练 E 个 epoch：w_t^k = LocalSGD(w_t, D_k)   │
│  - 上传 w_t^k - w_t（增量）给服务器                         │
└──────────────────────────────────────────────────────────┘
```

**关键思想**：
- **FedSGD**：每客户端只跑 1 个 mini-batch → 更新 = 梯度本身。
- **FedAvg**：每客户端跑 E 个 epoch → 减少通信轮次（核心优势）。

## 三、FedAvg 的 PyTorch 实现

```python
import torch
import copy

class FedAvgServer:
    def __init__(self, model, n_clients, sample_ratio=0.1):
        self.global_model = model
        self.n_clients = n_clients
        self.sample_ratio = sample_ratio
    
    def select_clients(self):
        k = max(1, int(self.n_clients * self.sample_ratio))
        return np.random.choice(self.n_clients, k, replace=False)
    
    def aggregate(self, client_states):
        """加权平均各客户端的模型参数。"""
        avg_state = copy.deepcopy(self.global_model.state_dict())
        for key in avg_state:
            avg_state[key] = torch.stack([
                state[key].float() for state in client_states
            ]).mean(dim=0)
        self.global_model.load_state_dict(avg_state)


class FedAvgClient:
    def __init__(self, model, data_loader, lr=1e-3, local_epochs=5):
        self.model = copy.deepcopy(model)
        self.data_loader = data_loader
        self.lr = lr
        self.local_epochs = local_epochs
    
    def train(self):
        """本地训练 E 个 epoch，返回新参数。"""
        optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        for _ in range(self.local_epochs):
            for x, y in self.data_loader:
                loss = torch.nn.functional.cross_entropy(self.model(x), y)
                optim.zero_grad(); loss.backward(); optim.step()
        return self.model.state_dict()


def federated_train(server, clients, n_rounds=100):
    for round_idx in range(n_rounds):
        selected = server.select_clients()
        client_states = []
        for cid in selected:
            # 客户端接收全局模型 → 本地训练 → 返回
            clients[cid].model.load_state_dict(server.global_model.state_dict())
            new_state = clients[cid].train()
            client_states.append(new_state)
        server.aggregate(client_states)
        if round_idx % 10 == 0:
            print(f"Round {round_idx} complete")
```

## 四、FL 的核心挑战

### 1. Non-IID 数据

每个客户端的数据分布不同（**Non-IID**）——有的客户端用英文，有的用中文；有的喜欢猫视频，有的喜欢体育。这是 FL 最大的难题，导致**客户端漂移（client drift）**：

```python
# 问题示意：客户端 A 一直在学英语，模型偏向英语
# 客户端 B 一直在学中文，模型偏向中文
# 平均后两边都学不好
```

**解决方案**：
- **FedProx**（Li et al. 2020）：加近端项 $\frac{\mu}{2} \|w - w_t\|^2$，防止本地偏离太远。
- **SCAFFOLD**（Karimireddy et al. 2020）：用**控制变量**修正客户端漂移。
- **FedNova**（Wang et al. 2020）：归一化不同客户端的更新步数。

```python
def fedprox_loss(local_model, global_model, mu=0.01):
    """FedProx 损失：原任务损失 + 近端正则。"""
    task_loss = compute_loss(local_model, batch)
    prox_loss = sum(
        ((p - g_p) ** 2).sum() 
        for p, g_p in zip(local_model.parameters(), global_model.parameters())
    )
    return task_loss + (mu / 2) * prox_loss
```

### 2. 通信成本

LLM 上 GB 级模型每轮传一遍——通信是瓶颈。

**解决方案**：
- **梯度压缩**：稀疏化（Sparse SGD）、量化（1-bit SGD）、低秩分解（LoRA）。
- **客户端选择**：每轮只选部分客户端（**重要性采样**）。
- **异步 FL**：不等所有客户端，慢客户端不拖累。
- **模型分割**：把模型切成 client-side + server-side，client 端只算小部分。

### 3. 系统异构性

不同客户端算力不同——可能手机、嵌入式设备、服务器并存。**Stragglers**（慢客户端）拖慢整体训练。

**解决方案**：
- **客户端能力建模**：根据历史时间分配数据量。
- **早停机制**：客户端跑 E epoch 或超时即停。
- **层级 FL**：边缘服务器先聚合，再上传到中心。

### 4. 隐私推断攻击

**重要：FL ≠ 隐私保护**。

梯度本身可能泄漏训练数据：
- **DLG（Deep Leakage from Gradients, Zhu et al. 2019）**：从梯度反演出原始图像。
- **iDLG**：更高效的数据重建。
- **Membership Inference**：从梯度推断某条记录是否在客户端。

**解决方案**：
- **Secure Aggregation**（Bonawitz et al. 2017）：服务器只能看到聚合后的梯度，看不到单个客户端。
- **差分隐私**：上传梯度前加噪。
- **同态加密**：梯度加密后再上传，服务器在密文上做聚合。

## 五、Secure Aggregation

Bonawitz et al. (2017) 的 Secure Aggregation 让服务器**只看到 K 个客户端的平均梯度**，看不到单个：

```text
Client i 上传：E_i(grad_i + r_i + Σ_secret_share_from_j(r_j))
                                          ↑ 与其它客户端的秘密分享
其中 r_i 是 Client i 的随机掩码，
通过 Shamir 秘密分享让 Σ r_i = 0（聚合时抵消）。

服务器看到的：Σ grad_i + Σ r_i = Σ grad_i
```

服务器能恢复聚合，但**单客户端梯度被掩码保护**。

## 六、DP-FL：差分隐私 + 联邦学习

FL 提供"数据不出本地"的**输入隐私**，但**梯度**仍可能泄漏信息。在梯度上加噪 → DP-FL：

```python
def dp_fl_client(model, global_state, clip_norm=1.0, noise_multiplier=1.1):
    """DP-FL 客户端：本地训练 + 梯度裁剪 + 加噪。"""
    model.load_state_dict(global_state)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for x, y in dataloader:
        optim.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        
        # 1) 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        # 2) 加噪
        for p in model.parameters():
            p.grad += torch.randn_like(p.grad) * noise_multiplier * clip_norm
        
        optim.step()
    
    return model.state_dict()
```

**注意**：服务器能看到的**聚合梯度**有更强隐私——K 个客户端的噪声叠加，**总噪声方差为 $\sigma^2 / K$**，比单客户端小 K 倍。

**Google Gboard** 用的是 DP-FL + Secure Aggregation，$\epsilon \approx 8$ 对用户隐私有数学保证。

## 七、LLM 上的联邦学习挑战

### 1. 模型规模

7B 参数的 LLM 每轮传 14GB（FP16）。**不可能每轮全传**。

**解决方案**：
- **FedLLM（FedAdapter）**：客户端只训 LoRA / Adapter，几 MB 量级。
- **LoRA 聚合**：服务器只聚合 LoRA 参数。
- **梯度稀疏化**：仅传 top-K 梯度。

### 2. 训练-服务分离

生产 LLM 服务中，**推理在客户端、训练在中心**。FL 可让**推理模型个性化**——客户端用本地数据微调 LoRA，定期上传到中心聚合。

```text
中央模型（base）         客户端个性化
     │                       │
     └──── 共享 base ────────┘
              │
              └── 上传本地 LoRA
                    聚合到中心
                    下发新的聚合 LoRA
```

这就是 **Federated LoRA / FedIT** 等工作。

### 3. 客户端数据规模小

每个客户端可能只有几十到几百条本地对话，远小于 LLM 训练需求。**Few-shot + FL**：每个客户端只有少量样本，但仍能从 FL 聚合中受益。

### 4. 通信轮次 vs 性能

LLM 微调通常需要数千步；FL 每轮只有少量更新。**通信-性能权衡**比传统 FL 更突出。

## 八、联邦学习的工业实践

| 项目 | 场景 | 技术 |
|---|---|---|
| **Google Gboard** | 移动输入法 | DP-FL + Secure Aggregation |
| **Apple QuickType** | iOS 键盘 | DP + 联邦 |
| **Meta Ads** | 广告排序 | FedAvg + 差分隐私 |
| **NVIDIA Clara** | 医疗影像 | 联邦 + 同态加密 |
| **IBM FL** | 跨医院医疗 | 联邦学习 + 可信执行环境 |
| **OpenFL**（Intel） | 通用 FL 框架 | 多策略支持 |
| **Flower**（学术） | 通用 FL 框架 | 易扩展 |

## 九、FL 的隐私-性能权衡

| 方案 | 隐私 | 性能 | 计算开销 |
|---|---|---|---|
| 集中训练 | 无 | 最好 | 低 |
| FedAvg | 数据不出本地 | 中（Non-IID 时差） | 中 |
| FedAvg + SecureAgg | 单客户端梯度不可见 | 中 | 中 |
| DP-FL | **数学可证明** | 中-差（噪声影响） | 中 |
| 同态加密 + FL | 强 | 中-差 | **高** |
| TEE（可信执行环境） | 强 | 中 | 中 |

**典型选择**：
- 大众消费应用（Gboard, QuickType）：**DP-FL + SecureAgg**。
- 跨机构（医院、银行）：**HE + FL** 或 **TEE + FL**。
- 高敏感（国防、生物）：**TEE + DP-FL** 多重保险。

## 十、FL 与 LLM 的"前沿"组合

### 1. FedLLM / FedIT

客户端用本地数据 LoRA 微调，中心聚合 LoRA。论文显示在 100~1000 客户端规模下，能达到集中训练的 90~95% 性能。

### 2. Personalized FL

每个客户端不仅共享全局知识，还保留**个性化参数**：

```python
def personalized_local(global_model, personal_layers, client_data):
    """全局模型 + 个性化 head。"""
    global_features = global_model.encoder(client_data)
    logits = personal_layers(global_features)
    return logits
```

### 3. Split Learning（拆分学习）

把模型切成 client-side + server-side，**中间激活**而非梯度传给服务器：

```text
Client:                Server:
Input                  ↑
  ↓                     │
Client_Encoder(x) → 中间激活 → Server_Decoder(h) → Loss
```

隐私比 FL 更强（不暴露梯度），但**中间激活仍可能泄漏**。

### 4. FFA-LoRA（联邦全参数聚合）

每个客户端 LoRA 微调，但**中心聚合 base model + LoRA**：

- 客户端算力低，LoRA 训练。
- 中心聚合全模型更新。
- 比纯 LoRA 联邦性能更好。

## 十一、FL 的根本局限

### 1. 隐私保证弱

"数据不出本地"≠"梯度不出泄漏"——DLG / iDLG 等攻击能从梯度恢复图像。**必须配合 DP / HE / SecureAgg** 才能有真正的隐私。

### 2. 系统复杂度高

跨组织 FL 需要：
- 客户端注册、认证。
- 加密通信。
- 故障容忍（掉线客户端）。
- 模型版本管理。

**部署成本远高于集中训练**。

### 3. 性能损失

Non-IID 数据 + 通信限制 → FL 模型通常**比集中训练差 5~15%**。在 LLM 上更严重。

### 4. 公平性

**FedAvg 在异构数据下可能偏向"客户端多的群体"**。少数客户端（如小语种）的贡献可能被淹没。**公平聚合** 是开放问题。

## 十二、给工程团队的清单

1. **明确威胁模型**：谁可能看梯度？谁可能做成员推断？决定要 DP / HE / TEE 中哪个。
2. **数据分布审计**：每个客户端的数据分布如何？Non-IID 程度？
3. **选 FL 框架**：Flower（研究）、FATE / FedML（生产）、OpenFL（硬件）。
4. **通信预算**：定通信轮次上限 + 每轮通信量。
5. **隐私会计**：用 RDP accountant 追踪 DP 累计 $\epsilon$。
6. **监控异构**：监控 stragglers、掉线率、客户端算力差异。
7. **联邦 + 集中混合**：核心任务用集中训练，仅敏感环节用 FL。

## 小结

联邦学习通过"数据不动模型动"实现**法规合规与基础隐私**，但**梯度本身仍有泄漏风险**——必须配合 DP / HE / Secure Aggregation 才能获得可证明的隐私保护。在 LLM 场景下，**FedAvg + LoRA** 是当前最实用的组合（FedLLM / FedIT），但 Non-IID、通信成本、系统异构仍是根本挑战。下一篇我们将看到更"直接"的隐私攻击——**成员推断攻击**：从模型本身推断"某条记录是否在训练集中"，以及如何防御。
