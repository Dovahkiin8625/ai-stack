# AI 制药：从靶点发现到分子设计

新药研发平均耗资 **$2.6B、耗时 10~15 年**，成功率仅 **~10%**——是 AI 最有价值也最具挑战的应用领域。从 AlphaFold 2 解决蛋白质结构预测，到 Insilico Medicine 用 AI 找到新靶点，再到 Chemistry42、RoseTTAFold 推动分子设计，AI 制药已经从概念走向落地。本文系统介绍 AI 制药的核心任务（靶点发现、分子生成、ADMET 预测、临床前优化）、代表平台（AlphaFold、Insilico、Atomwise、Isomorphic Labs）、关键算法（Diffusion、Graph Neural Network），以及与传统制药的对比。

## 一、新药研发的传统流程

```text
靶点发现        →    3~5 年    →   $50M
  ↓
先导化合物      →    2~3 年    →   $100M
  ↓
临床前研究      →    1~2 年    →   $50M
  ↓
临床 Ⅰ 期       →    1~2 年    →   $100M
  ↓
临床 Ⅱ 期       →    2~3 年    →   $300M
  ↓
临床 Ⅲ 期       →    3~5 年    →   $1B
  ↓
审批上市        →    1~2 年    →   $50M
────────────────────────────────────
总计            →    10~15 年  →   $2.6B+
```

**核心痛点**：
1. 靶点发现慢且失败率高。
2. 先导化合物筛选周期长。
3. ADMET 性质（吸收、分布、代谢、排泄、毒性）优化难。
4. 临床试验成本极高。

## 二、AI 在制药中的应用图谱

```text
┌────────────────────────────────────────────────────────┐
│  1. 靶点发现（Target Discovery）                          │
│     - 多组学数据分析                                      │
│     - 蛋白质结构预测（AlphaFold）                          │
│     - 蛋白质-蛋白质相互作用                                │
└────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│  2. 分子生成（Molecule Generation）                       │
│     - 骨架跃迁                                          │
│     - 分子对接                                          │
│     - 反向合成规划                                       │
└────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│  3. ADMET 预测                                           │
│     - 吸收、分布、代谢、排泄、毒性                          │
│     - 药代动力学（PK）                                    │
│     - 药物-药物相互作用                                   │
└────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│  4. 临床前优化                                           │
│     - 虚拟筛选                                          │
│     - 剂量预测                                          │
│     - 适应症拓展                                        │
└────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│  5. 临床试验优化                                         │
│     - 患者分层                                          │
│     - 试验设计                                          │
│     - 招募优化                                          │
└────────────────────────────────────────────────────────┘
```

## 三、AlphaFold：蛋白质结构预测的革命

### 1. AlphaFold 2（DeepMind, 2020）

**解决了困扰生物学 50 年的"蛋白质折叠问题"**：

```python
import torch

# AlphaFold 2 的核心思想（简化）
class AlphaFold2(nn.Module):
    """AlphaFold 2 简化版。"""
    def __init__(self):
        # 1) Evoformer: 进化 + 多序列比对 (MSA) + 几何
        self.evoformer = EvoformerBlock(
            n_layers=48,
            d_pair=128,
            d_msa=256,
        )
        # 2) Structure module: 输出 3D 坐标
        self.structure_module = StructureModule(
            n_layers=8,
            d_ipa=128,  # Invariant Point Attention
        )
    
    def forward(self, msa, pair_features):
        """
        msa: (B, N_seq, N_res, d_msa) 多序列比对
        pair_features: (B, N_res, N_res, d_pair) 残基对特征
        """
        # 1) Evoformer 推理
        msa_repr, pair_repr = self.evoformer(msa, pair_features)
        
        # 2) Structure module 输出 3D 结构
        coords = self.structure_module(pair_repr, msa_repr)
        return coords  # (B, N_res, 3) 3D 坐标
```

**结果**：在 CASP14 上达到 **92.4 GDT**——接近实验精度。

### 2. AlphaFold 3（2024）

扩展到**蛋白质-配体、蛋白质-DNA、复合物**预测：

```python
class AlphaFold3(nn.Module):
    """AlphaFold 3：泛分子预测。"""
    def __init__(self):
        # 处理任意分子组合
        self.token_embed = MultiModalTokenEmbed(...)
        # 共享 Transformer
        self.diffusion = DiffusionModule(...)  # 用 Diffusion 输出结构
```

### 3. 开源生态

```python
OPEN_SOURCE_PROTEIN_TOOLS = {
    "AlphaFold 2/3": "DeepMind 官方（部分开源）",
    "ESMFold": "Meta ESM-2 + 结构预测",
    "RoseTTAFold": "David Baker Lab",
    "OmegaFold": "清华",
    "Helixon": "百度",
    "Protenix": "字节跳动",
}
```

## 四、分子生成

### 1. 分子表示

```python
# 1) SMILES 字符串
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # 阿司匹林

# 2) 分子图
# 节点：原子，边：键
graph = {
    "atoms": ["C", "C", "O", "c", "c", "c", "c", "c", "c", "C", "O", "O"],
    "bonds": [[0,1], [1,2], [1,3], [3,4], [4,5], ...],
    "features": [...],
}

# 3) 3D 构象
# 原子坐标 + 键长 + 键角 + 二面角
```

### 2. Graph Neural Network 分子生成

```python
class MolecularGNN(nn.Module):
    """分子 GNN。"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.atom_embed = nn.Embedding(100, hidden_dim)  # 原子种类
        self.bond_embed = nn.Embedding(10, hidden_dim)   # 键类型
        self.gnn = GINConv(hidden_dim)                  # Graph Isomorphism
        self.readout = nn.Linear(hidden_dim, 1)          # 性质预测
    
    def forward(self, molecule_graph):
        # 节点嵌入
        x = self.atom_embed(molecule_graph.atom_types)
        
        # 消息传递
        for layer in self.gnn_layers:
            x = layer(x, molecule_graph.edge_index, molecule_graph.edge_attr)
        
        # 图级池化
        graph_repr = global_mean_pool(x, molecule_graph.batch)
        return self.readout(graph_repr)  # 预测性质
```

### 3. Diffusion 分子生成

```python
class MolecularDiffusion(nn.Module):
    """用 Diffusion 生成分子 3D 结构。"""
    def __init__(self):
        # Equivariant diffusion: 保持平移/旋转不变性
        self.equivariant_net = EquivariantGNN()
    
    def forward(self, protein_pocket):
        """给定蛋白质口袋，生成结合的配体。"""
        # 1) 初始化随机 3D 坐标 + 原子类型
        coords = torch.randn(N_atoms, 3)
        atom_types = torch.randint(0, 100, (N_atoms,))
        
        # 2) 迭代去噪
        for t in reversed(range(n_steps)):
            coords, atom_types = self.equivariant_net.denoise_step(
                coords, atom_types, protein_pocket, t
            )
        
        return coords, atom_types
```

代表工作：**DiffDock**（MIT, 2023）、**TargetDiff**、**MolDiff**。

### 4. 强化学习分子生成

```python
class MolecularRL:
    """用 RL 优化分子。"""
    
    def __init__(self, generator, predictor):
        self.generator = generator  # 生成 SMILES
        self.predictor = predictor  # 预测活性/性质
    
    def step(self):
        # 1) 生成候选分子
        smiles = self.generator.generate()
        
        # 2) 预测性质
        activity = self.predictor(smiles)
        drug_likeness = self.drug_likeness_score(smiles)
        novelty = self.novelty_score(smiles)
        
        # 3) 奖励
        reward = (
            2.0 * activity +
            1.0 * drug_likeness +
            1.0 * novelty
        )
        
        # 4) 更新
        return reward
```

## 五、ADMET 预测

### 1. 核心 ADMET 任务

| 任务 | 含义 | 失败影响 |
|---|---|---|
| **吸收（A）** | 口服生物利用度 | 药效不足 |
| **分布（D）** | 组织分布、血脑屏障 | 副作用、毒性 |
| **代谢（M）** | 肝脏代谢稳定性 | 剂量、相互作用 |
| **排泄（E）** | 肾脏清除 | 蓄积毒性 |
| **毒性（T）** | 急性/慢毒、致畸、肝毒 | 临床失败主因 |

### 2. 经典 ADMET 数据集

```python
ADMET_BENCHMARKS = {
    "Aqueous Solubility": "Delaney ESOL dataset, ~1100 molecules",
    "BBB Penetration": "血脑屏障渗透性, ~2000 molecules",
    "CYP Inhibition": "细胞色素 P450 抑制, 多亚型",
    "hERG Inhibition": "心脏毒性, ~650 molecules",
    "AMES Mutagenicity": "致突变性, ~6500 molecules",
    "Hepatotoxicity": "肝毒性, ~950 drugs",
    "Skin Sensitization": "皮肤致敏",
    "LogP / Lipophilicity": "脂溶性",
}
```

### 3. 多任务 ADMET 模型

```python
class MultiTaskADMET(nn.Module):
    """多任务 ADMET 预测。"""
    def __init__(self, encoder, n_tasks=12):
        super().__init__()
        self.encoder = encoder  # GNN 或 Transformer
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder.out_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )
            for _ in range(n_tasks)
        ])
    
    def forward(self, molecules):
        shared_repr = self.encoder(molecules)
        # 每个任务有自己的 head，但共享底层表示
        return [head(shared_repr) for head in self.task_heads]


# 训练：每个任务加权 loss
loss = sum(
    task_weight[i] * F.binary_cross_entropy(preds[i], targets[i])
    for i in range(n_tasks)
)
```

## 六、代表公司

### 1. Insilico Medicine（英矽智能）

**AI 驱动药物发现**：

```text
里程碑:
- 2020: 用 AI 发现 DDR1 激酶抑制剂（治疗纤维化）
  从立项到临床前候选物仅 21 天
- 2022: INS018_055 抗纤维化药物进入临床 Ⅰ 期
  全球首个 AI 全流程发现的药物
```

技术栈：

```python
INSILICO_PLATFORM = {
    "PandaOmics": "靶点发现（多组学 + 知识图谱）",
    "Chemistry42": "分子生成（Transformer + RL）",
    "InClinico": "临床试验预测",
    "生成式 AI": "小分子、抗体、RNA 设计",
}
```

### 2. Atomwise

**AtomNet** 平台：卷积神经网络做虚拟筛选：

```python
class AtomNet(nn.Module):
    """3D CNN 分子-蛋白相互作用预测。"""
    def __init__(self):
        # 输入：3D 网格化的"配体 + 口袋"
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # ...
        )
        self.fc = nn.Linear(512, 1)  # 结合亲和力
```

### 3. Recursion Pharmaceuticals

**用高通量影像 + AI 发现新药**：

```text
工作流：
1. 细胞表型实验 → 高内涵成像（每化合物几百个参数）
2. 深度学习提取形态学特征
3. 找相似形态的化合物 → 推断相似机制
4. 推断新靶点 → 化合物-表型-靶点映射
```

### 4. Isomorphic Labs（Alphabet 子公司）

DeepMind 母公司成立——把 AlphaFold 商业化制药。

### 5. 国内公司

- **晶泰科技（XtalPi）**：晶体结构预测 + 药物设计。
- **望石智慧（StoneWise）**：AI 制药平台。
- **百图生科（BioMap）**：AI 生物制药。
- **云深智药**：阿里 + 高校联合。
- **华为 EIHealth**：盘古药物分子大模型。

## 七、关键算法

### 1. Graph Neural Network

```python
class GINLayer(nn.Module):
    """Graph Isomorphism Network 层。"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, edge_index):
        # x: (N_nodes, hidden_dim)
        # edge_index: (2, N_edges)
        
        # 聚合邻居
        row, col = edge_index
        neighbor_sum = scatter_add(x[col], row, dim=0)
        
        # 更新
        out = self.mlp((1 + self.eps) * x + neighbor_sum)
        return out
```

### 2. Transformer for Chemistry

```python
class ChemFormer(nn.Module):
    """化学 Transformer。"""
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model, nhead=8, num_layers=12
        )
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        return self.output(self.transformer(src, tgt))
```

### 3. Equivariant Neural Network

3D 分子结构需要**旋转/平移不变性**：

```python
class EquivariantGNN(nn.Module):
    """SE(3)-等变图神经网络。"""
    def __init__(self):
        # 同时输出标量 (invariant) + 向量 (equivariant)
        self.scalar_head = nn.Linear(...)
        self.vector_head = nn.Linear(...)
    
    def forward(self, coords, atom_types):
        """
        coords: (N_atoms, 3)
        atom_types: (N_atoms,)
        """
        # 标量特征：旋转不变
        scalars = self.scalar_head(atom_types)
        
        # 向量特征：旋转等变
        vectors = coords.unsqueeze(-1) * self.vector_head(atom_types).unsqueeze(1)
        
        # 消息传递时同时更新标量和向量
        return scalars, vectors
```

## 八、AI vs 传统制药

### 1. 速度对比

| 阶段 | 传统 | AI | 加速倍数 |
|---|---|---|---|
| 靶点发现 | 3~5 年 | 0.5~1 年 | 5~10x |
| 先导优化 | 2~3 年 | 6~12 月 | 4~6x |
| 临床 Ⅰ 期 | 1~2 年 | 1~1.5 年 | 1.5x |

### 2. 成功率对比

传统制药：临床 Ⅰ→上市 ~10%。

AI 制药（截至 2024）：

| 公司 | 阶段 | 进展 |
|---|---|---|
| Insilico INS018_055 | 临床 Ⅱ | 进展良好 |
| Recursion | 临床 Ⅰ/Ⅱ | 多管线 |
| AbCellera | 已上市 | COVID 抗体快速发现 |
| Exscientia | 临床 Ⅰ/Ⅱ | DSP-1181 等 |

**样本量仍小**——AI 是否真的提升成功率仍待观察。

### 3. 成本对比

| 阶段 | 传统 | AI | 节约 |
|---|---|---|---|
| 早期发现 | $50M | $5~10M | 80%+ |
| 先导优化 | $100M | $20~30M | 70%+ |
| 临床 | $1.4B | （基本不变） | <10% |

**AI 主要加速前期——临床阶段仍是最贵最慢**。

## 九、AI 制药的挑战

### 1. 数据稀缺与质量

```python
DATA_CHALLENGES = {
    "labeled_data": "高质量活性数据稀缺（公开数据集 ~10^5 量级，私有更大）",
    "noisy_labels": "活性数据常含实验噪声",
    "out_of_distribution": "AI 在训练分布外表现差",
    "data_privacy": "制药公司不愿共享核心数据",
}
```

**缓解**：
- 主动学习选择最有价值的实验。
- 自监督学习用未标注数据。
- 联邦学习（多公司合作）。

### 2. 可合成性

AI 生成的分子可能**根本合不出来**：

```python
def synthetic_accessibility_score(molecule):
    """可合成性评分。"""
    # 用 SAS (Synthetic Accessibility Score)
    # 基于片段出现频率 + 复杂度
    return SA_Score(molecule)  # 1=容易, 10=很难
```

**解决方案**：
- **可合成性约束**：把 SAS 分数加入奖励。
- **反向合成规划**：先生成，再规划合成路线。

### 3. 生物学复杂性

```text
靶点蛋白 → 调控细胞通路 → 影响疾病
       ↓ 但实际还有：
   - 蛋白质相互作用网络
   - 反馈机制
   - 代谢适应性
   - 异质性
```

**AI 模型多在分子层面有效，但生物学系统复杂**——临床失败率高。

### 4. 监管不确定性

FDA 对 AI 制药的**审批路径**仍在演化：

- 2024 年 FDA 讨论草案。
- 关键问题：AI 设计的分子 vs 人类设计的分子，监管是否一致？

## 十、未来方向

### 1. 基础模型

```python
class MolecularFoundationModel:
    """分子基础模型。"""
    def __init__(self):
        # 用 10 亿+ 分子训练
        self.backbone = PretrainedTransformer(...)
    
    def few_shot_adapt(self, new_task_data, n_shot=20):
        """用 20 个样本适配新任务。"""
        # Adapter / LoRA 微调
        return self.fine_tune_with_lora(new_task_data, n_shot)
```

代表：**Uni-Mol**（微软 + 清华）、**MolBERT**、**ChemBERTa**、**MoleculeGPT**。

### 2. 抗体设计

```python
class AntibodyDesigner:
    """AI 设计抗体。"""
    def design(self, antigen_3d):
        # 1) 抗原表面分析
        epitope = self.find_epitope(antigen_3d)
        
        # 2) 抗体 CDR 区设计
        cdr_sequences = self.generate_cdr(epitope)
        
        # 3) 亲和力优化
        optimized = self.optimize_affinity(cdr_sequences, antigen_3d)
        
        return optimized
```

代表：**AbodyBuilder**、**IgFold**、**Chai-1**（用于抗体）。

### 3. RNA 设计

```python
class RNADesigner:
    """AI 设计 RNA 药物（如 mRNA、siRNA）。"""
    def design(self, target_protein):
        # 1) RNA 序列设计
        sequences = self.generate_rna(target_protein)
        
        # 2) 二级/三级结构预测
        structures = self.predict_structure(sequences)
        
        # 3) 优化稳定性 + 免疫原性
        return self.optimize(sequences, structures)
```

代表：**LinearDesign**、**mRNAid**、**RhoFold**。

### 4. 自动化实验室

```text
AI + 机器人自动化:
- AI 设计分子
- 机器人自动合成 + 测试
- 数据反馈到 AI
- 闭环 → 加速 10~100x
```

代表：**Emerald Cloud Lab**、**Strateos**、**Insilico 的机器人实验室**。

## 十一、给 AI 制药团队的清单

1. **从明确靶点开始**：不要做"全靶点扫描"——有针对性。
2. **与药化专家深度协作**：AI 不能替代药化经验。
3. **湿实验验证**：所有 AI 预测必须经过实验验证。
4. **多任务 ADMET**：不要只看活性，要综合 ADMET。
5. **可合成性优先**：生成的分子能合出来才有意义。
6. **数据治理**：高质量私有数据 + 公开数据结合。
7. **持续迭代**：把实验数据持续反馈到模型。
8. **多模态融合**：分子 + 蛋白 + 表型 + 组学。
9. **合规与伦理**：动物福利、临床试验伦理。
10. **耐心**：药物研发仍是"长跑"——AI 不是银弹。

## 小结

AI 制药已经从概念走向落地——**AlphaFold 解决了蛋白质结构预测，Insilico 把 AI 全流程药物推到临床 Ⅱ 期，Atomwise 等平台缩短早期发现周期 5~10x**。核心任务包括靶点发现、分子生成、ADMET 预测、临床前优化；核心算法包括 GNN、Transformer、Diffusion、Equivariant Network。**核心挑战**是数据稀缺、可合成性、生物学复杂性、监管不确定性。**AI 的真正价值**不在"加速一切"，而在**让小团队能做大公司能做的事**——降低新药研发的门槛，让罕见病、个体化治疗更可行。三篇文章覆盖了 ai-for-healthcare 的核心：医疗 LLM、医学影像 AI、AI 制药。下一篇我们将转向 **ai-for-science**：AI 在基础科学研究中的应用。
