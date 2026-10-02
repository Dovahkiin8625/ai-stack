# Vision Transformer 与现代视觉架构：从 ViT 到 ConvNeXt

2020 年，Google 的 Dosovitskiy 等人用一篇《An Image is Worth 16x16 Words》彻底改变了视觉模型的设计范式——他们证明：**只要把图像切成小块（patch），直接套用 NLP 领域的 Transformer 编码器，就能在 ImageNet 上达到甚至超过 CNN 的精度**。Vision Transformer（ViT）撕掉了"图像必须用卷积处理"的标签，但同时也暴露了 Transformer 在视觉任务上的若干痛点：数据饥饿、缺乏归纳偏置、计算量随分辨率平方增长。本文从 ViT 的核心机制出发，沿着 DeiT（数据高效）、Swin Transformer（层次化）、ConvNeXt（卷积复兴）的演进路线，把"Transformer 时代"的视觉架构梳理清楚，最后用 PyTorch 从零实现一个最小可用的 ViT block。

## 一、ViT 的核心思想：图像 = 一串 patch tokens

Transformer 处理的是"序列"。要把图像喂给 Transformer，必须先把图像"序列化"。ViT 的做法非常暴力：把 $H \times W \times C$ 的图像均匀切成 $N = \frac{HW}{P^2}$ 个 $P \times P$ 的 patch（$P=16$ 是常见选择），再把每个 patch 拉平成一个 $P^2 \cdot C$ 维向量。这一步叫 **patch embedding**，由一个卷积层（stride = kernel size = $P$）高效实现：

$$
z_0 = [x_{\text{cls}}; \, x_p^1 E; \, x_p^2 E; \dots; x_p^N E] + E_{\text{pos}}
$$

其中：

- $x_p^i \in \mathbb{R}^{P^2 \cdot C}$ 是第 $i$ 个 patch 的像素向量；
- $E \in \mathbb{R}^{P^2 C \times D}$ 是 patch embedding 矩阵，把每个 patch 投影到 $D$ 维；
- $x_{\text{cls}} \in \mathbb{R}^{D}$ 是一个**可学习的分类 token**（借鉴 BERT），其最终输出用作整图分类；
- $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 是可学习的位置编码，让模型感知 patch 的空间顺序。

直觉：**patch 就是图像的"词元"（token）**，后续的 Transformer encoder 就像在 NLP 里一样，用 self-attention 让任意两个 patch 直接交互。

## 二、ViT 的 Encoder Block

ViT block 与原始 Transformer encoder 完全一致：交替的 Multi-Head Self-Attention（MHA）与 Feed-Forward Network（FFN），外加残差与 LayerNorm（Pre-Norm）：

$$
z'_\ell = \text{MHA}\!\left(\text{LN}(z_{\ell-1})\right) + z_{\ell-1}
$$
$$
z_\ell = \text{FFN}\!\left(\text{LN}(z'_\ell)\right) + z'_\ell
$$

其中 FFN 是逐位置的两层 MLP，扩展比通常为 4：

$$
\text{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2, \quad \sigma = \text{GELU}
$$

物理直觉：MHA 让每个 patch 都能"看到"所有 patch 的信息，从而捕获长程依赖；FFN 在每个 patch 内部做非线性变换。MHA 的 $O(N^2)$ 计算量是 ViT 的主要瓶颈——当 $H=W=224, P=16$ 时 $N=196$，还好；但若改成 $P=8$，$N=784$，计算量变成 4 倍。

## 三、ViT vs CNN：归纳偏置的拉锯战

CNN 假设"局部 + 平移不变"，所以：

- **局部性（Locality）**：$3\times3$ 卷积只看邻域。
- **平移等变性（Translation Equivariance）**：把猫平移一下，特征图整体平移，类别不变。
- **层次化（Hierarchy）**：通过池化逐层扩大感受野。

ViT 不假设这些：

- **没有局部性假设**：第一个 attention 层就让任意两个 patch 直接相连。
- **没有平移等变**：必须靠数据学习（或靠位置编码人为注入）。
- **所有层感受野都是全局**：patch 之间距离远近不影响 attention 的"路径长度"。

这意味着 ViT 比 CNN **更灵活**，但**需要更多数据**才能学到这些先验。论文里的经验数字很说明问题：在 ImageNet-1K（1.3M 图）上从头训练，ViT 比 ResNet 差几个点；但在 JFT-300M（300M 图）预训练后，ViT 反超 CNN。这就是所谓**数据效率问题**。

## 四、DeiT（2021）：用蒸馏让 ViT "吃得起" ImageNet

为了缓解 ViT 的数据饥饿，Meta 的 Touvron 等人提出 **Data-efficient image Transformer（DeiT）**。两个核心技巧：

1. **强数据增强**：RandAugment、Mixup、Cutmix、随机擦除——把 CNN 训练里的"配方"借给 Transformer。
2. **蒸馏 Token（Distillation Token）**：在输入序列里再加一个可学习的 $x_{\text{dist}}$，最终它的输出通过交叉熵与教师模型的预测对齐。教师既可以是 CNN（硬标签），也可以是预训练好的 Transformer（软标签，对应 `DeiT-B↑`）。

蒸馏损失：

$$
\mathcal{L} = \frac{1}{2}\mathcal{L}_{\text{CE}}(\text{softmax}(z_L^{\text{cls}}), y) + \frac{1}{2}\mathcal{L}_{\text{CE}}(\text{softmax}(z_L^{\text{dist}}), y_{\text{teacher}})
$$

DeiT-B 在 ImageNet-1K 上从头训练就能达到 83.1% top-1，几乎与 EfficientNet 持平，且训练成本远低于 JFT 预训练。这是 ViT 走向"工业可用"的关键一步。

## 五、Swin Transformer（2021）：层次化的视觉 Transformer

ViT 的"一把梭"结构（patch 数量全程不变）有两个缺陷：

1. **特征图分辨率固定**：无法直接用于检测、分割等需要多尺度特征的下游任务。
2. **全局 attention 的 $O(N^2)$ 计算量**：高分辨率图像吃不起。

Swin Transformer 通过两个关键设计解决了这两个问题：

1. **Patch Merging**：每经过一个 stage，把相邻 $2\times2$ patch 的特征拼接后用线性层降维，等价于 2x 下采样。空间分辨率逐 stage 减半（$\frac{H}{4} \to \frac{H}{8} \to \frac{H}{16} \to \frac{H}{32}$），通道数加倍——和 CNN 经典 backbone 一样的层次化结构。
2. **Window Attention**：在每个 stage 内只对局部 $M \times M$（默认 $M=7$）窗口内的 patch 做 self-attention，计算量从 $O(N^2)$ 降到 $O(M^2 N)$。
3. **Shifted Window**：相邻的两个 block 在窗口划分上"错位"半个窗口，从而让信息在不同窗口间流动——这是 Swin 名字的由来。

$$
\hat{z}^l = \text{W-MSA}(\text{LN}(z^{l-1})) + z^{l-1}
$$
$$
z^l = \text{MLP}(\text{LN}(\hat{z}^l)) + \hat{z}^l
$$
$$
\hat{z}^{l+1} = \text{SW-MSA}(\text{LN}(z^l)) + z^l
$$
$$
z^{l+1} = \text{MLP}(\text{LN}(\hat{z}^{l+1})) + \hat{z}^{l+1}
$$

物理直觉：Swin 把 CNN 的"局部-全局层次化"用 attention 重新实现了一次，既享受 Transformer 的全局建模能力，又保留 CNN 的尺度金字塔。Swin Transformer 在 COCO 检测、ADE20K 分割上一举夺魁，成为视觉 backbone 的新标杆。

## 六、ConvNeXt（2022）：用 CNN 借鉴 Transformer

2022 年 Meta 的 Liu 等人做了件有趣的事：**从纯 ResNet-50 出发，逐步把 Transformer 的"设计配方"搬进卷积网络**，最终得到一个不输 Swin Transformer 的纯卷积模型 ConvNeXt。

逐步改造清单（每一项贡献几个点的精度）：

1. **训练配方**：把 ResNet 的 90 epoch 训练扩展到 300 epoch，使用 AdamW + cosine schedule + 强数据增强（和 DeiT 一致）。
2. **Inverted Bottleneck**：将 bottleneck 块的维度从"宽-窄-宽"改成"窄-宽-窄"，参考 MobileNetV2——这与 Transformer FFN "升维-降维"的结构同源。
3. **大卷积核**：把 $3\times3$ 替换成 $7\times7$ depthwise 卷积，对应 ViT/Swin 的"全局感受野"。
4. **Depthwise 卷积替代分组归一化**：把组数 $G$ 设成等于通道数，等价于 depthwise，再与 $1\times1$ pointwise 组合——这正是 MobileNet 的核心。
5. **LayerNorm 替代 BatchNorm**：对 batch size 不敏感，训练推理行为更一致。
6. **少激活函数**：每个块只用一个 GELU，与 Transformer block 的 FFN 数量一致。
7. **Patchify Stem**：把 stem 的 $7\times7$ + maxpool 替换成一个 $4\times4$ stride=4 的卷积——和 ViT 的 patchify 同源。

最终 ConvNeXt-T 在 ImageNet-1K 上达到 82.1% top-1，与 Swin-T 持平，推理速度还更快。这个工作的意义是：**架构的胜负不在"卷积 vs attention"，而在"哪种设计哲学更适配任务"**。

## 七、现代视觉架构横向对比

| 模型 | 类型 | 参数量 | ImageNet top-1 | 特点 |
| --- | --- | --- | --- | --- |
| ResNet-50 | CNN | 25.6M | 76.1% | 残差学习的代表，工业标配 |
| EfficientNet-B3 | CNN | 12M | 81.7% | 复合缩放（深度+宽度+分辨率） |
| DeiT-B | Transformer | 86M | 83.1% | 蒸馏 ViT，1K 数据从零训练 |
| Swin-T | Transformer | 28M | 81.3% | 层次化 + 滑动窗口 |
| Swin-B | Transformer | 88M | 83.5% | 大数据 / 大模型更优 |
| ConvNeXt-T | CNN | 29M | 82.1% | 现代化 ResNet |
| ConvNeXt-B | CNN | 89M | 83.8% | 与 Swin-B 持平甚至略胜 |

观察：

- **同参数量下差距在 ±1%**——架构之间的"代差"远小于训练数据/配方的代差。
- **Transformer 的优势在高分辨率/大模型**时更明显；中等规模下，ConvNeXt 是非常务实的选择。
- **FLOPs 与推理速度非线性相关**：Swin 的 window attention 看似省计算，但实现中的 memory access pattern 不友好，实际推理可能比 ConvNeXt 慢。

## 八、PyTorch 实现：最小可用的 ViT Block

下面是一个独立的 ViT encoder block，可直接 `python vit_block.py` 运行。把它堆叠 $L$ 次 + CLS token 分类头 = 一个完整的 ViT：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """把 (B, 3, H, W) 切成 (B, N, D)，N = (H*W)/P^2。"""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)             # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MLP(nn.Module):
    """Transformer block 中的逐位置前馈网络（FFN）。"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Attention(nn.Module):
    """多头自注意力（无 GQA，无 RoPE；ViT 原始实现）。"""

    def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)        # (B, H, N, Dh)

        # PyTorch 2.0+ 内置高效实现，等价于手写 scaled-dot-product
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    """ViT Encoder Block: Pre-Norm + MHA + Pre-Norm + FFN。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.,
                 dropout: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, num_classes: int = 1000,
                 embed_dim: int = 384, depth: int = 12,
                 num_heads: int = 6, mlp_ratio: float = 4.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                     # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, D)
        x = torch.cat((cls, x), dim=1)             # (B, N+1, D)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                  # CLS token 输出


if __name__ == "__main__":
    # 烟测：构造一个小模型，跑一次 forward
    model = SimpleViT(img_size=224, patch_size=16,
                      embed_dim=384, depth=6, num_heads=6)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("output shape:", out.shape)              # torch.Size([2, 1000])
    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
```

代码里几个值得注意的点：

- **Patch Embedding 用 `nn.Conv2d(stride=kernel_size)`**：天然把切块和线性投影融合，等价于把 patch 拉平再乘 $E$。
- **CLS token 与位置编码**：`nn.Parameter` 注册为可学习参数，`trunc_normal_` 初始化避免激活值过大。
- **Pre-Norm**：`norm1 → attn → residual`、`norm2 → mlp → residual`，与原始 ViT 论文一致。
- **`scaled_dot_product_attention`**：PyTorch 2.0 之后的高效实现，比手写 softmax 节省显存并跑得快。

## 九、训练 ViT 的工程经验

ViT 的训练比 ResNet 更加"娇贵"。下面这些经验是大量实验总结出来的共识。

### 9.1 优化器与学习率

- **AdamW**：几乎是 ViT 训练的标配。$\text{lr}=1e-4$（DeiT-B）到 $3e-4$（ViT-L），$\text{weight decay}=0.05$。
- **Warm-up**：前 5-10 epoch 用线性 warm-up 避免初始梯度爆炸；之后用 cosine 衰减到 $1e-6$。
- **梯度裁剪**：clip 到 1.0，防止极端 batch 导致训练发散。
- **Layer-wise LR Decay（LLRD）**：给浅层设更小的学习率（如 $\times 0.9^{i}$），浅层参数更稳定。ViT finetune 时 LLRD 比统一 lr 提升 0.5-1%。

### 9.2 数据增强

- **强增强**：RandAugment + Mixup（$\alpha=0.8$）+ Cutmix（$\alpha=1.0$）+ Random Erasing 是 Deit 论文里的标准配置。
- **RandomErasing / CutOut**：在 patch 层面或像素层面随机遮盖一部分，对 ViT 尤其有效——它强迫模型不要过度依赖单一 patch。
- **ColorJitter**：色彩抖动 + 灰度化，能提升模型对光照变化的鲁棒性。

### 9.3 正则化

- **Dropout**：`attn_drop=0`、`proj_drop=0.1` 是默认值。
- **DropPath / Stochastic Depth**：在 FFN 或 attention 后随机丢掉整个分支，对 ViT 比 Dropout 更有效。DeiT-B 用 0.1，Swin 接近 0.2-0.4。
- **Mixup 概率**：与 label smoothing（$\epsilon=0.1$）结合，能再涨 0.5-1%。

### 9.4 推理技巧

- **TTA（Test-Time Augmentation）**：推理时对原图、水平翻转、多尺度裁剪各跑一次再取平均，能涨 0.5-1%。但代价是推理时延翻倍。
- **模型集成**：DeiT 论文里 $\text{DeiT-B}\uparrow$ 用 CNN 教师，$\text{DeiT-B}\ddagger$ 用 Transformer 教师，多教师集成是工业界常见做法。
- **Token 剪枝**：DynamicViT 等工作通过预测 token 重要性，跳过不重要的 patch，推理 FLOPs 可降 30-50% 而精度损失 < 1%。

下面这段 finetune 脚本展示了核心流程：

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 假设 model 是预训练 ViT，train_loader 提供目标域数据
optimizer = torch.optim.AdamW([
    {'params': model.patch_embed.parameters(), 'lr': 1e-5},
    {'params': model.blocks[:6].parameters(),  'lr': 5e-5},
    {'params': model.blocks[6:].parameters(),  'lr': 1e-4},
    {'params': model.head.parameters(),        'lr': 1e-4},
], weight_decay=0.05)

scaler = GradScaler()
for imgs, labels in train_loader:
    imgs, labels = imgs.cuda(), labels.cuda()
    optimizer.zero_grad(set_to_none=True)
    with autocast(dtype=torch.float16):
        loss = torch.nn.functional.cross_entropy(model(imgs), labels)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

注意这里的关键是**分层学习率**——backbone 浅层用更小 lr，分类头用更激进 lr，是 ViT finetune 的"标配配方"。

## 十、可视化与可解释性

ViT 的可解释性是一个研究热点。几个经典手段：

### 10.1 Attention Map 可视化

ViT 的 self-attention 权重本身就能解释"模型在看哪里"。把最后一层（或最后几层）CLS token 对所有 patch 的 attention 权重上采样到原图分辨率，能直接画出"模型关注的热力图"。

但要注意：ViT 的注意力并不是"模型真实在用的东西"，它是 softmax 后的一个中间表示，与最终预测的相关性是经验性的。

### 10.2 DINO / DINOv2 的自监督 attention

Meta 的 DINO 用自蒸馏训练 ViT，学到的注意力图非常干净——不需要任何标注就能在前景分割上达到 SOTA。这与 ViT 的全局感受野有关：注意力本身就隐含了"哪些 patch 属于同一物体"的信息。

### 10.3 MAE（Masked Autoencoder）

Kaiming He 等人把 NLP 的 BERT 预训练范式搬到视觉上：随机遮住 75% 的 patch，让 ViT 重建被遮住的像素。MAE 的可视化非常震撼——模型能根据可见 patch 重建出几乎完整原图，证明了 ViT 学到了真正的语义而非纹理。

## 十一、在下游任务中使用 ViT 系列 backbone

ViT / Swin 系列不只用于分类，下面是几个常见下游用法。

### 11.1 目标检测

- **DETR**（Facebook AI, 2020）：首次用 Transformer 编码-解码结构端到端做检测。backbone 是 ResNet 或 Swin，decoder 输出固定数量的 object queries。
- **Swin Transformer + RetinaNet / Mask R-CNN**：Swin 的层次化特征天然兼容 FPN，是检测的"开箱即用"backbone。
- **Co-DETR / DINO**：近两年的 SOTA 检测器，进一步改进 DETR 的 query 设计。

### 11.2 语义分割

- **SegFormer**：用 Mix-Transformer (MiT) 做 encoder，简单的 MLP decoder，简洁高效。
- **Mask2Former**：统一了检测、分割、实例分割，用 masked attention 替代 cross-attention。

### 11.3 多模态模型

- **CLIP / DINOv2**：ViT-L/14 是 CLIP 的标准 backbone，对比学习预训练后可直接做 zero-shot 分类。
- **BLIP / LLaVA**：用 ViT 提取图像特征，作为 LLM 的视觉输入，是当前多模态 LLM 的核心组件。

## 十二、常见疑问与误区

1. **"ViT 一定比 CNN 好"**：错。在中小数据集上 ResNet/ConvNeXt 往往更稳。ViT 的优势要在 JFT-300M 规模预训练后才完全显现。
2. **"ViT 没有归纳偏置就一定差"**：错。ViT 通过位置编码和大量数据"学到"了等变性。DeiT 证明在足够强的增强下，ViT 在 1K 数据集也能超过 CNN。
3. **"Patch 越小越好"**：不是。Patch 越小，序列越长，attention 的 $O(N^2)$ 计算量剧增。实践中 $16\times16$ 是分类的常用选择，$4\times4$ 多用于分割。
4. **"CLS token 是必须的"**：不一定。Average Pooling（去掉 CLS token）对所有 patch 输出取平均，效果几乎一样。但 CLS token 对下游检测/分割有兼容性优势（需要额外的 query token）。
5. **"Swin 全面超越 ViT"**：在检测/分割上是的；在纯分类任务上 ViT-L/14 仍然很能打，两者各有优势场景。

## 十三、前沿趋势：MAE、DINOv2 与多模态预训练

2022 年之后，视觉模型的研究重心从"架构创新"明显转向"预训练范式"。几个里程碑工作值得了解。

### 13.1 MAE（Masked Autoencoders Are Scalable Vision Learners）

何恺明等人把 NLP 的 BERT 范式搬到视觉：随机遮住输入 patch 的 75%，让 ViT 重建被遮住的像素。MAE 的关键设计：

- **编码器只看可见 patch**：把 25% 可见 patch 喂给 ViT encoder，省 3-4 倍计算。
- **轻量解码器**：一个小型 Transformer 把可见 patch 表示 + mask token 一起解码成像素。
- **像素回归损失**：MSE 直接对像素值回归，比 patch 分类简单。

MAE 证明 ViT 在 ImageNet-1K 上用自监督预训练后，仅 finetune 50 epoch 就能达到 84%+ top-1，几乎与全监督持平。这把"数据标注成本"的天花板又抬高了一截。

### 13.2 DINOv2（Meta, 2023）

DINOv2 用 142M 张无标签图像训练 ViT-L/14，得到的特征几乎"开箱即用"——零样本分类、深度估计、语义分割都能直接用。它的核心技巧是"无标签自蒸馏 + 多裁剪增强 + 动量编码器"，把对比学习与 masked modeling 结合。

DINOv2 的意义：**让通用视觉特征（foundation model for vision）第一次接近 NLP 时代的 GPT-3**。后续 SAM（Segment Anything）、Grounding DINO 等工作都建立在 DINOv2 的特征之上。

### 13.3 多模态 LLM 与 ViT

ChatGPT 出现后，几乎所有多模态 LLM（LLaVA、Qwen-VL、InternVL、GPT-4V）都用 ViT 提取图像特征。典型流程：

1. 用 CLIP / SigLIP 预训练的 ViT 把图像编码成 visual tokens；
2. 通过 projector（MLP 或 Q-Former）把 visual tokens 映射到 LLM 的 embedding 空间；
3. 与文本 tokens 一起送入 LLM，做下一 token 预测。

ViT 在这一波多模态浪潮中扮演了"视觉编码器"的角色，几乎是 LLaMA 系 LLM 的标配搭档。

### 13.4 当下活跃的研究方向

- **混合架构**：CoAtNet、CM3Leon、InternImage 用卷积做浅层、用 attention 做深层。
- **高效 attention**：线性 attention、LongFormer-style 局部+全局、Flash Attention 让高分辨率视觉成为可能。
- **视频 Transformer**：TimeSformer、Video Swin 把时间维度纳入 attention。
- **3D 视觉**：把 ViT 扩展到点云（Point Transformer）、NeRF（pixelNeRF）。

## 十四、回到原点：选哪个 backbone？

最后给一个简短的"选型指南"——面对实际问题时该选什么：

| 任务场景 | 推荐 backbone | 理由 |
| --- | --- | --- |
| 中小数据集分类（< 100K 图） | ResNet-50 / ConvNeXt-T | 归纳偏置强，finetune 稳定 |
| 大规模分类预训练 | ViT-L / Swin-B / ConvNeXt-B | 三者精度接近，按喜好选择 |
| 目标检测 | Swin-T/B 或 ConvNeXt | 层次化特征 + 强下游兼容性 |
| 语义分割 | Swin / ConvNeXt + UPerNet | 同上 |
| 多模态 LLM | CLIP ViT-L/14 / SigLIP | 与文本预训练对齐 |
| 自监督预训练 | ViT-L + MAE / DINOv2 | 当前最强的视觉 foundation model |
| 移动端 / 边缘部署 | MobileNetV3 / EfficientNet | 速度优先 |

## 十五、小结

从 ViT 到 ConvNeXt，过去五年视觉架构的演进给我们的启示是：

1. **架构是手段，归纳偏置是核心**：ViT 把归纳偏置完全交给数据，Swin 把局部性还回来，ConvNeXt 用卷积显式表达局部性——三种哲学各有所长。
2. **训练配方比架构更重要**：DeiT 用更好的数据增强 + 蒸馏让 ViT 在 1K 上起飞；ConvNeXt 用更长 schedule + AdamW 让 ResNet 复活。架构差异常常被"配方"抹平。
3. **层次化是刚需**：纯 ViT 在分类上很好，但在检测/分割上必须依赖 FPN 之类的外部结构；Swin / ConvNeXt 的层次化是"通用 backbone"的关键。
4. **未来是混合架构**：CoAtNet、CM3Leon、InternImage 等近期工作把卷积的局部性与 attention 的全局性融合，正在成为新的研究主流。
5. **预训练范式才是当下决胜关键**：MAE / DINOv2 让 ViT 在无标注数据上达到前所未有的表示能力，多模态 LLM 进一步把视觉模型与语言模型深度融合。

理解 ViT 等于理解了"Transformer 范式如何跨界"，而理解 ConvNeXt 则理解了"为什么卷积不会消亡"。下一篇我们将进一步探索自监督预训练（MAE、DINO）如何让视觉模型在没有标签的海量数据上学到更好的表示——这将是 ViT 之后视觉领域的下一个主战场。
