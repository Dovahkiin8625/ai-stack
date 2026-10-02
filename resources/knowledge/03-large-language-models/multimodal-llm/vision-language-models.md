# 视觉-语言模型：ViT、CLIP、BLIP

视觉-语言模型（Vision-Language Model, VLM）是当前所有多模态应用的基石。它的核心问题是：**如何把图像和文字映射到同一个语义空间**？本文从 ViT 出发，沿 CLIP → BLIP → Flamingo → InstructBLIP 这条主线，拆解它们的架构与训练目标。

## 一、ViT：用 Transformer 统治视觉

2020 年 Google 的 Vision Transformer（ViT）证明：**只要把图像切成 patch，再用 Transformer 编码，就能在 ImageNet 上击败 CNN**。

### 1.1 Patch 切分

给定一张 $H \times W \times 3$ 的图像，ViT 把它切成 $N = \frac{HW}{P^2}$ 个 $P \times P$ 的小方块（patch），每个 patch 线性映射成 $d$ 维 embedding：

$$
\mathbf{x}_i = \mathbf{E} \cdot \text{flatten}(\text{patch}_i) + \mathbf{e}_{\text{pos},i}
$$

其中 $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot 3) \times d}$ 是线性投影，$\mathbf{e}_{\text{pos}}$ 是可学习的位置编码。

### 1.2 [CLS] token 与分类

按 BERT 的范式，预留一个可学习的 `[CLS]` token 拼在序列最前面，Transformer 编码后用 `[CLS]` 位置的特征做分类：

$$
[\text{CLS}, \mathbf{x}_1, \dots, \mathbf{x}_N] \xrightarrow{\text{Transformer}} [\mathbf{c}, \mathbf{h}_1, \dots, \mathbf{h}_N]
$$

最终 $\mathbf{c}$ 拿去接 MLP 分类头。

### 1.3 一个最小 ViT 实现

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)             # (B, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2)   # (B, N, D)


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, n_heads=12):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, self.patch.n_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, n_heads,
                                           dim_feedforward=embed_dim * 4,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        B = x.size(0)
        h = self.patch(x)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos
        return self.encoder(h)[:, 0]      # 取 [CLS]
```

训练好的 ViT 输出的每个 patch token 都是"该区域高层语义"的向量——这正是下游 VLM 想要的视觉表征。

## 二、CLIP：双塔对比学习

CLIP（2021）首次让视觉-语言对齐在工业规模上稳定。结构上，**两个独立编码器**互不通信，只在最后的向量空间里通过余弦相似度相连。

### 2.1 训练目标

给定 batch $B$ 内 $N$ 对 (image, text)，构建一个 $N \times N$ 的相似度矩阵，对角线是正样本，其余是负样本：

$$
\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} + \log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}\right]
$$

直觉：**让匹配对的相似度趋近 1，让不匹配对的相似度趋近 0**。对称化是为了让 image→text 和 text→image 两个方向都学。

### 2.2 简化双塔

```python
class CLIP(nn.Module):
    def __init__(self, vision, text, embed_dim=512, temperature_init=0.07):
        super().__init__()
        self.vision = vision                # ViT 输出维度 d_v
        self.text = text                    # Text 编码器输出维度 d_t
        self.vision_proj = nn.Linear(d_v, embed_dim, bias=False)
        self.text_proj = nn.Linear(d_t, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/temperature_init)))

    def forward(self, image, text_ids):
        v = F.normalize(self.vision_proj(self.vision(image)), dim=-1)
        t = F.normalize(self.text_proj(self.text(text_ids)), dim=-1)
        logits = v @ t.T * self.logit_scale.exp()
        return logits
```

零样本分类时，把"a photo of {class}"模板填进文本编码器，取 logits 最大值即可。

## 三、BLIP：ITC + ITM + LM 三任务统一

Salesforce 的 BLIP（2022）发现 CLIP 缺一个**生成式**任务，于是把三个目标统一到同一个模型：

| 任务 | 全称 | 目的 |
| --- | --- | --- |
| **ITC** | Image-Text Contrastive | 共享空间对齐（同 CLIP） |
| **ITM** | Image-Text Matching | 二分类：图与文是否匹配 |
| **LM** | Language Modeling | 给定图像生成描述 |

结构上，BLIP 用了一个**共享的 image encoder** + 一个**文本 encoder**（用于 ITC/ITM）+ 一个**文本 decoder**（用于 LM）。这种"encoder-decoder 同体"让 BLIP 既能检索又能 caption。

## 四、Flamingo：少样本视觉-语言模型

DeepMind 的 Flamingo（2022）首次让 VLM 在**少样本**设定下比肩 GPT-3。三个关键设计：

1. **冻结预训练视觉编码器（NFNet-F6）和冻结预训练 LLM（Chinchilla）**——只训练中间的桥。
2. **Perceiver Resampler**：把大量视觉 token 压缩为固定数量（如 64 个）的 latent query，让 LLM 看得过来。
3. **Gated Cross-Attention**：在 LLM 每一层（或每隔几层）插入带 sigmoid 门的 cross-attn，让视觉特征流式注入。

直觉：Flamingo 不重新发明 Transformer，而是在两个已训练好的塔之间"铺电缆"。

## 五、InstructBLIP：给 VLM 加指令微调

InstructBLIP（2023）在 BLIP-2 的基础上做指令微调，引入两个技巧：

1. **Instruction-aware Q-Former**：把指令文本也喂给 Q-Former，让它"按问题挑视觉特征"。
2. **指令微调数据集**：把 26 个公开数据集转成 instruction-tuning 格式，统一训练。

效果：在多个 VQA 任务上显著优于 BLIP-2，且对未见过的指令格式（如中文）也具备一定泛化能力。

## 六、模型能力对比

| 模型 | 视觉编码器 | LLM | 训练范式 | 核心贡献 |
| --- | --- | --- | --- | --- |
| CLIP | ViT-L/14 | — | 对比学习 | 零样本分类 |
| BLIP | ViT-B/L | BERT → OPT | ITC + ITM + LM | 三任务统一 |
| Flamingo | NFNet-F6 | Chinchilla | 冻结 + 桥接 | 少样本视觉-语言 |
| BLIP-2 | EVA-CLIP | OPT / FlanT5 | Q-Former 桥 | 高效训练（参数量 < 1%） |
| InstructBLIP | EVA-CLIP | FlanT5 | 指令微调 | 通用指令遵循 |

## 小结

视觉-语言模型的发展可以浓缩成三条主线：**更稳的对齐（CLIP）→ 更强的生成（BLIP）→ 更深的对接（Flamingo）**。共同的设计哲学是"**预训练 + 冻结 + 桥接**"——尊重 ViT 和 LLM 各自的能力，只训练中间的轻量适配器。下一篇我们将深入这些桥接器的具体形态：投影、Q-Former 与 Cross-Attention。
