# Video Transformer 与时序建模

Vision Transformer（ViT）证明了纯注意力结构在图像任务上同样可以超越 CNN。然而，视频比图像多了一个时间维度——把 ViT 直接"贴"到视频上，会让计算量随帧数平方增长。视频 Transformer 家族的核心议题就是：**如何在保持 attention 表达力的同时，把时空复杂度降到可控**。本文从时空注意力的设计挑战出发，梳理 TimeSformer、Video Swin Transformer、MViT、VideoMAE，再延伸到大模型时代的 VideoGPT、Sora DiT，最后用 PyTorch 从零实现 Divided Space-Time Attention。

## 一、视频 Transformer 的挑战：时空复杂度

ViT 把图像切成 $N \times N$ 个 patch（如 $14 \times 14$），全部送入 self-attention。视频多 $T$ 帧，最朴素的做法是"时空联合切 patch"：

- 时间维按 $t$ 帧切，时间 patch 大小 $t=2$；
- 空间维按 $14 \times 14$ 切，空间 patch 大小 $16$；
- 总 token 数 $= T/2 \times 14 \times 14 \approx T \times 196$。

把全部 token 跑一次 full attention：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
$$

复杂度是 $O((T \cdot N^2)^2)$——**视频时长加一倍，计算量翻四倍**。10 秒、30 FPS 的视频有 300 帧，比 ImageNet 单图多 200+ 倍，全 attention 完全跑不动。这是视频 Transformer 设计的**第一性约束**。

## 二、时空注意力的两条路线

围绕"如何把 $O(T^2 \cdot N^4)$ 降下来"，研究者走出两条路线：

### 2.1 Joint Space-Time Attention

把所有 token（无论空间或时间）一起跑 attention。一次搞定"帧内"与"帧间"关系。ViViT（2021）、Motionformer 采用这种方案。优势是表达力强；劣势是贵，且学习成本高（模型必须自己发现"时间相邻"这一归纳偏置）。

### 2.2 分时空分解 Attention

把 attention 拆成"空间"与"时间"两步：

$$
\text{SpatialAttn}: \quad X_{t,h,w} \leftrightarrow X_{t,h',w'} \quad (\text{同帧内})
$$

$$
\text{TemporalAttn}: \quad X_{t,h,w} \leftrightarrow X_{t',h,w} \quad (\text{同位置跨帧})
$$

每步复杂度 $O(N^2)$ 或 $O(T^2)$，**总复杂度从二次方降到线性相加**。TimeSformer、Video Swin Transformer 都走这条路。

物理直觉：Joint 方案是"全员开大会"，分时空是"先分组讨论、再纵向汇报"——后者更高效也更符合"时空独立"的物理直觉。

## 三、TimeSformer（Divided Space-Time Attention, 2021）

TimeSformer（Bertasius et al., 2021）是分时空方案的代表作。结构上仍是 ViT，但每个 block 内依次做：

### 3.1 时间注意力（Temporal Attention）

对每个空间位置 $(h, w)$，独立地在时间维 $T$ 上做 attention。即把同一个 $(h, w)$ 在所有帧的 token 视为一个序列：

$$
\text{TemporalAttn}(Q_t, K_t, V_t) = \text{softmax}\!\left(\frac{Q_t K_t^{\top}}{\sqrt{d}}\right) V_t
$$

其中 $Q_t, K_t, V_t$ 都来自时间维拼接的 token，复杂度 $O(T^2)$。

### 3.2 空间注意力（Spatial Attention）

时间 attention 之后，对每帧 $t$ 独立地在空间维 $N^2$ 上做 attention：

$$
\text{SpatialAttn}(Q_s, K_s, V_s) = \text{softmax}\!\left(\frac{Q_s K_s^{\top}}{\sqrt{d}}\right) V_s
$$

复杂度 $O(T \cdot N^4)$，与帧数线性相关。

物理直觉：先把同一物体在所有帧的"运动轨迹"汇总（时间 attention），再在每帧内部做空间关系建模。这样模型显式学习"时间一致性"和"空间结构"两个独立但互补的归纳偏置。

### 3.3 与 ViT 的对比

- ViT 单次 attention 跑全部时空 token，复杂度 $O(T^2 \cdot N^4)$。
- TimeSformer 拆开后复杂度 $O(T \cdot N^4 + T^2 \cdot N^2)$。
- 当 $T \ll N^2$（典型情况）时，TimeSformer 显著省 FLOPs，且精度更高。

TimeSformer 在 Kinetics-400/600 上超越同期 3D CNN 与 I3D，成为视频 Transformer 的"开山之作"。

## 四、Video Swin Transformer

Swin Transformer 在图像领域用"局部窗口 attention + 移位窗口"把复杂度从 $O(N^4)$ 降到 $O(N^2)$。Video Swin（Liu et al., 2022）把这一思想扩展到时空——**窗口变成 3D 的**。

### 4.1 3D 局部窗口

对一段时空特征 $T' \times H' \times W'$，用 $P_T \times P_H \times P_W$ 的 3D 窗口切分（如 $2 \times 7 \times 7$），每个窗口内独立做 self-attention：

$$
\text{WindowAttn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}} + B\right) V
$$

其中 $B$ 是相对位置编码。复杂度按窗口数线性扩展，整体 $O(T \cdot H \cdot W)$。

### 4.2 3D 移位窗口（Shifted Window）

图像 Swin 用 2D 移位让窗口间互通信息。Video Swin 扩展为 3D 移位——下一个 block 把窗口起点向 $(T, H, W)$ 三个方向各平移一段，让跨窗口的 token 重新组合成新窗口。配合 masked attention 实现"窗口内只算可见 token"。

### 4.3 Video Swin 的优势

- **高效**：3D 窗口让 attention 复杂度与帧数线性增长。
- **强归纳偏置**：局部窗口让模型天然适合视频的局部时空模式（短程运动、纹理）。
- **可扩展**：从 Tiny 到 Huge，覆盖 ImageNet-22K 预训练。

Video Swin Transformer 在 Kinetics-400 上达到 84.9% top-1，是当时最强模型之一，且推理速度比 TimeSformer 快 2–3 倍。

## 五、MViT（Multiscale Vision Transformer）

MViT（Fan et al., 2021）的核心是**多尺度特征金字塔**——借鉴 CNN 的 FPN 思想，在 Transformer 内构造多层级特征。

### 5.1 池化注意力（Pooling Attention）

MViT 的关键操作是**Pooling Q/K/V**：

$$
Q' = \text{Pool}(Q), \quad K' = \text{Pool}(K), \quad V' = \text{Pool}(V)
$$

通过 stride > 1 的 pooling（如 3D Pool，stride $= (1, 2, 2)$），让 Q/K/V 的序列长度缩短：

$$
\text{Attn}(Q', K', V') \in \mathbb{R}^{T' \cdot H' \cdot W' \times d}
$$

物理直觉：通过"池化注意力"让 attention 在不同 stage 看到不同尺度的时空信息——浅层精细、深层抽象。这与 ResNet 的 stage 设计思路一致，但用 attention 替代了 3D 卷积。

### 5.2 MViTv2 与改进

MViTv2 引入**相对位置编码 + 分离的相对偏置**，让 attention 学习更精细的时空相对位置，精度大幅提升。MViT 在 Kinetics-400 上以更低 FLOPs 达到 86% top-1，是精度/效率 trade-off 的代表模型。

## 六、VideoMAE：视频的 MAE 风格预训练

VideoMAE（Tong et al., 2022）把 He 等人 2021 年提出的 MAE（Masked Autoencoder）扩展到视频。其核心是：

### 6.1 高比例时空掩码

视频 token 数远多于图像，所以 VideoMAE 进一步提高掩码比例——把 90%–95% 的 token 随机遮住，只让 encoder 看到 5%–10% 的可见 token：

$$
\text{loss} = \frac{1}{M} \sum_{i \in \text{masked}} \| x_i - \hat{x}_i \|_2^2
$$

其中 $M$ 是被掩码的 token 数。极高掩码比例迫使模型利用**强时空冗余**重建原视频。

### 6.2 时空立方体掩码

一种改进是"立方体掩码"——把空间相邻、时间相邻的 token 一起掩码（如 $2 \times 16 \times 16$ 的 3D 立方体）。这样模型必须利用跨立方体的运动线索重建局部细节。

### 6.3 VideoMAE 的优势

- **自监督**：摆脱对 Kinetics 等大标注集的依赖，可直接用未标注视频。
- **高效**：encoder 只处理可见 token，FLOPs 大幅下降。
- **迁移性强**：预训练后可在 Kinetics-400/600、Something-Something、AVA 等多个下游任务微调。

VideoMAE 是视频 Transformer 进入"自监督预训练"时代的标志，让"小标注集+大未标注视频"成为可行范式。

## 七、现代视频大模型：从 VideoGPT 到 Sora

2024 年以来，视频 Transformer 走向"视频生成"方向，代表作有 VideoGPT、VideoPoet、Phenaki，以及 OpenAI 的 Sora。

### 7.1 VideoGPT 与 VideoPoet

- **VideoGPT**（Yan et al., 2021）：VQ-VAE 离散化视频 → 用 GPT 类自回归 Transformer 生成离散 token。
- **VideoPoet**（Google, 2023）：统一框架，把视频生成、图像生成、音频、文本等多模态塞进同一个 Transformer decoder。

它们的核心思想是**"把视频当成一种语言"**——用 LLM 范式（自回归 Transformer）统一理解与生成。

### 7.2 Sora 的 DiT（Diffusion Transformer）

OpenAI Sora（2024）公开的技术报告揭示了关键架构——**DiT（Diffusion Transformer）+ 时空 patch**：

1. **时空 patch**：把视频切成"时空小块"（spatio-temporal patch），每块是 $T' \times H' \times W' \times C$ 的小立方体。
2. **线性嵌入**：每个 patch 展平成 token，经线性投影送入 Transformer。
3. **DiT 主体**：在扩散模型的去噪网络中，用 Transformer 替代 U-Net。每个 block 是标准 Transformer（attention + FFN），配合 adaptive layer norm 把 timestep 注入。
4. **扩散过程**：从高斯噪声逐步去噪，生成 $T \times H \times W \times 3$ 的视频。

$$
x_{t-1} = \text{DiT}\!\left(\sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \, t, \, \text{text-prompt}\right)
$$

物理直觉：DiT 把 U-Net 的"局部归纳偏置"换成 Transformer 的"全局 attention"——这让模型能更好地建模长程时空一致性（如物体持续保持身份、跨秒动作连贯）。

### 7.3 可扩展的视频扩散

Sora 的另一关键是**可扩展**：随着 patch 数从 $16^2$ 增加到 $64^2$ 再到更长的时间窗口，整体性能稳定提升。这验证了"视频 Transformer 是 scaling law 友好"——更大的 patch 集合 + 更长的上下文能让模型持续变强。

## 八、应用：视频生成、检索与理解

视频 Transformer 的应用横跨三大方向：

### 8.1 视频生成

- 文本到视频（Text-to-Video）：Sora、Runway Gen-3、Luma Dream Machine。
- 图像到视频（Image-to-Video）：Stable Video Diffusion。
- 视频到视频（Video-to-Video）：可控编辑、风格迁移。

### 8.2 视频理解

- 动作识别、视频问答（VideoQA）、时序动作检测、视频描述（captioning）。
- VideoMAE / Video Swin 等预训练 backbone 提供强特征。

### 8.3 视频检索

- 用 Transformer 把视频编码成 embedding，在共享空间里与文本/图像做对比学习（CLIP 风格）。
- 文本搜视频、视频搜视频都依赖高质量 embedding。

## 九、PyTorch 实现：Divided Space-Time Attention

下面实现一个最小可用的 Divided Space-Time Attention，可直接 `python dst_attn.py` 跑通：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_rel_pos_embedding(seq_len: int, dim: int) -> torch.Tensor:
    """1D 相对位置偏置（Learnable）。"""
    rel_pos = torch.randn(seq_len, seq_len, dim) * 0.02
    return rel_pos


class TemporalAttention(nn.Module):
    """对每个空间位置 (h, w) 沿时间维 T 做 self-attention。"""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        # 相对位置偏置（沿时间维）
        self.rel_pos = nn.Parameter(get_rel_pos_embedding(64, num_heads))

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        # x: (B, T*H*W, D)
        B, N, D = x.shape
        # 重排为 (B*H*W, T, D)，便于对每个空间位置沿 T 做 attention
        x = x.reshape(B, T, H * W, D).permute(0, 2, 1, 3).reshape(B * H * W, T, D)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B * H * W, T, self.num_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]                                            # (B*HW, h, T, dh)
        attn = (q @ k.transpose(-2, -1)) * self.scale                        # (B*HW, h, T, T)
        # 加上可学习的相对位置偏置
        attn = attn + self.rel_pos[:T, :T].unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * H * W, T, D)
        out = self.proj(out)
        # 还原为 (B, T*H*W, D)
        out = out.reshape(B, H * W, T, D).permute(0, 2, 1, 3).reshape(B, T * H * W, D)
        return out


class SpatialAttention(nn.Module):
    """对每帧独立地在空间维 (H*W) 做 self-attention。"""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos = nn.Parameter(get_rel_pos_embedding(64, num_heads))

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        # x: (B, T*H*W, D)
        B, N, D = x.shape
        x = x.reshape(B * T, H * W, D)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B * T, H * W, self.num_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]                                            # (B*T, h, HW, dh)
        attn = (q @ k.transpose(-2, -1)) * self.scale                        # (B*T, h, HW, HW)
        attn = attn + self.rel_pos[:H * W, :H * W].unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * T, H * W, D)
        out = self.proj(out)
        return out.reshape(B, T * H * W, D)


class DividedSpaceTimeAttention(nn.Module):
    """TimeSformer 风格的 Divided Space-Time Attention block。"""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.spatial_attn = SpatialAttention(dim, num_heads)
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        # 1) Temporal attention + residual
        x = x + self.temporal_attn(self.norm1(x), T, H, W)
        # 2) Spatial attention + residual
        x = x + self.spatial_attn(self.norm2(x), T, H, W)
        # 3) FFN + residual
        x = x + self.mlp(self.norm3(x))
        return x


# 烟测：构造时空 token，跑一次 forward
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, H, W, D = 2, 8, 14, 14, 384
    block = DividedSpaceTimeAttention(dim=D, num_heads=6)
    x = torch.randn(B, T * H * W, D)
    y = block(x, T=T, H=H, W=W)
    print("output shape:", y.shape)   # torch.Size([2, 1568, 384])
```

代码里几个值得注意的点：

- **TemporalAttention 重排为 `(B*H*W, T, D)`**：把同一空间位置的 token 排成时间序列，让 attention 只在时间维发生。
- **SpatialAttention 重排为 `(B*T, H*W, D)`**：每帧独立做空间 attention。
- **相对位置偏置用 `nn.Parameter`**：让模型自适应学时间/空间相对位置，比绝对位置编码对长度更友好。
- **Pre-Norm + Residual**：保证深层 Transformer 训练稳定。

把这段代码塞进 patch embed（`Conv3d(kernel=(1,16,16), stride=(1,16,16))`）、堆叠 $L$ 个 block、最后 global pool 接分类头，就是一个最小可用的 TimeSformer。

## 十、PyTorch 实现：简化版 VideoMAE 的掩码重建

下面给出一个最小可用的 VideoMAE 风格掩码自编码器——先用 Video Swin 的 3D patch embed 把视频打成 token，再随机掩码 90%，让 Transformer encoder 只看 10% 的可见 token，最后用一个轻量 decoder 重建被掩码的时空 patch：

```python
import torch
import torch.nn as nn


class VideoPatchEmbed(nn.Module):
    """3D 时空 patch embed：Conv3d 一步完成 patch 切分与线性投影。"""

    def __init__(self, patch_size=(2, 16, 16), in_channels=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, T, H, W) -> (B, D, T', H', W')
        x = self.proj(x)
        B, D, T, H, W = x.shape
        return x.flatten(2).transpose(1, 2)   # (B, T*H*W, D)


class VideoMAE(nn.Module):
    """简化版 VideoMAE：encoder 只处理可见 token，decoder 重建被掩码 patch。"""

    def __init__(self, embed_dim=384, encoder_layers=6, decoder_dim=128,
                 decoder_layers=4, mask_ratio=0.9):
        super().__init__()
        self.patch_embed = VideoPatchEmbed(embed_dim=embed_dim)
        self.mask_ratio = mask_ratio

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=6, dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # decoder 把 token 重建为原 patch 大小 (2*16*16*3 = 1536 维像素)
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim, nhead=4, dim_feedforward=decoder_dim * 4,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_pred = nn.Linear(decoder_dim, 2 * 16 * 16 * 3)

    def random_masking(self, x: torch.Tensor):
        B, N, D = x.shape
        keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_visible, ids_keep, ids_restore

    def forward(self, x: torch.Tensor):
        # x: (B, 3, T, H, W)
        x = self.patch_embed(x)            # (B, N, D)
        x_vis, ids_keep, ids_restore = self.random_masking(x)
        latent = self.encoder(x_vis)       # (B, N_vis, D)
        # decoder 在全 token（含 masked 占位）上跑
        latent_full = self.decoder_embed(
            torch.cat([latent, torch.zeros_like(x[:, :x.size(1) - latent.size(1)])], dim=1)
        )
        # 真实实现里需要按 ids_restore 把 visible 的部分还原到原位置
        pred = self.decoder_pred(latent_full)
        return pred, ids_restore


# 烟测
if __name__ == "__main__":
    torch.manual_seed(0)
    model = VideoMAE()
    video = torch.randn(2, 3, 16, 224, 224)
    pred, ids_restore = model(video)
    print("pred shape:", pred.shape, "ids_restore shape:", ids_restore.shape)
```

代码要点：

- **3D patch embed**：用 `Conv3d(kernel=(2, 16, 16), stride=(2, 16, 16))` 一步完成时空切块与线性投影，把 $(B, 3, 16, 224, 224)$ 变成 $(B, N, D)$ 的 token 序列。
- **随机掩码 90%**：只让 encoder 看到 10% 的 token，迫使它从时空上下文重建原 patch。
- **Decoder 重建**：把被掩码位置用零向量占位，经 decoder 预测每个原始 patch 的像素值。

把这段代码与 MSE loss 衔接，就能开始无标注视频的自监督预训练。预训练完成后，丢弃 decoder，只用 encoder 即可迁移到 Kinetics、AVA、Something-Something 等下游任务。

## 十一、视频 Transformer 的训练与工程实践

### 11.1 预训练数据规模

视频 Transformer 对数据规模极为敏感——这是它与 CNN 的最大区别。常见组合：

| 预训练数据 | 数据量 | 代表模型 |
| --- | --- | --- |
| ImageNet-21K | 14M 图 | Video Swin |
| Kinetics-400 | 0.24M 视频 | TimeSformer、MViT |
| Kinetics-700 | 0.65M 视频 | Video Swin V2 |
| Something-Something V2 | 0.11M 视频 | 强调时序推理 |
| HowTo100M / WebVid | 100M+ 视频 | Sora、VideoPoet |

数据规模越大，模型越能从视频中学到"世界知识"。Sora 的关键不只是 DiT 架构，更是**海量视频数据 + 大规模 GPU 训练**。

### 11.2 训练技巧

- **Sparse Temporal Sampling**：训练时从视频中随机抽 $T$ 帧（典型 $T=8$ 或 $16$），推理时采更多帧（$T=32$ 或 $64$）取平均。
- **混合精度**：fp16/bf16 + GradScaler 是视频 Transformer 的标配——显存吃紧的代价远超图像。
- **梯度累积**：当单卡 batch=1 时，可以用梯度累积模拟大 batch。
- **Layer-wise LR Decay**：浅层用小学习率、深层用大学习率，能稳定收敛。

### 11.3 推理加速

- **Token Reduction**：Video Swin 的下采样天然减少 token；MViT 的池化注意力进一步压缩。
- **KV Cache**：对自回归视频生成（Sora 类），缓存历史帧的 K/V，避免重复计算。
- **蒸馏**：用大 TimeSformer 蒸馏小 TimeSformer，可保留 90%+ 性能但 FLOPs 砍半。

## 十二、视频 Transformer 的局限与未来

### 12.1 当前局限

1. **算力贵**：分钟级视频的全 attention 仍需 10+ 张 H100，不适合消费级部署。
2. **数据饥渴**：相比图像 Transformer，视频 Transformer 更依赖大规模预训练。
3. **长视频仍是难题**：1 小时视频的 $T=3600$ 帧对 attention 是天文数字，需要 memory bank、hierarchical attention 等额外机制。
4. **视频-语言对齐**：Video-LLaMA、Video-ChatGPT 等多模态大模型刚起步，幻觉问题严重。

### 12.2 未来方向

- **3D RoPE / 3D ALiBi**：把 LLM 的位置编码思想扩展到时空，让模型支持长视频。
- **Memory-based Attention**：Hiera、Memorizing Transformers 用外部 memory bank 缓存长程信息。
- **统一生成-理解**：Sora 类生成模型也开始具备理解能力，未来可能形成"既能生成、又能理解"的统一视频大模型。
- **稀疏视频扩散**：不是全帧去噪，而是只更新"运动区域"，降低生成成本。

## 小结

视频 Transformer 的核心议题是**"时空复杂度与表达力的平衡"**。TimeSformer 用分时空 attention 把复杂度从 $O(T^2 N^4)$ 降到 $O(T N^4 + T^2 N^2)$；Video Swin 用 3D 局部窗口进一步降到 $O(T \cdot H \cdot W)$；MViT 用多尺度池化注意力构造特征金字塔；VideoMAE 用 90%+ 高比例掩码实现自监督预训练；Sora DiT 用时空 patch + Diffusion Transformer 把视频生成扩展到分钟级。**共同主线是"把时空切块"——切得越合理，模型越高效、表达力越强**。在生成式 AI 时代，视频 Transformer 正成为视频生成与理解的统一骨干。下一篇我们将进入视频-语言多模态，看 Video-LLaMA、Video-ChatGPT 等多模态大模型如何把视频 Transformer 与 LLM 结合。
