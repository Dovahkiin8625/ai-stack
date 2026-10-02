# 多模态融合架构：投影、Q-Former、Cross-Attention

视觉编码器（ViT、CLIP）和大语言模型（LLM）分别在自己的领域已经"训练充分"，真正决定多模态系统能力的是**它们之间的桥**。本文拆解三种主流融合架构：线性投影（MLP Projector）、注意力桥（Q-Former）和跨注意力注入（Cross-Attention），并给出 LLaVA 风格 projector 的最小可运行代码。

## 一、为什么"融合"是多模态的核心难题

一个 224×224 的图像经 ViT-L/14 切分后会产生 **256 个 token**（14×14 patch），而每个 token 维度是 1024。LLM 的输入通常是 2048-8192 维的高维语义向量。要把视觉特征送进 LLM，必须解决：

1. **维度对齐**：视觉编码器输出维度 ≠ LLM 输入维度。
2. **token 数压缩**：256 个视觉 token 直接拼进 prompt，序列长度爆炸。
3. **语义对齐**：视觉 patch 是"局部纹理级"，LLM 需要"物体-关系级"语义。

这就是"融合架构"要解决的问题。

## 二、方案 A：MLP Projector（LLaVA 风格）

LLaVA 的方案最简单：**冻结 ViT 和 LLM，只训练一个两层 MLP** 把视觉特征投影到 LLM 的 embedding 空间。

### 2.1 结构

```
ViT 输出 patch tokens (B, N, D_v)
        │
   LayerNorm
        │
   Linear(D_v → D_llm)
        │
       GELU
        │
   Linear(D_llm → D_llm)
        │
visual tokens (B, N, D_llm)
        │
       + 文本 tokens → LLM
```

视觉 token 数与 ViT 的 patch 数一致。LLaVA-1.5 用 336×336 输入 + ViT-L/14，产生 576 个视觉 token。

### 2.2 最小可运行实现

```python
import torch
import torch.nn as nn

class MLPProjector(nn.Module):
    """LLaVA 风格两层 MLP projector。"""
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.norm = nn.LayerNorm(vision_dim)
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, visual_tokens):
        # visual_tokens: (B, N, D_v)  例如 (B, 576, 1024)
        x = self.norm(visual_tokens)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)        # (B, N, D_llm)
        return x


class LLaVALikeFusion(nn.Module):
    """完整的 LLaVA 风格融合：ViT → MLP → LLM。"""
    def __init__(self, vit, llm_embed, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.vit = vit
        for p in self.vit.parameters():        # 冻结 ViT
            p.requires_grad = False
        self.projector = MLPProjector(vision_dim, llm_dim)
        self.llm_embed = llm_embed             # 文本 embedding 表

    def forward(self, images, input_ids):
        # 1) 视觉侧
        with torch.no_grad():
            visual_tokens = self.vit.forward_features(images)   # (B, N, D_v)
        visual_embeds = self.projector(visual_tokens)           # (B, N, D_llm)

        # 2) 文本侧
        text_embeds = self.llm_embed(input_ids)                  # (B, L, D_llm)

        # 3) 拼接，喂给 LLM
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        return inputs_embeds    # 后续交给 LLM 的 transformer body
```

### 2.3 优点与局限

- **优点**：结构简单、训练快（只训 projector）、显存友好。
- **局限**：视觉 token 数量大（576），长序列推理慢；细节信息在 MLP 中可能损失。

## 三、方案 B：Q-Former（BLIP-2 风格）

Q-Former 是 BLIP-2 引入的关键设计。它本质是一个**轻量 transformer**，中间插一组可学习的 "query tokens"，通过 cross-attention 把视觉特征"压缩"到固定数量（如 32 个）的 latent。

### 3.1 工作机制

```
可学习 query (B, K, D)  ──┐
                          ├── Cross-Attention → Self-Attention → 输出 (B, K, D)
ViT 视觉特征 (B, N, D) ───┘
```

其中 $K \ll N$，所以 $K=32$ 时视觉 token 数从 256 压到 32，**节省 8 倍推理显存**。同时这些 query 通过自监督学习（ITC/ITM/ITG）学会了"挑出与任务相关的视觉信息"。

### 3.2 简化 Q-Former

```python
class QFormerBlock(nn.Module):
    def __init__(self, dim=768, n_heads=12):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        # q: (B, K, D)  kv: (B, N, D)
        q1, _ = self.cross_attn(self.norm1(q), kv, kv)
        q = q + q1
        q2, _ = self.self_attn(self.norm2(q), q, q)
        q = q + q2
        q = q + self.ffn(self.norm3(q))
        return q
```

Q-Former 的训练目标比纯投影复杂——它要在 ITC、ITM、ITG（image-grounded text generation）三任务上联合优化。

## 四、方案 C：Cross-Attention 注入（Flamingo 风格）

Flamingo 不再把视觉 token "硬塞"进 LLM 的输入序列，而是在 LLM 每一层（或每隔几层）插入**带 sigmoid 门控的 cross-attention**，让视觉信息"软注入"到 hidden state。

### 4.1 数学表达

设 LLM 第 $l$ 层的 hidden state 为 $h_l$，视觉特征为 $V$：

$$
h_l' = h_l + \sigma(\alpha_l) \cdot \text{CrossAttn}(h_l, V)
$$

其中 $\sigma(\alpha_l)$ 是可学习的**门控因子**，初始化为接近 0——训练初期几乎不影响 LLM，从而**保护预训练好的语言能力不被破坏**。

### 4.2 实现片段

```python
class GatedCrossAttention(nn.Module):
    def __init__(self, dim=4096, n_heads=32):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))   # 初始化为 0，门接近关闭

    def forward(self, x, visual_kv):
        # x: (B, L, D)   visual_kv: (B, N, D)
        out, _ = self.attn(x, visual_kv, visual_kv)
        return x + torch.sigmoid(self.gate) * out
```

把这种层插入 LLM 的第 4、8、12、16、20 层，就得到 Flamingo 风格的"多模态 LLM"。训练时**只更新 cross-attn 的参数和门**，保持 LLM 冻结。

## 五、三种方案对比

| 维度 | MLP Projector | Q-Former | Cross-Attention |
| --- | --- | --- | --- |
| **代表模型** | LLaVA、MiniGPT-4 | BLIP-2、InstructBLIP | Flamingo、IDEFICS |
| **可训练参数量** | ~10M | ~100M | ~300M |
| **视觉 token 数** | 等于 patch 数（256-576） | 固定 K（32） | 全部保留 |
| **训练成本** | 最低 | 中等 | 较高 |
| **LLM 是否冻结** | 通常冻结 | 冻结 | 冻结 |
| **细节保留** | 中（受 MLP 容量限制） | 强（可挑细粒度） | 强（全程可见） |
| **推理速度** | 中等（视觉 token 多） | 最快（视觉 token 少） | 较慢（每层多算一次） |
| **代表问题** | 简单指令 | 细粒度定位 | 复杂多步推理 |

## 六、现代模型的混合方案

新一代 MLLM 倾向于**多种方案的组合**：

- **CogVLM**：在 LLM 内部增加一个"视觉专家" FFN，专门处理视觉 token——本质是方案 A + 局部方案 C。
- **DeepSeek-VL**：高分辨率图像切 tile，每 tile 用独立 ViT，再用 projector 拼接（方案 A 的扩展）。
- **InternVL**：用 Q-Former 风格的 token 压缩 + cross-attention 注入。
- **MiniCPM-V / Llama 3.2 Vision**：LLaVA 风格 MLP + cross-attention 增强。

## 七、选型建议

| 任务 | 推荐方案 | 理由 |
| --- | --- | --- |
| 通用对话、OCR、文档理解 | MLP Projector | 简单稳定，已足够强 |
| 视频、长上下文多图 | Q-Former | token 压缩显著 |
| 精细空间推理（GUI Agent、机器人） | Cross-Attention | 视觉 token 不可压缩 |
| 资源有限的小模型（< 4B） | MLP Projector | 训练成本最低 |

## 小结

融合架构的本质是**"在什么位置、以什么粒度、把多少视觉信息送进 LLM"**。MLP Projector 最朴素但够用，Q-Former 巧妙地压缩视觉 token，Cross-Attention 则最贴近"无损接入"。工程上没有绝对最优，理解三种方案的取舍，再结合任务特点选择或混合，才是落地之道。下一篇我们将走出视觉，进入音频/语音大模型的世界。
