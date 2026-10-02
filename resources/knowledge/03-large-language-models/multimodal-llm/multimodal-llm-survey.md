# 多模态大模型概览：从 CLIP 到 GPT-4V

2021 年 CLIP 的横空出世把"图像 + 文本"对齐学习推到了工业可用的高度；三年后，GPT-4V、Gemini、LLaVA、Qwen-VL 让多模态对话成为大模型的标配能力。本文按"范式演进"梳理这条主线：对比学习奠基 → 生成式扩散 → 视觉-语言对话 → 统一原生多模态，并给出一个可独立运行的简化 CLIP 双塔 PyTorch 实现。

## 一、为什么需要多模态

纯文本 LLM 面对真实世界时存在两大盲区：

1. **缺乏对视觉/音频/视频的原生理解**——它只能"听别人描述"，无法直接"看图"。
2. **人类常识多以视觉形式存在**——"红色 + 圆形 + 顶部一片绿叶 = 苹果"，光靠文字描述需要数千 token。

多模态大模型（Multimodal Large Language Model, MLLM）的目标就是把这些异构信号**统一在一个语言空间**里，让模型能"看、听、读、写"。

## 二、五大主流范式

下面按时间线梳理主流路线。

### 1. 对比学习范式：CLIP / ALIGN

OpenAI 的 CLIP（2021）和 Google 的 ALIGN 用 4 亿对"图像-文本"做对比学习，把图像编码器和文本编码器映射到同一个向量空间。这种"双塔"结构天然适合检索、零样本分类。

**代表模型**：CLIP、ALIGN、Florence。

### 2. 视觉编码器 + LLM 适配范式：LLaVA / MiniGPT-4 / Qwen-VL

冻结一个预训练 ViT，再冻结一个 LLM，中间用一个轻量"投影器"（线性层、MLP 或 Q-Former）把视觉 token 翻译成 LLM 能理解的 token。这是当前**最主流**的工业方案。

**代表模型**：LLaVA、MiniGPT-4、Qwen-VL、InternVL、CogVLM。

### 3. 端到端生成范式：CogView / DALL-E / Stable Diffusion

把多模态理解为"从文本生成像素"。这类模型不是用来"理解"图像，而是"画"出图像，常用扩散模型或自回归 transformer。

**代表模型**：DALL-E 2/3、Stable Diffusion、Kandinsky。

### 4. 统一多模态范式：Chameleon / Show-o / Janus

用一个**单一 transformer** 同时处理文本 token 和图像 token（图像被切成 VQ 编码）。这类架构的优势是"一套模型、一个训练目标"，缺点是图像分辨率和细节能力受限于 VQ。

**代表模型**：Chameleon（Meta）、Show-o、Janus（DeepSeek）、Transfusion。

### 5. 原生多模态范式：Gemini / GPT-4V

从预训练第一天就把图像、音频、视频一起喂进去，模型权重里同时存在多种模态的"神经元"。技术上属于黑盒，但效果惊人。

**代表模型**：Gemini 1.5、GPT-4V/o。

## 三、对比学习的核心：CLIP 训练目标

CLIP 的训练目标是：对一个 batch 内 $N$ 对 (image, text)，让匹配的对得分高、不匹配的对得分低。

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)} + \log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ji}/\tau)}\right]
$$

其中 $s_{ij} = \cos(\mathbf{I}_i, \mathbf{T}_j)$ 是余弦相似度，$\tau$ 是温度。这种"对称 InfoNCE"损失让两边的表示在共享空间里对齐。

## 四、简化 CLIP 双塔实现

下面是一个教学用的最小 PyTorch 实现，逻辑等同于原版 CLIP，去掉了分布式训练和 gradient checkpointing 等工程细节。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImageEncoder(nn.Module):
    """简化版图像编码器：ResNet50 backbone + 投影头。"""
    def __init__(self, embed_dim=256):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B, 2048, 1, 1)
        self.proj = nn.Linear(2048, embed_dim)

    def forward(self, x):
        h = self.backbone(x).flatten(1)        # (B, 2048)
        return F.normalize(self.proj(h), dim=-1)


class TextEncoder(nn.Module):
    """简化版文本编码器：词嵌入 + Transformer + 投影头。"""
    def __init__(self, vocab_size=10000, max_len=64, embed_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(embed_dim, n_heads, dim_feedforward=512,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, ids):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, self.token(ids) + self.pos(pos)], dim=1)
        h = self.encoder(h)
        return F.normalize(self.proj(h[:, 0]), dim=-1)        # 取 [CLS]


def clip_loss(image_feats, text_feats, temperature=0.07):
    """对称 InfoNCE 损失。"""
    logits = image_feats @ text_feats.T / temperature   # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

训练循环：

```python
image_enc = ImageEncoder()
text_enc = TextEncoder()
opt = torch.optim.AdamW(list(image_enc.parameters()) + list(text_enc.parameters()), lr=1e-4)

for images, token_ids in dataloader:
    i_feat = image_enc(images)
    t_feat = text_enc(token_ids)
    loss = clip_loss(i_feat, t_feat)
    opt.zero_grad(); loss.backward(); opt.step()
```

这个实现丢掉了分布式、混合精度、梯度缓存、mask 等工程细节，但保留了三件事：**双编码器**、**共享向量空间**、**对称 InfoNCE**——这就是 CLIP 的灵魂。

## 五、演进全景图

```
2021  CLIP / ALIGN         对比学习双塔，零样本分类可用
2022  Flamingo / BLIP      冻结 LLM，注入视觉特征
2023  LLaVA / MiniGPT-4    视觉编码器 + 投影 + LLM，对话能力起飞
2023  GPT-4V / Gemini      原生多模态，闭源旗舰
2024  InternVL / Qwen-VL   高分辨率 + 强指令，开源 SOTA
2024  Chameleon / Show-o   统一 tokenizer，单一 transformer
2024  Janus / Transfusion  理解/生成解耦的视觉编码器
```

## 六、选型建议

| 场景 | 推荐路线 | 理由 |
| --- | --- | --- |
| 图像检索 / 零样本分类 | CLIP 系（双塔） | 训练简单，检索 O(1) embedding |
| 文档/图表问答 | LLaVA / Qwen-VL / InternVL | OCR + 视觉 + 推理三位一体 |
| 文生图 / 设计 | Stable Diffusion / DALL-E 3 | 像素级生成 |
| 多模态 Agent | GPT-4V / Claude 3.5V | 工具调用 + 视觉理解最稳定 |
| 移动端 / 端侧 | MobileVLM / MiniCPM-V | 参数量 1-3B，可本地跑 |

## 小结

多模态大模型的演进可以归纳为五条路线——对比学习、生成式、视觉+LLM 拼接、统一 transformer、原生融合。当前工业界**最实用**的是"ViT 视觉编码器 + MLP 投影 + LLM"的组合；研究界正在向"统一 tokenizer"和"原生多模态"收敛。下一篇我们将深入剖析这条主线最核心的视觉编码器——ViT 以及它在 CLIP、BLIP 中的角色。
