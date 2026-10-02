# 语义分割：从 FCN 到 SegFormer

语义分割（Semantic Segmentation）是计算机视觉里"像素级理解"的核心任务——给一张图，模型要回答"每个像素属于哪个类别"。和分类只输出一张图的标签不同，分割要求输出**一张和输入同分辨率的类别图**。本文从任务定义出发，梳理 FCN、U-Net、SegNet、DeepLab 系列、到 SegFormer / Mask2Former 这条主线上的关键设计，并给出 PyTorch 实现简化版 ASPP 模块的代码。

## 一、什么是语义分割

给定一张 $H \times W \times 3$ 的 RGB 图像，语义分割模型输出 $H \times W \times C$ 的概率图，其中 $C$ 是类别数，每个位置 $(i, j)$ 上的 $C$ 维向量表示该像素属于各类别的概率。最终预测：

$$
\hat{Y}_{i,j} = \arg\max_{c \in \{1, \dots, C\}} P_{i,j}(c)
$$

物理直觉：**把分类"展开"成每个像素的分类**。所以也叫 **dense prediction（密集预测）** 或 **pixel-wise classification（逐像素分类）**。

注意一个容易混淆的边界：

- 语义分割：同一类别的多个实例（两只猫）会被涂成**同一个颜色**——它只关心"是什么"，不关心"是哪一只"。
- 实例分割（下一篇）：每只猫都要单独 mask + 类别。
- 全景分割（下一篇）：把 stuff（天空、道路）和 things（行人、汽车）一起做。

## 二、评估指标：IoU、mIoU、Dice 与 Pixel Accuracy

### 2.1 Pixel Accuracy

最朴素的指标：分类正确的像素数占总像素数的比例。

$$
\text{Pixel Accuracy} = \frac{\sum_{c} TP_c}{\sum_{c} (TP_c + FP_c + FN_c)}
$$

缺点：**类别不均衡时几乎不可用**。例如背景占 95%，模型只预测背景就能拿到 95% 的 pixel accuracy，但完全没用。

### 2.2 IoU（Intersection over Union）

对每个类别分别计算预测 mask 与真值 mask 的交并比：

$$
\text{IoU}_c = \frac{|P_c \cap G_c|}{|P_c \cup G_c|} = \frac{TP_c}{TP_c + FP_c + FN_c}
$$

### 2.3 mIoU（mean IoU）

对所有类别 IoU 取平均，是 PASCAL VOC、Cityscapes、ADE20K 等数据集的**官方标准指标**：

$$
\text{mIoU} = \frac{1}{C}\sum_{c=1}^{C} \text{IoU}_c
$$

### 2.4 Dice Coefficient

医疗影像里更常用，等价于 F1-score 的 mask 版本：

$$
\text{Dice}_c = \frac{2 |P_c \cap G_c|}{|P_c| + |G_c|} = \frac{2 \, TP_c}{2 \, TP_c + FP_c + FN_c}
$$

Dice 与 IoU 的关系（非线性）：$\text{Dice} = \frac{2 \, \text{IoU}}{1 + \text{IoU}}$。

PyTorch 最小实现：

```python
import torch


def iou_per_class(pred, target, num_classes, eps=1e-6):
    """pred: (N, H, W) int64; target: (N, H, W) int64."""
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        g = (target == c)
        inter = (p & g).sum().float()
        union = (p | g).sum().float()
        if union < eps:
            ious.append(torch.tensor(float("nan")))  # 跳过空类别
        else:
            ious.append(inter / (union + eps))
    return torch.stack(ious)              # (C,)
```

## 三、FCN：把分类网络改造成分割网络

FCN（Fully Convolutional Network, Long et al. 2015）是一切的起点。它做了两件关键事：

1. **用 1×1 卷积替换全连接层**：把 ImageNet 预训练的 VGG / AlexNet 末尾的 `fc` 层换成 $1 \times 1$ conv，输出从 `(N, 1000)` 变成 `(N, C, H/32, W/32)`。
2. **双线性上采样到原图分辨率**：用 `nn.Upsample(mode="bilinear")` 把特征图放大回 $H \times W$。
3. **Skip connections**：把 pool3（stride 8）、pool4（stride 16）的特征也接到 decoder 上，得到 FCN-32s / 16s / 8s，越精细结果越好。

物理直觉：分类网络的下采样学到的是"语义强但分辨率低"的特征，FCN 把它们"展开"回原图尺寸。skip connection 负责把"边缘、纹理"等浅层细节补回来。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class FCN32s(nn.Module):
    """最朴素的 FCN：VGG16 backbone + 1x1 conv + 双线性上采样 32 倍。"""
    def __init__(self, num_classes: int = 21):
        super().__init__()
        backbone = vgg16(weights=None).features
        self.encoder = backbone                       # 输出 stride 32
        self.score = nn.Conv2d(512, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)

    def forward(self, x):
        feat = self.encoder(x)                        # (N, 512, H/32, W/32)
        logits = self.score(feat)                     # (N, C, H/32, W/32)
        return self.upsample(logits)                  # (N, C, H, W)
```

FCN 的根本问题：32 倍上采样太粗，物体边界模糊。它奠定了"编码器-解码器 + skip"这个范式，但精度不够。

## 四、U-Net：对称的编码器-解码器

U-Net（Ronneberger et al. 2015）原本是为医疗影像（细胞分割）设计的，但很快在所有分割任务上爆火。它的结构对称、优雅：

- **收缩路径（encoder）**：每次下采样都把通道翻倍、空间减半，捕获上下文。
- **扩张路径（decoder）**：每次上采样把空间翻倍、通道减半，恢复定位。
- **Skip connections（关键）**：把 encoder 对应层的特征**裁剪/对齐后 concat**到 decoder，给 decoder 提供"我从哪里来"的细粒度信息。

物理直觉：**decoder 知道"这块是肝脏"，但不知道"边界在哪里"；encoder 的浅层知道"边界在这"，但不知道"是什么"。skip 把两者拼起来。**

U-Net 的对称结构还有一个工程上的好处：编码器可以用 ImageNet 预训练，decoder 是从头训的小网络，整体仍能很快收敛。

```python
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """最小可用的 U-Net（4 层下采样）。"""
    def __init__(self, in_ch=3, num_classes=21, base=64):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bot  = DoubleConv(base * 8, base * 16)
        self.up4  = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3  = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1  = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out_conv = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x);  p1 = F.max_pool2d(e1, 2)
        e2 = self.enc2(p1); p2 = F.max_pool2d(e2, 2)
        e3 = self.enc3(p2); p3 = F.max_pool2d(e3, 2)
        e4 = self.enc4(p3); p4 = F.max_pool2d(e4, 2)
        b  = self.bot(p4)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)               # (N, C, H, W)
```

U-Net 至今仍是医学影像、缺陷检测等中小数据集上的"默认配置"。

## 五、SegNet：用池化索引上采样

SegNet（Badrinarayanan et al. 2017）是一个细节优化：decoder 上采样时**直接使用 encoder 的 max-pooling 索引**（哪个位置被选中过）把特征"扔回去"。

$$
\text{Up}(x, \text{idx})_{i,j} = \begin{cases} x_{i',j'} & \text{if } \text{idx}_{i,j} = (i', j') \\ 0 & \text{otherwise} \end{cases}
$$

优点：**省参数、省显存**（不用学 deconv），边界也更锐利。缺点：丢了 max 之外的位置信息，精度通常略低于 U-Net / DeepLab。

## 六、DeepLab 系列：空洞卷积与 ASPP

DeepLab（Chen et al., Google）走的是另一条路——**不缩小分辨率，但扩大感受野**。

### 6.1 空洞卷积（Dilated / Atrous Convolution）

普通卷积 stride=1 时感受野线性增长；下采样 stride=2 时分辨率减半、感受野翻倍，但**空间信息丢失**。空洞卷积在两者之间：

$$
y[i] = \sum_{k} x[i + r \cdot k] \cdot w[k]
$$

其中 $r$ 是**dilation rate**。$r=1$ 时就是普通 $3\times3$；$r=2$ 时感受野扩大到 $5\times5$，但参数只算 9 个。**物理直觉：在不缩小分辨率的前提下"看更远"。**

DeepLabv3+ 的 backbone（Xception / ResNet）最后几层把 stride 从 2 改成 1，配合 dilation rate 2、4，保证 output stride = 8 而感受野足够大。

### 6.2 CRF（条件随机场）后处理

DeepLabv1/v2 时代，CNN 输出后还会接一个**全连接 CRF**（Fully Connected CRF, Krähenbühl & Koltun）：

$$
E(y) = \sum_{i} \psi_u(y_i) + \sum_{i < j} \psi_p(y_i, y_j)
$$

其中一元项 $\psi_u$ 是 CNN 输出 logits，二元项 $\psi_p$ 鼓励"颜色相近、空间相邻"的像素同标签。CRF 能把粗糙的 CNN 输出"细化"到接近像素级，但**推理慢**，到 DeepLabv3 就被抛弃了。

### 6.3 ASPP（Atrous Spatial Pyramid Pooling）

DeepLabv2 引入、v3 完善。核心想法：**用多个不同 dilation rate 的并行分支捕捉多尺度上下文**。

$$
\text{ASPP}(x) = \text{Concat}\big(\text{Conv}_{1\times 1}(x),\ \text{Conv}_{3\times 3, r=6}(x),\ \text{Conv}_{3\times 3, r=12}(x),\ \text{Conv}_{3\times 3, r=18}(x),\ \text{GlobalPool}(x)\big)
$$

最后一支是 **image-level pooling**（v3 加的），把全局上下文也喂进来。物理直觉：**"近看一个点 + 远看一大片"是多尺度感知的标配**。

PyTorch 实现一个简化版 ASPP（去掉 BN 是为了少占行数，工程上当然要加）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, dilation=1):
        super().__init__()
        if k == 1:
            padding = 0
        else:
            padding = dilation                                # 保持分辨率
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ASPP(nn.Module):
    """简化版 ASPP：1x1 + 3x3 r=6/12/18 + 全局池化。"""
    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()
        self.b0 = _ConvBNReLU(in_channels, out_channels, k=1)
        self.b1 = _ConvBNReLU(in_channels, out_channels, k=3, dilation=6)
        self.b2 = _ConvBNReLU(in_channels, out_channels, k=3, dilation=12)
        self.b3 = _ConvBNReLU(in_channels, out_channels, k=3, dilation=18)
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(in_channels, out_channels, k=1),
        )
        self.project = nn.Sequential(
            _ConvBNReLU(out_channels * 5, out_channels, k=1),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [self.b0(x), self.b1(x), self.b2(x), self.b3(x)]
        gp = self.b4(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(gp)
        return self.project(torch.cat(feats, dim=1))         # (N, 256, H, W)
```

DeepLabv3+ 在 v3 的基础上加了**轻量级 decoder**，把 Xception backbone 输出的低层特征 concat 进来，进一步锐化边界。

## 七、分割里的 Transformer：SegFormer 与 Mask2Former

CNN 的根本瓶颈：**感受野受限于层数 / dilation**，且**缺乏全局注意力**。Transformer 的 self-attention 天然全局，于是"分割 Transformer"在 2021 年后爆发。

### 7.1 SegFormer（Xie et al. 2021）

SegFormer 设计简洁、效果好，工程友好：

- **层次化 Transformer encoder**：用 Mix Transformer (MiT) 系列，把图分成 4 个 stage，每个 stage 输出 stride 4 / 8 / 16 / 32 的特征，类似 ResNet 的多尺度。
- **轻量 MLP decoder**：4 个 stage 的特征先 `PatchEmbed` 到同一维度，concat 后用几个 MLP 融合，最后上采样 4 倍出 logits。

物理直觉：层次化 encoder 让不同 stage 学不同尺度的语义（浅层局部、深层全局），decoder 把它们"对齐"到同一空间再融合——本质是 **CNN FPN 的 Transformer 版本**。

SegFormer-B0 在 Cityscapes 上跑得飞快且 mIoU 接近 76%，是边缘部署的常见选择。

### 7.2 Mask2Former（Cheng et al. 2022）

Mask2Former 是当前最通用的"分割大一统"框架，**同一套架构同时支持语义、实例、全景三种分割**。核心组件：

- **像素级特征** + **Transformer 风格的 mask attention**：query 是一个可学习的 mask embedding。
- **Masked attention**：让 decoder 的 cross-attention 只在当前 mask 预测区域内做，**强制 query 专注于自己负责的区域**。
- **Hungarian matching**：在训练时用最优匹配把预测 query 与真值实例一一对齐，得到 multi-task loss。

物理直觉：**不再预测"每个像素属于哪一类"，而是预测"每个 query 负责画哪一片"**。query 之间的角色完全由匹配动态决定，所以**同一架构可以统一三种任务**。

Mask2Former 在 COCO panoptic 上 57.8 PQ，是 2022–2024 年各大 benchmark 的常胜将军。

## 八、损失函数：CE、Focal、Dice

### 8.1 Cross-Entropy（逐像素）

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{HW}\sum_{i,j}\sum_{c} y_{i,j,c}\log \hat{p}_{i,j,c}
$$

直观：每个像素独立做 softmax + NLL。**问题是类别严重不均衡时，背景像素主导 loss**。

### 8.2 Focal Loss（Lin et al. 2017，RetinaNet）

对易分类样本降权，让模型专注于难样本：

$$
\mathcal{L}_{\text{Focal}} = -\alpha_c (1 - \hat{p}_{i,j,c})^{\gamma} \log \hat{p}_{i,j,c}
$$

$\gamma = 2$ 时易样本权重衰减到约 $1/100$，效果显著。语义分割里通常 $\gamma = 2$。

### 8.3 Dice Loss（医疗影像标配）

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum \hat{p}_{i,j,c} \cdot y_{i,j,c}}{\sum \hat{p}_{i,j,c} + \sum y_{i,j,c} + \epsilon}
$$

直接优化 IoU 的近似，对**前景/背景极不均衡**的小目标特别有效。实践中常用 **CE + Dice** 组合：

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{Dice}}
$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss：对每个类别独立算，可与 CE 叠加。"""
    def __init__(self, smooth: float = 1.0, include_bg: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_bg = include_bg

    def forward(self, logits, target):
        # logits: (N, C, H, W); target: (N, H, W) int64
        prob = F.softmax(logits, dim=1)
        n_classes = prob.size(1)
        target_oh = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (prob * target_oh).sum(dims)
        union = (prob + target_oh).sum(dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        if not self.include_bg:
            loss = loss[1:]
        return loss.mean()
```

## 九、训练与推理的工程细节

- **输入分辨率**：常用 $512 \times 512$ 或 $768 \times 768$。Cityscapes 用 $1024 \times 2048$ 训练；ADE20K 用 $512 \times 512$。
- **数据增强**：随机水平翻转、随机缩放（0.5–2.0）、随机裁剪、色彩抖动。医学影像还会有弹性形变（elastic deformation）。
- **OHEM（在线难例挖掘）**：对 CE loss 排序，只取 top-k 像素回传，能显著改善小目标分割。
- **滑窗推理**：大图（如遥感）训练用 $512$，推理用滑窗 + overlap + 加权融合。
- **TTA（Test-Time Augmentation）**：原图 + 水平翻转各跑一次，再平均 logits，一般能涨 0.5–1.5 mIoU。
- **混合精度**：分割模型基本都吃显存，`torch.cuda.amp` 是必备。

## 十、未来方向

- **大模型预训练**：SAM（Segment Anything, Meta 2023）用 11 亿 mask 训练 ViT-H/16，能 zero-shot 分割几乎任何东西。
- **视频分割**：把时序信息也融进 attention，处理运动模糊与遮挡。
- **开放词汇分割（Open-Vocabulary）**：用 CLIP 文本 embedding 当分类器，类别数不再固定。
- **高效分割**：Mobile-friendly 的 STDC、TopFormer、SeaFormer 在边缘设备跑 100+ FPS。

## 十一、常用数据集一览

| 数据集 | 类别数 | 训练规模 | 主要用途 |
|---|---|---|---|
| PASCAL VOC 2012 | 21 | 10K | 老牌 benchmark |
| Cityscapes | 19 | 5K（精细）| 城市驾驶 |
| ADE20K | 150 | 20K | 场景理解，Swin/SegFormer 的标准训练集 |
| COCO-Stuff | 171 | 118K | stuff + things |
| Mapillary Vistas | 66 | 18K | 高分辨率街景 |
| BDD100K | 40 | 100K | 驾驶 + 天气 |
| LoveDA | 7 | 5K | 跨域遥感 |

物理直觉：**数据集决定上限**——Cityscapes 训练出的模型在跨城市迁移时掉 mIoU 经常 > 10%，数据集 bias 是分割领域长期难题。

## 小结

语义分割的核心是"逐像素分类"，但真正决定效果的是**感受野、多尺度、上下文融合**三件事：

- **FCN** 用 1×1 conv + 上采样完成"分类到分割"的范式转换；
- **U-Net** 的对称 encoder-decoder + skip 是中小数据下的默认答案；
- **DeepLab** 用空洞卷积 + ASPP 在不缩小分辨率的前提下扩大感受野；
- **SegFormer / Mask2Former** 把 Transformer 引入分割，前者轻量高效，后者统一了语义/实例/全景三种任务；
- 损失函数层面，**CE + Dice** 仍是大部分场景的最优组合。

下一篇我们将进入"实例分割 + 全景分割"——不仅要知道"是什么"，还要知道"是哪一只"。

## 附：参考资源

- **Segmentation Models PyTorch**：Pavel Yakubovskiy 维护的轻量库，30+ 编码器 + 5+ 解码器一键组合。
- **MMSegmentation**（OpenMMLab）：覆盖 FCN / PSPNet / DeepLab / SegFormer / Mask2Former，配置化训练，工业标准。
- **Detectron2**：Facebook Research 官方库，语义 / 实例 / 全景分割统一支持。
- **Hugging Face Transformers**：`SegformerForSemanticSegmentation` 等模型一行加载。
- **Cityscapes 官方 benchmark**：脚本化评估 mIoU / per-class IoU，论文必备。
- **torch.optim.lr_scheduler**：CosineAnnealingLR 是分割训练的最稳选择。
