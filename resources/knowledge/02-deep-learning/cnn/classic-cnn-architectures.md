# 经典 CNN 架构：从 LeNet 到 ResNet

从 1998 年的 LeNet 到 2015 年的 ResNet，八年时间 CNN 在 ImageNet 上的 top-5 错误率从 26% 降到 3.57%——首次低于人类水平（约 5%）。本文按时间顺序梳理五座里程碑：LeNet-5、AlexNet、VGG、GoogLeNet、ResNet，讲清楚每一代"解决了什么问题、带来了什么新机制"，最后给出一个 ImageNet 性能对比表与 ResNet 迁移学习的 PyTorch 示例。

## 一、为什么需要"架构"——CNN 演进的两条主线

CNN 架构不是凭空变深的，**两条主线**贯穿始终：

1. **更深更大**：2012 年 AlexNet 用两块 GPU 训出 8 层；2014 年 VGG 把网络推到 16/19 层；2015 年 ResNet 直接突破 100 层。深度带来更强的表达能力。
2. **更聪明的连接**：GoogLeNet 用并行的多尺度卷积（Inception 模块）、ResNet 用残差连接绕开深度瓶颈、MobileNet 用深度可分离卷积压缩计算量。

回顾这段历史，你会发现**深度学习的很多"标准技巧"都是被某一篇经典论文首次提出并证明有效的**——理解它们有助于读现代论文时知道"这个 trick 最早从哪来"。

## 二、LeNet-5（1998）：CNN 的开山之作

LeCun 等人 1998 年提出 LeNet-5，用于手写数字识别（MNIST），是 CNN 第一次真正落地。它确立了"**卷积 + 池化 + 全连接**"的基本范式。

```text
输入 (32x32x1)                  C1: 5x5 conv, 6 通道   →  (28x28x6)
   ↓                                          ↓
   ↓                                  S2: 2x2 avg pool      →  (14x14x6)
   ↓                                          ↓
   ↓                                  C3: 5x5 conv, 16 通道  →  (10x10x16)
   ↓                                          ↓
   ↓                                  S4: 2x2 avg pool      →  (5x5x16)
   ↓                                          ↓
   ↓                                  C5: 5x5 conv, 120 通道 →  (1x1x120)
   ↓                                          ↓
   ↓                                  F6: 全连接 84 神经元
   ↓                                          ↓
输出                                输出: 10 类 softmax
```

关键创新：

- 交替使用 **卷积 + 平均池化**（当时还没有 ReLU，用的是 sigmoid / tanh）。
- 最后用全连接层输出分类。
- MNIST 上达到 ~99.2% 准确率——为后来所有视觉模型奠基。

但 LeNet 也有局限：网络太浅（5 层可学习层）、激活函数饱和、没有 GPU 训练、没有数据增强、只能跑小图（32×32 灰度）。

## 三、AlexNet（2012）：深度学习复兴的号角

Krizhevsky 等人 2012 年用 AlexNet 把 ImageNet top-5 错误率从 26% 降到 16.4%——**远超第二名**（传统视觉方法）。这是深度学习真正进入主流的转折点。

```text
输入 (227x227x3)
   ↓
Conv 11x11, stride 4, 64 通道    →  (55x55x64)
   ↓
MaxPool 3x3, stride 2            →  (27x27x64)
   ↓
Conv 5x5, padding 2, 192 通道    →  (27x27x192)
   ↓
MaxPool 3x3, stride 2            →  (13x13x192)
   ↓
3× Conv 3x3, padding 1, 384/256/256  →  (13x13x256)
   ↓
MaxPool 3x3, stride 2            →  (6x6x256)
   ↓
FC 4096 → FC 4096 → FC 1000 (softmax)

共 8 层可学习：5 conv + 3 fc，参数约 60M
```

AlexNet 的关键创新：

1. **ReLU 激活**：解决了 sigmoid 在深网络中的梯度饱和，让训练深网络成为可能。
2. **Dropout**（$p=0.5$）：全连接层随机丢弃 50% 神经元，缓解过拟合。
3. **GPU 训练**：用两块 GTX 580（各 3GB 显存）并行，把训练时间从 CPU 的几周压到几天。
4. **数据增强**：随机裁剪、水平翻转、PCA 颜色增强，相当于免费扩大训练集。
5. **Local Response Normalization（LRN）**：局部神经元互相抑制，今天基本已被 BatchNorm 取代。

**历史地位**：自 AlexNet 起，几乎所有视觉任务都开始用 GPU + CNN。

## 四、VGG（2014）：小核堆叠的胜利

VGG（Visual Geometry Group, Oxford）的洞察：**两个 3×3 卷积堆叠的感受野等于一个 5×5，但参数更少、非线性更多**。

参数对比：

$$
5\times5 \text{ conv on C channels}: \quad 25 C^2 \text{ params}
\text{两个 } 3\times3 \text{ conv on C channels}: \quad 2 \times 9 C^2 = 18 C^2 \text{ params}
$$

3 个 3×3 卷积 = 一个 7×7，但参数只有 $27C^2$ vs $49C^2$，还多两次非线性激活。

```text
VGG-16 结构（每段 conv 数量, 输出通道）:
  Block1: 2× [Conv3-64]      → MaxPool    (224→112)
  Block2: 2× [Conv3-128]     → MaxPool    (112→56)
  Block3: 3× [Conv3-256]     → MaxPool    (56→28)
  Block4: 3× [Conv3-512]     → MaxPool    (28→14)
  Block5: 3× [Conv3-512]     → MaxPool    (14→7)
  FC:    4096 → 4096 → 1000

共 16 层可学习，参数约 138M
```

VGG 的优点是**结构简洁、规整**，全是 3×3 conv + 2×2 max pool，几乎成了深度学习课程的"教学网络"。缺点是参数太多（138M）、计算贵、显存占用大——后来很多架构都在想办法"用更少的参数达到 VGG 的精度"。

## 五、GoogLeNet / Inception（2014）：多尺度并行

Google 的 Szegedy 等人提出了 **Inception 模块**——在同一层里**并行**用 1×1、3×3、5×5 卷积和 3×3 池化，再把结果沿通道拼起来。这样网络能"自动选择"合适尺度的特征。

```text
Inception v1 模块（朴素版）:
                                 输入 (28x28x256)
                                       │
        ┌──────────┬──────────┬────────┴────────┐
        │          │          │                 │
   Conv 1x1     Conv 3x3    Conv 5x5        MaxPool 3x3
   (32 通道)    (32 通道)   (32 通道)        (32 通道)
        │          │          │                 │
        └──────────┴──────────┴────────┬────────┘
                                       │
                                  filter concat
                                       │
                                 输出 (28x28x128)
```

但朴素版的 5×5 卷积在 256 通道输入上参数爆炸（$5^2 \times 32 \times 256 = 204800$）。**关键改进**：在 3×3/5×5 卷积之前先做 **1×1 卷积降维**。

```text
带降维的 Inception 模块:

           ┌──── Conv 1x1 (32) ────┐
           │                       │
   Conv 1x1 (32)             Conv 1x1 (16) → Conv 3x3 (128)   ← 降维后 5×5 更便宜
           │                       │
           │             Conv 1x1 (16) → Conv 5x5 (32)        ← 降维后 5×5 更便宜
           │                       │
           └──── MaxPool 3x3 → Conv 1x1 (32) ────┘
                                       │
                                  filter concat
```

GoogLeNet 共 22 层（含辅助分类头），但参数只有约 5M——比 AlexNet 还少。**1×1 卷积**也由此成为 CNN 的标准工具，既能降维又能加非线性。

## 六、ResNet（2015）：残差连接与"深度天花板"的突破

单纯堆叠更多层会遇到**退化问题（degradation）**：训练误差先降后升，20 层之后网络反而比浅层更难训——不是过拟合，是优化难度本身在增长。

He 等人 2015 年的核心洞察：**让网络学习"残差"$F(x) = H(x) - x$，而不是直接学 $H(x)$**。前向传播变成：

$$
y = F(x, \{W_i\}) + x
$$

其中 $F$ 是两到三个卷积层堆叠，$x$ 是恒等映射（**shortcut / skip connection**）。

```text
Residual Block（残差块）:

        x ─────────────────────────────────────┐
                         │                      │
                         ↓                      │
                    Conv 3x3, BN, ReLU         │
                         │                      │
                    Conv 3x3, BN                │
                         │                      │
                         ↓                      │
                      ReLU ←── + ───────────────┘
                          y = F(x) + x
```

直觉：如果某层"什么都不需要做"，让 $F(x) \to 0$，输出就是 $y = x$——等价于恒等映射；网络至少不会变差。反之，需要复杂变换时，$F(x)$ 自然会学出来。

ResNet-50/101/152 等变体在 ImageNet 上把 top-5 错误率推到 3.57%——**首次超越人类水平**。后续的 ResNeXt、DenseNet 等都是这条线的延伸。

```text
ResNet-50 整体结构:
  Conv 7x7, 64 通道, stride 2                →  (112x112x64)
  MaxPool 3x3, stride 2                     →  (56x56x64)
  3 × Residual Block (64→64→256, bottleneck)
  4 × Residual Block (128→128→512)
  6 × Residual Block (256→256→1024)
  3 × Residual Block (512→512×2048)
  Global Average Pool                        →  (1x1x2048)
  FC 1000
```

## 七、经典架构对比

| 网络 | 年份 | 层数 | Top-5 错误率 | 参数量 | 关键创新 |
|---|---|---|---|---|---|
| LeNet-5 | 1998 | 5 | (MNIST) | ~60K | CNN 范式奠基 |
| AlexNet | 2012 | 8 | 16.4% | 60M | ReLU + Dropout + GPU |
| VGG-16 | 2014 | 16 | 7.3% | 138M | 3×3 卷积堆叠 |
| GoogLeNet | 2014 | 22 | 6.7% | 5M | Inception + 1×1 降维 |
| ResNet-152 | 2015 | 152 | 3.57% | 60M | 残差连接 |

**趋势**：错误率快速下降，但参数量在 ResNet 后反而持平甚至下降——架构创新替代了暴力加深。

## 八、用 ResNet 做迁移学习

预训练 + 微调是工业界最常用的模式：拿 ImageNet 训好的 ResNet，去掉最后分类层，换成自己的分类头，只微调最后几层就能拿到很强的小数据精度。

```python
import torch
import torch.nn as nn
from torchvision import models

# 加载预训练 ResNet-50
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# 冻结所有卷积层参数
for param in backbone.parameters():
    param.requires_grad = False

# 替换最后的分类头：原 fc 输出 1000 类，改成 10 类
in_features = backbone.fc.in_features       # 2048
backbone.fc = nn.Linear(in_features, 10)

# 只有 backbone.fc 这 2048*10+10 个参数会训练
trainable = sum(p.numel() for p in backbone.fc.parameters())
print(f'trainable params: {trainable:,}')    # 20,490

model = backbone.to('cuda')
```

数据少时这样做又快又稳；如果数据充足，可以**解冻最后几层残差块**一起微调，精度通常更高。

```python
# 解冻最后两个 layer（layer3, layer4）一起训练
for name, param in backbone.named_parameters():
    if 'layer4' in name or 'layer3' in name or 'fc' in name:
        param.requires_grad = True
```

## 九、补充：DenseNet 与轻量化前奏

- **DenseNet（2017）**：把残差连接的"加"换成"通道拼接"，每层都直接接收前面所有层的特征，梯度流通更顺畅。
- **MobileNet v1（2017）**：提出**深度可分离卷积（depthwise separable convolution）**——把标准卷积拆成 depthwise + pointwise，大幅压缩参数量与 FLOPs，是移动端模型的标配。

这两条线直接通向现代 CNN：前者探索极致精度，后者探索极致效率。

## 小结

经典 CNN 架构的演进是一条"**加深 + 加巧**"的双线：AlexNet 用 ReLU+GPU 把网络推到 8 层，VGG 用 3×3 卷积堆到 16/19 层，GoogLeNet 用 Inception 多尺度并行把精度顶上去而不增加参数，ResNet 用残差连接解决了"深到 100+ 层还能训"的优化难题。读懂这五个里程碑，你就能读懂 90% 的现代视觉论文——ConvNeXt 是"用 ResNet 的思路回看 Transformer 的设计"、EfficientNet 是"用 NAS 搜索更优的缩放策略"，它们都建立在这一代架构奠定的地基之上。