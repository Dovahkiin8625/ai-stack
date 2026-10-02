# CNN 演进之路：从 LeNet 到 ResNet 的架构与思想

卷积神经网络（Convolutional Neural Network, CNN）是过去十年计算机视觉领域最具影响力的架构。从 1998 年 Yann LeCun 提出的 LeNet-5 手写数字识别，到 2012 年 AlexNet 在 ImageNet 上一举夺魁，再到 2015 年 ResNet（Residual Network, 残差网络）突破 152 层深度，CNN 用一条清晰的演进线告诉我们：**更好的视觉特征 = 合理的归纳偏置（inductive bias）+ 更深的网络 + 更稳定的优化**。本文沿时间线梳理 LeNet、AlexNet、VGGNet、ResNet 及其变体，配以 PyTorch 从零实现，让你在读完之后既能"看清历史"，也能"写出一个能跑的 ResNet"。

## 一、为什么 CNN 适合图像

图像具有三个核心性质：局部相关性、平移不变性、层次化结构。CNN 恰好把这三个性质"硬编码"进了架构本身：

- **局部连接（Local Connectivity）**：单个神经元只看输入的一小块区域（例如 $3\times3$），大幅减少参数。
- **权值共享（Weight Sharing）**：同一个卷积核（kernel / filter）在整张图上滑动，参数数量与图像大小无关。
- **池化（Pooling）**：通过下采样（downsampling）聚合邻域信息，使特征对小幅平移鲁棒。

设输入为 $X \in \mathbb{R}^{H \times W \times C_{in}}$，卷积核为 $K \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$，则二维卷积的输出特征图 $Y$ 在空间位置 $(i,j)$、输出通道 $c$ 上的值为：

$$
Y_{i,j,c} = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1}\sum_{c_{in}=0}^{C_{in}-1} K_{u,v,c_{in},c} \cdot X_{i+u, j+v, c_{in}} + b_c
$$

物理直觉：卷积核就是一个"滑动的模式探测器"——边缘检测核在经过物体的轮廓时会产生高响应。下游的更深层会把这些边缘组合成纹理、部件，直到最终的语义概念。

## 二、LeNet-5（1998）：CNN 的奠基之作

LeNet-5 由 Yann LeCun 在 1998 年提出，用于手写数字识别（MNIST），其结构是"卷积-池化-卷积-池化-全连接"的经典范式：

| 层 | 类型 | 输出尺寸 | 参数量 |
| --- | --- | --- | --- |
| C1 | Conv $5\times5$, 6 filters | $28\times28\times6$ | 156 |
| S2 | Avg Pool $2\times2$ | $14\times14\times6$ | 0 |
| C3 | Conv $5\times5$, 16 filters | $10\times10\times16$ | 2,416 |
| S4 | Avg Pool $2\times2$ | $5\times5\times16$ | 0 |
| C5 | Conv $5\times5$, 120 filters | $1\times1\times120$ | 48,120 |
| F6 | Fully Connected | 84 | 10,164 |
| Output | Fully Connected | 10 | 850 |

三个关键设计沿用至今：

1. **卷积 + 池化交替**：先用卷积提取局部模式，再用池化压缩空间维度，扩大后续层的感受野（receptive field）。
2. **tanh / sigmoid 激活**：早期 CNN 的标配，后被 ReLU 取代。
3. **稀疏连接 + 权值共享**：参数量比同等表达能力的全连接网络少一个数量级。

## 三、AlexNet（2012）：深度学习的"破冰船"

2012 年，Alex Krizhevsky 等人用 AlexNet 在 ImageNet ILSVRC 比赛上把 top-5 错误率从 26% 降到 15.3%，相当于一次"范式跃迁"。它的成功不是单一技术，而是多项工程创新的叠加：

1. **ReLU 激活**：用 $\text{ReLU}(x) = \max(0, x)$ 替代 tanh/sigmoid，缓解梯度消失，训练速度提升约 6 倍。
2. **GPU 训练**：把网络拆分到两块 GTX 580 显卡上并行训练，开启了"GPU + 深度学习"时代。
3. **Dropout**：在最后两个全连接层以 $p=0.5$ 的概率随机丢弃神经元，缓解过拟合。
4. **数据增强**：随机裁剪、水平翻转、PCA 颜色扰动，把数据集"扩大"成原来的 2048 倍。
5. **Local Response Normalization (LRN)**：对相邻通道做归一化（后续被 BN 取代）。

AlexNet 还首次使用了重叠池化（stride < kernel size），让特征的"颗粒度"更细。

## 四、VGGNet（2014）：小卷积核的胜利

VGG（Visual Geometry Group）网络来自牛津大学的 Karen Simonyan 和 Andrew Zisserman。它论证了一个简单却深刻的事实：**用 $3\times3$ 小卷积核堆叠，能用更少参数达到与大卷积核相当的感受野**。

直观推导：两个 $3\times3$ 卷积堆叠的等效感受野是 $5\times5$，参数从 $5^2 \cdot C^2 = 25C^2$ 降到 $2 \cdot 3^2 \cdot C^2 = 18C^2$，而非线性次数还多了一次。三个 $3\times3$ 等效 $7\times7$，参数从 $49C^2$ 降到 $27C^2$。

VGG-16 的结构非常规整，全部使用 $3\times3$ 卷积和 $2\times2$ 最大池化，每经过一个 stage 通道数翻倍（64 → 128 → 256 → 512），空间尺寸减半。它赢得了 2014 年 ILSVRC 分类亚军（冠军是 GoogLeNet）和定位冠军。

但 VGG 的问题也显而易见：**参数量大**（VGG-16 约 138M，大部分集中在全连接层）、**显存占用高**、**训练慢**。这促使研究者探索更高效的架构——Inception、ResNet 等。

## 五、ResNet（2015）：用残差学习打破深度诅咒

理论上，深网络应当至少与浅网络等价——把多余的层设为恒等映射（identity mapping）即可。但实践中，**单纯加深网络反而让训练误差变大**，这就是著名的"网络退化"（degradation）问题：不是过拟合，而是优化器难以拟合恒等映射。

何恺明等人给出的答案是**残差学习（residual learning）**：让每层不再直接拟合目标 $H(x)$，而是拟合残差 $F(x) = H(x) - x$，最终输出为：

$$
H(x) = F(x) + x
$$

对应到网络结构上，就是**快捷连接（shortcut connection）**——把输入 $x$ 直接加到卷积输出上：

$$
y = F(x, \{W_i\}) + x
$$

其中 $F(x, \{W_i\})$ 是若干卷积 + 归一化 + 激活的复合函数。当残差 $F(x)=0$ 时，就是恒等映射——网络只需"什么都不做"就能保持不退化。

### 5.1 残差块的两种形态

**BasicBlock**（ResNet-18/34）：

$$
y = \text{ReLU}\!\left(\text{BN}\!\left(W_2 \cdot \text{ReLU}\!\left(\text{BN}\!\left(W_1 x\right)\right)\right) + x\right)
$$

**Bottleneck**（ResNet-50/101/152）：先用 $1\times1$ 卷积降维，再用 $3\times3$ 卷积，最后用 $1\times1$ 卷积升维，整体计算量更小：

$$
y = W_1 \cdot \text{ReLU}\!\left(\text{BN}\!\left(W_2 \cdot \text{ReLU}\!\left(\text{BN}\!\left(W_3 \cdot \text{ReLU}\!\left(\text{BN}(W_1 x)\right)\right)\right)\right)\right) + x
$$

### 5.2 为什么残差能缓解梯度消失

反向传播时，$\frac{\partial \mathcal{L}}{\partial x}$ 经残差块有两条路径：一条穿过卷积（容易梯度消失），另一条直接穿过 shortcut（恒等）。求和之后：

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}\left(\frac{\partial F}{\partial x} + 1\right)
$$

那个 "+1" 让梯度多了一条"高速公路"，即使卷积路径上梯度接近 0，shortcut 也能把梯度无损地传回去。

### 5.3 ResNet 的 PyTorch 实现

下面是从零实现 BasicBlock 与 ResNet-18 的完整代码：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet-18/34 使用的基础残差块（无 bottleneck）。"""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 当维度不匹配时，用 1x1 卷积调整 shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    model = ResNet18(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    print("output:", model(x).shape)   # torch.Size([2, 1000])
```

注意 `shortcut` 的处理：当 stride 不为 1 或通道数变化时，必须用一个 $1\times1$ 卷积把 $x$ 投影到目标维度，否则 `out + identity` 会因尺寸不匹配而报错。

## 六、Batch Normalization：让深层网络"住得舒服"

在 ResNet 之前，深度网络训练时每一层的输入分布都会随参数更新而漂移（internal covariate shift），迫使学习率必须设得很小。**Batch Normalization（BN, 批量归一化）**在 mini-batch 维度上对每个特征通道做归一化：

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta
$$

其中 $\mu_{\mathcal{B}}$ 和 $\sigma_{\mathcal{B}}^2$ 是当前 mini-batch 在该通道上的均值和方差，$\gamma$、$\beta$ 是可学习的缩放与偏移。BN 的三大好处：

1. **允许更大学习率**：归一化后梯度更稳定，训练速度大幅提升。
2. **正则化效果**：mini-batch 统计量引入噪声，相当于轻度 Dropout。
3. **缓解梯度消失/爆炸**：把激活值拉回到近似零均值单位方差区间。

BN 的副作用是**依赖 batch 大小**——batch 过小时统计量不准，过小时效果退化甚至崩溃，这也是后续 LayerNorm、GroupNorm 等替代方案的动因。

## 七、ResNet 变体与现代化

ResNet 之后，研究者从三个方向继续推进：更宽（width）、更密（dense）、更高效。

### 7.1 Wide ResNet（Zagoruyko & Komodakis, 2016）

把通道数扩大 2-4 倍（例如 WRN-50-2），同时减少深度，发现**宽度比深度更划算**——在 CIFAR 上用更少训练时间达到更高精度。

### 7.2 ResNeXt（Xie et al., 2017）

引入"基数"（cardinality）——即分组卷积的组数 $C$。单个瓶颈块内部做 $C$ 路并行卷积再求和。ResNeXt-50 在参数量相近的情况下比 ResNet-50 精度更高。

$$
y = x + \sum_{i=1}^{C} \mathcal{T}_i(x)
$$

其中 $\mathcal{T}_i$ 是同构的变换分支。

### 7.3 DenseNet（Huang et al., 2017）

极致版的 shortcut：每一层都接收前面所有层的特征图作为输入（通过通道拼接）。它在参数量更小的前提下实现了更强的特征复用，缓解了 vanishing-gradient。

$$
x_\ell = H_\ell([x_0, x_1, \dots, x_{\ell-1}])
$$

$[.,.,.]$ 表示沿通道维度的拼接。DenseNet 在小数据集上很受欢迎，但显存占用较大。

### 7.4 ResNet 的现代化改造

2020 年代的研究把 ResNet 的若干超参数按 Transformer 的设计哲学改造，得到 **ResNet-50 + 现代化配方**：

- 把 $7\times7$ stem 改成三组 $3\times3$ 卷积。
- 用 depthwise $3\times3$ + $1\times1$ 替换 bottleneck（参考 MobileNetV2）。
- 替换 ReLU 为 SiLU / GELU。
- LayerNorm 替代 BatchNorm（对 batch size 不敏感）。

这套改造直接启发了后来的 **ConvNeXt**——"用纯卷积复刻 Transformer 的设计哲学"。

## 八、训练 ResNet 的工程经验

光有架构还不够，把 ResNet 在自定义数据集上训好还需要若干工程细节。下面这些经验来自工业界多年实践。

### 8.1 数据预处理与增强

- **输入尺寸**：ImageNet 预训练时常用 $224\times224$，推理时可用 $256, 320, 384$ 等多尺度测试。更大尺寸通常带来 1-2% 的精度提升，代价是 FLOPs 平方级增长。
- **训练增强**：随机裁剪 + 水平翻转是基线；RandAugment、Mixup、Cutmix 在 ResNet 上同样有效，但收益不如 ViT 那么显著。
- **归一化**：ImageNet 的均值 $(0.485, 0.456, 0.406)$、标准差 $(0.229, 0.224, 0.225)$——这一对值已成为事实标准。

### 8.2 优化器与学习率

- **SGD + Momentum**：$\text{lr}=0.1$（batch=256 起步）、$\text{momentum}=0.9$、$\text{weight decay}=1e-4$。在多个 GPU 上按线性规则缩放学习率。
- **AdamW**：ViT 训练更常用，但 ResNet 用 AdamW 也完全可行。学习率通常设成 SGD 的 $\frac{1}{10}$。
- **Cosine Decay**：训练 ResNet-50 on ImageNet（90 epoch）时的标准 schedule 是先 warm-up 5 epoch，然后 cosine 衰减到 0。
- **Warm-up**：前几百个 step 用很小的学习率，避免 BN 统计量在初始化阶段剧烈波动。

### 8.3 BN 的工程注意事项

- **train / eval 切换**：`model.train()` 时 BN 用 mini-batch 统计量；`model.eval()` 时用累计的 moving average。忘了切 eval 是新手最常见的 bug。
- **batch size**：BN 在 batch < 16 时统计量噪声过大，精度明显下降。GroupNorm / LayerNorm 是不依赖 batch 的替代品。
- **冻结 BN**：finetune 时常常冻结前面几个 stage 的 BN（`param.requires_grad=False`），减少小数据集上过拟合。

下面是一段典型的 ImageNet 训练循环核心代码：

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None,
                    use_amp: bool = True):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
```

注意点：`set_to_none=True` 比 `zero_grad()` 略快；`non_blocking=True` 让 CPU→GPU 传输与计算重叠；`GradScaler` 在 FP16 训练时避免数值下溢。

## 九、可视化：理解 CNN 学到了什么

很多人以为"训练完就完事了"，但可视化能让模型从"黑盒"变成"可以解释的工具"。

### 9.1 第一层卷积核

ResNet 的 stem 是一个 $7\times7$ 卷积，可以直接可视化。训练有素的核会呈现出"边缘检测器"的模样——水平、垂直、对角线、颜色对比。这是因为第一层只需要提取最基础的低级特征。

### 9.2 Grad-CAM

对于分类问题，Grad-CAM 通过计算最后一层特征图对目标类别的梯度，得到一张"热力图"：

$$
\alpha_k^c = \frac{1}{HW}\sum_{i,j}\frac{\partial y^c}{\partial A_{i,j}^k},\quad
L_{\text{Grad-CAM}}^c = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)
$$

物理直觉：$\alpha_k^c$ 表示通道 $k$ 对类别 $c$ 的"重要程度"，加权求和后得到的热力图就标出了图像中"对分类贡献最大的区域"。Grad-CAM 在医疗影像、可解释 AI 中应用极广。

### 9.3 特征图演化

把 ResNet-50 的四个 stage 输出可视化，可以看到层级化特征的清晰过程：

- Stage 1：边缘、纹理、颜色；
- Stage 2：重复的纹理 pattern、简单部件；
- Stage 3：物体部件（车轮、动物头部）；
- Stage 4：完整语义（"狗"、"汽车"）。

这种"由低到高"的特征组装正是 CNN 强大表示能力的来源，也是它与 ViT 全局特征的关键区别。

## 十、ResNet 在下游任务的应用

ResNet 不仅用于分类，更常常作为下游任务的 backbone。

### 10.1 目标检测

Faster R-CNN 用 ResNet 替换 VGG 后，mAP 显著提升。后来的 Mask R-CNN、Cascade R-CNN 都默认使用 ResNet-50 / ResNeXt-101。检测任务需要多尺度特征图，因此常用 **FPN（Feature Pyramid Network）** 在 ResNet 的四阶段输出上构建自顶向下的特征金字塔。

### 10.2 语义分割

DeepLab 系列把 ResNet 的最后几层改成 atrous（空洞）卷积，保持分辨率的同时扩大感受野。`output_stride=8`（输出是输入的 1/8）配合 ASPP（Atrous Spatial Pyramid Pooling）是标准配置。

### 10.3 自监督预训练

SimCLR、MoCo v2、BYOL 等对比学习方法几乎都用 ResNet-50 作为 backbone。学到的表示在下游分类、检测、分割上迁移性极强。直到 MAE、DINO 出现，ViT 才在自监督领域反超 CNN。

## 十一、常见疑问与误区

最后列出几个初学者经常踩的坑：

1. **"ResNet 越深越好"**：不一定。在 ImageNet-1K 上，ResNet-152 的精度略高于 ResNet-50，但提升有限而计算成本翻倍；很多场景下 ResNet-50 才是性价比最优解。
2. **"BN 在推理时和训练时一样"**：错。推理时 BN 使用 moving average 的全局统计量，而不是当前 batch 的统计量。这是 `.eval()` 必须正确调用的根本原因。
3. **"残差连接只为了解决梯度消失"**：过于简化。残差连接还把"学什么"从"完整映射"简化为"残差"，让优化景观更平滑，这才是它强大的根源。
4. **"下采样只能用 max pooling"**：ResNet 在某些下采样位置用 stride=2 的卷积代替 max pooling，让网络自己学"如何降采样"。

## 十二、ResNet 之后：CNN 还有什么可挖？

尽管 Vision Transformer 抢走了大量注意力，但 CNN 在过去几年并没有停滞。几个值得关注的演进方向：

### 12.1 RepLKNet：超大卷积核的复兴

2022 年的 RepLKNet 证明：在现代训练配方下，把深度可分离卷积核做到 $31\times31$ 甚至 $51\times51$，性能反而更好。这与"大卷积核 ≈ 大感受野"的直觉一致——ViT 用 attention 实现的事，CNN 也可以用大卷积核实现，且计算更便宜。

### 12.2 ConvNeXt v2 与 MAE-CNN

Meta 把 MAE（Masked Autoencoder）预训练范式搬到 ConvNeXt 上，得到 ConvNeXt v2。在 ImageNet-1K 上用 MAE 预训练后，FCMAE 头微调，全量 ConvNeXt-B 达到 87% top-1。这证明了 CNN 在自监督时代仍能保持竞争力。

### 12.3 EfficientNet 系列

Google 的 EfficientNet 提出**复合缩放**（compound scaling）：同时按特定比例缩放深度、宽度、分辨率。EfficientNet-V2 进一步引入训练感知的 NAS（神经架构搜索），把 MBConv 与 Fused-MBConv 混合，在精度和速度之间取得极好的平衡。

### 12.4 MobileNet 与端侧部署

MobileNetV3 / MobileOne 把深度可分离卷积、轻量注意力（SE 模块）、结构重参数化结合，是移动端推理的事实标准。RepVGG 的"训练多分支、推理单分支"技巧也在工业界广泛使用。

## 十三、ResNet 的精神遗产

即使今天我们用 ViT、ConvNeXt、Swin 处理大部分视觉任务，ResNet 的精神遗产仍在影响每一个新架构：

1. **残差连接（skip connection）几乎是所有现代架构的标配**——ViT、ConvNeXt、Swin 都有它。
2. **层次化设计**：把网络分成 4 个 stage，每个 stage 内部结构相似，stage 间降采样——这是几乎所有视觉 backbone 的"标准模板"。
3. **BN / LN 归一化**：现代 Transformer 普遍使用 LayerNorm，但其动机与 BN 几乎一致——让梯度在深层网络中稳定流动。
4. **End-to-End 训练**：ResNet 之后，分类、检测、分割都尽量端到端训练，手工设计特征 + SVM 的时代彻底结束。

### 13.5 一个完整 ResNet 训练脚本示例

把上面提到的所有要点串成一个最小可用的训练脚本，方便上手实验：

```python
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. 数据：ImageNet 风格的数据增强
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 2. 模型：从 torchvision 加载预训练 ResNet-50，替换分类头
model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.cuda()

# 3. 优化器 + 余弦退火
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                             momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=90)

# 4. 训练循环（伪代码，省略 loader / save 等细节）
for epoch in range(90):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        loss = nn.functional.cross_entropy(model(imgs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
```

注意：`weights="IMAGENET1K_V2"` 是 torchvision 1.13+ 的新写法，比旧的 `pretrained=True` 更清晰。`IMAGENET1K_V2` 是 ResNet-50 的改进版预训练权重（80.4% top-1），比 V1（76.1%）强 4 个点。

## 十四、结语：从 CNN 的视角理解"什么是好的视觉模型"

ResNet 之所以成为分水岭，不仅因为它的精度，更因为它提供了一个思考视觉模型的统一框架：**通过残差连接 + BN + 层次化设计，把"深度学习的优化困难"与"视觉特征的层次化"两个问题同时解决**。后续的几乎所有视觉模型——无论是 ConvNeXt 的"现代化卷积"、Swin 的"层次化 attention"、MAE 的"视觉 BERT"——都在这个框架上做加减法。

理解 ResNet，等于理解了视觉深度学习的"语法"；剩下的就是学习如何在这个语法上写更好的"句子"。

## 小结

从 LeNet 到 ResNet，CNN 的演进有三条清晰的主线：

1. **更深的网络**：从 7 层到 152 层。残差学习是关键一跃，它让"加深网络"从"增加过拟合风险"变成"提升表达能力的捷径"。
2. **更优的归纳偏置**：小卷积核堆叠（VGG）、BN 归一化、瓶颈结构（Bottleneck）共同把"能不能训"变成"训得好"。
3. **更高效的连接**：ResNeXt 的分组、DenseNet 的密集连接，都在用更廉价的"信息通路"换取特征复用。

理解 ResNet 的残差思想，是理解后续 Vision Transformer、Swin Transformer、ConvNeXt 等现代视觉架构的钥匙——它们本质上都在问同一个问题：**"如何让信号在很深的网络中顺畅流动？"** 下一篇我们将看到 Transformer 如何用 attention 给这个问题另一种回答。
