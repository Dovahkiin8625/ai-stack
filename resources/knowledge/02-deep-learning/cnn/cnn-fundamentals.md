# 卷积神经网络基础：从卷积操作到 PyTorch 实现

卷积神经网络（CNN）是处理图像、视频等网格结构数据的标配架构。本文从「为什么 MLP 处理图像不行」出发，系统讲解卷积操作、步长、填充、池化、通道、特征层级等核心概念，最后用 PyTorch 实现一个完整的 CIFAR-10 分类网络，并对比 MLP 与 CNN 的参数规模。

## 一、为什么 MLP 在图像上力不从心

把一张 $224 \times 224 \times 3$ 的彩色图片送入 MLP（多层感知机），最简单的做法是把它**展平**（flatten）成一个长度为 $224 \times 224 \times 3 = 150528$ 的向量。如果下一层有 1024 个神经元，参数矩阵就有 $150528 \times 1024 \approx 1.5$ 亿。还没到 ImageNet 这种真实任务，模型就已经膨胀到不可接受。

更深的问题有三点：

1. **参数爆炸**。每一像素都和每个隐藏神经元全连接，输入稍大就爆炸。
2. **没有平移不变性**。同一只猫出现在图像左上角和右下角，MLP 学到的是两套完全不同的权重——必须用海量数据才能勉强覆盖所有位置。
3. **忽略局部结构**。相邻像素之间的空间关系被展平破坏，但"边缘""角点""纹理"这些视觉特征恰恰是局部像素的统计。

CNN 用三个核心思想解决这些问题：**局部连接**（每个神经元只看一小块）、**权值共享**（同一个卷积核扫遍整张图）、**池化**（逐步降低空间分辨率）。这三件事加起来，把参数从亿级降到百万级，并自然引入平移不变性。

## 二、卷积操作：滑动的小窗口

**卷积**的直观解释：一个小的权重矩阵（**卷积核 / 滤波器**）在输入图像上**滑动**，每到一处就做一次点积，得到一个输出值。所有输出值拼起来就是**特征图（feature map）**。

```text
输入 (5x5):                卷积核 (3x3):        输出 (3x3):
+----+----+----+----+----+   +----+----+----+    +----+----+----+
|  1 |  2 |  3 |  0 |  1 |   |  1 |  0 | -1 |    | ?  | ?  | ?  |
+----+----+----+----+----+   +----+----+----+    +----+----+----+
|  0 |  1 |  2 |  3 |  1 |   |  1 |  0 | -1 |    | ?  | ?  | ?  |
+----+----+----+----+----+   +----+----+----+    +----+----+----+
|  1 |  2 |  0 |  2 |  0 |   |  1 |  0 | -1 |
+----+----+----+----+----+   作用：纵向边缘检测
|  2 |  1 |  3 |  1 |  1 |   （中心列与左右两列的差）
+----+----+----+----+----+
|  1 |  0 |  1 |  2 |  3 |
+----+----+----+----+----+

左上角点积 = 1*1+2*0+3*(-1)+0*1+1*0+2*(-1)+1*1+2*0+0*(-1)
           = 1+0-3+0+0-2+1+0+0 = -3
```

数学上，离散卷积在位置 $(i, j)$ 的输出：

$$
y_{i,j} = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} w_{u,v} \cdot x_{i+u,\, j+v} + b
$$

其中 $w$ 是卷积核、$b$ 是偏置、$k$ 是核大小。**注意**：深度学习中通常把这一步叫 "convolution"，但严格说它是"互相关"（cross-correlation），不做核翻转；不影响学习结果，但读经典信号处理文献时要知道区别。

## 三、步长与填充：控制输出尺寸

输出特征图的尺寸由三个超参决定：

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1
$$

- **$k$**：卷积核大小（常用 $3\times3$ 或 $5\times5$）。
- **$s$（stride）**：步长。$s=1$ 时输出与输入同尺寸；$s=2$ 时输出尺寸减半。
- **$p$（padding）**：边界补零。$p=0$ 叫 **valid**（不做填充，输出会缩小）；$p=(k-1)/2$ 叫 **same**（输入输出同尺寸，对 $k=3, p=1$）。

实际工程里常见两个约定：

| 设定 | 输入 $32 \times 32$ + 核 $3\times3$ | 输出尺寸 | 适用场景 |
|---|---|---|---|
| valid ($p=0$) | $30 \times 30$ | 缩小 | 想要快速降采样 |
| same ($p=1$) | $32 \times 32$ | 同尺寸 | 主流，几乎所有经典网络 |

**感受野（receptive field）** 指的是输出特征图上某个点"看到"输入图像的范围。$3\times3$ 卷积核的输出点看到 $3\times3$；两层堆叠后看到 $5\times5$——这正是为什么堆叠小核比单层大核更有效（见 VGG 那篇）。

## 四、池化：降采样与小幅平移不变

**池化（pooling）** 在每个小窗口里取一个汇总值，降低空间分辨率。两个常用算子：

- **最大池化（Max Pooling）**：取窗口内最大值。保留"有没有该模式"的强响应。
- **平均池化（Average Pooling）**：取窗口内平均值。更平滑，保留整体亮度信息。

```text
MaxPool 2x2, stride=2:     AvgPool 2x2, stride=2:

+----+----+----+----+       +----+----+----+----+
|  1 |  3 |  2 |  1 |       |  1 |  3 |  2 |  1 |
+----+----+----+----+       +----+----+----+----+
|  2 |  9 |  1 |  4 |  -->  |  9 |             (max 9)
+----+----+----+----+       +----+----+----+----+
|  0 |  6 |  3 |  2 |       |  0 |  6 |  3 |  2 |
+----+----+----+----+       +----+----+----+----+
|  5 |  1 |  4 |  8 |  -->  |  5 |  8 |  (avg 5, 8) 取均值
+----+----+----+----+       +----+----+----+----+
```

池化的作用：

1. **降维**：$2\times2$ stride-2 的池化把特征图缩小到 1/4，后续层计算量大幅下降。
2. **小幅平移不变**：猫的耳朵往左挪 1 像素，最大池化后输出基本不变。
3. **扩大感受野**：每做一次池化，下一层的"视野"就扩大一倍。

近年来也出现了 **global average pooling**：对整张特征图取平均，直接替代全连接层，大幅减少参数（见 GoogLeNet）。

## 五、通道与特征层级

单个卷积核只能提取一种模式（比如"纵向边缘"）。要同时检测横向、斜向、纹理等多种模式，就要用**多个卷积核**，每个核产生一个独立的特征图。这些特征图叠在一起就形成**通道（channel）**维度。

- **输入图像**：3 通道（R/G/B），尺寸 $H \times W$。
- **第一次卷积**：64 个 $3\times3$ 卷积核，输出 $H \times W \times 64$。
- **第二次卷积**：128 个 $3\times3$ 卷积核（输入 64 通道，所以每个核实际是 $3\times3\times64$），输出 $H/2 \times W/2 \times 128$。

注意**每多一个卷积核就多一组可学习参数**，通道数是性能和参数量的关键旋钮。

**特征层级**是 CNN 的核心优势——浅层学简单模式，深层组合出复杂语义：

```text
Layer 1 (浅):  边缘、颜色斑块、方向
              ↓
Layer 2-3:    纹理、角点、简单形状
              ↓
Layer 4-5:    物体部件（眼睛、轮子、轮辐）
              ↓
深层:          完整物体（人脸、汽车）
```

这就是为什么 CNN 能在视觉任务上如此成功——它天然地把"视觉皮层"的层次结构嵌入了架构。

## 六、PyTorch 实现：一个 CIFAR-10 分类 CNN

下面是一个完整可跑的 CIFAR-10 分类网络：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 三个 conv block：conv + ReLU + pool
        self.features = nn.Sequential(
            # 输入: (B, 3, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B, 32, 32, 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # (B, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (B, 64, 16, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # (B, 64, 8, 8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128, 8, 8)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                      # (B, 128, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # (B, 128)
            nn.Linear(128, num_classes),   # (B, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN()
x = torch.randn(8, 3, 32, 32)
print(model(x).shape)   # torch.Size([8, 10])
```

几个关键点：

- **`nn.Conv2d(in, out, k, padding)`**：`in` 是输入通道数，`out` 是输出通道数（卷积核个数）。
- **`nn.MaxPool2d(2)`**：默认 stride=2，等价于 `MaxPool2d(2, stride=2)`。
- **`nn.AdaptiveAvgPool2d(1)`**：自适应地把任意尺寸的特征图压成 $1\times1$，免去手算尺寸的麻烦。
- **不用全连接层也能分类**：global average pooling + linear 在很多架构里效果一样好且参数更少。

训练循环片段：

```python
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f'epoch {epoch}  loss {loss.item():.4f}')
```

## 七、MLP vs CNN：参数规模对比

同样把 $32\times32\times3$ 的 CIFAR-10 图片分类成 10 类：

```text
MLP（两隐层 256 神经元）:
  flatten -> 3072
  fc1     -> 256    3072*256 + 256      ≈ 786k
  fc2     -> 256     256*256 + 256       ≈  66k
  fc3     -> 10      256*10  + 10        ≈   2.5k
  ---------------------------------  合计 ≈ 854k 参数

SimpleCNN（上面那一个）:
  conv1  (3->32, 3x3)        3*32*9 + 32         =    896
  conv2  (32->64, 3x3)       32*64*9 + 64        =  18,496
  conv3  (64->128, 3x3)      64*128*9 + 128      = 147,584
  fc     (128->10)           128*10 + 10         =   1,290
  ---------------------------------------------  合计 ≈ 168k 参数
```

CNN 参数约为 MLP 的 **1/5**，但准确率通常**高出 15–25 个百分点**——因为 CNN 充分利用了图像的局部结构与平移不变性。

## 八、训练 CNN 的常见工程要点

- **数据增强**：`RandomCrop(32, padding=4)`、`RandomHorizontalFlip()`、`ColorJitter`，几乎免费的精度提升。
- **BatchNorm**：在 conv 和 ReLU 之间插入，稳定训练、允许更大学习率。
- **学习率调度**：`CosineAnnealingLR` 或 `MultiStepLR`，训练末期衰减 10–100 倍。
- **权重初始化**：Kaiming（`nn.init.kaiming_normal_`）配合 ReLU，几乎不会失败。
- **不要一上来就 MLP**：处理图像默认从 CNN 起步——除非你确定图像极小、极简单。

## 小结

CNN 通过**局部连接 + 权值共享 + 池化**三个核心机制，把视觉任务中至关重要的"平移不变性"和"局部结构"嵌入了架构本身：浅层学边缘与纹理，深层组合出物体部件与完整对象；参数规模比同等输入的 MLP 小一个数量级，准确率却高出一截。掌握卷积、步长、填充、池化、通道与感受野这些基础概念后，下一步就是看 LeNet/AlexNet/VGG/ResNet 等经典架构如何把 CNN 推到 SOTA，以及现代架构（ConvNeXt、EfficientNet）又如何应对 Transformer 的挑战。