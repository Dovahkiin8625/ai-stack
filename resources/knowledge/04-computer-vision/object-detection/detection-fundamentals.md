# 目标检测基础：从 R-CNN 到 YOLO 的两阶段与单阶段范式

目标检测（Object Detection）是计算机视觉的"重工业"任务——它不仅要回答"图里有什么"，还要回答"在哪里"。给定一张输入图像，模型需要输出一组 **bounding box**（边界框）+ **class label**（类别标签），并且每个框都要配一个置信度分数。这个看似简单的"分类 + 定位"组合，背后却涉及 IoU、anchor 设计、NMS、特征金字塔、损失函数平衡等大量工程细节。本文从评价指标出发，沿两条主线——**两阶段检测器**（R-CNN 系列）与**单阶段检测器**（YOLO 系列）——系统梳理十年来的演进路线，最后用 PyTorch 实现一个最小可用的 YOLO detection head。

## 一、任务定义与评价指标

### 1.1 任务形式

目标检测的输入是图像 $I \in \mathbb{R}^{H \times W \times 3}$，输出是一组检测结果：

$$
\mathcal{D} = \{(b_i, c_i, s_i)\}_{i=1}^{N}
$$

其中 $b_i = (x, y, w, h)$ 是边界框（中心坐标 + 宽高，或左上角 + 右下角），$c_i \in \{1, \dots, K\}$ 是类别，$s_i \in [0,1]$ 是置信度。一个边界框也可以用 $(\ell, t, r, b)$ 的"左上右下"形式表示，两种形式可互相转换：

$$
x = \ell + \tfrac{1}{2}r, \quad y = t + \tfrac{1}{2}b, \quad w = r, \quad h = b
$$

物理直觉：边界框就是一个"在哪里"的紧凑表达——只用 4 个数就框住了一个物体。

### 1.2 IoU（Intersection over Union）

IoU（Intersection over Union，交并比）是检测领域最基础的几何度量。它衡量两个边界框的重合程度：

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{交集面积}}{\text{并集面积}}
$$

取值范围 $[0,1]$，1 表示完全重合，0 表示完全分离。IoU 既是训练时的回归目标（通过 $\mathcal{L}_{IoU} = 1 - \text{IoU}$ 转化为损失），也是推理时判断"是否匹配某个真值"的关键阈值（通常取 0.5）。

后续工作还提出了 **GIoU**、**DIoU**、**CIoU** 等改进版：当两个框不相交时，IoU 永远是 0、没有梯度信号；GIoU 引入最小外接矩形 $C$ 解决了这一问题：

$$
\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}
$$

### 1.3 Precision / Recall / mAP

和分类任务不同，检测任务要对每个类别独立计算：

- **TP（True Positive）**：IoU $\geq$ 阈值且类别正确的检测。
- **FP（False Positive）**：IoU $<$ 阈值，或重复检测（同一目标被多次检出）。
- **FN（False Negative）**：未被检出的真值。

**Precision**（精度）衡量"检出来的东西有多少是对的"：

$$
P = \frac{TP}{TP + FP}
$$

**Recall**（召回率）衡量"该检出的东西检出了多少"：

$$
R = \frac{TP}{TP + FN}
$$

**AP（Average Precision）**是 P-R 曲线下的面积。**mAP（mean Average Precision）**是所有类别 AP 的平均值。常用基准是 COCO mAP：在 IoU 从 0.5 到 0.95 每 0.05 取一个阈值求平均（mAP@0.5:0.95），比 VOC 的单阈值 mAP@0.5 更严苛。

### 1.4 速度指标：FPS 与 Latency

实时检测还需要关注 **FPS**（Frames Per Second，每秒帧数）和 **Latency**（单帧端到端耗时）。一般认为 FPS $>$ 30 才算"实时"，FPS $>$ 60 才算"流畅"。但 FPS 高度依赖硬件，论文里通常会标明是在哪张 GPU 上测得的。

### 1.5 COCO vs VOC：评估基准的差异

两个最常用的检测基准：

- **PASCAL VOC**：20 个类别，IoU 阈值 0.5 算 mAP。门槛较低，目前已基本被淘汰。
- **MS COCO**：80 个类别，引入 mAP@0.5:0.95（10 个 IoU 阈值求平均）、mAP@small/medium/large（按目标尺寸分组）、AR@1/10/100（不同最大检测数下的召回率）。COCO 对小目标的评判更严苛，是当前主流基准。

COCO 训练集 118k 张图、验证集 5k 张图、测试集 20k 张图，标注了 150 万个目标实例。"在 COCO 上 mAP 达到 50"已经是工业级强检测器的水平。

### 1.6 训练与推理的不对称

另一个工程上经常被忽略的事实：**检测的训练和推理存在天然不对称**。训练时一张图可能有 10 个目标，但网络会输出 100~100k 个候选；推理时我们需要从中挑出最可信的几个。这种不对称带来两个挑战：

- **正负样本失衡**：候选框里 99% 都是背景，需要 Focal Loss、难例挖掘（OHEM）等手段平衡。
- **推理冗余**：大量候选框都是"低置信度噪声"，NMS 是必须的清理动作。

## 二、两阶段检测器：R-CNN 家族

两阶段检测器（Two-stage Detector）的核心思想是**先粗后精**：第一阶段产生候选区域（region proposals），第二阶段对每个候选做分类与回归。代表工作就是 Ross Girshick 一脉相承的 R-CNN、Fast R-CNN、Faster R-CNN。

### 2.1 R-CNN（2014）

R-CNN（Regions with CNN features）的流程很直接：

1. **Selective Search**：用传统图像算法从一张图里提取约 2000 个候选区域。
2. **CNN 特征提取**：把每个候选区域**裁剪（crop）**并 **warp** 到 $227 \times 227$，逐个送入 AlexNet 提特征。
3. **SVM 分类**：每个类别训练一个线性 SVM 判断候选框是背景还是该类。
4. **bbox 回归**：训练一个线性回归器修正候选框位置。

R-CNN 在 VOC 2007 上把 mAP 从传统方法的 $\sim$33% 拉到 58.5%，但**速度极慢**——一张图要做 2000 次 CNN 前向，GPU 上也要几十秒。

### 2.2 Fast R-CNN（2015）

Fast R-CNN 的两项关键改进解决了 R-CNN 的速度与训练难题：

1. **整图卷积 + RoI Pooling**：把整张图一次性过 CNN 得到特征图，再用 **RoI Pooling**（Region of Interest Pooling）把每个候选区域对应的特征图区域池化到固定尺寸（如 $7 \times 7$）。这样 2000 个候选框共享一次卷积，速度大幅提升。
2. **多任务损失**：用 softmax 替代 SVM，并把分类损失与 bbox 回归损失合并到一个网络中，端到端训练。

RoI Pooling 的物理直觉：候选框是图像坐标，需要先按 CNN 的步长映射到特征图坐标，再把映射后的区域均匀分成 $H \times W$ 个格子，每个格子做 max pooling，得到固定大小的特征。

RoI Pooling 有一个小问题：**两次量化**（候选框到特征图的映射 + 区域到格子的划分）会引入像素级偏差。后续 **RoI Align**（Mask R-CNN）通过双线性插值彻底解决了这个问题。

### 2.3 Faster R-CNN（2015）：RPN 让两阶段真正端到端

Faster R-CNN 的核心贡献是 **RPN（Region Proposal Network，区域提议网络）**——用一个轻量神经网络替代 Selective Search，让"提议区域"也变成可学习的过程。整个流程变为：

1. 共享 CNN backbone 提取特征图。
2. RPN 在特征图上以**滑动窗口**方式预测"这里是否有物体"以及"框该往哪边修"。
3. 把 RPN 输出的候选框送入第二阶段的检测头，做精细分类与回归。

**anchor（锚框）** 是 RPN 的关键设计：在特征图的每个位置，预设 $k$ 个不同尺寸与长宽比的参考框（如 $128^2, 256^2, 512^2$ 配 1:1, 1:2, 2:1），RPN 预测的是这些 anchor 与真值之间的偏移。这种"在固定模板上做回归"的思想深刻影响了后续五六年的检测器设计。

Faster R-CNN 在 VOC 2007 上 mAP 达到 78.8%，并且推理速度提升到 5 FPS（在当时的 GPU 上），第一次让两阶段检测器具备实用价值。

### 2.4 关键组件：Anchor、NMS 与 FPN

**Anchor Boxes** 的几何含义：把图像空间离散化为一组"候选模板"，后续只需预测相对偏移量。YOLOv2/v3/v4/v5、SSD、RetinaNet 全部沿用了这一思想。它的问题也很明显——anchor 的尺寸与长宽比需要精心设计，且正负样本严重不平衡（绝大多数 anchor 都是背景）。

**NMS（Non-Maximum Suppression，非极大值抑制）** 是检测后处理的标准操作：对每个类别，把所有检测按分数排序，依次保留分数最高的框，剔除与它 IoU $>$ 阈值（通常 0.5）的相邻框。它消除"同一目标被多次检出"的问题，但也带来"遮挡目标被吞掉"的隐患——Soft-NMS、DIoU-NMS 是常见的改进。

**FPN（Feature Pyramid Network，特征金字塔网络）** 解决了"小目标难检"的痛点：CNN 的深层特征语义强但分辨率低，浅层特征分辨率高但语义弱。FPN 用"自顶向下 + 横向连接"把两者融合：

$$
P_i = \text{Conv}_{1\times1}(C_i) + \text{Upsample}(P_{i+1})
$$

其中 $C_i$ 是 backbone 第 $i$ 阶段的特征图，$P_i$ 是融合后的金字塔层。检测头在不同尺度的 $P_i$ 上分别预测大、中、小目标，显著提升召回率。

## 三、单阶段检测器：YOLO 系列

两阶段检测器虽然精度高，但"先提议再分类"的流程注定跑不快。**单阶段检测器（One-stage Detector）**把"提议 + 分类"合并成一次回归：直接在特征图上预测类别与边界框，没有 RPN 也没有二次精细化。

### 3.1 YOLOv1（2016）：把检测当作回归

YOLO（You Only Look Once）的核心思想极具颠覆性：把输入图像划分成 $S \times S$ 网格，每个网格直接预测 $B$ 个边界框、置信度与 $C$ 个类别概率。整张图的检测只需要**一次** CNN 前向。

YOLOv1 的 backbone 是类 GoogLeNet 的 24 层卷积 + 2 层全连接，最后输出一个 $S \times S \times (B \times 5 + C)$ 的张量。在 VOC 上以 $S=7, B=2, C=20$ 计算，每个网格预测 2 个框 + 20 个类别，总共 49 个网格 → 98 个候选。

YOLOv1 的优势是**速度极快**——45 FPS，远超 Faster R-CNN。代价是精度不如两阶段（VOC mAP 63.4%），且对小目标、密集目标表现较差（每个网格只能预测 2 个框，且网格划分粒度有限）。

### 3.2 YOLOv3/v4/v5：多尺度与 Anchor-based 的成熟

**YOLOv3**（2018）做了几项关键改进：

1. **Darknet-53 backbone**：借鉴 ResNet 的残差结构，53 层卷积，无全连接层。
2. **多尺度预测**：在 3 个不同尺度的特征图（步长 32、16、8）上分别检测大、中、小目标，每个尺度 3 个 anchor。
3. **独立的 logistic 分类器**：用 sigmoid 替代 softmax，支持多标签（一个目标可以同时属于多个标签）。

YOLOv3 的损失函数把回归、置信度、分类三部分加权求和，anchor 与真值的匹配采用"哪个 anchor 与真值 IoU 最大，就由它负责预测"。

**YOLOv4**（2020）和 **YOLOv5**（2020）是工程优化的集大成：Mosaic 数据增强、CIoU 损失、PANet 特征融合、CSPDarknet、Focus 结构、自动 anchor 学习……YOLOv5 至今仍是工业界最常用的目标检测器之一，因为它在 COCO 上 mAP@0.5 接近 55%，而速度能跑到 100+ FPS。

### 3.3 YOLOv8/v9/v10/v11：Anchor-free 与现代改进

2023 年发布的 **Ultralytics YOLOv8** 把整个流程简化到一个 `ultralytics` 包里，并引入了若干新设计：

1. **Anchor-free 检测头**：不再预定义 anchor，而是直接预测中心点距离四条边的距离。这一思想借鉴自 FCOS、CenterNet。
2. **解耦头（Decoupled Head）**：分类与回归用两个独立的卷积分支，避免任务冲突。
3. **TAL（Task-Aligned Assigner）**：用分类得分与回归精度的几何平均做标签分配，比传统 IoU 阈值更鲁棒。

**YOLOv9**（2024）提出 **GELAN**（Generalized Efficient Layer Aggregation Network）和 **PGI**（Programmable Gradient Information），进一步提升参数效率。**YOLOv10**（2024）引入 **NMS-free 训练**，用一致双标签分配让模型在推理时不再需要 NMS。**YOLOv11**（2024）则把 backbone 与 head 都做了轻量化重设计，在保持精度的同时进一步压低延迟。

整体趋势是：**Anchor-free + Decoupled Head + Advanced Label Assignment + NMS-free** 成为 YOLO 系列的现代配方。

## 四、关键组件深度解析

### 4.1 Anchor 的前世今生

Anchor 的设计目标是"把检测问题转成对固定模板的偏移回归"。给定 anchor $(x_a, y_a, w_a, h_a)$ 和真值框 $(x_g, y_g, w_g, h_g)$，预测偏移量 $(t_x, t_y, t_w, t_h)$ 的关系是：

$$
x_g = x_a + t_x \cdot w_a,\quad y_g = y_a + t_y \cdot h_a
$$
$$
w_g = w_a \cdot e^{t_w},\quad h_g = h_a \cdot e^{t_h}
$$

这样设计的好处是数值范围稳定——$t$ 通常在 $[-1, 1]$ 之间，便于优化。

Anchor 的核心缺陷是**超参数敏感**（尺寸、长宽比、正负样本阈值都得调）和**不平衡**（一张图里 100k+ anchor，可能只有几十个真值框）。FCOS、CenterNet、YOLOv8 等后续工作通过 anchor-free 彻底绕开了这些问题。

### 4.2 NMS 与 Soft-NMS

NMS 的算法伪代码：

```text
输入：boxes (N,4), scores (N,), iou_threshold
1. 按 scores 降序排列
2. while 仍有未处理的框：
3.     取出分数最高的框 b_max，加入 keep 列表
4.     计算 b_max 与剩余所有框的 IoU
5.     剔除 IoU > 阈值 的框
6. 返回 keep
```

**Soft-NMS**（2017）不直接剔除相邻框，而是把它们的分数按 IoU 衰减：

$$
s_i = s_i \cdot (1 - \text{IoU}(b_i, b_{\max})) \quad \text{(线性版本)}
$$

物理直觉：被遮挡的物体虽然和第一选择高度重叠，但分数调低后仍可能进入最终结果。Soft-NMS 在密集目标场景下提升明显。

### 4.3 损失函数：分类 + 回归 + 平衡

现代检测器的损失通常是三部分加权：

$$
\mathcal{L} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{obj}} \mathcal{L}_{\text{obj}}
$$

- $\mathcal{L}_{\text{cls}}$：Focal Loss（RetinaNet 提出）解决正负样本不平衡：

$$
\text{FL}(p) = -\alpha (1-p)^\gamma \log p
$$

$(1-p)^\gamma$ 让"易分样本"（$p$ 大）的梯度被压制，"难分样本"（$p$ 小）获得更多关注。

- $\mathcal{L}_{\text{box}}$：L1、L2、Smooth L1、IoU、GIoU、CIoU 损失。CIoU 同时考虑重叠面积、中心点距离、长宽比一致性，是当前主流。
- $\mathcal{L}_{\text{obj}}$：判断"这里是物体还是背景"的二分类损失。

## 五、PyTorch 实现：最小 YOLO 检测头

下面给出一个**简化版 YOLO 检测头**——输入是 FPN 的某一层特征图，输出是每个网格的预测张量：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BN + SiLU 的标准卷积块（YOLOv5 风格）。"""

    def __init__(self, in_c, out_c, k=1, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))


class YOLODetectionHead(nn.Module):
    """简化版 YOLO 检测头（解耦头 + anchor-free）。

    输入：FPN 某一层特征图 (B, C_in, H, W)
    输出：3 个张量
      - cls: (B, num_classes, H, W)
      - reg: (B, 4, H, W)            预测中心点到四条边的距离
      - obj: (B, 1, H, W)            objectness
    """

    def __init__(self, in_channels: int, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes

        # 分类分支：3 个 3x3 卷积 + 1x1 投影
        self.cls_branch = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3),
            ConvBNAct(in_channels, in_channels, k=3),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )
        # 回归分支：4 个距离值
        self.reg_branch = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3),
            ConvBNAct(in_channels, in_channels, k=3),
            nn.Conv2d(in_channels, 4, kernel_size=1),
        )
        # objectness 分支
        self.obj_branch = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3),
            ConvBNAct(in_channels, in_channels, k=3),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        cls = self.cls_branch(x)              # (B, num_classes, H, W)
        reg = self.reg_branch(x)              # (B, 4, H, W)
        obj = self.obj_branch(x)              # (B, 1, H, W)
        return cls, reg, obj


class SimpleYOLO(nn.Module):
    """超简版 YOLO：backbone 用一层 stride=32 的卷积代替 ResNet。"""

    def __init__(self, num_classes: int = 80):
        super().__init__()
        # 简化 backbone：实际工程里换成 CSPDarknet / ResNet
        self.backbone = nn.Sequential(
            ConvBNAct(3, 64, k=7, s=2, p=3),                  # /2
            ConvBNAct(64, 128, k=3, s=2),                     # /4
            ConvBNAct(128, 256, k=3, s=2),                    # /8
            ConvBNAct(256, 512, k=3, s=2),                    # /16
            ConvBNAct(512, 1024, k=3, s=2),                   # /32
        )
        # 检测头：在最高层做预测
        self.head = YOLODetectionHead(in_channels=1024, num_classes=num_classes)
        # 网格坐标（注册为 buffer，自动跟随 .to(device)）
        self.stride = 32

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)              # (B, 1024, H/32, W/32)
        cls, reg, obj = self.head(feat)
        B, _, H, W = cls.shape
        # 物理直觉：每个像素点代表原图一个 self.stride×self.stride 的格子
        return {"cls": cls, "reg": reg, "obj": obj, "stride": self.stride}


# 烟测
if __name__ == "__main__":
    model = SimpleYOLO(num_classes=80)
    x = torch.randn(2, 3, 640, 640)
    out = model(x)
    print("cls:", out["cls"].shape)   # (2, 80, 20, 20)
    print("reg:", out["reg"].shape)   # (2, 4, 20, 20)
    print("obj:", out["obj"].shape)   # (2, 1, 20, 20)
```

代码里几个值得注意的点：

- **解耦头**：分类、回归、objectness 用三条独立卷积分支，避免任务间梯度相互干扰。
- **SiLU 激活**：比 ReLU 更平滑，YOLOv5/v8 标配。
- **Anchor-free**：回归分支直接预测中心点到四条边的距离，简化了标签分配。
- **真实 backbone**：这里用 5 层卷积代替，工程上要换成 CSPDarknet、ResNet 等；多尺度预测需要对应多个 head。

## 六、R-CNN vs YOLO：如何选型

| 维度 | 两阶段（Faster R-CNN） | 单阶段（YOLOv8） |
| --- | --- | --- |
| **精度** | 高（COCO mAP@0.5:0.95 约 40-45%） | 中高（约 50%，但 v8/v10 已经接近） |
| **速度** | 较慢（5-15 FPS） | 很快（50-150 FPS） |
| **小目标** | 配合 FPN 表现好 | 多尺度 FPN/PAN 同样出色 |
| **密集目标** | NMS 后处理是关键 | NMS-free (v10) 或 Soft-NMS 更鲁棒 |
| **训练稳定性** | 较稳，提案质量有保证 | 对数据增强、正则化更敏感 |
| **部署友好** | 候选框生成复杂，工程量大 | 端到端一个 .pt，易部署 |

**选型经验**：

- **离线精度优先**（医学影像、工业质检）：Faster R-CNN 或 Cascade R-CNN。
- **实时视频**（自动驾驶、监控）：YOLOv8/v10/v11。
- **资源受限的边缘设备**：YOLO-Nano、PP-PicoDet、MobileDet。
- **旋转框 / 关键点**：两阶段的 Oriented R-CNN、单阶段的 RTMDet。

## 七、其他重要单阶段检测器

除 YOLO 之外，单阶段检测器还有两个绕不开的名字：SSD 与 RetinaNet。

### 7.1 SSD（Single Shot MultiBox Detector, 2016）

SSD 的核心思想是**多尺度特征图预测**：在 backbone（VGG-16）的多个不同分辨率层上分别接检测头，每个层负责预测特定尺寸的目标。浅层特征检测小目标，深层特征检测大目标。

SSD 的损失函数是 MultiBox Loss：

$$
\mathcal{L} = \frac{1}{N}\left(\mathcal{L}_{\text{conf}} + \alpha \mathcal{L}_{\text{loc}}\right)
$$

其中 $N$ 是正样本数，$\mathcal{L}_{\text{conf}}$ 是多分类 softmax 损失，$\mathcal{L}_{\text{loc}}$ 是 Smooth L1 回归损失，$\alpha$ 控制两者的平衡（默认 1）。

SSD 的 anchor 设计与 Faster R-CNN 类似，但每层 anchor 的尺寸通过公式 $s_k = s_{\min} + \frac{s_{\max} - s_{\min}}{m-1}(k-1)$ 控制，$s_{\min}=0.2, s_{\max}=0.9$。

### 7.2 RetinaNet（2017）：Focal Loss 解决类别不平衡

RetinaNet 的核心贡献是 **Focal Loss**——在标准交叉熵基础上加上一个调制因子：

$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

其中 $p_t$ 是预测正确类别的概率，$\gamma$ 通常取 2。

物理直觉：当样本被正确分类且置信度高（$p_t$ 接近 1）时，$(1-p_t)^\gamma$ 接近 0，损失被大幅压制；当样本被错分或置信度低时，调制因子接近 1，损失不被削弱。这让模型把更多"注意力"放在难分样本上。

RetinaNet 的 backbone 是 ResNet-FPN，检测头是分类子网 + 回归子网的并行结构。在 COCO 上达到 40.8% mAP，比同期 YOLOv2 高出约 7 个点，证明了单阶段检测器也能达到两阶段的精度。

## 八、数据增强与训练技巧

检测模型的精度不仅取决于架构，还取决于**数据增强**和**训练策略**。

### 8.1 几何增强

- **Random Crop**：随机裁剪区域作为新的输入图，需要同步处理对应的边界框。
- **Random Flip**：水平翻转是最便宜的增强，垂直翻转用得少。
- **Multi-Scale Training**：每隔若干 iter 随机选一个训练尺寸（如 320~640），让模型对尺度鲁棒。

### 8.2 颜色增强

- **HSV 抖动**：随机调整图像的色调（H）、饱和度（S）、明度（V）。YOLOv5 默认启用。
- **Mosaic**：把 4 张图拼接成 1 张，等价于"小批量内的裁剪"。YOLOv4 引入，能让模型看到更多小目标。
- **MixUp / CutMix**：两张图按比例混合，对应框也按比例混合。

### 8.3 正则化与优化

- **EMA（Exponential Moving Average）**：对模型参数做指数滑动平均，推理时用平均后的参数，通常带来 1-2 个点的 mAP 提升。
- **Cosine LR**：学习率按余弦曲线下降，比 Step LR 更平滑。
- **Warmup**：训练初期学习率从 0 线性上升到目标值，避免前期梯度爆炸。
- **AutoAnchor**：训练前用 k-means 在数据集上聚类 anchor 尺寸，让预设 anchor 更贴合数据分布。

## 九、小结

目标检测十年来的演进可以浓缩成三条主线：

1. **从粗到精再到统一**：R-CNN → Fast → Faster 是"提议+精修"流程的极致；YOLO 直接把检测变成单次回归。DETR（下一篇要讲的）则更进一步，把 anchor、NMS、标签分配全部用 set prediction 替代。
2. **从手工设计到端到端学习**：Selective Search → RPN → Anchor-free → NMS-free，模型在越来越少依赖人工先验的同时，精度越来越高。
3. **从单一 backbone 到多尺度 FPN**：从 VGG/ResNet 的单层特征，到 FPN/PAN 的多尺度融合，再到 BiFPN 的加权融合，"如何用好不同尺度的特征"始终是检测的核心命题。

掌握 R-CNN 的"提议+精修"两阶段范式和 YOLO 的"单次回归"单阶段范式，是理解 DETR、ViT-based 检测器等更现代方法的基石。下一篇我们将看到 DETR 如何用 Transformer 的 set prediction 范式，把整个检测 pipeline 重新设计一遍。
