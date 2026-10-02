# DETR 与端到端检测 Transformer：把检测变成 Set Prediction

2020 年 Facebook AI 提出的 **DETR（DEtection TRansformer）** 是目标检测领域的一次范式跃迁。它用一套简洁到令人惊讶的架构——CNN backbone + Transformer encoder-decoder + 一组可学习的 object queries——直接把检测问题转化为**集合预测（set prediction）**问题，从此摆脱了 anchor 设计、NMS 后处理、手工标签分配这些沿用了六七年的"工程包袱"。本文从 DETR 的核心思想出发，逐步拆解它的架构、训练目标、标签分配算法，并讨论 Deformable DETR、DINO 等改进如何让这套范式真正实用。

## 一、为什么要"端到端"

回顾上一篇 R-CNN/YOLO 的设计，有几处明显的"非端到端"环节：

1. **NMS 后处理**：必须人工设置 IoU 阈值，且阈值的选择会影响结果。
2. **Anchor 设计**：anchor 的尺寸、长宽比、IoU 阈值都需要调。
3. **标签分配**：Faster R-CNN 用 MaxIoU、YOLO 用 Anchor-Positive 匹配、RetinaNet 用 Focal Loss——这些都是人为规则。

这些组件的本质是：**检测任务输出的是一个集合（多个目标），而深度学习默认的输出是固定大小的向量或网格**。人工设计的 anchor、NMS 就是为了在"网络输出"和"集合"之间搭建桥梁。

DETR 的洞察是：**既然输出是集合，那就用集合预测的范式——直接让网络输出 $N$ 个预测，与真值集合做最优二分图匹配**。这样 anchor、NMS、标签分配规则全部消失，整个 pipeline 就变成纯端到端。

物理直觉：把 $N$ 个 "object queries" 看作 $N$ 个"槽位"，每个槽位通过注意力机制学会"对图中某个区域负责"。训练时通过 Hungarian 算法把每个槽位与最合适的真值框匹配，匹配上的槽位负责回归对应框，没匹配上的槽位就预测"无目标"。

## 二、DETR 的整体架构

### 2.1 三段式流水线

DETR 的架构出奇地简单，可以拆成三段：

```
[CNN Backbone] → [Transformer Encoder-Decoder] → [FFN 检测头]
   ResNet-50        6 + 6 层                          共享 MLP
```

1. **Backbone**：标准 CNN（通常是 ResNet-50），输入图像 $x \in \mathbb{R}^{3 \times H_0 \times W_0}$，输出特征图 $f \in \mathbb{R}^{C \times H \times W}$，其中 $C=2048$，$H=H_0/32, W=W_0/32$。
2. **Transformer 编码器-解码器**：把 $f$ 加上位置编码，展平为 $(HW, C)$ 的序列，过 6 层 encoder + 6 层 decoder。解码器的查询由 $N=100$ 个 **object queries** 提供。
3. **检测头**：每个 object query 经过一个共享的 3 层 FFN，输出 (class, bbox) 二元组。

$N=100$ 表示 DETR 最多检测 100 个目标；少于 100 个真值的图像，剩余 query 预测"无目标"。

### 2.2 Object Queries：可学习的目标槽位

Object queries 是 DETR 引入的新概念，本质是 $N$ 个可学习的位置向量 $\{(q_1, \dots, q_N)\}$，它们的角色类似 NLP Transformer 里的"解码器 token"。每个 query 通过交叉注意力去"问"编码器的特征，"问"到感兴趣的区域后，再通过 FFN 预测一个边界框。

直觉上，每个 query 会通过训练分化出不同"专长"：一些 query 学会检测大目标，一些关注小目标，一些偏好图像中央，一些偏好边角。论文可视化发现 object queries 在不同训练轮次的注意力分布逐渐聚焦到不同物体上。

Object queries 没有显式的位置编码——它们自身的位置信息就是可学习的。

### 2.3 编码器的位置编码

编码器处理的是 CNN 特征图序列 $(HW, C)$，没有时序结构，必须用二维空间位置编码。DETR 用一组**固定 sinusoidal** 位置编码，沿着 $H$ 和 $W$ 两个轴生成：

$$
PE_{(x,y,2i)} = \sin(x / 10000^{2i/C}),\quad PE_{(x,y,2i+1)} = \cos(y / 10000^{2i/C})
$$

这个位置编码直接加到特征图上，让 Transformer 知道每个 patch 在图像里的空间位置。

## 三、匈牙利匹配：把"集合"对齐到"真值"

### 3.1 为什么需要 Hungarian

DETR 的核心数学问题是：**模型预测了 $N$ 个结果，真值只有 $M$ 个（$M \le N$），如何把预测和真值一一对应起来**？这是一个经典的**二分图最小权匹配**问题，用 Hungarian 算法可以在 $O(N^3)$ 时间内求解。

构造一个 $N \times M$ 的代价矩阵 $\mathcal{C}$，元素 $(i, j)$ 表示把第 $i$ 个预测分配给第 $j$ 个真值的"代价"：

$$
\mathcal{C}_{i,j} = -\mathbb{1}_{\{c_j \neq \varnothing\}} \hat{p}_{\sigma(j)}(c_j) + \mathbb{1}_{\{c_j \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_j)
$$

其中 $\hat{p}_{\sigma(j)}(c_j)$ 是第 $i$ 个预测对真值类别 $c_j$ 的概率，$\mathcal{L}_{\text{box}}$ 是 bbox 回归损失。"$\varnothing$"（no object）是额外的虚拟类别，保证预测数与真值数对齐。

Hungarian 算法找到的最优匹配 $\hat{\sigma}$ 让总代价最小：

$$
\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{j=1}^{M} \mathcal{C}_{\sigma(j), j}
$$

物理直觉：匹配完成后，每个真值框由"最合适"的预测负责预测；剩余的预测只能预测"无目标"，否则就会被惩罚。

### 3.2 损失函数

匹配完成后，DETR 的总损失是分类损失 + bbox 损失的加权和：

$$
\mathcal{L}_{\text{DETR}}(y, \hat{y}) = \sum_{i=1}^{N}\left[-\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)})\right]
$$

分类用普通的 **cross-entropy**。"无目标"对应一个特殊的 $\varnothing$ 类别。

bbox 损失是 **L1 + GIoU** 的组合：

$$
\mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) = \lambda_{\text{L1}} \|b_i - \hat{b}_{\hat{\sigma}(i)}\|_1 + \lambda_{\text{iou}} \mathcal{L}_{\text{GIoU}}(b_i, \hat{b}_{\hat{\sigma}(i)})
$$

L1 让绝对坐标误差最小，GIoU 让重叠度最大。两者互补——L1 对小框更敏感，GIoU 对位置不重合的情况能给出梯度。

### 3.3 GIoU Loss 详解

GIoU 在 IoU 基础上引入"最小外接矩形" $C$：

$$
\text{GIoU}(A, B) = \text{IoU}(A, B) - \frac{|C \setminus (A \cup B)|}{|C|}
$$

当 $A$ 与 $B$ 完全不相交时，$\text{IoU}=0$ 但 $C$ 的"空隙"很大，GIoU 退化为一个负值，给优化器留出了方向。对应的损失：

$$
\mathcal{L}_{\text{GIoU}}(A, B) = 1 - \text{GIoU}(A, B)
$$

物理直觉：GIoU = IoU − 罚项。当两个框分离时，罚项为正、IoU 为 0，GIoU 为负；模型要让 GIoU 变大，必须让两个框越来越接近——这就解决了"不相交时 IoU 无梯度"的问题。

### 3.4 L1 Loss vs Smooth L1 Loss

DETR 的 L1 损失定义是简单的绝对值差：

$$
\mathcal{L}_{L1}(b, \hat{b}) = \sum_{i=1}^{4} |b_i - \hat{b}_i|
$$

但朴素的 L1 损失在零点附近梯度不连续，且对异常值（如尺寸特别大的框）不够鲁棒。Faster R-CNN 用了 **Smooth L1**：

$$
\mathcal{L}_{\text{smooth-L1}}(x) = \begin{cases} 0.5 x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}
$$

DETR 没用 Smooth L1，而是直接用 L1 + GIoU 的组合——因为 GIoU 已经提供了对距离的连续梯度，L1 只需负责"绝对坐标精度"即可。

### 3.5 标签分配的"去重"机制

匈牙利匹配本质上是一种**一一对应**分配：每个真值只能匹配一个预测。这天然避免了 R-CNN 系列里"一个目标被多个 anchor 重复预测"的问题。所以 DETR 在推理时不需要 NMS——每个 query 天然只负责一个目标，不需要后处理去重。

但"一一对应"也带来副作用：当两个目标高度重叠时（如密集行人），匈牙利匹配可能在两个可行匹配之间反复跳变，导致训练不稳定。**DINO 的去噪训练**就是专门为了缓解这一问题而设计的。

## 四、Deformable DETR：解决 DETR 的收敛慢

### 4.1 DETR 的两个核心痛点

DETR 的论文漂亮，但工程上有两个绕不开的问题：

1. **收敛慢**：在 COCO 上，DETR 需要 500 个 epoch 才能达到 Faster R-CNN 用 36 个 epoch 达到的精度。
2. **小目标差**：特征图的全局自注意力对小目标不友好——它们的空间位置容易被淹没。

根因是**全局自注意力的复杂度**为 $O(H^2 W^2)$，在 800×800 的图像上相当于 40000 个 patch 互相 attend，计算和优化都很吃力。

### 4.2 Deformable Attention 的核心思想

**Deformable DETR**（ICLR 2021）借鉴 Deformable Convolution 的思路，提出了 **Deformable Attention**：每个 query 不再 attend 所有 patch，而是只 attend **少量学习到的"采样点"**。

具体做法：每个 query 通过一个线性层预测 $K$ 个 2D 偏移量 $\{\Delta p_{qk}\}_{k=1}^K$，然后只在这 $K$ 个位置采样特征做加权：

$$
\text{DeformAttn}(z_q, p_q, x) = \sum_{k=1}^{K} A_{qk} \cdot W x(p_q + \Delta p_{qk})
$$

其中 $A_{qk}$ 是 query 学到的注意力权重（通过 softmax 归一化），$W$ 是值投影矩阵。

复杂度从 $O(H^2 W^2)$ 降到 $O(HWK)$，$K$ 通常取 4 或 8。物理直觉：让每个 query "自适应地"看向图像里最有信息量的几个点，而不是对所有 patch 做无差别加权。

### 4.3 多尺度 Deformable Attention

Deformable DETR 进一步把多尺度特征金字塔纳入：

$$
\text{MS-DeformAttn}(z_q, \hat{p}_q, \{x^l\}) = \sum_{m=1}^{M} W_m \sum_{k=1}^{K} A_{qmk} \cdot x^m(\phi_l(\hat{p}_q) + \Delta p_{qmk})
$$

其中 $m$ 遍历 FPN 的多个尺度（通常 4 层），$\phi_l$ 是从 query 参考点映射到第 $l$ 层特征图的尺度归一化。

这相当于让 query 能在不同分辨率的"望远镜"下看图——既能看大目标，也能看小目标。

Deformable DETR 把 DETR 的训练收敛时间从 500 epoch 缩短到 50 epoch，COCO mAP 从 44.5 提升到 49.0。

## 五、DETR 的现代变体：DINO 与后续

### 5.1 DINO（2022）

DINO（DETR with Improved DeNoising Anchor Boxes）是 DETR 系列当前最强的开源版本，COCO mAP 达到 63.3。它的核心改进：

1. **对比去噪训练（Contrastive DeNoising, CDN）**：在训练时，把真值框加少量噪声生成"正样本"，把真值框加更大噪声或替换成其他框生成"负样本"，让模型学会区分"接近真值的框"和"远离真值的框"。这相当于用一种生成式的方式教模型做 NMS。
2. **混合查询选择（Mixed Query Selection）**：encoder 的输出特征作为 decoder 的初始 query，让 query 起点不再是随机可学习的，而是基于图像内容的"proposal"。
3. **Look Forward Twice**：在 decoder 的更新规则中加入"前一层预测"作为参考，让回归更稳定。

### 5.2 DN-DETR（2022）

DN-DETR（Denoising DETR）单独提出了去噪训练。它观察到 DETR 训练初期匹配不稳定，于是把"加噪的真值框"作为额外 query 喂给 decoder，强制模型学会"把带噪声的框恢复到真值"。这一招把收敛速度从 50 epoch 进一步压到 12-36 epoch。

### 5.3 其他变体

- **Conditional DETR**：用条件注意力机制，让 query 的空间注意力更精准。
- **SMCA（2021）**：把 query 的空间先验显式编码进 attention 权重。
- **Efficient DETR**：简化 encoder-decoder 流程，去掉冗余结构。
- **D-FINE**（2024）：进一步把 DETR 推到实时/边缘场景，FPS 突破 100。

整体趋势是：**DETR 范式已经从"理论上优雅、工程上难用"走向"工业上也能落地"**。

## 六、PyTorch 实现：简化版 Hungarian Matcher

下面给出 DETR 中 Hungarian matching 的简化 PyTorch 实现，可直接接入任何 detection head：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """DETR 风格的二分图匹配器。

    输入：网络预测 (cls_logits, pred_boxes) 与真值 (target_classes, target_boxes)。
    输出：每个 batch 内 (pred_idx, gt_idx) 的索引对。
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs:
            cls_logits: (B, N, num_classes + 1)   最后一类为 no-object
            pred_boxes: (B, N, 4)                 (cx, cy, w, h)，已归一化到 [0, 1]
        targets:
            list[dict]，每个 dict 含 'labels' (M,) 与 'boxes' (M, 4)
        """
        bs, num_queries = outputs["cls_logits"].shape[:2]

        # 1) 分类代价 = -prob[gt_class]
        cls_prob = outputs["cls_logits"].flatten(0, 1).softmax(-1)              # (B*N, K+1)
        cost_class_list = []
        for b in range(bs):
            tgt_cls = targets[b]["labels"]                                       # (M,)
            cost_class_list.append(-cls_prob[b * num_queries:(b + 1) * num_queries, tgt_cls])
        # 每张图的 cost_class 形状 (N, M)

        # 2) L1 bbox 代价
        pred_boxes = outputs["pred_boxes"]                                       # (B, N, 4)
        cost_bbox_list = []
        for b in range(bs):
            tgt_box = targets[b]["boxes"]                                        # (M, 4)
            cost_bbox_list.append(
                torch.cdist(pred_boxes[b], tgt_box, p=1)                         # (N, M)
            )

        # 3) GIoU 代价（注意：GIoU 取负，cost = -GIoU；最小化代价 = 最大化 GIoU）
        cost_giou_list = []
        for b in range(bs):
            tgt_box = targets[b]["boxes"]                                        # (M, 4)
            cost_giou_list.append(-_generalized_box_iou(
                _box_cxcywh_to_xyxy(pred_boxes[b]),
                _box_cxcywh_to_xyxy(tgt_box),
            ))                                                                    # (N, M)

        # 4) 合并代价并求解 Hungarian
        indices = []
        for b in range(bs):
            C = (self.cost_class * cost_class_list[b]
                 + self.cost_bbox * cost_bbox_list[b]
                 + self.cost_giou * cost_giou_list[b])
            # linear_sum_assignment 要求 numpy 或 tensor
            idx = linear_sum_assignment(C.cpu())
            indices.append((torch.as_tensor(idx[0], dtype=torch.int64),
                            torch.as_tensor(idx[1], dtype=torch.int64)))
        return indices


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) → (x1, y1, x2, y2)。"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组框之间的 GIoU（Generalized IoU）。"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 交集
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # 并集
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    # 最小外接矩形
    lt_enclose = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enclose = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_enclose = (rb_enclose - lt_enclose).clamp(min=0)
    area_enclose = wh_enclose[:, :, 0] * wh_enclose[:, :, 1]

    giou = iou - (area_enclose - union) / area_enclose.clamp(min=1e-6)
    return giou


# 烟测
if __name__ == "__main__":
    matcher = HungarianMatcher()
    outputs = {
        "cls_logits": torch.randn(2, 10, 6),         # B=2, N=10, K+1=6（5 类 + no-object）
        "pred_boxes": torch.rand(2, 10, 4),          # 归一化到 [0,1] 的 cxcywh
    }
    targets = [
        {"labels": torch.tensor([0, 2]), "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.3], [0.2, 0.7, 0.3, 0.2]])},
        {"labels": torch.tensor([1, 3, 4]), "boxes": torch.rand(3, 4)},
    ]
    indices = matcher(outputs, targets)
    for b, (p, g) in enumerate(indices):
        print(f"batch {b}: pred_idx={p.tolist()}, gt_idx={g.tolist()}")
```

代码里几个关键点：

- **代价矩阵构造**：分类代价用 `-prob`（最大化概率 = 最小化负概率），bbox 代价用 L1，GIoU 代价用 `-GIoU`。三者加权和作为最终代价。
- **`torch.cdist(p=1)`**：等价于 L1 距离，复杂度 $O(NM)$。
- **`scipy.optimize.linear_sum_assignment`**：Hungarian 算法的标准实现。注意它只支持 CPU，所以代价矩阵要 `.cpu()`。
- **逐图求解**：batch 内每张图独立做匹配，因为不同图的 M 不同。
- **`@torch.no_grad()`**：匹配本身不需要梯度。

把这段 Hungarian matcher 与 detection head 接在一起，就是一个最简版 DETR 训练框架的核心。再把分类损失 + L1 + GIoU 加权求和，就得到 DETR 的训练目标。

## 七、对比：DETR vs Faster R-CNN vs YOLO

| 维度 | Faster R-CNN | YOLOv8/v10 | DETR / DINO |
| --- | --- | --- | --- |
| **Pipeline** | 两阶段：RPN + RoI Head | 单阶段：单次回归 | 端到端：集合预测 |
| **核心组件** | Anchor + RPN + RoI Align + NMS | Anchor-free + 解耦头 + NMS | Object Queries + Hungarian + 无 NMS |
| **标签分配** | MaxIoU / OTA | TAL / SimOTA | Hungarian 二分图匹配 |
| **小目标** | FPN 后表现良好 | FPN/PAN 多尺度 | 多尺度 Deformable Attention 优秀 |
| **速度** | 中（5-15 FPS） | 快（50-150 FPS） | 中-慢（10-30 FPS，Deformable / DINO 可达 25+） |
| **精度** | 高 | 中-高 | 高（DINO COCO mAP 63+） |
| **训练稳定性** | 较稳 | 较稳 | 早期不稳，需 warmup + 大量 epoch |
| **可解释性** | 中（anchor 物理意义清晰） | 中 | 较高（attention 可视化） |
| **架构创新** | RPN + RoI Pooling | Anchor-free / NMS-free | Set Prediction + Hungarian |

**如何选型**：

- **追求极致精度 + 资源充足**：DINO、Deformable DETR。
- **追求实时性 + 中等精度**：YOLOv8/v10/v11。
- **追求训练简洁 + 高质量 proposal**：DETR 系（无需设计 anchor）。
- **追求稳定训练 + 工业部署**：Faster R-CNN 或 RTMDet。

## 八、DETR 训练中的工程细节

### 8.1 辅助损失（Auxiliary Loss）

DETR 在每个 decoder 层都加了辅助损失，强制每一层都做"完整的预测"。具体做法：把每一层 decoder 的输出接到一个共享的 FFN 头上，计算分类 + bbox 损失，与最终层损失求和：

$$
\mathcal{L}_{\text{aux}} = \sum_{l=1}^{L} \mathcal{L}_{\text{DETR}}(y, \hat{y}^{(l)})
$$

其中 $\hat{y}^{(l)}$ 是第 $l$ 层 decoder 的输出。辅助损失让每一层都"逼着自己"学会预测，加快收敛，也稳定了训练。

### 8.2 学习率调度

DETR 的 backbone 用很小的学习率（$\text{lr}_{\text{backbone}} = 10^{-5}$），其他部分用 $\text{lr} = 10^{-4}$，transformer 部分的 weight decay 设为 $10^{-4}$。学习率在最后几个 epoch 衰减到 $10^{-5}$。

### 8.3 数据增强

DETR 默认使用标准的几何增强：random crop（裁剪后保证最小 box 仍可见）、horizontal flip、multi-scale（训练时随机选 [480, 800] 区间的尺寸）。颜色增强使用 photometric distortion（如色调抖动、亮度抖动）。

### 8.4 与 ViT-based 检测器的衔接

DETR 之后出现的 **ViT-based 检测器**（如 ViTDet、Cascade Mask R-CNN with ViT）通常把 CNN backbone 换成 Vision Transformer，保留 DETR 的检测头或换成 Cascade 结构。ViTDet 的关键观察是：**只要 mask 窗口 attention 足够大，ViT 就能在不依赖 FPN 的情况下达到甚至超过 CNN-FPN 的精度**。这进一步模糊了"detection backbone"与"classification backbone"的边界。

## 九、扩展：DETR 在其他任务的应用

DETR 的 set prediction 范式并不局限于物体检测，它的思想已经被推广到多个视觉任务：

### 9.1 分割任务：MaskFormer / Mask2Former

**MaskFormer**（2021）把语义分割、实例分割、全景分割统一为"mask classification"任务：

1. 用 DETR 的解码器输出一组 mask embedding + class label。
2. 把每个 embedding 与 backbone 特征图做点积，得到一个二值 mask。
3. mask × class label = 一组实例。

**Mask2Former**（2022）进一步引入 **masked attention**——每个 query 只 attend 自己当前预测的 mask 区域，大幅提升收敛速度和精度。在全景分割上达到 57.8 PQ、实例分割 50.1 mAP，成为分割任务的新 SOTA。

### 9.2 多目标跟踪：TrackFormer

**TrackFormer** 把 DETR 扩展到多目标跟踪：在不同帧之间复用 object query 作为"track query"，让同一目标在不同帧共享一个 query id。query 的"身份"自然地编码了跟踪 ID，无需额外的 re-ID 网络。

### 9.3 开放词汇检测：Grounding DINO / OWL-ViT

**Grounding DINO** 把 DETR 与文本编码器结合：把"类别标签"从固定 80 个扩展到任意自然语言描述（如"红色汽车"）。它把每个类别名编码成 text embedding，与图像特征在 cross-attention 中融合，实现开放词汇检测。

### 9.4 3D 检测：DETR3D / PETR

**DETR3D**（2021）首次把 DETR 范式扩展到 3D 物体检测。它把 2D 图像 query 映射到 3D 参考点，再从多视角图像中采样 3D 特征，避免显式构建 3D 体素特征。**PETR**（2022）进一步把 3D 位置编码直接加到 2D 特征上，让 query 能直接"看到"3D 空间。

## 十、小结

DETR 的意义不只是提出一个新模型，而是**重新定义了一个范式**：

1. **Set Prediction 取代了手工规则**：anchor 设计、NMS、IoU 阈值这些"启发式"被 Hungarian 匹配取代；分类与回归通过 set matching 自动对齐。
2. **端到端成为可能**：从图像输入到边界框输出，整个 pipeline 没有任何手工设计的非可微环节，可以纯反向传播优化。
3. **Transformer 给检测带来新工具**：cross-attention 让 query 能"全局扫描"图像，object queries 让模型自适应地分工，多尺度 deformable attention 让小目标问题有了优雅解。

但 DETR 也并非银弹——它的训练代价、对初始化的敏感性、对大图的二次复杂度，都让后续 Deformable DETR、DINO、D-FINE 等工作继续深挖。当前的共识是：**DETR 范式更适合精度优先的中大规模场景，YOLO 范式更适合实时性优先的工业部署**。

从 DETR 出发，set prediction 的思想正在向分割、跟踪、3D 检测、开放词汇检测全面渗透。掌握 DETR 的 set prediction、object queries、Hungarian matching 三大核心思想，是理解现代检测器——尤其是 ViT-based 检测器、Grounding DINO、开放词汇检测（OWL-ViT）——的钥匙。后续我们将沿着这个方向，看到 Transformer 如何从检测扩展到更多视觉任务。
