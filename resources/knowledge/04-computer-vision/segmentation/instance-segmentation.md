# 实例分割与全景分割：从 Mask R-CNN 到 SAM

语义分割知道"每个像素是哪一类"，但当图里有多只猫、多辆车时，它会把它们涂成同色——**实例分割（Instance Segmentation）**就是补齐这一环：把"是哪一只"也预测出来；**全景分割（Panoptic Segmentation）**则更进一步，把 stuff（天空、道路这种无边界的背景）与 things（行人、汽车这种离散个体）**统一**在一张图里。本文梳理 Mask R-CNN 的奠基设计、单阶段实例分割、Panoptic FPN / UPSNet、Mask2Former 的统一范式，最后落到 SAM 等现代大模型。

## 一、任务定义：实例、全景、与语义分割的差异

### 1.1 实例分割

输入一张图，输出**一组实例**：

$$
\mathcal{Y} = \{ (c_k, m_k, b_k) \}_{k=1}^{K}
$$

其中 $c_k$ 是类别标签，$m_k \in \{0,1\}^{H \times W}$ 是二值 mask，$b_k \in \mathbb{R}^4$ 是 bounding box，$K$ 是实例数（每张图不一样）。

**关键区别**：

| 任务 | 输出粒度 | 区分个体 |
|---|---|---|
| 语义分割 | 每个像素 → 类别 | 否（同类的所有个体合并） |
| 实例分割 | 每个像素 → 实例 ID + 类别 | 是 |
| 全景分割 | 每个像素 → (类别, 实例 ID) | 是，且包含 stuff |

### 1.2 全景分割（Panoptic）

Kirillov et al.（2019）提出"全景分割"概念，**联合**输出两类：

- **Things**（离散个体，如人、车、猫）：每个实例有独立 mask 和类别标签。
- **Stuff**（无定形背景，如天空、道路、草地）：只有类别，没有个体。

$$
\mathcal{Y}_{\text{panoptic}} = \bigcup_{c \in \text{things}} \{(c, \text{instance}_k)\}_k \cup \bigcup_{c \in \text{stuff}} \{(c, \text{stuff mask})\}
$$

每个像素**有且只有一个**类别 + 实例 ID。

物理直觉：全景分割 = **实例分割（things）+ 语义分割（stuff）**。评估指标 PQ（Panoptic Quality）也是这两类的加权和。

## 二、评估指标：COCO mask mAP 与 PQ

### 2.1 COCO mask mAP（实例分割）

对每张图，按 mask IoU 阈值从 0.5 到 0.95（步长 0.05）取平均：

$$
\text{mAP} = \frac{1}{10}\sum_{t \in \{0.5, 0.55, \dots, 0.95\}} \text{AP}_{t}
$$

每个阈值 $t$ 下，预测 mask 与真值 mask 的 IoU $\ge t$ 算 TP。**COCO 难度大，但更接近实际部署需求**。

### 2.2 PQ（Panoptic Quality）

全景分割官方指标：

$$
\text{PQ} = \frac{\sum_{(c, k)} \text{IoU}(P_{c,k}, G_{c,k})}{\text{TP} + \frac{1}{2}\text{FP} + \frac{1}{2}\text{FN}}
$$

分子是匹配的 (类别, 实例) 对的 IoU 之和，分母是匹配数 + 未匹配对的一半（FP/FN 各罚 0.5）。

物理直觉：**PQ ≈ mAP × 匹配率**，把"分类对不对"和"分割精不精"用同一个数表达。

## 三、Mask R-CNN：在 Faster R-CNN 上加 mask 分支

Mask R-CNN（He et al. 2017）是实例分割的奠基工作。它的结构是 **Faster R-CNN + 一个并行的 mask 分支**。

### 3.1 流水线

1. **Backbone**（ResNet-50/101 + FPN）：提取多尺度特征图。
2. **RPN（Region Proposal Network）**：在特征图上生成候选框（objectness + 粗 box 回归）。
3. **RoIAlign**：把每个 proposal 对应的特征区域抠出来，统一成 $7 \times 7$（box head）和 $14 \times 14$（mask head）。
4. **Box head**：分类 + bbox 回归，得到最终检测结果。
5. **Mask head**：对每个 RoI 预测 $C$ 个 $28 \times 28$ 的二值 mask（每个类别一张），只取预测类对应的那一张作为输出。

物理直觉：**检测 + 像素分类**两个任务共用 backbone，权重互相促进。Mask 分支只跑在已经分类好的 RoI 上，所以小、轻量。

### 3.2 RoIAlign：解决量化的关键

RoIPool 为了把不同大小的 RoI 统一到 $7 \times 7$，要做两次量化（候选框坐标取整 / 内部 bin 划分取整），**最大引入 1 像素误差**——对分类无所谓，对 mask 是致命的：mask 边缘会偏 1–2 像素。

**RoIAlign** 的做法：**不量化**，用双线性插值在浮点坐标上采样到固定大小。物理直觉：让"特征图对齐"和"几何对齐"一致。

$$
\text{RoIAlign}(F, b, \text{size})[i, j] = \text{BilinearInterp}\!\left(F,\ \frac{b_{x_0}}{W} + (j + 0.5)\frac{b_w / W}{w},\ \dots \right)
$$

其中 $W, H$ 是特征图尺寸，$w, h$ 是输出尺寸。代码上就是 `torchvision.ops.roi_align`。

### 3.3 损失函数

Mask R-CNN 是**多任务损失**：

$$
\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}
$$

其中：

- $\mathcal{L}_{\text{cls}}$：交叉熵，$K+1$ 类（含背景）。
- $\mathcal{L}_{\text{box}}$：smooth L1（与 Faster R-CNN 相同）。
- $\mathcal{L}_{\text{mask}}$：对每个 RoI，预测 $C$ 张 $28\times28$ 的二值 mask，**只对真值类别那张算 BCE**——这就是 "decoupled mask" 思想，**避免类别间竞争**。

$$
\mathcal{L}_{\text{mask}} = -\frac{1}{m^2}\sum_{i,j}\left[y_{i,j}\log \hat{m}_{c^*, i,j} + (1 - y_{i,j})\log (1 - \hat{m}_{c^*, i,j})\right]
$$

其中 $c^*$ 是该 RoI 的真值类别，$m = 28$。

简化版 mask head 的 PyTorch 实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskHead(nn.Module):
    """简化版 Mask R-CNN mask head：4 层 3x3 conv + deconv + 1x1 预测每类 mask。"""
    def __init__(self, in_channels: int = 256, num_classes: int = 80, hidden: int = 256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden,     hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden,     hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden,     hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.deconv = nn.ConvTranspose2d(hidden, hidden, 2, stride=2)   # 14 -> 28
        self.predictor = nn.Conv2d(hidden, num_classes, 1)              # 每类一张 mask

    def forward(self, roi_feats):
        x = self.block(roi_feats)               # (N, hidden, 14, 14)
        x = F.relu(self.deconv(x), inplace=True)  # (N, hidden, 28, 28)
        return self.predictor(x)                  # (N, num_classes, 28, 28)
```

完整 Mask R-CNN 直接用 `torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")` 即可。

### 3.4 Cascade Mask R-CNN：Cascade 思想

Mask R-CNN 一个常见升级是 **Cascade**：把单个 box head 换成 3 个串行的 head，每个用更严格的 IoU 阈值（0.5 / 0.6 / 0.7）训练。物理直觉：**先用宽松标准找头，再用更精细的回归"打磨"**——Cascade Mask R-CNN 在 COCO test-dev 上能到 ~40 mAP，比基础版涨 2–3 点。

## 四、单阶段实例分割：YOLACT 与 SOLO

两阶段（proposal + refine）虽然准但慢。单阶段方法想做 **"端到端一次出 mask"**。

### 4.1 YOLACT（Bolya et al. 2019）

YOLACT = You Only Look At Coefficients。核心思想：**把 mask 拆成"原型 + 系数"**。

$$
\text{Mask}_k = \sigma\!\left(\sum_{p=1}^{P} c_{k,p} \cdot P_p\right)
$$

其中 $P_p \in \mathbb{R}^{H \times W}$ 是 $P$ 个**原型 mask**（全图共享），$c_{k,p}$ 是每个实例的 $P$ 维系数（box head 同时预测）。

物理直觉：**"哪些区域属于实例"由原型决定，"哪些原型属于我这个实例"由系数决定**。组合是线性的，推理极快。YOLACT-550 在 COCO 上 29.8 mAP，跑 33 FPS。

YOLACT 的另一个工程 trick 是 **Fast NMS**：把 N×N IoU 矩阵一次矩阵化算出来，比传统 NMS 快 12 ms。

### 4.2 SOLO（Wang et al. 2020）

SOLO 提出了更优雅的"位置即类别"思路：**把每个像素按"它属于哪个 cell"分类，cell 大小由 instance 尺寸决定**。

- **Semantic category**：每个像素预测 $C$ 维类别。
- **Instance category**：把图分成 $S \times S$ 个 cell，每个像素预测自己在 $x$ 方向属于哪个 cell（$S$ 类）、$y$ 方向属于哪个 cell（$S$ 类）。两者组合得到 instance ID。

物理直觉：**"我在图的左上角第三格 + 类别是猫"就足以定义一个实例**，完全不需要 box / anchor / proposal。

SOLOv2 把 mask 预测也按位置 query 做，推理更快、精度更高。SOLO 还启发了后续的 **CondInst**（用 dynamic mask head）和 **BoxInst**（用 box 投影损失替代 mask loss）。

### 4.3 QueryInst（2021）

把单阶段实例分割推得更远：**直接用 Transformer 解码器预测 mask**，query 是检测 box 的 embedding。物理直觉：**"检测 + mask"不再是两个独立分支，而是同一组 query 的两种输出**。

## 五、全景分割：Panoptic FPN 与 UPSNet

### 5.1 Panoptic FPN（Kirillov et al. 2019）

最简单的全景分割实现：**FPN 上接两个并行的 head**。

- **Instance head**：用 Mask R-CNN 风格的 box + mask 分支，处理 things。
- **Semantic head**：用 FPN 自顶向下融合的特征做逐像素分类，处理 stuff。

最后用一个简单的融合规则：things 的 mask 优先于 stuff，其余像素用 stuff 类别填满。

物理直觉：**把"两套系统"拼起来**，但训练仍是各自损失相加，没有跨任务的端到端协同。

### 5.2 UPSNet（Xiong et al. 2019）

UPSNet（Unified Panoptic Segmentation Network）做得更紧：

- 共享 FPN backbone。
- 三个 head：semantic、instance（Mask R-CNN 风格）、stuff 类别 NMS。
- 加一个 **thing-stuff 融合 module**：当 things mask 互相重叠时按置信度仲裁；当 things mask 没覆盖到的像素就归 stuff。

UPernet 进一步把 instance 分支拆成"无参数 instance embedding"——同一实例的像素 embedding 接近，不同实例远离，再用 mean-shift 聚类。**完全不用 box / anchor**，对东西类别都很灵活。

### 5.3 Panoptic-DeepLab（Cheng et al. 2020）

Panoptic-DeepLab 走"双 head"路线：

- **Instance head**：预测每个像素到实例中心点的 offset（类似 VarNet），然后 mean-shift 聚类。
- **Semantic head**：双 DeepLab 卷积头。

优势：**没有 box / anchor / NMS**，训练和推理都比 Mask R-CNN 风格的框架更简洁。在 Cityscapes panoptic 上 65 PQ。

## 六、Mask2Former：统一三任务的大一统框架

Mask2Former（Cheng et al. 2022）的设计哲学：**任何分割任务 = 给一组 query 分配 mask**。

### 6.1 三任务统一

| 任务 | Query 数 | 输出 |
|---|---|---|
| 语义分割 | $C$（每类一个 query）| 每个 query 一张 mask |
| 实例分割 | $N$（远大于实例数，如 100）| query + mask，Hungarian 匹配 |
| 全景分割 | 上面两者结合 | things 用 instance 分支，stuff 用 semantic 分支 |

### 6.2 Masked Attention

普通 cross-attention 让每个 query 关注整张特征图。Mask2Former 改成**只在自己预测的 mask 区域**做 attention：

$$
\text{MA}(q, F, M) = \text{softmax}\!\left(\mathcal{M} + \frac{q K^{\top}}{\sqrt{d}}\right) V
$$

其中 $\mathcal{M}_{i,j} = 0$ if 像素 $j$ 在上一轮预测的 mask $M$ 内，否则 $-\infty$。

物理直觉：**"先画个圈，再看圈里"——强制 query 只关注自己负责的区域，避免不同 query 之间的"注意力打架"**。

### 6.3 Hungarian Matching

训练时把 $N$ 个预测 query 与真值实例做最优匹配（$\arg\min$ 总 cost），匹配上的对算 loss，未匹配预测当 background。cost = 分类 loss + mask loss + box loss 加权和。

同一套代码在 COCO panoptic、COCO instance、ADE20K semantic 三套 benchmark 上都能跑出 SOTA，是"分割 Transformer"的**事实标准**。

```python
# 伪代码：Mask2Former 一个解码层的核心
def masked_cross_attention(query, key, value, mask):
    # mask: (Nq, H, W) bool/float，上一轮预测的二值 mask
    scores = einsum("qd, kd -> qk", query, key) / sqrt(d)         # (Nq, H*W)
    if mask is not None:
        scores = scores.masked_fill(~mask.flatten(1), float("-inf"))
    attn = scores.softmax(-1)
    return einsum("qk, kd -> qd", attn, value)
```

Mask2Former 在 COCO panoptic 上 57.8 PQ，是 2022–2024 年各大 benchmark 的常胜将军。

### 6.4 OneFormer

OneFormer（2023）是 Mask2Former 的进一步泛化：**单个模型同时支持语义 / 实例 / 全景**，且只需**单一训练任务**——通过 task-conditioned query 和 text-conditioned query 让模型自动适配任务。这是"分割大一统"的终极形态之一。其核心 trick：

- **Task token**：在 query 里加一个 learned task token（"semantic" / "instance" / "panoptic"），让模型知道当前该输出什么。
- **Text query**：把类别名也作为 query 喂进去，模型自动把"语义知识"和"几何信息"对齐。

### 6.5 Mask2Former 的工程取舍

| 项 | 选择 | 原因 |
|---|---|---|
| Backbone | Swin-L / InternImage | 多尺度特征 + 强表征 |
| Decoder 层数 | 9 层 | 平衡精度与速度 |
| Query 数 | 100（instance）| 远大于实例数上界 |
| 训练 epoch | 50–100（panoptic）| Transformer 需要长训 |
| Loss 权重 | cls:mask = 5:5:5:5 | 经验值 |

## 七、现代发展：SAM 与开放词汇分割

### 7.1 SAM（Segment Anything Model, Meta 2023）

SAM 走的是另一条路：**prompt-based 分割**。给定 prompt（点 / box / mask），输出对应 mask。

- **Image encoder**：ViT-H/16（632M 参数）。
- **Prompt encoder**：把点 / box / mask 转成 token embedding。
- **Mask decoder**：轻量 Transformer，两轮 cross-attention 输出 mask。

SAM 在 SA-1B（11 亿 mask）上训练，**zero-shot 即可分割从未见过的类别**，是"分割基础模型"的代表。

物理直觉：**"分割"被定义成"找出 prompt 指向的区域"**——把语义类别从模型里剥离，prompt 来决定切什么。

SAM 的关键工程 trick：

- **Amber-size mask decoder**：只有 4M 参数，比 image encoder 小 100 倍——保证交互式响应的关键。
- **Multi-mask 输出**：每个 prompt 输出 3 张 mask + IoU 预测，取 confidence 最高的。处理 ambiguous prompts（如一个点可能对应多个目标）。
- **Focal loss + Dice loss 混合**：默认 20:1 权重，兼顾全局与边界。

### 7.2 SAM 2 与 HQ-SAM

- **SAM 2（Meta 2024）**：把 SAM 扩到**视频**，引入 memory attention 模块传播跨帧 mask。在 SA-V 数据集（5090 万 mask）上训练。
- **HQ-SAM（2023）**：用 Hi-Fi 模块替换 mask decoder，输出分辨率从 $256\times256$ 提到 $1024\times1024$，细化小目标 / 细线（电网、钢丝、玻璃裂纹）。

### 7.3 开放词汇分割（Open-Vocabulary）

- **CLIP-Surgery / MaskCLIP**：把 CLIP 的 patch embedding 当 mask 特征，类别由文本 prompt 提供。
- **ODISE / FC-CLIP**：用 diffusion 模型的 text-to-image prior 当分类器。
- **Grounded-SAM**：结合 Grounding DINO 的文本 grounding，能"分割图里所有出现'红色气球'的东西"。

开放词汇分割让"类别数"不再是模型设计的瓶颈——这是 2024 年以来最热的分割方向。

## 八、训练与推理细节

- **多任务 loss 权重**：Mask R-CNN 里一般 cls : box : mask = 1 : 1 : 1；Mask2Former 里 mask loss 权重更高。
- **数据增强**：实例分割对 box 抖动 + 随机裁剪很敏感；Mask2Former 默认 strong augmentation（scale jitter 0.1–2.0）。
- **推理 trick**：
  - Mask R-CNN：NMS（box）+ mask 阈值化（0.5）。
  - Mask2Former：置信度阈值（0.8）+ mask NMS（按像素 IoU）。
  - SAM：multi-mask 输出（3 张），取 lowest loss 的那张。
- **显存技巧**：大实例分割 batch=2 也行，gradient accumulation 凑等效 batch。
- **半监督**：用 SAM 给 COCO 无标注图打 pseudo mask，再训 Mask R-CNN，能涨 2–3 mAP。

### 8.1 PointRend 与 Boundary 优化

普通 mask head 输出 $28 \times 28$ 的低分辨率 mask，上采样后边缘模糊。**PointRend**（Kirillov et al. 2020）的做法：

1. 在低分辨率 mask 上挑"不确定像素"（mask 边界附近、概率接近 0.5 的点）。
2. 把这些点 + 高分辨率特征送入小 MLP，逐点预测类别。
3. 迭代细化，直到 mask 收敛。

物理直觉：**"不在所有像素上算 mask，只在边界上精修"**——以 4 倍的算力换 2 mAP 提升。

Boundary IoU 也成为新指标，**专门惩罚 mask 边界错误**，比 mask IoU 更敏感于分割质量。

### 8.2 半监督与弱监督

- **NoisyStudent / BoxInst**：只用 box 标注训练 mask（弱监督），loss 用 box 投影一致性。
- **UniverSeg / SAM2Long**：用 SAM 做 cross-image 的 pseudo-label 蒸馏。

### 8.3 推理加速

- **TensorRT 部署**：Mask R-CNN TensorRT 加速后可达 60 FPS。
- **Mask 量化**：把 mask head 量化到 INT8，loss < 1 mAP。
- **Mask 剪枝**：按 mask 重要性剪掉贡献小的 query。

## 九、常见失败模式与诊断

| 现象 | 可能原因 | 调试思路 |
|---|---|---|
| mask 边缘模糊 | backbone 分辨率太低 / RoIAlign 误差 | 改 output stride=8 或用 DeepLab 风格的 dilated backbone |
| 同类实例互相吃掉 | NMS 阈值太严 | 调 IoU 阈值（0.5 → 0.6）或换 soft-NMS |
| 小实例漏检 | FPN 低层信息没进 mask head | 加 P2 分支、focal loss |
| stuff / things 重叠 | 后处理冲突 | 用 Panoptic-DeepLab 的 stuff/thing 双 head + 仲裁 |
| SAM 在细线 / 玻璃上失败 | ViT patch 大小 | 换 SAM2 / HQ-SAM（高清） |
| mask head 显存爆炸 | RoI 数过多 | 加 mask head 量化 / 减小 batch / 梯度累积 |
| 全景分割 RQ 低 | stuff 漏检 | 增加 stuff 类别权重 / 改用 Panoptic-DeepLab 双 head |
| Box AP 高 Mask AP 低 | mask 分支欠拟合 | 加大 mask head 通道 / 多训 30 epoch |
| 大目标 mask 锯齿 | mask head 分辨率太低 | 28×28 → 56×56（PointRend 思路）|
| 训练 loss 抖动 | Hungarian matching 不稳定 | 改 cost weight / 增加 query 数 |

## 十、未来方向

- **分割基础模型**：SAM 2 把 SAM 扩到视频；SAM 3 引入文本 prompt。
- **3D / 4D 分割**：点云、体素、4D 时空。
- **交互式分割**：用户几次点击就修正结果（医疗标注友好）。
- **多模态融合**：用 DINOv2 / CLIP 的强语义特征做 mask 分类，零样本即可。
- **端侧实时**：YOLOv8-seg、RTMDet-Seg 在移动端能跑 30+ FPS。

## 十一、常用数据集一览

| 数据集 | 任务 | 类别数 | 规模 | 特点 |
|---|---|---|---|---|
| COCO | instance / panoptic | 80 / 53 stuff | 200K 图 | 通用最权威 |
| Cityscapes | semantic / panoptic | 19 / 19 stuff + 8 things | 5K 图（精细） | 城市驾驶 |
| LVIS | instance | 1203 | 164K 图 | 长尾实例分割 |
| ADE20K | semantic / panoptic | 150 / 100 stuff | 25K 图 | 场景理解 |
| SA-1B | prompt / SAM | 开放 | 11M 图 + 1.1B mask | SAM 训练集 |
| Mapillary Vistas | semantic | 66 | 25K 图 | 街景高分辨率 |
| YouTube-VIS / OVIS | video instance | 40 / 25 | 4K / 1K 视频 | 视频实例 |
| BDD100K | panoptic | 40 | 100K 图 | 驾驶 + 天气多样 |

物理直觉：**数据集决定任务难度上限**——长尾、密集、遮挡、模糊都靠数据来"教"模型。

## 十二、与上下游任务的关系

实例分割是"检测 + 分割"的合体，常常和检测、深度估计、姿态估计组合部署：

- **Mask R-CNN + keypoint**：在 mask head 旁加 keypoint head，做关键点检测（人体姿态）。
- **Panoptic + depth**：用 depth 估计辅助 stuff 边界（道路/天空的远近）。
- **Track + segment**：把实例分割加 temporal matching，就是 video instance segmentation（VIS）。

工程上：**实例分割是"最贵"的任务之一**——mask head $28 \times 28$ 上采样 + 大量 RoI 都吃显存，所以 batch size 一般 1–2，gradient accumulation 凑等效 batch。

部署时要权衡：

- **Mask head 量化到 INT8**：loss < 1 mAP，显存减半。
- **Mask head 替换成 depthwise conv**：MobileMask 在 iPhone 上 30+ FPS。
- **Query 蒸馏**：把 Mask2Former 蒸馏到 ResNet-50 backbone，推理快 3 倍，仅掉 2 mAP。

## 小结

实例分割的核心挑战是**"区分同类个体"**，全景分割则把 stuff 也纳入统一框架。从 2017 到 2024，这条主线的演进是：

- **Mask R-CNN** 用 RoIAlign + decoupled mask 把两阶段检测扩展成实例分割，是当前最成熟的 baseline；
- **YOLACT / SOLO** 把 mask 预测提速到单阶段，证明了"线性原型组合"和"位置即类别"的可行性；
- **Panoptic FPN / UPSNet** 是全景分割的初代框架，但仍是两套 head 拼起来；
- **Mask2Former** 用 masked attention + Hungarian matching 把三任务统一到同一套 Transformer 里；
- **SAM / 开放词汇分割** 把"类别"从模型内搬到了 prompt 端，定义了"分割基础模型"的新范式。

实务上：中小数据、追求稳定就用 Mask R-CNN；想要 SOTA 上 Mask2Former；想做交互 / 零样本直接上 SAM。下一篇我们将进入"3D 视觉"——从 2D 像素走向点云与体素的几何理解。

## 附：参考资源

- **Detectron2**（Facebook Research）：Mask R-CNN / Panoptic FPN 的官方实现，工业级代码库。
- **MMDetection**（OpenMMLab）：覆盖 Mask R-CNN / SOLO / QueryInst / Mask2Former 等几十种模型，配置化训练。
- **Segmentation Models PyTorch**：轻量语义分割库（Pavel Yakubovskiy, GitHub），FCN / U-Net / DeepLab 一键调用。
- **SAM 官方仓库**：Meta AI 提供 ViT-B/L/H 三个尺寸，含 SA-1B 数据集下载链接。
- **Detectron2 / Hugging Face Transformers**：`Mask2FormerForUniversalSegmentation` 已集成，开箱即用。
