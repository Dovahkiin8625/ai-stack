# 现代 CNN 趋势：从 ViT 冲击到 EfficientNet 与 NAS

2020 年 Vision Transformer（ViT）横空出世，在 ImageNet 上以 88.55% top-1 准确率压制了所有 CNN——但代价是**需要 3 亿张图（JFT-300M）预训练**。本文顺着这条线讲清楚：CNN 是怎么回应挑战的（ConvNeXt 把"Transformer 化"的 trick 反向用到 CNN）、MobileNet 如何用深度可分离卷积统治移动端、EfficientNet 用复合缩放和 NAS 找到精度-效率的帕累托前沿，最后给出按部署场景选骨干网络的实战指南。

## 一、ViT 的冲击：CNN 被"超车"了吗？

Dosovitskiy 等人 2020 年的 ViT 把图像切成 16×16 的 patch，当作"词"送入标准 Transformer Encoder。在 JFT-300M 这种巨型数据集上预训练后，ViT-Large 在 ImageNet 上达到 88.55% top-1——**当时所有 CNN 都达不到**。

但这个结论有两个关键限定：

1. **数据规模**：ViT 没有 CNN 的"归纳偏置"（局部性、平移不变性），需要海量数据"自学"这些先验。ImageNet 1.3M 张图训练，ViT 其实比 ResNet 略差。
2. **推理成本**：ViT 的 self-attention 是 $O(N^2)$（$N$ 是 patch 数），高分辨率输入时代价爆炸。

这给 CNN 留下两条反击路线：

- **数据不足时 CNN 仍占优**（工业部署常态）。
- **CNN 可以借鉴 Transformer 的设计**（ConvNeXt）。

## 二、深度可分离卷积：标准卷积的"低秩分解"

标准卷积的计算量（FLOPs）：

$$
\text{FLOPs}_{\text{std}} = H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot k^2
$$

**深度可分离卷积**把它分成两步：

```text
标准卷积:                                深度可分离卷积（两步）:
  一步搞定 (k×k)                          1. Depthwise (k×k, 每个通道独立):
       │                                       │
       ↓                                       ↓
输出 (H×W×Cout)                          输出 (H×W×Cin)
                                        2. Pointwise (1×1, 跨通道混合):
                                              │
                                              ↓
                                        输出 (H×W×Cout)
```

数学上的 FLOPs 比：

$$
\frac{\text{FLOPs}_{\text{DW+P}}}{\text{FLOPs}_{\text{std}}} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}
$$

$k=3$、$C_{\text{out}}=64$ 时，深度可分离卷积的计算量只有标准卷积的约 **1/9**，参数也压缩到约 1/9——代价是略低的精度（一般 1–2%）。这种"极低成本"是 MobileNet 在端侧可行的根本。

## 三、MobileNet 家族：v1 → v2 → v3 的进化

### v1（2017）：深度可分离 + 1×1 升维 + ReLU6

基础块：Depthwise 3×3 → Pointwise 1×1。ReLU6 把输出限制在 [0, 6]，对低精度推理友好。

### v2（2018）：Inverted Residual + Linear Bottleneck

v2 发现：**bottleneck 里不要用 ReLU**——低维空间里的 ReLU 会破坏信息。建议：

- **升维**用 1×1 conv 把通道数扩 6 倍（t=6）。
- 中间用 Depthwise 3×3 提取特征。
- **降维**用 1×1 conv，但**不用** ReLU（线性瓶颈）。
- 输入输出是低维，用 shortcut 连接——方向和 ResNet 相反，故称 **Inverted Residual**。

```text
MobileNetV2 Block:

       x (低维 h×w×c_in)
        │
        │ 1×1 conv (升维, ReLU6)
        ↓
       6c_in (h×w×6c_in)
        │
        │ 3×3 depthwise (ReLU6)
        ↓
       6c_in (h×w×6c_in)
        │
        │ 1×1 conv (降维, Linear!)
        ↓
       c_out (h×w×c_out)
        │
   ───── + ─────  (x 的残差)
        │
        ↓
       输出
```

### v3（2019）：NAS 搜索 + Squeeze-and-Excitation

v3 用**平台感知 NAS（platform-aware NAS）**搜索出最优结构，并加入：

- **Squeeze-and-Excitation（SE）**：全局平均池化 → 两个 FC → sigmoid 得到通道权重，对特征图做通道级加权。
- **h-swish** 激活：`x * \text{ReLU6}(x+3)/6`，比 swish 计算更便宜。
- **MobileDet** 风格的轻量检测头。

MobileNet v3-Small 在 ImageNet 上达到 67.5% top-1，速度比 v2 还快——是 Android/iOS 上的事实标准骨干网络。

## 四、EfficientNet：复合缩放与 NAS

### 动机：单一维度缩放的局限

传统做法是"要更准就加深度、要更快就减宽度"——但**深度、宽度、分辨率三者一起按比例放大更有效**。

$$
\text{depth: } d = \alpha^\phi, \quad
\text{width: } w = \beta^\phi, \quad
\text{resolution: } r = \gamma^\phi
\quad \text{with} \quad \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
$$

$\phi$ 是用户指定的"计算预算"，$\alpha, \beta, \gamma$ 由小网格搜索确定。EfficientNet-B0 是基础网络，B1–B7 用同样的复合系数放大。

### NAS 的基本原理

EfficientNet 的骨架不是手写的，而是**神经架构搜索（NAS）**出来的：

```text
NAS 流程（高层视角）:

  ┌─────────────┐    ┌─────────────────┐    ┌───────────────┐
  │ 搜索空间     │ →  │ 控制器 RNN /     │ →  │ 子网络 A      │
  │ (候选算子)   │    │ 强化学习策略      │    │ (训练+评估)    │
  └─────────────┘    └─────────────────┘    └───────┬───────┘
                                                     │
                                                     ↓
                                              回报 r → 更新控制器
                                                     │
                                                     ↓
                                       ┌─────────────┴─────────────┐
                                       │ 反复数千次，保留 top 配置   │
                                       └───────────────────────────┘
```

简单说：让一个"控制器网络"（通常也是 RNN 或 Transformer）**生成子网络的结构描述**，子网络训练后回报一个精度，把回报反馈给控制器，重复数千次。计算代价很高（EfficientNet-B0 用了 ~3000 GPU days），但找出来的架构通常优于人工设计。

### 性能

EfficientNet-B7 在 ImageNet 上达到 84.3% top-1，参数量 66M——比同精度的 GPipe 小 8.4 倍。

## 五、ConvNeXt：CNN 的"Transformer 化"

Facebook 的 Liu 等人 2022 年提出 ConvNeXt——**用纯卷积网络达到 Swin Transformer 的精度**。方法是从 ResNet-50 出发，逐步"借鉴" Transformer 的设计：

```text
现代化 ResNet 的设计选择 (每一项都提升精度):

  ① Macro 设计:  stem 改 4×4 patchify (像 ViT)
                 block 比例从 (3,4,6,3) → (3,3,9,3) (像 Swin)
  ② ResNeXt 化:  用 depthwise 3×3 + 1×1 (分组卷积)
  ③ Inverted bottleneck:  维度先升后降 (像 MobileNetV2)
  ④ 大核 7×7:   depthwise 核从 3×3 改 7×7
  ⑤ 微观细节:   减少激活函数、LN 替代 BN、GELU 替代 ReLU
```

ConvNeXt-Tiny 在 ImageNet 上达到 82.1% top-1，**接近 Swin-T 但全部是卷积**——这说明 Transformer 的精度优势并不必然来自 attention，而是来自一系列**可独立移植的设计选择**。

> 启示：当数据规模是中等到大（ImageNet 级别）时，CNN 和 Transformer 各有所长；选哪个更多取决于工程栈、推理框架和团队熟悉度。

## 六、现代架构性能对比

| 模型 | 年份 | Top-1 | 参数量 | FLOPs | 适用场景 |
|---|---|---|---|---|---|
| ResNet-50 | 2015 | 76.1% | 25.6M | 4.1G | 通用基线 |
| EfficientNet-B0 | 2019 | 77.1% | 5.3M | 0.4G | 移动端 |
| EfficientNet-B7 | 2019 | 84.3% | 66M | 37G | 服务器高精度 |
| ViT-L/16 | 2020 | 87.76% | 304M | 190G | 大数据预训练 |
| Swin-T | 2021 | 81.3% | 28M | 4.5G | 通用 |
| ConvNeXt-T | 2022 | 82.1% | 28M | 4.5G | CNN 升级首选 |
| MobileNetV3-S | 2019 | 67.5% | 2.5M | 0.06G | 端侧 / 移动 |
| MobileNetV3-L | 2019 | 75.2% | 5.4M | 0.22G | 移动 + 服务端 |

注：ViT-L 的精度依赖 JFT-300M 预训练；纯 ImageNet 训下来通常 76–78%。

## 七、对象检测为什么依赖这些骨干

骨干网络（backbone）是检测 / 分割模型的"特征提取器"。常见组合：

```text
检测器                典型 backbone         应用
─────────────────────────────────────────────
Faster R-CNN         ResNet-50 / ResNet-101
YOLOv5 / v11         CSP-DarkNet / MobileNet
DETR                 ResNet-50 / Swin-T
Mask R-CNN           ResNet-101 + FPN
CenterNet             Hourglass / DLA
EfficientDet          EfficientNet-B0..B3
```

EfficientDet 还做了一件有意思的事：把 **EfficientNet 的复合缩放扩展到检测器**——同时缩放 backbone、BiFPN、检测头，得到 B0–B7 的一套"端到端最优化"模型家族。

## 八、PyTorch：用 EfficientNet-B0 做推理

```python
import torch
from torchvision.io import read_image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import functional as F

# 加载预训练模型 + 官方推荐的预处理
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights).eval()
categories = weights.meta['categories']

# 读图 + 预处理
img = read_image('cat.jpg')                       # (3, H, W)
img = F.resize(img, [256])                        # 短边 resize
img = F.center_crop(img, [224])                   # 中心裁剪
img = F.to_dtype(img, torch.float32, scale=True)  # → [0, 1]
img = F.normalize(img,
    mean=[0.485, 0.456, 0.406],                   # ImageNet 均值
    std =[0.229, 0.224, 0.225])                   # ImageNet 标准差
img = img.unsqueeze(0)                            # (1, 3, 224, 224)

with torch.inference_mode():
    logits = model(img)                           # (1, 1000)
    prob   = logits.softmax(dim=1)                # 概率

top5 = torch.topk(prob, 5)
for score, idx in zip(top5.values[0], top5.indices[0]):
    print(f'{score.item():.4f}  {categories[idx]}')
```

EfficientNet-B0 模型约 5.3M 参数，CPU 上单张图推理通常 < 30ms——可以放心跑在普通服务器甚至端侧。

## 九、实战选型：什么时候选哪个骨干

```text
场景                                推荐骨干
─────────────────────────────────────────────
移动 App / 端侧实时推理              MobileNetV3-S  / EfficientNet-B0
嵌入式设备 / 低算力                  MobileNetV3-S  (≈2.5M 参数)
服务器高精分类                       EfficientNet-B5/B7  或 ConvNeXt-Large
通用研究 / 论文基线                  ResNet-50  (可复现性最好)
要追赶 SOTA 且数据大                ConvNeXt-Large  或 Swin-Large
目标检测 backbone                    EfficientNet  / ResNet  / Swin
语义分割 backbone                    ResNet + U-Net  / ConvNeXt
小数据集 (<10K 图)                   ResNet-50 预训练 + 冻结微调
```

**经验法则**：

1. **先跑 ResNet-50 / EfficientNet-B0 当 baseline**，再考虑升级。
2. **精度-效率权衡**：EfficientNet-B 系列几乎是当前性价比最优解。
3. **端侧部署**：必选 MobileNetV3-S 或 EfficientNet-B0，不要直接上 ResNet-50。
4. **极端轻量**（<1M 参数）：考虑 MCUNet、TinyNAS 等面向微控制器的 NAS 架构。
5. **数据量决定上层架构**：<10K 张图不要轻易用 ViT/ConvNeXt-Large，CNN 小骨干 + 强数据增强 + 预训练才是稳妥方案。

## 小结

现代 CNN 的两大主线是**效率**与**精度-效率权衡**：MobileNet 用深度可分离卷积统治了移动端，EfficientNet 用 NAS + 复合缩放在服务器侧达到最优精度-参数比，ConvNeXt 则证明 CNN 在借鉴 Transformer 设计后仍可达到顶级精度。下次面对一个新视觉任务，先确定**部署环境**（端侧 / 服务器 / 云）和**精度目标**，再在表格里选骨干：移动端选 MobileNetV3-S，服务器高精度选 EfficientNet-B5/B7 或 ConvNeXt-Large，要追赶 SOTA 且数据充足时再考虑 ViT/Swin 系列。选对骨干，后续训练 / 微调的难度就降一半。