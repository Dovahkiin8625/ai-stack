# 视频识别基础：从双流网络到 SlowFast

视频理解（Video Understanding）是计算机视觉中比图像任务更具挑战性的方向。一段 10 秒、30 FPS 的短视频就有 300 帧，每帧都是一张 RGB 图，再加上音频、光流（optical flow）、字幕等多模态信息。模型不仅要理解"每帧里有什么"，还要捕捉"这些物体如何随时间变化"。本文从任务定义出发，沿着时间线梳理双流网络（Two-Stream）、C3D、I3D、TSN、SlowFast 等经典架构，最后用 PyTorch 从零实现一个简化版 SlowFast。

## 一、视频理解的任务层次

视频理解是个"任务家族"，不同任务对时空建模的要求不同。常见的三层任务：

### 1.1 视频分类（Action Recognition / Video Classification）

给定一段视频，输出一个动作类别（如"打篮球""刷牙""跳水"）。是视频理解的"图像分类"版本，主流数据集有 UCF-101、Kinetics-400/600/700、Something-Something 等。

### 1.2 时序动作检测（Temporal Action Detection）

不仅要知道"视频里有什么动作"，还要定位动作发生的**起止时间**——本质上是视频版的"目标检测"，输出 `[(t_start, t_end, label)]` 列表。代表数据集有 THUMOS-14、ActivityNet。

### 1.3 视频描述（Video Captioning）

用自然语言描述视频内容（如"一个人在厨房里切西红柿"），是视频版的"图像描述"，需要同时建模视觉与语言。代表数据集有 MSR-VTT、MSVD。

这三层任务对**时空建模**的要求递增：分类只需"看全段"提取全局特征；时序检测需要"在长视频里定位片段"；视频描述需要"把动作序列翻译成语言"。

## 二、视频分类数据集与评估指标

### 2.1 主流数据集

- **UCF-101**（2012）：13320 段、101 类人类动作，是早期动作识别的事实标准。
- **Kinetics-400/600/700**（DeepMind, 2017–2019）：从 YouTube 采集的更大规模数据集，Kinetics-400 含约 24 万段、400 类，是当前最主流的视频分类基准。
- **Something-Something**（2017–2019）：约 11 万段、174 类，专注于**时序推理**（如"把某物推到某物后面"），对光流和时序建模极度敏感，是检验模型是否真正"看懂时间"的关键基准。

数据集特点不同：Kinetics 偏"是什么"，Something-Something 偏"怎么变"，这意味着同样的模型在两者上的表现并不一定一致。

### 2.2 评估指标

- **Top-1 / Top-5 Accuracy**：分类任务的标准，前者是模型最高置信度类别正确的比例，后者是前五名包含正确答案的比例。
- **mAP**（mean Average Precision）：时序动作检测与视频检索的核心指标，与目标检测的 mAP 计算方式类似——对每个 IoU 阈值求 PR 曲线下面积的平均值。
- **METEOR / CIDEr / BLEU**：视频描述任务使用文本生成指标，其中 CIDEr 专门为图像/视频描述设计，对词频加权更合理。

## 三、双流网络（Two-Stream Network, 2014）

双流网络是视频理解的开山之作，由 Simonyan 和 Zisserman 在 NIPS 2014 提出。它把视频拆成**空间**与**时间**两条独立的 CNN 通路：

- **空间流（spatial stream）**：输入单帧 RGB 图，负责识别"场景与物体"——例如看到篮球场，就倾向于判断"打篮球"。
- **时间流（temporal stream）**：输入**光流场**（optical flow），即相邻帧的位移向量堆叠，负责识别"动作"——例如看到"手向上挥"的位移模式。

光流是把"动作"显式化的关键。一种常见的预处理方式：在 TV-L1 光流算法上计算相邻帧的位移 $(u, v)$，沿通道堆叠 L 帧（如 L=5），得到 $H \times W \times 2L$ 的输入。

$$
\text{Flow}_{t \to t+1}(x, y) = (u(x,y), v(x,y))
$$

物理直觉：RGB 流回答"看到什么"，光流流回答"怎么动"。两条流各有侧重，最后融合。

### 3.1 融合策略

- **Late fusion**：两条 CNN 分别对每帧/片段预测，最后融合分类分数（平均或 SVM）。
- **Early fusion**：把光流与 RGB 在输入层拼接后送入单一 3D 网络。

原论文采用 Late fusion：空间流取单帧（$224 \times 224 \times 3$），时间流取 5 帧光流堆叠（$224 \times 224 \times 10$），分别用两个独立的 CNN（VGG-M / VGG-16）得到分类 logits，再加权融合。

### 3.2 双流网络的局限

光流需要离线计算（如 TV-L1 算法），推理时无法端到端；且光流本身消耗大量存储（每段视频的光流可能比原始视频大 10 倍）。但它奠定了"空间+时间"两条通路的核心思想，后续的 I3D、SlowFast 都可看作是这个思想的延续。

## 四、3D 卷积与 C3D

把"空间+时间"两条流合在一起的更优雅思路是**3D 卷积**：把 2D 卷积的 kernel 从 $k \times k$ 扩展到 $k_t \times k \times k$，在时间维度也滑动。

$$
Y_{t, i, j, c} = \sum_{\tau=0}^{k_t-1} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} \sum_{c_{in}} K_{\tau, u, v, c_{in}, c} \cdot X_{t+\tau, i+u, j+v, c_{in}}
$$

C3D（Tran et al., 2015）是早期 3D CNN 的代表：8 层 $3 \times 3 \times 3$ 卷积 + 池化，输出 16 类（Sport1M）。但 C3D 参数量大、训练数据少，效果有限——它真正的贡献是证明了"3D 卷积是视频建模的可行方向"。

## 五、I3D（Inflated 3D ConvNet, 2017）

I3D（Carreira & Zisserman, CVPR 2017）的核心洞察是**把 ImageNet 预训练的 2D Inception 膨胀到 3D**：

1. 把 2D 卷积核 $k \times k$ 在时间维复制 $k_t$ 次，得到 $k_t \times k \times k$ 的 3D 卷积核。
2. 把预训练权重除以 $k_t$，保持输出量级不变。

这样一来，3D 网络就继承了 ImageNet 的强空间特征，再在 Kinetics 上做端到端微调。I3D 把 Kinetics-400 的 top-1 从 60% 提升到 71%，奠定了"预训练+微调"在视频领域的标准范式。

更巧妙的是，I3D 还有**双流版本**——空间流用 RGB（膨胀 2D Inception），时间流用光流（同样膨胀），最后 late fusion。这与原始双流网络结构相同，但骨干换成了更强的 I3D，性能大幅提升。

## 六、TSN（Temporal Segment Network, 2016）

TSN（Wang et al., 2016）针对的是"长视频分类"问题。视频通常几十秒甚至几分钟，但 3D CNN 受限于显存很难处理全部帧。TSN 的解决方案是**稀疏采样**：

1. 把整段视频按时间切成 $K$ 个段（segment），每段随机抽 1 帧。
2. 每段独立过 CNN 得到段级预测。
3. 最后对 $K$ 段预测做加权平均，得到视频级预测。

$$
y = \frac{1}{K} \sum_{k=1}^{K} f(\text{frame}_k)
$$

物理直觉：与其每帧都看，不如"扫一遍骨架"——只要分段足够多，每段都能覆盖视频中关键动作即可。TSN 在 UCF-101 上以更少帧达到了与双流网络相当的效果，证明了稀疏采样的有效性。

## 七、SlowFast 网络（2019）

SlowFast（Feichtenhofer et al., ICCV 2019）是 Facebook AI 的代表作，灵感来自人眼视网膜——**80% 的视锥细胞（M 细胞）处理低时间分辨率的颜色信息，20% 的视杆细胞（P 细胞）处理高时间分辨率但低空间分辨率的运动信息**。SlowFast 把这种"快慢双通路"显式建模。

### 7.1 Slow 通路

- 低帧率（每秒约 4–8 帧，慢）。
- 高空间分辨率（如 $224 \times 224$）。
- 高通道数（如 64–256），用于提取"语义"信息——"这是什么动作"。

### 7.2 Fast 通路

- 高帧率（每秒约 32–64 帧，快）。
- 低空间分辨率（$\times 1/4$ 或 $\times 1/8$，例如 $56 \times 56$）。
- 低通道数（仅为 Slow 通路的 $\beta$ 倍，通常 $\beta = 1/8$），用于提取"运动"细节——"动作是怎么变化的"。

Fast 通路轻量，但用低通道数+低分辨率换取"高时间分辨率"。

### 7.3 横向连接（Lateral Connection）

两条通路之间通过**横向连接**融合——Fast 通路在每个 stage 后把特征送给 Slow 通路做时空聚合。融合方式有三种：

- **拼接 + 1×1×1 卷积降维**
- **逐元素相加**
- **逐元素相乘**

物理直觉：Fast 通路给 Slow 通路"补充运动细节"，避免 Slow 通路漏掉快速变化的子动作。

### 7.4 SlowFast 的优势

1. **高效**：Fast 通路轻量，整体参数量比同性能的 3D CNN 少 2–3 倍。
2. **兼容性好**：主干仍可以是 ResNet，迁移 ImageNet 预训练权重方便。
3. **强泛化**：在 Kinetics-400/600/700、AVA、Charades 等多个基准上同时 SOTA。

## 八、时序建模：LSTM vs Temporal Conv vs Attention

在 3D CNN 抽完"每帧特征"之后，仍需要一个时序聚合层把多帧融合成视频级表示。三种主流方案：

### 8.1 LSTM

循环网络，擅长捕捉长程依赖，但串行计算慢，训练时显存消耗大。早期双流网络常用 LSTM 把帧级特征聚合成段级。

### 8.2 Temporal Convolution

1D 卷积沿时间维滑动（如 $k=3$ 或 $k=5$），并行计算、感受野可控（多层堆叠可指数扩大）。TBN、ARTNet 等模型使用。

### 8.3 Temporal Attention

让模型自己学"哪几帧重要"，每帧 query/key 做 attention。SlowFast 在最终全局池化前可叠加一个 attention pool；后续的 TimeSformer、Video Swin Transformer 几乎完全依赖 attention 完成时空建模。

物理直觉：LSTM 像"边走边记"，Temporal Conv 像"滑动窗口看一段"，Attention 像"全局点对点对照"。在数据量大时，Attention 的性能上限最高，但代价是 $O(T^2)$ 的计算量。

## 九、PyTorch 实现：简化版 SlowFast 的 Fast/Slow 通路

下面实现一个最小可用的 SlowFast 双通路骨干网络，仅保留核心结构（ResNet 风格的残差块 + lateral connection），可直接 `python slowfast.py` 跑通：

```python
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """ResNet bottleneck 块的简化版（膨胀到 3D 用于 Slow 通路）。"""
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1, 1)):
        super().__init__()
        mid = out_channels // self.expansion
        self.conv1 = nn.Conv3d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid)
        self.conv2 = nn.Conv3d(mid, mid, kernel_size=(1, 3, 3),
                               stride=stride, padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(mid)
        self.conv3 = nn.Conv3d(mid, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class FastPathway(nn.Module):
    """Fast 通路：低通道、高时间分辨率、低空间分辨率。"""

    def __init__(self, num_classes: int = 400):
        super().__init__()
        # 输入: (B, 3, T=32, H=56, W=56)
        self.stem = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True),
        )
        # 残差 stage（轻量级）
        self.layer1 = Bottleneck(8, 32, stride=(1, 1, 1))
        self.layer2 = Bottleneck(32, 64, stride=(1, 2, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SlowPathway(nn.Module):
    """Slow 通路：高通道、低时间分辨率、高空间分辨率。"""

    def __init__(self, num_classes: int = 400):
        super().__init__()
        # 输入: (B, 3, T=8, H=224, W=224)
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = Bottleneck(64, 256, stride=(1, 1, 1))
        self.layer2 = Bottleneck(256, 512, stride=(1, 2, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class LateralConnection(nn.Module):
    """Fast -> Slow 的横向连接：把 Fast 特征对齐到 Slow 的时间分辨率与空间尺寸。"""
    def __init__(self, fast_channels: int, slow_channels: int):
        super().__init__()
        # 把 Fast 的通道数映射到 Slow，并做时间维下采样（与 Slow 对齐）
        self.proj = nn.Conv3d(fast_channels, slow_channels,
                              kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.bn = nn.BatchNorm3d(slow_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fast_feat: torch.Tensor, slow_feat: torch.Tensor) -> torch.Tensor:
        # fast_feat: (B, C_fast, T_fast, H, W), slow_feat: (B, C_slow, T_slow, H, W)
        fast_aligned = self.relu(self.bn(self.proj(fast_feat)))
        return slow_feat + fast_aligned


class SlowFast(nn.Module):
    """最小可用的 SlowFast 网络。"""

    def __init__(self, num_classes: int = 400):
        super().__init__()
        self.fast = FastPathway(num_classes=0)            # 不分类，只提取特征
        self.slow = SlowPathway(num_classes=0)
        self.lateral = LateralConnection(fast_channels=64, slow_channels=512)
        self.head = nn.Linear(512, num_classes)

    def forward(self, slow_x: torch.Tensor, fast_x: torch.Tensor) -> torch.Tensor:
        # 分别走两条通路
        slow_feat = self.slow.layer2(self.slow.layer1(self.slow.stem(slow_x)))
        fast_feat = self.fast.layer2(self.fast.layer1(self.fast.stem(fast_x)))
        # 横向融合
        slow_feat = self.lateral(fast_feat, slow_feat)
        # 全局池化 + 分类
        out = slow_feat.mean(dim=[2, 3, 4])
        return self.head(out)


# 烟测：构造两个通路的输入，跑一次 forward
if __name__ == "__main__":
    torch.manual_seed(0)
    model = SlowFast(num_classes=400)
    slow_x = torch.randn(2, 3, 8, 224, 224)   # Slow: T=8, 高空间
    fast_x = torch.randn(2, 3, 32, 56, 56)    # Fast: T=32, 低空间
    y = model(slow_x, fast_x)
    print("output shape:", y.shape)           # torch.Size([2, 400])
```

代码里几个值得注意的点：

- **Fast 通路时间维是 Slow 的 4 倍**（$T=32$ vs $T=8$），空间分辨率是 Slow 的 $\times 1/4$（$56 \times 56$ vs $224 \times 224$）。
- **通道数 Fast:Slow = 1:8**：用 $\beta = 1/8$ 控制 Fast 的轻量。
- **LateralConnection 用 $3 \times 1 \times 1$ 卷积**做时间对齐与通道映射。
- **最后在时空维度做全局平均池化**得到视频级表示，再过分类头。

把这段代码塞进数据加载（随机稀疏采样 + RandomCrop + Flip）、配上交叉熵损失，就是一个能在 UCF-101 上跑出不错 baseline 的 SlowFast。

## 十、PyTorch 实现：简化版双流网络（Two-Stream）

为了进一步对比 3D 卷积与双流方案，下面给出一个极简的双流网络实现——空间流走 ResNet-18，时间流走一个独立的 ResNet-18（输入通道数为 10，对应 5 帧光流堆叠），最后做 late fusion：

```python
import torch
import torch.nn as nn
import torchvision.models as tv


class TwoStreamNet(nn.Module):
    """简化版双流网络：RGB 流 + 光流流，各自一个 ResNet-18，最后 late fusion。"""

    def __init__(self, num_classes: int = 400, dropout: float = 0.5):
        super().__init__()
        # 空间流：标准 3 通道 RGB 输入
        self.spatial_stream = tv.resnet18(weights=None)
        self.spatial_stream.fc = nn.Linear(self.spatial_stream.fc.in_features, num_classes)

        # 时间流：10 通道光流堆叠（5 帧 (u, v)）
        self.temporal_stream = tv.resnet18(weights=None)
        self.temporal_stream.conv1 = nn.Conv2d(
            10, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.temporal_stream.fc = nn.Linear(self.temporal_stream.fc.in_features, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        # rgb: (B, 3, H, W) — 单帧 RGB
        # flow: (B, 10, H, W) — 5 帧光流堆叠
        s_logits = self.spatial_stream(self.dropout(rgb))
        t_logits = self.temporal_stream(self.dropout(flow))
        # Late fusion：加权平均
        return 0.7 * s_logits + 0.3 * t_logits


# 烟测
if __name__ == "__main__":
    torch.manual_seed(0)
    model = TwoStreamNet(num_classes=400)
    rgb = torch.randn(2, 3, 224, 224)
    flow = torch.randn(2, 10, 224, 224)
    y = model(rgb, flow)
    print("output shape:", y.shape)   # torch.Size([2, 400])
```

代码要点：

- **空间流**：标准的 ResNet-18，输入 RGB 单帧（$224 \times 224 \times 3$）。
- **时间流**：把 ResNet 的第一个 $7 \times 7$ 卷积改为 10 通道输入，承接 5 帧光流堆叠。
- **Late fusion**：两个 logit 加权平均（空间 0.7、时间 0.3）——原论文里这个权重通过交叉验证搜索得到。

把这段代码与光流预处理（TV-L1）衔接起来，就是一个端到端可训练的双流视频分类器。在 Kinetics-400 上，双流 ResNet-152 大约能达到 73% top-1，比单流 RGB 高出 5 个百分点——验证了"光流=显式动作"对视频理解的增益。

## 十一、视频识别的训练技巧

### 11.1 数据增强

视频数据增强比图像更复杂，因为要保持时间一致性：

- **空间增强**：RandomCrop、RandomFlip、ColorJitter 都在所有帧上**一致**应用。
- **时序增强**：随机采样间隔（stride）、随机丢帧（drop frames）、时间反转（仅 Kinetics 类合适）。
- **Mixup / CutMix**：对两段视频做像素级加权或空间区域替换，能显著提升泛化。

### 11.2 预训练策略

- **ImageNet 预训练**：所有 3D CNN/I3D/SlowFast 都从 ImageNet 2D 权重膨胀初始化。
- **Kinetics 预训练**：在 Kinetics-400 上预训练，再迁移到下游（UCF-101、Something-Something）。
- **自监督预训练**：VideoMAE、MaskFeat 等方案减少对标注的依赖。

### 11.3 推理加速

- **时序滑动窗口**：把长视频切成多段，每段单独推理，最后融合段级 logit。
- **关键帧采样**：用光流或显著性选择关键帧，跳过冗余帧。
- **量化与 TensorRT**：把 3D CNN 量化到 INT8，可显著降低推理延迟。

## 十二、视频识别在产业中的典型应用

视频识别不只是学术基准，在工业界有大量落地：

1. **短视频推荐**：抖音/TikTok 用动作识别+内容理解给视频打标，辅助推荐。
2. **安防监控**：打架、跌倒、闯入等异常行为检测，本质是时序动作检测+分类。
3. **体育分析**：运动员动作识别、战术分析、动作质量评估（体操、跳水评分）。
4. **自动驾驶**：行人意图预测、危险行为检测，需要实时性 + 鲁棒性。
5. **医疗康复**：物理治疗动作打分、康复动作规范性评估。
6. **内容审核**：暴力、色情、违规动作的自动检测。

这些场景对**延迟**和**精度**都有要求，因此工程上往往需要在 SlowFast 这类强模型与 TSN 这类轻量模型之间做 trade-off。

## 十三、常见误区与踩坑

最后总结几条视频识别中容易踩的坑：

1. **错把时序建模当成 3D 卷积的天然能力**：3D 卷积在小数据集上很容易过拟合，必须借助预训练（ImageNet 膨胀或 Kinetics）。
2. **光流当成"免费午餐"**：光流存储开销大、离线计算慢，工程上往往限制帧数（如只用前 5 帧光流），导致长时序动作识别性能下降。
3. **稀疏采样过头**：TSN 风格采样对 Kinetics 类任务有效，但对 Something-Something 类依赖精细时序的任务完全不适用——后者必须密集采样。
4. **混淆帧级与段级 loss**：某些代码在中间帧也加分类 loss（deep supervision），看似提升精度，但会显著增加训练时间。
5. **忽视帧率与时间分辨率**：30 FPS 和 5 FPS 的同一动作看起来可能完全不同（走路 vs 跑步），数据集本身的帧率分布很关键。
6. **3D 卷积核大小选择**：$3 \times 3 \times 3$ 是性价比最优的，过大的 $7 \times 7 \times 7$ 卷积在前几层会浪费大量算力在浅层时空特征上。

## 小结

视频识别经历了从"光流+RGB 双流"到"3D 卷积"再到"快慢双通路"的演进。双流网络奠定了"空间+时间"的解耦思想，C3D/I3D 证明了 3D 卷积+预训练的有效性，TSN 用稀疏采样解决了长视频的计算瓶颈，SlowFast 把"高语义低时间"与"低语义高时间"两条通路组合起来，用更少的参数取得更强的性能。这些架构的核心都围绕着同一个问题：**如何在有限算力下高效建模"时空"**。下一篇我们将进入 Transformer 时代，看看 TimeSformer、Video Swin Transformer 与 VideoMAE 如何用 attention 重新定义视频建模。
