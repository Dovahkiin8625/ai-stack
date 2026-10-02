# GAN 基础与演进：从博弈论到 StyleGAN

生成对抗网络（Generative Adversarial Network, GAN）是 2014 年由 Ian Goodfellow 提出的生成模型框架，过去十年间深刻改变了图像合成、风格迁移和数据增广的研究范式。其核心思想是让两个神经网络在「零和博弈（zero-sum game）」中相互对抗：生成器（Generator）试图以假乱真，判别器（Discriminator）则努力鉴别真伪。本文从生成模型家族的横向对比出发，逐节推演 GAN 的数学基础、训练难题与关键改进，最后用 PyTorch 实现一个最小可用的 DCGAN 训练循环。

## 一、生成模型三大类：GAN、VAE、Diffusion

在深度生成模型领域，最具影响力的三条技术路线分别是 GAN、变分自编码器（Variational Autoencoder, VAE）与扩散模型（Diffusion Model）。它们的目标都是从数据分布 $p_{\text{data}}(x)$ 中采样，但建模思路截然不同。

| 维度 | GAN | VAE | Diffusion |
| --- | --- | --- | --- |
| 建模对象 | 隐式（implicit） | 显式（ELBO 下界） | 显式（噪声预测） |
| 损失函数 | 对抗式 min-max | 负 ELBO + KL | 噪声 MSE / 简化 ELBO |
| 采样速度 | 极快（一步前向） | 快（一次 decode） | 慢（多步迭代） |
| 训练稳定性 | 较差 | 稳定 | 较稳定 |
| 样本多样性 | 易 mode collapse | 易模糊 | 高质量 + 多样 |
| 似然估计 | 不可直接计算 | 可估计下界 | 可估计下界 |

物理直觉上，GAN 把生成器视作「造假工厂」，判别器是「质检员」；VAE 则让编码器把图片压缩成均值和方差，再用解码器从正态分布中重建；Diffusion 模型则是反复「加噪—去噪」，把图片逐步还原。从 2022 年起，Diffusion 模型（Stable Diffusion、DALL·E 3）成为高质量图像生成的主流，但 GAN 仍在实时合成、风格迁移等场景中扮演重要角色。

一个常被忽略的细节是，**这三条路线的数学本质其实是相通的**：它们都可以被理解为在某个特定函数族上拟合 $\log p_{\text{data}}(x)$ 的下界或替代目标。GAN 通过判别器隐式估计密度比，VAE 通过重构 + KL 直接逼近似然，Diffusion 则在每一个噪声尺度上做变分推断。理解这层关系，就能在工程中根据"质量 vs 多样性 vs 速度"的优先级选择合适的路线。

## 二、GAN 的博弈论基础：min-max 优化

GAN 的核心是一个双人极小极大（min-max）博弈：

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

其中 $G$ 是生成器，输入噪声 $z \sim p_z$（通常为 $\mathcal{N}(0, I)$），输出合成图像 $G(z)$；$D$ 是判别器，输出 $x$ 为真实图片的概率 $D(x) \in [0, 1]$。

**判别器的目标**：最大化 $\log D(x)$（给真图高分）并最小化 $\log D(G(z))$（给假图低分），即最大化整体对数似然。

**生成器的目标**：让 $D(G(z))$ 趋近于 1，等价于最小化 $\log(1 - D(G(z)))$。但实践发现这个目标在训练初期梯度极弱（判别器轻松识别假图，$\log(1 - D(G(z))) \approx 0$），Goodfellow 建议改为**最大化 $\log D(G(z))$**（即「骗过判别器」），这个非饱和目标能提供更强的早期梯度。

理论分析（Goodfellow 2014）证明：当判别器达到最优 $D^*$ 时，生成器的目标等价于最小化 $p_g$（生成分布）与 $p_{\text{data}}$ 之间的 **JS 散度（Jensen-Shannon Divergence）**。具体推导：固定 $G$，对 $D$ 求最优解：

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

代回 $V$ 可得 $\min_G V(D^*, G) = -\log 4 + 2 \cdot \text{JS}(p_{\text{data}} \| p_g)$。因此 GAN 训练最优时等价于**最小化 JS 散度**。这是 GAN 后续改进的源头——能否用更稳定的距离度量替换 JS？

## 三、原始 GAN 的训练不稳定性与 Mode Collapse

虽然原始 GAN 理论优雅，但实际训练中暴露了三大难题：

1. **模式坍缩（Mode Collapse）**：生成器只学会输出某几种「安全」的样本（恰好骗过判别器），无法覆盖真实数据的所有模式。典型表现：训练人脸 GAN 时，所有生成图都是同一个人脸的微调版本。
2. **训练震荡**：$G$ 和 $D$ 的损失曲线此起彼伏，难以收敛到纳什均衡。
3. **梯度消失**：判别器过强时，$G$ 的梯度接近 0；判别器过弱时，又失去对生成器的引导信号。

JS 散度的根本问题是：当 $p_g$ 与 $p_{\text{data}}$ 的支撑集（support）几乎不重叠时，JS 近似为常数 $\log 2$，梯度消失。生成器的梯度公式可写为：

$$
\nabla_{\theta_G} V = \mathbb{E}_{z \sim p_z}\!\left[\nabla_{\theta_G} \log(1 - D(G(z)))\right]
$$

当 $D$ 接近最优且两个分布不重叠时，这一梯度近似为 0，训练停滞。

**缓解 mode collapse 的实用技巧**：

- **小批量判别（Minibatch Discrimination, Salimans 2016）**：让 $D$ 不仅看单样本，还看同一 batch 内的样本统计量，鼓励生成器输出有差异的样本。
- **Unrolled GAN**：在更新 $G$ 时把 $D$ 后续几步的更新"展开"，让 $G$ 提前看到 $D$ 的反应。
- **经验回放（Experience Replay）**：给判别器喂一些历史生成的样本，防止其过度拟合当前生成器。
- **多生成器 / 多判别器**：用 $G_1, ..., G_k$ 和 $D_1, ..., D_m$ 组成集成，分散模式覆盖。

这些问题催生了 WGAN、条件 GAN 和 StyleGAN 等改进。

## 四、DCGAN：用 CNN 替代 MLP

Radford et al. (2015) 提出的 DCGAN（Deep Convolutional GAN）是第一个把 GAN 真正"工业化"的工作，其架构指南后来成为社区的事实标准：

- 用 **strided conv** 替代 pooling 做下采样
- 用 **transposed conv** 做上采样
- 生成器与判别器都用 **BatchNorm**
- 去掉全连接层
- 生成器除输出层外用 **ReLU**，输出层用 **Tanh**
- 判别器全部用 **LeakyReLU**

```python
import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    """DCGAN 生成器：以 100 维噪声生成 64x64 RGB 图像。"""

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: (B, nz, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (B, ngf*8, 4, 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (B, ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (B, ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (B, ngf, 32, 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),  # 输出范围 [-1, 1]
            # (B, nc, 64, 64)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z.view(z.size(0), -1, 1, 1))
```

DCGAN 证明只要架构合理，无监督训练就能学到层次化的视觉特征（早期层学到边缘、纹理，深层学到物体部件），为后续所有 GAN 变体奠定基础。DCGAN 的一个意外副产品是：把生成器的中间激活可视化可以发现"单元神经元对应特定特征"，这启发了后续的可解释性研究。

## 五、条件 GAN（cGAN）：让生成可控

原始 GAN 从纯噪声生成图像，无法控制生成什么。Mirza & Osindero (2014) 提出把类别标签 $y$ 同时喂给生成器和判别器：

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x,y}[\log D(x|y)] + \mathbb{E}_{z,y}[\log(1 - D(G(z|y)))]
$$

实现上，$y$ 通常先编码为 embedding，再与噪声 $z$ 拼接或相加送入生成器。这种「条件化」后来演化出文本到图像（text-to-image）、图像到图像（pix2pix）等众多方向。

### 1. InfoGAN（Chen et al. 2016）

InfoGAN 把隐码拆成两部分：

- **不可压缩噪声** $z$（保持多样性）
- **结构化潜码** $c$（被约束与生成样本互信息最大化）

$$
\min_G \max_D \; V_{\text{Info}}(D, G) - \lambda \, I(c;\, G(z, c))
$$

训练后 $c$ 的不同维度会自动对应"光照方向"、"形状"、"写字风格"等可解释属性，无需任何标签。

### 2. ACGAN（Auxiliary Classifier GAN）

在判别器上加一个**辅助分类头**，要求判别器同时判断"真伪 + 类别"。这比 InfoGAN 弱（需要标签），但更直接地提升了类别一致性。

### 3. Pix2Pix 与 CycleGAN

**Pix2Pix（Isola et al. 2017）**：用 cGAN 做图像翻译（image-to-image translation），条件是另一张图而非标签。损失为 L1 + 对抗损失的加权和。

**CycleGAN（Zhu et al. 2017）**：用两个 GAN 做无配对的图像翻译（如马↔斑马）。关键技巧是 **cycle consistency**：$F(G(x)) \approx x$ 且 $G(F(y)) \approx y$，循环一致性约束让无配对翻译成为可能。这两类工作直接催生了 2018~2020 年的图像编辑潮流。

## 六、WGAN：用 Wasserstein 距离改善训练

Arjovsky et al. (2017) 提出用 **Earth-Mover（Wasserstein-1）距离** 替代 JS 散度：

$$
W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x, y) \sim \gamma}\!\left[\|x - y\|\right]
$$

直觉：把一堆土从一个分布"搬运"到另一个分布所需的最小平均距离。即使两个分布支撑不重叠，$W$ 距离仍然能给出有意义的梯度。

为了让 $W$ 距离可优化，WGAN 要求判别器（此时改称 **critic**）满足 **1-Lipschitz 约束**。原始 WGAN 通过**权重裁剪**（把所有参数 clamp 到 $[-c, c]$）强制 Lipschitz，但这往往导致容量不足。Gulrajani et al. (2017) 改用 **梯度惩罚（gradient penalty）**：

$$
L_D = \mathbb{E}_{z}[D(G(z))] - \mathbb{E}_{x}[D(x)] + \lambda \, \mathbb{E}_{\hat{x}}\!\left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]
$$

其中 $\hat{x}$ 是真实与生成样本的随机插值。WGAN-GP 通常能稳定训练，显著缓解 mode collapse，且 loss 数值与生成质量高度相关——这是工程上一个非常重要的特性。

### 谱归一化（Spectral Normalization, Miyato et al. 2018）

另一种保证 Lipschitz 的做法是**直接约束每一层权重矩阵的谱范数**：

$$
\bar{W} = W / \sigma(W),\quad \sigma(W) = \max_{\|x\|_2 = 1} \|W x\|_2
$$

实现上只需在每一层用 power iteration 估计 $\sigma(W)$，然后除之。相比梯度惩罚，谱归一化计算更便宜，已被广泛集成进现代 GAN 库（如 `torch.nn.utils.spectral_norm`）。

## 七、StyleGAN 系列：分层风格控制

### 1. StyleGAN（Karras et al. 2019）

StyleGAN 把生成器从「噪声→图像」改造为「常量输入 + 风格调制」：

- **常量输入**：用一个 $4\times4\times512$ 的可学习张量作为起点，而非从噪声开始。
- **映射网络 $f$**：把潜码 $z \in \mathcal{Z}$ 通过 8 层 MLP 映射到中间潜空间 $\mathcal{W}$。$\mathcal{W}$ 解耦了潜码中的纠缠因子，使后续控制更精细。
- **自适应实例归一化（AdaIN）**：在每一层把 $\mathcal{W}$ 中的风格向量注入特征图：

$$
\text{AdaIN}(x_i, y) = y_{s,i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}
$$

- **随机变化（stochastic variation）**：在每个层级加入 per-pixel 噪声，生成头发、皮肤等细节。
- **混合正则化**：训练时随机把两张图的某层风格交叉，强化 $\mathcal{W}$ 各层之间的解耦。

直觉：$\mathcal{W}$ 中**较粗粒度**的层（低分辨率 4×4、8×8）控制姿态、脸型；**中粒度**层（16×16、32×32）控制发型、五官；**细粒度**层（64×64 以上）控制肤色纹理、微表情。StyleGAN 让"控制风格"变成在不同层级注入不同潜码。

### 2. StyleGAN2（2020）

StyleGAN2 发现 AdaIN 会产生类似"水滴"（droplet）的伪影，原因是 AdaIN 破坏了特征图相对于输入的相对信息。StyleGAN2 用 **权重调制 + 去调制（weight demodulation）** 取代 AdaIN：

$$
w'_{ijk} = s_i \cdot w_{ijk},\quad s_i = \sqrt{\mathbb{E}_j(w_{ijk}^2) + \epsilon}
$$

$$
w''_{ijk} = w'_{ijk} / \sqrt{\sum_{i,k} w'_{ijk}^2 + \epsilon}
$$

并加入 **路径长度正则化（path length regularization）**，让潜码到图像的映射局部更光滑，便于潜空间编辑。StyleGAN2 在人脸、动物、动漫等数据集上达到 FID 历史新低。

### 3. StyleGAN3（2021）

StyleGAN3 进一步解决了"纹理黏附"问题：当图像在潜空间平滑插值时，细节纹理本应随之移动，但实际会出现纹理"黏"在同一像素位置的现象。StyleGAN3 引入 **傅里叶特征** 与**等变约束（equivariance）**，让生成器对平移和旋转具有显式的不变性。技术上用「全局采样 + 滤波上采样」替代了 StyleGAN2 的逐层 transposed conv，并在网络中显式消除 aliasing。

StyleGAN 系列至今仍是人脸与高保真图像生成的标杆，并衍生出大量图像编辑工具（StyleGAN-NADA、StyleCLIP、DragGAN 等）。

## 八、评估指标：IS 与 FID

### 1. Inception Score（IS, Salimans et al. 2016）

$$
\text{IS}(G) = \exp\!\left(\mathbb{E}_{x \sim p_g}\!\left[D_{\text{KL}}\!\left(p(y|x) \,\|\, p(y)\right)\right]\right)
$$

直觉：好的图像应当**被 Inception 高置信地识别为某一类**（$p(y|x)$ 尖锐），且**所有类都被均匀覆盖**（$p(y)$ 接近均匀）。IS 越高越好，但它**不比较真实分布**，对 mode collapse 不敏感——一组 100 张完全相同的猫图，IS 也能很高。

### 2. Fréchet Inception Distance（FID, Heusel et al. 2017）

把真实图片和生成图片分别用 Inception 网络提取特征（通常是 pool3 层 2048 维向量），然后假设两者都服从高斯分布，计算 Frechet 距离：

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

FID 同时考虑**质量**和**多样性**，是目前最常用的评估指标。需要注意的是，FID 对样本数和数据集敏感，论文里必须报告相同条件下的对比。

### 3. KID（Kernel Inception Distance）

FID 假设两个分布都是高斯，对小样本偏差较大。KID 用 MMD（Maximum Mean Discrepancy）替代，更适合**样本量较小**的场景。Bińkowski et al. (2018) 的实验表明，KID 的偏差比 FID 小一个数量级。

### 4. Precision-Recall 曲线（Sajjadi et al. 2018）

把"生成样本是否落在真实分布的 manifold 上"（precision）和"真实分布的 manifold 是否被生成样本覆盖"（recall）分别度量，避免 IS / FID 把质量与多样性耦合在一个数里。Image-to-Image GAN 中常见 $P \uparrow R \uparrow$ 两条曲线。

## 九、应用与 PyTorch 实现简化版 DCGAN 训练循环

GAN 的应用覆盖图像编辑（CycleGAN、StarGAN）、数据增广（医学图像）、超分辨率（SRGAN）、图像修复（inpainting）等众多领域。下面是一个最小化的 DCGAN 训练循环，演示对抗更新的核心逻辑：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 沿用上文 DCGANGenerator；判别器为对称的卷积网络
class DCGANDiscriminator(nn.Module):
    def __init__(self, nc: int = 3, ndf: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1)


def train_step(G, D, real, z, opt_G, opt_D, criterion, device):
    """一个标准 DCGAN 训练 step: 先更新 D，再更新 G。"""
    real = real.to(device)
    bs = real.size(0)

    # ---- (1) 更新判别器 ----
    D.zero_grad()
    label_real = torch.ones(bs, device=device)
    label_fake = torch.zeros(bs, device=device)
    out_real = D(real)
    loss_real = criterion(out_real, label_real)
    fake = G(z)
    out_fake = D(fake.detach())
    loss_fake = criterion(out_fake, label_fake)
    loss_D = loss_real + loss_fake
    loss_D.backward()
    opt_D.step()

    # ---- (2) 更新生成器 ----
    G.zero_grad()
    out_fake2 = D(fake)
    loss_G = criterion(out_fake2, label_real)  # 骗过判别器
    loss_G.backward()
    opt_G.step()

    return loss_D.item(), loss_G.item()


# 优化器配置：G 与 D 都用 Adam，lr=2e-4, betas=(0.5, 0.999)
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()
```

训练注意事项：

- **判别器与生成器更新次数**保持 1:1，不要让 D 过强。
- **图像归一化到 $[-1, 1]$**（与 Tanh 输出匹配）。
- **不均衡的数据增强**会让判别器走捷径。
- 监控 FID 而非 loss，**loss 数值本身不能反映生成质量**。

### 渐进式训练（ProGAN, Karras et al. 2017）

除了 DCGAN，另一项里程碑式的 GAN 架构改进是 ProGAN：从 4×4 开始训练，逐步加入更高分辨率的 block（8×8, 16×16, ..., 1024×1024）。这种"由粗到细"训练大幅稳定了高分辨率 GAN 训练，是 StyleGAN 的前置工作。

### BigGAN（Brock et al. 2018）

把 GAN 推到 ImageNet 规模的关键工作：

- 更大的 batch（2048）+ 更宽的通道
- **类别条件 BatchNorm**：把类别嵌入到 BN 的 affine 参数
- **截断技巧（truncation trick）**：采样时把 $z$ 截断到 $\mathcal{N}(0, \sigma^2 I)$，$\sigma$ 越小样本越"典型"
- 引入了 **Self-Attention** 模块（SAGAN）

BigGAN 在 ImageNet 128×128 上达到 FID ≈ 9，超过了同类 Diffusion 模型，证明了 GAN 仍然能在特定场景下与 Diffusion 抗衡。

### SAGAN（Self-Attention GAN, Zhang et al. 2019）

普通卷积只能捕获**局部**感受野，GAN 难以建模长程依赖（如对称结构、跨区域的纹理）。SAGAN 在 U-Net 风格的 backbone 中加入 **self-attention 层**：

$$
\text{Attn}(x) = \text{softmax}\!\left(\frac{f(x) g(x)^\top}{\sqrt{d_k}}\right) h(x)
$$

其中 $f, g, h$ 是 $1 \times 1$ 卷积得到的 Q、K、V。Self-attention 让远距离像素直接交互，再配合 **Spectral Normalization + Two-Timescale Update Rule（TTUR, 不同学习率）**，ImageNet 128×128 的 FID 从 18 降到 14 左右。

### ProGAN（Progressive Growing, Karras et al. 2017）

ProGAN 是 StyleGAN 的前置工作。训练从 4×4 开始，逐层加入更高分辨率的 block（8×8, 16×16, ..., 1024×1024）。关键技巧：

- 过渡阶段用一个 $\alpha$ 加权混合新旧分辨率输出，避免跳跃式崩坏
- 每一层新加入时使用 **fade-in** 卷积
- 用 WGAN-GP loss + pixel norm

渐进式训练大幅稳定了高分辨率 GAN 训练，是 StyleGAN 的前置工作，也为后来的 Diffusion 模型"由粗到细"采样器提供了灵感。

### StyleGAN 的图像编辑应用

StyleGAN 的 $\mathcal{W}$ 空间高度解耦，是 GAN 编辑研究的黄金平台：

- **StyleGAN-NADA（Gal et al. 2022）**：用 CLIP 引导，把预训练 StyleGAN 适配到新领域（"皮克斯风"、"油画风"）。
- **StyleCLIP**：在 $\mathcal{W}$ 空间沿 CLIP 文本方向平移，实现文本驱动的图像编辑。
- **DragGAN（Pan et al. 2023）**：用户拖动几个控制点，迭代地把图像变形到目标位置，效果直观。
- **PTI / HyperStyle**：把真实图像 inversion 到 $\mathcal{W}$ 空间，再编辑后再解码。

这些工作构成了 2020~2023 年图像编辑研究的主流，与 Diffusion 的 inpainting / InstructPix2Pix 形成两条互补路线。

### GAN 与 Diffusion 的工程取舍

一个常被忽视的事实是：GAN 与 Diffusion 的工程取舍并不像表面上那么绝对。

- **实时应用**（滤镜、视频通话背景替换）：GAN 仍然是首选，单步推理的延迟远低于 Diffusion 的 20+ 步。
- **可控图像编辑**（人脸属性调整、姿态变化）：StyleGAN + $\mathcal{W}$ 空间编辑比 Diffusion 的 inpainting 更稳定，prompt 不需要细调。
- **多模态生成**（文生图、图生图）：Diffusion 占绝对优势，CLIP 文本条件集成天然。
- **长尾类别与文本对齐**：Diffusion 借助大规模图文预训练，远胜 GAN。

因此在工业落地中**两种范式长期共存**：GAN 守住实时+可解释的"边缘"，Diffusion 占领高质量+多模态的"中心"。

## 十、GAN 在工业落地中的考量

工程上部署 GAN 时还要关注几点：

1. **潜空间搜索 vs 随机采样**：StyleGAN 系列通过在 $\mathcal{W}$ 中沿特定方向平移，能定向生成"微笑 / 男性化 / 年轻化"等属性，远比纯随机采样可控。
2. **Inversion**：把真实图片映射回潜码 $w^+$，通常用 encoder 或多次迭代优化（e.g., e4e, pSp）。Inversion 质量决定后续编辑能力。
3. **轻量化**：StyleGAN2 的 30M 参数 + 单卡 GPU 即可达到实时推理，比 Diffusion 模型便宜很多，特别适合手机端实时滤镜。
4. **可控生成 vs 多样性**：GAN 牺牲多样性换取"可控"。如果业务需要 N 种风格确定输出，GAN 通常比 Diffusion 更合适。

### 3D-aware GAN：EG3D（Chan et al. 2022）

GAN 还可以拓展到 3D-aware 生成：只训练 2D 图片，但通过 NeRF 风格的 3D 表示隐式建立相机视角一致性。EG3D（Efficient Geometry-aware 3D GAN）的核心组件：

- **Tri-plane representation**：用一个三平面特征张量代替 3D 体素
- **轻量 NeRF decoder**：从 tri-plane + 视角向量合成 RGB + density
- **dual discriminator**：分别监督 2D 图像质量与 3D 几何一致性

EG3D 在 FFHQ 上能生成多视角一致的 1024×1024 人脸，证明了 GAN 的能力边界远不止"2D 图像合成"。

## 小结

GAN 通过生成器与判别器的对抗博弈，为深度生成模型提供了一条全新的路线。从原始 GAN 的 JS 散度困境，到 DCGAN 的 CNN 化、再到 WGAN 的 Wasserstein 距离、StyleGAN 的分层风格控制，每一次演进都对应着对训练稳定性和可控性的工程回应。从 SAGAN 的 self-attention、BigGAN 的 ImageNet 规模化，到 ProGAN 的渐进式训练、EG3D 的 3D 感知生成，每一项技术都在解决"对抗训练不稳"这一根本难题。

然而，GAN 的对抗训练本质上不稳定的根本原因没有完全消失——判别器的容量、训练目标的局部最优、潜空间的纠缠问题，都使得"高质量 + 高多样性 + 可控性"难以兼得。这正是 Diffusion 模型在 2020 年后迅速崛起的契机：把生成问题从"对抗博弈"转化为"去噪回归"，绕开了 GAN 训练的所有坑。下一篇我们将进入扩散模型与 Stable Diffusion 的世界，看看"加噪—去噪"这一思路如何把图像生成的稳定性与多样性推向新高度。
