# 扩散模型与 Stable Diffusion：从 DDPM 到 SD3

扩散模型（Diffusion Model）在 2020 年由 Ho et al. 提出后，短短几年内迅速取代 GAN 成为高质量图像合成的事实标准。Stable Diffusion、SDXL、DALL·E 3、Midjourney 等明星产品的核心技术，都建立在一个优雅的概率框架之上：把数据分布通过逐步加噪变成各向同性的高斯，再训练一个神经网络反向去噪。本文从 DDPM（Denoising Diffusion Probabilistic Models）的数学推导出发，依次展开改进采样器、Classifier-Free Guidance、Latent Diffusion（Stable Diffusion）、再到 SDXL / SD3 / ControlNet 的现代变体，最后用 PyTorch 实现一个最简 DDPM 训练 step。

## 一、扩散模型的核心思想：前向加噪 + 逆向去噪

扩散模型借鉴了非平衡热力学的思想。一个直观类比是：往一杯清水中滴入一滴墨水，墨水分子会从「聚拢」（$x_0$）经随机扩散变得「均匀分散」（$x_T$）；反过来，如果我们能学会这个"逆过程"，就能从均匀的墨水分子重新聚拢出任意一滴原始墨水。

**前向过程（forward / diffusion process）**：对真实图像 $x_0$ 在 $T$ 步内逐步加入高斯噪声，最终得到接近 $\mathcal{N}(0, I)$ 的 $x_T$。每一步的转移分布为：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{\alpha_t}\, x_{t-1},\, (1 - \alpha_t) I\right)
$$

其中 $\alpha_t \in (0, 1)$ 是每一步的"保留比例"。通常设 $\beta_t$ 为方差 schedule（linear 或 cosine），$\alpha_t = 1 - \beta_t$。

**逆向过程（reverse process）**：学习一个参数化的高斯分布 $p_\theta(x_{t-1} \mid x_t)$ 来逐步去噪：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\right)
$$

关键的工程化选择是：**不去预测 $x_{t-1}$ 或 $\mu_\theta$，而是预测加进去的噪声 $\epsilon_\theta(x_t, t)$**。这就是 DDPM 的精髓。

### 1. 与 VAE 的关系

DDPM 的形式化推导与 VAE 高度相似：正向过程 $q$ 是固定的后验分布，逆向过程 $p_\theta$ 是参数化的变分后验。我们同样可以写出 ELBO：

$$
\log p_\theta(x_0) \geq \mathbb{E}_q\!\left[\log p_\theta(x_0 \mid x_1)\right] - \sum_{t=2}^{T} \mathbb{E}_q\!\left[D_{\text{KL}}\!\left(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\right)\right]
$$

扩散模型的独特之处在于：

- $T$ 通常取 1000 以上，远大于 VAE 中的 1 个隐层；
- 每一步的"潜变量" $x_t$ 与数据维度相同，是逐步加噪的版本；
- 通过把噪声预测重参数化，得到了极其简洁的训练目标。

### 2. 三种等价的目标参数化

DDPM 原始论文里讨论了三种参数化方式：**预测 $\epsilon$**、**预测 $x_0$**、**预测 $\mu_\theta$**。它们在数学上等价（可以通过公式互相换算），但在工程上差异巨大：

- 预测 $\epsilon$：数值范围始终是 $\mathcal{N}(0, I)$，训练最稳定，DDPM 原文采用。
- 预测 $x_0$：尺度随 $t$ 变化大，但在 $t \to 0$ 时预测更直接，常用于 DDIM。
- 预测 $\mu_\theta$：与 ELBO 直接对应，但与 $\epsilon$ 仅相差缩放，无显著优势。

$$
\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

## 二、DDPM 的数学推导

### 1. 前向过程的"重参数化"公式

利用重参数化技巧，可以直接从 $x_0$ 跳到任意 $x_t$，无需逐步迭代。令 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，有：

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t) I\right)
$$

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

物理直觉：$\sqrt{\bar{\alpha}_t}$ 随 $t$ 增大而衰减（信号被稀释），$\sqrt{1 - \bar{\alpha}_t}$ 随 $t$ 增大而增长（噪声占主导）。当 $T$ 足够大时，$x_T \approx \mathcal{N}(0, I)$。

### 2. 简化的训练损失

如果用 $\mu_\theta$ 直接参数化，训练目标是负 ELBO：

$$
L_{\text{ELBO}} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\frac{1}{2 \| \Sigma_\theta \|^2_2}\, \| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \|^2\right] + C
$$

其中 $\tilde{\mu}_t$ 是真实后验均值：

$$
\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon\right)
$$

Ho et al. 做了一个天才的简化：**固定 $\Sigma_\theta = \sigma_t^2 I$（与 $t$ 相关的常数），改让网络预测噪声 $\epsilon$**。此时 $\mu_\theta$ 与 $\epsilon_\theta$ 存在解析对应关系：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\right)
$$

损失函数简化为：

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\right]
$$

直觉：给定加噪后的图像 $x_t$ 和时间步 $t$，网络只要预测出"被加进去的噪声 $\epsilon$"就够了。优化目标就是一个普通的 MSE，对工程实现非常友好。

### 3. 训练算法

```
# 训练 DDPM（伪代码）
repeat:
    x_0 ~ p_data                              # 采样真实图像
    t ~ Uniform(1, ..., T)                    # 随机采样时间步
    ε ~ N(0, I)                               # 采样噪声
    x_t = sqrt(α̅_t) x_0 + sqrt(1 - α̅_t) ε  # 直接跳到 x_t
    take gradient step on:
        ∇_θ || ε - ε_θ(x_t, t) ||^2
```

实现上几个关键技巧：

- **时间步 $t$ 均匀采样**比按 schedule 加权采样效果更好。
- **损失可以加 SNR 权重**：$\text{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$，最小化 $\|\epsilon - \epsilon_\theta\|^2 / \text{SNR}(t)$ 可改善样本质量（"min-SNR weighting"）。
- **EMA**：对网络参数做指数滑动平均（$\theta_{\text{ema}} \leftarrow 0.9999 \theta_{\text{ema}} + 0.0001 \theta$），采样时用 EMA 权重，质量显著提升。

### 4. 噪声 schedule 的选择

DDPM 原文用**线性 schedule** $\beta_t: 10^{-4} \to 0.02$。实践中常用改进版：

- **Cosine schedule**（Nichol & Dhariwal 2021）：在低分辨率时变化更平滑，避免最后几步"突然丢失信息"。
- **Scaled linear**：$\beta_t$ 在两端变化更温和，适合大 $T$（如 4000）。
- **Learned schedule**：把 $\beta_t$ 也变成可学习参数（少见）。

## 三、采样算法：从噪声到图像

训练完成后，采样即"逆向迭代"：

```
# 采样（DDPM）
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    z ~ N(0, I)  if t > 1 else z = 0
    x_{t-1} = (1/√α_t)(x_t - (1 - α_t)/√(1 - α̅_t) · ε_θ(x_t, t)) + σ_t z
return x_0
```

DDPM 原始采样需要 $T = 1000$ 步，速度非常慢。优化方向有三：**(a)** 让网络预测 $x_0$（即去噪结果），再用 $\hat{x}_0$ 重新加噪；**(b)** 用更大的步长（strided sampling）；**(c)** 完全跳出马尔可夫假设（DDIM）。

### 从 score 的视角看扩散

DDPM 与 **Score-based Generative Model**（Song et al. 2021）有深层联系。噪声预测与 score 函数成正比：

$$
\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t}\, \nabla_{x_t} \log p_t(x_t)
$$

也就是说，**去噪网络学到的本质是"加噪后分布的 score 函数"**。这个视角下，采样可以用**随机微分方程（SDE）或 ODE 求解器**统一处理——这就是 DDIM 与 DPMSolver 的理论基础。

## 四、改进的采样器：DDIM 与 DPMSolver

### 1. DDIM（Denoising Diffusion Implicit Models, Song et al. 2020）

DDIM 证明：**不需要遵循严格的马尔可夫逆向过程**，只要每一步从同一个预测 $\epsilon_\theta$ 反推 $\hat{x}_0$，就能写出确定性采样公式：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\, \epsilon_\theta(x_t, t)
$$

$$
\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

DDIM 的好处是：(1) **采样步数可从 1000 降到 20~50**；(2) 同样的 $\epsilon_\theta$ 在确定性采样下保证**语义一致的潜空间插值**（$\epsilon_t$ 固定 → 路径唯一）；(3) 与 classifier-free guidance 配合效果极佳。

### 2. DPMSolver（Lu et al. 2022）

把 ODE 视角的扩散采样器（如 Heun、multistep）系统化，在 10~20 步内就能达到与 DDPM 1000 步相当的图像质量。`diffusers` 库默认的 `DPMSolverMultistepScheduler` 即为此类实现。DPMSolver 的核心是利用 ODE 解的高阶泰勒展开，把多步信息融合：

$$
x_{t_{i-1}} = \frac{\sigma_{t_{i-1}}}{\sigma_{t_i}} x_{t_i} + (\exp(h_i) - 1) \cdot \text{model}(x_{t_i}, t_i)
$$

其中 $h_i = \log(\sigma_{t_i} / \sigma_{t_{i-1}})$ 是对数信噪比变化。

### 3. Consistency Models（Song et al. 2023）

更进一步，直接学习「任意 $x_t$ 一步映射到 $x_0$」的映射，把采样压缩到 **1~4 步**，但对训练稳定性要求较高。Consistency 模型的关键约束是**自洽性**：对于任意在同一 ODE 轨迹上的 $(x_t, x_{t'})$，网络输出相同。这让单步采样成为可能。

### 4. 各采样器对比

| 采样器 | 步数 | 质量 | 速度 | 多样性 | 适用 |
| --- | --- | --- | --- | --- | --- |
| DDPM | 1000 | 高 | 慢 | 高 | 学术 |
| DDIM | 20~50 | 高 | 快 | 中 | 实时应用 |
| DPMSolver | 10~20 | 高 | 更快 | 中 | Stable Diffusion 默认 |
| Consistency | 1~4 | 中 | 极快 | 中 | 移动端 |

## 五、Classifier-Free Guidance：无分类器条件生成

传统做法是训练一个**额外**的分类器 $p(y \mid x_t)$，然后用梯度 $\nabla_{x_t} \log p(y \mid x_t)$ 引导采样走向目标类别。Ho & Salimans (2022) 提出更简洁的方案：**训练一个既能条件又能无条件生成的统一模型**。

具体做法：训练时以概率 $p_{\text{uncond}}$（如 10%）把条件标签 $c$ 替换为「空」（null embedding）。推理时同时跑两个前向：

- 无条件预测：$\epsilon_\theta(x_t, t, c = \emptyset)$
- 有条件预测：$\epsilon_\theta(x_t, t, c)$

最终采样时按权重混合：

$$
\hat{\epsilon}_\theta = (1 + w)\, \epsilon_\theta(x_t, t, c) - w\, \epsilon_\theta(x_t, t, \emptyset)
$$

$w$ 称为 **guidance scale**，$w > 1$ 时图像更符合提示词但多样性下降；$w < 1$ 时图像更多样但可能偏离意图。这是 Stable Diffusion 默认的引导方式（典型 $w = 7.5$）。

### 为什么 CFG 有效？

直觉上，CFG 把"无条件生成方向"减去，迫使模型沿着"条件更鲜明的方向"前进。从梯度角度看：

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t))
$$

第一项是"无条件预测"，后面是"条件方向"。$w$ 越大，相当于在条件方向上加力。代价是**过引导**会让图像过饱和、细节崩坏；欠引导则 prompt 跟随性差。

## 六、Latent Diffusion / Stable Diffusion

直接在像素空间训练扩散模型计算量极大。Rombach et al. (2022) 提出 **Latent Diffusion Model (LDM)**——也叫 **Stable Diffusion**——把扩散过程搬到 VAE 的潜空间（latent space）中：

### 1. 为什么在 Latent Space？

- 像素空间中相邻像素高度冗余，扩散模型在每个噪声尺度都重复处理这种冗余。
- VAE 把 $3 \times 256 \times 256$ 图像压缩到 $4 \times 32 \times 32$（共 48 倍压缩），**关键语义被保留但空间维度大幅降低**。
- 在更小的潜空间上跑 U-Net，**训练成本和显存都成倍下降**。原本需要 8 张 A100 训 30 天的像素扩散模型，Stable Diffusion 1.4 只需单卡 A100 训几天。

### 2. 三大组件

Stable Diffusion 包含三个独立的预训练模型：

**(a) VAE（变分自编码器）**
- 编码器：$x \in \mathbb{R}^{3 \times 256 \times 256} \mapsto z \in \mathbb{R}^{4 \times 32 \times 32}$
- 解码器：$z \mapsto \hat{x}$，把潜码还原成图像
- VAE 决定最终图像的细节质量与色彩还原度
- SD 1.x 用 KL-regularized VAE；SDXL 改用 8 倍下采样的 VAE，进一步压缩

**(b) CLIP Text Encoder**
- 把 prompt（"a corgi wearing a red hat, studio lighting"）编码为 $77 \times 768$ 维的 token embedding 序列
- CLIP 的关键性质是**文本—图像对齐空间**，让 U-Net 能"读懂"文字
- Stable Diffusion 2.x 之后也开始用更大的 OpenCLIP-H/14
- 77 tokens 的限制来自 CLIP 上下文长度，对长 prompt 通常需要截断或拆分

**(c) U-Net 去噪网络**
- 时间步 $t$ 通过 sinusoidal 位置编码注入
- **Cross-attention** 把文本 embedding 作为 K、V，图像特征作为 Q，实现文本条件控制：

$$
\text{CrossAttn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

- 输出仍是对潜码 $z_t$ 加噪的预测 $\epsilon_\theta$
- SD 1.x U-Net 约 860M 参数，深度残差 + spatial transformer block

### 3. 文本到图像的完整流程

```
prompt ──→ CLIP ──→ text_emb (77, 768)
                          │
z_T ~ N(0, I) ──→ U-Net(ε_θ(z_t, t, text_emb)) ──→ ε_pred
                          │
            x_{t-1} = scheduler.step(ε_pred, t, x_t)
                          │
                  （循环 T 步后得到 z_0）
                          │
                       VAE.decode(z_0) ──→ image
```

### 4. 推理时的实用参数

- **CFG scale** $w$：典型 7~12，越大越"听话"但过饱和。
- **Sampler**：DPMSolver / Euler / DPM++ 等。
- **Steps**：20~30 步足够，质量几乎饱和。
- **Negative prompt**：用 $\epsilon_\theta(x_t, t, \emptyset)$ 替换为"对负向词的预测"，相当于引导远离负向特征。
- **Seed**：固定种子可复现结果，方便调试。

## 七、现代变体：SDXL、SD3、ControlNet

### 1. SDXL（Stable Diffusion XL, Podell et al. 2023）

- **双文本编码器**：CLIP-ViT/L + OpenCLIP-ViT/bigG（拼接使用）
- **双 U-Net 结构**：base model + refiner，先用 base 出粗图，再用 refiner 精修
- **更大的 VAE** 与更高的 latent 分辨率
- **多 aspect ratio 训练**：原生支持 1024×1024、1152×896 等多比例
- **微条件（micro-conditioning）**：把原始图像尺寸、裁剪坐标也作为额外输入，让模型对分辨率和构图更鲁棒

### 2. SD3 / DiT（Diffusion Transformer, Peebles & Xie 2023）

SD3 把 U-Net 替换为 **DiT（Diffusion Transformer）**：把潜码切成 patch token，用 Transformer 主干 + adaLN-Zero 做条件注入。优势：

- **可扩展性强**：随数据/算力 scaling 表现单调提升。论文实验显示 DiT-XL 比 DiT-S 显著更好，且仍没看到饱和迹象。
- **MMDiT（Multi-Modal DiT）**：文本与图像 token 在 SD3 中通过**双向 self-attention** 融合，而不再是 cross-attention。每个 block 都同时处理文本 token 与图像 token，让文本—图像的对齐更自然。
- 训练与推理都更高效，质量大幅超越 SDXL

### 3. ControlNet（Zhang et al. 2023）

ControlNet 把额外的**结构条件**（Canny 边缘、深度图、姿态关键点、分割图等）注入到 U-Net 的每个 block：

- 复制 U-Net 的权重 $W$ 为可训练副本 $W_c$
- 通过**零卷积（zero convolution）** $z(\cdot; \theta_z)$ 注入：$\text{out} = W(x) + z(W_c(x); \theta_z)$
- 训练时 $\theta_z$ 从零初始化，保证初始输出与原 U-Net 一致

物理直觉：ControlNet 像给 U-Net 加了一个"条件通道"，让用户精确控制生成图像的几何结构，是 AIGC 落地不可或缺的一环。ControlNet 已被广泛用于：

- **Canny / Depth / Normal**：边缘、深度、法线图控制结构
- **OpenPose**：姿态关键点控制人物动作
- **Segmentation**：语义分割图控制物体布局
- **IP-Adapter**：图像 prompt 风格迁移

### 4. T2I-Adapter 与 IP-Adapter

- **T2I-Adapter**：比 ControlNet 更轻量的条件注入方法，把条件图降采样后与 U-Net 的中间特征相加。
- **IP-Adapter**：把"图像 prompt"作为条件，让模型参考给定图的风格/人物/构图。是 AIGC 角色一致性的关键技术之一。

### 5. LoRA 微调扩散模型

Stable Diffusion 训练成本高，社区普遍用 **LoRA（Low-Rank Adaptation）** 做轻量微调：冻结原权重 $W$，引入低秩增量 $W + \Delta W$，$\Delta W = A B$，$A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。训练 50 张图、几小时即可学会特定风格/人物。`diffusers` 库的 `LoraLoaderMixin` 直接支持。

## 八、PyTorch 实现：简化版 DDPM 训练 step

下面是一个最小化的 DDPM 训练 step（约 30 行），展示核心数学流程：

```python
import math
import torch
import torch.nn as nn


class DDPM(nn.Module):
    """简化版 DDPM: 给定 U-Net ε_θ 与 β schedule, 完成训练 step。"""

    def __init__(self, eps_model: nn.Module, T: int = 1000):
        super().__init__()
        self.eps_model = eps_model
        # 线性 β schedule: β_1=1e-4, β_T=2e-2
        beta = torch.linspace(1e-4, 2e-2, T)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)  # ᾱ_t

        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt())
        self.T = T

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """前向加噪: 直接跳到 x_t。"""
        sa = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        so = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sa * x0 + so * noise

    def training_step(self, x0: torch.Tensor) -> torch.Tensor:
        """单个训练 step: 随机 t, 随机 ε, 预测噪声并计算 MSE。"""
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.eps_model(x_t, t)          # U-Net 预测噪声
        loss = torch.nn.functional.mse_loss(eps_pred, noise)
        return loss


# 一个最小化的 U-Net ε_θ (示意，真实模型通常 ~860M 参数)
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, base),
        )
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1),
        )
        self.up = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        h = self.down(x)
        h = h + self.time_mlp(_sinusoidal(t, h.size(1))).unsqueeze(-1).unsqueeze(-1)
        h = self.mid(h)
        return self.up(h)


def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

实现要点：

- **重参数化 `q_sample`**：直接由 $x_0$ 和 $\epsilon$ 跳到 $x_t$，避免循环 $T$ 步。
- **时间步编码**：用 sinusoidal 编码 $t$ 后注入 U-Net，让同一套网络处理不同噪声尺度。
- **MSE 损失在 $\epsilon$ 空间**：简化训练目标，无需手动推导 $\mu_\theta$ 与 $\Sigma_\theta$。
- 真实的 Stable Diffusion U-Net 在每个 block 都有 **cross-attention** 用于接收文本 embedding，本简化版本省略。

### 进阶训练技巧

- **Mixed precision（AMP）**：fp16/bf16 训练，速度 +50%，显存减半。
- **Gradient checkpointing**：U-Net 深层残差 block 太多，启用 checkpoint 可省下大量显存。
- **EMA（Exponential Moving Average）**：维护 $\theta_{\text{ema}}$，推理时使用。
- **Min-SNR weighting**：给不同 $t$ 的损失加权（小 $t$ 权重小，大 $t$ 权重大），收敛更快、样本质量更好。

## 小结

扩散模型通过"前向加噪—逆向去噪"的对称框架，把复杂的图像分布学习问题转化为"预测每一步的噪声"这一简单的回归任务。DDPM 用 $L_{\text{simple}} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$ 让训练稳定，DDIM/DPMSolver 把采样从 1000 步压缩到 20 步，Latent Diffusion 通过 VAE 把这一过程搬到潜空间，Classifier-Free Guidance 让文本控制简洁优雅。从 SDXL 的双 U-Net，到 SD3 的 DiT 主干，再到 ControlNet 的结构条件注入，扩散模型的工程化已经形成相对完整的生态。理解这一整套链路，是继续深入 AIGC 工程（如 LoRA 微调、AnimateDiff 视频生成）的基础。
