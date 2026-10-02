# NeRF 与 3D 表示：从体渲染到 3D Gaussian Splatting

三维重建与新视角合成（Novel View Synthesis, NVS）是计算机视觉最具想象力的方向之一：从一组稀疏的照片里"复原"一个可以自由漫游的 3D 场景。2020 年 Mildenhall 等人提出的 **NeRF**（Neural Radiance Fields，神经辐射场）用一个 MLP 把 3D 场景"装"进神经网络权重，开创了隐式 3D 表示的浪潮；2023 年 Kerbl 等人的 **3D Gaussian Splatting**（3DGS）则用显式的高斯椭球把训练和实时渲染推到新高度。本文从 3D 表示的横向对比出发，逐层推导 NeRF 的体渲染公式、MLP 结构与训练策略，再介绍 Instant-NGP、Mip-NeRF、NeuS 等改进路线，最后落到 3DGS 的革新与 PyTorch 实现。

## 一、3D 表示方式的横向对比

把"一个 3D 场景"装进计算，需要选一种**表示**（representation）。四类主流方案各有取舍：

| 表示方式 | 数据结构 | 典型应用 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| **点云**（Point Cloud） | $(N, 3)$ 或 $(N, 6)$ | LiDAR 感知、3D 打印 | 简单、与传感器直出对齐 | 无拓扑、稀疏处空洞 |
| **体素网格**（Voxel Grid） | 规则 3D 栅格 | 占据预测、医学影像 | 规整、可用 3D CNN | 分辨率受显存限制，$O(N^3)$ |
| **三角网格**（Mesh） | 顶点 + 三角形面 | 游戏、CAD、动画 | 紧凑、GPU 友好 | 拓扑固定、非水密难处理 |
| **隐式表示**（Implicit） | 函数 $f_\theta: \mathbb{R}^3 \to \text{属性}$ | NeRF、SDF 重建 | 连续、可微分、与分辨率解耦 | 采样密集、训练慢 |

NeRF 走的是最后一条路：用一个 MLP 表示连续的体密度场（density field）和颜色场（color field），从而把"3D 场景"变成一组权重。这条路的代价是**必须用体渲染去查询它**，带来了 NeRF 标志性的"沿光线积分"。

## 二、NeRF 的核心思想

Mildenhall et al. (2020, *Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV) 的核心假设非常简洁：

> 任意 3D 点的颜色和密度，**只取决于这个点的空间位置与观察方向**。

把这条假设写成一个 MLP：

$$
F_\theta: (\mathbf{x}, \mathbf{d}) \mapsto (\mathbf{c}, \sigma)
$$

其中：

- $\mathbf{x} = (x, y, z) \in \mathbb{R}^3$，3D 坐标；
- $\mathbf{d} \in \mathbb{S}^2$，单位视角方向（射线方向）；
- $\sigma \in \mathbb{R}_{\geq 0}$，体素密度（volume density），相当于该点"对光的不透明度"；
- $\mathbf{c} \in [0,1]^3$，RGB 颜色。

物理直觉：体密度 $\sigma$ 反映"这条光线在此处有多大概率被挡住"——可以把它想成沿光线走过的"小段吸光度"；颜色 $\mathbf{c}$ 则是"被挡住那一刻，被这个点反射出的光"。

## 三、体渲染：从 3D 场到 2D 像素

有了 $F_\theta$，NeRF 的关键问题就变成：**给定一条相机光线 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$，它在像素上的颜色是多少？** 这正是经典的体渲染方程（volume rendering equation）：

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt
$$

其中 **透射率**（transmittance）：

$$
T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\, ds\right)
$$

物理直觉：$T(t)$ 是"光线从起点走到 $t$ 还没被吸收"的概率；$\sigma\,dt$ 是"在 $t$ 附近这一小段被吸收的概率"；两者相乘，再乘上该处反射的颜色 $\mathbf{c}$，积分得到最终像素颜色。这套公式本质就是 Beer-Lambert 吸收定律的离散版本。

### 1. 离散化与分层采样

连续积分没法在 MLP 上直接求，所以 NeRF 把光线切成 $N$ 段，每段用一个点采样近似：

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i\,(1 - e^{-\sigma_i \delta_i})\,\mathbf{c}_i
$$

$$
T_i = \exp\!\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right),\quad \delta_i = t_{i+1} - t_i
$$

这就是经典的 **alpha compositing**（从后向前的透明度合成）。NeRF 进一步用**粗-细两阶段采样**（coarse-to-fine）：先用一个粗网络在 $N_c$ 个均匀采样点上预测密度，做 importance sampling 得到 $N_f$ 个"贴近表面"的细采样点，再把 $N_c + N_f$ 个点一起送入细网络计算颜色。这种做法把算力集中在贡献大的位置，是 NeRF 早期效果出色的关键 trick。

### 2. 离散化中的几个细节

把连续积分切成 $N$ 段时，有几个工程细节值得展开：

- **采样区间 $[t_n, t_f]$**：每条光线有自己的近平面和远平面，由相机参数 + 场景包围盒决定。超出区间的 $\sigma$ 一律按 0 处理，避免远处噪声干扰。
- **数值稳定**：`cumprod` 内部如果连续乘 0 会快速下溢，所以代码里加了一个 `1e-10` 的 epsilon。
- **背景建模**：NeRF 假设场景外是固定颜色（白或黑），但如果背景复杂（例如户外天空），可以在末端额外加一个"无穷远背景颜色"，或者改用透明背景 + 合成。
- **白点一致性**（white background）：训练时若背景是白色，渲染时也要把 `1 - T_\text{final}` 那一部分补成白色，避免暗角。

### 3. 简化版 PyTorch 体渲染

下面这段代码把上面的离散公式直接实现出来，可以独立运行做烟测：

```python
import torch


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Importance sampling：按 weights 的 CDF 在 [0,1] 均匀采样，再映射回 t。"""
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    u = torch.rand(*cdf.shape[:-1], n_samples, device=bins.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(min=0)
    above = inds.clamp(max=cdf.shape[-1] - 1)
    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    bins_below = torch.gather(bins, -1, below)
    bins_above = torch.gather(bins, -1, above)
    denom = (cdf_above - cdf_below).clamp(min=1e-5)
    t = (u - cdf_below) / denom
    return bins_below + t * (bins_above - bins_below)


def volume_render(sigma: torch.Tensor, rgb: torch.Tensor,
                  delta: torch.Tensor) -> torch.Tensor:
    """离散体渲染：sigma (N_rays, N_samples), rgb (N_rays, N_samples, 3), delta (N_rays, N_samples)。"""
    alpha = 1.0 - torch.exp(-sigma * delta)                  # 每段不透明度
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),
                                     1.0 - alpha[..., :-1] + 1e-10], dim=-1), dim=-1)
    weights = alpha * trans                                   # 每段对最终颜色的贡献
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)          # 加权求和
    return rgb_map, weights
```

`volume_render` 给出的 `weights` 就是该光线在每个采样点的不透明度分布——既是颜色合成权重，也是后面提取 mesh、做 depth 估计的中间量。把 `weights` 沿 $t$ 做加权平均，就能得到 **expected depth**：

$$
\hat{D}(\mathbf{r}) = \sum_i w_i \cdot t_i
$$

这把 NeRF 从"看"扩展到"测距"，是后续做 SLAM（如 NICE-SLAM、iMAP）的关键桥梁。

## 四、MLP 网络结构与位置编码

NeRF 的 MLP 很小（典型 8 层、256 维），但有两条关键设计。

### 1. 分支结构

密度 $\sigma$ **只取决于位置**，颜色 $\mathbf{c}$ **取决于位置 + 视角方向**。所以网络是：

$$
\mathbf{h} \to \sigma,\quad (\mathbf{h}, \mathbf{d}) \to \mathbf{c}
$$

也就是把方向在 MLP 中段接入，让前面的隐藏层只对几何敏感，对颜色解耦。这条设计在 NeRF 之后的几乎所有工作中都保留了下来。

### 2. 位置编码（Positional Encoding）

直接喂 $(x, y, z)$ 进 MLP 效果很差——MLP 倾向学低频函数。NeRF 用高频正弦编码把坐标"升维"：

$$
\gamma(p) = \left(\sin(2^0 \pi p),\ \cos(2^0 \pi p),\ \dots,\ \sin(2^{L-1} \pi p),\ \cos(2^{L-1} \pi p)\right)
$$

通常 $L = 10$ 用于位置，$L = 4$ 用于方向。这条 trick 把 MLP 强行拉到高频空间，是 NeRF 能"看清"细节的核心。Transformer 里的位置编码思路在精神上与之相似：**显式注入高维周期信号，让浅层网络也能拟合高频**。

下面是一个完整的微型 NeRF MLP：

```python
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        scales = 2.0 ** torch.arange(num_freqs) * torch.pi
        self.register_buffer("scales", scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D) -> (..., D * 2 * num_freqs)
        scaled = x[..., None, :] * self.scales[:, None]   # (..., F, D)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-2).flatten(-2)


class NeRF(nn.Module):
    def __init__(self, d_hidden=256, n_layers=8, L_xyz=10, L_dir=4):
        super().__init__()
        self.pe_xyz = PositionalEncoding(L_xyz)
        self.pe_dir = PositionalEncoding(L_dir)
        in_xyz = 3 * 2 * L_xyz
        in_dir = 3 * 2 * L_dir

        # 前半段：只用位置，逐步压缩到瓶颈
        self.geo_layers = nn.ModuleList(
            [nn.Linear(in_xyz if i == 0 else d_hidden, d_hidden) for i in range(n_layers)]
        )
        # 几何输出层：只决定 sigma
        self.sigma_head = nn.Linear(d_hidden, 1)
        # 颜色分支：把瓶颈 + 方向再走两层
        self.color_pre = nn.Linear(d_hidden, d_hidden)
        self.color_layers = nn.Sequential(
            nn.Linear(d_hidden + in_dir, d_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        h = self.pe_xyz(x)
        for layer in self.geo_layers:
            h = torch.relu(layer(h))
        sigma = torch.relu(self.sigma_head(h))[..., 0]            # (...,)
        feat = self.color_pre(h)
        d_enc = self.pe_dir(d)
        rgb = self.color_layers(torch.cat([feat, d_enc], dim=-1))  # (..., 3)
        return rgb, sigma
```

`sigma` 通过 `ReLU` 保证非负；颜色用 `Sigmoid` 收敛到 $[0, 1]$，与监督信号对齐。

## 五、训练流程与损失函数

NeRF 的训练极简：每个 batch 取若干条光线，每条光线上采样若干点，用上面的 `volume_render` 得到预测颜色，与 ground-truth 像素做像素级 L2 损失：

$$
\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{B}} \left\| \hat{C}(\mathbf{r}) - C_{\text{gt}}(\mathbf{r}) \right\|_2^2
$$

注意是**对像素颜色回归**，不是分类损失。这条损失在 NeRF 全程没有换过，但它能稳定工作，依赖两点：

1. **体渲染的可微分性**：MLP 的梯度可以直接穿过 $\sigma, \mathbf{c}$ 传到位置 $\mathbf{x}$，相当于隐式地在做"该处几何应该往哪挪"的监督。
2. **粗-细两阶段采样的协同训练**：粗网络与细网络共享同一损失函数，但粗网络只是"采样器"，它学到的密度分布本身就是有用的副产品。

训练数据通常是一组已知相机位姿的多视角照片（COLMAP 跑出来的 $\text{SfM}$ 即可），大约 100 张就能合成高画质新视角。

## 六、改进路线：从 Mip-NeRF 到 Instant-NGP

原始 NeRF 训练一个场景要 1–2 天、渲染一张图要几十秒，这催生了一系列改进：

### 1. Mip-NeRF（Barron et al. 2021）

NeRF 把每条光线当成"无穷细的针"，遇到像素与像素之间不同覆盖区域时会出现锯齿。Mip-NeRF 用**圆锥近似**（conical frustum）替代单条射线，并在**积分位置编码**（integrated positional encoding, IPE）上做解析积分，本质上把"位置编码"从点变成一个高斯分布：

$$
\gamma_{\text{IPE}}(\mathbf{x}) = \int \gamma(\mathbf{x} + \mathbf{y})\, \mathcal{N}(\mathbf{y}; 0, \sigma^2)\, d\mathbf{y}
$$

物理直觉：每个像素真正"看到"的是一个圆锥台，覆盖了一团高斯分布的点，所以应该用高斯加权的编码。效果是抗锯齿能力大幅提升，PSNR 提升 1–2 dB。

### 2. Instant-NGP（Mueller et al. 2022, NVIDIA）

原始 NeRF 的最大瓶颈是**位置编码在 MLP 内计算**——每次前向都要算几千次 $\sin/\cos$。Instant-NGP 的核心思想是：

> 用多分辨率**哈希表**（multi-resolution hash grid）替代 MLP 内的位置编码。

具体做法是把空间分成若干个不同分辨率的网格，每格维护一个可学习的特征向量，通过哈希查表 + 线性插值就能 $O(1)$ 得到位置特征，再喂给一个非常小的 MLP 输出 $\sigma, \mathbf{c}$。结果：**训练从 1 天降到 5 分钟**（百倍提速），渲染质量不输原始 NeRF。

直觉对比：原始 NeRF 是"用 MLP 学一个连续函数"，Instant-NGP 是"用一个稀疏哈希表 + 局部 MLP 学残差"——把"先验结构"显式塞进表示里。

### 3. NeuS / Neuralangelo（曲面重建）

原始 NeRF 给的是密度场，提取 mesh 需要 Marching Cubes + 等值面阈值，效果常带噪声。**NeuS**（Wang et al. 2021）改用 **SDF**（Signed Distance Function，有符号距离函数）做隐式表示，把不透明度与 SDF 通过一个 Sigmoid 映射联系起来，让表面自然出现在 SDF 的零等值面处。**Neuralangelo**（Li et al. 2023, NVIDIA）则把 Instant-NGP 的哈希编码与多分辨率 SDF 结合，能从视频中恢复高细节的 3D 表面。

这类工作的共同点是**把"场景的物理结构"显式建模**——密度场适合"看"，SDF 适合"摸 / 测"。

## 七、3D Gaussian Splatting：显式表示的复兴

2023 年 Kerbl 等人的 **3D Gaussian Splatting (3DGS)** 在 SIGGRAPH 拿下最佳论文。它的核心理念反转了 NeRF：

> 不再"用一个 MLP 表示 3D 场"，而是**用一组显式的 3D 高斯椭球**直接渲染。

每个高斯椭球有 7 个属性：位置 $\mathbf{x}$、协方差 $\Sigma$（由缩放 + 旋转参数化）、不透明度 $\alpha$、球谐系数（Spherical Harmonics, SH）表示视角相关的颜色。渲染时把高斯**投影**到图像平面（splatting），用类似 $\alpha$-blending 的方式合成像素：

$$
C = \sum_{i} \mathbf{c}_i \alpha_i' \prod_{j<i} (1 - \alpha_j')
$$

物理直觉：每个高斯是一团有方向的"彩色雾"，投影到屏幕上是一个椭圆；多个椭圆按深度顺序合成，就成了最终像素。

相比 NeRF，3DGS 的优势：

- **训练快**：几分钟到一个小时，单卡即可。
- **渲染快**：实时（30 FPS 以上），适合交互。
- **编辑友好**：显式几何，可以直接拖动、删除、添加高斯。
- **质量高**：在多个数据集上 PSNR / SSIM 接近甚至超过 NeRF。

代价是显存占用大、需要周期性 densify / prune 来控制高斯数量。它的出现把 NVS 从"实验室技术"推向"实时产品"。

## 八、评估指标与常见陷阱

评估 NeRF / 3DGS 的输出质量，通常用三类指标：

- **PSNR**（Peak Signal-to-Noise Ratio）：直接衡量 RGB 重建误差，越高越好。NeRF 论文一般在 25–32 dB 之间，3DGS 多在 27–35 dB。
- **SSIM**（Structural Similarity）：在结构 / 亮度 / 对比度三个维度上衡量相似度，比 PSNR 更贴近人眼。
- **LPIPS**（Learned Perceptual Image Patch Similarity）：用预训练 CNN 抽特征再算距离，捕捉高层语义相似度。

公式上：

$$
\text{PSNR} = 10 \log_{10} \frac{1}{\text{MSE}}, \quad \text{MSE} = \frac{1}{N}\sum_{i}(I_i - \hat{I}_i)^2
$$

$$
\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

$$
\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l}\sum_{h,w}\| w_l \odot (\phi_l(x)_{h,w} - \phi_l(y)_{h,w}) \|_2^2
$$

实际训练中几个常见坑：

1. **训练集 / 测试集划分**：必须按视角划分（同视角只取一张），不能按像素划分，否则会严重高估 PSNR。
2. **相机位姿噪声**：NeRF 对 SfM 出的相机位姿非常敏感，COLMAP 失败的场景基本无法训练。
3. **背景建模**：户外场景若不显式处理背景，无限远处会"吞掉"密度，导致云雾状伪影。
4. **图像分辨率**：过高分辨率会让 MLP 拟合成本暴涨，通常先下采样到 1–2 MP 再训练。
5. **场景复杂度**：高度反光（玻璃、镜面）、动态物体（行人、树叶飘动）是 NeRF 的天敌，需要专门改进。

## 九、典型应用场景

把 NeRF / 3DGS 落地的几个代表性方向：

1. **新视角合成**：从一组照片生成可漫游的 3D 场景，应用于 VR 看房、文物保护、电商 3D 展示。
2. **虚拟数字人**：GaussianHead、GaussianAvatar 等工作用 3DGS 表示人脸，可驱动、可重光照。
3. **自动驾驶仿真**：用 NeRF / 3DGS 重建真实道路，做闭环仿真（CARLA、Waymax 等已开始接入）。
4. **机器人抓取**：重建工件的精细几何，用 SDF / NeRF 给出可微的"接触场"，辅助规划抓取位姿。
5. **SLAM 加速**：iMAP、NICE-SLAM 等把 NeRF 作为稠密建图模块，与视觉里程计融合做实时定位与建图。

## 十、PyTorch 实战：3D 高斯溅射的最小核心

3DGS 完整实现上千行，但最核心的"高斯投影 + 排序 + 合成"可以浓缩成下面这个简版：

```python
import math
import torch


def project_gaussians(mean3d, cov3d, view_matrix, fx, fy, cx, cy, img_w, img_h):
    """把 3D 高斯 (mean3d: (N,3), cov3d: (N,3,3)) 投影到图像平面，
    返回 2D 均值 (N,2)、2D 协方差 (N,2,2) 和深度 (N,)。"""
    # 世界 -> 相机
    mean_cam = (view_matrix[:, :3] @ mean3d.T).T                 # (N,3)
    mean_cam = mean_cam[:, :3]
    # Jacobian of perspective projection
    x, y, z = mean_cam[:, 0], mean_cam[:, 1], mean_cam[:, 2]
    z2 = z.clamp(min=1e-6) ** 2
    J = torch.zeros(mean_cam.size(0), 2, 3, device=mean_cam.device, dtype=mean_cam.dtype)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / z2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / z2
    R = view_matrix[:, :3, :3]                                    # (3,3)
    cov_cam = R @ cov3d @ R.transpose(-1, -2)                     # (N,3,3)
    cov2d = J @ cov_cam @ J.transpose(-1, -2)                     # (N,2,2)
    # 屏幕坐标
    mean2d = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)
    depth = z
    return mean2d, cov2d + torch.eye(2, device=cov2d.device) * 0.3, depth


def render_gaussians(mean2d, cov2d, color, alpha, depth, img_w, img_h):
    """简化版 alpha-blending：按深度从远到近合成。"""
    N = mean2d.size(0)
    # 视锥裁剪
    in_screen = (mean2d[:, 0] >= 0) & (mean2d[:, 0] < img_w) & \
                (mean2d[:, 1] >= 0) & (mean2d[:, 1] < img_h) & (depth > 0)
    mean2d, cov2d, color, alpha, depth = (
        mean2d[in_screen], cov2d[in_screen], color[in_screen],
        alpha[in_screen], depth[in_screen]
    )
    order = torch.argsort(depth, descending=True)                 # 远 -> 近
    mean2d, cov2d, color, alpha = mean2d[order], cov2d[order], color[order], alpha[order]

    canvas = torch.zeros(img_h, img_w, 3, device=mean2d.device, dtype=mean2d.dtype)
    trans = torch.ones(img_h, img_w, 1, device=mean2d.device)
    for i in range(mean2d.size(0)):
        u, v = int(mean2d[i, 0].item()), int(mean2d[i, 1].item())
        if 0 <= u < img_w and 0 <= v < img_h:
            a = alpha[i].clamp(0, 1)
            canvas[v, u] += trans[v, u] * a * color[i]
            trans[v, u] *= (1.0 - a)
    return canvas
```

这段代码省去了原版 3DGS 里的 tile-based 光栅化、SH 颜色评估与自适应密度控制，但保留了**核心思想**：3D 高斯 → 2D 椭圆 → 按深度排序的 $\alpha$-blending。把它接上 COLMAP 出来的 SfM 点云作为初始位置，再加一个 L1 + SSIM 的渲染损失做端到端优化，就是一个最小可跑的 3DGS pipeline。

## 十一、动态场景：从 3D 到 4D

把时间维度加进来，就是 **4D NeRF**（或 Dynamic NeRF）。最直接的思路是把时间 $t$ 也作为输入：

$$
F_\theta: (\mathbf{x}, \mathbf{d}, t) \mapsto (\mathbf{c}, \sigma)
$$

代表工作：

- **D-NeRF**（Pumarola et al. 2021）：把场景拆成"模板 NeRF + 位移场"，让模板随时间变形。
- **Nerfies / HyperNeRF**（Park et al. 2021）：引入 per-frame 的潜码（latent code）建模非刚性形变。
- **HexPlane**（Cao & Johnson 2023）：用六个 2D 平面特征（$XY, XZ, YZ, Xt, Yt, Zt$）做 4D 张量分解，效率极高。
- **4D Gaussian Splatting**（Wu et al. 2024）：把 3DGS 的高斯加一个时间维度，专门做动态场景重建。

物理直觉：动态 NeRF 之所以难，是因为同一空间位置在不同时间的密度和颜色都要变。如果不显式建模运动，先验不够，MLP 容易"糊"掉。引入变形场或显式时间特征，相当于在 4D 空间里给了模型一个"骨架"。

实际工程上，4D 重建面临三重挑战：

1. **数据量大**：从短视频到多相机阵列，输入数据比静态 NeRF 多一个量级。
2. **运动模糊**：长曝光或快速运动会模糊监督信号，需要在损失里加时序正则。
3. **存储开销**：4D 哈希表或 4D 高斯的内存占用随时间线性增长，常常需要分块、剪枝或低秩分解。

HexPlane 之所以能用"6 个平面"装下 4D 场景，本质是借助了 4D 张量的低秩先验——真实运动往往只在少数几个轴上显著，把这种结构塞进模型能极大压缩参数。

## 十二、选型建议

工程上选 NeRF 还是 3DGS，可以按下面几条粗略判断：

- **场景静态、追求画质**：NeRF / Instant-NGP / Mip-NeRF。
- **场景静态、追求实时**：3DGS。
- **需要可编辑的显式几何**：3DGS 或 Voxel Hashing。
- **需要精确表面（CAD、打印）**：NeuS / Neuralangelo。
- **需要动态场景**：HexPlane / 4D Gaussian Splatting。
- **训练数据极少（< 20 张）**：RegNeRF / SparseNeRF 等带正则项的 NeRF。

不论选哪条路线，**相机位姿准确**是所有 NVS 方法的"硬地基"——COLMAP 跑不出来的场景，几乎没有 NVS 方法能直接救回来。

## 小结

3D 场景的表示选择决定了整套算法的形状：点云/体素/网格是"传统几何"，隐式 MLP（NeRF）是"把场景学成函数"，显式高斯（3DGS）是"用一堆彩色云团近似"。NeRF 用 **体渲染方程 + MLP + 像素 L2** 三件套，把 NVS 推到了第一个工程里程碑；Instant-NGP 用哈希编码、Mip-NeRF 用圆锥近似解决了 NeRF 的训练慢与抗锯齿问题；NeuS 类工作把隐式表示从"看"延伸到"摸"；3DGS 则用显式表示 + 实时渲染把整条赛道重新洗牌。下一步无论是与神经场的混合建模（4D 动态场）、还是与生成式先验结合（文本驱动 3D 生成），都建立在"如何表示 3D"这个根本选择之上。下一篇将沿着 3D 表示进一步延伸到**点云网络与多模态 3D 预训练**——看 PointNet、PCT、OpenShape 是如何把 3D 与语言对齐成"可被理解的场景"的。
