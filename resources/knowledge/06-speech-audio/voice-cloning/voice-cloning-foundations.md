# 声音克隆基础：说话人嵌入与少样本学习

声音克隆（Voice Cloning）的目标是给定一段目标说话人的参考语音，让 TTS 系统用"这个人"的声音读出任意文本。它与一般 TTS 的核心区别是**说话人身份（speaker identity）的可控性**。从工程上看，声音克隆可分为三种范式：**零样本（zero-shot）**——仅给一段几秒到几十秒的参考音频，不做任何训练；**少样本（few-shot）**——给几条到几十条同说话人样本做轻量适配；**微调（fine-tuning）**——用目标说话人较长时间（30 分钟到几小时）语料对预训练模型做全量或部分微调。三者对应不同的数据门槛与质量上限，而贯穿其中的关键技术就是**说话人嵌入（speaker embedding）**——一段与文本内容无关、固定长度的向量，用以表征"说话人是谁"。

## 一、声音克隆的三种工程范式

设目标说话人为 $s$，参考音频为 $X_{\text{ref}}$，待合成文本为 $Y$，输出语音为 $\hat{X}$。三种范式可形式化为：

$$
\hat{X} = f_\theta(Y, X_{\text{ref}}, s)
$$

其中 $s$ 的来源决定了范式类别：

| 范式 | 所需 $s$ | 数据需求 | 推理延迟 | 典型 MOS |
|------|------------|----------|----------|----------|
| 零样本 | 实时编码 $X_{\text{ref}}$ | 仅推理时几秒参考音频 | 低 | 3.8-4.2 |
| 少样本 | 5-30 条样本原型 | 几十秒-几分钟 | 低 | 4.0-4.4 |
| 微调 | 模型权重本身被适配 | 30 分钟-数小时 | 训练后低 | 4.4-4.7 |

零样本路径的关键在于把 $f_\theta$ 中的"说话人编码器"训练成**与文本无关、与说话人强相关**的度量空间。少样本则借原型网络（Prototypical Networks）的思路，从多条参考样本聚合一个更鲁棒的嵌入。微调路径则把嵌入"吸收"到声学模型权重中，质量最高但代价最大。

## 二、说话人嵌入：d-vector、x-vector、ECAPA-TDNN

### 2.1 d-vector 与 GE2E 损失（Google, 2017）

最早的说话人嵌入称为 d-vector（Variani et al., 2014），Google 的端到端版本则把 LSTM 的最后帧向量作为说话人表征，并提出 **GE2E 损失（Generalized End-to-End loss）**。给定一个 batch 中 $N$ 个说话人、每人 $M$ 句话，GE2E 让**同说话人句子的嵌入靠近、不同说话人句子的嵌入远离**：

$$
\mathcal{L}_{\text{GE2E}} = \sum_{i,j} \sigma\left( \left\| \mathbf{e}_{ij} - \mathbf{c}_{i^{-j}} \right\|^2 - \left\| \mathbf{e}_{ij} - \mathbf{c}_j \right\|^2 + \alpha \right)
$$

其中 $\mathbf{e}_{ij}$ 是说话人 $i$ 第 $j$ 句的嵌入，$\mathbf{c}_j$ 是说话人 $j$ 所有句嵌入的质心，$\mathbf{c}_{i^{-j}}$ 是说话人 $i$ **除第 $j$ 句外**的质心，$\alpha$ 是 margin。GE2E 比传统的 Triplet Loss 更稳定，是说话人验证的奠基性损失。

### 2.2 x-vector（Kaldi / Snyder et al., 2018）

Snyder 等人 2018 年在 Kaldi 工具链中提出的 x-vector 用 TDNN（时延神经网络）抽取说话人嵌入，其结构是：帧级 5 层 TDNN → 拼接均值与标准差 → 2 层全连接 → softmax 说话人分类。训练完后取倒数第二层 512 维向量作为嵌入。x-vector 在 VoxCeleb 数据集上把说话人验证的 EER 从 d-vector 的 8.8% 降到 3.4%，成为 ASV 领域的工业标配。

### 2.3 ECAPA-TDNN：信道鲁棒设计（Desplanques et al., 2020）

x-vector 在信道差异（不同麦克风、不同录音环境）下表现退化。ECAPA-TDNN（**Emphasized Channel Attention, Propagation and Aggregation TDNN**）做了三项关键改进：

1. **通道注意力（SE-block）**：对每一层 TDNN 输出，自适应地为不同频带加权，强调说话人相关频带（如 200-4000 Hz 内的共振峰）。
2. **多层特征聚合（Multi-layer Feature Aggregation, MFA）**：把每一层 TDNN 的输出都汇聚到池化层，而不是只用最后一层。
3. **统计池化改进**：拼接均值 $\mu$ 与标准差 $\sigma$ 后加一个注意力池化：

$$
\mathbf{w}_t = \text{softmax}\left( \mathbf{v}^\top \tanh(\mathbf{W}\mathbf{h}_t + \mathbf{b}) + \mathbf{c} \right), \quad \mathbf{e} = \sum_t w_t \mathbf{h}_t
$$

ECAPA-TDNN 在 VoxCeleb1-E / VoxCeleb1-H 上把 EER 进一步降到 0.87% / 1.16%，并通过 **AAM-Softmax（Additive Angular Margin）** 损失：

$$
\mathcal{L}_{\text{AAM}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s\cos\theta_j}}
$$

把同类夹角收紧 $\theta + m$ 度（典型 $m=0.2$，$s=30$），使嵌入空间更紧凑、更可分。

## 三、说话人条件 TTS：Speaker Encoder + 声学模型

把说话人嵌入接入 TTS 的标准做法是 **Speaker Encoder + Acoustic Model + Vocoder** 三段式。Speaker Encoder 在**预训练阶段**用 ASV 任务（GE2E / AAM-Softmax）训练好，**推理时冻结**；声学模型（Mel 预测器，如 Tacotron2、FastSpeech2）把文本与说话人嵌入拼成条件：

$$
\mathbf{h}_t = \text{FastSpeech2}(Y; \theta) + \text{MLP}(\mathbf{e}_s)
$$

声码器（HiFi-GAN、Vocos）把 mel 谱转成波形。说话人嵌入 $\mathbf{e}_s$ 通常作为 **adapter 或 FiLM 条件**注入声学模型的每一处归一化层。

```python
import torch
import torch.nn as nn


class SpeakerFiLM(nn.Module):
    """把说话人嵌入作为 FiLM 条件调制声学模型的中间特征。"""

    def __init__(self, spk_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(spk_dim, feat_dim)
        self.beta = nn.Linear(spk_dim, feat_dim)

    def forward(self, x: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        # x:      (B, T, F) 声学特征
        # spk_emb:(B, D)    说话人嵌入
        gamma = self.gamma(spk_emb).unsqueeze(1)  # (B, 1, F)
        beta = self.beta(spk_emb).unsqueeze(1)
        return gamma * x + beta


class ECAPA-TDNN(nn.Module):
    """简化版 ECAPA-TDNN：3 层 SE-Res2Net + MFA + AAM-Softmax。"""

    def __init__(self, in_dim: int = 80, emb_dim: int = 192, num_speakers: int = 7205):
        super().__init__()
        # 帧级编码：3 层 1D 卷积 + SE 注意力
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_dim if i == 0 else 512, 512, kernel_size=3, dilation=2**i, padding=2**i),
                nn.BatchNorm1d(512), nn.ReLU(),
            ) for i in range(3)
        ])
        # 统计池化 + 注意力
        self.attn = nn.Linear(512 * 3, 1)  # 输入：mean+std+raw 的拼接
        self.pool_mlp = nn.Sequential(nn.Linear(1536, 512), nn.ReLU())
        # 嵌入层
        self.emb = nn.Linear(512, emb_dim)
        # AAM-Softmax 分类器
        self.classifier = nn.Linear(emb_dim, num_speakers, bias=False)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        # 统计池化
        mu = x.mean(dim=-1)
        std = x.std(dim=-1)
        feats = torch.cat([x, mu.unsqueeze(-1).expand_as(x), std.unsqueeze(-1).expand_as(x)], dim=1)
        w = torch.softmax(self.attn(feats.transpose(1, 2)), dim=1)   # (B, T, 1)
        pooled = (w * x.transpose(1, 2)).sum(dim=1)                  # (B, F)
        h = self.pool_mlp(torch.cat([pooled, mu, std], dim=-1))
        e = self.emb(h)                                              # (B, D)
        logits = self.classifier(e)                                  # (B, num_speakers)
        if labels is None:
            return e
        # AAM-Softmax：把 logits 改为 cos(theta + m)
        with torch.no_grad():
            w = self.classifier.weight
            w_norm = F.normalize(w, dim=-1)
            e_norm = F.normalize(e, dim=-1)
            cos = torch.clamp(e_norm @ w_norm.T, -1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target_cos = torch.cos(theta + m_with) * s_scale            # s=30, m=0.2
        one_hot = F.one_hot(labels, num_speakers)
        logits = torch.where(one_hot > 0, target_cos, e_norm @ w_norm.T)
        return logits, e
```

## 四、少样本克隆的元学习思路

零样本要求 Speaker Encoder 在没见过的说话人上也能泛化，但单条参考音频可能含信道噪声。少样本路径借 **Prototypical Networks（Snell et al., 2017）** 的思路：给定目标说话人的 $K$ 条参考音频，各自通过 Speaker Encoder 得到嵌入 $\{\mathbf{e}_1, \dots, \mathbf{e}_K\}$，原型向量就是质心：

$$
\mathbf{c}_s = \frac{1}{K}\sum_{i=1}^{K} \mathbf{e}_i
$$

测试时把参考嵌入替换为原型向量。实验上 $K=5$ 就比 $K=1$（单条参考）把说话人相似度 SVS 提升约 0.05（余弦）。实际工程中常在 $K$ 条参考间加一个 **self-attention 聚合**：

$$
\mathbf{c}_s = \text{AttnPool}(\{\mathbf{e}_1, \dots, \mathbf{e}_K\})
$$

让模型自己学会给"更干净"的句子更高权重。

## 五、与传统方法的对比

| 维度 | 传统 HMM 拼接合成 | Speaker Encoder + TTS | ECAPA-TDNN 条件 TTS |
|------|---------------------|-----------------------|----------------------|
| 数据需求 | 数小时-数天对齐标注 | 几十小时多说话人 | 数千小时多说话人 |
| 推理延迟 | 实时 | 实时（取决于声码器） | 实时 |
| 可迁移性 | 差换说话人需重做 | 强 | 极强 |
| 自然度 MOS | 3.0-3.5 | 4.0-4.3 | 4.3-4.6 |
| 相似度 SVS | 0.85-0.90 | 0.90-0.93 | 0.93-0.96 |

传统 HMM 拼接合成（如 Festival、MaryTTS）需要为目标说话人录制数小时语料并做音素级 force-alignment，开发周期以"天"计。基于深度学习的 Speaker Encoder + TTS 把开发周期降到"分钟"，且自然度与相似度大幅超越拼接法。

## 六、Speaker Encoder 训练与评估的工程细节

### 6.1 训练数据集

主流 Speaker Encoder 训练语料：

| 数据集 | 规模 | 说话人数 | 场景 |
|--------|------|----------|------|
| VoxCeleb1（2017） | 352 小时 | 1,251 | 公开访谈 |
| VoxCeleb2（2018） | 2,442 小时 | 6,112 | 公开访谈 |
| CNCeleb（2020） | 1,300 小时 | 1,000 | 中文娱乐 |
| LibriSpeech（2015） | 1,000 小时 | 2,438 | 有声书 |

### 6.2 评估指标

**EER（Equal Error Rate）**：使假阳性率（FPR）等于假阴性率（FNR）的工作点。EER 越低越好，VoxCeleb1-E 上 ECAPA-TDNN 达到 0.87%。

**minDCF（minimum Detection Cost Function）**：加权考虑 FPR 与 FNR 的业务成本，常用 $\text{minDCF}_{0.05}$。

### 6.3 多说话人 TTS 中 Speaker Encoder 的复用

工程上 Speaker Encoder 与声学模型是**联合训练**还是**预训练后冻结**？两条路线的实践差异：

- **联合训练**：声学模型的损失能反向传到 Speaker Encoder，让说话人表征对齐到声学合成的需求。优点是相似度高，缺点是 Encoder 不再通用。
- **冻结预训练**：用 ASV 数据集训练的 Encoder 不再被流。优点是 Embedding 通用、可解释，缺点是相似度略低。

VITS、NaturalSpeech 2 多采用冻结预训练；F5-TTS、CosyVoice 多采用联合训练。

## 七、局限与展望

说话人嵌入路线有两个开放问题：（1）**情感与韵律的解耦**——同一说话人嵌入通常耦合了"音色+韵律习惯"，导致克隆出"音色对、韵律像原说话人、情感却不能自主控制"。（2）**跨语言一致性**——多语言预训练里同一说话人不同语言下嵌入可能不聚类，需引入语言无关的因子化设计。零样本声音克隆将在下一篇中通过 YourTTS、VALL-E 等神经编解码语言模型范式进一步突破数据门槛。

## 小结

| 概念 | 关键点 | 公式/方法 |
|------|--------|-----------|
| 声音克隆范式 | 零样本 / 少样本 / 微调 | $f_\theta(Y, X_{\text{ref}}, s)$ |
| d-vector | LSTM + GE2E 损失 | softmax 相似度对比 |
| x-vector | TDNN + 统计池化 | VoxCeleb EER 3.4% |
| ECAPA-TDNN | SE 注意力 + MFA + AAM-Softmax | VoxCeleb1-E EER 0.87% |
| 少样本原型 | Prototypical Networks | $\mathbf{c}_s = \frac{1}{K}\sum \mathbf{e}_i$ |
| 工业现状 | Speaker Encoder + TTS + Vocoder | 开发周期从"天"降到"分钟" |