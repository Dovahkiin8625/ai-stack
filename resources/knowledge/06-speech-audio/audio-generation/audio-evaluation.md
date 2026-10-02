# 音频生成评估：客观指标与主观听测

一个 24 kHz 语音波形展开是 24000 维的连续向量，"这张生成的音频好不好"没有唯一答案。**图像领域有 FID 和 IS，但音频更复杂**——人耳对相位、谐波、噪声敏感，对小幅度失真有非线性容忍。音频生成评估需要同时回答三个问题：

1. **音质**：波形听起来是否清晰、无噪声、无失真？
2. **相似度**：生成结果与目标参考有多接近？
3. **可控性**：提示词（文本/类别/旋律）是否被准确反映？

这一篇系统梳理客观指标（FAD、IS、PESQ、STOI、MCD）与主观听测（MOS、CMOS、ABX），并讨论评估协议的统计学标准——为什么单一指标永远不够。

## 一、评估的三个层次

### 1.1 音质（Quality）

衡量生成波形本身的**绝对听感**，不依赖任何参考。好的音质意味着"听起来像专业录音"——无背景噪声、无金属感、无失真颤动。客观上用 FAD、IS、ViSQOL 度量，主观上用 MOS 度量。

### 1.2 相似度（Similarity / Fidelity）

衡量生成结果与**参考目标**的距离。TTS 要求生成的"hello"听起来像标准发音；声音克隆要求生成结果与目标说话人音色一致；音频重建（codec）要求重建与原波形尽可能相同。客观上用 PESQ、STOI、MCD 度量。

### 1.3 可控性（Controllability）

衡量提示词对生成结果的影响。文本提示要求生成"狗叫"时，结果真的听起来像狗叫。客观上用 CLAPScore、CIDEr 度量。**可控性是文本-音频对齐独有的维度**，也是当前研究的痛点。

## 二、客观指标族谱

### 2.1 FAD：Frechet Audio Distance

FAD（Kilgour et al., 2019）借鉴图像领域的 FID（Heusel et al., 2017），把生成音频和真实音频分别编码成 embedding，计算两组高斯分布的 Frechet 距离：

$$
\text{FAD} = \|\mu_g - \mu_r\|^2 + \text{Tr}\left(\Sigma_g + \Sigma_r - 2(\Sigma_g \Sigma_r)^{1/2}\right)
$$

其中 $(\mu_g, \Sigma_g)$ 是生成集 embedding 的均值与协方差，$(\mu_r, \Sigma_r)$ 是真实集的。FAD 越小，生成集分布越接近真实集分布。

**关键选择：用什么 embedding？** 三种主流方案：

| Embedding | 来源 | 优点 | 缺点 |
| --- | --- | --- | --- |
| VGGish | Hershey et al., 2017 | 经典 baseline | 8 kHz 限制，音频通用 |
| PANN | Kong et al., 2020 | 音频事件分类强 | 对音乐过拟合 |
| CLAP | Elizalde et al., 2023 | 跨模态对齐 | 对齐误差会进入 FAD |

AudioSet 训练集规模越大、CLAP 微调越充分，FAD 与 MOS 的相关性越高。**重要警告**：FAD 对样本数非常敏感，< 500 条样本时 FAD 数值不稳定，建议至少 1000 条。

### 2.2 IS：Inception Score

IS（Salimans et al., 2016）从图像领域迁移而来，计算生成样本在分类器上的**边缘分布与条件分布的 KL 散度**：

$$
\text{IS} = \exp\left( \mathbb{E}_x \left[ D_{\text{KL}}(p(y \mid x) \| p(y)) \right] \right)
$$

其中 $p(y \mid x)$ 是分类器对生成样本 $x$ 的类别分布，$p(y)$ 是边缘分布。直觉：好的生成样本应该"明确属于某一类"（$p(y \mid x)$ 尖锐）且"类别多样"（$p(y)$ 均匀），所以 IS 越大越好。

**音频专用分类器**：AudioSet 的 PANNs CNN14 是事实标准。**IS 的局限**：只看分类多样性，不看样本质量；可能"模型把所有生成都识别为 dog bark，IS 也很高"。

### 2.3 KL 散度与 KL-PASS

KL 散度直接比较两个分布：

$$
D_{\text{KL}}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i}
$$

PASS（Patron et al., 2021）把生成集与真实集的频谱包络做 KL 比较，比 FAD 更轻量。适合工业场景下的快速 A/B。

### 2.4 PESQ：语音感知质量

PESQ（Perceptual Evaluation of Speech Quality, ITU-T P.862, 2001）是**专门为语音设计的客观指标**，模拟人耳对语音的感知：

$$
\text{PESQ} = a_0 + a_1 \cdot D_{\text{ind}} + a_2 \cdot D_{\text{ase}}
$$

其中 $D_{\text{ind}}$ 是失真距离，$D_{\text{ase}}$ 是非对称失真距离。PESQ 取值 -0.5 到 4.5，4.5 接近原始波形。**PESQ 只适用于语音**，对音乐和声音事件无意义。

### 2.5 STOI：短时客观可懂度

STOI（Short-Time Objective Intelligibility, Taal et al., 2011）衡量**语音可懂度**，与单词识别正确率高度相关：

$$
\text{STOI} = \frac{1}{T} \sum_{t=1}^{T} \frac{(\mathbf{x}_t - \mu_x)^\top (\mathbf{y}_t - \mu_y)}{\|\mathbf{x}_t - \mu_x\| \cdot \|\mathbf{y}_t - \mu_y\|}
$$

其中 $\mathbf{x}_t, \mathbf{y}_t$ 是时频表示的短时帧。STOI 取值 0-1，越接近 1 越清晰。STOI 不关心"音质好坏"，只关心"听不听得清"。

### 2.6 MCD：梅尔倒谱失真

MCD（Mel Cepstral Distortion）是**语音合成的事实标准**：

$$
\text{MCD} = \frac{10}{\ln 10} \sqrt{2 \sum_{d=1}^{D} (\text{mc}_d^x - \text{mc}_d^y)^2}
$$

其中 $\text{mc}_d^x$ 是生成语音的第 $d$ 维梅尔倒谱系数。MCD 单位是 dB，越小越好；典型的 Tacotron 2 + WaveNet 系统 MCD 在 3-5 dB 之间。

### 2.7 ViSQOL：虚拟语音质量

ViQSOL（Virtual Speech Quality Objective Listener, Hines et al., 2015）是 Google 提出的语音质量指标，模拟人耳听觉模型，输出 1-5 分。**比 PESQ 更鲁棒**，对噪声、低码率、丢包容忍度更好。

## 三、主观听测

### 3.1 MOS：Mean Opinion Score

MOS 是音频评估的**金标准**。受试者对每条音频按 1-5 分打分：

| 分值 | 含义 |
| --- | --- |
| 5 | Excellent（优秀，听不出差异） |
| 4 | Good（好，但能听出区别） |
| 3 | Fair（一般，明显是合成） |
| 2 | Poor（差，明显失真） |
| 1 | Bad（极差，几乎听不懂） |

MOS 的标准协议：

1. **样本数**：每系统至少 20-40 条样本；
2. **受试者数**：每条样本至少 10-15 名受试者（避免个体差异）；
3. **随机化**：样本顺序打乱，避免顺序偏差；
4. **环境**：安静室内、监听耳机或标准化扬声器；
5. **听前训练**：先放 5 条样本让受试者熟悉评分标准。

MOS 95% 置信区间通常在 ±0.1-0.2，**差异 < 0.2 通常不显著**。

### 3.2 CMOS：比较 MOS

CMOS（Comparative MOS）让受试者**对同一内容的两个版本直接打分对比**：

$$
\text{CMOS} = \frac{1}{N} \sum_{i=1}^{N} (s_i^A - s_i^B)
$$

取值 -3 到 +3，正值表示 A 优于 B。CMOS 比 MOS 更稳定（受试者直接对比），是工业界 A/B 测试的主流。

### 3.3 ABX 偏好测试

ABX 让受试者听 A、B、X 三条音频，判断 X 听起来更接近 A 还是 B。ABX 适合"音色克隆"、"声音转换"这类**相似度判断**任务。统计上用二项检验判断偏好是否显著。

## 四、文本对齐评估

### 4.1 CLAPScore

CLAPScore（Elizalde et al., 2023）用 CLAP 的相似度直接打分文本-音频对齐：

$$
\text{CLAPScore} = \frac{1}{N} \sum_{i=1}^{N} \text{sim}(t_i, a_i)
$$

其中 $t_i$ 是文本嵌入，$a_i$ 是生成音频嵌入。CLAPScore 越高，文本-音频对齐越好。**局限**：CLAP 本身的偏差会传递；CLAP 训练时未见的概念（如"婴儿哭声"）分数不可靠。

### 4.2 AudioCaps 上的 CIDEr / SPIDEr

AudioCaps（Kim et al., 2019）是音频描述数据集，1 万条 YouTube 音频 + 人工描述。生成音频后用 AudioCaps 验证集测试：

1. **CIDEr**：基于 n-gram TF-IDF 余弦，衡量生成描述与参考描述的相似度；
2. **SPIDEr**：SPICE（语义命题）+ CIDEr 的混合评分。

但这是**间接评估**——文本-音频生成需要先写一段描述，再与参考比较。

### 4.3 文本-音频检索精度

另一种方案：给定一批 (文本, 生成音频) 对，训练一个**文本-音频检索模型**（通常就是 CLAP），计算 top-K 检索精度。这种评估更鲁棒，因为不依赖单次相似度。

## 五、评估协议的统计学标准

### 5.1 样本数

FAD 官方建议 **≥ 2048** 条样本；MOS **≥ 40 条样本 × 10 受试者**；CMOS **≥ 20 对**。低于这个数量，结果的**统计功效（statistical power）**不够，结论不可靠。

### 5.2 显著性检验

| 检验 | 适用场景 |
| --- | --- |
| 配对 t 检验 | MOS / CMOS 两组对比 |
| Wilcoxon 符号秩检验 | 顺序尺度、不正态分布 |
| 费舍尔精确检验 | ABX 偏好（分类数据） |
| bootstrap | FAD 置信区间 |

显著性水平通常取 $\alpha = 0.05$，多重比较时做 **Bonferroni 校正**。

### 5.3 置信区间

报告结果时**必须**给出置信区间，不能只给均值。例如：

> "System A achieves MOS 4.32 ± 0.08 (95% CI), which is significantly higher than System B (MOS 4.18 ± 0.09, p < 0.01)."

### 5.4 公开测试集

不同数据集评估结果**不可直接比较**。AudioCaps、VCTK、LibriTTS 的 MOS 分数不在同一量级。学术界应尽量用统一测试集：

- **VCTK**：英语多说话人 TTS 标准；
- **LibriTTS**：长语音 TTS；
- **AudioCaps**：音频描述；
- **FSD50K / ESC-50**：声音事件分类。

## 六、为什么单一指标不够

### 6.1 FAD 与 MOS 的相关性不是 1

大量实证（e.g., Vinay & Lerch, 2022）显示 FAD 与 MOS 的 Spearman 相关在 0.5-0.7 之间，**还有大量 FAD 看不到的信息**：MOS 反映的整体听感包含语义、情绪、节奏等因素。AudioLDM 的 FAD 优秀但人声可能失真；WaveGrad 的 FAD 较差但 TTS 自然度高。

### 6.2 不同任务需要不同指标

- **TTS 重建**：PESQ + STOI + MCD + MOS；
- **声音克隆**：余弦说话人相似度 + MOS + ABX；
- **文本-音频生成**：CLAPScore + FAD + MOS；
- **音频 codec**：ViSQOL + STFT 距离 + MOS；
- **音乐生成**：CLAPScore + FAD + 节拍对齐度 + MOS。

### 6.3 推荐组合方案

一个**最简稳健**的音频生成评估协议应包括：

1. **客观音质**：FAD（CLAP 版）+ PESQ（语音）/ ViSQOL；
2. **客观对齐**：CLAPScore + 文本-音频检索 top-1；
3. **客观相似度**：说话人余弦相似度（克隆）/ MCD（TTS）；
4. **主观音质**：MOS（≥ 40 样本 × 15 受试者）；
5. **统计显著性**：配对 t 检验 + 95% CI。

任何**只用单一指标的论文**都应被视为"探索性"，需要至少三种以上指标 + 主观听测才能下结论。

## 小结

| 指标 | 维度 | 取值 | 任务 | 主观相关性 |
| --- | --- | --- | --- | --- |
| FAD | 分布距离 | 0-∞ | 通用 | 0.6-0.7 |
| IS | 多样性 | 1-∞ | 通用 | 0.4-0.5 |
| PESQ | 语音质量 | -0.5-4.5 | 语音 | 0.85+ |
| STOI | 语音可懂度 | 0-1 | 语音 | 0.9+ |
| MCD | 梅尔失真 | dB | TTS | 0.7-0.8 |
| ViSQOL | 语音质量 | 1-5 | 语音 | 0.85+ |
| CLAPScore | 文本对齐 | 0-1 | 文本→音频 | 0.5-0.6 |
| MOS | 整体听感 | 1-5 | 通用 | 1.0（金标准） |

音频生成评估是工程与统计学的结合：**没有万能指标，只有组合协议**。论文应明确报告样本数、受试者数、显著性检验、置信区间；工业部署应建立内部 MOS 评测流水线，并辅以 FAD / CLAPScore 做快速回归。下一步将是把这套评估协议与具体的可复现实验脚本结合——这需要工程化的评测平台，超出本篇的范畴。
