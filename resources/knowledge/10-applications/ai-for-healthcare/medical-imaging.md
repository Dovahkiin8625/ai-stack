# 医学影像 AI：从检测到诊断

**医学影像 AI** 是医疗 AI 中商业化最成熟的赛道——DeepMind 的视网膜病变检测、IDx-DR 的自主糖尿病筛查、Viz.ai 的卒中检测，数十家公司已经获得 FDA 批准。本文系统介绍医学影像 AI 的核心任务（分类、检测、分割、配准、生成）、深度学习架构（U-Net、Transformer）、数据挑战（标注稀缺、长尾、隐私）、代表系统，以及从研究到临床的转化路径。

## 一、医学影像 AI 的应用场景

```text
         影像类型          AI 任务            临床价值
────────────────────────────────────────────────────
眼底    视网膜相机      病变检测分级    糖网筛查、防盲
CT      胸部 CT        肺结节检测      肺癌早筛
CT      头部 CT        卒中检测        缩短溶栓时间
MRI     脑 MRI        肿瘤分割        放疗规划
MRI     心脏 MRI      功能评估        心肌活力
X-ray   胸部 X-ray    多病检测        基层医院辅助
病理    全切片图像     癌细胞检测      病理诊断
超声    心脏超声       射血分数        心功能评估
皮肤    皮肤镜图像     黑色素瘤识别    皮肤癌早筛
```

**市场规模**：全球医学影像 AI 市场预计 2030 年达到 **$30B+**。

## 二、核心计算机视觉任务

### 1. 分类（Classification）

判断影像属于哪一类：

```python
import torch
import torch.nn as nn
import timm


class ChestXRayClassifier(nn.Module):
    """胸部 X-ray 多病分类。"""
    def __init__(self, n_classes=14):
        super().__init__()
        # 用预训练 backbone
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=True, num_classes=0
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return torch.sigmoid(self.head(features))  # 多标签 sigmoid


# ChestX-ray14 数据集有 14 类疾病
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
```

### 2. 检测（Detection）

定位影像中的病变（bounding box）：

```python
import torchvision


def get_detection_model(num_classes=2):
    """肺结节检测。"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        num_classes=num_classes + 1,  # +1 for background
        pretrained=True,
    )
    return model


def train_one_epoch(model, dataloader, optimizer):
    model.train()
    for images, targets in dataloader:
        # targets: list of dicts with 'boxes' and 'labels'
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3. 分割（Segmentation）

像素级标注——器官、肿瘤分割：

```python
class UNet(nn.Module):
    """经典 U-Net 分割网络。"""
    def __init__(self, in_channels=1, n_classes=2, base=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, base)
        self.enc2 = self._block(base, base * 2)
        self.enc3 = self._block(base * 2, base * 4)
        self.enc4 = self._block(base * 4, base * 8)
        
        # Bottleneck
        self.bottleneck = self._block(base * 8, base * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = self._block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = self._block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = self._block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = self._block(base * 2, base)
        
        # Output
        self.out = nn.Conv2d(base, n_classes, 1)
    
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
        e4 = self.enc4(nn.functional.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(nn.functional.max_pool2d(e4, 2))
        
        # Decoder + skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)
```

### 4. 配准（Registration）

把不同时间 / 不同模态的影像对齐：

```python
# VoxelMorph 风格
class RegistrationNet(nn.Module):
    """医学影像配准。"""
    def __init__(self):
        super().__init__()
        # U-Net 估计变形场
        self.unet = UNet(in_channels=2, n_classes=2)  # 输出 2D 位移
    
    def forward(self, moving, fixed):
        # 输入：moving 和 fixed 拼接
        x = torch.cat([moving, fixed], dim=1)
        flow = self.unet(x)  # 位移场
        
        # 用位移场 warp moving
        warped = warp_image(moving, flow)
        return warped, flow
```

### 5. 生成（Generation）

用 GAN / Diffusion 生成 / 增强影像：

```python
# 用 Diffusion 做超分辨率 / 重建
class MedicalSRDiffusion:
    """医学影像超分辨率 Diffusion。"""
    def __init__(self, model):
        self.model = model
    
    def super_resolve(self, low_res, n_steps=50):
        """低分辨率 → 高分辨率。"""
        x = low_res
        for t in reversed(range(n_steps)):
            x = self.model.denoise_step(x, t)
        return x
```

## 三、代表系统案例

### 1. IDx-DR（2018，FDA 批准）

**首个自主 AI 诊断**——糖尿病视网膜病变筛查：

```text
工作流：
1. 护士用视网膜相机拍患者眼底
2. 上传至 IDx-DR 系统
3. AI 自动分析：
   - 无明显病变 → 12 个月内复查
   - 中度/重度病变 → 转眼科医生
4. 系统生成报告，无需眼科医生在场
```

**意义**：基层医院、全科诊所可独立完成糖网筛查——缓解眼科医生不足。

### 2. Viz.ai（LVO 卒中检测）

```python
def stroke_detection_pipeline(ct_image):
    """LVO（大血管闭塞）检测。"""
    # 1) 自动检测 LVO
    has_lvo = lvo_detector.predict(ct_image)
    
    # 2) 如果检测到 LVO，立即通知
    if has_lvo:
        # 触发卒中团队
        notify_stroke_team(patient_id, ct_image, timestamp)
        # 同时建议转院到有血管内治疗能力的中心
        suggest_transfer()
```

**临床价值**：缩短"门到针"时间（door-to-needle），每分钟约损失 190 万神经元——这是真正的"与时间赛跑"。

### 3. Paige.AI（前列腺癌）

**首个 FDA 批准的病理 AI**：

```text
工作流：
1. 病理学家扫描组织切片（WSI）
2. Paige Prostate 标识可疑区域
3. 病理学家 review AI 标注
4. 确认/否定癌症诊断

效果：在大型研究中，与病理学家准确率相当，但速度更快。
```

### 4. Google ARDA（糖尿病视网膜病变）

Google 在印度、泰国部署的糖尿病视网膜病变筛查 AI——已在临床使用。

### 5. Aidoc / Annalise.ai

综合影像 AI 平台——多病种、多模态：

```python
class AidocPlatform:
    """综合医学影像 AI。"""
    def __init__(self):
        self.models = {
            "PE": pulmonary_embolism_detector,        # 肺栓塞
            "stroke": stroke_detector,                  # 卒中
            "rib_fracture": rib_fracture_detector,      # 肋骨骨折
            "intracranial_hemorrhage": ich_detector,   # 颅内出血
            "abdominal_free_air": free_air_detector,   # 腹腔游离气体
        }
    
    def triage(self, ct_image, clinical_context):
        """分诊：哪些需要立即 review。"""
        findings = {}
        for condition, model in self.models.items():
            if model.predict(ct_image):
                findings[condition] = {
                    "confidence": model.confidence,
                    "urgency": URGENCY_LEVELS[condition],
                }
        return sorted(findings.items(), key=lambda x: x[1]["urgency"])
```

## 四、数据挑战

### 1. 标注稀缺

医疗影像标注**极昂贵**——需要专家：

```python
def estimate_annotation_cost(dataset_size, expert_hourly_rate=200):
    """估算标注成本。"""
    minutes_per_image = {
        "classification": 0.5,
        "detection": 2,
        "segmentation": 10,    # 像素级标注
        "wsi": 30,             # 全切片图像
    }
    hours = dataset_size * minutes_per_image["segmentation"] / 60
    cost = hours * expert_hourly_rate
    return cost

# 1000 张 CT 分割标注 = 167 小时 × $200 = $33,400
```

**解决方案**：

```python
# 1) 主动学习
class ActiveLearningPipeline:
    def __init__(self, model, unlabeled_data):
        self.model = model
        self.unlabeled = unlabeled_data
    
    def select_to_label(self, n_samples=100):
        """选最值得标注的样本。"""
        uncertainty_scores = []
        for img in self.unlabeled:
            # 多模型集成的不确定性
            preds = [m.predict(img) for m in self.ensemble]
            disagreement = compute_disagreement(preds)
            uncertainty_scores.append(disagreement)
        
        # 选不确定性最高的 n_samples
        idx = np.argsort(uncertainty_scores)[-n_samples:]
        return [self.unlabeled[i] for i in idx]
```

### 2. 长尾分布

罕见病标注极少——**few-shot learning**：

```python
class FewShotLearner:
    """用 1-5 个样本学习新病种。"""
    def __init__(self, base_model):
        self.model = base_model  # 在常见病上预训练
    
    def adapt(self, support_set, n_shot=5):
        """用 support set 中的 n_shot 个样本微调。"""
        # prototype network
        prototypes = {}
        for class_id, samples in support_set.items():
            features = [self.model.encode(s) for s in samples[:n_shot]]
            prototypes[class_id] = torch.stack(features).mean(0)
        return prototypes
    
    def predict(self, test_image):
        features = self.model.encode(test_image)
        # 与 prototype 计算距离
        distances = {
            cls: -torch.norm(features - proto) 
            for cls, proto in self.prototypes.items()
        }
        return min(distances, key=distances.get)
```

### 3. 数据隐私

医疗数据不能外传——**联邦学习** + **差分隐私**：

```python
class FederatedMedicalTraining:
    """跨医院联邦训练。"""
    def __init__(self, global_model, hospital_clients):
        self.global = global_model
        self.clients = hospital_clients  # 各医院本地模型
    
    def train_round(self):
        # 各医院本地训练
        local_updates = []
        for client in self.clients:
            update = client.train_one_epoch()  # 数据不出医院
            
            # 差分隐私：梯度加噪
            noisy_update = add_gaussian_noise(update, sigma=0.1)
            local_updates.append(noisy_update)
        
        # 中心聚合
        self.global.aggregate(local_updates)
```

### 4. 数据标准化

不同医院、不同设备的影像差异大：

```python
class DomainAdaptation:
    """跨医院 / 跨设备域适应。"""
    
    def histogram_match(self, source, target):
        """直方图匹配：让 source 分布接近 target。"""
        # 把 source 的灰度分布映射到 target 的分布
        sorted_target = np.sort(target.flatten())
        sorted_source = np.sort(source.flatten())
        
        # 建立映射
        mapping = np.interp(
            np.linspace(0, 1, len(sorted_source)),
            np.linspace(0, 1, len(sorted_target)),
            sorted_target
        )
        # 应用
        return mapping[np.searchsorted(sorted_source, source)]
```

## 五、模型架构演进

### 1. CNN 时代（2015~2020）

- **U-Net**（Ronneberger 2015）：医学影像分割的"标配"。
- **ResNet** / **DenseNet**：分类 backbone。
- **Faster R-CNN** / **YOLO**：检测。

### 2. Transformer 时代（2020~）

```python
class SwinUNet:
    """Swin Transformer + U-Net 混合架构。"""
    def __init__(self):
        # Encoder 用 Swin Transformer
        self.encoder = SwinTransformer(...)
        # Decoder 用 CNN 上采样
        self.decoder = UNetDecoder(...)
```

代表：
- **TransUNet**：Transformer + U-Net。
- **SwinUNet**：纯 Swin。
- **nnFormer**：3D 医学影像 Transformer。
- **Medical SAM**：基于 SAM 的医学影像分割。

### 3. 基础模型时代（2023~）

```python
class MedicalFoundationModel:
    """医学影像基础模型。"""
    def __init__(self):
        # 在大规模医学影像上预训练
        self.backbone = VisionTransformer(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            n_heads=16,
        )
        # 多种下游任务适配器
        self.adapters = {
            "classification": ClassificationHead(...),
            "detection": DetectionHead(...),
            "segmentation": SegmentationHead(...),
        }
```

代表：
- **MedSAM**（2023）：基于 SAM 的医学影像分割。
- **RadFM**（2023）：放射学基础模型。
- **LLaVA-Med**（2023）：医学影像 + 视觉问答。
- **BiomedCLIP**（2023）：医学 CLIP。

### 4. 多模态时代

```python
class MultimodalMedicalAI:
    """多模态医学影像 AI。"""
    def __init__(self):
        self.ct_encoder = ViT_CT()
        self.mri_encoder = ViT_MRI()
        self.xray_encoder = ViT_XRay()
        self.text_encoder = BioBERT()
        self.fusion = MultimodalFusion()
    
    def analyze(self, image=None, text=None, lab=None, genetic=None):
        """融合多种数据综合诊断。"""
        features = []
        if image:
            features.append(self.encode_image(image))
        if text:
            features.append(self.text_encoder(text))
        if lab:
            features.append(self.lab_encoder(lab))
        
        return self.fusion(features)
```

## 六、模型评估

### 1. 关键指标

```python
def medical_ai_metrics(y_true, y_pred, threshold=0.5):
    """医学 AI 关键指标。"""
    tp = ((y_pred > threshold) & (y_true == 1)).sum()
    fp = ((y_pred > threshold) & (y_true == 0)).sum()
    fn = ((y_pred <= threshold) & (y_true == 1)).sum()
    tn = ((y_pred <= threshold) & (y_true == 0)).sum()
    
    return {
        "sensitivity": tp / (tp + fn + 1e-6),  # 召回率（漏诊率 = 1 - sensitivity）
        "specificity": tn / (tn + fp + 1e-6),  # 特异度
        "ppv": tp / (tp + fp + 1e-6),          # 阳性预测值
        "npv": tn / (tn + fn + 1e-6),          # 阴性预测值
        "auc": compute_auc(y_true, y_pred),
    }
```

### 2. 临床场景的指标权衡

**关键：根据临床场景选指标**

| 场景 | 关键指标 | 原因 |
|---|---|---|
| **癌症筛查** | 高敏感度（低漏诊） | 漏诊 = 致命 |
| **急性病检测（卒中）** | 高敏感度 | 时间敏感 |
| **病理辅助** | 高 PPV | 减少假阳性 |
| **慢病监测** | 综合 | 平衡漏诊与误诊 |

### 3. 跨数据集泛化

模型在一个医院训练，**在另一个医院往往效果下降**——是部署最大障碍：

```python
def evaluate_cross_site(model, datasets):
    """评估跨数据集泛化。"""
    results = {}
    for site_name, dataset in datasets.items():
        metrics = medical_ai_metrics(dataset.y_true, model.predict(dataset.X))
        results[site_name] = metrics
    return results


# 通常结果：
# In-domain AUC: 0.95
# Out-of-domain AUC: 0.82  ← 显著下降
```

**缓解**：
- 多医院数据联合训练（联邦学习）。
- 域适应技术。
- 风格归一化。

## 七、从研究到临床的转化

### 1. 临床验证步骤

```text
研究阶段:
  - 回顾性研究（已有影像 + 真实标签）
  - 与医生对比

验证阶段:
  - 前瞻性多中心研究
  - 真实临床环境测试

监管阶段:
  - FDA / NMPA 提交
  - 510(k) / De Novo / PMA

部署阶段:
  - 医院 IT 集成（PACS、HIS）
  - 用户培训
  - 持续监控
```

### 2. 真实临床部署的挑战

```python
class ClinicalIntegration:
    """临床 IT 集成。"""
    
    def integrate_with_pacs(self, ai_model, pacs_url):
        """与医院 PACS 系统集成。"""
        # 1) 监听新影像
        self.pacs_listener = PACSListener(pacs_url)
        self.pacs_listener.on_new_study(self.process_study)
    
    def process_study(self, study):
        """处理一份新影像。"""
        # 1) 下载 DICOM
        dicom_files = study.download()
        image = dicom_to_array(dicom_files)
        
        # 2) AI 分析
        findings = self.ai_model.predict(image)
        
        # 3) 生成 DICOM SR（结构化报告）
        sr = self.generate_dicom_sr(findings, study)
        
        # 4) 推回 PACS
        self.pacs_listener.upload_report(sr, study)
        
        # 5) 触发警报（如需要）
        if findings["urgency"] == "STAT":
            self.notify_radiologist(study, findings)
```

### 3. FDA 510(k) 流程

```text
510(k) 申请材料：
1. 设备描述
2. 性能测试报告（与对照设备对比）
3. 软件文档（IEC 62304）
4. 风险管理（ISO 14971）
5. 标签与说明书
6. 临床数据（如适用）
```

## 八、伦理与监管

### 1. 偏见与公平

医学影像 AI 已被证明存在**群体差异**：

```python
# Google 研究: 胸部 X-ray 模型在不同族群上表现差异
# Nature Medicine 2020: 严重皮肤烧伤检测模型在深色皮肤上表现差
```

**缓解**：
- 训练数据**多样化**。
- 跨群体性能测试。
- 持续监控不同人群效果。

### 2. 透明度

医生需要理解 AI 决策——**可解释性**：

```python
def gradcam_visualization(model, image, target_class):
    """Grad-CAM 可视化。"""
    # 计算目标类的梯度
    grad = compute_gradient(model, image, target_class)
    
    # 权重化激活图
    weights = grad.mean(dim=(2, 3), keepdim=True)
    activation_map = (weights * activations).sum(dim=1)
    
    # 上采样到原图大小
    heatmap = nn.functional.interpolate(
        activation_map.unsqueeze(0),
        size=image.shape[-2:],
        mode='bilinear',
    )
    return heatmap.squeeze().cpu().numpy()
```

### 3. 持续监控

模型部署后必须**持续监控**：

```python
class ProductionMonitoring:
    """生产环境监控。"""
    
    def monitor_drift(self, predictions, ground_truth_delayed):
        """监控模型漂移。"""
        # 1) 数据漂移：输入分布变化
        # 2) 概念漂移：标签与预测关系变化
        # 3) 性能漂移：实际准确率变化
        
        recent_auc = compute_auc(ground_truth_delayed, predictions)
        if recent_auc < 0.85:  # 阈值
            alert("MODEL_DEGRADATION")
            trigger_retraining()
```

## 九、未来方向

### 1. 全切片图像（WSI）AI

病理 WSI 是 GB 级别的超大图像——需要**多实例学习（MIL）**：

```python
class MILModel:
    """多实例学习：把整个 WSI 切成小块，再聚合判断。"""
    
    def __init__(self, patch_encoder):
        self.encoder = patch_encoder  # 编码每个 patch
    
    def forward(self, wsi):
        """WSI: (N_patches, C, H, W) 一堆 patch。"""
        # 1) 编码每个 patch
        features = [self.encoder(p) for p in wsi]
        features = torch.stack(features)  # (N_patches, D)
        
        # 2) 注意力聚合
        attention = self.attention_net(features)  # (N_patches, 1)
        weighted_features = (features * attention).sum(dim=0)
        
        # 3) 分类
        return self.classifier(weighted_features)
```

### 2. 3D 影像 AI

CT / MRI 是 3D 数据——需要 **3D 网络**：

```python
class Medical3DNet(nn.Module):
    """3D 医学影像网络。"""
    def __init__(self):
        # 3D 卷积 / 3D Transformer
        self.conv3d = nn.Conv3d(...)
        # Memory-efficient 实现
```

### 3. 实时影像分析

术中导航 / 介入治疗——需要**毫秒级**推理：

```python
# 用 TensorRT / ONNX Runtime 优化
# 模型量化 INT8
# 模型剪枝
```

### 4. 数字病理学 + 多组学

```text
影像 + 基因组 + 蛋白质组 + 临床数据
        ↓
  综合诊断与个性化治疗
```

## 十、给医学影像 AI 团队的清单

1. **临床问题驱动**：从临床需求出发，不只是"用 AI"。
2. **多中心数据**：避免单中心偏差。
3. **放射科医生协作**：从数据标注到模型评估全程参与。
4. **FDA 路径**：明确产品分类与监管路径。
5. **真实临床测试**：前瞻性研究，不是只回顾性。
6. **可解释性**：Grad-CAM、attention 可视化等。
7. **公平性审计**：跨族群、跨设备、跨中心。
8. **持续监控**：生产环境漂移检测。
9. **PACS 集成**：与医院现有系统无缝衔接。
10. **法规合规**：HIPAA、GDPR、当地数据法规。

## 小结

医学影像 AI 是医疗 AI 中**商业化最成熟、监管最严格**的赛道——U-Net 是标配，Transformer 与基础模型是前沿。**六大应用**——眼底筛查、肺结节检测、卒中检测、肿瘤分割、病理诊断、皮肤癌识别——都已 FDA 批准。**核心挑战**是标注稀缺、跨域泛化、公平性、持续监控。**真正成功的医学影像 AI** 是"AI 辅助医生、医生最终决策、严格监管、持续改进"的闭环。下一篇我们将看到医疗 AI 的另一前沿方向——**AI for Drug Discovery**：用 AI 加速新药研发。
