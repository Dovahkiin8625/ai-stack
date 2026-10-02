# 计算机视觉评测基准：从分类到干到 3D 与视频

视觉模型的"刷榜史"几乎就是深度学习的发展史——从 ImageNet 大规模分类，到 COCO 检测分割，到 ADE3、Cityscapes 3D 重建，再到 SAM202 等视觉基础模型的零样本监督。理解这些 benchmark 的设计意图，能帮助我们更客观地判断"哪个模型真的更好"。

## 一、为什么视觉评测比 LLM 评测更"成熟"

视觉任务天然有"ground truth"——图像标签、检测框、分割掩码都是像素级标注，所以评测方式相对客观：

```python
# 视觉评测相对客观的核心原因
# 1. 输出结构固定（标签、mask、3D box）
# 2. 标注可由人或工具精确生成
# 3. 评测指标可数学化（IoU、mAP、PSNR）
# 4. 不依赖自然语言的主观判断
```

但视觉评测也有自己的"暗礁"——数据集偏置、标注噪声、长尾问题让"刷高分"不等于"用得好"。

## 二、图像分类基准

### ImageNet（ILSVRC）

**120 万张图、1000 类**——深度学习的奠基性数据集：

```python
# ImageNet 结构
imagenet_stats = {
    "train_images": 1_281_167,
    "val_images": 50_000,
    "test_images": 100_000,
    "classes": 1000,
    "metric": "Top-1 / Top-5 Accuracy",
}

# Top-5: 模型输出概率最高的 5 个类别中包含正确类别的比例
# Top-1: 模型输出概率最高的 1 个类别等于正确类别的比例
```

**历史意义**：2012 年 AlexNet 将 Top-5 错误率从 26% 降到 16%，开启了深度学习时代。今天 SOTA 已达 91%+ Top-1 准确率，**已进入饱和阶段**——靠刷 ImageNet 不再能区分前沿模型。

**局限**：类别分布偏（很多类别相关自然图不充分）、标注噪声、单一分辨率（多为 224×224）。

### ImageNet-Real / ImageNet-V2

为修正 ImageNet 评测的早期噪声而生：

```python
# ImageNet-Real
# - 重新清洗了验证集的标签
# - 测试时用多标签平均精度
# - 比原版 ImageNet 难 ~3%，区分度更好

# ImageNet-V2
# - 重新按原 ImageNet 流程收集的验证集
# - 16k 上下采样规模
# - 用以检测模型泛化是否依赖原版验证集的特定样本
```

### ObjectNet

**反转现实控制的物体识别**——让物体处于"反常"姿态（如倾斜的椅子）：

```python
# ObjectNet 的设计理念
# - 物体处于反常姿态（如椅子斜放、杯子放在机柜上）
# - 测试模型对"概念"vs"记忆"的区分
# - Inception-style fusion at unusual views
```

测试模型是否真正"理解"物体，而非依赖训练集常见的相机角度。

## 三、目标检测基准

### COCO（Common Objects in Context）

**20 万图、80 类、1500 万标注框**——检测领域的"事实标准"：

```python
# COCO 标注格式
{
    "image_id": 12345,
    "annotations": [
        {
            "bbox": [x, y, w, h],      # 左上角 + 宽高
            "category_id": 3,          # car
            "area": 3500.5,
            "iscrowd": 0,
            "segmentation": [[...]],   # 多边形
            "opacity": 0.95,
        }
    ],
}
```

**关键指标**：

```python
# COCO mAP 计算
def compute_coco_map(preds, gts):
    """
    mAP@.5: IoU 阈值 0.5 的 mAP
    mAP@.75: IoU 阈值 0.75 的 mAP（更严苛）
    mAP@[.5:.95]: 10 个 IoU 阈值的平均 mAP（COCO 主指标）
    """
    iou_thresholds = [0.5, 0.55, 0.6, ..., 0.95]
    aps = [compute_ap_at_iou(preds, gts, t) for t in iou_thresholds]
    return np.mean(aps)
```

**局限**：类别分布长尾（小目标类高被遗漏）、标注成本高（皮图/框）。

### LVIS（Large Vocabulary Instance Segmentation）

**1200 类**——属于大类别词表实例分割：

```python
# LVIS vs COCO
# COCO: 80 类（常见）
# LVIS: 1200+ 类（长尾）
# LVIS 包含很多"few-shot"类别（仅几张训练样本）
```

LVIS 揭示了模型的"长尾能力"——COCO 满分但在 LVIS（长尾）可能掉到 30%。

### Objects365

**365 类、2200 万框**——中文场景大规模检测数据集。

### OpenImages V3

**190 万图、600 类**——Google 发布的大规模检测数据集。

## 四、分割基准

### ADE20k 语义分割

**2 万图、150 类像素级标注**：

```python
# ADE20k 语义分割标注
{
    "image": "ADE_train_00000001.jpg",
    "mask": "ADE_train_00000001.png",   # 每像素一个类别 ID
    "classes": ["wall", "building", "sky", "tree", "road", "person", ...],
}
```

**指标**：mIoU（mean Intersection over Union）。

### Cityscapes 城市驾驶场景

**5000 张精细标注 + 20000 张粗标注**，19 类：

```python
# Cityscapes 类别
cityscapes_classes = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]
```

**关键评测**：自动驾驶场景的真实表现。

## 六、3D 视觉基准

### KITTI

自动驾驶领域的经典 3D 基准：

```python
# KITTI 任务
kitti_tasks = [
    "stereo",           # 立体匹配
    "flow",             # 光流
    "scene_flow",      # 场景流
    "depth",            # 深度估计
    "object_3d",        # 3D 目标检测
    "tracking",         # 多目标跟踪
    "road",             # 道路/车道检测
    "semantics",        # 半语义分割
]
```

### nuScenes

**40k 帧、23 类、6 相机 + 1 雷达**——比 3D 传感器更新更好的现代数据集：

```python
# nuScenes 特点
n3_samples = {
    "cameras": 6,
    "radar": 1,
    "lidar": 1,
    "frames": 40_000,
    "annotations_3d": 1.4_000_000,  # 1.4M 3D 框
    "classes": 23,
}
```

### Waymo Open

**1000 段、200 万 3D 框**——大规模 3D 检测基准。

### ShapeNet / PartNet / ModelNet40

**3D 形状分类/检索**：

```python
# ModelNet40
# - 40 类（如椅子、桌子、飞机）
# - 12,311 个 CAD 模型
# - 用于 3D 分类、检索、补全
```

## 七、视频理解基准

### Kinetics-400/600/700

**大规模动作识别**：

```python
# Kinetics
kinetics = {
    "400": {"videos": 400_000, "classes": 400},
    "600": {"videos": 500_000, "classes": 600},
    "700": {"videos": 650_000, "classes": 700},
}
```

### Something-Something-V2

**细粒度动作**（如"把 X 推到 Y 后面"）：

```python
# Something-Something-V2 强调"动作理解"
# 而不是物体识别
# 例如："把某个东西推到另一个东西的后面"
# 模型必须理解动作语义，而非仅靠物体外观猜测
```

### UCF-101 / HMDB-51

视频动作识别经典数据集，已进入饱和。

### AVA / AVA-Actions

**时空动作检测**——检测视频中每个人在做什么：

```python
# AVA
# - 每 1 秒采样标注 1 个关键帧
# - 80 类原子动作（走、跑、打电话）
# - 多标签（一个人可同时多个动作）
```

## 八、视觉基础模型基准

### Visual Genome（VG）

**10 万图、540 万区域描述、380 万问答对**——视觉基础模型常用训练集。

### SAM-1B / SA-1B

**Segment Anything 数据集**——1100 万图、11 亿掩码。

### LAION-5B

**58.5 亿图文对**——CLIP 类模型的核心训练数据。

### COCO Captions / Flickr30k

**图像描述**基准。

## 九、人脸关键评测集

### FFHQ（Flickr-Faces-HQ）

**7 万张高质量人脸**——StyleGAN 等生成模型的训练集。

### CelebA-HQ / CelebA

**20 万张名人脸**——含 40 个属性标注。

### WIDER FACE / WIDER Pedestrian

**人脸/行人检测**基准。

## 十、视觉评测的"暗礁"

### 数据集偏置

```python
# ImageNet 的偏置
# - 西方场景多；
# - 偏向特定摄影风格；
# - 类别不平衡。

# 缓解：使用更平衡的数据集或多源融合
```

### 标注噪声

```python
# COCO 标注问题
# - 部分小目标框被遗漏；
# - 遮挡目标的边界标注不一致；
# - 类别边界模糊（如"餐桌"vs"椅子"）。
```

### 评测脚本不一致

```python
# 同一模型，不同评测脚本可能差 2-3 mAP
# 建议：固定使用官方评测脚本 + 标准数据划分
```

## 十一、评测实操建议

```python
# 1. 选 benchmark 与业务场景对齐
# 2. 关注"饱和"信号——多个模型都 90%+ 时不再区分
# 3. 多个 benchmark 组合判断
# 4. 自建内部 benchmark 反映业务真实分布
# 5. 关注模型的"失败模式"而非仅高分样本
```

## 小结

视觉评测比 LLM 评测更"客观"，但仍有数据集偏置、标注噪声、长尾问题等"暗礁"。ImageNet（分类）、COCO（检测分割）、KITTI/nuScenes（3D）、Kinetics（视频）构成视觉评测的骨架。下一篇我们将讨论**多模态评测基准**——图文理解、视频问答、视觉推理等跨模态任务如何评测。