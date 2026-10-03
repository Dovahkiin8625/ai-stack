# 开源仓库：CNN / 视觉骨干

> 与本目录主题相关的开源仓库索引。点击仓库名直达 GitHub。

## PyTorch Image Models (timm)

- 仓库：https://github.com/huggingface/pytorch-image-models
- 维护：Hugging Face（原作者 Ross Wightman）
- 简介：当前最全面的视觉骨干库之一，覆盖 CNN（ResNet/EfficientNet/ConvNeXt/RegNet）与 ViT 家族，配套训练脚本、数据增强（Mixup/CutMix/RandAugment）和预训练权重。
- 适用：图像分类骨干选型、ImageNet 预训练、迁移到下游检测/分割任务。

## torchvision

- 仓库：https://github.com/pytorch/vision
- 简介：PyTorch 官方视觉库，包含经典 CNN 模型定义（AlexNet/VGG/ResNet 等）、常用数据集加载器与标准变换。
- 适用：教学、基准对照、快速实验。

## MMPreTrain (OpenMMLab)

- 仓库：https://github.com/open-mmlab/mmpretrain
- 简介：OpenMMLab 预训练仓库，统一封装 50+ 骨干网络与训练配方，支持分类、自监督、MAE 等。
- 适用：需要多模型对照实验、生产级训练 pipeline。
