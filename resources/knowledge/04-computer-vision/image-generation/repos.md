# 开源仓库：图像生成

## diffusers (Hugging Face)

- 仓库：https://github.com/huggingface/diffusers
- 简介：当前最主流的扩散模型库，覆盖 DDPM / Stable Diffusion / SDXL / ControlNet / LoRA 训练；统一 Pipeline 接口易用。
- 适用：文生图、图生图、Inpainting、ControlNet 控制生成。

## stable-diffusion-webui (AUTOMATIC1111)

- 仓库：https://github.com/AUTOMATIC1111/stable-diffusion-webui
- 简介：功能最丰富的 SD Web UI，插件生态庞大（LoRA、ControlNet、SDXL、AnimateDiff 等）。
- 适用：交互式出图、模型/插件调试。

## StyleGAN (NVIDIA)

- 仓库：https://github.com/NVlabs/stylegan3
- 简介：StyleGAN3 官方实现，潜空间可控的高质量人脸/风格生成。
- 适用：GAN-based 高质量图像合成研究。

## ControlNet

- 仓库：https://github.com/lllyasviel/ControlNet
- 简介：通过 Canny/Depth/Pose 等条件控制扩散模型生成，论文 *Adding Conditional Control to Text-to-Image Diffusion Models* 官方实现。
