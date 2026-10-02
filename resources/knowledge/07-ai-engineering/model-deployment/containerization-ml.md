# ML 容器化：从训练到推理的可移植镜像

传统软件用 Docker 打包相对简单——选个 base image、装依赖、复制代码。ML 系统的容器化难点在于：模型权重（GB 级）、GPU 驱动、CUDA / cuDNN 版本、数据依赖、以及"训练 vs 推理"环境的差异。本文梳理 ML 容器化的最佳实践：基础镜像选择、多阶段构建、模型权重管理、镜像优化与安全。

## 一、ML 镜像的分层设计

```text
┌──────────────────────────────────────────┐
│ Layer 5: 模型权重 + 业务代码（频繁变更） │
├──────────────────────────────────────────┤
│ Layer 4: 业务依赖（每周/月变更）         │
├──────────────────────────────────────────┤
│ Layer 3: ML 框架（PyTorch / TF）         │
├──────────────────────────────────────────┤
│ Layer 2: CUDA / cuDNN / 系统库           │
├──────────────────────────────────────────┤
│ Layer 1: 基础 OS（Ubuntu / Debian）       │
└──────────────────────────────────────────┘
```

把变更频率低的放底层，频繁变更的放上层，最大化 Docker 缓存复用——同样 1GB 的代码更新，build 时间可以从 30 分钟降到 30 秒。

## 二、基础镜像选型

### 训练环境

```dockerfile
# NVIDIA 官方镜像，自带 CUDA + cuDNN
FROM nvcr.io/nvidia/pytorch:24.01-py3

# 已包含：
# - Ubuntu 22.04
# - CUDA 12.3, cuDNN 8.9
# - Python 3.10
# - PyTorch 2.3
```

其他选择：

- **`nvidia/cuda:12.x-cudnn-runtime`**：仅 CUDA 运行时，自己装框架（更小）。
- **`python:3.11-slim`**：纯 CPU 训练或调试。

### 推理环境

```dockerfile
# 推理镜像要比训练镜像小 5-10×
FROM nvcr.io/nvidia/tritonserver:24.01-py3   # Triton
# 或
FROM pytorch/torchserve:latest               # TorchServe
# 或自建轻量镜像
```

对 LLM 推理，**vLLM / TGI / TensorRT-LLM** 都自带官方 Docker 镜像：

```bash
docker run --gpus all -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

## 三、多阶段构建：训练 vs 推理分离

把训练环境"重型"包（git、wandb、jupyter）和推理"无用"包（开发工具）剥离：

```dockerfile
# ========== Stage 1: 训练 ==========
FROM nvcr.io/nvidia/pytorch:24.01-py3 AS trainer

RUN pip install transformers datasets accelerate wandb

WORKDIR /workspace
COPY src/ ./src/
COPY configs/ ./configs/

RUN python src/train.py --config configs/lora.yaml
# 产出 model.safetensors 到 /workspace/output/

# ========== Stage 2: 推理 ==========
FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS server

# 只复制最终产物
COPY --from=trainer /workspace/output/model.safetensors /models/model/1/
COPY serving/ /models/

# 推理镜像完全没有训练依赖，省 5-10 GB
```

`COPY --from=trainer` 只复制构建产物，镜像层轻巧。

## 四、模型权重管理：放在哪

### 4.1 三种策略

| 策略 | 优点 | 缺点 |
|---|---|---|
| **镜像内 COPY** | 简单、自包含 | 镜像巨大、每次权重变更要重 build |
| **镜像启动时下载** | 镜像小、版本化 | 启动慢、需要凭据管理 |
| **挂载 PV / S3** | 完全解耦 | 运维复杂 |

### 4.2 推荐做法：OCI 镜像 + 独立权重

```dockerfile
FROM python:3.11-slim
RUN pip install vllm
COPY serve.py /app/serve.py
# 不在镜像里塞权重
```

```bash
# 启动时通过环境变量指定模型来源
docker run -e MODEL_PATH=s3://bucket/models/llama3-8b-q4.gguf vllm-image
```

或者把模型推到 **OCI registry**（Docker Hub / GHCR）：

```bash
# 模型作为 OCI artifact 推送
oras push ghcr.io/myorg/llama3-8b:q4 llama3-8b-q4.gguf
```

启动时拉取：

```dockerfile
FROM python:3.11-slim
RUN pip install vllm oras
COPY entrypoint.sh /app/
# entrypoint.sh: oras pull ghcr.io/myorg/llama3-8b:q4 && vllm serve ...
```

### 4.3 大型模型：分片与流式

>100 GB 的模型不适合直接 COPY 进镜像。用 **git-lfs**、**Hugging Face Hub**、**ModelScope** 等托管，启动时拉取或流式加载。

```python
from transformers import AutoModelForCausalLM
# 启动时下载（仅首次）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B-Instruct",
    cache_dir="/data/hf-cache",        # 挂 PV 做缓存
    torch_dtype=torch.bfloat16,
)
```

## 五、镜像优化技巧

### 5.1 减小镜像尺寸

```dockerfile
# 不好：装完不清理
RUN apt-get update && apt-get install -y git
# → 缓存、列表都留下来了

# 好：合并 + 清理
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
```

### 5.2 用 .dockerignore

```
.git
__pycache__
*.pyc
tests/
docs/
*.log
.DS_Store
```

### 5.3 利用 BuildKit 缓存

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install vllm transformers
```

`--mount=type=cache` 让 pip 缓存跨 build 复用，第二次 build 几乎瞬时完成。

### 5.4 Distroless / 极简基础

```dockerfile
# 极端精简：完全没有 shell / 包管理器
FROM gcr.io/distroless/python3-debian12
COPY app.py /app.py
ENTRYPOINT ["python", "/app.py"]
```

适合生产推理，但调试困难。

## 六、安全与合规

### 6.1 镜像签名

```bash
# cosign 签名
cosign sign --key cosign.key myregistry/myimage:v1

# 验证
cosign verify --key cosign.pub myregistry/myimage:v1
```

### 6.2 扫描漏洞

```bash
trivy image myregistry/myimage:v1
# 报告 OS 包、Python 依赖的 CVE
```

集成到 CI：

```yaml
- name: Trivy scan
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: myregistry/myimage:${{ github.sha }}
    severity: 'CRITICAL,HIGH'
    exit-code: '1'    # 严重漏洞阻断合并
```

### 6.3 非 root 运行

```dockerfile
RUN groupadd -r app && useradd -r -g app app
USER app     # 不再用 root
```

## 七、Kubernetes 上的部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 1     # 申请 1 张 GPU
            memory: 32Gi
          requests:
            cpu: 4
            memory: 16Gi
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "s3://bucket/models/llama3-8b"
        volumeMounts:
        - name: hf-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: hf-cache
        persistentVolumeClaim:
          claimName: hf-cache-pvc
```

关键点：

- `resources.limits.nvidia.com/gpu`：申请 GPU。
- 挂 PV 缓存模型权重（避免重复下载）。
- `replicas > 1`：水平扩展。

## 小结

ML 容器化的核心是**分层 + 分离**——基础镜像、ML 框架、业务代码、模型权重各管各的。多阶段构建把训练与推理环境解耦，OCI artifact 把模型权重独立版本化。生产环境要叠加镜像签名、漏洞扫描、非 root 运行、Kubernetes 资源管控。下一篇我们将进入 **CI/CD for ML**——把训练、评估、部署全部流水线化的工程框架。
