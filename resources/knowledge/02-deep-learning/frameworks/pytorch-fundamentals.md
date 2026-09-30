# PyTorch 基础：从张量到训练循环

PyTorch 是当前深度学习研究的事实标准。它凭借**动态计算图**（define-by-run）、直观的 Pythonic API、强大的 GPU 加速以及丰富的生态系统，几乎垄断了学术论文与开源模型实现。本文系统讲解 PyTorch 的核心抽象、训练循环、设备管理与 `torch.compile`，并对比 PyTorch eager / `torch.compile` / TensorFlow 静态图三种执行模式，让你对这套"研究主流框架"形成完整的心智模型。

## 一、为什么 PyTorch 赢得了研究界

2016 年之前，深度学习研究主流是 TensorFlow 1.x。它的执行模式是**静态图**：先 `tf.placeholder` 定义输入，再 `tf.Session()` 启动图，最后 `sess.run` 取结果。这种模式有三个痛点：

1. **调试困难**：报错是堆栈在 C++ 运行时里，Python `print` 打不出中间值。
2. **控制流笨重**：`tf.cond`、`tf.while_loop` 这种图内控制流写起来绕。
3. **条件分支不直观**：同一个输入想走不同网络结构得手动拼图。

PyTorch 反其道而行——**每次前向传播都即时构建计算图**（define-by-run / dynamic graph）：

```python
import torch
x = torch.randn(3, requires_grad=True)
y = x * 2
if y.sum() > 0:        # 普通 Python 控制流
    z = y * 3
else:
    z = y / 3
z.sum().backward()     # 反向传播自动穿越 if/else
```

这种"图跟着代码走"的范式让 PyTorch 像普通 Python 库一样好调试：`pdb.set_trace()`、`print(x.shape)`、`if/else` 全部天然可用。2019 年 TensorFlow 2.0 才在 `tf.enable_eager_execution`（默认开启）之后补上了这一课，但社区惯性已经形成。

## 二、核心抽象

### `torch.Tensor`

`Tensor` 是 numpy `ndarray` 的 GPU 升级版，并内置自动求导。关键差异：

| 特性 | numpy.ndarray | torch.Tensor |
|---|---|---|
| GPU 加速 | 无 | `.to('cuda')` 即可 |
| 自动求导 | 无 | `requires_grad=True` |
| 与 numpy 互转 | — | `tensor.numpy()` / `torch.from_numpy(arr)` |
| 默认 dtype | float64 | float32 |

```python
import torch
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 显式指定 dtype
x = x.to('cuda')               # 移到 GPU
print(x.requires_grad)         # False，普通张量
y = x * 2 + 1                  # 普通运算
```

### `torch.nn.Module`

所有模型都继承 `torch.nn.Module`。两个核心方法：

- **`__init__`**：定义层（子模块）。PyTorch 自动追踪所有 `nn.Module` / `nn.Parameter`，注册到 `model.parameters()`。
- **`forward(x)`**：定义前向计算。**不要直接调用 `model.forward(x)`**，而是 `model(x)`（会顺带跑 hooks）。

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x.flatten(1))     # (B, 784) -> (B, 10)

model = MLP()
print(sum(p.numel() for p in model.parameters()))   # ~204k 参数
```

### `torch.optim`

优化器持有模型参数的引用，按算法（`SGD` / `Adam` / `AdamW`）更新梯度：

```python
import torch.optim as optim
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss.backward()
opt.step()      # 用梯度更新参数
opt.zero_grad()  # 清零梯度（PyTorch 默认累加）
```

### `torch.utils.data.Dataset` + `DataLoader`

数据流水线两层抽象：

- **`Dataset`**：单样本抽象。`__len__` 返回样本数，`__getitem__(i)` 返回第 i 个样本。
- **`DataLoader`**：在 `Dataset` 上做批采样、打乱、多进程加载：

```python
from torch.utils.data import DataLoader, TensorDataset
ds = TensorDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
```

### `torch.autograd`

自动微分引擎。只要 `requires_grad=True` 的张量参与了运算，PyTorch 就记录一条计算图，`loss.backward()` 会沿图反向传播梯度。

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)      # tensor([2*x0, 2*x1, 2*x2])
```

推理时不需要梯度，用 `torch.no_grad()` 上下文跳过记录，节省显存：

```python
with torch.no_grad():
    pred = model(x_test)             # 不构建计算图
```

## 三、标准训练循环

下面是一段**经典 15 行训练循环**，几乎所有 PyTorch 项目都长这样：

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP().to(device)
opt   = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()                    # 训练模式：启用 dropout / 更新 BN 统计
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()              # 关键：清零上一步梯度
        loss.backward()              # 反向传播
        opt.step()                   # 优化器更新参数

    model.eval()                     # 推理模式：关闭 dropout / 用 BN running stats
    val_loss = 0
    with torch.no_grad():            # 推理不记录梯度
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_loss += loss_fn(model(x), y).item()
    print(f'epoch {epoch}  val_loss {val_loss/len(val_loader):.4f}')
```

几个**最容易踩的坑**：

- **忘记 `model.eval()`**：BatchNorm 会继续用当前 batch 的均值/方差更新 running stats，导致推理结果不稳。
- **忘记 `opt.zero_grad()`**：PyTorch 梯度默认累加（方便梯度累积实现大 batch），所以每步训练前必须清零。
- **在 `no_grad` 里调用 `loss.backward()`**：会直接报错 `RuntimeError`，因为没有计算图。
- **`model(x)` vs `model.forward(x)`**：前者会跑 hooks，后者不会——用前者。

## 四、设备管理

### 单 GPU / CPU 切换

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x, y = x.to(device), y.to(device)
```

`.to(device)` 是**in-place**地改 device 引用，模型参数和数据都搬到同一设备才能算。

### 简易多卡：`nn.DataParallel`

```python
model = nn.DataParallel(model, device_ids=[0, 1])   # 自动切分 batch 到两张卡
```

**注意**：主卡（GPU 0）承担更多工作（汇总梯度、输出），是常见瓶颈。生产环境已逐渐被更高效的 `DistributedDataParallel`（DDP）取代。

## 五、保存与加载

### 只保存参数（推荐）

```python
# 保存
torch.save(model.state_dict(), 'model.pt')

# 加载
model = MLP()                              # 必须先实例化同结构
model.load_state_dict(torch.load('model.pt'))
model.eval()
```

`state_dict()` 是一个 `{层名: 参数张量}` 的有序字典，**只含参数不含结构**，因此必须先重建模型。这是迁移学习、模型共享的标准做法。

### 完整检查点（断点续训）

```python
# 保存
torch.save({
    'epoch': epoch,
    'model_state': model.state_dict(),
    'opt_state':   opt.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'best_acc':    best_acc,
}, 'checkpoint.pt')

# 加载
ckpt = torch.load('checkpoint.pt')
model.load_state_dict(ckpt['model_state'])
opt.load_state_dict(ckpt['opt_state'])
start_epoch = ckpt['epoch'] + 1
```

**为什么要存 optimizer state**：Adam 这种自适应优化器有动量项 `m` / `v`，中断后如果重新初始化，相当于用错了历史统计，loss 会突然飙升。

## 六、`torch.compile`：JIT 编译加速

PyTorch 2.0 引入 `torch.compile`，把 Python 级的 eager 执行**JIT 编译**成融合 CUDA kernel：

```python
model = torch.compile(model, mode='reduce-overhead')   # 一行启用
# 后续 model(x) 会在第二次调用时触发编译
```

`mode` 三个常用档位：

| mode | 行为 | 适用 |
|---|---|---|
| `default` | 平衡 | 一般训练 |
| `reduce-overhead` | 用 CUDA Graph 减 launch 开销 | 小模型、batch 大 |
| `max-autotune` | 搜索最优 kernel 组合 | 编译耗时可接受时 |

**典型加速**：在 Transformer、CNN 上 `1.3× – 2×` 的 throughput 提升，几乎免费。但有动态 shape（变长输入）、大量 Python 控制流时可能失效或回退。

## 七、Eager vs `torch.compile` vs TF 静态图

| 维度 | PyTorch eager | `torch.compile` | TF 1.x 静态图 |
|---|---|---|---|
| 执行时机 | 即时 | 第一次调用时编译 | sess.run 时 |
| 调试 | 直接 `print` / `pdb` | 部分可调试 | 困难 |
| 性能 | baseline | 1.3×–2× 加速 | 接近 `torch.compile` |
| 动态控制流 | 自然 | 部分支持 | 需用 `tf.cond` |
| 部署 | TorchScript / ONNX | 同左 | SavedModel 原生 |

简而言之：研究 / 快速迭代用 eager；追求性能加 `torch.compile`；部署可用 TorchScript 导出或转 ONNX。

## 八、加载预训练模型并微调

HuggingFace / `torchvision` 提供的预训练模型，标准用法是冻结 backbone、只训最后一层：

```python
import torchvision.models as models

# 1. 加载预训练 ResNet18，替换最后的 fc
model = models.resnet18(weights='IMAGENET1K_V1')
for p in model.parameters():             # 冻结 backbone
    p.requires_grad = False
model.fc = nn.Linear(512, 2)             # 二分类（猫/狗）
model.fc.requires_grad = True             # 只解冻最后一层

# 2. 优化器只传需要更新的参数
opt = optim.AdamW(model.fc.parameters(), lr=1e-3)
```

这种"冻结 + 小学习率"模式在小数据上几乎不会过拟合，是迁移学习的事实默认。

## 九、PyTorch 生态

| 子库 | 用途 |
|---|---|
| `torchvision` | 图像数据集、模型、数据增强 |
| `torchaudio` | 音频 I/O、特征提取 |
| `torchtext` | 文本数据集、tokenizer |
| `torchserve` | 生产环境部署（TorchServe） |
| `torch.distributed` | DDP / FSDP / RPC 分布式训练 |
| `torch.fx` | 图变换与量化 |

加上 **HuggingFace transformers / accelerate / peft**，整个 PyTorch 生态几乎覆盖了从研究到生产的全部场景。

## 十、常见陷阱速查

- **BatchNorm / Dropout 忘记切模式**：`model.train()` / `model.eval()`。
- **梯度累加未清零**：每步前 `opt.zero_grad()`。
- **`torch.no_grad()` 里调用 `backward()`**：去掉 `with` 或用 `torch.enable_grad()`。
- **`requires_grad=False` 的参数被训**：检查是否传错了 `opt = optim.AdamW(...)`。
- **GPU 内存爆**：`batch_size` 太大 / 没 `del` 中间变量 / 没调 `torch.compile`。

## 小结

PyTorch 用**动态图 + Pythonic API** 赢得了研究界，核心抽象是 `Tensor`（带自动求导的 GPU 数组）、`nn.Module`（可组合的模型层）、`optim`（优化器）、`DataLoader`（数据流水线）、`autograd`（反向传播引擎）。掌握"模型 → 损失 → `zero_grad` → `backward` → `step`"这条 15 行训练循环主干，再配合 `state_dict` 检查点、`torch.compile` 加速、以及 torchvision/HuggingFace 的迁移学习模板，就能覆盖绝大多数深度学习研究与中小规模生产任务。下一篇文章将对比 TensorFlow 和 JAX，看看生产部署与函数式编程视角下另外两套生态的设计哲学。