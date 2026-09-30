# TensorFlow 与 JAX：生产部署与函数式编程视角

PyTorch 在研究界一家独大，但深度学习框架不止一个生态：**TensorFlow** 仍然是生产部署（特别是移动端、服务端）的事实标准；**JAX** 在 Google 内部与前沿研究中凭借函数式编程和 XLA 编译迅速崛起。本文从 TF 2.x 的 Keras / `tf.data` / SavedModel 体系出发，再讲 JAX 的四大函数式原语与 Flax / Haiku，最后给出三大框架的横向对比与"何时选谁"的工程建议。

## 一、TensorFlow 2.x：从静态图到 eager 默认

### TF 1.x vs TF 2.x 的核心变化

```python
# TF 1.x: 先建图，再 sess.run
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y = tf.matmul(x, W)
sess = tf.Session()
print(sess.run(y, feed_dict={x: x_data}))

# TF 2.x: 默认 eager，像 numpy 一样写
import tensorflow as tf
W = tf.Variable(tf.zeros([784, 10]))
y = tf.matmul(x_data, W)        # 立即得到结果
```

TF 2.0 把 eager execution 设为默认，整合了 `tf.keras` 作为高级 API，删掉了 `tf.Session` / `tf.placeholder`。`@tf.function` 装饰器仍可把 Python 函数编译成图：

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

### Keras 三种建模方式

| API | 写法 | 适用 |
|---|---|---|
| **Sequential** | `tf.keras.Sequential([...])` | 单输入单输出、层叠结构 |
| **Functional** | `inputs = Input(...); x = Dense(...)(x); Model(inputs, outputs)` | 多输入/多输出、跳跃连接 |
| **Model 子类化** | 继承 `tf.keras.Model`，自定义 `__init__` / `call` | 复杂控制流、研究型自定义 |

```python
# Functional API：带跳跃连接的 MLP
from tensorflow.keras import layers, Model, Input

def residual_mlp(in_dim, hidden, out_dim):
    inputs = Input(shape=(in_dim,))
    x = layers.Dense(hidden, activation='relu')(inputs)
    h = layers.Dense(hidden)(x)
    x = layers.Add()([x, h])               # 跳跃连接
    x = layers.Activation('relu')(x)
    outputs = layers.Dense(out_dim)(x)
    return Model(inputs, outputs)
```

## 二、`tf.data`：高性能数据管线

`tf.data.Dataset` 是 TF 的标准数据抽象，比 PyTorch `DataLoader` 更激进地做图优化：

```python
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds = ds.shuffle(10000) \
       .map(lambda x, y: (tf.image.random_flip_left_right(x), y),  # 增强
            num_parallel_calls=tf.data.AUTOTUNE) \
       .batch(128) \
       .prefetch(tf.data.AUTOTUNE)         # 后台预取下一 batch
```

三个性能旋钮：

- **`map(..., num_parallel_calls=AUTOTUNE)`**：CPU 端并行做数据增强。
- **`batch(..., drop_remainder=True)`**：固定 batch 形状，便于 XLA 编译。
- **`prefetch(buffer_size=AUTOTUNE)`**：让 GPU 算当前 batch 时 CPU 已经在准备下一个 batch。

典型收益：把 `model.fit()` 的 step time 从数据加载瓶颈压到接近 GPU 峰值。

## 三、`model.fit()` vs 自定义训练循环

两种训练范式：

```python
# 1. 高级：model.fit() 一行搞定
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(ds_train, epochs=10, validation_data=ds_val)

# 2. 低级：tf.GradientTape 自定义循环（GAN、RL、混合精度时必需）
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

**经验法则**：标准监督学习用 `model.fit()`；GAN、强化学习、需要梯度惩罚或多优化器时用 `tf.GradientTape`。

## 四、SavedModel 与部署生态

TF 真正的杀手锏是部署链：

```python
# 训练完保存
model.save('export/')                      # 生成 SavedModel 目录

# 任何语言加载
restored = tf.saved_model.load('export/')
```

部署选项：

| 工具 | 场景 |
|---|---|
| **TF Serving** | 服务端 gRPC / REST 部署 |
| **TF Lite** | 移动端 / 嵌入式（量化、算子优化） |
| **TF.js** | 浏览器端推理 |
| **TFX** | 端到端 ML Pipeline |

SavedModel 是**自包含**的（计算图 + 权重 + 签名），是 TF 在工业界依然首选的核心原因。

## 五、分布式策略

```python
# 多卡同步训练
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    model.compile(...)

# 多机
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# TPU
strategy = tf.distribute.TPUStrategy(tpu)
```

`strategy.scope()` 把模型构建和优化器包装在分布上下文里，**零代码改动**就支持多卡 / 多机 / TPU。这是 TF 比 PyTorch 早期更省心的部分。

## 六、JAX：NumPy + 自动求导 + XLA

JAX 由 Google 的 `autograd` 演化而来，定位是"高性能数值计算 + 函数式变换"。它的 API 与 NumPy 几乎一致，但自带求导和编译：

```python
import jax
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.sin(x) @ jnp.cos(x)            # 像 numpy，但跑在 GPU/TPU
```

### 四大原语

| 函数 | 作用 |
|---|---|
| **`jax.grad(f)`** | 对标量输出求梯度 |
| **`jax.jit(f)`** | JIT 编译为 XLA kernel |
| **`jax.vmap(f)`** | 自动向量化（手写 batch 维度消失） |
| **`jax.pmap(f)`** | 多设备并行（一个设备一个 replica） |

```python
def loss(w, x, y):
    pred = jnp.dot(x, w)
    return jnp.mean((pred - y) ** 2)

grad_fn = jax.grad(loss)            # 自动求导
fast_fn = jax.jit(loss)             # 编译加速
batch_fn = jax.vmap(loss, in_axes=(None, 0, 0))   # 自动批处理
```

### 函数式约束

JAX 的核心哲学是**纯函数**：

- **无 in-place 修改**：所有操作返回新数组。
- **无副作用**：函数应只依赖输入与输出。
- **PRNG 显式传**：`key = jax.random.PRNGKey(0); k1, k2 = jax.random.split(key)`。

这让 `jit` / `vmap` / `grad` 可以安全组合，也让大规模并行训练（如 TPU Pod）变得可控。

## 七、JAX 上的神经网络库

JAX 本身只是"数值计算 + 自动微分"，神经网络库是上层：

| 库 | 风格 |
|---|---|
| **Flax**（Google） | `nn.Module` 子类化，生态最完整 |
| **Haiku**（DeepMind） | 函数式 `def forward` + `hk.transform`，类 PyTorch 体验 |
| **Optax** | 解耦的优化器集合（与 Flax 配套） |
| **Equinox** | 更纯粹的函数式 nn 库 |

```python
# Flax 示例
from flax import linen as nn

class MLP(nn.Module):
    hidden: int
    out: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out)(x)
        return x
```

## 八、横向对比

| 维度 | PyTorch | TensorFlow | JAX |
|---|---|---|---|
| 主 API 风格 | OOP，`nn.Module` | OOP + Functional，`tf.keras` | 纯函数 + 变换 |
| 默认执行 | eager | eager（@tf.function 可编译图） | 通过 `jit` 显式编译 |
| 编译器 | `torch.compile` (TorchInductor) | XLA | XLA |
| GPU/TPU | GPU 强，TPU 次之 | GPU + TPU 都好 | GPU + TPU 都好（TPU 最强） |
| 部署生态 | TorchServe / ONNX / 移动端较弱 | TF Serving / Lite / JS 全栈 | 较薄，多靠 TF / PyTorch 转换 |
| 研究论文占比 | 约 70%+ | 约 15% | 约 10%，多在 Google 内部 |
| 学习曲线 | 平缓 | 中等（概念多） | 较陡（函数式 + 显式 PRNG） |
| 调试体验 | 最好 | 中等 | 较好（jit 内部仍难调） |

## 九、何时选谁

```text
场景                          推荐
─────────────────────────────────────
研究 / 论文复现 / 新模型        PyTorch (首选) 或 JAX (若有偏好)
开源模型 / HuggingFace 生态    PyTorch (绝大多数都是 PyTorch-first)
生产部署 (移动 / Web / 服务)   TensorFlow (TF Lite / TF.js / TF Serving)
大规模 TPU 训练                JAX / Flax (Google 内部标配)
教学入门                       PyTorch (API 最直观)
需要严格函数式 / 可微分编程    JAX
已有 TF 2.x 代码库             继续 TF, 别强行迁移
```

**经验法则**：研究用 PyTorch；上线用 TF；玩 TPU / 函数式编程用 JAX。

## 十、迷你代码对比：三层 MLP

### PyTorch（OOP、`nn.Module`）

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in=784, d_h=256, d_out=10):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h), nn.Linear(d_h, d_out)
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### TensorFlow（`tf.keras.Sequential`）

```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10),
])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x_train, y_train, epochs=5, batch_size=128)
```

### JAX（函数式、`jit` + `grad`）

```python
import jax, jax.numpy as jnp
from jax import random, jit, grad

def init_params(key, d_in=784, d_h=256, d_out=10):
    k1, k2, k3 = random.split(key, 3)
    return [random.normal(k1, (d_in, d_h)),
            random.normal(k2, (d_h, d_h)),
            random.normal(k3, (d_h, d_out))]

def forward(params, x):
    for w in params[:-1]:
        x = jnp.dot(x, w); x = jnp.relu(x)
    return jnp.dot(x, params[-1])

@jit
def update(params, x, y, lr=1e-3):
    grads = grad(lambda p: jnp.mean((forward(p, x) - y) ** 2))(params)
    return [w - lr * g for w, g in zip(params, grads)]

params = init_params(random.PRNGKey(0))
```

三段代码的**控制流方向完全不同**：PyTorch 用类继承描述"模型长什么样"，TF 用声明式层序列，JAX 用纯函数 + 变换组合——这也是三个框架哲学差异的缩影。

## 十一、生态收敛：ONNX 与 HuggingFace

虽然三大框架各有偏好，但生态正在**互相通约**：

- **ONNX**：开放的模型中间格式，可在 PyTorch / TF / ONNX Runtime / TensorRT 之间互转。导出：

  ```python
  torch.onnx.export(model, dummy_input, 'model.onnx')
  ```

- **HuggingFace transformers**：几乎所有 SOTA 模型都有 PyTorch 与 TF 两种 checkpoint（`from_pretrained()` 自动选）。这意味着**选模型时可以不必先选框架**。

- **Flax ↔ PyTorch ↔ TF** 的 checkpoint 转换工具（`flax` 的 `from_pt` 等）也越来越成熟。

## 小结

**TensorFlow** 用 `tf.keras` + `tf.data` + SavedModel + TF Lite / TF.js / TF Serving 构成了**最完整的生产部署栈**，仍然是企业级 ML 的首选；**JAX** 用 `grad` / `jit` / `vmap` / `pmap` 四大原语 + 函数式约束在 Google 内部与前沿研究里开辟了独特阵地，特别擅长 TPU 大规模训练。PyTorch、TF、JAX 三大框架并未走向"赢者通吃"，而是通过 ONNX 与 HuggingFace 这类**跨框架中间层**逐步融合：选框架的真正依据是"我在做什么任务、在哪个环境部署、团队熟悉哪种风格"——而不是哪个框架"更好"。