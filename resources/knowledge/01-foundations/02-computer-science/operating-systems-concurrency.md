# 操作系统与并发

## 一、进程、线程与协程

**进程**是资源分配的最小单位，每个进程有独立虚拟地址空间、文件描述符表、信号处理。**线程**是 CPU 调度的最小单位，同一进程的线程共享地址空间但各自有栈、寄存器。**协程**（如 Python `asyncio`、Go goroutine）是用户态的轻量"线程"，由程序自己调度，开销极小。

| 维度 | 进程 | 线程 | 协程 |
|---|---|---|---|
| 切换开销 | 高（内核态） | 中（内核态） | 低（用户态） |
| 通信方式 | 管道、socket、共享内存 | 共享变量 | Channel / await |
| 并行（多核） | ✅ | ✅ | ❌（单线程内） |
| 隔离性 | 强 | 弱 | 弱 |
| 典型用例 | 多服务、数据并行 worker | I/O 并发、PyTorch DataLoader | 高并发网络服务 |

Python 由于 GIL（全局解释器锁），**多线程无法跑 CPU 密集任务**，所以 PyTorch 的 `DataLoader` 用多进程（`num_workers > 0`）做 CPU 端的数据增强；`num_workers = 0` 时主进程串行加载，会成为 GPU 利用率的瓶颈。

```python
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,           # 4 个 worker 进程
    pin_memory=True,         # 把数据预加载到 CUDA pinned memory
    prefetch_factor=2,       # 每个 worker 预取 2 个 batch
    persistent_workers=True, # 跨 epoch 复用 worker，避免重启开销
)
```

## 二、同步原语

多线程/多进程共享数据时，必须用同步原语避免竞态。

**互斥锁（mutex）**：同一时刻只允许一个线程进入临界区。

```python
import threading
lock = threading.Lock()
counter = 0
def inc():
    global counter
    with lock:
        counter += 1   # 读-改-写是原子的
```

**读写锁（RWLock）**：读多写少场景下允许多读者并发，写时独占。

**条件变量（Condition）**：让线程阻塞等待某个条件成立。`producer-consumer` 队列的经典实现。

**信号量（Semaphore）**：维护一个计数器，限制同时访问资源的线程数（例如限制并发下载数为 16）。

**原子操作**（CAS）：硬件支持的 compare-and-swap，是无锁队列、计数器的基础。Python 没有原生的 CAS，但可以用 `queue.Queue`、Rust/Go 有专门的原子类型。

ML 场景中常见并发陷阱：

- **共享 GPU 上下文**：多进程共享同一个 CUDA tensor 会触发序列化错误，应避免。
- **`+=` 不是原子**：Python 中 `counter += 1` 实际是 load-add-store 多步，需加锁。
- **回调中改共享状态**：Dataloader 的 `collate_fn` 在 worker 进程跑，不要用主进程的全局对象。

## 三、内存模型与可见性

JVM、C++、Rust、Go 都有自己的内存模型。核心问题：**一个线程的写入何时对其他线程可见？**

- **CPU 缓存**：每个核有自己的 L1/L2 缓存，主存是最终一致来源。如果不强制刷新，另一个核看到的可能是旧值。
- **内存屏障（memory barrier）**：强制 CPU 把缓存刷新到主存并禁止重排。
- **Java `volatile`** / **C++ `atomic`** / **Go `atomic`** 都通过屏障保证可见性。

```go
var flag atomic.Bool
go func() {
    flag.Store(true)   // 写屏障
}()
if flag.Load() {      // 读屏障
    // 保证看到 Store 之前的写入
}
```

Python 的 GIL 实际上意外提供了"内存一致性"——大多数 CPython 实现下，简单的 list/dict 操作在 GIL 释放/获取时是可见的。但跨进程必须用 `multiprocessing.Manager`、`Pipe` 或 `Queue` 显式通信。

## 四、I/O 模型

**同步阻塞**：发起 I/O 后线程被挂起直到完成。简单但浪费 CPU。

**同步非阻塞**：I/O 立即返回 `EAGAIN`，线程轮询。`O_NONBLOCK` socket 是这种。

**I/O 多路复用**：单线程监听多个 fd 的事件。Linux 上的 `select`、`poll`、`epoll`；BSD/macOS 的 `kqueue`。**epoll** 是 Linux 高并发服务器的核心（NGINX、Redis 用它）。

**异步 I/O**：发起 I/O 后立即返回，内核完成后通过回调/事件通知完成。Linux `io_uring`、Windows IOCP、Node.js 事件循环。

```python
# asyncio 高并发 HTTP
import asyncio, aiohttp

async def fetch(url):
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            return await r.text()

async def main():
    results = await asyncio.gather(*[fetch(u) for u in urls])
```

ML 场景：

- **异步日志**：写日志时阻塞主循环会拖慢推理，用 `aiofiles` 或队列 + 后台线程。
- **批量推理服务**：用 `asyncio` 处理成千上万的并发请求，每请求 ms 级响应。
- **流式 LLM 输出**：服务端用 SSE / WebSocket 流式把 token 推给客户端。

## 五、虚拟内存与进程隔离

现代 OS 用**虚拟内存**给每个进程一个独立的地址空间：

- **页表**（page table）把虚拟页映射到物理页框。
- **TLB**（Translation Lookaside Buffer）缓存最近翻译，加速地址翻译。
- **缺页中断**：访问未映射页时内核会从磁盘加载（swap）或分配新页。
- **OOM Killer**：物理内存耗尽时内核按 oom_score 杀进程。

ML 工程的几个关键点：

- **大模型的 KV-cache**：占用大量"虚拟内存"，要预留足够 RAM 或用 offloading。
- **mmap 数据集**：HuggingFace `datasets` 用 mmap 把数据文件映射进内存，多进程共享而不复制。
- **交换（swap）**：绝对不要让 GPU 训练进程 swap——PCIe 带宽会让 swap 比正常计算慢 1000 倍。

```python
# mmap 读取大文件，避免一次性载入
import mmap, os
with open('big.bin', 'rb') as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # 随机访问 mm[offset:offset+size] 类似 numpy 切片
        chunk = mm[offset:offset+4096]
```

## 六、文件系统与持久化

**文件系统**把字节组织到文件、提供目录、权限。Linux ext4 / XFS、Windows NTFS、macOS APFS。ML 数据常用：

- **大文件顺序读**：训练 checkpoint 通常几 GB 到几十 GB，顺序 IO 远快于随机。
- **小文件随机读**：ImageNet 那种百万张小图就是 IO 瓶颈，需要打包成 tar / TFRecord / WebDataset。
- **POSIX 语义**：原子重命名（`rename(src, dst)` 替换 dst）是安全的；直接 `open(dst, 'w')` 然后 `write` 不是（崩溃会留半截文件）。

```python
# 安全的 checkpoint 保存：先写临时文件，再 rename
import torch, tempfile, os
tmp = model_path.with_suffix('.tmp')
torch.save(model.state_dict(), tmp)
tmp.rename(model_path)   # 原子替换
```

## 七、CUDA 与异构计算视角

GPU 不是 CPU，调度模型完全不同：

- **Host（CPU）** 启动 kernel 到 **device（GPU）**，kernel 在 SM（streaming multiprocessor）上并行执行数千线程。
- **kernel 异步**：启动后 host 不等完成；用 `torch.cuda.synchronize()` 或 `event` 显式同步。
- **Stream**：不同 stream 上的 kernel 可以并发；同一 stream 串行。
- **内存传输**：H2D、D2H、D2D 都是显式的，PCIe / NVLink 带宽远低于 GPU 显存带宽。

```python
# 多 stream 并行数据传输与计算
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
with torch.cuda.stream(s1):
    next_batch = next_batch.to('cuda', non_blocking=True)
# 主 stream 等 s1
torch.cuda.current_stream().wait_stream(s1)
```

**统一内存**（Unified Memory）让 CPU/GPU 共享地址空间，页按需迁移，对超大模型友好但显式管理困难。

## 八、常见并发模式与陷阱

- **线程池 vs 进程池**：CPU 密集用进程池，I/O 密集用线程池。
- **Future 与 Promise**：把"未来的结果"作为对象传递，便于组合（`concurrent.futures`、`asyncio.Future`）。
- **Actor 模型**：每个 actor 单线程，消息驱动（Erlang、Akka）。LLM agent 框架某种意义上就是 actor。
- **背压（backpressure）**：生产快于消费时让生产者慢下来，避免 OOM（Kafka 流处理核心概念）。

```python
# 经典死锁：两个锁互相等待
# 避免：始终按固定顺序加锁；或用 try_acquire + 超时回滚
```

ML 实战中的死锁案例：

- **多卡 NCCL**：如果各卡进入 barrier 的顺序不一致可能死锁；要保证每张卡都执行同样的 collective。
- **Dataloader + 锁**：自定义 collate_fn 中加锁可能与 PyTorch 内部锁冲突，导致 hang。
- **Async + sync 混用**：在 asyncio 循环里调阻塞 IO 会卡死整个 event loop；用 `loop.run_in_executor` 切到线程池。

## 小结

操作系统与并发的核心是"**共享 + 同步 + 隔离**"——共享数据要同步，访问共享资源要隔离。ML 工程尤其需要理解：Python GIL 让线程不能跑 CPU 密集、CUDA stream 决定 GPU 并发粒度、mmap 是大文件 IO 的关键、mmap + pin_memory + 多 worker 是训练 IO 的标准配方、checkpoint 写临时文件再 rename 保证一致性。掌握这些，你就能写出既快又稳的训练和推理服务代码，而不是在生产事故时手忙脚乱地查 hang。