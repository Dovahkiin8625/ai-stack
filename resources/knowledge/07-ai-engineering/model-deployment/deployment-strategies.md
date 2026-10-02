# 模型部署策略：影子、A/B、Canary 与蓝绿

模型部署不是"把权重文件复制到服务器"那么简单。一次仓促的全量上线可能让线上指标腰斩。成熟的部署策略用**流量分配 + 渐进切换 + 实时回滚**，把发布风险降到最低。本文梳理四种主流策略——影子、A/B、Canary、蓝绿——以及它们的适用场景与工程实现。

## 一、四种策略概览

```text
策略      │ 流量切换   │ 风险隔离   │ 适用场景
──────────┼────────────┼────────────┼─────────────────────────
影子发布  │ 0%（异步） │ 完全隔离   │ 验证模型行为、性能基准
A/B 测试  │ 5-50%      │ 用户分组   │ 对比新旧模型/算法
Canary    │ 1-10%      │ 实例隔离   │ 渐进放量，监控告警
蓝绿发布  │ 100%瞬切   │ 双倍资源   │ 快速回滚的版本切换
```

## 二、影子发布：零风险的并行验证

影子发布把新模型的请求与旧模型并行执行，**但只把旧模型的响应返回给用户**——新模型的输出仅用于观察和评估。

```text
用户请求 ──┬──→ 旧模型 v1 ──→ 返回用户
           │
           └──→ 新模型 v2 ──→ 异步记录到评估系统
```

```python
async def shadow_predict(request):
    # 主路径：旧模型
    response = await old_model.predict(request)

    # 影子路径：新模型，不阻塞用户
    asyncio.create_task(
        new_model.predict(request)
        .then(lambda pred: log_for_eval(request, pred, response))
    )
    return response
```

**优点**：

- 零风险：用户感知不到任何变化。
- 真实流量：拿到的样本分布与生产一致。
- 可对比：直接比对 v1 和 v2 的输出差异。

**缺点**：

- 双倍算力成本。
- 不能直接测"用户对 v2 的真实反馈"（点击、转化）。
- 不适合对延迟极敏感的场景。

## 三、A/B 测试：业务指标驱动的对比

A/B 测试把流量按用户 ID 哈希分成多组（通常 50/50 或 90/10），**两组用户都得到真实响应**，靠统计显著性判断哪个版本胜出。

```python
import hashlib

def assign_bucket(user_id: str, salt: str = "v2") -> str:
    h = hashlib.md5(f"{user_id}{salt}".encode()).hexdigest()
    return "v2" if int(h, 16) % 100 < 50 else "v1"   # 50/50 切分

async def serve(request):
    user_id = request.user_id
    bucket = assign_bucket(user_id)

    if bucket == "v1":
        return await v1_model.predict(request)
    else:
        return await v2_model.predict(request)
```

**关键要素**：

- **样本量**：上线前要算 power analysis，避免"还没等到结论就下掉"。
- **一致性**：同一用户始终在同一组（粘性 bucketing）。
- **业务指标**：CTR、转化率、人均时长、留存——比 accuracy 更能反映真实价值。
- **SRM 检查**：Sample Ratio Mismatch，统计两组实际流量比例是否严重偏离设定。

**优点**：直接测业务效果。**缺点**：需要足够流量支撑、等待统计显著需要时间、流量被分成两半会拉长实验周期。

## 四、Canary 渐进发布：金丝雀

Canary（金丝雀）把少量流量（1-5%）先切到新模型，观察延迟、错误率、业务指标，全部健康后逐步扩大到 10% → 50% → 100%。

```text
Step 1 (5 min):   99% ──→ v1,  1% ──→ v2
Step 2 (15 min):  90% ──→ v1, 10% ──→ v2
Step 3 (30 min):  50% ──→ v1, 50% ──→ v2
Step 4 (确认 OK): 0%  ──→ v1, 100% ──→ v2
```

**流量切换的实现**：

- **API 网关层**：Kong / Nginx / Envoy 按路由权重分配。
- **服务网格**：Istio / Linkerd 用 VirtualService 配置权重。
- **Kubernetes**：用 Ingress + 服务多版本 + HPA。

**回滚机制**：

```yaml
# Istio VirtualService
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ml-model
spec:
  hosts:
  - ml-model
  http:
  - route:
    - destination:
        host: ml-model-v1
      weight: 90
    - destination:
        host: ml-model-v2
      weight: 10
```

触发回滚的告警：

- 错误率 > 1%
- P99 延迟 > 2× 基线
- 业务指标下降 > 5%

**优点**：渐进、可控、自动回滚。**缺点**：配置复杂、需要监控 + 告警 + 自动化编排。

## 五、蓝绿发布：瞬时切换 + 快速回滚

蓝绿维护两套完全相同的生产环境（蓝色 = 旧版本，绿色 = 新版本）。流量通过负载均衡器切换，瞬时切到绿色，发现问题秒级切回蓝色。

```text
                  Load Balancer
                       │
        ┌──────────────┴──────────────┐
        ↓                             ↓
   蓝色（旧 v1）                  绿色（新 v2）
   100% 流量                       0% 流量
```

```bash
# 上线：流量从蓝色切到绿色
kubectl patch service ml-model -p '{"spec":{"selector":{"version":"v2"}}}'

# 回滚：切回蓝色（秒级）
kubectl patch service ml-model -p '{"spec":{"selector":{"version":"v1"}}}'
```

**优点**：切换 / 回滚都在秒级。**缺点**：双倍资源成本、数据库 schema 变更需要兼容设计。

## 六、策略组合：现代发布流水线

实际生产往往**组合使用**：

```text
1. 影子发布 1-3 天（验证模型行为、对比指标）
       ↓
2. Canary 5% × 1 小时（监控延迟、错误率）
       ↓
3. A/B 50/50 × 1 周（验证业务指标）
       ↓
4. 全量蓝绿切换（确认胜出后）
```

每一步失败都回滚，损失最小化。

## 七、关键工程能力

### 7.1 模型路由

```python
# 简单的路由层
class ModelRouter:
    def __init__(self):
        self.routes = {
            "v1": ModelEndpoint("v1", weight=90),
            "v2": ModelEndpoint("v2", weight=10),
        }

    async def predict(self, request):
        endpoint = self.weighted_select(request.user_id)
        return await endpoint.predict(request)
```

更复杂的需求（基于地理位置、用户分层、上下文特征）用配置中心 + 动态路由。

### 7.2 监控 + 自动回滚

```python
async def canary_monitor():
    while True:
        v1_metrics = await get_metrics("v1")
        v2_metrics = await get_metrics("v2")
        if v2_metrics.error_rate > v1_metrics.error_rate * 1.5:
            await rollback_to("v1")
            alert("Canary v2 触发自动回滚")
            return
        await asyncio.sleep(10)
```

### 7.3 数据库 Schema 兼容

新模型可能需要新字段（如更细粒度的预测类别）。要设计**向后兼容**的存储：

```sql
-- 新模型需要的字段，nullable，旧模型不写也不报错
ALTER TABLE predictions ADD COLUMN new_category VARCHAR(50);
```

## 小结

模型部署的核心是把"上线风险"变成"可控的渐进过程"。影子发布用于行为验证、A/B 测试用于业务对比、Canary 用于平滑放量、蓝绿用于瞬时回滚。生产环境通常把四种策略组合使用，配合自动监控 + 回滚，才能既快又稳地把模型推到全量。下一篇我们将进入 **容器化 ML**——把模型 + 依赖打包成可移植镜像的工程实践。
