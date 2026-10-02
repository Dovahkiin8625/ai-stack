# AI 天气与气候建模：从 GraphCast 到 Pangu-Weather

天气预报是**数值预报（NWP）的最后堡垒**——过去 70 年都靠物理方程 + 超级计算机。2022~2023 年，DeepMind 的 **GraphCast**、华为的 **Pangu-Weather**、英伟达的 **FourCastNet** 几乎同时发布，**AI 天气模型首次在准确度上超越传统 ECMWF IFS**——10 天预测准确度提升 10~25%。本文系统介绍 AI 天气建模的原理（GraphCast、Pangu、FourCastNet）、与传统数值预报的对比、商业应用（能源、农业、灾害预警），以及局限与未来。

## 一、传统数值天气预报

### 1. 基本流程

```text
1. 数据同化：把全球观测数据（卫星、探空、地面）融合成初始场
2. 数值积分：求解大气运动方程
   - 动量方程（牛顿第二定律）
   - 连续方程（质量守恒）
   - 热力学方程（能量守恒）
   - 状态方程（理想气体）
   + 辐射、湍流、对流等参数化
3. 后处理：插值到网格、生成预报产品
```

### 2. 局限

| 维度 | 传统数值预报 | 现状 |
|---|---|---|
| **计算成本** | 数千 CPU 核心 + 数小时 | 巨大 |
| **延迟** | 6 小时/12 小时更新 | 较慢 |
| **空间分辨率** | 9km / 25km 全球网格 | 较粗 |
| **极值预报** | 极端天气常不准 | 弱 |
| **小尺度** | 雷暴、龙卷难预报 | 难 |
| **数据同化** | 复杂反演 | 高门槛 |

## 二、AI 天气模型的革命

### 1. GraphCast（DeepMind, 2022）

**首个超越 ECMWF 的 AI 天气模型**：

```python
class GraphCast(nn.Module):
    """GraphCast：基于 Graph Neural Network 的全球天气模型。"""
    def __init__(self):
        # 多分辨率图（高层稀疏，地表密集）
        self.encoder = GraphEncoder(...)
        self.processor = GraphProcessor(
            n_layers=16,
            hidden_dim=512,
        )
        self.decoder = GraphDecoder(...)
    
    def forward(self, state_t):
        """
        state_t: 当前时刻全球天气状态
        返回: 未来 6 小时的预测
        """
        # 1) 编码到 latent
        h = self.encoder(state_t)
        # 2) GNN 处理
        h = self.processor(h)
        # 3) 解码为预测
        return self.decoder(h)
```

**关键创新**：
- **多分辨率 graph**：减少计算量。
- **Encoder-Processor-Decoder 架构**。
- **直接预测未来状态**而非 delta。
- 在 **ERA5 数据**上训练（1979~2021）。

**成果**：
- 10 天准确度提升 **20~25%**。
- 计算速度提升 **1000x**（1 GPU vs 数千 CPU）。
- 2023 年被 ECMWF 集成到 Coperniqus 系统。

### 2. Pangu-Weather（华为, 2023）

```python
class PanguWeather(nn.Module):
    """Pangu：3D Earth-Specific Transformer。"""
    def __init__(self):
        # 3D Swin Transformer + 地球特定设计
        self.encoder = EarthSpecific3DEncoder(...)
        self.decoder = EarthSpecific3DDecoder(...)
    
    def forward(self, input_fields):
        """
        input_fields: 多层高空 + 地表变量
        返回: 各层预测
        """
        return self.decoder(self.encoder(input_fields))
```

**创新**：
- **3D Swin Transformer**：垂直分层处理。
- **层级时间聚合**：1h / 3h / 6h / 24h 多尺度预测。
- **Hierarchical Temporal Aggregation (HTA)**：用 4 个不同时间分辨率模型级联。

**成果**：
- 准确度与 GraphCast 相当，某些指标超过。
- 极端天气（台风路径）更准。

### 3. FourCastNet（Nvidia, 2023）

```python
class FourCastNet(nn.Module):
    """FourCastNet：Adaptive Fourier Neural Operator。"""
    def __init__(self):
        # AFNO + Vision Transformer
        self.spectral_blocks = AdaptiveFourierNeuralOperator(
            n_blocks=8,
            modes=20,
        )
    
    def forward(self, state):
        """Adaptive Fourier Neural Operator。"""
        for block in self.spectral_blocks:
            state = block(state)  # 频域混合
        return state
```

**创新**：
- **Fourier Neural Operator**：在频域做混合。
- 比 GNN 更快。
- 用 ERA5 数据训练。

### 4. 其它代表

```python
AI_WEATHER_MODELS = {
    "GraphCast": "DeepMind, GNN",
    "Pangu-Weather": "华为, 3D Transformer",
    "FourCastNet": "Nvidia, Fourier Neural Operator",
    "NowcastNet": "DeepMind, 短临预报",
    "MetNet-2/3": "Google, 美国本土短临",
    "FuXi": "复旦, 中国团队",
    "FengWu": "上海 AI Lab",
    "Weyn et al.": "Microsoft + 学术",
    "Keisler et al.": "Graph Nets (GraphCast 前身)",
}
```

## 三、技术细节

### 1. 数据：ERA5

所有 AI 天气模型都用 **ERA5** 训练——ECMWF 1979 年至今的**再分析数据**：

```python
ERA5_VARIABLES = {
    # 地表（地表变量）
    "2m_temperature": "K",
    "10m_u_wind": "m/s",
    "10m_v_wind": "m/s",
    "mean_sea_level_pressure": "Pa",
    "total_precipitation": "m",
    
    # 高空（约 13 个气压层）
    "geopotential": "m²/s²",
    "u_wind": "m/s",
    "v_wind": "m/s",
    "temperature": "K",
    "specific_humidity": "kg/kg",
    
    # 共 5 * 13 + ~10 = ~75 个变量
}
```

分辨率：0.25° × 0.25° × 1 小时。

### 2. 训练目标

```python
class WeatherLoss(nn.Module):
    """训练 AI 天气模型的损失。"""
    def forward(self, pred, target):
        # 1) MSE 损失
        mse_loss = F.mse_loss(pred, target)
        
        # 2) 加权（不同变量 / 不同高度）
        # 高空变量更重要
        # 极端天气权重更大
        weighted_loss = (pred - target) ** 2 * self.weights
        
        # 3) 梯度损失（保留空间结构）
        grad_pred = spatial_gradient(pred)
        grad_target = spatial_gradient(target)
        grad_loss = F.mse_loss(grad_pred, grad_target)
        
        return weighted_loss.mean() + 0.5 * grad_loss
```

### 3. 推理速度对比

```python
INFERENCE_PERFORMANCE = {
    "ECMWF IFS": "8 小时 / 1 次预报 / 数千 CPU",
    "GraphCast": "1 分钟 / 1 次预报 / 1 GPU",      # 480x 加速
    "Pangu-Weather": "1~2 分钟 / 1 次预报 / 1 GPU",
    "FourCastNet": "30 秒 / 1 次预报 / 1 GPU",
}
```

**意义**：可以在**几分钟内**给出全球 10 天预报——传统需要数小时。

## 四、性能对比

### 1. 10 天准确度

```python
ACCURACY_METRICS = {
    "Z500 (500hPa 位势高度)": {
        "ECMWF IFS": "baseline",
        "GraphCast": "+20% RMSE 改善",
        "Pangu-Weather": "+22% 改善",
    },
    "T850 (850hPa 温度)": {
        "ECMWF IFS": "baseline",
        "GraphCast": "+15% 改善",
        "Pangu-Weather": "+18% 改善",
    },
    "Surface pressure": {
        "ECMWF IFS": "baseline",
        "GraphCast": "+12% 改善",
    },
}
```

### 2. 极端天气

| 极端天气 | 传统 | AI 模型 | 提升 |
|---|---|---|---|
| **台风路径** | 较好 | **Pangu 更准** | 显著 |
| **极端高温** | 偏差较大 | 显著改善 | 显著 |
| **强降水** | 难 | 改善但仍有限 | 中等 |
| **龙卷** | 极难 | 仍难 | 有限 |

### 3. 长期预测（>10 天）

```text
AI 模型优势:
- 前 5 天：显著优于传统
- 5~10 天：略优于传统
- >10 天：与 ECMWF 相当或略好
```

### 4. 数据同化（DA）

**GenCast（DeepMind, 2024）**——把数据同化也用 AI 做：

```python
class GenCast:
    """生成式 AI 集合预报。"""
    def __init__(self):
        self.diffusion = DiffusionModel(...)
    
    def forecast_ensemble(self, initial_state, n_samples=50):
        """生成 50 个可能未来——做集合预报。"""
        return [self.diffusion.sample(initial_state) for _ in range(n_samples)]
```

**优势**：集合预报**天然集成**——传统 NWP 需要 50 次模拟，AI 一次生成 50 个样本。

## 五、与其他领域的交叉

### 1. 能源

```python
class EnergyForecasting:
    """能源天气预报。"""
    def predict_solar(self, location, hours):
        # 云量直接影响太阳能发电
        cloud_forecast = ai_weather_model.predict_clouds(location, hours)
        solar_yield = self.solar_model.predict(cloud_forecast)
        return solar_yield
    
    def predict_wind(self, location, hours):
        # 风速直接影响风电
        wind_forecast = ai_weather_model.predict_wind(location, hours)
        return self.wind_model.predict(wind_forecast)
```

**意义**：可再生能源（风电、光伏）发电预测——电网调度依赖。

### 2. 农业

```python
class AgricultureWeather:
    """农业气象预报。"""
    def predict_growing_season(self, region):
        # 降水、温度、霜冻预测
        forecast = ai_weather_model.predict(region, days=120)
        
        # 灌溉决策
        irrigation_schedule = self.optimize_irrigation(forecast)
        
        # 收获时间
        harvest_time = self.predict_harvest(forecast)
        
        return {
            "irrigation": irrigation_schedule,
            "harvest": harvest_time,
            "frost_warning": forecast.min_temp < 0,
        }
```

### 3. 灾害预警

```python
class DisasterWarning:
    """灾害预警。"""
    def predict_cyclone(self, region):
        # 台风路径 + 强度
        forecast = ai_weather_model.predict(region, hours=120)
        cyclone = self.extract_cyclone(forecast)
        
        # 发布预警
        if cyclone.wind_speed > 119:  # km/h
            return "RED_ALERT"
    
    def predict_flood(self, basin):
        # 降水 + 河流模型
        rainfall = ai_weather_model.predict_precip(basin, days=7)
        flood_risk = self.hydrology_model(rainfall)
        
        if flood_risk > 0.7:
            return "FLOOD_WARNING"
```

### 4. 航空

```python
class AviationWeather:
    """航空气象。"""
    def predict_turbulence(self, route):
        # 高空风场 + 湍流指数
        forecast = ai_weather_model.predict_high_alt(route)
        turbulence = self.compute_turbulence_index(forecast)
        return turbulence
    
    def plan_fuel(self, route):
        # 顶风/顺风影响油耗
        winds = ai_weather_model.predict_winds_along_route(route)
        fuel_needed = self.compute_fuel(route, winds)
        return fuel_needed
```

## 六、模型的局限

### 1. 物理一致性

AI 模型**不保证物理守恒**：

```python
ISSUES = {
    "energy_conservation": "可能不守恒",
    "mass_conservation": "可能不守恒",
    "realistic_extremes": "极端值可能不物理",
    "out_of_distribution": "未见过的气候模式可能出错",
}
```

**缓解**：
- 物理损失函数。
- 后处理校准。
- 与 NWP 集合结合。

### 2. 气候变化

```text
❌ 担忧: AI 模型在训练分布外表现差
   - 训练用 1979~2021 数据
   - 2024 气候可能不在分布内
   - 极端气候事件可能预测失败

✅ 实测: 多数模型对气候变化仍能给出"持续趋势"
   - 但极端事件尾部可能不准
```

### 3. 数据偏差

```python
DATA_BIASES = {
    "era5_reanalysis": "依赖 ECMWF 同化数据",
    "satellite": "海洋、极地观测稀疏",
    "radiosonde": "集中在北半球发达国家",
    "urban_bias": "城市观测密集，乡村稀疏",
}
```

### 4. 黑盒问题

```text
传统数值预报:
  可解释（每个方程都有物理意义）

AI 模型:
  黑盒——为什么这个预测？
  不能直接归因
```

## 七、与传统数值预报的融合

### 1. Hybrid 模式

```python
class HybridNWP:
    """混合数值 + AI 模型。"""
    def forecast(self, initial_state, hours):
        # 1) 用 NWP 做初值
        nwp_state = nwp_model.run(initial_state, hours=3)
        
        # 2) 用 AI 修正
        ai_correction = ai_model.correct(nwp_state)
        return nwp_state + ai_correction
```

代表：**ECMWF AIFS**（2024）——用 GraphCast 替换部分 NWP 模块。

### 2. AI 替代

```text
完全替代 NWP:
- 短期（< 10 天）：已可行
- 长期：仍需 NWP
- 数据同化：AI 仍在追赶

AI 辅助 NWP:
- 参数化（辐射、对流）加速
- 后处理校准
- 极端事件预报
```

### 3. 数据同化

```python
class AIDataAssimilation:
    """AI 数据同化。"""
    def assimilate(self, observations, background):
        # 1) 估计最优初值
        analysis_state = self.da_model(observations, background)
        return analysis_state
```

代表：**GenCast**、**GraphDOP**。

## 八、商业产品

### 1. 大型 AI 天气预报服务

- **Google Weather**：基于 GraphCast + MetNet。
- **华为云天气**：基于 Pangu-Weather。
- **Nvidia Earth-2**：基于 FourCastNet。
- **DeepMind**：GraphCast + GenCast。

### 2. 行业应用

```python
INDUSTRY_AI_WEATHER = {
    "energy": ["Tomorrow.io", "DTN", "IBM Environmental Intelligence"],
    "agriculture": ["aWhere", "Climate FieldView", "Taranis"],
    "logistics": ["FedEx SenseAware", "UPS Weather"],
    "insurance": ["Zurich Climate Risk", "Aon"],
    "retail": ["Walmart AI weather"],  # 经典案例：根据天气进货
}
```

### 3. 政府气象机构

- **ECMWF**：与 Google 合作，把 GraphCast 集成到 Coperniqus。
- **NOAA**：评估 AI 模型。
- **中国气象局**：Pangu 在业务中试用。
- **英国 Met Office**：GraphDOP 测试。

## 九、技术挑战与未来

### 1. 多尺度统一

```text
挑战: 
- 全球模型分辨率（0.25°）vs 区域高分辨率（3km）
- 短期（小时）vs 中期（10 天）vs 长期（季-年）
- 对流尺度 vs 行星尺度

未来: 多尺度统一模型
```

### 2. 与气候模型结合

```python
class ClimateModel:
    """气候预测模型。"""
    def predict_climate(self, years=80):
        # 当前 AI 模型多在 10 天预测
        # 气候需要月-年-十年尺度
        # 是重大挑战
```

### 3. 不确定性量化

```python
class EnsembleForecast:
    """AI 集合预报。"""
    def forecast_with_uncertainty(self, initial_state, n_samples=100):
        # 多次预测 + 统计
        samples = [self.forecast(initial_state) for _ in range(n_samples)]
        
        return {
            "mean": np.mean(samples),
            "std": np.std(samples),    # 不确定性
            "p10": np.percentile(samples, 10),  # 较冷结果
            "p90": np.percentile(samples, 90),  # 较暖结果
        }
```

代表：**GenCast**（DeepMind）原生输出集合预报。

### 4. 极端天气专用

```python
class ExtremeWeatherModel:
    """极端天气专用模型。"""
    def predict_tropical_cyclone(self, current_state):
        # 专门针对台风/飓风的精细模型
        ...
```

代表：**Pangu-Weather 极端版**、**NowcastNet**（短临强对流）。

### 5. 与地球系统耦合

```text
未来: 大气 + 海洋 + 冰冻圈 + 碳循环 + 生态系统
     全耦合 AI 模型
```

代表：**AI-GOMS**、**Nvidia Earth-2** 长期愿景。

## 十、对科研和行业的启示

### 1. 对地球科学

```text
传统:
  数值预报 → 大气动力学 → 物理理解
  
AI 时代:
  数据驱动 → 大气统计模式 → 隐式物理理解
```

**挑战**：AI 是否能反过来帮助理解大气物理？这是开放问题。

### 2. 对能源行业

```python
def optimize_energy_grid(weather_forecast, demand_forecast):
    """优化电网调度。"""
    # 风电 / 光伏预测
    renewable = predict_renewable(weather_forecast)
    
    # 化石能源调度
    fossil = demand_forecast - renewable
    
    # 储能
    battery = optimize_battery(renewable, demand)
    
    return {
        "renewable": renewable,
        "fossil": fossil,
        "battery": battery,
    }
```

### 3. 对农业

- **精准农业**：AI 天气 + IoT + 农业模型。
- **气候适应**：选种、灌溉、收获时间。
- **保险**：天气衍生品、作物保险。

## 十一、给 AI 天气团队的清单

1. **使用 ERA5 数据**：标准训练数据。
2. **多模型集成**：GraphCast + Pangu + ECMWF 集成。
3. **物理一致性损失**：在训练中加入守恒约束。
4. **极端事件专项评估**：台风、寒潮、热浪。
5. **气候漂移监控**：持续评估模型在新气候下的表现。
6. **不确定性量化**：用集合预报。
7. **下游应用**：与具体业务（能源、农业）结合。
8. **开源贡献**：模型权重公开（GraphCast、Pangu、FourCastNet 都已开源）。
9. **跨学科**：AI + 大气科学 + 地球系统。
10. **伦理**：避免"控制天气"等滥用风险。

## 小结

AI 天气建模是 AI for Science 的**又一里程碑**——GraphCast、Pangu-Weather、FourCastNet 在 2022~2023 年几乎同时超越传统 ECMWF IFS，10 天预测准确度提升 10~25%，计算速度提升 100~1000 倍。**核心创新**是 GNN、3D Transformer、Fourier Neural Operator 在 ERA5 数据上的训练；**未来方向**是多尺度统一、长期气候预测、不确定性量化、极端事件预报、地球系统耦合。**应用价值**已经从科研扩展到能源、农业、灾害预警、航空等具体行业。**真正成功的 AI 天气模型**不是替代 NWP，而是与之融合，**加速物理理解 + 提供新能力**。三篇文章覆盖了 ai-for-science 的核心：AlphaFold、科学 LLM、AI 天气。下一篇（也是最后一篇）我们将转向 **robotics**：AI 在机器人中的应用。
