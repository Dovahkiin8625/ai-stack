# 算法交易与量化研究：AI 在金融市场的自动化决策

**算法交易（Algorithmic Trading）** 是金融 AI 中**利润最高、监管最严、风险最隐蔽**的应用——用数学模型和 AI 系统自动做出买卖决策。从文艺复兴的 Jim Simons 到今天的 Two Sigma、DE Shaw、Citadel，再到人人可用的开源工具（Zipline、Backtrader、QuantConnect），量化交易已经走过三个时代。本文梳理算法交易的核心策略、LLM 在量化研究中的新角色、风控要点，以及零售可用的入门路径。

## 一、算法交易的历史与现状

### 时代 1：传统统计套利（1980~2000）

- **统计套利**：基于历史相关性的配对交易。
- **工具**：Excel、Matlab、早期 Python。
- **代表**：长期资本管理公司（LTCM, 1998 年崩盘）。

### 时代 2：机器学习量化（2000~2015）

- **特征工程**：技术指标、基本面、宏观。
- **模型**：随机森林、GBDT、SVM。
- **代表**：Two Sigma、DE Shaw、Renaissance Technologies。

### 时代 3：深度学习与 LLM 时代（2015~）

- **深度学习**：CNN（K线图）、RNN（时序）、Transformer（多资产）。
- **另类数据**：卫星图像、信用卡、社交媒体。
- **LLM**：研报解析、新闻情感、自动化研究。
- **代表**：多数顶级对冲基金都在用，但具体策略保密。

### 市场规模

- 全球算法交易占交易量 **60~75%**（美国股市）。
- 美国高频交易公司每天处理 **数十亿股**。
- 全球量化基金 AUM 估计 **$1.5T+**。

## 二、核心交易策略类型

### 1. 趋势跟随（Trend Following）

```python
def trend_following_strategy(prices: pd.Series, fast_window=20, slow_window=50):
    """经典双均线趋势策略。"""
    fast_ma = prices.rolling(fast_window).mean()
    slow_ma = prices.rolling(slow_window).mean()
    
    signals = pd.Series(0, index=prices.index)
    signals[fast_ma > slow_ma] = 1    # 金叉 → 买入
    signals[fast_ma < slow_ma] = -1   # 死叉 → 卖出
    
    return signals
```

**核心思想**：价格有动量——涨势会持续一段时间。

### 2. 均值回归（Mean Reversion）

```python
def mean_reversion_strategy(prices: pd.Series, lookback=20, threshold=2):
    """Z-score 均值回归。"""
    ma = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    z = (prices - ma) / std
    
    signals = pd.Series(0, index=prices.index)
    signals[z < -threshold] = 1   # 偏低 → 买入
    signals[z > threshold] = -1  # 偏高 → 卖出
    
    return signals
```

**核心思想**：价格偏离均值后会回归。

### 3. 配对交易（Pairs Trading）

```python
def pairs_trading_strategy(prices_a, prices_b, lookback=60, threshold=2):
    """配对交易：基于价差统计套利。"""
    spread = prices_a - prices_b
    z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    
    signals = pd.Series(0, index=spread.index)
    signals[z < -threshold] = 1   # spread 低 → 做多 A 做空 B
    signals[z > threshold] = -1  # spread 高 → 做空 A 做多 B
    
    return signals
```

**核心思想**：两只相关性强的股票，价差长期稳定。

### 4. 统计套利（Statistical Arbitrage）

```python
def stat_arb_signals(returns: pd.DataFrame, residual_window=60):
    """
    残差统计套利：用因子模型拟合后做残差交易。
    """
    # 1) 估计因子载荷（市场、规模、价值）
    factors = pd.DataFrame({
        "market": get_market_returns(),
        "size": get_size_factor(),
        "value": get_value_factor(),
    })
    
    # 2) 对每只股票拟合因子模型
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
    for stock in returns.columns:
        betas = np.linalg.lstsq(factors, returns[stock], rcond=None)[0]
        residuals[stock] = returns[stock] - factors @ betas
    
    # 3) 残差 z-score 排序，做空残差高的、做多残差低的
    z_scores = (residuals - residuals.mean()) / residuals.std()
    
    # 排名靠前 20% 做多，排名靠后 20% 做空
    longs = z_scores.rank(axis=1, ascending=False) <= len(z_scores.columns) * 0.2
    shorts = z_scores.rank(axis=1, ascending=False) > len(z_scores.columns) * 0.8
    
    signals = longs.astype(int) - shorts.astype(int)
    return signals
```

### 5. 高频交易（HFT）

```python
# HFT 通常在 FPGA / C++ 上实现
# 这里只是示意
class HFTMarketMaker:
    """做市商策略。"""
    def __init__(self):
        self.inventory = 0
        self.pnl = 0
    
    def quote(self, mid_price, volatility, inventory):
        """根据市场状态和市场库存出报价。"""
        spread = max(0.01, volatility * 0.5)  # 价差
        
        # 库存偏多 → 倾向卖
        skew = -inventory * 0.01
        
        bid = mid_price - spread/2 + skew
        ask = mid_price + spread/2 + skew
        return bid, ask
```

### 6. 强化学习策略（前沿）

```python
class TradingEnv:
    """交易强化学习环境。"""
    def __init__(self, prices, initial_cash=10000):
        self.prices = prices
        self.initial_cash = initial_cash
        self.reset()
    
    def reset(self):
        self.cash = self.initial_cash
        self.position = 0
        self.step_idx = 0
        return self._get_state()
    
    def _get_state(self):
        # 状态：价格窗口 + 持仓 + 现金
        window = self.prices[self.step_idx:self.step_idx+20]
        return np.concatenate([
            window,
            [self.position, self.cash / self.initial_cash],
        ])
    
    def step(self, action):
        """
        action: -1 (sell), 0 (hold), 1 (buy)
        """
        price = self.prices[self.step_idx]
        if action == 1 and self.cash >= price:
            self.position += 1
            self.cash -= price
        elif action == -1 and self.position > 0:
            self.position -= 1
            self.cash += price
        
        self.step_idx += 1
        new_price = self.prices[self.step_idx]
        
        # 奖励：净值变化
        portfolio_value = self.cash + self.position * new_price
        reward = (portfolio_value - self.initial_cash) / self.initial_cash
        
        done = self.step_idx >= len(self.prices) - 1
        return self._get_state(), reward, done, {}
```

```python
# PPO 算法训练
from stable_baselines3 import PPO

env = TradingEnv(prices)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## 三、LLM 在量化研究中的角色

### 1. 研报情感分析

```python
def research_sentiment(ticker: str, days=7) -> dict:
    """分析最近研报对股票的情感。"""
    reports = retriever.retrieve(
        query=f"{ticker} earnings outlook",
        filter={"type": "research_report", "days_back": days},
        k=20,
    )
    
    sentiment = llm.generate(f"""分析以下 {len(reports)} 份研报对 {ticker} 的整体情感：

{reports}

输出：
- 整体情感（看多 / 中性 / 看空）
- 信心程度 (0-1)
- 关键看多逻辑（3 条）
- 关键看空逻辑（3 条）
- 共识与分歧""")
    return sentiment
```

### 2. 自动化 alpha 因子挖掘

```python
def auto_alpha_mining(universe: list, n_iterations=100) -> list:
    """用 LLM 自动生成 alpha 因子。"""
    alphas = []
    for i in range(n_iterations):
        # 让 LLM 生成一个候选 alpha 表达式
        expr = llm.generate(f"""基于以下数据，生成一个量化 alpha 因子表达式：

可用数据: open, high, low, close, volume, vwap

要求：
- 输出 Python 表达式
- 简洁（不超过 3 行）
- 符合市场直觉
- 表达式:""")
        
        try:
            # 评估 IC（信息系数）
            alpha_signal = eval_alpha(expr, universe)
            ic = alpha_signal.corr(future_returns)
            if abs(ic) > 0.02:
                alphas.append({"expression": expr, "ic": ic})
        except Exception as e:
            continue
    
    return sorted(alphas, key=lambda x: abs(x["ic"]), reverse=True)
```

### 3. 财报电话会议分析

```python
def earnings_call_analysis(ticker: str) -> dict:
    """分析财报电话会议文本。"""
    transcript = get_transcript(ticker)
    
    return llm.generate(f"""分析 {ticker} 最新财报电话会议：

{transcript}

提取：
1. 管理层语调（信心、谨慎、乐观）
2. 与上季度相比的关键变化
3. 提及的具体风险
4. 分析师关注的关键问题
5. 隐含的业绩指引

输出结构化 JSON。""")
```

### 4. 自动化研究助理

```python
def quant_research_agent(hypothesis: str) -> dict:
    """自动化研究：从假设到回测。"""
    # 1) LLM 拆解假设
    factors = llm.generate(f"把以下交易假设拆解为可测试的因子：{hypothesis}")
    
    # 2) 自动写代码
    code = llm.generate(f"""根据以下因子写回测代码：

{factors}

用 Backtrader 框架。""")
    
    # 3) 执行回测
    try:
        result = execute_backtest(code)
        return {
            "hypothesis": hypothesis,
            "code": code,
            "sharpe": result["sharpe"],
            "returns": result["returns"],
        }
    except Exception as e:
        return {"error": str(e)}
```

## 四、回测框架

### 1. Backtrader 入门

```python
import backtrader as bt


class SMAStrategy(bt.Strategy):
    """双均线策略。"""
    params = (
        ("fast", 10),
        ("slow", 30),
    )
    
    def __init__(self):
        self.fast_ma = bt.indicators.SMA(period=self.p.fast)
        self.slow_ma = bt.indicators.SMA(period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
    
    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.sell()


# 运行回测
cerebro = bt.Cerebro()
cerebro.addstrategy(SMAStrategy)

data = bt.feeds.YahooFinanceData(dataname="AAPL", fromdate=..., todate=...)
cerebro.adddata(data)
cerebro.broker.set_cash(100000)
cerebro.broker.setcommission(commission=0.001)

results = cerebro.run()
cerebro.plot()
```

### 2. 重要回测陷阱

#### 前视偏差（Look-ahead Bias）

```python
# ❌ 错误：用了未来信息
def bad_strategy(prices):
    # 用了当天的 close 决定当天的 trade（不可能）
    return prices.pct_change().apply(lambda x: 1 if x > 0 else -1)

# ✅ 正确：用当天开盘价或前一天收盘价
def good_strategy(prices):
    # signal 在 close 时计算，next day open 时执行
    signal = prices.shift(1).pct_change().apply(lambda x: 1 if x > 0 else -1)
    return signal
```

#### 生存者偏差

```python
# ❌ 错误：只回测当前存在的股票（高估收益）
def load_prices_wrong():
    return yfinance.download("AAPL MSFT GOOGL")  # 都是赢家

# ✅ 正确：包括已退市的股票
def load_prices_correct():
    return load_survivorship_bias_free("SP500_2010_2024")
```

#### 过拟合

```python
# ❌ 错误：在测试集上反复调参
for param in params:
    if backtest(param)["sharpe"] > 0.5:
        best_param = param  # 多次检验后选最好 → 过拟合

# ✅ 正确：用样本外测试
train_period = "2010-2018"
test_period = "2018-2024"
# 只在 train 上调参，在 test 上报告
```

### 3. 关键回测指标

```python
def backtest_metrics(returns: pd.Series) -> dict:
    """计算回测关键指标。"""
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # 最大回撤
    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }
```

## 五、风控系统

### 1. 仓位管理

```python
class RiskManager:
    """交易风控。"""
    def __init__(self, max_position=0.1, max_drawdown=0.2, max_leverage=2):
        self.max_position = max_position  # 单仓位上限 10%
        self.max_drawdown = max_drawdown  # 最大回撤 20%
        self.max_leverage = max_leverage
    
    def check_trade(self, signal, current_positions, portfolio_value):
        """检查交易是否符合风控。"""
        # 1) 仓位上限
        proposed_position = signal["size"] * portfolio_value
        if abs(proposed_position) > self.max_position * portfolio_value:
            return False, "Position size exceeds limit"
        
        # 2) 杠杆上限
        total_exposure = sum(abs(p) for p in current_positions.values())
        if total_exposure > self.max_leverage * portfolio_value:
            return False, "Leverage exceeds limit"
        
        return True, "OK"
    
    def check_drawdown(self, portfolio_history):
        """检查回撤是否超限。"""
        peak = portfolio_history.cummax()
        drawdown = (portfolio_history - peak) / peak
        if drawdown.min() < -self.max_drawdown:
            return False, f"Drawdown {drawdown.min():.2%} exceeds {self.max_drawdown:.2%}"
        return True, "OK"
```

### 2. 实时监控

```python
def real_time_monitor():
    """实时交易监控。"""
    while True:
        # 1) PnL 监控
        pnl = get_portfolio_pnl()
        if pnl < -0.05 * initial_capital:  # 日亏 5%
            alert("DAILY_LOSS_LIMIT", pnl)
        
        # 2) 持仓监控
        positions = get_positions()
        for ticker, position in positions.items():
            if position.risk > max_risk:
                alert("SINGLE_POSITION_RISK", ticker)
        
        # 3) 系统监控
        if not is_market_open():
            alert("MARKET_CLOSED", "系统仍在运行")
        
        time.sleep(60)  # 每分钟检查
```

### 3. 模型监控

```python
def monitor_strategy_drift():
    """监控策略表现漂移。"""
    recent_sharpe = calculate_sharpe(recent_30_days)
    expected_sharpe = 1.5
    
    if recent_sharpe < 0.5 * expected_sharpe:
        alert("STRATEGY_UNDERPERFORMING", recent_sharpe)
    
    # 监控 alpha 衰减
    recent_ic = calculate_ic(recent_data)
    if recent_ic < 0.5 * historical_ic:
        alert("ALPHA_DECAY", recent_ic)
```

## 六、关键风险

### 1. 模型风险

- **过拟合**：在历史数据上找到的规律在未来失效。
- **概念漂移**：市场结构变化导致模型失效（如 2020 年疫情）。
- **黑天鹅**：极端事件下所有模型同时失效（LTCM 1998, 长期资本）。

### 2. 系统风险

- **技术故障**：服务器宕机、网络中断。
- **数据错误**：行情延迟、价格错误。
- **流动性风险**：无法按预期价格成交。

### 3. 监管风险

- **市场操纵**：算法合谋（spoofing）、幌骗（layering）。
- **内幕交易**：AI 可能无意中利用内幕信息。
- **系统重要性**：算法故障可能引发市场动荡（2010 Flash Crash）。

### 4. 运营风险

- **回测与实盘差距**：滑点、手续费、冲击成本。
- **心理压力**：连续亏损时的纪律性。

## 七、实盘部署

### 1. 经纪商 API

```python
import alpaca_trade_api as tradeapi

api = tradeapi.REST(
    key_id="YOUR_API_KEY",
    secret_key="YOUR_SECRET",
    base_url="https://paper-api.alpaca.markets",
)


def execute_trade(symbol, qty, side):
    """执行交易。"""
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,  # 'buy' or 'sell'
        type='market',
        time_in_force='day',
    )
```

### 2. 滑点与冲击

```python
def estimate_slippage(symbol, qty, side):
    """估算滑点。"""
    orderbook = get_orderbook(symbol)
    if side == "buy":
        # 买 → 消耗 ask 侧
        return estimate_impact(orderbook['asks'], qty)
    else:
        return estimate_impact(orderbook['bids'], qty)
```

### 3. 监控告警

```python
import logging

def setup_alerting():
    """设置告警系统。"""
    logging.basicConfig(level=logging.INFO)
    
    # Slack 告警
    from slack_sdk import WebClient
    slack = WebClient(token=os.environ["SLACK_TOKEN"])
    
    def send_alert(channel, message):
        slack.chat_postMessage(channel=channel, text=message)
    
    return send_alert
```

## 八、合规与监管

### 1. 美国监管

- **SEC**（Securities and Exchange Commission）。
- **FINRA**（金融业监管局）。
- **CFTC**（商品期货交易委员会）。

### 2. 关键规则

- **Reg NMS**：保护投资者获得最优价格。
- **Reg SHO**：裸卖空限制。
- **Reg SCI**：系统合规与完整性。

### 3. 算法交易合规要求

```python
def compliance_check(signal):
    """交易前的合规检查。"""
    # 1) 不在禁止交易的股票清单中
    if signal["symbol"] in restricted_list:
        return False
    
    # 2) 不在冷静期内（重大事件）
    if in_quiet_period(signal["symbol"]):
        return False
    
    # 3) 符合仓位限制
    if signal["size"] > max_position_limit:
        return False
    
    # 4) 不是幌骗行为
    if is_layering_signal(signal):
        return False
    
    return True
```

## 九、零售入门路径

### 1. 学习路线

```text
Level 1: 基础
  - Python 编程
  - Pandas / NumPy
  - 基础金融知识

Level 2: 量化基础
  - 统计学基础
  - 时间序列分析
  - Backtrader / Zipline

Level 3: 机器学习
  - scikit-learn
  - 特征工程
  - 模型评估

Level 4: 高级
  - 深度学习（PyTorch）
  - 自然语言处理（LLM）
  - 另类数据

Level 5: 实盘
  - 经纪商 API
  - 风控系统
  - 监控告警
```

### 2. 推荐工具

```python
# 数据
PANDAS_DATAREADER, YFINANCE, TUSHARE, AKSHARE

# 回测
BACKTRADER, ZIPLINE-RELOADED, QUANTCONNECT, LEAN

# 实盘
ALPACA, INTERACTIVE_BROKERS, BINANCE, CCXT

# LLM 应用
LANGCHAIN, LLAMAINDEX, AUTOGEN
```

### 3. 重要警告

**多数零售量化交易者亏损**：

- 缺乏机构级数据和基础设施。
- 交易成本高（小额账户）。
- 心理纪律差。
- 模型过拟合普遍。

**不要用输不起的钱做交易**。

## 十、未来方向

### 1. LLM-驱动的研究自动化

```text
传统量化研究: 
  研究员提假设 → 写代码 → 跑回测 → 分析 → 迭代（数周）

LLM 驱动:
  研究员提假设 → LLM 写代码 → LLM 跑回测 → LLM 分析 → LLM 迭代
```

### 2. 多智能体交易系统

```python
# 多个 Agent 协作
data_agent = Agent("拉取并清洗市场数据")
research_agent = Agent("分析研报、新闻")
strategy_agent = Agent("生成交易信号")
risk_agent = Agent("风控检查")
execution_agent = Agent("执行交易")
```

### 3. 实时另类数据整合

- 卫星图像 + 财报 + 信用卡 → 实时销售预测。
- 社交媒体 + 新闻 + 政策 → 实时行业情绪。

### 4. 监管技术（RegTech）

AI 自动监控交易合规、检测异常行为。

## 十一、给量化团队的清单

1. **数据合规**：只用合规数据源，避免 MNPI。
2. **回测严谨**：避免前视偏差、生存者偏差、过拟合。
3. **样本外测试**：把数据严格分 train/test/out-of-sample。
4. **风控优先**：仓位、杠杆、回撤多重限制。
5. **实时监控**：PnL、持仓、系统、模型表现。
6. **灾备方案**：系统故障时人工接管。
7. **成本管理**：滑点、手续费、冲击成本精细建模。
8. **压力测试**：极端市场下的策略表现。
9. **团队纪律**：策略、代码、决策可追溯。
10. **持续学习**：策略定期回顾、迭代、淘汰。

## 小结

算法交易是金融 AI 的最高形态——**用数学和代码自动化决策**。从趋势跟随、均值回归、统计套利到强化学习、LLM 驱动的研究自动化，量化策略不断演化。**核心挑战是模型风险、系统风险、监管风险、运营风险**——技术只是门槛，风控才是护城河。**真正成功的量化基金** 不是"找到圣杯 alpha"，而是**持续迭代、风控严谨、成本可控、纪律严明**。三篇文章覆盖了 ai-for-finance 的核心：LLM 应用、金融 RAG、算法交易。下一篇我们将转向 **ai-for-healthcare**：AI 在医疗中的应用与挑战。
