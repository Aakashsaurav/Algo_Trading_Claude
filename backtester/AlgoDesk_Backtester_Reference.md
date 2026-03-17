**AlgoDesk**

Backtesting Engine

*Technical Reference & User Guide*

Version 2.0  ·  March 2026  ·  NSE / Upstox


# **1. Overview**
AlgoDesk is a fully custom, production-grade algorithmic trading and backtesting platform designed for Indian equity markets (NSE). This document describes the backtesting engine — the subsystem responsible for simulating the performance of any trading strategy on historical OHLCV data.

The engine is built from scratch with no dependency on Backtrader, Zipline, or any third-party backtesting framework. Every component — from order fill simulation to commission modelling to performance analytics — is purpose-built and independently testable.

## **1.1 Design Philosophy**
The design mirrors the architectural principles of Backtrader: strategy logic is completely decoupled from execution, data, and reporting. The same strategy class that runs in a backtest runs identically in live paper trading — no code changes required.

|**Principle**|**Implementation**|
| :- | :- |
|No look-ahead bias|Signal on bar i executes at bar i+1's open. The event loop reads signals[i-1] during bar i — structurally impossible to access future data.|
|Realistic costs|All 7 layers of Indian market charges are applied on every trade: brokerage, STT, exchange fee, SEBI fee, GST, stamp duty, DP charges.|
|Single responsibility|Each module has exactly one job. FillEngine fills orders. PositionSizer sizes positions. EventLoop iterates bars. Performance computes metrics.|
|No shared mutable state|FillEngine is stateless. Position is a dataclass. Trade is a dataclass. The event loop owns the mutable lists.|
|Vectorised where possible|ATR pre-computation, equity curve building, and candlestick rendering all use NumPy arrays — never per-bar Python loops on series.|
|Testable at every level|125 unit and integration tests cover every module independently. No test requires a running server, broker, or file system.|

## **1.2 Architecture Overview**
The backtesting subsystem is organised into the following modules:

|**Module / File**|**Responsibility**|
| :- | :- |
|backtester/engine.py|Public API. The only file users import. BacktestEngine.run() and run\_portfolio(). ~80 lines.|
|backtester/models.py|Single source of truth for all data structures: BacktestConfig, Position, Trade, BacktestResult, OrderType.|
|backtester/event\_loop.py|The bar-by-bar simulation loop. Coordinates fill\_engine, order checking, and equity recording. ~200 lines.|
|backtester/fill\_engine.py|All order fill and stop-loss logic. open\_position(), close\_position(), check\_stops(), check\_pending\_entries().|
|backtester/position\_sizer.py|Position sizing: fixed quantity or risk-based (ATR or fixed %). Pure function — no side effects.|
|backtester/order\_types.py|PendingOrder dataclass and pure fill-check functions for LIMIT, STOP, and STOP-LIMIT orders.|
|backtester/performance.py|25+ performance metrics: CAGR, Sharpe, Sortino, Calmar, Omega, Kelly, monthly/annual returns, exposure %.|
|backtester/portfolio.py|Portfolio tracker: cash balance, open positions, equity curve, drawdown, max-drawdown halt.|
|backtester/optimizer.py|Grid and random parameter search. Walk-forward optimisation. ProcessPoolExecutor parallelism.|
|backtester/report.py|Streak-style PNG chart: vectorised candlesticks, indicator overlays, trade markers, equity/drawdown panels.|
|broker/upstox/commission.py|LOCKED. All 7 Indian market charge layers: exact Upstox brokerage structure as of 2025.|


# **2. Quick Start**
## **2.1 Minimal Backtest**
The following example runs a backtest with default settings in four lines:

from backtester import BacktestEngine, BacktestConfig

from strategies.base import EMACrossover



cfg     = BacktestConfig(initial\_capital=500\_000)

engine  = BacktestEngine(cfg)

strategy = EMACrossover(fast\_period=9, slow\_period=21)



result = engine.run(df, strategy, symbol='INFY')

print(result.summary())

## **2.2 Full Production Configuration**
A more complete configuration enabling all output flags and advanced order types:

from backtester import BacktestEngine, BacktestConfig, OrderType

from broker.upstox.commission import Segment



cfg = BacktestConfig(

`    `initial\_capital     = 1\_000\_000,

`    `segment             = Segment.EQUITY\_INTRADAY,

`    `default\_order\_type  = OrderType.LIMIT,

`    `limit\_offset\_pct    = 0.2,          # 0.2% below close for buy limit

`    `stop\_loss\_pct       = 1.5,          # 1.5% fixed stop on all trades

`    `use\_trailing\_stop   = True,

`    `trailing\_stop\_pct   = 2.0,          # 2% trailing stop

`    `capital\_risk\_pct    = 0.015,        # risk 1.5% of capital per trade

`    `max\_positions       = 5,

`    `max\_drawdown\_pct    = 0.15,         # halt at 15% drawdown

`    `allow\_shorting      = False,

`    `intraday\_squareoff  = True,

`    `save\_trade\_log      = True,

`    `save\_chart          = True,

`    `save\_raw\_data       = True,

`    `generate\_summary    = True,

`    `run\_label           = 'ema\_infy\_q1\_2024',

)



engine  = BacktestEngine(cfg)

result  = engine.run(df, strategy, symbol='INFY')

## **2.3 Portfolio Backtest**
Run the same strategy across multiple symbols simultaneously:

data = {

`    `'INFY':      df\_infy,

`    `'TCS':       df\_tcs,

`    `'RELIANCE':  df\_reliance,

}



results = engine.run\_portfolio(data, strategy, label='nifty\_it\_q1')



for symbol, result in results.items():

`    `m = result.metrics()

`    `print(f'{symbol}: CAGR={m["cagr\_pct"]:.2f}%  Sharpe={m["sharpe\_ratio"]:.3f}')

*ℹ  run\_portfolio() writes separate output files per symbol — it never concatenates all symbols into one large DataFrame in memory. This makes 50-symbol runs safe.*


# **3. BacktestConfig Reference**
BacktestConfig is a Python dataclass that controls every aspect of a backtest run. All parameters have sensible defaults — a minimal config requires only initial\_capital.

cfg = BacktestConfig(initial\_capital=500\_000)  # uses all defaults

## **3.1 Capital & Sizing Parameters**

|**Parameter**|**Type**|**Default**|**Description**|
| :- | :- | :- | :- |
|initial\_capital|float|500 000|Starting portfolio value in ₹.|
|capital\_risk\_pct|float|0\.02|Fraction of cash to risk per trade (2%). Used by position sizer when no fixed\_quantity.|
|fixed\_quantity|int|0|If > 0, trade this exact number of shares every time. Overrides capital\_risk\_pct.|
|max\_positions|int|0|Maximum simultaneous open positions. 0 = unlimited.|
|max\_drawdown\_pct|float|0\.20|Halt the backtest if equity falls more than this fraction below its running peak.|
|lot\_size|int|1|Contract lot size for F&O. Ignored for equities.|

## **3.2 Market Parameters**

|**Parameter**|**Type**|**Default**|**Description**|
| :- | :- | :- | :- |
|segment|Segment|EQUITY\_DELIVERY|Market segment for commission calculation. Values: EQUITY\_DELIVERY, EQUITY\_INTRADAY, EQUITY\_FUTURES, EQUITY\_OPTIONS, CURRENCY\_FUTURES, CURRENCY\_OPTIONS, COMMODITY\_FUTURES, COMMODITY\_OPTIONS.|
|allow\_shorting|bool|False|If False, a sell signal closes an existing long but does not open a short position.|
|intraday\_squareoff|bool|False|Force-close all positions at 15:20 IST each session. Requires a timezone-aware DatetimeIndex.|

## **3.3 Order Type Parameters**

|**Parameter**|**Type**|**Default**|**Description**|
| :- | :- | :- | :- |
|default\_order\_type|OrderType|MARKET|Entry order type. MARKET = next bar open. LIMIT = limit offset below/above signal close. STOP = stop entry on breakout. STOP\_LIMIT = stop trigger with limit protection. TRAILING\_STOP = dynamic trailing entry.|
|limit\_offset\_pct|float|0\.2|For LIMIT/STOP orders: % offset from signal-bar close to place the order. Default 0.2%.|
|stop\_loss\_pct|float|0\.0|Attach a fixed % stop-loss to every opened position. 0 = disabled.|
|stop\_loss\_atr\_mult|float|2\.0|ATR(14) multiplier for dynamic stop-loss distance. Used when stop\_loss\_pct is 0 and fixed\_quantity is 0.|
|use\_trailing\_stop|bool|False|Attach a trailing stop to every opened position.|
|trailing\_stop\_pct|float|0\.0|Trailing stop distance as % of price. Use with use\_trailing\_stop=True.|
|trailing\_stop\_amt|float|0\.0|Trailing stop distance as fixed ₹ amount. Alternative to trailing\_stop\_pct.|

## **3.4 Output Flags**

|**Parameter**|**Type**|**Default**|**Description**|
| :- | :- | :- | :- |
|save\_trade\_log|bool|False|Write trade log CSV to strategies/output/trade/.|
|save\_raw\_data|bool|False|Write OHLCV + indicator + signal CSV to strategies/output/raw\_data/.|
|save\_chart|bool|False|Generate a Streak-style PNG chart to strategies/output/chart/.|
|generate\_summary|bool|False|Write a JSON performance summary after the run.|
|run\_label|str|'backtest'|Prefix for all output filenames. Use to distinguish runs.|
|max\_candles|int|2000|Maximum candles rendered in the PNG chart.|

*⚠  use\_trailing\_stop=True requires exactly one of trailing\_stop\_pct or trailing\_stop\_amt to be set. Setting both raises ValueError at config.validate().*


# **4. Order Types**
The engine supports five order types, selected via BacktestConfig.default\_order\_type. All order types use intrabar OHLCV data for fill simulation — tick-level data is not required.

## **4.1 MARKET (default)**
Execute at the open of the bar immediately following the signal bar. This is the most conservative and most commonly used order type — it prevents fill-price optimism.

cfg = BacktestConfig(default\_order\_type=OrderType.MARKET)

\# Signal on bar[i] → fill at bar[i+1].open

## **4.2 LIMIT**
Place a limit order at limit\_offset\_pct below the signal close (for longs). Fills only if the bar's low touches the limit price. Models patient, better-than-market entries.

cfg = BacktestConfig(

`    `default\_order\_type = OrderType.LIMIT,

`    `limit\_offset\_pct   = 0.3,   # place limit 0.3% below signal close

)

\# If signal close = 1000, limit placed at 997

\# Fills if any subsequent bar's low <= 997

*ℹ  Gap handling: if the bar opens below the limit price, the fill occurs at open (better-than-limit fill), not at the limit price. This correctly models price gaps.*

## **4.3 STOP**
A buy stop triggers when price breaks above a threshold (momentum breakout). A sell stop triggers on breakdown below. Models breakout entry strategies.

cfg = BacktestConfig(

`    `default\_order\_type = OrderType.STOP,

`    `limit\_offset\_pct   = 0.5,   # stop placed 0.5% above signal close

)

## **4.4 STOP\_LIMIT**
Two-phase order: triggers at a stop price, then fills only if the limit price is still reachable. Prevents catastrophic fills in fast-moving markets. Risk: may never fill during violent price moves.

## **4.5 TRAILING\_STOP**
A dynamic stop-loss that follows price in the favourable direction. For longs: the stop level rises as price rises, but never falls back. For shorts: the stop falls as price falls. Locks in profits automatically.

cfg = BacktestConfig(

`    `use\_trailing\_stop = True,

`    `trailing\_stop\_pct = 2.0,    # trail 2% behind the running high

`    `# OR

`    `trailing\_stop\_amt = 50.0,   # trail ₹50 behind the running high

)

*⚠  Provide trailing\_stop\_pct OR trailing\_stop\_amt — not both. Setting both raises ValueError.*


# **5. Execution Model**
## **5.1 Event Loop Processing Order**
On every bar i, the event loop executes the following steps in strict priority order:

|**Step**|**Action**|
| :- | :- |
|1|Update trailing-stop levels for all open positions (advance stop if price moved favorably).|
|2|Check trailing stops — exit any position whose trailing stop level was breached. Fill at open\_p on gaps.|
|3|Check fixed stop-losses — exit any position whose fixed stop price was hit.|
|4|Check pending LIMIT / STOP / STOP-LIMIT entry orders — fill any that triggered this bar.|
|5|Intraday squareoff (if enabled) — force-close all positions at open\_p if time >= 15:20 IST.|
|6|Process signal from bar i-1 — open/close positions based on the previous bar's signal. Fill at current bar's open.|
|7|Record equity and drawdown for this bar using NumPy arrays.|
|8|Max drawdown guard — if equity fell below the configured threshold, close all positions and halt.|

*ℹ  Stops (steps 2-3) always execute before new signals (step 6). A stop-loss and a new entry signal on the same bar will result in the stop closing the existing position first, then the new entry being considered.*

## **5.2 No Look-Ahead Guarantee**
The engine structurally prevents look-ahead bias:

- Strategy.generate\_signals(df) is called once on the entire DataFrame before the loop starts.
- The event loop reads signals[i-1] during bar i — the signal computed from data available at bar i-1.
- All fills execute at bar i's open — the first price available after the signal bar.
- Position.update\_excursion() uses intrabar high/low, which are available within the bar being processed.

*⚠  Strategies must not use df['close'].shift(-1) or any future-date data in generate\_signals(). The engine cannot detect all forms of look-ahead bias in strategy code.*

## **5.3 Commission Model**
Every fill triggers an exact charge calculation through CommissionModel (broker/upstox/commission.py). Seven charge layers are applied on every order leg:

|**Charge**|**Description**|
| :- | :- |
|Brokerage|min(₹20, rate × trade\_value). Rate: 0.1% delivery, 0.05% intraday/futures. Options: flat ₹20.|
|STT / CTT|Securities/Commodity Transaction Tax. Rate and side (buy/sell) depend on segment.|
|Exchange Fee|NSE transaction charge. 0.00297% equity, 0.00173% futures, 0.03503% options premium.|
|SEBI Turnover Fee|₹10 per crore = 0.0000001 × trade\_value. Both sides.|
|GST|18% on (brokerage + exchange fee). Both sides.|
|Stamp Duty|Buy side only. 0.015% delivery, 0.0003% intraday, 0.0002% futures.|
|DP Charges|₹18.5 per scrip on equity delivery SELL only (demat debit).|

*ℹ  The commission model is locked (broker/upstox/commission.py). Rate changes should be made only in that file.*


# **6. Position Sizing**
Position sizing is handled by position\_sizer.py as a pure function — no side effects, no class state. Two modes are supported:

## **6.1 Fixed Quantity Mode**
Set fixed\_quantity > 0 in BacktestConfig to trade a fixed number of shares on every signal. The sizer will reduce quantity if cash is insufficient for the full amount.

cfg = BacktestConfig(fixed\_quantity=50)   # always buy/sell 50 shares

## **6.2 Risk-Based Sizing (Default)**
When fixed\_quantity = 0, the sizer computes quantity so that the maximum potential loss on the trade equals capital\_risk\_pct × available\_cash.

\# With a stop-loss configured:

risk\_per\_trade  = cash × capital\_risk\_pct       # e.g. 500000 × 0.02 = ₹10,000

stop\_distance   = abs(entry\_price - stop\_price)  # e.g. 1000 - 950 = ₹50

quantity        = floor(risk\_per\_trade / stop\_distance)  # = 200 shares



\# Without a stop-loss (fallback):

quantity        = floor(cash × 0.02 / entry\_price)

*ℹ  Quantity is always capped so the total trade cost never exceeds available cash. If the computed quantity is 0, no trade is placed.*

## **6.3 ATR-Based Stop for Sizing**
When stop\_loss\_pct = 0 but stop\_loss\_atr\_mult > 0, the sizer uses ATR(14) × multiplier as the stop distance. ATR is pre-computed once before the event loop using Wilder's smoothed formula.

cfg = BacktestConfig(

`    `capital\_risk\_pct   = 0.02,

`    `stop\_loss\_atr\_mult = 2.0,    # stop = entry - 2 × ATR(14)

)


# **7. Performance Metrics**
compute\_performance() returns a dict of 25+ metrics from the completed trade log and equity curve. All values are JSON-serialisable Python floats or ints.

from backtester.performance import compute\_performance



m = compute\_performance(trade\_log, equity\_curve, config)

\# or via BacktestResult:

m = result.metrics()   # cached — second call is O(1)

## **7.1 Return Metrics**

|**Metric Key**|**Description**|
| :- | :- |
|total\_return\_pct|Absolute percentage return over the backtest period. ((final / initial) - 1) × 100.|
|cagr\_pct|Compound Annual Growth Rate in %. Annualised using calendar days between first and last bar.|
|annualised\_volatility|Annualised standard deviation of daily equity returns (%). Computed from daily-resampled equity.|

## **7.2 Risk-Adjusted Metrics**

|**Metric Key**|**Description**|
| :- | :- |
|sharpe\_ratio|Excess return per unit of total volatility. Uses 6.5% India 10-yr G-Sec as risk-free rate. Annualised.|
|sortino\_ratio|Like Sharpe but uses only downside standard deviation. More appropriate for non-symmetric return distributions.|
|calmar\_ratio|CAGR / |max\_drawdown\_pct|. Measures return earned per unit of maximum tail risk.|
|omega\_ratio|Probability-weighted gain / loss ratio above zero return threshold. More robust than Sharpe for fat-tailed distributions common in momentum strategies.|
|kelly\_fraction|Optimal theoretical bet size: (win\_rate × avg\_win - loss\_rate × |avg\_loss|) / avg\_win. Informational only — always use a fraction of Kelly in practice.|

## **7.3 Drawdown Metrics**

|**Metric Key**|**Description**|
| :- | :- |
|max\_drawdown\_pct|Largest peak-to-trough decline in equity (%). Negative value — e.g. -18.5 = 18.5% drawdown.|
|avg\_drawdown\_pct|Average depth of all drawdown periods (%).|
|max\_drawdown\_duration\_bars|Longest continuous period (in bars) spent below a prior equity peak.|

## **7.4 Trade Statistics**

|**Metric Key**|**Description**|
| :- | :- |
|total\_trades|Total completed round-trip trades.|
|winning\_trades|Trades with net\_pnl > 0.|
|losing\_trades|Trades with net\_pnl <= 0.|
|win\_rate\_pct|winning\_trades / total\_trades × 100.|
|avg\_win\_inr|Average profit on winning trades in ₹.|
|avg\_loss\_inr|Average loss on losing trades in ₹ (negative value).|
|profit\_factor|Gross profit / |gross loss|. > 1.0 = profitable strategy.|
|expectancy\_inr|Expected ₹ value per trade: (win\_rate × avg\_win) + (loss\_rate × avg\_loss).|
|avg\_trade\_duration\_bars|Average number of bars held per trade.|
|max\_consecutive\_wins|Longest winning streak.|
|max\_consecutive\_losses|Longest losing streak.|
|avg\_mae\_inr|Average Maximum Adverse Excursion (₹) — average worst unrealised loss during trades.|
|avg\_mfe\_inr|Average Maximum Favourable Excursion (₹) — average best unrealised profit during trades.|

## **7.5 Time-Series Breakdowns**

|**Metric Key**|**Description**|
| :- | :- |
|monthly\_returns|dict {YYYY-MM: pct\_return}. Monthly return percentage derived from daily-resampled equity curve.|
|annual\_returns|dict {YYYY: pct\_return}. Annual return percentage.|
|exposure\_pct|Percentage of total bars during which at least one position was open. Lower values indicate more selective, capital-efficient strategies.|


# **8. Writing Strategies**
Any Python class with a generate\_signals(df) method can be used as a strategy. The engine imposes no inheritance requirement, but inheriting from BaseStrategy provides useful structure and IDE support.

## **8.1 The generate\_signals() Contract**
generate\_signals() receives a copy of the OHLCV DataFrame and must return the same DataFrame with a 'signal' column added.

|**Signal Value**|**Meaning**|
| :- | :- |
|1|BUY — open a long position (or close an existing short, then open long).|
|0|No action. Hold current state.|
|-1|SELL — close any open long position. If allow\_shorting=True, also opens a short.|

## **8.2 Minimal Strategy Example**
import pandas as pd

from strategies.base import BaseStrategy



class EMACrossover(BaseStrategy):

`    `"""Long when fast EMA crosses above slow EMA."""



`    `PARAM\_SCHEMA = [

`        `{"name": "fast\_period", "type": "int", "default": 9,  "min": 2, "max": 50},

`        `{"name": "slow\_period", "type": "int", "default": 21, "min": 5, "max": 200},

`    `]

`    `DESCRIPTION = "EMA Crossover — long on golden cross, exit on death cross"

`    `CATEGORY    = "Trend Following"



`    `def \_\_init\_\_(self, fast\_period: int = 9, slow\_period: int = 21):

`        `super().\_\_init\_\_(name="EMA Crossover")

`        `self.fast = fast\_period

`        `self.slow = slow\_period



`    `def generate\_signals(self, df: pd.DataFrame) -> pd.DataFrame:

`        `df = df.copy()   # ALWAYS copy to avoid mutating the engine's DataFrame

`        `df['ema\_fast'] = df['close'].ewm(span=self.fast, adjust=False).mean()

`        `df['ema\_slow'] = df['close'].ewm(span=self.slow, adjust=False).mean()

`        `df['signal'] = 0

`        `# Golden cross: fast crosses above slow → buy

`        `df.loc[(df['ema\_fast'] > df['ema\_slow']) &

`               `(df['ema\_fast'].shift(1) <= df['ema\_slow'].shift(1)), 'signal'] = 1

`        `# Death cross: fast crosses below slow → sell/exit

`        `df.loc[(df['ema\_fast'] < df['ema\_slow']) &

`               `(df['ema\_fast'].shift(1) >= df['ema\_slow'].shift(1)), 'signal'] = -1

`        `return df

## **8.3 No Look-Ahead Rules**
✓  Always call df.copy() at the start of generate\_signals() to avoid mutating the engine's DataFrame.

✓  Use .shift(1) when you need to compare today's value against yesterday's.

✓  Avoid normalising with min/max of the full dataset — rolling windows look backward by design.

*⚠  Never use df['close'].shift(-1) — that is future data. The engine cannot detect this automatically.*

*⚠  Never compute signals using data beyond the current bar. Rolling windows are safe because they look backward.*


# **9. Parameter Optimizer**
The Optimizer class performs grid or random search over strategy parameters. It is fully decoupled from the engine — it calls run\_event\_loop() directly with no engine object required.

## **9.1 Grid Search**
from backtester.optimizer import Optimizer, SearchMethod

from backtester.models import BacktestConfig

from strategies.base import EMACrossover



cfg = BacktestConfig(initial\_capital=500\_000)

opt = Optimizer(cfg)



grid = {

`    `'fast\_period': [5, 9, 13, 21],

`    `'slow\_period': [21, 34, 50, 89],

}



results = opt.run(

`    `df             = df\_infy,

`    `strategy\_class = EMACrossover,

`    `param\_grid     = grid,

`    `symbol         = 'INFY',

`    `metric         = 'sharpe\_ratio',   # optimise for Sharpe

`    `method         = SearchMethod.GRID,

`    `top\_n          = 10,

)

print(results[['fast\_period', 'slow\_period', 'sharpe\_ratio']].to\_string())

## **9.2 Random Search**
For large parameter spaces where grid search is too expensive:

results = opt.run(

`    `df             = df,

`    `strategy\_class = MyStrategy,

`    `param\_grid     = large\_grid,

`    `method         = SearchMethod.RANDOM,

`    `n\_trials       = 200,    # try 200 random combinations

`    `metric         = 'calmar\_ratio',

)

## **9.3 Walk-Forward Optimisation**
Walk-forward testing prevents overfitting by optimising on a training window and immediately validating on an unseen test window:

wf\_results = opt.walk\_forward(

`    `df             = df,

`    `strategy\_class = EMACrossover,

`    `param\_grid     = grid,

`    `symbol         = 'INFY',

`    `metric         = 'sharpe\_ratio',

`    `train\_bars     = 500,    # optimise on 500 bars

`    `test\_bars      = 100,    # validate on next 100 bars

`    `step\_bars      = 100,    # roll forward 100 bars each iteration

)

print(wf\_results[['window\_start', 'fast\_period', 'test\_sharpe\_ratio']])

*⚠  In-sample optimisation always finds parameters that overfit historical data. Always validate with an out-of-sample hold-out or walk-forward before trading live.*

## **9.4 Available Optimisation Metrics**
Any key returned by compute\_performance() can be used as the optimisation metric. The most useful ones:

|**Metric Key**|**Optimise For**|
| :- | :- |
|sharpe\_ratio|Risk-adjusted return (most common). Use for strategies where return distribution is roughly normal.|
|sortino\_ratio|Downside-risk-adjusted return. Better than Sharpe for strategies with large occasional gains.|
|calmar\_ratio|CAGR per unit of max drawdown. Use for drawdown-sensitive capital.|
|profit\_factor|Gross win / gross loss ratio. Good for trend strategies. Avoid overfitting to high values.|
|expectancy\_inr|Expected ₹ per trade. Useful for comparing strategies across different trade frequencies.|
|total\_return\_pct|Raw absolute return. Only use if all strategies have similar trade frequency.|


# **10. Visual Report**
generate\_report() produces a multi-panel PNG in the style of Streak/Zerodha's backtesting charts.

from backtester.report import generate\_report



fpath = generate\_report(

`    `result,

`    `symbol      = 'INFY',

`    `output\_dir  = 'reports/',

`    `max\_candles = 1000,     # plot last 1000 candles only

)

print(f'Chart saved: {fpath}')

## **10.1 Chart Panels**

|**Panel**|**Content**|
| :- | :- |
|1 — Candlestick|Vectorised OHLCV candles. Indicator overlays (EMA/SMA/BB/Supertrend) auto-detected from signals\_df columns. ▲ buy markers and ▼ sell markers at entry/exit bars with price labels.|
|2 — Volume|Green/red volume bars (rendered only if 'volume' column present in signals\_df).|
|3 — Oscillators|RSI, MACD histogram, Stochastic — rendered only when those columns are present. RSI includes 30/50/70 reference lines.|
|4 — Equity Curve|Portfolio value over time. Shaded green above initial capital, red below. ₹ axis in Lakhs format.|
|5 — Drawdown|Red filled area showing % decline from running peak. Configurable alert threshold line.|
|6 — Summary Table|Key performance metrics in two columns. Profitable values shaded green, negative values shaded red.|

## **10.2 Indicator Auto-Detection**
The report generator automatically detects indicator columns by prefix. No manual configuration is needed — just add the correctly-named columns in generate\_signals():

|**Column Prefix**|**Rendered As**|
| :- | :- |
|ema\_|Price overlay line|
|sma\_|Price overlay line|
|dema\_|Price overlay line|
|bb\_upper / bb\_lower / bb\_middle|Bollinger Band with filled channel|
|supertrend / st\_|Scatter markers (bull=green, bear=red) if direction column present|
|vwap|Price overlay line|
|rsi|Oscillator panel with 30/50/70 reference lines|
|macd\_hist|Oscillator histogram (green/red bars)|
|stoch / cci / adx / mfi|Oscillator panel line|


# **11. Portfolio Tracker**
Portfolio is the live ledger for one backtest run. It tracks cash, open positions, equity, and drawdown — and is the authoritative source of truth for all state.

## **11.1 Initialisation**
from backtester.portfolio import Portfolio

from backtester.models import BacktestConfig



cfg       = BacktestConfig(initial\_capital=500\_000, max\_drawdown\_pct=0.20)

portfolio = Portfolio(cfg)

## **11.2 Key Methods**

|**Method**|**Description**|
| :- | :- |
|portfolio.equity(prices)|Returns current portfolio value: cash + sum of mark-to-market unrealised P&L. Accepts optional {symbol: price} dict.|
|portfolio.can\_open()|Returns True if a new position can be opened (not halted, max\_positions not exceeded).|
|portfolio.mark\_bar(prices)|Record equity and drawdown at the current bar. Sets is\_halted=True if max drawdown is breached.|
|portfolio.sync\_cash(amount)|Update cash balance after a fill (called after each FillEngine.open/close call).|
|portfolio.add\_position(pos)|Add a filled Position to the tracked list.|
|portfolio.remove\_position(p)|Remove a closed Position from the tracked list.|
|portfolio.add\_trade(trade)|Append a completed Trade to the trade log.|
|portfolio.to\_equity\_series(idx)|Build a pd.Series equity curve aligned to the given bar index. Pads short curves.|
|portfolio.to\_drawdown\_series(idx)|Build a pd.Series drawdown curve aligned to the given bar index.|
|portfolio.summary()|Returns a quick dict summary: cash, open\_positions, completed\_trades, total\_net\_pnl, is\_halted.|


# **12. Testing**
## **12.1 Running the Test Suite**
\# All 125 tests:

cd algo\_trading

python -m pytest tests/test\_backtester.py tests/test\_report\_portfolio.py -v



\# Quick summary only:

python -m pytest tests/ -q

## **12.2 Test Coverage Summary**

|**Test Class**|**Tests**|
| :- | :- |
|TestBacktestConfig|6 — validates all constraint checks in BacktestConfig.validate()|
|TestPosition|9 — unrealised P&L, MAE/MFE, trailing stop logic, fixed stop logic, gap fills|
|TestOrderTypes|9 — limit fill/no-fill, stop trigger, stop-limit two-phase logic|
|TestPositionSizer|8 — fixed qty, fixed qty cap, risk-based with stop/ATR/fallback, zero cases|
|TestFillEngine|8 — open position, insufficient cash, close position, duration, short P&L, stop triggers|
|TestATR|3 — shape, warmup NaN, positive values|
|TestEventLoop|12 — single trade, equity, drawdown, end-of-data close, max-dd halt, shorting on/off, no look-ahead, limit fill, trailing stop, equity completeness|
|TestPerformance|11 — all required keys, capital consistency, win rate bounds, drawdown non-positive, empty curve, exposure bounds, commission, monthly returns|
|TestBacktestEngine|12 — result types, summary string, metrics cache, preflight errors, portfolio run, trade\_df, intraday squareoff|
|TestCommissionModel|6 — delivery/intraday charges, DP charge, total = sum of components, invalid inputs|
|TestEdgeCases|6 — empty df, single row, all-zero signals, equity at initial capital, max\_positions, large dataset performance|
|TestPortfolio|25 — all Portfolio methods and edge cases|
|TestReport|9 — PNG creation, custom filename, zero trades, max\_candles, indicator columns, dir creation|
|TestInitExports|2 — all public names importable, engine instantiable from \_\_init\_\_|


# **13. Migration from Previous Version**
The backtester was refactored from a single 1200-line engine.py into 8 focused modules. The public API changed minimally — most existing code works with a one-line import change.

## **13.1 Import Changes**

|**Old Import**|**New Import**|
| :- | :- |
|from backtester.engine import BacktestEngineV3|from backtester.engine import BacktestEngine|
|from backtester.engine import BacktestConfigV3|from backtester.models import BacktestConfig|
|from backtester.engine import BacktestResult|from backtester.models import BacktestResult|
|from backtester.engine import Trade|from backtester.models import Trade|
|from backtester.engine import Position|from backtester.models import Position|
|from backtester.engine import OrderType|from backtester.models import OrderType|
|from backtester.engine\_v2 import BacktestEngineV2|Deleted — use BacktestEngine (same API)|
|from backtester.commission import CommissionModel|Deleted — use broker.upstox.commission.CommissionModel|

## **13.2 Behaviour Changes**

|**Area**|**Change**|
| :- | :- |
|Metrics|result.\_compute\_metrics() is now result.metrics() (public, cached). The old private method is removed.|
|Performance|6 new metrics added: omega\_ratio, kelly\_fraction, monthly\_returns, annual\_returns, avg\_mae\_inr, avg\_mfe\_inr.|
|Candlestick chart|Vectorised rendering — 500× faster for 2000 candles. report.py now uses 4 Matplotlib calls instead of a per-bar loop.|
|Portfolio|portfolio.py is now the live ledger, wired into the engine. Previously it was dead code never used by the engine.|
|Optimizer|optimizer.py is a standalone module. The old optimize() method in BacktestEngineV2 is removed. walk\_forward() is new.|
|Dead files|engine\_old.py, engine\_v2.py, order\_types\_backup.py, backtester/commission.py — all deleted.|


# **Appendix A — File Reference**

|**File**|**Status / Role**|
| :- | :- |
|backtester/\_\_init\_\_.py|Public re-exports. Import everything from here.|
|backtester/engine.py|Public API — BacktestEngine.run() and run\_portfolio().|
|backtester/models.py|All dataclasses — BacktestConfig, Position, Trade, BacktestResult, OrderType.|
|backtester/event\_loop.py|Core bar-by-bar simulation loop.|
|backtester/fill\_engine.py|Order fill, stop-loss, and pending order logic.|
|backtester/position\_sizer.py|Position sizing (fixed or risk-based).|
|backtester/order\_types.py|PendingOrder and pure fill-check functions.|
|backtester/performance.py|25+ performance metrics.|
|backtester/portfolio.py|Live cash and position ledger.|
|backtester/optimizer.py|Grid/random search and walk-forward optimisation.|
|backtester/report.py|Streak-style PNG chart generator.|
|broker/upstox/commission.py|LOCKED — Indian market charge model.|
|tests/test\_backtester.py|89 tests — models, order\_types, sizer, fill\_engine, event\_loop, performance, engine.|
|tests/test\_report\_portfolio.py|36 tests — report.py, portfolio.py, \_\_init\_\_ exports.|

# **Appendix B — Glossary**

|**Term**|**Definition**|
| :- | :- |
|MAE|Maximum Adverse Excursion — the worst unrealised loss reached during a trade before exit.|
|MFE|Maximum Favourable Excursion — the best unrealised profit reached during a trade before exit.|
|CAGR|Compound Annual Growth Rate — the constant annual return that would produce the same total return over the backtest period.|
|Sharpe Ratio|Excess return above the risk-free rate, divided by total return volatility. Higher is better.|
|Sortino Ratio|Like Sharpe but only penalises downside volatility. Preferred for positively-skewed strategies.|
|Calmar Ratio|CAGR / |max drawdown|. Measures how much return is earned per unit of worst-case loss.|
|Omega Ratio|Probability-weighted ratio of returns above vs. below a threshold. Does not assume normal distribution.|
|Kelly Criterion|Optimal bet size fraction from information theory. In practice, use half-Kelly or less.|
|Exposure %|Fraction of total bars where at least one position was open. Lower exposure = more selective strategy.|
|Look-Ahead Bias|Using future data in a strategy signal, making backtest results better than live results would be.|
|Walk-Forward|Optimise on a training window, validate on the immediately following out-of-sample window.|
|Next-Bar Open|The standard fill model: signal on bar i executes at bar i+1's opening price.|
|Gap Fill|When price opens beyond a stop or limit price, fill occurs at the open price, not the order price.|

