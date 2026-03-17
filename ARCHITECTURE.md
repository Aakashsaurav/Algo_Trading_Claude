# AlgoDesk — Algorithmic Trading Platform
## Phase 1–4 Complete Architecture Reference

---

## Folder Structure

```
algo_trading/                          ← Project root
│
├── .env.example                       ← Copy to .env, fill credentials
├── config.py                          ← AppConfig: all paths, env vars, logging setup
├── requirements.txt                   ← All Python dependencies
├── ARCHITECTURE.md                    ← This file
│
├── broker/                            ── BROKER LAYER (Phase 1)
│   ├── __init__.py
│   ├── auth.py                        ← Base auth (mirrors upstox/auth.py)
│   ├── instrument_manager.py          ← Base instrument manager
│   ├── market_data.py                 ← Base market data (mirrors upstox/market_data.py)
│   └── upstox/                        ← Upstox-specific implementation
│       ├── __init__.py
│       ├── auth.py                    ← AuthManager: OAuth2 login, token storage, refresh
│       ├── data_manager.py            ← get_ohlcv(): fetch+cache OHLCV as Parquet
│       ├── instrument_manager.py      ← Instrument key lookup, NSE FO list
│       └── market_data.py             ← MarketDataManager: REST calls for OHLCV + holidays
│
├── data/                              ── DATA LAYER (Phase 1)
│   ├── __init__.py
│   ├── data_manager.py                ← get_ohlcv(): unified data fetch (broker → Parquet cache)
│   ├── fetcher.py                     ← DataFetcher: download OHLCV via Upstox SDK
│   ├── cleaner.py                     ← DataCleaner: validate, deduplicate, fill gaps
│   ├── parquet_store.py               ← ParquetStore: read/write Parquet files
│   ├── universe.py                    ← UniverseManager: Nifty500, FO list, market holidays
│   └── ohlcv/                         ← Parquet storage
│       ├── daily/                     ← daily/SYMBOL.parquet
│       ├── minute/                    ← minute/SYMBOL/MMYY.parquet
│       └── weekly/                    ← weekly/SYMBOL.parquet
│
├── indicators/                        ── INDICATOR LAYER (Phase 2)
│   ├── __init__.py
│   ├── technical.py                   ← Master indicator module (used by strategies/base.py)
│   │                                    sma, ema, dema, rsi, macd, bollinger_bands, atr,
│   │                                    supertrend, stochastic, roc, vwap, obv, adx,
│   │                                    keltner_channels, crossover, crossunder
│   ├── bridge.py                      ← IndicatorBridge: auto-selects pandas-ta / TA-Lib / custom
│   ├── moving_averages.py             ← sma, ema, dema, wma, vwap (pure functions)
│   ├── oscillators.py                 ← rsi, stochastic, macd, roc, cci
│   ├── volatility.py                  ← atr, bollinger_bands, keltner_channels, bb_squeeze
│   ├── trend.py                       ← supertrend, adx
│   └── statistics.py                  ← zscore, rolling_correlation, rolling_beta, cointegration
│
├── strategies/                        ── STRATEGY LAYER (Phase 2)
│   ├── __init__.py
│   ├── base.py                        ← BaseStrategy ABC + 5 built-in strategies
│   │                                    EMACrossover, RSIMeanReversion, BollingerBandStrategy
│   │                                    MACDStrategy, SupertrendStrategy
│   │                                    ← These are used by the backtest engine + registry
│   ├── base_strategy.py               ← Alternative BaseStrategy (on_bar/prepare pattern)
│   │                                    BaseStrategy, Signal, Action, PortfolioState
│   │                                    ← Used by the strategy module files below
│   ├── registry.py                    ← get_strategy_registry(), load_strategy()
│   │                                    Auto-discovers all BaseStrategy subclasses
│   │
│   ├── momentum/                      ← Momentum strategies (on_bar + generate_signals)
│   │   ├── __init__.py
│   │   ├── ema_crossover.py           ← EMACrossoverStrategy
│   │   └── macd_crossover.py          ← MACDCrossoverStrategy
│   │
│   ├── mean_reversion/                ← Mean reversion strategies
│   │   ├── __init__.py
│   │   ├── rsi_reversion.py           ← RSIReversionStrategy
│   │   └── bollinger_squeeze.py       ← BollingerSqueezeStrategy
│   │
│   ├── trend/                         ← Trend-following strategies
│   │   ├── __init__.py
│   │   └── supertrend_strategy.py     ← SupertrendStrategy
│   │
│   └── output/                        ← Backtest artifacts (auto-created)
│       ├── trade/                     ← trade_log CSV files
│       ├── raw_data/                  ← OHLCV + signal CSVs
│       └── chart/                     ← PNG chart exports
│
├── backtester/                        ── BACKTEST ENGINE (Phase 3)
│   ├── __init__.py
│   ├── engine.py                      ← BacktestEngine (primary), BacktestConfig, BacktestResult
│   │                                    Trade, Position dataclasses
│   │                                    result._compute_metrics() → 20 performance metrics
│   ├── engine_v2.py                   ← BacktestEngineV2 + BacktestConfigV2
│   │                                    engine.optimize() for grid/random param search
│   ├── commission.py                  ← CommissionModel, Segment enum
│   │                                    Segments: EQUITY_DELIVERY, EQUITY_INTRADAY,
│   │                                    FUTURES, OPTIONS_BUY, OPTIONS_SELL
│   ├── order_types.py                 ← OrderType enum, StopTracker, PendingOrder
│   ├── performance.py                 ← compute_performance(): CAGR, Sharpe, Sortino,
│   │                                    Max Drawdown, Win Rate, Expectancy, Exposure
│   ├── portfolio.py                   ← Portfolio: equity curve, drawdown, position tracking
│   ├── trade_log.py                   ← TradeLog, TradeRecord, OpenPosition
│   │                                    to_dataframe(), to_csv(), summary()
│   ├── report.py                      ← generate_report(): matplotlib chart PNG
│   └── order_types_backup.py          ← Backup (can be deleted)
│
├── screener/                          ── SCREENER (Phase 3)
│   ├── __init__.py
│   ├── screener.py                    ← Screener (v1): run(), scan(), export_csv()
│   │                                    ScreenerConfig dataclass
│   │                                    Loads from Parquet, runs any strategy
│   ├── screener_v2.py                 ← Screener (v2, used by dashboard): scan()
│   │                                    ScreenerConfig dataclass, ThreadPoolExecutor
│   └── output/                        ← Screener CSV results (auto-saved)
│
├── dashboard/                         ── WEB DASHBOARD (Phase 4)
│   ├── __init__.py
│   ├── app.py                         ← FastAPI app (primary server)
│   │                                    _safe_scalar(), _serialize_trade() ← JSON-safe trade output
│   │                                    Routes: /backtester, /screener, /strategy-builder, /live-bot
│   │                                    API: /api/backtest, /api/optimize, /api/screener/scan
│   │                                         /api/strategies, /api/data/ohlcv/{symbol}, /health
│   │
│   ├── templates/                     ← Jinja2 HTML templates
│   │   ├── base.html                  ← Sidebar nav layout (all pages inherit)
│   │   ├── backtester.html            ← Strategy config + run + results display
│   │   ├── screener.html              ← Universe scan + results table
│   │   ├── strategy_builder.html      ← Monaco code editor for custom strategies
│   │   └── live_bot.html              ← Live / paper trading bot control panel
│   │
│   └── static/
│       ├── css/
│       │   └── main.css               ← Full dark theme UI (variables, sidebar, cards, badges)
│       └── js/
│           ├── chart.js               ← TradingView Lightweight Charts
│           │                            initChart(), updateChart(bars, trades), fetchAndRenderChart()
│           ├── backtest_results.js    ← runBacktest(), renderBacktestResults(), runOptimizer()
│           ├── code_editor.js         ← Monaco editor integration, saveStrategy()
│           ├── screener_table.js      ← runScreener(), scSortBy(), CSV download
│           └── live_bot_panel.js      ← startBot(), stopBot(), checkAuth(), _botLog()
│
├── logs/                              ← Log files (auto-created)
│   └── app.log
│
├── reports/                           ← Generated report exports
│
└── tests/                             ── TEST SUITE
    ├── __init__.py
    ├── test_phase1.py                 ← Broker, data, indicator unit tests
    └── test_phase2_phase3.py          ← Strategy, engine, screener integration tests
```

---

## File Count by Module

| Module        | Files | Status      |
|---------------|-------|-------------|
| Root config   | 3     | ✅ Complete |
| broker/       | 7     | ✅ Complete |
| data/         | 6     | ✅ Complete |
| indicators/   | 7     | ✅ Complete |
| strategies/   | 11    | ✅ Complete |
| backtester/   | 9     | ✅ Complete |
| screener/     | 3     | ✅ Complete |
| dashboard/    | 13    | ✅ Complete |
| tests/        | 3     | ✅ Complete |
| **TOTAL**     | **62**| ✅ All OK   |

---

## Key Design Decisions

### Two Strategy Base Classes (intentional)

**`strategies/base.py → BaseStrategy`** — used by the backtest engine
- Implements `generate_signals(df) → df` (vectorised, returns signal column)
- 5 built-in strategies live here: EMACrossover, RSIMeanReversion, etc.
- Discovered by `strategies/registry.py` → shown in dashboard dropdown

**`strategies/base_strategy.py → BaseStrategy`** — used by strategy module files
- Implements `prepare(df)` + `on_bar(index, row, portfolio) → [Signal]`
- Also implements `generate_signals(df)` which calls `prepare()` first
- Pattern is ready for live engine (Phase 5 event-driven bar-by-bar processing)
- Strategy files: `momentum/`, `mean_reversion/`, `trend/`

Both patterns produce the same output: a DataFrame with a `signal` column.

### Two Screener Classes (intentional)

**`screener/screener.py`** — v1, full-featured
- Loads OHLCV from local Parquet
- Concurrent scanning with ThreadPoolExecutor
- Has `run()`, `scan()`, `export_csv()`, `ScreenerConfig`

**`screener/screener_v2.py`** — v2, API-optimised
- Accepts pre-loaded `{symbol: df}` dict (data already in memory)
- Used by `dashboard/app.py /api/screener/scan`
- Has `scan()`, `ScreenerConfig`

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up credentials
cp .env.example .env
# Edit .env with your Upstox API key and secret

# 3. Start dashboard
cd algo_trading
uvicorn dashboard.app:app --reload --port 8080

# 4. Open browser
# http://localhost:8080
```

---

## Phase Roadmap

| Phase | Status     | Contents |
|-------|------------|----------|
| 1     | ✅ Done    | Broker (Upstox auth, data, instruments), Data layer (Parquet, universe) |
| 2     | ✅ Done    | Indicators (8 modules), Strategy base classes, 5 built-in strategies |
| 3     | ✅ Done    | Backtest engine, performance analytics, screener, report generation |
| 4     | ✅ Done    | FastAPI dashboard, 5 HTML pages, 5 JS modules, CSS dark theme |
| 5     | 🔜 Next    | Live trading engine, WebSocket, Upstox order routing, risk management |
