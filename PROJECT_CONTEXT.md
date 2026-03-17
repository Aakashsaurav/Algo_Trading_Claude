# PROJECT_CONTEXT.md

## AlgoDesk — Algorithmic Trading Platform

This file provides **AI agents and developers with a complete understanding of the project architecture, rules, and development guidelines.**

All AI coding agents must read this file before modifying the repository.

---

# 1. Project Overview

AlgoDesk is a **self-hosted modular algorithmic trading platform for Indian equity markets**.

The platform allows users to:

• Develop strategies
• Backtest them with realistic cost modelling
• Screen markets
• Run paper trading bots using live market data
• Eventually deploy live trading

The system is designed so that:

**The same strategy class runs in backtest and live trading without modification.**

This ensures high reliability between research and execution.

---

# 2. Core Design Principles

These rules must **never be violated by AI-generated code.**

### Broker Independence

All strategy, backtest, and engine code must remain **broker-agnostic**.

Broker-specific logic must stay inside:

```
broker/
live_bot/feeds/
```

This allows future support for:

• Zerodha Kite
• Angel One
• Fyers

without modifying core logic.

---

### Strategy Reusability

The same strategy must run in:

```
BacktestEngine
LiveBotEngine
```

Strategies must not include broker calls or API calls.

Strategies should operate only on **dataframes or candle objects.**

---

### Replaceable Modules

Each layer must remain independent.

```
data layer
indicator layer
strategy layer
engine layer
broker layer
dashboard layer
```

No module should directly depend on another layer except through defined interfaces.

---

### No Look-Ahead Bias

Backtest logic must ensure:

```
orders fill at next bar
strategies see only past data
```

Never allow strategies to access future bars.

---

### Thread Safety

Live trading components must remain thread-safe.

Shared state is handled only through:

```
LiveState
CandleRegistry
```

All writes must be protected using locks.

---

### Deterministic Backtests

Backtest results must always be reproducible.

Random behaviour must be seeded.

---

# 3. High-Level System Architecture

The system consists of 9 layers.

```
Frontend Layer
REST API Layer
Backtest Engine
Live Bot Engine
Strategy Layer
Indicator Layer
Broker Layer
Data Layer
Storage Layer
```

Data flows differently depending on mode.

### Backtest Mode

```
Data → Indicators → Strategy → BacktestEngine → Metrics
```

### Live Trading Mode

```
WebSocket Feed → CandleBuilder → Strategy → RiskGuard → Broker
```

---

# 4. Repository Structure

```
algo_trading/

broker/
    upstox/

data/
    parquet_store.py
    cleaner.py
    fetcher.py
    universe.py

indicators/
    technical.py
    oscillators.py
    volatility.py
    statistics.py

strategies/
    base.py
    base_strategy.py
    momentum/
    trend/
    mean_reversion/

backtester/
    engine.py
    engine_v2.py
    portfolio.py
    commission.py
    performance.py

live_bot/
    engine.py
    state.py
    candle_builder.py
    feeds/
    orders/
    risk/

dashboard/
    FastAPI server

tests/
```

AI agents must **not restructure folders without explicit request.**

---

# 5. Core Components

## Data Layer

Responsible for:

• downloading OHLCV data
• validating data
• storing data in Parquet
• providing cached access

Primary entry point:

```
data_manager.get_ohlcv()
```

---

## Indicator Layer

Indicators must be:

• vectorized
• pandas compatible
• reusable

Indicators must **never call broker APIs**.

---

## Strategy Layer

Two patterns exist:

### Vectorised Strategy

```
generate_signals(df)
```

Used by:

```
BacktestEngine
```

---

### Event Driven Strategy

```
prepare(df)
on_bar(index,row,portfolio)
```

Used by:

```
LiveBotEngine
```

Strategies must return **Signal objects**.

---

## Backtest Engine

Responsible for:

• order simulation
• position tracking
• commission modelling
• performance metrics

Core file:

```
backtester/engine.py
```

Important rule:

Backtest behaviour must **match live trading behaviour**.

---

## Live Trading Engine

Components:

```
MarketFeed
PortfolioFeed
CandleBuilder
StrategyLoop
PaperBroker
RiskGuard
```

Threads running simultaneously:

```
market feed thread
portfolio feed thread
strategy thread
fastapi thread
```

---

# 6. Risk Management

All orders must pass through:

```
RiskGuard.check_order()
```

Checks include:

• kill switch
• market hours
• daily loss limit
• drawdown limit
• max positions
• duplicate positions

Risk checks must always run **before order placement.**

---

# 7. Data Storage

OHLCV data is stored in **Apache Parquet**.

Example structure:

```
data/ohlcv/daily/
data/ohlcv/minute/SYMBOL/
```

Advantages:

• fast loading
• partial reads
• compressed storage

---

# 8. Performance Metrics

Backtests compute:

```
CAGR
Sharpe Ratio
Sortino Ratio
Calmar Ratio
Max Drawdown
Profit Factor
Expectancy
Win Rate
```

All metrics are calculated inside:

```
backtester/performance.py
```

---

# 9. Testing Requirements

Tests exist for:

```
broker layer
data layer
indicator calculations
backtest engine
live trading components
risk guard
paper broker
websocket parsing
```

Currently the project contains **94 automated tests.**

AI must ensure **all tests pass before merging changes.**

---

# 10. Coding Guidelines

AI must follow these rules.

### Python Version

```
Python 3.11
```

---

### Code Style

• Use type hints
• Use dataclasses when appropriate
• Avoid global variables
• Prefer pure functions

---

### Logging

Use:

```
logging module
```

Do not use print statements.

---

### Performance

Critical sections:

```
backtest engine
indicator calculations
live tick processing
```

Avoid heavy loops where vectorization is possible.

---

# 11. Future Roadmap

### Phase 8 ✅

Live order execution via Upstox API has been implemented in Phase 8.
The paper broker has been replaced by a `LiveBroker` that uses the
`upstox_client.OrderApiV3` SDK. Unit tests now cover both paper and live
modes, and the engine selects the appropriate broker based on
`config.PAPER_TRADE`.

```
PaperBroker → LiveBroker
```

---

### Phase 9

AI-assisted features:

• strategy generation
• signal explanation
• anomaly detection

---

### Phase 10

Web-application to be hosted on server:

• Streak (Zerodha) like UI 
• A backtester, screener, paper trading and live trading tool
• Configurable for different brokers
• Have user login/sign up feature

---

# 12. Rules for AI Agents

When modifying the repository:

1. Do not break module boundaries.
2. Do not introduce broker logic into strategies.
3. Do not introduce look-ahead bias.
4. Maintain compatibility between backtest and live engines.
5. Ensure tests pass.

---

# 13. Typical Developer Tasks

Examples of safe tasks:

```
Add new indicator
Add new strategy
Improve backtest engine
Add optimisation algorithms
Improve risk management
Add analytics
```

---

# 14. Unsafe Modifications (Require Human Approval)

AI agents must **not perform these automatically**:

```
changing threading model
rewriting broker layer
modifying data schema
removing risk guard
changing commission model
```

---

# 15. AI Prompt Starter

When interacting with AI coding agents, start with:

"Read PROJECT_CONTEXT.md before making changes."

Example:

Understand this repository using PROJECT_CONTEXT.md and implement a new RSI breakout strategy.

---
