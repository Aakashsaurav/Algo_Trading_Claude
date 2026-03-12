# TASK_QUEUE.md

## AlgoDesk Development Backlog

This file lists development tasks for AI agents and developers.

AI agents should implement tasks **one at a time**.

Always read:

PROJECT_CONTEXT.md
AI_DEVELOPMENT_GUIDE.md

before starting work.

---

# HIGH PRIORITY

### 1. Walk-Forward Optimisation

Add walk-forward testing capability to the backtest engine.

Requirements:

* train window
* test window
* rolling optimization
* performance summary

Files likely involved:

backtester/engine_v2.py
backtester/performance.py

---

### 2. Portfolio Level Backtesting

Allow multiple strategies to run simultaneously.

Requirements:

* portfolio allocation
* strategy capital allocation
* combined equity curve

Modules:

backtester/portfolio.py
backtester/engine.py

---

### 3. Trailing Stop Loss

Add trailing stop logic.

Requirements:

* ATR based trailing stop
* percentage trailing stop
* backtest support
* live engine support

---

# MEDIUM PRIORITY

### 4. Strategy Parameter Optimiser

Implement parameter search.

Support:

* grid search
* random search
* multiprocessing

---

### 5. Screener Performance Optimisation

Improve screener performance.

Possible methods:

* multiprocessing
* caching
* incremental scanning

---

### 6. Telegram Alerts

Send notifications when:

* strategy enters trade
* stop loss triggered
* daily drawdown exceeded

---

# LOW PRIORITY

### 7. Strategy Marketplace

Allow importing/exporting strategies.

---

### 8. Monte Carlo Simulation

Add robustness testing.

---

### 9. Dashboard UI Improvements

Improve charts and layout.

---

# END OF TASK QUEUE
