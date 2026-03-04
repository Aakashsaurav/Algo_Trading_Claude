# Algo_Trading_Claude

This repository provides infrastructure for algorithmic trading backtests.

## Exporting Signals to CSV ✅

After running a backtest you can save the OHLCV data along with any
indicator columns and the generated signal column to a CSV file. For
example:

```python
from backtester.engine import BacktestEngine, BacktestConfig
from backtester.commission import Segment
from strategies.base import EMACrossover

engine = BacktestEngine(BacktestConfig(...))
df = ...  # fetch OHLCV data
strategy = EMACrossover(9, 21)
result = engine.run(df, strategy, symbol="RELIANCE-1m")

# write output to disk
result.export_signals_csv("signals_RELIANCE_2026-03-03.csv")
```

You can also manually export from the sample script at
`test_manually.py`, which now writes a CSV named according to the
strategy and date.
