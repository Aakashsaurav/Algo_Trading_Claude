# Quick Reference: EMACrossover RELIANCE Backtest

## TL;DR - Run in 2 Steps

### Step 1: Fetch Data (First time only, takes ~5-10 minutes)
```bash
cd /workspaces/Algo_Trading_Claude
python run_ema_backtest.py --fetch
```

### Step 2: Run Backtest (takes ~1-2 minutes)
```bash
cd /workspaces/Algo_Trading_Claude
python run_ema_backtest.py
```

---

## Backtest Specifications

| Parameter | Value |
|-----------|-------|
| **Symbol** | RELIANCE |
| **Timeframe** | Daily (1D) |
| **Period** | Last 15 years |
| **Strategy** | EMACrossover |
| **Fast EMA** | 10 periods |
| **Slow EMA** | 21 periods |
| **Initial Capital** | Rs 500,000 |
| **Position Size** | 50% of available capital |
| **Max Open Positions** | 1 |
| **Execution** | Market orders at next bar open |
| **Stop Loss** | Entry - (2.0 × ATR) |

---

## Entry and Exit Rules

### Entry Signal (BUY)
- **Trigger**: Fast EMA (10) crosses ABOVE Slow EMA (21)
- **Execution**: Entry at NEXT bar's OPEN price
- **Position Size**: 50% of current portfolio value ÷ Entry price
- **Stop Loss**: Entry price - (2.0 × 14-period ATR)

### Exit Signal (SELL)
- **Trigger**: Fast EMA (10) crosses BELOW Slow EMA (21)
- **Execution**: Exit at NEXT bar's OPEN price

---

## Script Options

### Option 1: Quick Start (Recommended)
```bash
python run_ema_backtest.py              # Auto-detects if data needs fetching
```

### Option 2: Fetch Fresh Data + Backtest
```bash
python run_ema_backtest.py --fetch      # Force fetch + run backtest
```

### Option 3: Only Fetch Data
```bash
python run_ema_backtest.py --fetch-only # Fetch data without backtesting
```

### Option 4: Backtest Only (Data must exist)
```bash
python tests/test_ema_crossover_reliance.py
```

### Option 5: Fetch Data Only
```bash
python fetch_reliance_data.py
```

---

## What Gets Generated

After running backtest, you'll get:

### Console Output
✅ Summary statistics (CAGR, Sharpe, Max Drawdown, Win Rate, etc.)
✅ First 10 trades with entry/exit dates and P&L
✅ Total number of trades generated

### Files Generated

1. **Trade Log** (CSV)
   - Path: `strategies/output/trade/ema_crossover_reliance_trade_log.csv`
   - Content: All trades with entry/exit prices, P&L, charges, etc.
   - Rows: One per trade

2. **Raw Data** (CSV)
   - Path: `strategies/output/raw_data/ema_crossover_reliance_raw_data.csv`
   - Content: All OHLCV bars with indicator values and signals
   - Rows: One per daily bar (3,652+ rows)

3. **Interactive Chart** (HTML)
   - Path: `strategies/output/chart/ema_crossover_reliance_chart.html`
   - Content: Candlestick chart with entry/exit markers
   - Usage: Open in web browser to visualize trades

---

## Key Metrics Explained

| Metric | Definition | Good Value |
|--------|-----------|-----------|
| **CAGR** | Annual compound growth rate | > 15% |
| **Total Return** | Overall % profit/loss | > 0% |
| **Sharpe Ratio** | Risk-adjusted returns | > 1.0 |
| **Max Drawdown** | Worst peak-to-trough loss | < 30% |
| **Win Rate** | % of profitable trades | > 50% |
| **Profit Factor** | Total wins / Total losses | > 1.5 |
| **Risk/Reward** | Avg Win / Avg Loss | > 1.0 |

---

## Position Sizing Example

**Initial Capital**: Rs 500,000

### First Trade
- Entry Price: Rs 2,000
- Position Size = (500,000 × 0.50) / 2,000 = **250 shares**
- Capital Used: 250 × 2,000 = Rs 500,000 (50% of portfolio)

### If Portfolio grows to Rs 700,000
- Entry Price: Rs 2,000
- Position Size = (700,000 × 0.50) / 2,000 = **175 shares**
- Capital Used: 175 × 2,000 = Rs 350,000 (50% of new portfolio)

### If Portfolio shrinks to Rs 400,000
- Entry Price: Rs 2,000
- Position Size = (400,000 × 0.50) / 2,000 = **100 shares**
- Capital Used: 100 × 2,000 = Rs 200,000 (50% of new portfolio)

---

## Customization Examples

### Change Position Sizing to 30%
Edit `tests/test_ema_crossover_reliance.py`:
```python
capital_risk_pct=0.30,  # Change from 0.50 to 0.30
```

### Change to 10 Years Instead of 15
Edit `tests/test_ema_crossover_reliance.py`:
```python
df = load_reliance_data(years=10)  # Change from 15 to 10
```

### Change Strategy Parameters
Edit `tests/test_ema_crossover_reliance.py`:
```python
strategy = EMACrossoverStrategy(
    params={
        "fast_period": 8,      # Changed from 10
        "slow_period": 20,     # Changed from 21
        "atr_period": 14,
        "atr_multiplier": 2.0,
    }
)
```

### Change Initial Capital
Edit `tests/test_ema_crossover_reliance.py`:
```python
initial_capital=1_000_000.0,  # Changed from 500_000.0
```

---

## Troubleshooting

### "No RELIANCE data found"
```bash
# Run this:
python run_ema_backtest.py --fetch
```

### Takes too long / hangs
- First fetch can take 5-10 minutes (depends on internet)
- First backtest can take 1-2 minutes (normal for 15 years)
- Subsequent runs are faster

### API Authentication Error
- Check `config.py` for Upstox credentials
- Verify API key and secret are correct
- Ensure Upstox account is active

### Zero Trades Generated
- Check if you have the right symbols
- Try fewer years first for testing
- Verify data was loaded correctly

---

## File Locations

| File | Purpose |
|------|---------|
| `tests/test_ema_crossover_reliance.py` | Main backtest script |
| `run_ema_backtest.py` | Quick-start wrapper |
| `fetch_reliance_data.py` | Data fetcher |
| `BACKTEST_GUIDE.md` | Detailed documentation |
| `QUICK_REFERENCE.md` | This file |
| `strategies/momentum/ema_crossover.py` | Strategy code |
| `backtester/engine_v3.py` | Backtesting engine |
| `data/parquet_store.py` | Data storage |

---

## Data Storage Location

RELIANCE data is cached here:
```
data/ohlcv/daily/NSE_EQ_RELIANCE.parquet
```

Once fetched, it's reusable across multiple backtests!

---

## Performance Tips

- First backtest of 15 years takes ~2 minutes (expected)
- Subsequent backtests are instant (data cached)
- Data fetching only needed once
- For testing, use fewer years (5 or 10) for faster results

---

## Notes

✅ **What's Automated**:
- Data fetching from Upstox
- Indicator calculation (EMA, ATR)
- Signal generation (crossovers)
- Position sizing (50% capital)
- Order execution simulation (next bar open)
- Trade log generation
- Performance metric calculation

⚠️ **What's Simulated**:
- Market orders at next bar open (realistic execution)
- Upstox commission (real brokerage fees)
- Stop-loss orders (not triggered in backtest, only in trade log)
- Slippage: None (assumes perfect fills at open price)

---

## Next Steps

1. ✅ Run backtest with default parameters
2. 📊 Analyze results (metrics, trade log, chart)
3. 🔧 Customize strategy parameters
4. 📈 Optimize for better results
5. 📝 Document findings
6. 🎯 Paper trade before going live

---

**Last Updated**: March 2024  
**Strategy Version**: 1.0  
**Engine Version**: v3
