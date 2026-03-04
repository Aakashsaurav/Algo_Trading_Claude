#from data.parquet_store import parquet_store
from backtester.engine import BacktestEngine, BacktestConfig
from broker.upstox.data_manager import get_ohlcv
from backtester.commission import Segment
from strategies.base import EMACrossover, RSIMeanReversion
import pandas as pd

today = pd.Timestamp.today(tz="Asia/Kolkata").date().isoformat()
df = get_ohlcv(
      instrument_type="EQUITY", exchange="NSE", trading_symbol="RELIANCE",
      unit="minutes", interval=5, from_date="2022-01-01"
    )

if df.empty:
    raise RuntimeError("missing minute data for RELIANCE")

#strat = EMACrossover(9, 21)
strat = RSIMeanReversion(14, 30, 70, 200)
config = BacktestConfig(
    initial_capital=500_000,
    segment=Segment.EQUITY_DELIVERY,
    capital_risk_pct=0.1,
    allow_shorting=False,
    max_positions=5,
)
engine = BacktestEngine(config)

result = engine.run(df, strat, symbol="RELIANCE-1m")
print(result.summary())

# -----------------------------------------
# Debug: Check for early short signals
# -----------------------------------------
signals_df = result.signals_df
print("\n=== SIGNAL ANALYSIS ===")
print(f"Total signals generated: {len(signals_df)}")

# Find first non-zero signals
first_signals = signals_df[signals_df["signal"] != 0]
if len(first_signals) > 0:
    print(f"\nFirst 10 non-zero signals:")
    print(first_signals[["close", "signal"]].head(10))
    
    # Check if there's a -1 before the first +1
    first_buy_idx = None
    first_sell_idx = None
    for idx, row in signals_df.iterrows():
        if row["signal"] == 1 and first_buy_idx is None:
            first_buy_idx = idx
        if row["signal"] == -1 and first_sell_idx is None:
            first_sell_idx = idx
        if first_buy_idx is not None and first_sell_idx is not None:
            break
    
    if first_buy_idx is not None and first_sell_idx is not None:
        print(f"\nFirst BUY signal at index: {signals_df.index.tolist().index(first_buy_idx)}")
        print(f"First SELL signal at index: {signals_df.index.tolist().index(first_sell_idx)}")
        if first_sell_idx < first_buy_idx:
            print("⚠️ WARNING: First SELL signal comes BEFORE first BUY signal")
            print("   With allow_shorting=False, these early sells are ignored by the engine")

# -----------------------------------------
# Export signals (ohlcv + indicators + signal column) to CSV
# -----------------------------------------
signals_df = result.signals_df
csv_path = f"signals_{strat.__class__.__name__}_{today}.csv"
# ensure we're not overwriting something sensitive: drop multi-index if present
try:
    signals_df.to_csv(csv_path, index=True)
    print(f"\nSignals exported to {csv_path}")
except Exception as e:
    print(f"Failed to write CSV: {e}")