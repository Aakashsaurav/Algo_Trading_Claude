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