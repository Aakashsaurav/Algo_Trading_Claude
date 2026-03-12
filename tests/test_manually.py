
#Test 1: broker/upstox/data_manager.py -> get_ohlcv() -> check if we have minute data for RELIANCE
from broker.upstox.data_manager import get_ohlcv
df = get_ohlcv(
      instrument_type="EQUITY", exchange="NSE", trading_symbol="RELIANCE",
      unit="days", interval=1, from_date="2015-01-03"
    )
#print(df.head(10))

'''
#Test2
from data.universe import universe_manager

nifty500 = universe_manager.get_nifty500()
fo_stocks = universe_manager.get_fo_stocks()

print(nifty500)
print(fo_stocks)
'''
'''
#Test 3
from data.stock_universe import stock_universe_manager
stocks = stock_universe_manager.get_nifty500_detailed()
print(stocks)
'''
'''
#Test 4
from indicators.technical import sma, ema, rsi
import pandas as pd
from broker.upstox.data_manager import get_ohlcv
df = get_ohlcv(
      instrument_type="EQUITY", exchange="NSE", trading_symbol="SBIN",
      unit="hours", interval=1, from_date="2020-01-03"
    )

df_sma = sma(df['close'], 20)
print(df_sma.head)
'''
'''
#Test 5
from indicators.bridge import IndicatorBridge
bridge = IndicatorBridge()
rsi = bridge.rsi(df['close'], 14, library='pandas_ta')
print(rsi)
'''
'''
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

#Test 6
def test_ema_crossover_strategy():
    """
    Test the EMACrossover strategy on RELIANCE data.
    Generates:
    1. CSV with OHLCV data, indicator values, and buy/sell signals (1/-1)
    2. CSV with trade log
    3. CSV with final report (metrics)
    4. PNG chart of trades like TradingView
    """
    import pandas as pd
    from broker.upstox.data_manager import get_ohlcv
    from backtester.engine import BacktestEngine, BacktestConfig
    from backtester.commission import Segment
    from strategies.base import EMACrossover
    from backtester.report import generate_report
    import os

    # Get data
    print("Fetching data...")
    df = get_ohlcv(
        instrument_type="EQUITY", 
        exchange="NSE", 
        trading_symbol="INFY",
        unit="minutes", 
        interval=5, 
        from_date="2022-01-01"
    )

    if df.empty:
        raise RuntimeError("No data available for RELIANCE")

    print(f"Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Create strategy
    strat = EMACrossover(fast_period=9, slow_period=21)
    print(f"Strategy: {strat.name}")

    # Create backtest config
    config = BacktestConfig(
        initial_capital=500_000,
        segment=Segment.EQUITY_DELIVERY,
        capital_risk_pct=0.1,
        allow_shorting=False,
        max_positions=5,
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(df, strat, symbol="RELIANCE-5m")
    
    print("Backtest completed!")
    print(result.summary())

    # Create output directory
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export signals CSV (OHLCV + indicators + signals)
    signals_csv_path = os.path.join(output_dir, "signals_ema_crossover.csv")
    result.signals_df.to_csv(signals_csv_path, index=True)
    print(f"Signals CSV exported to: {signals_csv_path}")

    # 2. Export trade log CSV
    trade_log_csv_path = os.path.join(output_dir, "trade_log_ema_crossover.csv")
    trade_df = result.trade_df()
    if not trade_df.empty:
        trade_df.to_csv(trade_log_csv_path, index=False)
        print(f"Trade log CSV exported to: {trade_log_csv_path}")
    else:
        print("No trades generated - trade log CSV not created")

    # 3. Export final report CSV (metrics)
    metrics_csv_path = os.path.join(output_dir, "report_ema_crossover.csv")
    metrics_df = pd.DataFrame(list(result.metrics_dict().items()), columns=['Metric', 'Value'])
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Report CSV exported to: {metrics_csv_path}")

    # 4. Generate PNG chart
    png_path = os.path.join(output_dir, "trades_chart_ema_crossover.png")
    try:
        generate_report(result, symbol="RELIANCE-5m", output_dir=output_dir, 
                       filename="trades_chart_ema_crossover.png", show=False)
        print(f"Trades chart PNG saved to: {png_path}")
    except Exception as e:
        print(f"Failed to generate PNG chart: {e}")

    print("\nAll outputs saved to directory:", output_dir)
'''
'''
#Test 7
def test_rsi_supertrend_rs_strategy():
    """
    Comprehensive backtest of RSI + Supertrend + Relative Strength strategy.
    
    This strategy goes LONG when ALL THREE conditions are met:
      1. RSI(14) > 50 (momentum positive)
      2. Supertrend direction = +1 (bullish trend)
      3. Relative Strength vs Nifty > 0 (stock outperforming benchmark)
    
    EXITS when ANY condition fails.
    
    Generates:
      1. Signals CSV (OHLCV + indicators + signals)
      2. Trade log CSV
      3. Metrics report CSV
      4. PNG chart
    """
    import pandas as pd
    from broker.upstox.data_manager import get_ohlcv
    from backtester.engine_v2 import BacktestEngineV2, BacktestConfigV2
    from backtester.commission import Segment
    from backtester.order_types import OrderType
    from strategies.base_strategy import RSISupertrendRelativeStrength
    from backtester.report import generate_report
    import os

    # Get data for stock
    print("Fetching stock data (RELIANCE)...")
    stock_df = get_ohlcv(
        instrument_type="EQUITY",
        exchange="NSE",
        trading_symbol="RELIANCE",
        unit="days",
        interval=1,
        from_date="2007-01-01"
    )
    
    if stock_df.empty:
        raise RuntimeError("No data available for RELIANCE")
    
    print(f"Stock data loaded: {len(stock_df)} bars from {stock_df.index[0]} to {stock_df.index[-1]}")

    # Get data for benchmark (Nifty 50)
    print("Fetching benchmark data (Nifty 50)...")
    nifty_df = get_ohlcv(
        instrument_type="INDEX",
        exchange="NSE",
        trading_symbol="NIFTY",
        unit="days",
        interval=1,
        from_date="2007-01-01"
    )
    
    if nifty_df.empty:
        raise RuntimeError("No data available for NIFTY_50")
    
    print(f"Nifty data loaded: {len(nifty_df)} bars")

    # Merge stock and benchmark on timestamp
    print("Merging stock and benchmark data...")
    
    # Ensure both dataframes have datetime index for proper alignment
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        if 'timestamp' in stock_df.columns:
            stock_df = stock_df.set_index('timestamp')
        else:
            stock_df = stock_df.reset_index().set_index('timestamp')
            
    if not isinstance(nifty_df.index, pd.DatetimeIndex):
        if 'timestamp' in nifty_df.columns:
            nifty_df = nifty_df.set_index('timestamp')
        else:
            nifty_df = nifty_df.reset_index().set_index('timestamp')
    
    # Merge on index (timestamp) with inner join to keep only dates in BOTH datasets
    # Use 'inner' to ensure we only process dates where both stock & benchmark exist
    merged_df = stock_df.merge(
        nifty_df[['close']].rename(columns={'close': 'bench_close'}),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # Get date ranges for reporting
    stock_start = stock_df.index[0] if isinstance(stock_df.index, pd.DatetimeIndex) else pd.to_datetime(stock_df.index[0])
    stock_end = stock_df.index[-1] if isinstance(stock_df.index, pd.DatetimeIndex) else pd.to_datetime(stock_df.index[-1])
    nifty_start = nifty_df.index[0] if isinstance(nifty_df.index, pd.DatetimeIndex) else pd.to_datetime(nifty_df.index[0])
    nifty_end = nifty_df.index[-1] if isinstance(nifty_df.index, pd.DatetimeIndex) else pd.to_datetime(nifty_df.index[-1])
    merged_start = merged_df.index[0] if isinstance(merged_df.index, pd.DatetimeIndex) else pd.to_datetime(merged_df.index[0])
    merged_end = merged_df.index[-1] if isinstance(merged_df.index, pd.DatetimeIndex) else pd.to_datetime(merged_df.index[-1])
    
    print(f"Stock data range: {stock_start.date()} to {stock_end.date()} ({len(stock_df)} bars)")
    print(f"Benchmark data range: {nifty_start.date()} to {nifty_end.date()} ({len(nifty_df)} bars)")
    print(f"Merged overlap: {merged_start.date()} to {merged_end.date()} ({len(merged_df)} bars)")
    
    # Verify we have enough data
    if len(merged_df) == 0:
        raise RuntimeError(
            "No overlapping dates between stock and benchmark data. "
            f"Stock: {stock_start} to {stock_end}, "
            f"Benchmark: {nifty_start} to {nifty_end}"
        )

    # Create strategy with RS lookback = 55 bars
    strat = RSISupertrendRelativeStrength(
        rsi_period=14,
        super_period=10,
        super_multiplier=3.0,
        rs_period=55
    )
    print(f"Strategy: {strat.name}")

    # Create backtest config with engine_v2 features
    config = BacktestConfigV2(
        initial_capital=500_000,
        segment=Segment.EQUITY_DELIVERY,
        capital_risk_pct=0.02,
        allow_shorting=False,
        max_positions=1,
        # engine_v2 specific options
        default_order_type=OrderType.MARKET,
        save_trade_log=True,
        save_raw_data=True,
        save_chart=True,
        run_label="rsi_st_rs_backtest",
    )

    # Run backtest
    print("Running backtest with engine_v2...")
    engine = BacktestEngineV2(config)
    result = engine.run(merged_df, strat, symbol="RELIANCE")
    
    print("Backtest completed!")
    print(result.summary())

    # Create output directory
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export signals CSV
    signals_csv_path = os.path.join(output_dir, "signals_rsi_st_rs.csv")
    result.signals_df.to_csv(signals_csv_path, index=True)
    print(f"Signals CSV exported to: {signals_csv_path}")

    # 2. Export trade log CSV
    trade_log_csv_path = os.path.join(output_dir, "trade_log_rsi_st_rs.csv")
    trade_df = result.trade_df()
    if not trade_df.empty:
        trade_df.to_csv(trade_log_csv_path, index=False)
        print(f"Trade log CSV exported to: {trade_log_csv_path}")
        print(f"\nTrades summary:")
        print(trade_df[["entry_time", "exit_time", "entry_price", "exit_price", 
                        "quantity", "net_pnl", "pnl_pct", "duration_bars"]].head())
    else:
        print("No trades generated - trade log CSV not created")

    # 3. Export metrics report CSV
    metrics_csv_path = os.path.join(output_dir, "report_rsi_st_rs.csv")
    metrics_df = pd.DataFrame(list(result.metrics_dict().items()), columns=['Metric', 'Value'])
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Report CSV exported to: {metrics_csv_path}")

    # 4. Generate PNG chart
    png_path = os.path.join(output_dir, "trades_chart_rsi_st_rs.png")
    try:
        # Use max_candles=None to show the entire trade duration
        generate_report(result, symbol="RELIANCE", output_dir=output_dir,
                       filename="trades_chart_rsi_st_rs.png", show=False, max_candles=4670)
        print(f"Trades chart PNG saved to: {png_path}")
    except Exception as e:
        print(f"Failed to generate PNG chart: {e}")

    print("\nAll outputs saved to directory:", output_dir)


# Uncomment the line below to run the test
# test_ema_crossover_strategy()
test_rsi_supertrend_rs_strategy()
'''
'''
#Test 8
from broker.upstox.auth import AuthManager
auth = AuthManager()

# Step 1: Get the login URL and open it in browser
url = auth.get_login_url()

# Step 2: After redirect, paste the full redirect URL or just the code
token = auth.generate_token(auth_code="BEuoBt")
https://127.0.0.1:5000/?code=
token = auth.generate_token_from_url("https://api-v2.upstox.com/login/authorization/redirect?code=J4AqmJ&ucc=728042")
# All subsequent uses — just get a valid token:
token = auth.get_valid_token()
'''
#Test 9
from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3
from backtester.order_types import OrderType
from strategies.base_strategy_github import EMACrossover, RSIMeanReversion

config = BacktestConfigV3(
    initial_capital    = 500_000,
    default_order_type = OrderType.MARKET,
    capital_risk_pct   = 0.50,
    max_positions      = 1,
    fixed_quantity     = 0,
    max_drawdown_pct   = 0.50,
    save_trade_log     = True,
    save_raw_data      = True,
    save_chart         = True,
    generate_summary   = True,
    run_label          = "ema_crossover_reliance",
    #limit_offset_pct   = 0.2,
    #stop_loss_pct      = 2.0,
    #trailing_stop_pct  = 1.5,
    )
engine   = BacktestEngineV3(config)
strat = EMACrossover(fast_period=9, slow_period=21)

# Single symbol
result = engine.run(df, strat, symbol="RELIANCE")

# Multi-symbol portfolio
#results = engine.run_portfolio({symbol: df_dict[symbol] for symbol in symbols}, strategy)

# Parameter optimization
#param_grid = {'fast_period': [5,9,13], 'slow_period': [21,34,50]}
#top_results = engine.optimize(df, StrategyClass, param_grid, symbol='INFY')