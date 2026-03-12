#!/usr/bin/env python3
"""
test_ema_crossover_reliance.py
-------------------------------
Backtest EMACrossover strategy on RELIANCE daily data for the last 15 years.

BACKTEST PARAMETERS:
  • Symbol: RELIANCE
  • Strategy: EMACrossover (fast_period=10, slow_period=21)
  • Data: Daily OHLCV for last 15 years
  • Position Sizing: 50% of available capital at any point
  • Max Positions: 1 open position at a time
  • Execution: Next bar opening (order=MARKET)
  • Initial Capital: Rs 500,000

EXPECTED BEHAVIOR:
  ✓ Buy signal when fast EMA crosses above slow EMA
  ✓ Sell signal when fast EMA crosses below slow EMA
  ✓ Position size = (Available Capital * 0.5) / Entry Price
  ✓ Maximum 1 open position at any time
  ✓ Execution at next bar's opening price
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/workspaces/Algo_Trading_Claude')

import pandas as pd
import logging

from data.parquet_store import parquet_store
from strategies.momentum.ema_crossover import EMACrossoverStrategy
from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3
from backtester.order_types import OrderType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_reliance_data(years: int = 15) -> pd.DataFrame:
    """
    Load RELIANCE daily OHLCV data for the specified number of years.
    
    Args:
        years (int): Number of years of historical data to load. Defaults to 15.
    
    Returns:
        pd.DataFrame: OHLCV data indexed by datetime.
        
    Raises:
        ValueError: If no data is found.
    """
    logger.info(f"Loading RELIANCE daily data for last {years} years...")
    
    # Calculate date range
    to_date = datetime.now().date().isoformat()
    from_date = (datetime.now() - timedelta(days=years * 365)).date().isoformat()
    
    logger.info(f"Date range: {from_date} to {to_date}")
    
    # Load from parquet store
    df = parquet_store.load_daily(
        exchange="NSE_EQ",
        symbol="RELIANCE",
        from_date=from_date,
        to_date=to_date
    )
    
    if df.empty:
        raise ValueError(
            "No RELIANCE data found in parquet store. "
            "Please fetch the data first using data_fetcher.fetch_and_save_daily()"
        )
    
    logger.info(f"Loaded {len(df)} daily bars for RELIANCE")
    logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    return df


def run_backtest():
    """
    Run EMACrossover backtest on RELIANCE with specified parameters.
    """
    try:
        # Load data
        df = load_reliance_data(years=15)
        
        # Configure backtest
        config = BacktestConfigV3(
            # Capital and sizing
            initial_capital=500_000.0,              # Rs 500,000 initial capital
            capital_risk_pct=0.50,                  # Use 50% of available capital per trade
            
            # Position management
            max_positions=1,                        # Maximum 1 open position at a time
            fixed_quantity=0,                       # 0 = calculate from capital_risk_pct
            
            # Order execution
            default_order_type=OrderType.MARKET,    # Market orders at next bar open
            
            # Risk management
            max_drawdown_pct=0.50,                  # Stop if drawdown exceeds 50%
            
            # Output flags
            save_trade_log=True,                    # Save trade log
            save_raw_data=True,                     # Save raw signals/data
            save_chart=True,                        # Generate chart
            generate_summary=True,                  # Print summary to console
            run_label="ema_crossover_reliance",     # Output file label
        )
        
        # Create strategy instance with default parameters
        strategy = EMACrossoverStrategy(
            params={
                "fast_period": 10,
                "slow_period": 21,
                "atr_period": 14,
                "atr_multiplier": 2.0,
            }
        )
        
        logger.info(f"Strategy: {strategy.name}")
        logger.info(f"Parameters: fast_period=10, slow_period=21")
        logger.info(f"Position sizing: 50% of available capital")
        logger.info(f"Max positions: 1")
        logger.info("")
        
        # Create engine and run backtest
        engine = BacktestEngineV3(config)
        result = engine.run(df, strategy, symbol="RELIANCE")
        
        # Print detailed summary
        print(result.summary())
        
        # Print trade log if trades were generated
        trade_df = result.trade_df()
        if not trade_df.empty:
            print("\n" + "=" * 120)
            print("TRADE LOG (First 10 trades)")
            print("=" * 120)
            print(trade_df.head(10).to_string())
            
            if len(trade_df) > 10:
                print(f"\n... and {len(trade_df) - 10} more trades")
        else:
            logger.warning("No trades were generated during backtest.")
        
        # Return result for further analysis if needed
        return result
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" RELIANCE EMACrossover Backtest (15 Years Daily Data)")
    print("=" * 80 + "\n")
    
    result = run_backtest()
    
    print("\n" + "=" * 80)
    print(" Backtest Complete")
    print("=" * 80 + "\n")
    
    # Output file locations
    logger.info("Output files:")
    logger.info("  • Trade Log: strategies/output/trade/ema_crossover_reliance_trade_log.csv")
    logger.info("  • Raw Data:  strategies/output/raw_data/ema_crossover_reliance_raw_data.csv")
    logger.info("  • Chart:     strategies/output/chart/ema_crossover_reliance_chart.html")
