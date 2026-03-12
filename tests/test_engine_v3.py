#!/usr/bin/env python3
"""
test_engine_v3.py
------------------
Basic test script to verify engine_v3.py works correctly.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, '/workspaces/Algo_Trading_Claude')

from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3
from backtester.order_types import OrderType

# Create sample OHLCV data
def create_sample_data(n_bars=100):
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='D')
    np.random.seed(42)

    # Generate realistic price data
    prices = []
    price = 1000.0
    for i in range(n_bars):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        price *= (1 + change)
        prices.append(price)

    # Create OHLCV from close prices
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = data[-1]['close'] if i > 0 else close * (1 + np.random.normal(0, 0.005))
        volume = np.random.randint(10000, 100000)

        data.append({
            'open': open_price,
            'high': max(open_price, high),
            'low': min(open_price, low),
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data, index=dates)
    return df

# Simple strategy class for testing
class SimpleTestStrategy:
    def __init__(self, fast_period=5, slow_period=20):
        self.name = f"SimpleTestStrategy({fast_period},{slow_period})"
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df):
        """Generate simple crossover signals."""
        df = df.copy()

        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(self.slow_period).mean()

        # Generate signals
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1

        return df

def test_basic_functionality():
    """Test basic backtesting functionality."""
    print("Testing basic functionality...")

    # Create sample data
    df = create_sample_data(200)
    print(f"Created sample data: {len(df)} bars")

    # Create strategy
    strategy = SimpleTestStrategy()

    # Test with basic config (market orders)
    config = BacktestConfigV3(
        initial_capital=100000,
        default_order_type=OrderType.MARKET,
        save_trade_log=False,
        save_raw_data=False,
        save_chart=False
    )

    # Create engine
    engine = BacktestEngineV3(config)

    # Run backtest
    result = engine.run(df, strategy, symbol="TEST")

    # Check results
    print(f"Trades generated: {len(result.trade_log)}")
    print(f"Final portfolio value: {result.equity_curve.dropna().iloc[-1]:.2f}")

    # Check metrics
    metrics = result.metrics_dict()
    print(f"Total return: {metrics.get('Total Return', 'N/A')}")
    print(f"Win rate: {metrics.get('Win Rate', 'N/A')}")

    return result

def test_advanced_features():
    """Test advanced features like limit orders and stops."""
    print("\nTesting advanced features...")

    # Create sample data
    df = create_sample_data(200)

    # Create strategy
    strategy = SimpleTestStrategy()

    # Test with limit orders and stops
    config = BacktestConfigV3(
        initial_capital=100000,
        default_order_type=OrderType.LIMIT,
        limit_offset_pct=0.5,  # 0.5% limit offset
        stop_loss_pct=2.0,     # 2% stop loss
        use_trailing_stop=True,
        trailing_stop_pct=1.5, # 1.5% trailing stop
        save_trade_log=False,
        save_raw_data=False,
        save_chart=False
    )

    # Create engine
    engine = BacktestEngineV3(config)

    # Run backtest
    result = engine.run(df, strategy, symbol="TEST_ADVANCED")

    # Check results
    print(f"Trades generated: {len(result.trade_log)}")
    print(f"Final portfolio value: {result.equity_curve.dropna().iloc[-1]:.2f}")

    return result

def test_portfolio_run():
    """Test multi-symbol portfolio run."""
    print("\nTesting portfolio run...")

    # Create sample data for multiple symbols
    symbols_data = {}
    for symbol in ['INFY', 'RELIANCE', 'TCS']:
        df = create_sample_data(150)
        symbols_data[symbol] = df

    # Create strategy
    strategy = SimpleTestStrategy()

    # Config for portfolio run
    config = BacktestConfigV3(
        initial_capital=300000,  # More capital for portfolio
        default_order_type=OrderType.MARKET,
        save_trade_log=True,
        save_raw_data=True,
        save_chart=False,
        generate_summary=True,
        run_label="test_portfolio"
    )

    # Create engine
    engine = BacktestEngineV3(config)

    # Run portfolio backtest
    results = engine.run_portfolio(symbols_data, strategy, label="test_portfolio")

    # Check results
    total_trades = sum(len(result.trade_log) for result in results.values())
    print(f"Portfolio run completed: {len(results)} symbols, {total_trades} total trades")

    for symbol, result in results.items():
        final_value = result.equity_curve.dropna().iloc[-1]
        print(f"  {symbol}: {len(result.trade_log)} trades, final value: {final_value:.2f}")

    return results

if __name__ == "__main__":
    print("Testing engine_v3.py...")
    print("=" * 50)

    try:
        # Test basic functionality
        basic_result = test_basic_functionality()
        assert len(basic_result.trade_log) > 0, "No trades generated in basic test"
        print("✓ Basic functionality test passed")

        # Test advanced features
        advanced_result = test_advanced_features()
        assert len(advanced_result.trade_log) > 0, "No trades generated in advanced test"
        print("✓ Advanced features test passed")

        # Test portfolio run
        portfolio_results = test_portfolio_run()
        assert len(portfolio_results) > 0, "No portfolio results generated"
        print("✓ Portfolio run test passed")

        print("\n" + "=" * 50)
        print("🎉 All tests passed! engine_v3.py is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)