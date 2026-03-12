#!/usr/bin/env python3
"""
simple_test_engine_v3.py
------------------------
Simple test script to verify engine_v3.py basic functionality.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/workspaces/Algo_Trading_Claude')

def create_sample_data(n_bars=100):
    """Create sample OHLCV data."""
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='D')
    np.random.seed(42)

    prices = []
    price = 1000.0
    for i in range(n_bars):
        change = np.random.normal(0, 0.02)
        price *= (1 + change)
        prices.append(price)

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

    return pd.DataFrame(data, index=dates)

class SimpleStrategy:
    def __init__(self):
        self.name = "SimpleStrategy"

    def generate_signals(self, df):
        df = df.copy()
        df['fast_ma'] = df['close'].rolling(5).mean()
        df['slow_ma'] = df['close'].rolling(20).mean()
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1
        return df

def test_basic_import():
    """Test if engine_v3 can be imported."""
    try:
        from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3
        from backtester.order_types import OrderType
        print("✓ Import successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_backtest():
    """Test basic backtesting functionality."""
    try:
        from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3
        from backtester.order_types import OrderType

        # Create test data
        df = create_sample_data(100)
        strategy = SimpleStrategy()

        # Create config
        config = BacktestConfigV3(
            initial_capital=100000,
            default_order_type=OrderType.MARKET,
            save_trade_log=False,
            save_raw_data=False,
            save_chart=False
        )

        # Create engine and run backtest
        engine = BacktestEngineV3(config)
        result = engine.run(df, strategy, symbol="TEST")

        # Check results
        trades = len(result.trade_log)
        final_value = result.equity_curve.dropna().iloc[-1]

        print(f"✓ Basic backtest successful: {trades} trades, final value: {final_value:.2f}")
        return True

    except Exception as e:
        print(f"✗ Basic backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_features():
    """Test advanced features."""
    try:
        from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3
        from backtester.order_types import OrderType

        df = create_sample_data(100)
        strategy = SimpleStrategy()

        # Test limit orders
        config_limit = BacktestConfigV3(
            initial_capital=100000,
            default_order_type=OrderType.LIMIT,
            limit_offset_pct=0.5,
            save_trade_log=False
        )

        engine = BacktestEngineV3(config_limit)
        result = engine.run(df, strategy, symbol="TEST_LIMIT")

        print(f"✓ Limit orders test successful: {len(result.trade_log)} trades")
        return True

    except Exception as e:
        print(f"✗ Advanced features test failed: {e}")
        return False

def test_portfolio():
    """Test portfolio functionality."""
    try:
        from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3

        # Create data for multiple symbols
        symbols_data = {
            'INFY': create_sample_data(50),
            'RELIANCE': create_sample_data(50)
        }
        strategy = SimpleStrategy()

        config = BacktestConfigV3(
            initial_capital=200000,
            save_trade_log=False,
            save_raw_data=False,
            save_chart=False
        )

        engine = BacktestEngineV3(config)
        results = engine.run_portfolio(symbols_data, strategy)

        total_trades = sum(len(result.trade_log) for result in results.values())
        print(f"✓ Portfolio test successful: {len(results)} symbols, {total_trades} total trades")
        return True

    except Exception as e:
        print(f"✗ Portfolio test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases."""
    try:
        from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3

        # Test empty dataframe
        config = BacktestConfigV3(initial_capital=100000)
        engine = BacktestEngineV3(config)
        strategy = SimpleStrategy()

        try:
            empty_df = pd.DataFrame()
            result = engine.run(empty_df, strategy, symbol="EMPTY")
            print("✗ Empty dataframe test failed: should have raised error")
            return False
        except ValueError:
            print("✓ Empty dataframe correctly raises ValueError")

        # Test dataframe with NaN
        df_nan = create_sample_data(10)
        df_nan.loc[5, 'close'] = np.nan
        result_nan = engine.run(df_nan, strategy, symbol="NAN")
        print(f"✓ NaN handling test successful: {len(result_nan.trade_log)} trades")

        return True

    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Simple Engine V3 Test Suite")
    print("=" * 40)

    tests = [
        ("Import Test", test_basic_import),
        ("Basic Backtest", test_basic_backtest),
        ("Advanced Features", test_advanced_features),
        ("Portfolio Test", test_portfolio),
        ("Edge Cases", test_edge_cases),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Engine V3 is working correctly.")
        return True
    else:
        print("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)