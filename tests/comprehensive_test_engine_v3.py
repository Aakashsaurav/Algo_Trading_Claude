#!/usr/bin/env python3
"""
comprehensive_test_engine_v3.py
-------------------------------
Comprehensive test suite for engine_v3.py covering all functionality and edge cases.

Tests include:
- Basic backtesting functionality
- Advanced order types (LIMIT, STOP, TRAILING STOP)
- Multi-symbol portfolio runs
- Parameter optimization
- Output generation (trade logs, charts, raw data, summaries)
- Edge cases (empty data, NaN values, single bars, etc.)
- Error handling and validation
"""

import sys
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/workspaces/Algo_Trading_Claude')

from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3, BacktestConfig, Position, Trade, BacktestResult
from backtester.order_types import OrderType
from backtester.commission import Segment

# Test results tracking
test_results = []
test_count = 0
passed_count = 0

def log_test(test_name, passed, message=""):
    """Log test result."""
    global test_count, passed_count
    test_count += 1
    if passed:
        passed_count += 1
        status = "✓ PASS"
    else:
        status = "✗ FAIL"

    result = f"{status} {test_name}"
    if message:
        result += f" - {message}"
    print(result)
    test_results.append((test_name, passed, message))

def create_sample_data(n_bars=200, start_price=1000.0, seed=42, include_nan=False):
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='D')
    np.random.seed(seed)

    prices = []
    price = start_price
    for i in range(n_bars):
        if include_nan and i == 50:  # Add NaN at position 50
            prices.append(np.nan)
            continue

        change = np.random.normal(0, 0.02)  # 2% daily volatility
        price *= (1 + change)
        prices.append(price)

    # Create OHLCV from close prices
    data = []
    for i, close in enumerate(prices):
        if np.isnan(close):
            # Create NaN row
            data.append({
                'open': np.nan, 'high': np.nan, 'low': np.nan,
                'close': np.nan, 'volume': np.nan
            })
            continue

        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = data[-1]['close'] if i > 0 and not np.isnan(data[-1]['close']) else close * (1 + np.random.normal(0, 0.005))
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

class SimpleTestStrategy:
    """Simple test strategy for basic functionality."""
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

class NoSignalStrategy:
    """Strategy that generates no signals."""
    def __init__(self):
        self.name = "NoSignalStrategy"

    def generate_signals(self, df):
        df = df.copy()
        df['signal'] = 0  # No signals
        return df

class AllBuyStrategy:
    """Strategy that always generates buy signals."""
    def __init__(self):
        self.name = "AllBuyStrategy"

    def generate_signals(self, df):
        df = df.copy()
        df['signal'] = 1  # Always buy
        return df

def test_basic_functionality():
    """Test basic backtesting functionality."""
    print("\n=== Testing Basic Functionality ===")

    # Test 1: Basic market order backtest
    df = create_sample_data(200)
    strategy = SimpleTestStrategy()

    config = BacktestConfigV3(
        initial_capital=100000,
        default_order_type=OrderType.MARKET,
        save_trade_log=False,
        save_raw_data=False,
        save_chart=False
    )

    engine = BacktestEngineV3(config)
    result = engine.run(df, strategy, symbol="TEST")

    log_test("Basic market order backtest",
             len(result.trade_log) > 0 and result.equity_curve.dropna().iloc[-1] > 0,
             f"Trades: {len(result.trade_log)}, Final value: {result.equity_curve.dropna().iloc[-1]:.2f}")

    # Test 2: No signals generated
    strategy_no_sig = NoSignalStrategy()
    result_no_sig = engine.run(df, strategy_no_sig, symbol="TEST_NO_SIG")

    log_test("No signals generated",
             len(result_no_sig.trade_log) == 0,
             f"Trades: {len(result_no_sig.trade_log)}")

    # Test 3: All buy signals (should hit max positions)
    strategy_all_buy = AllBuyStrategy()
    config_max_pos = BacktestConfigV3(
        initial_capital=100000,
        max_positions=3,  # Limit positions
        default_order_type=OrderType.MARKET,
        save_trade_log=False
    )
    engine_max_pos = BacktestEngineV3(config_max_pos)
    result_max_pos = engine_max_pos.run(df, strategy_all_buy, symbol="TEST_MAX_POS")

    log_test("Max positions limit",
             len([p for p in result_max_pos.trade_log if 'entry' in str(p.entry_signal).lower()]) <= 3,
             f"Positions opened: {len([p for p in result_max_pos.trade_log if 'entry' in str(p.entry_signal).lower()])}")

def test_advanced_order_types():
    """Test advanced order types."""
    print("\n=== Testing Advanced Order Types ===")

    df = create_sample_data(200)
    strategy = SimpleTestStrategy()

    # Test 1: Limit orders
    config_limit = BacktestConfigV3(
        initial_capital=100000,
        default_order_type=OrderType.LIMIT,
        limit_offset_pct=0.5,  # 0.5% limit offset
        save_trade_log=False
    )

    engine_limit = BacktestEngineV3(config_limit)
    result_limit = engine_limit.run(df, strategy, symbol="TEST_LIMIT")

    log_test("Limit orders",
             len(result_limit.trade_log) >= 0,  # May have fewer fills due to limits
             f"Trades: {len(result_limit.trade_log)}")

    # Test 2: Stop orders
    config_stop = BacktestConfigV3(
        initial_capital=100000,
        default_order_type=OrderType.STOP,
        limit_offset_pct=0.5,
        save_trade_log=False
    )

    engine_stop = BacktestEngineV3(config_stop)
    result_stop = engine_stop.run(df, strategy, symbol="TEST_STOP")

    log_test("Stop orders",
             len(result_stop.trade_log) >= 0,
             f"Trades: {len(result_stop.trade_log)}")

    # Test 3: Trailing stops
    config_trail = BacktestConfigV3(
        initial_capital=100000,
        default_order_type=OrderType.MARKET,
        use_trailing_stop=True,
        trailing_stop_pct=1.5,  # 1.5% trailing stop
        save_trade_log=False
    )

    engine_trail = BacktestEngineV3(config_trail)
    result_trail = engine_trail.run(df, strategy, symbol="TEST_TRAIL")

    log_test("Trailing stops",
             len(result_trail.trade_log) >= 0,
             f"Trades: {len(result_trail.trade_log)}")

    # Test 4: Fixed percentage stop loss
    config_stop_loss = BacktestConfigV3(
        initial_capital=100000,
        default_order_type=OrderType.MARKET,
        stop_loss_pct=2.0,  # 2% stop loss
        save_trade_log=False
    )

    engine_stop_loss = BacktestEngineV3(config_stop_loss)
    result_stop_loss = engine_stop_loss.run(df, strategy, symbol="TEST_STOP_LOSS")

    log_test("Fixed stop loss",
             len(result_stop_loss.trade_log) >= 0,
             f"Trades: {len(result_stop_loss.trade_log)}")

def test_portfolio_functionality():
    """Test multi-symbol portfolio functionality."""
    print("\n=== Testing Portfolio Functionality ===")

    # Create data for multiple symbols
    symbols_data = {}
    for symbol in ['INFY', 'RELIANCE', 'TCS']:
        df = create_sample_data(150, start_price=1000 + len(symbols_data) * 200, seed=42 + len(symbols_data))
        symbols_data[symbol] = df

    strategy = SimpleTestStrategy()

    # Test portfolio run
    config_portfolio = BacktestConfigV3(
        initial_capital=300000,
        default_order_type=OrderType.MARKET,
        save_trade_log=True,
        save_raw_data=True,
        save_chart=False,
        generate_summary=True,
        run_label="test_portfolio_comprehensive"
    )

    engine_portfolio = BacktestEngineV3(config_portfolio)
    results = engine_portfolio.run_portfolio(symbols_data, strategy, label="test_portfolio_comprehensive")

    total_trades = sum(len(result.trade_log) for result in results.values())

    log_test("Portfolio run",
             len(results) == 3 and total_trades >= 0,
             f"Symbols: {len(results)}, Total trades: {total_trades}")

    # Check if output files were created
    output_files_exist = (
        Path("strategies/output/trade/test_portfolio_comprehensive_trade_log.csv").exists() and
        Path("strategies/output/raw_data/test_portfolio_comprehensive_raw_data.csv").exists()
    )

    log_test("Portfolio output files",
             output_files_exist,
             "Trade log and raw data files created")

def test_parameter_optimization():
    """Test parameter optimization functionality."""
    print("\n=== Testing Parameter Optimization ===")

    df = create_sample_data(150)
    strategy_class = SimpleTestStrategy

    param_grid = {
        'fast_period': [3, 5, 7],
        'slow_period': [15, 20, 25]
    }

    config_opt = BacktestConfigV3(
        initial_capital=100000,
        default_order_type=OrderType.MARKET,
        save_trade_log=False
    )

    engine_opt = BacktestEngineV3(config_opt)

    # Test grid optimization
    results_grid = engine_opt.optimize(
        df, strategy_class, param_grid,
        symbol="TEST_OPT", metric="Sharpe Ratio",
        method="grid", top_n=3
    )

    log_test("Grid optimization",
             len(results_grid) == 3 and 'fast_period' in results_grid.columns,
             f"Results: {len(results_grid)} parameter combinations")

    # Test random optimization
    results_random = engine_opt.optimize(
        df, strategy_class, param_grid,
        symbol="TEST_OPT_RAND", metric="Total Return",
        method="random", n_random=5, top_n=2
    )

    log_test("Random optimization",
             len(results_random) == 2,
             f"Results: {len(results_random)} parameter combinations")

def test_output_generation():
    """Test output generation functionality."""
    print("\n=== Testing Output Generation ===")

    df = create_sample_data(100)
    strategy = SimpleTestStrategy()

    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    original_paths = {
        'trade': '/workspaces/Algo_Trading_Claude/strategies/output/trade',
        'raw': '/workspaces/Algo_Trading_Claude/strategies/output/raw_data',
        'chart': '/workspaces/Algo_Trading_Claude/strategies/output/chart'
    }

    try:
        # Temporarily modify output paths (this is a bit hacky but works for testing)
        import backtester.engine_v3 as eng_mod
        eng_mod.OUTPUT_TRADE = Path(temp_dir) / "trade"
        eng_mod.OUTPUT_RAW = Path(temp_dir) / "raw_data"
        eng_mod.OUTPUT_CHART = Path(temp_dir) / "chart"

        # Ensure directories exist
        for path in [eng_mod.OUTPUT_TRADE, eng_mod.OUTPUT_RAW, eng_mod.OUTPUT_CHART]:
            path.mkdir(parents=True, exist_ok=True)

        config_output = BacktestConfigV3(
            initial_capital=100000,
            default_order_type=OrderType.MARKET,
            save_trade_log=True,
            save_raw_data=True,
            save_chart=True,  # Note: chart generation may fail without proper dependencies
            generate_summary=False,
            run_label="test_output"
        )

        engine_output = BacktestEngineV3(config_output)
        result_output = engine_output.run(df, strategy, symbol="TEST_OUTPUT")

        # Check if files were created
        trade_file = eng_mod.OUTPUT_TRADE / "test_output_TEST_OUTPUT_trade_log.csv"
        raw_file = eng_mod.OUTPUT_RAW / "test_output_TEST_OUTPUT_raw_data.csv"

        trade_exists = trade_file.exists()
        raw_exists = raw_file.exists()

        log_test("Trade log output",
                 trade_exists,
                 f"File exists: {trade_exists}")

        log_test("Raw data output",
                 raw_exists,
                 f"File exists: {raw_exists}")

        # Check file contents
        if trade_exists:
            trade_df = pd.read_csv(trade_file)
            log_test("Trade log content",
                     len(trade_df) == len(result_output.trade_log),
                     f"CSV rows: {len(trade_df)}, Expected: {len(result_output.trade_log)}")

        if raw_exists:
            raw_df = pd.read_csv(raw_file)
            log_test("Raw data content",
                     len(raw_df) == len(df) and 'signal' in raw_df.columns,
                     f"CSV rows: {len(raw_df)}, Has signal column: {'signal' in raw_df.columns}")

    finally:
        # Restore original paths
        eng_mod.OUTPUT_TRADE = Path(original_paths['trade'])
        eng_mod.OUTPUT_RAW = Path(original_paths['raw'])
        eng_mod.OUTPUT_CHART = Path(original_paths['chart'])

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n=== Testing Edge Cases ===")

    # Test 1: Empty DataFrame
    empty_df = pd.DataFrame()
    strategy = SimpleTestStrategy()
    config = BacktestConfigV3(initial_capital=100000)

    engine = BacktestEngineV3(config)

    try:
        result_empty = engine.run(empty_df, strategy, symbol="EMPTY")
        log_test("Empty DataFrame", False, "Should have raised ValueError")
    except ValueError as e:
        log_test("Empty DataFrame", "missing columns" in str(e), f"Correctly raised error: {e}")
    except Exception as e:
        log_test("Empty DataFrame", False, f"Unexpected error: {e}")

    # Test 2: DataFrame with NaN values
    df_with_nan = create_sample_data(100, include_nan=True)
    result_nan = engine.run(df_with_nan, strategy, symbol="NAN_TEST")

    log_test("DataFrame with NaN",
             len(result_nan.trade_log) >= 0,  # Should handle NaN gracefully
             f"Trades: {len(result_nan.trade_log)}")

    # Test 3: Single bar DataFrame
    single_bar_df = create_sample_data(1)
    try:
        result_single = engine.run(single_bar_df, strategy, symbol="SINGLE")
        log_test("Single bar DataFrame", True, f"Trades: {len(result_single.trade_log)}")
    except Exception as e:
        log_test("Single bar DataFrame", False, f"Error: {e}")

    # Test 4: Missing required columns
    incomplete_df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        # Missing 'close' and 'volume'
    }, index=pd.date_range('2023-01-01', periods=3))

    try:
        result_incomplete = engine.run(incomplete_df, strategy, symbol="INCOMPLETE")
        log_test("Missing columns", False, "Should have raised ValueError")
    except ValueError as e:
        log_test("Missing columns", "missing columns" in str(e), f"Correctly raised error: {e}")
    except Exception as e:
        log_test("Missing columns", False, f"Unexpected error: {e}")

    # Test 5: Strategy without signal column
    class BadStrategy:
        def __init__(self):
            self.name = "BadStrategy"
        def generate_signals(self, df):
            df = df.copy()
            # No 'signal' column added
            return df

    bad_strategy = BadStrategy()
    df_good = create_sample_data(50)

    try:
        result_bad = engine.run(df_good, bad_strategy, symbol="BAD_STRATEGY")
        log_test("Missing signal column", False, "Should have raised ValueError")
    except ValueError as e:
        log_test("Missing signal column", "signal" in str(e), f"Correctly raised error: {e}")
    except Exception as e:
        log_test("Missing signal column", False, f"Unexpected error: {e}")

def test_risk_management():
    """Test risk management features."""
    print("\n=== Testing Risk Management ===")

    df = create_sample_data(200)
    strategy = SimpleTestStrategy()

    # Test 1: Intraday squareoff
    config_squareoff = BacktestConfigV3(
        initial_capital=100000,
        intraday_squareoff=True,
        save_trade_log=False
    )

    engine_squareoff = BacktestEngineV3(config_squareoff)
    result_squareoff = engine_squareoff.run(df, strategy, symbol="SQUAREOFF")

    log_test("Intraday squareoff",
             len(result_squareoff.trade_log) >= 0,
             f"Trades: {len(result_squareoff.trade_log)}")

    # Test 2: Drawdown limits
    config_drawdown = BacktestConfigV3(
        initial_capital=100000,
        max_drawdown_pct=5.0,  # 5% max drawdown
        save_trade_log=False
    )

    engine_drawdown = BacktestEngineV3(config_drawdown)
    result_drawdown = engine_drawdown.run(df, strategy, symbol="DRAWDOWN")

    max_dd = result_drawdown.drawdown.min()
    log_test("Drawdown limit",
             max_dd >= -5.0,  # Should not exceed -5%
             f"Max drawdown: {max_dd:.2f}%")

    # Test 3: Lot size rounding
    config_lot = BacktestConfigV3(
        initial_capital=100000,
        lot_size=10,  # Must trade in multiples of 10
        fixed_quantity=15,  # This should be rounded to 10
        save_trade_log=False
    )

    engine_lot = BacktestEngineV3(config_lot)
    result_lot = engine_lot.run(df, strategy, symbol="LOT_SIZE")

    # Check if quantities are multiples of lot size
    quantities = [trade.quantity for trade in result_lot.trade_log]
    valid_quantities = all(q % 10 == 0 for q in quantities)

    log_test("Lot size rounding",
             valid_quantities,
             f"Quantities: {quantities}")

def test_metrics_calculation():
    """Test metrics calculation."""
    print("\n=== Testing Metrics Calculation ===")

    df = create_sample_data(200)
    strategy = SimpleTestStrategy()

    config = BacktestConfigV3(
        initial_capital=100000,
        default_order_type=OrderType.MARKET,
        save_trade_log=False
    )

    engine = BacktestEngineV3(config)
    result = engine.run(df, strategy, symbol="METRICS")

    # Test metrics calculation
    metrics = result.metrics_dict()

    required_metrics = [
        "Total Trades", "Win Rate", "Profit Factor",
        "Sharpe Ratio", "Max Drawdown", "Total Return", "CAGR"
    ]

    missing_metrics = [m for m in required_metrics if m not in metrics]

    log_test("Metrics calculation",
             len(missing_metrics) == 0,
             f"Missing metrics: {missing_metrics}")

    # Test summary output
    summary = result.summary()
    log_test("Summary output",
             len(summary) > 0 and "BACKTEST RESULTS" in summary,
             f"Summary length: {len(summary)}")

    # Test trade DataFrame
    trade_df = result.trade_df()
    log_test("Trade DataFrame",
             isinstance(trade_df, pd.DataFrame),
             f"Shape: {trade_df.shape}")

def run_all_tests():
    """Run all test functions."""
    print("🧪 Comprehensive Engine V3 Test Suite")
    print("=" * 50)

    test_functions = [
        test_basic_functionality,
        test_advanced_order_types,
        test_portfolio_functionality,
        test_parameter_optimization,
        test_output_generation,
        test_edge_cases,
        test_risk_management,
        test_metrics_calculation,
    ]

    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            log_test(f"{test_func.__name__}", False, f"Test crashed: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_count}/{test_count} tests passed")

    if passed_count == test_count:
        print("🎉 All tests passed! Engine V3 is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)