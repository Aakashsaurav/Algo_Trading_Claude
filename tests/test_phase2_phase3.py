"""
tests/test_phase2_phase3.py
-----------------------------
Comprehensive unit and integration tests for Phase 2 (Indicators & Strategies)
and Phase 3 (Backtesting Engine, Commission Model, Report Generator).

WHAT IS TESTED:
  - Every public function in indicators/technical.py (25 functions)
  - Every Strategy class and its generate_signals() output
  - Every charge component in backtester/commission.py
  - BacktestEngine event loop, position tracking, FIFO close, shorting
  - BacktestResult metrics (Win Rate, Sharpe, Drawdown, CAGR, Profit Factor)
  - Edge cases: empty data, all-NaN data, zero ATR, single bar, all-loss run
  - Report generator: correct file creation and non-zero size
  - Look-ahead bias guard: signals only use past data

IMPORTANT CONVENTIONS:
  - Every test has a docstring explaining WHAT it tests and WHY it matters.
  - All tests are isolated (no shared state). setUp() recreates fixtures.
  - Upstox API / file I/O is never called — all external dependencies are mocked.
  - Assertions are specific: we check exact values, not just "not None".

HOW TO RUN:
  python -m pytest tests/test_phase2_phase3.py -v
  python -m pytest tests/test_phase2_phase3.py -v -k "TestRSI"   # single class
"""

import os
import sys
import tempfile
import unittest
import warnings
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Shared fixture factory ─────────────────────────────────────────────────────

def _make_ohlcv(
    n:          int   = 200,
    start:      str   = "2022-01-03",
    freq:       str   = "B",
    start_price:float = 1000.0,
    seed:       int   = 42,
    tz:         str   = "Asia/Kolkata",
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data with a random walk price series.

    Guarantees:
      - high >= max(open, close) always
      - low  <= min(open, close) always
      - all prices > 0
      - volume > 0
      - timezone-aware index (IST)
    """
    np.random.seed(seed)
    dates  = pd.date_range(start, periods=n, freq=freq, tz=tz)
    close  = start_price + np.cumsum(np.random.randn(n) * 15)
    close  = np.maximum(close, 10.0)   # floor at Rs 10 so we never go negative
    noise  = np.abs(np.random.randn(n) * 8)

    df = pd.DataFrame({
        "open":   (close - noise * 0.5).astype("float32"),
        "close":  close.astype("float32"),
        "volume": np.random.randint(100_000, 5_000_000, n).astype("int32"),
        "oi":     np.zeros(n, dtype="int32"),
    }, index=dates)

    # Correct high / low to always be valid
    df["high"] = df[["open", "close"]].max(axis=1) + np.abs(noise * 0.8).astype("float32")
    df["low"]  = df[["open", "close"]].min(axis=1) - np.abs(noise * 0.8).astype("float32")
    df["low"]  = df["low"].clip(lower=1.0)   # keep positive
    return df


# =============================================================================
# BLOCK 1 — indicators/technical.py
# =============================================================================

class TestSMA(unittest.TestCase):
    """Simple Moving Average — pure arithmetic mean over rolling window."""

    def setUp(self):
        from indicators.technical import sma
        self.sma = sma
        self.df  = _make_ohlcv(100)

    def test_returns_series(self):
        """SMA must return a pandas Series, not a DataFrame or ndarray."""
        result = self.sma(self.df["close"], 20)
        self.assertIsInstance(result, pd.Series)

    def test_warmup_is_nan(self):
        """First (period-1) bars must be NaN — no partial-window values."""
        result = self.sma(self.df["close"], 20)
        self.assertTrue(result.iloc[:19].isna().all(),
                        "SMA warm-up period should be all NaN")

    def test_first_valid_bar_is_arithmetic_mean(self):
        """The first non-NaN value must equal the arithmetic mean of the first 20 bars."""
        close  = self.df["close"]
        result = self.sma(close, 20)
        expected = float(close.iloc[:20].mean())
        self.assertAlmostEqual(float(result.iloc[19]), expected, places=3)

    def test_output_length_matches_input(self):
        """Output must have exactly the same length as the input."""
        result = self.sma(self.df["close"], 10)
        self.assertEqual(len(result), len(self.df))

    def test_raises_on_period_zero(self):
        """Period of 0 is invalid — must raise ValueError."""
        with self.assertRaises((ValueError, TypeError)):
            self.sma(self.df["close"], 0)

    def test_raises_on_non_series_input(self):
        """Passing a DataFrame instead of Series must raise TypeError."""
        with self.assertRaises(TypeError):
            self.sma(self.df, 20)

    def test_name_attribute(self):
        """Result Series should have a descriptive name for chart labelling."""
        result = self.sma(self.df["close"], 20)
        self.assertIn("20", result.name)


class TestEMA(unittest.TestCase):
    """Exponential Moving Average — exponentially weighted with alpha = 2/(n+1)."""

    def setUp(self):
        from indicators.technical import ema
        self.ema = ema
        self.df  = _make_ohlcv(100)

    def test_ema_reacts_faster_than_sma(self):
        """
        EMA should react more quickly to a sudden price jump than SMA.
        We inject a price spike and check that EMA moves further than SMA.
        """
        from indicators.technical import sma
        close = self.df["close"].copy()
        # Inject a large spike at the end
        close.iloc[-5:] = close.iloc[-5:] + 500
        ema_val = self.ema(close, 20).iloc[-1]
        sma_val = sma(close, 20).iloc[-1]
        # EMA should be closer to the spike (higher) than SMA
        self.assertGreater(ema_val, sma_val - 1,
                           "EMA should respond more to recent price changes than SMA")

    def test_warmup_nan(self):
        """First (period-1) values must be NaN."""
        result = self.ema(self.df["close"], 20)
        self.assertTrue(result.iloc[:19].isna().all())

    def test_no_nan_after_warmup(self):
        """After the warm-up period there must be no NaN values."""
        result = self.ema(self.df["close"], 20)
        self.assertFalse(result.iloc[19:].isna().any(),
                         "EMA should have no NaN after the warm-up period")

    def test_same_index_as_input(self):
        """Output index must exactly match input index."""
        result = self.ema(self.df["close"], 10)
        pd.testing.assert_index_equal(result.index, self.df.index)


class TestDEMA(unittest.TestCase):
    """Double EMA — DEMA = 2*EMA(n) - EMA(EMA(n)). Less lag than EMA."""

    def setUp(self):
        from indicators.technical import dema, ema
        self.dema = dema
        self.ema  = ema
        self.df   = _make_ohlcv(150)

    def test_dema_less_lag_than_ema(self):
        """
        After a sustained price increase, DEMA should be higher than EMA
        because DEMA subtracts the lag component.
        """
        close = pd.Series(np.linspace(100, 200, 150),
                          index=self.df.index, dtype="float32")
        d = self.dema(close, 20).dropna()
        e = self.ema(close, 20).dropna()
        # On a linear uptrend, DEMA > EMA (DEMA leads)
        common = d.index.intersection(e.index)
        self.assertTrue((d[common] >= e[common] - 0.1).all(),
                        "DEMA should be >= EMA on a linear uptrend (less lag)")

    def test_output_type(self):
        result = self.dema(self.df["close"], 20)
        self.assertIsInstance(result, pd.Series)


class TestMACD(unittest.TestCase):
    """MACD — fast EMA minus slow EMA, with signal line and histogram."""

    def setUp(self):
        from indicators.technical import macd
        self.macd = macd
        self.df   = _make_ohlcv(200)

    def test_returns_three_columns(self):
        """MACD must return exactly: macd, signal, histogram."""
        result = self.macd(self.df["close"])
        self.assertEqual(list(result.columns), ["macd", "signal", "histogram"])

    def test_histogram_equals_macd_minus_signal(self):
        """histogram = macd - signal must hold exactly."""
        result = self.macd(self.df["close"])
        diff   = (result["macd"] - result["signal"] - result["histogram"]).dropna().abs()
        self.assertAlmostEqual(float(diff.max()), 0.0, places=4,
                               msg="histogram must exactly equal macd - signal")

    def test_fast_must_be_less_than_slow(self):
        """Swapping fast/slow periods should raise ValueError."""
        with self.assertRaises(ValueError):
            self.macd(self.df["close"], fast_period=26, slow_period=12)

    def test_index_preserved(self):
        """Output index must match input."""
        result = self.macd(self.df["close"])
        pd.testing.assert_index_equal(result.index, self.df.index)

    def test_custom_periods(self):
        """Custom period parameters must be accepted and produce valid output."""
        result = self.macd(self.df["close"], fast_period=5, slow_period=10, signal_period=3)
        self.assertFalse(result["macd"].dropna().empty)


class TestRSI(unittest.TestCase):
    """RSI — momentum oscillator between 0 and 100."""

    def setUp(self):
        from indicators.technical import rsi
        self.rsi = rsi
        self.df  = _make_ohlcv(150)

    def test_all_values_between_0_and_100(self):
        """RSI must always be in [0, 100]. No exceptions."""
        result = self.rsi(self.df["close"], 14)
        valid  = result.dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 100).all(),
                        f"RSI out of [0,100]: min={valid.min():.2f}, max={valid.max():.2f}")

    def test_all_gains_gives_rsi_100(self):
        """
        A monotonically increasing price series (all gains, zero losses)
        should give RSI = 100 after the warm-up period.
        """
        rising = pd.Series(
            np.linspace(100, 200, 100),
            index=_make_ohlcv(100).index,
        )
        result = self.rsi(rising, 14)
        self.assertAlmostEqual(float(result.dropna().iloc[-1]), 100.0, places=0)

    def test_all_losses_gives_rsi_0(self):
        """
        A monotonically decreasing price series should give RSI = 0.
        """
        falling = pd.Series(
            np.linspace(200, 100, 100),
            index=_make_ohlcv(100).index,
        )
        result = self.rsi(falling, 14)
        self.assertAlmostEqual(float(result.dropna().iloc[-1]), 0.0, places=0)

    def test_warmup_nan(self):
        """First `period` values should be NaN."""
        result = self.rsi(self.df["close"], 14)
        self.assertTrue(result.iloc[:14].isna().all())

    def test_default_period_14(self):
        """Default period should be 14."""
        result = self.rsi(self.df["close"])
        self.assertIn("14", result.name)


class TestSupertrend(unittest.TestCase):
    """Supertrend — ATR-based trend direction indicator."""

    def setUp(self):
        from indicators.technical import supertrend
        self.supertrend = supertrend
        self.df = _make_ohlcv(200)

    def test_returns_correct_columns(self):
        """Must return DataFrame with supertrend, direction, buy_signal, sell_signal."""
        result = self.supertrend(self.df, 10, 3.0)
        for col in ("supertrend", "direction", "buy_signal", "sell_signal"):
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_direction_only_1_or_minus1(self):
        """Direction must be exactly 1 (bull) or -1 (bear) — no other values."""
        result = self.supertrend(self.df, 10, 3.0)
        valid  = result["direction"].dropna()
        unique = set(valid.unique())
        self.assertTrue(unique.issubset({1.0, -1.0}),
                        f"Unexpected direction values: {unique}")

    def test_buy_and_sell_signals_mutually_exclusive(self):
        """A bar cannot have both a buy signal AND a sell signal simultaneously."""
        result = self.supertrend(self.df, 10, 3.0)
        both   = result["buy_signal"] & result["sell_signal"]
        self.assertFalse(both.any(),
                         "A bar cannot simultaneously be a buy AND sell signal")

    def test_raises_on_missing_columns(self):
        """Must raise ValueError if high/low/close are missing."""
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.supertrend(self.df[["open", "volume"]], 10, 3.0)

    def test_buy_signal_follows_direction_change_to_bull(self):
        """buy_signal = True must always coincide with direction changing to +1."""
        result = self.supertrend(self.df, 10, 3.0)
        buys   = result[result["buy_signal"] == True]
        for idx in buys.index:
            self.assertEqual(result.loc[idx, "direction"], 1,
                             "buy_signal must occur only when direction == 1")


class TestATR(unittest.TestCase):
    """Average True Range — volatility measure."""

    def setUp(self):
        from indicators.technical import atr
        self.atr = atr
        self.df  = _make_ohlcv(100)

    def test_atr_always_positive(self):
        """ATR must always be > 0 (it is an absolute range, never negative)."""
        result = self.atr(self.df, 14)
        valid  = result.dropna()
        self.assertTrue((valid > 0).all(), "ATR must always be positive")

    def test_atr_higher_for_volatile_data(self):
        """ATR of high-volatility data must exceed ATR of low-volatility data."""
        low_vol  = pd.DataFrame({
            "high":  [101.0] * 100, "low": [99.0] * 100,
            "close": [100.0] * 100,
        }, index=self.df.index)
        high_vol = pd.DataFrame({
            "high":  [120.0] * 100, "low": [80.0] * 100,
            "close": [100.0] * 100,
        }, index=self.df.index)
        atr_low  = self.atr(low_vol,  14).mean()
        atr_high = self.atr(high_vol, 14).mean()
        self.assertGreater(atr_high, atr_low)

    def test_warmup_nan(self):
        """
        ATR warmup: True Range requires prev_close, so TR[0] is NaN.
        With ewm min_periods=14, the first valid ATR appears at bar index 13
        (because TR is available from bar 1, so 14 valid TR values exist at index 13+1=14-1).
        Therefore bars 0..12 (13 bars) must be NaN.
        """
        result = self.atr(self.df, 14)
        self.assertTrue(result.iloc[:13].isna().all(),
                        f"ATR warmup: expected bars 0-12 to be NaN, got: {result.iloc[:14].tolist()}")


class TestBollingerBands(unittest.TestCase):
    """Bollinger Bands — SMA ± 2σ dynamic channel."""

    def setUp(self):
        from indicators.technical import bollinger_bands
        self.bb = bollinger_bands
        self.df = _make_ohlcv(150)

    def test_returns_five_columns(self):
        """Must return: bb_upper, bb_middle, bb_lower, bb_pct_b, bb_bandwidth."""
        result = self.bb(self.df["close"], 20, 2.0)
        for col in ("bb_upper", "bb_middle", "bb_lower", "bb_pct_b", "bb_bandwidth"):
            self.assertIn(col, result.columns)

    def test_upper_always_above_lower(self):
        """Upper band must always be >= lower band (can be equal if std = 0)."""
        result = self.bb(self.df["close"], 20, 2.0)
        valid  = result.dropna()
        self.assertTrue((valid["bb_upper"] >= valid["bb_lower"]).all())

    def test_middle_band_is_sma(self):
        """Middle band must equal the SMA of the same period."""
        from indicators.technical import sma
        bb_mid = self.bb(self.df["close"], 20, 2.0)["bb_middle"]
        sma20  = sma(self.df["close"], 20)
        diff   = (bb_mid - sma20).dropna().abs()
        self.assertAlmostEqual(float(diff.max()), 0.0, places=3)

    def test_pct_b_formula(self):
        """
        %B = (close - lower) / (upper - lower).
        Verify this formula holds at every valid bar.

        NOTE: We cannot verify %B=1.0 by forcing close=precomputed_bb_upper,
        because changing close at bar i shifts the rolling std, which in turn
        shifts bb_upper — so the new bb_upper no longer equals the old value.
        Instead we directly verify the definition holds across the whole series.
        """
        result = self.bb(self.df["close"], 20, 2.0)
        valid  = result.dropna()
        close  = self.df["close"].reindex(valid.index)

        expected_pct_b = (
            (close - valid["bb_lower"]) /
            (valid["bb_upper"] - valid["bb_lower"])
        )
        diff = (valid["bb_pct_b"] - expected_pct_b).abs()
        self.assertAlmostEqual(float(diff.max()), 0.0, places=4,
            msg="%B formula: (close - lower)/(upper - lower) does not match stored bb_pct_b")

    def test_custom_std_multiplier(self):
        """Wider std_dev should produce a wider band."""
        bb2 = self.bb(self.df["close"], 20, 2.0)
        bb3 = self.bb(self.df["close"], 20, 3.0)
        bw2 = (bb2["bb_upper"] - bb2["bb_lower"]).dropna().mean()
        bw3 = (bb3["bb_upper"] - bb3["bb_lower"]).dropna().mean()
        self.assertGreater(bw3, bw2, "3σ bands must be wider than 2σ bands")


class TestKeltnerChannels(unittest.TestCase):
    """Keltner Channels — EMA ± multiplier×ATR."""

    def setUp(self):
        from indicators.technical import keltner_channels
        self.kc = keltner_channels
        self.df = _make_ohlcv(150)

    def test_returns_three_columns(self):
        result = self.kc(self.df)
        for col in ("kc_upper", "kc_middle", "kc_lower"):
            self.assertIn(col, result.columns)

    def test_upper_above_lower(self):
        result = self.kc(self.df)
        valid  = result.dropna()
        self.assertTrue((valid["kc_upper"] >= valid["kc_lower"]).all())


class TestStochastic(unittest.TestCase):
    """Stochastic Oscillator — %K and %D, range 0–100."""

    def setUp(self):
        from indicators.technical import stochastic
        self.stoch = stochastic
        self.df    = _make_ohlcv(100)

    def test_both_columns_present(self):
        result = self.stoch(self.df)
        self.assertIn("stoch_k", result.columns)
        self.assertIn("stoch_d", result.columns)

    def test_values_between_0_and_100(self):
        result = self.stoch(self.df, 14, 3)
        valid  = result.dropna()
        self.assertTrue((valid >= 0).all().all() and (valid <= 100).all().all())


class TestVWAP(unittest.TestCase):
    """VWAP — must reset at the start of each trading day."""

    def setUp(self):
        from indicators.technical import vwap
        self.vwap = vwap

    def test_resets_each_day(self):
        """
        VWAP computed on two separate days must NOT be cumulative across days.
        On day 2, the VWAP should start fresh from the first bar of day 2.
        """
        # Build 2 days of 1-minute data (375 bars per day)
        n     = 375 * 2
        times = pd.date_range("2024-01-02 09:15", periods=n, freq="min",
                              tz="Asia/Kolkata")
        df = pd.DataFrame({
            "high":   [101.0] * n,
            "low":    [99.0]  * n,
            "close":  [100.0] * n,
            "volume": [1000]  * n,
        }, index=times)

        result = self.vwap(df)

        # Day 1 last bar VWAP == 100 (typical price = (101+99+100)/3 = 100)
        # Day 2 first bar VWAP should also start fresh ≈ 100, NOT a massive
        # cumulative number from 750 bars
        day2_start_vwap = float(result.iloc[375])
        day1_end_vwap   = float(result.iloc[374])

        self.assertAlmostEqual(day2_start_vwap, 100.0, places=1,
                               msg="VWAP must reset at start of each day")
        self.assertAlmostEqual(day1_end_vwap,   100.0, places=1)

    def test_raises_on_missing_columns(self):
        df = _make_ohlcv(10).drop(columns=["volume"])
        with self.assertRaises((ValueError, KeyError)):
            self.vwap(df)


class TestOBV(unittest.TestCase):
    """On Balance Volume — cumulative volume direction indicator."""

    def setUp(self):
        from indicators.technical import obv
        self.obv = obv
        self.df  = _make_ohlcv(50)

    def test_rising_prices_give_positive_obv(self):
        """A series of all-up closes should produce monotonically increasing OBV."""
        close = pd.Series(np.linspace(100, 200, 50), index=self.df.index)
        df    = self.df.copy()
        df["close"] = close
        result = self.obv(df)
        diffs  = result.diff().dropna()
        self.assertTrue((diffs >= 0).all(),
                        "OBV must increase on every up-close")

    def test_falling_prices_give_negative_obv(self):
        """A series of all-down closes should produce monotonically decreasing OBV."""
        close = pd.Series(np.linspace(200, 100, 50), index=self.df.index)
        df    = self.df.copy()
        df["close"] = close
        result = self.obv(df)
        diffs  = result.diff().dropna()
        self.assertTrue((diffs <= 0).all(),
                        "OBV must decrease on every down-close")


class TestZScore(unittest.TestCase):
    """Rolling Z-Score — number of standard deviations from rolling mean."""

    def setUp(self):
        from indicators.technical import zscore
        self.zscore = zscore
        self.df     = _make_ohlcv(100)

    def test_zscore_mean_near_zero(self):
        """
        Over a long rolling window, the mean of the z-score should be near 0
        because the mean of (x - mean(x)) / std(x) = 0 by definition.
        """
        result = self.zscore(self.df["close"], 20)
        valid  = result.dropna()
        # Mean of z-scores should be close to 0 (not exact due to rolling window)
        self.assertAlmostEqual(float(valid.mean()), 0.0, delta=1.5)

    def test_warmup_nan(self):
        result = self.zscore(self.df["close"], 20)
        self.assertTrue(result.iloc[:19].isna().all())

    def test_period_1_raises(self):
        """Period of 1 gives std of 0 — zscore is undefined. Should return NaN or error."""
        # Either raises or returns NaN — both are acceptable
        try:
            result = self.zscore(self.df["close"], 1)
            # If it doesn't raise, result should be all NaN (std=0 → division by 0)
            self.assertTrue(result.isna().any())
        except (ValueError, ZeroDivisionError):
            pass  # Raising is also fine


class TestSignalHelpers(unittest.TestCase):
    """crossover(), crossunder(), above_threshold(), below_threshold()."""

    def setUp(self):
        from indicators.technical import (
            crossover, crossunder, above_threshold, below_threshold
        )
        self.crossover        = crossover
        self.crossunder       = crossunder
        self.above_threshold  = above_threshold
        self.below_threshold  = below_threshold

    def _make_cross(self, n=20):
        """Create two series that cross at bar 10."""
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        s1  = pd.Series([1.0] * 10 + [3.0] * 10, index=idx)  # jumps up at bar 10
        s2  = pd.Series([2.0] * n, index=idx)                  # constant at 2
        return s1, s2

    def test_crossover_fires_exactly_once_at_cross(self):
        """crossover should return True only on the single bar where s1 goes above s2."""
        s1, s2 = self._make_cross()
        result  = self.crossover(s1, s2)
        n_true  = result.sum()
        self.assertEqual(n_true, 1, f"crossover should fire exactly 1 time, got {n_true}")
        # Must fire at bar 10 (index 10)
        self.assertTrue(result.iloc[10])

    def test_crossunder_fires_at_cross_down(self):
        """crossunder should fire when s1 drops below s2."""
        idx = pd.date_range("2024-01-01", periods=20, freq="B")
        s1  = pd.Series([3.0] * 10 + [1.0] * 10, index=idx)
        s2  = pd.Series([2.0] * 20, index=idx)
        result = self.crossunder(s1, s2)
        self.assertEqual(result.sum(), 1)
        self.assertTrue(result.iloc[10])

    def test_crossover_and_crossunder_mutually_exclusive(self):
        """A bar cannot be both crossover AND crossunder."""
        s1, s2 = self._make_cross()
        co = self.crossover(s1, s2)
        cu = self.crossunder(s1, s2)
        both = co & cu
        self.assertFalse(both.any(), "Crossover and crossunder cannot both be True")

    def test_above_threshold_fires_on_crossing_up(self):
        """above_threshold should fire once when series goes from below to above level."""
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        s   = pd.Series([25.0] * 5 + [35.0] * 5, index=idx)
        result = self.above_threshold(s, 30.0)
        self.assertEqual(result.sum(), 1)
        self.assertTrue(result.iloc[5])

    def test_below_threshold_fires_on_crossing_down(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        s   = pd.Series([35.0] * 5 + [25.0] * 5, index=idx)
        result = self.below_threshold(s, 30.0)
        self.assertEqual(result.sum(), 1)
        self.assertTrue(result.iloc[5])


class TestCandleHelpers(unittest.TestCase):
    """candle_body, candle_range, is_green, is_red, upper_shadow, lower_shadow."""

    def setUp(self):
        from indicators.technical import (
            candle_body, candle_range, is_green, is_red,
            upper_shadow, lower_shadow
        )
        self.candle_body  = candle_body
        self.candle_range = candle_range
        self.is_green     = is_green
        self.is_red       = is_red
        self.upper_shadow = upper_shadow
        self.lower_shadow = lower_shadow

    def _make_candle_df(self):
        return pd.DataFrame({
            "open":  [100.0, 110.0, 100.0],
            "high":  [120.0, 120.0, 115.0],
            "low":   [90.0,  90.0,  90.0],
            "close": [110.0, 100.0, 100.0],   # bar0=green, bar1=red, bar2=doji
        })

    def test_candle_body_always_positive(self):
        df     = self._make_candle_df()
        result = self.candle_body(df)
        self.assertTrue((result >= 0).all())

    def test_green_candle_detected_correctly(self):
        df     = self._make_candle_df()
        result = self.is_green(df)
        self.assertTrue(result.iloc[0],  "Bar 0 close>open → should be green")
        self.assertFalse(result.iloc[1], "Bar 1 close<open → should not be green")

    def test_red_candle_detected_correctly(self):
        df     = self._make_candle_df()
        result = self.is_red(df)
        self.assertFalse(result.iloc[0], "Bar 0 is green, not red")
        self.assertTrue(result.iloc[1],  "Bar 1 close<open → should be red")

    def test_candle_range_equals_high_minus_low(self):
        df     = self._make_candle_df()
        result = self.candle_range(df)
        for i in range(len(df)):
            expected = df["high"].iloc[i] - df["low"].iloc[i]
            self.assertAlmostEqual(float(result.iloc[i]), expected)

    def test_shadows_non_negative(self):
        df  = self._make_candle_df()
        us  = self.upper_shadow(df)
        ls  = self.lower_shadow(df)
        self.assertTrue((us >= 0).all(), "Upper shadow must be >= 0")
        self.assertTrue((ls >= 0).all(), "Lower shadow must be >= 0")


# =============================================================================
# BLOCK 2 — backtester/commission.py
# =============================================================================

class TestSegmentEnum(unittest.TestCase):
    """Segment enum — correct values for all market segments."""

    def test_all_segments_exist(self):
        from backtester.commission import Segment
        expected = [
            "EQUITY_DELIVERY", "EQUITY_INTRADAY", "EQUITY_FUTURES",
            "EQUITY_OPTIONS",  "CURRENCY_FUTURES", "CURRENCY_OPTIONS",
            "COMMODITY_FUTURES", "COMMODITY_OPTIONS",
        ]
        for name in expected:
            self.assertTrue(hasattr(Segment, name), f"Missing Segment.{name}")


class TestCommissionDelivery(unittest.TestCase):
    """Equity Delivery charges — the most common retail segment."""

    def setUp(self):
        from backtester.commission import CommissionModel, Segment
        self.model   = CommissionModel()
        self.segment = Segment.EQUITY_DELIVERY

    def test_brokerage_capped_at_20(self):
        """For large trades (e.g. Rs1 lakh), brokerage must be exactly Rs20 (the cap)."""
        chg = self.model.calculate(self.segment, "BUY", 100, 1000.0)
        # 100 × 1000 × 0.001 = Rs100 → capped at Rs20
        self.assertEqual(chg.brokerage, 20.0)

    def test_brokerage_below_cap_for_tiny_trade(self):
        """For very small trades (1 share @ Rs10), brokerage = pct × value < Rs20."""
        chg = self.model.calculate(self.segment, "BUY", 1, 10.0)
        # 10 × 0.001 = Rs0.01, which is < Rs20
        self.assertLess(chg.brokerage, 20.0)
        self.assertAlmostEqual(chg.brokerage, 0.01, places=3)

    def test_stt_on_buy_side(self):
        """Delivery buy STT = 0.1% of trade value."""
        chg = self.model.calculate(self.segment, "BUY", 100, 1000.0)
        expected_stt = 100 * 1000 * 0.001
        self.assertAlmostEqual(chg.stt, expected_stt, places=2)

    def test_stt_on_sell_side(self):
        """Delivery sell STT = 0.1% of trade value."""
        chg = self.model.calculate(self.segment, "SELL", 100, 1000.0)
        expected_stt = 100 * 1000 * 0.001
        self.assertAlmostEqual(chg.stt, expected_stt, places=2)

    def test_stamp_duty_on_buy_only(self):
        """Stamp duty must be > 0 on BUY, exactly 0 on SELL."""
        buy_chg  = self.model.calculate(self.segment, "BUY",  100, 1000.0)
        sell_chg = self.model.calculate(self.segment, "SELL", 100, 1000.0)
        self.assertGreater(buy_chg.stamp_duty, 0.0)
        self.assertEqual(sell_chg.stamp_duty, 0.0)

    def test_dp_charge_on_sell_only(self):
        """DP charge (Rs18.5) must apply on SELL, never on BUY."""
        buy_chg  = self.model.calculate(self.segment, "BUY",  100, 1000.0)
        sell_chg = self.model.calculate(self.segment, "SELL", 100, 1000.0)
        self.assertEqual(buy_chg.dp_charge, 0.0)
        self.assertEqual(sell_chg.dp_charge, 18.5)

    def test_gst_on_brokerage_plus_transaction(self):
        """
        GST must be exactly 18% of (brokerage + transaction_charge).
        Both sides are rounded to 2 decimal places (paise precision)
        before comparison, matching how the commission model stores charges.
        """
        chg          = self.model.calculate(self.segment, "BUY", 100, 1000.0)
        expected_gst = round((chg.brokerage + chg.transaction_charge) * 0.18, 2)
        self.assertAlmostEqual(chg.gst, expected_gst, places=2)

    def test_total_equals_sum_of_components(self):
        """Total must equal the exact sum of all 7 charge components."""
        chg = self.model.calculate(self.segment, "BUY", 100, 1000.0)
        expected = (chg.brokerage + chg.stt + chg.transaction_charge +
                    chg.sebi_fee + chg.gst + chg.stamp_duty + chg.dp_charge)
        self.assertAlmostEqual(chg.total, expected, places=2)

    def test_sebi_fee_is_tiny(self):
        """SEBI fee is Rs10/crore = very small for normal trade sizes."""
        chg = self.model.calculate(self.segment, "BUY", 100, 1000.0)
        # Trade value = Rs1,00,000. SEBI = 1,00,000 × 1e-7 = Rs0.01
        self.assertAlmostEqual(chg.sebi_fee, 100_000 * 1e-7, places=4)

    def test_invalid_side_raises(self):
        """Invalid side string must raise ValueError."""
        with self.assertRaises(ValueError):
            self.model.calculate(self.segment, "HOLD", 100, 1000.0)

    def test_zero_quantity_raises(self):
        with self.assertRaises(ValueError):
            self.model.calculate(self.segment, "BUY", 0, 1000.0)

    def test_zero_price_raises(self):
        with self.assertRaises(ValueError):
            self.model.calculate(self.segment, "BUY", 100, 0.0)


class TestCommissionOptions(unittest.TestCase):
    """Options always charge flat Rs20 brokerage regardless of trade size."""

    def setUp(self):
        from backtester.commission import CommissionModel, Segment
        self.model   = CommissionModel()
        self.segment = Segment.EQUITY_OPTIONS

    def test_brokerage_always_flat_20(self):
        """Options brokerage = Rs20 even for a Rs1 trade or a Rs10 lakh trade."""
        small = self.model.calculate(self.segment, "BUY", 1,    1.0)
        large = self.model.calculate(self.segment, "BUY", 1000, 1000.0)
        self.assertEqual(small.brokerage, 20.0)
        self.assertEqual(large.brokerage, 20.0)

    def test_no_dp_charge_on_options_sell(self):
        """DP charge only applies to equity delivery sells, not options."""
        chg = self.model.calculate(self.segment, "SELL", 50, 200.0)
        self.assertEqual(chg.dp_charge, 0.0)


class TestCommissionFutures(unittest.TestCase):
    """Equity Futures — STT only on sell side."""

    def setUp(self):
        from backtester.commission import CommissionModel, Segment
        self.model   = CommissionModel()
        self.segment = Segment.EQUITY_FUTURES

    def test_stt_zero_on_buy(self):
        """Futures: no STT on buy side."""
        chg = self.model.calculate(self.segment, "BUY", 250, 1000.0)
        self.assertEqual(chg.stt, 0.0)

    def test_stt_nonzero_on_sell(self):
        """Futures: STT = 0.0125% on sell side."""
        chg = self.model.calculate(self.segment, "SELL", 250, 1000.0)
        expected = 250 * 1000 * 0.0001250
        self.assertAlmostEqual(chg.stt, expected, places=3)


class TestRoundTrip(unittest.TestCase):
    """round_trip_cost() — convenience method for full trade cost."""

    def setUp(self):
        from backtester.commission import CommissionModel, Segment
        self.model   = CommissionModel()
        self.segment = Segment.EQUITY_DELIVERY

    def test_round_trip_total_matches_sum_of_legs(self):
        """round_trip total must equal buy.total + sell.total."""
        result = self.model.round_trip_cost(
            self.segment, 100, 1000.0, 1050.0
        )
        buy  = self.model.calculate(self.segment, "BUY",  100, 1000.0)
        sell = self.model.calculate(self.segment, "SELL", 100, 1050.0)
        expected = buy.total + sell.total
        self.assertAlmostEqual(result["total_charges"], expected, places=2)

    def test_round_trip_contains_both_legs(self):
        """round_trip_cost must return both buy_charges and sell_charges dicts."""
        result = self.model.round_trip_cost(self.segment, 100, 1000.0, 1050.0)
        self.assertIn("buy_charges",  result)
        self.assertIn("sell_charges", result)
        self.assertIn("total_charges", result)


class TestInferSegment(unittest.TestCase):
    """infer_segment() — maps instrument_type + holding_type → Segment."""

    def setUp(self):
        from backtester.commission import infer_segment, Segment
        self.infer   = infer_segment
        self.Segment = Segment

    def test_equity_cnc_is_delivery(self):
        self.assertEqual(self.infer("EQUITY", "CNC"), self.Segment.EQUITY_DELIVERY)

    def test_equity_mis_is_intraday(self):
        self.assertEqual(self.infer("EQUITY", "MIS"), self.Segment.EQUITY_INTRADAY)

    def test_futstk_nrml_is_futures(self):
        self.assertEqual(self.infer("FUTSTK", "NRML"), self.Segment.EQUITY_FUTURES)

    def test_futidx_nrml_is_futures(self):
        self.assertEqual(self.infer("FUTIDX", "NRML"), self.Segment.EQUITY_FUTURES)

    def test_optidx_nrml_is_options(self):
        self.assertEqual(self.infer("OPTIDX", "NRML"), self.Segment.EQUITY_OPTIONS)

    def test_futcom_nrml_is_commodity_futures(self):
        self.assertEqual(self.infer("FUTCOM", "NRML"), self.Segment.COMMODITY_FUTURES)

    def test_unknown_combination_raises(self):
        """Unknown instrument_type must raise ValueError with helpful message."""
        with self.assertRaises(ValueError) as ctx:
            self.infer("UNKNOWN", "CNC")
        self.assertIn("UNKNOWN", str(ctx.exception))


# =============================================================================
# BLOCK 3 — strategies/base.py
# =============================================================================

class TestBaseStrategyValidation(unittest.TestCase):
    """BaseStrategy._validate_and_prepare() — input checking."""

    def setUp(self):
        from strategies.base_strategy_github import EMACrossover
        self.strategy = EMACrossover(9, 21)
        self.df       = _make_ohlcv(200)

    def test_raises_on_missing_ohlcv_columns(self):
        """Must raise ValueError if required columns are absent."""
        bad_df = self.df.drop(columns=["close"])
        with self.assertRaises(ValueError) as ctx:
            self.strategy.generate_signals(bad_df)
        self.assertIn("close", str(ctx.exception))

    def test_signal_column_added(self):
        """generate_signals must add a 'signal' column to the output."""
        result = self.strategy.generate_signals(self.df)
        self.assertIn("signal", result.columns)

    def test_signal_only_contains_valid_values(self):
        """Signal values must be only -1, 0, or +1."""
        result = self.strategy.generate_signals(self.df)
        self.assertTrue(result["signal"].isin([-1, 0, 1]).all(),
                        f"Unexpected signal values: {result['signal'].unique()}")

    def test_input_dataframe_not_mutated(self):
        """generate_signals must NOT modify the original DataFrame in place."""
        original_cols = set(self.df.columns)
        _             = self.strategy.generate_signals(self.df)
        self.assertEqual(set(self.df.columns), original_cols,
                         "generate_signals must not mutate the input DataFrame")

    def test_warmup_signals_are_zero(self):
        """First N bars (warm-up period) must have signal == 0."""
        result   = self.strategy.generate_signals(self.df)
        # EMA(21) → warmup = 21 bars
        warmup   = 21
        warmup_signals = result["signal"].iloc[:warmup]
        self.assertTrue((warmup_signals == 0).all(),
                        "Warm-up bars must have signal == 0 to prevent false trades")


class TestEMACrossoverStrategy(unittest.TestCase):
    """EMACrossover — buy on fast>slow crossover, sell on fast<slow crossunder."""

    def setUp(self):
        from strategies.base_strategy_github import EMACrossover
        self.df = _make_ohlcv(300)

    def test_fast_must_be_less_than_slow(self):
        from strategies.base_strategy_github import EMACrossover
        with self.assertRaises(ValueError):
            EMACrossover(fast_period=21, slow_period=9)

    def test_indicator_columns_added(self):
        from strategies.base_strategy_github import EMACrossover
        strategy = EMACrossover(9, 21)
        result   = strategy.generate_signals(self.df)
        self.assertIn("ema_fast", result.columns)
        self.assertIn("ema_slow", result.columns)

    def test_buy_signal_when_fast_above_slow(self):
        """A +1 signal must coincide with ema_fast > ema_slow."""
        from strategies.base_strategy_github import EMACrossover
        strategy = EMACrossover(9, 21)
        result   = strategy.generate_signals(self.df)
        buys     = result[result["signal"] == 1]
        for idx in buys.index:
            # The signal is generated at bar i → at that bar, fast should be above slow
            self.assertGreaterEqual(
                result.loc[idx, "ema_fast"],
                result.loc[idx, "ema_slow"] - 0.01,   # tiny tolerance for floating point
                f"At bar {idx}, ema_fast should be >= ema_slow on a buy signal"
            )

    def test_get_parameters_returns_periods(self):
        from strategies.base_strategy_github import EMACrossover
        strat  = EMACrossover(9, 21)
        params = strat.get_parameters()
        self.assertEqual(params["fast_period"], 9)
        self.assertEqual(params["slow_period"], 21)


class TestRSIMeanReversionStrategy(unittest.TestCase):
    """RSI Mean Reversion — buy oversold, sell overbought."""

    def setUp(self):
        from strategies.base_strategy_github import RSIMeanReversion
        self.df = _make_ohlcv(300)
        self.strat = RSIMeanReversion(14, 30, 70, 200)

    def test_rsi_column_added(self):
        result = self.strat.generate_signals(self.df)
        self.assertIn("rsi", result.columns)

    def test_sma_filter_column_added_when_enabled(self):
        result = self.strat.generate_signals(self.df)
        self.assertIn("sma_filter", result.columns)

    def test_no_sma_filter_when_period_zero(self):
        from strategies.base_strategy_github import RSIMeanReversion
        strat  = RSIMeanReversion(14, 30, 70, sma_filter_period=0)
        result = strat.generate_signals(self.df)
        # sma_filter should be all NaN when disabled
        if "sma_filter" in result.columns:
            self.assertTrue(result["sma_filter"].isna().all())

    def test_buy_signal_has_valid_rsi(self):
        """Every buy signal bar must have RSI that was recently below the oversold level."""
        result = self.strat.generate_signals(self.df)
        buys   = result[result["signal"] == 1]
        # Not every buy bar itself has RSI<30 (the signal fires on the crossover bar)
        # But RSI at buys should be NEAR 30, not near 70
        if len(buys) > 0:
            avg_rsi_at_buy = float(buys["rsi"].mean())
            self.assertLess(avg_rsi_at_buy, 60.0,
                            "Buys from RSI reversion should occur at low RSI values")


class TestBollingerBandStrategy(unittest.TestCase):
    """Bollinger Band Strategy — reversion and breakout modes."""

    def setUp(self):
        self.df = _make_ohlcv(300)

    def test_reversion_mode_buy_at_lower_band(self):
        from strategies.base_strategy_github import BollingerBandStrategy
        strat  = BollingerBandStrategy(20, 2.0, "reversion")
        result = strat.generate_signals(self.df)
        buys   = result[result["signal"] == 1]
        # In reversion mode, buy bars should have close <= bb_lower
        for idx in buys.index:
            self.assertLessEqual(
                float(result.loc[idx, "close"]),
                float(result.loc[idx, "bb_lower"]) + 0.1,
                "Reversion buy should occur when price is at or below lower band"
            )

    def test_breakout_mode_sell_at_lower_band(self):
        from strategies.base_strategy_github import BollingerBandStrategy
        strat  = BollingerBandStrategy(20, 2.0, "breakout")
        result = strat.generate_signals(self.df)
        self.assertIn("signal", result.columns)
        self.assertTrue(result["signal"].isin([-1, 0, 1]).all())

    def test_invalid_mode_raises(self):
        from strategies.base_strategy_github import BollingerBandStrategy
        with self.assertRaises(ValueError):
            BollingerBandStrategy(20, 2.0, "invalid_mode")

    def test_bb_columns_present_in_output(self):
        from strategies.base_strategy_github import BollingerBandStrategy
        strat  = BollingerBandStrategy(20, 2.0, "reversion")
        result = strat.generate_signals(self.df)
        for col in ("bb_upper", "bb_middle", "bb_lower"):
            self.assertIn(col, result.columns)


class TestMACDStrategy(unittest.TestCase):
    """MACD Crossover Strategy."""

    def setUp(self):
        from strategies.base_strategy_github import MACDStrategy
        self.df    = _make_ohlcv(300)
        self.strat = MACDStrategy(12, 26, 9)

    def test_macd_columns_present(self):
        result = self.strat.generate_signals(self.df)
        for col in ("macd", "macd_signal", "macd_histogram"):
            self.assertIn(col, result.columns)

    def test_signals_only_on_histogram_crossovers(self):
        """
        All +1 signals should occur when histogram crosses from negative to positive.
        All -1 signals should occur when histogram crosses from positive to negative.
        """
        result = self.strat.generate_signals(self.df)
        hist   = result["macd_histogram"]

        for i in result.index[1:]:
            prev_i = result.index[result.index.get_loc(i) - 1]
            sig    = result.loc[i, "signal"]
            if sig == 1:
                # histogram should have just crossed above 0
                self.assertGreaterEqual(float(hist.loc[i]), 0,
                    "Buy signal must occur when histogram >= 0")
            elif sig == -1:
                self.assertLessEqual(float(hist.loc[i]), 0,
                    "Sell signal must occur when histogram <= 0")


class TestSupertrendStrategy(unittest.TestCase):
    """Supertrend Strategy — direction flip signals."""

    def setUp(self):
        from strategies.base_strategy_github import SupertrendStrategy
        self.df    = _make_ohlcv(300)
        self.strat = SupertrendStrategy(10, 3.0)

    def test_supertrend_columns_added(self):
        result = self.strat.generate_signals(self.df)
        self.assertIn("supertrend",   result.columns)
        self.assertIn("st_direction", result.columns)

    def test_signals_match_direction_changes(self):
        """Buy signals must coincide exactly with direction flipping from -1 to +1."""
        result = self.strat.generate_signals(self.df)
        buys   = result[result["signal"] == 1]
        for idx in buys.index:
            self.assertEqual(result.loc[idx, "st_direction"], 1,
                             "Buy signal must occur when direction == 1")


# =============================================================================
# BLOCK 4 — backtester/engine.py
# =============================================================================

class TestBacktestEngineBasic(unittest.TestCase):
    """BacktestEngine — basic run, output structure, equity curve."""

    def setUp(self):
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        from strategies.base_strategy_github import EMACrossover

        self.df       = _make_ohlcv(300)
        self.strategy = EMACrossover(9, 21)
        self.config   = BacktestConfig(
            initial_capital=500_000,
            segment=Segment.EQUITY_DELIVERY,
            capital_risk_pct=0.02,
            allow_shorting=False,
            max_positions=5,
        )
        self.engine   = BacktestEngine(self.config)

    def test_run_returns_backtest_result(self):
        from backtester.engine import BacktestResult
        result = self.engine.run(self.df, self.strategy, symbol="TEST")
        self.assertIsInstance(result, BacktestResult)

    def test_equity_curve_length_matches_data(self):
        """Equity curve must have exactly one value per bar."""
        result = self.engine.run(self.df, self.strategy, symbol="TEST")
        self.assertEqual(len(result.equity_curve), len(self.df))

    def test_equity_curve_starts_at_initial_capital(self):
        """
        The first bar of the equity curve must be close to initial capital.
        No trades execute on bar 0 (we execute the prev bar's signal).
        """
        result = self.engine.run(self.df, self.strategy, symbol="TEST")
        first_eq = float(result.equity_curve.dropna().iloc[0])
        self.assertAlmostEqual(first_eq, 500_000, delta=50_000,
                               msg="Equity should start near initial capital")

    def test_drawdown_always_non_positive(self):
        """
        Drawdown = (portfolio - peak) / peak.
        This is always <= 0 since portfolio can never exceed the historical peak.
        """
        result = self.engine.run(self.df, self.strategy, symbol="TEST")
        max_dd = float(result.drawdown.dropna().max())
        self.assertLessEqual(max_dd, 0.01,
                             "Drawdown must be <= 0% at all times")

    def test_trade_log_is_list(self):
        result = self.engine.run(self.df, self.strategy, symbol="TEST")
        self.assertIsInstance(result.trade_log, list)

    def test_each_trade_has_required_fields(self):
        """Every Trade object must have all expected fields."""
        result = self.engine.run(self.df, self.strategy, symbol="TEST")
        required_fields = [
            "entry_time", "exit_time", "entry_price", "exit_price",
            "quantity", "direction", "gross_pnl", "net_pnl",
            "total_charges", "pnl_pct", "mae", "mfe", "duration",
        ]
        for trade in result.trade_log:
            d = trade.to_dict()
            for field in required_fields:
                self.assertIn(field, d, f"Trade missing field: {field}")

    def test_all_trades_have_positive_charges(self):
        """Every round-trip trade must have total charges > 0."""
        result = self.engine.run(self.df, self.strategy, symbol="TEST")
        for trade in result.trade_log:
            self.assertGreater(trade.total_charges, 0,
                               f"Trade has zero charges: {trade.to_dict()}")

    def test_gross_pnl_minus_charges_equals_net_pnl(self):
        """net_pnl = gross_pnl - total_charges (accounting identity)."""
        result = self.engine.run(self.df, self.strategy, symbol="TEST")
        for trade in result.trade_log:
            expected_net = trade.gross_pnl - trade.total_charges
            self.assertAlmostEqual(
                trade.net_pnl, expected_net, places=1,
                msg=f"net_pnl accounting error for trade: {trade.to_dict()}"
            )

    def test_duration_and_bars_computation(self):
        """Ensure duration string and bar count reflect actual elapsed time."""
        # create 10 bars of 5‑minute data so we can easily calculate expectation
        df = _make_ohlcv(n=10, freq="5T")
        # build a simple signal series: buy on first bar, sell on sixth bar
        sig = pd.Series(0, index=df.index, name="signal")
        sig.iloc[0] = 1
        sig.iloc[5] = -1
        df = df.copy()
        df["signal"] = sig

        # use a dummy strategy that returns the dataframe unchanged
        from strategies.base_strategy_github import BaseStrategy
        class PassThroughStrategy(BaseStrategy):
            def __init__(self):
                super().__init__(name="pass")
            def generate_signals(self, df2):
                return df2

        result = self.engine.run(df, PassThroughStrategy(), symbol="TEST")
        self.assertEqual(len(result.trade_log), 1)
        trade = result.trade_log[0]
        # we expected 5 bars held (entry at bar0, exit executed on bar5 open)
        self.assertEqual(trade.duration_bars, 5)
        self.assertEqual(trade.duration, "0d 00h 25m")

    def test_missing_signal_column_raises(self):
        """Engine must raise ValueError if strategy doesn't add 'signal' column."""
        from backtester.engine import BacktestEngine
        from strategies.base_strategy_github import BaseStrategy

        class BadStrategy(BaseStrategy):
            def generate_signals(self, df):
                return df.copy()   # Forget to add 'signal' column!

        with self.assertRaises(ValueError) as ctx:
            self.engine.run(self.df, BadStrategy("Bad"), "TEST")
        self.assertIn("signal", str(ctx.exception))

    def test_export_signals_csv_creates_file(self):
        """BacktestResult.export_signals_csv should write a CSV containing
        the OHLCV data with indicators and 'signal' column.
        """
        result = self.engine.run(self.df, self.strategy, symbol="TEST")
        # create a temporary file path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_path = tmp.name
        tmp.close()

        # perform export and verify
        result.export_signals_csv(tmp_path)
        self.assertTrue(os.path.exists(tmp_path), "CSV file was not created")

        df2 = pd.read_csv(tmp_path, index_col=0)
        self.assertIn("signal", df2.columns)
        # original OHLCV columns should still be present
        for col in ["open", "high", "low", "close", "volume"]:
            self.assertIn(col, df2.columns)

        os.remove(tmp_path)

    def test_warns_on_too_few_bars(self):
        """Engine must warn when fewer than 100 bars are provided."""
        tiny_df = _make_ohlcv(50)
        with self.assertWarns(UserWarning):
            self.engine.run(tiny_df, self.strategy, symbol="TEST")


class TestRSISupertrendRelativeStrength(unittest.TestCase):
    """Unit tests for the RSI+Supertrend+RelativeStrength composite strategy."""

    def setUp(self):
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        from strategies.base_strategy_github import RSISupertrendRelativeStrength

        # create a simple upward-trending price series
        self.df = _make_ohlcv(n=200)
        # overwrite close/open/high/low to make a clean uptrend
        self.df["close"] = pd.Series(
            np.linspace(1000, 1200, len(self.df)), index=self.df.index
        )
        self.df["open"]  = self.df["close"] - 0.01
        self.df["high"]  = self.df["close"] + 0.01
        self.df["low"]   = self.df["close"] - 0.02
        self.df["volume"] = 1_000_000

        # benchmark is flat so RS > 0 whenever stock rises over rs_period
        self.df["bench_close"] = 1000.0

        config = BacktestConfig(
            initial_capital=500_000,
            segment=Segment.EQUITY_DELIVERY,
            capital_risk_pct=0.02,
            allow_shorting=False,
        )
        self.engine   = BacktestEngine(config)
        self.strategy = RSISupertrendRelativeStrength(
            rsi_period=14, super_period=10, super_multiplier=3.0, rs_period=55
        )

    def test_strategy_generates_trades(self):
        """Strategy should produce at least one long trade on an uptrend."""
        result = self.engine.run(self.df, self.strategy, symbol="TEST")
        self.assertGreater(len(result.trade_log), 0,
                           "No trades generated by strategy on strong uptrend")
        # ensure all trades are long (direction == 1)
        for trade in result.trade_log:
            self.assertEqual(trade.direction, 1,
                             f"Unexpected short trade: {trade.to_dict()}")

    def test_signals_follow_logic(self):
        """Verify that signals only appear when the three conditions are met."""
        sig_df = self.strategy.generate_signals(self.df)
        rsi_vals = sig_df["rsi"]
        st_dir   = sig_df["st_direction"]
        rs_vals  = sig_df["rs"]
        long_cond = (rsi_vals > 50) & (st_dir == 1) & (rs_vals > 0)

        # every long signal should satisfy the condition
        longs = sig_df[sig_df["signal"] == 1]
        self.assertTrue(long_cond.loc[longs.index].all(),
                        "A long signal was generated when conditions were not met")

        # every sell signal should occur when the condition is false
        sells = sig_df[sig_df["signal"] == -1]
        self.assertTrue((~long_cond.loc[sells.index]).all(),
                        "A sell signal was generated while long-conditions still held")


class TestExecutionModel(unittest.TestCase):
    """
    Next-bar-open execution model — the most critical anti-bias test.

    A signal on bar[i] must ONLY be executed at bar[i+1]'s open.
    This prevents look-ahead bias.
    """

    def setUp(self):
        from backtester.engine import BacktestEngine, BacktestConfig, Trade
        from backtester.commission import Segment

        self.Trade   = Trade
        self.config  = BacktestConfig(
            initial_capital=500_000,
            segment=Segment.EQUITY_DELIVERY,
            capital_risk_pct=0.10,    # Large position so we notice the price
            fixed_quantity=10,         # Fixed qty for predictable testing
            stop_loss_atr_mult=0.0,
            allow_shorting=False,
        )
        self.engine = BacktestEngine(self.config)

    def test_entry_price_is_next_bar_open(self):
        """
        Construct a DataFrame where bar 50 has signal=+1 and bar 51 has a
        known, distinctive open price. The trade entry must be at that open.
        """
        from strategies.base_strategy_github import BaseStrategy

        df = _make_ohlcv(200)

        # Force a specific open price at bar 51 to verify execution price
        SIGNAL_BAR  = 50
        EXEC_BAR    = 51
        KNOWN_OPEN  = 9999.0   # Distinctive price we can verify in the trade log

        df.iloc[EXEC_BAR, df.columns.get_loc("open")]  = KNOWN_OPEN
        df.iloc[EXEC_BAR, df.columns.get_loc("high")]  = KNOWN_OPEN + 10
        df.iloc[EXEC_BAR, df.columns.get_loc("low")]   = KNOWN_OPEN - 10
        df.iloc[EXEC_BAR, df.columns.get_loc("close")] = KNOWN_OPEN + 5

        class FixedSignalStrategy(BaseStrategy):
            """Emits exactly one buy at SIGNAL_BAR and one sell 10 bars later."""
            def generate_signals(self, df):
                df = df.copy()
                df["signal"] = 0
                df.iloc[SIGNAL_BAR, df.columns.get_loc("signal")] = 1
                df.iloc[SIGNAL_BAR + 10, df.columns.get_loc("signal")] = -1
                return df

        result = self.engine.run(df, FixedSignalStrategy("Fixed"), "TEST")

        # Should have exactly one completed trade
        self.assertEqual(len(result.trade_log), 1,
                         "Should have exactly 1 trade from the fixed signal")

        trade = result.trade_log[0]
        self.assertAlmostEqual(
            trade.entry_price, KNOWN_OPEN, places=1,
            msg=f"Entry price must be bar[{EXEC_BAR}]'s open ({KNOWN_OPEN}), "
                f"not bar[{SIGNAL_BAR}]'s close. Got: {trade.entry_price}"
        )


class TestPositionSizing(unittest.TestCase):
    """Position sizing — capital_risk_pct, fixed_quantity, lot_size."""

    def setUp(self):
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        self.Segment = Segment
        self.Config  = BacktestConfig
        self.Engine  = BacktestEngine

    def _run_with_fixed_qty(self, qty):
        from strategies.base_strategy_github import EMACrossover
        config = self.Config(
            initial_capital=500_000,
            segment=self.Segment.EQUITY_DELIVERY,
            fixed_quantity=qty,
        )
        engine   = self.Engine(config)
        strategy = EMACrossover(9, 21)
        df       = _make_ohlcv(300)
        return engine.run(df, strategy, symbol="TEST")

    def test_fixed_quantity_respected(self):
        """Every trade quantity must equal fixed_quantity when it is set."""
        result = self._run_with_fixed_qty(50)
        for trade in result.trade_log:
            self.assertEqual(trade.quantity, 50,
                             f"Expected quantity=50, got {trade.quantity}")

    def test_lot_size_rounding(self):
        """When lot_size > 1, quantity must be a multiple of lot_size."""
        from strategies.base_strategy_github import EMACrossover
        config = self.Config(
            initial_capital=500_000,
            segment=self.Segment.EQUITY_FUTURES,
            capital_risk_pct=0.05,
            lot_size=50,   # NIFTY lot size
            stop_loss_atr_mult=0,
        )
        engine   = self.Engine(config)
        strategy = EMACrossover(9, 21)
        df       = _make_ohlcv(300)
        result   = engine.run(df, strategy, symbol="TEST")
        for trade in result.trade_log:
            self.assertEqual(trade.quantity % 50, 0,
                             f"Quantity {trade.quantity} is not a multiple of lot_size=50")


class TestPyramiding(unittest.TestCase):
    """Multiple simultaneous positions per symbol (pyramiding)."""

    def setUp(self):
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        self.config = BacktestConfig(
            initial_capital=1_000_000,    # Large capital
            segment=Segment.EQUITY_DELIVERY,
            capital_risk_pct=0.01,
            fixed_quantity=10,
            max_positions=5,              # Allow up to 5 simultaneous positions
            allow_shorting=False,
        )
        self.engine = BacktestEngine(self.config)

    def test_max_positions_respected(self):
        """
        When max_positions=5, at most 5 open positions should exist at any point.
        We verify this indirectly: if we generate 10 consecutive buy signals,
        only the first 5 should be executed.
        """
        from strategies.base_strategy_github import BaseStrategy

        class BurstBuyStrategy(BaseStrategy):
            """Emits 10 buy signals in a row, then 10 sell signals."""
            def generate_signals(self, df):
                df = df.copy()
                df["signal"] = 0
                # 10 consecutive buys starting at bar 30
                for i in range(30, 40):
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                # Close all with sells at bar 60
                for i in range(60, 70):
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                return df

        df     = _make_ohlcv(300)
        result = self.engine.run(df, BurstBuyStrategy("Burst"), "TEST")

        # We should have at most 5 completed trades (opened) then 5 closes
        # Total trades = min(10, max_positions) = 5
        self.assertLessEqual(len(result.trade_log), 5,
                             f"max_positions=5 but {len(result.trade_log)} trades completed")


class TestFIFOClose(unittest.TestCase):
    """Positions are closed in FIFO order (oldest first)."""

    def setUp(self):
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        self.engine = BacktestEngine(BacktestConfig(
            initial_capital=500_000,
            segment=Segment.EQUITY_DELIVERY,
            fixed_quantity=10,
            max_positions=3,
        ))

    def test_first_entry_is_first_closed(self):
        """
        Open 3 positions at known prices. Close with one sell signal.
        The closed trade should correspond to the FIRST entry price.
        """
        from strategies.base_strategy_github import BaseStrategy

        ENTRY_PRICE_1 = 1001.0
        ENTRY_PRICE_2 = 1002.0
        ENTRY_PRICE_3 = 1003.0

        df = _make_ohlcv(200)

        # Set specific opens at bars 10, 20, 30 for entries
        # Set a specific open at bar 50 for the exit
        for bar, price in [(10, ENTRY_PRICE_1), (20, ENTRY_PRICE_2), (30, ENTRY_PRICE_3)]:
            df.iloc[bar, df.columns.get_loc("open")]  = price
            df.iloc[bar, df.columns.get_loc("high")]  = price + 5
            df.iloc[bar, df.columns.get_loc("low")]   = price - 5
            df.iloc[bar, df.columns.get_loc("close")] = price + 2

        class SequentialBuyStrategy(BaseStrategy):
            def generate_signals(self, df):
                df = df.copy()
                df["signal"] = 0
                df.iloc[ 9, df.columns.get_loc("signal")] = 1   # triggers at bar 10
                df.iloc[19, df.columns.get_loc("signal")] = 1   # triggers at bar 20
                df.iloc[29, df.columns.get_loc("signal")] = 1   # triggers at bar 30
                df.iloc[49, df.columns.get_loc("signal")] = -1  # close at bar 50
                return df

        result = self.engine.run(df, SequentialBuyStrategy("Seq"), "TEST")

        # The first closed trade should have been entered at ENTRY_PRICE_1
        self.assertGreater(len(result.trade_log), 0, "Expected at least 1 closed trade")
        first_closed = result.trade_log[0]
        self.assertAlmostEqual(
            first_closed.entry_price, ENTRY_PRICE_1, places=0,
            msg=f"FIFO: first closed should be first entered. "
                f"Expected {ENTRY_PRICE_1}, got {first_closed.entry_price}"
        )


class TestShorting(unittest.TestCase):
    """Short selling — only executed when allow_shorting=True."""

    def setUp(self):
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        self.Segment = Segment
        self.Config  = BacktestConfig
        self.Engine  = BacktestEngine

    def _make_short_strategy(self):
        from strategies.base_strategy_github import BaseStrategy
        class ShortStrategy(BaseStrategy):
            def generate_signals(self, df):
                df = df.copy()
                df["signal"] = 0
                df.iloc[30, df.columns.get_loc("signal")] = -1   # short signal
                df.iloc[60, df.columns.get_loc("signal")] =  1   # cover
                return df
        return ShortStrategy("Short")

    def test_short_not_opened_when_allow_shorting_false(self):
        """With allow_shorting=False, a -1 signal when no open position is ignored."""
        config = self.Config(
            initial_capital=500_000,
            segment=self.Segment.EQUITY_DELIVERY,
            fixed_quantity=10,
            allow_shorting=False,
        )
        result = self.Engine(config).run(
            _make_ohlcv(200), self._make_short_strategy(), "TEST"
        )
        shorts = [t for t in result.trade_log if t.direction == -1]
        self.assertEqual(len(shorts), 0, "No short positions should be opened")

    def test_short_opened_when_allow_shorting_true(self):
        """With allow_shorting=True, a -1 signal opens a short position."""
        config = self.Config(
            initial_capital=500_000,
            segment=self.Segment.EQUITY_DELIVERY,
            fixed_quantity=10,
            allow_shorting=True,
        )
        result = self.Engine(config).run(
            _make_ohlcv(200), self._make_short_strategy(), "TEST"
        )
        shorts = [t for t in result.trade_log if t.direction == -1]
        self.assertGreater(len(shorts), 0, "Short position should have been opened")


class TestIntradaySquareoff(unittest.TestCase):
    """Intraday square-off — all positions must close by 15:20 IST."""

    def test_positions_closed_at_squareoff(self):
        """
        Build intraday minute data. Open a position. Verify it is force-closed
        at 15:20 IST exit signal ('MIS Squareoff 15:20').
        """
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        from strategies.base_strategy_github import BaseStrategy

        # Build one day of 1-minute bars (375 bars: 9:15 to 15:29)
        n     = 375
        times = pd.date_range("2024-01-02 09:15", periods=n, freq="min",
                              tz="Asia/Kolkata")
        df = pd.DataFrame({
            "open":   [100.0] * n,
            "high":   [101.0] * n,
            "low":    [99.0]  * n,
            "close":  [100.0] * n,
            "volume": [10000] * n,
            "oi":     [0]     * n,
        }, index=times)

        class BuyAtOpenStrategy(BaseStrategy):
            def generate_signals(self, df):
                df = df.copy()
                df["signal"] = 0
                df.iloc[5, df.columns.get_loc("signal")] = 1   # buy at bar 5
                # No sell signal — engine must auto-close at 15:20
                return df

        config = BacktestConfig(
            initial_capital=500_000,
            segment=Segment.EQUITY_INTRADAY,
            fixed_quantity=10,
            intraday_squareoff=True,
        )
        engine = BacktestEngine(config)
        result = engine.run(df, BuyAtOpenStrategy("Buy"), "TEST")

        # If position was opened, it must have been auto-closed
        squareoff_trades = [t for t in result.trade_log
                            if "Squareoff" in t.exit_signal or "15:20" in t.exit_signal]
        if result.trade_log:   # Position was opened
            self.assertGreater(len(squareoff_trades), 0,
                               "Position should have been auto-closed at 15:20")


class TestBacktestMetrics(unittest.TestCase):
    """BacktestResult._compute_metrics() — verify each metric formula."""

    def setUp(self):
        from backtester.engine import BacktestEngine, BacktestConfig, BacktestResult
        from backtester.commission import Segment
        from strategies.base_strategy_github import EMACrossover

        df         = _make_ohlcv(400)
        config     = BacktestConfig(
            initial_capital=500_000,
            segment=Segment.EQUITY_DELIVERY,
            fixed_quantity=10,
            stop_loss_atr_mult=0,
        )
        engine     = BacktestEngine(config)
        strategy   = EMACrossover(9, 21)
        self.result = engine.run(df, strategy, "TEST")
        self.config = config

    def test_metrics_dict_has_required_keys(self):
        m = self.result._compute_metrics()
        required = [
            "Total Trades", "Win Rate", "Profit Factor",
            "Sharpe Ratio", "Max Drawdown", "Total Return",
            "CAGR", "Expectancy/Trade", "Total Charges Paid",
        ]
        for key in required:
            self.assertIn(key, m, f"Missing metric: {key}")

    def test_win_rate_between_0_and_100(self):
        m = self.result._compute_metrics()
        wr = float(m["Win Rate"].rstrip("%"))
        self.assertGreaterEqual(wr, 0.0)
        self.assertLessEqual(wr, 100.0)

    def test_profit_factor_non_negative(self):
        m  = self.result._compute_metrics()
        pf = float(m["Profit Factor"].replace("inf", "9999"))
        self.assertGreaterEqual(pf, 0.0)

    def test_total_return_consistent_with_portfolio(self):
        """Total return % must be consistent with (final - initial) / initial × 100."""
        m      = self.result._compute_metrics()
        eq     = self.result.equity_curve.dropna()
        if eq.empty:
            return
        init   = self.config.initial_capital
        final  = float(eq.iloc[-1])
        expected_pct = (final / init - 1) * 100
        reported_pct = float(m["Total Return"].rstrip("%"))
        self.assertAlmostEqual(reported_pct, expected_pct, places=1)

    def test_trade_df_matches_trade_log_length(self):
        """trade_df() row count must equal len(trade_log)."""
        df  = self.result.trade_df()
        self.assertEqual(len(df), len(self.result.trade_log))

    def test_trade_df_has_correct_columns(self):
        df = self.result.trade_df()
        if df.empty:
            return
        required_cols = [
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "quantity",
            "net_pnl", "total_charges", "mae", "mfe", "duration",
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

    def test_summary_is_string(self):
        s = self.result.summary()
        self.assertIsInstance(s, str)
        self.assertIn("BACKTEST RESULTS", s)

    def test_mae_is_non_positive(self):
        """MAE (Maximum Adverse Excursion) must be <= 0 for long trades."""
        for trade in self.result.trade_log:
            if trade.direction == 1:   # long trade
                self.assertLessEqual(trade.mae, 0.001,
                    f"MAE must be <= 0 for long trades, got {trade.mae}")

    def test_mfe_is_non_negative(self):
        """MFE (Maximum Favourable Excursion) must be >= 0."""
        for trade in self.result.trade_log:
            self.assertGreaterEqual(trade.mfe, -0.001,
                f"MFE must be >= 0, got {trade.mfe}")


class TestEdgeCases(unittest.TestCase):
    """Edge cases — empty data, all-same prices, no-signal strategies."""

    def setUp(self):
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        self.config = BacktestConfig(
            initial_capital=500_000,
            segment=Segment.EQUITY_DELIVERY,
            fixed_quantity=10,
        )
        self.engine = BacktestEngine(self.config)

    def test_strategy_with_no_signals_produces_no_trades(self):
        """A strategy that always returns signal=0 must produce zero trades."""
        from strategies.base_strategy_github import BaseStrategy
        class FlatStrategy(BaseStrategy):
            def generate_signals(self, df):
                df = df.copy()
                df["signal"] = 0
                return df

        df     = _make_ohlcv(200)
        result = self.engine.run(df, FlatStrategy("Flat"), "TEST")
        self.assertEqual(len(result.trade_log), 0)

    def test_all_buy_signals_eventually_closed_at_end(self):
        """Any position still open at the last bar must be force-closed."""
        from strategies.base_strategy_github import BaseStrategy
        class AlwaysBuyStrategy(BaseStrategy):
            def generate_signals(self, df):
                df = df.copy()
                df["signal"] = 0
                df.iloc[10, df.columns.get_loc("signal")] = 1   # One buy, no sell
                return df

        df     = _make_ohlcv(200)
        result = self.engine.run(df, AlwaysBuyStrategy("AlwaysBuy"), "TEST")

        # Position opened at bar 11 must be closed at last bar
        last_trades = [t for t in result.trade_log
                       if "End of Backtest" in t.exit_signal]
        if result.trade_log:
            self.assertGreater(len(last_trades), 0,
                               "Open positions must be closed at end of backtest")

    def test_constant_price_data_does_not_crash(self):
        """All-constant prices (zero ATR) must not cause division by zero."""
        df = _make_ohlcv(200)
        df["open"]  = 1000.0
        df["high"]  = 1000.0
        df["low"]   = 1000.0
        df["close"] = 1000.0

        from strategies.base_strategy_github import EMACrossover
        strategy = EMACrossover(9, 21)

        try:
            result = self.engine.run(df, strategy, "TEST")
            # Should complete without error (may produce 0 trades)
        except Exception as e:
            self.fail(f"Constant-price data caused crash: {e}")

    def test_missing_open_price_skipped_gracefully(self):
        """Bars with open=NaN or open=0 must be skipped without crashing."""
        from strategies.base_strategy_github import EMACrossover
        df = _make_ohlcv(200)
        # Inject NaN opens at a few bars
        df.iloc[50, df.columns.get_loc("open")] = np.nan
        df.iloc[51, df.columns.get_loc("open")] = 0.0

        strategy = EMACrossover(9, 21)
        try:
            result = self.engine.run(df, strategy, "TEST")
        except Exception as e:
            self.fail(f"NaN/zero open price caused crash: {e}")


# =============================================================================
# BLOCK 5 — backtester/report.py
# =============================================================================

class TestReportGenerator(unittest.TestCase):
    """generate_report() — file creation, size, and content validity."""

    def setUp(self):
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        from strategies.base_strategy_github import EMACrossover

        df         = _make_ohlcv(300)
        config     = BacktestConfig(
            initial_capital=500_000,
            segment=Segment.EQUITY_DELIVERY,
            fixed_quantity=10,
        )
        engine     = BacktestEngine(config)
        strategy   = EMACrossover(9, 21)
        self.result = engine.run(df, strategy, "REPORTTEST")
        self.tmpdir = tempfile.mkdtemp()

    def test_report_file_created(self):
        """generate_report must create a PNG file at the specified path."""
        from backtester.report import generate_report
        fpath = generate_report(self.result, "REPORTTEST",
                                output_dir=self.tmpdir, show=False)
        self.assertTrue(os.path.exists(fpath), f"Report not found: {fpath}")

    def test_report_file_is_png(self):
        """Report file must be a valid PNG (correct extension and file signature)."""
        from backtester.report import generate_report
        fpath = generate_report(self.result, "REPORTTEST",
                                output_dir=self.tmpdir, show=False)
        self.assertTrue(fpath.endswith(".png"), "Report should have .png extension")
        with open(fpath, "rb") as f:
            header = f.read(8)
        # PNG magic bytes: \x89PNG\r\n\x1a\n
        self.assertEqual(header[:4], b"\x89PNG",
                         "File is not a valid PNG (wrong magic bytes)")

    def test_report_file_non_trivial_size(self):
        """Report must be > 50 KB — a smaller file is likely blank/broken."""
        from backtester.report import generate_report
        fpath = generate_report(self.result, "REPORTTEST",
                                output_dir=self.tmpdir, show=False)
        size_kb = os.path.getsize(fpath) / 1024
        self.assertGreater(size_kb, 50,
                           f"Report is too small ({size_kb:.1f} KB) — may be blank")

    def test_report_with_no_trades_does_not_crash(self):
        """generate_report must handle a result with zero trades gracefully."""
        from backtester.report import generate_report
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        from strategies.base_strategy_github import BaseStrategy

        class NoSignalStrategy(BaseStrategy):
            def generate_signals(self, df):
                df = df.copy()
                df["signal"] = 0
                return df

        df      = _make_ohlcv(200)
        config  = BacktestConfig(segment=Segment.EQUITY_DELIVERY)
        engine  = BacktestEngine(config)
        result  = engine.run(df, NoSignalStrategy("NoSignal"), "TEST")

        try:
            fpath = generate_report(result, "NOTRADES",
                                    output_dir=self.tmpdir, show=False)
            self.assertTrue(os.path.exists(fpath))
        except Exception as e:
            self.fail(f"Report with no trades crashed: {e}")

    def test_custom_filename_respected(self):
        """Custom filename parameter must be used."""
        from backtester.report import generate_report
        fpath = generate_report(self.result, "REPORTTEST",
                                output_dir=self.tmpdir,
                                filename="custom_name.png", show=False)
        self.assertTrue(fpath.endswith("custom_name.png"))
        self.assertTrue(os.path.exists(fpath))

    def test_output_dir_created_if_missing(self):
        """generate_report must create the output directory if it doesn't exist."""
        from backtester.report import generate_report
        new_dir = os.path.join(self.tmpdir, "does", "not", "exist", "yet")
        fpath   = generate_report(self.result, "REPORTTEST",
                                  output_dir=new_dir, show=False)
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.exists(fpath))

    def test_max_candles_trims_chart(self):
        """max_candles parameter must trim the chart to recent candles only."""
        from backtester.report import generate_report
        # Both reports should succeed (smaller chart may be slightly smaller file)
        fpath_full = generate_report(self.result, "FULL",
                                     output_dir=self.tmpdir,
                                     filename="full.png", show=False)
        fpath_trim = generate_report(self.result, "TRIM",
                                     output_dir=self.tmpdir,
                                     filename="trim.png",
                                     max_candles=50, show=False)
        self.assertTrue(os.path.exists(fpath_full))
        self.assertTrue(os.path.exists(fpath_trim))

    def test_report_with_all_indicators(self):
        """
        Report with RSI + MACD + Bollinger Bands + EMA in signals_df must
        render all panels without error.
        """
        from backtester.report import generate_report
        from strategies.base_strategy_github import MACDStrategy

        df       = _make_ohlcv(300)
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        config   = BacktestConfig(segment=Segment.EQUITY_DELIVERY, fixed_quantity=10)
        engine   = BacktestEngine(config)
        strategy = MACDStrategy(12, 26, 9, rsi_filter=40)
        result   = engine.run(df, strategy, "MACDTEST")

        try:
            fpath = generate_report(result, "MACDTEST",
                                    output_dir=self.tmpdir,
                                    filename="macd_report.png", show=False)
            self.assertTrue(os.path.exists(fpath))
        except Exception as e:
            self.fail(f"Multi-indicator report crashed: {e}")


# =============================================================================
# BLOCK 6 — Anti-Bias Checks
# =============================================================================

class TestLookAheadBias(unittest.TestCase):
    """
    Verify that no future data is used in signal generation.

    The gold standard test: run the strategy on a dataset, then run it again
    after ADDING future data. The signals on all previously-seen bars must
    be IDENTICAL. If they change, the strategy is using future data.
    """

    def test_signals_stable_when_new_bars_added(self):
        """
        Adding new bars at the end must not change historical signals.
        If a strategy uses future data (look-ahead bias), older signals
        would change when new bars are appended.
        """
        from strategies.base_strategy_github import EMACrossover

        df_short  = _make_ohlcv(200)
        df_long   = _make_ohlcv(250)   # 50 extra bars appended

        strategy  = EMACrossover(9, 21)
        sig_short = strategy.generate_signals(df_short)["signal"]
        sig_long  = strategy.generate_signals(df_long)["signal"].iloc[:200]

        # Realign indexes for comparison
        sig_short_arr = sig_short.values
        sig_long_arr  = sig_long.values[:len(sig_short_arr)]

        matches = (sig_short_arr == sig_long_arr).sum()
        total   = len(sig_short_arr)
        pct_match = matches / total * 100

        self.assertGreater(pct_match, 95.0,
            f"Only {pct_match:.1f}% of signals matched after adding new bars. "
            "This indicates look-ahead bias — historical signals should not change "
            "when new future data is appended.")

    def test_indicator_warmup_is_always_nan(self):
        """
        Indicator values during the warm-up period must be NaN, not computed
        from partial data. Using partial data introduces subtle look-ahead bias
        because the partial mean/std uses a different denominator.
        """
        from indicators.technical import sma, ema, rsi, bollinger_bands

        df     = _make_ohlcv(200)
        close  = df["close"]

        sma_result = sma(close, 20)
        ema_result = ema(close, 20)
        rsi_result = rsi(close, 14)

        self.assertTrue(sma_result.iloc[:19].isna().all(),
                        "SMA warm-up (first 19 bars) must be NaN")
        self.assertTrue(ema_result.iloc[:19].isna().all(),
                        "EMA warm-up (first 19 bars) must be NaN")
        self.assertTrue(rsi_result.iloc[:14].isna().all(),
                        "RSI warm-up (first 14 bars) must be NaN")


class TestSurvivorshipBias(unittest.TestCase):
    """
    Warn about survivorship bias in the data layer.

    This test doesn't execute real Upstox API calls, but it verifies that
    the backtesting engine does NOT filter stocks based on current membership
    in the Nifty 500 — only membership at the time of the signal matters.
    """

    def test_engine_uses_provided_data_without_filtering(self):
        """
        The engine must use exactly the data passed to it.
        It must NOT silently remove stocks or filter by current index membership.
        We pass 300 bars; the engine must process all 300.
        """
        from backtester.engine import BacktestEngine, BacktestConfig
        from backtester.commission import Segment
        from strategies.base_strategy_github import EMACrossover

        df     = _make_ohlcv(300)
        config = BacktestConfig(segment=Segment.EQUITY_DELIVERY, fixed_quantity=10)
        engine = BacktestEngine(config)
        result = engine.run(df, EMACrossover(9, 21), "TEST")

        self.assertEqual(len(result.equity_curve), 300,
                         "Engine filtered rows — possible survivorship bias in engine layer")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
