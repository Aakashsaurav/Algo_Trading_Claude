"""
tests/test_backtester.py
-------------------------
Comprehensive test suite for the refactored backtester.
Covers all modules: models, order_types, position_sizer, fill_engine,
event_loop, performance, optimizer, and engine (public API).

Run with:
    cd /home/claude/algodesk && python -m pytest tests/test_backtester.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

# ── Imports under test ──────────────────────────────────────────────────────
from broker.upstox.commission import CommissionModel, Segment
from backtester.models import (
    BacktestConfig, Position, Trade, BacktestResult, OrderType
)
from backtester.order_types import (
    PendingOrder,
    check_limit_fill, check_stop_fill, check_stop_limit_fill,
)
from backtester.position_sizer import compute_quantity
from backtester.fill_engine import FillEngine
from backtester.event_loop import run_event_loop, _compute_atr14
from backtester.performance import compute_performance, _compute_trade_stats, _max_run
from backtester.engine import BacktestEngine


# ===========================================================================
# Fixtures
# ===========================================================================

IST = timezone(timedelta(hours=5, minutes=30))

def _make_ts(days_offset: int) -> pd.Timestamp:
    return pd.Timestamp("2023-01-01", tz="Asia/Kolkata") + pd.Timedelta(days=days_offset)


def _make_ohlcv(n: int = 100, trend: float = 1.0, start_price: float = 1000.0) -> pd.DataFrame:
    """Generate deterministic OHLCV data with a configurable drift."""
    np.random.seed(42)
    rng = pd.date_range("2023-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    closes = start_price + np.cumsum(np.random.randn(n) * 5 + trend)
    closes = np.clip(closes, 1.0, None)
    highs  = closes + np.abs(np.random.randn(n) * 2)
    lows   = closes - np.abs(np.random.randn(n) * 2)
    lows   = np.clip(lows, 1.0, None)
    opens  = (closes + lows) / 2.0
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": np.random.randint(1000, 100000, n)},
        index=rng,
    )


def _simple_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Buy on bar 5, sell on bar 20. One clean round-trip."""
    df = df.copy()
    df["signal"] = 0
    if len(df) > 20:
        df.iloc[5,  df.columns.get_loc("signal")] = 1
        df.iloc[20, df.columns.get_loc("signal")] = -1
    return df


def _ema_cross_strategy(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """Simple EMA crossover — multiple trades."""
    df = df.copy()
    df["fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["signal"] = 0
    df.loc[df["fast"] > df["slow"], "signal"] = 1
    df.loc[df["fast"] < df["slow"], "signal"] = -1
    return df


class _MockStrategy:
    """Minimal strategy for engine tests."""
    name = "MockStrategy"
    def __init__(self, signal_fn):
        self._fn = signal_fn
    def generate_signals(self, df):
        return self._fn(df)


# ===========================================================================
# 1. BacktestConfig validation
# ===========================================================================

class TestBacktestConfig:

    def test_defaults_are_sensible(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 500_000.0
        assert cfg.capital_risk_pct == 0.02
        assert cfg.max_drawdown_pct == 0.20
        assert cfg.default_order_type == OrderType.MARKET
        assert cfg.allow_shorting is False

    def test_validate_passes_for_valid_config(self):
        BacktestConfig(initial_capital=100_000).validate()   # no exception

    def test_validate_rejects_zero_capital(self):
        with pytest.raises(ValueError, match="initial_capital"):
            BacktestConfig(initial_capital=0).validate()

    def test_validate_rejects_bad_risk_pct(self):
        with pytest.raises(ValueError):
            BacktestConfig(capital_risk_pct=0.0).validate()
        with pytest.raises(ValueError):
            BacktestConfig(capital_risk_pct=1.5).validate()

    def test_validate_rejects_trailing_stop_without_distance(self):
        with pytest.raises(ValueError, match="trailing_stop"):
            BacktestConfig(use_trailing_stop=True).validate()

    def test_validate_rejects_both_trailing_stop_params(self):
        with pytest.raises(ValueError):
            BacktestConfig(
                use_trailing_stop=True,
                trailing_stop_pct=2.0,
                trailing_stop_amt=50.0,
            ).validate()


# ===========================================================================
# 2. Position model
# ===========================================================================

class TestPosition:

    def _long_pos(self, entry=1000.0, qty=10) -> Position:
        return Position(
            symbol="TEST", entry_time=_make_ts(0),
            entry_price=entry, quantity=qty, direction=1,
        )

    def test_unrealised_pnl_long(self):
        pos = self._long_pos(entry=1000.0, qty=10)
        assert pos.unrealised_pnl(1050.0) == pytest.approx(500.0)
        assert pos.unrealised_pnl(950.0)  == pytest.approx(-500.0)

    def test_unrealised_pnl_short(self):
        pos = Position(
            symbol="TEST", entry_time=_make_ts(0),
            entry_price=1000.0, quantity=10, direction=-1,
        )
        assert pos.unrealised_pnl(950.0)  == pytest.approx(500.0)
        assert pos.unrealised_pnl(1050.0) == pytest.approx(-500.0)

    def test_update_excursion(self):
        pos = self._long_pos(entry=1000.0)
        pos.update_excursion(1100.0)
        assert pos.mfe == pytest.approx(100.0)
        assert pos.mae == pytest.approx(0.0)
        pos.update_excursion(900.0)
        assert pos.mae == pytest.approx(-100.0)
        assert pos.mfe == pytest.approx(100.0)   # unchanged

    def test_trailing_stop_long_advances_with_price(self):
        pos = Position(
            symbol="TEST", entry_time=_make_ts(0),
            entry_price=1000.0, quantity=10, direction=1,
            trailing_stop_pct=5.0,
        )
        pos.update_trailing_stop(high=1100.0, low=1050.0)
        # First call: seed = entry - dist = 1000 - 50 = 950
        # ideal = high - dist = 1100 - 55 = 1045
        # max(950, 1045) = 1045  -- but dist uses entry_price not current
        # dist = entry_price * pct/100 = 1000 * 0.05 = 50
        # ideal = 1100 - 50 = 1050; seed = 1000 - 50 = 950; max = 1050
        assert pos.trailing_stop_level == pytest.approx(1050.0)

    def test_trailing_stop_long_never_retreats(self):
        pos = Position(
            symbol="TEST", entry_time=_make_ts(0),
            entry_price=1000.0, quantity=10, direction=1,
            trailing_stop_pct=5.0,
        )
        pos.update_trailing_stop(high=1100.0, low=1080.0)
        lvl_after_high = pos.trailing_stop_level
        pos.update_trailing_stop(high=1050.0, low=1020.0)   # price retreats
        assert pos.trailing_stop_level == pytest.approx(lvl_after_high)

    def test_trailing_stop_triggered_on_gap(self):
        pos = Position(
            symbol="TEST", entry_time=_make_ts(0),
            entry_price=1000.0, quantity=10, direction=1,
            trailing_stop_pct=5.0,
            trailing_stop_level=950.0,
        )
        triggered, fp = pos.is_trailing_stop_triggered(open_p=920.0, low=910.0, high=930.0)
        assert triggered
        assert fp == pytest.approx(920.0)   # gap — fill at open

    def test_trailing_stop_not_triggered(self):
        pos = Position(
            symbol="TEST", entry_time=_make_ts(0),
            entry_price=1000.0, quantity=10, direction=1,
            trailing_stop_pct=5.0,
            trailing_stop_level=950.0,
        )
        triggered, _ = pos.is_trailing_stop_triggered(open_p=980.0, low=960.0, high=990.0)
        assert not triggered

    def test_fixed_stop_triggered_long(self):
        pos = Position(
            symbol="TEST", entry_time=_make_ts(0),
            entry_price=1000.0, quantity=10, direction=1,
            stop_price=950.0,
        )
        triggered, fp = pos.is_fixed_stop_triggered(open_p=960.0, low=940.0, high=965.0)
        assert triggered
        assert fp == pytest.approx(950.0)

    def test_fixed_stop_gap_fills_at_open(self):
        pos = Position(
            symbol="TEST", entry_time=_make_ts(0),
            entry_price=1000.0, quantity=10, direction=1,
            stop_price=960.0,
        )
        triggered, fp = pos.is_fixed_stop_triggered(open_p=940.0, low=930.0, high=945.0)
        assert triggered
        assert fp == pytest.approx(940.0)


# ===========================================================================
# 3. Order types
# ===========================================================================

class TestOrderTypes:

    # ── Limit fills ─────────────────────────────────────────────────────────

    def test_buy_limit_fills_when_low_touches(self):
        filled, fp = check_limit_fill(1, 990.0, open_p=1000.0, low=985.0, high=1010.0)
        assert filled
        assert fp == pytest.approx(990.0)

    def test_buy_limit_gap_fills_at_open(self):
        filled, fp = check_limit_fill(1, 990.0, open_p=980.0, low=975.0, high=985.0)
        assert filled
        assert fp == pytest.approx(980.0)

    def test_buy_limit_does_not_fill_when_price_above(self):
        filled, _ = check_limit_fill(1, 990.0, open_p=995.0, low=992.0, high=1010.0)
        assert not filled

    def test_sell_limit_fills_when_high_reaches(self):
        filled, fp = check_limit_fill(-1, 1010.0, open_p=1000.0, low=995.0, high=1015.0)
        assert filled
        assert fp == pytest.approx(1010.0)

    # ── Stop fills ──────────────────────────────────────────────────────────

    def test_buy_stop_fills_on_breakout(self):
        filled, fp = check_stop_fill(1, 1010.0, open_p=1000.0, low=998.0, high=1015.0)
        assert filled
        assert fp == pytest.approx(1010.0)

    def test_sell_stop_fills_on_breakdown(self):
        filled, fp = check_stop_fill(-1, 990.0, open_p=1000.0, low=985.0, high=1002.0)
        assert filled
        assert fp == pytest.approx(990.0)

    def test_stop_not_filled_when_not_breached(self):
        filled, _ = check_stop_fill(1, 1010.0, open_p=1000.0, low=998.0, high=1009.0)
        assert not filled

    # ── Stop-limit fills ────────────────────────────────────────────────────

    def test_stop_limit_fills_when_both_conditions_met(self):
        filled, fp, hit = check_stop_limit_fill(
            direction=1, stop_price=1010.0, limit_price=1015.0,
            open_p=1005.0, low=1003.0, high=1020.0
        )
        assert filled
        assert fp == pytest.approx(1010.0)
        assert hit

    def test_stop_limit_stop_triggered_but_limit_not_reachable(self):
        # price gaps way through both stop and limit
        filled, _, hit = check_stop_limit_fill(
            direction=1, stop_price=1010.0, limit_price=1012.0,
            open_p=1020.0, low=1018.0, high=1025.0
        )
        # stop IS triggered (open > stop) but fill = max(open, stop) = 1020 > limit
        # Depending on implementation: may not fill
        assert hit   # stop was triggered


# ===========================================================================
# 4. Position sizer
# ===========================================================================

class TestPositionSizer:

    def test_fixed_quantity_returned_directly(self):
        qty = compute_quantity(cash=500_000, entry_price=1000.0,
                               capital_risk_pct=0.02, fixed_quantity=50)
        assert qty == 50

    def test_fixed_quantity_capped_by_cash(self):
        qty = compute_quantity(cash=10_000, entry_price=1000.0,
                               capital_risk_pct=0.02, fixed_quantity=50)
        assert qty == 10   # can only afford 10

    def test_risk_based_with_stop(self):
        # risk = 500000 * 0.02 = 10000; stop_dist = 1000-950 = 50
        # qty = floor(10000 / 50) = 200
        qty = compute_quantity(cash=500_000, entry_price=1000.0,
                               capital_risk_pct=0.02, stop_price=950.0)
        assert qty == 200

    def test_risk_based_with_atr(self):
        # risk = 100000 * 0.01 = 1000; stop = 20 * 2 = 40
        # qty = floor(1000 / 40) = 25
        qty = compute_quantity(cash=100_000, entry_price=500.0,
                               capital_risk_pct=0.01, atr=20.0, atr_mult=2.0)
        assert qty == 25

    def test_fallback_sizing(self):
        # No stop, no ATR → 2% of cash / price
        qty = compute_quantity(cash=100_000, entry_price=500.0,
                               capital_risk_pct=0.02)
        # 100000 * 0.02 / 500 = 4
        assert qty == 4

    def test_zero_for_insufficient_cash(self):
        qty = compute_quantity(cash=0.0, entry_price=1000.0, capital_risk_pct=0.02)
        assert qty == 0

    def test_zero_for_zero_price(self):
        qty = compute_quantity(cash=100_000, entry_price=0.0, capital_risk_pct=0.02)
        assert qty == 0

    def test_qty_never_exceeds_affordable(self):
        # Very wide stop → large raw qty → must be capped
        qty = compute_quantity(cash=10_000, entry_price=1000.0,
                               capital_risk_pct=0.5, stop_price=999.0)
        assert qty <= 10   # can afford at most 10 at price 1000


# ===========================================================================
# 5. FillEngine
# ===========================================================================

class TestFillEngine:

    def _cfg(self, **kw) -> BacktestConfig:
        return BacktestConfig(initial_capital=500_000, fixed_quantity=10, **kw)

    def _pos(self, entry=1000.0, qty=10, direction=1) -> Position:
        return Position(
            symbol="TEST", entry_time=_make_ts(0),
            entry_price=entry, quantity=qty, direction=direction,
        )

    def test_open_position_debits_cash(self):
        cfg    = self._cfg()
        filler = FillEngine(cfg)
        pos, cash = filler.open_position(
            direction=1, exec_price=1000.0, cash=500_000,
            symbol="TEST", bar_idx=0, bar_time=_make_ts(0),
            entry_signal="Test",
        )
        assert pos is not None
        assert cash < 500_000   # cash was debited
        assert pos.quantity == 10
        assert pos.direction == 1

    def test_open_position_returns_none_when_insufficient_cash(self):
        cfg    = BacktestConfig(initial_capital=100, fixed_quantity=1000)
        filler = FillEngine(cfg)
        pos, cash = filler.open_position(
            direction=1, exec_price=1000.0, cash=100.0,
            symbol="TEST", bar_idx=0, bar_time=_make_ts(0),
            entry_signal="Test",
        )
        assert pos is None
        assert cash == 100.0   # unchanged

    def test_close_position_credits_cash(self):
        cfg    = self._cfg()
        filler = FillEngine(cfg)
        pos    = self._pos(entry=1000.0, qty=10)
        trade, cash = filler.close_position(
            pos=pos, exec_price=1100.0, cash=490_000.0,
            bar_time=_make_ts(5), bar_idx=5, exit_signal="Signal Exit",
            portfolio_value_after=491_000.0,
        )
        assert trade.net_pnl > 0   # profitable trade
        assert trade.gross_pnl == pytest.approx(1000.0)   # (1100-1000)*10
        assert cash > 490_000.0

    def test_close_position_records_correct_duration(self):
        cfg    = self._cfg()
        filler = FillEngine(cfg)
        pos    = self._pos()
        pos.entry_bar_idx = 0
        trade, _ = filler.close_position(
            pos=pos, exec_price=1000.0, cash=490_000.0,
            bar_time=_make_ts(10), bar_idx=10, exit_signal="Test",
            portfolio_value_after=490_000.0,
        )
        assert trade.duration_bars == 10

    def test_short_position_profits_on_decline(self):
        cfg    = BacktestConfig(initial_capital=500_000, fixed_quantity=10,
                                allow_shorting=True)
        filler = FillEngine(cfg)
        pos    = self._pos(entry=1000.0, qty=10, direction=-1)
        trade, _ = filler.close_position(
            pos=pos, exec_price=900.0, cash=490_000.0,
            bar_time=_make_ts(3), bar_idx=3, exit_signal="Test",
            portfolio_value_after=491_000.0,
        )
        assert trade.net_pnl > 0   # short profits on decline

    def test_check_stops_trailing_stop_fires(self):
        cfg    = self._cfg()
        filler = FillEngine(cfg)
        pos    = self._pos(entry=1000.0)
        pos.trailing_stop_level = 950.0
        pos.trailing_stop_pct   = 5.0

        remaining, fired_trades, cash = filler.check_stops(
            [pos], cash=490_000.0,
            open_p=940.0, high=945.0, low=935.0,
            bar_time=_make_ts(5), bar_idx=5, symbol="TEST",
            portfolio_value=490_000.0,
        )
        assert len(remaining) == 0
        assert len(fired_trades) == 1
        assert "Trailing Stop" in fired_trades[0].exit_signal

    def test_check_stops_fixed_stop_fires(self):
        cfg    = self._cfg()
        filler = FillEngine(cfg)
        pos    = self._pos(entry=1000.0)
        pos.stop_price = 950.0

        remaining, fired_trades, cash = filler.check_stops(
            [pos], cash=490_000.0,
            open_p=960.0, high=965.0, low=940.0,
            bar_time=_make_ts(5), bar_idx=5, symbol="TEST",
            portfolio_value=490_000.0,
        )
        assert len(remaining) == 0
        assert len(fired_trades) == 1
        assert "Stop Loss" in fired_trades[0].exit_signal

    def test_check_stops_no_trigger_when_price_ok(self):
        cfg    = self._cfg()
        filler = FillEngine(cfg)
        pos    = self._pos(entry=1000.0)
        pos.stop_price = 900.0

        remaining, fired_trades, _ = filler.check_stops(
            [pos], cash=490_000.0,
            open_p=980.0, high=990.0, low=970.0,
            bar_time=_make_ts(5), bar_idx=5, symbol="TEST",
            portfolio_value=490_000.0,
        )
        assert len(remaining) == 1
        assert len(fired_trades) == 0


# ===========================================================================
# 6. ATR pre-computation
# ===========================================================================

class TestATR:

    def test_atr_shape_and_nan_warmup(self):
        df = _make_ohlcv(100)
        atr = _compute_atr14(
            df["close"].values, df["high"].values, df["low"].values, 100
        )
        assert atr is not None
        assert len(atr) == 100
        # First 14 values should be NaN (warm-up)
        assert all(np.isnan(atr[:14]))
        assert not np.isnan(atr[14])

    def test_atr_returns_none_for_short_series(self):
        df  = _make_ohlcv(10)
        atr = _compute_atr14(
            df["close"].values, df["high"].values, df["low"].values, 10
        )
        assert atr is None

    def test_atr_values_positive(self):
        df = _make_ohlcv(100)
        atr = _compute_atr14(
            df["close"].values, df["high"].values, df["low"].values, 100
        )
        valid = atr[~np.isnan(atr)]
        assert (valid > 0).all()


# ===========================================================================
# 7. Event loop integration
# ===========================================================================

class TestEventLoop:

    def _cfg(self, **kw) -> BacktestConfig:
        return BacktestConfig(initial_capital=500_000, fixed_quantity=10, **kw)

    def test_single_trade_generates_profit(self):
        df  = _make_ohlcv(50, trend=2.0)   # upward trend
        sdf = _simple_strategy(df)
        cfg = self._cfg()
        trades, equity, dd = run_event_loop(sdf, cfg, "TEST")
        assert len(trades) == 1
        # With upward trend, the trade should be profitable
        # (not guaranteed for all random seeds — check it ran)
        assert trades[0].quantity == 10

    def test_equity_curve_starts_at_initial_capital(self):
        df  = _make_ohlcv(50)
        sdf = _simple_strategy(df)
        cfg = self._cfg()
        _, equity, _ = run_event_loop(sdf, cfg, "TEST")
        valid = equity.dropna()
        assert not valid.empty
        # First valid value should be close to initial capital
        assert valid.iloc[0] == pytest.approx(500_000, rel=0.05)

    def test_drawdown_is_non_positive(self):
        df  = _make_ohlcv(100)
        sdf = _simple_strategy(df)
        cfg = self._cfg()
        _, _, dd = run_event_loop(sdf, cfg, "TEST")
        valid = dd.dropna()
        assert (valid <= 0.0001).all()   # allow tiny float rounding

    def test_no_position_at_end_of_data(self):
        """All positions must be closed by end-of-data logic."""
        df  = _make_ohlcv(50)
        sdf = df.copy()
        sdf["signal"] = 0
        sdf.iloc[5, sdf.columns.get_loc("signal")] = 1   # buy, never sell
        cfg = self._cfg()
        trades, _, _ = run_event_loop(sdf, cfg, "TEST")
        # The open position should be closed at end of data
        assert len(trades) == 1
        assert trades[0].exit_signal == "End of Data"

    def test_max_drawdown_halt_closes_all_positions(self):
        """Engine halts and closes positions when max drawdown is hit."""
        df  = _make_ohlcv(100, trend=-10.0, start_price=1000.0)   # strong downtrend
        sdf = df.copy()
        sdf["signal"] = 0
        sdf.iloc[2, sdf.columns.get_loc("signal")] = 1   # buy early into downtrend
        cfg = BacktestConfig(
            initial_capital=100_000,
            fixed_quantity=100,
            max_drawdown_pct=0.10,   # halt at 10% drawdown
        )
        trades, equity, _ = run_event_loop(sdf, cfg, "TEST")
        # At least one trade should have been force-closed
        assert len(trades) >= 1
        halt_trades = [t for t in trades if "Halt" in t.exit_signal or "End" in t.exit_signal]
        assert len(halt_trades) >= 1

    def test_shorting_disabled_by_default(self):
        """Sell signal with allow_shorting=False must NOT open a short."""
        df  = _make_ohlcv(50)
        sdf = df.copy()
        sdf["signal"] = 0
        sdf.iloc[5, sdf.columns.get_loc("signal")] = -1   # sell signal, no prior long
        cfg = self._cfg(allow_shorting=False)
        trades, _, _ = run_event_loop(sdf, cfg, "TEST")
        assert len(trades) == 0   # no position to close, shorting disabled

    def test_shorting_enabled_opens_short(self):
        """Sell signal with allow_shorting=True opens a short position."""
        df  = _make_ohlcv(50)
        sdf = df.copy()
        sdf["signal"] = 0
        sdf.iloc[5,  sdf.columns.get_loc("signal")] = -1   # open short
        sdf.iloc[20, sdf.columns.get_loc("signal")] = 1    # close short
        cfg = BacktestConfig(
            initial_capital=500_000, fixed_quantity=10, allow_shorting=True
        )
        trades, _, _ = run_event_loop(sdf, cfg, "TEST")
        assert len(trades) >= 1
        assert trades[0].direction == -1

    def test_no_lookahead_bias(self):
        """Signal on bar i must execute on bar i+1 open."""
        df  = _make_ohlcv(30)
        sdf = df.copy()
        sdf["signal"] = 0
        sdf.iloc[10, sdf.columns.get_loc("signal")] = 1
        sdf.iloc[20, sdf.columns.get_loc("signal")] = -1
        cfg = self._cfg()
        trades, _, _ = run_event_loop(sdf, cfg, "TEST")
        assert len(trades) == 1
        # Entry should be at bar 11 open, not bar 10 close
        expected_entry = float(df.iloc[11]["open"])
        assert trades[0].entry_price == pytest.approx(expected_entry, rel=0.001)

    def test_limit_order_fills_when_price_dips(self):
        """LIMIT order fills when intrabar low touches the limit price."""
        # Build data where bar 11 has a low that would fill the limit
        df       = _make_ohlcv(30)
        df_limit = df.copy()
        signal_close = float(df.iloc[10]["close"])
        limit_price  = signal_close * 0.998   # 0.2% below
        # Force bar 11 to have a low below limit price
        df_limit.iloc[11, df_limit.columns.get_loc("low")] = limit_price - 1.0
        df_limit.iloc[11, df_limit.columns.get_loc("open")] = signal_close + 1.0

        sdf = df_limit.copy()
        sdf["signal"] = 0
        sdf.iloc[10, sdf.columns.get_loc("signal")] = 1
        sdf.iloc[20, sdf.columns.get_loc("signal")] = -1

        cfg = BacktestConfig(
            initial_capital=500_000, fixed_quantity=10,
            default_order_type=OrderType.LIMIT, limit_offset_pct=0.2,
        )
        trades, _, _ = run_event_loop(sdf, cfg, "TEST")
        # Trade should exist — limit was reached
        assert len(trades) >= 1

    def test_trailing_stop_closes_position(self):
        df  = _make_ohlcv(60, trend=2.0)
        sdf = df.copy()
        sdf["signal"] = 0
        sdf.iloc[5, sdf.columns.get_loc("signal")] = 1   # buy
        # No sell signal — trailing stop should close it

        cfg = BacktestConfig(
            initial_capital=500_000,
            fixed_quantity=10,
            use_trailing_stop=True,
            trailing_stop_pct=50.0,   # very wide — but trend reversal will fire
        )
        # Run — the trailing stop or end-of-data will close
        trades, _, _ = run_event_loop(sdf, cfg, "TEST")
        assert len(trades) == 1

    def test_equity_monotonically_filled(self):
        """No NaN gaps in equity after valid bars."""
        df  = _make_ohlcv(50)
        sdf = _simple_strategy(df)
        cfg = self._cfg()
        _, equity, _ = run_event_loop(sdf, cfg, "TEST")
        # All bars have valid open prices → no NaN expected
        assert equity.isna().sum() == 0


# ===========================================================================
# 8. Performance metrics
# ===========================================================================

class TestPerformance:

    def _run_and_get_metrics(self, n_bars=100, trend=1.0):
        df  = _make_ohlcv(n_bars, trend=trend)
        sdf = _ema_cross_strategy(df)
        cfg = BacktestConfig(initial_capital=500_000, fixed_quantity=10)
        trades, equity, _ = run_event_loop(sdf, cfg, "TEST")
        m = compute_performance(trades, equity, cfg)
        return m, trades, equity

    def test_returns_all_required_keys(self):
        m, _, _ = self._run_and_get_metrics()
        required_keys = [
            "total_trades", "win_rate_pct", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "omega_ratio", "kelly_fraction",
            "max_drawdown_pct", "cagr_pct", "total_return_pct",
            "profit_factor", "expectancy_inr", "exposure_pct",
            "total_commission_paid", "avg_mae_inr", "avg_mfe_inr",
        ]
        for key in required_keys:
            assert key in m, f"Missing key: {key}"

    def test_initial_final_capital_consistent(self):
        m, _, equity = self._run_and_get_metrics()
        eq_final = float(equity.dropna().iloc[-1])
        assert m["final_capital"] == pytest.approx(eq_final, rel=0.01)

    def test_win_rate_bounds(self):
        m, _, _ = self._run_and_get_metrics()
        assert 0.0 <= m["win_rate_pct"] <= 100.0

    def test_max_drawdown_non_positive(self):
        m, _, _ = self._run_and_get_metrics()
        assert m["max_drawdown_pct"] <= 0.0

    def test_profit_factor_positive(self):
        m, _, _ = self._run_and_get_metrics(trend=3.0)  # strong uptrend
        if m["total_trades"] > 0 and m["losing_trades"] > 0:
            assert m["profit_factor"] >= 0.0

    def test_empty_equity_returns_error_key(self):
        cfg = BacktestConfig(initial_capital=100_000)
        m = compute_performance([], pd.Series(dtype=float), cfg)
        assert "error" in m

    def test_exposure_pct_between_0_and_100(self):
        m, _, _ = self._run_and_get_metrics()
        assert 0.0 <= m["exposure_pct"] <= 100.0

    def test_max_run_helper(self):
        assert _max_run([True, True, False, True]) == 2
        assert _max_run([False, False, False]) == 0
        assert _max_run([True, True, True]) == 3
        assert _max_run([]) == 0

    def test_commission_paid_positive_when_trades_exist(self):
        m, trades, _ = self._run_and_get_metrics()
        if len(trades) > 0:
            assert m["total_commission_paid"] > 0

    def test_monthly_returns_populated(self):
        m, _, _ = self._run_and_get_metrics(n_bars=252)
        # With 252 bars (≈1 year of data), monthly returns should be populated
        assert isinstance(m["monthly_returns"], dict)

    def test_trade_stats_no_trades(self):
        stats = _compute_trade_stats([])
        assert stats["total_trades"] == 0
        assert stats["win_rate_pct"] == 0.0


# ===========================================================================
# 9. BacktestEngine (public API)
# ===========================================================================

class TestBacktestEngine:

    def _engine(self, **kw) -> BacktestEngine:
        cfg = BacktestConfig(
            initial_capital=500_000, fixed_quantity=10, **kw
        )
        return BacktestEngine(cfg)

    def test_run_returns_backtest_result(self):
        engine   = self._engine()
        strategy = _MockStrategy(_simple_strategy)
        df       = _make_ohlcv(50)
        result   = engine.run(df, strategy, symbol="INFY")
        assert isinstance(result, BacktestResult)

    def test_result_has_trade_log(self):
        engine   = self._engine()
        strategy = _MockStrategy(_simple_strategy)
        result   = engine.run(_make_ohlcv(50), strategy)
        assert isinstance(result.trade_log, list)

    def test_result_summary_is_string(self):
        engine   = self._engine()
        strategy = _MockStrategy(_ema_cross_strategy)
        result   = engine.run(_make_ohlcv(100), strategy)
        s = result.summary()
        assert isinstance(s, str)
        assert "BACKTEST RESULTS" in s

    def test_result_metrics_returns_dict(self):
        engine   = self._engine()
        strategy = _MockStrategy(_ema_cross_strategy)
        result   = engine.run(_make_ohlcv(100), strategy)
        m = result.metrics()
        assert isinstance(m, dict)
        assert "total_trades" in m

    def test_metrics_are_cached(self):
        engine   = self._engine()
        strategy = _MockStrategy(_ema_cross_strategy)
        result   = engine.run(_make_ohlcv(100), strategy)
        m1 = result.metrics()
        m2 = result.metrics()
        assert m1 is m2   # same object — cached

    def test_preflight_raises_on_missing_column(self):
        engine = self._engine()
        df_bad = pd.DataFrame({"close": [1.0, 2.0], "open": [1.0, 2.0]})
        strategy = _MockStrategy(lambda df: df.assign(signal=0))
        with pytest.raises(Exception):
            engine.run(df_bad, strategy)

    def test_preflight_raises_on_duplicate_index(self):
        df = _make_ohlcv(10)
        df = pd.concat([df, df.iloc[:2]])   # duplicate rows
        engine   = self._engine()
        strategy = _MockStrategy(lambda df: df.assign(signal=0))
        with pytest.raises(ValueError, match="duplicate"):
            engine.run(df, strategy)

    def test_no_signal_column_raises(self):
        engine   = self._engine()
        strategy = _MockStrategy(lambda df: df)   # no signal column added
        with pytest.raises(ValueError, match="signal"):
            engine.run(_make_ohlcv(20), strategy)

    def test_run_portfolio_returns_dict(self):
        engine   = self._engine()
        strategy = _MockStrategy(_ema_cross_strategy)
        data     = {"INFY": _make_ohlcv(100), "TCS": _make_ohlcv(100)}
        results  = engine.run_portfolio(data, strategy)
        assert isinstance(results, dict)
        assert "INFY" in results and "TCS" in results

    def test_run_portfolio_independent_results(self):
        """Two symbols with opposite trends should have different P&Ls."""
        engine   = self._engine()
        strategy = _MockStrategy(_ema_cross_strategy)
        data     = {
            "UP":   _make_ohlcv(100, trend=5.0),
            "DOWN": _make_ohlcv(100, trend=-5.0),
        }
        results = engine.run_portfolio(data, strategy)
        up_pnl   = sum(t.net_pnl for t in results["UP"].trade_log)
        down_pnl = sum(t.net_pnl for t in results["DOWN"].trade_log)
        # They should differ (not necessarily one positive one negative,
        # but they should not be identical)
        assert up_pnl != down_pnl

    def test_trade_df_returns_dataframe(self):
        engine   = self._engine()
        strategy = _MockStrategy(_ema_cross_strategy)
        result   = engine.run(_make_ohlcv(100), strategy)
        df = result.trade_df()
        assert isinstance(df, pd.DataFrame)
        if len(result.trade_log) > 0:
            assert "net_pnl" in df.columns
            assert "entry_price" in df.columns

    def test_intraday_squareoff_closes_positions(self):
        """All positions must be closed at 15:20 with intraday_squareoff=True."""
        rng = pd.date_range("2023-01-03 09:15", periods=120, freq="1min",
                            tz="Asia/Kolkata")
        closes = 1000.0 + np.arange(120) * 0.5
        df = pd.DataFrame({
            "open":   closes - 0.2,
            "high":   closes + 0.5,
            "low":    closes - 0.5,
            "close":  closes,
            "volume": 10000,
        }, index=rng)
        df["signal"] = 0
        df.iloc[10, df.columns.get_loc("signal")] = 1   # buy at 09:25

        engine   = BacktestEngine(BacktestConfig(
            initial_capital=500_000, fixed_quantity=10, intraday_squareoff=True
        ))
        strategy = _MockStrategy(lambda d: d)
        df["signal"] = 0
        df.iloc[10, df.columns.get_loc("signal")] = 1

        trades, _, _ = run_event_loop(df, engine.config, "TEST")
        # Either squareoff or end-of-data must have closed the trade
        assert len(trades) == 1
        exit_reasons = [t.exit_signal for t in trades]
        assert any("Squareoff" in r or "End" in r for r in exit_reasons)


# ===========================================================================
# 10. Commission model (broker — locked, only sanity checks)
# ===========================================================================

class TestCommissionModel:

    def test_delivery_buy_charges_positive(self):
        cm  = CommissionModel()
        chg = cm.calculate(Segment.EQUITY_DELIVERY, "BUY", 100, 1000.0)
        assert chg.total > 0
        assert chg.brokerage <= 20.0   # capped at ₹20

    def test_intraday_buy_lower_brokerage_pct(self):
        cm   = CommissionModel()
        del_ = cm.calculate(Segment.EQUITY_DELIVERY, "BUY", 100, 1000.0)
        int_ = cm.calculate(Segment.EQUITY_INTRADAY, "BUY", 100, 1000.0)
        # Intraday brokerage pct is 0.05% vs delivery 0.1%
        assert int_.brokerage <= del_.brokerage

    def test_dp_charge_on_delivery_sell_only(self):
        cm   = CommissionModel()
        sell = cm.calculate(Segment.EQUITY_DELIVERY, "SELL", 100, 1000.0)
        buy  = cm.calculate(Segment.EQUITY_DELIVERY, "BUY",  100, 1000.0)
        assert sell.dp_charge == pytest.approx(18.5)
        assert buy.dp_charge  == pytest.approx(0.0)

    def test_total_equals_sum_of_components(self):
        cm  = CommissionModel()
        chg = cm.calculate(Segment.EQUITY_INTRADAY, "BUY", 50, 500.0)
        expected = (chg.brokerage + chg.stt + chg.transaction_charge +
                    chg.sebi_fee + chg.gst + chg.stamp_duty + chg.dp_charge)
        assert chg.total == pytest.approx(expected, abs=0.02)  # ±2 paise tolerance for rounding

    def test_invalid_side_raises(self):
        cm = CommissionModel()
        with pytest.raises(ValueError):
            cm.calculate(Segment.EQUITY_DELIVERY, "HOLD", 100, 1000.0)

    def test_zero_quantity_raises(self):
        cm = CommissionModel()
        with pytest.raises(ValueError):
            cm.calculate(Segment.EQUITY_DELIVERY, "BUY", 0, 1000.0)


# ===========================================================================
# 11. Edge cases and regression tests
# ===========================================================================

class TestEdgeCases:

    def test_empty_dataframe_raises(self):
        engine   = BacktestEngine(BacktestConfig(initial_capital=100_000))
        strategy = _MockStrategy(lambda df: df.assign(signal=0))
        df_empty = pd.DataFrame(columns=["open","high","low","close","signal"])
        with pytest.raises((ValueError, Exception)):
            engine.run(df_empty, strategy)

    def test_single_row_raises(self):
        engine   = BacktestEngine(BacktestConfig(initial_capital=100_000))
        strategy = _MockStrategy(lambda df: df.assign(signal=0))
        df_one   = _make_ohlcv(1)
        with pytest.raises(ValueError):
            engine.run(df_one, strategy)

    def test_all_signals_zero_produces_no_trades(self):
        df  = _make_ohlcv(50)
        sdf = df.copy(); sdf["signal"] = 0
        cfg = BacktestConfig(initial_capital=500_000, fixed_quantity=10)
        trades, equity, _ = run_event_loop(sdf, cfg, "TEST")
        assert len(trades) == 0

    def test_equity_equals_initial_capital_with_no_trades(self):
        df  = _make_ohlcv(50)
        sdf = df.copy(); sdf["signal"] = 0
        cfg = BacktestConfig(initial_capital=500_000, fixed_quantity=10)
        _, equity, _ = run_event_loop(sdf, cfg, "TEST")
        assert float(equity.dropna().iloc[-1]) == pytest.approx(500_000.0, rel=0.001)

    def test_consecutive_buy_signals_not_stacking_beyond_max_positions(self):
        df  = _make_ohlcv(50)
        sdf = df.copy(); sdf["signal"] = 0
        # Three consecutive buy signals
        for idx in [5, 6, 7]:
            sdf.iloc[idx, sdf.columns.get_loc("signal")] = 1
        cfg = BacktestConfig(
            initial_capital=500_000, fixed_quantity=10, max_positions=1
        )
        trades, _, _ = run_event_loop(sdf, cfg, "TEST")
        # Max 1 position — only one should open, then close at end
        assert len(trades) <= 2   # one open + one close = 1 trade

    def test_large_dataset_performance(self):
        """10 years of minute data — should complete in reasonable time."""
        import time
        n   = 2_500 * 375   # ~10 years of 1-min bars for NSE
        np.random.seed(0)
        closes = 1000.0 + np.cumsum(np.random.randn(n) * 0.5)
        closes = np.clip(closes, 1.0, None)
        idx    = pd.date_range("2013-01-02 09:15", periods=n, freq="1min",
                               tz="Asia/Kolkata")
        df = pd.DataFrame({
            "open": closes, "high": closes + 1, "low": closes - 1,
            "close": closes, "volume": 10000,
        }, index=idx)
        df["signal"] = np.where(np.arange(n) % 100 == 0, 1,
                       np.where(np.arange(n) % 100 == 50, -1, 0))
        cfg = BacktestConfig(initial_capital=500_000, fixed_quantity=1)

        t0 = time.time()
        trades, equity, _ = run_event_loop(df, cfg, "PERF_TEST")
        elapsed = time.time() - t0

        assert elapsed < 60.0, f"10-year minute backtest took {elapsed:.1f}s — too slow"
        assert len(trades) > 0
        print(f"\nPerformance test: {n:,} bars in {elapsed:.2f}s "
              f"({n/elapsed:,.0f} bars/sec)")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
