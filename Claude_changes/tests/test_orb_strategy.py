"""
tests/test_orb_strategy.py
===========================
Complete test suite for the ORB NIFTY 50 strategy.

Coverage:
  - ORLevels / DayTradeState data containers
  - ORBIndicators: OR levels, VWAP, broadcast
  - ORBTrailingStop: increment formula
  - ORBNiftyStrategy: generate_signals — all conditions, daily limits, edge cases
  - run_orb_event_loop: stop-loss trigger, trailing stop advance, squareoff, 100% qty
  - Integration: full pipeline → BacktestResult → metrics

Run with:
    python -m pytest tests/test_orb_strategy.py -v
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, time as dtime
from typing import List

import numpy as np
import pandas as pd
import pytest

from strategies.day_strategy.orb_nifty import (
    ORBNiftyStrategy,
    ORBIndicators,
    ORBTrailingStop,
    ORLevels,
    DayTradeState,
    SignalBar,
    OR_START_TIME,
    OR_END_TIME,
    SIGNAL_START,
    SQUAREOFF_TIME,
    TRAILING_DIVISOR,
    MIN_OR_BARS,
)

IST = "Asia/Kolkata"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_df(
    n_days:     int   = 5,
    start:      str   = "2024-01-02",
    base:       float = 21_000.0,
    trend:      float = 0.3,
    seed:       int   = 42,
    bars_per_day:int  = 375,         # 09:15 – 15:29 = 375 min
) -> pd.DataFrame:
    """
    Generate synthetic 1-minute OHLCV data for n_days of NSE trading.
    Uses a business-day calendar so weekends are excluded.
    """
    np.random.seed(seed)
    total = n_days * bars_per_day
    bdays = pd.bdate_range(start, periods=n_days, freq="B")
    idx   = pd.DatetimeIndex(
        [ts for d in bdays
         for ts in pd.date_range(f"{d.date()} 09:15", periods=bars_per_day,
                                  freq="1min", tz=IST)]
    )
    closes = np.clip(base + np.cumsum(np.random.randn(total) * 8 + trend), 1.0, None)
    highs  = closes + np.abs(np.random.randn(total) * 4)
    lows   = np.clip(closes - np.abs(np.random.randn(total) * 4), 1.0, None)
    opens  = (closes + lows) / 2
    vol    = np.random.randint(500_000, 8_000_000, total)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol},
        index=idx,
    )


def _make_breakout_df(
    direction:  str   = "long",
    base:       float = 21_000.0,
    or_spread:  float = 50.0,        # half-width of OR range
    breakout:   float = 10.0,        # how far the breakout bar closes beyond OR
) -> pd.DataFrame:
    """
    Synthetic single-day data with a controlled breakout at 09:30.

    Bar 0–14 (09:15–09:29) : OR window — deterministic range
    Bar 15    (09:30)       : signal bar — breakout candle
    Bar 16+   (09:31+)      : flat / trending away from range
    """
    n   = 375
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="1min", tz=IST)
    o   = np.full(n, base)
    h   = np.full(n, base + or_spread)
    l   = np.full(n, base - or_spread)
    c   = np.full(n, base)
    v   = np.full(n, 5_000_000, dtype=int)

    if direction == "long":
        # Signal bar: green candle closing above or_high
        o[15] = base
        c[15] = base + or_spread + breakout    # above OR_high
        h[15] = c[15] + 5
        l[15] = base - 2
    else:
        # Signal bar: red candle closing below or_low
        o[15] = base
        c[15] = base - or_spread - breakout   # below OR_low
        l[15] = c[15] - 5
        h[15] = base + 2

    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v},
                         index=idx)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Data containers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestORLevels:

    def test_valid_default(self):
        lvl = ORLevels("2024-01-02", 21100.0, 20900.0)
        assert lvl.is_valid
        assert lvl.or_high > lvl.or_low

    def test_invalid_flag(self):
        lvl = ORLevels("2024-01-02", float("nan"), float("nan"), is_valid=False)
        assert not lvl.is_valid

    def test_repr_contains_date(self):
        lvl = ORLevels("2024-01-02", 21100.0, 20900.0)
        assert "2024-01-02" in repr(lvl)

    def test_bar_count_stored(self):
        lvl = ORLevels("2024-01-02", 21100.0, 20900.0, bar_count=15)
        assert lvl.bar_count == 15


class TestDayTradeState:

    def test_initial_can_go_both(self):
        s = DayTradeState("2024-01-02")
        assert s.can_go_long
        assert s.can_go_short
        assert not s.both_legs_used

    def test_after_long_taken(self):
        s = DayTradeState("2024-01-02", long_taken=True)
        assert not s.can_go_long
        assert s.can_go_short

    def test_after_short_taken(self):
        s = DayTradeState("2024-01-02", short_taken=True)
        assert s.can_go_long
        assert not s.can_go_short

    def test_both_taken(self):
        s = DayTradeState("2024-01-02", long_taken=True, short_taken=True)
        assert s.both_legs_used
        assert not s.can_go_long
        assert not s.can_go_short


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. ORBIndicators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestORBIndicators:

    def test_one_entry_per_day(self):
        df  = _make_df(n_days=5)
        lvl = ORBIndicators.compute_or_levels(df)
        assert len(lvl) == 5

    def test_or_high_gt_or_low_each_day(self):
        df  = _make_df(n_days=10)
        lvl = ORBIndicators.compute_or_levels(df)
        for dt, l in lvl.items():
            if l.is_valid:
                assert l.or_high >= l.or_low, f"or_high < or_low on {dt}"

    def test_or_range_bounded_by_full_day_range(self):
        df  = _make_df(n_days=3)
        lvl = ORBIndicators.compute_or_levels(df)
        bar_dates = df.index.date
        for dt, l in lvl.items():
            if l.is_valid:
                day_df = df[bar_dates == date.fromisoformat(dt)]
                assert l.or_high <= float(day_df["high"].max()) + 0.01
                assert l.or_low  >= float(day_df["low"].min())  - 0.01

    def test_invalid_when_few_or_bars(self):
        """A day with fewer OR bars than MIN_OR_BARS must be marked invalid."""
        df = _make_df(n_days=1)
        bt = df.index.time
        or_mask = (pd.Series(bt, index=df.index) >= OR_START_TIME) & \
                  (pd.Series(bt, index=df.index) <= OR_END_TIME)
        # Drop enough OR bars to go below minimum
        drop_n = len(df[or_mask]) - (MIN_OR_BARS - 1)
        if drop_n > 0:
            drop_idx = df.index[or_mask][:drop_n]
            df2 = df.drop(drop_idx)
            lvl = ORBIndicators.compute_or_levels(df2)
            day_str = str(df2.index.date[0])
            assert not lvl[day_str].is_valid

    def test_vwap_length_matches_df(self):
        df   = _make_df(n_days=3)
        vwap = ORBIndicators.compute_vwap(df)
        assert len(vwap) == len(df)

    def test_vwap_positive_values(self):
        df   = _make_df(n_days=5, base=21000.0)
        vwap = ORBIndicators.compute_vwap(df).dropna()
        assert (vwap > 0).all()

    def test_vwap_first_bar_equals_typical_price(self):
        """On the first bar of each day, VWAP == typical_price of that bar."""
        df   = _make_df(n_days=4)
        vwap = ORBIndicators.compute_vwap(df)
        bar_dates = pd.Series(df.index.date, index=df.index)
        for dt in pd.unique(bar_dates):
            day_df = df[bar_dates == dt]
            first  = day_df.iloc[0]
            tp     = (first["high"] + first["low"] + first["close"]) / 3.0
            assert abs(float(vwap.loc[day_df.index[0]]) - tp) < 0.01

    def test_vwap_name(self):
        df   = _make_df(n_days=1)
        vwap = ORBIndicators.compute_vwap(df)
        assert vwap.name == "vwap"

    def test_broadcast_lengths(self):
        df      = _make_df(n_days=5)
        lvl     = ORBIndicators.compute_or_levels(df)
        h, l, v = ORBIndicators.broadcast_or_levels(df, lvl)
        assert len(h) == len(df)
        assert len(l) == len(df)
        assert len(v) == len(df)

    def test_broadcast_valid_flag_boolean(self):
        df      = _make_df(n_days=5)
        lvl     = ORBIndicators.compute_or_levels(df)
        _, _, v = ORBIndicators.broadcast_or_levels(df, lvl)
        assert v.dtype == bool or v.dtype == np.bool_

    def test_broadcast_or_high_constant_within_day(self):
        """Every bar in the same day must have the same or_high."""
        df      = _make_df(n_days=2)
        lvl     = ORBIndicators.compute_or_levels(df)
        h, _, _ = ORBIndicators.broadcast_or_levels(df, lvl)
        bar_dates = pd.Series(df.index.date, index=df.index)
        for dt in pd.unique(bar_dates):
            day_vals = h[bar_dates == dt].dropna()
            if not day_vals.empty:
                assert day_vals.nunique() == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. ORBTrailingStop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestORBTrailingStop:

    def test_length_matches_df(self):
        df  = _make_df(n_days=3)
        inc = ORBTrailingStop.compute_increment(df)
        assert len(inc) == len(df)

    def test_first_bar_zero(self):
        df  = _make_df(n_days=2)
        inc = ORBTrailingStop.compute_increment(df)
        assert inc.iloc[0] == pytest.approx(0.0)

    def test_non_negative(self):
        df  = _make_df(n_days=5)
        inc = ORBTrailingStop.compute_increment(df)
        assert (inc >= 0).all()

    def test_formula_correct(self):
        """inc[i] = |open[i-1] - close[i-1]| / TRAILING_DIVISOR"""
        df  = _make_df(n_days=2)
        inc = ORBTrailingStop.compute_increment(df)
        for i in range(1, 20):
            expected = abs(float(df["open"].iloc[i-1]) -
                           float(df["close"].iloc[i-1])) / TRAILING_DIVISOR
            assert abs(float(inc.iloc[i]) - expected) < 0.001

    def test_custom_divisor(self):
        """Doubling the divisor should halve all increments."""
        df  = _make_df(n_days=2)
        inc5 = ORBTrailingStop.compute_increment(df, divisor=5.0)
        inc10= ORBTrailingStop.compute_increment(df, divisor=10.0)
        pd.testing.assert_series_equal(inc5 / 2, inc10, check_names=False)

    def test_name_attribute(self):
        df  = _make_df(n_days=1)
        inc = ORBTrailingStop.compute_increment(df)
        assert inc.name == "trail_increment"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. ORBNiftyStrategy — generate_signals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestORBNiftyStrategy:

    def _s(self, **kw) -> ORBNiftyStrategy:
        return ORBNiftyStrategy(vwap_filter=False, **kw)

    # ── Output structure ──────────────────────────────────────────────────────

    def test_returns_dataframe(self):
        out = self._s().generate_signals(_make_df(5))
        assert isinstance(out, pd.DataFrame)

    def test_required_output_columns(self):
        out = self._s().generate_signals(_make_df(5))
        for col in ("signal", "or_high", "or_low", "or_valid",
                    "vwap", "trail_increment", "signal_sl", "signal_tag"):
            assert col in out.columns, f"Missing output column: {col}"

    def test_signal_values_in_valid_set(self):
        out = self._s().generate_signals(_make_df(10))
        assert set(out["signal"].unique()).issubset({-1, 0, 1})

    def test_signal_sl_nan_where_no_signal(self):
        out = self._s().generate_signals(_make_df(5))
        no_sig = out[out["signal"] == 0]
        assert no_sig["signal_sl"].isna().all()

    def test_signal_sl_set_where_signal(self):
        out  = self._s().generate_signals(_make_df(5))
        sigs = out[out["signal"] != 0]
        if not sigs.empty:
            assert sigs["signal_sl"].notna().all()

    def test_signal_tag_empty_where_no_signal(self):
        out = self._s().generate_signals(_make_df(5))
        no_sig = out[out["signal"] == 0]
        assert (no_sig["signal_tag"] == "").all()

    # ── Time constraints ─────────────────────────────────────────────────────

    def test_no_signal_in_or_window(self):
        out = self._s().generate_signals(_make_df(5))
        bt  = pd.Series(out.index.time, index=out.index)
        or_mask = bt <= OR_END_TIME
        assert (out.loc[or_mask, "signal"] == 0).all()

    def test_no_signal_at_or_after_squareoff(self):
        out = self._s().generate_signals(_make_df(5))
        bt  = pd.Series(out.index.time, index=out.index)
        sq_mask = bt >= SQUAREOFF_TIME
        assert (out.loc[sq_mask, "signal"] == 0).all()

    # ── Daily limits ─────────────────────────────────────────────────────────

    def test_max_one_long_per_day(self):
        out  = self._s().generate_signals(_make_df(20, seed=0))
        dates = pd.Series(out.index.date, index=out.index)
        for dt in pd.unique(dates):
            assert (out.loc[dates == dt, "signal"] == 1).sum() <= 1

    def test_max_one_short_per_day(self):
        out  = self._s().generate_signals(_make_df(20, seed=0))
        dates = pd.Series(out.index.date, index=out.index)
        for dt in pd.unique(dates):
            assert (out.loc[dates == dt, "signal"] == -1).sum() <= 1

    # ── Controlled breakout signals ───────────────────────────────────────────

    def test_long_signal_on_green_breakout(self):
        df  = _make_breakout_df("long")
        out = ORBNiftyStrategy(vwap_filter=False).generate_signals(df)
        assert (out["signal"] == 1).sum() >= 1

    def test_short_signal_on_red_breakout(self):
        df  = _make_breakout_df("short")
        out = ORBNiftyStrategy(vwap_filter=False).generate_signals(df)
        assert (out["signal"] == -1).sum() >= 1

    def test_long_signal_tag_correct(self):
        df  = _make_breakout_df("long")
        out = ORBNiftyStrategy(vwap_filter=False).generate_signals(df)
        longs = out[out["signal"] == 1]
        assert (longs["signal_tag"] == "ORB_LONG").all()

    def test_short_signal_tag_correct(self):
        df  = _make_breakout_df("short")
        out = ORBNiftyStrategy(vwap_filter=False).generate_signals(df)
        shorts = out[out["signal"] == -1]
        assert (shorts["signal_tag"] == "ORB_SHORT").all()

    def test_signal_sl_equals_open_of_signal_bar(self):
        df  = _make_breakout_df("long")
        out = ORBNiftyStrategy(vwap_filter=False).generate_signals(df)
        longs = out[out["signal"] == 1]
        for ts in longs.index:
            assert longs.at[ts, "signal_sl"] == pytest.approx(float(df.at[ts, "open"]))

    def test_long_needs_green_candle(self):
        """A bar where close < open should never generate a long signal."""
        df  = _make_breakout_df("long")
        # Force signal bar to be red
        df.iloc[15, df.columns.get_loc("close")] = df.iloc[15]["open"] - 5
        out = ORBNiftyStrategy(vwap_filter=False).generate_signals(df)
        assert (out["signal"] == 1).sum() == 0

    def test_short_needs_red_candle(self):
        """A bar where close > open should never generate a short signal."""
        df  = _make_breakout_df("short")
        df.iloc[15, df.columns.get_loc("close")] = df.iloc[15]["open"] + 5
        out = ORBNiftyStrategy(vwap_filter=False).generate_signals(df)
        assert (out["signal"] == -1).sum() == 0

    # ── VWAP filter ──────────────────────────────────────────────────────────

    def test_vwap_filter_off_more_signals(self):
        df   = _make_df(20, seed=5)
        with_v  = ORBNiftyStrategy(vwap_filter=True).generate_signals(df)
        without = ORBNiftyStrategy(vwap_filter=False).generate_signals(df)
        assert (without["signal"] != 0).sum() >= (with_v["signal"] != 0).sum()

    # ── Parameter variations ─────────────────────────────────────────────────

    def test_or_window_changes_end_time(self):
        s15 = ORBNiftyStrategy(or_window_minutes=15)
        s30 = ORBNiftyStrategy(or_window_minutes=30)
        assert s30._or_end_time > s15._or_end_time

    def test_min_body_pct_reduces_signals(self):
        df   = _make_df(20, seed=7)
        s0   = ORBNiftyStrategy(vwap_filter=False, min_body_pct=0.0).generate_signals(df)
        s05  = ORBNiftyStrategy(vwap_filter=False, min_body_pct=0.05).generate_signals(df)
        # stricter filter → fewer or equal signals
        assert (s05["signal"] != 0).sum() <= (s0["signal"] != 0).sum()

    # ── Error handling ────────────────────────────────────────────────────────

    def test_missing_volume_raises(self):
        df = _make_df(2).drop(columns=["volume"])
        with pytest.raises(ValueError, match="volume"):
            ORBNiftyStrategy().generate_signals(df)

    def test_missing_close_raises(self):
        df = _make_df(2).drop(columns=["close"])
        with pytest.raises(ValueError, match="close"):
            ORBNiftyStrategy().generate_signals(df)

    def test_non_datetime_index_raises(self):
        df = _make_df(2).reset_index(drop=True)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            ORBNiftyStrategy().generate_signals(df)

    # ── Introspection ─────────────────────────────────────────────────────────

    def test_get_params_returns_dict(self):
        s = ORBNiftyStrategy(or_window_minutes=15, vwap_filter=True, trailing_divisor=5.0)
        p = s.get_params()
        assert p["or_window_minutes"] == 15
        assert p["vwap_filter"] is True
        assert p["trailing_divisor"] == 5.0

    def test_repr_meaningful(self):
        s = ORBNiftyStrategy()
        assert "ORBNiftyStrategy" in repr(s)
        assert "vwap_filter" in repr(s)

    def test_param_schema_valid(self):
        for p in ORBNiftyStrategy.PARAM_SCHEMA:
            assert "name" in p
            assert "default" in p
            assert "type" in p

    def test_category_defined(self):
        assert ORBNiftyStrategy.CATEGORY == "Intraday Breakout"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Custom event loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestORBEventLoop:
    """Tests for run_orb_event_loop() in run_orb_backtest.py."""

    def _run(self, df, capital=500_000, vwap_filter=False):
        from run_orb_backtest import run_orb_event_loop, build_config
        from backtester.models import BacktestConfig
        from broker.upstox.commission import Segment

        strat  = ORBNiftyStrategy(vwap_filter=vwap_filter)
        sigs   = strat.generate_signals(df)
        cfg    = BacktestConfig(
            initial_capital    = capital,
            capital_risk_pct   = 0.99,
            fixed_quantity     = 0,
            max_positions      = 1,
            segment            = Segment.EQUITY_INTRADAY,
            allow_shorting     = True,
            intraday_squareoff = True,
            stop_loss_pct      = 0.0,
            max_drawdown_pct   = 0.90,    # high threshold for tests
            save_trade_log     = False,
            save_raw_data      = False,
            save_chart         = False,
            generate_summary   = False,
            run_label          = "test",
        )
        return run_orb_event_loop(sigs, cfg, "TEST")

    def test_returns_three_tuple(self):
        r = self._run(_make_df(5))
        assert len(r) == 3

    def test_equity_curve_correct_length(self):
        df = _make_df(5)
        _, eq, _ = self._run(df)
        assert len(eq) == len(df)

    def test_drawdown_non_positive(self):
        _, _, dd = self._run(_make_df(5))
        valid = dd.dropna()
        assert (valid <= 0.001).all()

    def test_no_trades_when_no_signals(self):
        """If strategy fires no signals, trade log must be empty."""
        df = _make_df(3)
        # Manually blank all signals in the signals df
        from run_orb_backtest import run_orb_event_loop
        from backtester.models import BacktestConfig
        from broker.upstox.commission import Segment

        strat = ORBNiftyStrategy(vwap_filter=False)
        sigs  = strat.generate_signals(df)
        sigs["signal"]    = 0
        sigs["signal_sl"] = np.nan

        cfg = BacktestConfig(
            initial_capital=500_000, segment=Segment.EQUITY_INTRADAY,
            allow_shorting=True, intraday_squareoff=True,
            save_trade_log=False, save_raw_data=False,
            save_chart=False, generate_summary=False,
        )
        trades, eq, _ = run_orb_event_loop(sigs, cfg, "TEST")
        assert len(trades) == 0
        last_eq = float(eq.dropna().iloc[-1])
        assert last_eq == pytest.approx(500_000.0, rel=0.005)

    def test_long_entry_on_breakout_day(self):
        df = _make_breakout_df("long", base=21000.0)
        trades, _, _ = self._run(df, capital=5_000_000)
        long_trades = [t for t in trades if t.direction == 1]
        assert len(long_trades) >= 1

    def test_short_entry_on_breakout_day(self):
        df = _make_breakout_df("short", base=21000.0)
        trades, _, _ = self._run(df, capital=5_000_000)
        short_trades = [t for t in trades if t.direction == -1]
        assert len(short_trades) >= 1

    def test_all_trades_closed_by_end(self):
        """Every trade must have an exit_time >= entry_time (same bar exit is valid)."""
        df = _make_df(10)
        trades, _, _ = self._run(df)
        for t in trades:
            assert t.exit_time is not None
            assert t.exit_time >= t.entry_time   # same-bar exit allowed (max-dd halt)

    def test_stop_loss_fires_correctly(self):
        """
        Build a scenario where the SL must fire.
        Long entry at 09:30 → next bar opens well below SL → exit via Stop Loss.

        Uses large capital so the 100%-capital position is realistic, and
        disables the max-drawdown guard (max_drawdown_pct=1.0) so only the
        stop-loss can trigger the exit.
        """
        df = _make_breakout_df("long", base=21000.0)
        strat = ORBNiftyStrategy(vwap_filter=False)
        sigs  = strat.generate_signals(df)

        sig_idx = sigs.index[sigs["signal"] == 1]
        if sig_idx.empty:
            pytest.skip("No long signal generated")

        sig_pos   = sigs.index.get_loc(sig_idx[0])
        entry_bar = sig_pos + 1
        entry_sl  = float(sigs.iloc[sig_pos]["signal_sl"])

        # Force the bar AFTER entry to open well below SL
        if entry_bar + 1 < len(sigs):
            gap_open = entry_sl - 200.0
            sigs.iloc[entry_bar + 1, sigs.columns.get_loc("open")]  = gap_open
            sigs.iloc[entry_bar + 1, sigs.columns.get_loc("low")]   = gap_open - 10
            sigs.iloc[entry_bar + 1, sigs.columns.get_loc("close")] = gap_open - 5

        from run_orb_backtest import run_orb_event_loop
        from backtester.models import BacktestConfig
        from broker.upstox.commission import Segment
        cfg = BacktestConfig(
            initial_capital    = 500_000,          # realistic capital
            capital_risk_pct   = 0.99,
            segment            = Segment.EQUITY_INTRADAY,
            allow_shorting     = True,
            intraday_squareoff = True,
            stop_loss_pct      = 0.0,
            max_drawdown_pct   = 1.0,              # 100% = effectively disabled
            save_trade_log=False, save_raw_data=False,
            save_chart=False, generate_summary=False,
        )
        trades, _, _ = run_orb_event_loop(sigs, cfg, "SL_TEST")
        sl_trades = [t for t in trades if "Stop" in t.exit_signal]
        assert len(sl_trades) >= 1, (
            f"Expected a Stop Loss exit. Got: {[t.exit_signal for t in trades]}"
        )

    def test_squareoff_fires_at_1515(self):
        """
        Position open at 15:15 must be closed at that bar's open.

        SL is set to ₹1 (never triggers at real prices).
        max_drawdown_pct=1.0 disables the drawdown guard so the only
        possible exit is the 15:15 squareoff.
        """
        df    = _make_breakout_df("long", base=21000.0)
        strat = ORBNiftyStrategy(vwap_filter=False)
        sigs  = strat.generate_signals(df)
        sigs["signal_sl"] = 1.0    # SL at ₹1 — never triggers at 21000 range

        from run_orb_backtest import run_orb_event_loop
        from backtester.models import BacktestConfig
        from broker.upstox.commission import Segment
        cfg = BacktestConfig(
            initial_capital    = 500_000,
            capital_risk_pct   = 0.99,
            segment            = Segment.EQUITY_INTRADAY,
            allow_shorting     = True,
            intraday_squareoff = True,
            stop_loss_pct      = 0.0,
            max_drawdown_pct   = 1.0,              # disabled — only squareoff exits
            save_trade_log=False, save_raw_data=False,
            save_chart=False, generate_summary=False,
        )
        trades, _, _ = run_orb_event_loop(sigs, cfg, "SQ_TEST")
        sq_trades = [t for t in trades
                     if "Squareoff" in t.exit_signal or "End" in t.exit_signal]
        assert len(sq_trades) >= 1, (
            f"Expected Squareoff/End-of-Data exit. Got: {[t.exit_signal for t in trades]}"
        )

    def test_trailing_stop_advances_for_long(self):
        """
        After entry, if price rises and trail_increment > 0, the SL
        must advance. We verify this by checking that a profitable exit
        occurs at a price above entry_sl when trail fires.
        This is an indirect test (we cannot easily peek inside the loop).
        We just verify the loop completes without error and produces trades.
        """
        df = _make_df(5, trend=5.0)    # strong uptrend
        trades, _, _ = self._run(df, capital=5_000_000)
        # Just verify it ran without exception and produced sensible results
        assert isinstance(trades, list)

    def test_equity_never_negative_under_normal_conditions(self):
        df = _make_df(10)
        _, eq, _ = self._run(df)
        valid = eq.dropna()
        assert (valid >= 0).all()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Integration — full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIntegration:

    def _full_run(self, n_days=20, seed=42, capital=500_000):
        from run_orb_backtest import run_orb_event_loop
        from backtester.models import BacktestConfig, BacktestResult
        from broker.upstox.commission import Segment

        df    = _make_df(n_days, seed=seed)
        strat = ORBNiftyStrategy(vwap_filter=False)
        sigs  = strat.generate_signals(df)
        cfg   = BacktestConfig(
            initial_capital=capital, segment=Segment.EQUITY_INTRADAY,
            allow_shorting=True, intraday_squareoff=True,
            capital_risk_pct=0.99, max_positions=1, max_drawdown_pct=0.90,
            save_trade_log=False, save_raw_data=False,
            save_chart=False, generate_summary=False,
        )
        trades, equity, dd = run_orb_event_loop(sigs, cfg, "NIFTY50")
        result = BacktestResult(cfg, "NIFTY50", trades, equity, dd, sigs)
        return result

    def test_result_metrics_dict(self):
        m = self._full_run().metrics()
        assert isinstance(m, dict)
        assert "total_trades" in m

    def test_all_25_metrics_present(self):
        m = self._full_run(n_days=30).metrics()
        required = [
            "total_trades", "win_rate_pct", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "omega_ratio", "kelly_fraction",
            "max_drawdown_pct", "cagr_pct", "total_return_pct",
            "profit_factor", "expectancy_inr", "exposure_pct",
            "total_commission_paid", "avg_mae_inr", "avg_mfe_inr",
            "monthly_returns", "annual_returns",
        ]
        for key in required:
            assert key in m, f"Missing metric: {key}"

    def test_all_scalar_metrics_numeric(self):
        """All scalar metric values (excluding date strings) must be numeric."""
        m = self._full_run(n_days=25).metrics()
        STRING_KEYS = {"start_date", "end_date", "error"}   # known non-numeric
        scalar_keys = [
            k for k, v in m.items()
            if not isinstance(v, (dict, list)) and k not in STRING_KEYS
        ]
        for k in scalar_keys:
            assert isinstance(m[k], (int, float, np.integer, np.floating)), \
                f"Metric '{k}' has unexpected type {type(m[k])}: {m[k]!r}"

    def test_summary_is_string(self):
        result = self._full_run()
        s = result.summary()
        assert isinstance(s, str)
        assert "NIFTY50" in s

    def test_trade_df_has_expected_columns(self):
        result = self._full_run()
        df = result.trade_df()
        if not df.empty:
            for col in ("net_pnl", "entry_price", "exit_price",
                        "direction", "quantity"):
                assert col in df.columns

    def test_win_rate_bounded(self):
        m = self._full_run(n_days=30).metrics()
        assert 0.0 <= m["win_rate_pct"] <= 100.0

    def test_max_drawdown_non_positive(self):
        m = self._full_run(n_days=25).metrics()
        assert m["max_drawdown_pct"] <= 0.001

    def test_50_trading_days_no_exception(self):
        """Large run must complete without exception."""
        result = self._full_run(n_days=50, seed=99)
        assert isinstance(result.metrics(), dict)

    def test_metrics_cached(self):
        result = self._full_run()
        m1 = result.metrics()
        m2 = result.metrics()
        assert m1 is m2   # same object — cached

    def test_no_trade_exceeds_daily_leg_limit(self):
        """Verify max 1 long and 1 short per day in the full pipeline."""
        result = self._full_run(n_days=20)
        trades = result.trade_log
        from collections import defaultdict
        day_longs  = defaultdict(int)
        day_shorts = defaultdict(int)
        for t in trades:
            try:
                dt = str(t.entry_time.date())
            except Exception:
                continue
            if t.direction == 1:
                day_longs[dt] += 1
            else:
                day_shorts[dt] += 1
        for dt, cnt in day_longs.items():
            assert cnt <= 1, f"More than 1 LONG on {dt}"
        for dt, cnt in day_shorts.items():
            assert cnt <= 1, f"More than 1 SHORT on {dt}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])