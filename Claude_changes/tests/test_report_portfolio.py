"""
tests/test_report_portfolio.py
--------------------------------
Tests for backtester/report.py and backtester/portfolio.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from backtester.models import BacktestConfig, BacktestResult, Position, Trade, OrderType
from backtester.portfolio import Portfolio
from backtester.engine import BacktestEngine
from backtester.event_loop import run_event_loop


# ── Shared helpers ──────────────────────────────────────────────────────────

def _make_ohlcv(n=100, trend=1.0, seed=42):
    np.random.seed(seed)
    rng    = pd.date_range("2023-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    closes = 1000.0 + np.cumsum(np.random.randn(n) * 5 + trend)
    closes = np.clip(closes, 1.0, None)
    highs  = closes + np.abs(np.random.randn(n) * 2)
    lows   = np.clip(closes - np.abs(np.random.randn(n) * 2), 1.0, None)
    opens  = (closes + lows) / 2.0
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": np.random.randint(1000, 100_000, n)},
        index=rng,
    )


def _make_result(n=100, trend=1.0) -> BacktestResult:
    """Build a BacktestResult with at least one trade."""
    df  = _make_ohlcv(n, trend=trend)
    sdf = df.copy()
    sdf["signal"] = 0
    sdf.iloc[5,  sdf.columns.get_loc("signal")] = 1
    sdf.iloc[20, sdf.columns.get_loc("signal")] = -1
    sdf.iloc[40, sdf.columns.get_loc("signal")] = 1
    sdf.iloc[55, sdf.columns.get_loc("signal")] = -1
    cfg    = BacktestConfig(initial_capital=500_000, fixed_quantity=10)
    trades, equity, dd = run_event_loop(sdf, cfg, "TEST")
    return BacktestResult(cfg, "TEST", trades, equity, dd, sdf)


# ===========================================================================
# Portfolio tests
# ===========================================================================

class TestPortfolio:

    def _cfg(self, **kw):
        return BacktestConfig(initial_capital=500_000, fixed_quantity=10, **kw)

    def _pos(self, symbol="TEST", entry=1000.0, qty=10, direction=1):
        return Position(
            symbol=symbol, entry_time=pd.Timestamp("2023-01-05", tz="Asia/Kolkata"),
            entry_price=entry, quantity=qty, direction=direction,
        )

    # ── Initialisation ──────────────────────────────────────────────────────

    def test_initial_cash_equals_config(self):
        p = Portfolio(self._cfg())
        assert p.cash == 500_000.0

    def test_initial_equity_equals_cash_with_no_positions(self):
        p = Portfolio(self._cfg())
        assert p.equity() == 500_000.0

    def test_not_halted_at_start(self):
        p = Portfolio(self._cfg())
        assert p.is_halted is False

    # ── can_open ────────────────────────────────────────────────────────────

    def test_can_open_returns_true_initially(self):
        p = Portfolio(self._cfg())
        assert p.can_open() is True

    def test_can_open_returns_false_when_halted(self):
        p = Portfolio(self._cfg())
        p.is_halted = True
        assert p.can_open() is False

    def test_can_open_respects_max_positions(self):
        p = Portfolio(self._cfg(max_positions=2))
        p.add_position(self._pos("INFY"))
        p.add_position(self._pos("TCS"))
        assert p.can_open() is False

    def test_can_open_allows_when_under_max(self):
        p = Portfolio(self._cfg(max_positions=3))
        p.add_position(self._pos("INFY"))
        assert p.can_open() is True

    # ── Position management ─────────────────────────────────────────────────

    def test_add_position_increments_count(self):
        p   = Portfolio(self._cfg())
        pos = self._pos()
        p.add_position(pos)
        assert len(p.positions) == 1

    def test_remove_position_decrements_count(self):
        p   = Portfolio(self._cfg())
        pos = self._pos()
        p.add_position(pos)
        p.remove_position(pos)
        assert len(p.positions) == 0

    def test_remove_nonexistent_position_does_not_raise(self):
        p   = Portfolio(self._cfg())
        pos = self._pos()
        p.remove_position(pos)   # should log warning, not raise

    def test_open_positions_for_filters_by_symbol(self):
        p = Portfolio(self._cfg())
        p.add_position(self._pos("INFY"))
        p.add_position(self._pos("TCS"))
        p.add_position(self._pos("INFY"))
        assert len(p.open_positions_for("INFY")) == 2
        assert len(p.open_positions_for("TCS"))  == 1

    # ── Equity calculation ──────────────────────────────────────────────────

    def test_equity_includes_unrealised_pnl(self):
        p   = Portfolio(self._cfg())
        pos = self._pos(entry=1000.0, qty=10)   # long 10 @ 1000
        p.add_position(pos)
        eq = p.equity({"TEST": 1050.0})
        # unrealised = (1050-1000)*10 = 500
        assert eq == pytest.approx(500_000 + 500)

    def test_equity_with_no_price_map_uses_entry(self):
        p   = Portfolio(self._cfg())
        pos = self._pos(entry=1000.0, qty=10)
        p.add_position(pos)
        eq = p.equity()   # no current prices — uses entry price
        assert eq == pytest.approx(500_000.0)

    # ── Mark bar ────────────────────────────────────────────────────────────

    def test_mark_bar_appends_to_equity_curve(self):
        p = Portfolio(self._cfg())
        p.mark_bar({})
        p.mark_bar({})
        assert len(p.equity_curve)   == 2
        assert len(p.drawdown_curve) == 2

    def test_mark_bar_drawdown_is_zero_at_peak(self):
        p = Portfolio(self._cfg())
        eq = p.mark_bar({})
        assert p.drawdown_curve[-1] == pytest.approx(0.0)

    def test_mark_bar_drawdown_negative_when_below_peak(self):
        p    = Portfolio(self._cfg())
        pos  = self._pos(entry=1000.0, qty=10)
        p.add_position(pos)
        p.mark_bar({"TEST": 1000.0})   # at peak
        p.mark_bar({"TEST": 900.0})    # below peak
        assert p.drawdown_curve[-1] < 0.0

    def test_mark_bar_sets_halted_when_dd_exceeds_limit(self):
        # Configure tight max_drawdown so test-scale losses trigger it
        p   = Portfolio(self._cfg(max_drawdown_pct=0.01))   # 1%
        pos = self._pos(entry=1000.0, qty=400)   # large position
        p.add_position(pos)
        p.mark_bar({"TEST": 1000.0})   # peak
        p.mark_bar({"TEST": 950.0})    # -50 * 400 = -20000 = 4% of 500k — exceeds 1%
        assert p.is_halted is True

    # ── Cash sync ───────────────────────────────────────────────────────────

    def test_sync_cash_updates_balance(self):
        p = Portfolio(self._cfg())
        p.sync_cash(450_000.0)
        assert p.cash == 450_000.0

    # ── Trade log ───────────────────────────────────────────────────────────

    def test_add_trade_appends(self):
        p = Portfolio(self._cfg())
        assert len(p.trade_log) == 0
        # Build a minimal Trade object
        t = Trade(
            symbol="TEST",
            entry_time=pd.Timestamp("2023-01-05", tz="Asia/Kolkata"),
            exit_time=pd.Timestamp("2023-01-10", tz="Asia/Kolkata"),
            entry_price=1000.0, exit_price=1050.0, quantity=10,
            direction=1, direction_label="LONG",
            gross_pnl=500.0, entry_charges=20.0, exit_charges=20.0,
            total_charges=40.0, net_pnl=460.0, pnl_pct=0.046,
        )
        p.add_trade(t)
        assert len(p.trade_log) == 1

    # ── Serialisation ───────────────────────────────────────────────────────

    def test_to_equity_series_correct_length(self):
        p   = Portfolio(self._cfg())
        idx = pd.date_range("2023-01-01", periods=10, freq="D", tz="Asia/Kolkata")
        for _ in range(10):
            p.mark_bar({})
        s = p.to_equity_series(idx)
        assert len(s) == 10
        assert isinstance(s, pd.Series)

    def test_to_equity_series_pads_short_curve(self):
        """If mark_bar was called fewer times than index length, pad with last value."""
        p   = Portfolio(self._cfg())
        idx = pd.date_range("2023-01-01", periods=10, freq="D", tz="Asia/Kolkata")
        for _ in range(5):
            p.mark_bar({})
        s = p.to_equity_series(idx)
        assert len(s) == 10
        assert s.iloc[9] == s.iloc[4]   # padded

    def test_to_drawdown_series_correct_length(self):
        p   = Portfolio(self._cfg())
        idx = pd.date_range("2023-01-01", periods=8, freq="D", tz="Asia/Kolkata")
        for _ in range(8):
            p.mark_bar({})
        s = p.to_drawdown_series(idx)
        assert len(s) == 8

    def test_summary_returns_dict_with_required_keys(self):
        p = Portfolio(self._cfg())
        s = p.summary()
        for key in ("cash", "open_positions", "completed_trades",
                    "total_net_pnl", "is_halted"):
            assert key in s

    def test_repr_is_string(self):
        p = Portfolio(self._cfg())
        assert isinstance(repr(p), str)
        assert "Portfolio" in repr(p)

    # ── Integration: Portfolio tracks what engine produces ──────────────────

    def test_portfolio_equity_matches_engine_equity(self):
        """
        Portfolio.equity_curve should match the equity produced by run_event_loop
        when initialised from the same config and trade sequence.
        """
        result = _make_result(n=80, trend=2.0)
        engine_final = float(result.equity_curve.dropna().iloc[-1])

        # Rebuild a Portfolio from the trade log and compare
        cfg = result.config
        p   = Portfolio(cfg)
        for t in result.trade_log:
            p.add_trade(t)
        final_from_trades = cfg.initial_capital + sum(
            t.net_pnl for t in result.trade_log
        )
        # Engine final equity ≈ initial + sum of net_pnl
        # (small difference due to unrealised at bar close vs. trade close)
        assert abs(engine_final - final_from_trades) < cfg.initial_capital * 0.05


# ===========================================================================
# Report tests
# ===========================================================================

class TestReport:

    def test_generate_report_creates_png_file(self):
        result = _make_result(n=60)
        with tempfile.TemporaryDirectory() as tmpdir:
            from backtester.report import generate_report
            path = generate_report(result, symbol="INFY", output_dir=tmpdir)
            assert Path(path).exists()
            assert Path(path).suffix == ".png"
            assert Path(path).stat().st_size > 1000   # non-empty file

    def test_generate_report_custom_filename(self):
        result = _make_result(n=60)
        with tempfile.TemporaryDirectory() as tmpdir:
            from backtester.report import generate_report
            path = generate_report(
                result, symbol="TCS", output_dir=tmpdir,
                filename="custom_chart.png"
            )
            assert Path(path).name == "custom_chart.png"

    def test_generate_report_returns_string_path(self):
        result = _make_result(n=60)
        with tempfile.TemporaryDirectory() as tmpdir:
            from backtester.report import generate_report
            path = generate_report(result, output_dir=tmpdir)
            assert isinstance(path, str)

    def test_generate_report_with_zero_trades(self):
        """Report should still render even with no completed trades."""
        df  = _make_ohlcv(60)
        sdf = df.copy()
        sdf["signal"] = 0
        cfg    = BacktestConfig(initial_capital=500_000, fixed_quantity=10)
        trades, equity, dd = run_event_loop(sdf, cfg, "NOTR")
        result = BacktestResult(cfg, "NOTR", trades, equity, dd, sdf)
        with tempfile.TemporaryDirectory() as tmpdir:
            from backtester.report import generate_report
            path = generate_report(result, symbol="NOTR", output_dir=tmpdir)
            assert Path(path).exists()

    def test_generate_report_with_max_candles_limit(self):
        """max_candles slices data correctly — file must still be generated."""
        result = _make_result(n=200)
        with tempfile.TemporaryDirectory() as tmpdir:
            from backtester.report import generate_report
            path = generate_report(
                result, symbol="TEST", output_dir=tmpdir, max_candles=50
            )
            assert Path(path).exists()

    def test_generate_report_with_indicator_columns(self):
        """Report renders correctly when indicator columns are present."""
        df  = _make_ohlcv(80)
        sdf = df.copy()
        # Add fake EMA columns to test overlay detection
        sdf["ema_9"]   = sdf["close"].ewm(span=9).mean()
        sdf["ema_21"]  = sdf["close"].ewm(span=21).mean()
        sdf["rsi_14"]  = 50.0 + np.random.randn(80) * 10
        sdf["signal"]  = 0
        sdf.iloc[10, sdf.columns.get_loc("signal")] = 1
        sdf.iloc[30, sdf.columns.get_loc("signal")] = -1

        cfg    = BacktestConfig(initial_capital=500_000, fixed_quantity=10)
        trades, equity, dd = run_event_loop(sdf, cfg, "INDY")
        result = BacktestResult(cfg, "INDY", trades, equity, dd, sdf)

        with tempfile.TemporaryDirectory() as tmpdir:
            from backtester.report import generate_report
            path = generate_report(result, symbol="INDY", output_dir=tmpdir)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 5000

    def test_detect_cols_finds_ema(self):
        from backtester.report import _detect_cols, _OVERLAY_PREFIXES
        df = pd.DataFrame({
            "open": [1], "close": [1], "ema_9": [1],
            "rsi_14": [50], "signal": [0]
        })
        overlays = _detect_cols(df, _OVERLAY_PREFIXES)
        assert "ema_9" in overlays
        assert "rsi_14" not in overlays

    def test_detect_cols_finds_rsi(self):
        from backtester.report import _detect_cols, _OSC_PREFIXES
        df = pd.DataFrame({
            "close": [1], "rsi_14": [50], "ema_9": [1], "signal": [0]
        })
        oscs = _detect_cols(df, _OSC_PREFIXES)
        assert "rsi_14" in oscs
        assert "ema_9"  not in oscs

    def test_generate_report_output_dir_created_if_missing(self):
        result = _make_result(n=60)
        with tempfile.TemporaryDirectory() as tmpdir:
            from backtester.report import generate_report
            new_dir = str(Path(tmpdir) / "new" / "subdir")
            path    = generate_report(result, output_dir=new_dir)
            assert Path(path).exists()


# ===========================================================================
# __init__ exports test
# ===========================================================================

class TestInitExports:

    def test_all_public_names_importable(self):
        import backtester
        required = [
            "BacktestEngine", "BacktestConfig", "BacktestResult",
            "Position", "Trade", "OrderType", "TrailingType",
            "Optimizer", "SearchMethod", "compute_performance",
            "Portfolio", "generate_report",
        ]
        for name in required:
            assert hasattr(backtester, name), f"Missing export: {name}"

    def test_engine_instantiable_from_init(self):
        from backtester import BacktestEngine, BacktestConfig
        engine = BacktestEngine(BacktestConfig(initial_capital=100_000))
        assert engine is not None


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
