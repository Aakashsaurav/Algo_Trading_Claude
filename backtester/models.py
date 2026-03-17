"""
backtester/models.py
---------------------
Single source of truth for every data structure used across the backtesting engine.

DESIGN RATIONALE
================
Previously, ``Trade``, ``Position``, ``BacktestConfig``, and ``BacktestResult`` were
scattered across ``engine.py``, ``engine_v2.py``, ``engine_old.py``, and
``portfolio.py`` — creating four partially-overlapping definitions with subtle
field differences. This module consolidates them into one canonical set.

No logic lives here. These are pure data containers. The engine, fill logic,
portfolio tracker, and performance module all import from here — ensuring that
a field rename requires only one edit in one file.

CONTENTS
========
  BacktestConfig    — all engine parameters in one place
  Trade             — one completed round-trip (entry + exit)
  Position          — one open position currently held
  BacktestResult    — output container from engine.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import time as dtime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from broker.upstox.commission import CommissionModel, Segment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INTRADAY_SQUAREOFF = dtime(15, 20)
_HERE = Path(__file__).resolve().parent.parent   # project root


# ---------------------------------------------------------------------------
# OrderType enum (kept in models so everything can import from one place)
# ---------------------------------------------------------------------------

class OrderType(Enum):
    """
    Supported order types for backtest simulation.

    ``MARKET``
        Execute at the open of the next bar after the signal bar.
        This is the default and the most conservative (no fill-price optimism).

    ``LIMIT``
        Place a limit order below (long) or above (short) the signal close
        by ``limit_offset_pct`` percent.  Fills only if the bar's low
        (long) or high (short) reaches the limit price.

    ``STOP``
        Enter once price breaches a stop level.  Models momentum breakout
        entries.

    ``STOP_LIMIT``
        Trigger at stop price, fill only if limit price is still reachable.
        Prevents catastrophic fills in fast markets.

    ``TRAILING_STOP``
        Dynamic stop-loss that follows price in the favourable direction,
        never retreating against the trade.
    """
    MARKET        = "MARKET"
    LIMIT         = "LIMIT"
    STOP          = "STOP"
    STOP_LIMIT    = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class TrailingType(Enum):
    """Whether trailing stop distance is measured in % or fixed ₹ amount."""
    PERCENT = "PERCENT"
    AMOUNT  = "AMOUNT"


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Complete configuration for one backtest run.

    All parameters have sensible defaults so a minimal config is::

        cfg = BacktestConfig(initial_capital=500_000)

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value in ₹.  Default 500 000.
    capital_risk_pct : float
        Maximum fraction of capital to risk on a single trade (for
        ATR-based sizing).  Default 0.02 (2 %).
    fixed_quantity : int
        If > 0, use this exact share count for every trade and ignore
        ``capital_risk_pct``.  0 = use dynamic risk-based sizing.
    max_positions : int
        Maximum simultaneous open positions (0 = unlimited).
    max_drawdown_pct : float
        Halt the backtest if equity falls more than this fraction below
        its peak.  Default 0.20 (20 %).
    segment : Segment
        Market segment for charge calculation (delivery, intraday, futures …).
    allow_shorting : bool
        If False, ``signal == -1`` closes an existing long but does NOT
        open a short.  Default False.
    intraday_squareoff : bool
        Force-close all positions at 15:20 IST on each session.
        Requires a timezone-aware DatetimeIndex.  Default False.
    lot_size : int
        Lot size for F&O contracts.  Ignored for equities.  Default 1.
    stop_loss_atr_mult : float
        Attach an ATR-based fixed stop-loss.  Stop = entry ± mult × ATR(14).
        0 = disabled.  Default 2.0.
    default_order_type : OrderType
        Entry order type.  Default MARKET.
    limit_offset_pct : float
        For LIMIT entries: % offset from signal-bar close.  Default 0.2.
    stop_loss_pct : float
        Fixed % stop-loss on all trades (overrides ATR when > 0).  Default 0.
    use_trailing_stop : bool
        Attach a trailing stop to every opened position.  Default False.
    trailing_stop_pct : float
        Trailing stop distance in % of price (use with use_trailing_stop).
    trailing_stop_amt : float
        Trailing stop distance in fixed ₹ (alternative to trailing_stop_pct).
    save_trade_log : bool
        Write a trade-log CSV to ``strategies/output/trade/``.
    save_raw_data : bool
        Write OHLCV + indicator + signal CSV to ``strategies/output/raw_data/``.
    save_chart : bool
        Write a Streak-style PNG chart to ``strategies/output/chart/``.
    generate_summary : bool
        Write a performance summary JSON/CSV after the run.
    run_label : str
        Prefix for all output filenames.  Default ``"backtest"``.
    max_candles : int
        Maximum candles rendered in the PNG chart.  Default 2000.
    commission_model : CommissionModel
        Commission calculator instance.  Defaults to a fresh CommissionModel().
    """
    # ── Capital & sizing ────────────────────────────────────────────────────
    initial_capital:    float           = 500_000.0
    capital_risk_pct:   float           = 0.02
    fixed_quantity:     int             = 0
    max_positions:      int             = 0
    max_drawdown_pct:   float           = 0.20
    # ── Market parameters ───────────────────────────────────────────────────
    segment:            Segment         = Segment.EQUITY_DELIVERY
    allow_shorting:     bool            = False
    intraday_squareoff: bool            = False
    lot_size:           int             = 1
    # ── Stop / size helpers ─────────────────────────────────────────────────
    stop_loss_atr_mult: float           = 2.0
    # ── Order type settings ─────────────────────────────────────────────────
    default_order_type:  OrderType      = OrderType.MARKET
    limit_offset_pct:    float          = 0.2
    stop_loss_pct:       float          = 0.0
    use_trailing_stop:   bool           = False
    trailing_stop_pct:   float          = 0.0
    trailing_stop_amt:   float          = 0.0
    # ── Output flags ────────────────────────────────────────────────────────
    save_trade_log:      bool           = False
    save_raw_data:       bool           = False
    save_chart:          bool           = False
    generate_summary:    bool           = False
    run_label:           str            = "backtest"
    max_candles:         int            = 2000
    # ── Commission ──────────────────────────────────────────────────────────
    commission_model:   CommissionModel = field(default_factory=CommissionModel)

    def validate(self) -> None:
        """Raise ValueError for obviously wrong configurations."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not (0.0 < self.capital_risk_pct <= 1.0):
            raise ValueError("capital_risk_pct must be in (0, 1]")
        if self.fixed_quantity < 0:
            raise ValueError("fixed_quantity must be >= 0")
        if self.max_drawdown_pct <= 0 or self.max_drawdown_pct > 1:
            raise ValueError("max_drawdown_pct must be in (0, 1]")
        if self.use_trailing_stop:
            if self.trailing_stop_pct == 0 and self.trailing_stop_amt == 0:
                raise ValueError(
                    "use_trailing_stop=True requires trailing_stop_pct or trailing_stop_amt > 0"
                )
            if self.trailing_stop_pct > 0 and self.trailing_stop_amt > 0:
                raise ValueError(
                    "Provide trailing_stop_pct OR trailing_stop_amt, not both"
                )


# ---------------------------------------------------------------------------
# Position  (one open trade currently held)
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    Represents a single open position.

    Created by :class:`backtester.fill_engine.FillEngine` when an entry order
    fills, and destroyed when the corresponding exit fills.

    Attributes
    ----------
    symbol : str
        Instrument symbol.
    entry_time : pd.Timestamp
        Bar timestamp at which the position was opened.
    entry_price : float
        Actual fill price (next-bar open after signal).
    quantity : int
        Number of shares / units held.
    direction : int
        +1 for LONG, -1 for SHORT.
    entry_signal : str
        Human-readable label for the entry reason (e.g. ``"Market Signal"``).
    entry_charges : float
        Total brokerage + taxes paid on entry.
    entry_bar_idx : int
        Integer index of the entry bar in the signal DataFrame.
    mae : float
        Maximum Adverse Excursion — worst unrealised loss since entry.
        Updated every bar by :meth:`update_excursion`.
    mfe : float
        Maximum Favourable Excursion — best unrealised profit since entry.
    stop_price : float or None
        Fixed stop-loss price (None if not configured).
    trailing_stop_pct : float
        Trailing stop % (0 = disabled).
    trailing_stop_amt : float
        Trailing stop ₹ amount (0 = disabled).
    trailing_stop_level : float
        Current trailing stop price (updated each bar).
    order_type : OrderType
        How the position was entered.
    """
    symbol:              str
    entry_time:          pd.Timestamp
    entry_price:         float
    quantity:            int
    direction:           int                 # +1 LONG, -1 SHORT
    entry_signal:        str   = ""
    entry_charges:       float = 0.0
    entry_bar_idx:       int   = 0
    mae:                 float = 0.0
    mfe:                 float = 0.0
    stop_price:          Optional[float] = None
    trailing_stop_pct:   float = 0.0
    trailing_stop_amt:   float = 0.0
    trailing_stop_level: float = 0.0
    order_type:          OrderType = OrderType.MARKET

    # ------------------------------------------------------------------
    def unrealised_pnl(self, current_price: float) -> float:
        """Mark-to-market unrealised profit/loss at ``current_price``."""
        return (current_price - self.entry_price) * self.direction * self.quantity

    def update_excursion(self, price: float) -> None:
        """Update MAE / MFE with the latest price."""
        move = (price - self.entry_price) * self.direction
        self.mfe = max(self.mfe, move)
        self.mae = min(self.mae, move)

    def update_trailing_stop(self, high: float, low: float) -> None:
        """
        Advance the trailing stop level if price moved in our favour.

        The stop NEVER moves against the trade.
        """
        if self.trailing_stop_pct > 0:
            dist = self.entry_price * (self.trailing_stop_pct / 100.0)
        elif self.trailing_stop_amt > 0:
            dist = self.trailing_stop_amt
        else:
            return   # no trailing stop configured

        if self.direction == 1:   # LONG — trail behind high
            ideal = high - dist
            if self.trailing_stop_level == 0.0:
                self.trailing_stop_level = self.entry_price - dist
            self.trailing_stop_level = max(self.trailing_stop_level, ideal)
        else:                     # SHORT — trail above low
            ideal = low + dist
            if self.trailing_stop_level == 0.0:
                self.trailing_stop_level = self.entry_price + dist
            self.trailing_stop_level = min(self.trailing_stop_level, ideal)

    def is_trailing_stop_triggered(self, open_p: float, low: float, high: float):
        """
        Returns ``(triggered: bool, fill_price: float)``.

        Fills at ``open_p`` when price gaps through the stop, otherwise at
        the stop level.
        """
        if self.trailing_stop_level == 0.0:
            return False, 0.0
        if self.direction == 1:   # LONG exit if low < stop
            if open_p <= self.trailing_stop_level:
                return True, open_p    # gap-through — fill at open
            if low <= self.trailing_stop_level:
                return True, self.trailing_stop_level
        else:                     # SHORT exit if high > stop
            if open_p >= self.trailing_stop_level:
                return True, open_p
            if high >= self.trailing_stop_level:
                return True, self.trailing_stop_level
        return False, 0.0

    def is_fixed_stop_triggered(self, open_p: float, low: float, high: float):
        """Returns ``(triggered: bool, fill_price: float)`` for fixed stop."""
        if self.stop_price is None:
            return False, 0.0
        if self.direction == 1:
            if open_p <= self.stop_price:
                return True, open_p
            if low <= self.stop_price:
                return True, self.stop_price
        else:
            if open_p >= self.stop_price:
                return True, open_p
            if high >= self.stop_price:
                return True, self.stop_price
        return False, 0.0

    @property
    def direction_label(self) -> str:
        return "LONG" if self.direction == 1 else "SHORT"


# ---------------------------------------------------------------------------
# Trade  (one completed round-trip)
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """
    Represents one completed round-trip trade.

    Created by :class:`backtester.fill_engine.FillEngine` when an exit fills,
    by closing the matching :class:`Position`.

    All monetary values are in ₹.

    Attributes
    ----------
    symbol : str
    entry_time, exit_time : pd.Timestamp
    entry_price, exit_price : float
    quantity : int
    direction : int  (+1 LONG / -1 SHORT)
    direction_label : str  ("LONG" / "SHORT")
    gross_pnl : float
        (exit_price - entry_price) × qty × direction — before costs.
    entry_charges, exit_charges, total_charges : float
    net_pnl : float
        gross_pnl − total_charges.
    pnl_pct : float
        net_pnl / (entry_price × qty).
    entry_signal, exit_signal : str
    duration : str
        Human-readable duration (e.g. ``"3d 2h 15m"``).
    duration_bars : int
    mae, mfe : float
        Maximum adverse / favourable excursion in ₹.
    cumulative_portfolio : float
        Total portfolio value immediately after this trade closes.
    """
    symbol:               str
    entry_time:           pd.Timestamp
    exit_time:            pd.Timestamp
    entry_price:          float
    exit_price:           float
    quantity:             int
    direction:            int
    direction_label:      str
    gross_pnl:            float
    entry_charges:        float
    exit_charges:         float
    total_charges:        float
    net_pnl:              float
    pnl_pct:              float
    entry_signal:         str   = ""
    exit_signal:          str   = ""
    duration:             str   = ""
    duration_bars:        int   = 0
    mae:                  float = 0.0
    mfe:                  float = 0.0
    cumulative_portfolio: float = 0.0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (used for CSV export)."""
        return {
            "symbol":           self.symbol,
            "entry_time":       str(self.entry_time),
            "exit_time":        str(self.exit_time),
            "direction":        self.direction_label,
            "entry_price":      round(self.entry_price,  2),
            "exit_price":       round(self.exit_price,   2),
            "quantity":         self.quantity,
            "gross_pnl":        round(self.gross_pnl,    2),
            "entry_charges":    round(self.entry_charges, 2),
            "exit_charges":     round(self.exit_charges,  2),
            "total_charges":    round(self.total_charges, 2),
            "net_pnl":          round(self.net_pnl,       2),
            "pnl_pct":          round(self.pnl_pct,       4),
            "entry_signal":     self.entry_signal,
            "exit_signal":      self.exit_signal,
            "duration":         self.duration,
            "duration_bars":    self.duration_bars,
            "mae":              round(self.mae * self.quantity, 2),
            "mfe":              round(self.mfe * self.quantity, 2),
            "portfolio_value":  round(self.cumulative_portfolio, 2),
        }


# ---------------------------------------------------------------------------
# BacktestResult  (engine output container)
# ---------------------------------------------------------------------------

class BacktestResult:
    """
    Container returned by :meth:`backtester.engine.BacktestEngine.run`.

    Do not construct manually — the engine builds this for you.

    Attributes
    ----------
    config : BacktestConfig
    symbol : str
    trade_log : list[Trade]
    equity_curve : pd.Series
        Portfolio value at each bar (float64, same index as signals_df).
    drawdown : pd.Series
        Drawdown from peak at each bar, expressed as a negative fraction
        (e.g. −0.15 = 15 % below peak).
    signals_df : pd.DataFrame
        The DataFrame returned by the strategy, containing OHLCV columns,
        indicator columns, and the ``signal`` column.
    """

    def __init__(
        self,
        config:       BacktestConfig,
        symbol:       str,
        trade_log:    List[Trade],
        equity_curve: pd.Series,
        drawdown:     pd.Series,
        signals_df:   pd.DataFrame,
    ) -> None:
        self.config       = config
        self.symbol       = symbol
        self.trade_log    = trade_log
        self.equity_curve = equity_curve
        self.drawdown     = drawdown
        self.signals_df   = signals_df
        self._metrics: Optional[Dict] = None   # lazy-computed

    # ------------------------------------------------------------------
    def trade_df(self) -> pd.DataFrame:
        """Return all trades as a DataFrame (one row per trade)."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trade_log])

    def metrics(self) -> Dict:
        """
        Return the full performance metrics dict.

        Computed lazily and cached — subsequent calls are O(1).
        """
        if self._metrics is None:
            from backtester.performance import compute_performance
            self._metrics = compute_performance(
                trade_log    = self.trade_log,
                equity_curve = self.equity_curve,
                config       = self.config,
            )
        return self._metrics

    def summary(self) -> str:
        """Return a formatted text summary of performance metrics."""
        m = self.metrics()
        if "error" in m:
            return f"Backtest — {self.symbol} — {m['error']}"
        width = 55
        lines = [
            "", "=" * width,
            f"  BACKTEST RESULTS — {self.symbol}",
            "=" * width,
        ]
        order = [
            ("Start Date",             "start_date"),
            ("End Date",               "end_date"),
            ("Initial Capital",        "initial_capital"),
            ("Final Capital",          "final_capital"),
            ("Total Net P&L",          "total_net_pnl"),
            ("Total Return",           "total_return_pct"),
            ("CAGR",                   "cagr_pct"),
            ("Sharpe Ratio",           "sharpe_ratio"),
            ("Sortino Ratio",          "sortino_ratio"),
            ("Calmar Ratio",           "calmar_ratio"),
            ("Max Drawdown",           "max_drawdown_pct"),
            ("Total Trades",           "total_trades"),
            ("Win Rate",               "win_rate_pct"),
            ("Profit Factor",          "profit_factor"),
            ("Expectancy / Trade",     "expectancy_inr"),
            ("Avg Win",                "avg_win_inr"),
            ("Avg Loss",               "avg_loss_inr"),
            ("Max Consec. Wins",       "max_consecutive_wins"),
            ("Max Consec. Losses",     "max_consecutive_losses"),
            ("Exposure %",             "exposure_pct"),
            ("Total Commission Paid",  "total_commission_paid"),
        ]
        for label, key in order:
            val = m.get(key, "N/A")
            if isinstance(val, float):
                if key in ("total_return_pct", "cagr_pct", "win_rate_pct",
                           "max_drawdown_pct", "exposure_pct"):
                    val = f"{val:.2f}%"
                elif key in ("sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor"):
                    val = f"{val:.3f}"
                else:
                    val = f"₹{val:,.2f}"
            lines.append(f"  {label:<26}: {val}")
        lines.append("=" * width)
        return "\n".join(lines)

    def export_signals_csv(self, path: str) -> None:
        """Write OHLCV + indicators + signal column to a CSV file."""
        if self.signals_df is None or self.signals_df.empty:
            raise ValueError("No signal data available.")
        self.signals_df.to_csv(path)
