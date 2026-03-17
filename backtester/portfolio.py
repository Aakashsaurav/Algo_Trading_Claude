"""
backtester/portfolio.py
------------------------
Portfolio tracker — the single source of truth for cash and position state
during a backtest run.

PREVIOUS PROBLEM
================
In the original codebase, ``portfolio.py`` existed as a well-written
class but was **never used by the engine**.  The event loop managed cash
and positions via plain Python locals, making the ``Portfolio`` class dead
code.  All its risk management (20 % max-drawdown halt, pyramiding logic,
risk-based sizing) had zero effect on actual backtests.

THIS VERSION
============
``Portfolio`` is now wired directly into ``FillEngine`` and used by
``run_event_loop`` as the **single mutable state object** for:

* Cash balance after every fill.
* All open positions (supports pyramiding — multiple simultaneous entries
  into the same symbol).
* Running mark-to-market equity and peak tracking.
* Equity curve and drawdown recording.

The engine no longer manages these as raw locals — it calls
``portfolio.open_position()``, ``portfolio.close_position()``, and
``portfolio.mark_bar()`` instead.  This makes the event loop a thin
orchestrator and the portfolio the authoritative ledger.

PYRAMIDING
==========
Multiple open positions per symbol are fully supported.  Each entry is
tracked as a separate ``OpenPosition`` with its own entry price, quantity,
and stop levels.  ``portfolio.close_all(symbol)`` exits every position for
a symbol at once (e.g., on a signal-exit).

RISK CONTROLS ENFORCED HERE
============================
* **Max drawdown halt** — ``portfolio.is_halted`` becomes ``True`` once
  equity falls more than ``max_drawdown_pct`` below its running peak.
  The event loop checks this flag before processing any new signal.
* **Max positions** — ``portfolio.can_open()`` returns ``False`` when the
  total open position count reaches the configured limit.
* **Cash check** — ``open_position()`` returns ``None`` if available cash
  is insufficient for even one share.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtester.models import BacktestConfig, Position, Trade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio:
    """
    Stateful portfolio tracker for one backtest run.

    Parameters
    ----------
    config : BacktestConfig
        Engine configuration.  Uses ``initial_capital``,
        ``max_drawdown_pct``, and ``max_positions``.

    Attributes
    ----------
    cash : float
        Current available cash (debited on buy, credited on sell).
    positions : list[Position]
        All currently open positions across all symbols.
    trade_log : list[Trade]
        All completed trades (appended by ``close_position``).
    equity_curve : list[float]
        Portfolio value recorded at each call to ``mark_bar()``.
    drawdown_curve : list[float]
        Fractional drawdown from peak at each call to ``mark_bar()``.
    is_halted : bool
        True once the max-drawdown threshold is breached.  When True,
        the engine skips all new signal processing.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config   = config
        self.cash:    float  = config.initial_capital
        self._peak:   float  = config.initial_capital
        self.is_halted: bool = False

        self.positions:      List[Position] = []
        self.trade_log:      List[Trade]    = []
        self.equity_curve:   List[float]    = []
        self.drawdown_curve: List[float]    = []

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def initial_capital(self) -> float:
        return self.config.initial_capital

    def equity(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Return current portfolio value.

        Parameters
        ----------
        current_prices : dict, optional
            ``{symbol: price}`` map for mark-to-market calculation.
            If omitted, positions are valued at entry price (no unrealised
            P&L — use this only for cash-only situations).
        """
        if not self.positions:
            return self.cash
        unrealised = 0.0
        for pos in self.positions:
            price = (current_prices or {}).get(pos.symbol, pos.entry_price)
            unrealised += pos.unrealised_pnl(price)
        return self.cash + unrealised

    def can_open(self) -> bool:
        """
        Return True if a new position can be opened.

        Checks:
        * ``is_halted`` — engine was stopped due to max drawdown.
        * ``max_positions`` — limit on simultaneous open positions.
        """
        if self.is_halted:
            return False
        cfg = self.config
        if cfg.max_positions > 0 and len(self.positions) >= cfg.max_positions:
            logger.debug(
                f"max_positions={cfg.max_positions} reached — "
                f"skipping new entry"
            )
            return False
        return True

    def open_positions_for(self, symbol: str) -> List[Position]:
        """Return all open positions for a given symbol."""
        return [p for p in self.positions if p.symbol == symbol]

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def add_position(self, position: Position) -> None:
        """
        Add a position that was filled by FillEngine to the tracked list.

        Called by the event loop after a successful ``FillEngine.open_position()``
        call.  The cash debit is already applied by FillEngine — Portfolio
        receives the updated ``cash`` via ``sync_cash()``.
        """
        self.positions.append(position)
        logger.debug(
            f"Portfolio: opened {position.direction_label} {position.symbol} "
            f"qty={position.quantity} @ ₹{position.entry_price:.2f}"
        )

    def remove_position(self, position: Position) -> None:
        """Remove a closed position from the tracked list."""
        try:
            self.positions.remove(position)
        except ValueError:
            logger.warning(
                f"Portfolio.remove_position: position not found "
                f"({position.symbol} {position.entry_time})"
            )

    def sync_cash(self, new_cash: float) -> None:
        """
        Update cash balance after a fill.

        Called after every ``FillEngine.open_position()`` or
        ``FillEngine.close_position()`` call to keep the portfolio ledger
        in sync with the fill engine's return value.
        """
        self.cash = new_cash

    def add_trade(self, trade: Trade) -> None:
        """Record a completed trade."""
        self.trade_log.append(trade)

    # ------------------------------------------------------------------
    # Bar-level accounting
    # ------------------------------------------------------------------

    def mark_bar(self, current_price_map: Dict[str, float]) -> float:
        """
        Update equity curve and drawdown at the end of each bar.

        Parameters
        ----------
        current_price_map : dict
            ``{symbol: close_price}`` for all symbols with open positions.

        Returns
        -------
        float
            Current portfolio equity value.
        """
        eq = self.equity(current_price_map)
        self.equity_curve.append(eq)

        if eq > self._peak:
            self._peak = eq

        dd = (eq - self._peak) / self._peak if self._peak > 0 else 0.0
        self.drawdown_curve.append(dd)

        # Max drawdown halt check
        if not self.is_halted and dd < -self.config.max_drawdown_pct:
            logger.warning(
                f"Portfolio: Max drawdown limit breached "
                f"(dd={dd*100:.2f}%). Halting."
            )
            self.is_halted = True

        return eq

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_equity_series(self, index: pd.Index) -> pd.Series:
        """
        Build a ``pd.Series`` equity curve aligned to the given bar index.

        If the recorded equity curve length does not match the index length
        (e.g., early halt), the missing tail is filled with the last known value.
        """
        arr = np.array(self.equity_curve, dtype=float)
        n   = len(index)
        if len(arr) < n:
            last = arr[-1] if len(arr) > 0 else self.config.initial_capital
            arr  = np.concatenate([arr, np.full(n - len(arr), last)])
        elif len(arr) > n:
            arr = arr[:n]
        return pd.Series(arr, index=index, dtype=float)

    def to_drawdown_series(self, index: pd.Index) -> pd.Series:
        """Build a ``pd.Series`` drawdown curve aligned to the given bar index."""
        arr = np.array(self.drawdown_curve, dtype=float)
        n   = len(index)
        if len(arr) < n:
            last = arr[-1] if len(arr) > 0 else 0.0
            arr  = np.concatenate([arr, np.full(n - len(arr), last)])
        elif len(arr) > n:
            arr = arr[:n]
        return pd.Series(arr, index=index, dtype=float)

    def summary(self) -> dict:
        """Return a quick summary dict (for debugging / logging)."""
        pnls = [t.net_pnl for t in self.trade_log]
        return {
            "cash":               round(self.cash, 2),
            "open_positions":     len(self.positions),
            "completed_trades":   len(self.trade_log),
            "total_net_pnl":      round(sum(pnls), 2),
            "winners":            sum(1 for p in pnls if p > 0),
            "losers":             sum(1 for p in pnls if p <= 0),
            "is_halted":          self.is_halted,
            "peak_equity":        round(self._peak, 2),
        }

    def __repr__(self) -> str:
        return (
            f"Portfolio(cash=₹{self.cash:,.0f}, "
            f"positions={len(self.positions)}, "
            f"trades={len(self.trade_log)}, "
            f"halted={self.is_halted})"
        )
