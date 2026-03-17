"""
backtester/portfolio.py
------------------------
<<<<<<< HEAD
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
=======
Tracks cash, open positions, equity curve, and drawdown
throughout the backtest simulation.

RESPONSIBILITIES:
    - Maintain cash balance after every fill
    - Track all open positions (supports pyramiding)
    - Compute mark-to-market portfolio value at every bar
    - Record equity curve (portfolio value at each bar)
    - Compute running drawdown from peak
    - Enforce the 20% max drawdown rule (alert + halt if breached)
    - Position sizing: risk-based (never risk > 2% of capital per trade)

POSITION SIZING FORMULA:
    Risk per trade = capital × risk_pct  (default 1.5%)
    Position size  = Risk per trade / (entry_price - stop_loss)
    If no stop_loss given: fallback to ATR-based or fixed % of price

PYRAMIDING:
    Multiple open positions per symbol are allowed.
    Each entry is tracked separately with its own entry price and trade_id.
    All positions for a symbol can be closed at once with EXIT_ALL.
"""

import logging
from collections import defaultdict
from typing import Optional
import pandas as pd

from backtester.trade_log import TradeLog, TradeRecord, OpenPosition
from commission.base_commission import TradeContext, OrderType, Side
from strategies.base_strategy_github import PortfolioState

logger = logging.getLogger(__name__)

# Risk parameters
DEFAULT_RISK_PCT    = 0.015   # 1.5% of capital per trade
MAX_DRAWDOWN_LIMIT  = 0.20    # 20% max drawdown before halting
DEFAULT_FALLBACK_RISK_PCT = 0.02  # if no stop, size = 2% of capital ÷ price


class Portfolio:
    """
    Simulated portfolio for backtesting.

    Usage (by engine, not directly):
        portfolio = Portfolio(initial_capital=500_000)
        # On each bar:
        portfolio.mark_to_market(symbol, current_price)
        # On entry fill:
        qty = portfolio.compute_position_size(symbol, entry_price, stop_loss)
        portfolio.open_position(symbol, entry_price, qty, "LONG", tag, commission)
        # On exit fill:
        portfolio.close_position(position_id, exit_price, exit_tag, commission)
    """

    def __init__(
        self,
        initial_capital: float,
        commission_model=None,
        risk_pct:        float = DEFAULT_RISK_PCT,
    ):
        """
        Args:
            initial_capital (float): Starting capital in ₹.
            commission_model       : Instance of BaseCommission subclass.
            risk_pct (float)       : Max fraction of capital to risk per trade.
        """
        self.initial_capital  = initial_capital
        self.cash             = initial_capital
        self.commission_model = commission_model
        self.risk_pct         = risk_pct

        # Open positions: symbol → list of OpenPosition
        # (list because pyramiding allows multiple entries per symbol)
        self._open: dict[str, list[OpenPosition]] = defaultdict(list)

        # Completed trade log
        self.trade_log = TradeLog()

        # Equity curve: list of (timestamp, portfolio_value)
        self.equity_curve: list[tuple] = []

        # Peak value for drawdown computation
        self._peak_value = initial_capital

        # Drawdown series: list of (timestamp, drawdown_pct)
        self.drawdown_series: list[tuple] = []

        # Halted flag: set to True when max drawdown breached
        self.halted = False
        self.halt_reason = ""

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_position_size(
        self,
        entry_price: float,
        stop_loss:   Optional[float],
        capital_override: Optional[float] = None,
    ) -> int:
        """
        Compute how many units to buy based on risk rules.

        Risk-based sizing (when stop_loss is provided):
            risk_amount = capital × risk_pct
            risk_per_unit = |entry_price - stop_loss|
            quantity = floor(risk_amount / risk_per_unit)

        Fallback (no stop_loss):
            quantity = floor((capital × DEFAULT_FALLBACK_RISK_PCT) / entry_price)

        Args:
            entry_price      : Expected fill price.
            stop_loss        : Stop-loss price. None = use fallback sizing.
            capital_override : Use this capital instead of self.cash (optional).

        Returns:
            int: Position size in units. Minimum 1. Returns 0 if entry_price <= 0.
        """
        if entry_price <= 0:
            return 0

        capital = capital_override if capital_override is not None else self.cash
        risk_amount = capital * self.risk_pct

        if stop_loss is not None and stop_loss > 0:
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit < 0.01:
                # Stop too close to entry — use fallback
                qty = max(1, int((capital * DEFAULT_FALLBACK_RISK_PCT) / entry_price))
            else:
                qty = max(1, int(risk_amount / risk_per_unit))
        else:
            # No stop-loss: size by capital percentage
            qty = max(1, int((capital * DEFAULT_FALLBACK_RISK_PCT) / entry_price))

        # Hard cap: never use more than 20% of capital on a single trade
        max_qty = max(1, int(capital * 0.20 / entry_price))
        qty = min(qty, max_qty)

        # Never buy more than available cash allows
        affordable = max(1, int(self.cash / entry_price))
        qty = min(qty, affordable)

        return qty

    # ------------------------------------------------------------------
    # Opening and closing positions
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol:      str,
        entry_time:  pd.Timestamp,
        entry_price: float,
        quantity:    int,
        side:        str,           # "LONG" or "SHORT"
        entry_tag:   str,
        bar_idx:     int,
        commission_amount: float = 0.0,
    ) -> OpenPosition:
        """
        Record a new open position and deduct cash + commission.

        Args:
            symbol          : Trading symbol.
            entry_time      : Bar timestamp of the fill.
            entry_price     : Fill price.
            quantity        : Units bought/sold.
            side            : "LONG" or "SHORT".
            entry_tag       : Signal label.
            bar_idx         : Bar index in DataFrame (for duration calc).
            commission_amount: Total entry commission in ₹.

        Returns:
            OpenPosition: The newly created position object.
        """
        trade_id = self.trade_log.next_trade_id()

        pos = OpenPosition(
            trade_id      = trade_id,
            symbol        = symbol,
            entry_time    = entry_time,
            entry_price   = entry_price,
            quantity      = quantity,
            side          = side,
            entry_tag     = entry_tag,
            entry_bar_idx = bar_idx,
            commission_entry = commission_amount,
            worst_price   = entry_price,
            best_price    = entry_price,
        )

        self._open[symbol].append(pos)

        # Deduct cash (for long: buy stock; for short: margin deposit approx)
        cost = entry_price * quantity + commission_amount
        self.cash -= cost

        logger.debug(
            f"Opened {side} #{trade_id}: {symbol} qty={quantity} "
            f"@{entry_price:.2f} commission={commission_amount:.2f} "
            f"cash_remaining={self.cash:.2f}"
        )
        return pos

    def close_position(
        self,
        position:     OpenPosition,
        exit_time:    pd.Timestamp,
        exit_price:   float,
        exit_bar_idx: int,
        exit_tag:     str,
        commission_amount: float = 0.0,
    ) -> TradeRecord:
        """
        Close an open position, compute PnL, record in trade log.

        Args:
            position      : The OpenPosition to close.
            exit_time     : Bar timestamp of the exit fill.
            exit_price    : Exit fill price.
            exit_bar_idx  : Bar index of exit.
            exit_tag      : Signal label for exit.
            commission_amount: Total exit commission in ₹.

        Returns:
            TradeRecord: The completed trade record.
        """
        symbol = position.symbol

        # Remove from open positions
        if position in self._open[symbol]:
            self._open[symbol].remove(position)

        # Return cash from closing position
        proceeds = exit_price * position.quantity
        if position.side == "LONG":
            self.cash += proceeds - commission_amount
        else:
            # Short: cash was held as margin; return margin + profit/loss
            short_pnl = (position.entry_price - exit_price) * position.quantity
            self.cash += position.entry_price * position.quantity + short_pnl - commission_amount

        # Compute P&L
        direction = position.direction
        gross_pnl = (exit_price - position.entry_price) * position.quantity * direction
        total_commission = position.commission_entry + commission_amount
        net_pnl = gross_pnl - total_commission

        entry_value = position.entry_price * position.quantity
        pnl_pct = (net_pnl / entry_value * 100) if entry_value > 0 else 0.0

        # Duration
        duration_bars = exit_bar_idx - position.entry_bar_idx
        duration_str  = _format_duration(position.entry_time, exit_time)

        # MAE and MFE
        mae_val = position.mae()
        mfe_val = position.mfe()

        trade = TradeRecord(
            trade_id         = position.trade_id,
            symbol           = symbol,
            entry_time       = position.entry_time,
            exit_time        = exit_time,
            entry_price      = round(position.entry_price, 4),
            exit_price       = round(exit_price, 4),
            quantity         = position.quantity,
            side             = position.side,
            entry_tag        = position.entry_tag,
            exit_tag         = exit_tag,
            gross_pnl        = round(gross_pnl, 2),
            commission_entry = round(position.commission_entry, 2),
            commission_exit  = round(commission_amount, 2),
            net_pnl          = round(net_pnl, 2),
            pnl_pct          = round(pnl_pct, 4),
            duration_bars    = duration_bars,
            duration_str     = duration_str,
            mae              = round(mae_val, 2),
            mfe              = round(mfe_val, 2),
            portfolio_value  = round(self.total_value_approx(), 2),
        )

        self.trade_log.add(trade)
        return trade

    def close_all_positions(
        self,
        symbol:       str,
        exit_time:    pd.Timestamp,
        exit_price:   float,
        exit_bar_idx: int,
        exit_tag:     str,
        commission_model=None,
        order_type_enum=None,
    ) -> list[TradeRecord]:
        """
        Close ALL open positions for a given symbol at once.
        Used for EXIT_ALL signal, intraday square-off, expiry exit.
        """
        positions = list(self._open.get(symbol, []))
        if not positions:
            return []

        trades = []
        for pos in positions:
            comm = 0.0
            if commission_model and order_type_enum:
                ctx = TradeContext(
                    order_type = order_type_enum,
                    side       = Side.SELL if pos.side == "LONG" else Side.BUY,
                    quantity   = pos.quantity,
                    price      = exit_price,
                )
                comm = commission_model.calculate(ctx).total
            t = self.close_position(pos, exit_time, exit_price, exit_bar_idx, exit_tag, comm)
            trades.append(t)
        return trades

    # ------------------------------------------------------------------
    # Mark-to-market and equity curve
    # ------------------------------------------------------------------

    def mark_to_market(
        self,
        prices:    dict[str, float],
        timestamp: pd.Timestamp,
    ) -> float:
        """
        Compute total portfolio value (cash + open position market values)
        and record in equity curve. Update excursion trackers.

        Args:
            prices    : {symbol: current_price} for all open symbols.
            timestamp : Current bar timestamp.

        Returns:
            float: Total portfolio value.
        """
        open_value = 0.0
        for symbol, positions in self._open.items():
            price = prices.get(symbol)
            if price is None:
                continue
            for pos in positions:
                pos.update_excursions(price)
                open_value += pos.unrealised_pnl(price)
                # Unrealised = change from entry; add back the original cost
                open_value += pos.entry_price * pos.quantity

        total = self.cash + open_value
        self.equity_curve.append((timestamp, round(total, 2)))

        # Update peak and drawdown
        if total > self._peak_value:
            self._peak_value = total
        current_dd = (self._peak_value - total) / self._peak_value if self._peak_value > 0 else 0.0
        self.drawdown_series.append((timestamp, round(current_dd * 100, 4)))

        # Check max drawdown breach
        if current_dd >= MAX_DRAWDOWN_LIMIT and not self.halted:
            self.halted = True
            self.halt_reason = (
                f"MAX DRAWDOWN BREACHED: {current_dd*100:.1f}% "
                f"(limit {MAX_DRAWDOWN_LIMIT*100:.0f}%) at {timestamp}"
            )
            logger.warning(f"⚠️  BACKTEST HALTED — {self.halt_reason}")

        return total

    def total_value_approx(self) -> float:
        """
        Quick estimate of total portfolio value using last equity curve entry.
        Returns initial_capital if equity curve is empty.
        """
        if self.equity_curve:
            return self.equity_curve[-1][1]
        return self.initial_capital

    # ------------------------------------------------------------------
    # PortfolioState snapshot (passed to strategy on_bar)
    # ------------------------------------------------------------------

    def get_state(self, symbol: str = None) -> PortfolioState:
        """Build a PortfolioState snapshot for the strategy."""
        open_pos  = {}
        open_pnl  = {}

        for sym, positions in self._open.items():
            total_qty = sum(p.quantity * p.direction for p in positions)
            open_pos[sym] = total_qty

        total_val = self.total_value_approx()
        peak      = self._peak_value
        dd        = (peak - total_val) / peak if peak > 0 else 0.0

        return PortfolioState(
            cash              = round(self.cash, 2),
            total_value       = round(total_val, 2),
            open_positions    = open_pos,
            open_position_pnl = open_pnl,
            peak_value        = round(peak, 2),
            current_drawdown  = round(dd, 6),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def has_open_position(self, symbol: str) -> bool:
        return bool(self._open.get(symbol))

    def get_open_positions(self, symbol: str) -> list[OpenPosition]:
        return self._open.get(symbol, [])


def _format_duration(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Format duration between two timestamps as a human-readable string."""
    if pd.isna(start) or pd.isna(end):
        return "N/A"
    delta = end - start
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "0s"
    days    = total_seconds // 86400
    hours   = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    parts = []
    if days:    parts.append(f"{days}d")
    if hours:   parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "<1m"
>>>>>>> 8d072798ed841b92b7056b98b3d612023cbaf223
