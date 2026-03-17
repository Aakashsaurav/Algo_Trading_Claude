"""
backtester/fill_engine.py
--------------------------
Order fill and position lifecycle logic.

RESPONSIBILITIES
================
* Open a new position (entry fill): debit cash, build a :class:`Position`.
* Close an existing position (exit fill): credit cash, build a :class:`Trade`.
* Check and trigger fixed stop-losses.
* Check and trigger trailing stops.
* Check and fill pending LIMIT / STOP / STOP-LIMIT entry orders.

DESIGN
======
``FillEngine`` is a *stateless helper* — it receives all state as arguments
and returns results as plain Python values.  This means:

* The event loop owns all mutable state (``cash``, ``positions``, ``trades``).
* ``FillEngine`` methods are pure functions of their inputs.
* Unit testing is trivial: no mocking needed.

FILL PRICE MODEL
================
All fills occur at prices derived from bar OHLCV data:

* **Market entry** → next bar's ``open`` (conservative; no fill-price optimism).
* **Limit entry** → ``limit_price`` when ``low <= limit_price`` for buys.
* **Stop entry** → ``stop_price`` when ``high >= stop_price`` for buy stops.
* **Stop-loss exit** → ``open`` if bar opens through the stop, else ``stop_price``.
* **Trailing stop exit** → same gap logic as stop-loss.
* **Signal exit** → next bar's ``open``.

Gap handling is applied consistently: when price opens beyond an order
price, the fill is at ``open_p`` (not at the order price), preventing
unrealistic fills.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import List, Optional, Tuple

import pandas as pd

from backtester.models import BacktestConfig, Position, Trade, OrderType
from backtester.order_types import (
    PendingOrder,
    check_limit_fill,
    check_stop_fill,
    check_stop_limit_fill,
)
from backtester.position_sizer import compute_quantity
from broker.upstox.commission import CommissionModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FillEngine
# ---------------------------------------------------------------------------

class FillEngine:
    """
    Stateless helper that handles all order fill mechanics.

    Instantiate once per :class:`backtester.engine.BacktestEngine` run and
    pass ``config`` at construction time.  All methods are O(1) per call.

    Parameters
    ----------
    config : BacktestConfig
        The same config object used by the engine.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.cfg        = config
        self.commission = config.commission_model

    # ------------------------------------------------------------------
    # Entry fill
    # ------------------------------------------------------------------

    def open_position(
        self,
        direction:    int,
        exec_price:   float,
        cash:         float,
        symbol:       str,
        bar_idx:      int,
        bar_time:     pd.Timestamp,
        entry_signal: str,
        atr:          Optional[float] = None,
        stop_price:   Optional[float] = None,
    ) -> Tuple[Optional[Position], float]:
        """
        Simulate a market-order entry fill.

        Parameters
        ----------
        direction : int
            +1 LONG, -1 SHORT.
        exec_price : float
            Fill price (next bar's open for market orders).
        cash : float
            Available capital before this trade.
        symbol : str
        bar_idx : int
        bar_time : pd.Timestamp
        entry_signal : str
            Human-readable label (e.g. ``"Market Signal"``).
        atr : float, optional
            Current ATR(14) — used for risk-based sizing when no stop price.
        stop_price : float, optional
            Fixed stop-loss price pre-computed by the caller.

        Returns
        -------
        (position, new_cash)
            ``position`` is None if the order could not be filled
            (insufficient capital, zero quantity, etc.).
        """
        cfg = self.cfg

        qty = compute_quantity(
            cash             = cash,
            entry_price      = exec_price,
            capital_risk_pct = cfg.capital_risk_pct,
            fixed_quantity   = cfg.fixed_quantity,
            stop_price       = stop_price,
            atr              = atr,
            atr_mult         = cfg.stop_loss_atr_mult,
        )
        if qty <= 0:
            return None, cash

        order_side = "BUY" if direction == 1 else "SELL"
        chg = self.commission.calculate(cfg.segment, order_side, qty, exec_price)
        total_cost = exec_price * qty + chg.total if direction == 1 else chg.total

        if cash < total_cost:
            logger.debug(f"Insufficient cash: need ₹{total_cost:.0f}, have ₹{cash:.0f}")
            return None, cash

        new_cash = cash - total_cost

        # Build trailing stop level if configured
        trail_level = 0.0
        if cfg.use_trailing_stop:
            pct = cfg.trailing_stop_pct
            amt = cfg.trailing_stop_amt
            if pct > 0:
                dist = exec_price * (pct / 100.0)
            else:
                dist = amt
            trail_level = (exec_price - dist) if direction == 1 else (exec_price + dist)

        pos = Position(
            symbol              = symbol,
            entry_time          = bar_time,
            entry_price         = exec_price,
            quantity            = qty,
            direction           = direction,
            entry_signal        = entry_signal,
            entry_charges       = chg.total,
            entry_bar_idx       = bar_idx,
            stop_price          = stop_price,
            trailing_stop_pct   = cfg.trailing_stop_pct,
            trailing_stop_amt   = cfg.trailing_stop_amt,
            trailing_stop_level = trail_level,
            order_type          = cfg.default_order_type,
        )
        logger.debug(
            f"OPEN  {direction:+d} {symbol} qty={qty} @ ₹{exec_price:.2f} "
            f"charges=₹{chg.total:.2f}  cash_remaining=₹{new_cash:.0f}"
        )
        return pos, new_cash

    # ------------------------------------------------------------------
    # Exit fill
    # ------------------------------------------------------------------

    def close_position(
        self,
        pos:         Position,
        exec_price:  float,
        cash:        float,
        bar_time:    pd.Timestamp,
        bar_idx:     int,
        exit_signal: str,
        portfolio_value_after: float,
    ) -> Tuple[Trade, float]:
        """
        Simulate an exit fill, closing ``pos`` at ``exec_price``.

        Returns
        -------
        (trade, new_cash)
        """
        cfg = self.cfg
        order_side = "SELL" if pos.direction == 1 else "BUY"
        chg = self.commission.calculate(cfg.segment, order_side, pos.quantity, exec_price)

        gross_pnl = (exec_price - pos.entry_price) * pos.direction * pos.quantity
        net_pnl   = gross_pnl - pos.entry_charges - chg.total
        pnl_pct   = net_pnl / (pos.entry_price * pos.quantity) if pos.entry_price > 0 else 0.0

        # Proceeds back to cash
        if pos.direction == 1:   # long exit — receive proceeds, pay charges
            new_cash = cash + exec_price * pos.quantity - chg.total
        else:                    # short exit — return margin, receive short gain/loss
            new_cash = cash + pos.entry_price * pos.quantity + gross_pnl - chg.total

        # Duration
        try:
            delta: timedelta = bar_time - pos.entry_time
            total_s = int(delta.total_seconds())
            d = total_s // 86400
            h = (total_s % 86400) // 3600
            m = (total_s % 3600)  // 60
            parts = []
            if d: parts.append(f"{d}d")
            if h: parts.append(f"{h}h")
            if m: parts.append(f"{m}m")
            duration_str = " ".join(parts) if parts else "< 1m"
        except Exception:
            duration_str = ""

        trade = Trade(
            symbol               = pos.symbol,
            entry_time           = pos.entry_time,
            exit_time            = bar_time,
            entry_price          = pos.entry_price,
            exit_price           = exec_price,
            quantity             = pos.quantity,
            direction            = pos.direction,
            direction_label      = pos.direction_label,
            gross_pnl            = round(gross_pnl, 2),
            entry_charges        = round(pos.entry_charges, 2),
            exit_charges         = round(chg.total, 2),
            total_charges        = round(pos.entry_charges + chg.total, 2),
            net_pnl              = round(net_pnl, 2),
            pnl_pct              = round(pnl_pct, 6),
            entry_signal         = pos.entry_signal,
            exit_signal          = exit_signal,
            duration             = duration_str,
            duration_bars        = bar_idx - pos.entry_bar_idx,
            mae                  = round(pos.mae, 4),
            mfe                  = round(pos.mfe, 4),
            cumulative_portfolio = round(portfolio_value_after, 2),
        )
        logger.debug(
            f"CLOSE {pos.direction:+d} {pos.symbol} @ ₹{exec_price:.2f} "
            f"net_pnl=₹{net_pnl:.2f}  [{exit_signal}]"
        )
        return trade, new_cash

    # ------------------------------------------------------------------
    # Stop-loss and trailing-stop checks
    # ------------------------------------------------------------------

    def check_stops(
        self,
        positions: List[Position],
        cash:      float,
        open_p:    float,
        high:      float,
        low:       float,
        bar_time:  pd.Timestamp,
        bar_idx:   int,
        symbol:    str,
        portfolio_value: float,
    ) -> Tuple[List[Position], List[Trade], float]:
        """
        Check all open positions for stop-loss and trailing-stop triggers.

        Evaluates **trailing stop first**, then **fixed stop** (trailing stop
        takes precedence because it is dynamic and may be tighter).

        Parameters
        ----------
        positions : list[Position]
            All currently open positions for this symbol.
        cash : float
            Current cash before any stops fire.
        open_p, high, low : float
            Current bar OHLCV values.
        bar_time : pd.Timestamp
        bar_idx : int
        symbol : str
        portfolio_value : float
            Total portfolio value at this bar (used for cumulative_portfolio).

        Returns
        -------
        (remaining_positions, new_trades, new_cash)
        """
        remaining: List[Position] = []
        new_trades: List[Trade]   = []

        for pos in positions:
            # Update excursion metrics every bar
            mid = (high + low) / 2.0
            pos.update_excursion(mid)

            # Update trailing stop level
            pos.update_trailing_stop(high, low)

            fired    = False
            fill_price = 0.0
            reason   = ""

            # 1. Trailing stop (checked first — may be tighter than fixed)
            triggered, fp = pos.is_trailing_stop_triggered(open_p, low, high)
            if triggered:
                fired = True; fill_price = fp; reason = "Trailing Stop"

            # 2. Fixed stop-loss (only if trailing stop did not fire)
            if not fired:
                triggered, fp = pos.is_fixed_stop_triggered(open_p, low, high)
                if triggered:
                    fired = True; fill_price = fp; reason = "Stop Loss"

            if fired:
                trade, cash = self.close_position(
                    pos, fill_price, cash, bar_time, bar_idx, reason, portfolio_value
                )
                new_trades.append(trade)
            else:
                remaining.append(pos)

        return remaining, new_trades, cash

    # ------------------------------------------------------------------
    # Pending limit / stop entry orders
    # ------------------------------------------------------------------

    def check_pending_entries(
        self,
        pending:   List[PendingOrder],
        cash:      float,
        open_p:    float,
        high:      float,
        low:       float,
        bar_time:  pd.Timestamp,
        bar_idx:   int,
        symbol:    str,
        atr:       Optional[float],
    ) -> Tuple[List[PendingOrder], List[Position], float]:
        """
        Iterate pending entry orders and fill any that are triggered.

        Expired orders (``expires_after`` exceeded) are silently dropped.

        Returns
        -------
        (remaining_pending, new_positions, new_cash)
        """
        remaining_pending: List[PendingOrder] = []
        new_positions:     List[Position]     = []

        for order in pending:
            # Expiry check
            if order.expires_after > 0 and (bar_idx - order.signal_bar) > order.expires_after:
                logger.debug(f"Pending {order.order_type.value} order expired at bar {bar_idx}")
                continue

            filled = False
            fill_price = 0.0
            stop_already_triggered = getattr(order, "_stop_triggered", False)

            if order.order_type == OrderType.LIMIT:
                filled, fill_price = check_limit_fill(
                    order.direction, order.limit_price, open_p, low, high
                )

            elif order.order_type == OrderType.STOP:
                filled, fill_price = check_stop_fill(
                    order.direction, order.stop_price, open_p, low, high
                )

            elif order.order_type == OrderType.STOP_LIMIT:
                filled, fill_price, hit = check_stop_limit_fill(
                    order.direction, order.stop_price, order.limit_price,
                    open_p, low, high, stop_already_triggered
                )
                order._stop_triggered = hit  # type: ignore[attr-defined]

            if filled:
                # Compute stop price for the new position
                stop_for_pos: Optional[float] = None
                if self.cfg.stop_loss_pct > 0:
                    dist = fill_price * (self.cfg.stop_loss_pct / 100.0)
                    stop_for_pos = fill_price - dist if order.direction == 1 else fill_price + dist
                elif atr and self.cfg.stop_loss_atr_mult > 0:
                    dist = atr * self.cfg.stop_loss_atr_mult
                    stop_for_pos = fill_price - dist if order.direction == 1 else fill_price + dist

                pos, cash = self.open_position(
                    direction    = order.direction,
                    exec_price   = fill_price,
                    cash         = cash,
                    symbol       = symbol,
                    bar_idx      = bar_idx,
                    bar_time     = bar_time,
                    entry_signal = f"{order.order_type.value} Fill",
                    atr          = atr,
                    stop_price   = stop_for_pos,
                )
                if pos:
                    new_positions.append(pos)
            else:
                remaining_pending.append(order)

        return remaining_pending, new_positions, cash
