"""
backtester/event_loop.py
-------------------------
The core bar-by-bar simulation loop.

This module has exactly ONE responsibility: iterate over ``signals_df``
one bar at a time and coordinate the fill engine and portfolio tracker.
All order-fill math lives in :mod:`fill_engine`.  All sizing math lives
in :mod:`position_sizer`.  All data structures live in :mod:`models`.

EXECUTION ORDER PER BAR
========================
On each bar ``i``:

1. **Update trailing-stop levels** and **check stop triggers** (trailing
   and fixed) via :meth:`FillEngine.check_stops`.
2. **Check pending limit / stop / stop-limit entry orders**.
3. **Intraday squareoff** (if enabled and time >= 15:20 IST).
4. **Process signal from the previous bar** — entries and exits execute
   at the *current* bar's open (next-bar execution model, no look-ahead).
5. **Record equity and drawdown** at this bar using NumPy arrays
   (converted to pd.Series only at the end — zero Pandas overhead per bar).
6. **Max drawdown guard** — halt if equity fell below the configured
   threshold, closing all positions immediately.

NO LOOK-AHEAD BIAS
==================
Signals are generated on bar ``i`` but executed on bar ``i+1``'s open.
The loop acts on ``signals[i-1]`` during bar ``i`` — strategies cannot
use bar ``i``'s OHLCV to fill at bar ``i``'s prices.
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from backtester.models import BacktestConfig, Position, Trade, OrderType
from backtester.fill_engine import FillEngine
from backtester.order_types import PendingOrder
from backtester.position_sizer import compute_quantity

logger = logging.getLogger(__name__)

_SQUAREOFF_TIME = dtime(15, 20)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_event_loop(
    signals_df: pd.DataFrame,
    config:     BacktestConfig,
    symbol:     str,
) -> Tuple[List[Trade], pd.Series, pd.Series]:
    """
    Run the bar-by-bar backtest simulation for one symbol.

    Parameters
    ----------
    signals_df : pd.DataFrame
        OHLCV data with a ``signal`` column appended by the strategy.
        Required columns: ``open``, ``high``, ``low``, ``close``, ``signal``.
    config : BacktestConfig
    symbol : str

    Returns
    -------
    trade_log : list[Trade]
    equity_curve : pd.Series  (same index as signals_df)
    drawdown : pd.Series      (fractional, <= 0)
    """
    cfg    = config
    filler = FillEngine(cfg)

    # Extract NumPy arrays — avoids per-bar DataFrame attribute lookup
    closes  = signals_df["close"].values.astype(float)
    highs   = signals_df["high"].values.astype(float)
    lows    = signals_df["low"].values.astype(float)
    opens   = signals_df["open"].values.astype(float)
    signals = signals_df["signal"].fillna(0).astype(int).values
    times   = signals_df.index
    n       = len(signals_df)

    # Pre-compute ATR(14) once — used by position sizer each bar
    atr_vals = _compute_atr14(closes, highs, lows, n)

    # Mutable state
    cash:      float                = cfg.initial_capital
    positions: List[Position]       = []
    pending:   List[PendingOrder]   = []
    trade_log: List[Trade]          = []
    halted:    bool                 = False

    # Pre-allocated output arrays (NumPy write-by-index is ~100x faster than
    # pd.Series.iloc assignment inside a loop)
    equity_arr   = np.full(n, np.nan, dtype=float)
    drawdown_arr = np.full(n, np.nan, dtype=float)
    peak_equity  = cfg.initial_capital

    # ── Main event loop ────────────────────────────────────────────────────
    for i in range(n):
        op = opens[i];  hp = highs[i];  lp = lows[i];  cp = closes[i]
        ct = times[i]
        atr_i: Optional[float] = (
            float(atr_vals[i])
            if (atr_vals is not None and not np.isnan(atr_vals[i]))
            else None
        )

        # Skip bars with invalid open (corporate action gaps, bad data)
        if np.isnan(op) or op <= 0:
            equity_arr[i] = cash
            continue

        if halted:
            equity_arr[i] = cash
            continue

        # Step 1: trailing + fixed stop checks
        port_val = _pv(cash, positions, cp)
        positions, fired, cash = filler.check_stops(
            positions, cash, op, hp, lp, ct, i, symbol, port_val
        )
        trade_log.extend(fired)

        # Step 2: pending entry orders
        pending, new_pos, cash = filler.check_pending_entries(
            pending, cash, op, hp, lp, ct, i, symbol, atr_i
        )
        positions.extend(new_pos)

        # Step 3: intraday squareoff
        if cfg.intraday_squareoff and _past_squareoff(ct):
            cash, positions, trade_log = _close_all(
                positions, cash, op, ct, i, filler, trade_log, "Intraday Squareoff"
            )

        # Step 4: signal from PREVIOUS bar (next-bar execution — no look-ahead)
        if i > 0 and signals[i - 1] != 0:
            cash, positions, pending, trade_log = _handle_signal(
                signal=int(signals[i - 1]),
                exec_price=op, prev_close=closes[i - 1],
                bar_time=ct, bar_idx=i, cash=cash, atr=atr_i,
                positions=positions, pending=pending,
                trade_log=trade_log, filler=filler, cfg=cfg, symbol=symbol,
            )

        # Step 5: record equity and drawdown
        equity = _pv(cash, positions, cp)
        equity_arr[i] = equity
        if equity > peak_equity:
            peak_equity = equity
        drawdown_arr[i] = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0

        # Step 6: max drawdown halt
        if peak_equity > 0 and (equity / peak_equity - 1.0) < -cfg.max_drawdown_pct:
            logger.warning(f"[{symbol}] Max drawdown limit breached at bar {i}. Halting.")
            cash, positions, trade_log = _close_all(
                positions, cash, cp, ct, i, filler, trade_log, "Max Drawdown Halt"
            )
            halted = True

    # Force-close any surviving positions at end of data
    if positions:
        lp_ = closes[-1]
        lt_ = times[-1]
        pv_ = _pv(cash, positions, lp_)
        for pos in list(positions):
            trade, cash = filler.close_position(
                pos, lp_, cash, lt_, n - 1, "End of Data", pv_
            )
            trade_log.append(trade)

    eq_series = pd.Series(equity_arr,   index=times, dtype=float)
    dd_series = pd.Series(drawdown_arr, index=times, dtype=float)

    final_eq = eq_series.dropna()
    if not final_eq.empty:
        logger.info(
            f"[{symbol}] {len(trade_log)} trades | "
            f"final equity=₹{float(final_eq.iloc[-1]):,.2f}"
        )
    return trade_log, eq_series, dd_series


# ---------------------------------------------------------------------------
# Signal processing helper
# ---------------------------------------------------------------------------

def _handle_signal(
    signal:     int,
    exec_price: float,
    prev_close: float,
    bar_time,
    bar_idx:    int,
    cash:       float,
    atr:        Optional[float],
    positions:  List[Position],
    pending:    List[PendingOrder],
    trade_log:  List[Trade],
    filler:     FillEngine,
    cfg:        BacktestConfig,
    symbol:     str,
) -> Tuple[float, List[Position], List[PendingOrder], List[Trade]]:
    """Translate a strategy signal into order actions."""
    ot = cfg.default_order_type

    # Close opposing positions
    remaining: List[Position] = []
    port_val = _pv(cash, positions, exec_price)
    for pos in positions:
        should_close = (
            (signal ==  1 and pos.direction == -1) or
            (signal == -1 and pos.direction ==  1)
        )
        if should_close:
            trade, cash = filler.close_position(
                pos, exec_price, cash, bar_time, bar_idx, "Signal Exit", port_val
            )
            trade_log.append(trade)
        else:
            remaining.append(pos)
    positions = remaining

    # Guards
    if cfg.max_positions > 0 and len(positions) >= cfg.max_positions:
        return cash, positions, pending, trade_log
    if signal == -1 and not cfg.allow_shorting:
        return cash, positions, pending, trade_log

    # Stop price for new entry
    stop_price = _entry_stop(signal, exec_price, atr, cfg)

    # Route by order type
    if ot == OrderType.MARKET:
        pos, cash = filler.open_position(
            direction=signal, exec_price=exec_price, cash=cash,
            symbol=symbol, bar_idx=bar_idx, bar_time=bar_time,
            entry_signal="Market Signal", atr=atr, stop_price=stop_price,
        )
        if pos:
            positions.append(pos)

    else:
        pct = cfg.limit_offset_pct / 100.0
        qty = compute_quantity(
            cash, exec_price, cfg.capital_risk_pct,
            cfg.fixed_quantity, stop_price, atr, cfg.stop_loss_atr_mult,
        )
        if qty > 0:
            if ot == OrderType.LIMIT:
                lp = prev_close * (1.0 - pct) if signal == 1 else prev_close * (1.0 + pct)
                pending.append(PendingOrder(
                    direction=signal, order_type=OrderType.LIMIT,
                    quantity=qty, signal_bar=bar_idx, limit_price=lp,
                ))

            elif ot == OrderType.STOP:
                sp = prev_close * (1.0 + pct) if signal == 1 else prev_close * (1.0 - pct)
                pending.append(PendingOrder(
                    direction=signal, order_type=OrderType.STOP,
                    quantity=qty, signal_bar=bar_idx, stop_price=sp,
                ))

            elif ot == OrderType.STOP_LIMIT:
                sp  = prev_close * (1.0 + pct) if signal == 1 else prev_close * (1.0 - pct)
                lim = sp * (1.0 + pct) if signal == 1 else sp * (1.0 - pct)
                pending.append(PendingOrder(
                    direction=signal, order_type=OrderType.STOP_LIMIT,
                    quantity=qty, signal_bar=bar_idx,
                    stop_price=sp, limit_price=lim,
                ))

    return cash, positions, pending, trade_log


# ---------------------------------------------------------------------------
# Micro-helpers (keep hot-path readable)
# ---------------------------------------------------------------------------

def _pv(cash: float, positions: List[Position], price: float) -> float:
    """Portfolio value = cash + sum of mark-to-market unrealised P&L."""
    return cash + sum(p.unrealised_pnl(price) for p in positions)


def _close_all(
    positions:  List[Position],
    cash:       float,
    fill_price: float,
    bar_time,
    bar_idx:    int,
    filler:     FillEngine,
    trade_log:  List[Trade],
    reason:     str,
) -> Tuple[float, List[Position], List[Trade]]:
    """Close every open position at ``fill_price``."""
    port_val = _pv(cash, positions, fill_price)
    for pos in list(positions):
        trade, cash = filler.close_position(
            pos, fill_price, cash, bar_time, bar_idx, reason, port_val
        )
        trade_log.append(trade)
    return cash, [], trade_log


def _past_squareoff(bar_time) -> bool:
    try:
        return bar_time.time() >= _SQUAREOFF_TIME
    except AttributeError:
        return False


def _entry_stop(
    direction: int, price: float, atr: Optional[float], cfg: BacktestConfig
) -> Optional[float]:
    """Compute fixed stop-loss price for a new entry (None if not configured)."""
    if cfg.stop_loss_pct > 0:
        dist = price * (cfg.stop_loss_pct / 100.0)
        return price - dist if direction == 1 else price + dist
    if atr and cfg.stop_loss_atr_mult > 0:
        dist = atr * cfg.stop_loss_atr_mult
        return price - dist if direction == 1 else price + dist
    return None


def _compute_atr14(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, n: int
) -> Optional[np.ndarray]:
    """
    Wilder's smoothed ATR(14) — vectorised pre-computation.

    Returns None for series shorter than 15 bars or on error.
    The first 14 values are NaN (warm-up period).
    """
    try:
        if n < 15:
            return None
        pc   = closes[:-1]
        tr   = np.maximum(highs[1:] - lows[1:],
               np.maximum(np.abs(highs[1:] - pc), np.abs(lows[1:] - pc)))
        atr  = np.full(n, np.nan, dtype=float)
        # Seed ATR with simple mean of first 14 TR values
        atr[14] = float(tr[:14].mean())
        for k in range(15, n):
            atr[k] = (atr[k - 1] * 13.0 + tr[k - 1]) / 14.0
        return atr
    except Exception as exc:
        logger.debug(f"ATR pre-computation error: {exc}")
        return None
