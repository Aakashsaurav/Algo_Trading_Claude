"""
backtester/order_types.py
--------------------------
Pending order dataclass and fill-check functions for LIMIT, STOP, and
STOP-LIMIT orders.

All functions are pure (no side-effects) and operate on scalar values —
making them trivially unit-testable and reusable by both the event loop
and a future tick-replay engine.

The ``Position`` class owns its own trailing-stop logic (see models.py).
This module handles only *pending entry* orders that have not yet filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backtester.models import OrderType


# ---------------------------------------------------------------------------
# Pending entry order
# ---------------------------------------------------------------------------

@dataclass
class PendingOrder:
    """
    An entry order that has been placed but not yet filled.

    Attributes
    ----------
    direction : int
        +1 buy, -1 sell.
    order_type : OrderType
    quantity : int
    signal_bar : int
        Bar index at which the signal was generated.
    limit_price : float or None
        Fill price cap for LIMIT and STOP_LIMIT orders.
    stop_price : float or None
        Trigger price for STOP and STOP_LIMIT orders.
    expires_after : int
        Cancel after this many bars (0 = Good-Till-Cancelled).
    """
    direction:     int
    order_type:    OrderType
    quantity:      int
    signal_bar:    int
    limit_price:   Optional[float] = None
    stop_price:    Optional[float] = None
    expires_after: int = 0


# ---------------------------------------------------------------------------
# Fill-check functions  (all pure, all O(1))
# ---------------------------------------------------------------------------

def check_limit_fill(
    direction:   int,
    limit_price: float,
    open_p:      float,
    low:         float,
    high:        float,
) -> tuple[bool, float]:
    """
    Determine whether a pending LIMIT entry order fills on this bar.

    For a **buy limit** the order fills when ``low <= limit_price`` (price
    came down to the limit).  For a **sell limit** the order fills when
    ``high >= limit_price``.

    Gap handling: when the bar opens on the wrong side of the limit, we fill
    at open_p (better-than-limit fill), not at limit_price.

    Returns
    -------
    (filled, fill_price)
    """
    if direction == 1:   # BUY limit
        if open_p <= limit_price:              # gap down — fill at open
            return True, open_p
        if low <= limit_price:
            return True, limit_price
    else:                # SELL limit (short entry)
        if open_p >= limit_price:              # gap up — fill at open
            return True, open_p
        if high >= limit_price:
            return True, limit_price
    return False, 0.0


def check_stop_fill(
    direction:  int,
    stop_price: float,
    open_p:     float,
    low:        float,
    high:       float,
) -> tuple[bool, float]:
    """
    Determine whether a pending STOP entry order triggers on this bar.

    A buy stop triggers when ``high >= stop_price`` (breakout above stop).
    A sell stop triggers when ``low <= stop_price`` (breakdown below stop).

    Gap handling: when the bar opens through the stop, we fill at open_p.

    Returns
    -------
    (filled, fill_price)
    """
    if direction == 1:   # BUY stop (breakout)
        if open_p >= stop_price:
            return True, open_p
        if high >= stop_price:
            return True, stop_price
    else:                # SELL stop (breakdown)
        if open_p <= stop_price:
            return True, open_p
        if low <= stop_price:
            return True, stop_price
    return False, 0.0


def check_stop_limit_fill(
    direction:       int,
    stop_price:      float,
    limit_price:     float,
    open_p:          float,
    low:             float,
    high:            float,
    stop_triggered:  bool = False,
) -> tuple[bool, float, bool]:
    """
    Two-phase STOP-LIMIT fill check.

    Phase 1 — Stop trigger: is the stop level breached?
    Phase 2 — Limit check: if triggered, can we still fill at the limit?

    Returns
    -------
    (filled, fill_price, stop_hit)
        ``stop_hit`` remains True across bars once the stop is triggered,
        allowing the limit phase to try again on subsequent bars.
    """
    # ── Phase 1: stop trigger ──────────────────────────────────────────────
    if not stop_triggered:
        if direction == 1:
            stop_hit = open_p >= stop_price or high >= stop_price
        else:
            stop_hit = open_p <= stop_price or low <= stop_price
    else:
        stop_hit = True

    if not stop_hit:
        return False, 0.0, False

    # ── Phase 2: limit fill ────────────────────────────────────────────────
    if direction == 1:   # BUY stop-limit
        fill_price = max(open_p, stop_price)
        if fill_price <= limit_price:
            return True, fill_price, True
    else:                # SELL stop-limit
        fill_price = min(open_p, stop_price)
        if fill_price >= limit_price:
            return True, fill_price, True

    # Stop triggered but limit not reachable yet
    return False, 0.0, True
