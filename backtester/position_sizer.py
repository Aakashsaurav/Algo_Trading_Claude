"""
backtester/position_sizer.py
-----------------------------
Position sizing logic, completely decoupled from the event loop.

SIZING METHODS
==============
Two methods are supported, selected automatically based on ``BacktestConfig``:

1. **Fixed quantity** (``config.fixed_quantity > 0``)
   Trade exactly ``fixed_quantity`` shares every time, regardless of capital.

2. **Risk-based sizing** (default)
   Compute the maximum position whose potential loss equals
   ``capital_risk_pct × available_cash``.

   If a stop-loss price is available (ATR-based or fixed %)::

       risk_per_trade  = cash × capital_risk_pct
       stop_distance   = |entry_price - stop_price|
       quantity        = floor(risk_per_trade / stop_distance)

   If no stop-loss is configured, fall back to a conservative
   2 % of cash / price estimate::

       quantity = floor(cash × 0.02 / entry_price)

INSTITUTION-GRADE SAFEGUARDS
=============================
* Quantity is always floored to a whole number of shares.
* Quantity is capped so the total trade cost never exceeds available cash.
* A maximum position size of ``max_positions`` is enforced at engine level
  (not here — the engine checks open position count before calling sizer).
* If any computation yields quantity <= 0, returns 0 (no trade placed).
"""

from __future__ import annotations

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_quantity(
    cash:             float,
    entry_price:      float,
    capital_risk_pct: float,
    fixed_quantity:   int = 0,
    stop_price:       Optional[float] = None,
    atr:              Optional[float] = None,
    atr_mult:         float = 2.0,
) -> int:
    """
    Calculate how many shares to buy/sell for one trade.

    Parameters
    ----------
    cash : float
        Available capital at the moment of order placement.
    entry_price : float
        Expected fill price (next bar's open).
    capital_risk_pct : float
        Fraction of cash to risk per trade.  E.g. 0.02 = 2 %.
    fixed_quantity : int
        If > 0, return this value directly (ignores all other params).
    stop_price : float, optional
        Fixed stop-loss price.  Used to compute stop distance.
    atr : float, optional
        ATR(14) at the signal bar.  Used when stop_price is None.
    atr_mult : float
        Multiplier applied to ATR for stop distance.  Default 2.0.

    Returns
    -------
    int
        Number of shares to trade.  Always >= 0.  Returns 0 when:
        - entry_price is zero or negative
        - cash is insufficient even for one share
        - stop distance is zero (degenerate case)
    """
    # Guard: invalid price
    if entry_price <= 0 or cash <= 0:
        return 0

    # ── Fixed quantity mode ────────────────────────────────────────────────
    if fixed_quantity > 0:
        cost = entry_price * fixed_quantity
        if cost > cash:
            affordable = int(cash / entry_price)
            logger.debug(
                f"Fixed qty {fixed_quantity} costs ₹{cost:.0f} > cash ₹{cash:.0f}. "
                f"Reducing to {affordable}."
            )
            return max(0, affordable)
        return fixed_quantity

    # ── Risk-based sizing ──────────────────────────────────────────────────
    risk_rupees = cash * capital_risk_pct

    # Determine stop distance
    stop_distance: Optional[float] = None

    if stop_price is not None and stop_price > 0:
        stop_distance = abs(entry_price - stop_price)

    elif atr is not None and atr > 0:
        stop_distance = atr * atr_mult

    if stop_distance and stop_distance > 0:
        qty = int(math.floor(risk_rupees / stop_distance))
    else:
        # Fallback: size so that position = 2% of cash
        qty = int(math.floor(cash * 0.02 / entry_price))

    # Cap to available cash
    if qty <= 0:
        return 0
    max_affordable = int(math.floor(cash / entry_price))
    qty = min(qty, max_affordable)

    if qty <= 0:
        logger.debug(f"position_sizer: qty=0 (cash={cash:.0f}, price={entry_price:.2f})")
    return max(0, qty)
