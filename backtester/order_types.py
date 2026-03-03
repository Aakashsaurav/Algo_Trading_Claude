"""
backtester/order_types.py
--------------------------
Advanced order types for the backtesting engine.

THEORY — WHY ORDER TYPES MATTER IN BACKTESTING:
================================================
Market orders (execute at next bar's open) are the SIMPLEST model but not always
the most realistic. Real traders use:

  • LIMIT ORDER  — "Buy only if price drops to Rs X"
    Simulates patient entry. Only fills if the price reaches your target.
    Risk: may never fill if price moves away.

  • STOP-LOSS ORDER  — "Exit if price falls below Rs X" (for longs)
    Simulates automatic risk management. Executes at market once triggered.
    Risk: gaps — price may open below your stop (gap risk).

  • STOP-LIMIT ORDER  — "Exit no worse than Rs Y after stop Rs X is hit"
    Prevents horrible fills in fast markets. But may leave you stuck in
    a position if the limit is never reached.

  • TRAILING STOP-LOSS  — "Follow the price up, exit if it reverses by N%"
    Locks in profits dynamically. The stop price rises as the stock rises
    but never falls back.

HOW THIS MODULE WORKS:
  Each order type is evaluated per bar in the event loop BEFORE signals.
  This means stop orders can exit positions even on bars with no signal.

  Priority of evaluation:
    1. Trailing stop-loss update (adjust stop level)
    2. Stop-loss check (exit if price breached stop)
    3. Stop-limit check (exit if price is between stop and limit)
    4. Limit entry order fill check (enter if price reached limit)
    5. Signal-based market orders (next-bar open)

IMPORTANT CAVEATS (we tell you so you know your results are realistic):
  • All order fills are simulated against intrabar high/low, not tick data.
    Real fills may differ slightly.
  • Gaps at open can cause slippage through stops — we model this correctly
    (stop fills at open price when price opens BELOW stop for longs).
  • Limit orders do NOT guarantee fills — we fill at the limit price only if
    high >= limit (buy limit) or low <= limit (sell limit) that bar.

USAGE:
    from backtester.order_types import (
        OrderType, PendingOrder, StopTracker,
        check_limit_fill, check_stop_fill, check_stop_limit_fill,
        update_trailing_stop
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# =============================================================================
# Enums
# =============================================================================

class OrderType(Enum):
    """
    Supported order types.

    MARKET       — Execute immediately at next bar's open (classic backtester default).
    LIMIT        — Execute only if price reaches the specified limit price.
    STOP         — Execute at market once price breaches the stop level.
    STOP_LIMIT   — Trigger at stop price, then fill only if limit price is reachable.
    TRAILING_STOP— Dynamic stop that follows price by a fixed % or Rs amount.
    """
    MARKET       = "MARKET"
    LIMIT        = "LIMIT"
    STOP         = "STOP"
    STOP_LIMIT   = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class TrailingType(Enum):
    """Whether trailing stop is measured as a % of price or a fixed Rs amount."""
    PERCENT      = "PERCENT"
    AMOUNT       = "AMOUNT"


# =============================================================================
# Pending Order — placed but not yet filled
# =============================================================================

@dataclass
class PendingOrder:
    """
    Represents a pending order that was placed but not yet filled.

    Fields:
      direction   : +1 = buy order, -1 = sell order
      order_type  : OrderType enum
      limit_price : Fill price for LIMIT and STOP_LIMIT (optional for others)
      stop_price  : Trigger price for STOP and STOP_LIMIT
      quantity    : Number of units to trade
      signal_bar  : Bar index where the signal was generated
      expires_after: If > 0, cancel order after this many bars (0 = GTC)
    """
    direction:     int
    order_type:    OrderType
    quantity:      int
    signal_bar:    int
    limit_price:   Optional[float] = None   # For LIMIT and STOP_LIMIT orders
    stop_price:    Optional[float] = None   # For STOP and STOP_LIMIT orders
    expires_after: int = 0                  # 0 = Good Till Cancelled


# =============================================================================
# Stop Tracker — tracks trailing stop for an open position
# =============================================================================

@dataclass
class StopTracker:
    """
    Tracks a trailing stop-loss level for one open position.

    The stop level is updated each bar as price moves in favour of the trade.
    It NEVER moves against the trade (a long's trailing stop only rises).

    Args:
        direction      : +1 for long, -1 for short
        initial_price  : Entry price of the position
        trail_type     : PERCENT or AMOUNT
        trail_value    : Either pct (e.g. 2.0 = 2%) or Rs amount (e.g. 50.0)
        current_stop   : Current stop level (initialised from initial_price)
    """
    direction:     int
    initial_price: float
    trail_type:    TrailingType
    trail_value:   float
    current_stop:  float = field(init=False)

    def __post_init__(self):
        """Set initial stop level at time of entry."""
        self.current_stop = self._compute_stop(self.initial_price)

    def _compute_stop(self, reference_price: float) -> float:
        """
        Compute the stop level based on current reference price.
        For a LONG: stop = reference_price - trail_distance
        For a SHORT: stop = reference_price + trail_distance
        """
        if self.trail_type == TrailingType.PERCENT:
            dist = reference_price * (self.trail_value / 100.0)
        else:
            dist = self.trail_value

        return (reference_price - dist) if self.direction == 1 else (reference_price + dist)

    def update(self, current_high: float, current_low: float) -> None:
        """
        Update the trailing stop based on the current bar's price action.

        For LONG positions: trail stop follows HIGH upward (never downward).
        For SHORT positions: trail stop follows LOW downward (never upward).

        Args:
            current_high: Bar's high price
            current_low:  Bar's low price
        """
        if self.direction == 1:
            # Long: raise stop if high moved our stop up
            new_stop = self._compute_stop(current_high)
            self.current_stop = max(self.current_stop, new_stop)
        else:
            # Short: lower stop if low moved our stop down
            new_stop = self._compute_stop(current_low)
            self.current_stop = min(self.current_stop, new_stop)

    def is_triggered(self, open_p: float, low_p: float, high_p: float) -> tuple:
        """
        Check if trailing stop was triggered this bar.

        Uses open price for gap-down/gap-up scenarios (realistic).
        If open already breaches stop, fill at open (gap risk modelled).

        Returns:
            (triggered: bool, fill_price: float)
        """
        if self.direction == 1:
            # Long: triggered if low <= stop OR open gapped below stop
            effective_price = min(open_p, low_p)
            if effective_price <= self.current_stop:
                fill = min(open_p, self.current_stop)  # Gap-aware fill
                return True, fill
        else:
            # Short: triggered if high >= stop OR open gapped above stop
            effective_price = max(open_p, high_p)
            if effective_price >= self.current_stop:
                fill = max(open_p, self.current_stop)  # Gap-aware fill
                return True, fill
        return False, 0.0


# =============================================================================
# Order Fill Check Functions
# =============================================================================

def check_limit_fill(
    direction:   int,
    limit_price: float,
    open_p:      float,
    high_p:      float,
    low_p:       float,
) -> tuple:
    """
    Check if a pending LIMIT order fills this bar.

    A BUY LIMIT fills if the bar's LOW touches or goes below the limit price.
    (You want to buy cheaply, so you wait for price to come down to you.)

    A SELL LIMIT fills if the bar's HIGH touches or goes above the limit price.
    (You want to sell at a good price, so you wait for price to come up to you.)

    Gap handling: if open already satisfies the limit, fill at open (better fill).

    Args:
        direction:   +1 = buy limit, -1 = sell limit
        limit_price: Specified limit price
        open_p:      Bar open price
        high_p:      Bar high price
        low_p:       Bar low price

    Returns:
        (filled: bool, fill_price: float)
    """
    if direction == 1:  # BUY LIMIT — want price to come DOWN
        if open_p <= limit_price:
            # Gap-down open: filled at open (better than limit)
            return True, open_p
        elif low_p <= limit_price:
            # Price touched limit intrabar: fill at limit
            return True, limit_price
    else:               # SELL LIMIT — want price to go UP
        if open_p >= limit_price:
            # Gap-up open: filled at open (better than limit)
            return True, open_p
        elif high_p >= limit_price:
            return True, limit_price
    return False, 0.0


def check_stop_fill(
    direction:  int,
    stop_price: float,
    open_p:     float,
    high_p:     float,
    low_p:      float,
) -> tuple:
    """
    Check if a STOP order (stop-market) triggers and fills this bar.

    A BUY STOP fills if price rises above the stop level.
    (Used for breakout entries: "buy if price breaks above resistance".)

    A SELL STOP fills if price falls below the stop level.
    (Used for stop-losses on long positions: "sell if price drops too far".)

    Gap handling: if open already through the stop, fill at open (gap risk).

    Returns:
        (triggered: bool, fill_price: float)
    """
    if direction == 1:  # BUY STOP (breakout entry or cover short)
        if open_p >= stop_price:
            return True, open_p   # Gapped through stop
        elif high_p >= stop_price:
            return True, stop_price
    else:               # SELL STOP (stop-loss on long)
        if open_p <= stop_price:
            return True, open_p   # Gapped through stop (worst case)
        elif low_p <= stop_price:
            return True, stop_price
    return False, 0.0


def check_stop_limit_fill(
    direction:   int,
    stop_price:  float,
    limit_price: float,
    open_p:      float,
    high_p:      float,
    low_p:       float,
    stop_triggered: bool = False,
) -> tuple:
    """
    Check if a STOP-LIMIT order fills this bar.

    Phase 1 — Trigger: check if price breached the stop level.
    Phase 2 — Fill:    once triggered, only fill if limit price is reachable.

    For SELL STOP-LIMIT (protecting a long):
      - Stop: sell if price drops to stop_price (e.g. Rs 1000)
      - Limit: but don't accept worse than limit_price (e.g. Rs 990)
      - Risk: if gap takes price directly to Rs 950, order WON'T fill → stuck in loss

    Args:
        stop_triggered: True if stop was already triggered in a prior bar

    Returns:
        (filled: bool, fill_price: float, stop_hit: bool)
    """
    # Step 1: Was stop triggered this bar (or already triggered)?
    if not stop_triggered:
        if direction == 1:  # Buy stop
            stop_hit = open_p >= stop_price or high_p >= stop_price
        else:               # Sell stop
            stop_hit = open_p <= stop_price or low_p <= stop_price
    else:
        stop_hit = True

    if not stop_hit:
        return False, 0.0, False

    # Step 2: Stop triggered — now check if limit price is reachable
    if direction == 1:
        # BUY STOP-LIMIT: triggered (price rose to stop), now fill only if ask ≤ limit
        # Approximation: fill at stop_price if stop_price <= limit_price
        fill_price = max(open_p, stop_price)
        if fill_price <= limit_price:
            return True, fill_price, True
    else:
        # SELL STOP-LIMIT: triggered (price fell to stop), now fill only if bid ≥ limit
        fill_price = min(open_p, stop_price)
        if fill_price >= limit_price:
            return True, fill_price, True

    # Stop triggered but limit not reachable → order pending, may fill next bar
    return False, 0.0, True


# =============================================================================
# Convenience Builder Functions
# =============================================================================

def make_trailing_stop(
    direction:     int,
    entry_price:   float,
    trail_pct:     Optional[float] = None,
    trail_amount:  Optional[float] = None,
) -> StopTracker:
    """
    Create a trailing stop tracker for a newly-opened position.

    Provide EITHER trail_pct (e.g. 2.0 for 2%) OR trail_amount (e.g. 50.0 for Rs 50).

    Args:
        direction:    +1 for long, -1 for short
        entry_price:  Entry price of the position
        trail_pct:    Trail distance as percentage of price
        trail_amount: Trail distance as fixed Rs amount

    Returns:
        StopTracker instance ready to update each bar

    Raises:
        ValueError: if neither or both trail_pct and trail_amount are provided
    """
    if trail_pct is not None and trail_amount is not None:
        raise ValueError("Provide trail_pct OR trail_amount, not both.")
    if trail_pct is None and trail_amount is None:
        raise ValueError("Provide either trail_pct (%) or trail_amount (Rs).")

    if trail_pct is not None:
        return StopTracker(
            direction=direction,
            initial_price=entry_price,
            trail_type=TrailingType.PERCENT,
            trail_value=trail_pct,
        )
    else:
        return StopTracker(
            direction=direction,
            initial_price=entry_price,
            trail_type=TrailingType.AMOUNT,
            trail_value=trail_amount,
        )
