"""
live_bot/risk/risk_guard.py
----------------------------
Real-time risk management for the live trading engine.

CHECKS PERFORMED BEFORE EVERY ORDER:
    1. Kill switch — is trading globally halted?
    2. Daily loss limit — has today's loss exceeded the threshold?
    3. Max drawdown — has portfolio dropped too far from peak?
    4. Max open positions — are we already at the limit?
    5. Max position size — would this order exceed per-symbol limit?
    6. Market hours — is the market open?
    7. Squareoff time — must we close all MIS positions?
    8. Capital check — do we have enough cash for this order?
    9. Duplicate position guard — already long/short this symbol?

All checks run before every order. If ANY check fails, the order is blocked
and the reason is logged. The strategy is notified so it can handle it.

SQUAREOFF LOGIC:
    At 15:20 IST, all MIS (intraday) positions must be closed. The risk guard
    fires a squareoff event that the engine handles by sending EXIT_ALL.
"""

import logging
from datetime import datetime, time, timezone, timedelta
from typing import Optional, Tuple

import live_bot.state as _state_module

# Always access the live singleton via _state_module.state, not a cached copy.
# This ensures tests can replace live_bot.state.state and the guard picks it up.

logger = logging.getLogger(__name__)

# India Standard Time
IST = timezone(timedelta(hours=5, minutes=30))

# Market hours
MARKET_OPEN_TIME    = time(9, 15)
MARKET_CLOSE_TIME   = time(15, 30)
INTRADAY_SQUAREOFF  = time(15, 20)   # All MIS positions must close by this time
PRE_MARKET_START    = time(9,  0)    # Pre-market session starts


class RiskGuard:
    """
    Evaluates risk conditions before every trade action.

    All methods are synchronous and fast (no I/O). They read from LiveState
    which is already in memory.

    Usage:
        guard = RiskGuard(
            daily_loss_limit_pct=2.0,
            max_drawdown_pct=10.0,
            max_open_positions=5,
            max_position_pct=10.0,
        )
        allowed, reason = guard.check_order("RELIANCE", "BUY", qty=10, price=1234.5)
    """

    def __init__(
        self,
        daily_loss_limit_pct: float  = 2.0,    # Halt trading if day loss > 2%
        max_drawdown_pct:     float  = 10.0,   # Halt if portfolio drawdown > 10%
        max_open_positions:   int    = 5,      # Max simultaneous open trades
        max_position_pct:     float  = 20.0,   # Max % of capital in one position
        allow_short:          bool   = False,  # Short-selling allowed?
    ):
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct     = max_drawdown_pct
        self.max_open_positions   = max_open_positions
        self.max_position_pct     = max_position_pct
        self.allow_short          = allow_short

        # Track whether squareoff has been triggered today
        self._squareoff_triggered = False
        self._squareoff_date      = None

        logger.info(
            f"[RiskGuard] Initialised: "
            f"daily_loss_limit={daily_loss_limit_pct}% "
            f"max_dd={max_drawdown_pct}% "
            f"max_positions={max_open_positions} "
            f"max_pos_size={max_position_pct}%"
        )

    def check_order(
        self,
        symbol:     str,
        action:     str,       # "BUY", "SELL", "SHORT", "COVER", "EXIT_ALL"
        quantity:   int,
        price:      float,
    ) -> Tuple[bool, str]:
        """
        Run all pre-order risk checks.

        Returns:
            (True, "ok") if the order is allowed.
            (False, "reason") if the order is blocked, with a human-readable reason.
        """
        # ── 1. Kill switch (highest priority) ────────────────────────────────
        if _state_module.state.kill_switch:
            return False, "KILL_SWITCH: Trading halted by kill switch."

        # ── 2. Market hours check ─────────────────────────────────────────────
        now_ist = datetime.now(tz=IST)
        current_time = now_ist.time()

        # Allow EXIT/SELL/COVER outside market hours (to close stuck positions)
        is_exit_action = action in ("SELL", "COVER", "EXIT_ALL")

        if not is_exit_action:
            if current_time < MARKET_OPEN_TIME:
                return False, f"MARKET_CLOSED: Market opens at {MARKET_OPEN_TIME}. Current: {current_time}"
            if current_time >= MARKET_CLOSE_TIME:
                return False, f"MARKET_CLOSED: Market closed at {MARKET_CLOSE_TIME}. Current: {current_time}"

        # ── 3. Intraday squareoff time ────────────────────────────────────────
        if current_time >= INTRADAY_SQUAREOFF and not is_exit_action:
            return (
                False,
                f"SQUAREOFF_TIME: No new entries allowed after {INTRADAY_SQUAREOFF}. "
                f"Close positions only."
            )

        # ── 4. Cash sufficiency check (early — before drawdown/loss checks) ──
        # Checking cash before drawdown ensures INSUFFICIENT_CASH is the reported
        # reason when the portfolio is depleted, not MAX_DRAWDOWN.
        if action == "BUY" and price > 0 and quantity > 0:
            required  = price * quantity
            available = _state_module.state.cash
            if required > available:
                return (
                    False,
                    f"INSUFFICIENT_CASH: Need ₹{required:.0f}, "
                    f"have ₹{available:.0f}."
                )

        # ── 5. Daily loss limit ───────────────────────────────────────────────
        if _state_module.state._daily_loss_hit:
            return False, "DAILY_LOSS_LIMIT: Daily loss limit already hit."

        day_pnl = _state_module.state.day_pnl
        initial  = _state_module.state._initial_capital
        if initial > 0:
            day_loss_pct = abs(min(0, day_pnl)) / initial * 100
            if day_loss_pct >= self.daily_loss_limit_pct and not is_exit_action:
                _state_module.state.set_daily_loss_hit()
                msg = (
                    f"DAILY_LOSS_LIMIT: Day P&L ₹{day_pnl:.2f} "
                    f"({day_loss_pct:.2f}%) exceeds {self.daily_loss_limit_pct}% limit."
                )
                logger.warning(f"[RiskGuard] {msg}")
                _state_module.state.log_activity("RISK_DAILY_LOSS", msg, level="WARNING")
                return False, msg

        # ── 6. Max portfolio drawdown ─────────────────────────────────────────
        if _state_module.state._max_dd_hit:
            return False, "MAX_DRAWDOWN: Portfolio drawdown limit already hit."

        drawdown = _state_module.state.drawdown_pct
        if drawdown >= self.max_drawdown_pct and not is_exit_action:
            _state_module.state.set_max_dd_hit()
            msg = (
                f"MAX_DRAWDOWN: Portfolio drawdown {drawdown:.2f}% "
                f"exceeds {self.max_drawdown_pct}% limit."
            )
            logger.warning(f"[RiskGuard] {msg}")
            _state_module.state.log_activity("RISK_MAX_DRAWDOWN", msg, level="WARNING")
            return False, msg

        # ── 7. Short selling guard ────────────────────────────────────────────
        if action == "SHORT" and not self.allow_short:
            return False, "SHORT_NOT_ALLOWED: Short selling is disabled."

        # ── 8. Duplicate position guard ───────────────────────────────────────
        # Prevent opening a position if we're already in one for this symbol
        if action == "BUY" and _state_module.state.has_position(symbol):
            existing = _state_module.state.get_position(symbol)
            if existing and existing.direction > 0:
                return False, f"DUPLICATE_LONG: Already long {symbol}."

        if action == "SHORT" and _state_module.state.has_position(symbol):
            existing = _state_module.state.get_position(symbol)
            if existing and existing.direction < 0:
                return False, f"DUPLICATE_SHORT: Already short {symbol}."

        # ── 9. Max open positions ─────────────────────────────────────────────
        if action in ("BUY", "SHORT"):
            n_positions = len(_state_module.state.get_all_positions())
            if n_positions >= self.max_open_positions:
                return (
                    False,
                    f"MAX_POSITIONS: Already at max {self.max_open_positions} open positions "
                    f"(currently {n_positions})."
                )

        # ── 10. Max position size (% of capital) ──────────────────────────────
        if action in ("BUY", "SHORT") and price > 0 and quantity > 0:
            order_value = price * quantity
            total_val   = _state_module.state.total_value
            if total_val > 0:
                position_pct = order_value / total_val * 100
                if position_pct > self.max_position_pct:
                    return (
                        False,
                        f"MAX_POSITION_SIZE: Order value ₹{order_value:.0f} "
                        f"is {position_pct:.1f}% of portfolio, "
                        f"exceeds {self.max_position_pct}% limit."
                    )

        return True, "ok"

    def compute_position_size(
        self,
        price:              float,
        stop_loss:          Optional[float],
        risk_pct_per_trade: float = 1.5,
    ) -> int:
        """
        Calculate position size using the fixed-fractional risk model.

        Risk per trade = risk_pct_per_trade% of total portfolio value.
        Position size  = (portfolio * risk_pct) / (price - stop_loss)

        If stop_loss is None, uses 2% of price as default risk distance.

        Args:
            price:              Current market price.
            stop_loss:          Stop-loss price for this trade.
            risk_pct_per_trade: % of portfolio to risk per trade (e.g. 1.5).

        Returns:
            int: Number of shares to trade. 1 if calculation results in 0.
        """
        if price <= 0:
            return 1

        portfolio_value = _state_module.state.total_value
        if portfolio_value <= 0:
            return 1

        risk_amount = portfolio_value * (risk_pct_per_trade / 100)

        if stop_loss is not None and stop_loss > 0:
            risk_per_share = abs(price - stop_loss)
        else:
            risk_per_share = price * 0.02  # Default: 2% of price

        if risk_per_share <= 0:
            return 1

        qty = int(risk_amount / risk_per_share)
        qty = max(1, qty)  # At least 1 share

        logger.debug(
            f"[RiskGuard] Position size: ₹{risk_amount:.0f} risk / "
            f"₹{risk_per_share:.2f} per share = {qty} shares @ ₹{price:.2f}"
        )
        return qty

    def is_market_open(self) -> bool:
        """
        Returns True if the current IST time is within normal market hours.
        Used by the engine and dashboard to display market status.

        Market hours: 09:15 to 15:30 IST (Monday to Friday).
        Note: Does NOT check for NSE holidays — add that in a future enhancement.
        """
        now_ist = datetime.now(tz=IST)
        # Check weekday first (Mon=0 ... Fri=4)
        if now_ist.weekday() > 4:
            return False
        current_time = now_ist.time()
        return MARKET_OPEN_TIME <= current_time < MARKET_CLOSE_TIME

    def should_squareoff_now(self) -> bool:
        """
        Returns True if it's time to squareoff all intraday positions.
        Only fires once per day to avoid repeated triggers.
        """
        from datetime import date as date_cls
        today = date_cls.today()
        now_time = datetime.now(tz=IST).time()

        if self._squareoff_date == today and self._squareoff_triggered:
            return False  # Already triggered today

        if now_time >= INTRADAY_SQUAREOFF:
            self._squareoff_triggered = True
            self._squareoff_date = today
            return True

        return False

    def reset_daily_state(self) -> None:
        """Call at the start of each trading day to reset daily flags."""
        self._squareoff_triggered = False
        self._squareoff_date      = None
        logger.info("[RiskGuard] Daily state reset.")
