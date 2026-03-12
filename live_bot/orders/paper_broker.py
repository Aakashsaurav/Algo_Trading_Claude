"""
live_bot/orders/paper_broker.py
--------------------------------
Paper trading order simulator for Phase 7.

WHAT THIS IS:
    In paper trade mode, all orders are SIMULATED. No real orders are sent
    to Upstox. This lets you:
        - Test your strategy logic with live market data feeds.
        - See how signals would have executed in real market conditions.
        - Track a hypothetical portfolio with realistic fills.

HOW FILLS ARE SIMULATED:
    - MARKET orders: Fill immediately at current LTP + slippage.
    - LIMIT orders:  Fill when market price crosses the limit price.
    - Slippage:      0.05% of price added to BUY, subtracted from SELL.
                     This simulates the bid-ask spread cost.

COMMISSION MODEL:
    Equity Intraday: 0.03% per side (both buy and sell)
    Equity Delivery: 0.1% buy only
    Both have minimum ₹20 per order.

POSITION MANAGEMENT:
    The broker manages the transition from order → position:
        BUY order FILLED  → open LONG position in LiveState
        SELL order FILLED → close LONG position, record P&L
        SHORT order FILLED→ open SHORT position (if allowed)
        COVER order FILLED→ close SHORT position

EDGE CASES HANDLED:
    - SELL with no open position → order rejected.
    - Quantity mismatch (sell more than owned) → quantity clamped.
    - Zero LTP at fill time → order rejected with "no_price" reason.
    - Duplicate order IDs → UUID-based so collision is practically impossible.
"""

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import live_bot.state as _state_module
from live_bot.state import (
    LiveOrder,
    LivePosition,
    ClosedTrade,
)

# Always access the live singleton via _state_module.state (not a cached copy).
# This allows tests to replace live_bot.state.state and have the broker pick it up.

def _get_state():
    """Return the current live state singleton. Never cache this reference."""
    return _state_module.state

logger = logging.getLogger(__name__)

# India Standard Time
IST = timezone(timedelta(hours=5, minutes=30))

# Slippage: % added/subtracted to simulate bid-ask spread
SLIPPAGE_PCT = 0.0005   # 0.05%

# Commission rates
COMMISSION_INTRADAY_PCT = 0.0003  # 0.03% per side
COMMISSION_DELIVERY_PCT = 0.001   # 0.1% buy-only
MIN_COMMISSION          = 20.0    # ₹20 minimum per order


def _compute_slippage(price: float, action: str) -> float:
    """
    Compute fill price with slippage applied.
    BUY  → price slightly higher (market impact of buying)
    SELL → price slightly lower  (market impact of selling)
    """
    if action in ("BUY", "SHORT"):
        return round(price * (1 + SLIPPAGE_PCT), 2)
    else:
        return round(price * (1 - SLIPPAGE_PCT), 2)


def _compute_commission(price: float, quantity: int, product: str = "I") -> float:
    """
    Compute brokerage commission.

    Args:
        price:    Fill price.
        quantity: Number of shares.
        product:  "I" for MIS (intraday), "D" for CNC (delivery).
    """
    trade_value = price * quantity
    if product == "D":
        commission = trade_value * COMMISSION_DELIVERY_PCT
    else:
        commission = trade_value * COMMISSION_INTRADAY_PCT
    return max(commission, MIN_COMMISSION)


class PaperBroker:
    """
    Simulates order execution for paper trading.

    The engine calls place_order() when a strategy fires a signal.
    The broker checks the current LTP, applies slippage, and immediately
    fills market orders. Limit orders are queued and checked on each tick.

    IMPORTANT: This is NOT the live broker (Phase 8). In Phase 8, this
    class will be replaced by LiveBroker which actually calls Upstox's
    OrderApiV3 to place real orders.
    """

    def __init__(self, product: str = "I"):
        """
        Args:
            product: "I" for MIS (intraday auto-squareoff),
                     "D" for CNC (delivery / positional).
        """
        self.product = product
        logger.info(f"[PaperBroker] Initialised. Product={product}. Mode=PAPER TRADE")

    def place_order(
        self,
        symbol:          str,
        instrument_key:  str,
        action:          str,          # "BUY", "SELL", "SHORT", "COVER"
        quantity:        int,
        order_type:      str = "MARKET",
        limit_price:     Optional[float] = None,
        stop_loss:       Optional[float] = None,
        take_profit:     Optional[float] = None,
        strategy_tag:    str = "",
    ) -> Optional[LiveOrder]:
        """
        Place a paper trade order.

        For MARKET orders: fills immediately at LTP + slippage.
        For LIMIT orders:  creates a pending order, filled by check_pending_orders().

        Returns:
            LiveOrder if order was created, None if rejected.
        """
        # Get current price for this symbol
        tick = _get_state().get_tick(symbol)
        current_ltp = tick.ltp if tick else 0.0

        if current_ltp <= 0:
            logger.warning(
                f"[PaperBroker] Rejecting {action} {symbol}: "
                "No price data available (LTP=0)."
            )
            _get_state().log_activity(
                "ORDER_REJECTED",
                f"Order rejected: {symbol} {action} — no price data.",
                level="WARNING",
            )
            return None

        # Validate sell/cover against existing position
        if action in ("SELL", "COVER"):
            position = _get_state().get_position(symbol)
            if position is None:
                logger.warning(
                    f"[PaperBroker] Rejecting {action} {symbol}: No open position to close."
                )
                _get_state().log_activity(
                    "ORDER_REJECTED",
                    f"Order rejected: {symbol} {action} — no open position.",
                    level="WARNING",
                )
                return None
            # Clamp quantity to actual position size
            if quantity > position.quantity:
                logger.warning(
                    f"[PaperBroker] Clamping {symbol} {action} qty "
                    f"{quantity} → {position.quantity} (actual position size)."
                )
                quantity = position.quantity

        order_id = str(uuid.uuid4())[:16]  # Short unique ID

        order = LiveOrder(
            order_id       = order_id,
            symbol         = symbol,
            instrument_key = instrument_key,
            action         = action,
            quantity       = quantity,
            order_type     = order_type,
            limit_price    = limit_price,
            status         = "PENDING",
            created_at     = datetime.now(tz=IST),
            strategy_tag   = strategy_tag,
        )

        _get_state().add_order(order)
        logger.info(
            f"[PaperBroker] Order created: {order_id} | "
            f"{action} {symbol} x{quantity} [{order_type}]"
        )

        # Market orders fill immediately
        if order_type == "MARKET":
            fill_price = _compute_slippage(current_ltp, action)
            self._fill_order(order, fill_price, stop_loss, take_profit)

        # Limit orders go into pending queue
        # They are processed in check_pending_orders() called by the engine on each tick

        return order

    def check_pending_orders(self, symbol: str) -> None:
        """
        Check if any pending limit orders for this symbol should be filled.
        Called by the engine on every tick for each watched symbol.

        Fill condition:
            BUY limit:  LTP <= limit_price  (price came down to our buy level)
            SELL limit: LTP >= limit_price  (price rose to our sell level)
        """
        tick = _get_state().get_tick(symbol)
        if tick is None:
            return

        ltp = tick.ltp
        all_orders = _get_state().get_all_orders()

        for order_id, order in all_orders.items():
            if (
                order.symbol == symbol
                and order.status == "PENDING"
                and order.order_type == "LIMIT"
                and order.limit_price is not None
            ):
                should_fill = (
                    (order.action == "BUY"   and ltp <= order.limit_price) or
                    (order.action == "SELL"  and ltp >= order.limit_price) or
                    (order.action == "SHORT" and ltp >= order.limit_price) or
                    (order.action == "COVER" and ltp <= order.limit_price)
                )

                if should_fill:
                    fill_price = _compute_slippage(ltp, order.action)
                    # Retrieve the position's stop/target (stored when order was created)
                    position = _get_state().get_position(symbol)
                    sl = position.stop_loss   if position else None
                    tp = position.take_profit if position else None
                    self._fill_order(order, fill_price, sl, tp)

    def check_stop_loss_take_profit(self, symbol: str) -> None:
        """
        Check if any open position's stop-loss or take-profit was hit.
        Called by the engine on every tick.

        This is the core risk management loop for paper trading.
        In live trading (Phase 8), GTT orders on Upstox handle this server-side.
        """
        tick     = _get_state().get_tick(symbol)
        position = _get_state().get_position(symbol)

        if tick is None or position is None:
            return

        ltp = tick.ltp

        # ── Stop loss check ───────────────────────────────────────────────────
        if position.stop_loss is not None and position.stop_loss > 0:
            sl_hit = (
                (position.direction > 0 and ltp <= position.stop_loss) or  # Long SL
                (position.direction < 0 and ltp >= position.stop_loss)      # Short SL
            )
            if sl_hit:
                logger.warning(
                    f"[PaperBroker] STOP LOSS HIT: {symbol} "
                    f"LTP={ltp:.2f} SL={position.stop_loss:.2f}"
                )
                self._exit_position(symbol, ltp, "STOP_LOSS")
                return

        # ── Take profit check ─────────────────────────────────────────────────
        if position.take_profit is not None and position.take_profit > 0:
            tp_hit = (
                (position.direction > 0 and ltp >= position.take_profit) or  # Long TP
                (position.direction < 0 and ltp <= position.take_profit)      # Short TP
            )
            if tp_hit:
                logger.info(
                    f"[PaperBroker] TAKE PROFIT HIT: {symbol} "
                    f"LTP={ltp:.2f} TP={position.take_profit:.2f}"
                )
                self._exit_position(symbol, ltp, "TAKE_PROFIT")

    def squareoff_all(self) -> None:
        """
        Close all open positions immediately at current LTP.
        Called at 15:20 IST for intraday squareoff.
        """
        positions = _get_state().get_all_positions()
        if not positions:
            logger.info("[PaperBroker] Squareoff: No open positions.")
            return

        logger.info(f"[PaperBroker] Squareoff: Closing {len(positions)} position(s).")
        for symbol in list(positions.keys()):
            tick = _get_state().get_tick(symbol)
            ltp  = tick.ltp if tick and tick.ltp > 0 else None
            if ltp:
                self._exit_position(symbol, ltp, "SQUAREOFF")
            else:
                logger.warning(
                    f"[PaperBroker] Squareoff: No price for {symbol}. "
                    "Cannot close position — manual intervention needed."
                )

    def _fill_order(
        self,
        order:       LiveOrder,
        fill_price:  float,
        stop_loss:   Optional[float],
        take_profit: Optional[float],
    ) -> None:
        """
        Simulate an order fill. Updates order state and opens/closes positions.
        """
        commission = _compute_commission(fill_price, order.quantity, self.product)

        _get_state().update_order_status(
            order_id   = order.order_id,
            status     = "FILLED",
            fill_price = fill_price,
            filled_at  = datetime.now(tz=IST),
        )

        logger.info(
            f"[PaperBroker] FILLED: {order.order_id} | "
            f"{order.action} {order.symbol} x{order.quantity} "
            f"@ ₹{fill_price:.2f} | Commission: ₹{commission:.2f}"
        )

        if order.action == "BUY":
            cost = fill_price * order.quantity + commission
            try:
                _get_state().debit_cash(cost)
            except ValueError as exc:
                # Not enough cash — reject the fill after the order was created.
                logger.warning(
                    f"[PaperBroker] Fill rejected for {order.symbol}: {exc}"
                )
                _get_state().update_order_status(order.order_id, "REJECTED")
                _get_state().log_activity(
                    "ORDER_REJECTED",
                    f"Fill rejected: {order.symbol} — {exc}",
                    level="WARNING",
                )
                return

            position = LivePosition(
                symbol         = order.symbol,
                instrument_key = order.instrument_key,
                direction      = 1,
                quantity       = order.quantity,
                entry_price    = fill_price,
                entry_time     = datetime.now(tz=IST),
                stop_loss      = stop_loss,
                take_profit    = take_profit,
                strategy_tag   = order.strategy_tag,
            )
            _get_state().add_position(position)
            _get_state().log_activity(
                "TRADE_ENTRY",
                f"📈 BUY {order.symbol} x{order.quantity} @ ₹{fill_price:.2f} "
                f"| SL={stop_loss} TP={take_profit}",
            )

        elif order.action in ("SELL", "COVER"):
            self._close_position_on_fill(order, fill_price, commission)

        elif order.action == "SHORT":
            proceeds = fill_price * order.quantity - commission
            _get_state().credit_cash(proceeds)

            position = LivePosition(
                symbol         = order.symbol,
                instrument_key = order.instrument_key,
                direction      = -1,
                quantity       = order.quantity,
                entry_price    = fill_price,
                entry_time     = datetime.now(tz=IST),
                stop_loss      = stop_loss,
                take_profit    = take_profit,
                strategy_tag   = order.strategy_tag,
            )
            _get_state().add_position(position)
            _get_state().log_activity(
                "TRADE_ENTRY",
                f"📉 SHORT {order.symbol} x{order.quantity} @ ₹{fill_price:.2f}",
            )

    def _close_position_on_fill(
        self,
        order:      LiveOrder,
        fill_price: float,
        commission: float,
    ) -> None:
        """Close an existing position and record the closed trade."""
        position = _get_state().close_position(order.symbol)
        if position is None:
            logger.error(
                f"[PaperBroker] _close_position_on_fill: "
                f"No position found for {order.symbol} during fill."
            )
            return

        proceeds = fill_price * order.quantity - commission
        _get_state().credit_cash(proceeds)

        pnl     = (fill_price - position.entry_price) * order.quantity * position.direction
        pnl_net = pnl - commission
        pnl_pct = pnl_net / (position.entry_price * order.quantity) * 100

        trade = ClosedTrade(
            symbol       = order.symbol,
            direction    = "LONG" if position.direction > 0 else "SHORT",
            quantity     = order.quantity,
            entry_price  = position.entry_price,
            exit_price   = fill_price,
            entry_time   = position.entry_time,
            exit_time    = datetime.now(tz=IST),
            pnl          = round(pnl_net, 2),
            pnl_pct      = round(pnl_pct, 2),
            strategy_tag = order.strategy_tag or position.strategy_tag,
            exit_reason  = "SIGNAL",
        )
        _get_state().record_closed_trade(trade)

        emoji = "✅" if pnl_net >= 0 else "🔴"
        _get_state().log_activity(
            "TRADE_EXIT",
            f"{emoji} EXIT {order.symbol} x{order.quantity} @ ₹{fill_price:.2f} "
            f"| P&L: ₹{pnl_net:+.2f} ({pnl_pct:+.2f}%)",
            level="INFO" if pnl_net >= 0 else "WARNING",
        )

    def _exit_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """Exit a position for reasons other than a strategy signal (SL/TP/squareoff)."""
        position = _get_state().get_position(symbol)
        if position is None:
            return

        action    = "SELL" if position.direction > 0 else "COVER"
        fill_px   = _compute_slippage(exit_price, action)
        commission = _compute_commission(fill_px, position.quantity, self.product)

        pnl       = (fill_px - position.entry_price) * position.quantity * position.direction
        pnl_net   = pnl - commission
        pnl_pct   = pnl_net / (position.entry_price * position.quantity) * 100

        _get_state().close_position(symbol)
        proceeds = fill_px * position.quantity
        _get_state().credit_cash(proceeds - commission)

        trade = ClosedTrade(
            symbol       = symbol,
            direction    = "LONG" if position.direction > 0 else "SHORT",
            quantity     = position.quantity,
            entry_price  = position.entry_price,
            exit_price   = fill_px,
            entry_time   = position.entry_time,
            exit_time    = datetime.now(tz=IST),
            pnl          = round(pnl_net, 2),
            pnl_pct      = round(pnl_pct, 2),
            strategy_tag = position.strategy_tag,
            exit_reason  = reason,
        )
        _get_state().record_closed_trade(trade)

        emoji = "✅" if pnl_net >= 0 else "🔴"
        _get_state().log_activity(
            "TRADE_EXIT",
            f"{emoji} {reason}: {symbol} x{position.quantity} @ ₹{fill_px:.2f} "
            f"| P&L: ₹{pnl_net:+.2f} ({pnl_pct:+.2f}%)",
            level="INFO" if pnl_net >= 0 else "WARNING",
        )
        logger.info(
            f"[PaperBroker] {reason}: {symbol} "
            f"{'LONG' if position.direction > 0 else 'SHORT'} "
            f"x{position.quantity} @ ₹{fill_px:.2f} | P&L: ₹{pnl_net:+.2f}"
        )
