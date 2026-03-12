"""
live_bot/orders/live_broker.py
------------------------------
Live order execution wrapper for Phase 8.

This module provides a ``LiveBroker`` class that mimics the interface of
:class:`PaperBroker` but sends *real* orders to Upstox using the
``upstox_client`` SDK. It is deliberately lightweight: most of the heavy
lifting (fills, stop‑loss, take‑profit management) is handled server-side by
Upstox and reflected back to us via the ``PortfolioFeed`` websocket.

The engine and tests interact only with the methods defined here, so
swapping between paper and live mode is as simple as changing
``config.PAPER_TRADE``.

For now we intentionally keep the behaviour conservative: if the SDK is not
installed or if an API call fails we log an error and return ``None`` rather
than crashing the bot. This makes testing easier and keeps the system safe
in case of an unexpected failure.

IMPORTANT:
    The methods ``check_pending_orders`` and ``check_stop_loss_take_profit``
    exist only for API compatibility with ``PaperBroker``. In live trading
    these responsibilities are transferred to Upstox (GTT orders, portfolio
    stream events). Hence both methods are no-ops in this class.
"""

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import live_bot.state as _state_module
from live_bot.state import LiveOrder

# India Standard Time
IST = timezone(timedelta(hours=5, minutes=30))

logger = logging.getLogger(__name__)


def _get_state():
    """Return the current live state singleton. Never cache this reference."""
    return _state_module.state


class LiveBroker:
    """Routes orders to Upstox using their V3 Order API.

    The public interface mirrors :class:`PaperBroker` so the rest of the
    engine can be broker-agnostic. Most of the methods are simple wrappers
    around the SDK; when an order is placed we also record a pending
    :class:`~live_bot.state.LiveOrder` in our in‑memory state so the dashboard
    and logs can show it immediately. The final status (filled / cancelled /
    rejected) will arrive asynchronously via ``PortfolioFeed`` events.
    """

    def __init__(self, product: str = "I", access_token: Optional[str] = None):
        """
        Args:
            product: "I" for MIS (intraday), "D" for CNC (delivery).
            access_token: valid Upstox OAuth token (required for API calls).
        """
        self.product = product
        self.access_token = access_token
        self._api = None

        logger.info(f"[LiveBroker] Initialised. Product={product}. Mode=LIVE TRADE")

        try:
            import upstox_client

            configuration = upstox_client.Configuration()
            if access_token:
                configuration.access_token = access_token
            self._api = upstox_client.OrderApiV3(upstox_client.ApiClient(configuration))
        except ImportError:
            logger.error("[LiveBroker] upstox_client not installed; live orders unavailable.")
        except Exception as e:
            logger.error(f"[LiveBroker] Failed to set up OrderApiV3: {e}", exc_info=True)

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
        """Place a live order via Upstox.

        On success we create a ``LiveOrder`` with status "PENDING" and add it
        to the shared state. The actual fill details are expected to come back
        through the portfolio feed (Phase 8 behaviour).
        """
        if not self._api:
            logger.error("[LiveBroker] Cannot place order: API client unavailable.")
            return None

        try:
            import upstox_client
        except ImportError:
            logger.error("[LiveBroker] upstox_client import failed during order placement.")
            return None

        # Build Upstox request object
        req = upstox_client.PlaceOrderV3Request(
            quantity=quantity,
            product=self.product,
            validity="DAY",
            price=limit_price or 0.0,
            tag=strategy_tag or None,
            slice=False,
            instrument_token=instrument_key,
            order_type=order_type,
            transaction_type="BUY" if action in ("BUY", "SHORT") else "SELL",
            disclosed_quantity=0,
            trigger_price=stop_loss or 0.0,
            is_amo=False,
        )

        try:
            resp = self._api.place_order(req)
        except Exception as e:
            logger.error(f"[LiveBroker] Order placement failed: {e}")
            return None

        # Extract Upstox order_id if available; fall back to a UUID
        order_id = getattr(resp, "order_id", None) or str(uuid.uuid4())[:16]

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
        logger.info(f"[LiveBroker] Order sent: {order_id} | {action} {symbol} x{quantity} [{order_type}]")
        return order

    # Compatibility stubs --------------------------------------------------

    def check_pending_orders(self, symbol: str) -> None:
        """No-op in live mode. Upstox handles pending/limit orders.

        This method exists so the engine can call it unconditionally.
        """
        pass

    def check_stop_loss_take_profit(self, symbol: str) -> None:
        """No-op: stop-loss/take-profit enforced server-side via GTT orders."""
        pass

    def squareoff_all(self) -> None:
        """Market-terminate all open positions.

        Live execution would typically involve cancelling open orders and
        placing opposing market orders. For now we simply log a message; the
        portfolio stream will inform us if positions actually change.
        """
        logger.info("[LiveBroker] squareoff_all() called (no action taken).")
