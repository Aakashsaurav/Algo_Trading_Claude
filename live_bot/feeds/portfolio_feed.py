"""
live_bot/feeds/portfolio_feed.py
---------------------------------
Wraps the Upstox PortfolioDataStreamer WebSocket.

WHAT THIS DOES:
    Listens for real-time updates from Upstox about:
        1. Order updates  — status changes (OPEN→COMPLETE, REJECTED, etc.)
        2. Position updates — quantity/avg_price changes
        3. Holding updates — CNC holding changes
        4. GTT updates     — Good-Till-Trigger order events

    In PAPER TRADE mode (Phase 7):
        - The portfolio feed is used only for MONITORING.
        - All order state management is done by paper_broker.py.
        - Real Upstox fills are NOT expected (no real orders placed).
        - We still connect to verify WebSocket plumbing works correctly.

    In LIVE mode (Phase 8):
        - Portfolio feed is critical — it confirms actual order fills.
        - paper_broker.py is replaced by live_broker.py.

MESSAGE FORMAT (Upstox V3 Portfolio Stream):
    {
      "type": "order_update",
      "data": {
        "order_id": "2409...",
        "status": "complete",
        "instrument_token": "NSE_EQ|INE020B01018",
        "transaction_type": "BUY",
        "quantity": 10,
        "average_price": 1234.50,
        "filled_quantity": 10,
        "pending_quantity": 0,
        "order_type": "MARKET",
        "product": "I",
        "placed_by": "client_id",
        "exchange_order_id": "...",
        "exchange_time": "2024-01-01 10:17:00"
      }
    }

EDGE CASES HANDLED:
    - Message type not recognised → log warning, continue.
    - order_id not in our order book (manual order from app) → log, continue.
    - Missing fields → safe .get() with defaults.
    - Connection drops → auto_reconnect handles it.
    - Duplicate messages → order status update is idempotent.
"""

import logging
import threading
from datetime import datetime
from typing import Callable, Optional

from live_bot.state import state as live_state

logger = logging.getLogger(__name__)


def _parse_order_update(data: dict) -> dict:
    """
    Parse an order_update message from the portfolio stream.

    Returns a clean dict with all relevant fields, or None if parsing fails.
    All fields use safe .get() to avoid KeyError on missing data.
    """
    if not isinstance(data, dict):
        return {}

    return {
        "order_id":          str(data.get("order_id",          "")),
        "status":            str(data.get("status",            "unknown")).lower(),
        "instrument_token":  str(data.get("instrument_token",  "")),
        "transaction_type":  str(data.get("transaction_type",  "")).upper(),
        "quantity":          int(data.get("quantity",          0) or 0),
        "average_price":     float(data.get("average_price",   0) or 0),
        "filled_quantity":   int(data.get("filled_quantity",   0) or 0),
        "pending_quantity":  int(data.get("pending_quantity",  0) or 0),
        "order_type":        str(data.get("order_type",        "")),
        "product":           str(data.get("product",           "")),
        "exchange_order_id": str(data.get("exchange_order_id", "")),
        "exchange_time":     str(data.get("exchange_time",     "")),
        "tag":               str(data.get("tag",               "")),
    }


def _parse_position_update(data: dict) -> dict:
    """Parse a position_update message."""
    if not isinstance(data, dict):
        return {}
    return {
        "instrument_token": str(data.get("instrument_token", "")),
        "average_price":    float(data.get("average_price",  0) or 0),
        "quantity":         int(data.get("quantity",         0) or 0),
        "buy_value":        float(data.get("buy_value",      0) or 0),
        "sell_value":       float(data.get("sell_value",     0) or 0),
        "product":          str(data.get("product",          "")),
    }


class PortfolioFeed:
    """
    Manages the Upstox PortfolioDataStreamer WebSocket connection.

    Receives real-time order and position updates from Upstox and
    routes them to the appropriate handlers.
    """

    def __init__(
        self,
        access_token: str,
        on_order_update:    Optional[Callable[[dict], None]] = None,
        on_position_update: Optional[Callable[[dict], None]] = None,
        auto_reconnect_interval: int = 5,
        auto_reconnect_retries:  int = 50,
    ):
        """
        Args:
            access_token             : Valid Upstox access token.
            on_order_update          : Callback(order_dict) when order status changes.
            on_position_update       : Callback(position_dict) when position changes.
            auto_reconnect_interval  : Seconds between reconnect retries.
            auto_reconnect_retries   : Maximum reconnect attempts.
        """
        self._access_token        = access_token
        self._on_order_update     = on_order_update
        self._on_position_update  = on_position_update
        self._reconnect_interval  = auto_reconnect_interval
        self._reconnect_retries   = auto_reconnect_retries

        self._streamer            = None
        self._is_running          = False
        self._thread: Optional[threading.Thread] = None

        logger.info("[PortfolioFeed] Initialised.")

    def start(self) -> None:
        """Start the portfolio feed in a background daemon thread."""
        if self._is_running:
            logger.warning("[PortfolioFeed] Already running.")
            return

        self._is_running = True
        self._thread = threading.Thread(
            target=self._run,
            name="PortfolioFeedThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("[PortfolioFeed] Feed thread started.")

    def _run(self) -> None:
        """Main run loop in the background thread."""
        try:
            import upstox_client  # noqa

            configuration = upstox_client.Configuration()
            configuration.access_token = self._access_token

            # Enable all update types — we want full visibility
            self._streamer = upstox_client.PortfolioDataStreamer(
                upstox_client.ApiClient(configuration),
                order_update    = True,
                position_update = True,
                holding_update  = True,
                gtt_update      = True,
            )

            # Configure auto-reconnect
            self._streamer.auto_reconnect(
                True,
                self._reconnect_interval,
                self._reconnect_retries,
            )

            # Attach event handlers
            self._streamer.on("open",               self._on_open)
            self._streamer.on("message",            self._on_message)
            self._streamer.on("error",              self._on_error)
            self._streamer.on("close",              self._on_close)
            self._streamer.on("reconnecting",       self._on_reconnecting)
            self._streamer.on("autoReconnectStopped", self._on_reconnect_stopped)

            logger.info("[PortfolioFeed] Connecting to Upstox Portfolio WebSocket...")
            self._streamer.connect()  # Blocking

        except ImportError:
            logger.error(
                "[PortfolioFeed] upstox_client not installed. "
                "Run: pip install upstox-python-sdk"
            )
            live_state.set_portfolio_feed_status(False)
        except Exception as e:
            logger.error(
                f"[PortfolioFeed] Fatal error in feed thread: {e}",
                exc_info=True,
            )
            live_state.set_portfolio_feed_status(False)

    # ── Event Handlers ────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        logger.info("[PortfolioFeed] ✅ Portfolio WebSocket connected.")
        live_state.set_portfolio_feed_status(True)
        live_state.log_activity("PORTFOLIO_CONNECTED", "Portfolio stream WebSocket connected.")

    def _on_message(self, message: dict) -> None:
        """
        Route portfolio messages to the appropriate handler.

        Upstox sends different message types on the same channel:
            - "order_update"
            - "position_update"
            - "holding_update"
            - "gtt_update"
        """
        if not message or not isinstance(message, dict):
            return

        msg_type = message.get("type", "")

        try:
            if msg_type == "order_update":
                self._handle_order_update(message.get("data", {}))
            elif msg_type == "position_update":
                self._handle_position_update(message.get("data", {}))
            elif msg_type == "holding_update":
                self._handle_holding_update(message.get("data", {}))
            elif msg_type == "gtt_update":
                self._handle_gtt_update(message.get("data", {}))
            else:
                # Heartbeat, status messages, or unknown types
                logger.debug(
                    f"[PortfolioFeed] Unhandled message type: '{msg_type}'. "
                    f"Full message: {str(message)[:200]}"
                )
        except Exception as e:
            logger.error(
                f"[PortfolioFeed] Error processing message type '{msg_type}': {e}",
                exc_info=False,
            )

    def _handle_order_update(self, data: dict) -> None:
        """Process order status change."""
        parsed = _parse_order_update(data)
        if not parsed.get("order_id"):
            return

        order_id = parsed["order_id"]
        status   = parsed["status"]

        logger.info(
            f"[PortfolioFeed] Order update: {order_id} "
            f"{parsed['transaction_type']} {parsed['instrument_token']} "
            f"qty={parsed['filled_quantity']} avg_price={parsed['average_price']:.2f} "
            f"status={status}"
        )

        # Update our order book
        live_state.update_order_status(
            order_id  = order_id,
            status    = status,
            fill_price= parsed["average_price"] if status == "complete" else None,
            filled_at = datetime.now() if status == "complete" else None,
        )

        # Log to activity stream
        live_state.log_activity(
            "ORDER_UPDATE",
            f"Order {order_id[:8]}... {status.upper()} | "
            f"{parsed['transaction_type']} {parsed['quantity']}x "
            f"@ ₹{parsed['average_price']:.2f}",
            level="INFO" if status in ("complete", "open") else "WARNING",
        )

        # Forward to engine callback (for position management)
        if self._on_order_update:
            try:
                self._on_order_update(parsed)
            except Exception as e:
                logger.error(f"[PortfolioFeed] on_order_update callback error: {e}")

    def _handle_position_update(self, data: dict) -> None:
        """Process position change."""
        parsed = _parse_position_update(data)
        logger.debug(f"[PortfolioFeed] Position update: {parsed}")

        if self._on_position_update:
            try:
                self._on_position_update(parsed)
            except Exception as e:
                logger.error(f"[PortfolioFeed] on_position_update callback error: {e}")

    def _handle_holding_update(self, data: dict) -> None:
        """Log holding updates (CNC positions settling)."""
        logger.info(f"[PortfolioFeed] Holding update: {data}")
        live_state.log_activity(
            "HOLDING_UPDATE",
            f"Holding updated: {data.get('instrument_token', 'unknown')}",
        )

    def _handle_gtt_update(self, data: dict) -> None:
        """Log GTT (Good-Till-Trigger) order updates."""
        gtt_id     = data.get("id", "unknown")
        gtt_status = data.get("status", "unknown")
        logger.info(f"[PortfolioFeed] GTT update: id={gtt_id} status={gtt_status}")
        live_state.log_activity(
            "GTT_UPDATE",
            f"GTT order {gtt_id} → {gtt_status}",
        )

    def _on_error(self, error) -> None:
        logger.error(f"[PortfolioFeed] WebSocket error: {error}")
        live_state.set_portfolio_feed_status(False)
        live_state.log_activity("PORTFOLIO_ERROR", f"Portfolio feed error: {error}", level="ERROR")

    def _on_close(self) -> None:
        logger.warning("[PortfolioFeed] Portfolio WebSocket connection closed.")
        live_state.set_portfolio_feed_status(False)
        live_state.log_activity("PORTFOLIO_DISCONNECTED", "Portfolio feed disconnected.")

    def _on_reconnecting(self) -> None:
        logger.info("[PortfolioFeed] Reconnecting portfolio feed...")
        live_state.log_activity("PORTFOLIO_RECONNECTING", "Portfolio feed reconnecting...")

    def _on_reconnect_stopped(self, msg=None) -> None:
        error_msg = f"Portfolio feed auto-reconnect stopped after {self._reconnect_retries} attempts."
        logger.critical(f"[PortfolioFeed] {error_msg}")
        live_state.set_portfolio_feed_status(False)
        live_state.log_activity("PORTFOLIO_RECONNECT_FAILED", error_msg, level="CRITICAL")

    def stop(self) -> None:
        """Gracefully disconnect."""
        self._is_running = False
        if self._streamer:
            try:
                self._streamer.disconnect()
                logger.info("[PortfolioFeed] Disconnected gracefully.")
            except Exception as e:
                logger.warning(f"[PortfolioFeed] Disconnect error: {e}")
        live_state.set_portfolio_feed_status(False)
