"""
live_bot/feeds/market_feed.py
------------------------------
Wraps the Upstox MarketDataStreamerV3 WebSocket.

WHAT THIS DOES:
    1. Connects to Upstox's live market WebSocket (V3 — V2 is discontinued).
    2. Subscribes to a list of instrument keys in "full" mode
       (live OHLC candles + bid/ask depth + OI).
    3. Decodes each incoming message (the SDK handles Protobuf decoding).
    4. Extracts ALL useful fields (LTP, OHLC, volume, OI, etc.).
    5. Updates the shared state (LiveState) and CandleRegistry on every tick.
    6. Handles auto-reconnect with configurable retry count/interval.
    7. Logs all connection events to the activity log for the dashboard.

CRITICAL EDGE CASES HANDLED:
    - Feed sends None or empty message → silently skipped.
    - Missing fields in message dict → safe .get() with defaults everywhere.
    - Instrument key not in our watch list → still processed (logged as warning).
    - Connection drops → SDK auto_reconnect handles it; we log the event.
    - Market closed → ticks will stop naturally; the feed stays connected.
    - marketOHLC may be absent on LTPC-only instruments → handled gracefully.
    - ltt (last trade time) may be None or a timestamp integer → both handled.
    - Volume can be None → defaulted to 0.
    - OI is not present for equity stocks → defaulted to 0.

SANDBOX MODE:
    When PAPER_TRADE=True and a SANDBOX_ACCESS_TOKEN is set in .env,
    this feed connects to Upstox sandbox which streams simulated ticks.
    The interface is identical — no code changes needed.
"""

import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional

from live_bot.state import TickData, state as live_state
from live_bot.candle_builder import candle_registry

logger = logging.getLogger(__name__)

# India Standard Time
IST = timezone(timedelta(hours=5, minutes=30))

# Upstox instrument key → short symbol map (populated at startup)
_KEY_TO_SYMBOL: Dict[str, str] = {}


def _parse_ltt(ltt_value) -> datetime:
    """
    Parse Last Trade Time from the Upstox feed.

    Upstox V3 returns ltt as:
        - An integer epoch in milliseconds (common)
        - A datetime object (rare)
        - None (when no trade has occurred)

    Returns current IST time as fallback.
    """
    if ltt_value is None:
        return datetime.now(tz=IST)

    if isinstance(ltt_value, (int, float)):
        try:
            # Upstox uses millisecond epoch
            return datetime.fromtimestamp(ltt_value / 1000, tz=IST)
        except (ValueError, OSError, OverflowError):
            return datetime.now(tz=IST)

    if isinstance(ltt_value, datetime):
        if ltt_value.tzinfo is None:
            return ltt_value.replace(tzinfo=IST)
        return ltt_value

    # String format fallback
    try:
        return datetime.fromisoformat(str(ltt_value))
    except ValueError:
        return datetime.now(tz=IST)


def _extract_ohlc_from_feed(market_ff: dict) -> tuple:
    """
    Extract 1-minute candle OHLC from the Upstox V3 feed message.

    IMPORTANT: This function receives the `marketFF` dict directly (already unwrapped
    by the caller). It must NOT try to unwrap "marketFF" again.

    The structure inside market_ff:
        market_ff["marketOHLC"]["ohlc"]
        → list of OHLC objects, each with {"interval": "I1", "open":..., "high":..., ...}

    Intervals provided:
        "1d"  → full day candle (one bar = yesterday)
        "I30" → 30-minute candle (current + previous)
        "I1"  → 1-minute candle (current + previous) ← we use this

    Returns:
        (open, high, low, close, volume) as floats/ints, or (0,0,0,0,0) if absent.

    EDGE CASES:
        - marketOHLC key absent → return zeros
        - ohlc list empty → return zeros
        - I1 interval not present → try first available interval
        - Any field is None → substitute 0
    """
    try:
        # market_ff is already the marketFF dict — access marketOHLC directly
        market_ohlc = market_ff.get("marketOHLC", {})
        ohlc_list = market_ohlc.get("ohlc", [])

        if not ohlc_list:
            return 0.0, 0.0, 0.0, 0.0, 0

        # Find the 1-minute candle first, fall back to first available
        target = None
        for candle in ohlc_list:
            if candle.get("interval") == "I1":
                target = candle
                break
        if target is None:
            target = ohlc_list[0]

        return (
            float(target.get("open",   0) or 0),
            float(target.get("high",   0) or 0),
            float(target.get("low",    0) or 0),
            float(target.get("close",  0) or 0),
            int(  target.get("volume", 0) or 0),
        )

    except (AttributeError, TypeError, ValueError, KeyError) as e:
        logger.debug(f"OHLC extraction error (non-critical): {e}")
        return 0.0, 0.0, 0.0, 0.0, 0


def _parse_message(message: dict) -> List[TickData]:
    """
    Parse a raw WebSocket message from MarketDataStreamerV3 into TickData objects.

    Upstox V3 message structure (full mode):
    {
      "feeds": {
        "NSE_EQ|INE020B01018": {
          "fullFeed": {
            "marketFF": {
              "ltpc": {"ltp": 1234.5, "ltt": 1700000000000, "ltq": 50, "cp": 1220.0},
              "marketDFF": {"atBuyPrice": [...], "atSellPrice": [...]},
              "marketOHLC": {"ohlc": [{"interval":"I1","open":...,"high":...}, ...]},
              "eFeedDetails": {"atp": ..., "cp": ..., "vtt": ..., "tbq": ..., "tsq": ..., "oi": ..., "lowerCB": ..., "upperCB": ...},
            }
          }
        },
        ...
      },
      "currentTs": 1700000000000,
      "type": "market_data"
    }

    Returns:
        List of TickData (one per instrument in the message).
    """
    results = []

    if not isinstance(message, dict):
        return results

    feeds = message.get("feeds", {})
    if not feeds:
        return results

    for instrument_key, feed_data in feeds.items():
        try:
            symbol = _KEY_TO_SYMBOL.get(instrument_key, instrument_key.split("|")[-1])

            full_feed   = feed_data.get("fullFeed", {})
            market_ff   = full_feed.get("marketFF", {})

            # ── LTPC (Last Trade Price + Close) ───────────────────────────────
            ltpc = market_ff.get("ltpc", {})
            ltp  = float(ltpc.get("ltp",  0) or 0)
            ltt  = _parse_ltt(ltpc.get("ltt"))
            ltq  = int(ltpc.get("ltq",    0) or 0)
            cp   = float(ltpc.get("cp",   0) or 0)   # previous close

            # Skip ticks with zero price (pre-market or error)
            if ltp <= 0:
                logger.debug(f"Skipping zero-price tick for {instrument_key}")
                continue

            # ── Extended feed details ─────────────────────────────────────────
            e_feed = market_ff.get("eFeedDetails", {})
            atp    = float(e_feed.get("atp",      0) or 0)  # Average Traded Price
            vtt    = int(  e_feed.get("vtt",      0) or 0)  # Total volume
            oi     = float(e_feed.get("oi",       0) or 0)  # Open Interest

            # Day OHLC comes from eFeedDetails or we can derive from ltpc
            # Upstox sends day open/high/low in eFeedDetails
            day_open  = float(e_feed.get("open",  ltp) or ltp)
            day_high  = float(e_feed.get("high",  ltp) or ltp)
            day_low   = float(e_feed.get("low",   ltp) or ltp)

            # ── 1-min candle from feed ─────────────────────────────────────────
            c_open, c_high, c_low, c_close, c_vol = _extract_ohlc_from_feed(market_ff)

            tick = TickData(
                instrument_key = instrument_key,
                symbol         = symbol,
                ltp            = ltp,
                ltt            = ltt,
                ltq            = ltq,
                close_price    = cp,
                open_price     = day_open,
                high_price     = day_high,
                low_price      = day_low,
                volume         = vtt,
                oi             = oi,
                candle_open    = c_open,
                candle_high    = c_high,
                candle_low     = c_low,
                candle_close   = c_close,
                candle_volume  = c_vol,
            )
            results.append(tick)

        except Exception as e:
            # Never let one bad instrument kill the whole message processing
            logger.warning(
                f"Error parsing feed for {instrument_key}: {e}",
                exc_info=False,
            )
            continue

    return results


class MarketFeed:
    """
    Manages the Upstox MarketDataStreamerV3 WebSocket connection.

    This class owns the WebSocket streamer, processes all incoming messages,
    and routes tick data to both the shared LiveState and CandleRegistry.

    Usage:
        feed = MarketFeed(access_token, instrument_map, on_candle_complete_cb)
        feed.start()      # begins connecting in background thread
        feed.stop()       # graceful disconnect
    """

    def __init__(
        self,
        access_token: str,
        instrument_map: Dict[str, str],   # {instrument_key: symbol}
        on_candle_complete: Optional[Callable] = None,
        mode: str = "full",
        auto_reconnect_interval: int = 5,
        auto_reconnect_retries:  int = 50,
    ):
        """
        Args:
            access_token             : Valid Upstox OAuth2 access token.
            instrument_map           : Dict mapping instrument_key → symbol name.
            on_candle_complete       : Callback(symbol, candle_dict) called when
                                       a 1-min candle completes. Used by the engine
                                       to trigger strategy evaluation.
            mode                     : Subscription mode: "full", "ltpc",
                                       "option_greeks", "full_d30".
            auto_reconnect_interval  : Seconds between reconnect attempts.
            auto_reconnect_retries   : Max number of reconnect attempts.
        """
        self._access_token        = access_token
        self._instrument_map      = instrument_map
        self._on_candle_complete  = on_candle_complete
        self._mode                = mode
        self._reconnect_interval  = auto_reconnect_interval
        self._reconnect_retries   = auto_reconnect_retries

        self._streamer            = None
        self._is_running          = False
        self._thread: Optional[threading.Thread] = None

        # Populate the global key→symbol map
        global _KEY_TO_SYMBOL
        _KEY_TO_SYMBOL.update(instrument_map)

        logger.info(
            f"[MarketFeed] Initialised. Mode={mode}. "
            f"Instruments={list(instrument_map.values())}"
        )

    def start(self) -> None:
        """
        Start the market feed in a background daemon thread.
        The streamer's connect() call is blocking, so it runs in its own thread.
        """
        if self._is_running:
            logger.warning("[MarketFeed] Already running. Ignoring start().")
            return

        self._is_running = True
        self._thread = threading.Thread(
            target=self._run,
            name="MarketFeedThread",
            daemon=True,   # Daemon: exits automatically when main program exits
        )
        self._thread.start()
        logger.info("[MarketFeed] Feed thread started.")

    def _run(self) -> None:
        """
        Main run loop — called in the background thread.
        Creates the streamer, attaches event handlers, and connects.
        """
        try:
            import upstox_client  # noqa: import inside thread for safety

            configuration = upstox_client.Configuration()
            configuration.access_token = self._access_token

            instrument_keys = list(self._instrument_map.keys())

            # Create streamer with initial instrument list and mode
            self._streamer = upstox_client.MarketDataStreamerV3(
                upstox_client.ApiClient(configuration),
                instrument_keys,
                self._mode,
            )

            # Configure auto-reconnect BEFORE connecting
            self._streamer.auto_reconnect(
                True,
                self._reconnect_interval,
                self._reconnect_retries,
            )

            # Attach all event handlers
            self._streamer.on("open",               self._on_open)
            self._streamer.on("message",            self._on_message)
            self._streamer.on("error",              self._on_error)
            self._streamer.on("close",              self._on_close)
            self._streamer.on("reconnecting",       self._on_reconnecting)
            self._streamer.on("autoReconnectStopped", self._on_reconnect_stopped)

            logger.info("[MarketFeed] Connecting to Upstox WebSocket V3...")
            self._streamer.connect()   # Blocking call — runs until disconnect

        except ImportError:
            logger.error(
                "[MarketFeed] upstox_client package not installed. "
                "Run: pip install upstox-python-sdk"
            )
            live_state.set_market_feed_status(False)
            live_state.log_activity(
                "FEED_ERROR",
                "upstox_client not installed. Cannot start market feed.",
                level="ERROR",
            )
        except Exception as e:
            logger.error(f"[MarketFeed] Fatal error in feed thread: {e}", exc_info=True)
            live_state.set_market_feed_status(False)
            live_state.log_activity("FEED_ERROR", f"Feed thread crashed: {e}", level="ERROR")

    # ── Event Handlers ────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        """Called when WebSocket connection is established."""
        logger.info("[MarketFeed] ✅ WebSocket connected.")
        live_state.set_market_feed_status(True)
        live_state.log_activity("FEED_CONNECTED", "Market data WebSocket connected.")

    def _on_message(self, message: dict) -> None:
        """
        Called for EVERY tick message from the WebSocket.
        This is the hot path — keep it fast, defer heavy work.

        EDGE CASES:
            - message is None → skip.
            - message is not a dict (heartbeat ping) → skip.
            - message has no "feeds" key → skip (could be status message).
            - Individual instrument parse error → skip that instrument only.
        """
        if not message:
            return

        try:
            ticks = _parse_message(message)
        except Exception as e:
            logger.debug(f"[MarketFeed] Message parse error: {e}")
            return

        for tick in ticks:
            # 1. Update shared state with latest tick
            live_state.update_tick(tick.symbol, tick)

            # 2. Feed tick into candle builder and check for completed candle
            try:
                completed_candle = candle_registry.on_tick(tick.symbol, tick)
            except Exception as e:
                logger.debug(f"[MarketFeed] Candle build error for {tick.symbol}: {e}")
                completed_candle = None

            # 3. Notify strategy engine when a candle completes
            if completed_candle is not None and self._on_candle_complete:
                try:
                    self._on_candle_complete(tick.symbol, completed_candle.to_dict())
                except Exception as e:
                    logger.error(
                        f"[MarketFeed] on_candle_complete callback error for "
                        f"{tick.symbol}: {e}",
                        exc_info=False,
                    )

    def _on_error(self, error) -> None:
        """Called when the WebSocket encounters an error."""
        logger.error(f"[MarketFeed] WebSocket error: {error}")
        live_state.set_market_feed_status(False)
        live_state.log_activity("FEED_ERROR", f"WebSocket error: {error}", level="ERROR")

    def _on_close(self) -> None:
        """Called when the WebSocket connection is closed."""
        logger.warning("[MarketFeed] WebSocket connection closed.")
        live_state.set_market_feed_status(False)
        live_state.log_activity("FEED_DISCONNECTED", "Market data WebSocket disconnected.")

    def _on_reconnecting(self) -> None:
        """Called when auto-reconnect starts a new attempt."""
        logger.info("[MarketFeed] Attempting to reconnect WebSocket...")
        live_state.log_activity("FEED_RECONNECTING", "Attempting to reconnect market feed...")

    def _on_reconnect_stopped(self, msg=None) -> None:
        """Called when all reconnect retries are exhausted."""
        error_msg = f"Auto-reconnect stopped after {self._reconnect_retries} attempts."
        logger.critical(f"[MarketFeed] {error_msg}")
        live_state.set_market_feed_status(False)
        live_state.log_activity("FEED_RECONNECT_FAILED", error_msg, level="CRITICAL")
        # Activate kill switch — we cannot trade without market data
        live_state.activate_kill_switch("Market feed permanently disconnected.")

    # ── Control methods ───────────────────────────────────────────────────────

    def subscribe(self, instrument_keys: List[str]) -> None:
        """
        Subscribe to additional instruments on the live connection.
        Safe to call after connect(). Must not be called before connection opens.
        """
        if self._streamer is None:
            logger.warning("[MarketFeed] Cannot subscribe — streamer not initialised.")
            return
        try:
            self._streamer.subscribe(instrument_keys, self._mode)
            logger.info(f"[MarketFeed] Subscribed to: {instrument_keys}")
        except Exception as e:
            logger.error(f"[MarketFeed] Subscribe error: {e}")

    def unsubscribe(self, instrument_keys: List[str]) -> None:
        """Unsubscribe from instruments."""
        if self._streamer is None:
            return
        try:
            self._streamer.unsubscribe(instrument_keys)
            logger.info(f"[MarketFeed] Unsubscribed: {instrument_keys}")
        except Exception as e:
            logger.error(f"[MarketFeed] Unsubscribe error: {e}")

    def change_mode(self, instrument_keys: List[str], new_mode: str) -> None:
        """Switch subscription mode for specific instruments."""
        if self._streamer is None:
            return
        try:
            self._streamer.change_mode(instrument_keys, new_mode)
            logger.info(f"[MarketFeed] Mode changed to '{new_mode}' for {instrument_keys}")
        except Exception as e:
            logger.error(f"[MarketFeed] change_mode error: {e}")

    def stop(self) -> None:
        """Gracefully disconnect the WebSocket and stop the feed thread."""
        self._is_running = False
        if self._streamer:
            try:
                self._streamer.disconnect()
                logger.info("[MarketFeed] WebSocket disconnected gracefully.")
            except Exception as e:
                logger.warning(f"[MarketFeed] Error during disconnect: {e}")
        live_state.set_market_feed_status(False)

    @property
    def is_connected(self) -> bool:
        return live_state._market_feed_connected
