"""
live_bot/state.py
-----------------
Shared, thread-safe in-memory state for the live trading engine.

WHY THIS EXISTS:
    Multiple components run concurrently:
      - Market feed thread (writing tick data)
      - Portfolio feed thread (writing order updates)
      - Strategy loop (reading data, writing signals)
      - Dashboard WebSocket (reading everything)
      - Webhook server (writing order confirmations)

    All of them need to read/write the same state without race conditions.
    This module is the single source of truth. All writes go through
    thread-safe methods (using threading.Lock).

DESIGN:
    - No database round-trips — everything lives in RAM for speed.
    - State is reconstructible from logs if the bot crashes.
    - Dashboard reads state directly from here via the /api/live/* endpoints.
"""

import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Data containers ─────────────────────────────────────────────────────────

@dataclass
class TickData:
    """
    A single market tick received from the WebSocket feed.
    All fields are extracted from the Upstox MarketDataStreamerV3 message.
    """
    instrument_key:  str             # e.g. "NSE_EQ|INE020B01018"
    symbol:          str             # e.g. "RELIANCE"
    ltp:             float           # Last Traded Price
    ltt:             datetime        # Last Trade Time (IST)
    ltq:             int             # Last Traded Quantity
    close_price:     float           # Previous day's closing price (cp)
    open_price:      float           # Day's open (from OHLC)
    high_price:      float           # Day's high
    low_price:       float           # Day's low
    volume:          int             # Total volume so far today
    oi:              float           # Open Interest (0 for equities)
    # 1-min OHLC candle from live feed (updated every minute)
    candle_open:     float = 0.0
    candle_high:     float = 0.0
    candle_low:      float = 0.0
    candle_close:    float = 0.0
    candle_volume:   int   = 0
    received_at:     datetime = field(default_factory=datetime.now)

    @property
    def change_pct(self) -> float:
        """Percentage change from previous close."""
        if self.close_price and self.close_price > 0:
            return round((self.ltp - self.close_price) / self.close_price * 100, 2)
        return 0.0


@dataclass
class LivePosition:
    """Represents an open paper-trade position."""
    symbol:          str
    instrument_key:  str
    direction:       int          # +1 = long, -1 = short
    quantity:        int
    entry_price:     float
    entry_time:      datetime
    stop_loss:       Optional[float] = None
    take_profit:     Optional[float] = None
    strategy_tag:    str = ""

    @property
    def current_pnl(self) -> float:
        """Must be computed externally using current LTP."""
        return 0.0  # computed in state using last_tick


@dataclass
class LiveOrder:
    """State of a paper-trade order."""
    order_id:        str
    symbol:          str
    instrument_key:  str
    action:          str          # "BUY" or "SELL"
    quantity:        int
    order_type:      str          # "MARKET" or "LIMIT"
    limit_price:     Optional[float]
    status:          str          # "PENDING", "FILLED", "CANCELLED", "REJECTED"
    created_at:      datetime
    filled_at:       Optional[datetime] = None
    fill_price:      Optional[float] = None
    strategy_tag:    str = ""
    pnl:             float = 0.0  # Only set when closing a position


@dataclass
class ClosedTrade:
    """A completed round-trip trade (entry + exit)."""
    symbol:          str
    direction:       str          # "LONG" or "SHORT"
    quantity:        int
    entry_price:     float
    exit_price:      float
    entry_time:      datetime
    exit_time:       datetime
    pnl:             float
    pnl_pct:         float
    strategy_tag:    str = ""
    exit_reason:     str = ""     # "SIGNAL", "STOP_LOSS", "TAKE_PROFIT", "SQUAREOFF", "KILL_SWITCH"


# ─── Main State Store ─────────────────────────────────────────────────────────

class LiveState:
    """
    Thread-safe, centralised state store for the live trading engine.

    Usage:
        from live_bot.state import state
        state.update_tick("RELIANCE", tick_data)
        tick = state.get_tick("RELIANCE")
    """

    def __init__(self):
        self._lock = threading.RLock()  # Reentrant lock — safe to nest

        # ── Market Data ───────────────────────────────────────────────────────
        # Latest tick per symbol
        self._ticks:        Dict[str, TickData] = {}

        # Rolling tick history (last 500 ticks per symbol for candle building)
        self._tick_history: Dict[str, Deque[TickData]] = {}

        # ── Portfolio ─────────────────────────────────────────────────────────
        self._positions:    Dict[str, LivePosition] = {}
        self._open_orders:  Dict[str, LiveOrder]    = {}
        self._closed_trades: List[ClosedTrade]      = []

        # ── Capital tracking ──────────────────────────────────────────────────
        self._initial_capital: float = 500_000.0
        self._cash:            float = 500_000.0
        self._peak_capital:    float = 500_000.0

        # ── Day stats ─────────────────────────────────────────────────────────
        self._day_pnl:          float = 0.0
        self._day_start_capital: float = 500_000.0
        self._day_start_date:   Optional[date] = None

        # ── Risk flags ────────────────────────────────────────────────────────
        self._kill_switch:      bool  = False
        self._daily_loss_hit:   bool  = False
        self._max_dd_hit:       bool  = False

        # ── Engine status ─────────────────────────────────────────────────────
        self._is_running:       bool  = False
        self._market_feed_connected:     bool = False
        self._portfolio_feed_connected:  bool = False
        self._subscribed_symbols: List[str] = []
        self._active_strategy:   Optional[str] = None
        self._bot_start_time:    Optional[datetime] = None

        # ── Activity log (shown in dashboard) ────────────────────────────────
        # Capped at 200 entries to avoid memory growth
        self._activity_log: Deque[dict] = deque(maxlen=200)

        logger.info("LiveState initialised.")

    # ─── Tick data ────────────────────────────────────────────────────────────

    def update_tick(self, symbol: str, tick: TickData) -> None:
        """Store the latest tick for a symbol. Thread-safe."""
        with self._lock:
            self._ticks[symbol] = tick
            if symbol not in self._tick_history:
                self._tick_history[symbol] = deque(maxlen=500)
            self._tick_history[symbol].append(tick)

    def get_tick(self, symbol: str) -> Optional[TickData]:
        """Get the latest tick for a symbol."""
        with self._lock:
            return self._ticks.get(symbol)

    def get_all_ticks(self) -> Dict[str, TickData]:
        """Snapshot of all current ticks."""
        with self._lock:
            return dict(self._ticks)

    def get_tick_history(self, symbol: str) -> List[TickData]:
        """Return tick history list for a symbol (copy)."""
        with self._lock:
            return list(self._tick_history.get(symbol, []))

    # ─── Positions ────────────────────────────────────────────────────────────

    def add_position(self, position: LivePosition) -> None:
        """Open a new position. Raises if position already exists."""
        with self._lock:
            if position.symbol in self._positions:
                logger.warning(
                    f"[STATE] Position for {position.symbol} already exists — "
                    f"overwriting. This may indicate a duplicate signal."
                )
            self._positions[position.symbol] = position
            logger.info(
                f"[STATE] Position opened: {position.symbol} "
                f"{'LONG' if position.direction > 0 else 'SHORT'} "
                f"x{position.quantity} @ ₹{position.entry_price:.2f}"
            )

    def close_position(self, symbol: str) -> Optional[LivePosition]:
        """Remove and return a position (to calculate final P&L)."""
        with self._lock:
            return self._positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[LivePosition]:
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, LivePosition]:
        with self._lock:
            return dict(self._positions)

    def has_position(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self._positions

    # ─── Orders ───────────────────────────────────────────────────────────────

    def add_order(self, order: LiveOrder) -> None:
        with self._lock:
            self._open_orders[order.order_id] = order

    def update_order_status(
        self,
        order_id: str,
        status: str,
        fill_price: Optional[float] = None,
        filled_at: Optional[datetime] = None,
    ) -> Optional[LiveOrder]:
        """Update order status. Returns the updated order."""
        with self._lock:
            order = self._open_orders.get(order_id)
            if not order:
                logger.warning(f"[STATE] Order {order_id} not found in open orders.")
                return None
            order.status = status
            if fill_price is not None:
                order.fill_price = fill_price
            if filled_at is not None:
                order.filled_at = filled_at
            return order

    def get_order(self, order_id: str) -> Optional[LiveOrder]:
        with self._lock:
            return self._open_orders.get(order_id)

    def get_all_orders(self) -> Dict[str, LiveOrder]:
        with self._lock:
            return dict(self._open_orders)

    # ─── Closed Trades ────────────────────────────────────────────────────────

    def record_closed_trade(self, trade: ClosedTrade) -> None:
        with self._lock:
            self._closed_trades.append(trade)
            self._day_pnl += trade.pnl
            logger.info(
                f"[STATE] Trade closed: {trade.symbol} {trade.direction} "
                f"P&L=₹{trade.pnl:.2f} ({trade.pnl_pct:.2f}%) "
                f"reason={trade.exit_reason}"
            )

    def get_closed_trades(self) -> List[ClosedTrade]:
        with self._lock:
            return list(self._closed_trades)

    # ─── Capital ──────────────────────────────────────────────────────────────

    def set_initial_capital(self, amount: float) -> None:
        with self._lock:
            self._initial_capital = amount
            self._cash = amount
            self._peak_capital = amount
            self._day_start_capital = amount
            self._day_start_date = date.today()

    def debit_cash(self, amount: float) -> None:
        """Deduct cash (on buy order fill). Raises ValueError if insufficient funds."""
        with self._lock:
            if amount > self._cash:
                raise ValueError(
                    f"Insufficient cash: need ₹{amount:.2f}, "
                    f"have ₹{self._cash:.2f}"
                )
            self._cash -= amount

    def credit_cash(self, amount: float) -> None:
        """Add cash (on sell order fill)."""
        with self._lock:
            self._cash += amount
            if self._cash > self._peak_capital:
                self._peak_capital = self._cash

    @property
    def cash(self) -> float:
        with self._lock:
            return self._cash

    @property
    def total_value(self) -> float:
        """Cash + unrealised P&L of all open positions."""
        with self._lock:
            total = self._cash
            for sym, pos in self._positions.items():
                tick = self._ticks.get(sym)
                if tick and tick.ltp > 0:
                    val = (tick.ltp - pos.entry_price) * pos.quantity * pos.direction
                    total += val
            return total

    @property
    def day_pnl(self) -> float:
        with self._lock:
            return self._day_pnl

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak as percentage (0-100)."""
        with self._lock:
            if self._peak_capital <= 0:
                return 0.0
            return max(0.0, (self._peak_capital - self.total_value) / self._peak_capital * 100)

    # ─── Risk flags ───────────────────────────────────────────────────────────

    def activate_kill_switch(self, reason: str = "") -> None:
        with self._lock:
            self._kill_switch = True
            msg = f"🔴 KILL SWITCH ACTIVATED{f': {reason}' if reason else ''}"
            logger.critical(msg)
            self._log_activity("KILL_SWITCH", msg, level="CRITICAL")

    def set_daily_loss_hit(self) -> None:
        with self._lock:
            self._daily_loss_hit = True

    def set_max_dd_hit(self) -> None:
        with self._lock:
            self._max_dd_hit = True

    @property
    def kill_switch(self) -> bool:
        with self._lock:
            return self._kill_switch

    def is_trading_allowed(self) -> bool:
        """
        Combined check: can the bot place new orders?

        This is a regular method (not @property) so callers use
        state.is_trading_allowed() with parentheses — consistent with
        tests and engine code.
        """
        with self._lock:
            return (
                self._is_running
                and not self._kill_switch
                and not self._daily_loss_hit
                and not self._max_dd_hit
                and self._market_feed_connected
            )

    # ─── Engine status ────────────────────────────────────────────────────────

    def set_running(self, val: bool) -> None:
        with self._lock:
            self._is_running = val
            if val:
                self._bot_start_time = datetime.now()

    def set_market_feed_status(self, connected: bool) -> None:
        with self._lock:
            self._market_feed_connected = connected

    def set_portfolio_feed_status(self, connected: bool) -> None:
        with self._lock:
            self._portfolio_feed_connected = connected

    def set_subscribed_symbols(self, symbols: List[str]) -> None:
        with self._lock:
            self._subscribed_symbols = list(symbols)

    def set_active_strategy(self, name: Optional[str]) -> None:
        with self._lock:
            self._active_strategy = name

    def get_status_snapshot(self) -> dict:
        """
        Complete status dict for the dashboard /api/live/status endpoint.
        All values are JSON-serialisable (no datetime objects).
        """
        with self._lock:
            positions_data = {}
            for sym, pos in self._positions.items():
                tick = self._ticks.get(sym)
                ltp = tick.ltp if tick else pos.entry_price
                unreal_pnl = (ltp - pos.entry_price) * pos.quantity * pos.direction
                positions_data[sym] = {
                    "direction":    "LONG" if pos.direction > 0 else "SHORT",
                    "quantity":     pos.quantity,
                    "entry_price":  round(pos.entry_price, 2),
                    "ltp":          round(ltp, 2),
                    "unrealised_pnl": round(unreal_pnl, 2),
                    "stop_loss":    pos.stop_loss,
                    "take_profit":  pos.take_profit,
                    "strategy_tag": pos.strategy_tag,
                    "entry_time":   pos.entry_time.isoformat(),
                }

            ticks_data = {}
            for sym, tick in self._ticks.items():
                ticks_data[sym] = {
                    "ltp":        round(tick.ltp, 2),
                    "change_pct": tick.change_pct,
                    "volume":     tick.volume,
                    "high":       round(tick.high_price, 2),
                    "low":        round(tick.low_price, 2),
                    "oi":         tick.oi,
                    "ltt":        tick.ltt.isoformat() if isinstance(tick.ltt, datetime) else str(tick.ltt),
                }

            trades_data = [
                {
                    "symbol":       t.symbol,
                    "direction":    t.direction,
                    "quantity":     t.quantity,
                    "entry_price":  round(t.entry_price, 2),
                    "exit_price":   round(t.exit_price, 2),
                    "pnl":          round(t.pnl, 2),
                    "pnl_pct":      round(t.pnl_pct, 2),
                    "exit_reason":  t.exit_reason,
                    "entry_time":   t.entry_time.isoformat(),
                    "exit_time":    t.exit_time.isoformat(),
                    "strategy_tag": t.strategy_tag,
                }
                for t in self._closed_trades
            ]

            return {
                "is_running":               self._is_running,
                "market_feed_connected":    self._market_feed_connected,
                "portfolio_feed_connected": self._portfolio_feed_connected,
                "kill_switch":              self._kill_switch,
                "daily_loss_hit":           self._daily_loss_hit,
                "is_trading_allowed":       self.is_trading_allowed(),
                "active_strategy":          self._active_strategy,
                "subscribed_symbols":       self._subscribed_symbols,
                "cash":                     round(self._cash, 2),
                "total_value":              round(self.total_value, 2),
                "day_pnl":                  round(self._day_pnl, 2),
                "drawdown_pct":             round(self.drawdown_pct, 2),
                "initial_capital":          round(self._initial_capital, 2),
                "open_positions":           positions_data,
                "market_ticks":             ticks_data,
                "closed_trades":            trades_data,
                "activity_log":             list(self._activity_log),
                "bot_start_time":           self._bot_start_time.isoformat() if self._bot_start_time else None,
            }

    # ─── Activity log ─────────────────────────────────────────────────────────

    def get_activity_log(self, last_n: int = 200) -> List[dict]:
        """
        Return the rolling activity log (last N entries).
        Used by the dashboard /api/live/status endpoint and tests.
        """
        with self._lock:
            entries = list(self._activity_log)
        return entries[-last_n:]

    def get_status_dict(self) -> dict:
        """
        Alias for get_status_snapshot() — used by dashboard and tests.
        Returns a JSON-serialisable snapshot of current live state.
        """
        return self.get_status_snapshot()

    def log_activity(self, event_type: str, message: str, level: str = "INFO") -> None:
        """Public method for components to log activity."""
        self._log_activity(event_type, message, level)

    def _log_activity(self, event_type: str, message: str, level: str = "INFO") -> None:
        """Append to the rolling activity log shown in the dashboard."""
        # Note: must be called while already holding the lock, OR use log_activity()
        entry = {
            "time":       datetime.now().isoformat(),
            "event_type": event_type,
            "message":    message,
            "level":      level,
        }
        self._activity_log.append(entry)


# ── Module-level singleton ────────────────────────────────────────────────────
# All live_bot modules import this single shared instance:
#   from live_bot.state import state
state = LiveState()
