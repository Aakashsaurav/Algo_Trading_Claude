"""
live_bot/candle_builder.py
---------------------------
Assembles raw tick data (LTP stream) into OHLCV candles.

WHY THIS IS NEEDED:
    The Upstox MarketDataStreamerV3 "full" mode provides a live 1-min candle
    inside each tick message (marketOHLC.I1). However, that candle object may
    not always be populated for every tick, and the field structure can vary.

    This module acts as a DUAL-SOURCE candle builder:
        1. PRIMARY:  Use the live OHLC values from the "full" mode feed directly
                     (candle_open, candle_high, candle_low, candle_close).
        2. FALLBACK: If the feed candle is absent or zero, build candles from
                     raw LTP ticks ourselves — grouping by minute boundary.

    Both methods produce a pandas DataFrame with columns:
        [datetime, open, high, low, close, volume]

    The strategy's generate_signals() and on_bar() consume these DataFrames.

THREAD SAFETY:
    Each symbol has its own CandleBuilder instance. The strategy loop
    reads completed candles. The feed thread writes ticks. We protect
    the shared buffer with a threading.Lock.
"""

import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from live_bot.state import TickData

logger = logging.getLogger(__name__)

# India Standard Time offset (+05:30)
IST = timezone(timedelta(hours=5, minutes=30))


class MinuteCandle:
    """
    Accumulates ticks for a single 1-minute OHLCV bar.

    Attributes:
        minute_start : The datetime rounded down to the minute (IST).
        open, high, low, close, volume : OHLCV values.
        tick_count   : How many raw ticks contributed to this candle.
    """

    __slots__ = ["minute_start", "open", "high", "low", "close", "volume", "tick_count"]

    def __init__(self, minute_start: datetime, first_price: float, first_volume: int):
        self.minute_start = minute_start
        self.open         = first_price
        self.high         = first_price
        self.low          = first_price
        self.close        = first_price
        self.volume       = first_volume
        self.tick_count   = 1

    def update(self, price: float, volume_delta: int) -> None:
        """Update candle with a new tick."""
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close   = price
        self.volume  += max(0, volume_delta)  # guard against negative delta
        self.tick_count += 1

    def to_dict(self) -> dict:
        return {
            "datetime": self.minute_start,
            "open":     round(self.open,  2),
            "high":     round(self.high,  2),
            "low":      round(self.low,   2),
            "close":    round(self.close, 2),
            "volume":   self.volume,
        }


class CandleBuilder:
    """
    Per-symbol OHLCV candle builder.

    Usage:
        builder = CandleBuilder("RELIANCE", history_df)
        builder.on_tick(tick_data)           # call from feed thread
        df = builder.get_candles_df()        # call from strategy thread
        completed = builder.get_new_candles() # only newly completed candles
    """

    def __init__(
        self,
        symbol: str,
        seed_df: Optional[pd.DataFrame] = None,
        max_history_bars: int = 500,
    ):
        """
        Args:
            symbol           : Trading symbol (for logging only).
            seed_df          : Historical OHLCV DataFrame to pre-populate history.
                               Must have columns [datetime, open, high, low, close, volume].
            max_history_bars : Maximum candles to keep in memory.
        """
        self.symbol            = symbol
        self.max_history_bars  = max_history_bars
        self._lock             = threading.Lock()

        # Completed candles list (seed + live)
        self._completed: List[dict] = []

        # The candle currently being built (not yet completed)
        self._current: Optional[MinuteCandle] = None

        # Track last seen cumulative volume to compute per-tick volume delta
        self._last_volume: int = 0

        # Track newly completed candles since last get_new_candles() call
        self._new_candle_idx: int = 0

        # Seed with historical data if provided
        if seed_df is not None and not seed_df.empty:
            self._seed_from_df(seed_df)

        logger.info(
            f"[CandleBuilder:{symbol}] Initialised with "
            f"{len(self._completed)} historical candles."
        )

    def _seed_from_df(self, df: pd.DataFrame) -> None:
        """
        Pre-populate completed candles from a historical DataFrame.
        Validates column presence and skips malformed rows.
        """
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            logger.warning(
                f"[CandleBuilder:{self.symbol}] Seed df missing columns: {missing}. "
                "Skipping seed."
            )
            return

        # Ensure datetime index or column
        if "datetime" in df.columns:
            dt_col = df["datetime"]
        elif isinstance(df.index, pd.DatetimeIndex):
            dt_col = df.index.to_series()
        else:
            logger.warning(
                f"[CandleBuilder:{self.symbol}] No datetime column/index found. Skipping seed."
            )
            return

        # Reset index so iloc-based iteration is safe regardless of original index type
        df = df.reset_index(drop=True)
        if "datetime" in df.columns:
            dt_col = df["datetime"]
        elif isinstance(df.index, pd.DatetimeIndex):
            dt_col = df.index.to_series().reset_index(drop=True)
        else:
            logger.warning(
                f"[CandleBuilder:{self.symbol}] No datetime column/index found after reset. Skipping seed."
            )
            return

        count = 0
        for i in range(len(df)):
            try:
                row = df.iloc[i]
                dt = dt_col.iloc[i]
                if pd.isna(dt):
                    continue
                self._completed.append({
                    "datetime": pd.Timestamp(dt),
                    "open":     float(row["open"]),
                    "high":     float(row["high"]),
                    "low":      float(row["low"]),
                    "close":    float(row["close"]),
                    "volume":   int(row.get("volume", 0)),
                })
                count += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"[CandleBuilder:{self.symbol}] Skipped malformed row {i}: {e}")

        # Keep only the most recent N bars
        if len(self._completed) > self.max_history_bars:
            self._completed = self._completed[-self.max_history_bars:]

        # New candle pointer starts after seed data
        self._new_candle_idx = len(self._completed)
        logger.info(f"[CandleBuilder:{self.symbol}] Seeded {count} historical candles.")

    def on_tick(self, tick: TickData) -> Optional[MinuteCandle]:
        """
        Process one incoming tick.

        Strategy:
            - If the tick has non-zero candle OHLC from the feed → use those
              values directly (preferred — data comes from Upstox).
            - Otherwise → build from raw LTP.

        Returns the COMPLETED candle if a new minute started, else None.
        """
        with self._lock:
            # Determine which minute this tick belongs to
            if isinstance(tick.ltt, datetime):
                tick_dt = tick.ltt
            else:
                tick_dt = datetime.now(tz=IST)

            # Round down to minute
            minute_start = tick_dt.replace(second=0, microsecond=0)

            # Compute volume delta (Upstox sends cumulative day volume)
            vol_delta = max(0, tick.volume - self._last_volume)
            self._last_volume = tick.volume

            # Use feed-provided candle OHLC if available and non-zero
            use_feed_candle = (
                tick.candle_open > 0
                and tick.candle_high > 0
                and tick.candle_low > 0
                and tick.candle_close > 0
            )

            completed_candle = None

            if self._current is None:
                # First tick ever — start a new candle
                price = tick.candle_close if use_feed_candle else tick.ltp
                self._current = MinuteCandle(minute_start, price, vol_delta)

            elif minute_start > self._current.minute_start:
                # New minute started — complete the old candle
                completed_candle = self._current
                self._completed.append(completed_candle.to_dict())

                # Trim to max history
                if len(self._completed) > self.max_history_bars:
                    self._completed = self._completed[-self.max_history_bars:]

                logger.debug(
                    f"[CandleBuilder:{self.symbol}] Candle completed: "
                    f"{completed_candle.minute_start} "
                    f"O={completed_candle.open} H={completed_candle.high} "
                    f"L={completed_candle.low} C={completed_candle.close} "
                    f"V={completed_candle.volume}"
                )

                # Start new candle for the new minute
                price = tick.candle_close if use_feed_candle else tick.ltp
                self._current = MinuteCandle(minute_start, price, vol_delta)

            else:
                # Same minute — update the current candle
                if use_feed_candle:
                    # Prefer feed OHLC for accuracy
                    self._current.high  = max(self._current.high, tick.candle_high)
                    self._current.low   = min(self._current.low, tick.candle_low)
                    self._current.close = tick.candle_close
                else:
                    self._current.update(tick.ltp, vol_delta)

            return completed_candle

    def get_candles_df(self) -> pd.DataFrame:
        """
        Return all candles (historical + live completed) as a DataFrame.

        IMPORTANT: Does NOT include the currently-forming candle — only
        completed candles are returned. This prevents look-ahead bias.

        Returns:
            pd.DataFrame with columns: [datetime, open, high, low, close, volume]
            Empty DataFrame if no candles yet.
        """
        with self._lock:
            if not self._completed:
                return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

            df = pd.DataFrame(self._completed)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            return df

    def get_new_candles(self) -> List[dict]:
        """
        Return only candles completed SINCE the last call to this method.
        Used by the strategy loop to process only new bars.
        """
        with self._lock:
            new = self._completed[self._new_candle_idx:]
            self._new_candle_idx = len(self._completed)
            return list(new)

    def get_current_bar(self) -> Optional[dict]:
        """Return the currently forming (incomplete) candle, or None."""
        with self._lock:
            if self._current is None:
                return None
            return self._current.to_dict()

    def bar_count(self) -> int:
        """Number of completed candles available."""
        with self._lock:
            return len(self._completed)


class CandleRegistry:
    """
    Registry that owns one CandleBuilder per symbol.
    The market feed calls registry.on_tick(symbol, tick) for every message.
    The strategy calls registry.get_df(symbol) to read candle history.
    """

    def __init__(self):
        self._builders: Dict[str, CandleBuilder] = {}
        self._lock = threading.Lock()

    def register(
        self,
        symbol: str,
        seed_df: Optional[pd.DataFrame] = None,
        max_history_bars: int = 500,
    ) -> CandleBuilder:
        """
        Register a symbol and optionally seed it with historical data.
        Safe to call multiple times for the same symbol.
        """
        with self._lock:
            if symbol not in self._builders:
                self._builders[symbol] = CandleBuilder(symbol, seed_df, max_history_bars)
                logger.info(f"[CandleRegistry] Registered symbol: {symbol}")
            return self._builders[symbol]

    def on_tick(self, symbol: str, tick: TickData) -> Optional[MinuteCandle]:
        """
        Route a tick to the correct CandleBuilder.
        If the symbol is not registered, it is auto-registered (no seed).
        """
        with self._lock:
            if symbol not in self._builders:
                logger.warning(
                    f"[CandleRegistry] Symbol {symbol} not pre-registered. "
                    "Auto-registering without seed data."
                )
                self._builders[symbol] = CandleBuilder(symbol)
        return self._builders[symbol].on_tick(tick)

    def get_df(self, symbol: str) -> pd.DataFrame:
        """Return the full OHLCV DataFrame for a symbol."""
        with self._lock:
            builder = self._builders.get(symbol)
        if builder is None:
            logger.warning(f"[CandleRegistry] {symbol} not registered — returning empty df.")
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        return builder.get_candles_df()

    def get_new_candles(self, symbol: str) -> List[dict]:
        """Return newly completed candles since last check for a symbol."""
        with self._lock:
            builder = self._builders.get(symbol)
        if builder is None:
            return []
        return builder.get_new_candles()

    def bar_counts(self) -> Dict[str, int]:
        """Return {symbol: bar_count} for all registered symbols."""
        with self._lock:
            return {sym: b.bar_count() for sym, b in self._builders.items()}

    def get_symbols(self) -> List[str]:
        with self._lock:
            return list(self._builders.keys())


# Module-level registry singleton
candle_registry = CandleRegistry()
