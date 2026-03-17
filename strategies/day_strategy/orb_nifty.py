"""
strategies/day_strategy/orb_nifty.py
======================================
Opening Range Breakout (ORB) — NIFTY 50 Index
Institutional-grade intraday strategy module.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Opening Range (OR):
    OR_high = max(high)  of 1-min bars from 09:15 to 09:29 (inclusive)
    OR_low  = min(low)   of 1-min bars from 09:15 to 09:29 (inclusive)

LONG entry (max 1 per day):
    ① Time >= 09:30 IST
    ② Candle is green   → close > open
    ③ Close > OR_high   → breakout above the range
    ④ Close > VWAP      → price is above volume-weighted average (trend filter)
    Stop-loss   : open of the signal candle (green bar that triggered)
    Trailing SL : advance stop by abs(prev_open − prev_close) / 5 each bar

SHORT entry (max 1 per day):
    ① Time >= 09:30 IST
    ② Candle is red     → close < open
    ③ Close < OR_low    → breakdown below the range
    ④ Close < VWAP      → price is below volume-weighted average (trend filter)
    Stop-loss   : open of the signal candle (red bar that triggered)
    Trailing SL : advance stop by abs(prev_open − prev_close) / 5 each bar

EXIT rules:
    • Stop-loss hit (fixed, set at entry to signal candle's open)
    • Trailing stop advances each bar (never retreats against the trade)
    • Hard squareoff at 15:15 IST — open of the 15:15 bar
    • No take-profit target (trend-following; let winners run)

Daily limits:
    • Maximum 1 LONG per day
    • Maximum 1 SHORT per day
    • Once a leg is used, it cannot be re-entered on the same day

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULAR DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each sub-component is a standalone class. To refine the strategy:

  Change OR window       → edit OR_END_TIME constant or or_window_minutes param
  Change VWAP filter     → set vwap_filter=False or override _check_vwap()
  Add RSI / ATR filter   → add a column in ORBIndicators.compute_indicators()
                           and reference it in ORBNiftyStrategy._apply_daily_limits()
  Change trailing formula→ edit ORBTrailingStop.compute_increment()
  Change squareoff time  → edit SQUAREOFF_TIME constant

None of these changes touch the runner (run_orb_backtest.py) or the engine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NIFTY 50 SPOT TRADING NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NIFTY 50 spot (NSE_INDEX) is NOT directly tradeable. In production:
  • Use NIFTY Futures (FUTIDX) — most liquid, negligible basis in near-month
  • Use NIFTYBEES ETF (NSE_EQ)  — tradeable, tracks index closely
  • Use NIFTY Options (OPTIDX)  — strategy-specific

This module uses spot price as the research proxy (standard quant practice).
Adjust instrument_type in the runner for live deployment.

AUTHOR: AlgoDesk — Institutional Strategy Research
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base_strategy_github import BaseStrategy

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module-level constants — single place to change timing rules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OR_START_TIME    = dtime(9, 15)   # Opening Range window start (inclusive)
OR_END_TIME      = dtime(9, 29)   # Opening Range window end   (inclusive) → 15 min
SIGNAL_START     = dtime(9, 30)   # Earliest bar eligible for a signal
SQUAREOFF_TIME   = dtime(15, 15)  # Hard close at this time (fills at open of bar)
MARKET_CLOSE     = dtime(15, 30)  # Last bar of the session

MIN_OR_BARS      = 10             # Min bars required in OR window (data quality guard)
TRAILING_DIVISOR = 5.0            # Trail increment = |prev_open − prev_close| / this


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pure data containers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ORLevels:
    """
    Opening Range high/low for one trading session.

    Attributes
    ----------
    date     : str   — 'YYYY-MM-DD'
    or_high  : float — max(high) of 09:15–09:29 bars
    or_low   : float — min(low)  of 09:15–09:29 bars
    is_valid : bool  — False if OR window had fewer than MIN_OR_BARS bars
    bar_count: int   — actual number of bars in the OR window
    """
    date:      str
    or_high:   float
    or_low:    float
    is_valid:  bool = True
    bar_count: int  = 0

    def __repr__(self) -> str:
        return (
            f"ORLevels(date={self.date!r}, "
            f"high={self.or_high:.2f}, low={self.or_low:.2f}, "
            f"valid={self.is_valid}, bars={self.bar_count})"
        )


@dataclass
class DayTradeState:
    """
    Per-day entry-limit tracker.

    Prevents more than one long and one short per trading day.
    Instantiated fresh for each trading day inside the signal scanner.
    """
    date:        str
    long_taken:  bool = False
    short_taken: bool = False

    @property
    def can_go_long(self) -> bool:
        return not self.long_taken

    @property
    def can_go_short(self) -> bool:
        return not self.short_taken

    @property
    def both_legs_used(self) -> bool:
        return self.long_taken and self.short_taken


@dataclass
class SignalBar:
    """
    Complete information about one signal bar.
    Used by the custom event loop to open positions.
    """
    timestamp:  pd.Timestamp
    bar_index:  int
    direction:  int           # +1 LONG / -1 SHORT
    stop_loss:  float         # open of signal candle
    signal_tag: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sub-component 1: Indicator computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ORBIndicators:
    """
    Stateless indicator calculator for the ORB strategy.

    All methods are static — no internal state. Thread-safe by design.
    """

    @staticmethod
    def compute_or_levels(df: pd.DataFrame) -> Dict[str, ORLevels]:
        """
        Compute OR high and OR low for every trading day in df.

        Groups by calendar date and finds the maximum high / minimum low
        within the OR window (09:15 to 09:29 inclusive).

        Parameters
        ----------
        df : pd.DataFrame
            1-minute OHLCV with IST-aware DatetimeIndex.

        Returns
        -------
        dict  {date_str: ORLevels}
        """
        result: Dict[str, ORLevels] = {}

        # Vectorised time / date extraction
        bar_times = df.index.time
        bar_dates = df.index.date

        for dt in np.unique(bar_dates):
            date_str = str(dt)

            # Select only OR-window bars for this day
            day_mask = bar_dates == dt
            or_mask  = (
                (bar_times >= OR_START_TIME) &
                (bar_times <= OR_END_TIME)
            )
            or_bars = df.iloc[day_mask & or_mask]

            if len(or_bars) < MIN_OR_BARS:
                result[date_str] = ORLevels(
                    date      = date_str,
                    or_high   = float("nan"),
                    or_low    = float("nan"),
                    is_valid  = False,
                    bar_count = len(or_bars),
                )
                logger.debug(
                    f"ORB {date_str}: only {len(or_bars)} OR bars "
                    f"(need {MIN_OR_BARS}) — day skipped"
                )
                continue

            result[date_str] = ORLevels(
                date      = date_str,
                or_high   = float(or_bars["high"].max()),
                or_low    = float(or_bars["low"].min()),
                is_valid  = True,
                bar_count = len(or_bars),
            )

        logger.info(
            f"ORBIndicators: {len(result)} days processed, "
            f"{sum(1 for v in result.values() if v.is_valid)} valid OR days"
        )
        return result

    @staticmethod
    def compute_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Compute intraday VWAP reset at the start of each calendar day.

        VWAP = Σ(typical_price × volume) / Σ(volume)
        where typical_price = (high + low + close) / 3

        The cumulative sum resets at midnight so each day starts fresh.
        On the first bar of the day, VWAP equals the typical price of
        that bar (Σvolume = volume of bar 0).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.Series  (name='vwap', same index as df)
        """
        typical   = (df["high"] + df["low"] + df["close"]) / 3.0
        tp_vol    = typical * df["volume"]
        bar_dates = pd.Series(df.index.date, index=df.index)

        cum_tp_vol = tp_vol.groupby(bar_dates).cumsum()
        cum_vol    = df["volume"].groupby(bar_dates).cumsum()

        vwap = (cum_tp_vol / cum_vol.replace(0, np.nan)).rename("vwap")
        return vwap

    @staticmethod
    def broadcast_or_levels(
        df:        pd.DataFrame,
        or_levels: Dict[str, ORLevels],
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Broadcast per-day OR levels to a full-length Series aligned to df.

        Every bar in the day gets the same OR high/low for that day.
        Days with invalid OR levels get NaN (no signal is possible).

        Parameters
        ----------
        df        : pd.DataFrame
        or_levels : dict from compute_or_levels()

        Returns
        -------
        (or_high, or_low, or_valid)
            Three pd.Series of the same length as df.
            or_valid is boolean: True where OR levels are usable.
        """
        n          = len(df)
        bar_dates  = df.index.date
        high_arr   = np.full(n, np.nan, dtype=float)
        low_arr    = np.full(n, np.nan, dtype=float)
        valid_arr  = np.zeros(n, dtype=bool)

        for i, dt in enumerate(bar_dates):
            lvl = or_levels.get(str(dt))
            if lvl and lvl.is_valid:
                high_arr[i]  = lvl.or_high
                low_arr[i]   = lvl.or_low
                valid_arr[i] = True

        return (
            pd.Series(high_arr,  index=df.index, name="or_high"),
            pd.Series(low_arr,   index=df.index, name="or_low"),
            pd.Series(valid_arr, index=df.index, name="or_valid"),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sub-component 2: Trailing stop increment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ORBTrailingStop:
    """
    Computes the per-bar trailing stop increment.

    Formula (per the strategy specification):
        increment[i] = abs(open[i-1] − close[i-1]) / TRAILING_DIVISOR

    The increment is always non-negative.
    The event loop applies this increment directionally:
        LONG  → new_sl = current_sl + increment  (if new_sl < current_price)
        SHORT → new_sl = current_sl − increment  (if new_sl > current_price)

    The stop NEVER retreats against the trade.
    """

    @staticmethod
    def compute_increment(df: pd.DataFrame, divisor: float = TRAILING_DIVISOR) -> pd.Series:
        """
        Compute trailing stop increment for every bar.

        Parameters
        ----------
        df      : pd.DataFrame  — OHLCV
        divisor : float         — trail_increment = body / divisor

        Returns
        -------
        pd.Series  (name='trail_increment', first value = 0.0)
        """
        prev_body = (df["open"] - df["close"]).abs().shift(1)
        increment = (prev_body / divisor).fillna(0.0)
        return increment.rename("trail_increment")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main strategy class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ORBNiftyStrategy(BaseStrategy):
    """
    Opening Range Breakout strategy for NIFTY 50 Index.

    Implements the vectorised BaseStrategy interface:
    the BacktestEngine calls generate_signals(df) → df with 'signal' column.

    The custom event loop (in run_orb_backtest.py) reads additional
    columns ('signal_sl', 'trail_increment') to enforce per-trade
    stop-loss and trailing stop rules.

    Parameters
    ----------
    or_window_minutes : int
        Length of the Opening Range window in minutes starting at 09:15.
        Default 15 → window is 09:15 to 09:29 inclusive.
    vwap_filter : bool
        Require close > VWAP for longs and close < VWAP for shorts.
        Disabling this increases signal frequency at the cost of more
        false breakouts. Default True.
    trailing_divisor : float
        Divisor for the trailing stop increment formula.
        trail_inc = abs(prev_open − prev_close) / trailing_divisor.
        Larger divisor → smaller increments → tighter trailing stop.
        Default 5.0.
    min_body_pct : float
        Minimum candle body as a percentage of close price.
        Filters out near-doji candles that produce false breakouts.
        0.0 = disabled. Default 0.0.

    Examples
    --------
    >>> strategy = ORBNiftyStrategy(vwap_filter=True, trailing_divisor=5.0)
    >>> signals_df = strategy.generate_signals(df_1min)
    >>> n_long  = (signals_df['signal'] ==  1).sum()
    >>> n_short = (signals_df['signal'] == -1).sum()
    """

    # ── Dashboard / registry metadata ─────────────────────────────────────────
    PARAM_SCHEMA = [
        {"name": "or_window_minutes", "type": "int",   "default": 15,  "min": 5,   "max": 60},
        {"name": "vwap_filter",       "type": "bool",  "default": True},
        {"name": "trailing_divisor",  "type": "float", "default": 5.0, "min": 1.0, "max": 20.0},
        {"name": "min_body_pct",      "type": "float", "default": 0.0, "min": 0.0, "max": 2.0},
    ]
    DESCRIPTION = (
        "Opening Range Breakout (ORB) — Long on green candle crossing OR_high + VWAP; "
        "Short on red candle crossing OR_low + below VWAP. "
        "Max 1 trade per leg per day. Hard squareoff at 15:15 IST."
    )
    CATEGORY = "Intraday Breakout"

    def __init__(
        self,
        or_window_minutes: int   = 15,
        vwap_filter:       bool  = True,
        trailing_divisor:  float = TRAILING_DIVISOR,
        min_body_pct:      float = 0.0,
    ) -> None:
        super().__init__(
            name        = "ORB NIFTY 50",
            description = self.DESCRIPTION,
        )
        self.or_window_minutes = int(or_window_minutes)
        self.vwap_filter       = bool(vwap_filter)
        self.trailing_divisor  = float(trailing_divisor)
        self.min_body_pct      = float(min_body_pct)

        # Derive the OR end time from the window parameter
        # OR_START = 09:15, so end minute = 14 + window
        end_minute = 14 + self.or_window_minutes
        end_hour   = 9 + end_minute // 60
        end_minute = end_minute % 60
        self._or_end_time = dtime(end_hour, end_minute)

    # ─────────────────────────────────────────────────────────────────────────
    # Primary API — called by BacktestEngine
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators and generate entry signals.

        Calls each sub-component in order, then enforces daily limits
        via a chronological scan to avoid look-ahead bias.

        Output columns added to df:
            or_high          : OR high for this day (NaN = invalid day)
            or_low           : OR low  for this day (NaN = invalid day)
            or_valid         : bool — True where OR levels are usable
            vwap             : intraday VWAP (day-reset)
            trail_increment  : |prev_open−prev_close| / trailing_divisor
            signal           : +1 LONG / -1 SHORT / 0 no-action
            signal_sl        : stop-loss price (open of signal candle)
            signal_tag       : 'ORB_LONG' / 'ORB_SHORT' / ''

        Parameters
        ----------
        df : pd.DataFrame
            1-minute OHLCV. Must have columns: open, high, low, close, volume.
            Must have a timezone-aware DatetimeIndex (IST recommended).

        Returns
        -------
        pd.DataFrame
        """
        # ── Validate and copy ──────────────────────────────────────────────
        df = self._validate_and_prepare(df)

        # ── Step 1: Opening Range levels ───────────────────────────────────
        ind      = ORBIndicators()
        or_lvls  = ind.compute_or_levels(df)
        or_h, or_l, or_v = ind.broadcast_or_levels(df, or_lvls)

        df["or_high"]  = or_h
        df["or_low"]   = or_l
        df["or_valid"] = or_v

        # ── Step 2: VWAP (day-reset) ───────────────────────────────────────
        df["vwap"] = ind.compute_vwap(df)

        # ── Step 3: Trailing stop increment ───────────────────────────────
        df["trail_increment"] = ORBTrailingStop.compute_increment(
            df, divisor=self.trailing_divisor
        )

        # ── Step 4: Time masks ─────────────────────────────────────────────
        bar_times       = df.index.time
        after_or_window = pd.Series(bar_times, index=df.index) >= SIGNAL_START
        before_squareoff= pd.Series(bar_times, index=df.index) <  SQUAREOFF_TIME
        tradeable       = after_or_window & before_squareoff

        # ── Step 5: Candle direction ───────────────────────────────────────
        is_green = df["close"] > df["open"]
        is_red   = df["close"] < df["open"]

        # Optional minimum body filter
        if self.min_body_pct > 0:
            body      = (df["close"] - df["open"]).abs()
            min_body  = df["close"] * (self.min_body_pct / 100.0)
            is_green  = is_green & (body >= min_body)
            is_red    = is_red   & (body >= min_body)

        # ── Step 6: Raw breakout conditions ───────────────────────────────
        long_break = (
            tradeable
            & df["or_valid"]
            & is_green
            & (df["close"] > df["or_high"])
        )
        short_break = (
            tradeable
            & df["or_valid"]
            & is_red
            & (df["close"] < df["or_low"])
        )

        # ── Step 7: VWAP filter (optional) ────────────────────────────────
        if self.vwap_filter:
            long_break  = long_break  & (df["close"] > df["vwap"])
            short_break = short_break & (df["close"] < df["vwap"])

        # ── Step 8: Initialise output columns ─────────────────────────────
        df["signal_sl"]  = np.nan
        df["signal_tag"] = ""
        # signal column already initialised to 0 by _validate_and_prepare

        # ── Step 9: Enforce daily limits (chronological scan) ──────────────
        self._apply_daily_limits(df, long_break, short_break)

        # ── Log summary ───────────────────────────────────────────────────
        n_long  = (df["signal"] ==  1).sum()
        n_short = (df["signal"] == -1).sum()
        n_days  = len(np.unique(df.index.date))
        logger.info(
            f"ORBNiftyStrategy: {len(df):,} bars | {n_days} days | "
            f"LONG signals={n_long} | SHORT signals={n_short} | "
            f"vwap_filter={self.vwap_filter} | "
            f"OR window={self.or_window_minutes}min"
        )

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_daily_limits(
        self,
        df:          pd.DataFrame,
        long_break:  pd.Series,
        short_break: pd.Series,
    ) -> None:
        """
        Chronological scan that allows only the FIRST qualifying signal
        per leg per day. Modifies df['signal'], df['signal_sl'],
        df['signal_tag'] in-place.

        Cannot be vectorised without look-ahead bias — the rule
        'fire on the FIRST qualifying bar' requires forward scanning.
        Per-day overhead is small: each day has at most ~285 tradeable
        bars, and we break early once both legs are used.

        Parameters
        ----------
        df          : DataFrame (modified in-place)
        long_break  : boolean Series — raw long condition per bar
        short_break : boolean Series — raw short condition per bar
        """
        bar_dates = df.index.date

        for dt in np.unique(bar_dates):
            day_mask = bar_dates == dt
            day_idx  = df.index[day_mask]

            state = DayTradeState(date=str(dt))

            for ts in day_idx:
                if state.both_legs_used:
                    break

                is_long_sig  = bool(long_break.loc[ts])  if not state.long_taken  else False
                is_short_sig = bool(short_break.loc[ts]) if not state.short_taken else False

                if is_long_sig:
                    df.at[ts, "signal"]    = 1
                    df.at[ts, "signal_sl"] = float(df.at[ts, "open"])
                    df.at[ts, "signal_tag"]= "ORB_LONG"
                    state.long_taken       = True
                    logger.debug(
                        f"ORB LONG signal: {ts} | "
                        f"close={df.at[ts,'close']:.2f} > "
                        f"or_high={df.at[ts,'or_high']:.2f} | "
                        f"SL={df.at[ts,'signal_sl']:.2f}"
                    )

                elif is_short_sig:
                    df.at[ts, "signal"]    = -1
                    df.at[ts, "signal_sl"] = float(df.at[ts, "open"])
                    df.at[ts, "signal_tag"]= "ORB_SHORT"
                    state.short_taken      = True
                    logger.debug(
                        f"ORB SHORT signal: {ts} | "
                        f"close={df.at[ts,'close']:.2f} < "
                        f"or_low={df.at[ts,'or_low']:.2f} | "
                        f"SL={df.at[ts,'signal_sl']:.2f}"
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # Validation override
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the OHLCV DataFrame and return a clean copy.

        Raises
        ------
        ValueError
            If required columns are missing or the index has no time component.
        """
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"ORBNiftyStrategy: DataFrame missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        if not hasattr(df.index, "time"):
            raise ValueError(
                "ORBNiftyStrategy requires a DatetimeIndex with a .time attribute. "
                "Ensure the DataFrame has a timezone-aware IST DatetimeIndex "
                "(e.g. pd.date_range(..., tz='Asia/Kolkata'))."
            )
        df = df.copy()
        df["signal"] = 0
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Introspection helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        """Return current strategy parameters as a dict."""
        return {
            "or_window_minutes": self.or_window_minutes,
            "vwap_filter":       self.vwap_filter,
            "trailing_divisor":  self.trailing_divisor,
            "min_body_pct":      self.min_body_pct,
        }

    def __repr__(self) -> str:
        return (
            f"ORBNiftyStrategy("
            f"or_window={self.or_window_minutes}min, "
            f"vwap_filter={self.vwap_filter}, "
            f"trail_div={self.trailing_divisor}, "
            f"min_body={self.min_body_pct}%)"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience export for the event loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    "ORBNiftyStrategy",
    "ORBIndicators",
    "ORBTrailingStop",
    "ORLevels",
    "DayTradeState",
    "SignalBar",
    "OR_START_TIME",
    "OR_END_TIME",
    "SIGNAL_START",
    "SQUAREOFF_TIME",
    "TRAILING_DIVISOR",
    "MIN_OR_BARS",
]