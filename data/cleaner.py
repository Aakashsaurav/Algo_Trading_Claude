"""
data/cleaner.py
----------------
Handles data quality issues in OHLCV time-series data.

WHY DATA CLEANING MATTERS IN TRADING:
---------------------------------------
Raw market data is messier than you think. Common problems:

1. MISSING DATES — Market holidays, trading halts, circuit breakers, or API
   gaps can create holes in your time series. A rolling average on data with
   gaps silently produces wrong values.

2. ZERO/NULL PRICES — Occasionally the API returns 0 for open/close.
   This would make your strategy think a stock crashed to zero.

3. PRICE SPIKES — Erroneous data points (e.g. 10,000 for a ₹100 stock).
   These destroy Bollinger Bands, ATR, and any volatility-based signal.

4. OHLC RELATIONSHIP VIOLATIONS — Sometimes data has close > high or open < low,
   which is physically impossible and will break candlestick pattern detection.

5. DUPLICATE TIMESTAMPS — Can occur when merging data from multiple sources.

6. NON-TRADING HOURS IN MINUTE DATA — NSE trades 9:15 AM–3:30 PM.
   Some feeds include pre-market or post-market junk rows.

This module detects and fixes all these issues before data is used for backtesting.

USAGE:
    from data.cleaner import cleaner
    df_clean = cleaner.clean_daily(df_raw, symbol="INFY")
    df_clean = cleaner.clean_minute(df_raw, symbol="INFY")
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# NSE trading hours (for filtering minute data)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Maximum allowed single-candle price change (as a fraction)
# If a candle moves >50% from previous close, flag it as a potential spike
MAX_SPIKE_THRESHOLD = 0.5


class DataCleaner:
    """
    Cleans and validates OHLCV DataFrames before they're used in strategies.

    All methods follow the same pattern:
    - Input: raw DataFrame (possibly with issues)
    - Output: clean DataFrame with issues fixed or removed
    - The original DataFrame is never modified (returns a copy)
    """

    def clean_daily(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        fill_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Full cleaning pipeline for daily OHLCV data.

        Applies these fixes in order:
        1. Remove duplicate timestamps
        2. Remove rows with zero/null prices
        3. Fix OHLC relationship violations
        4. Remove price spikes (optional warning)
        5. Fill missing trading days (if fill_missing=True)

        Args:
            df (pd.DataFrame): Raw daily OHLCV data.
            symbol (str): Symbol name for logging context.
            fill_missing (bool): If True, fill missing weekdays using forward fill.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        if df is None or df.empty:
            logger.warning(f"[{symbol}] clean_daily received empty DataFrame.")
            return pd.DataFrame()

        df = df.copy()
        original_len = len(df)

        df = self._remove_duplicates(df, symbol)
        df = self._remove_invalid_prices(df, symbol)
        df = self._fix_ohlc_violations(df, symbol)
        df = self._flag_price_spikes(df, symbol)

        if fill_missing:
            df = self._fill_missing_trading_days(df, symbol)

        cleaned_len = len(df)
        removed = original_len - cleaned_len

        if removed > 0:
            logger.info(
                f"[{symbol}] Daily cleaning: {original_len} → {cleaned_len} rows "
                f"({removed} rows removed/fixed)"
            )

        return df

    def clean_minute(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> pd.DataFrame:
        """
        Full cleaning pipeline for 1-minute OHLCV data.

        Additional steps vs daily:
        - Filter out candles outside NSE trading hours (9:15 AM – 3:30 PM IST)
        - More aggressive spike detection (minute data is noisier)

        Args:
            df (pd.DataFrame): Raw 1-minute OHLCV data.
            symbol (str): Symbol name for logging.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        if df is None or df.empty:
            logger.warning(f"[{symbol}] clean_minute received empty DataFrame.")
            return pd.DataFrame()

        df = df.copy()
        original_len = len(df)

        df = self._remove_duplicates(df, symbol)
        df = self._filter_trading_hours(df, symbol)
        df = self._remove_invalid_prices(df, symbol)
        df = self._fix_ohlc_violations(df, symbol)

        cleaned_len = len(df)
        removed = original_len - cleaned_len

        if removed > 0:
            logger.info(
                f"[{symbol}] Minute cleaning: {original_len} → {cleaned_len} rows "
                f"({removed} removed)"
            )

        return df

    # ── Individual Cleaning Steps ─────────────────────────────────────────────

    def _remove_duplicates(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Remove rows with duplicate timestamps.
        Keeps the last occurrence (most recently received data wins).
        """
        dupes = df.index.duplicated(keep="last")
        count = dupes.sum()

        if count > 0:
            logger.warning(f"[{symbol}] Removing {count} duplicate timestamps.")
            df = df[~dupes]

        return df

    def _remove_invalid_prices(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Remove rows where OHLC prices are zero, negative, or null.

        These rows would cause division-by-zero errors and destroy indicators
        like RSI, Bollinger Bands, etc.
        """
        price_cols = ["open", "high", "low", "close"]
        available_cols = [c for c in price_cols if c in df.columns]

        # Mask: True for rows where any price is null, zero, or negative
        invalid_mask = (
            df[available_cols].isnull().any(axis=1) |
            (df[available_cols] <= 0).any(axis=1)
        )
        count = invalid_mask.sum()

        if count > 0:
            logger.warning(
                f"[{symbol}] Removing {count} rows with zero/null/negative prices."
            )
            df = df[~invalid_mask]

        return df

    def _fix_ohlc_violations(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Fix OHLC relationship violations — physically impossible candles.

        Rules that must hold:
        - high >= open, high >= close, high >= low
        - low <= open, low <= close, low <= high

        Fix strategy: if high is too low, raise it to max(open, close).
                       if low is too high, lower it to min(open, close).

        We fix rather than remove to preserve the time series continuity.
        """
        if not all(c in df.columns for c in ["open", "high", "low", "close"]):
            return df

        # High must be the maximum of open, high, low, close
        correct_high = df[["open", "high", "low", "close"]].max(axis=1)
        high_violations = df["high"] < correct_high
        if high_violations.sum() > 0:
            logger.warning(
                f"[{symbol}] Fixing {high_violations.sum()} 'high' violations."
            )
            df.loc[high_violations, "high"] = correct_high[high_violations]

        # Low must be the minimum of open, high, low, close
        correct_low = df[["open", "high", "low", "close"]].min(axis=1)
        low_violations = df["low"] > correct_low
        if low_violations.sum() > 0:
            logger.warning(
                f"[{symbol}] Fixing {low_violations.sum()} 'low' violations."
            )
            df.loc[low_violations, "low"] = correct_low[low_violations]

        return df

    def _flag_price_spikes(
        self,
        df: pd.DataFrame,
        symbol: str,
        threshold: float = MAX_SPIKE_THRESHOLD,
    ) -> pd.DataFrame:
        """
        Detect and log (but NOT auto-remove) suspicious price spikes.

        A spike is flagged when close-to-close change exceeds `threshold`
        (default: 50%) in a single day. This could be:
        - Legitimate (earnings, corporate action, circuit breaker)
        - Erroneous data

        We log it for review but don't auto-remove — a real 50% move
        (e.g. surprise earnings, bonus announcement) should be preserved.
        This requires human/scheduled review.
        """
        if "close" not in df.columns or len(df) < 2:
            return df

        pct_change = df["close"].pct_change().abs()
        spikes = pct_change > threshold

        if spikes.sum() > 0:
            spike_dates = df.index[spikes].tolist()
            logger.warning(
                f"[{symbol}] ⚠️  {spikes.sum()} potential price spike(s) detected. "
                f"Dates: {[str(d.date()) for d in spike_dates[:5]]}. "
                f"Please review — could be legitimate corporate action or bad data."
            )

        return df

    def _filter_trading_hours(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Filter 1-minute data to only include NSE trading hours: 9:15 AM – 3:30 PM IST.

        Remove pre-market, post-market, and any erroneous off-hours candles.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Extract hour and minute from the datetime index
        hours = df.index.hour
        minutes = df.index.minute

        # Time as a comparable integer: HHMM (e.g. 9:15 = 915, 15:30 = 1530)
        time_val = hours * 100 + minutes

        market_open_val = MARKET_OPEN_HOUR * 100 + MARKET_OPEN_MINUTE    # 915
        market_close_val = MARKET_CLOSE_HOUR * 100 + MARKET_CLOSE_MINUTE  # 1530

        in_hours = (time_val >= market_open_val) & (time_val <= market_close_val)
        removed = (~in_hours).sum()

        if removed > 0:
            logger.debug(
                f"[{symbol}] Removed {removed} candles outside trading hours."
            )

        return df[in_hours]

    def _fill_missing_trading_days(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Fill gaps in daily data caused by holidays, trading halts, or API issues.

        Strategy: Forward-fill — missing days get the previous day's OHLCV.
        Volume is set to 0 for synthetic days (no actual trading occurred).

        Why forward-fill: For backtesting, indicators need a continuous time
        series. A 5-day EMA with a gap in the middle produces incorrect values.

        Note: This fills weekdays only. Weekends are left as-is.

        Args:
            df (pd.DataFrame): Daily OHLCV data.
            symbol (str): For logging.

        Returns:
            pd.DataFrame: Data with missing weekdays filled.
        """
        if df.empty:
            return df

        # Generate a complete weekday range from first to last date in the data
        full_range = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            freq="B",   # "B" = business days (Mon–Fri, excludes weekends)
            tz=df.index.tz,
        )

        # Reindex to the full range — creates NaN rows for missing days
        df_reindexed = df.reindex(full_range)

        missing_count = df_reindexed.isnull().any(axis=1).sum()

        if missing_count > 0:
            logger.info(
                f"[{symbol}] Forward-filling {missing_count} missing trading days."
            )

            # Forward-fill OHLC prices
            for col in ["open", "high", "low", "close"]:
                if col in df_reindexed.columns:
                    df_reindexed[col] = df_reindexed[col].ffill()

            # Set volume to 0 for synthetic days (no real trades happened)
            if "volume" in df_reindexed.columns:
                df_reindexed["volume"] = df_reindexed["volume"].fillna(0)

            # Set OI to 0 for synthetic days
            if "oi" in df_reindexed.columns:
                df_reindexed["oi"] = df_reindexed["oi"].fillna(0)

        # Drop any remaining NaN rows (shouldn't happen, but safety net)
        df_reindexed.dropna(subset=["close"], inplace=True)

        return df_reindexed

    # ── Utility: Validate a DataFrame ────────────────────────────────────────

    def get_quality_report(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> dict:
        """
        Generate a data quality report for a DataFrame.

        Useful to call after loading data to understand its quality
        before running a backtest. Shown in the dashboard's data view.

        Args:
            df (pd.DataFrame): OHLCV data to inspect.
            symbol (str): Symbol name for the report.

        Returns:
            dict: Quality metrics.
        """
        if df is None or df.empty:
            return {"symbol": symbol, "status": "empty", "rows": 0}

        price_cols = ["open", "high", "low", "close"]
        available = [c for c in price_cols if c in df.columns]

        # Count issues
        null_count = df[available].isnull().any(axis=1).sum()
        zero_count = (df[available] <= 0).any(axis=1).sum()
        dupe_count = df.index.duplicated().sum()

        pct_change = df["close"].pct_change().abs() if "close" in df.columns else pd.Series()
        spike_count = (pct_change > MAX_SPIKE_THRESHOLD).sum()

        report = {
            "symbol": symbol,
            "rows": len(df),
            "date_from": str(df.index[0].date()) if not df.empty else None,
            "date_to": str(df.index[-1].date()) if not df.empty else None,
            "null_price_rows": int(null_count),
            "zero_price_rows": int(zero_count),
            "duplicate_timestamps": int(dupe_count),
            "potential_spikes": int(spike_count),
            "status": "clean" if (null_count + zero_count + dupe_count) == 0 else "issues found",
        }

        return report


# ── Module-level singleton ────────────────────────────────────────────────────
cleaner = DataCleaner()
