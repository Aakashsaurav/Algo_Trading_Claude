"""
indicators/moving_averages.py
------------------------------
Pure functions for moving average indicators.
All functions take a pandas Series and return a pandas Series.
No state, no side effects — safe to use in both backtester and live bot.

INDICATORS:
    sma(series, period)            Simple Moving Average
    ema(series, period)            Exponential Moving Average
    dema(series, period)           Double EMA (reduces lag vs EMA)
    wma(series, period)            Weighted Moving Average
    vwap(high, low, close, volume) Volume Weighted Avg Price (intraday)
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Arithmetic mean of the last `period` values. Equal weight to all bars.
    First (period-1) values will be NaN — this is correct behaviour.

    Args:
        series (pd.Series): Price series (typically close prices).
        period (int):       Lookback window. Must be >= 1.

    Returns:
        pd.Series: SMA values, same index as input.

    Example:
        sma_20 = sma(df["close"], 20)
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int, adjust: bool = False) -> pd.Series:
    """
    Exponential Moving Average.

    Gives more weight to recent prices. The smoothing factor (alpha) is
    2 / (period + 1). Recent values have greater influence than older ones.

    WHY adjust=False: This matches the industry-standard EMA calculation
    used by TradingView, Bloomberg, and NSE. adjust=True uses a different
    formula that produces different values for early data points.

    Args:
        series (pd.Series): Price series.
        period (int):       EMA span. Must be >= 1.
        adjust (bool):      Use pandas EWM adjust parameter. Default False.

    Returns:
        pd.Series: EMA values, same index as input.

    Example:
        ema_50 = ema(df["close"], 50)
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return series.ewm(span=period, adjust=adjust, min_periods=period).mean()


def dema(series: pd.Series, period: int) -> pd.Series:
    """
    Double Exponential Moving Average.

    Formula: DEMA = 2 * EMA(n) - EMA(EMA(n))
    Reduces the lag of a standard EMA by applying EMA twice and correcting.

    WHY DEMA: For trend-following strategies, DEMA reacts faster to price
    changes than EMA while producing fewer whipsaws than simply halving
    the EMA period.

    Args:
        series (pd.Series): Price series.
        period (int):       EMA period used internally.

    Returns:
        pd.Series: DEMA values, same index as input.

    Example:
        dema_20 = dema(df["close"], 20)
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    return 2 * ema1 - ema2


def wma(series: pd.Series, period: int) -> pd.Series:
    """
    Weighted Moving Average.

    Assigns linearly increasing weights: the most recent bar gets weight
    `period`, second-most-recent gets `period-1`, ..., oldest gets 1.
    More responsive than SMA, less than EMA.

    Args:
        series (pd.Series): Price series.
        period (int):       Lookback window. Must be >= 1.

    Returns:
        pd.Series: WMA values, same index as input.

    Edge case: Returns NaN for the first (period-1) values (insufficient data).

    Example:
        wma_10 = wma(df["close"], 10)
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    weights = np.arange(1, period + 1, dtype=float)  # [1, 2, ..., period]

    def _weighted_mean(window):
        if len(window) < period:
            return np.nan
        return np.dot(window, weights) / weights.sum()

    return series.rolling(window=period, min_periods=period).apply(
        _weighted_mean, raw=True
    )


def vwap(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Volume Weighted Average Price (VWAP).

    VWAP = Cumulative(Typical Price × Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3

    IMPORTANT: VWAP resets every trading day. This implementation computes
    a running intraday VWAP that resets at each new date in the index.
    For daily/weekly/monthly data, VWAP is computed over the entire series
    (no intraday reset).

    WHY VWAP: Institutional traders use VWAP as a benchmark. Price above
    VWAP = bullish bias; price below VWAP = bearish bias.

    Args:
        high   (pd.Series): High prices.
        low    (pd.Series): Low prices.
        close  (pd.Series): Close prices.
        volume (pd.Series): Volume.

    Returns:
        pd.Series: VWAP values, same index as input.

    Edge case: If volume is 0 for an entire day, VWAP for that day = NaN.

    Example:
        df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])
    """
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume

    # Detect if this is intraday data (index has time component)
    has_time = hasattr(close.index, 'date')

    if has_time:
        # Reset cumulative sum at each new date
        date_col = pd.Series(close.index.date, index=close.index)
        cum_tp_vol = tp_vol.groupby(date_col).cumsum()
        cum_vol    = volume.groupby(date_col).cumsum()
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_vol    = volume.cumsum()

    return cum_tp_vol / cum_vol.replace(0, np.nan)
