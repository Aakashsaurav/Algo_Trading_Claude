"""
indicators/oscillators.py
--------------------------
Momentum and oscillator indicators.
All functions take pandas Series/DataFrame and return pandas Series.

INDICATORS:
    rsi(series, period)                        Relative Strength Index
    stochastic(high, low, close, k, d)         Stochastic %K and %D
    macd(series, fast, slow, signal)           MACD line, signal, histogram
    roc(series, period)                        Rate of Change (momentum %)
    cci(high, low, close, period)              Commodity Channel Index
"""

import numpy as np
import pandas as pd
from indicators.moving_averages import ema, sma


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's RSI).

    RSI measures the speed and magnitude of price changes.
    Range: 0 to 100.
    Convention: > 70 = overbought, < 30 = oversold.

    IMPLEMENTATION NOTE:
    Wilder's original smoothing uses a modified EMA with alpha = 1/period,
    NOT the standard 2/(period+1) alpha. This is computed via
    ewm(alpha=1/period, adjust=False) — matching TradingView's RSI.

    Args:
        series (pd.Series): Price series (typically close).
        period (int):       Lookback period. Default 14 (Wilder's original).

    Returns:
        pd.Series: RSI values 0–100, NaN for first `period` rows.

    Edge cases:
        - If all gains are 0 for a period → RSI = 0.
        - If all losses are 0 for a period → RSI = 100.
        - Empty series → returns empty Series.

    Example:
        rsi_14 = rsi(df["close"], 14)
        overbought = rsi_14 > 70
        oversold   = rsi_14 < 30
    """
    if series.empty:
        return pd.Series(dtype=float)
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    delta = series.diff()

    gain = delta.clip(lower=0)   # positive changes only
    loss = (-delta).clip(lower=0)  # absolute value of negative changes

    # Wilder's smoothing: alpha = 1/period (NOT 2/(period+1))
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # avoid division by zero
    rsi_val = 100 - (100 / (1 + rs))

    # When avg_loss is 0 (all gains), RSI should be 100
    rsi_val[avg_loss == 0] = 100.0
    # When avg_gain is 0 (all losses), RSI should be 0
    rsi_val[avg_gain == 0] = 0.0

    return rsi_val


def stochastic(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K and %D).

    %K shows where the close is relative to the high-low range over k_period.
    %D is a smoothed version of %K (acts as signal line).

    Formula:
        Raw %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        %K     = SMA(Raw %K, smooth_k)      [smooth_k=1 gives fast stochastic]
        %D     = SMA(%K, d_period)

    Range: 0–100.
    Convention: %K > 80 = overbought, %K < 20 = oversold.

    Args:
        high     : High price series.
        low      : Low price series.
        close    : Close price series.
        k_period : Lookback for raw %K. Default 14.
        d_period : Smoothing for %D. Default 3.
        smooth_k : Smoothing for %K (3 = slow stochastic). Default 3.

    Returns:
        pd.DataFrame with columns ["stoch_k", "stoch_d"].

    Edge cases:
        - If high == low for all bars in window → %K = NaN (no range to measure).

    Example:
        stoch = stochastic(df["high"], df["low"], df["close"])
        buy_signal = (stoch["stoch_k"] < 20) & (stoch["stoch_k"] > stoch["stoch_d"])
    """
    lowest_low   = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    price_range = highest_high - lowest_low
    raw_k = 100 * (close - lowest_low) / price_range.replace(0, np.nan)

    k = sma(raw_k, smooth_k) if smooth_k > 1 else raw_k
    d = sma(k, d_period)

    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=close.index)


def macd(
    series:        pd.Series,
    fast_period:   int = 12,
    slow_period:   int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    MACD Line     = EMA(fast) - EMA(slow)
    Signal Line   = EMA(MACD Line, signal_period)
    Histogram     = MACD Line - Signal Line

    Convention:
        MACD crosses above Signal → bullish momentum building.
        Histogram turns positive  → trend strengthening.
        Histogram turns negative  → trend weakening.

    Args:
        series       : Price series (typically close).
        fast_period  : Short EMA period. Default 12.
        slow_period  : Long EMA period. Default 26.
        signal_period: Signal line EMA period. Default 9.

    Returns:
        pd.DataFrame with columns ["macd", "signal", "histogram"].

    Edge cases:
        - fast_period must be < slow_period.
        - NaN values for first (slow_period + signal_period - 1) bars.

    Example:
        m = macd(df["close"])
        crossover_up = (m["macd"] > m["signal"]) & (m["macd"].shift(1) <= m["signal"].shift(1))
    """
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be less than "
            f"slow_period ({slow_period})"
        )

    fast_ema   = ema(series, fast_period)
    slow_ema   = ema(series, slow_period)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram  = macd_line - signal_line

    return pd.DataFrame({
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": histogram,
    }, index=series.index)


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """
    Rate of Change — momentum as percentage.

    ROC = 100 * (Close - Close[period ago]) / Close[period ago]

    Positive ROC = price has risen relative to `period` bars ago.
    ROC crossing zero from below = bullish signal.

    Args:
        series (pd.Series): Price series.
        period (int):       Lookback bars. Default 12.

    Returns:
        pd.Series: ROC percentage values.

    Example:
        roc_12 = roc(df["close"], 12)
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return series.pct_change(periods=period) * 100


def cci(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Commodity Channel Index.

    CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)

    Range: Unbounded, but typically oscillates ±100 to ±200.
    Convention: > +100 = overbought, < -100 = oversold.

    The constant 0.015 ensures ~75% of values fall within ±100.

    Args:
        high   : High price series.
        low    : Low price series.
        close  : Close price series.
        period : Lookback window. Default 20.

    Returns:
        pd.Series: CCI values.

    Example:
        cci_20 = cci(df["high"], df["low"], df["close"], 20)
    """
    typical_price = (high + low + close) / 3
    tp_sma = sma(typical_price, period)

    # Mean absolute deviation
    mean_dev = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )

    return (typical_price - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))
