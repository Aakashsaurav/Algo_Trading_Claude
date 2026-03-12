"""
indicators/trend.py
--------------------
Trend-following indicators.

INDICATORS:
    supertrend(high, low, close, period, multiplier)   Supertrend
    adx(high, low, close, period)                      Average Directional Index
"""

import numpy as np
import pandas as pd
from indicators.volatility import atr


def supertrend(
    high:       pd.Series,
    low:        pd.Series,
    close:      pd.Series,
    period:     int   = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend Indicator.

    Supertrend determines trend direction using ATR-based bands around
    the midpoint of the High-Low range.

    Basic Upper Band = (High + Low) / 2 + multiplier × ATR
    Basic Lower Band = (High + Low) / 2 - multiplier × ATR

    The final Supertrend line flips between the upper and lower band
    based on close price position.

    Signal:
        direction = +1 → Uptrend (price above Supertrend line)
        direction = -1 → Downtrend (price below Supertrend line)

    WHY SUPERTREND: Clean binary signal (long/flat). Self-adjusting
    stop-loss. Works well on trending markets. Used widely on NSE equities.

    Args:
        high, low, close : OHLC series.
        period           : ATR period. Default 10.
        multiplier       : ATR multiplier for band width. Default 3.0.

    Returns:
        pd.DataFrame with columns:
            supertrend     : The Supertrend line value (acts as trailing stop)
            direction      : +1 = uptrend, -1 = downtrend
            buy_signal     : True on the bar where trend flips to +1
            sell_signal    : True on the bar where trend flips to -1

    Edge cases:
        - First `period` rows are NaN (ATR not yet available).
        - If close equals the Supertrend exactly → treated as uptrend.

    Example:
        st = supertrend(df["high"], df["low"], df["close"], 10, 3.0)
        buy_entry  = st["buy_signal"]
        sell_entry = st["sell_signal"]
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")

    atr_val  = atr(high, low, close, period)
    hl_mid   = (high + low) / 2

    basic_upper = hl_mid + multiplier * atr_val
    basic_lower = hl_mid - multiplier * atr_val

    # Final bands and direction are computed iteratively — cannot be vectorised
    # because each bar's final band depends on the previous bar's final band.
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend_line = pd.Series(np.nan, index=close.index)
    direction       = pd.Series(0,      index=close.index, dtype=int)

    for i in range(1, len(close)):
        if pd.isna(atr_val.iloc[i]):
            continue

        # Final Upper Band:
        # If previous final upper < current basic upper, OR
        # if previous close was > previous final upper → reset to basic
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or \
                close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Final Lower Band:
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or \
                close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Determine direction
        prev_dir = direction.iloc[i - 1]

        if prev_dir == -1 and close.iloc[i] > final_upper.iloc[i]:
            direction.iloc[i] = 1   # flip to uptrend
        elif prev_dir == 1 and close.iloc[i] < final_lower.iloc[i]:
            direction.iloc[i] = -1  # flip to downtrend
        else:
            direction.iloc[i] = prev_dir if prev_dir != 0 else 1

        # Supertrend line = lower band in uptrend, upper band in downtrend
        supertrend_line.iloc[i] = (
            final_lower.iloc[i] if direction.iloc[i] == 1
            else final_upper.iloc[i]
        )

    # Buy signal: direction flips from -1 to +1
    # Sell signal: direction flips from +1 to -1
    buy_signal  = (direction == 1)  & (direction.shift(1) == -1)
    sell_signal = (direction == -1) & (direction.shift(1) == 1)

    return pd.DataFrame({
        "supertrend":  supertrend_line,
        "direction":   direction,
        "buy_signal":  buy_signal,
        "sell_signal": sell_signal,
    }, index=close.index)


def adx(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with +DI and -DI.

    ADX measures TREND STRENGTH (not direction).
    +DI measures upward movement strength.
    -DI measures downward movement strength.

    Interpretation:
        ADX > 25 → Strong trend (worth following)
        ADX < 20 → Weak/no trend (avoid trend strategies)
        +DI > -DI → Uptrend
        -DI > +DI → Downtrend
        +DI crosses above -DI → Buy signal
        -DI crosses above +DI → Sell signal

    Args:
        high, low, close : OHLC series.
        period           : Wilder smoothing period. Default 14.

    Returns:
        pd.DataFrame with columns: adx, plus_di, minus_di.

    Edge cases:
        - First (2 × period) rows are NaN.
        - If True Range is 0 (no price movement) → DI = NaN.

    Example:
        adx_df = adx(df["high"], df["low"], df["close"])
        strong_uptrend = (adx_df["adx"] > 25) & (adx_df["plus_di"] > adx_df["minus_di"])
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move   = high - prev_high
    down_move = prev_low - low

    # +DM: upward move larger and positive
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm,  index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    # Wilder's smoothing for TR, +DM, -DM
    smooth_tr      = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    smooth_minus_dm= minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    plus_di  = 100 * smooth_plus_dm  / smooth_tr.replace(0, np.nan)
    minus_di = 100 * smooth_minus_dm / smooth_tr.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({
        "adx":      adx_val,
        "plus_di":  plus_di,
        "minus_di": minus_di,
    }, index=close.index)
