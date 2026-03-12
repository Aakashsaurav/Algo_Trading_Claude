"""
indicators/volatility.py
--------------------------
Volatility-based indicators.

INDICATORS:
    atr(high, low, close, period)         Average True Range
    bollinger_bands(series, period, std)  Bollinger Bands (upper, mid, lower, %B, bandwidth)
    keltner_channels(high,low,close,...)  Keltner Channels
    bb_squeeze(df, ...)                   Bollinger Band / Keltner squeeze detector
"""

import numpy as np
import pandas as pd
from indicators.moving_averages import ema, sma


def atr(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range (Wilder's ATR).

    True Range = max(
        High - Low,
        |High - Previous Close|,
        |Low  - Previous Close|
    )
    ATR = Wilder's EMA(True Range, period)

    WHY ATR: ATR captures volatility regardless of direction. Used for:
    - Position sizing (risk per trade = 1 ATR)
    - Stop-loss placement (e.g. 2× ATR below entry)
    - Supertrend calculation

    Args:
        high   : High prices.
        low    : Low prices.
        close  : Close prices.
        period : Smoothing period. Default 14 (Wilder's original).

    Returns:
        pd.Series: ATR values. First value is NaN (needs previous close).

    Edge cases:
        - Single-bar series → returns NaN.
        - All identical prices → ATR = 0.

    Example:
        atr_14 = atr(df["high"], df["low"], df["close"], 14)
        stop_loss = entry_price - 2 * atr_14.iloc[-1]
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing: same as RSI smoothing (alpha = 1/period)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(
    series: pd.Series,
    period: int   = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Middle Band = SMA(period)
    Upper Band  = Middle Band + num_std × rolling std deviation
    Lower Band  = Middle Band - num_std × rolling std deviation
    %B          = (Price - Lower) / (Upper - Lower)   [0=lower, 1=upper, 0.5=middle]
    Bandwidth   = (Upper - Lower) / Middle             [squeeze when very low]

    WHY BOLLINGER BANDS: Price tends to stay within the bands (~95% of time
    with 2σ). Breakouts signal strong moves; contractions signal upcoming
    breakouts.

    Args:
        series  : Price series (typically close).
        period  : SMA period. Default 20.
        num_std : Number of standard deviations. Default 2.0.

    Returns:
        pd.DataFrame with columns:
            bb_upper, bb_middle, bb_lower, bb_pct_b, bb_bandwidth

    Edge cases:
        - When bandwidth = 0 (all prices identical) → %B = NaN.
        - First (period-1) rows are NaN.

    Example:
        bb = bollinger_bands(df["close"])
        df["bb_upper"] = bb["bb_upper"]
        buy  = df["close"] < bb["bb_lower"]    # price touches lower band
        sell = df["close"] > bb["bb_upper"]    # price touches upper band
    """
    if period < 2:
        raise ValueError(f"period must be >= 2 for std deviation, got {period}")
    if num_std <= 0:
        raise ValueError(f"num_std must be > 0, got {num_std}")

    middle = series.rolling(window=period, min_periods=period).mean()
    std    = series.rolling(window=period, min_periods=period).std(ddof=1)

    upper = middle + num_std * std
    lower = middle - num_std * std

    band_width = upper - lower
    pct_b     = (series - lower) / band_width.replace(0, np.nan)
    bandwidth  = band_width / middle.replace(0, np.nan)

    return pd.DataFrame({
        "bb_upper":     upper,
        "bb_middle":    middle,
        "bb_lower":     lower,
        "bb_pct_b":     pct_b,
        "bb_bandwidth": bandwidth,
    }, index=series.index)


def keltner_channels(
    high:       pd.Series,
    low:        pd.Series,
    close:      pd.Series,
    ema_period: int   = 20,
    atr_period: int   = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Keltner Channels.

    Middle = EMA(close, ema_period)
    Upper  = Middle + multiplier × ATR(atr_period)
    Lower  = Middle - multiplier × ATR(atr_period)

    WHY KELTNER vs BOLLINGER: Keltner uses ATR (true volatility) instead of
    standard deviation. It's smoother and less prone to sudden widening from
    single spike bars.

    Args:
        high       : High prices.
        low        : Low prices.
        close      : Close prices.
        ema_period : EMA period for middle line. Default 20.
        atr_period : ATR period. Default 10.
        multiplier : ATR multiplier. Default 2.0.

    Returns:
        pd.DataFrame with columns: kc_upper, kc_middle, kc_lower.

    Example:
        kc = keltner_channels(df["high"], df["low"], df["close"])
    """
    middle = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)

    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val

    return pd.DataFrame({
        "kc_upper":  upper,
        "kc_middle": middle,
        "kc_lower":  lower,
    }, index=close.index)


def bb_squeeze(
    high:        pd.Series,
    low:         pd.Series,
    close:       pd.Series,
    bb_period:   int   = 20,
    bb_std:      float = 2.0,
    kc_ema:      int   = 20,
    kc_atr:      int   = 10,
    kc_mult:     float = 1.5,
) -> pd.Series:
    """
    Bollinger Band / Keltner Channel Squeeze detector.

    A "squeeze" occurs when Bollinger Bands are INSIDE Keltner Channels,
    indicating low volatility compression before an explosive move.

    Returns True (squeeze on) when BB is inside KC, False when released.

    WHY SQUEEZE: After a period of compression, price tends to break out
    with higher-than-average momentum. The squeeze release is a high-
    probability entry signal.

    Args:
        high, low, close : OHLC series.
        bb_period        : Bollinger Bands period.
        bb_std           : Bollinger Bands std dev multiplier.
        kc_ema           : Keltner EMA period.
        kc_atr           : Keltner ATR period.
        kc_mult          : Keltner multiplier (lower = tighter KC = fewer squeezes).

    Returns:
        pd.Series of bool: True = squeeze active, False = squeeze released.

    Example:
        squeeze = bb_squeeze(df["high"], df["low"], df["close"])
        squeeze_released = squeeze.shift(1) & ~squeeze   # squeeze just ended
    """
    bb = bollinger_bands(close, bb_period, bb_std)
    kc = keltner_channels(high, low, close, kc_ema, kc_atr, kc_mult)

    # Squeeze = BB upper is below KC upper AND BB lower is above KC lower
    squeeze_on = (bb["bb_upper"] < kc["kc_upper"]) & \
                 (bb["bb_lower"] > kc["kc_lower"])

    return squeeze_on
