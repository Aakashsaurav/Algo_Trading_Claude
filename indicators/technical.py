"""
indicators/technical.py
------------------------
Complete technical indicator library implemented in pure NumPy/pandas.

No external dependencies (pandas-ta, TA-Lib, etc.) required.
All functions accept pandas Series or DataFrame and return pandas Series/DataFrame.

DESIGN PRINCIPLES:
  1. Each function is a pure function — takes data in, returns data out.
     No global state. Safe to call multiple times in any order.
  2. All outputs are pandas Series with the same index as the input.
     This makes alignment with OHLCV data automatic and error-free.
  3. NaN is used for the warm-up period (first N bars where the indicator
     doesn't have enough data). Never fill with zeros — that creates
     false signals at the start of the data.
  4. All functions validate inputs and raise clear ValueError messages.

LOOK-AHEAD BIAS NOTE:
  Every indicator here is strictly backward-looking.
  The value at index i is computed using ONLY data up to and including i.
  shift(1) is used where needed to prevent using future data.

AVAILABLE INDICATORS:
  Trend:
    sma()       - Simple Moving Average
    ema()       - Exponential Moving Average
    dema()      - Double EMA (faster, less lag than EMA)
    supertrend() - Supertrend with ATR-based bands
    macd()      - MACD line, signal, histogram

  Momentum:
    rsi()       - Relative Strength Index
    stochastic() - Stochastic %K and %D
    roc()       - Rate of Change (price momentum %)

  Volatility:
    atr()       - Average True Range
    bollinger_bands() - Upper, middle, lower bands + %B + bandwidth
    keltner_channels() - EMA ± (multiplier × ATR)

  Volume:
    vwap()      - Volume Weighted Average Price (intraday reset per day)
    obv()       - On Balance Volume

  Statistical:
    zscore()    - Rolling z-score of a series
    rolling_correlation() - Rolling Pearson correlation between two series

  Pattern helpers:
    candle_body()   - Absolute candle body size (|close - open|)
    candle_range()  - High - Low
    is_green()      - Boolean: close > open
    is_red()        - Boolean: close < open
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_series(s: pd.Series, name: str = "input") -> None:
    """Raise clear errors for common bad inputs."""
    if not isinstance(s, pd.Series):
        raise TypeError(
            f"'{name}' must be a pandas Series, got: {type(s).__name__}. "
            "Pass a single column like df['close'], not the full DataFrame."
        )
    if s.empty:
        raise ValueError(f"'{name}' Series is empty — no data to compute indicator.")
    if s.isnull().all():
        raise ValueError(f"'{name}' Series is all NaN — cannot compute indicator.")


def _validate_period(period: int, name: str = "period") -> None:
    """Period must be a positive integer."""
    if not isinstance(period, int) or period < 1:
        raise ValueError(
            f"'{name}' must be a positive integer, got: {period}."
        )


# ===========================================================================
# TREND INDICATORS
# ===========================================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Theory: Arithmetic mean of the last `period` values.
    SMA is slower to react to price changes but less noisy than EMA.
    Use it to identify trend direction and support/resistance levels.

    Args:
        series (pd.Series): Price series (typically 'close').
        period (int):       Lookback window in bars.

    Returns:
        pd.Series: SMA values. First (period-1) values are NaN.

    Example:
        df['sma20'] = sma(df['close'], 20)
    """
    _validate_series(series, "series")
    _validate_period(period, "period")

    result = series.rolling(window=period, min_periods=period).mean()
    result.name = f"SMA_{period}"
    return result


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Theory: Weighted average giving more weight to recent prices.
    Weight of each bar decays exponentially: multiplier = 2 / (period + 1).
    EMA reacts faster to price changes than SMA — better for signals.

    Args:
        series (pd.Series): Price series.
        period (int):       Number of periods. The span of the EMA.

    Returns:
        pd.Series: EMA values. First (period-1) values are NaN.

    Example:
        df['ema9']  = ema(df['close'], 9)
        df['ema21'] = ema(df['close'], 21)
        # Signal: go long when ema9 crosses above ema21
    """
    _validate_series(series, "series")
    _validate_period(period, "period")

    # adjust=False gives the standard EMA formula used in trading platforms
    # min_periods ensures NaN for the warm-up period instead of partial EMAs
    result = series.ewm(span=period, adjust=False, min_periods=period).mean()
    result.name = f"EMA_{period}"
    return result


def dema(series: pd.Series, period: int) -> pd.Series:
    """
    Double Exponential Moving Average (DEMA).

    Theory: DEMA = 2 × EMA(n) - EMA(EMA(n))
    Reduces the lag of a standard EMA by subtracting the 'double-smoothed'
    version. Reacts faster but generates more false signals in sideways markets.

    Args:
        series (pd.Series): Price series.
        period (int):       EMA period.

    Returns:
        pd.Series: DEMA values.

    Example:
        df['dema20'] = dema(df['close'], 20)
    """
    _validate_series(series, "series")
    _validate_period(period, "period")

    ema1 = series.ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    result = 2 * ema1 - ema2
    result.name = f"DEMA_{period}"
    return result


def macd(
    series: pd.Series,
    fast_period:   int = 12,
    slow_period:   int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    Theory:
      MACD Line  = EMA(fast) - EMA(slow)
      Signal Line = EMA(MACD Line, signal_period)
      Histogram  = MACD Line - Signal Line

    Signals:
      BUY  when MACD line crosses above Signal line (histogram turns positive)
      SELL when MACD line crosses below Signal line (histogram turns negative)
      Divergence between price and MACD signals trend weakness/reversal.

    Args:
        series (pd.Series):  Close price series.
        fast_period (int):   Fast EMA period (default 12).
        slow_period (int):   Slow EMA period (default 26).
        signal_period (int): Signal line EMA period (default 9).

    Returns:
        pd.DataFrame with columns: ['macd', 'signal', 'histogram']

    Example:
        macd_df = macd(df['close'])
        df['macd']      = macd_df['macd']
        df['macd_sig']  = macd_df['signal']
        df['macd_hist'] = macd_df['histogram']
    """
    _validate_series(series, "series")
    for p, name in [(fast_period, "fast_period"), (slow_period, "slow_period"),
                    (signal_period, "signal_period")]:
        _validate_period(p, name)
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be less than "
            f"slow_period ({slow_period})."
        )

    fast_ema   = series.ewm(span=fast_period,   adjust=False, min_periods=fast_period).mean()
    slow_ema   = series.ewm(span=slow_period,   adjust=False, min_periods=slow_period).mean()
    macd_line  = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False,
                                 min_periods=signal_period).mean()
    histogram  = macd_line - signal_line

    result = pd.DataFrame({
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": histogram,
    }, index=series.index)
    return result


def supertrend(
    df: pd.DataFrame,
    period:     int   = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend indicator.

    Theory:
      Based on ATR (Average True Range) to set dynamic support/resistance bands.
      Upper Band = (high + low) / 2 + multiplier × ATR(period)
      Lower Band = (high + low) / 2 - multiplier × ATR(period)

      When price is above the Supertrend line → Uptrend (go long / stay long).
      When price is below the Supertrend line → Downtrend (go short / stay out).

    Args:
        df (pd.DataFrame):  Must have 'high', 'low', 'close' columns.
        period (int):        ATR period (default 10).
        multiplier (float):  ATR multiplier for bands (default 3.0).

    Returns:
        pd.DataFrame with columns:
          'supertrend'   : The actual Supertrend line value
          'direction'    : 1 = uptrend (bullish), -1 = downtrend (bearish)
          'buy_signal'   : True on the bar direction flips to bullish
          'sell_signal'  : True on the bar direction flips to bearish

    Example:
        st = supertrend(df, period=10, multiplier=3.0)
        df['supertrend'] = st['supertrend']
        df['st_dir']     = st['direction']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame with 'high', 'low', 'close' columns.")
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must have '{col}' column.")

    _validate_period(period, "period")

    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    # Smooth the bands: they can only move in one direction to avoid whipsaws
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    direction   = pd.Series(np.nan, index=df.index)
    supertrend_line = pd.Series(np.nan, index=df.index)

    close = df["close"]

    for i in range(1, len(df)):
        if pd.isna(atr_val.iloc[i]):
            continue

        # Upper band: only decreases if new upper < previous upper OR prev close > prev upper
        prev_upper = final_upper.iloc[i - 1]
        if upper_band.iloc[i] < prev_upper or close.iloc[i - 1] > prev_upper:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = prev_upper

        # Lower band: only increases if new lower > previous lower OR prev close < prev lower
        prev_lower = final_lower.iloc[i - 1]
        if lower_band.iloc[i] > prev_lower or close.iloc[i - 1] < prev_lower:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = prev_lower

        # Direction: if close > upper_band → uptrend; if close < lower_band → downtrend
        prev_dir = direction.iloc[i - 1]
        if pd.isna(prev_dir):
            prev_dir = 1  # Start assuming uptrend

        if close.iloc[i] > final_upper.iloc[i]:
            direction.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_dir

        # The actual Supertrend line
        if direction.iloc[i] == 1:
            supertrend_line.iloc[i] = final_lower.iloc[i]
        else:
            supertrend_line.iloc[i] = final_upper.iloc[i]

    # Detect direction changes for signals
    buy_signal  = (direction == 1)  & (direction.shift(1) == -1)
    sell_signal = (direction == -1) & (direction.shift(1) == 1)

    return pd.DataFrame({
        "supertrend":  supertrend_line,
        "direction":   direction,
        "buy_signal":  buy_signal,
        "sell_signal": sell_signal,
    }, index=df.index)


# ===========================================================================
# MOMENTUM INDICATORS
# ===========================================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Theory:
      RSI = 100 - (100 / (1 + RS))
      where RS = Average Gain / Average Loss over the last `period` bars.

      RSI ranges from 0 to 100.
      > 70 → Overbought (potential sell signal)
      < 30 → Oversold   (potential buy signal)

    This implementation uses Wilder's smoothing (standard for RSI):
    First average uses simple mean; subsequent bars use exponential decay.

    Args:
        series (pd.Series): Price series (usually 'close').
        period (int):       RSI period (default 14).

    Returns:
        pd.Series: RSI values (0–100). First `period` values are NaN.

    Example:
        df['rsi14'] = rsi(df['close'], 14)
        # Buy when rsi crosses above 30 (oversold recovery)
        # Sell when rsi crosses below 70 (overbought reversal)
    """
    _validate_series(series, "series")
    _validate_period(period, "period")

    delta = series.diff(1)

    # Separate gains and losses
    gains  = delta.clip(lower=0)   # Positive changes only
    losses = (-delta).clip(lower=0) # Absolute negative changes only

    # Wilder's smoothed moving average (equivalent to EMA with alpha=1/period)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
    rsi_values = 100 - (100 / (1 + rs))

    # When avg_loss is 0 (all gains), RSI = 100; when avg_gain is 0, RSI = 0
    rsi_values = rsi_values.fillna(
        pd.Series(np.where(avg_loss == 0, 100.0, 0.0), index=series.index)
    )

    result = pd.Series(rsi_values, index=series.index, name=f"RSI_{period}")

    # Mask warm-up period with NaN
    result.iloc[:period] = np.nan
    return result


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K and %D).

    Theory:
      %K = (Close - Lowest Low over k_period) / (Highest High - Lowest Low) × 100
      %D = SMA(%K, d_period)   ← signal line

      %K and %D range from 0 to 100.
      > 80 → Overbought; < 20 → Oversold.
      Signal: buy when %K crosses above %D in oversold territory.

    Args:
        df (pd.DataFrame): Must have 'high', 'low', 'close' columns.
        k_period (int):    Lookback for highest high and lowest low (default 14).
        d_period (int):    Smoothing period for %D (default 3).

    Returns:
        pd.DataFrame with columns: ['stoch_k', 'stoch_d']

    Example:
        stoch_df = stochastic(df, k_period=14, d_period=3)
        df['%K'] = stoch_df['stoch_k']
        df['%D'] = stoch_df['stoch_d']
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must have '{col}' column.")
    _validate_period(k_period, "k_period")
    _validate_period(d_period, "d_period")

    low_k  = df["low"].rolling(k_period, min_periods=k_period).min()
    high_k = df["high"].rolling(k_period, min_periods=k_period).max()

    denom = (high_k - low_k).replace(0, np.nan)  # Avoid division by zero
    k = ((df["close"] - low_k) / denom) * 100
    d = k.rolling(d_period, min_periods=d_period).mean()

    return pd.DataFrame({
        "stoch_k": k,
        "stoch_d": d,
    }, index=df.index)


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change (momentum as % change over N bars).

    Theory:
      ROC = (Close - Close[period bars ago]) / Close[period bars ago] × 100

      Positive ROC → upward momentum.
      Negative ROC → downward momentum.
      ROC crossing zero → trend reversal signal.

    Args:
        series (pd.Series): Price series.
        period (int):       Lookback period.

    Returns:
        pd.Series: ROC values in percentage.
    """
    _validate_series(series, "series")
    _validate_period(period, "period")

    result = ((series - series.shift(period)) / series.shift(period)) * 100
    result.name = f"ROC_{period}"
    return result


# ===========================================================================
# VOLATILITY INDICATORS
# ===========================================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR).

    Theory:
      True Range = max(high-low, |high-prev_close|, |low-prev_close|)
      ATR = EMA(True Range, period) using Wilder's smoothing

      ATR measures volatility. Higher ATR = more volatile market.
      Use ATR for:
        - Setting stop-loss distances (e.g., stop = entry - 2×ATR)
        - Position sizing (risk-based sizing)
        - Supertrend and Keltner Channel bands

    Args:
        df (pd.DataFrame): Must have 'high', 'low', 'close' columns.
        period (int):      ATR period (default 14).

    Returns:
        pd.Series: ATR values.
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must have '{col}' column.")
    _validate_period(period, "period")

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing: alpha = 1/period
    result = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    result.name = f"ATR_{period}"
    return result


def bollinger_bands(
    series:     pd.Series,
    period:     int   = 20,
    std_dev:    float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Theory:
      Middle Band = SMA(period)
      Upper Band  = SMA + std_dev × Rolling StdDev
      Lower Band  = SMA - std_dev × Rolling StdDev

      Bands expand during high volatility and contract during low volatility.
      Price touching upper band → overbought signal.
      Price touching lower band → oversold signal.
      'Bollinger Squeeze' (narrow bandwidth) → low volatility, expect breakout.

      %B = (Price - Lower Band) / (Upper Band - Lower Band)
        %B > 1 → price above upper band
        %B < 0 → price below lower band

    Args:
        series (pd.Series): Price series (usually close).
        period (int):       SMA period for middle band (default 20).
        std_dev (float):    Number of standard deviations (default 2.0).

    Returns:
        pd.DataFrame with columns:
          ['bb_upper', 'bb_middle', 'bb_lower', 'bb_pct_b', 'bb_bandwidth']
    """
    _validate_series(series, "series")
    _validate_period(period, "period")

    rolling     = series.rolling(window=period, min_periods=period)
    middle      = rolling.mean()
    std         = rolling.std(ddof=0)   # Population std (ddof=0), standard in TA

    upper       = middle + std_dev * std
    lower       = middle - std_dev * std
    bandwidth   = (upper - lower) / middle.replace(0, np.nan)
    pct_b       = (series - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame({
        "bb_upper":     upper,
        "bb_middle":    middle,
        "bb_lower":     lower,
        "bb_pct_b":     pct_b,
        "bb_bandwidth": bandwidth,
    }, index=series.index)


def keltner_channels(
    df:         pd.DataFrame,
    ema_period: int   = 20,
    atr_period: int   = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Keltner Channels.

    Theory:
      Middle = EMA(close, ema_period)
      Upper  = EMA + multiplier × ATR
      Lower  = EMA - multiplier × ATR

      Similar to Bollinger Bands but uses ATR instead of StdDev.
      Less prone to false breakouts in trending markets.
      Often used together with Bollinger Bands:
        If BB is inside Keltner → 'squeeze' = low volatility, breakout expected.

    Args:
        df (pd.DataFrame): Must have 'high', 'low', 'close'.
        ema_period (int):  EMA period for middle line (default 20).
        atr_period (int):  ATR period for band width (default 10).
        multiplier (float): ATR multiplier (default 2.0).

    Returns:
        pd.DataFrame with columns: ['kc_upper', 'kc_middle', 'kc_lower']
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must have '{col}' column.")

    middle      = ema(df["close"], ema_period)
    atr_val     = atr(df, atr_period)
    upper       = middle + multiplier * atr_val
    lower       = middle - multiplier * atr_val

    return pd.DataFrame({
        "kc_upper":  upper,
        "kc_middle": middle,
        "kc_lower":  lower,
    }, index=df.index)


# ===========================================================================
# VOLUME INDICATORS
# ===========================================================================

def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price (VWAP).

    Theory:
      VWAP = Cumulative(Price × Volume) / Cumulative(Volume)
      where Price = (high + low + close) / 3  (Typical Price)

      VWAP resets to zero at the start of EACH trading day.
      Institutions use VWAP as a benchmark — executing above VWAP (buy) or
      below VWAP (sell) is considered a good execution.

      Signals:
        Price > VWAP → bullish bias (market paying above average)
        Price < VWAP → bearish bias

      IMPORTANT: VWAP is meaningful ONLY for intraday data (minute/hour).
      It has no meaning on daily or weekly data.

    Args:
        df (pd.DataFrame): Must have 'high', 'low', 'close', 'volume' columns.
                           Index must be timezone-aware datetime (IST) so we
                           can detect day boundaries correctly.

    Returns:
        pd.Series: VWAP values that reset each day.
    """
    for col in ("high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must have '{col}' column.")

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume     = typical_price * df["volume"]

    # Group by date to reset cumulation each day
    dates         = df.index.date
    vwap_values   = pd.Series(np.nan, index=df.index, name="VWAP")

    for day in np.unique(dates):
        mask = dates == day
        day_tp_vol    = tp_volume.loc[mask]
        day_vol       = df["volume"].loc[mask]
        cum_tp_vol    = day_tp_vol.cumsum()
        cum_vol       = day_vol.cumsum()
        vwap_values.loc[mask] = cum_tp_vol / cum_vol.replace(0, np.nan)

    return vwap_values


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On Balance Volume (OBV).

    Theory:
      If close > prev_close: OBV += volume
      If close < prev_close: OBV -= volume
      If close == prev_close: OBV unchanged

      OBV is a cumulative indicator. It tracks if volume is flowing
      into (accumulation) or out of (distribution) a security.

      Signals:
        Rising OBV with rising price   → confirmed uptrend (strong)
        Rising OBV with falling price  → bullish divergence (reversal likely)
        Falling OBV with rising price  → bearish divergence (reversal likely)

    Args:
        df (pd.DataFrame): Must have 'close' and 'volume' columns.

    Returns:
        pd.Series: OBV values.
    """
    for col in ("close", "volume"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must have '{col}' column.")

    direction = np.sign(df["close"].diff(1).fillna(0))
    result    = (direction * df["volume"]).cumsum()
    result.name = "OBV"
    return result


# ===========================================================================
# STATISTICAL INDICATORS
# ===========================================================================

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Rolling Z-Score.

    Theory:
      Z = (Price - Rolling Mean) / Rolling StdDev

      A z-score measures how many standard deviations the current price is
      from the rolling mean. Used in mean reversion strategies.

      |z| > 2   → price is unusually far from mean (potential reversion)
      |z| > 3   → extreme deviation (stronger reversion signal)
      z = 0     → price is exactly at the rolling mean

      Trading idea:
        z < -2 → buy (price below mean, expect reversion up)
        z >  2 → sell/short (price above mean, expect reversion down)

    Args:
        series (pd.Series): Price or spread series.
        period (int):       Rolling window (default 20).

    Returns:
        pd.Series: Z-score values.

    Example:
        df['zscore'] = zscore(df['close'], period=20)
    """
    _validate_series(series, "series")
    _validate_period(period, "period")

    rolling = series.rolling(window=period, min_periods=period)
    mean    = rolling.mean()
    std     = rolling.std(ddof=0).replace(0, np.nan)
    result  = (series - mean) / std
    result.name = f"ZScore_{period}"
    return result


def rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    period:  int = 20,
) -> pd.Series:
    """
    Rolling Pearson Correlation between two series.

    Theory:
      Measures linear relationship between two assets over a rolling window.
      Value ranges from -1 to +1.
        +1 → perfectly positively correlated (move together)
         0 → no correlation
        -1 → perfectly negatively correlated (move opposite)

      Used in:
        - Pairs trading (find/maintain high-correlation pairs)
        - Portfolio diversification (prefer low/negative correlation)
        - Arbitrage (signal when correlation breaks down)

    Args:
        series1, series2 (pd.Series): Must have the same index.
        period (int):                 Rolling window.

    Returns:
        pd.Series: Rolling correlation values.
    """
    _validate_series(series1, "series1")
    _validate_series(series2, "series2")
    _validate_period(period, "period")

    result = series1.rolling(period).corr(series2)
    result.name = f"Corr_{period}"
    return result


# ===========================================================================
# SIGNAL GENERATION HELPERS
# ===========================================================================

def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Returns True on the bar where series1 crosses ABOVE series2.

    Use case: detect EMA crossover signals.

    Example:
        signal_buy = crossover(ema9, ema21)   # True when fast crosses above slow
    """
    _validate_series(series1, "series1")
    _validate_series(series2, "series2")

    above_now  = series1 > series2
    below_prev = series1.shift(1) <= series2.shift(1)
    return above_now & below_prev


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Returns True on the bar where series1 crosses BELOW series2.

    Example:
        signal_sell = crossunder(ema9, ema21)   # True when fast crosses below slow
    """
    _validate_series(series1, "series1")
    _validate_series(series2, "series2")

    below_now  = series1 < series2
    above_prev = series1.shift(1) >= series2.shift(1)
    return below_now & above_prev


def above_threshold(series: pd.Series, level: float) -> pd.Series:
    """Returns True when series crosses ABOVE the given level."""
    return (series > level) & (series.shift(1) <= level)


def below_threshold(series: pd.Series, level: float) -> pd.Series:
    """Returns True when series crosses BELOW the given level."""
    return (series < level) & (series.shift(1) >= level)


# ===========================================================================
# CANDLESTICK PATTERN HELPERS
# ===========================================================================

def candle_body(df: pd.DataFrame) -> pd.Series:
    """Absolute candle body size: |close - open|"""
    return (df["close"] - df["open"]).abs()


def candle_range(df: pd.DataFrame) -> pd.Series:
    """High - Low (full candle range)"""
    return df["high"] - df["low"]


def is_green(df: pd.DataFrame) -> pd.Series:
    """True if close > open (bullish candle)"""
    return df["close"] > df["open"]


def is_red(df: pd.DataFrame) -> pd.Series:
    """True if close < open (bearish candle)"""
    return df["close"] < df["open"]


def upper_shadow(df: pd.DataFrame) -> pd.Series:
    """Upper shadow length: high - max(open, close)"""
    return df["high"] - df[["open", "close"]].max(axis=1)


def lower_shadow(df: pd.DataFrame) -> pd.Series:
    """Lower shadow length: min(open, close) - low"""
    return df[["open", "close"]].min(axis=1) - df["low"]
