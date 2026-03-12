"""
indicators/statistics.py
--------------------------
Statistical indicators for mean reversion and pairs trading strategies.

INDICATORS:
    zscore(series, period)                  Rolling Z-score
    rolling_correlation(s1, s2, period)     Rolling Pearson correlation
    rolling_beta(s1, s2, period)            Rolling OLS beta
    spread(s1, s2, hedge_ratio)             Price spread between two instruments
    half_life(spread_series)               Mean reversion half-life (Ornstein-Uhlenbeck)
    cointegration_test(s1, s2)             Engle-Granger cointegration test
"""

import numpy as np
import pandas as pd


def zscore(series: pd.Series, period: int) -> pd.Series:
    """
    Rolling Z-score.

    z = (Price - Rolling Mean) / Rolling Std Dev

    Interpretation:
        z > +2  → Price is 2 standard deviations above mean → overbought
        z < -2  → Price is 2 standard deviations below mean → oversold
        z = 0   → Price is at the rolling mean

    WHY Z-SCORE: The z-score normalises price deviation regardless of the
    absolute price level. Used heavily in mean reversion and pairs trading.

    Args:
        series (pd.Series): Price or spread series.
        period (int):       Rolling window size.

    Returns:
        pd.Series: Z-score values.

    Edge cases:
        - If rolling std = 0 (all prices identical) → NaN (avoid div by zero).
        - First (period-1) values are NaN.

    Example:
        z = zscore(df["close"], 20)
        buy_signal  = z < -2
        sell_signal = z > +2
    """
    if period < 2:
        raise ValueError(f"period must be >= 2 to compute std dev, got {period}")

    roll_mean = series.rolling(window=period, min_periods=period).mean()
    roll_std  = series.rolling(window=period, min_periods=period).std(ddof=1)

    return (series - roll_mean) / roll_std.replace(0, np.nan)


def rolling_correlation(
    s1:     pd.Series,
    s2:     pd.Series,
    period: int,
) -> pd.Series:
    """
    Rolling Pearson correlation between two price series.

    Range: -1 (perfectly inverse) to +1 (perfectly correlated).

    Used for pairs trading to verify the ongoing correlation between
    two historically correlated instruments.

    Args:
        s1, s2  : Two price series with aligned indices.
        period  : Rolling window.

    Returns:
        pd.Series: Correlation values in [-1, 1].

    Edge cases:
        - If either series is constant in the window → NaN.
        - Indices must be aligned; misaligned indices will produce NaN.

    Example:
        corr = rolling_correlation(df["RELIANCE"], df["ONGC"], 60)
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    return s1.rolling(window=period, min_periods=period).corr(s2)


def rolling_beta(
    s1:     pd.Series,
    s2:     pd.Series,
    period: int,
) -> pd.Series:
    """
    Rolling OLS Beta (hedge ratio).

    Beta = Cov(s1, s2) / Var(s2)

    Used in pairs trading to determine how many units of s2 to short
    for every unit of s1 held long (the hedge ratio).

    Args:
        s1     : Dependent series (the instrument you're long).
        s2     : Independent series (the instrument you're short).
        period : Rolling window.

    Returns:
        pd.Series: Rolling beta (hedge ratio) values.

    Example:
        beta = rolling_beta(df["HDFC"], df["ICICIBANK"], 60)
        spread = df["HDFC"] - beta * df["ICICIBANK"]
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")

    cov = s1.rolling(window=period, min_periods=period).cov(s2)
    var = s2.rolling(window=period, min_periods=period).var(ddof=1)

    return cov / var.replace(0, np.nan)


def spread(
    s1:          pd.Series,
    s2:          pd.Series,
    hedge_ratio: float = 1.0,
) -> pd.Series:
    """
    Compute the price spread between two instruments.

    spread = s1 - hedge_ratio × s2

    When the spread is stationary (cointegrated), it reverts to its mean.
    The hedge_ratio is typically computed via rolling_beta().

    Args:
        s1           : Series 1 (instrument you're long).
        s2           : Series 2 (instrument you're short/hedge).
        hedge_ratio  : Number of s2 units per s1 unit. Default 1.0.

    Returns:
        pd.Series: Spread values.

    Example:
        hr     = rolling_beta(df["HDFC"], df["ICICIBANK"], 60)
        sp     = spread(df["HDFC"], df["ICICIBANK"], hr)
        z      = zscore(sp, 20)
    """
    if hedge_ratio == 0:
        raise ValueError("hedge_ratio cannot be 0")
    return s1 - hedge_ratio * s2


def half_life(spread_series: pd.Series) -> float:
    """
    Estimate the mean reversion half-life using the Ornstein-Uhlenbeck model.

    Half-life = the expected number of bars for the spread to revert
    halfway back to its mean after a deviation.

    Formula:
        Regress: Δspread_t = λ × spread_(t-1) + ε
        half_life = -ln(2) / λ

    Interpretation:
        Short half-life (e.g. 5 bars) → fast mean reversion → better for trading
        Long half-life (> 252 bars)   → very slow reversion → not tradeable

    Args:
        spread_series (pd.Series): The spread or price series to test.

    Returns:
        float: Half-life in number of bars. Returns inf if no mean reversion detected.

    Edge cases:
        - If λ >= 0 (no mean reversion) → returns infinity.
        - Needs at least 30 observations for a reliable estimate.

    Example:
        hl = half_life(sp)
        if hl < 30:
            print("Fast mean-reverting — good for pairs trading")
    """
    if len(spread_series.dropna()) < 30:
        return float("inf")

    spread_lag   = spread_series.shift(1).dropna()
    spread_diff  = spread_series.diff().dropna()

    # Align (diff removes first row, shift removes first row)
    spread_lag, spread_diff = spread_lag.align(spread_diff, join="inner")

    if len(spread_lag) < 10:
        return float("inf")

    # OLS: spread_diff = λ × spread_lag + constant
    # Using numpy polyfit for the regression
    x = spread_lag.values
    y = spread_diff.values

    # Add constant column
    X = np.column_stack([x, np.ones(len(x))])
    try:
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        lam = coefs[0]
    except np.linalg.LinAlgError:
        return float("inf")

    if lam >= 0:
        return float("inf")   # No mean reversion (λ must be negative)

    return float(-np.log(2) / lam)


def cointegration_test(
    s1: pd.Series,
    s2: pd.Series,
    significance: float = 0.05,
) -> dict:
    """
    Engle-Granger two-step cointegration test.

    Tests whether two non-stationary price series share a long-run
    equilibrium relationship (i.e., their spread is stationary).

    Step 1: OLS regression of s1 on s2 to get residuals (the spread).
    Step 2: ADF test on residuals to check for stationarity.

    Args:
        s1, s2        : Two price series (same length, aligned index).
        significance  : p-value threshold. Default 0.05 (95% confidence).

    Returns:
        dict with keys:
            is_cointegrated (bool)   : True if cointegrated at given significance.
            p_value (float)          : ADF test p-value of residuals.
            hedge_ratio (float)      : OLS regression coefficient (β).
            adf_statistic (float)    : ADF test statistic.

    NOTE: This uses statsmodels for the ADF test. If statsmodels is not
    installed, raises ImportError with installation instructions.

    Edge cases:
        - Requires at least 30 aligned observations.
        - Series must have overlapping non-NaN values.

    Example:
        result = cointegration_test(df["HDFC"], df["ICICIBANK"])
        if result["is_cointegrated"]:
            hr     = result["hedge_ratio"]
            sp     = spread(df["HDFC"], df["ICICIBANK"], hr)
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        raise ImportError(
            "statsmodels is required for cointegration_test(). "
            "Install it with: pip install statsmodels"
        )

    # Align and drop NaN
    df_aligned = pd.concat([s1, s2], axis=1).dropna()
    if len(df_aligned) < 30:
        raise ValueError(
            f"Need at least 30 observations for cointegration test, "
            f"got {len(df_aligned)}"
        )

    y = df_aligned.iloc[:, 0].values
    x = df_aligned.iloc[:, 1].values

    # Step 1: OLS regression
    X = np.column_stack([x, np.ones(len(x))])
    try:
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "is_cointegrated": False,
            "p_value":         1.0,
            "hedge_ratio":     1.0,
            "adf_statistic":   0.0,
        }

    hedge_ratio = coefs[0]
    residuals   = y - X @ coefs

    # Step 2: ADF test on residuals
    adf_result = adfuller(residuals, autolag="AIC")
    adf_stat   = float(adf_result[0])
    p_value    = float(adf_result[1])

    return {
        "is_cointegrated": p_value < significance,
        "p_value":         round(p_value, 6),
        "hedge_ratio":     round(hedge_ratio, 6),
        "adf_statistic":   round(adf_stat, 6),
    }
