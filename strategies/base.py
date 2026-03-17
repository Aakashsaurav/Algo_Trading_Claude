"""
strategies/base.py
-------------------
Base class for all trading strategies.

THEORY — HOW STRATEGIES WORK IN THIS SYSTEM:
=============================================
Every strategy is a Python class that:
  1. Receives OHLCV data (a pandas DataFrame of historical candles).
  2. Adds indicator columns to the DataFrame.
  3. Returns a 'signal' column: +1 = BUY, -1 = SELL, 0 = no action.

The Backtester engine then takes these signals and simulates trades:
  - Signal +1 on bar i → opens a LONG position at bar i+1's OPEN
    (next-bar-open slippage — the most realistic approach)
  - Signal -1 on bar i → closes the open LONG at bar i+1's OPEN
  - Short selling is supported: -1 can also open a SHORT if allowed

WHY EVENT-DRIVEN + VECTORISED BOTH:
  We compute signals vectorially (on the full DataFrame at once) for speed.
  The backtester READS these signals one bar at a time (event-driven loop)
  to correctly simulate next-bar execution and avoid look-ahead bias.

PYRAMIDING (multiple positions per symbol):
  Strategies return a signal Series where values > 1 or < -1 indicate
  adding to an existing position. The backtester manages position tracking.

CRITICAL — NO LOOK-AHEAD BIAS:
  When computing signals at bar i, you must ONLY use data available at bar i.
  Common mistakes to avoid:
    ✗ Using df['close'].shift(-1) (future close)
    ✗ normalising with min/max of the full dataset
    ✓ Use .shift(1) when you need previous bar's value in a comparison
    ✓ All rolling calculations look backward by design

WRITING YOUR OWN STRATEGY:
  1. Subclass BaseStrategy
  2. Implement generate_signals(df) → returns the df with 'signal' column added
  3. Optional: override get_parameters() to document configurable params
  4. Pass your class to BacktestEngine.run()

  class MyStrategy(BaseStrategy):
      def __init__(self, fast=9, slow=21):
          super().__init__(name="My EMA Cross", description="...")
          self.fast = fast
          self.slow = slow

      def generate_signals(self, df):
          df = df.copy()
          df['fast_ema'] = ema(df['close'], self.fast)
          df['slow_ema'] = ema(df['close'], self.slow)
          df['signal']   = 0
          df.loc[crossover(df['fast_ema'], df['slow_ema']), 'signal'] = 1
          df.loc[crossunder(df['fast_ema'], df['slow_ema']), 'signal'] = -1
          return df
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from indicators.technical import (
    sma, ema, dema, macd, supertrend,
    rsi, stochastic, roc,
    atr, bollinger_bands, keltner_channels,
    vwap, obv,
    zscore, rolling_correlation,
    crossover, crossunder, above_threshold, below_threshold,
    is_green, is_red,
)

logger = logging.getLogger(__name__)

# Minimum number of bars required before strategy should start generating signals.
# Signals before this bar index are forced to 0 to prevent warm-up artifacts.
MIN_WARMUP_BARS = 50


# ===========================================================================
# Base Strategy
# ===========================================================================

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies MUST implement generate_signals().
    All other methods have sensible defaults that can be overridden.
    """

    def __init__(
        self,
        name:        str = "Unnamed Strategy",
        description: str = "",
    ) -> None:
        self.name        = name
        self.description = description
        self.logger      = logging.getLogger(f"strategy.{self.__class__.__name__}")

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute indicators and return signals.

        Args:
            df (pd.DataFrame): OHLCV DataFrame with columns:
                               open, high, low, close, volume, oi
                               Index: timezone-aware timestamp (IST).

        Returns:
            pd.DataFrame: Same df with additional columns including
                          'signal': int column where:
                            +1 = Buy / Go Long
                            -1 = Sell / Go Short (or close long)
                             0 = No action
                          May also include indicator columns for charting.

        CRITICAL: Call super()._validate_and_prepare(df) at the start
        to get a validated, copied DataFrame.
        """
        ...

    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the input DataFrame and return a safe copy.
        Call this at the start of every generate_signals() implementation.
        """
        required = ["open", "high", "low", "close", "volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Strategy '{self.name}': DataFrame is missing columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        if len(df) < MIN_WARMUP_BARS:
            self.logger.warning(
                f"Only {len(df)} bars available. Strategy needs at least "
                f"{MIN_WARMUP_BARS} bars for reliable signals. "
                "Results may be unreliable."
            )

        df = df.copy()
        df["signal"] = 0   # Default: no action on every bar
        return df

    def _suppress_warmup_signals(
        self,
        df:         pd.DataFrame,
        warmup_bars: int,
    ) -> pd.DataFrame:
        """
        Zero out signals during the warm-up period.

        The first `warmup_bars` rows are NaN for most indicators.
        Any signals there would be based on incomplete data and should
        be suppressed to avoid false trades at the start of a backtest.
        """
        if warmup_bars > 0 and len(df) > warmup_bars:
            df.iloc[:warmup_bars, df.columns.get_loc("signal")] = 0
        return df

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return a dict of strategy parameters. Override in subclasses.
        Used by the backtester for logging and report generation.
        """
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ===========================================================================
# Strategy 1: EMA Crossover (Momentum / Trend Following)
# ===========================================================================

class EMACrossover(BaseStrategy):
    """
    Dual EMA Crossover Strategy.

    THEORY:
      Two EMAs of different periods are computed. When the faster EMA
      crosses above the slower EMA, it signals upward momentum → BUY.
      When the fast EMA crosses below the slow EMA → SELL.

      Classic setup: 9/21, 20/50, or 50/200 (the 'Golden Cross').

    ENTRY:
      Long: fast_ema crosses above slow_ema → signal = +1
    EXIT:
      Close long: fast_ema crosses below slow_ema → signal = -1
      (if shorting allowed: open short on the same bar)

    PARAMETERS:
      fast_period: Short EMA (default 9)
      slow_period: Long EMA (default 21)

    INDICATOR COLUMNS ADDED:
      'ema_fast', 'ema_slow'

    EXAMPLE:
      strategy = EMACrossover(fast_period=9, slow_period=21)
      df_with_signals = strategy.generate_signals(df)
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
    ) -> None:
        super().__init__(
            name=f"EMA Crossover ({fast_period}/{slow_period})",
            description=(
                f"Buy when EMA{fast_period} crosses above EMA{slow_period}. "
                f"Sell when EMA{fast_period} crosses below EMA{slow_period}."
            ),
        )
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than "
                f"slow_period ({slow_period})."
            )
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_and_prepare(df)

        df["ema_fast"] = ema(df["close"], self.fast_period)
        df["ema_slow"] = ema(df["close"], self.slow_period)

        df.loc[crossover(df["ema_fast"],  df["ema_slow"]), "signal"] = 1
        df.loc[crossunder(df["ema_fast"], df["ema_slow"]), "signal"] = -1

        return self._suppress_warmup_signals(df, self.slow_period)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name":        self.name,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
        }


# ===========================================================================
# Strategy 2: RSI Mean Reversion
# ===========================================================================

class RSIMeanReversion(BaseStrategy):
    """
    RSI Oversold/Overbought Mean Reversion Strategy.

    THEORY:
      When RSI drops below the oversold level (e.g. 30), the asset has
      likely been sold too aggressively and should revert upward → BUY.
      When RSI rises above the overbought level (e.g. 70), the asset has
      been bought too aggressively → SELL.

      Adding an SMA filter (only buy if price > SMA) avoids buying into
      persistent downtrends (catching falling knives).

    ENTRY:
      Long: RSI crosses above oversold_level while price > SMA(sma_filter_period)
    EXIT:
      Close long: RSI crosses below overbought_level

    PARAMETERS:
      rsi_period:          RSI lookback (default 14)
      oversold_level:      RSI threshold for buy signal (default 30)
      overbought_level:    RSI threshold for sell signal (default 70)
      sma_filter_period:   Trend filter — only buy above this SMA (default 200)
                           Set to 0 to disable the filter.

    INDICATOR COLUMNS ADDED:
      'rsi', 'sma_filter'
    """

    def __init__(
        self,
        rsi_period:         int   = 14,
        oversold_level:     float = 30.0,
        overbought_level:   float = 70.0,
        sma_filter_period:  int   = 200,
    ) -> None:
        super().__init__(
            name=f"RSI Mean Reversion ({rsi_period})",
            description=(
                f"Buy when RSI({rsi_period}) crosses above {oversold_level} "
                f"(oversold). Sell when RSI crosses below {overbought_level} "
                f"(overbought). Trend filter: SMA({sma_filter_period})."
            ),
        )
        self.rsi_period        = rsi_period
        self.oversold_level    = oversold_level
        self.overbought_level  = overbought_level
        self.sma_filter_period = sma_filter_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_and_prepare(df)

        df["rsi"] = rsi(df["close"], self.rsi_period)

        # SMA trend filter (if period > 0)
        if self.sma_filter_period > 0:
            df["sma_filter"] = sma(df["close"], self.sma_filter_period)
            trend_ok = df["close"] > df["sma_filter"]
        else:
            df["sma_filter"] = np.nan
            trend_ok = pd.Series(True, index=df.index)

        # Buy: RSI crossing back above oversold level (with trend filter)
        rsi_buy  = above_threshold(df["rsi"], self.oversold_level)
        # Sell: RSI crossing below overbought level
        rsi_sell = below_threshold(df["rsi"], self.overbought_level)

        df.loc[rsi_buy  & trend_ok, "signal"] = 1
        df.loc[rsi_sell,            "signal"] = -1

        warmup = max(self.rsi_period, self.sma_filter_period)
        return self._suppress_warmup_signals(df, warmup)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name":              self.name,
            "rsi_period":        self.rsi_period,
            "oversold_level":    self.oversold_level,
            "overbought_level":  self.overbought_level,
            "sma_filter_period": self.sma_filter_period,
        }


# ===========================================================================
# Strategy 3: Bollinger Band Mean Reversion / Breakout
# ===========================================================================

class BollingerBandStrategy(BaseStrategy):
    """
    Bollinger Band Strategy — configurable for mean reversion OR breakout.

    THEORY:
      Bollinger Bands define a dynamic range of 'normal' price behaviour.

      Mean Reversion mode (mode='reversion'):
        When price touches the lower band, it's oversold → BUY.
        When price touches the upper band, it's overbought → SELL.
        Works well in sideways/ranging markets.

      Breakout mode (mode='breakout'):
        When price CLOSES ABOVE the upper band, it's breaking out → BUY.
        When price CLOSES BELOW the lower band → SELL.
        Works well in trending markets after a 'squeeze' (narrow bands).

    PARAMETERS:
      period:     Bollinger Band SMA period (default 20)
      std_dev:    Standard deviation multiplier (default 2.0)
      mode:       'reversion' or 'breakout' (default 'reversion')

    INDICATOR COLUMNS ADDED:
      'bb_upper', 'bb_middle', 'bb_lower', 'bb_pct_b', 'bb_bandwidth'
    """

    def __init__(
        self,
        period:  int   = 20,
        std_dev: float = 2.0,
        mode:    str   = "reversion",
    ) -> None:
        if mode not in ("reversion", "breakout"):
            raise ValueError("mode must be 'reversion' or 'breakout'.")
        super().__init__(
            name=f"Bollinger Band {mode.title()} ({period}, {std_dev}σ)",
            description=(
                f"Bollinger Bands ({period} period, {std_dev}σ). "
                f"Mode: {mode}."
            ),
        )
        self.period  = period
        self.std_dev = std_dev
        self.mode    = mode

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_and_prepare(df)

        bb = bollinger_bands(df["close"], self.period, self.std_dev)
        df["bb_upper"]     = bb["bb_upper"]
        df["bb_middle"]    = bb["bb_middle"]
        df["bb_lower"]     = bb["bb_lower"]
        df["bb_pct_b"]     = bb["bb_pct_b"]
        df["bb_bandwidth"] = bb["bb_bandwidth"]

        if self.mode == "reversion":
            # Buy when price touches/crosses below lower band
            df.loc[df["close"] <= df["bb_lower"], "signal"] = 1
            # Sell when price touches/crosses above upper band
            df.loc[df["close"] >= df["bb_upper"], "signal"] = -1

        elif self.mode == "breakout":
            # Breakout: buy when close CROSSES above upper band
            df.loc[crossover(df["close"], df["bb_upper"]), "signal"] = 1
            # Sell: close CROSSES below lower band
            df.loc[crossunder(df["close"], df["bb_lower"]), "signal"] = -1

        return self._suppress_warmup_signals(df, self.period)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name":    self.name,
            "period":  self.period,
            "std_dev": self.std_dev,
            "mode":    self.mode,
        }


# ===========================================================================
# Strategy 4: MACD Signal Line Crossover
# ===========================================================================

class MACDStrategy(BaseStrategy):
    """
    MACD Signal Line Crossover Strategy.

    THEORY:
      MACD Histogram = MACD Line - Signal Line
      When histogram crosses above 0 (MACD crosses above Signal) → BUY.
      When histogram crosses below 0 (MACD crosses below Signal) → SELL.

      Optional: add RSI filter to avoid trading in weak momentum conditions.

    PARAMETERS:
      fast_period:    Fast EMA period (default 12)
      slow_period:    Slow EMA period (default 26)
      signal_period:  Signal line EMA period (default 9)
      rsi_filter:     Only buy if RSI > this level (default 0 = disabled)

    INDICATOR COLUMNS ADDED:
      'macd', 'macd_signal', 'macd_histogram'
    """

    def __init__(
        self,
        fast_period:   int = 12,
        slow_period:   int = 26,
        signal_period: int = 9,
        rsi_filter:    int = 0,    # 0 = disabled; e.g. 40 = only buy if RSI > 40
    ) -> None:
        super().__init__(
            name=f"MACD ({fast_period}/{slow_period}/{signal_period})",
            description=(
                f"Buy when MACD({fast_period}) crosses above Signal({signal_period}). "
                f"Sell when MACD crosses below Signal."
            ),
        )
        self.fast_period   = fast_period
        self.slow_period   = slow_period
        self.signal_period = signal_period
        self.rsi_filter    = rsi_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_and_prepare(df)

        macd_df = macd(df["close"], self.fast_period, self.slow_period, self.signal_period)
        df["macd"]           = macd_df["macd"]
        df["macd_signal"]    = macd_df["signal"]
        df["macd_histogram"] = macd_df["histogram"]

        # RSI filter (optional)
        if self.rsi_filter > 0:
            df["rsi_filter"] = rsi(df["close"], 14)
            rsi_ok = df["rsi_filter"] > self.rsi_filter
        else:
            rsi_ok = pd.Series(True, index=df.index)

        # Buy: histogram crosses above 0
        buy_signal  = above_threshold(df["macd_histogram"], 0)
        # Sell: histogram crosses below 0
        sell_signal = below_threshold(df["macd_histogram"], 0)

        df.loc[buy_signal  & rsi_ok, "signal"] = 1
        df.loc[sell_signal,          "signal"] = -1

        warmup = self.slow_period + self.signal_period
        return self._suppress_warmup_signals(df, warmup)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name":          self.name,
            "fast_period":   self.fast_period,
            "slow_period":   self.slow_period,
            "signal_period": self.signal_period,
            "rsi_filter":    self.rsi_filter,
        }


# ===========================================================================
# Strategy 5: Supertrend
# ===========================================================================

class SupertrendStrategy(BaseStrategy):
    """
    Supertrend Trend-Following Strategy.

    THEORY:
      The Supertrend indicator uses ATR to dynamically place a trailing
      stop-loss line above or below price depending on the trend direction.

      When Supertrend direction flips from -1 to +1 (downtrend → uptrend):
        The line moves from above price to below price → BUY signal.
      When direction flips from +1 to -1:
        The line moves from below price to above price → SELL signal.

      The Supertrend line itself acts as a dynamic stop-loss:
      - In a long trade, set stop = Supertrend line value.

      Works best on trending instruments. Prone to whipsaws in ranging markets.

    PARAMETERS:
      period:      ATR period (default 10)
      multiplier:  ATR multiplier for bands (default 3.0)

    INDICATOR COLUMNS ADDED:
      'supertrend', 'st_direction'
    """

    def __init__(
        self,
        period:     int   = 10,
        multiplier: float = 3.0,
    ) -> None:
        super().__init__(
            name=f"Supertrend ({period}, {multiplier}x)",
            description=(
                f"Buy when Supertrend({period}, {multiplier}x) flips bullish. "
                f"Sell when it flips bearish."
            ),
        )
        self.period     = period
        self.multiplier = multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_and_prepare(df)

        st = supertrend(df, self.period, self.multiplier)
        df["supertrend"]  = st["supertrend"]
        df["st_direction"] = st["direction"]

        df.loc[st["buy_signal"],  "signal"] = 1
        df.loc[st["sell_signal"], "signal"] = -1

        return self._suppress_warmup_signals(df, self.period * 2)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name":       self.name,
            "period":     self.period,
            "multiplier": self.multiplier,
        }
