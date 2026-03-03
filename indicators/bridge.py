"""
indicators/bridge.py
---------------------
Universal indicator bridge — use ANY indicator library in your strategies.

THEORY — WHY A BRIDGE?
========================
Our system has its own pure-numpy indicators in indicators/technical.py.
But the world has thousands of battle-tested indicators in libraries like:
  • pandas-ta  — 130+ indicators, pip install pandas-ta
  • TA-Lib     — C-based, blazing fast, pip install TA-Lib
  • ta          — lightweight, pip install ta

This bridge lets your strategy use indicators from ANY of these sources
transparently, with a uniform interface and graceful fallback:

  1. If the requested library is available → use it directly
  2. If not available → fall back to our built-in indicators/technical.py
  3. If no fallback exists → raise a clear, actionable error

USAGE IN A STRATEGY:
    from indicators.bridge import IndicatorBridge
    bridge = IndicatorBridge()

    # Will use pandas-ta if installed, else our built-in:
    rsi = bridge.rsi(df['close'], 14)

    # Explicitly use pandas-ta:
    rsi = bridge.rsi(df['close'], 14, library='pandas_ta')

    # Explicitly use talib:
    atr = bridge.atr(df, 14, library='talib')

    # Use any pandas-ta indicator by name:
    result = bridge.pandas_ta_indicator(df, 'bbands', length=20, std=2)

    # Use any talib indicator by name:
    result = bridge.talib_indicator(df['close'], 'DEMA', timeperiod=20)

    # Register your own custom indicator:
    def my_custom_rsi(close, period):
        # your logic here
        return ...
    bridge.register('my_rsi', my_custom_rsi)
    rsi = bridge.call('my_rsi', df['close'], 14)
"""

import importlib
import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Library Detection
# =============================================================================

def _is_available(library_name: str) -> bool:
    """Check if a library is installed without importing it (fast check)."""
    try:
        importlib.import_module(library_name)
        return True
    except ImportError:
        return False


class LibraryStatus:
    """
    Checks and caches the availability of external indicator libraries.
    This is evaluated once at startup — no repeated import overhead.
    """
    _cache: Dict[str, bool] = {}

    @classmethod
    def check(cls, name: str) -> bool:
        if name not in cls._cache:
            cls._cache[name] = _is_available(name)
            if cls._cache[name]:
                logger.debug(f"Library available: {name}")
            else:
                logger.debug(f"Library NOT available: {name} (will use fallback)")
        return cls._cache[name]

    @classmethod
    def summary(cls) -> str:
        """Print a summary of which indicator libraries are available."""
        lines = ["", "=" * 50, "  INDICATOR LIBRARY STATUS", "=" * 50]
        for lib in ["pandas_ta", "talib", "ta"]:
            status = "✅ Available" if cls.check(lib) else "❌ Not installed"
            lines.append(f"  {lib:<15}: {status}")
        lines.append("  (built-in)     : ✅ Always available")
        lines.append("=" * 50)
        return "\n".join(lines)


# =============================================================================
# IndicatorBridge — main class
# =============================================================================

class IndicatorBridge:
    """
    Universal indicator bridge.

    Provides:
      - Built-in wrappers for common indicators (rsi, ema, atr, etc.)
        with library fallback chain: preferred → pandas_ta → talib → built-in
      - Direct access to pandas_ta / talib by indicator name
      - Custom indicator registration
    """

    def __init__(self, preferred_library: str = "auto"):
        """
        Args:
            preferred_library: 'auto' (detect best available), 'pandas_ta',
                               'talib', 'built_in', or 'ta'
        """
        self.preferred = preferred_library
        self._custom: Dict[str, Callable] = {}   # User-registered indicators
        self._log_library_status()

    def _log_library_status(self):
        """Log which libraries are available at construction time."""
        available = []
        for lib in ["pandas_ta", "talib", "ta"]:
            if LibraryStatus.check(lib):
                available.append(lib)
        available.append("built_in")
        logger.info(f"IndicatorBridge: available libraries = {available}")

    def _best_library(self) -> str:
        """Return the best available library based on preference."""
        if self.preferred != "auto":
            return self.preferred
        for lib in ["pandas_ta", "talib", "built_in"]:
            if lib == "built_in" or LibraryStatus.check(lib):
                return lib
        return "built_in"

    # -------------------------------------------------------------------------
    # Custom indicator registration
    # -------------------------------------------------------------------------

    def register(self, name: str, func: Callable) -> None:
        """
        Register a custom indicator function.

        Args:
            name: Name to call it by (e.g. 'my_rsi')
            func: Callable that accepts (data, *args, **kwargs) and returns Series

        Example:
            bridge.register('zscore_rsi', lambda c, p: zscore(rsi(c, p), 20))
            values = bridge.call('zscore_rsi', df['close'], 14)
        """
        self._custom[name] = func
        logger.info(f"Registered custom indicator: '{name}'")

    def call(self, name: str, *args, **kwargs) -> Any:
        """
        Call a registered custom indicator by name.

        Raises:
            KeyError: if the indicator name is not registered
        """
        if name not in self._custom:
            raise KeyError(
                f"Custom indicator '{name}' not registered. "
                f"Available: {list(self._custom.keys())}"
            )
        return self._custom[name](*args, **kwargs)

    # -------------------------------------------------------------------------
    # Direct library access
    # -------------------------------------------------------------------------

    def pandas_ta_indicator(
        self,
        df:        pd.DataFrame,
        indicator: str,
        **kwargs
    ) -> Any:
        """
        Call any pandas-ta indicator directly by name.

        Args:
            df:        OHLCV DataFrame (pandas-ta uses df.ta accessor)
            indicator: Indicator name string (e.g. 'bbands', 'ema', 'rsi')
            **kwargs:  Parameters passed to the indicator

        Returns:
            DataFrame or Series with indicator values

        Raises:
            ImportError: if pandas-ta is not installed

        Example:
            bb = bridge.pandas_ta_indicator(df, 'bbands', length=20, std=2)
            df['bb_upper'] = bb['BBU_20_2.0']
        """
        if not LibraryStatus.check("pandas_ta"):
            raise ImportError(
                "pandas-ta is not installed. Run: pip install pandas-ta\n"
                "Or use bridge.rsi() which falls back to built-in automatically."
            )
        import pandas_ta as pta
        func = getattr(pta, indicator, None)
        if func is None:
            raise ValueError(
                f"pandas-ta does not have indicator: '{indicator}'. "
                f"See: https://github.com/twopirllc/pandas-ta#indicators"
            )
        return func(
            close=df.get("close"),
            high=df.get("high"),
            low=df.get("low"),
            open_=df.get("open"),
            volume=df.get("volume"),
            **kwargs
        )

    def talib_indicator(
        self,
        data,
        indicator: str,
        **kwargs
    ) -> Any:
        """
        Call any TA-Lib indicator directly by name.

        Args:
            data:      Series (for single-input) or DataFrame (for OHLCV)
            indicator: TA-Lib function name (e.g. 'RSI', 'BBANDS', 'ATR')
            **kwargs:  Parameters passed to the function

        Returns:
            ndarray or tuple of ndarrays

        Raises:
            ImportError: if TA-Lib is not installed

        Example:
            upper, mid, lower = bridge.talib_indicator(df, 'BBANDS', timeperiod=20)
        """
        if not LibraryStatus.check("talib"):
            raise ImportError(
                "TA-Lib is not installed. Run: pip install TA-Lib\n"
                "(Note: TA-Lib requires C library — see talib.github.io for setup)"
            )
        import talib
        func = getattr(talib, indicator.upper(), None)
        if func is None:
            raise ValueError(
                f"TA-Lib does not have function: '{indicator}'. "
                f"See: https://ta-lib.github.io/ta-lib-python/doc_index.html"
            )
        if isinstance(data, pd.DataFrame):
            return func(
                data.get("close", data.iloc[:, 0]).values.astype(float),
                **kwargs
            )
        return func(data.values.astype(float) if hasattr(data, "values") else data,
                    **kwargs)

    # -------------------------------------------------------------------------
    # Common indicators with library fallback chain
    # -------------------------------------------------------------------------

    def rsi(self, close: pd.Series, period: int = 14,
            library: str = "auto") -> pd.Series:
        """
        RSI — Relative Strength Index.

        Fallback chain: pandas_ta → talib → built_in
        """
        lib = library if library != "auto" else self._best_library()

        if lib == "pandas_ta" and LibraryStatus.check("pandas_ta"):
            import pandas_ta as pta
            result = pta.rsi(close, length=period)
            return result.rename(f"RSI_{period}")

        if lib == "talib" and LibraryStatus.check("talib"):
            import talib
            arr = talib.RSI(close.values.astype(float), timeperiod=period)
            return pd.Series(arr, index=close.index, name=f"RSI_{period}")

        # Built-in fallback
        from indicators.technical import rsi as _rsi
        return _rsi(close, period)

    def ema(self, close: pd.Series, period: int,
            library: str = "auto") -> pd.Series:
        """EMA — Exponential Moving Average."""
        lib = library if library != "auto" else self._best_library()

        if lib == "pandas_ta" and LibraryStatus.check("pandas_ta"):
            import pandas_ta as pta
            return pta.ema(close, length=period).rename(f"EMA_{period}")

        if lib == "talib" and LibraryStatus.check("talib"):
            import talib
            arr = talib.EMA(close.values.astype(float), timeperiod=period)
            return pd.Series(arr, index=close.index, name=f"EMA_{period}")

        from indicators.technical import ema as _ema
        return _ema(close, period)

    def sma(self, close: pd.Series, period: int,
            library: str = "auto") -> pd.Series:
        """SMA — Simple Moving Average."""
        lib = library if library != "auto" else self._best_library()

        if lib == "pandas_ta" and LibraryStatus.check("pandas_ta"):
            import pandas_ta as pta
            return pta.sma(close, length=period).rename(f"SMA_{period}")

        if lib == "talib" and LibraryStatus.check("talib"):
            import talib
            arr = talib.SMA(close.values.astype(float), timeperiod=period)
            return pd.Series(arr, index=close.index, name=f"SMA_{period}")

        from indicators.technical import sma as _sma
        return _sma(close, period)

    def atr(self, df: pd.DataFrame, period: int = 14,
            library: str = "auto") -> pd.Series:
        """ATR — Average True Range."""
        lib = library if library != "auto" else self._best_library()

        if lib == "pandas_ta" and LibraryStatus.check("pandas_ta"):
            import pandas_ta as pta
            return pta.atr(df["high"], df["low"], df["close"], length=period)

        if lib == "talib" and LibraryStatus.check("talib"):
            import talib
            arr = talib.ATR(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                timeperiod=period,
            )
            return pd.Series(arr, index=df.index, name=f"ATR_{period}")

        from indicators.technical import atr as _atr
        return _atr(df, period)

    def macd(self, close: pd.Series, fast: int = 12, slow: int = 26,
             signal: int = 9, library: str = "auto") -> pd.DataFrame:
        """MACD — Returns DataFrame with macd, signal, histogram columns."""
        lib = library if library != "auto" else self._best_library()

        if lib == "pandas_ta" and LibraryStatus.check("pandas_ta"):
            import pandas_ta as pta
            result = pta.macd(close, fast=fast, slow=slow, signal=signal)
            # pandas-ta returns columns like MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            cols = result.columns.tolist()
            result.columns = ["macd", "histogram", "signal"]
            return result

        if lib == "talib" and LibraryStatus.check("talib"):
            import talib
            m, s, h = talib.MACD(
                close.values.astype(float),
                fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            return pd.DataFrame({
                "macd":      m, "signal": s, "histogram": h
            }, index=close.index)

        from indicators.technical import macd as _macd
        return _macd(close, fast, slow, signal)

    def bollinger_bands(self, close: pd.Series, period: int = 20,
                        std_dev: float = 2.0,
                        library: str = "auto") -> pd.DataFrame:
        """Bollinger Bands — Returns DataFrame with bb_upper, bb_middle, bb_lower."""
        lib = library if library != "auto" else self._best_library()

        if lib == "pandas_ta" and LibraryStatus.check("pandas_ta"):
            import pandas_ta as pta
            result = pta.bbands(close, length=period, std=std_dev)
            result.columns = ["bb_lower", "bb_middle", "bb_upper",
                               "bb_bandwidth", "bb_pct_b"]
            return result

        if lib == "talib" and LibraryStatus.check("talib"):
            import talib
            upper, mid, lower = talib.BBANDS(
                close.values.astype(float),
                timeperiod=period,
                nbdevup=std_dev, nbdevdn=std_dev,
            )
            df = pd.DataFrame({
                "bb_upper": upper, "bb_middle": mid, "bb_lower": lower
            }, index=close.index)
            df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_pct_b"]     = ((close - df["bb_lower"]) /
                                   (df["bb_upper"] - df["bb_lower"]))
            return df

        from indicators.technical import bollinger_bands as _bb
        return _bb(close, period, std_dev)

    def supertrend(self, df: pd.DataFrame, period: int = 10,
                   multiplier: float = 3.0,
                   library: str = "auto") -> pd.DataFrame:
        """Supertrend — always uses built-in (pandas-ta version differs)."""
        from indicators.technical import supertrend as _st
        return _st(df, period, multiplier)

    def vwap(self, df: pd.DataFrame, library: str = "auto") -> pd.Series:
        """VWAP — Volume Weighted Average Price (daily reset)."""
        if library == "pandas_ta" and LibraryStatus.check("pandas_ta"):
            import pandas_ta as pta
            return pta.vwap(df["high"], df["low"], df["close"], df["volume"])

        from indicators.technical import vwap as _vwap
        return _vwap(df)

    def stochastic(self, df: pd.DataFrame, k_period: int = 14,
                   d_period: int = 3, library: str = "auto") -> pd.DataFrame:
        """Stochastic Oscillator — %K and %D."""
        if library == "talib" and LibraryStatus.check("talib"):
            import talib
            k, d = talib.STOCH(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                fastk_period=k_period, slowd_period=d_period,
            )
            return pd.DataFrame({
                "stoch_k": k, "stoch_d": d
            }, index=df.index)

        from indicators.technical import stochastic as _stoch
        return _stoch(df, k_period, d_period)

    def zscore(self, series: pd.Series, period: int = 20) -> pd.Series:
        """Rolling Z-Score — always uses built-in."""
        from indicators.technical import zscore as _z
        return _z(series, period)

    # -------------------------------------------------------------------------
    # Status helpers
    # -------------------------------------------------------------------------

    def library_status(self) -> str:
        """Print a formatted summary of which indicator libraries are available."""
        return LibraryStatus.summary()

    def list_custom(self) -> list:
        """List names of all registered custom indicators."""
        return list(self._custom.keys())
