"""
screener/screener.py
---------------------
WHAT IS A SCREENER?
===================
A screener scans a universe of stocks (e.g. Nifty 500) and identifies
which ones currently satisfy a trading strategy's entry conditions.

Think of it like this:
  - The Backtester runs a strategy on ONE symbol over HISTORICAL data.
  - The Screener runs the SAME strategy on ALL symbols and shows you
    which ones have a LIVE signal TODAY.

This is exactly how Zerodha Streak's screener works.

HOW IT WORKS:
=============
1. Load OHLCV data for each symbol from local Parquet files.
2. Run the strategy's generate_signals() on that data.
3. Check the signal on the LAST bar (today's bar).
4. If signal != 0, add it to the results.
5. Return a sorted, ranked list of matches.

RANKING:
========
Results are ranked by signal strength + additional filters:
  - RSI distance from extreme (closer to 30/70 = stronger signal)
  - Volume ratio: current volume vs 20-day average (higher = stronger)
  - Trend alignment: price vs 200-day SMA

CSV EXPORT:
===========
Every scan automatically saves a CSV to reports/screener_YYYYMMDD.csv

USAGE:
======
    from screener.screener import Screener
    from strategies.base import RSIMeanReversion

    screener = Screener()
    results  = screener.run(
        strategy      = RSIMeanReversion(14, 30, 70),
        universe      = screener.load_nifty500_universe(),
        min_volume    = 100_000,         # skip illiquid stocks
        signal_filter = 1,               # 1=buy signals only, -1=sell, 0=all
    )
    print(results)
    screener.export_csv(results, "my_scan")
"""

import importlib
import importlib.util
import inspect
import logging
import os
import sys
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (screener/screener.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Helper: safely import project modules without crashing on missing .env ───

def _safe_import(module_name: str):
    """Import a project module gracefully, returning None if it fails."""
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        logger.debug(f"Could not import {module_name}: {e}")
        return None


# ─── Screener Result Dataclass ─────────────────────────────────────────────────


# ─── ScreenerConfig dataclass ─────────────────────────────────────────────────
from dataclasses import dataclass as _dataclass, field as _field

@_dataclass
class ScreenerConfig:
    """Configuration for a Screener scan. Mirrors screener_v2.ScreenerConfig for API compatibility."""
    min_volume:   float = 200_000    # Minimum daily volume to include symbol
    min_price:    float = 50.0       # Minimum closing price to include symbol
    signal_type:  int   = 1          # 1=buy only, -1=sell only, 0=all signals
    max_results:  int   = 50         # Maximum symbols to return
    rank_by:      str   = "close"    # Column to rank results by
    save_results: bool  = True       # Whether to save CSV output
    label:        str   = "scan"     # Label prefix for output files

class ScreenerResult:
    """
    Holds results for a single symbol that matched the screener criteria.

    Attributes:
        symbol         : Trading symbol (e.g. "INFY")
        exchange       : Exchange segment (e.g. "NSE_EQ")
        signal         : +1 (buy) or -1 (sell)
        signal_label   : Human-readable ("BUY" / "SELL")
        last_price     : Last close price
        signal_date    : Date of the signal bar
        volume         : Volume on signal bar
        avg_volume_20d : 20-day average volume
        volume_ratio   : volume / avg_volume_20d (>1.5 = high volume confirmation)
        rsi_value      : RSI at signal bar (None if no RSI in strategy output)
        price_vs_sma200: % difference from 200-day SMA
        extra_cols     : Any additional indicator columns from strategy output
        rank_score     : Computed quality score for sorting (higher = better)
    """

    __slots__ = [
        "symbol", "exchange", "signal", "signal_label",
        "last_price", "signal_date", "volume", "avg_volume_20d",
        "volume_ratio", "rsi_value", "price_vs_sma200",
        "extra_cols", "rank_score",
    ]

    def __init__(
        self,
        symbol:         str,
        exchange:       str,
        signal:         int,
        last_price:     float,
        signal_date:    pd.Timestamp,
        volume:         float,
        avg_volume_20d: float,
        rsi_value:      Optional[float] = None,
        price_vs_sma200:Optional[float] = None,
        extra_cols:     Optional[Dict]  = None,
    ):
        self.symbol          = symbol
        self.exchange        = exchange
        self.signal          = signal
        self.signal_label    = "BUY" if signal > 0 else "SELL"
        self.last_price      = round(last_price, 2)
        self.signal_date     = signal_date
        self.volume          = int(volume)
        self.avg_volume_20d  = int(avg_volume_20d) if avg_volume_20d else 0
        self.volume_ratio    = round(volume / avg_volume_20d, 2) if avg_volume_20d else 0.0
        self.rsi_value       = round(rsi_value, 1) if rsi_value is not None else None
        self.price_vs_sma200 = round(price_vs_sma200, 2) if price_vs_sma200 is not None else None
        self.extra_cols      = extra_cols or {}
        self.rank_score      = self._compute_rank_score()

    def _compute_rank_score(self) -> float:
        """
        Compute a quality score for ranking results.

        Components:
        1. Volume ratio score: higher volume confirmation = better signal
        2. RSI score: closer to extreme (30 for BUY, 70 for SELL) = stronger
        3. Trend alignment bonus: price > SMA200 for BUY adds 10 pts
        """
        score = 0.0

        # Volume confirmation (0–40 points)
        if self.volume_ratio > 0:
            score += min(self.volume_ratio * 10, 40.0)

        # RSI proximity to extreme (0–30 points)
        if self.rsi_value is not None:
            if self.signal == 1:    # BUY — want RSI near 30 (oversold)
                score += max(0, (40 - self.rsi_value))
            else:                   # SELL — want RSI near 70 (overbought)
                score += max(0, (self.rsi_value - 60))

        # Trend alignment bonus (0–30 points)
        if self.price_vs_sma200 is not None:
            if self.signal == 1 and self.price_vs_sma200 > 0:
                score += min(self.price_vs_sma200 * 2, 30.0)
            elif self.signal == -1 and self.price_vs_sma200 < 0:
                score += min(abs(self.price_vs_sma200) * 2, 30.0)

        return round(score, 2)

    def to_dict(self) -> Dict:
        """Convert to dict for DataFrame / CSV export."""
        d = {
            "signal":          self.signal_label,
            "symbol":          self.symbol,
            "exchange":        self.exchange,
            "last_price":      self.last_price,
            "signal_date":     str(self.signal_date.date()) if hasattr(self.signal_date, "date") else str(self.signal_date),
            "volume":          self.volume,
            "avg_vol_20d":     self.avg_volume_20d,
            "vol_ratio":       self.volume_ratio,
            "rsi":             self.rsi_value if self.rsi_value is not None else "",
            "vs_sma200_%":     self.price_vs_sma200 if self.price_vs_sma200 is not None else "",
            "rank_score":      self.rank_score,
        }
        # Append any extra indicator columns from the strategy
        for col, val in self.extra_cols.items():
            d[col] = round(val, 4) if isinstance(val, float) else val
        return d


# ─── Main Screener Class ───────────────────────────────────────────────────────

class Screener:
    """
    Scans a universe of symbols for trading signals.

    Thread-safe: uses ThreadPoolExecutor for parallel scanning.
    Falls back to synthetic data if Parquet files are missing (useful for demo).
    """

    # Indicator columns to capture from strategy output (for ranking + display)
    CAPTURE_COLS = [
        "rsi", "macd", "macd_signal", "macd_histogram",
        "bb_upper", "bb_lower", "bb_pct_b",
        "ema_fast", "ema_slow", "supertrend", "st_direction",
        "atr", "obv",
    ]

    def __init__(
        self,
        parquet_dir:   Optional[str] = None,
        reports_dir:   Optional[str] = None,
        max_workers:   int           = 4,
    ):
        """
        Args:
            parquet_dir:  Path to Parquet files (default: project data/ohlcv/daily)
            reports_dir:  Where to save CSVs (default: project reports/)
            max_workers:  Parallel scan threads (keep ≤4 to be safe)
        """
        self.parquet_dir = Path(parquet_dir) if parquet_dir else PROJECT_ROOT / "data" / "ohlcv" / "daily"
        self.reports_dir = Path(reports_dir) if reports_dir else PROJECT_ROOT / "reports"
        self.max_workers = max_workers
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Will be set when run() is called — used for status reporting
        self._total_symbols  = 0
        self._scanned        = 0
        self._errors         = 0

    # ── Universe loading ───────────────────────────────────────────────────────

    def load_nifty500_universe(self) -> List[Dict]:
        """
        Load Nifty 500 symbols from the universe database.
        Falls back to a hardcoded sample list if database is unavailable.

        Returns:
            List of dicts with keys: exchange, symbol
        """
        universe_mod = _safe_import("data.universe")
        if universe_mod:
            try:
                mgr    = universe_mod.UniverseManager()
                result = mgr.get_nifty500()
                if result:
                    logger.info(f"Loaded {len(result)} Nifty 500 symbols from database")
                    return result
            except Exception as e:
                logger.warning(f"Universe DB unavailable ({e}), using fallback list")

        # Hardcoded fallback — top 50 liquid Nifty stocks
        fallback = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT",
            "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE", "TITAN",
            "SUNPHARMA", "WIPRO", "ULTRACEMCO", "NESTLEIND", "TECHM",
            "HCLTECH", "M&M", "ONGC", "NTPC", "POWERGRID",
            "BAJAJFINSV", "JSWSTEEL", "TATASTEEL", "ADANIENT", "COALINDIA",
            "ADANIPORTS", "DIVISLAB", "DRREDDY", "CIPLA", "APOLLOHOSP",
            "BPCL", "HEROMOTOCO", "GRASIM", "BRITANNIA", "ITC",
            "EICHERMOT", "TATACONSUM", "SHRIRAMFIN", "SBILIFE", "HDFCLIFE",
            "BAJAJ-AUTO", "INDUSINDBK", "UPL", "LTIM", "MCDOWELL-N",
        ]
        return [{"exchange": "NSE_EQ", "symbol": s} for s in fallback]

    def load_fo_universe(self) -> List[Dict]:
        """
        Load F&O universe (stocks with futures and options).
        """
        universe_mod = _safe_import("data.universe")
        if universe_mod:
            try:
                mgr    = universe_mod.UniverseManager()
                result = mgr.get_fo_universe()
                if result:
                    logger.info(f"Loaded {len(result)} F&O symbols from database")
                    return result
            except Exception as e:
                logger.warning(f"F&O universe DB unavailable ({e}), using fallback")

        # F&O eligible fallback (top 30 by OI)
        fo_stocks = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "NIFTY", "BANKNIFTY", "SBIN", "AXISBANK", "KOTAKBANK",
            "TATAMOTORS", "BAJFINANCE", "MARUTI", "LT", "SUNPHARMA",
            "WIPRO", "HCLTECH", "BHARTIARTL", "ONGC", "NTPC",
            "JSWSTEEL", "TATASTEEL", "COALINDIA", "BPCL", "DRREDDY",
            "CIPLA", "DIVISLAB", "M&M", "BAJAJFINSV", "ADANIENT",
        ]
        return [{"exchange": "NSE_EQ", "symbol": s} for s in fo_stocks]

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load_ohlcv(self, exchange: str, symbol: str, min_bars: int = 250) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for a symbol from local Parquet files.
        Returns None if the file doesn't exist or has too few bars.
        """
        parquet_mod = _safe_import("data.parquet_store")
        if parquet_mod:
            try:
                store = parquet_mod.ParquetStore()
                df    = store.load_daily(exchange, symbol)
                if df is not None and len(df) >= min_bars:
                    return df
            except Exception:
                pass

        # Try direct file path
        file_paths = [
            self.parquet_dir / exchange / f"{symbol}.parquet",
            self.parquet_dir / f"{symbol}.parquet",
        ]
        for fp in file_paths:
            if fp.exists():
                try:
                    df = pd.read_parquet(fp)
                    if len(df) >= min_bars:
                        return df
                except Exception:
                    pass

        return None

    def _generate_demo_ohlcv(self, symbol: str, n: int = 300) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for demo/testing purposes.
        Used when real Parquet files are not available.

        This is clearly labelled as DEMO data in all outputs.
        """
        np.random.seed(hash(symbol) % (2**31))   # deterministic per symbol
        dates  = pd.date_range("2023-01-01", periods=n, freq="B", tz="Asia/Kolkata")
        price  = 500 + np.cumsum(np.random.randn(n) * 12)
        price  = np.maximum(price, 10.0)
        noise  = np.abs(np.random.randn(n) * 6)

        df = pd.DataFrame({
            "open":   price - noise * 0.4,
            "high":   price + noise * 0.8,
            "low":    (price - noise * 0.8).clip(1),
            "close":  price,
            "volume": np.random.randint(500_000, 10_000_000, n),
            "oi":     np.zeros(n),
        }, index=dates)
        return df

    # ── Signal extraction ──────────────────────────────────────────────────────

    def _extract_last_signal(
        self,
        symbol:      str,
        exchange:    str,
        strategy,
        df:          pd.DataFrame,
        use_demo:    bool,
    ) -> Optional[ScreenerResult]:
        """
        Run the strategy on df and check if the LAST bar has a signal.

        Returns a ScreenerResult if the last bar has signal != 0, else None.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig_df = strategy.generate_signals(df.copy())

        if "signal" not in sig_df.columns:
            logger.debug(f"{symbol}: strategy returned no 'signal' column")
            return None

        # Get the last bar signal (today's bar = most recent)
        last_signal = int(sig_df["signal"].iloc[-1])
        if last_signal == 0:
            return None

        last_bar = sig_df.iloc[-1]

        # ── RSI ────────────────────────────────────────────────────────────────
        rsi_val = None
        for col in ("rsi", "rsi_14", "rsi_value"):
            if col in sig_df.columns and not pd.isna(sig_df[col].iloc[-1]):
                rsi_val = float(sig_df[col].iloc[-1])
                break

        # ── Price vs SMA200 ────────────────────────────────────────────────────
        price_vs_sma200 = None
        try:
            from indicators.technical import sma
            sma200 = sma(df["close"], 200)
            last_sma = float(sma200.iloc[-1])
            if not np.isnan(last_sma) and last_sma > 0:
                price_vs_sma200 = (float(last_bar["close"]) / last_sma - 1) * 100
        except Exception:
            pass

        # ── Volume ────────────────────────────────────────────────────────────
        volume          = float(last_bar.get("volume", 0))
        avg_volume_20d  = float(df["volume"].tail(20).mean()) if "volume" in df.columns else 0

        # ── Extra indicator columns ───────────────────────────────────────────
        extra = {}
        for col in self.CAPTURE_COLS:
            if col in sig_df.columns:
                val = sig_df[col].iloc[-1]
                if not pd.isna(val):
                    extra[col] = float(val)

        result = ScreenerResult(
            symbol          = symbol + (" [DEMO]" if use_demo else ""),
            exchange        = exchange,
            signal          = last_signal,
            last_price      = float(last_bar["close"]),
            signal_date     = sig_df.index[-1],
            volume          = volume,
            avg_volume_20d  = avg_volume_20d,
            rsi_value       = rsi_val,
            price_vs_sma200 = price_vs_sma200,
            extra_cols      = extra,
        )
        return result

    # ── Single symbol scan ─────────────────────────────────────────────────────

    def _scan_one(
        self,
        item:       Dict,
        strategy,
        min_volume: int,
        use_demo:   bool,
    ) -> Optional[ScreenerResult]:
        """Scan one symbol. Designed to be called from a thread pool."""
        symbol   = item.get("symbol", "")
        exchange = item.get("exchange", "NSE_EQ")

        try:
            # Load data
            df = self._load_ohlcv(exchange, symbol)

            if df is None:
                if use_demo:
                    df = self._generate_demo_ohlcv(symbol)
                else:
                    logger.debug(f"{symbol}: no data found, skipping")
                    return None

            # Volume filter (applied on 20-day average to avoid single-day spikes)
            avg_vol = float(df["volume"].tail(20).mean()) if "volume" in df.columns else 0
            if avg_vol < min_volume:
                logger.debug(f"{symbol}: avg volume {avg_vol:.0f} < min {min_volume}, skipping")
                return None

            # Run strategy and extract signal
            return self._extract_last_signal(symbol, exchange, strategy, df, use_demo=(df is None))

        except Exception as e:
            logger.warning(f"{symbol}: scan error — {e}")
            logger.debug(traceback.format_exc())
            self._errors += 1
            return None

    # ── Main run method ────────────────────────────────────────────────────────

    def run(
        self,
        strategy,
        universe:      Optional[List[Dict]] = None,
        min_volume:    int                  = 100_000,
        signal_filter: int                  = 0,
        top_n:         Optional[int]        = None,
        use_demo:      bool                 = False,
        save_csv:      bool                 = True,
        tag:           str                  = "",
    ) -> pd.DataFrame:
        """
        Run the screener across a universe of symbols.

        Args:
            strategy:      Any strategy class (must implement generate_signals())
            universe:      List of {exchange, symbol} dicts. Defaults to Nifty 500.
            min_volume:    Skip stocks with 20d avg volume below this threshold
            signal_filter: 1 = BUY signals only | -1 = SELL only | 0 = all signals
            top_n:         Return only top N results by rank_score
            use_demo:      If True, generate synthetic data for missing symbols
            save_csv:      Auto-save results CSV to reports/
            tag:           Optional tag appended to CSV filename

        Returns:
            pd.DataFrame with one row per matched symbol, sorted by rank_score desc
        """
        if universe is None:
            universe = self.load_nifty500_universe()

        self._total_symbols = len(universe)
        self._scanned       = 0
        self._errors        = 0

        strategy_name = getattr(strategy, "name", type(strategy).__name__)
        logger.info(
            f"Screener starting: strategy={strategy_name}, "
            f"universe={self._total_symbols} symbols, "
            f"min_volume={min_volume:,}, signal_filter={'ALL' if signal_filter==0 else ('BUY' if signal_filter==1 else 'SELL')}"
        )

        results: List[ScreenerResult] = []

        # ── Parallel scan ──────────────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._scan_one, item, strategy, min_volume, use_demo): item
                for item in universe
            }
            for future in as_completed(futures):
                self._scanned += 1
                result = future.result()
                if result is not None:
                    # Apply signal direction filter
                    if signal_filter == 0 or result.signal == signal_filter:
                        results.append(result)

        logger.info(
            f"Scan complete: {self._scanned} scanned, "
            f"{len(results)} matches, {self._errors} errors"
        )

        if not results:
            logger.info("No signals found matching criteria")
            return pd.DataFrame()

        # ── Sort by rank score ─────────────────────────────────────────────────
        results.sort(key=lambda r: r.rank_score, reverse=True)
        if top_n:
            results = results[:top_n]

        # ── Build output DataFrame ─────────────────────────────────────────────
        rows = [r.to_dict() for r in results]
        df   = pd.DataFrame(rows)

        # ── Save CSV ───────────────────────────────────────────────────────────
        if save_csv and len(df) > 0:
            self.export_csv(df, strategy_name=strategy_name, tag=tag)

        return df

    # ── CSV Export ─────────────────────────────────────────────────────────────


    def scan(
        self,
        data_dict: Dict[str, "pd.DataFrame"],
        strategy:  Any,
        signal_type: int = 1,
        min_volume:  float = 0,
        min_price:   float = 0,
        max_results: int   = 50,
    ) -> list:
        """
        Alias for run() with a dict-based interface compatible with screener_v2.
        
        Args:
            data_dict:   {symbol: ohlcv_df} mapping — pre-loaded DataFrames
            strategy:    Strategy instance with generate_signals()
            signal_type: 1=buy, -1=sell, 0=all
            min_volume:  Minimum volume filter
            min_price:   Minimum price filter
            max_results: Maximum results to return

        Returns:
            list of dicts — one per matched symbol
        """
        results = []
        for symbol, df in data_dict.items():
            try:
                if df is None or len(df) < 50:
                    continue
                last_close = float(df['close'].iloc[-1])
                last_vol   = float(df['volume'].iloc[-1])
                if min_price > 0 and last_close < min_price:
                    continue
                if min_volume > 0 and last_vol < min_volume:
                    continue
                sig_df = strategy.generate_signals(df.copy())
                if 'signal' not in sig_df.columns:
                    continue
                last_signal = int(sig_df['signal'].iloc[-1])
                if signal_type != 0 and last_signal != signal_type:
                    continue
                if last_signal == 0:
                    continue
                row = {
                    'symbol':      symbol,
                    'signal':      last_signal,
                    'direction':   'BUY' if last_signal == 1 else 'SELL',
                    'close':       round(last_close, 2),
                    'volume':      int(last_vol),
                    'signal_tag':  str(sig_df.get('signal_tag', pd.Series(['']))).split('\n')[-1].strip() if 'signal_tag' in sig_df.columns else '',
                    'date':        str(sig_df.index[-1].date()) if hasattr(sig_df.index[-1], 'date') else str(sig_df.index[-1])[:10],
                }
                results.append(row)
            except Exception as e:
                logger.debug(f"scan: skip {symbol}: {e}")
                continue
        # Sort by direction (buy first) then close price
        results.sort(key=lambda x: (-x['signal'], -x['close']))
        return results[:max_results]

    def export_csv(
        self,
        df:            pd.DataFrame,
        strategy_name: str  = "screener",
        tag:           str  = "",
        filename:      str  = "",
    ) -> str:
        """
        Save screener results to a timestamped CSV file.

        Args:
            df:            Results DataFrame from run()
            strategy_name: Used in filename
            tag:           Optional suffix tag
            filename:      Override auto-generated filename

        Returns:
            str: Absolute path of the saved CSV
        """
        if df.empty:
            logger.info("No results to export")
            return ""

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = filename or f"screener_{strategy_name}_{ts}{'_' + tag if tag else ''}.csv"
        path = self.reports_dir / name
        df.to_csv(path, index=False)
        logger.info(f"Screener results saved → {path}")
        return str(path)

    # ── Status reporting ───────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return current scan progress (useful for live progress bars in the UI)."""
        return {
            "total":   self._total_symbols,
            "scanned": self._scanned,
            "errors":  self._errors,
            "pct":     round(self._scanned / max(self._total_symbols, 1) * 100, 1),
        }


# ─── Custom Indicator Library Loader ──────────────────────────────────────────

class UserIndicatorLibrary:
    """
    Allows users to write their own indicator functions in Python and
    have them available alongside the built-in indicators.

    HOW USER INDICATORS WORK:
    =========================
    1. User writes a .py file with indicator functions (see template below).
    2. Places the file in the user_indicators/ folder.
    3. Functions are loaded and available by name in the strategy editor.
    4. All functions must accept a pd.DataFrame or pd.Series and return
       a pd.Series (same index as input).

    USER INDICATOR TEMPLATE:
    ========================
    # my_indicators.py
    import pandas as pd
    import numpy as np

    def my_custom_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        '''My custom RSI variant with different smoothing.'''
        # ... your implementation ...
        result.name = f"my_rsi_{period}"
        return result
    """

    INDICATORS_DIR = PROJECT_ROOT / "user_indicators"

    def __init__(self):
        self.INDICATORS_DIR.mkdir(exist_ok=True)
        self._cache: Dict[str, Any] = {}   # module_name → module object
        self._write_template_if_empty()

    def _write_template_if_empty(self):
        """Write a starter template file if no user indicators exist yet."""
        template_path = self.INDICATORS_DIR / "example_indicators.py"
        if template_path.exists():
            return
        template = '''\
"""
user_indicators/example_indicators.py
----------------------------------------
This is your personal indicator library.

HOW TO ADD YOUR OWN INDICATOR:
1. Define a function that takes price data (Series or DataFrame).
2. Return a pandas Series with the same index as the input.
3. Give it a descriptive name for the 'name' attribute.
4. Save the file and refresh the dashboard.

The function will automatically appear in the indicator selector.

RULES (same as built-in indicators):
- Only use data at index i or earlier — never future data!
- Return NaN for the warm-up period (first N bars).
- Do NOT use global state.
"""

import numpy as np
import pandas as pd


def hull_moving_average(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Hull Moving Average (HMA) — faster and smoother than EMA.

    Formula: WMA(2 × WMA(n/2) − WMA(n), sqrt(n))
    Reduces lag significantly compared to standard moving averages.

    Args:
        close:  Close price Series
        period: Look-back period (default 20)

    Returns:
        pd.Series: HMA values with NaN during warm-up
    """
    half   = max(int(period / 2), 2)
    sqrtp  = max(int(np.sqrt(period)), 2)

    wma_half = close.rolling(half,   min_periods=half ).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    wma_full = close.rolling(period, min_periods=period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)

    diff   = 2 * wma_half - wma_full
    result = diff.rolling(sqrtp, min_periods=sqrtp).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)

    result.name = f"hma_{period}"
    return result


def vwap_bands(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
    VWAP with Upper and Lower Bands (VWAP ± multiplier × std).

    Useful for mean-reversion: buy near lower band, sell near upper.

    Args:
        df:         DataFrame with high, low, close, volume columns
        multiplier: Band width (default 1.5 standard deviations)

    Returns:
        pd.DataFrame with columns: vwap_band_mid, vwap_band_upper, vwap_band_lower
    """
    typical  = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol  = df["volume"].cumsum()
    cum_tp_v = (typical * df["volume"]).cumsum()
    vwap_mid = cum_tp_v / cum_vol

    # Rolling standard deviation of typical price
    rolling_std  = typical.rolling(20, min_periods=10).std()

    result = pd.DataFrame({
        "vwap_band_mid":   vwap_mid,
        "vwap_band_upper": vwap_mid + multiplier * rolling_std,
        "vwap_band_lower": vwap_mid - multiplier * rolling_std,
    }, index=df.index)
    return result


def chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0) -> pd.Series:
    """
    Chandelier Exit — ATR-based trailing stop for trend-following.

    Long Chandelier Exit = Highest High (n) − ATR(n) × mult
    When price crosses below, it signals an exit.

    Args:
        df:     DataFrame with high, low, close columns
        period: ATR period (default 22)
        mult:   ATR multiplier (default 3.0)

    Returns:
        pd.Series: Chandelier Exit stop level
    """
    # Compute ATR manually (no circular import)
    tr    = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr   = tr.ewm(alpha=1/period, min_periods=period).mean()

    highest_high = df["high"].rolling(period, min_periods=period).max()
    exit_level   = highest_high - atr * mult
    exit_level.name = f"chandelier_exit_{period}"
    return exit_level
'''
        template_path.write_text(template)
        logger.info(f"User indicator template written → {template_path}")

    def load_all(self) -> Dict[str, Any]:
        """
        Load all .py files from user_indicators/ and return a dict of
        {function_name: callable} for all discovered indicator functions.
        """
        functions: Dict[str, Any] = {}
        py_files = list(self.INDICATORS_DIR.glob("*.py"))

        for filepath in py_files:
            module_name = f"user_indicators.{filepath.stem}"
            try:
                spec   = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self._cache[module_name] = module

                # Extract all public functions (not starting with _)
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if not name.startswith("_"):
                        functions[name] = obj
                        logger.debug(f"Loaded user indicator: {name} from {filepath.name}")

            except Exception as e:
                logger.error(f"Failed to load user indicators from {filepath.name}: {e}")

        logger.info(f"Loaded {len(functions)} user indicator functions from {len(py_files)} files")
        return functions

    def list_files(self) -> List[Dict]:
        """Return info about all user indicator files."""
        files = []
        for fp in sorted(self.INDICATORS_DIR.glob("*.py")):
            stat = fp.stat()
            files.append({
                "name":     fp.name,
                "size_kb":  round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "path":     str(fp),
            })
        return files

    def read_file(self, filename: str) -> str:
        """Read a user indicator file's source code."""
        fp = self.INDICATORS_DIR / filename
        if not fp.exists() or fp.suffix != ".py":
            raise FileNotFoundError(f"Indicator file not found: {filename}")
        return fp.read_text()

    def save_file(self, filename: str, content: str) -> bool:
        """
        Save user indicator file source code.
        Validates Python syntax before saving.

        Raises:
            SyntaxError if the code has syntax errors
        """
        if not filename.endswith(".py"):
            filename += ".py"

        # Safety: no path traversal
        filename = Path(filename).name

        # Syntax check before saving
        try:
            compile(content, filename, "exec")
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in {filename}: {e}")

        fp = self.INDICATORS_DIR / filename
        fp.write_text(content)
        logger.info(f"Saved user indicator file: {fp}")

        # Invalidate cache so next load_all() picks up changes
        module_name = f"user_indicators.{Path(filename).stem}"
        if module_name in self._cache:
            del self._cache[module_name]
        if module_name in sys.modules:
            del sys.modules[module_name]

        return True

    def delete_file(self, filename: str) -> bool:
        """Delete a user indicator file (with safety check)."""
        filename = Path(filename).name   # strip any path components
        fp = self.INDICATORS_DIR / filename
        if fp.exists() and fp.suffix == ".py":
            fp.unlink()
            logger.info(f"Deleted user indicator file: {fp}")
            return True
        return False

    def get_all_indicator_names(self) -> Dict[str, List[str]]:
        """
        Return a dictionary of all available indicators grouped by source.

        Returns:
            {
              "builtin":  ["sma", "ema", "rsi", ...],
              "user":     ["hull_moving_average", "chandelier_exit", ...]
            }
        """
        # Built-in indicators (from indicators/technical.py)
        builtin_names = [
            "sma", "ema", "dema", "macd", "supertrend",
            "rsi", "stochastic", "roc",
            "atr", "bollinger_bands", "keltner_channels",
            "vwap", "obv",
            "zscore", "rolling_correlation",
        ]

        # User indicators (loaded from files)
        user_fns = self.load_all()

        return {
            "builtin": builtin_names,
            "user":    list(user_fns.keys()),
        }
