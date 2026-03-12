"""
screener/screener_v2.py
------------------------
Phase 4 Screener — scans the Nifty 500 / F&O universe daily for strategy signals.

THEORY — WHAT A SCREENER DOES:
================================
A backtester asks: "How would this strategy have performed in the past?"
A screener asks:   "Which stocks are showing this strategy's entry signal RIGHT NOW?"

The screener is the bridge between backtesting and live trading:
  1. Load today's OHLCV data for all Nifty 500 stocks
  2. Run each stock through the strategy's generate_signals() function
  3. Collect all stocks where signal == +1 (or -1 for shorts)
  4. Rank them by signal strength (configurable)
  5. Output: console table, CSV file, or FastAPI endpoint (for the dashboard)

KEY DESIGN CHOICES:
  • Parallel processing (ThreadPoolExecutor) — 500 stocks in ~10-15 seconds
  • Graceful error handling — one bad data file never crashes the full scan
  • Results include indicator values (so you can see WHY a stock was selected)
  • Configurable filters: min volume, min price, min ATR (avoid illiquid stocks)
  • Saves results to screener/output/ as CSV + JSON for the web dashboard

USAGE:
    from screener.screener_v2 import Screener, ScreenerConfig
    from strategies.base import RSIMeanReversion

    screener = Screener(ScreenerConfig(
        min_volume    = 500_000,   # Minimum daily volume (avoid illiquid)
        min_price     = 50.0,      # Minimum stock price
        signal_type   = 1,         # +1 = buy signals only, -1 = sell, 0 = both
        max_results   = 20,        # Return top 20 matches
        save_results  = True,
    ))

    strategy = RSIMeanReversion(14, 30, 70)
    hits     = screener.scan(data_dict, strategy)

    for hit in hits:
        print(f"{hit['symbol']}: RSI={hit['rsi']:.1f}, Close={hit['close']:.2f}")
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HERE          = Path(__file__).resolve().parent.parent
OUTPUT_SCREENER = _HERE / "screener" / "output"
OUTPUT_SCREENER.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Config
# =============================================================================

@dataclass
class ScreenerConfig:
    """
    Configuration for the screener.

    Filters (applied BEFORE running the strategy — saves compute time):
      min_volume     : Skip stocks with 20-day avg volume below this
      min_price      : Skip penny stocks below this price
      max_price      : Skip stocks above this price (optional)
      min_atr_pct    : Skip low-volatility stocks (ATR/price < threshold)
      min_bars       : Minimum number of bars required (for indicator warm-up)

    Signal settings:
      signal_type    : +1 = only buy signals, -1 = only sell, 0 = both
      max_results    : Maximum hits to return (ranked by rank_by metric)
      rank_by        : Column name to sort results by (e.g. 'rsi', 'volume')
      rank_ascending : True = lowest value first, False = highest first

    Output:
      save_results   : Save CSV + JSON to screener/output/
      label          : Prefix for output filenames
      n_workers      : Parallel workers (default 8)
    """
    min_volume:      float = 100_000
    min_price:       float = 10.0
    max_price:       float = 0.0         # 0 = no max
    min_atr_pct:     float = 0.0         # 0 = no filter
    min_bars:        int   = 100
    signal_type:     int   = 1           # +1 buy / -1 sell / 0 both
    max_results:     int   = 50
    rank_by:         str   = "close"
    rank_ascending:  bool  = True
    save_results:    bool  = True
    label:           str   = "screener"
    n_workers:       int   = 8


# =============================================================================
# Screener
# =============================================================================

class Screener:
    """
    Multi-threaded strategy screener for Nifty 500 / F&O universe.

    Scans a dictionary of {symbol: OHLCV_DataFrame} and returns all
    symbols where the strategy generates a signal matching signal_type.
    """

    def __init__(self, config: Optional[ScreenerConfig] = None) -> None:
        self.config = config or ScreenerConfig()

    def scan(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy,
        extra_filters: Optional[List[Callable]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scan all symbols in data_dict for strategy signals.

        Args:
            data_dict:      {symbol: ohlcv_df} — at least 100 bars recommended
            strategy:       Any BaseStrategy subclass instance
            extra_filters:  Optional list of callables (symbol, df, signals_df) → bool
                            Return True to KEEP the stock, False to exclude it.

        Returns:
            List of result dicts, sorted by rank_by column.
            Each dict contains: symbol, signal, close, volume, and all
            indicator columns added by the strategy.
        """
        cfg       = self.config
        symbols   = list(data_dict.keys())
        t_start   = time.time()
        hits:     List[Dict[str, Any]] = []
        errors:   List[str]             = []

        logger.info(f"Screener: scanning {len(symbols)} symbols | "
                    f"signal={cfg.signal_type} | workers={cfg.n_workers}")

        def _process_one(symbol: str) -> Optional[Dict]:
            """Process a single symbol — designed to run in a thread pool."""
            try:
                df = data_dict[symbol]

                # ── Pre-flight filters ──────────────────────────────────
                if len(df) < cfg.min_bars:
                    return None

                last = df.iloc[-1]
                close = float(last.get("close", 0))
                vol   = float(df["volume"].tail(20).mean())  # 20-day avg volume

                if close < cfg.min_price:
                    return None
                if cfg.max_price > 0 and close > cfg.max_price:
                    return None
                if vol < cfg.min_volume:
                    return None

                # ATR filter — skip stocks with very low volatility
                if cfg.min_atr_pct > 0:
                    from indicators.technical import atr as _atr
                    atr_val = float(_atr(df, 14).iloc[-1])
                    if not np.isnan(atr_val) and (atr_val / close * 100) < cfg.min_atr_pct:
                        return None

                # ── Run strategy ────────────────────────────────────────
                signals_df = strategy.generate_signals(df)
                if "signal" not in signals_df.columns:
                    return None

                last_signal = int(signals_df["signal"].iloc[-1])

                # ── Signal type filter ───────────────────────────────────
                if cfg.signal_type != 0:
                    if last_signal != cfg.signal_type:
                        return None
                else:
                    if last_signal == 0:
                        return None

                # ── Extra custom filters ─────────────────────────────────
                if extra_filters:
                    for filt in extra_filters:
                        if not filt(symbol, df, signals_df):
                            return None

                # ── Collect result row ───────────────────────────────────
                row: Dict[str, Any] = {
                    "symbol":       symbol,
                    "signal":       last_signal,
                    "close":        round(close, 2),
                    "volume":       int(vol),
                    "scan_date":    str(df.index[-1].date()),
                }

                # Add ALL indicator columns from the strategy output
                indicator_cols = [c for c in signals_df.columns
                                  if c not in ("open", "high", "low", "close",
                                               "volume", "oi", "signal")]
                for col in indicator_cols:
                    val = signals_df[col].iloc[-1]
                    if isinstance(val, (np.floating, float)):
                        row[col] = round(float(val), 4) if not np.isnan(val) else None
                    elif isinstance(val, (np.integer, int)):
                        row[col] = int(val)
                    else:
                        row[col] = val

                return row

            except Exception as e:
                logger.debug(f"  {symbol}: error — {e}")
                errors.append(symbol)
                return None

        # ── Parallel execution ────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=cfg.n_workers) as executor:
            futures = {executor.submit(_process_one, sym): sym for sym in symbols}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    hits.append(result)

        elapsed = time.time() - t_start

        # ── Sort and truncate ─────────────────────────────────────────────
        if hits and cfg.rank_by in hits[0]:
            hits.sort(
                key=lambda r: r.get(cfg.rank_by, 0) or 0,
                reverse=not cfg.rank_ascending,
            )
        hits = hits[:cfg.max_results]

        logger.info(f"Screener done: {len(hits)} hits from {len(symbols)} symbols "
                    f"in {elapsed:.1f}s | {len(errors)} errors")

        if cfg.save_results:
            self._save_results(hits, strategy)

        return hits

    def scan_parallel(
        self,
        data_dict:   Dict[str, pd.DataFrame],
        strategies:  List,
        labels:      Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Run multiple strategies over the same universe simultaneously.

        Returns a dict {strategy_name: list_of_hits} so you can see which
        stocks are confirmed by multiple strategies (confluence).

        Args:
            data_dict:   {symbol: ohlcv_df}
            strategies:  List of strategy instances
            labels:      Optional names for each strategy

        Returns:
            {strategy_label: [hit_dicts]}
        """
        results = {}
        for i, strategy in enumerate(strategies):
            label = (labels[i] if labels and i < len(labels)
                     else getattr(strategy, "name", f"strategy_{i}"))
            logger.info(f"Multi-strategy scan: {label}")
            old_label         = self.config.label
            self.config.label = label
            results[label]    = self.scan(data_dict, strategy)
            self.config.label = old_label
        return results

    def confluence(
        self,
        multi_results: Dict[str, List[Dict]],
        min_count: int = 2,
    ) -> List[Dict]:
        """
        Find symbols that appear in results from multiple strategies.

        This is "confluence screening" — only trade stocks where 2+ strategies
        independently agree on the direction.

        Args:
            multi_results: Output from scan_parallel()
            min_count:     Minimum number of strategies that must agree

        Returns:
            List of symbols with how many strategies flagged them
        """
        from collections import Counter
        all_symbols = []
        for strategy_name, hits in multi_results.items():
            all_symbols.extend([h["symbol"] for h in hits])

        counts = Counter(all_symbols)
        confirmed = [
            {"symbol": sym, "strategy_count": cnt, "strategies": [
                s for s, hits in multi_results.items()
                if any(h["symbol"] == sym for h in hits)
            ]}
            for sym, cnt in counts.items()
            if cnt >= min_count
        ]
        confirmed.sort(key=lambda r: r["strategy_count"], reverse=True)

        logger.info(f"Confluence: {len(confirmed)} symbols confirmed by "
                    f"{min_count}+ strategies")
        return confirmed

    # =========================================================================
    # Output Helpers
    # =========================================================================

    def _save_results(self, hits: List[Dict], strategy) -> None:
        """Save screener results to CSV + JSON."""
        cfg       = self.config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label     = f"{cfg.label}_{timestamp}"

        if not hits:
            logger.info("Screener: no hits to save.")
            return

        # CSV
        csv_path = OUTPUT_SCREENER / f"{label}.csv"
        df_out   = pd.DataFrame(hits)
        df_out.to_csv(csv_path, index=False)
        logger.info(f"Screener CSV: {csv_path}")

        # JSON (for web dashboard)
        json_path = OUTPUT_SCREENER / f"{label}.json"
        with open(json_path, "w") as f:
            json.dump({
                "scan_time":    timestamp,
                "strategy":     getattr(strategy, "name", "unknown"),
                "signal_type":  cfg.signal_type,
                "total_hits":   len(hits),
                "results":      hits,
            }, f, indent=2, default=str)
        logger.info(f"Screener JSON: {json_path}")

    def print_results(self, hits: List[Dict], max_cols: int = 8) -> None:
        """
        Print a formatted table of screener results to the console.

        Args:
            hits:     Output from scan()
            max_cols: Maximum number of columns to display (truncates to fit terminal)
        """
        if not hits:
            print("No signals found.")
            return

        df = pd.DataFrame(hits)
        cols_to_show = list(df.columns[:max_cols])
        display_df   = df[cols_to_show].copy()

        # Format numeric columns to 2 decimal places
        for col in display_df.select_dtypes(include=[float]).columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if x else "N/A")

        print("\n" + "=" * 80)
        print(f"  SCREENER RESULTS — {len(hits)} signals")
        print("=" * 80)
        print(display_df.to_string(index=False))
        print("=" * 80)
