"""
backtester/engine.py
---------------------
Public API for the AlgoDesk backtesting engine.

This is the only file you need to import.  All implementation is in the
sub-modules it orchestrates:

    models.py         — data structures (BacktestConfig, Trade, Position, …)
    event_loop.py     — bar-by-bar simulation
    fill_engine.py    — order fill and stop logic
    position_sizer.py — position sizing
    performance.py    — metrics computation
    optimizer.py      — parameter search

QUICK START
===========
::

    from backtester.engine import BacktestEngine
    from backtester.models import BacktestConfig, OrderType
    from broker.upstox.commission import Segment
    from strategies.base import EMACrossover

    cfg = BacktestConfig(
        initial_capital    = 500_000,
        segment            = Segment.EQUITY_DELIVERY,
        default_order_type = OrderType.MARKET,
        stop_loss_pct      = 2.0,
        save_trade_log     = True,
        save_chart         = True,
        run_label          = "ema_infy_2023",
    )
    engine   = BacktestEngine(cfg)
    strategy = EMACrossover(fast_period=9, slow_period=21)

    result = engine.run(df, strategy, symbol="INFY")
    print(result.summary())

MULTI-SYMBOL
============
::

    results = engine.run_portfolio(
        {"INFY": df_infy, "TCS": df_tcs},
        strategy,
        label="portfolio_run",
    )

OPTIMISATION
============
::

    from backtester.optimizer import Optimizer
    opt = Optimizer(cfg)
    top = opt.run(df, EMACrossover, {"fast_period": [5,9,13], "slow_period": [21,34]})
    print(top)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backtester.models import BacktestConfig, BacktestResult
from backtester.event_loop import run_event_loop

from config import config

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent.parent   # project root
#OUTPUT_TRADE = _HERE / "strategies" / "output" / "trade"
#OUTPUT_RAW   = _HERE / "strategies" / "output" / "raw_data"
#OUTPUT_CHART = _HERE / "strategies" / "output" / "chart"

OUTPUT_TRADE = config.OUTPUT_TRADE
OUTPUT_RAW   = config.OUTPUT_RAW
OUTPUT_CHART = config.OUTPUT_CHART

for _d in (OUTPUT_TRADE, OUTPUT_RAW, OUTPUT_CHART):
    _d.mkdir(parents=True, exist_ok=True)


class BacktestEngine:
    """
    Main backtesting engine.

    Parameters
    ----------
    config : BacktestConfig
        All engine parameters.

    Notes
    -----
    The engine is stateless between runs — the same instance can be used
    for multiple ``run()`` calls with different data or strategies.
    """

    def __init__(self, config: BacktestConfig) -> None:
        config.validate()
        self.config = config

    # ------------------------------------------------------------------
    def run(
        self,
        df:       pd.DataFrame,
        strategy,
        symbol:   str = "SYMBOL",
    ) -> BacktestResult:
        """
        Run a backtest on a single symbol.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.  Must contain ``open``, ``high``, ``low``,
            ``close`` columns.  A timezone-aware DatetimeIndex (IST) is
            recommended for intraday work.
        strategy
            Any instance whose ``generate_signals(df)`` method returns
            the DataFrame with a ``signal`` column added.
        symbol : str
            Used for logging and output filenames.

        Returns
        -------
        BacktestResult
        """
        self._preflight(df)
        logger.info(
            f"[BacktestEngine] {getattr(strategy, 'name', strategy.__class__.__name__)} "
            f"on {symbol} ({len(df)} bars) | "
            f"order={self.config.default_order_type.value}"
        )

        signals_df = strategy.generate_signals(df.copy())
        if "signal" not in signals_df.columns:
            raise ValueError(
                f"strategy.generate_signals() must return a DataFrame with a "
                f"'signal' column.  Got columns: {list(signals_df.columns)}"
            )

        trade_log, equity, drawdown = run_event_loop(signals_df, self.config, symbol)

        result = BacktestResult(
            config       = self.config,
            symbol       = symbol,
            trade_log    = trade_log,
            equity_curve = equity,
            drawdown     = drawdown,
            signals_df   = signals_df,
        )
        self._handle_outputs(result, symbol)
        return result

    # ------------------------------------------------------------------
    def run_portfolio(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy,
        label:     str = "",
    ) -> Dict[str, BacktestResult]:
        """
        Run the same strategy on a portfolio of symbols.

        Output files (trade log, raw data, summary) are written as
        *separate per-symbol files*, not concatenated into one large
        file.  This keeps memory usage O(1 symbol) rather than O(N
        symbols) — critical for 50+ symbol runs.

        Parameters
        ----------
        data_dict : dict
            ``{symbol: ohlcv_df}`` mapping.
        strategy
            Strategy instance (same object used for all symbols).
        label : str
            Override ``config.run_label`` for this portfolio run.

        Returns
        -------
        dict
            ``{symbol: BacktestResult}``
        """
        run_label = label or self.config.run_label
        logger.info(
            f"[BacktestEngine] Portfolio run: {len(data_dict)} symbols | "
            f"label={run_label}"
        )
        results: Dict[str, BacktestResult] = {}

        for symbol, df in data_dict.items():
            try:
                self._preflight(df)
                signals_df = strategy.generate_signals(df.copy())
                if "signal" not in signals_df.columns:
                    logger.warning(f"{symbol}: no 'signal' column — skipped")
                    continue

                trade_log, equity, drawdown = run_event_loop(
                    signals_df, self.config, symbol
                )
                result = BacktestResult(
                    config       = self.config,
                    symbol       = symbol,
                    trade_log    = trade_log,
                    equity_curve = equity,
                    drawdown     = drawdown,
                    signals_df   = signals_df,
                )
                results[symbol] = result
                self._handle_outputs(result, symbol, label=run_label)

                net = sum(t.net_pnl for t in trade_log)
                logger.info(
                    f"  {symbol}: {len(trade_log)} trades | net=₹{net:+,.0f}"
                )
            except Exception as exc:
                logger.error(f"  {symbol}: ERROR — {exc}", exc_info=True)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preflight(df: pd.DataFrame) -> None:
        """Validate the OHLCV DataFrame before running the loop."""
        required = {"open", "high", "low", "close"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        if len(df) < 2:
            raise ValueError("DataFrame must have at least 2 rows.")
        if df.index.duplicated().any():
            raise ValueError(
                "DataFrame index has duplicates. Run DataCleaner first."
            )
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            logger.warning(
                "DataFrame index is not datetime — intraday squareoff and "
                "time-based checks will be disabled."
            )

    def _handle_outputs(
        self,
        result: BacktestResult,
        symbol: str,
        label:  str = "",
    ) -> None:
        """Write output files based on config flags."""
        cfg   = self.config
        label = label or cfg.run_label

        if cfg.save_trade_log and result.trade_log:
            path = OUTPUT_TRADE / f"{label}_{symbol}_trade_log.csv"
            result.trade_df().to_csv(path, index=False)
            logger.info(f"Trade log → {path}")

        if cfg.save_raw_data and result.signals_df is not None:
            path = OUTPUT_RAW / f"{label}_{symbol}_raw.csv"
            result.signals_df.to_csv(path)
            logger.info(f"Raw data → {path}")

        if cfg.save_chart:
            try:
                from backtester.report import generate_report
                generate_report(
                    result,
                    symbol      = symbol,
                    output_dir  = str(OUTPUT_CHART),
                    filename    = f"{label}_{symbol}_chart.png",
                    max_candles = cfg.max_candles,
                )
            except Exception as exc:
                logger.warning(f"Chart generation failed for {symbol}: {exc}")

        if cfg.generate_summary:
            path = OUTPUT_TRADE / f"{label}_{symbol}_summary.json"
            import json
            m = result.metrics()
            # Remove non-serialisable nested dicts for top-level summary
            safe = {k: v for k, v in m.items()
                    if not isinstance(v, (dict, list))}
            with open(path, "w") as fh:
                json.dump(safe, fh, indent=2, default=str)
            logger.info(f"Summary → {path}")


# ---------------------------------------------------------------------------
# Convenience re-exports so users can ``from backtester.engine import *``
# ---------------------------------------------------------------------------
from backtester.models import BacktestConfig, BacktestResult, Trade, Position, OrderType  # noqa: F401, E402
from backtester.optimizer import Optimizer, SearchMethod  # noqa: F401, E402
