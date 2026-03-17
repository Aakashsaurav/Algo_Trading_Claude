"""
backtester/optimizer.py
------------------------
Standalone parameter optimizer for any :class:`backtester.models.BacktestConfig`-
compatible strategy.

DESIGN
======
``Optimizer`` is completely decoupled from the engine — it imports
:func:`backtester.event_loop.run_event_loop` directly and builds a
minimal pipeline: generate signals → run loop → extract metric.

This decoupling means:

* The engine class stays thin (≈ 60 lines).
* The optimizer can be swapped, extended, or replaced without touching
  the event loop.
* Walk-forward testing can call the same optimizer in rolling windows.

SEARCH METHODS
==============
``GRID``
    Exhaustive search over the Cartesian product of all parameter lists.
    Complexity: O(∏ len(values_i)).

``RANDOM``
    Randomly sample ``n_trials`` parameter combinations.  Faster for
    large search spaces; no guarantee of finding the global optimum.

PARALLELISM
===========
Both methods use :class:`concurrent.futures.ProcessPoolExecutor` with
pickle-based IPC.  To reduce pickling overhead on large DataFrames, the
raw OHLCV arrays are sent once per worker via ``initializer``/
``initargs`` and stored in a module-level global — avoiding re-pickling
the same DataFrame for every parameter combination.

OVERFITTING WARNING
===================
In-sample optimization finds the parameters that best fit the
*historical* data you provided.  These parameters will almost always
look worse on new data.  Always validate with:

* An out-of-sample hold-out period.
* Walk-forward testing (``walk_forward()`` below).
* Monte Carlo simulation on shuffled trade returns.
"""

from __future__ import annotations

import itertools
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from backtester.models import BacktestConfig
from backtester.event_loop import run_event_loop

logger = logging.getLogger(__name__)

# Module-level worker state — set once per process via initializer
_worker_df:     Optional[pd.DataFrame] = None
_worker_config: Optional[BacktestConfig] = None
_worker_symbol: str = "SYMBOL"


class SearchMethod(Enum):
    GRID   = "grid"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Worker initializer and task (called in subprocess)
# ---------------------------------------------------------------------------

def _init_worker(df_bytes: bytes, config: BacktestConfig, symbol: str) -> None:
    """Initializer: deserialise the DataFrame once per worker process."""
    global _worker_df, _worker_config, _worker_symbol
    import io
    _worker_df     = pd.read_parquet(io.BytesIO(df_bytes))
    _worker_config = config
    _worker_symbol = symbol


def _run_one(args: tuple) -> Dict[str, Any]:
    """
    Run a single backtest for one parameter combination.
    Called in a subprocess — reads shared worker globals.
    """
    strategy_class, params, metric = args
    try:
        strategy = strategy_class(**params)
        signals_df = strategy.generate_signals(_worker_df.copy())
        if "signal" not in signals_df.columns:
            return {**params, metric: np.nan, "error": "no signal column"}
        trade_log, equity, _ = run_event_loop(signals_df, _worker_config, _worker_symbol)

        from backtester.performance import compute_performance
        m = compute_performance(trade_log, equity, _worker_config)
        val = m.get(metric, np.nan)
        row = {**params, metric: round(float(val), 6)}
        # Include a few bonus metrics for context
        for extra in ("total_trades", "win_rate_pct", "max_drawdown_pct", "total_return_pct"):
            if extra != metric:
                row[extra] = round(float(m.get(extra, np.nan)), 4)
        return row
    except Exception as exc:
        logger.debug(f"Optimizer worker error ({params}): {exc}")
        return {**params, metric: np.nan, "error": str(exc)}


# ---------------------------------------------------------------------------
# Optimizer class
# ---------------------------------------------------------------------------

class Optimizer:
    """
    Grid and random parameter search for any BaseStrategy subclass.

    Parameters
    ----------
    config : BacktestConfig
        Engine configuration.  ``initial_capital``, ``segment``, and all
        risk/order settings are taken from here.  Do NOT pass
        ``strategy_class`` or ``params`` here — those go to :meth:`run`.
    max_workers : int
        Number of parallel worker processes.  Default: CPU count.

    Examples
    --------
    >>> from backtester.models import BacktestConfig
    >>> from backtester.optimizer import Optimizer
    >>> from strategies.base import EMACrossover
    >>>
    >>> config = BacktestConfig(initial_capital=500_000)
    >>> opt    = Optimizer(config)
    >>>
    >>> grid = {"fast_period": [5, 9, 13], "slow_period": [21, 34, 50]}
    >>> results = opt.run(df, EMACrossover, grid, symbol="INFY")
    >>> print(results.head())
    """

    def __init__(
        self,
        config:      BacktestConfig,
        max_workers: Optional[int] = None,
    ) -> None:
        self.config      = config
        self.max_workers = max_workers

    def run(
        self,
        df:             pd.DataFrame,
        strategy_class: Type,
        param_grid:     Dict[str, List[Any]],
        symbol:         str       = "SYMBOL",
        metric:         str       = "sharpe_ratio",
        method:         SearchMethod = SearchMethod.GRID,
        n_trials:       int       = 50,
        top_n:          int       = 10,
    ) -> pd.DataFrame:
        """
        Run the parameter search.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (same format as used in engine.run).
        strategy_class : type
            The strategy class (not an instance).
        param_grid : dict
            Mapping from parameter name to list of values to try.
            Example: ``{"fast": [5, 9], "slow": [21, 34]}``.
        symbol : str
        metric : str
            Performance metric to optimise.  Any key returned by
            :func:`backtester.performance.compute_performance`.
            Default: ``"sharpe_ratio"``.
        method : SearchMethod
            ``GRID`` (default) or ``RANDOM``.
        n_trials : int
            Number of random combinations to try (RANDOM only).
        top_n : int
            Return only the top N rows sorted by ``metric``.

        Returns
        -------
        pd.DataFrame
            Sorted by ``metric`` descending.  Each row is one parameter
            combination with its performance metrics.
        """
        # Build combination list
        combos = self._build_combos(param_grid, method, n_trials)
        if not combos:
            logger.warning("Optimizer: empty parameter grid — nothing to search.")
            return pd.DataFrame()

        logger.info(
            f"Optimizer: {len(combos)} combinations | metric={metric} | "
            f"method={method.value} | strategy={strategy_class.__name__}"
        )

        # Serialise the DataFrame once (shared across workers via initializer)
        import io
        buf = io.BytesIO()
        df.to_parquet(buf, index=True)
        df_bytes = buf.getvalue()

        args_list = [(strategy_class, params, metric) for params in combos]
        rows: List[Dict] = []

        with ProcessPoolExecutor(
            max_workers  = self.max_workers,
            initializer  = _init_worker,
            initargs     = (df_bytes, self.config, symbol),
        ) as pool:
            futures = {pool.submit(_run_one, a): a for a in args_list}
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    logger.warning(f"Optimizer future error: {exc}")

        if not rows:
            return pd.DataFrame()

        result = (
            pd.DataFrame(rows)
            .dropna(subset=[metric])
            .sort_values(metric, ascending=False)
            .reset_index(drop=True)
        )
        logger.info(
            f"Optimizer complete. Best {metric}={result[metric].iloc[0]:.4f} "
            f"→ {dict(result.iloc[0][list(param_grid.keys())])}"
        )
        return result.head(top_n)

    # ------------------------------------------------------------------
    def walk_forward(
        self,
        df:             pd.DataFrame,
        strategy_class: Type,
        param_grid:     Dict[str, List[Any]],
        symbol:         str       = "SYMBOL",
        metric:         str       = "sharpe_ratio",
        train_bars:     int       = 500,
        test_bars:      int       = 100,
        step_bars:      int       = 100,
        method:         SearchMethod = SearchMethod.GRID,
        n_trials:       int       = 30,
    ) -> pd.DataFrame:
        """
        Rolling walk-forward optimisation.

        Trains on a ``train_bars``-wide window, tests on the subsequent
        ``test_bars`` bars, then rolls forward by ``step_bars``.

        Returns
        -------
        pd.DataFrame
            One row per walk-forward window with the best parameters and
            the out-of-sample ``metric`` achieved in the test period.
        """
        n = len(df)
        results: List[Dict] = []
        start = 0

        while start + train_bars + test_bars <= n:
            train_df = df.iloc[start : start + train_bars]
            test_df  = df.iloc[start + train_bars : start + train_bars + test_bars]

            logger.info(
                f"Walk-forward window [{start}:{start+train_bars}] train, "
                f"[{start+train_bars}:{start+train_bars+test_bars}] test"
            )

            # Optimise on training data
            opt_results = self.run(
                df=train_df, strategy_class=strategy_class,
                param_grid=param_grid, symbol=symbol, metric=metric,
                method=method, n_trials=n_trials, top_n=1,
            )
            if opt_results.empty:
                start += step_bars
                continue

            best_params = {k: opt_results.iloc[0][k] for k in param_grid.keys()}

            # Evaluate best params on out-of-sample test data
            try:
                strategy   = strategy_class(**best_params)
                signals_df = strategy.generate_signals(test_df.copy())
                trade_log, equity, _ = run_event_loop(signals_df, self.config, symbol)

                from backtester.performance import compute_performance
                m = compute_performance(trade_log, equity, self.config)

                row = {
                    "window_start": start,
                    "window_end":   start + train_bars + test_bars,
                    **best_params,
                    f"train_{metric}": float(opt_results.iloc[0][metric]),
                    f"test_{metric}":  float(m.get(metric, np.nan)),
                    "test_total_trades":  m.get("total_trades", 0),
                    "test_total_return":  m.get("total_return_pct", 0.0),
                }
                results.append(row)
            except Exception as exc:
                logger.warning(f"Walk-forward test evaluation failed: {exc}")

            start += step_bars

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_combos(
        param_grid: Dict[str, List[Any]],
        method:     SearchMethod,
        n_trials:   int,
    ) -> List[Dict[str, Any]]:
        keys   = list(param_grid.keys())
        values = list(param_grid.values())

        if method == SearchMethod.GRID:
            return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        # RANDOM
        all_combos = list(itertools.product(*values))
        k = min(n_trials, len(all_combos))
        return [dict(zip(keys, c)) for c in random.sample(all_combos, k)]
