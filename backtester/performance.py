"""
backtester/performance.py
--------------------------
<<<<<<< HEAD
Computes the full suite of performance metrics from a completed backtest.

All computation is vectorised (NumPy).  The function accepts a
:class:`backtester.models.BacktestResult` (or its components directly)
and returns a plain ``dict`` of JSON-serialisable scalars.

METRICS RETURNED
================
Returns
~~~~~~~
total_return_pct, cagr_pct
    Absolute and compounded annual return.

annualised_volatility
    Annualised standard deviation of daily equity returns.

sharpe_ratio
    Risk-adjusted return using the Indian 10-yr G-Sec proxy (6.5 % p.a.).

sortino_ratio
    Like Sharpe but penalises only downside volatility.

calmar_ratio
    CAGR / |max_drawdown_pct|.  Measures return per unit of tail risk.

omega_ratio
    Probability-weighted ratio of gains to losses above/below a
    threshold (0 % return).  More robust than Sharpe for non-normal
    return distributions.

kelly_fraction
    Optimal bet size fraction: (win_rate × avg_win − loss_rate × |avg_loss|)
    / avg_win.  Informational only — always size smaller in practice.

Drawdown
~~~~~~~~
max_drawdown_pct, avg_drawdown_pct
    Largest and average peak-to-trough equity drop as a percentage.

max_drawdown_duration_bars
    Longest continuous period spent below a prior peak (in bars).

Trade Statistics
~~~~~~~~~~~~~~~~
total_trades, winning_trades, losing_trades
win_rate_pct, avg_win_inr, avg_loss_inr
profit_factor
    gross_wins / |gross_losses|.
expectancy_inr
    Expected ₹ per trade.
avg_trade_duration_bars
max_consecutive_wins, max_consecutive_losses
avg_mae_inr, avg_mfe_inr

Monthly / Annual Breakdown
~~~~~~~~~~~~~~~~~~~~~~~~~~
monthly_returns : dict  ``{YYYY-MM: pct_return}``
annual_returns  : dict  ``{YYYY: pct_return}``

Capital
~~~~~~~
initial_capital, final_capital, total_net_pnl
total_commission_paid
exposure_pct
    Fraction of total bars during which at least one position was open.
    Lower is better for capital-efficient strategies.

Dates
~~~~~
start_date, end_date
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtester.models import BacktestConfig, Trade

logger = logging.getLogger(__name__)

RISK_FREE_RATE_ANNUAL = 0.065   # 6.5% India 10-yr G-Sec proxy
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_performance(
    trade_log:    List["Trade"],
    equity_curve: pd.Series,
    config:       "BacktestConfig",
=======
Computes all strategy performance metrics from the trade log and equity curve.

METRICS COMPUTED:
  Returns:
    total_return_pct         Total % return over the backtest period
    cagr_pct                 Compound Annual Growth Rate
    annualised_volatility    Annualised std dev of daily returns

  Risk-Adjusted:
    sharpe_ratio             (Return - RFR) / Volatility. Uses 6% India RFR.
    sortino_ratio            Like Sharpe but only penalises downside volatility
    calmar_ratio             CAGR / Max Drawdown

  Drawdown:
    max_drawdown_pct         Largest peak-to-trough equity drop (%)
    avg_drawdown_pct         Average drawdown depth (%)
    max_drawdown_duration    Longest drawdown period (bars)

  Trade Statistics:
    total_trades             Total completed round-trip trades
    winning_trades           Trades with net_pnl > 0
    losing_trades            Trades with net_pnl <= 0
    win_rate_pct             winning_trades / total_trades × 100
    avg_win                  Average profit on winning trades (₹)
    avg_loss                 Average loss on losing trades (₹)
    profit_factor            Sum(wins) / |Sum(losses)|
    expectancy               Expected ₹ per trade = (win_rate × avg_win) - (loss_rate × |avg_loss|)
    avg_trade_duration       Average bars held per trade
    max_consecutive_wins     Longest winning streak
    max_consecutive_losses   Longest losing streak

  Capital:
    initial_capital          Starting capital
    final_capital            Ending capital
    total_commission_paid    Sum of all commissions deducted
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Indian risk-free rate for Sharpe ratio (approx 10-yr Gsec yield)
RISK_FREE_RATE_ANNUAL = 0.065   # 6.5%
TRADING_DAYS_PER_YEAR = 252


def compute_performance(
    trade_log_df:   pd.DataFrame,
    equity_curve:   list[tuple],
    initial_capital: float,
    start_date:     Optional[pd.Timestamp] = None,
    end_date:       Optional[pd.Timestamp] = None,
>>>>>>> 8d072798ed841b92b7056b98b3d612023cbaf223
) -> dict:
    """
    Compute all performance metrics.

<<<<<<< HEAD
    Parameters
    ----------
    trade_log : list[Trade]
        Completed trades returned by the engine.
    equity_curve : pd.Series
        Portfolio value at each bar, indexed by bar timestamps.
    config : BacktestConfig
        Engine configuration (needed for ``initial_capital``).

    Returns
    -------
    dict
        All metrics.  All values are Python floats or ints (JSON-safe).
    """
    initial_capital = config.initial_capital

    if equity_curve.empty or equity_curve.dropna().empty:
        logger.warning("Empty equity curve — returning zero metrics.")
        return _empty_metrics(initial_capital)

    eq = equity_curve.dropna()

    # ── Return metrics ──────────────────────────────────────────────────────
    start_v = initial_capital
    end_v   = float(eq.iloc[-1])
    total_return_pct = ((end_v / start_v) - 1.0) * 100.0

    start_date = eq.index[0]
    end_date   = eq.index[-1]
    try:
        years = max((end_date - start_date).days / 365.25, 1 / 365.25)
    except Exception:
        years = 1.0

    cagr_pct = (((end_v / start_v) ** (1.0 / years)) - 1.0) * 100.0

    # ── Daily returns for vol / Sharpe / Sortino / Omega ───────────────────
    # Resample equity to daily to avoid intraday bar inflation
    if hasattr(eq.index, "date"):
        try:
            eq_daily = eq.resample("D").last().dropna()
        except Exception:
            eq_daily = eq
    else:
        eq_daily = eq

    daily_returns = eq_daily.pct_change().dropna()

    ann_vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) * 100.0

    rfr_daily = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = daily_returns - rfr_daily
    sharpe = (
        float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if excess.std() > 0 else 0.0
    )

    downside = daily_returns[daily_returns < 0]
    sortino = (
        float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(downside) > 0 and downside.std() > 0 else 0.0
    )

    # ── Drawdown ────────────────────────────────────────────────────────────
    eq_arr  = eq.values.astype(float)
    peak    = np.maximum.accumulate(eq_arr)
    dd_arr  = np.where(peak > 0, (eq_arr - peak) / peak, 0.0)

    max_dd  = float(dd_arr.min()) * 100.0  # negative %
    avg_dd  = float(dd_arr[dd_arr < 0].mean()) * 100.0 if (dd_arr < 0).any() else 0.0

    # Max drawdown duration (bars)
    in_dd = dd_arr < 0
    max_dd_dur = _max_run(in_dd)

    calmar = abs(cagr_pct / max_dd) if max_dd != 0 else 0.0

    # ── Omega ratio ─────────────────────────────────────────────────────────
    threshold = 0.0
    gains  = daily_returns[daily_returns > threshold] - threshold
    losses = threshold - daily_returns[daily_returns <= threshold]
    omega  = float(gains.sum() / losses.sum()) if losses.sum() > 0 else float("inf")

    # ── Monthly / annual breakdowns ─────────────────────────────────────────
    monthly_returns: dict = {}
    annual_returns:  dict = {}
    try:
        if hasattr(eq_daily.index, "to_period"):
            monthly_eq = eq_daily.resample("ME").last().dropna()
            m_ret      = monthly_eq.pct_change().dropna() * 100.0
            for ts, val in m_ret.items():
                monthly_returns[str(ts)[:7]] = round(float(val), 2)

            annual_eq = eq_daily.resample("YE").last().dropna()
            a_ret     = annual_eq.pct_change().dropna() * 100.0
            for ts, val in a_ret.items():
                annual_returns[str(ts)[:4]] = round(float(val), 2)
    except Exception as exc:
        logger.debug(f"Monthly/annual breakdown skipped: {exc}")

    # ── Trade statistics ─────────────────────────────────────────────────────
    trade_stats = _compute_trade_stats(trade_log)

    # Kelly fraction (informational)
    wr   = trade_stats["win_rate_pct"] / 100.0
    lr   = 1.0 - wr
    aw   = trade_stats["avg_win_inr"]
    al   = abs(trade_stats["avg_loss_inr"])
    kelly = ((wr * aw - lr * al) / aw) if aw > 0 else 0.0

    # ── Exposure % ──────────────────────────────────────────────────────────
    exposure_pct = _compute_exposure(trade_log, eq) if trade_log else 0.0

    # ── Commission ──────────────────────────────────────────────────────────
    total_commission = sum(t.total_charges for t in trade_log) if trade_log else 0.0

    return {
        # Dates
        "start_date":                str(start_date)[:10],
        "end_date":                  str(end_date)[:10],
        # Capital
        "initial_capital":           round(initial_capital, 2),
        "final_capital":             round(end_v, 2),
        "total_net_pnl":             round(end_v - initial_capital, 2),
        # Returns
        "total_return_pct":          round(total_return_pct, 4),
        "cagr_pct":                  round(cagr_pct, 4),
        "annualised_volatility":     round(ann_vol, 4),
        # Risk-adjusted
        "sharpe_ratio":              round(sharpe, 4),
        "sortino_ratio":             round(sortino, 4),
        "calmar_ratio":              round(calmar, 4),
        "omega_ratio":               round(omega, 4),
        "kelly_fraction":            round(kelly, 4),
        # Drawdown
        "max_drawdown_pct":          round(max_dd, 4),
        "avg_drawdown_pct":          round(avg_dd, 4),
        "max_drawdown_duration_bars": int(max_dd_dur),
        # Trade stats (merged from helper)
        **trade_stats,
        # Monthly/annual
        "monthly_returns":           monthly_returns,
        "annual_returns":            annual_returns,
        # Misc
        "exposure_pct":              round(exposure_pct, 4),
        "total_commission_paid":     round(total_commission, 2),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_trade_stats(trade_log: List["Trade"]) -> dict:
    """Compute all trade-level statistics from the completed trade list."""
    empty = _empty_trade_stats()
    if not trade_log:
        return empty

    net_pnls    = np.array([t.net_pnl       for t in trade_log], dtype=float)
    charges     = np.array([t.total_charges  for t in trade_log], dtype=float)
    durations   = np.array([t.duration_bars  for t in trade_log], dtype=float)
    mae_vals    = np.array([t.mae * t.quantity for t in trade_log], dtype=float)
    mfe_vals    = np.array([t.mfe * t.quantity for t in trade_log], dtype=float)

    winners = net_pnls[net_pnls > 0]
    losers  = net_pnls[net_pnls <= 0]
    n_total = len(net_pnls)
    n_win   = len(winners)
    n_loss  = len(losers)
    win_rate = (n_win / n_total) * 100.0 if n_total > 0 else 0.0

    avg_win  = float(winners.mean()) if n_win > 0 else 0.0
    avg_loss = float(losers.mean())  if n_loss > 0 else 0.0

    gross_wins  = winners.sum() if n_win > 0 else 0.0
    gross_loss  = abs(losers.sum()) if n_loss > 0 else 0.0
    pf = (gross_wins / gross_loss) if gross_loss > 0 else float("inf")

    wr_frac  = win_rate / 100.0
    lr_frac  = 1.0 - wr_frac
    expectancy = (wr_frac * avg_win) + (lr_frac * avg_loss)  # avg_loss is negative

    is_win = net_pnls > 0
    is_loss = net_pnls <= 0

    return {
        "total_trades":            n_total,
        "winning_trades":          int(n_win),
        "losing_trades":           int(n_loss),
        "win_rate_pct":            round(win_rate, 4),
        "avg_win_inr":             round(avg_win,  2),
        "avg_loss_inr":            round(avg_loss, 2),
        "profit_factor":           round(pf,       4) if pf != float("inf") else 999.0,
        "expectancy_inr":          round(expectancy, 2),
        "avg_trade_duration_bars": round(float(durations.mean()), 2) if len(durations) > 0 else 0.0,
        "max_consecutive_wins":    int(_max_run(is_win)),
        "max_consecutive_losses":  int(_max_run(is_loss)),
        "avg_mae_inr":             round(float(mae_vals.mean()), 2) if len(mae_vals) > 0 else 0.0,
        "avg_mfe_inr":             round(float(mfe_vals.mean()), 2) if len(mfe_vals) > 0 else 0.0,
    }


def _compute_exposure(trade_log: List["Trade"], equity_curve: pd.Series) -> float:
    """
    Estimate what % of total bars had at least one open position.

    Uses the entry/exit times in the trade log to mark bars as "in trade".
    Returns 0.0 if the equity curve has no timestamp index.
    """
    if not trade_log or equity_curve.empty:
        return 0.0
    try:
        total_bars = len(equity_curve.dropna())
        if total_bars == 0:
            return 0.0

        idx = equity_curve.index
        in_trade = pd.Series(False, index=idx)

        for t in trade_log:
            mask = (idx >= t.entry_time) & (idx <= t.exit_time)
            in_trade = in_trade | mask

        return round(in_trade.sum() / total_bars * 100.0, 2)
    except Exception as exc:
        logger.debug(f"exposure_pct computation failed: {exc}")
        return 0.0


def _max_run(bool_arr) -> int:
    """Return the length of the longest contiguous True run."""
    max_run = cur_run = 0
    for val in bool_arr:
=======
    Args:
        trade_log_df    : DataFrame from TradeLog.to_dataframe().
        equity_curve    : List of (timestamp, portfolio_value) tuples.
        initial_capital : Starting capital in ₹.
        start_date      : Backtest start (for CAGR calculation).
        end_date        : Backtest end (for CAGR calculation).

    Returns:
        dict: All metrics. Values are floats or ints (JSON-serialisable).
    """
    metrics = {}

    # ------------------------------------------------------------------
    # Equity curve → Series
    # ------------------------------------------------------------------
    if not equity_curve:
        logger.warning("Empty equity curve — returning zero metrics.")
        return _empty_metrics(initial_capital)

    eq_times  = [t for t, v in equity_curve]
    eq_values = [v for t, v in equity_curve]
    eq_series = pd.Series(eq_values, index=pd.to_datetime(eq_times))
    eq_series = eq_series[~eq_series.index.duplicated(keep="last")]
    eq_series = eq_series.sort_index()

    final_capital = float(eq_series.iloc[-1])
    metrics["initial_capital"]  = round(initial_capital, 2)
    metrics["final_capital"]    = round(final_capital, 2)

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------
    total_return = (final_capital - initial_capital) / initial_capital
    metrics["total_return_pct"] = round(total_return * 100, 4)

    # CAGR
    if start_date is None:
        start_date = eq_series.index[0]
    if end_date is None:
        end_date = eq_series.index[-1]

    years = (end_date - start_date).days / 365.25
    if years > 0 and initial_capital > 0:
        cagr = (final_capital / initial_capital) ** (1 / years) - 1
    else:
        cagr = 0.0
    metrics["cagr_pct"] = round(cagr * 100, 4)

    # Daily returns (resample equity to daily; handles intraday data)
    daily_eq = eq_series.resample("1B").last().dropna()
    daily_returns = daily_eq.pct_change().dropna()
    metrics["annualised_volatility"] = round(
        float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100), 4
    ) if len(daily_returns) > 1 else 0.0

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------
    rfr_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
    excess_returns = daily_returns - rfr_daily

    if len(excess_returns) > 1 and daily_returns.std() > 0:
        sharpe = float(excess_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sharpe = 0.0
    metrics["sharpe_ratio"] = round(sharpe, 4)

    # Sortino: only downside volatility
    downside_returns = daily_returns[daily_returns < rfr_daily]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        sortino = float(excess_returns.mean() / downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sortino = 0.0
    metrics["sortino_ratio"] = round(sortino, 4)

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------
    rolling_max = eq_series.cummax()
    drawdown     = (eq_series - rolling_max) / rolling_max * 100
    max_dd       = float(drawdown.min())
    avg_dd       = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0
    metrics["max_drawdown_pct"] = round(max_dd, 4)
    metrics["avg_drawdown_pct"] = round(avg_dd, 4)

    # Calmar ratio
    if abs(max_dd) > 0:
        metrics["calmar_ratio"] = round(cagr * 100 / abs(max_dd), 4)
    else:
        metrics["calmar_ratio"] = 0.0

    # Max drawdown duration (consecutive bars below peak)
    in_dd = drawdown < 0
    max_dd_dur = 0
    cur_dur    = 0
    for flag in in_dd:
        if flag:
            cur_dur += 1
            max_dd_dur = max(max_dd_dur, cur_dur)
        else:
            cur_dur = 0
    metrics["max_drawdown_duration_bars"] = max_dd_dur

    # ------------------------------------------------------------------
    # Trade statistics
    # ------------------------------------------------------------------
    if trade_log_df.empty:
        metrics.update(_empty_trade_stats())
        return metrics

    tl = trade_log_df.copy()
    total_trades  = len(tl)
    winning       = tl[tl["net_pnl"] > 0]
    losing        = tl[tl["net_pnl"] <= 0]

    metrics["total_trades"]    = total_trades
    metrics["winning_trades"]  = len(winning)
    metrics["losing_trades"]   = len(losing)
    metrics["win_rate_pct"]    = round(len(winning) / total_trades * 100, 2) if total_trades > 0 else 0.0

    avg_win  = float(winning["net_pnl"].mean()) if len(winning) > 0 else 0.0
    avg_loss = float(losing["net_pnl"].mean())  if len(losing)  > 0 else 0.0
    metrics["avg_win_inr"]  = round(avg_win, 2)
    metrics["avg_loss_inr"] = round(avg_loss, 2)

    total_wins   = float(winning["net_pnl"].sum()) if len(winning) > 0 else 0.0
    total_losses = float(losing["net_pnl"].sum())  if len(losing)  > 0 else 0.0
    metrics["total_commission_paid"] = round(
        float(tl["commission_entry"].sum() + tl["commission_exit"].sum()), 2
    )

    if abs(total_losses) > 0:
        metrics["profit_factor"] = round(total_wins / abs(total_losses), 4)
    else:
        metrics["profit_factor"] = float("inf") if total_wins > 0 else 0.0

    win_rate  = len(winning) / total_trades if total_trades > 0 else 0.0
    loss_rate = 1 - win_rate
    metrics["expectancy_inr"] = round(
        win_rate * avg_win + loss_rate * avg_loss, 2
    )

    metrics["avg_trade_duration_bars"] = round(float(tl["duration_bars"].mean()), 1) if total_trades > 0 else 0.0

    # Consecutive wins/losses
    metrics["max_consecutive_wins"]   = _max_consecutive(tl["net_pnl"] > 0)
    metrics["max_consecutive_losses"] = _max_consecutive(tl["net_pnl"] <= 0)

    # MAE/MFE averages
    metrics["avg_mae_inr"] = round(float(tl["mae"].mean()), 2) if "mae" in tl.columns else 0.0
    metrics["avg_mfe_inr"] = round(float(tl["mfe"].mean()), 2) if "mfe" in tl.columns else 0.0

    return metrics


def _max_consecutive(bool_series: pd.Series) -> int:
    """Count longest consecutive True run."""
    max_run = cur_run = 0
    for val in bool_series:
>>>>>>> 8d072798ed841b92b7056b98b3d612023cbaf223
        if val:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


<<<<<<< HEAD
def _empty_trade_stats() -> dict:
    return {
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "win_rate_pct": 0.0, "avg_win_inr": 0.0, "avg_loss_inr": 0.0,
        "profit_factor": 0.0, "expectancy_inr": 0.0,
        "avg_trade_duration_bars": 0.0, "max_consecutive_wins": 0,
        "max_consecutive_losses": 0, "avg_mae_inr": 0.0, "avg_mfe_inr": 0.0,
    }


def _empty_metrics(initial_capital: float) -> dict:
    m = _empty_trade_stats()
    m.update({
        "start_date": "", "end_date": "",
        "initial_capital": round(initial_capital, 2),
        "final_capital": round(initial_capital, 2),
        "total_net_pnl": 0.0,
        "total_return_pct": 0.0, "cagr_pct": 0.0,
        "annualised_volatility": 0.0, "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0, "calmar_ratio": 0.0,
        "omega_ratio": 0.0, "kelly_fraction": 0.0,
        "max_drawdown_pct": 0.0, "avg_drawdown_pct": 0.0,
        "max_drawdown_duration_bars": 0,
        "monthly_returns": {}, "annual_returns": {},
        "exposure_pct": 0.0, "total_commission_paid": 0.0,
        "error": "No trades generated.",
    })
    return m
=======
def _empty_metrics(initial_capital: float) -> dict:
    m = _empty_trade_stats()
    m.update({
        "initial_capital": round(initial_capital, 2),
        "final_capital": round(initial_capital, 2),
        "total_return_pct": 0.0,
        "cagr_pct": 0.0,
        "annualised_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_drawdown_pct": 0.0,
        "max_drawdown_duration_bars": 0,
        "total_commission_paid": 0.0,
    })
    return m


def _empty_trade_stats() -> dict:
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": 0.0,
        "avg_win_inr": 0.0,
        "avg_loss_inr": 0.0,
        "profit_factor": 0.0,
        "expectancy_inr": 0.0,
        "avg_trade_duration_bars": 0.0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "avg_mae_inr": 0.0,
        "avg_mfe_inr": 0.0,
    }
>>>>>>> 8d072798ed841b92b7056b98b3d612023cbaf223
