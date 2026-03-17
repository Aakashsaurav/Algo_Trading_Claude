"""
backtester/performance.py
--------------------------
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
) -> dict:
    """
    Compute all performance metrics.

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
        if val:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


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
