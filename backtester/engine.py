"""
Renamed from backtester/engine_v3.py
backtester/engine.py
------------------------
Merged backtesting engine combining engine.py and engine_v2.py.

FEATURES:
  • Basic backtesting (from engine.py)
  • Advanced order types: LIMIT, STOP, STOP-LIMIT, TRAILING STOP
  • Output flags (trade log, raw data, chart, summary)
  • Multi-security portfolio runs
  • Parameter optimization (grid/random search)
  • Full Upstox commission modeling

USAGE:
    from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3
    from backtester.order_types import OrderType

    config = BacktestConfigV3(
        initial_capital   = 500_000,
        default_order_type = OrderType.LIMIT,
        limit_offset_pct   = 0.2,
        stop_loss_pct      = 2.0,
        trailing_stop_pct  = 1.5,
        save_trade_log     = True,
        save_raw_data      = True,
        save_chart         = True,
        generate_summary   = True,
        run_label          = "merged_engine_test",
    )
    engine   = BacktestEngineV3(config)
    strategy = SomeStrategy()

    # Single symbol
    result = engine.run(df, strategy, symbol="INFY")

    # Multi-symbol portfolio
    results = engine.run_portfolio({symbol: df_dict[symbol] for symbol in symbols}, strategy)

    # Parameter optimization
    param_grid = {'fast_period': [5,9,13], 'slow_period': [21,34,50]}
    top_results = engine.optimize(df, StrategyClass, param_grid, symbol='INFY')
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import time as dtime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from broker.upstox.commission import CommissionModel, Segment
from backtester.order_types import (
    OrderType, PendingOrder, StopTracker, TrailingType,
    check_limit_fill, check_stop_fill, check_stop_limit_fill,
    make_trailing_stop,
)

logger = logging.getLogger(__name__)

INTRADAY_SQUAREOFF = dtime(15, 20)


# =============================================================================
# Core Data Classes (from engine.py)
# =============================================================================

@dataclass
class BacktestConfig:
    initial_capital:    float          = 500_000.0
    capital_risk_pct:   float          = 0.02
    stop_loss_atr_mult: float          = 2.0
    fixed_quantity:     int            = 0
    max_drawdown_pct:   float          = 0.20
    segment:            Segment        = Segment.EQUITY_DELIVERY
    allow_shorting:     bool           = False
    intraday_squareoff: bool           = False
    max_positions:      int            = 0
    lot_size:           int            = 1
    commission_model:   CommissionModel = field(default_factory=CommissionModel)


@dataclass
class Position:
    entry_time:    pd.Timestamp
    entry_price:   float
    quantity:      int
    direction:     int
    entry_signal:  str   = ""
    entry_charges: float = 0.0
    # index of bar at which the position was opened (for computing duration)
    entry_bar_idx: int   = 0
    mae:           float = 0.0
    mfe:           float = 0.0

    def update_excursion(self, price: float) -> None:
        move = (price - self.entry_price) * self.direction
        self.mfe = max(self.mfe, move)
        self.mae = min(self.mae, move)

    def unrealised_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.direction * self.quantity


@dataclass
class Trade:
    symbol:              str
    entry_time:          pd.Timestamp
    exit_time:           pd.Timestamp
    entry_price:         float
    exit_price:          float
    quantity:            int
    direction:           int
    direction_label:     str
    gross_pnl:           float
    entry_charges:       float
    exit_charges:        float
    total_charges:       float
    net_pnl:             float
    pnl_pct:             float
    entry_signal:        str   = ""
    exit_signal:         str   = ""
    duration:            str   = ""
    duration_bars:       int   = 0
    mae:                 float = 0.0
    mfe:                 float = 0.0
    cumulative_portfolio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "entry_time":      self.entry_time,
            "exit_time":       self.exit_time,
            "direction":       self.direction_label,
            "entry_price":     round(self.entry_price,  2),
            "exit_price":      round(self.exit_price,   2),
            "quantity":        self.quantity,
            "gross_pnl":       round(self.gross_pnl,    2),
            "entry_charges":   round(self.entry_charges, 2),
            "exit_charges":    round(self.exit_charges,  2),
            "total_charges":   round(self.total_charges, 2),
            "net_pnl":         round(self.net_pnl,       2),
            "pnl_pct":         round(self.pnl_pct,       4),
            "entry_signal":    self.entry_signal,
            "exit_signal":     self.exit_signal,
            "duration":        self.duration,
            "duration_bars":   self.duration_bars,
            "mae":             round(self.mae, 2),
            "mfe":             round(self.mfe, 2),
            "portfolio_value": round(self.cumulative_portfolio, 2),
        }


class BacktestResult:
    def __init__(self, config, trade_log, equity_curve, drawdown, signals_df, symbol):
        self.config       = config
        self.trade_log    = trade_log
        self.equity_curve = equity_curve
        self.drawdown     = drawdown
        self.signals_df   = signals_df
        self.symbol       = symbol
        self._metrics     = None

    def _compute_metrics(self) -> Dict:
        if self._metrics is not None:
            return self._metrics
        tl = self.trade_log
        if not tl:
            self._metrics = {"error": "No trades generated."}
            return self._metrics

        net_pnls = np.array([t.net_pnl      for t in tl])
        charges  = np.array([t.total_charges for t in tl])
        winners  = net_pnls[net_pnls > 0]
        losers   = net_pnls[net_pnls < 0]
        n_total  = len(tl)
        n_win    = len(winners)
        win_rate = n_win / n_total if n_total else 0
        avg_win  = winners.mean() if len(winners) else 0
        avg_loss = losers.mean()  if len(losers)  else 0
        pf       = (abs(winners.sum()) / abs(losers.sum())
                    if len(losers) and losers.sum() != 0 else float("inf"))
        rr       = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        exp      = net_pnls.mean()

        eq = self.equity_curve.dropna()
        max_dd   = self.drawdown.min() if not self.drawdown.empty else 0

        daily_ret = eq.resample("B").last().pct_change(fill_method=None).dropna() if len(eq) > 1 else pd.Series()
        sharpe    = ((daily_ret.mean() / daily_ret.std()) * (252**0.5)
                     if len(daily_ret) > 1 and daily_ret.std() > 0 else 0)

        start_v = self.config.initial_capital
        end_v   = eq.iloc[-1] if not eq.empty else start_v
        s_date  = eq.index[0].date()  if not eq.empty else None
        e_date  = eq.index[-1].date() if not eq.empty else None
        yrs     = max((e_date - s_date).days / 365.25, 1/365.25) if s_date and e_date else 1
        cagr    = ((end_v / start_v) ** (1 / yrs) - 1) * 100

        self._metrics = {
            "Symbol":              self.symbol,
            "Start Date":          str(s_date),
            "End Date":            str(e_date),
            "Initial Capital":     f"Rs {start_v:,.0f}",
            "Final Portfolio":     f"Rs {end_v:,.0f}",
            "Total Net P&L":       f"Rs {net_pnls.sum():,.2f}",
            "Total Return":        f"{((end_v/start_v)-1)*100:.2f}%",
            "CAGR":                f"{cagr:.2f}%",
            "Sharpe Ratio":        f"{sharpe:.2f}",
            "Max Drawdown":        f"{max_dd:.2f}%",
            "Total Trades":        n_total,
            "Winning Trades":      n_win,
            "Losing Trades":       len(losers),
            "Win Rate":            f"{win_rate*100:.1f}%",
            "Profit Factor":       f"{pf:.2f}",
            "Risk/Reward Ratio":   f"{rr:.2f}",
            "Avg Win":             f"Rs {avg_win:,.2f}",
            "Avg Loss":            f"Rs {avg_loss:,.2f}",
            "Expectancy/Trade":    f"Rs {exp:,.2f}",
            "Total Charges Paid":  f"Rs {charges.sum():,.2f}",
        }
        return self._metrics

    def summary(self) -> str:
        m = self._compute_metrics()
        if "error" in m:
            return f"Backtest completed — {m['error']}"
        lines = ["", "=" * 55, f"  BACKTEST RESULTS — {m['Symbol']}", "=" * 55]
        for k, v in m.items():
            if k == "Symbol":
                continue
            lines.append(f"  {k:<26}: {v}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def trade_df(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trade_log])

    def metrics_dict(self) -> Dict:
        return self._compute_metrics()

    # ------------------------------------------------------------------
    # Data export helpers
    # ------------------------------------------------------------------
    def export_signals_csv(self, path: str) -> None:
        """Write the signals DataFrame (ohlcv + indicators + signal column) to
        a CSV file at the specified ``path``.

        The DataFrame is exactly what the strategy returned in
        ``BacktestEngine.run``; it already includes the original OHLCV columns
        plus any indicator columns added by the strategy and the final
        ``signal`` column.
        """
        if self.signals_df is None or self.signals_df.empty:
            raise ValueError("No signal data available to export.")
        self.signals_df.to_csv(path)


# =============================================================================
# Enhanced Config (from engine_v2.py)
# =============================================================================

@dataclass
class BacktestConfigV3(BacktestConfig):
    """Extended BacktestConfig with advanced order types and output flags."""
    # ── Advanced Order Types ──────────────────────────────────────────────────
    default_order_type:  OrderType     = OrderType.MARKET
    limit_offset_pct:    float         = 0.2       # % below close for buy limit
    stop_loss_pct:       float         = 0.0       # 0 = disabled
    use_trailing_stop:   bool          = False
    trailing_stop_pct:   float         = 0.0       # 0 = use trailing_stop_amt
    trailing_stop_amt:   float         = 0.0       # 0 = use trailing_stop_pct

    # ── Output Flags ─────────────────────────────────────────────────────────
    save_trade_log:      bool          = False
    save_raw_data:       bool          = False
    save_chart:          bool          = False
    generate_summary:    bool          = False
    run_label:           str           = "backtest"
    max_candles:         int           = 2000      # Max candles to plot in chart


# =============================================================================
# Extended Position (from engine_v2.py)
# =============================================================================

@dataclass
class PositionV3(Position):
    """Position extended with stop tracker and pending order type."""
    stop_tracker:     Optional[StopTracker] = None
    fixed_stop_price: Optional[float]       = None   # Hard stop-loss price
    order_type:       OrderType             = OrderType.MARKET


# =============================================================================
# Output directory setup (from engine_v2.py)
# =============================================================================

_HERE = Path(__file__).resolve().parent.parent
OUTPUT_TRADE    = _HERE / "strategies" / "output" / "trade"
OUTPUT_RAW      = _HERE / "strategies" / "output" / "raw_data"
OUTPUT_CHART    = _HERE / "strategies" / "output" / "chart"
for _d in (OUTPUT_TRADE, OUTPUT_RAW, OUTPUT_CHART):
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Merged Backtest Engine
# =============================================================================

class BacktestEngineV3:
    """
    Merged backtesting engine combining features from engine.py and engine_v2.py.

    Features:
      • Basic backtesting with market orders (from engine.py)
      • Advanced order types (LIMIT, STOP, STOP-LIMIT, TRAILING STOP)
      • Multi-security portfolio runs with aggregated outputs
      • Parameter optimizer (grid search + random search)
      • All output flags (trade log, raw data, chart, summary)
      • Full Upstox commission modeling
    """

    def __init__(self, config: BacktestConfigV3) -> None:
        self.config = config

    # =========================================================================
    # Public API
    # =========================================================================

    def run(self, df: pd.DataFrame, strategy, symbol: str = "SYMBOL") -> BacktestResult:
        """
        Run backtest on a single symbol.

        Args:
            df:       OHLCV DataFrame (timezone-aware index recommended)
            strategy: Any BaseStrategy subclass instance
            symbol:   Symbol name for labelling

        Returns:
            BacktestResult
        """
        self._preflight_checks(df)
        logger.info(f"[V3] Running {strategy.name} on {symbol} ({len(df)} bars) | "
                    f"order={self.config.default_order_type.value}")

        signals_df = strategy.generate_signals(df)
        if "signal" not in signals_df.columns:
            raise ValueError("Strategy must add a 'signal' column.")

        result = self._event_loop(signals_df, symbol)
        self._handle_outputs(result, symbol)
        return result

    def run_portfolio(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy,
        label:     str = "",
    ) -> Dict[str, BacktestResult]:
        """
        Run the same strategy on a portfolio of symbols.

        All output files (trade log, raw data, chart, summary) are aggregated
        into SINGLE files across all symbols when output flags are True.

        Args:
            data_dict: {symbol: ohlcv_df} mapping
            strategy:  Strategy instance (same for all symbols)
            label:     Override run_label for this portfolio run

        Returns:
            {symbol: BacktestResult} mapping
        """
        run_label = label or self.config.run_label
        logger.info(f"[V3] Portfolio run: {len(data_dict)} symbols | label={run_label}")

        results:     Dict[str, BacktestResult] = {}
        all_trades:  List[dict]                = []
        all_raw:     List[pd.DataFrame]        = []
        summary_rows:List[dict]                = []

        for symbol, df in data_dict.items():
            try:
                self._preflight_checks(df)
                signals_df = strategy.generate_signals(df)
                if "signal" not in signals_df.columns:
                    logger.warning(f"No 'signal' column for {symbol} — skipped")
                    continue

                result = self._event_loop(signals_df, symbol)
                results[symbol] = result

                # Collect trade log rows
                if self.config.save_trade_log:
                    for t in result.trade_log:
                        all_trades.append(t.to_dict())

                # Collect raw data rows
                if self.config.save_raw_data:
                    raw = signals_df.copy()
                    raw.insert(0, "symbol", symbol)
                    all_raw.append(raw)

                # Summary row
                if self.config.generate_summary:
                    m = result._compute_metrics()
                    summary_rows.append(self._metrics_to_row(m, symbol))

                # Individual chart
                if self.config.save_chart:
                    self._save_chart(result, symbol, run_label)

                logger.info(f"  {symbol}: {len(result.trade_log)} trades | "
                            f"net={sum(t.net_pnl for t in result.trade_log):+,.0f}")

            except Exception as e:
                logger.error(f"  {symbol}: ERROR — {e}")

        # ── Write aggregated files ────────────────────────────────────────────
        if self.config.save_trade_log and all_trades:
            fpath = OUTPUT_TRADE / f"{run_label}_trade_log.csv"
            pd.DataFrame(all_trades).to_csv(fpath, index=False)
            logger.info(f"Trade log saved: {fpath} ({len(all_trades)} rows)")

        if self.config.save_raw_data and all_raw:
            combined = pd.concat(all_raw, axis=0)
            fpath = OUTPUT_RAW / f"{run_label}_raw_data.csv"
            combined.to_csv(fpath)
            logger.info(f"Raw data saved: {fpath}")

        if self.config.generate_summary and summary_rows:
            fpath = OUTPUT_TRADE / f"{run_label}_summary.csv"
            pd.DataFrame(summary_rows).to_csv(fpath, index=False)
            logger.info(f"Summary saved: {fpath}")

        logger.info(f"Portfolio run complete: {len(results)}/{len(data_dict)} symbols OK")
        return results

    # =========================================================================
    # Parameter Optimizer
    # =========================================================================

    def optimize(
        self,
        df:             pd.DataFrame,
        strategy_class,
        param_grid:     Dict[str, list],
        symbol:         str  = "SYMBOL",
        metric:         str  = "Sharpe Ratio",
        method:         str  = "grid",        # 'grid' or 'random'
        n_random:       int  = 50,
        top_n:          int  = 5,
    ) -> pd.DataFrame:
        """
        Fast parameter optimizer using vectorised pre-computation.

        Args:
            df:             OHLCV DataFrame
            strategy_class: Strategy class (not instance) to instantiate per trial
            param_grid:     {'param_name': [value1, value2, ...], ...}
            symbol:         Symbol name
            metric:         Which metric to optimise (must be in metrics_dict())
            method:         'grid' or 'random'
            n_random:       Number of random trials (only for method='random')
            top_n:          Number of top results to return

        Returns:
            DataFrame with columns = param names + metric values, sorted by metric
        """
        import itertools, random

        self._preflight_checks(df)

        # Generate all combinations or random subset
        keys   = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        if method == "grid":
            combos = list(itertools.product(*values))
        else:
            all_combos = list(itertools.product(*values))
            combos     = random.sample(all_combos, min(n_random, len(all_combos)))

        logger.info(f"Optimizer: {len(combos)} trials | metric={metric} | "
                    f"method={method} | {symbol}")

        # Temporarily disable all output for speed during optimization
        orig_flags = (
            self.config.save_trade_log, self.config.save_raw_data,
            self.config.save_chart,     self.config.generate_summary,
        )
        self.config.save_trade_log  = False
        self.config.save_raw_data   = False
        self.config.save_chart      = False
        self.config.generate_summary = False

        rows = []
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                strategy = strategy_class(**params)
                signals_df = strategy.generate_signals(df)
                if "signal" not in signals_df.columns:
                    continue
                result  = self._event_loop(signals_df, symbol)
                metrics = result._compute_metrics()

                # Extract numeric metric value for sorting
                raw_val = metrics.get(metric, "0")
                try:
                    num_val = float(str(raw_val).replace("%", "")
                                                 .replace("Rs", "")
                                                 .replace(",", "").strip())
                except (ValueError, AttributeError):
                    num_val = 0.0

                row = {**params,
                       metric:          num_val,
                       "n_trades":      metrics.get("Total Trades", 0),
                       "win_rate":      metrics.get("Win Rate", "0%"),
                       "max_drawdown":  metrics.get("Max Drawdown", "0%"),
                       "total_return":  metrics.get("Total Return", "0%"),
                       "profit_factor": metrics.get("Profit Factor", "0"),
                       }
                rows.append(row)

            except Exception as e:
                logger.debug(f"  Trial {params} failed: {e}")

        # Restore output flags
        (self.config.save_trade_log, self.config.save_raw_data,
         self.config.save_chart, self.config.generate_summary) = orig_flags

        if not rows:
            logger.warning("Optimizer: no valid results produced.")
            return pd.DataFrame()

        result_df = pd.DataFrame(rows).sort_values(metric, ascending=False)
        result_df = result_df.reset_index(drop=True)

        logger.info(f"Optimizer complete. Best {metric}: "
                    f"{result_df[metric].iloc[0]:.3f} "
                    f"→ {dict(result_df.iloc[0][keys])}")

        # Optionally save optimizer results
        if self.config.save_trade_log:
            fpath = OUTPUT_TRADE / f"{self.config.run_label}_optimizer.csv"
            result_df.to_csv(fpath, index=False)
            logger.info(f"Optimizer results saved: {fpath}")

        return result_df.head(top_n)

    # =========================================================================
    # Event Loop (core engine - merged from both versions)
    # =========================================================================

    def _event_loop(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        cfg        = self.config
        commission = cfg.commission_model
        cash       = cfg.initial_capital
        positions: List[PositionV3] = []
        trade_log: List[Trade]      = []

        equity_curve = pd.Series(np.nan, index=df.index, dtype=float)
        drawdown     = pd.Series(np.nan, index=df.index, dtype=float)
        peak_equity  = cfg.initial_capital

        prices  = df["close"].values
        highs   = df["high"].values
        lows    = df["low"].values
        opens   = df["open"].values
        signals = df["signal"].fillna(0).astype(int).values
        times   = df.index
        n       = len(df)

        # Pre-compute ATR for risk-based sizing
        atr_vals = None
        if cfg.stop_loss_atr_mult > 0 and cfg.fixed_quantity == 0:
            from indicators.technical import atr as _atr
            atr_vals = _atr(df, 14).values

        # Pending limit/stop entry orders {bar_index: PendingOrder}
        pending_entries: List[PendingOrder] = []

        for i in range(n):
            ct  = times[i]
            cp  = prices[i]
            hp  = highs[i]
            lp  = lows[i]
            op  = opens[i]

            if np.isnan(op) or op <= 0:
                equity_curve.iloc[i] = cash
                continue

            # ── 1. Update trailing stops and check triggers ───────────────────
            for pos in list(positions):
                if pos.stop_tracker is not None:
                    pos.stop_tracker.update(hp, lp)

                    triggered, fill_price = pos.stop_tracker.is_triggered(op, lp, hp)
                    if triggered:
                        res  = self._close_pos(pos, fill_price, ct, cash, commission,
                                               symbol, "Trailing Stop", i,
                                               cash + sum(p.unrealised_pnl(cp) for p in positions))
                        cash = res["cash"]
                        trade_log.append(res["trade"])
                        positions.remove(pos)

            # ── 2. Check fixed stop-loss ───────────────────────────────────
            for pos in list(positions):
                if pos.fixed_stop_price is not None:
                    triggered, fill_price = check_stop_fill(
                        -pos.direction, pos.fixed_stop_price, op, hp, lp
                    )
                    if triggered:
                        res  = self._close_pos(pos, fill_price, ct, cash, commission,
                                               symbol, f"Stop-Loss @{pos.fixed_stop_price:.2f}",
                                               i, cash)
                        cash = res["cash"]
                        trade_log.append(res["trade"])
                        positions.remove(pos)

            # ── 3. Check pending limit entry orders ────────────────────────
            for porder in list(pending_entries):
                # Expire old orders
                bars_open = i - porder.signal_bar
                if porder.expires_after > 0 and bars_open > porder.expires_after:
                    pending_entries.remove(porder)
                    continue

                if porder.order_type == OrderType.LIMIT:
                    filled, fill_price = check_limit_fill(
                        porder.direction, porder.limit_price, op, hp, lp
                    )
                    if filled:
                        result = self._open_pos(
                            porder.direction, porder.quantity, fill_price, ct,
                            cash, commission, symbol, "Limit Fill", i, atr_vals,
                        )
                        if result:
                            cash, new_pos = result
                            positions.append(new_pos)
                        pending_entries.remove(porder)

            # ── 4. Intraday squareoff ──────────────────────────────────────
            if cfg.intraday_squareoff and hasattr(ct, "time"):
                if ct.time() >= INTRADAY_SQUAREOFF and positions:
                    for pos in list(positions):
                        res  = self._close_pos(pos, cp, ct, cash, commission,
                                               symbol, "MIS Squareoff 15:20", i, cash)
                        cash = res["cash"]
                        trade_log.append(res["trade"])
                    positions.clear()
                    pending_entries.clear()

            # ── 5. Update excursions + equity ──────────────────────────────
            for pos in positions:
                pos.update_excursion(cp)

            pos_value = sum(p.unrealised_pnl(cp) + p.entry_price * p.quantity
                            for p in positions)
            pv = cash + pos_value
            equity_curve.iloc[i] = pv
            peak_equity  = max(peak_equity, pv)
            dd = ((pv - peak_equity) / peak_equity) * 100
            drawdown.iloc[i] = dd

            if dd <= -(cfg.max_drawdown_pct * 100):
                logger.warning(f"DRAWDOWN ALERT {ct}: {dd:.1f}% from peak")

            if i == 0:
                continue   # No prior signal

            prev_sig   = signals[i - 1]
            exec_price = op
            prev_close = prices[i - 1]

            # ── 6. Execute prior bar's signal ──────────────────────────────
            if prev_sig == -1:
                # Close long positions (FIFO)
                if positions:
                    pos = positions.pop(0)
                    exit_price = self._get_exit_price(pos, exec_price, cfg)
                    res  = self._close_pos(pos, exit_price, ct, cash, commission,
                                           symbol, "Signal -1", i, pv)
                    cash = res["cash"]
                    trade_log.append(res["trade"])
                elif cfg.allow_shorting:
                    result = self._handle_entry_signal(-1, exec_price, prev_close, ct,
                                                       cash, commission, symbol, i,
                                                       atr_vals, positions, pending_entries)
                    if isinstance(result, tuple):
                        cash = result[0]

            elif prev_sig == 1:
                limit_hit = cfg.max_positions > 0 and len(positions) >= cfg.max_positions
                if not limit_hit:
                    result = self._handle_entry_signal(1, exec_price, prev_close, ct,
                                                       cash, commission, symbol, i,
                                                       atr_vals, positions, pending_entries)
                    if isinstance(result, tuple):
                        cash = result[0]

        # ── End of data — close all remaining positions ────────────────────
        for pos in positions:
            res = self._close_pos(pos, prices[-1], times[-1], cash, commission,
                                  symbol, "End of Backtest", n - 1,
                                  equity_curve.dropna().iloc[-1] if not equity_curve.dropna().empty else cash)
            cash = res["cash"]
            trade_log.append(res["trade"])

        logger.info(f"[V3] {symbol}: {len(trade_log)} trades | "
                    f"final={equity_curve.dropna().iloc[-1] if not equity_curve.dropna().empty else cash:,.2f}")
        return BacktestResult(cfg, trade_log, equity_curve, drawdown, df, symbol)

    # =========================================================================
    # Entry Signal Handler
    # =========================================================================

    def _handle_entry_signal(
        self, direction: int, exec_price: float, prev_close: float,
        ct, cash: float, commission, symbol: str, bar_idx: int,
        atr_vals, positions: list, pending_entries: list,
    ):
        cfg      = self.config
        ot       = cfg.default_order_type
        qty      = self._qty(cash, exec_price, atr_vals[bar_idx] if atr_vals is not None else None)
        if qty <= 0:
            return None

        if ot == OrderType.MARKET:
            result = self._open_pos(direction, qty, exec_price, ct, cash, commission,
                                    symbol, "Market Signal", bar_idx, atr_vals)
            if result:
                cash, new_pos = result
                positions.append(new_pos)
                return (cash,)

        elif ot == OrderType.LIMIT:
            # Place limit order below (buy) / above (sell) previous close
            offset = prev_close * (cfg.limit_offset_pct / 100.0)
            lim    = prev_close - offset if direction == 1 else prev_close + offset
            pending_entries.append(PendingOrder(
                direction=direction, order_type=OrderType.LIMIT,
                quantity=qty, signal_bar=bar_idx, limit_price=lim,
                expires_after=5,  # GTD: good for 5 bars
            ))
            logger.debug(f"  Limit order placed: {direction} {qty}@{lim:.2f}")

        elif ot == OrderType.STOP:
            # Stop entry: enter on breakout of signal close
            offset = prev_close * (cfg.limit_offset_pct / 100.0)
            stop   = prev_close + offset if direction == 1 else prev_close - offset
            result = self._open_pos(direction, qty, exec_price, ct, cash, commission,
                                    symbol, f"Stop Entry@{stop:.2f}", bar_idx, atr_vals)
            if result:
                cash, new_pos = result
                positions.append(new_pos)
                return (cash,)

        return None

    # =========================================================================
    # Open Position Helper
    # =========================================================================

    def _open_pos(
        self, direction: int, qty: int, exec_price: float, ct,
        cash: float, commission, symbol: str, entry_signal: str,
        bar_idx: int, atr_vals,
    ):
        """Open a new position and attach stop tracker / fixed stop if configured."""
        cfg        = self.config
        order_side = "BUY" if direction == 1 else "SELL"
        chg        = commission.calculate(cfg.segment, order_side, qty, exec_price)
        cost       = exec_price * qty + chg.total if direction == 1 else chg.total

        if cash < cost:
            return None

        cash -= cost

        new_pos = PositionV3(
            entry_time=ct, entry_price=exec_price, quantity=qty, direction=direction,
            entry_signal=entry_signal, entry_charges=chg.total,
            entry_bar_idx=bar_idx,
            order_type=cfg.default_order_type,
        )

        # Attach trailing stop if configured
        if cfg.use_trailing_stop:
            pct = cfg.trailing_stop_pct if cfg.trailing_stop_pct > 0 else None
            amt = cfg.trailing_stop_amt if cfg.trailing_stop_amt > 0 else None
            new_pos.stop_tracker = make_trailing_stop(direction, exec_price,
                                                      trail_pct=pct, trail_amount=amt)

        # Attach fixed percentage stop-loss
        if cfg.stop_loss_pct > 0:
            dist = exec_price * (cfg.stop_loss_pct / 100.0)
            new_pos.fixed_stop_price = (exec_price - dist if direction == 1
                                        else exec_price + dist)

        return cash, new_pos

    def _get_exit_price(self, pos: PositionV3, market_price: float,
                        cfg: BacktestConfigV3) -> float:
        """Determine exit price based on order type for closes."""
        # For now, market exit always uses next open (already exec_price)
        return market_price

    # =========================================================================
    # Output Handlers
    # =========================================================================

    def _handle_outputs(self, result: BacktestResult, symbol: str) -> None:
        """Route outputs to correct folders based on config flags."""
        cfg   = self.config
        label = cfg.run_label

        if cfg.save_trade_log:
            self._save_trade_log_single(result, symbol, label)

        if cfg.save_raw_data:
            self._save_raw_data_single(result, symbol, label)

        if cfg.save_chart:
            self._save_chart(result, symbol, label)

    def _save_trade_log_single(self, result: BacktestResult,
                                symbol: str, label: str) -> None:
        """Save trade log for a single symbol to strategies/output/trade/."""
        if not result.trade_log:
            return
        fpath = OUTPUT_TRADE / f"{label}_{symbol}_trade_log.csv"
        df    = result.trade_df()
        df.to_csv(fpath, index=False)
        logger.info(f"Trade log: {fpath}")

    def _save_raw_data_single(self, result: BacktestResult,
                               symbol: str, label: str) -> None:
        """Save OHLCV + indicators + signals CSV to strategies/output/raw_data/."""
        df = result.signals_df.copy()
        df.insert(0, "symbol", symbol)
        fpath = OUTPUT_RAW / f"{label}_{symbol}_raw_data.csv"
        df.to_csv(fpath)
        logger.info(f"Raw data: {fpath}")

    def _save_chart(self, result: BacktestResult,
                    symbol: str, label: str) -> Optional[str]:
        """Save Streak-style chart to strategies/output/chart/."""
        try:
            from backtester.report import generate_report
            fpath = generate_report(
                result, symbol=symbol,
                output_dir=str(OUTPUT_CHART),
                filename=f"{label}_{symbol}_chart.png",
                show=False,
                max_candles=self.config.max_candles,
            )
            logger.info(f"Chart: {fpath}")
            return fpath
        except Exception as e:
            logger.error(f"Chart generation failed for {symbol}: {e}")
            return None

    def _metrics_to_row(self, metrics: dict, symbol: str) -> dict:
        """Flatten a metrics dict to a single-row dict for the summary sheet."""
        def _num(v):
            try:
                return float(str(v).replace("%", "").replace("Rs", "")
                              .replace(",", "").strip())
            except (ValueError, TypeError):
                return v

        return {
            "symbol":          symbol,
            "start_date":      metrics.get("Start Date", ""),
            "end_date":        metrics.get("End Date", ""),
            "total_trades":    metrics.get("Total Trades", 0),
            "win_rate_pct":    _num(metrics.get("Win Rate", 0)),
            "profit_factor":   _num(metrics.get("Profit Factor", 0)),
            "sharpe_ratio":    _num(metrics.get("Sharpe Ratio", 0)),
            "max_drawdown_pct":_num(metrics.get("Max Drawdown", 0)),
            "total_return_pct":_num(metrics.get("Total Return", 0)),
            "cagr_pct":        _num(metrics.get("CAGR", 0)),
            "total_net_pnl":   _num(metrics.get("Total Net P&L", 0)),
            "total_charges":   _num(metrics.get("Total Charges Paid", 0)),
            "final_portfolio": _num(metrics.get("Final Portfolio", 0)),
            "expectancy":      _num(metrics.get("Expectancy/Trade", 0)),
        }

    # =========================================================================
    # Shared helpers from base engine
    # =========================================================================

    def _qty(self, cash: float, price: float, atr_val) -> int:
        cfg = self.config
        if cfg.fixed_quantity > 0:
            qty = cfg.fixed_quantity
        elif (atr_val is not None and not np.isnan(atr_val) and
              atr_val > 0 and cfg.stop_loss_atr_mult > 0):
            risk_amt  = cash * cfg.capital_risk_pct
            stop_dist = atr_val * cfg.stop_loss_atr_mult
            qty       = int(risk_amt / stop_dist)
        else:
            qty = int(cash * cfg.capital_risk_pct / price)

        if cfg.lot_size > 1:
            qty = (qty // cfg.lot_size) * cfg.lot_size

        return max(0, min(qty, int(cash / max(price, 1))))

    def _close_pos(self, pos, exit_price, exit_time, cash, commission,
                   symbol, exit_signal, bar_idx, portfolio_value) -> dict:
        cfg        = self.config
        exit_side  = "SELL" if pos.direction == 1 else "BUY"
        exit_chg   = commission.calculate(cfg.segment, exit_side, pos.quantity, exit_price)
        gross_pnl  = (exit_price - pos.entry_price) * pos.direction * pos.quantity
        total_chg  = pos.entry_charges + exit_chg.total
        net_pnl    = gross_pnl - total_chg

        if pos.direction == 1:
            cash += exit_price * pos.quantity - exit_chg.total
        else:
            cash += (pos.entry_price - exit_price) * pos.quantity - exit_chg.total

        td        = exit_time - pos.entry_time
        # compute elapsed time normally (24‑hour days)
        total_s   = td.total_seconds()
        days      = td.days
        hrs       = int((total_s - days * 86400) // 3600)
        mins      = int((total_s - days * 86400) % 3600 // 60)
        dur_str   = f"{days}d {hrs:02d}h {mins:02d}m"

        # duration in bars based on stored entry index
        duration_bars = bar_idx - pos.entry_bar_idx

        cost_basis = pos.entry_price * pos.quantity
        pnl_pct    = (net_pnl / cost_basis * 100) if cost_basis > 0 else 0

        trade = Trade(
            symbol=symbol, entry_time=pos.entry_time, exit_time=exit_time,
            entry_price=pos.entry_price, exit_price=exit_price,
            quantity=pos.quantity, direction=pos.direction,
            direction_label="LONG" if pos.direction == 1 else "SHORT",
            gross_pnl=round(gross_pnl, 2),
            entry_charges=round(pos.entry_charges, 2),
            exit_charges=round(exit_chg.total, 2),
            total_charges=round(total_chg, 2),
            net_pnl=round(net_pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            entry_signal=pos.entry_signal, exit_signal=exit_signal,
            duration=dur_str, duration_bars=duration_bars,
            mae=round(pos.mae, 4), mfe=round(pos.mfe, 4),
            cumulative_portfolio=round(portfolio_value, 2),
        )
        return {"cash": cash, "trade": trade}

    def _preflight_checks(self, df: pd.DataFrame) -> None:
        required = ["open", "high", "low", "close", "volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        if len(df) < 100:
            warnings.warn(
                f"Only {len(df)} bars — reliable backtests need 100+ bars.",
                UserWarning,
            )