"""
backtester/engine.py  — Core backtesting engine.
See module docstring in the file for full documentation.
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import time as dtime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtester.commission import CommissionModel, Segment

logger = logging.getLogger(__name__)
INTRADAY_SQUAREOFF = dtime(15, 20)


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


class BacktestEngine:
    """
    Main backtesting engine.

    Execution model: signal on bar[i] → execute on bar[i+1]'s open.
    Supports pyramiding, shorting, intraday squareoff, and full
    Upstox commission modelling.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(self, df: pd.DataFrame, strategy, symbol: str = "SYMBOL") -> BacktestResult:
        self._preflight_checks(df)
        logger.info(f"Running: {strategy.name} on {symbol} ({len(df)} bars)")
        signals_df = strategy.generate_signals(df)
        if "signal" not in signals_df.columns:
            raise ValueError("Strategy must add a 'signal' column to the DataFrame.")
        return self._event_loop(signals_df, symbol)

    def _event_loop(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        cfg        = self.config
        commission = cfg.commission_model
        cash       = cfg.initial_capital
        positions: List[Position] = []
        trade_log: List[Trade]   = []

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

        # ATR series for risk-based position sizing
        atr_vals = None
        if cfg.stop_loss_atr_mult > 0 and cfg.fixed_quantity == 0:
            from indicators.technical import atr as _atr
            atr_vals = _atr(df, 14).values

        for i in range(n):
            ct = times[i]
            cp = prices[i]

            # Update excursions for all open positions
            for pos in positions:
                pos.update_excursion(cp)

            # Intraday squareoff at 15:20 IST
            if cfg.intraday_squareoff and hasattr(ct, "time"):
                if ct.time() >= INTRADAY_SQUAREOFF and positions:
                    for pos in list(positions):
                        res  = self._close_pos(pos, cp, ct, cash, commission,
                                               symbol, "MIS Squareoff 15:20", i, cash)
                        cash = res["cash"]
                        trade_log.append(res["trade"])
                    positions.clear()

            # Portfolio value
            pos_value = sum(p.unrealised_pnl(cp) +
                            p.entry_price * p.quantity for p in positions)
            pv = cash + pos_value
            equity_curve.iloc[i] = pv
            peak_equity = max(peak_equity, pv)
            dd = ((pv - peak_equity) / peak_equity) * 100
            drawdown.iloc[i] = dd

            if dd <= -(cfg.max_drawdown_pct * 100):
                logger.warning(
                    f"DRAWDOWN ALERT {ct}: portfolio down {dd:.1f}% from peak."
                )

            if i == 0:
                continue  # No prior signal to execute yet

            prev_sig   = signals[i - 1]
            exec_price = opens[i]
            if exec_price <= 0 or np.isnan(exec_price):
                continue

            # --- CLOSE / SHORT signal ---
            if prev_sig == -1:
                if positions:
                    # Close oldest position (FIFO)
                    pos = positions.pop(0)
                    res  = self._close_pos(pos, exec_price, ct, cash, commission,
                                           symbol, "Signal -1", i, pv)
                    cash = res["cash"]
                    trade_log.append(res["trade"])
                elif cfg.allow_shorting:
                    qty = self._qty(cash, exec_price,
                                   atr_vals[i] if atr_vals is not None else None)
                    if qty > 0:
                        chg = commission.calculate(cfg.segment, "SELL", qty, exec_price)
                        if cash >= chg.total:
                            cash -= chg.total
                            positions.append(Position(
                                entry_time=ct, entry_price=exec_price,
                                quantity=qty, direction=-1,
                                entry_signal="Short -1", entry_charges=chg.total,
                                entry_bar_idx=i,
                            ))

            # --- BUY / LONG signal ---
            elif prev_sig == 1:
                limit_hit = cfg.max_positions > 0 and len(positions) >= cfg.max_positions
                if not limit_hit:
                    qty = self._qty(cash, exec_price,
                                   atr_vals[i] if atr_vals is not None else None)
                    if qty > 0:
                        chg  = commission.calculate(cfg.segment, "BUY", qty, exec_price)
                        cost = exec_price * qty + chg.total
                        if cash >= cost:
                            cash -= cost
                            positions.append(Position(
                                entry_time=ct, entry_price=exec_price,
                                quantity=qty, direction=1,
                                entry_signal="Long +1", entry_charges=chg.total,
                                entry_bar_idx=i,
                            ))

        # End of data — close remaining positions at last close
        for pos in positions:
            res = self._close_pos(pos, prices[-1], times[-1], cash, commission,
                                  symbol, "End of Backtest", n-1,
                                  equity_curve.iloc[-1])
            cash = res["cash"]
            trade_log.append(res["trade"])

        logger.info(f"Done: {len(trade_log)} trades, final={equity_curve.iloc[-1]:,.2f}")
        return BacktestResult(cfg, trade_log, equity_curve, drawdown, df, symbol)

    def _qty(self, cash: float, price: float, atr_val) -> int:
        cfg = self.config
        if cfg.fixed_quantity > 0:
            qty = cfg.fixed_quantity
        elif (atr_val is not None and not np.isnan(atr_val) and
              atr_val > 0 and cfg.stop_loss_atr_mult > 0):
            risk_amt  = cash * cfg.capital_risk_pct
            stop_dist = atr_val * cfg.stop_loss_atr_mult
            qty = int(risk_amt / stop_dist)
        else:
            qty = int(cash * cfg.capital_risk_pct / price)

        if cfg.lot_size > 1:
            qty = (qty // cfg.lot_size) * cfg.lot_size

        return max(0, min(qty, int(cash / price)))

    def _close_pos(self, pos: Position, exit_price: float, exit_time,
                   cash: float, commission: CommissionModel,
                   symbol: str, exit_signal: str, bar_idx: int,
                   portfolio_value: float) -> dict:
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
        # compute human-readable duration using real elapsed time
        total_s   = td.total_seconds()
        days      = td.days
        hrs       = int((total_s - days * 86400) // 3600)
        mins      = int((total_s - days * 86400) % 3600 // 60)
        dur_str   = f"{days}d {hrs:02d}h {mins:02d}m"

        # calculate bars held based on entry index stored in position
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
            raise ValueError(f"Input DataFrame missing columns: {missing}.")
        if len(df) < 100:
            warnings.warn(
                f"Only {len(df)} bars. Backtests need 100+ bars for reliability.",
                UserWarning,
            )
