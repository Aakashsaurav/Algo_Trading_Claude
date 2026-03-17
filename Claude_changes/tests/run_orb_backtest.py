"""
run_orb_backtest.py
====================
Opening Range Breakout (ORB) Backtest Runner — NIFTY 50 Index

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
From the project root (algo_trading/):

    python run_orb_backtest.py

All user-facing parameters are in the CONFIG section at the top.
No other file needs to change to adjust the backtest.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS SCRIPT DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Downloads / loads cached 1-minute NIFTY 50 data for 5 years via
   broker.upstox.data_manager.get_ohlcv()
2. Runs ORBNiftyStrategy.generate_signals() to compute OR levels,
   VWAP, trailing stop increments, and signal column
3. Executes the custom ORB event loop which enforces:
   - Per-trade stop-loss = open of signal candle (NOT a fixed %)
   - Trailing stop: advance by |prev_open−prev_close|/5 each bar
   - 100% capital utilisation per trade (qty = floor(cash / price))
   - Hard squareoff at 15:15 IST
4. Computes 25+ performance metrics via BacktestResult.metrics()
5. Prints the full report to the terminal
6. Saves all output files (trade log CSV, signal CSV, summary JSON, chart PNG)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY A CUSTOM EVENT LOOP?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BacktestEngine's built-in stop-loss is a FIXED % applied identically
to all trades. ORB requires a PER-TRADE stop (the open of each unique
signal candle). A custom loop reads 'signal_sl' from the strategy output
and applies it individually per entry. It still uses FillEngine for all
commission calculations — no duplication of that logic.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, time as dtime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import config, setup_logging
from backtester.models import BacktestConfig, BacktestResult, OrderType, Trade
from backtester.fill_engine import FillEngine
from backtester.models import Position
from broker.upstox.commission import Segment
from strategies.day_strategy.orb_nifty import (
    ORBNiftyStrategy,
    SQUAREOFF_TIME,
)

setup_logging()
logger = logging.getLogger("orb_runner")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USER CONFIGURATION — edit here only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Instrument
INSTRUMENT_TYPE  = "INDEX"
EXCHANGE         = "NSE"
TRADING_SYMBOL   = "NIFTY"
SYMBOL_LABEL     = "NIFTY 50"          # output file prefix (no spaces)

# Date range — last 5 years
TO_DATE   = date.today().isoformat()
FROM_DATE = (date.today() - timedelta(days=365 * 5)).isoformat()

# Capital
INITIAL_CAPITAL = 500_000.0           # ₹5,00,000

# Strategy parameters
OR_WINDOW_MINUTES = 15                # OR = first 15 min → 09:15 to 09:29
VWAP_FILTER       = True
TRAILING_DIVISOR  = 5.0
MIN_BODY_PCT      = 0.0               # 0 = disabled

# Run label (appears in output filenames)
RUN_LABEL = f"orb_nifty50_{FROM_DATE[:4]}_to_{TO_DATE[:4]}"

# Output directories (relative to project root)
OUT_TRADE = _ROOT / "strategies" / "output" / "trade"
OUT_RAW   = _ROOT / "strategies" / "output" / "raw_data"
OUT_CHART = _ROOT / "strategies" / "output" / "chart"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Data loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_nifty_data(from_date: str, to_date: str) -> pd.DataFrame:
    """
    Download / load cached 1-minute NIFTY 50 OHLCV data.

    Uses broker.upstox.data_manager.get_ohlcv() which:
      1. Checks local Parquet cache first (instant if available)
      2. Downloads any missing months from Upstox API (chunked, incremental)
      3. Returns a timezone-aware (Asia/Kolkata) 1-minute DataFrame

    The function also filters to NSE market hours (09:15–15:30) and
    removes any duplicate timestamps.

    Parameters
    ----------
    from_date : str   'YYYY-MM-DD'
    to_date   : str   'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
        1-minute OHLCV, DatetimeIndex (IST), columns: open high low close volume oi
    """
    logger.info(f"Loading NIFTY 50 1-min data: {from_date} → {to_date}")

    from broker.upstox.data_manager import get_ohlcv

    df = get_ohlcv(
        instrument_type = INSTRUMENT_TYPE,
        exchange        = EXCHANGE,
        trading_symbol  = TRADING_SYMBOL,
        unit            = "minutes",
        interval        = 1,
        from_date       = from_date,
        to_date         = to_date,
    )

    if df is None or df.empty:
        raise RuntimeError(
            "get_ohlcv() returned no data for NIFTY 50. "
            "Check Upstox API credentials and network."
        )

    # Normalise dtypes
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    if "volume" not in df.columns:
        df["volume"] = 0
        logger.warning("'volume' column absent — set to 0 (index data has no volume).")
    if "oi" not in df.columns:
        df["oi"] = 0

    # Filter to market hours only: 09:15–15:30
    bar_time = df.index.time
    df = df[
        (pd.Series(bar_time, index=df.index) >= dtime(9, 15))
        & (pd.Series(bar_time, index=df.index) <= dtime(15, 30))
    ]

    # Deduplicate and sort
    df = df[~df.index.duplicated(keep="last")].sort_index()

    n_days = len(np.unique(df.index.date))
    logger.info(
        f"Data ready: {len(df):,} bars | {n_days} trading days | "
        f"{from_date} → {to_date}"
    )
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: BacktestConfig
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_config() -> BacktestConfig:
    """
    Build the BacktestConfig for the ORB backtest.

    The config controls commission calculation, risk guards, and output
    flags. Stop-loss and trailing stop are managed by the custom event
    loop (not by the config parameters) because they are per-trade.
    """
    return BacktestConfig(
        initial_capital    = INITIAL_CAPITAL,
        # Risk — 100% capital utilisation computed in event loop
        capital_risk_pct   = 0.99,
        fixed_quantity     = 0,
        max_positions      = 1,
        max_drawdown_pct   = 0.60,        # halt if 60% of capital is lost
        # Market
        segment            = Segment.EQUITY_INTRADAY,
        allow_shorting     = True,
        intraday_squareoff = True,        # additional safety net in engine
        # Order type
        default_order_type = OrderType.MARKET,
        # Stop-loss: managed per-trade in the event loop, not by config
        stop_loss_pct      = 0.0,
        stop_loss_atr_mult = 0.0,
        use_trailing_stop  = False,
        # Outputs — all True
        save_trade_log     = True,
        save_raw_data      = True,
        save_chart         = True,
        generate_summary   = True,
        run_label          = RUN_LABEL,
        max_candles        = 2000,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Custom ORB event loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_orb_event_loop(
    signals_df: pd.DataFrame,
    cfg:        BacktestConfig,
    symbol:     str,
) -> Tuple[List[Trade], pd.Series, pd.Series]:
    """
    Custom bar-by-bar simulation enforcing ORB-specific execution rules.

    Differences from BacktestEngine's standard event loop:
    ┌─────────────────────────────┬───────────────────────────────────────┐
    │ Standard engine             │ This loop                             │
    ├─────────────────────────────┼───────────────────────────────────────┤
    │ Fixed SL % from config      │ Per-trade SL from signal_sl column    │
    │ Config trailing stop %      │ Per-bar trail from trail_increment col│
    │ Qty from capital_risk_pct   │ Qty = floor(cash / entry_price) → 100%│
    │ Squareoff at 15:20          │ Squareoff at 15:15                    │
    └─────────────────────────────┴───────────────────────────────────────┘

    All commission math is delegated to FillEngine (no duplication).

    Parameters
    ----------
    signals_df : pd.DataFrame
        Output of ORBNiftyStrategy.generate_signals(). Must contain:
        open, high, low, close, signal, signal_sl, trail_increment.
    cfg    : BacktestConfig
    symbol : str

    Returns
    -------
    (trade_log, equity_curve, drawdown)
        trade_log    : list[Trade]
        equity_curve : pd.Series — portfolio value at each bar close
        drawdown     : pd.Series — fractional dd from peak (≤ 0)
    """
    # ── Extract NumPy arrays (avoids per-bar Series attribute lookups) ───────
    n          = len(signals_df)
    times      = signals_df.index
    opens_arr  = signals_df["open"].values.astype(float)
    highs_arr  = signals_df["high"].values.astype(float)
    lows_arr   = signals_df["low"].values.astype(float)
    closes_arr = signals_df["close"].values.astype(float)
    sig_arr    = signals_df["signal"].fillna(0).astype(int).values
    sl_arr     = signals_df["signal_sl"].values.astype(float)
    trail_arr  = signals_df["trail_increment"].values.astype(float)

    # ── Initialise state ─────────────────────────────────────────────────────
    filler     = FillEngine(cfg)
    cash       = cfg.initial_capital
    position: Optional[Position] = None
    current_sl = np.nan

    # Per-day direction guard: {date_str: set of directions entered}
    day_dirs: dict = {}

    # Pre-allocated output arrays
    equity_arr   = np.full(n, np.nan, dtype=float)
    drawdown_arr = np.full(n, np.nan, dtype=float)
    peak_equity  = cfg.initial_capital
    trade_log: List[Trade] = []

    logger.info(f"[ORB] Event loop start: {n:,} bars | {symbol}")

    # ─────────────────────────────────────────────────────────────────────────
    for i in range(n):
        op = opens_arr[i]
        hp = highs_arr[i]
        lp = lows_arr[i]
        cp = closes_arr[i]
        ct = times[i]

        # Skip invalid bars (can occur at boundaries due to data issues)
        if np.isnan(op) or op <= 0:
            equity_arr[i] = cash
            continue

        # ── A. Update MAE / MFE for open position ─────────────────────────
        if position is not None:
            position.update_excursion((hp + lp) / 2.0)

        # ── B. Stop-loss check (evaluated at bar OPEN / INTRABAR) ─────────
        if position is not None and not np.isnan(current_sl):
            sl_hit   = False
            sl_price = 0.0

            if position.direction == 1:           # LONG: stop if low < SL
                if op <= current_sl:              # Gap open below stop
                    sl_hit = True; sl_price = op
                elif lp <= current_sl:
                    sl_hit = True; sl_price = current_sl
            else:                                 # SHORT: stop if high > SL
                if op >= current_sl:
                    sl_hit = True; sl_price = op
                elif hp >= current_sl:
                    sl_hit = True; sl_price = current_sl

            if sl_hit:
                port_val = cash + position.unrealised_pnl(sl_price)
                trade, cash = filler.close_position(
                    position, sl_price, cash, ct, i, "Stop Loss", port_val
                )
                trade_log.append(trade)
                logger.info(
                    f"  SL hit  : {ct} | "
                    f"{'LONG' if position.direction == 1 else 'SHORT'} "
                    f"@ ₹{sl_price:,.2f} | P&L ₹{trade.net_pnl:+,.0f}"
                )
                position   = None
                current_sl = np.nan

        # ── C. Hard squareoff at 15:15 IST ────────────────────────────────
        if position is not None:
            try:
                bar_t = ct.time()
            except AttributeError:
                bar_t = None
            if bar_t and bar_t >= SQUAREOFF_TIME:
                port_val = cash + position.unrealised_pnl(op)
                trade, cash = filler.close_position(
                    position, op, cash, ct, i, "Squareoff 15:15", port_val
                )
                trade_log.append(trade)
                logger.info(
                    f"  SQROFF  : {ct} | "
                    f"{'LONG' if position.direction == 1 else 'SHORT'} "
                    f"@ ₹{op:,.2f} | P&L ₹{trade.net_pnl:+,.0f}"
                )
                position   = None
                current_sl = np.nan

        # ── D. Process signal from PREVIOUS bar (next-bar execution) ───────
        #    Signal on bar[i-1] → fill at bar[i] open.
        #    This is the core no-look-ahead rule.
        if i > 0 and sig_arr[i - 1] != 0 and position is None:
            direction   = int(sig_arr[i - 1])
            entry_price = op
            entry_sl    = float(sl_arr[i - 1])

            # Per-day direction guard (belt-and-suspenders; strategy enforces too)
            try:
                bar_date = str(ct.date())
            except AttributeError:
                bar_date = "unknown"

            used = day_dirs.setdefault(bar_date, set())
            if direction in used:
                pass   # already used this leg today
            elif entry_price > 0 and not np.isnan(entry_sl):
                # ── Quantity: max affordable with 100% of available cash ────
                # We bypass FillEngine's open_position() here because it
                # combines qty-sizing and cash-debit atomically using
                # capital_risk_pct. For 100%-capital utilisation, we:
                #   1. Compute commission on our qty directly.
                #   2. Compute the total cost (price × qty + commission).
                #   3. Debit cash ourselves.
                #   4. Build the Position dataclass directly.
                # FillEngine is still used for EXITS (close_position).
                order_side = "BUY" if direction == 1 else "SELL"

                # Start with maximum affordable qty, back off if commission
                # pushes total cost over available cash.
                qty = max(1, int(cash / entry_price))
                chg = cfg.commission_model.calculate(
                    cfg.segment, order_side, qty, entry_price
                )
                total_cost = (entry_price * qty + chg.total) if direction == 1 \
                             else chg.total

                # If total_cost > cash (commission pushed us over), reduce qty
                if total_cost > cash and qty > 1:
                    qty = max(1, int((cash - chg.total) / entry_price))
                    chg = cfg.commission_model.calculate(
                        cfg.segment, order_side, qty, entry_price
                    )
                    total_cost = (entry_price * qty + chg.total) if direction == 1 \
                                 else chg.total

                if total_cost > cash:
                    logger.warning(
                        f"  SKIP    : {ct} | insufficient cash "
                        f"(need ₹{total_cost:,.0f}, have ₹{cash:,.0f})"
                    )
                else:
                    # Build trailing stop level
                    trail_level = 0.0
                    if cfg.use_trailing_stop:
                        pct = cfg.trailing_stop_pct
                        amt = cfg.trailing_stop_amt
                        dist = entry_price * (pct / 100.0) if pct > 0 else amt
                        trail_level = (entry_price - dist) if direction == 1 \
                                      else (entry_price + dist)

                    position = Position(
                        symbol              = symbol,
                        entry_time          = ct,
                        entry_price         = entry_price,
                        quantity            = qty,
                        direction           = direction,
                        entry_signal        = "ORB_LONG" if direction == 1
                                              else "ORB_SHORT",
                        entry_charges       = chg.total,
                        entry_bar_idx       = i,
                        stop_price          = entry_sl,
                        trailing_stop_level = trail_level,
                        order_type          = OrderType.MARKET,
                    )
                    cash       -= total_cost
                    current_sl  = entry_sl
                    used.add(direction)
                    logger.info(
                        f"  ENTRY   : {ct} | "
                        f"{'LONG' if direction == 1 else 'SHORT'} "
                        f"qty={qty} @ ₹{entry_price:,.2f} | "
                        f"SL ₹{entry_sl:,.2f} | "
                        f"cash ₹{cash:,.0f}"
                    )

        # ── E. Advance trailing stop ───────────────────────────────────────
        if position is not None and i > 0:
            inc = float(trail_arr[i]) if not np.isnan(trail_arr[i]) else 0.0
            if inc > 0 and not np.isnan(current_sl):
                if position.direction == 1:
                    proposed = current_sl + inc
                    # Stop may advance only if it stays below current price
                    if proposed < cp:
                        current_sl = max(current_sl, proposed)
                else:
                    proposed = current_sl - inc
                    if proposed > cp:
                        current_sl = min(current_sl, proposed)

        # ── F. Mark-to-market equity and drawdown ─────────────────────────
        unrealised    = position.unrealised_pnl(cp) if position else 0.0
        equity        = cash + unrealised
        equity_arr[i] = equity

        peak_equity = max(peak_equity, equity)
        drawdown_arr[i] = (
            (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        )

        # ── G. Max drawdown guard ──────────────────────────────────────────
        if (peak_equity > 0
                and drawdown_arr[i] < -cfg.max_drawdown_pct
                and position is not None):
            logger.warning(
                f"[ORB] Max drawdown {cfg.max_drawdown_pct*100:.0f}% breached "
                f"at bar {i} ({ct}). Force-closing position."
            )
            port_val = equity
            trade, cash = filler.close_position(
                position, cp, cash, ct, i, "Max Drawdown Halt", port_val
            )
            trade_log.append(trade)
            position   = None
            current_sl = np.nan

    # ── Force-close any surviving position at end of data ────────────────────
    if position is not None:
        lp_  = closes_arr[-1]
        lt_  = times[-1]
        pv_  = cash + position.unrealised_pnl(lp_)
        trade, cash = filler.close_position(
            position, lp_, cash, lt_, n - 1, "End of Data", pv_
        )
        trade_log.append(trade)

    eq_series = pd.Series(equity_arr,   index=times, dtype=float)
    dd_series = pd.Series(drawdown_arr, index=times, dtype=float)

    final_eq = float(eq_series.dropna().iloc[-1]) if not eq_series.dropna().empty else cash
    logger.info(
        f"[ORB] Loop complete: {len(trade_log)} trades | "
        f"Final equity ₹{final_eq:,.2f} | "
        f"Return {((final_eq / cfg.initial_capital) - 1) * 100:.2f}%"
    )
    return trade_log, eq_series, dd_series


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: Save outputs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_all_outputs(result: BacktestResult, signals_df: pd.DataFrame) -> None:
    """
    Save all output files and print their locations to the terminal.

    Files saved:
      strategies/output/trade/{RUN_LABEL}_{SYMBOL_LABEL}_trade_log.csv
      strategies/output/raw_data/{RUN_LABEL}_{SYMBOL_LABEL}_signals.csv
      strategies/output/trade/{RUN_LABEL}_{SYMBOL_LABEL}_summary.json
      strategies/output/chart/{RUN_LABEL}_{SYMBOL_LABEL}_chart.png
    """
    for d in (OUT_TRADE, OUT_RAW, OUT_CHART):
        d.mkdir(parents=True, exist_ok=True)

    prefix = f"{RUN_LABEL}_{SYMBOL_LABEL}"

    # ── Trade log CSV ─────────────────────────────────────────────────────────
    if result.trade_log:
        p = OUT_TRADE / f"{prefix}_trade_log.csv"
        result.trade_df().to_csv(p, index=False)
        print(f"  [SAVED] Trade log     → {p}")
        logger.info(f"Trade log saved: {p}")
    else:
        print("  [INFO ] No trades generated — trade log not saved.")

    # ── Signal / raw data CSV ─────────────────────────────────────────────────
    p = OUT_RAW / f"{prefix}_signals.csv"
    signals_df.to_csv(p)
    print(f"  [SAVED] Signal data   → {p}")
    logger.info(f"Signal CSV saved: {p}")

    # ── Performance summary JSON ──────────────────────────────────────────────
    m    = result.metrics()
    safe = {k: v for k, v in m.items() if not isinstance(v, (dict, list))}
    p    = OUT_TRADE / f"{prefix}_summary.json"
    with open(p, "w") as fh:
        json.dump(safe, fh, indent=2, default=str)
    print(f"  [SAVED] Summary JSON  → {p}")
    logger.info(f"Summary JSON saved: {p}")

    # ── Chart PNG ─────────────────────────────────────────────────────────────
    try:
        from backtester.report import generate_report
        p = generate_report(
            result,
            symbol      = SYMBOL_LABEL,
            output_dir  = str(OUT_CHART),
            filename    = f"{prefix}_chart.png",
            max_candles = 2000,
        )
        print(f"  [SAVED] Chart PNG     → {p}")
        logger.info(f"Chart saved: {p}")
    except Exception as exc:
        logger.warning(f"Chart generation failed: {exc}")
        print(f"  [WARN ] Chart failed  : {exc}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: Terminal report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_report(result: BacktestResult) -> None:
    """Print a full performance report to the terminal."""
    sep  = "=" * 72
    thin = "─" * 72

    print(f"\n{sep}")
    print("  OPENING RANGE BREAKOUT — NIFTY 50 — BACKTEST REPORT")
    print(f"  Period  : {FROM_DATE}  →  {TO_DATE}  (5 years)")
    print(f"  Capital : ₹{INITIAL_CAPITAL:,.0f}  |  Segment: Equity Intraday")
    print(f"  OR Window: First {OR_WINDOW_MINUTES} minutes  "
          f"| VWAP Filter: {'ON' if VWAP_FILTER else 'OFF'}")
    print(f"  Trailing Stop: |prev_body| / {int(TRAILING_DIVISOR)}"
          f"  |  Squareoff: 15:15 IST")
    print(sep)

    # ── Core metrics ──────────────────────────────────────────────────────────
    print(result.summary())

    m = result.metrics()

    # ── Additional institution-grade metrics ──────────────────────────────────
    print(f"\n{thin}")
    print("  ADDITIONAL METRICS")
    print(thin)
    rows = [
        ("Omega Ratio",               f"{m.get('omega_ratio', 0):.4f}"),
        ("Kelly Fraction",            f"{m.get('kelly_fraction', 0):.4f}"),
        ("Avg Trade Duration (bars)", f"{m.get('avg_trade_duration_bars', 0):.1f}"),
        ("Max Consecutive Wins",      f"{m.get('max_consecutive_wins', 0)}"),
        ("Max Consecutive Losses",    f"{m.get('max_consecutive_losses', 0)}"),
        ("Avg MAE (₹)",               f"₹{m.get('avg_mae_inr', 0):,.2f}"),
        ("Avg MFE (₹)",               f"₹{m.get('avg_mfe_inr', 0):,.2f}"),
        ("Exposure %",                f"{m.get('exposure_pct', 0):.2f}%"),
        ("Total Commission Paid",     f"₹{m.get('total_commission_paid', 0):,.2f}"),
    ]
    for label, val in rows:
        print(f"  {label:<30}: {val}")

    # ── Monthly returns ───────────────────────────────────────────────────────
    monthly = m.get("monthly_returns", {})
    if monthly:
        print(f"\n{thin}")
        print("  MONTHLY RETURNS (last 60 months)")
        print(thin)
        keys = sorted(monthly.keys())[-60:]
        for k in keys:
            v    = monthly[k]
            sign = "+" if v >= 0 else ""
            bar  = ("▓" * min(int(abs(v)), 25))
            col  = "▲" if v >= 0 else "▼"
            print(f"  {k}  {col}  {sign}{v:>7.2f}%  {bar}")

    # ── Annual returns ────────────────────────────────────────────────────────
    annual = m.get("annual_returns", {})
    if annual:
        print(f"\n{thin}")
        print("  ANNUAL RETURNS")
        print(thin)
        for yr, ret in sorted(annual.items()):
            sign = "+" if ret >= 0 else ""
            bar  = "▓" * min(int(abs(ret) / 2), 30)
            col  = "▲" if ret >= 0 else "▼"
            print(f"  {yr}  {col}  {sign}{ret:>7.2f}%  {bar}")

    print(f"\n{sep}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    """
    End-to-end ORB NIFTY 50 backtest pipeline.

    ┌───┬─────────────────────────────────────────────────────┐
    │ 1 │ Load 1-min NIFTY 50 data (5 years)                 │
    │ 2 │ Generate ORB signals (OR levels, VWAP, trail)      │
    │ 3 │ Run custom event loop (per-trade SL + trail + 100%)│
    │ 4 │ Compute BacktestResult + 25+ metrics               │
    │ 5 │ Print terminal report                              │
    │ 6 │ Save trade log, signal CSV, summary JSON, chart PNG│
    └───┴─────────────────────────────────────────────────────┘
    """
    _banner()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    _step(1, "Downloading / loading NIFTY 50 1-minute data (5 years)...")
    df = load_nifty_data(FROM_DATE, TO_DATE)
    n_days = len(np.unique(df.index.date))
    print(f"      ✓ {len(df):,} bars loaded across {n_days} trading days")

    # ── 2. Strategy signals ───────────────────────────────────────────────────
    _step(2, "Running ORB strategy — computing OR levels, VWAP, signals...")
    strategy   = ORBNiftyStrategy(
        or_window_minutes = OR_WINDOW_MINUTES,
        vwap_filter       = VWAP_FILTER,
        trailing_divisor  = TRAILING_DIVISOR,
        min_body_pct      = MIN_BODY_PCT,
    )
    signals_df = strategy.generate_signals(df)
    n_long     = (signals_df["signal"] ==  1).sum()
    n_short    = (signals_df["signal"] == -1).sum()
    print(f"      ✓ Long signals: {n_long}  |  Short signals: {n_short}")
    print(f"      ✓ OR levels, VWAP, and trailing increments computed")

    # ── 3. Event loop ─────────────────────────────────────────────────────────
    _step(3, "Executing custom ORB event loop...")
    cfg = build_config()
    trade_log, equity_curve, drawdown = run_orb_event_loop(
        signals_df = signals_df,
        cfg        = cfg,
        symbol     = SYMBOL_LABEL,
    )
    print(f"      ✓ {len(trade_log)} completed trades")

    # ── 4. Result + metrics ───────────────────────────────────────────────────
    _step(4, "Computing performance metrics...")
    result = BacktestResult(
        config       = cfg,
        symbol       = SYMBOL_LABEL,
        trade_log    = trade_log,
        equity_curve = equity_curve,
        drawdown     = drawdown,
        signals_df   = signals_df,
    )
    m = result.metrics()
    print(
        f"      ✓ CAGR {m.get('cagr_pct', 0):.2f}%  |  "
        f"Sharpe {m.get('sharpe_ratio', 0):.3f}  |  "
        f"Win Rate {m.get('win_rate_pct', 0):.1f}%  |  "
        f"Max DD {m.get('max_drawdown_pct', 0):.2f}%"
    )

    # ── 5. Terminal report ────────────────────────────────────────────────────
    _step(5, "Printing terminal report...")
    print_report(result)

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    _step(6, "Saving output files...")
    save_all_outputs(result, signals_df)

    print("\n✓ Backtest complete.\n")


def _banner() -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  AlgoDesk — Opening Range Breakout (ORB) Backtest")
    print(f"  Instrument : {TRADING_SYMBOL} (NIFTY 50 Index)")
    print(f"  Period     : {FROM_DATE}  →  {TO_DATE}")
    print(f"  Capital    : ₹{INITIAL_CAPITAL:,.0f}")
    print(f"  OR Window  : 09:15 – 09:{14+OR_WINDOW_MINUTES:02d} IST ({OR_WINDOW_MINUTES} min)")
    print(f"  VWAP Filter: {'ON' if VWAP_FILTER else 'OFF'}")
    print(f"  Trail      : |prev_body| / {int(TRAILING_DIVISOR)}")
    print(f"  Squareoff  : 15:15 IST")
    print(f"  Qty Rule   : 100% capital per trade")
    print(f"{sep}\n")


def _step(n: int, msg: str) -> None:
    print(f"[{n}/6] {msg}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Backtest failed: {exc}")
        print(f"\n[ERROR] {exc}")
        sys.exit(1)