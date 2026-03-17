"""
backtester/report.py  — Streak-style visual backtest report generator.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BG_COLOUR      = "#0d1117"
PANEL_BG       = "#161b22"
GRID_COLOUR    = "#21262d"
TEXT_COLOUR    = "#c9d1d9"
BULL_COLOUR    = "#26a641"
BEAR_COLOUR    = "#e3342f"
EQ_COLOUR      = "#58a6ff"
DD_COLOUR      = "#f97583"
VOLUME_BULL    = "#1f5c32"
VOLUME_BEAR    = "#5c1f1f"
INDICATOR_COLS = ["#79c0ff", "#d2a8ff", "#ffb17a", "#56d364", "#e3b341"]


def generate_report(result, symbol="SYMBOL", output_dir="reports",
                    filename=None, show=False, max_candles=2000):
    """
    Generate a Streak-style multi-panel backtest report (PNG).

    Panels (top to bottom):
      1. Candlestick chart + indicator overlays + buy/sell markers
      2. Volume bars (green/red)
      3. Oscillators: RSI / MACD / Stochastic (only if present in signals_df)
      4. Equity curve with profit/loss colouring vs initial capital
      5. Drawdown % from peak (red filled area)
      6. Performance metrics + trade log table

    Args:
        result:      BacktestResult from BacktestEngine.run()
        symbol:      Symbol name (used in title and filename)
        output_dir:  Folder to save PNG (created if missing)
        filename:    Override default filename
        show:        Display interactively if True
        max_candles: Limit candles plotted for readability (default 2000)

    Returns:
        str: Absolute path of saved PNG file
    """
    df        = result.signals_df.copy()
    trade_log = result.trade_log
    equity    = result.equity_curve.copy()
    drawdown  = result.drawdown.copy()
    config    = result.config

    if len(df) > max_candles:
        df       = df.iloc[-max_candles:]
        equity   = equity.iloc[-max_candles:]
        drawdown = drawdown.iloc[-max_candles:]

    osc_cols     = _detect_oscillators(df)
    overlay_cols = _detect_overlays(df)
    has_osc      = len(osc_cols) > 0

    n_panels = 5 + (1 if has_osc else 0)
    ratios   = [5, 1.5] + ([1.5] if has_osc else []) + [2, 1.5, 3]

    fig = plt.figure(figsize=(18, n_panels * 3.5), facecolor=BG_COLOUR)
    gs  = gridspec.GridSpec(n_panels, 1, figure=fig,
                             height_ratios=ratios, hspace=0.04)

    ax_price = fig.add_subplot(gs[0])
    ax_vol   = fig.add_subplot(gs[1], sharex=ax_price)
    pidx     = 2
    ax_osc   = None
    if has_osc:
        ax_osc = fig.add_subplot(gs[pidx], sharex=ax_price)
        pidx  += 1
    ax_eq    = fig.add_subplot(gs[pidx]);     pidx += 1
    ax_dd    = fig.add_subplot(gs[pidx], sharex=ax_eq); pidx += 1
    ax_tbl   = fig.add_subplot(gs[pidx])

    for ax in fig.get_axes():
        _style_axis(ax)

    # --- Panel 1: Candlestick
    _plot_candlestick(ax_price, df)
    _plot_overlays(ax_price, df, overlay_cols)
    _plot_trade_markers(ax_price, df, trade_log)
    ax_price.set_ylabel("Price (Rs)", color=TEXT_COLOUR, fontsize=9)
    ax_price.set_title(
        f"{symbol}  |  {config.segment.value.upper()}  |  Backtester",
        color=TEXT_COLOUR, fontsize=11, fontweight="bold", pad=10)
    _add_legend(ax_price)

    # --- Panel 2: Volume
    _plot_volume(ax_vol, df)
    ax_vol.set_ylabel("Volume", color=TEXT_COLOUR, fontsize=8)

    # --- Panel 3: Oscillators
    if ax_osc is not None:
        _plot_oscillators(ax_osc, df, osc_cols)
        ax_osc.set_ylabel("Indicators", color=TEXT_COLOUR, fontsize=8)

    # --- Panel 4: Equity
    _plot_equity(ax_eq, equity, config.initial_capital)
    ax_eq.set_ylabel("Portfolio (Rs)", color=TEXT_COLOUR, fontsize=8)
    ax_eq.set_title("Equity Curve", color=TEXT_COLOUR, fontsize=8, pad=4)

    # --- Panel 5: Drawdown
    _plot_drawdown(ax_dd, drawdown, config.max_drawdown_pct)
    ax_dd.set_ylabel("Drawdown %", color=TEXT_COLOUR, fontsize=8)
    ax_dd.set_title("Drawdown from Peak", color=TEXT_COLOUR, fontsize=8, pad=4)

    # --- Panel 6: Table
    _plot_summary_table(ax_tbl, result)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fname = filename or f"{symbol}_backtest.png"
    fpath = output_path / fname
    fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor=BG_COLOUR)
    logger.info(f"Report saved: {fpath}")
    if show:
        plt.show()
    plt.close(fig)
    return str(fpath)


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _style_axis(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOUR, labelsize=7)
    ax.yaxis.label.set_color(TEXT_COLOUR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOUR)
    ax.grid(True, color=GRID_COLOUR, linewidth=0.5, alpha=0.7)


def _plot_candlestick(ax, df):
    for i, (ts, row) in enumerate(df.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        colour  = BULL_COLOUR if c >= o else BEAR_COLOUR
        body_lo = min(o, c)
        body_h  = max(abs(c - o), 0.01)
        ax.bar(i, body_h, bottom=body_lo, color=colour,
               width=0.6, linewidth=0, zorder=2)
        ax.plot([i, i], [l, h], color=colour, linewidth=0.7, zorder=1)

    tick_freq = max(1, len(df) // 12)
    tick_pos  = list(range(0, len(df), tick_freq))
    tick_lbl  = [str(df.index[p])[:10] for p in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=30, ha="right",
                       fontsize=6, color=TEXT_COLOUR)
    ax.set_xlim(-1, len(df))


def _plot_overlays(ax, df, cols):
    for idx, col in enumerate(cols):
        colour = INDICATOR_COLS[idx % len(INDICATOR_COLS)]
        if col == "bb_upper" and "bb_lower" in df.columns:
            x = np.arange(len(df))
            ax.plot(x, df["bb_upper"],  color=colour, lw=0.8, ls="--", label="BB Upper")
            ax.plot(x, df["bb_middle"], color=colour, lw=1.0, label="BB Mid")
            ax.plot(x, df["bb_lower"],  color=colour, lw=0.8, ls="--", label="BB Lower")
            ax.fill_between(x, df["bb_lower"], df["bb_upper"], alpha=0.06, color=colour)
        elif col in ("bb_middle", "bb_lower"):
            continue
        elif "supertrend" in col.lower():
            dir_col = "st_direction"
            if dir_col in df.columns:
                x = np.arange(len(df))
                bull = (df[dir_col] == 1).values
                bear = (df[dir_col] == -1).values
                yv   = df[col].values
                ax.scatter(x[bull], yv[bull], color=BULL_COLOUR,
                           s=4, marker="_", label="ST Bull", zorder=3)
                ax.scatter(x[bear], yv[bear], color=BEAR_COLOUR,
                           s=4, marker="_", label="ST Bear", zorder=3)
        else:
            ax.plot(np.arange(len(df)), df[col],
                    color=colour, lw=1.2, label=col, zorder=4)


def _plot_trade_markers(ax, df, trade_log):
    if not trade_log:
        return
    idx_map = {ts: i for i, ts in enumerate(df.index)}
    for trade in trade_log:
        ei = idx_map.get(trade.entry_time)
        xi = idx_map.get(trade.exit_time)
        if ei is not None:
            y = df["low"].iloc[ei] * 0.999
            ax.scatter(ei, y, marker="^", color=BULL_COLOUR, s=90, zorder=5)
            ax.annotate(f"B {trade.entry_price:.1f}", (ei, y),
                        xytext=(3, -11), textcoords="offset points",
                        fontsize=5.5, color=BULL_COLOUR, zorder=6)
        if xi is not None:
            y = df["high"].iloc[xi] * 1.001
            ax.scatter(xi, y, marker="v", color=BEAR_COLOUR, s=90, zorder=5)
            ax.annotate(f"S {trade.exit_price:.1f}", (xi, y),
                        xytext=(3, 6), textcoords="offset points",
                        fontsize=5.5, color=BEAR_COLOUR, zorder=6)


def _add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=7,
                  facecolor=PANEL_BG, edgecolor=GRID_COLOUR,
                  labelcolor=TEXT_COLOUR)


def _plot_volume(ax, df):
    colours = [VOLUME_BULL if df["close"].iloc[i] >= df["open"].iloc[i]
               else VOLUME_BEAR for i in range(len(df))]
    ax.bar(np.arange(len(df)), df["volume"], color=colours,
           width=0.8, linewidth=0, zorder=2)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6
                          else f"{x/1e3:.0f}K" if x >= 1e3 else str(int(x))))


def _detect_overlays(df):
    prefixes = ("ema_", "sma_", "dema_", "EMA_", "SMA_",
                "bb_upper", "bb_middle", "bb_lower",
                "kc_upper", "kc_middle", "kc_lower", "supertrend")
    skip = {"bb_pct_b", "bb_bandwidth"}
    return [c for c in df.columns
            if any(c.startswith(p) or c == p for p in prefixes)
            and c not in skip]


def _detect_oscillators(df):
    prefixes = ("rsi", "RSI_", "stoch_k", "stoch_d",
                "macd", "macd_signal", "macd_histogram")
    return [c for c in df.columns
            if any(c.lower().startswith(p.lower()) for p in prefixes)]


def _plot_oscillators(ax, df, cols):
    x = np.arange(len(df))
    for idx, col in enumerate(cols):
        colour = INDICATOR_COLS[idx % len(INDICATOR_COLS)]
        data   = df[col].values
        if col.lower() == "macd_histogram":
            colours = [BULL_COLOUR if v >= 0 else BEAR_COLOUR for v in data]
            ax.bar(x, data, color=colours, width=0.6, alpha=0.8,
                   linewidth=0, label="MACD Hist", zorder=2)
        else:
            ax.plot(x, data, color=colour, lw=1.0, label=col, zorder=3)

    rsi_cols = [c for c in cols if c.lower().startswith("rsi")]
    if rsi_cols:
        ax.axhline(70, color=BEAR_COLOUR, lw=0.8, ls="--", alpha=0.7)
        ax.axhline(30, color=BULL_COLOUR, lw=0.8, ls="--", alpha=0.7)
        ax.set_ylim(0, 100)

    ax.axhline(0, color=GRID_COLOUR, lw=0.8, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=6,
                  facecolor=PANEL_BG, edgecolor=GRID_COLOUR,
                  labelcolor=TEXT_COLOUR)


def _plot_equity(ax, equity, initial_capital):
    eq = equity.dropna()
    if eq.empty:
        return
    x = np.arange(len(eq))
    ax.plot(x, eq.values, color=EQ_COLOUR, lw=1.5, zorder=3)
    ax.fill_between(x, initial_capital, eq.values,
                    where=(eq.values >= initial_capital),
                    alpha=0.15, color=BULL_COLOUR)
    ax.fill_between(x, initial_capital, eq.values,
                    where=(eq.values < initial_capital),
                    alpha=0.15, color=BEAR_COLOUR)
    ax.axhline(initial_capital, color=TEXT_COLOUR, lw=0.8, ls="--", alpha=0.6)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"Rs {v/1e5:.1f}L"))
    tick_step = max(1, len(eq) // 8)
    tick_pos  = list(range(0, len(eq), tick_step))
    tick_lbl  = [str(eq.index[p])[:10] for p in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=30, ha="right",
                       fontsize=6, color=TEXT_COLOUR)


def _plot_drawdown(ax, drawdown, threshold):
    dd = drawdown.dropna()
    if dd.empty:
        return
    x = np.arange(len(dd))
    ax.fill_between(x, 0, dd.values, color=DD_COLOUR, alpha=0.6, zorder=2)
    ax.plot(x, dd.values, color=DD_COLOUR, lw=0.8, zorder=3)
    ax.axhline(0, color=GRID_COLOUR, lw=0.8)
    ax.axhline(-(threshold * 100), color=BEAR_COLOUR, lw=1.0, ls="--",
               label=f"Alert {threshold*100:.0f}%")
    ax.set_ylim(min(dd.min() * 1.1, -1), 1)
    tick_step = max(1, len(dd) // 8)
    tick_pos  = list(range(0, len(dd), tick_step))
    tick_lbl  = [str(dd.index[p])[:10] for p in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=30, ha="right",
                       fontsize=6, color=TEXT_COLOUR)
    ax.legend(fontsize=6, facecolor=PANEL_BG,
              edgecolor=GRID_COLOUR, labelcolor=TEXT_COLOUR)


def _plot_summary_table(ax, result):
    ax.axis("off")
    ax.set_facecolor(PANEL_BG)
    m = result._compute_metrics()
    if "error" in m:
        ax.text(0.5, 0.5, "No trades generated.",
                color=TEXT_COLOUR, ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        return

    keys = [k for k in m.keys() if k != "Symbol"]
    vals = [str(m[k]) for k in keys]
    half = (len(keys) + 1) // 2
    row_h, sy = 0.05, 0.97

    def vc(v):
        return BEAR_COLOUR if (v.startswith("-") or "Rs -" in v) else TEXT_COLOUR

    ax.text(0.01, sy + 0.02, "PERFORMANCE METRICS",
            color=EQ_COLOUR, fontsize=8, fontweight="bold",
            transform=ax.transAxes)

    for r, (k, v) in enumerate(zip(keys[:half], vals[:half])):
        y = sy - r * row_h
        ax.text(0.01,  y, k + ":", color=TEXT_COLOUR, fontsize=6.5, transform=ax.transAxes)
        ax.text(0.21,  y, v, color=vc(v), fontsize=6.5, fontweight="bold", transform=ax.transAxes)

    for r, (k, v) in enumerate(zip(keys[half:], vals[half:])):
        y = sy - r * row_h
        ax.text(0.35,  y, k + ":", color=TEXT_COLOUR, fontsize=6.5, transform=ax.transAxes)
        ax.text(0.55,  y, v, color=vc(v), fontsize=6.5, fontweight="bold", transform=ax.transAxes)

    trades = result.trade_log
    if not trades:
        return

    ax.text(0.70, sy + 0.02, "TRADE LOG",
            color=EQ_COLOUR, fontsize=8, fontweight="bold",
            transform=ax.transAxes)

    headers = ["#", "Entry", "Exit", "Dir", "Qty",
               "Entry Rs", "Exit Rs", "Net P&L", "Charges", "Duration"]
    col_x   = [0.70, 0.73, 0.83, 0.92, 0.95, 0.99, 1.06, 1.13, 1.20, 1.27]

    for h, cx in zip(headers, col_x):
        ax.text(cx, sy, h, color=TEXT_COLOUR, fontsize=5.5,
                fontweight="bold", transform=ax.transAxes, clip_on=True)

    for r, t in enumerate(trades[:14]):
        y   = sy - (r + 1) * row_h
        win = t.net_pnl >= 0
        pnl_c = BULL_COLOUR if win else BEAR_COLOUR
        cells = [
            str(r+1),
            str(t.entry_time)[:16],
            str(t.exit_time)[:16],
            t.direction_label[:1],
            str(t.quantity),
            f"{t.entry_price:.1f}",
            f"{t.exit_price:.1f}",
            f"{'+' if win else ''}{t.net_pnl:.0f}",
            f"{t.total_charges:.0f}",
            t.duration,
        ]
        for ci, (cell, cx) in enumerate(zip(cells, col_x)):
            c = pnl_c if ci == 7 else TEXT_COLOUR
            ax.text(cx, y, cell, color=c, fontsize=5.5,
                    transform=ax.transAxes, clip_on=True)

    if len(trades) > 14:
        ax.text(0.70, sy - 15 * row_h,
                f"... and {len(trades)-14} more trades",
                color=TEXT_COLOUR, fontsize=5.5, transform=ax.transAxes)
