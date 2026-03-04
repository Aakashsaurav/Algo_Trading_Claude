"""
backtester/trade_log.py
------------------------
Records every completed trade with full metadata.

A "trade" in our system = one complete round-trip:
    Entry fill → Exit fill (however many bars later)

For pyramiding (multiple entries into the same symbol), each
entry-exit pair is recorded as a separate trade row.

COLUMNS RECORDED:
    trade_id          : Sequential integer ID
    symbol            : Trading symbol (e.g. "INFY")
    entry_time        : Timestamp of entry bar signal (fill at next open)
    exit_time         : Timestamp of exit fill
    entry_price       : Actual fill price on entry (next bar's open)
    exit_price        : Actual fill price on exit  (next bar's open)
    quantity          : Shares / units traded
    side              : "LONG" or "SHORT"
    entry_tag         : Signal label that triggered entry
    exit_tag          : Signal label that triggered exit
    gross_pnl         : (exit_price - entry_price) × qty × direction (no costs)
    commission_entry  : Total charges on entry fill
    commission_exit   : Total charges on exit fill
    net_pnl           : gross_pnl - commission_entry - commission_exit
    pnl_pct           : net_pnl / (entry_price × qty) × 100
    duration_bars     : Number of bars held
    duration_str      : Human-readable duration (e.g. "3d 2h")
    mae               : Maximum Adverse Excursion (worst unrealised loss during trade)
    mfe               : Maximum Favourable Excursion (best unrealised profit during trade)
    portfolio_value   : Total portfolio value AFTER this trade closes
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """One completed round-trip trade."""

    trade_id:         int
    symbol:           str
    entry_time:       pd.Timestamp
    exit_time:        pd.Timestamp
    entry_price:      float
    exit_price:       float
    quantity:         int
    side:             str              # "LONG" or "SHORT"
    entry_tag:        str = ""
    exit_tag:         str = ""

    # P&L
    gross_pnl:        float = 0.0
    commission_entry: float = 0.0
    commission_exit:  float = 0.0
    net_pnl:          float = 0.0
    pnl_pct:          float = 0.0

    # Duration
    duration_bars:    int   = 0
    duration_str:     str   = ""

    # Excursions
    mae:              float = 0.0      # Maximum Adverse Excursion (negative = loss)
    mfe:              float = 0.0      # Maximum Favourable Excursion (positive = gain)

    # Portfolio state after this trade
    portfolio_value:  float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert timestamps to strings for serialisation
        d["entry_time"] = str(self.entry_time)
        d["exit_time"]  = str(self.exit_time)
        return d


@dataclass
class OpenPosition:
    """
    Tracks a single open position (one entry that hasn't exited yet).
    Used internally by the engine to track MAE/MFE and compute PnL.
    """
    trade_id:      int
    symbol:        str
    entry_time:    pd.Timestamp
    entry_price:   float
    quantity:      int
    side:          str              # "LONG" or "SHORT"
    entry_tag:     str = ""
    entry_bar_idx: int = 0
    commission_entry: float = 0.0

    # Running MAE/MFE trackers (updated every bar)
    worst_price:   float = 0.0      # lowest close seen (long) / highest (short)
    best_price:    float = 0.0      # highest close seen (long) / lowest (short)

    def update_excursions(self, current_price: float) -> None:
        """Update worst/best price seen since entry."""
        if self.side == "LONG":
            self.worst_price = min(self.worst_price or current_price, current_price)
            self.best_price  = max(self.best_price  or current_price, current_price)
        else:  # SHORT
            self.worst_price = max(self.worst_price or current_price, current_price)
            self.best_price  = min(self.best_price  or current_price, current_price)

    @property
    def direction(self) -> int:
        return 1 if self.side == "LONG" else -1

    def unrealised_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.quantity * self.direction

    def mae(self) -> float:
        """MAE: worst unrealised loss. Negative value = loss."""
        return (self.worst_price - self.entry_price) * self.quantity * self.direction

    def mfe(self) -> float:
        """MFE: best unrealised gain. Positive value = gain."""
        return (self.best_price - self.entry_price) * self.quantity * self.direction


class TradeLog:
    """
    Container for all completed trades.
    Provides methods to query, analyse, and export the trade log.
    """

    def __init__(self):
        self._trades: list[TradeRecord] = []
        self._trade_counter: int = 0

    def next_trade_id(self) -> int:
        self._trade_counter += 1
        return self._trade_counter

    def add(self, trade: TradeRecord) -> None:
        """Append a completed trade to the log."""
        self._trades.append(trade)
        logger.debug(
            f"Trade #{trade.trade_id}: {trade.side} {trade.symbol} "
            f"qty={trade.quantity} entry={trade.entry_price:.2f} "
            f"exit={trade.exit_price:.2f} net_pnl={trade.net_pnl:.2f}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export all trades as a pandas DataFrame."""
        if not self._trades:
            return pd.DataFrame()
        rows = [t.to_dict() for t in self._trades]
        df = pd.DataFrame(rows)
        # Sort chronologically by exit time
        if "exit_time" in df.columns:
            df = df.sort_values("exit_time").reset_index(drop=True)
        return df


    def to_csv(self, filepath: str) -> str:
        """
        Export all trades to a CSV file.
        
        Args:
            filepath: Full path including filename, e.g. "output/trades.csv"
        
        Returns:
            filepath — the path written to
        """
        import os
        df = self.to_dataframe()
        if df.empty:
            logger.warning("TradeLog.to_csv: no trades to export")
            return filepath
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"TradeLog: {len(df)} trades exported to {filepath}")
        return filepath

    def summary(self) -> dict:
        """Return a quick summary dict of trade statistics."""
        if not self._trades:
            return {'total_trades': 0, 'winners': 0, 'losers': 0, 'net_pnl': 0.0}
        pnls = [t.net_pnl for t in self._trades]
        return {
            'total_trades': len(pnls),
            'winners':      sum(1 for p in pnls if p > 0),
            'losers':       sum(1 for p in pnls if p <= 0),
            'net_pnl':      round(sum(pnls), 2),
            'avg_pnl':      round(sum(pnls)/len(pnls), 2),
        }

    def __len__(self) -> int:
        return len(self._trades)

    def __iter__(self):
        return iter(self._trades)
