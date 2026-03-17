"""
strategies/momentum/ema_crossover.py
--------------------------------------
EMA Crossover Strategy.

CONCEPT:
    When a fast EMA crosses above a slow EMA, momentum is turning bullish.
    When it crosses below, momentum is turning bearish.

PARAMETERS:
    fast_period (int): Fast EMA period. Default 10.
    slow_period (int): Slow EMA period. Default 21.
    atr_period  (int): ATR period for stop-loss. Default 14.
    atr_multiplier (float): Stop = entry - atr_mult × ATR. Default 2.0.
    product_type (str): "MIS" for intraday, "NRML" for positional.
"""

import pandas as pd
from strategies.base_strategy_github import BaseStrategy, Signal, Action, PortfolioState
from indicators.moving_averages import ema
from indicators.volatility import atr


class EMACrossoverStrategy(BaseStrategy):
    name        = "EMA Crossover"
    description = "Buy when fast EMA crosses above slow EMA. Sell on reverse cross."
    version     = "1.0"

    def __init__(self, params: dict = None):
        defaults = {
            "fast_period":    10,
            "slow_period":    21,
            "atr_period":     14,
            "atr_multiplier": 2.0,
            "product_type":   "NRML",
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df["ema_fast"] = ema(df["close"], p["fast_period"])
        df["ema_slow"] = ema(df["close"], p["slow_period"])
        df["atr"]      = atr(df["high"], df["low"], df["close"], p["atr_period"])

        # Pre-compute crossover columns for vectorised mode
        df["cross_up"]   = (df["ema_fast"] > df["ema_slow"]) & \
                           (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
        df["cross_down"] = (df["ema_fast"] < df["ema_slow"]) & \
                           (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
        return df

    def on_bar(
        self,
        index:     int,
        row:       pd.Series,
        portfolio: PortfolioState,
    ) -> list[Signal]:
        # Skip bars before indicators are ready (NaN values)
        if pd.isna(row.get("ema_fast")) or pd.isna(row.get("ema_slow")):
            return []

        signals = []
        atr_val = row.get("atr", 0) or 0
        mult    = self.params["atr_multiplier"]

        # BUY signal: fast EMA crosses above slow EMA AND no long position
        if row.get("cross_up") and not portfolio.is_long("__symbol__"):
            stop = row["close"] - mult * atr_val if atr_val else None
            signals.append(Signal(
                action     = Action.BUY,
                stop_loss  = stop,
                tag        = f"EMA_CROSS_UP_{self.params['fast_period']}/{self.params['slow_period']}",
            ))

        # SELL signal: fast EMA crosses below slow EMA AND holding long
        elif row.get("cross_down") and portfolio.is_long("__symbol__"):
            signals.append(Signal(
                action = Action.SELL,
                tag    = f"EMA_CROSS_DOWN_{self.params['fast_period']}/{self.params['slow_period']}",
            ))

        return signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised version for fast backtesting."""
        df = self.prepare(df.copy())  # compute indicators before signals
        df["signal"] = 0
        df.loc[df["cross_up"],   "signal"] = 1
        df.loc[df["cross_down"], "signal"] = -1
        df["signal_tag"] = ""
        df.loc[df["cross_up"],   "signal_tag"] = "EMA_CROSS_UP"
        df.loc[df["cross_down"], "signal_tag"] = "EMA_CROSS_DOWN"
        return df
