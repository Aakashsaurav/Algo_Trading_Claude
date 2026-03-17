"""
strategies/momentum/macd_crossover.py
---------------------------------------
MACD Signal Line Crossover Strategy.

CONCEPT:
    Buy when MACD line crosses above signal line (bullish momentum).
    Sell when MACD line crosses below signal line (bearish momentum).
    Optional filter: only trade when histogram confirms direction.

PARAMETERS:
    fast_period   (int): Fast EMA period. Default 12.
    slow_period   (int): Slow EMA period. Default 26.
    signal_period (int): Signal line EMA period. Default 9.
    atr_period    (int): ATR for stop-loss. Default 14.
    atr_multiplier (float): Stop = entry - mult × ATR. Default 2.0.
"""

import pandas as pd
from strategies.base_strategy_github import BaseStrategy, Signal, Action, PortfolioState
from indicators.oscillators import macd
from indicators.volatility import atr


class MACDCrossoverStrategy(BaseStrategy):
    name        = "MACD Crossover"
    description = "Buy on MACD/signal bullish cross. Sell on bearish cross."
    version     = "1.0"

    def __init__(self, params: dict = None):
        defaults = {
            "fast_period":    12,
            "slow_period":    26,
            "signal_period":  9,
            "atr_period":     14,
            "atr_multiplier": 2.0,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        macd_df = macd(df["close"], p["fast_period"], p["slow_period"], p["signal_period"])
        df["macd"]      = macd_df["macd"]
        df["macd_sig"]  = macd_df["signal"]
        df["macd_hist"] = macd_df["histogram"]
        df["atr"]       = atr(df["high"], df["low"], df["close"], p["atr_period"])

        df["macd_cross_up"]   = (df["macd"] > df["macd_sig"]) & \
                                (df["macd"].shift(1) <= df["macd_sig"].shift(1))
        df["macd_cross_down"] = (df["macd"] < df["macd_sig"]) & \
                                (df["macd"].shift(1) >= df["macd_sig"].shift(1))
        return df

    def on_bar(self, index, row, portfolio: PortfolioState) -> list[Signal]:
        if pd.isna(row.get("macd")) or pd.isna(row.get("macd_sig")):
            return []

        signals = []
        atr_val = row.get("atr", 0) or 0
        mult    = self.params["atr_multiplier"]

        if row.get("macd_cross_up") and not portfolio.is_long("__symbol__"):
            stop = row["close"] - mult * atr_val if atr_val else None
            signals.append(Signal(action=Action.BUY, stop_loss=stop, tag="MACD_CROSS_UP"))

        elif row.get("macd_cross_down") and portfolio.is_long("__symbol__"):
            signals.append(Signal(action=Action.SELL, tag="MACD_CROSS_DOWN"))

        return signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare(df.copy())  # compute indicators before signals
        df["signal"] = 0
        df.loc[df["macd_cross_up"],   "signal"] = 1
        df.loc[df["macd_cross_down"], "signal"] = -1
        df["signal_tag"] = ""
        df.loc[df["macd_cross_up"],   "signal_tag"] = "MACD_CROSS_UP"
        df.loc[df["macd_cross_down"], "signal_tag"] = "MACD_CROSS_DOWN"
        return df
