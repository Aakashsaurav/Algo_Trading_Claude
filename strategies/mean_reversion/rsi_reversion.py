"""
strategies/mean_reversion/rsi_reversion.py
--------------------------------------------
RSI Mean Reversion Strategy.

CONCEPT:
    RSI < oversold_level → price likely to bounce up → BUY
    RSI > overbought_level → price likely to fall → SELL / EXIT

    Exit occurs when RSI crosses back to the neutral zone (midline).

PARAMETERS:
    rsi_period      (int): RSI period. Default 14.
    oversold        (int): RSI buy threshold. Default 30.
    overbought      (int): RSI sell/exit threshold. Default 70.
    exit_midline    (int): RSI level to exit the trade. Default 50.
    atr_period      (int): ATR period. Default 14.
    atr_multiplier (float): Stop = entry - mult × ATR. Default 1.5.
"""

import pandas as pd
from strategies.base_strategy import BaseStrategy, Signal, Action, PortfolioState
from indicators.oscillators import rsi
from indicators.volatility import atr


class RSIReversionStrategy(BaseStrategy):
    name        = "RSI Mean Reversion"
    description = "Buy on RSI oversold, sell on RSI overbought or midline cross."
    version     = "1.0"

    def __init__(self, params: dict = None):
        defaults = {
            "rsi_period":     14,
            "oversold":       30,
            "overbought":     70,
            "exit_midline":   50,
            "atr_period":     14,
            "atr_multiplier": 1.5,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df["rsi"] = rsi(df["close"], p["rsi_period"])
        df["atr"] = atr(df["high"], df["low"], df["close"], p["atr_period"])

        df["rsi_oversold"]   = df["rsi"] < p["oversold"]
        df["rsi_overbought"] = df["rsi"] > p["overbought"]
        # RSI crosses back above midline (exit long)
        df["rsi_mid_cross_up"]   = (df["rsi"] > p["exit_midline"]) & \
                                   (df["rsi"].shift(1) <= p["exit_midline"])
        return df

    def on_bar(self, index, row, portfolio: PortfolioState) -> list[Signal]:
        if pd.isna(row.get("rsi")):
            return []

        signals = []
        atr_val = row.get("atr", 0) or 0
        mult    = self.params["atr_multiplier"]

        # Entry: RSI oversold and not already long
        if row.get("rsi_oversold") and not portfolio.is_long("__symbol__"):
            stop = row["close"] - mult * atr_val if atr_val else None
            signals.append(Signal(
                action    = Action.BUY,
                stop_loss = stop,
                tag       = f"RSI_OVERSOLD_{row['rsi']:.1f}",
            ))

        # Exit: RSI crosses back above midline or hits overbought
        elif portfolio.is_long("__symbol__"):
            if row.get("rsi_overbought") or row.get("rsi_mid_cross_up"):
                signals.append(Signal(
                    action = Action.SELL,
                    tag    = f"RSI_EXIT_{row['rsi']:.1f}",
                ))

        return signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare(df.copy())  # compute indicators before signals
        df["signal"] = 0
        df.loc[df["rsi_oversold"],  "signal"] = 1
        df.loc[df["rsi_overbought"],"signal"] = -1
        df["signal_tag"] = ""
        df.loc[df["rsi_oversold"],   "signal_tag"] = "RSI_OVERSOLD"
        df.loc[df["rsi_overbought"], "signal_tag"] = "RSI_OVERBOUGHT"
        return df
