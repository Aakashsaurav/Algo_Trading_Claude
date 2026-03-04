"""
strategies/mean_reversion/bollinger_squeeze.py
------------------------------------------------
Bollinger Band Squeeze + Breakout Strategy.

CONCEPT:
    When Bollinger Bands are inside Keltner Channels (squeeze), volatility
    is compressed. When the squeeze releases AND price breaks out, we enter
    in the direction of the breakout.

    Entry: Squeeze was active last bar, now released.
           Price > BB upper → BUY (upside breakout).
           Price < BB lower → SHORT (downside breakout).
    Exit: Price crosses back to the middle BB band.

PARAMETERS:
    bb_period  (int)  : Bollinger period. Default 20.
    bb_std     (float): BB std dev multiplier. Default 2.0.
    kc_ema     (int)  : Keltner EMA period. Default 20.
    kc_atr     (int)  : Keltner ATR period. Default 10.
    kc_mult    (float): Keltner multiplier. Default 1.5.
    atr_period (int)  : ATR for stop. Default 14.
    atr_mult   (float): Stop multiplier. Default 2.0.
"""

import pandas as pd
from strategies.base_strategy import BaseStrategy, Signal, Action, PortfolioState
from indicators.volatility import bollinger_bands, keltner_channels, bb_squeeze, atr


class BollingerSqueezeStrategy(BaseStrategy):
    name        = "Bollinger Band Squeeze"
    description = "Enter on squeeze release in breakout direction."
    version     = "1.0"

    def __init__(self, params: dict = None):
        defaults = {
            "bb_period":  20,
            "bb_std":     2.0,
            "kc_ema":     20,
            "kc_atr":     10,
            "kc_mult":    1.5,
            "atr_period": 14,
            "atr_mult":   2.0,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        bb = bollinger_bands(df["close"], p["bb_period"], p["bb_std"])
        df["bb_upper"]  = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"]  = bb["bb_lower"]

        df["squeeze"] = bb_squeeze(
            df["high"], df["low"], df["close"],
            p["bb_period"], p["bb_std"],
            p["kc_ema"], p["kc_atr"], p["kc_mult"]
        )
        df["atr"] = atr(df["high"], df["low"], df["close"], p["atr_period"])

        # Squeeze just released: was True last bar, False this bar
        df["squeeze_release"] = df["squeeze"].shift(1) & ~df["squeeze"]
        return df

    def on_bar(self, index, row, portfolio: PortfolioState) -> list[Signal]:
        if pd.isna(row.get("bb_upper")) or pd.isna(row.get("squeeze")):
            return []

        signals = []
        atr_val = row.get("atr", 0) or 0
        mult    = self.params["atr_mult"]

        # Entry on squeeze release
        if row.get("squeeze_release"):
            if row["close"] > row["bb_upper"] and not portfolio.is_long("__symbol__"):
                stop = row["close"] - mult * atr_val if atr_val else None
                signals.append(Signal(action=Action.BUY, stop_loss=stop, tag="SQUEEZE_BREAKOUT_UP"))
            elif row["close"] < row["bb_lower"] and not portfolio.is_short("__symbol__"):
                stop = row["close"] + mult * atr_val if atr_val else None
                signals.append(Signal(action=Action.SHORT, stop_loss=stop, tag="SQUEEZE_BREAKOUT_DOWN"))

        # Exit: price returns to middle band
        if portfolio.is_long("__symbol__") and row["close"] < row["bb_middle"]:
            signals.append(Signal(action=Action.SELL, tag="BB_MID_EXIT"))
        elif portfolio.is_short("__symbol__") and row["close"] > row["bb_middle"]:
            signals.append(Signal(action=Action.COVER, tag="BB_MID_EXIT"))

        return signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare(df.copy())  # compute indicators before signals
        df["signal"] = 0
        buy  = df["squeeze_release"] & (df["close"] > df["bb_upper"])
        sell = df["squeeze_release"] & (df["close"] < df["bb_lower"])
        df.loc[buy,  "signal"] = 1
        df.loc[sell, "signal"] = -1
        df["signal_tag"] = ""
        df.loc[buy,  "signal_tag"] = "SQUEEZE_UP"
        df.loc[sell, "signal_tag"] = "SQUEEZE_DOWN"
        return df
