"""
strategies/trend/supertrend_strategy.py
-----------------------------------------
Supertrend Trend-Following Strategy.

CONCEPT:
    Supertrend acts as a trailing stop-and-reverse indicator.
    When price is above the Supertrend line → uptrend → hold long.
    When price crosses below → downtrend → exit long / enter short.

PARAMETERS:
    st_period     (int)  : ATR period for Supertrend. Default 10.
    st_multiplier (float): ATR multiplier. Default 3.0.
    allow_short   (bool) : Trade both sides. Default False (long only).
"""

import pandas as pd
from strategies.base_strategy import BaseStrategy, Signal, Action, PortfolioState
from indicators.trend import supertrend


class SupertrendStrategy(BaseStrategy):
    name        = "Supertrend"
    description = "Trend-following using Supertrend as entry and trailing stop."
    version     = "1.0"

    def __init__(self, params: dict = None):
        defaults = {
            "st_period":     10,
            "st_multiplier": 3.0,
            "allow_short":   False,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        p  = self.params
        st = supertrend(df["high"], df["low"], df["close"], p["st_period"], p["st_multiplier"])
        df["supertrend"]  = st["supertrend"]
        df["st_direction"]= st["direction"]
        df["st_buy"]      = st["buy_signal"]
        df["st_sell"]     = st["sell_signal"]
        return df

    def on_bar(self, index, row, portfolio: PortfolioState) -> list[Signal]:
        if pd.isna(row.get("supertrend")):
            return []

        signals = []
        allow_short = self.params["allow_short"]

        if row.get("st_buy"):
            if allow_short and portfolio.is_short("__symbol__"):
                signals.append(Signal(action=Action.COVER, tag="ST_COVER"))
            if not portfolio.is_long("__symbol__"):
                signals.append(Signal(
                    action    = Action.BUY,
                    stop_loss = float(row["supertrend"]),
                    tag       = "ST_BUY",
                ))

        elif row.get("st_sell"):
            if portfolio.is_long("__symbol__"):
                signals.append(Signal(action=Action.SELL, tag="ST_SELL"))
            if allow_short and not portfolio.is_short("__symbol__"):
                signals.append(Signal(
                    action    = Action.SHORT,
                    stop_loss = float(row["supertrend"]),
                    tag       = "ST_SHORT",
                ))

        return signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare(df.copy())  # compute indicators before signals
        df["signal"] = 0
        df.loc[df["st_buy"],  "signal"] = 1
        df.loc[df["st_sell"], "signal"] = -1
        df["signal_tag"] = ""
        df.loc[df["st_buy"],  "signal_tag"] = "ST_BUY"
        df.loc[df["st_sell"], "signal_tag"] = "ST_SELL"
        return df
