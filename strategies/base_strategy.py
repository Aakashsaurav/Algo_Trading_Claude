"""
strategies/base_strategy.py
-----------------------------
Abstract base class that every strategy must inherit from.

DESIGN PHILOSOPHY:
    The BaseStrategy defines the contract between your strategy logic and
    the backtester engine. You override only the methods you need.

    The engine calls:
        1. strategy.prepare(df)      → compute all indicators on full df
        2. For each bar i:
               signals = strategy.on_bar(i, row, portfolio)
        3. Engine processes signals, fills orders, updates portfolio
        4. strategy.on_fill(trade)   → called after each order fills

    WHY EVENT-DRIVEN (on_bar):
        Mirrors exactly how live trading works — you receive one bar at a
        time. This prevents look-ahead bias structurally (you can only see
        rows up to and including index i).

    WHY ALSO VECTORISED (generate_signals):
        For fast parameter sweeps and optimisation, computing signals on
        the full DataFrame at once is 10–100× faster than the bar loop.
        The backtester uses vectorised mode when you override generate_signals.

SIGNAL FORMAT:
    on_bar() returns a list of Signal objects (can be empty for no action).
    Signal = {action, quantity, order_type, stop_loss, take_profit, tag}

BUILT-IN RISK GUARDS (enforced by engine, not strategy):
    - Max portfolio drawdown: 20% (from config)
    - Max per-trade risk: 1–2% of capital (position sizing)
    - All MIS positions squared off by 15:20 IST
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class Action(str, Enum):
    """What the signal is instructing the engine to do."""
    BUY        = "BUY"         # Open long position (or add to existing if pyramiding)
    SELL       = "SELL"        # Close long / open short
    SHORT      = "SHORT"       # Open short position
    COVER      = "COVER"       # Close short position
    EXIT_ALL   = "EXIT_ALL"    # Close all open positions immediately


class OrderType(str, Enum):
    """How the order should be executed."""
    MARKET = "MARKET"   # Fill at next bar's open (realistic — eliminates look-ahead)
    LIMIT  = "LIMIT"    # Fill only if price reaches limit (not yet used in backtester v1)


@dataclass
class Signal:
    """
    A trading signal emitted by the strategy on a given bar.

    Attributes:
        action      : What to do (BUY, SELL, SHORT, COVER, EXIT_ALL).
        quantity    : Number of shares/units. If 0, engine uses position sizing.
        order_type  : MARKET (default) or LIMIT.
        limit_price : Limit price if order_type == LIMIT.
        stop_loss   : Stop-loss price. Engine exits if price crosses this.
        take_profit : Take-profit price. Engine exits if price reaches this.
        tag         : Label for this trade in the trade log (e.g. "EMA_CROSSOVER_ENTRY").
    """
    action:       Action
    quantity:     int            = 0       # 0 = auto-size based on risk rules
    order_type:   OrderType      = OrderType.MARKET
    limit_price:  Optional[float]= None
    stop_loss:    Optional[float]= None
    take_profit:  Optional[float]= None
    tag:          str            = ""


@dataclass
class PortfolioState:
    """
    Snapshot of portfolio state passed to on_bar() for strategy decisions.

    Attributes:
        cash              : Available cash in ₹.
        total_value       : Cash + market value of all open positions.
        open_positions    : Dict of {symbol: quantity}. Positive = long.
        open_position_pnl : Dict of {symbol: unrealised_pnl_in_inr}.
        peak_value        : Highest portfolio value ever (for drawdown calc).
        current_drawdown  : Current drawdown from peak (0.0 to 1.0).
    """
    cash:               float
    total_value:        float
    open_positions:     dict = field(default_factory=dict)
    open_position_pnl:  dict = field(default_factory=dict)
    peak_value:         float = 0.0
    current_drawdown:   float = 0.0

    def is_long(self, symbol: str) -> bool:
        """True if holding a long position in this symbol."""
        return self.open_positions.get(symbol, 0) > 0

    def is_short(self, symbol: str) -> bool:
        """True if holding a short position in this symbol."""
        return self.open_positions.get(symbol, 0) < 0

    def position_size(self, symbol: str) -> int:
        """Current position size (positive=long, negative=short, 0=flat)."""
        return self.open_positions.get(symbol, 0)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    To create a strategy:
        1. Inherit from BaseStrategy.
        2. Override prepare() to compute indicators.
        3. Override on_bar() to generate signals based on the current bar.
        4. Optionally override generate_signals() for vectorised mode.
        5. Optionally override on_fill() to react to order fills.

    Example minimal strategy:

        class MyStrategy(BaseStrategy):
            name = "My EMA Strategy"

            def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
                df["ema_fast"] = ema(df["close"], 10)
                df["ema_slow"] = ema(df["close"], 20)
                return df

            def on_bar(self, index, row, portfolio):
                if row["ema_fast"] > row["ema_slow"] and not portfolio.is_long("INFY"):
                    return [Signal(action=Action.BUY, tag="EMA_CROSS_UP")]
                if row["ema_fast"] < row["ema_slow"] and portfolio.is_long("INFY"):
                    return [Signal(action=Action.SELL, tag="EMA_CROSS_DOWN")]
                return []
    """

    # Override these class attributes in your strategy
    name:        str = "Unnamed Strategy"
    description: str = ""
    version:     str = "1.0"

    def __init__(self, params: Optional[dict] = None):
        """
        Args:
            params (dict): Strategy parameters e.g. {"fast_period": 10, "slow_period": 20}.
                           Access via self.params inside the strategy.
        """
        self.params = params or {}

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Called ONCE before the bar loop with the full historical DataFrame.
        Compute and attach all indicator columns here.

        IMPORTANT: You receive the FULL DataFrame here — but in on_bar()
        you will only see one row at a time. Do NOT make trading decisions
        in prepare(). Only compute indicators.

        Args:
            df (pd.DataFrame): Full OHLCV DataFrame with columns
                               [open, high, low, close, volume, oi].

        Returns:
            pd.DataFrame: Same DataFrame with indicator columns added.
        """
        ...

    @abstractmethod
    def on_bar(
        self,
        index:     int,
        row:       pd.Series,
        portfolio: PortfolioState,
    ) -> list[Signal]:
        """
        Called for EACH BAR in chronological order during the backtest.

        This is where your entry/exit logic lives.

        LOOK-AHEAD BIAS WARNING:
            You can only use data from row (the current bar) and earlier.
            Never reference df.iloc[index+1] or future data. The engine
            fills orders at the NEXT bar's open — you will never see the
            fill price inside this method.

        Args:
            index     (int):            Position of current bar in the DataFrame.
            row       (pd.Series):      Current bar (open, high, low, close,
                                        volume, oi, + your indicator columns).
            portfolio (PortfolioState): Current portfolio snapshot.

        Returns:
            list[Signal]: Zero or more signals for the engine to process.
                          Return [] for no action this bar.
        """
        ...

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OPTIONAL: Vectorised signal generation on the full DataFrame.

        Override this for faster backtesting or optimisation runs.
        If not overridden, the engine falls back to the on_bar() loop.

        When overridden, this method is called AFTER prepare(). The engine
        will use the 'signal' column to determine entries and exits.

        Expected output columns:
            signal : 1 = long entry, -1 = short entry, 0 = no signal
            signal_tag : string label for the trade log (optional)

        Args:
            df (pd.DataFrame): Full DataFrame with indicator columns from prepare().

        Returns:
            pd.DataFrame: Same df with 'signal' (and optionally 'signal_tag') column added.
        """
        # Default: no vectorised signals — engine uses on_bar() loop
        return df

    def on_fill(self, trade: dict) -> None:
        """
        OPTIONAL: Called by the engine after every order fill.

        Use this to update internal strategy state when an order executes,
        e.g. tracking which trades are open, logging fill prices.

        Args:
            trade (dict): Fill details with keys:
                action, symbol, quantity, fill_price, fill_time, tag,
                commission (dict), pnl (for closing trades).
        """
        pass

    def get_params(self) -> dict:
        """Return current strategy parameters (for display and logging)."""
        return self.params

    def set_params(self, **kwargs) -> None:
        """Update strategy parameters."""
        self.params.update(kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, params={self.params})"
