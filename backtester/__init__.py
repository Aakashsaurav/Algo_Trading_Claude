"""
backtester — AlgoDesk Backtesting Engine
=========================================

Public API::

    from backtester import BacktestEngine, BacktestConfig, OrderType
    from backtester import Optimizer, SearchMethod, BacktestResult

All implementation lives in sub-modules.  Only these names are considered
part of the stable public API.  Everything else is subject to change.
"""

from backtester.models import (
    BacktestConfig,
    BacktestResult,
    Position,
    Trade,
    OrderType,
    TrailingType,
)
from backtester.engine import BacktestEngine
from backtester.optimizer import Optimizer, SearchMethod
from backtester.performance import compute_performance
from backtester.portfolio import Portfolio
from backtester.report import generate_report

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "Position",
    "Trade",
    "OrderType",
    "TrailingType",
    "Optimizer",
    "SearchMethod",
    "compute_performance",
    "Portfolio",
    "generate_report",
]
