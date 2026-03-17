"""
strategies/day_strategy
========================
Intraday strategies package.

Contains strategies designed exclusively for same-day (MIS) trading.
All strategies in this package:
  - Generate signals only within NSE market hours (09:15–15:29 IST)
  - Hard-close all positions by 15:15 IST at the latest
  - Are backtested with Segment.EQUITY_INTRADAY commission model

Available strategies:
  - ORBNiftyStrategy : Opening Range Breakout for NIFTY 50 Index
"""

from strategies.day_strategy.orb_nifty import ORBNiftyStrategy

__all__ = ["ORBNiftyStrategy"]