"""
live_bot/
---------
Phase 7: Paper Trading Engine — Live market feed + simulated order execution.

Module layout:
    feeds/market_feed.py     — MarketDataStreamerV3 wrapper with reconnect + candle building
    feeds/portfolio_feed.py  — PortfolioDataStreamer wrapper for order/position updates
    feeds/webhook_server.py  — FastAPI webhook receiver (Upstox postback URL)
    risk/risk_guard.py       — Daily loss limit, max positions, kill switch
    orders/paper_broker.py   — Paper trading order simulation (no real money)
    orders/order_manager.py  — Order state machine + fill tracker
    engine.py                — LiveBotEngine: orchestrates all feeds + strategy
    candle_builder.py        — Assembles tick data into OHLCV 1-min bars
    state.py                 — Shared in-memory state (positions, P&L, ticks)
"""
