"""
live_bot/engine.py
-------------------
LiveBotEngine — the main orchestrator for Phase 7 paper trading.

WHAT THIS DOES:
    Coordinates all live_bot components:
        1. Loads historical data and seeds the CandleRegistry.
        2. Starts the MarketFeed WebSocket (live price stream).
        3. Starts the PortfolioFeed WebSocket (order/position updates).
        4. On each completed 1-minute candle:
              a. Runs the strategy's generate_signals() on updated candle data.
              b. Checks risk guard before every order.
              c. Routes orders to PaperBroker (Phase 7) or LiveBroker (Phase 8).
        5. On every tick: checks stop-loss and take-profit for open positions.
        6. At 15:20 IST: forces squareoff of all MIS positions.
        7. Exposes status via LiveState for the dashboard.

STRATEGY INTEGRATION:
    The engine works with ANY BaseStrategy subclass from strategies/.
    The same strategy runs in backtest (BacktestEngine) and live (LiveBotEngine)
    because both call strategy.prepare(df) + strategy.generate_signals(df).

    Signal format from generate_signals():
        df["signal"] = 1   → BUY
        df["signal"] = -1  → SHORT (if allowed)
        df["signal"] = 0   → no action

    The engine looks ONLY at the LAST ROW's signal to avoid re-processing
    historical bars on every new candle.

THREADING MODEL:
    - MarketFeed runs in its own daemon thread (blocking WebSocket).
    - PortfolioFeed runs in its own daemon thread.
    - Strategy evaluation runs in a THIRD thread (strategy_loop).
      It wakes up when a new candle is completed (Event-driven).
    - The main thread (FastAPI) reads state only — no writes.

PAPER TRADE MODE FLOW:
    WebSocket tick arrives
        → CandleBuilder updates
        → New 1-min candle completes
        → strategy_loop wakes up
        → strategy.generate_signals(full_df) runs
        → Signal detected at last row
        → RiskGuard.check_order() passes
        → PaperBroker.place_order() simulates fill
        → LiveState updates (position opened, cash debited)
        → Dashboard reads LiveState
"""

import logging
import queue
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Type

import pandas as pd

from config import config
from live_bot.state import state as live_state
from live_bot.candle_builder import candle_registry
from live_bot.feeds.market_feed import MarketFeed
from live_bot.feeds.portfolio_feed import PortfolioFeed
from live_bot.risk.risk_guard import RiskGuard
from live_bot.orders.paper_broker import PaperBroker
from strategies.base_strategy_github import BaseStrategy, Action, Signal

logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


class LiveBotConfig:
    """
    Configuration for one live bot run session.

    Attributes:
        strategy_class    : The strategy class to run (not instance).
        strategy_params   : Params dict passed to strategy constructor.
        instrument_map    : Dict of {instrument_key: symbol}.
        initial_capital   : Starting capital in ₹.
        product           : "I" for intraday MIS, "D" for CNC delivery.
        daily_loss_limit_pct : Stop trading if day loss > this %.
        max_drawdown_pct  : Stop trading if portfolio drawdown > this %.
        max_open_positions: Maximum simultaneous open trades.
        max_position_pct  : Max % of capital per single position.
        seed_lookback_days: How many days of history to seed candles with.
    """

    def __init__(
        self,
        strategy_class:       Type[BaseStrategy],
        strategy_params:      dict,
        instrument_map:       Dict[str, str],       # {instrument_key: symbol}
        initial_capital:      float = 500_000.0,
        product:              str   = "I",
        daily_loss_limit_pct: float = 2.0,
        max_drawdown_pct:     float = 10.0,
        max_open_positions:   int   = 5,
        max_position_pct:     float = 20.0,
        seed_lookback_days:   int   = 60,
        min_bars_required:    int   = 50,          # Min candles before strategy fires
        allow_short:          bool  = False,
    ):
        self.strategy_class       = strategy_class
        self.strategy_params      = strategy_params
        self.instrument_map       = instrument_map
        self.initial_capital      = initial_capital
        self.product              = product
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct     = max_drawdown_pct
        self.max_open_positions   = max_open_positions
        self.max_position_pct     = max_position_pct
        self.seed_lookback_days   = seed_lookback_days
        self.min_bars_required    = min_bars_required
        self.allow_short          = allow_short


class LiveBotEngine:
    """
    Main live trading engine (paper mode for Phase 7).

    Coordinates feeds, strategy, risk management, and order execution.
    """

    def __init__(self, bot_config: LiveBotConfig, access_token: str):
        """
        Args:
            bot_config   : LiveBotConfig with strategy, instruments, and risk settings.
            access_token : Valid Upstox OAuth2 access token.
        """
        self._config       = bot_config
        self._access_token = access_token

        # ── Component initialisation ──────────────────────────────────────────
        # Instantiate the strategy class using parameter dict (unpacked)
        params = bot_config.strategy_params or {}
        if isinstance(params, dict):
            self._strategy = bot_config.strategy_class(**params)
        else:
            # backwards compatibility: if a single positional arg was supplied
            self._strategy = bot_config.strategy_class(params)
        self._risk     = RiskGuard(
            daily_loss_limit_pct = bot_config.daily_loss_limit_pct,
            max_drawdown_pct     = bot_config.max_drawdown_pct,
            max_open_positions   = bot_config.max_open_positions,
            max_position_pct     = bot_config.max_position_pct,
            allow_short          = bot_config.allow_short,
        )
        # Choose broker implementation depending on mode
        if config.PAPER_TRADE:
            from live_bot.orders.paper_broker import PaperBroker
            self._broker = PaperBroker(product=bot_config.product)
            logger.info("[LiveBotEngine] Running in PAPER TRADE mode.")
        else:
            from live_bot.orders.live_broker import LiveBroker
            self._broker = LiveBroker(product=bot_config.product, access_token=self._access_token)
            logger.info("[LiveBotEngine] Running in LIVE TRADE mode.")

        # ── Feeds (created in start()) ────────────────────────────────────────
        self._market_feed:    Optional[MarketFeed]    = None
        self._portfolio_feed: Optional[PortfolioFeed] = None

        # ── Candle completion queue ───────────────────────────────────────────
        # The feed thread puts (symbol, candle) here;
        # the strategy thread reads from it. Bounded to 100 to avoid memory blow-up.
        self._candle_queue: queue.Queue = queue.Queue(maxsize=100)

        # ── Strategy thread ───────────────────────────────────────────────────
        self._strategy_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # ── Squareoff check timer ─────────────────────────────────────────────
        self._squareoff_thread: Optional[threading.Thread] = None

        logger.info(
            f"[LiveBotEngine] Initialised. "
            f"Strategy={self._strategy.__class__.__name__} "
            f"Paper={config.PAPER_TRADE} Capital=₹{bot_config.initial_capital:,.0f}"
        )

    def start(self) -> None:
        """
        Start the live bot:
            1. Set up shared state.
            2. Seed candle builders with historical data.
            3. Start market feed (WebSocket).
            4. Start portfolio feed (WebSocket).
            5. Start strategy evaluation thread.
            6. Start squareoff monitoring thread.
        """
        if live_state._is_running:
            logger.warning("[LiveBotEngine] Already running. Call stop() first.")
            return

        cfg = self._config
        symbols = list(cfg.instrument_map.values())

        # ── Initialise state ──────────────────────────────────────────────────
        live_state.set_initial_capital(cfg.initial_capital)
        live_state.set_active_strategy(self._strategy.__class__.__name__)
        live_state.set_subscribed_symbols(symbols)
        live_state.set_running(True)
        self._stop_event.clear()

        logger.info(f"[LiveBotEngine] Starting bot for symbols: {symbols}")
        live_state.log_activity(
            "BOT_START",
            f"🚀 Bot started | Strategy: {self._strategy.__class__.__name__} | "
            f"Symbols: {', '.join(symbols)} | Capital: ₹{cfg.initial_capital:,.0f}"
        )

        # ── Seed candle builders with historical data ─────────────────────────
        self._seed_candles()

        # ── Start market data WebSocket feed ──────────────────────────────────
        self._market_feed = MarketFeed(
            access_token    = self._access_token,
            instrument_map  = cfg.instrument_map,
            on_candle_complete = self._on_candle_complete,
            mode            = "full",
        )
        self._market_feed.start()

        # ── Start portfolio WebSocket feed ────────────────────────────────────
        self._portfolio_feed = PortfolioFeed(
            access_token       = self._access_token,
            on_order_update    = self._on_order_update,
            on_position_update = None,
        )
        self._portfolio_feed.start()

        # ── Start strategy evaluation thread ──────────────────────────────────
        self._strategy_thread = threading.Thread(
            target = self._strategy_loop,
            name   = "StrategyThread",
            daemon = True,
        )
        self._strategy_thread.start()

        # ── Start squareoff monitor thread ────────────────────────────────────
        self._squareoff_thread = threading.Thread(
            target = self._squareoff_monitor,
            name   = "SquareoffThread",
            daemon = True,
        )
        self._squareoff_thread.start()

        logger.info("[LiveBotEngine] ✅ All components started.")

    def stop(self) -> None:
        """Gracefully shut down all components."""
        logger.info("[LiveBotEngine] Stopping...")
        live_state.log_activity("BOT_STOP", "🛑 Bot stop requested.")

        self._stop_event.set()

        if self._market_feed:
            self._market_feed.stop()
        if self._portfolio_feed:
            self._portfolio_feed.stop()

        live_state.set_running(False)
        logger.info("[LiveBotEngine] Stopped.")

    def activate_kill_switch(self, reason: str = "") -> None:
        """Emergency stop — halt all trading immediately."""
        live_state.activate_kill_switch(reason)
        self._broker.squareoff_all()

    # ── Seed historical candles ───────────────────────────────────────────────

    def _seed_candles(self) -> None:
        """
        Load historical 1-minute OHLCV data for each symbol from Parquet store
        and seed the CandleRegistry. This gives the strategy enough history to
        compute indicators (e.g. a 50-period EMA needs at least 50 bars).
        """
        cfg = self._config
        try:
            from data.parquet_store import ParquetStore
            store = ParquetStore()
        except ImportError:
            logger.warning("[LiveBotEngine] ParquetStore not available. No seed data loaded.")
            for symbol in cfg.instrument_map.values():
                candle_registry.register(symbol, seed_df=None)
            return

        for instrument_key, symbol in cfg.instrument_map.items():
            try:
                df = store.read(
                    symbol    = symbol,
                    timeframe = "minute",
                )
                if df is not None and not df.empty:
                    # Keep only the most recent lookback_days worth of data
                    # 1-min bars * 375 minutes/day * lookback_days
                    max_rows = cfg.seed_lookback_days * 375
                    if len(df) > max_rows:
                        df = df.tail(max_rows).reset_index(drop=True)

                    logger.info(
                        f"[LiveBotEngine] Seeding {symbol} with "
                        f"{len(df)} historical minute bars."
                    )
                    candle_registry.register(symbol, seed_df=df)
                else:
                    logger.warning(
                        f"[LiveBotEngine] No historical data for {symbol}. "
                        "Strategy will need to warm up from live ticks."
                    )
                    candle_registry.register(symbol, seed_df=None)

            except Exception as e:
                logger.error(
                    f"[LiveBotEngine] Error seeding {symbol}: {e}. "
                    "Continuing without seed.",
                    exc_info=False,
                )
                candle_registry.register(symbol, seed_df=None)

    # ── Candle completion callback ────────────────────────────────────────────

    def _on_candle_complete(self, symbol: str, candle: dict) -> None:
        """
        Called by MarketFeed whenever a 1-minute candle completes.
        Puts (symbol, candle) into the queue for the strategy thread.

        EDGE CASE: If the queue is full (strategy thread too slow), drop the
        oldest item to prevent memory growth. This should never happen in
        normal operation but guards against strategy hangs.
        """
        try:
            self._candle_queue.put_nowait((symbol, candle))
        except queue.Full:
            try:
                self._candle_queue.get_nowait()   # Drop oldest
                self._candle_queue.put_nowait((symbol, candle))
                logger.warning(
                    f"[LiveBotEngine] Candle queue full — dropped oldest item. "
                    "Strategy may be too slow."
                )
            except queue.Empty:
                pass

    # ── Strategy evaluation loop ──────────────────────────────────────────────

    def _strategy_loop(self) -> None:
        """
        Strategy evaluation thread — runs until stop_event is set.

        Waits for candles from the queue, then evaluates the strategy.
        Also triggers stop-loss/take-profit checks.
        """
        logger.info("[StrategyThread] Started.")
        cfg = self._config

        while not self._stop_event.is_set():
            try:
                # Block for up to 1 second waiting for a new candle
                symbol, candle = self._candle_queue.get(timeout=1.0)
            except queue.Empty:
                # No new candle in 1s — check stop-loss/TP for existing positions
                self._check_all_sl_tp()
                continue

            try:
                self._evaluate_strategy(symbol)
            except Exception as e:
                logger.error(
                    f"[StrategyThread] Error evaluating strategy for {symbol}: {e}",
                    exc_info=True,
                )

            # Always check SL/TP after every candle, for ALL symbols
            self._check_all_sl_tp()

        logger.info("[StrategyThread] Stopped.")

    def _evaluate_strategy(self, symbol: str) -> None:
        """
        Run the strategy on the latest candle data for a symbol.

        Workflow:
            1. Get the full OHLCV DataFrame from CandleRegistry.
            2. Run strategy.prepare(df) to compute indicators.
            3. Run strategy.generate_signals(df) to get signal column.
            4. Check only the LAST row's signal.
            5. If signal == 1 (BUY) or -1 (SHORT), check risk and place order.
            6. If signal == 0 and we have an open position, check exit condition.
        """
        cfg = self._config

        df = candle_registry.get_df(symbol)

        if df.empty or len(df) < cfg.min_bars_required:
            logger.debug(
                f"[StrategyThread] {symbol}: Only {len(df)} bars, "
                f"need {cfg.min_bars_required}. Warming up..."
            )
            return

        # ── Prepare indicators ────────────────────────────────────────────────
        try:
            df = self._strategy.prepare(df.copy())
        except Exception as e:
            logger.error(f"[StrategyThread] prepare() error for {symbol}: {e}")
            return

        # ── Generate signals ──────────────────────────────────────────────────
        try:
            df = self._strategy.generate_signals(df)
        except Exception as e:
            logger.error(f"[StrategyThread] generate_signals() error for {symbol}: {e}")
            return

        if "signal" not in df.columns:
            logger.debug(f"[StrategyThread] No 'signal' column in df for {symbol}.")
            return

        # ── Inspect LAST bar only ─────────────────────────────────────────────
        last = df.iloc[-1]
        signal_value = int(last.get("signal", 0))
        signal_tag   = str(last.get("signal_tag", self._strategy.__class__.__name__))

        tick = live_state.get_tick(symbol)
        if tick is None:
            logger.debug(f"[StrategyThread] No tick data for {symbol}.")
            return

        ltp = tick.ltp
        instrument_key = tick.instrument_key

        # ── Compute stop loss & take profit ───────────────────────────────────
        stop_loss   = float(last.get("stop_loss",   0) or 0) or None
        take_profit = float(last.get("take_profit", 0) or 0) or None

        # Default stop loss: 2% below entry (for long)
        if stop_loss is None and signal_value == 1:
            stop_loss = round(ltp * 0.98, 2)

        # ── BUY signal ────────────────────────────────────────────────────────
        if signal_value == 1 and not live_state.has_position(symbol):
            qty = self._risk.compute_position_size(ltp, stop_loss)
            allowed, reason = self._risk.check_order(symbol, "BUY", qty, ltp)

            if allowed:
                self._broker.place_order(
                    symbol         = symbol,
                    instrument_key = instrument_key,
                    action         = "BUY",
                    quantity       = qty,
                    order_type     = "MARKET",
                    stop_loss      = stop_loss,
                    take_profit    = take_profit,
                    strategy_tag   = signal_tag,
                )
            else:
                logger.info(f"[StrategyThread] {symbol} BUY blocked: {reason}")

        # ── SELL signal (exit long) ───────────────────────────────────────────
        elif signal_value == -1 and live_state.has_position(symbol):
            position = live_state.get_position(symbol)
            if position and position.direction > 0:
                allowed, reason = self._risk.check_order(
                    symbol, "SELL", position.quantity, ltp
                )
                if allowed:
                    self._broker.place_order(
                        symbol         = symbol,
                        instrument_key = instrument_key,
                        action         = "SELL",
                        quantity       = position.quantity,
                        order_type     = "MARKET",
                        strategy_tag   = signal_tag,
                    )
                else:
                    logger.warning(f"[StrategyThread] {symbol} SELL blocked: {reason}")

        # ── SHORT signal ──────────────────────────────────────────────────────
        elif signal_value == -1 and not live_state.has_position(symbol) and cfg.allow_short:
            qty = self._risk.compute_position_size(ltp, stop_loss)
            allowed, reason = self._risk.check_order(symbol, "SHORT", qty, ltp)
            if allowed:
                self._broker.place_order(
                    symbol         = symbol,
                    instrument_key = instrument_key,
                    action         = "SHORT",
                    quantity       = qty,
                    order_type     = "MARKET",
                    stop_loss      = stop_loss,
                    take_profit    = take_profit,
                    strategy_tag   = signal_tag,
                )

    def _check_all_sl_tp(self) -> None:
        """Check stop-loss and take-profit for all open positions."""
        for symbol in list(live_state.get_all_positions().keys()):
            try:
                self._broker.check_stop_loss_take_profit(symbol)
                self._broker.check_pending_orders(symbol)
            except Exception as e:
                logger.error(f"[StrategyThread] SL/TP check error for {symbol}: {e}")

    # ── Squareoff monitor ─────────────────────────────────────────────────────

    def _squareoff_monitor(self) -> None:
        """
        Thread that fires once at 15:20 IST to close all MIS positions.
        Checks every 30 seconds after 15:15.
        """
        logger.info("[SquareoffThread] Monitor started.")
        while not self._stop_event.is_set():
            try:
                if self._risk.should_squareoff_now():
                    logger.info("[SquareoffThread] 15:20 reached — squaring off all positions.")
                    live_state.log_activity(
                        "SQUAREOFF",
                        "⏰ 15:20 IST reached. Squaring off all intraday positions.",
                        level="WARNING",
                    )
                    self._broker.squareoff_all()
            except Exception as e:
                logger.error(f"[SquareoffThread] Error: {e}")
            time.sleep(30)

        logger.info("[SquareoffThread] Stopped.")

    # ── Portfolio feed callbacks ──────────────────────────────────────────────

    def _on_order_update(self, order_data: dict) -> None:
        """
        Handle order status updates from the PortfolioDataStreamer.
        In paper trade mode: mostly for logging.
        In live trade mode (Phase 8): handles actual fills.
        """
        logger.debug(f"[LiveBotEngine] Portfolio order update: {order_data}")
        # In Phase 7 (paper mode), the PaperBroker manages all state.
        # This callback is reserved for Phase 8 live execution.

    @property
    def is_running(self) -> bool:
        return live_state._is_running
