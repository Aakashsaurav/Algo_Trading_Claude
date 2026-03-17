"""
tests/test_phase7.py
--------------------
Comprehensive test suite for Phase 7 — Paper Trading (Live Bot) Layer.

WHAT IS TESTED:
    1. WebSocket message parsing (market_feed)
    2. CandleBuilder + CandleRegistry
    3. LiveState (shared state store)
    4. PaperBroker
    5. RiskGuard
    6. Portfolio feed message parsing
    7. Webhook endpoints (via FastAPI test client)
    8. Full simulation: 5 symbols through the entire pipeline

HOW TO RUN:
    cd algo_trading/
    python -m pytest tests/test_phase7.py -v
    # Or without pytest:
    python tests/test_phase7.py
"""

import os
import sys
import threading
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pandas as pd

# ── Environment setup (must happen before any project imports) ─────────────
os.environ.setdefault("UPSTOX_API_KEY",    "test_key_for_unit_tests")
os.environ.setdefault("UPSTOX_API_SECRET", "test_secret_for_unit_tests")

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Project imports ────────────────────────────────────────────────────────
from live_bot.state import (
    TickData, LivePosition, LiveOrder, ClosedTrade, LiveState
)
from live_bot.candle_builder import MinuteCandle, CandleBuilder, CandleRegistry
from live_bot.feeds.market_feed import (
    _parse_ltt, _extract_ohlc_from_feed, _parse_message
)
from live_bot.feeds.portfolio_feed import (
    _parse_order_update, _parse_position_update
)
from live_bot.risk.risk_guard import RiskGuard

IST = timezone(timedelta(hours=5, minutes=30))

# ── 5 real NSE symbols ─────────────────────────────────────────────────────
SYMBOLS      = ["RELIANCE", "TCS", "INFY", "HDFC", "SBIN"]
KEYS         = [
    "NSE_EQ|INE020B01018",
    "NSE_EQ|INE467B01029",
    "NSE_EQ|INE009A01021",
    "NSE_EQ|INE001A01036",
    "NSE_EQ|INE062A01020",
]
BASE_PRICES  = [2550.0, 3800.0, 1450.0, 1600.0, 620.0]
PREV_CLOSES  = [2500.0, 3750.0, 1420.0, 1580.0, 610.0]
INSTRUMENT_MAP = {k: s for k, s in zip(KEYS, SYMBOLS)}


# ═══════════════════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════════════════

def make_tick(
    symbol: str, key: str, ltp: float, minute: int,
    second: int = 30, volume: int = 1_000_000,
    close_px: float = 0.0, with_candle: bool = True,
) -> TickData:
    cp = close_px if close_px else ltp - 20
    return TickData(
        instrument_key=key, symbol=symbol, ltp=ltp,
        ltt=datetime(2026, 3, 7, 10, minute, second, tzinfo=IST),
        ltq=100, close_price=cp,
        open_price=ltp - 30, high_price=ltp + 50, low_price=ltp - 50,
        volume=volume, oi=0.0,
        candle_open  =ltp - 2 if with_candle else 0.0,
        candle_high  =ltp + 3 if with_candle else 0.0,
        candle_low   =ltp - 3 if with_candle else 0.0,
        candle_close =ltp     if with_candle else 0.0,
        candle_volume=5000    if with_candle else 0,
    )


def make_full_feed_message(instruments_data: dict) -> dict:
    """Build a realistic Upstox V3 full-mode WebSocket message."""
    feeds = {}
    for key, (sym, ltp, cp) in instruments_data.items():
        feeds[key] = {
            "fullFeed": {
                "marketFF": {
                    "ltpc": {"ltp": ltp, "ltt": 1709797800000, "ltq": 100, "cp": cp},
                    "marketOHLC": {
                        "ohlc": [
                            {"interval": "1d",  "open": ltp-50, "high": ltp+60, "low": ltp-80, "close": ltp, "volume": 5_000_000},
                            {"interval": "I30", "open": ltp-10, "high": ltp+15, "low": ltp-15, "close": ltp, "volume": 200_000},
                            {"interval": "I1",  "open": ltp-2,  "high": ltp+3,  "low": ltp-3,  "close": ltp, "volume": 15_000},
                        ]
                    },
                    "eFeedDetails": {
                        "atp": ltp-1.5, "vtt": 8_000_000, "oi": 0,
                        "open": ltp-40, "high": ltp+60, "low": ltp-90,
                        "lowerCB": ltp*0.9, "upperCB": ltp*1.1,
                    },
                }
            }
        }
    return {"feeds": feeds, "currentTs": 1709797800000, "type": "market_data"}


def make_fresh_state(capital: float = 500_000.0) -> LiveState:
    s = LiveState()
    s.set_initial_capital(capital)
    return s


def seed_df(base_price: float, n_bars: int = 60) -> pd.DataFrame:
    """Synthetic historical OHLCV DataFrame."""
    import random
    rows = []
    base_dt = datetime(2026, 3, 7, 9, 15, tzinfo=IST)
    price = base_price
    for i in range(n_bars):
        price = max(price + random.uniform(-5, 5), 1.0)
        rows.append({
            "datetime": base_dt + timedelta(minutes=i),
            "open":   round(price - 1, 2), "high": round(price + 2, 2),
            "low":    round(price - 2, 2), "close": round(price, 2),
            "volume": 10_000 + i * 100,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 1. WebSocket Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestParseLtt(unittest.TestCase):

    def test_epoch_ms(self):
        result = _parse_ltt(1709777400000)
        self.assertIsInstance(result, datetime)
        self.assertIsNotNone(result.tzinfo)

    def test_none_fallback(self):
        result = _parse_ltt(None)
        self.assertIsInstance(result, datetime)

    def test_datetime_with_tz(self):
        dt = datetime(2026, 3, 7, 10, 0, tzinfo=IST)
        self.assertEqual(_parse_ltt(dt), dt)

    def test_naive_datetime_gets_tz(self):
        dt = datetime(2026, 3, 7, 10, 0)
        result = _parse_ltt(dt)
        self.assertIsNotNone(result.tzinfo)

    def test_float_epoch(self):
        self.assertIsInstance(_parse_ltt(1709777400000.5), datetime)

    def test_overflow_epoch_no_crash(self):
        result = _parse_ltt(999_999_999_999_999_999)
        self.assertIsInstance(result, datetime)


class TestExtractOhlcFromFeed(unittest.TestCase):

    def _mff(self, intervals=None):
        if intervals is None:
            intervals = [
                {"interval": "I1",  "open": 100.0, "high": 105.0, "low": 99.0,  "close": 103.0, "volume": 5000},
                {"interval": "I30", "open": 90.0,  "high": 110.0, "low": 88.0,  "close": 102.0, "volume": 50000},
                {"interval": "1d",  "open": 80.0,  "high": 120.0, "low": 75.0,  "close": 100.0, "volume": 500000},
            ]
        return {"marketOHLC": {"ohlc": intervals}}

    def test_i1_selected_correctly(self):
        o, h, l, c, v = _extract_ohlc_from_feed(self._mff())
        self.assertAlmostEqual(c, 103.0)
        self.assertEqual(v, 5000)

    def test_no_i1_falls_back_to_first(self):
        mff = {"marketOHLC": {"ohlc": [
            {"interval": "I30", "open": 90.0, "high": 110.0, "low": 88.0, "close": 102.0, "volume": 50000}
        ]}}
        o, _, _, c, _ = _extract_ohlc_from_feed(mff)
        self.assertAlmostEqual(o, 90.0)

    def test_missing_marketohlc_returns_zeros(self):
        self.assertEqual(_extract_ohlc_from_feed({}), (0.0, 0.0, 0.0, 0.0, 0))

    def test_empty_ohlc_list_returns_zeros(self):
        self.assertEqual(_extract_ohlc_from_feed({"marketOHLC": {"ohlc": []}}), (0.0, 0.0, 0.0, 0.0, 0))

    def test_none_fields_become_zero(self):
        mff = {"marketOHLC": {"ohlc": [
            {"interval": "I1", "open": None, "high": None, "low": None, "close": None, "volume": None}
        ]}}
        o, h, l, c, v = _extract_ohlc_from_feed(mff)
        self.assertEqual(c, 0.0)

    def test_all_5_symbols(self):
        for sym, price in zip(SYMBOLS, BASE_PRICES):
            mff = {"marketOHLC": {"ohlc": [
                {"interval": "I1", "open": price-2, "high": price+3, "low": price-3, "close": price, "volume": 5000}
            ]}}
            _, _, _, c, _ = _extract_ohlc_from_feed(mff)
            self.assertAlmostEqual(c, price, msg=f"Failed for {sym}")

    def test_i1_middle_of_list(self):
        """I1 candle found even when not first in list."""
        market_ff = {
            "marketOHLC": {
                "ohlc": [
                    {"interval": "1d",  "open": 2500.0, "high": 2610.0, "low": 2420.0, "close": 2550.0, "volume": 5_000_000},
                    {"interval": "I30", "open": 2540.0, "high": 2565.0, "low": 2535.0, "close": 2550.0, "volume": 200_000},
                    {"interval": "I1",  "open": 2548.0, "high": 2553.0, "low": 2547.0, "close": 2551.0, "volume": 15_000},
                ]
            }
        }
        o, h, l, c, v = _extract_ohlc_from_feed(market_ff)
        self.assertAlmostEqual(o, 2548.0)
        self.assertAlmostEqual(c, 2551.0)
        self.assertEqual(v, 15_000)


class TestParseMessage(unittest.TestCase):

    def setUp(self):
        import live_bot.feeds.market_feed as mf
        mf._KEY_TO_SYMBOL.update(INSTRUMENT_MAP)

    def test_parse_5_symbols(self):
        data = {k: (s, p, c) for k, s, p, c in zip(KEYS, SYMBOLS, BASE_PRICES, PREV_CLOSES)}
        ticks = _parse_message(make_full_feed_message(data))
        self.assertEqual(len(ticks), 5)

    def test_ltp_correct(self):
        data = {KEYS[0]: (SYMBOLS[0], 2550.0, 2500.0)}
        ticks = _parse_message(make_full_feed_message(data))
        self.assertAlmostEqual(ticks[0].ltp, 2550.0)

    def test_candle_ohlc_not_zero_after_bugfix(self):
        """BUG FIX: I1 candle must be non-zero when marketOHLC.I1 is present."""
        data = {KEYS[0]: (SYMBOLS[0], 2550.0, 2500.0)}
        ticks = _parse_message(make_full_feed_message(data))
        t = ticks[0]
        self.assertGreater(t.candle_close, 0.0,
            "candle_close is 0 — double-unwrap bug may not be fixed!")
        self.assertAlmostEqual(t.candle_close, 2550.0, places=0)

    def test_change_pct_correct(self):
        data = {KEYS[0]: (SYMBOLS[0], 2550.0, 2500.0)}
        ticks = _parse_message(make_full_feed_message(data))
        expected = round((2550.0 - 2500.0) / 2500.0 * 100, 2)
        self.assertAlmostEqual(ticks[0].change_pct, expected, places=1)

    def test_zero_price_skipped(self):
        feeds = {"NSE_EQ|ZERO": {"fullFeed": {"marketFF": {
            "ltpc": {"ltp": 0.0, "ltt": 1709797800000, "ltq": 0, "cp": 0.0}
        }}}}
        ticks = _parse_message({"feeds": feeds, "type": "market_data", "currentTs": 1709797800000})
        self.assertEqual(len(ticks), 0)

    def test_none_message(self):
        self.assertEqual(_parse_message(None), [])

    def test_empty_dict(self):
        self.assertEqual(_parse_message({}), [])

    def test_missing_feeds_key(self):
        self.assertEqual(_parse_message({"type": "market_status"}), [])

    def test_partial_no_ohlc(self):
        feeds = {"NSE_EQ|PARTIAL": {"fullFeed": {"marketFF": {
            "ltpc": {"ltp": 500.0, "ltt": 1709797800000, "ltq": 50, "cp": 490.0}
        }}}}
        ticks = _parse_message({"feeds": feeds, "type": "market_data", "currentTs": 1709797800000})
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].candle_close, 0.0)

    def test_one_bad_instrument_does_not_kill_rest(self):
        data = {k: (s, p, c) for k, s, p, c in zip(KEYS, SYMBOLS, BASE_PRICES, PREV_CLOSES)}
        msg = make_full_feed_message(data)
        msg["feeds"][KEYS[2]] = {"fullFeed": {"marketFF": "CORRUPT"}}
        ticks = _parse_message(msg)
        self.assertEqual(len(ticks), 4)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CandleBuilder Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMinuteCandle(unittest.TestCase):

    def _candle(self, price=100.0, vol=500):
        return MinuteCandle(datetime(2026, 3, 7, 10, 0, tzinfo=IST), price, vol)

    def test_init_sets_all_ohlc(self):
        c = self._candle(100.0)
        self.assertEqual(c.open, 100.0)
        self.assertEqual(c.high, 100.0)
        self.assertEqual(c.low,  100.0)
        self.assertEqual(c.close, 100.0)

    def test_update_tracks_high_low_close(self):
        c = self._candle(100.0)
        c.update(110.0, 200)
        c.update(90.0,  100)
        c.update(105.0, 150)
        self.assertEqual(c.high,  110.0)
        self.assertEqual(c.low,    90.0)
        self.assertEqual(c.close, 105.0)
        self.assertEqual(c.volume, 500 + 450)

    def test_negative_volume_clamped(self):
        c = self._candle(100.0, vol=1000)
        c.update(101.0, -500)
        self.assertEqual(c.volume, 1000)

    def test_to_dict_keys(self):
        c = self._candle()
        d = c.to_dict()
        for key in ("datetime", "open", "high", "low", "close", "volume"):
            self.assertIn(key, d)


class TestCandleBuilder(unittest.TestCase):

    def _builder(self, sym="RELIANCE", price=2550.0, n=50):
        return CandleBuilder(sym, seed_df=seed_df(price, n), max_history_bars=200)

    def test_seed_loads_bars(self):
        self.assertEqual(self._builder().bar_count(), 50)

    def test_bad_seed_no_crash(self):
        bad = pd.DataFrame({"a": [1, 2]})
        b = CandleBuilder("BAD", seed_df=bad)
        self.assertEqual(b.bar_count(), 0)

    def test_string_index_seed_no_crash(self):
        """BUG FIX: non-integer index should not cause iloc crash."""
        df = seed_df(1000.0, n_bars=20)
        df.index = [f"r{i}" for i in range(20)]
        b = CandleBuilder("STR_IDX", seed_df=df)
        self.assertEqual(b.bar_count(), 20)

    def test_same_minute_no_completion(self):
        b = self._builder()
        initial = b.bar_count()
        for i in range(5):
            b.on_tick(make_tick("RELIANCE", KEYS[0], 2550.0, minute=15, second=i*10))
        self.assertEqual(b.bar_count(), initial)

    def test_new_minute_completes_candle(self):
        b = self._builder()
        initial = b.bar_count()
        b.on_tick(make_tick("RELIANCE", KEYS[0], 2550.0, minute=15, second=30))
        completed = b.on_tick(make_tick("RELIANCE", KEYS[0], 2555.0, minute=16, second=5))
        self.assertIsNotNone(completed)
        self.assertEqual(b.bar_count(), initial + 1)

    def test_feed_candle_preferred(self):
        b = CandleBuilder("RELIANCE")
        tick = make_tick("RELIANCE", KEYS[0], 2550.0, minute=15)
        tick.candle_close = 2560.0
        b.on_tick(tick)
        self.assertAlmostEqual(b.get_current_bar()["close"], 2560.0)

    def test_ltp_fallback_when_no_candle_feed(self):
        b = CandleBuilder("RELIANCE")
        tick = make_tick("RELIANCE", KEYS[0], 2550.0, minute=15, with_candle=False)
        b.on_tick(tick)
        self.assertAlmostEqual(b.get_current_bar()["close"], 2550.0)

    def test_max_history_trimming(self):
        b = CandleBuilder("RELIANCE", max_history_bars=10)
        for m in range(15):
            b.on_tick(make_tick("RELIANCE", KEYS[0], 2550.0, minute=m,   second=5))
            b.on_tick(make_tick("RELIANCE", KEYS[0], 2550.0, minute=m+1, second=5))
        self.assertLessEqual(b.bar_count(), 10)

    def test_get_candles_df_sorted(self):
        b = self._builder()
        df = b.get_candles_df()
        self.assertTrue(df["datetime"].is_monotonic_increasing)

    def test_empty_df_correct_columns(self):
        b = CandleBuilder("EMPTY")
        df = b.get_candles_df()
        self.assertTrue(df.empty)
        self.assertEqual(set(df.columns), {"datetime", "open", "high", "low", "close", "volume"})

    def test_get_new_candles_incremental(self):
        b = self._builder()
        b.get_new_candles()   # consume seed
        for m in range(11, 14):
            b.on_tick(make_tick("RELIANCE", KEYS[0], 2550.0, minute=m, second=5))
        new = b.get_new_candles()
        self.assertEqual(len(new), 2)
        self.assertEqual(len(b.get_new_candles()), 0)

    def test_thread_safety(self):
        b = CandleBuilder("THREAD")
        errors = []
        def write(start):
            for m in range(start, start+5):
                try:
                    b.on_tick(make_tick("THREAD", KEYS[0], 2550.0, minute=m))
                    b.get_candles_df()
                except Exception as e:
                    errors.append(str(e))
        threads = [threading.Thread(target=write, args=(i*10,)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])


class TestCandleRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = CandleRegistry()

    def test_register_5_symbols(self):
        for sym, price in zip(SYMBOLS, BASE_PRICES):
            self.reg.register(sym, seed_df=seed_df(price, n_bars=30))
        counts = self.reg.bar_counts()
        for sym in SYMBOLS:
            self.assertEqual(counts[sym], 30)

    def test_auto_register_unknown(self):
        self.reg.on_tick("WIPRO", make_tick("WIPRO", "NSE_EQ|WIPRO", 400.0, minute=5))
        self.assertIn("WIPRO", self.reg.get_symbols())

    def test_get_df_unknown_returns_empty(self):
        df = self.reg.get_df("NONEXISTENT")
        self.assertTrue(df.empty)

    def test_candle_completion_via_registry(self):
        self.reg.register("RELIANCE", seed_df=seed_df(2550.0, n_bars=10))
        initial = self.reg.bar_counts()["RELIANCE"]
        self.reg.on_tick("RELIANCE", make_tick("RELIANCE", KEYS[0], 2550.0, minute=15, second=10))
        completed = self.reg.on_tick("RELIANCE", make_tick("RELIANCE", KEYS[0], 2555.0, minute=16, second=5))
        self.assertIsNotNone(completed)
        self.assertEqual(self.reg.bar_counts()["RELIANCE"], initial + 1)


# ═══════════════════════════════════════════════════════════════════════════
# 3. LiveState Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLiveState(unittest.TestCase):

    def setUp(self):
        self.state = make_fresh_state(500_000.0)

    def _inject(self, sym, key, ltp, cp=None):
        tick = make_tick(sym, key, ltp, minute=10, close_px=cp or ltp-20)
        self.state.update_tick(sym, tick)
        return tick

    def test_update_get_tick(self):
        self._inject("RELIANCE", KEYS[0], 2550.0)
        t = self.state.get_tick("RELIANCE")
        self.assertAlmostEqual(t.ltp, 2550.0)

    def test_unknown_tick_returns_none(self):
        self.assertIsNone(self.state.get_tick("UNKNOWN"))

    def test_5_symbols_independent(self):
        for sym, key, p, cp in zip(SYMBOLS, KEYS, BASE_PRICES, PREV_CLOSES):
            self._inject(sym, key, p, cp)
        for sym, p in zip(SYMBOLS, BASE_PRICES):
            self.assertAlmostEqual(self.state.get_tick(sym).ltp, p)

    def test_add_has_position(self):
        pos = LivePosition("RELIANCE", KEYS[0], 1, 10, 2550.0, datetime.now(IST))
        self.state.add_position(pos)
        self.assertTrue(self.state.has_position("RELIANCE"))
        self.assertFalse(self.state.has_position("TCS"))

    def test_close_position(self):
        self.state.add_position(LivePosition("TCS", KEYS[1], 1, 5, 3800.0, datetime.now(IST)))
        closed = self.state.close_position("TCS")
        self.assertIsNotNone(closed)
        self.assertFalse(self.state.has_position("TCS"))

    def test_debit_credit(self):
        initial = self.state.cash
        self.state.debit_cash(50_000.0)
        self.assertAlmostEqual(self.state.cash, initial - 50_000.0)
        self.state.credit_cash(20_000.0)
        self.assertAlmostEqual(self.state.cash, initial - 30_000.0)

    def test_debit_beyond_cash_raises(self):
        with self.assertRaises(ValueError):
            self.state.debit_cash(1_000_000_000.0)

    def test_order_lifecycle(self):
        order = LiveOrder(
            order_id="ORD001", symbol="INFY", instrument_key=KEYS[2],
            action="BUY", quantity=20, order_type="MARKET", limit_price=None,
            status="PENDING", created_at=datetime.now(IST)
        )
        self.state.add_order(order)
        self.state.update_order_status("ORD001", "FILLED", fill_price=1450.0, filled_at=datetime.now(IST))
        updated = self.state.get_order("ORD001")
        self.assertEqual(updated.status, "FILLED")
        self.assertAlmostEqual(updated.fill_price, 1450.0)

    def test_record_closed_trade(self):
        trade = ClosedTrade(
            symbol="HDFC", direction="LONG", quantity=5,
            entry_price=1600.0, exit_price=1650.0,
            entry_time=datetime.now(IST), exit_time=datetime.now(IST),
            pnl=250.0, pnl_pct=3.125, exit_reason="SIGNAL"
        )
        self.state.record_closed_trade(trade)
        trades = self.state.get_closed_trades()
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0].pnl, 250.0)

    def test_kill_switch(self):
        self.assertFalse(self.state.kill_switch)
        self.state.activate_kill_switch("Test")
        self.assertTrue(self.state.kill_switch)
        self.assertFalse(self.state.is_trading_allowed())

    def test_get_activity_log_exists(self):
        """BUG FIX: get_activity_log() must exist."""
        self.state.log_activity("EV1", "msg1", level="INFO")
        self.state.log_activity("EV2", "msg2", level="WARNING")
        log = self.state.get_activity_log()
        self.assertEqual(len(log), 2)
        self.assertEqual(log[1]["level"], "WARNING")

    def test_get_status_dict_alias(self):
        """BUG FIX: get_status_dict() alias must exist."""
        d = self.state.get_status_dict()
        self.assertIn("cash", d)
        self.assertIn("is_running", d)

    def test_get_status_snapshot(self):
        snap = self.state.get_status_snapshot()
        for key in ("open_positions", "market_ticks", "closed_trades", "activity_log"):
            self.assertIn(key, snap)

    def test_activity_log_max_size(self):
        for i in range(250):
            self.state.log_activity("SPAM", f"entry {i}")
        log = self.state.get_activity_log()
        self.assertLessEqual(len(log), 200)

    def test_total_value_includes_positions(self):
        pos = LivePosition("RELIANCE", KEYS[0], 1, 10, 2500.0, datetime.now(IST))
        self.state.add_position(pos)
        self.state.debit_cash(25_000.0)
        self.state.update_tick("RELIANCE", make_tick("RELIANCE", KEYS[0], 2550.0, minute=10))
        tv = self.state.total_value
        # cash 475000 + position 25500 = 500500
        self.assertGreater(tv, 475_000.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Portfolio Feed Parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestPortfolioFeedParsing(unittest.TestCase):

    def test_order_update_complete(self):
        data = {
            "order_id": "2024010101234", "status": "complete",
            "instrument_token": "NSE_EQ|INE020B01018", "transaction_type": "BUY",
            "quantity": 10, "average_price": 2550.50,
            "filled_quantity": 10, "pending_quantity": 0,
            "order_type": "MARKET", "product": "I",
        }
        p = _parse_order_update(data)
        self.assertEqual(p["status"], "complete")
        self.assertAlmostEqual(p["average_price"], 2550.50)
        self.assertEqual(p["filled_quantity"], 10)

    def test_order_update_missing_fields_defaults(self):
        p = _parse_order_update({"order_id": "12345"})
        self.assertEqual(p["status"], "unknown")
        self.assertEqual(p["quantity"], 0)

    def test_order_update_non_dict(self):
        self.assertEqual(_parse_order_update("bad"), {})
        self.assertEqual(_parse_order_update(None), {})

    def test_all_order_statuses(self):
        for status in ("open", "complete", "rejected", "cancelled", "trigger pending"):
            p = _parse_order_update({"order_id": "X", "status": status})
            self.assertEqual(p["status"], status.lower())

    def test_position_update(self):
        data = {
            "instrument_token": "NSE_EQ|INE020B01018",
            "average_price": 2555.0, "quantity": 10,
            "buy_value": 25550.0, "sell_value": 0.0, "product": "I",
        }
        p = _parse_position_update(data)
        self.assertAlmostEqual(p["average_price"], 2555.0)
        self.assertEqual(p["quantity"], 10)

    def test_position_update_defaults(self):
        p = _parse_position_update({})
        self.assertEqual(p["quantity"], 0)


# ═══════════════════════════════════════════════════════════════════════════
# 5. RiskGuard Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRiskGuard(unittest.TestCase):

    def setUp(self):
        import live_bot.state as s_mod
        s_mod.state = make_fresh_state(500_000.0)
        self.state = s_mod.state
        self.guard = RiskGuard(
            daily_loss_limit_pct=2.0, max_drawdown_pct=10.0,
            max_open_positions=3, max_position_pct=30.0, allow_short=False,
        )
        for sym, key, p, cp in zip(SYMBOLS, KEYS, BASE_PRICES, PREV_CLOSES):
            self.state.update_tick(sym, make_tick(sym, key, p, minute=10, close_px=cp))

    @patch("live_bot.risk.risk_guard.datetime")
    def test_buy_allowed_market_hours(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 10, 30, tzinfo=IST)
        allowed, reason = self.guard.check_order("RELIANCE", "BUY", 1, 2550.0)
        self.assertTrue(allowed, f"Expected allowed: {reason}")

    @patch("live_bot.risk.risk_guard.datetime")
    def test_buy_blocked_before_open(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 8, 0, tzinfo=IST)
        allowed, reason = self.guard.check_order("RELIANCE", "BUY", 1, 2550.0)
        self.assertFalse(allowed)
        self.assertIn("MARKET_CLOSED", reason)

    @patch("live_bot.risk.risk_guard.datetime")
    def test_buy_blocked_after_close(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 15, 35, tzinfo=IST)
        allowed, reason = self.guard.check_order("RELIANCE", "BUY", 1, 2550.0)
        self.assertFalse(allowed)
        self.assertIn("MARKET_CLOSED", reason)

    @patch("live_bot.risk.risk_guard.datetime")
    def test_sell_allowed_after_close(self, mock_dt):
        """SELL (exit) is allowed even outside market hours."""
        mock_dt.now.return_value = datetime(2026, 3, 7, 16, 0, tzinfo=IST)
        _, reason = self.guard.check_order("RELIANCE", "SELL", 5, 2550.0)
        self.assertNotIn("MARKET_CLOSED", reason)

    def test_kill_switch_blocks_all(self):
        self.state.activate_kill_switch("Test")
        allowed, reason = self.guard.check_order("RELIANCE", "BUY", 1, 2550.0)
        self.assertFalse(allowed)
        self.assertIn("KILL_SWITCH", reason)

    @patch("live_bot.risk.risk_guard.datetime")
    def test_max_positions(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 10, 30, tzinfo=IST)
        for sym, key, p in zip(SYMBOLS[:3], KEYS[:3], BASE_PRICES[:3]):
            self.state.add_position(LivePosition(sym, key, 1, 10, p, datetime.now(IST)))
        allowed, reason = self.guard.check_order("SBIN", "BUY", 10, 620.0)
        self.assertFalse(allowed)
        self.assertIn("MAX_POSITIONS", reason)

    @patch("live_bot.risk.risk_guard.datetime")
    def test_duplicate_long(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 10, 30, tzinfo=IST)
        self.state.add_position(LivePosition("RELIANCE", KEYS[0], 1, 10, 2550.0, datetime.now(IST)))
        allowed, reason = self.guard.check_order("RELIANCE", "BUY", 5, 2550.0)
        self.assertFalse(allowed)
        self.assertIn("DUPLICATE_LONG", reason)

    @patch("live_bot.risk.risk_guard.datetime")
    def test_short_not_allowed(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 10, 30, tzinfo=IST)
        allowed, reason = self.guard.check_order("RELIANCE", "SHORT", 5, 2550.0)
        self.assertFalse(allowed)
        self.assertIn("SHORT_NOT_ALLOWED", reason)

    @patch("live_bot.risk.risk_guard.datetime")
    def test_insufficient_cash(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 10, 30, tzinfo=IST)
        self.state.debit_cash(499_000.0)
        allowed, reason = self.guard.check_order("RELIANCE", "BUY", 10, 2550.0)
        self.assertFalse(allowed)
        self.assertIn("INSUFFICIENT_CASH", reason)

    @patch("live_bot.risk.risk_guard.datetime")
    def test_position_size_computed(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 10, 30, tzinfo=IST)
        qty = self.guard.compute_position_size(price=2550.0, stop_loss=2500.0)
        self.assertGreater(qty, 0)
        self.assertIsInstance(qty, int)

    def test_is_market_open_returns_bool(self):
        """BUG FIX: is_market_open() must exist and return bool."""
        result = self.guard.is_market_open()
        self.assertIsInstance(result, bool)

    @patch("live_bot.risk.risk_guard.datetime")
    def test_squareoff_time_blocks_buy(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 15, 21, tzinfo=IST)
        allowed, reason = self.guard.check_order("RELIANCE", "BUY", 1, 2550.0)
        self.assertFalse(allowed)
        self.assertIn("SQUAREOFF_TIME", reason)


# ═══════════════════════════════════════════════════════════════════════════
# 6. PaperBroker Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPaperBroker(unittest.TestCase):

    def setUp(self):
        import live_bot.state as s_mod
        s_mod.state = make_fresh_state(500_000.0)
        self.state = s_mod.state
        from live_bot.orders.paper_broker import PaperBroker
        self.broker = PaperBroker()
        for sym, key, p, cp in zip(SYMBOLS, KEYS, BASE_PRICES, PREV_CLOSES):
            self.state.update_tick(sym, make_tick(sym, key, p, minute=10, close_px=cp))

    def _buy(self, sym, key, price, qty=5, sl_offset=0.02, tp_offset=0.04):
        self.broker.place_order(
            symbol=sym, instrument_key=key, action="BUY", quantity=qty,
            order_type="MARKET", stop_loss=round(price*(1-sl_offset), 2),
            take_profit=round(price*(1+tp_offset), 2), strategy_tag="TEST"
        )

    def test_buy_creates_position_debits_cash(self):
        initial = self.state.cash
        self._buy("RELIANCE", KEYS[0], BASE_PRICES[0])
        self.assertTrue(self.state.has_position("RELIANCE"))
        self.assertLess(self.state.cash, initial)

    def test_sell_closes_position_records_trade(self):
        self._buy("TCS", KEYS[1], BASE_PRICES[1])
        self.assertTrue(self.state.has_position("TCS"))
        initial_trades = len(self.state.get_closed_trades())
        self.broker.place_order("TCS", KEYS[1], "SELL", 5, "MARKET", strategy_tag="TEST")
        self.assertFalse(self.state.has_position("TCS"))
        self.assertEqual(len(self.state.get_closed_trades()), initial_trades + 1)

    def test_sell_no_position_rejected(self):
        self.broker.place_order("INFY", KEYS[2], "SELL", 10, "MARKET", strategy_tag="TEST")
        self.assertFalse(self.state.has_position("INFY"))

    def test_slippage_on_buy(self):
        raw_ltp = self.state.get_tick("RELIANCE").ltp
        self._buy("RELIANCE", KEYS[0], raw_ltp)
        pos = self.state.get_position("RELIANCE")
        self.assertGreaterEqual(pos.entry_price, raw_ltp)

    def test_stop_loss_fires(self):
        self._buy("RELIANCE", KEYS[0], BASE_PRICES[0])
        sl_tick = make_tick("RELIANCE", KEYS[0], BASE_PRICES[0] * 0.96, minute=15)
        self.state.update_tick("RELIANCE", sl_tick)
        self.broker.check_stop_loss_take_profit("RELIANCE")
        self.assertFalse(self.state.has_position("RELIANCE"))
        self.assertTrue(any(t.exit_reason == "STOP_LOSS" for t in self.state.get_closed_trades()))

    def test_take_profit_fires(self):
        self._buy("TCS", KEYS[1], BASE_PRICES[1])
        tp_tick = make_tick("TCS", KEYS[1], BASE_PRICES[1] * 1.06, minute=16)
        self.state.update_tick("TCS", tp_tick)
        self.broker.check_stop_loss_take_profit("TCS")
        self.assertFalse(self.state.has_position("TCS"))
        self.assertTrue(any(t.exit_reason == "TAKE_PROFIT" for t in self.state.get_closed_trades()))

    def test_squareoff_all(self):
        for sym, key, p in zip(SYMBOLS[:3], KEYS[:3], BASE_PRICES[:3]):
            self._buy(sym, key, p, qty=2)
        self.assertEqual(len(self.state.get_all_positions()), 3)
        self.broker.squareoff_all()
        self.assertEqual(len(self.state.get_all_positions()), 0)

    def test_limit_order_not_filled_at_higher_price(self):
        ltp = self.state.get_tick("SBIN").ltp   # 620
        self.broker.place_order(
            "SBIN", KEYS[4], "BUY", 10, "LIMIT",
            limit_price=ltp - 20, strategy_tag="LIM"
        )
        self.assertFalse(self.state.has_position("SBIN"))

    def test_limit_order_fills_when_price_drops(self):
        ltp = self.state.get_tick("SBIN").ltp
        self.broker.place_order(
            "SBIN", KEYS[4], "BUY", 10, "LIMIT",
            limit_price=ltp - 20, strategy_tag="LIM"
        )
        # Drop price to fill
        self.state.update_tick("SBIN", make_tick("SBIN", KEYS[4], ltp - 25, minute=12))
        self.broker.check_pending_orders("SBIN")
        self.assertTrue(self.state.has_position("SBIN"))


# ═══════════════════════════════════════════════════════════════════════════
# 7. Webhook Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWebhookEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            import live_bot.state as s_mod
            s_mod.state = make_fresh_state()
            from fastapi import FastAPI
            from live_bot.feeds.webhook_server import webhook_router
            app = FastAPI()
            app.include_router(webhook_router, prefix="/webhook")
            cls.client = TestClient(app)
            cls.available = True
        except Exception as e:
            cls.available = False
            cls.skip_reason = str(e)

    def setUp(self):
        if not self.available:
            self.skipTest(f"FastAPI test client unavailable: {self.skip_reason}")

    def test_health_check(self):
        resp = self.client.get("/webhook/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_valid_order_update(self):
        payload = {
            "order_id": "2026030701234567", "status": "complete",
            "instrument_token": "NSE_EQ|INE020B01018",
            "transaction_type": "BUY", "quantity": 10,
            "average_price": 2550.50, "filled_quantity": 10,
        }
        resp = self.client.post("/webhook/order-update", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_order_update_missing_order_id(self):
        resp = self.client.post("/webhook/order-update", json={"status": "complete"})
        self.assertEqual(resp.status_code, 200)

    def test_order_update_invalid_json(self):
        resp = self.client.post(
            "/webhook/order-update",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status_code, 200)

    def test_token_empty_returns_400(self):
        resp = self.client.post("/webhook/token", json={"access_token": "", "user_id": "U1"})
        self.assertEqual(resp.status_code, 400)

    def test_all_order_statuses_no_crash(self):
        for status in ("open", "complete", "rejected", "cancelled", "trigger pending"):
            resp = self.client.post("/webhook/order-update", json={
                "order_id": f"ORD_{status}", "status": status,
                "average_price": 100.0, "filled_quantity": 1,
            })
            self.assertEqual(resp.status_code, 200, f"Status '{status}' caused non-200")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Full Integration: 5 Symbols Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipelineIntegration(unittest.TestCase):

    def setUp(self):
        import live_bot.state as s_mod
        s_mod.state = make_fresh_state(500_000.0)
        self.state = s_mod.state
        import live_bot.feeds.market_feed as mf
        mf._KEY_TO_SYMBOL.update(INSTRUMENT_MAP)
        self.reg = CandleRegistry()
        from live_bot.orders.paper_broker import PaperBroker
        self.broker = PaperBroker()

    def _complete_candle(self, sym, key, price, min_from, min_to):
        t1 = make_tick(sym, key, price, minute=min_from, second=10)
        self.state.update_tick(sym, t1)
        self.reg.on_tick(sym, t1)
        t2 = make_tick(sym, key, price+1, minute=min_to, second=5)
        self.state.update_tick(sym, t2)
        return self.reg.on_tick(sym, t2) is not None

    def test_full_pipeline_5_symbols(self):
        # 1. Register with historical data
        for sym, price in zip(SYMBOLS, BASE_PRICES):
            self.reg.register(sym, seed_df=seed_df(price, n_bars=30))

        # 2. Parse feed message for 5 symbols
        data = {k: (s, p, c) for k, s, p, c in zip(KEYS, SYMBOLS, BASE_PRICES, PREV_CLOSES)}
        ticks = _parse_message(make_full_feed_message(data))
        self.assertEqual(len(ticks), 5)
        for tick in ticks:
            self.state.update_tick(tick.symbol, tick)
            # Verify candle OHLC non-zero after bug fix
            self.assertGreater(tick.candle_close, 0.0, f"candle_close=0 for {tick.symbol}")

        # 3. Simulate candle completions
        completed = sum(
            self._complete_candle(sym, key, price, 10+i, 11+i)
            for i, (sym, key, price) in enumerate(zip(SYMBOLS, KEYS, BASE_PRICES))
        )
        self.assertGreater(completed, 0)

        # 4. Place BUY orders for 3 symbols
        for sym, key, price in zip(SYMBOLS[:3], KEYS[:3], BASE_PRICES[:3]):
            self.broker.place_order(
                sym, key, "BUY", 5, "MARKET",
                stop_loss=price*0.98, take_profit=price*1.04, strategy_tag="INT"
            )
        self.assertEqual(len(self.state.get_all_positions()), 3)

        # 5. Price moves → trigger TP + SL
        self.state.update_tick("RELIANCE", make_tick("RELIANCE", KEYS[0], BASE_PRICES[0]*1.05, minute=20))
        self.broker.check_stop_loss_take_profit("RELIANCE")
        self.state.update_tick("TCS", make_tick("TCS", KEYS[1], BASE_PRICES[1]*0.97, minute=20))
        self.broker.check_stop_loss_take_profit("TCS")

        # 6. Squareoff remaining
        self.broker.squareoff_all()
        self.assertEqual(len(self.state.get_all_positions()), 0)

        # 7. Verify 3 closed trades
        trades = self.state.get_closed_trades()
        self.assertEqual(len(trades), 3)

        # 8. Print summary
        print(f"\n[INTEGRATION TEST] {len(trades)} trades across {len(SYMBOLS)} symbols")
        for t in trades:
            print(f"  {t.symbol:10s} {t.direction:5s} {t.exit_reason:12s} P&L=₹{t.pnl:,.2f}")

    def test_capital_tracking_5_symbols(self):
        """Capital tracking correct across buy + squareoff of all 5 symbols."""
        initial_cash = self.state.cash
        for sym, key, p in zip(SYMBOLS, KEYS, BASE_PRICES):
            self.state.update_tick(sym, make_tick(sym, key, p, minute=10))
            self.broker.place_order(sym, key, "BUY", 2, "MARKET", strategy_tag="CAP")
        self.assertLess(self.state.cash, initial_cash)
        self.broker.squareoff_all()
        diff_pct = abs(self.state.cash - initial_cash) / initial_cash * 100
        self.assertLess(diff_pct, 5.0, f"Capital diff {diff_pct:.2f}% too large")

    def test_feed_drives_candle_builder_multiple_minutes(self):
        """3 minutes of ticks → each symbol accumulates candles."""
        for sym, price in zip(SYMBOLS, BASE_PRICES):
            self.reg.register(sym, seed_df=seed_df(price, n_bars=5))
        for minute in range(3):
            for sym, key, price in zip(SYMBOLS, KEYS, BASE_PRICES):
                tick = make_tick(sym, key, price + minute*2, minute=10+minute)
                self.state.update_tick(sym, tick)
                self.reg.on_tick(sym, tick)
        for sym in SYMBOLS:
            self.assertGreaterEqual(self.reg.bar_counts()[sym], 5)


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestParseLtt, TestExtractOhlcFromFeed, TestParseMessage,
        TestMinuteCandle, TestCandleBuilder, TestCandleRegistry,
        TestLiveState, TestPortfolioFeedParsing,
        TestRiskGuard, TestPaperBroker,
        TestWebhookEndpoints, TestFullPipelineIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    print(f"\n{'='*60}")
    print(f"PHASE 7 TESTS: {total - failed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
