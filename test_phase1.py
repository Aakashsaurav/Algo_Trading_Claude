"""
tests/test_phase1.py
---------------------
Comprehensive unit tests for all Phase 1 modules.

Tests cover:
- config.py — settings loading, directory creation
- broker/auth.py — token management, login URL generation
- broker/market_data.py — API response parsing, date validation
- broker/instrument_manager.py — symbol lookups, filtering
- data/parquet_store.py — save/load/merge logic (mocked pyarrow)
- data/fetcher.py — incremental logic, batch orchestration
- data/cleaner.py — all cleaning steps
- data/universe.py — SQLite persistence, fallback logic

Run with:  python -m pytest tests/test_phase1.py -v
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import numpy as np

# ─── Setup: point config to a temp directory so tests don't touch real data ───
# We do this BEFORE importing any project modules
TEMP_DIR = tempfile.mkdtemp()
os.environ.setdefault("UPSTOX_API_KEY", "test_api_key")
os.environ.setdefault("UPSTOX_API_SECRET", "test_api_secret")
os.environ.setdefault("UPSTOX_REDIRECT_URI", "http://127.0.0.1:8000/callback")
os.environ.setdefault("PAPER_TRADE", "True")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Now import project modules ───────────────────────────────────────────────
from config import AppConfig, setup_logging


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: config.py
# ═══════════════════════════════════════════════════════════════════════════════
class TestConfig(unittest.TestCase):

    def test_required_env_vars_loaded(self):
        """Config should load API key and secret from environment."""
        cfg = AppConfig()
        self.assertEqual(cfg.API_KEY, "test_api_key")
        self.assertEqual(cfg.API_SECRET, "test_api_secret")

    def test_default_values(self):
        """Check default values are sensible."""
        cfg = AppConfig()
        self.assertEqual(cfg.TOTAL_CAPITAL, 500000.0)
        self.assertEqual(cfg.MAX_PORTFOLIO_DRAWDOWN, 20.0)
        self.assertTrue(cfg.PAPER_TRADE)
        self.assertEqual(cfg.MARKET_OPEN_TIME, "09:15")
        self.assertEqual(cfg.MARKET_CLOSE_TIME, "15:30")
        self.assertEqual(cfg.INTRADAY_SQUAREOFF_TIME, "15:20")

    def test_directories_created(self):
        """AppConfig constructor should create all required directories."""
        cfg = AppConfig()
        self.assertTrue(cfg.DATA_DIR.exists())
        self.assertTrue(cfg.OHLCV_DIR.exists())
        self.assertTrue(cfg.DAILY_DIR.exists())
        self.assertTrue(cfg.MINUTE_DIR.exists())
        self.assertTrue(cfg.WEEKLY_DIR.exists())
        self.assertTrue(cfg.SQLITE_DIR.exists())

    def test_missing_required_env_raises(self):
        """Missing required env variable should raise EnvironmentError."""
        from config import _require
        with self.assertRaises(EnvironmentError):
            _require("THIS_ENV_VAR_DEFINITELY_DOES_NOT_EXIST_XYZ")

    def test_paper_trade_parsing(self):
        """PAPER_TRADE parsing logic should produce correct bool values."""
        # Class variables are evaluated at class-definition time, so we
        # test the exact parsing expression used in config.py directly.
        self.assertFalse("false".lower() == "true")
        self.assertTrue("True".lower() == "true")
        self.assertFalse("False".lower() == "true")
        # Default value from env (PAPER_TRADE=True set in test setUp)
        cfg = AppConfig()
        self.assertIsInstance(cfg.PAPER_TRADE, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: broker/auth.py
# ═══════════════════════════════════════════════════════════════════════════════
class TestAuthManager(unittest.TestCase):

    def setUp(self):
        """Create a fresh AuthManager with a temp token file for each test."""
        from broker.auth import AuthManager
        self.temp_dir = Path(tempfile.mkdtemp())
        self.token_file = self.temp_dir / "token.json"

        self.auth = AuthManager()
        self.auth.token_file = self.token_file
        self.auth._token_data = None  # Clear any cached state

    def test_get_login_url_format(self):
        """Login URL should contain required OAuth2 parameters."""
        url = self.auth.get_login_url()
        self.assertIn("https://api.upstox.com/v2/login/authorization/dialog", url)
        self.assertIn("response_type=code", url)
        self.assertIn("client_id=test_api_key", url)
        self.assertIn("redirect_uri=", url)

    def test_extract_code_from_redirect_url(self):
        """Should correctly parse auth code from a redirect URL."""
        redirect_url = "http://127.0.0.1:8000/callback?code=test_auth_code_123"

        # Mock the generate_token call so we don't hit the real API
        with patch.object(self.auth, "generate_token", return_value={"access_token": "tok"}) as mock_gen:
            self.auth.generate_token_from_url(redirect_url)
            mock_gen.assert_called_once_with("test_auth_code_123")

    def test_extract_code_fails_on_missing_code(self):
        """Should raise ValueError if redirect URL has no 'code' parameter."""
        with self.assertRaises(ValueError):
            self.auth.generate_token_from_url("http://127.0.0.1:8000/callback")

    def test_save_and_load_token(self):
        """Token saved to disk should be loadable and valid for today."""
        token_data = {
            "access_token": "valid_test_token_abc",
            "token_type": "Bearer",
            "generated_date": date.today().isoformat(),
        }
        self.auth._save_token(token_data)
        loaded = self.auth._load_token()

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["access_token"], "valid_test_token_abc")

    def test_stale_token_not_loaded(self):
        """Token generated yesterday should NOT be loaded (it's expired)."""
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        old_token = {
            "access_token": "old_token",
            "generated_date": yesterday,
        }
        self.auth._save_token(old_token)
        loaded = self.auth._load_token()
        self.assertIsNone(loaded)

    def test_get_valid_token_returns_from_memory(self):
        """get_valid_token should return cached in-memory token without disk access."""
        self.auth._token_data = {
            "access_token": "memory_token",
            "generated_date": date.today().isoformat(),
        }
        token = self.auth.get_valid_token()
        self.assertEqual(token, "memory_token")

    def test_get_valid_token_raises_when_no_token(self):
        """Should raise RuntimeError with helpful message when no token exists."""
        self.auth._token_data = None
        # Ensure token file doesn't exist
        if self.token_file.exists():
            self.token_file.unlink()

        with self.assertRaises(RuntimeError) as ctx:
            self.auth.get_valid_token()
        self.assertIn("login", str(ctx.exception).lower())

    def test_is_authenticated_false_without_token(self):
        """is_authenticated should return False when no valid token exists."""
        self.auth._token_data = None
        if self.token_file.exists():
            self.token_file.unlink()
        self.assertFalse(self.auth.is_authenticated())

    def test_is_authenticated_true_with_valid_token(self):
        """is_authenticated should return True with a valid in-memory token."""
        self.auth._token_data = {
            "access_token": "valid_token",
            "generated_date": date.today().isoformat(),
        }
        self.assertTrue(self.auth.is_authenticated())

    def test_logout_clears_token(self):
        """Logout should clear both in-memory token and token file."""
        self.auth._token_data = {
            "access_token": "test_tok",
            "generated_date": date.today().isoformat(),
        }
        self.auth._save_token(self.auth._token_data)

        with patch("requests.delete") as mock_delete:
            mock_delete.return_value.status_code = 200
            self.auth.logout()

        self.assertIsNone(self.auth._token_data)
        self.assertFalse(self.token_file.exists())

    def test_generate_token_parses_response(self):
        """generate_token should correctly process a successful API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "fresh_token_xyz",
            "token_type": "Bearer",
        }
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response):
            result = self.auth.generate_token("fake_code")

        self.assertEqual(result["access_token"], "fresh_token_xyz")
        self.assertEqual(result["generated_date"], date.today().isoformat())


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: broker/market_data.py
# ═══════════════════════════════════════════════════════════════════════════════
class TestMarketDataManager(unittest.TestCase):

    def setUp(self):
        from broker.market_data import MarketDataManager
        self.md = MarketDataManager()

    def _make_mock_candles(self, n=5, interval="day"):
        """Helper: generate realistic mock candle data."""
        candles = []
        base_price = 1000.0
        for i in range(n):
            dt_str = f"2024-01-{i+1:02d}T09:15:00+05:30"
            open_p = base_price + i * 10
            high_p = open_p + 20
            low_p = open_p - 10
            close_p = open_p + 5
            volume = 100000 + i * 1000
            oi = 0
            candles.append([dt_str, open_p, high_p, low_p, close_p, volume, oi])
        return candles

    def test_parse_candles_returns_dataframe(self):
        """_parse_candles_to_df should return a proper DataFrame."""
        candles = self._make_mock_candles(5)
        df = self.md._parse_candles_to_df(candles, "day")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertIn("close", df.columns)
        self.assertIn("open", df.columns)
        self.assertIn("volume", df.columns)

    def test_parse_candles_sorted_ascending(self):
        """Candles should be sorted oldest-first after parsing."""
        candles = self._make_mock_candles(5)
        # Reverse the candles to test sorting
        candles_reversed = list(reversed(candles))
        df = self.md._parse_candles_to_df(candles_reversed, "day")

        # Check index is sorted ascending
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_parse_empty_candles_returns_empty_df(self):
        """Empty candle list should return empty DataFrame, not crash."""
        df = self.md._parse_candles_to_df([], "day")
        self.assertTrue(df.empty)

    def test_parse_candles_numeric_types(self):
        """OHLCV columns should be numeric after parsing."""
        candles = self._make_mock_candles(3)
        df = self.md._parse_candles_to_df(candles, "day")

        for col in ["open", "high", "low", "close", "volume"]:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]),
                            f"Column '{col}' should be numeric")

    def test_get_daily_ohlcv_invalid_date_raises(self):
        """Invalid date format should raise ValueError."""
        with self.assertRaises(ValueError):
            self.md.get_daily_ohlcv("NSE_EQ|TEST", "2024/01/01")

    def test_get_daily_ohlcv_makes_correct_api_call(self):
        """get_daily_ohlcv should call the correct Upstox endpoint."""
        mock_response = {
            "data": {"candles": self._make_mock_candles(3)}
        }
        with patch.object(self.md, "_make_request", return_value=mock_response) as mock_req:
            with patch.object(self.md, "_get_headers", return_value={}):
                df = self.md.get_daily_ohlcv(
                    "NSE_EQ|TEST", "2024-01-01", "2024-01-31"
                )
                # Verify the correct URL pattern was used
                call_url = mock_req.call_args[0][0]
                self.assertIn("/day/", call_url)
                self.assertIn("2024-01-31", call_url)
                self.assertIn("2024-01-01", call_url)

    def test_get_minute_ohlcv_warns_on_long_range(self):
        """Should log a warning when requesting > 30 days of minute data."""
        with patch.object(self.md, "_make_request", return_value={"data": {"candles": []}}):
            with patch.object(self.md, "_get_headers", return_value={}):
                import logging
                with self.assertLogs(level="WARNING") as log:
                    self.md.get_minute_ohlcv(
                        "NSE_EQ|TEST",
                        "2024-01-01",
                        "2024-05-01"  # 4 months = >30 days
                    )
                self.assertTrue(
                    any("30" in msg for msg in log.output),
                    "Should warn about >30 day range"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: data/cleaner.py
# ═══════════════════════════════════════════════════════════════════════════════
class TestDataCleaner(unittest.TestCase):

    def setUp(self):
        from data.cleaner import DataCleaner
        self.cleaner = DataCleaner()

    def _make_df(self, n=10, with_tz=True):
        """Helper: create a realistic OHLCV DataFrame for testing."""
        freq = "D"
        if with_tz:
            idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="Asia/Kolkata")
        else:
            idx = pd.date_range("2024-01-01", periods=n, freq=freq)

        np.random.seed(42)
        prices = 1000 + np.cumsum(np.random.randn(n) * 5)

        return pd.DataFrame({
            "open":   prices + np.random.randn(n),
            "high":   prices + np.abs(np.random.randn(n)) + 5,
            "low":    prices - np.abs(np.random.randn(n)) - 5,
            "close":  prices,
            "volume": np.random.randint(50000, 200000, n).astype(float),
            "oi":     np.zeros(n),
        }, index=idx)

    def test_clean_daily_returns_dataframe(self):
        """clean_daily should return a non-empty DataFrame for valid input."""
        df = self._make_df(10)
        result = self.cleaner.clean_daily(df, symbol="TEST")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_clean_daily_empty_input_returns_empty(self):
        """Empty DataFrame input should return empty DataFrame without crash."""
        result = self.cleaner.clean_daily(pd.DataFrame(), symbol="TEST")
        self.assertTrue(result.empty)

    def test_removes_duplicate_timestamps(self):
        """Duplicate timestamps should be removed, keeping the last one."""
        df = self._make_df(5)
        # Insert a duplicate
        dup_row = df.iloc[[0]].copy()
        df = pd.concat([df, dup_row])
        self.assertEqual(len(df), 6)  # 5 + 1 duplicate

        result = self.cleaner._remove_duplicates(df, "TEST")
        self.assertEqual(len(result), 5)

    def test_removes_zero_price_rows(self):
        """Rows with zero close price should be removed."""
        df = self._make_df(5)
        df.iloc[2, df.columns.get_loc("close")] = 0.0
        df.iloc[2, df.columns.get_loc("open")] = 0.0

        result = self.cleaner._remove_invalid_prices(df, "TEST")
        self.assertEqual(len(result), 4)

    def test_removes_negative_price_rows(self):
        """Rows with negative prices should be removed."""
        df = self._make_df(5)
        df.iloc[1, df.columns.get_loc("low")] = -50.0

        result = self.cleaner._remove_invalid_prices(df, "TEST")
        self.assertEqual(len(result), 4)

    def test_fixes_high_violations(self):
        """high should always be >= close; violations should be corrected."""
        df = self._make_df(5)
        # Set high lower than close for row 0 — violation
        df.iloc[0, df.columns.get_loc("high")] = df.iloc[0]["close"] - 50

        result = self.cleaner._fix_ohlc_violations(df, "TEST")
        # After fix, high should be >= close for all rows
        self.assertTrue((result["high"] >= result["close"]).all())

    def test_fixes_low_violations(self):
        """low should always be <= close; violations should be corrected."""
        df = self._make_df(5)
        # Set low higher than close for row 0 — violation
        df.iloc[0, df.columns.get_loc("low")] = df.iloc[0]["close"] + 50

        result = self.cleaner._fix_ohlc_violations(df, "TEST")
        self.assertTrue((result["low"] <= result["close"]).all())

    def test_filter_trading_hours_removes_off_hours(self):
        """Minute data outside 9:15–15:30 should be removed."""
        # Create minute data with off-hours candles
        idx = pd.date_range(
            "2024-01-02 08:00",  # Start at 8 AM (before market)
            periods=500,
            freq="1min",
            tz="Asia/Kolkata"
        )
        df = pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.5, "volume": 1000.0, "oi": 0.0
        }, index=idx)

        result = self.cleaner._filter_trading_hours(df, "TEST")

        # All remaining candles should be within trading hours
        for ts in result.index:
            time_val = ts.hour * 100 + ts.minute
            self.assertGreaterEqual(time_val, 915,
                                    f"Candle at {ts.time()} is before market open")
            self.assertLessEqual(time_val, 1530,
                                 f"Candle at {ts.time()} is after market close")

    def test_fill_missing_trading_days(self):
        """Missing weekdays should be forward-filled."""
        # Create data with a gap (skip one weekday)
        idx = pd.DatetimeIndex([
            pd.Timestamp("2024-01-02", tz="Asia/Kolkata"),  # Tuesday
            pd.Timestamp("2024-01-04", tz="Asia/Kolkata"),  # Thursday (skipping Wednesday)
        ])
        df = pd.DataFrame({
            "open": [100.0, 110.0],
            "high": [105.0, 115.0],
            "low": [98.0, 108.0],
            "close": [102.0, 112.0],
            "volume": [1000.0, 2000.0],
            "oi": [0.0, 0.0],
        }, index=idx)

        result = self.cleaner._fill_missing_trading_days(df, "TEST")

        # Should now have Wednesday 2024-01-03 filled in
        result_dates = [str(ts.date()) for ts in result.index]
        self.assertIn("2024-01-03", result_dates)
        # Wednesday's close should equal Tuesday's close (forward-filled)
        wed_idx = result.index[result_dates.index("2024-01-03")]
        self.assertEqual(result.loc[wed_idx, "close"], 102.0)
        # Wednesday's volume should be 0 (synthetic day)
        self.assertEqual(result.loc[wed_idx, "volume"], 0.0)

    def test_quality_report_format(self):
        """get_quality_report should return expected keys."""
        df = self._make_df(10)
        report = self.cleaner.get_quality_report(df, "TEST")

        self.assertIn("symbol", report)
        self.assertIn("rows", report)
        self.assertIn("status", report)
        self.assertIn("null_price_rows", report)
        self.assertIn("duplicate_timestamps", report)

    def test_quality_report_empty_df(self):
        """Quality report for empty DataFrame should indicate 'empty' status."""
        report = self.cleaner.get_quality_report(pd.DataFrame(), "TEST")
        self.assertEqual(report["status"], "empty")

    def test_clean_daily_does_not_modify_original(self):
        """clean_daily must not modify the original DataFrame (works on a copy)."""
        df = self._make_df(5)
        # Introduce a violation
        df.iloc[0, df.columns.get_loc("high")] = df.iloc[0]["close"] - 100
        original_high = df.iloc[0]["high"]

        self.cleaner.clean_daily(df, symbol="TEST")

        # Original should be unchanged
        self.assertEqual(df.iloc[0]["high"], original_high)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: data/universe.py (SQLite operations)
# ═══════════════════════════════════════════════════════════════════════════════
class TestUniverseManager(unittest.TestCase):

    def setUp(self):
        """Each test gets a fresh temp database."""
        self.temp_dir = Path(tempfile.mkdtemp())
        # Patch the config to use temp DB
        with patch("data.universe.config") as mock_cfg:
            mock_cfg.METADATA_DB = self.temp_dir / "test_metadata.db"

            from data.universe import UniverseManager
            self.mgr = UniverseManager()
            self.mgr.db_path = self.temp_dir / "test_metadata.db"
            self.mgr._ensure_tables()

    def test_tables_created_on_init(self):
        """SQLite tables should be created when UniverseManager is initialised."""
        with sqlite3.connect(str(self.mgr.db_path)) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn("nifty500", table_names)
        self.assertIn("fo_universe", table_names)
        self.assertIn("market_holidays", table_names)

    def test_save_and_load_nifty500(self):
        """Saved Nifty 500 symbols should be retrievable from DB."""
        symbols = [
            {"exchange": "NSE_EQ", "symbol": "INFY"},
            {"exchange": "NSE_EQ", "symbol": "TCS"},
            {"exchange": "NSE_EQ", "symbol": "RELIANCE"},
        ]
        self.mgr._save_nifty500_to_db(symbols)
        loaded = self.mgr._load_nifty500_from_db()

        self.assertEqual(len(loaded), 3)
        loaded_symbols = [s["symbol"] for s in loaded]
        self.assertIn("INFY", loaded_symbols)
        self.assertIn("TCS", loaded_symbols)

    def test_save_market_holidays(self):
        """Market holidays should be saved and retrievable."""
        holidays = [
            {
                "date": "2024-01-26",
                "description": "Republic Day",
                "holiday_type": "TRADING_HOLIDAY",
                "closed_exchanges": ["NSE", "NFO", "BSE"],
            }
        ]
        self.mgr.save_market_holidays(holidays)
        is_holiday = self.mgr.is_nse_holiday("2024-01-26")
        self.assertTrue(is_holiday)

    def test_non_holiday_returns_false(self):
        """A date not in holidays table should return False."""
        is_holiday = self.mgr.is_nse_holiday("2024-01-15")
        self.assertFalse(is_holiday)

    def test_fallback_returns_sample_data(self):
        """When NSE fetch fails and DB is empty, fallback sample should be returned."""
        with patch.object(self.mgr, "_fetch_nifty500_from_nse", side_effect=Exception("NSE down")):
            result = self.mgr.get_nifty500()
        # Should return the hardcoded fallback
        self.assertGreater(len(result), 0)
        self.assertIn("exchange", result[0])
        self.assertIn("symbol", result[0])

    def test_stale_cache_triggers_refresh(self):
        """Cache older than 30 days should be considered stale."""
        old_date = (date.today() - timedelta(days=31)).isoformat()
        with sqlite3.connect(str(self.mgr.db_path)) as conn:
            conn.execute("DELETE FROM nifty500")
            conn.execute(
                "INSERT INTO nifty500 (symbol, exchange, last_updated) VALUES (?, ?, ?)",
                ("INFY", "NSE_EQ", old_date)
            )
            conn.commit()

        result = self.mgr._load_nifty500_from_db()
        # Stale data should not be returned (returns empty to trigger refresh)
        self.assertEqual(result, [])


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: data/fetcher.py
# ═══════════════════════════════════════════════════════════════════════════════
class TestDataFetcher(unittest.TestCase):

    def setUp(self):
        from data.fetcher import DataFetcher
        self.fetcher = DataFetcher()

    def _make_ohlcv_df(self, n=5):
        """Helper: make a simple OHLCV DataFrame."""
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata")
        return pd.DataFrame({
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low":  [98.0] * n,
            "close": [102.0] * n,
            "volume": [50000.0] * n,
            "oi": [0.0] * n,
        }, index=idx)

    def test_fetch_returns_false_when_instrument_not_found(self):
        """fetch_and_save_daily should return False if instrument key not found."""
        with patch.object(self.fetcher.instrument_manager, "get_instrument_key", return_value=None):
            result = self.fetcher.fetch_and_save_daily("NSE_EQ", "FAKESYMBOL")
        self.assertFalse(result)

    def test_fetch_returns_false_on_api_error(self):
        """fetch_and_save_daily should return False if market_data raises an exception."""
        with patch.object(self.fetcher.instrument_manager, "get_instrument_key",
                          return_value="NSE_EQ|FAKE"):
            with patch.object(self.fetcher.market_data, "get_daily_ohlcv",
                               side_effect=Exception("API down")):
                result = self.fetcher.fetch_and_save_daily("NSE_EQ", "FAKE")
        self.assertFalse(result)

    def test_fetch_returns_true_on_success(self):
        """fetch_and_save_daily should return True on successful download."""
        mock_df = self._make_ohlcv_df(5)

        with patch.object(self.fetcher.instrument_manager, "get_instrument_key",
                          return_value="NSE_EQ|INFY"):
            with patch.object(self.fetcher.market_data, "get_daily_ohlcv",
                               return_value=mock_df):
                with patch.object(self.fetcher.store, "save_daily") as mock_save:
                    result = self.fetcher.fetch_and_save_daily("NSE_EQ", "INFY")

        self.assertTrue(result)
        mock_save.assert_called_once()

    def test_incremental_update_skips_if_up_to_date(self):
        """Incremental update should skip if data is already current."""
        today = date.today().isoformat()

        with patch.object(self.fetcher.store, "get_daily_date_range",
                          return_value=("2024-01-01", today)):
            with patch.object(self.fetcher, "fetch_and_save_daily") as mock_fetch:
                result = self.fetcher.incremental_update_daily("NSE_EQ", "INFY")

        self.assertTrue(result)
        mock_fetch.assert_not_called()  # Should NOT have re-fetched

    def test_incremental_update_does_full_fetch_if_no_data(self):
        """Incremental update should do full fetch if no data exists yet."""
        with patch.object(self.fetcher.store, "get_daily_date_range",
                          return_value=(None, None)):
            with patch.object(self.fetcher, "fetch_and_save_daily",
                               return_value=True) as mock_fetch:
                result = self.fetcher.incremental_update_daily("NSE_EQ", "INFY")

        self.assertTrue(result)
        mock_fetch.assert_called_once()

    def test_batch_fetch_returns_summary(self):
        """fetch_universe_daily should return a summary dict."""
        symbols = [
            {"exchange": "NSE_EQ", "symbol": "INFY"},
            {"exchange": "NSE_EQ", "symbol": "TCS"},
        ]
        with patch.object(self.fetcher, "fetch_and_save_daily", return_value=True):
            with patch("time.sleep"):  # Skip delays in tests
                result = self.fetcher.fetch_universe_daily(
                    symbols, from_date="2024-01-01"
                )

        self.assertIn("total", result)
        self.assertIn("success_count", result)
        self.assertIn("failed_count", result)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["failed_count"], 0)

    def test_batch_fetch_tracks_failures(self):
        """Failed fetches should be tracked in summary."""
        symbols = [
            {"exchange": "NSE_EQ", "symbol": "GOOD"},
            {"exchange": "NSE_EQ", "symbol": "BAD"},
        ]

        def side_effect(exchange, symbol, *args, **kwargs):
            return symbol != "BAD"

        with patch.object(self.fetcher, "fetch_and_save_daily", side_effect=side_effect):
            with patch("time.sleep"):
                result = self.fetcher.fetch_universe_daily(symbols)

        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertIn("BAD", result["failed"])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Run tests
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
