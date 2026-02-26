"""
data/universe.py
-----------------
Maintains the trading universe — the list of symbols your system actively tracks.

TWO UNIVERSES:
1. Nifty 500 — 500 largest stocks by market cap on NSE. Used for the screener.
2. F&O Universe — Stocks approved by SEBI for futures and options trading.
   This list changes periodically as SEBI adds/removes stocks.

WHY THIS EXISTS:
- You need a consistent, refreshable list of symbols for:
  a) Batch data download
  b) Daily screener scans
  c) Strategy backtesting across the whole universe

HOW WE GET THE LISTS:
- Nifty 500: Fetched from NSE's index page (monthly refresh recommended)
- F&O List: Fetched from NSE's F&O underlyings page (monthly refresh)
- Both are cached locally in the SQLite metadata database

USAGE:
    from data.universe import universe_manager
    nifty500 = universe_manager.get_nifty500()
    fo_stocks = universe_manager.get_fo_stocks()
"""

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import config

logger = logging.getLogger(__name__)

# Hardcoded Nifty 500 — used as fallback if live fetch fails
# In production, this gets overwritten by the live fetch
# This is a small representative sample for development
FALLBACK_NIFTY500_SAMPLE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BAJFINANCE", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "TITAN", "NESTLEIND", "WIPRO", "ULTRACEMCO", "HCLTECH",
    "TECHM", "POWERGRID", "NTPC", "ONGC", "COALINDIA",
    "TATAMOTORS", "TATASTEEL", "HINDALCO", "GRASIM", "DIVISLAB",
    "DRREDDY", "CIPLA", "APOLLOHOSP", "BAJAJFINSV", "ADANIPORTS",
    "JSWSTEEL", "INDUSINDBK", "BRITANNIA", "EICHERMOT", "HEROMOTOCO",
    "M&M", "TATACONSUM", "BPCL", "IOC", "SHREECEM",
    "PIDILITIND", "DABUR", "MCDOWELL-N", "GODREJCP", "HAVELLS",
]


class UniverseManager:
    """
    Manages and caches the NSE 500 and F&O trading universes.

    Data is stored in SQLite for persistence across restarts.
    The lists are refreshed monthly (or on demand).
    """

    def __init__(self):
        self.db_path = config.METADATA_DB
        self._ensure_tables()

    # ── Database Setup ────────────────────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        """Create a SQLite connection to the metadata database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path))

    def _ensure_tables(self):
        """Create universe tables if they don't exist yet."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nifty500 (
                    symbol      TEXT PRIMARY KEY,
                    exchange    TEXT DEFAULT 'NSE_EQ',
                    last_updated DATE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fo_universe (
                    symbol      TEXT PRIMARY KEY,
                    exchange    TEXT DEFAULT 'NSE_FO',
                    last_updated DATE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_holidays (
                    holiday_date    TEXT PRIMARY KEY,
                    description     TEXT,
                    holiday_type    TEXT,
                    closed_exchanges TEXT
                )
            """)
            conn.commit()

    # ── Nifty 500 Universe ────────────────────────────────────────────────────

    def get_nifty500(self, force_refresh: bool = False) -> list[dict]:
        """
        Get the Nifty 500 universe.

        Returns cached data if available and less than 30 days old.
        Forces a refresh if force_refresh=True or data is stale.

        Args:
            force_refresh (bool): If True, download fresh data regardless of cache.

        Returns:
            list[dict]: List of dicts with "exchange" and "symbol" keys.
                        e.g. [{"exchange": "NSE_EQ", "symbol": "INFY"}, ...]
        """
        if not force_refresh:
            cached = self._load_nifty500_from_db()
            if cached:
                return cached

        # Try to load from NSE (may fail if NSE website changes structure)
        try:
            fresh_data = self._fetch_nifty500_from_nse()
            if fresh_data:
                self._save_nifty500_to_db(fresh_data)
                logger.info(f"Nifty 500 updated: {len(fresh_data)} symbols.")
                return fresh_data
        except Exception as e:
            logger.warning(f"Could not fetch Nifty 500 from NSE: {e}")

        # Fall back to any data in DB (even if stale)
        cached = self._load_nifty500_from_db(ignore_age=True)
        if cached:
            logger.warning("Using stale Nifty 500 data from DB (live fetch failed).")
            return cached

        # Last resort: hardcoded sample
        logger.warning(
            "Using hardcoded sample universe (NSE fetch failed, no DB cache). "
            "Run universe_manager.refresh_nifty500() to update."
        )
        return [{"exchange": "NSE_EQ", "symbol": s} for s in FALLBACK_NIFTY500_SAMPLE]

    def _fetch_nifty500_from_nse(self) -> list[dict]:
        """
        Attempt to fetch Nifty 500 constituents from NSE India.

        NSE provides this data via their index API.
        The URL may need updating if NSE changes their API structure.

        Returns:
            list[dict]: Universe symbols or empty list if fetch failed.
        """
        # NSE Index API endpoint for Nifty 500
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }

        # NSE requires a cookie — first visit the main page
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()
        stocks = data.get("data", [])

        # Filter out the index row itself (symbol starts with "NIFTY")
        symbols = [
            {"exchange": "NSE_EQ", "symbol": item["symbol"]}
            for item in stocks
            if not item.get("symbol", "").startswith("NIFTY")
        ]

        return symbols

    def _save_nifty500_to_db(self, symbols: list[dict]):
        """Save Nifty 500 list to SQLite."""
        today = date.today().isoformat()
        with self._get_connection() as conn:
            conn.execute("DELETE FROM nifty500")
            conn.executemany(
                "INSERT INTO nifty500 (symbol, exchange, last_updated) VALUES (?, ?, ?)",
                [(s["symbol"], s.get("exchange", "NSE_EQ"), today) for s in symbols]
            )
            conn.commit()

    def _load_nifty500_from_db(self, ignore_age: bool = False) -> list[dict]:
        """
        Load Nifty 500 from SQLite cache.

        Args:
            ignore_age: If True, return data even if it's older than 30 days.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT symbol, exchange, last_updated FROM nifty500"
            ).fetchall()

        if not rows:
            return []

        if not ignore_age:
            # Check if data is fresh (less than 30 days old)
            last_updated = rows[0][2]
            if last_updated:
                days_old = (date.today() - date.fromisoformat(last_updated)).days
                if days_old > 30:
                    logger.info(f"Nifty 500 cache is {days_old} days old. Refreshing...")
                    return []

        return [{"symbol": row[0], "exchange": row[1]} for row in rows]

    # ── F&O Universe ──────────────────────────────────────────────────────────

    def get_fo_stocks(self, force_refresh: bool = False) -> list[dict]:
        """
        Get the list of stocks permitted for F&O trading on NSE.

        Args:
            force_refresh (bool): Force a fresh download.

        Returns:
            list[dict]: List of F&O eligible stocks.
        """
        if not force_refresh:
            cached = self._load_fo_from_db()
            if cached:
                return cached

        try:
            fresh_data = self._fetch_fo_from_nse()
            if fresh_data:
                self._save_fo_to_db(fresh_data)
                logger.info(f"F&O universe updated: {len(fresh_data)} stocks.")
                return fresh_data
        except Exception as e:
            logger.warning(f"Could not fetch F&O list from NSE: {e}")

        cached = self._load_fo_from_db(ignore_age=True)
        if cached:
            logger.warning("Using stale F&O universe from DB.")
            return cached

        logger.warning("No F&O universe available.")
        return []

    def _fetch_fo_from_nse(self) -> list[dict]:
        """Fetch F&O eligible stocks from NSE."""
        url = "https://www.nseindia.com/api/master-quote?type=derivative"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()
        # NSE returns a list of symbol strings for this endpoint
        if isinstance(data, list):
            return [{"exchange": "NSE_FO", "symbol": s} for s in data if isinstance(s, str)]

        return []

    def _save_fo_to_db(self, symbols: list[dict]):
        """Save F&O universe to SQLite."""
        today = date.today().isoformat()
        with self._get_connection() as conn:
            conn.execute("DELETE FROM fo_universe")
            conn.executemany(
                "INSERT INTO fo_universe (symbol, exchange, last_updated) VALUES (?, ?, ?)",
                [(s["symbol"], s.get("exchange", "NSE_FO"), today) for s in symbols]
            )
            conn.commit()

    def _load_fo_from_db(self, ignore_age: bool = False) -> list[dict]:
        """Load F&O universe from SQLite cache."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT symbol, exchange, last_updated FROM fo_universe"
            ).fetchall()

        if not rows:
            return []

        if not ignore_age and rows[0][2]:
            days_old = (date.today() - date.fromisoformat(rows[0][2])).days
            if days_old > 30:
                return []

        return [{"symbol": row[0], "exchange": row[1]} for row in rows]

    # ── Market Holidays ───────────────────────────────────────────────────────

    def save_market_holidays(self, holidays: list[dict]):
        """
        Save market holidays fetched from Upstox to local SQLite.

        Args:
            holidays (list[dict]): Holiday data from market_data.get_market_holidays()
        """
        with self._get_connection() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO market_holidays
                   (holiday_date, description, holiday_type, closed_exchanges)
                   VALUES (?, ?, ?, ?)""",
                [
                    (
                        h.get("date"),
                        h.get("description"),
                        h.get("holiday_type"),
                        json.dumps(h.get("closed_exchanges", []))
                    )
                    for h in holidays
                ]
            )
            conn.commit()
        logger.info(f"Saved {len(holidays)} market holidays to DB.")

    def is_nse_holiday(self, check_date: Optional[str] = None) -> bool:
        """
        Check if a date is an NSE trading holiday from local DB.

        Args:
            check_date (str, optional): Date in "YYYY-MM-DD". Defaults to today.

        Returns:
            bool: True if NSE is closed.
        """
        if check_date is None:
            check_date = date.today().isoformat()

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT closed_exchanges FROM market_holidays WHERE holiday_date = ?",
                (check_date,)
            ).fetchone()

        if not row:
            return False

        closed = json.loads(row[0]) if row[0] else []
        return "NSE" in closed or "NFO" in closed

    def get_nse_holidays_this_year(self) -> list[str]:
        """Return all NSE holiday dates for the current year."""
        current_year = str(date.today().year)

        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT holiday_date, closed_exchanges FROM market_holidays
                   WHERE holiday_date LIKE ?""",
                (f"{current_year}%",)
            ).fetchall()

        return [
            row[0] for row in rows
            if "NSE" in json.loads(row[1] or "[]")
        ]

    # ── Convenience Methods ───────────────────────────────────────────────────

    def refresh_all(self):
        """
        Refresh both Nifty 500 and F&O universes from live sources.
        Call this monthly via the scheduler.
        """
        logger.info("Refreshing all universe data...")
        self.get_nifty500(force_refresh=True)
        self.get_fo_stocks(force_refresh=True)
        logger.info("Universe refresh complete.")

    def get_universe_summary(self) -> dict:
        """Return a summary of current universe data for the dashboard."""
        nifty500 = self.get_nifty500()
        fo_stocks = self.get_fo_stocks()

        return {
            "nifty500_count": len(nifty500),
            "fo_stocks_count": len(fo_stocks),
            "nifty500_sample": [s["symbol"] for s in nifty500[:5]],
        }


# ── Module-level singleton ────────────────────────────────────────────────────
universe_manager = UniverseManager()
