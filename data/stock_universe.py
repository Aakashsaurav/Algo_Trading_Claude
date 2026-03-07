"""
data/stock_universe.py
-----------------------
#Not fetching actual data

Maintains enriched metadata for Nifty 500 stocks — including company name, 
market cap, sector, and industry classification.

WHY THIS EXISTS:
- You need detailed information about stocks for:
  a) Sector-based filtering and analysis
  b) Market cap-weighted portfolio construction
  c) Industry-specific strategy backtesting
  d) Risk management and concentration tracking
  e) Fundamental-driven screeners

DATA ATTRIBUTES PER STOCK:
  symbol      : Trading symbol (e.g. "INFY", "RELIANCE")
  name        : Company name (e.g. "Infosys Limited", "Reliance Industries")
  market_cap  : Market cap in INR (numeric, in billions)
  sector      : Sector classification (e.g. "IT", "Energy", "Banking")
  industry    : Industry classification (e.g. "Software", "Oil & Gas", "Private Sector Banks")

HOW WE GET THE DATA:
- Nifty 500 constituents: Fetched from NSE's index API
- Company metadata: Fetched from NSE's equity infopage or similar endpoints
- All are cached locally in SQLite metadata database
- Cache is refreshed monthly (or on demand)

FALLBACK DATA:
- If live fetch fails, falls back to hardcoded sample data for development
- Stale data in DB is used if live fetch fails

USAGE:
    from data.stock_universe import stock_universe_manager
    
    # Get all Nifty 500 stocks with enriched data
    stocks = stock_universe_manager.get_nifty500_detailed()
    # Returns: [
    #   {
    #     "symbol": "INFY",
    #     "name": "Infosys Limited",
    #     "market_cap": 780.5,
    #     "sector": "IT",
    #     "industry": "Software"
    #   },
    #   ...
    # ]
    
    # Force refresh from live NSE data
    stocks = stock_universe_manager.get_nifty500_detailed(force_refresh=True)
    
    # Filter by sector
    banking_stocks = stock_universe_manager.get_by_sector("Banking")
    
    # Get top stocks by market cap
    top_10 = stock_universe_manager.get_by_market_cap(limit=10)
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

# Hardcoded Nifty 500 sample with enriched metadata — used as fallback
# This is a representative sample for development/testing
FALLBACK_NIFTY500_DETAILED = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Limited", "market_cap": 2150.5, "sector": "Energy", "industry": "Oil & Gas"},
    {"symbol": "TCS", "name": "Tata Consultancy Services Limited", "market_cap": 1450.2, "sector": "IT", "industry": "Software"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Limited", "market_cap": 1380.8, "sector": "Banking", "industry": "Private Sector Banks"},
    {"symbol": "INFY", "name": "Infosys Limited", "market_cap": 780.5, "sector": "IT", "industry": "Software"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank Limited", "market_cap": 850.3, "sector": "Banking", "industry": "Private Sector Banks"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Limited", "market_cap": 620.1, "sector": "FMCG", "industry": "Personal Care"},
    {"symbol": "SBIN", "name": "State Bank of India", "market_cap": 720.9, "sector": "Banking", "industry": "Public Sector Banks"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance Limited", "market_cap": 580.4, "sector": "NBFC", "industry": "Finance"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Limited", "market_cap": 550.2, "sector": "Telecom", "industry": "Telecom Services"},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank Limited", "market_cap": 540.8, "sector": "Banking", "industry": "Private Sector Banks"},
    {"symbol": "LT", "name": "Larsen & Toubro Limited", "market_cap": 520.6, "sector": "Engineering", "industry": "Heavy Engineering"},
    {"symbol": "AXISBANK", "name": "Axis Bank Limited", "market_cap": 490.3, "sector": "Banking", "industry": "Private Sector Banks"},
    {"symbol": "ASIANPAINT", "name": "Asian Paints (India) Limited", "market_cap": 450.7, "sector": "Chemicals", "industry": "Paints"},
    {"symbol": "MARUTI", "name": "Maruti Suzuki India Limited", "market_cap": 480.5, "sector": "Auto", "industry": "Automobiles"},
    {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries Limited", "market_cap": 420.2, "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "TITAN", "name": "Titan Company Limited", "market_cap": 380.9, "sector": "Consumer", "industry": "Retail & Watches"},
    {"symbol": "NESTLEIND", "name": "Nestle India Limited", "market_cap": 350.4, "sector": "FMCG", "industry": "Food Products"},
    {"symbol": "WIPRO", "name": "Wipro Limited", "market_cap": 340.8, "sector": "IT", "industry": "Software"},
    {"symbol": "ULTRACEMCO", "name": "UltraTech Cement Limited", "market_cap": 360.5, "sector": "Materials", "industry": "Cement"},
    {"symbol": "HCLTECH", "name": "HCL Technologies Limited", "market_cap": 320.3, "sector": "IT", "industry": "Software"},
]


class StockUniverseManager:
    """
    Manages enriched stock metadata for Nifty 500 — company names, market caps,
    sectors, and industries.

    Data is stored in SQLite for persistence across restarts.
    The list is refreshed monthly (or on demand).
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
        """Create stock metadata tables if they don't exist yet."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nifty500_detailed (
                    symbol          TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    market_cap      REAL,
                    sector          TEXT,
                    industry        TEXT,
                    last_updated    DATE
                )
            """)
            conn.commit()
        logger.debug("Stock universe tables ensured.")

    # ── Main Entry Point: Nifty 500 Detailed ──────────────────────────────────

    def get_nifty500_detailed(self, force_refresh: bool = False) -> list[dict]:
        """
        Get the Nifty 500 universe with enriched metadata.

        Returns cached data if available and less than 30 days old.
        Forces a refresh if force_refresh=True or data is stale.

        Args:
            force_refresh (bool): If True, download fresh data regardless of cache.

        Returns:
            list[dict]: List of dicts with keys:
                        ["symbol", "name", "market_cap", "sector", "industry"]
                        e.g. [
                            {
                                "symbol": "INFY",
                                "name": "Infosys Limited",
                                "market_cap": 780.5,
                                "sector": "IT",
                                "industry": "Software"
                            },
                            ...
                        ]
        """
        if not force_refresh:
            cached = self._load_from_db()
            if cached:
                logger.info(f"Loaded {len(cached)} stocks from cache.")
                return cached

        # Try to fetch fresh data from NSE
        try:
            fresh_data = self._fetch_nifty500_detailed_from_nse()
            if fresh_data:
                self._save_to_db(fresh_data)
                logger.info(f"Nifty 500 detailed metadata updated: {len(fresh_data)} stocks.")
                return fresh_data
        except Exception as e:
            logger.warning(f"Could not fetch Nifty 500 detailed data from NSE: {e}")

        # Fall back to any data in DB (even if stale)
        cached = self._load_from_db(ignore_age=True)
        if cached:
            logger.warning(
                f"Using stale cache ({len(cached)} stocks). "
                "Run stock_universe_manager.refresh() to update from live NSE."
            )
            return cached

        # Last resort: hardcoded sample
        logger.warning(
            "Using hardcoded sample data (NSE fetch failed, no DB cache). "
            "Run stock_universe_manager.refresh() to update."
        )
        return FALLBACK_NIFTY500_DETAILED

    def _fetch_nifty500_detailed_from_nse(self) -> list[dict]:
        """
        Fetch Nifty 500 constituents with enriched metadata from NSE.

        Strategy:
        1. Fetch Nifty 500 symbol list from NSE index API
        2. For each symbol, fetch company info from NSE equity infopage
           (name, market cap, sector, industry)
        3. Return consolidated list

        Returns:
            list[dict]: List of stocks with full metadata, or empty list if fetch fails.
        """
        try:
            # Step 1: Get symbol list
            symbols = self._fetch_nifty500_symbols()
            if not symbols:
                logger.warning("Could not fetch Nifty 500 symbol list.")
                return []

            logger.info(f"Fetched {len(symbols)} symbols. Enriching with metadata...")

            # Step 2: Enrich each symbol with metadata
            detailed_stocks = []
            session = requests.Session()
            session.get("https://www.nseindia.com", timeout=10)  # Set cookie

            for i, symbol in enumerate(symbols, 1):
                try:
                    stock_info = self._fetch_stock_info(session, symbol)
                    if stock_info:
                        detailed_stocks.append(stock_info)
                        if i % 50 == 0:
                            logger.debug(f"Enriched {i}/{len(symbols)} symbols...")
                except Exception as e:
                    logger.debug(f"Skipped {symbol}: {e}")
                    continue

            logger.info(f"Successfully enriched {len(detailed_stocks)} stocks.")
            return detailed_stocks

        except Exception as e:
            logger.error(f"Error fetching Nifty 500 detailed data: {e}")
            return []

    def _fetch_nifty500_symbols(self) -> list[str]:
        """
        Fetch just the symbol list from NSE Nifty 500 index.

        Returns:
            list[str]: List of trading symbols, e.g. ["INFY", "TCS", ...]
        """
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
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
        stocks = data.get("data", [])

        # Extract symbols, excluding NIFTY index row
        symbols = [
            item["symbol"]
            for item in stocks
            if not item.get("symbol", "").startswith("NIFTY")
        ]

        return symbols

    def _fetch_stock_info(self, session: requests.Session, symbol: str) -> Optional[dict]:
        """
        Fetch company metadata for a single stock from NSE.

        Attempts to extract: name, market cap, sector, industry.

        Args:
            session (requests.Session): Persistent session with NSE cookies
            symbol (str): Trading symbol (e.g. "INFY")

        Returns:
            dict: Stock info with keys ["symbol", "name", "market_cap", "sector", "industry"]
                  or None if fetch/parse fails.
        """
        try:
            # NSE quote API endpoint
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/",
            }

            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            info = data.get("data", {})

            # Extract fields (NSE API may return different keys depending on the endpoint)
            name = info.get("companyName") or info.get("name") or symbol
            market_cap = info.get("marketCap", 0) or self._parse_market_cap(info.get("mktCap", ""))
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")

            return {
                "symbol": symbol,
                "name": name,
                "market_cap": market_cap,
                "sector": sector,
                "industry": industry,
            }

        except Exception as e:
            logger.debug(f"Failed to fetch info for {symbol}: {e}")
            return None

    def _parse_market_cap(self, market_cap_str: str) -> float:
        """
        Parse market cap string into numeric value (in billions).

        Examples:
          "₹7,80,000 Cr" -> 780000 (crores, keep as-is)
          "7.8L Cr" -> 7800000 crores
          "780 Bn" -> 780000 crores (1 Bn = 1000 Cr)
        """
        if not market_cap_str or not isinstance(market_cap_str, str):
            return 0.0

        try:
            # Remove currency symbol and whitespace
            clean = market_cap_str.replace("₹", "").replace(",", "").strip()

            # Check for suffix
            if "Cr" in clean or "Crore" in clean:
                value = float(clean.replace("Cr", "").replace("Crore", "").strip())
                return value / 1000  # Convert to billions for storage
            elif "Bn" in clean or "Billion" in clean:
                value = float(clean.replace("Bn", "").replace("Billion", "").strip())
                return value * 100  # 1 Bn = 100 Cr, store in billions
            else:
                # Assume it's already in the intended unit
                return float(clean)

        except (ValueError, AttributeError):
            return 0.0

    # ── Database Operations ───────────────────────────────────────────────────

    def _save_to_db(self, stocks: list[dict]):
        """Save enriched stock list to SQLite."""
        today = date.today().isoformat()
        with self._get_connection() as conn:
            conn.execute("DELETE FROM nifty500_detailed")
            conn.executemany(
                """INSERT INTO nifty500_detailed 
                   (symbol, name, market_cap, sector, industry, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        s["symbol"],
                        s.get("name", ""),
                        s.get("market_cap", 0),
                        s.get("sector", "Unknown"),
                        s.get("industry", "Unknown"),
                        today,
                    )
                    for s in stocks
                ]
            )
            conn.commit()
        logger.info(f"Saved {len(stocks)} stocks to DB.")

    def _load_from_db(self, ignore_age: bool = False) -> list[dict]:
        """
        Load enriched stock list from SQLite cache.

        Args:
            ignore_age: If True, return data even if older than 30 days.

        Returns:
            list[dict]: List of stock dicts, or empty list if no cache.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT symbol, name, market_cap, sector, industry, last_updated
                   FROM nifty500_detailed"""
            ).fetchall()

        if not rows:
            return []

        # Check freshness (default: 30 days)
        if not ignore_age:
            last_updated = rows[0][5]
            if last_updated:
                days_old = (date.today() - date.fromisoformat(last_updated)).days
                if days_old > 30:
                    logger.info(f"Stock cache is {days_old} days old. Refresh needed.")
                    return []

        return [
            {
                "symbol": row[0],
                "name": row[1],
                "market_cap": row[2],
                "sector": row[3],
                "industry": row[4],
            }
            for row in rows
        ]

    # ── Query Methods ────────────────────────────────────────────────────────

    def get_by_sector(self, sector: str) -> list[dict]:
        """
        Get all stocks in a specific sector.

        Args:
            sector (str): Sector name (e.g. "IT", "Banking", "Energy")

        Returns:
            list[dict]: Filtered list of stocks.
        """
        stocks = self.get_nifty500_detailed()
        return [s for s in stocks if s.get("sector", "").lower() == sector.lower()]

    def get_by_market_cap(
        self, limit: Optional[int] = None, reverse: bool = True
    ) -> list[dict]:
        """
        Get stocks sorted by market cap.

        Args:
            limit (int, optional): Number of top stocks to return. None = all.
            reverse (bool): If True, sort descending (largest first). Default True.

        Returns:
            list[dict]: Sorted list of stocks.
        """
        stocks = self.get_nifty500_detailed()
        sorted_stocks = sorted(
            stocks,
            key=lambda s: s.get("market_cap", 0),
            reverse=reverse,
        )
        return sorted_stocks[:limit] if limit else sorted_stocks

    def get_sectors(self) -> list[str]:
        """Return unique list of all sectors in Nifty 500."""
        stocks = self.get_nifty500_detailed()
        sectors = set(s.get("sector", "Unknown") for s in stocks)
        return sorted(list(sectors))

    def get_industries(self) -> list[str]:
        """Return unique list of all industries in Nifty 500."""
        stocks = self.get_nifty500_detailed()
        industries = set(s.get("industry", "Unknown") for s in stocks)
        return sorted(list(industries))

    def get_summary(self) -> dict:
        """Return a summary of the stock universe."""
        stocks = self.get_nifty500_detailed()
        total_market_cap = sum(s.get("market_cap", 0) for s in stocks)

        return {
            "total_stocks": len(stocks),
            "total_market_cap_billions": round(total_market_cap, 2),
            "sectors": self.get_sectors(),
            "industries": len(self.get_industries()),
            "top_5_by_market_cap": [
                {
                    "symbol": s["symbol"],
                    "name": s["name"],
                    "market_cap": s["market_cap"],
                }
                for s in self.get_by_market_cap(limit=5)
            ],
        }

    # ── Maintenance ───────────────────────────────────────────────────────────

    def refresh(self):
        """
        Force refresh of Nifty 500 detailed metadata from live NSE.
        Call this periodically (monthly recommended) via the scheduler.
        """
        logger.info("Refreshing Nifty 500 detailed metadata...")
        self.get_nifty500_detailed(force_refresh=True)
        logger.info("Refresh complete.")


# ── Module-level singleton ────────────────────────────────────────────────────
stock_universe_manager = StockUniverseManager()
