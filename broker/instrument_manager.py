"""
broker/instrument_manager.py
------------------------------
Manages the Upstox instrument master — the complete list of all tradeable
instruments (stocks, futures, options, indices) with their unique keys.

WHY THIS EXISTS:
- Upstox API requires an instrument_key (like "NSE_EQ|INE009A01021") for every call.
- Traders know stocks by symbol (like "INFY"), not by ISIN.
- This module bridges the gap: symbol → instrument_key lookup.

HOW UPSTOX INSTRUMENT DATA WORKS:
- Upstox publishes a fresh instrument CSV every morning (before market open).
- We download it once per day and store it locally.
- The CSV contains: instrument_key, exchange, symbol, name, lot_size, expiry, etc.

USAGE:
    from broker.instrument_manager import instrument_manager
    key = instrument_manager.get_instrument_key("INFY", "NSE_EQ")
    print(key)  # → "NSE_EQ|INE009A01021"
"""

import gzip
import io
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import config

logger = logging.getLogger(__name__)


class InstrumentManager:
    """
    Downloads, caches, and provides lookup utilities for Upstox instruments.

    The instrument DataFrame (self._instruments) is the core data structure.
    It's loaded once per day and kept in memory for fast lookups.
    """

    # Relevant columns from the Upstox instrument CSV
    REQUIRED_COLUMNS = [
        "instrument_key",
        "exchange",
        "trading_symbol",
        "name",
        "instrument_type",
        "lot_size",
        "freeze_quantity",
        "expiry",
        "strike",
        "option_type",
        "tick_size",
        "isin",
        "segment",
    ]

    def __init__(self):
        self._instruments: Optional[pd.DataFrame] = None
        self._loaded_date: Optional[date] = None

    def _download_instruments(self) -> pd.DataFrame:
        """
        Download the latest instrument CSV from Upstox.

        The file is a gzip-compressed CSV updated each morning.
        We download, decompress, and parse it into a DataFrame.

        Returns:
            pd.DataFrame: Full instrument master.

        Raises:
            requests.RequestException: If download fails.
        """
        url = config.INSTRUMENT_CSV_URL
        logger.info(f"Downloading instrument master from Upstox...")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

        except requests.exceptions.Timeout:
            logger.error("Instrument download timed out. Check your internet connection.")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed to download instruments: {e}")
            raise

        # Decompress gzip content and parse as CSV
        try:
            with gzip.open(io.BytesIO(response.content), "rt", encoding="utf-8") as f:
                df = pd.read_csv(f, low_memory=False)

        except Exception as e:
            logger.error(f"Failed to decompress/parse instrument CSV: {e}")
            raise

        logger.info(f"Downloaded {len(df):,} instruments from Upstox.")
        return df

    def _load_or_refresh(self):
        """
        Load instruments into memory. Uses cached data if already loaded today.

        Logic:
        1. If in-memory data was loaded today → do nothing (already fresh).
        2. If a local CSV exists from today → load from disk (fast).
        3. Otherwise → download fresh from Upstox, save to disk.
        """
        today = date.today()

        # 1. Already loaded today — use in-memory cache
        if self._instruments is not None and self._loaded_date == today:
            return

        csv_path = config.INSTRUMENT_CSV_PATH

        # 2. Check if we already saved today's file on disk
        if csv_path.exists():
            file_date = date.fromtimestamp(csv_path.stat().st_mtime)
            if file_date == today:
                logger.info(f"Loading instruments from today's cached file: {csv_path}")
                self._instruments = pd.read_csv(csv_path, low_memory=False)
                self._loaded_date = today
                return

        # 3. Download fresh copy and save
        df = self._download_instruments()

        # Save to disk for reuse during the same day
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Instrument master saved to: {csv_path}")

        self._instruments = df
        self._loaded_date = today

    def refresh(self):
        """
        Force a fresh download of the instrument master, ignoring cache.
        Call this at market open (9:00 AM) each trading day.
        """
        logger.info("Force-refreshing instrument master...")
        self._loaded_date = None  # Invalidate cache
        self._instruments = None
        self._load_or_refresh()

    def get_all_instruments(self) -> pd.DataFrame:
        """
        Return the full instrument master as a DataFrame.

        Returns:
            pd.DataFrame: All instruments with full metadata.
        """
        self._load_or_refresh()
        return self._instruments.copy()

    def get_instrument_key(
        self,
        trading_symbol: str,
        exchange: str = "NSE_EQ"
    ) -> Optional[str]:
        """
        Get the Upstox instrument_key for a given trading symbol.

        Args:
            trading_symbol (str): The symbol as displayed on NSE. e.g. "INFY", "RELIANCE"
            exchange (str): Exchange segment. Common values:
                            "NSE_EQ"  → NSE equity (default)
                            "BSE_EQ"  → BSE equity
                            "NSE_FO"  → NSE F&O (futures and options)
                            "MCX_FO"  → MCX commodities

        Returns:
            str: instrument_key if found, None if not found.

        Example:
            key = instrument_manager.get_instrument_key("INFY", "NSE_EQ")
            # Returns: "NSE_EQ|INE009A01021"
        """
        self._load_or_refresh()

        mask = (
            (self._instruments["trading_symbol"].str.upper() == trading_symbol.upper()) &
            (self._instruments["exchange"].str.upper() == exchange.upper())
        )

        matches = self._instruments[mask]

        if matches.empty:
            logger.warning(
                f"Instrument not found: symbol='{trading_symbol}', exchange='{exchange}'"
            )
            return None

        if len(matches) > 1:
            # For equities, prefer the plain equity row (not derivatives)
            equity_matches = matches[matches["instrument_type"] == "EQ"]
            if not equity_matches.empty:
                return equity_matches.iloc[0]["instrument_key"]

        return matches.iloc[0]["instrument_key"]

    def search_instruments(
        self,
        query: str,
        exchange: Optional[str] = None,
        instrument_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Search instruments by partial symbol or name match.

        Useful when you're not sure of the exact symbol.

        Args:
            query (str): Partial symbol or name to search.
            exchange (str, optional): Filter by exchange. e.g. "NSE_EQ"
            instrument_type (str, optional): Filter by type. e.g. "EQ", "FUT", "OPT"

        Returns:
            pd.DataFrame: Matching instruments.

        Example:
            df = instrument_manager.search_instruments("TATA", exchange="NSE_EQ")
        """
        self._load_or_refresh()

        mask = (
            self._instruments["trading_symbol"].str.contains(query, case=False, na=False) |
            self._instruments["name"].str.contains(query, case=False, na=False)
        )
        results = self._instruments[mask].copy()

        if exchange:
            results = results[results["exchange"].str.upper() == exchange.upper()]

        if instrument_type:
            results = results[
                results["instrument_type"].str.upper() == instrument_type.upper()
            ]

        return results[["instrument_key", "exchange", "trading_symbol",
                         "name", "instrument_type", "lot_size", "expiry"]].head(20)

    def get_nse_equity_instruments(self) -> pd.DataFrame:
        """
        Return all NSE equity instruments (EQ segment only).
        This is the base universe for your NSE 500 screener.

        Returns:
            pd.DataFrame: NSE_EQ instruments.
        """
        self._load_or_refresh()

        mask = (
            (self._instruments["exchange"] == "NSE_EQ") &
            (self._instruments["instrument_type"] == "EQ")
        )
        return self._instruments[mask].copy()

    def get_fo_instruments(
        self,
        instrument_type: Optional[str] = None,
        expiry_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return NSE F&O instruments.

        Args:
            instrument_type (str, optional): "FUT" for futures, "OPT" for options.
                                              None returns all F&O.
            expiry_date (str, optional): Filter by expiry date "YYYY-MM-DD".

        Returns:
            pd.DataFrame: Matching F&O instruments.
        """
        self._load_or_refresh()

        mask = self._instruments["exchange"] == "NSE_FO"
        results = self._instruments[mask].copy()

        if instrument_type:
            results = results[
                results["instrument_type"].str.upper() == instrument_type.upper()
            ]

        if expiry_date:
            results = results[results["expiry"] == expiry_date]

        return results

    def get_index_instrument_key(self, index_name: str) -> Optional[str]:
        """
        Get the instrument key for major NSE indices.

        Args:
            index_name (str): Index name. Common values:
                              "NIFTY 50", "NIFTY BANK", "NIFTY 500",
                              "NIFTY MIDCAP 100", etc.

        Returns:
            str: instrument_key if found, None otherwise.
        """
        self._load_or_refresh()

        # Indices are in NSE_INDEX segment
        mask = (
            (self._instruments["exchange"].str.contains("INDEX", case=False, na=False)) &
            (self._instruments["trading_symbol"].str.upper() == index_name.upper())
        )
        matches = self._instruments[mask]

        if matches.empty:
            # Try broader name search
            mask2 = self._instruments["name"].str.upper() == index_name.upper()
            matches = self._instruments[mask2]

        if matches.empty:
            logger.warning(f"Index not found: '{index_name}'")
            return None

        return matches.iloc[0]["instrument_key"]

    def get_lot_size(self, instrument_key: str) -> int:
        """
        Get the lot size for an F&O instrument.
        Required for position sizing in F&O strategies.

        Args:
            instrument_key (str): Upstox instrument key.

        Returns:
            int: Lot size. Returns 1 for equities (no lot concept).
        """
        self._load_or_refresh()

        mask = self._instruments["instrument_key"] == instrument_key
        matches = self._instruments[mask]

        if matches.empty:
            logger.warning(f"Instrument not found for lot size: {instrument_key}")
            return 1

        lot_size = matches.iloc[0].get("lot_size", 1)
        return int(lot_size) if pd.notna(lot_size) else 1


# ── Module-level singleton ────────────────────────────────────────────────────
instrument_manager = InstrumentManager()
