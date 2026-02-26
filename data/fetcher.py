"""
data/fetcher.py
----------------
Orchestrates the download of OHLCV data from Upstox and saves it to Parquet.

This module is the bridge between broker/market_data.py (fetch from API)
and data/parquet_store.py (save to disk).

RESPONSIBILITIES:
- Download daily/minute/weekly OHLCV for one or many symbols
- Detect what data we already have and only fetch what's missing (incremental update)
- Handle rate limiting gracefully (Upstox: 25 req/sec for market data)
- Log progress for long batch downloads (downloading 500 stocks takes a while)

TYPICAL USAGE:
    from data.fetcher import data_fetcher

    # Download 2 years of daily data for a single stock
    data_fetcher.fetch_and_save_daily("NSE_EQ", "INFY", from_date="2022-01-01")

    # Download daily data for all NSE 500 stocks (runs in batch with rate limiting)
    data_fetcher.fetch_universe_daily(symbols_list, from_date="2022-01-01")

    # Incremental update — only downloads data after the last saved date
    data_fetcher.incremental_update_daily("NSE_EQ", "INFY")
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from broker.market_data import market_data
from broker.instrument_manager import instrument_manager
from data.parquet_store import parquet_store

logger = logging.getLogger(__name__)

# Delay between API calls in batch operations (seconds)
# 0.1s = ~10 requests/second, well within Upstox's 25 req/sec limit
BATCH_REQUEST_DELAY = 0.1

# How many days of daily history to fetch if no start date is given
DEFAULT_HISTORY_DAYS = 365 * 3  # 3 years


class DataFetcher:
    """
    Downloads OHLCV data from Upstox and persists it to local Parquet storage.

    Key design decision: all fetch methods are idempotent — you can call them
    multiple times safely. They only download what's actually missing.
    """

    def __init__(self):
        self.market_data = market_data
        self.instrument_manager = instrument_manager
        self.store = parquet_store

    # ── Single Symbol Fetch ───────────────────────────────────────────────────

    def fetch_and_save_daily(
        self,
        exchange: str,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        instrument_key: Optional[str] = None,
    ) -> bool:
        """
        Download daily OHLCV for a symbol and save to Parquet.

        Args:
            exchange (str): Exchange segment. e.g. "NSE_EQ"
            symbol (str): Trading symbol. e.g. "INFY"
            from_date (str, optional): Start date "YYYY-MM-DD". Defaults to 3 years ago.
            to_date (str, optional): End date "YYYY-MM-DD". Defaults to today.
            instrument_key (str, optional): If you already have the key, pass it directly
                                             to skip the lookup step.

        Returns:
            bool: True if successful, False if failed.
        """
        # Resolve instrument key if not provided
        if not instrument_key:
            instrument_key = self.instrument_manager.get_instrument_key(symbol, exchange)

        if not instrument_key:
            logger.error(f"Cannot find instrument key for {exchange}:{symbol}. Skipping.")
            return False

        # Default date range
        if from_date is None:
            from_date = (date.today() - timedelta(days=DEFAULT_HISTORY_DAYS)).isoformat()
        if to_date is None:
            to_date = date.today().isoformat()

        try:
            df = self.market_data.get_daily_ohlcv(instrument_key, from_date, to_date)

            if df.empty:
                logger.warning(f"No daily data returned for {symbol} ({from_date} → {to_date})")
                return False

            self.store.save_daily(df, exchange, symbol)
            return True

        except Exception as e:
            logger.error(f"Failed to fetch/save daily data for {symbol}: {e}")
            return False

    def fetch_and_save_minute(
        self,
        exchange: str,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        instrument_key: Optional[str] = None,
    ) -> bool:
        """
        Download 1-minute OHLCV for a symbol and save to Parquet.

        Automatically handles Upstox's 30-day limit by chunking requests.

        Args:
            exchange (str): Exchange segment.
            symbol (str): Trading symbol.
            from_date (str, optional): Start date. Defaults to 90 days ago.
            to_date (str, optional): End date. Defaults to today.
            instrument_key (str, optional): Pre-resolved instrument key.

        Returns:
            bool: True if successful, False if failed.
        """
        if not instrument_key:
            instrument_key = self.instrument_manager.get_instrument_key(symbol, exchange)

        if not instrument_key:
            logger.error(f"Cannot find instrument key for {exchange}:{symbol}. Skipping.")
            return False

        # Default: last 90 days of minute data (meaningful for intraday strategies)
        if from_date is None:
            from_date = (date.today() - timedelta(days=90)).isoformat()
        if to_date is None:
            to_date = date.today().isoformat()

        try:
            # This method handles chunking internally (30-day batches)
            df = self.market_data.get_minute_ohlcv_range(instrument_key, from_date, to_date)

            if df.empty:
                logger.warning(f"No minute data returned for {symbol}")
                return False

            self.store.save_minute(df, exchange, symbol)
            return True

        except Exception as e:
            logger.error(f"Failed to fetch/save minute data for {symbol}: {e}")
            return False

    def fetch_and_save_weekly(
        self,
        exchange: str,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        instrument_key: Optional[str] = None,
    ) -> bool:
        """Download weekly OHLCV for a symbol and save to Parquet."""
        if not instrument_key:
            instrument_key = self.instrument_manager.get_instrument_key(symbol, exchange)

        if not instrument_key:
            logger.error(f"Cannot find instrument key for {exchange}:{symbol}. Skipping.")
            return False

        if from_date is None:
            from_date = (date.today() - timedelta(days=DEFAULT_HISTORY_DAYS)).isoformat()
        if to_date is None:
            to_date = date.today().isoformat()

        try:
            df = self.market_data.get_weekly_ohlcv(instrument_key, from_date, to_date)

            if df.empty:
                logger.warning(f"No weekly data returned for {symbol}")
                return False

            self.store.save_weekly(df, exchange, symbol)
            return True

        except Exception as e:
            logger.error(f"Failed to fetch/save weekly data for {symbol}: {e}")
            return False

    # ── Incremental Update ────────────────────────────────────────────────────

    def incremental_update_daily(self, exchange: str, symbol: str) -> bool:
        """
        Smart update: only download daily data that's newer than what we have.

        This is what the EOD scheduler calls every evening at 6 PM.
        Instead of re-downloading everything, it just fetches today's new candles.

        Logic:
        1. Check the latest date in our stored data.
        2. Fetch from (latest_date + 1 day) to today.
        3. Merge and save.

        Args:
            exchange (str): Exchange segment.
            symbol (str): Trading symbol.

        Returns:
            bool: True if successful.
        """
        _, latest_date = self.store.get_daily_date_range(exchange, symbol)

        if latest_date is None:
            # No existing data — do a full fetch
            logger.info(f"No existing data for {symbol}. Doing full historical fetch.")
            return self.fetch_and_save_daily(exchange, symbol)

        # Start from the day after our latest stored candle
        from_date = (
            datetime.strptime(latest_date, "%Y-%m-%d") + timedelta(days=1)
        ).date().isoformat()

        today = date.today().isoformat()

        if from_date > today:
            logger.debug(f"{symbol} daily data is already up to date.")
            return True

        logger.info(f"Incremental update for {symbol}: {from_date} → {today}")
        return self.fetch_and_save_daily(exchange, symbol, from_date=from_date)

    # ── Universe Batch Fetch ──────────────────────────────────────────────────

    def fetch_universe_daily(
        self,
        symbols: list[dict],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> dict:
        """
        Download daily OHLCV for a list of symbols (batch operation).

        This is the main method for initial data download of your full universe
        (e.g., all 500 Nifty 500 stocks). Expect this to take 5–10 minutes
        for 500 stocks due to API rate limiting.

        Args:
            symbols (list[dict]): List of dicts with keys "exchange" and "symbol".
                                   e.g. [{"exchange": "NSE_EQ", "symbol": "INFY"}, ...]
            from_date (str, optional): Start date for all symbols.
            to_date (str, optional): End date for all symbols.

        Returns:
            dict: Summary with "success", "failed", and "skipped" lists.

        Example:
            symbols = [
                {"exchange": "NSE_EQ", "symbol": "INFY"},
                {"exchange": "NSE_EQ", "symbol": "TCS"},
            ]
            result = data_fetcher.fetch_universe_daily(symbols, from_date="2022-01-01")
            print(result)
            # {'success': ['INFY', 'TCS'], 'failed': [], 'skipped': []}
        """
        total = len(symbols)
        success = []
        failed = []

        logger.info(f"Starting batch daily fetch for {total} symbols...")
        print(f"\n{'='*55}")
        print(f"  Downloading daily data for {total} symbols")
        print(f"  Estimated time: ~{total * BATCH_REQUEST_DELAY / 60:.1f} minutes")
        print(f"{'='*55}")

        for i, item in enumerate(symbols, 1):
            exchange = item.get("exchange", "NSE_EQ")
            symbol = item.get("symbol", "")

            if not symbol:
                logger.warning(f"Empty symbol at index {i}. Skipping.")
                continue

            # Progress log every 50 symbols
            if i % 50 == 0 or i == total:
                pct = (i / total) * 100
                logger.info(f"Progress: {i}/{total} ({pct:.0f}%) | Last: {symbol}")
                print(f"  [{i:>4}/{total}] {pct:>5.1f}% | {symbol}")

            result = self.fetch_and_save_daily(exchange, symbol, from_date, to_date)

            if result:
                success.append(symbol)
            else:
                failed.append(symbol)

            # Rate limiting delay — be a good citizen with the API
            time.sleep(BATCH_REQUEST_DELAY)

        summary = {
            "total": total,
            "success": success,
            "failed": failed,
            "success_count": len(success),
            "failed_count": len(failed),
        }

        logger.info(
            f"Batch fetch complete: {len(success)}/{total} successful, "
            f"{len(failed)} failed."
        )

        if failed:
            logger.warning(f"Failed symbols: {failed}")

        return summary

    def incremental_update_universe(self, symbols: list[dict]) -> dict:
        """
        Run incremental daily updates for all symbols in the universe.

        Call this every evening at 6 PM via the scheduler.
        Only downloads today's new candles for each symbol.

        Args:
            symbols (list[dict]): Same format as fetch_universe_daily.

        Returns:
            dict: Summary of update results.
        """
        total = len(symbols)
        success = []
        failed = []

        logger.info(f"Starting incremental EOD update for {total} symbols...")

        for i, item in enumerate(symbols, 1):
            exchange = item.get("exchange", "NSE_EQ")
            symbol = item.get("symbol", "")

            if not symbol:
                continue

            result = self.incremental_update_daily(exchange, symbol)

            if result:
                success.append(symbol)
            else:
                failed.append(symbol)

            time.sleep(BATCH_REQUEST_DELAY)

        logger.info(
            f"Incremental update complete: {len(success)}/{total} successful."
        )

        return {
            "total": total,
            "success_count": len(success),
            "failed_count": len(failed),
            "failed": failed,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
data_fetcher = DataFetcher()
