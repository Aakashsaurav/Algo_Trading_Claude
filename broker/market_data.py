"""
broker/market_data.py
----------------------
Fetches OHLCV (Open, High, Low, Close, Volume) candlestick data from Upstox API.

WHAT THIS MODULE DOES:
- Fetch historical daily OHLCV candles for any NSE stock/index/F&O instrument
- Fetch historical 1-minute OHLCV candles (intraday data)
- Return data as clean Pandas DataFrames ready for analysis and storage

UPSTOX HISTORICAL DATA API:
- Endpoint: GET /v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}
- instrument_key format: NSE_EQ|ISIN  (e.g. "NSE_EQ|INE009A01021" for Infosys)
- Intervals supported: 1minute, 30minute, day, week, month
- Max 1-minute data: 30 days per request (Upstox limit)
- Max daily data: Up to 10 years per request

ABOUT INSTRUMENT KEYS:
- Each stock/F&O contract has a unique instrument_key on Upstox
- Format: {exchange}_{segment}|{ISIN or symbol}
- Examples:
    NSE_EQ|INE009A01021    → Infosys (equity, NSE)
    NSE_FO|...             → NSE Futures & Options
    BSE_EQ|...             → BSE equity
- instrument_manager.py handles all instrument lookups by symbol name

USAGE:
    from broker.market_data import MarketDataManager
    md = MarketDataManager()
    df = md.get_daily_ohlcv("NSE_EQ|INE009A01021", from_date="2023-01-01", to_date="2024-01-01")
"""

import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
import requests

from config import config
from broker.auth import auth_manager

logger = logging.getLogger(__name__)

# Upstox rate limit: 25 requests per second for market data
# We add a small delay between batch requests to be safe
REQUEST_DELAY_SECONDS = 0.1

# Valid interval strings accepted by Upstox API
VALID_INTERVALS = {"1minute", "30minute", "day", "week", "month"}

# Maximum days of 1-minute data per single API request (Upstox limit)
MAX_MINUTE_DAYS_PER_REQUEST = 30


class MarketDataManager:
    """
    Fetches historical OHLCV candle data from Upstox API.

    All methods return Pandas DataFrames with a clean, standardised structure:
        index: datetime (timezone-aware, IST)
        columns: open, high, low, close, volume (and optionally oi for F&O)
    """

    def __init__(self):
        self.base_url = config.BASE_URL
        self.session = requests.Session()  # Reuse HTTP connection for efficiency

    def _get_headers(self) -> dict:
        """
        Build the authorization headers required for every API call.
        Always fetches the latest valid token (handles daily expiry automatically).
        """
        token = auth_manager.get_valid_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def _make_request(self, url: str, params: Optional[dict] = None) -> dict:
        """
        Make a GET request to Upstox API with error handling.

        Args:
            url (str): Full API endpoint URL.
            params (dict, optional): Query string parameters.

        Returns:
            dict: Parsed JSON response body.

        Raises:
            requests.HTTPError: For non-200 responses.
            requests.Timeout: If the request takes longer than 20 seconds.
        """
        try:
            response = self.session.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=20
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out: {url}")
            raise

        except requests.exceptions.ConnectionError:
            logger.error("Network error: Cannot reach Upstox API. Check your connection.")
            raise

        except requests.exceptions.HTTPError as e:
            # Extract Upstox-specific error message
            try:
                error_body = response.json()
                error_msg = error_body.get("errors", [{}])[0].get("message", str(e))
            except Exception:
                error_msg = str(e)
            logger.error(f"Upstox API error for {url}: {error_msg}")
            raise

    def _parse_candles_to_df(self, candles: list, interval: str) -> pd.DataFrame:
        """
        Convert raw Upstox candle list to a clean, indexed Pandas DataFrame.

        Upstox returns candles as:
            [timestamp, open, high, low, close, volume, oi]
        where OI (Open Interest) is only relevant for F&O instruments.

        Args:
            candles (list): Raw list of candle arrays from Upstox.
            interval (str): The data interval (used to label the DataFrame).

        Returns:
            pd.DataFrame: Clean OHLCV DataFrame sorted by datetime ascending.
        """
        if not candles:
            logger.warning("No candle data returned from API.")
            return pd.DataFrame()

        # Upstox candle format: [datetime_str, open, high, low, close, volume, oi]
        df = pd.DataFrame(
            candles,
            columns=["datetime", "open", "high", "low", "close", "volume", "oi"]
        )

        # Convert datetime string to proper datetime objects
        # Upstox uses ISO 8601 format: "2023-01-02T09:15:00+05:30"
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)  # Oldest candle first (chronological order)

        # Ensure numeric types — Upstox sometimes returns strings
        for col in ["open", "high", "low", "close", "volume", "oi"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop any rows that are completely null (data gaps)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        logger.debug(f"Parsed {len(df)} candles | interval={interval} | "
                     f"range={df.index[0]} → {df.index[-1]}")

        return df

    # ── Public Methods ────────────────────────────────────────────────────────

    def get_daily_ohlcv(
        self,
        instrument_key: str,
        from_date: str,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV candles for a given instrument.

        Args:
            instrument_key (str): Upstox instrument key. e.g. "NSE_EQ|INE009A01021"
            from_date (str): Start date in "YYYY-MM-DD" format.
            to_date (str, optional): End date. Defaults to today.

        Returns:
            pd.DataFrame: Daily OHLCV data. Empty DataFrame if no data found.

        Example:
            df = md.get_daily_ohlcv("NSE_EQ|INE009A01021", "2022-01-01", "2024-01-01")
        """
        if to_date is None:
            to_date = date.today().isoformat()

        # Validate date format
        try:
            datetime.strptime(from_date, "%Y-%m-%d")
            datetime.strptime(to_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD'. Error: {e}")

        url = (
            f"{self.base_url}/v2/historical-candle/"
            f"{instrument_key}/day/{to_date}/{from_date}"
        )

        logger.info(f"Fetching daily OHLCV | {instrument_key} | {from_date} → {to_date}")

        response_data = self._make_request(url)
        candles = response_data.get("data", {}).get("candles", [])

        return self._parse_candles_to_df(candles, interval="day")

    def get_minute_ohlcv(
        self,
        instrument_key: str,
        from_date: str,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch 1-minute OHLCV candles for a given instrument.

        ⚠️ UPSTOX LIMIT: Maximum 30 days of 1-minute data per request.
        For more than 30 days, use get_minute_ohlcv_range() which
        automatically handles multiple requests.

        Args:
            instrument_key (str): Upstox instrument key.
            from_date (str): Start date in "YYYY-MM-DD" format.
            to_date (str, optional): End date. Defaults to today.

        Returns:
            pd.DataFrame: 1-minute OHLCV data.
        """
        if to_date is None:
            to_date = date.today().isoformat()

        # Validate date format
        try:
            start = datetime.strptime(from_date, "%Y-%m-%d")
            end = datetime.strptime(to_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD'. Error: {e}")

        # Warn if range exceeds Upstox 30-day limit
        if (end - start).days > MAX_MINUTE_DAYS_PER_REQUEST:
            logger.warning(
                f"Requested range {from_date} → {to_date} is more than "
                f"{MAX_MINUTE_DAYS_PER_REQUEST} days. Use get_minute_ohlcv_range() instead."
            )

        url = (
            f"{self.base_url}/v2/historical-candle/"
            f"{instrument_key}/1minute/{to_date}/{from_date}"
        )

        logger.info(f"Fetching 1min OHLCV | {instrument_key} | {from_date} → {to_date}")

        response_data = self._make_request(url)
        candles = response_data.get("data", {}).get("candles", [])

        return self._parse_candles_to_df(candles, interval="1minute")

    def get_minute_ohlcv_range(
        self,
        instrument_key: str,
        from_date: str,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch 1-minute OHLCV data for ranges longer than 30 days.

        Automatically splits the date range into 30-day chunks,
        makes multiple API requests, and combines the results.

        Args:
            instrument_key (str): Upstox instrument key.
            from_date (str): Start date in "YYYY-MM-DD" format.
            to_date (str, optional): End date. Defaults to today.

        Returns:
            pd.DataFrame: Combined 1-minute OHLCV data for the full range.
        """
        if to_date is None:
            to_date = date.today().isoformat()

        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = datetime.strptime(to_date, "%Y-%m-%d").date()

        all_chunks = []
        chunk_start = start

        logger.info(
            f"Fetching 1min OHLCV in chunks | {instrument_key} | {from_date} → {to_date}"
        )

        while chunk_start <= end:
            chunk_end = min(
                chunk_start + timedelta(days=MAX_MINUTE_DAYS_PER_REQUEST - 1),
                end
            )

            try:
                df_chunk = self.get_minute_ohlcv(
                    instrument_key,
                    chunk_start.isoformat(),
                    chunk_end.isoformat()
                )
                if not df_chunk.empty:
                    all_chunks.append(df_chunk)

            except Exception as e:
                logger.warning(
                    f"Failed to fetch chunk {chunk_start} → {chunk_end}: {e}. Skipping."
                )

            chunk_start = chunk_end + timedelta(days=1)

            # Respect Upstox rate limits between chunk requests
            time.sleep(REQUEST_DELAY_SECONDS)

        if not all_chunks:
            logger.warning(f"No 1-minute data fetched for {instrument_key}.")
            return pd.DataFrame()

        # Combine all chunks and remove any duplicate candles at chunk boundaries
        combined = pd.concat(all_chunks)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        logger.info(
            f"✅ Combined {len(combined)} 1-minute candles for {instrument_key}"
        )
        return combined

    def get_weekly_ohlcv(
        self,
        instrument_key: str,
        from_date: str,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch weekly OHLCV candles for a given instrument.

        Args:
            instrument_key (str): Upstox instrument key.
            from_date (str): Start date in "YYYY-MM-DD" format.
            to_date (str, optional): End date. Defaults to today.

        Returns:
            pd.DataFrame: Weekly OHLCV data.
        """
        if to_date is None:
            to_date = date.today().isoformat()

        url = (
            f"{self.base_url}/v2/historical-candle/"
            f"{instrument_key}/week/{to_date}/{from_date}"
        )

        logger.info(f"Fetching weekly OHLCV | {instrument_key} | {from_date} → {to_date}")

        response_data = self._make_request(url)
        candles = response_data.get("data", {}).get("candles", [])

        return self._parse_candles_to_df(candles, interval="week")

    def get_market_holidays(self, specific_date: Optional[str] = None) -> list:
        """
        Fetch market holidays for the current year from Upstox.

        Args:
            specific_date (str, optional): Check a specific date "YYYY-MM-DD".
                                           If None, returns all holidays this year.

        Returns:
            list: List of holiday dictionaries with date and description.
        """
        if specific_date:
            url = f"{self.base_url}/v2/market/holidays/{specific_date}"
        else:
            url = f"{self.base_url}/v2/market/holidays"

        logger.info(f"Fetching market holidays{f' for {specific_date}' if specific_date else ''}...")

        response_data = self._make_request(url)
        holidays = response_data.get("data", [])

        logger.info(f"Fetched {len(holidays)} market holiday entries.")
        return holidays

    def is_market_holiday(self, check_date: Optional[str] = None) -> bool:
        """
        Check if a given date is a market holiday for NSE.

        Args:
            check_date (str, optional): Date to check "YYYY-MM-DD". Defaults to today.

        Returns:
            bool: True if NSE is closed on that date.
        """
        if check_date is None:
            check_date = date.today().isoformat()

        try:
            holidays = self.get_market_holidays(specific_date=check_date)

            for holiday in holidays:
                closed = holiday.get("closed_exchanges", [])
                # Check if NSE or NFO is fully closed
                if "NSE" in closed or "NFO" in closed:
                    logger.info(
                        f"{check_date} is a market holiday: "
                        f"{holiday.get('description', 'Unknown')}"
                    )
                    return True

            return False

        except Exception as e:
            logger.warning(
                f"Could not verify holiday status for {check_date}: {e}. "
                f"Assuming it's a trading day."
            )
            return False


# ── Module-level singleton ────────────────────────────────────────────────────
market_data = MarketDataManager()
