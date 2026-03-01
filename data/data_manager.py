"""
data/data_manager.py
---------------------
Single entry point for all OHLCV data needs — downloading, caching, updating,
and returning resampled candles for any instrument and timeframe.

DESIGN PRINCIPLES:
  Storage layer  : Always stores raw 1-minute or 1-day Parquet files on disk.
                   minute/hour requests → 1-minute base data stored.
                   day/week/month requests → 1-day base data stored.
  Return layer   : Resamples raw stored data on-the-fly to the requested
                   unit+interval before returning. Resampled data is NEVER
                   written to disk — only the raw base resolution is cached.
  Full history   : On first request for an instrument, the full available
                   history is downloaded (Jan 2022 for minute, Jan 2000 for
                   day). Subsequent calls are instant cache hits or small
                   incremental updates.

FILE LOCATIONS:
  Minute data : data/ohlcv/minute/<SYMBOL>/<YYYY-MM>.parquet
                One Parquet file per calendar month per instrument.
                e.g. data/ohlcv/minute/RELIANCE/2024-01.parquet
  Day data    : data/ohlcv/day/<SYMBOL>.parquet
                One Parquet file per instrument, all history.
                e.g. data/ohlcv/day/RELIANCE.parquet

  SYMBOL = trading_symbol (e.g. RELIANCE, INFY, NIFTY) — NOT instrument_key.
  This makes filenames human-readable and instrument-type agnostic.

PARQUET SCHEMA (both minute and day):
  timestamp : datetime64[ns, Asia/Kolkata]   timezone-aware index
  open      : float32    (50% memory saving vs float64, sufficient precision)
  high      : float32
  low       : float32
  close     : float32
  volume    : int32
  oi        : int32      (open interest — 0 for equities)

API USED: Upstox HistoryV3Api (V3 historical candle endpoint)
  SDK call : HistoryV3Api.get_historical_candle_data1(
                 instrument_key, unit, interval, to_date, from_date)
  Auth     : Any non-empty string works for the historical endpoint as per
             Upstox documentation. "dummy_token" is used as the default.
             If a real access token is available it will be used automatically.

CHUNK LIMITS (from Upstox V3 API docs):
  minutes (interval 1-15)  : 1 month per call
  minutes (interval 16-300): 1 quarter per call   ← we always fetch interval=1
  hours                    : 1 quarter per call    ← we always fetch interval=1
  days                     : 10 years per call
  weeks / months           : No limit

  Since we always store base resolution (1-minute or 1-day), we only ever
  call the API with interval=1 and unit="minutes" or unit="days".
  Chunking: minute → monthly chunks; day → 10-year chunks.

WEEKEND / HOLIDAY LOGIC:
  NSE is closed on Saturdays and Sundays. If the query end-date falls on a
  weekend and the cache already covers through the last trading day (Friday),
  no API call is made. NSE holidays are also respected using the same logic —
  if the latest cache date equals the last working day before today, the
  cache is treated as current.

USAGE:
  from data.data_manager import get_ohlcv

  # 5-minute OHLCV for INFY for the last 3 months
  df = get_ohlcv(
      instrument_type="EQUITY", exchange="NSE", trading_symbol="INFY",
      unit="minutes", interval=5, period="3months"
  )

  # Daily candles for RELIANCE from a specific date range
  df = get_ohlcv(
      instrument_type="EQUITY", exchange="NSE", trading_symbol="RELIANCE",
      unit="days", interval=1,
      from_date="2023-01-01", to_date="2024-12-31"
  )

  # Weekly candles for BANKNIFTY futures
  df = get_ohlcv(
      instrument_type="FUTIDX", exchange="NSE", trading_symbol="BANKNIFTY",
      unit="weeks", interval=1, expiry="27MAR26", period="1year"
  )
"""

import time
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import upstox_client

from config import config
from broker.instrument_manager import get_instrument_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Earliest dates Upstox has data for (hard limits from their documentation)
MINUTE_DATA_START = date(2022, 1, 3)   # Jan 2022 — first trading day
DAY_DATA_START    = date(2000, 1, 3)   # Jan 2000 — first trading day

# IST timezone string used throughout for timezone-aware timestamps
IST = "Asia/Kolkata"

# Parquet base directories (from config)
MINUTE_DIR = config.MINUTE_DIR  # data/ohlcv/minute/
DAY_DIR    = config.DAILY_DIR   # data/ohlcv/daily/

# Pause between consecutive API calls (seconds) to respect Upstox rate limits.
# Upstox allows ~25 req/sec; 0.25s sleep keeps us safely under 4 req/sec.
API_CALL_SLEEP = 0.25

# Parquet dtypes — float32 and int32 halve memory vs default float64/int64
PARQUET_DTYPES = {
    "open":   "float32",
    "high":   "float32",
    "low":    "float32",
    "close":  "float32",
    "volume": "int32",
    "oi":     "int32",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_ohlcv(
    instrument_type:  str,
    exchange:         str,
    trading_symbol:   str,
    unit:             str,
    interval:         int,
    from_date:        Optional[str] = None,
    to_date:          Optional[str] = None,
    period:           Optional[str] = None,
    option_type:      Optional[str] = None,
    expiry:           Optional[str] = None,
    strike:           Optional[float] = None,
) -> pd.DataFrame:
    """
    Main entry point. Returns a resampled OHLCV DataFrame for the requested
    instrument, unit, and interval.

    Args:
        instrument_type (str): "EQUITY", "INDEX", "FUTSTK", "FUTIDX",
                               "OPTSTK", "OPTIDX", "FUTCOM", "OPTCOM", etc.
        exchange (str):        "NSE", "BSE", or "MCX".
        trading_symbol (str):  e.g. "INFY", "NIFTY", "RELIANCE".
        unit (str):            "minutes", "hours", "days", "weeks", "months".
        interval (int):        Candle size. e.g. 1, 3, 5, 15 for minutes.
        from_date (str):       Start date "YYYY-MM-DD". Optional if period given.
        to_date (str):         End date "YYYY-MM-DD". Defaults to today.
        period (str):          e.g. "3months", "6weeks", "2years", "90days".
                               Used when from_date is not given.
                               to_date defaults to today if also not given.
        option_type (str):     "CE" or "PE" — required for options.
        expiry (str):          "DDMONYY" — required for F&O. e.g. "27MAR26".
        strike (float):        Strike price — required for options.

    Returns:
        pd.DataFrame: Index = timezone-aware timestamp (IST).
                      Columns = [open, high, low, close, volume, oi].
                      Filtered to [from_date, to_date], resampled to
                      the requested unit and interval.

    Raises:
        ValueError: On invalid inputs or if instrument key cannot be resolved.
    """
    # ------------------------------------------------------------------
    # Step 1: Validate and resolve date range
    # ------------------------------------------------------------------
    query_start, query_end = _resolve_date_range(from_date, to_date, period)

    # ------------------------------------------------------------------
    # Step 2: Resolve instrument key using instrument_manager
    # ------------------------------------------------------------------
    instrument_key = get_instrument_key(
        instrument_type=instrument_type,
        exchange=exchange,
        trading_symbol=trading_symbol,
        option_type=option_type,
        expiry=expiry,
        strike=strike,
    )
    if not instrument_key:
        raise ValueError(
            f"Could not resolve instrument key for: {instrument_type} "
            f"{exchange} {trading_symbol} expiry={expiry} strike={strike}"
        )

    logger.info(
        f"get_ohlcv: {trading_symbol} | {unit}/{interval} | "
        f"{query_start} → {query_end} | key={instrument_key}"
    )

    # ------------------------------------------------------------------
    # Step 3: Decide storage unit (what we actually download & store)
    # ------------------------------------------------------------------
    storage_unit = _resolve_storage_unit(unit)
    # storage_unit is either "minutes" or "days"

    # ------------------------------------------------------------------
    # Step 4: Ensure local cache is current (download if needed)
    # ------------------------------------------------------------------
    _ensure_cache_current(
        instrument_key=instrument_key,
        trading_symbol=trading_symbol,
        storage_unit=storage_unit,
    )

    # ------------------------------------------------------------------
    # Step 5: Load only the date range we need from disk
    # ------------------------------------------------------------------
    raw_df = _load_from_cache(
        trading_symbol=trading_symbol,
        storage_unit=storage_unit,
        query_start=query_start,
        query_end=query_end,
    )

    if raw_df.empty:
        logger.warning(
            f"No data found in cache for {trading_symbol} "
            f"({query_start} → {query_end})"
        )
        return raw_df

    # ------------------------------------------------------------------
    # Step 6: Resample raw base-resolution data to requested unit+interval
    # ------------------------------------------------------------------
    result_df = _resample(raw_df, unit, interval)

    # ------------------------------------------------------------------
    # Step 7: Filter to the exact requested date range and return
    # ------------------------------------------------------------------
    tz_start = pd.Timestamp(query_start, tz=IST)
    tz_end   = pd.Timestamp(query_end,   tz=IST) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    result_df = result_df.loc[tz_start:tz_end]

    logger.info(
        f"Returning {len(result_df)} {interval}{unit} candles "
        f"for {trading_symbol}"
    )
    return result_df


# ---------------------------------------------------------------------------
# Step 1 Helper: Resolve date range
# ---------------------------------------------------------------------------

def _resolve_date_range(
    from_date: Optional[str],
    to_date:   Optional[str],
    period:    Optional[str],
) -> tuple[date, date]:
    """
    Convert the three possible input combinations into concrete start/end dates.

    Priority logic:
      - If from_date AND to_date given: use them directly.
      - If period given (with optional to_date): compute from_date = to_date - period.
      - If only from_date given: to_date = today.
      - If nothing given: raise ValueError.

    Period string format: "<number><unit>" where unit is one of:
      days, weeks, months, years.  e.g. "90days", "3months", "1year", "6weeks"
    """
    today = date.today()

    # Parse to_date (default: today)
    if to_date:
        end = datetime.strptime(to_date, "%Y-%m-%d").date()
    else:
        end = today

    # Cap end at today — we cannot request future data
    if end > today:
        logger.warning(f"to_date {end} is in the future. Capping at today ({today}).")
        end = today

    # Resolve start date
    if from_date:
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
    elif period:
        start = _subtract_period(end, period)
    else:
        raise ValueError(
            "Provide at least one of: from_date, period. "
            "Example: from_date='2024-01-01' or period='3months'."
        )

    if start > end:
        raise ValueError(f"from_date ({start}) must be before to_date ({end}).")

    return start, end


def _subtract_period(from_date: date, period: str) -> date:
    """
    Subtract a period string from a date.

    Supported formats: "90days", "6weeks", "3months", "2years"
    """
    period = period.strip().lower()

    # Split number from unit — handle both "3months" and "3 months"
    for unit in ("years", "year", "months", "month", "weeks", "week", "days", "day"):
        if period.endswith(unit):
            try:
                n = int(period[: -len(unit)].strip())
            except ValueError:
                raise ValueError(
                    f"Invalid period format: '{period}'. "
                    "Expected format: '<number><unit>' e.g. '3months', '90days'."
                )
            if unit in ("years", "year"):
                return from_date.replace(year=from_date.year - n)
            elif unit in ("months", "month"):
                # Handle month subtraction correctly (including year rollover)
                month = from_date.month - n
                year  = from_date.year
                while month <= 0:
                    month += 12
                    year  -= 1
                # Clamp day to valid range for the resulting month
                import calendar
                max_day = calendar.monthrange(year, month)[1]
                return from_date.replace(year=year, month=month, day=min(from_date.day, max_day))
            elif unit in ("weeks", "week"):
                return from_date - timedelta(weeks=n)
            elif unit in ("days", "day"):
                return from_date - timedelta(days=n)

    raise ValueError(
        f"Unrecognised period unit in '{period}'. "
        "Use: days, weeks, months, years. e.g. '3months', '90days', '2years'."
    )


# ---------------------------------------------------------------------------
# Step 2 Helper: Storage unit decision
# ---------------------------------------------------------------------------

def _resolve_storage_unit(unit: str) -> str:
    """
    Decide what base resolution to store on disk.

    User's unit   → Storage unit on disk
    ─────────────────────────────────────
    minutes       → "minutes"  (stores 1-min candles)
    hours         → "minutes"  (stores 1-min; resamples to hours on read)
    days          → "days"     (stores 1-day candles)
    weeks         → "days"     (stores 1-day; resamples to weekly on read)
    months        → "days"     (stores 1-day; resamples to monthly on read)
    """
    unit = unit.lower().strip()
    if unit in ("minutes", "hours"):
        return "minutes"
    elif unit in ("days", "weeks", "months"):
        return "days"
    else:
        raise ValueError(
            f"Invalid unit: '{unit}'. "
            "Valid values: minutes, hours, days, weeks, months."
        )


# ---------------------------------------------------------------------------
# Step 3 Helper: Last trading day (weekend / holiday awareness)
# ---------------------------------------------------------------------------

def _last_trading_day(reference: date = None) -> date:
    """
    Return the most recent trading day on or before `reference`.
    Rolls back from Saturday → Friday, Sunday → Friday.

    NOTE: NSE holidays are not hardcoded here. The check in _cache_is_current
    compares the latest cached date against the last trading day. If a holiday
    falls between them, the API will simply return an empty response for that
    day — Upstox returns empty candles for holidays, which is handled
    gracefully by our empty-response check in _fetch_chunks.
    """
    if reference is None:
        reference = date.today()

    d = reference
    # Weekday: 0=Monday, 5=Saturday, 6=Sunday
    while d.weekday() >= 5:  # Saturday or Sunday → step back
        d -= timedelta(days=1)
    return d


def _cache_is_current(latest_cached: date, storage_unit: str) -> bool:
    """
    Return True if the cache is up to date and no API call is needed.

    Logic:
    - Compute the last trading day up to and including today.
    - For day data: cache is current if latest_cached >= last_trading_day.
    - For minute data: cache is current if latest_cached >= last_trading_day
      AND (we are past 15:30 IST today OR last_trading_day < today).
      This handles the case where today is a trading day and market is still open
      — we don't want to fetch incomplete today's candles and cache them.
    """
    last_trade_day = _last_trading_day(date.today())

    if storage_unit == "days":
        return latest_cached >= last_trade_day

    # For minute data: if market is open right now (today is a trading day
    # and time is before 15:35 IST), treat yesterday as the last complete day.
    elif storage_unit == "minutes":
        now_ist = datetime.now(tz=pd.Timestamp.now(tz=IST).tzinfo)
        market_close_ist = now_ist.replace(hour=15, minute=35, second=0, microsecond=0)

        if date.today().weekday() < 5 and datetime.now() < market_close_ist.replace(tzinfo=None):
            # Market may still be open — last *complete* day is yesterday
            last_complete_day = _last_trading_day(date.today() - timedelta(days=1))
        else:
            last_complete_day = last_trade_day

        return latest_cached >= last_complete_day

    return False


# ---------------------------------------------------------------------------
# Step 4: Ensure cache is current
# ---------------------------------------------------------------------------

def _ensure_cache_current(
    instrument_key: str,
    trading_symbol: str,
    storage_unit:   str,
) -> None:
    """
    Check local cache and download any missing data.

    Cases handled:
    A) No cache at all        → download full history from absolute start to today.
    B) Cache is up to date    → do nothing.
    C) Cache exists but stale → download from (latest_cached + 1 day) to today.
    """
    latest_cached = _get_latest_cached_date(trading_symbol, storage_unit)

    if latest_cached is None:
        # Case A: No data at all — download everything from the absolute start
        abs_start = MINUTE_DATA_START if storage_unit == "minutes" else DAY_DATA_START
        logger.info(
            f"No cache for {trading_symbol} ({storage_unit}). "
            f"Downloading full history from {abs_start}."
        )
        _download_and_save(
            instrument_key=instrument_key,
            trading_symbol=trading_symbol,
            storage_unit=storage_unit,
            dl_start=abs_start,
            dl_end=date.today(),
        )
        return

    if _cache_is_current(latest_cached, storage_unit):
        # Case B: Cache is fresh — nothing to do
        logger.info(
            f"Cache is current for {trading_symbol} ({storage_unit}). "
            f"Latest cached: {latest_cached}."
        )
        return

    # Case C: Cache exists but stale — fetch only the missing tail
    dl_start = latest_cached + timedelta(days=1)
    dl_end   = date.today()

    logger.info(
        f"Updating cache for {trading_symbol} ({storage_unit}): "
        f"{dl_start} → {dl_end}."
    )
    _download_and_save(
        instrument_key=instrument_key,
        trading_symbol=trading_symbol,
        storage_unit=storage_unit,
        dl_start=dl_start,
        dl_end=dl_end,
    )


def _get_latest_cached_date(trading_symbol: str, storage_unit: str) -> Optional[date]:
    """
    Inspect the local Parquet files and return the most recent date present.
    Returns None if no cache files exist.
    """
    if storage_unit == "minutes":
        symbol_dir = MINUTE_DIR / trading_symbol
        if not symbol_dir.exists():
            return None

        parquet_files = sorted(symbol_dir.glob("*.parquet"))
        if not parquet_files:
            return None

        # Read only the last file (latest month) and get the max timestamp
        last_file = parquet_files[-1]
        try:
            df = pd.read_parquet(last_file, columns=["open"])  # minimal read
            if df.empty:
                return None
            return df.index.max().date()
        except Exception as e:
            logger.warning(f"Could not read {last_file}: {e}. Treating as no cache.")
            return None

    elif storage_unit == "days":
        day_file = DAY_DIR / f"{trading_symbol}.parquet"
        if not day_file.exists():
            return None
        try:
            df = pd.read_parquet(day_file, columns=["open"])
            if df.empty:
                return None
            return df.index.max().date()
        except Exception as e:
            logger.warning(f"Could not read {day_file}: {e}. Treating as no cache.")
            return None

    return None


# ---------------------------------------------------------------------------
# Step 4b: Download in chunks and save immediately per chunk
# ---------------------------------------------------------------------------

def _download_and_save(
    instrument_key: str,
    trading_symbol: str,
    storage_unit:   str,
    dl_start:       date,
    dl_end:         date,
) -> None:
    """
    Generate date chunks, call Upstox API for each, and save each chunk
    to Parquet immediately (so partial progress is preserved on interruption).
    """
    chunks = _generate_chunks(dl_start, dl_end, storage_unit)
    total  = len(chunks)

    logger.info(
        f"Downloading {total} chunk(s) for {trading_symbol} "
        f"({storage_unit}, {dl_start} → {dl_end})."
    )

    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        logger.info(
            f"  Chunk {i}/{total}: {chunk_start} → {chunk_end}"
        )

        raw_candles = _fetch_one_chunk(
            instrument_key=instrument_key,
            storage_unit=storage_unit,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )

        if raw_candles:
            chunk_df = _candles_to_dataframe(raw_candles)
            _append_to_parquet(chunk_df, trading_symbol, storage_unit)
        else:
            logger.info(
                f"  No data returned for chunk {chunk_start} → {chunk_end} "
                f"(holiday/weekend range or no trades)."
            )

        # Respect Upstox rate limits between consecutive API calls
        if i < total:
            time.sleep(API_CALL_SLEEP)


def _generate_chunks(
    start:        date,
    end:          date,
    storage_unit: str,
) -> list[tuple[date, date]]:
    """
    Split [start, end] into API-safe chunks based on the storage unit's limits.

    Chunk sizes:
      minutes : 1 calendar month  (limit: 1 month for intervals 1–15 min)
      days    : 10 calendar years (limit: 1 decade)
    """
    chunks = []
    cursor = start

    if storage_unit == "minutes":
        # Monthly chunks: [1st of month, last day of month]
        while cursor <= end:
            # End of current month
            if cursor.month == 12:
                chunk_end = date(cursor.year + 1, 1, 1) - timedelta(days=1)
            else:
                chunk_end = date(cursor.year, cursor.month + 1, 1) - timedelta(days=1)

            chunk_end = min(chunk_end, end)
            chunks.append((cursor, chunk_end))

            # Move to first day of next month
            if cursor.month == 12:
                cursor = date(cursor.year + 1, 1, 1)
            else:
                cursor = date(cursor.year, cursor.month + 1, 1)

    elif storage_unit == "days":
        # 10-year chunks
        while cursor <= end:
            chunk_end = date(cursor.year + 10, cursor.month, cursor.day) - timedelta(days=1)
            chunk_end = min(chunk_end, end)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end + timedelta(days=1)

    return chunks


def _fetch_one_chunk(
    instrument_key: str,
    storage_unit:   str,
    chunk_start:    date,
    chunk_end:      date,
) -> list:
    """
    Call the Upstox HistoryV3Api for a single date chunk.

    Always fetches with interval=1 (base resolution).
    Uses "dummy_token" as the access token — Upstox historical endpoint
    works with any non-empty token string as per documentation and user
    confirmation. A real token will be substituted when available.

    Returns:
        list: Raw candle arrays from the API response, or [] on failure.
    """
    # Use real token if available via config, else fall back to dummy
    try:
        from broker.auth import auth_manager
        token = auth_manager.get_access_token() or "dummy_token"
    except Exception:
        token = "dummy_token"

    configuration = upstox_client.Configuration()
    configuration.access_token = token

    api_instance = upstox_client.HistoryV3Api(
        upstox_client.ApiClient(configuration)
    )

    try:
        response = api_instance.get_historical_candle_data1(
            instrument_key=instrument_key,
            unit=storage_unit,           # "minutes" or "days"
            interval="1",                # always fetch base resolution
            to_date=chunk_end.strftime("%Y-%m-%d"),
            from_date=chunk_start.strftime("%Y-%m-%d"),
        )

        # SDK returns an object; candles are in response.data.candles
        candles = response.data.candles if response and response.data else []
        return candles

    except Exception as e:
        logger.error(
            f"API error fetching {instrument_key} "
            f"({chunk_start} → {chunk_end}): {e}"
        )
        return []


def _candles_to_dataframe(candles: list) -> pd.DataFrame:
    """
    Convert raw API candle arrays to a typed, timezone-aware DataFrame.

    API returns each candle as:
    [timestamp_str, open, high, low, close, volume, oi]

    Output columns: open, high, low, close, volume, oi
    Output index  : datetime64[ns, Asia/Kolkata]
    """
    if not candles:
        return pd.DataFrame()

    rows = []
    for c in candles:
        rows.append({
            "timestamp": pd.Timestamp(c[0]),
            "open":      float(c[1]),
            "high":      float(c[2]),
            "low":       float(c[3]),
            "close":     float(c[4]),
            "volume":    int(c[5]),
            "oi":        int(c[6]),
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    df = df.set_index("timestamp").sort_index()

    # Cast to compact dtypes to minimise memory and Parquet file size
    for col, dtype in PARQUET_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    return df


# ---------------------------------------------------------------------------
# Step 4c: Parquet save/append helpers
# ---------------------------------------------------------------------------

def _append_to_parquet(
    new_df:         pd.DataFrame,
    trading_symbol: str,
    storage_unit:   str,
) -> None:
    """
    Merge new_df into the appropriate Parquet file(s) on disk.

    For minute data: one file per calendar month.
        Groups new_df rows by (year, month) and merges each group into
        its corresponding monthly file.
    For day data: one file for the whole instrument.
        Merges new_df into the single existing file (or creates it).
    """
    if new_df is None or new_df.empty:
        return

    if storage_unit == "minutes":
        _append_minute_parquet(new_df, trading_symbol)
    elif storage_unit == "days":
        _append_day_parquet(new_df, trading_symbol)


def _append_minute_parquet(df: pd.DataFrame, trading_symbol: str) -> None:
    """
    Save minute data into monthly Parquet files.
    Groups rows by calendar month, then merges each group with existing file.

    File path: data/ohlcv/minute/<SYMBOL>/<YYYY-MM>.parquet
    """
    symbol_dir = MINUTE_DIR / trading_symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    # Group by year-month
    df["_ym"] = df.index.to_period("M")
    for period, group in df.groupby("_ym"):
        group = group.drop(columns=["_ym"])
        file_path = symbol_dir / f"{period}.parquet"
        _merge_and_write(existing_path=file_path, new_df=group)


def _append_day_parquet(df: pd.DataFrame, trading_symbol: str) -> None:
    """
    Save daily data into the single per-instrument Parquet file.

    File path: data/ohlcv/daily/<SYMBOL>.parquet
    """
    DAY_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DAY_DIR / f"{trading_symbol}.parquet"
    _merge_and_write(existing_path=file_path, new_df=df)


def _merge_and_write(existing_path: Path, new_df: pd.DataFrame) -> None:
    """
    Merge new_df with existing Parquet file (if present) and write result.

    Merge strategy:
    - Concatenate old and new rows.
    - Drop duplicate timestamps (new data wins — keeps last occurrence).
    - Sort chronologically.
    - Enforce compact dtypes.
    - Write back as Parquet with snappy compression.
    """
    if existing_path.exists():
        try:
            existing_df = pd.read_parquet(existing_path)
        except Exception as e:
            logger.warning(
                f"Could not read {existing_path} for merging ({e}). "
                "Writing new data only."
            )
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    if existing_df.empty:
        merged = new_df
    else:
        merged = pd.concat([existing_df, new_df])
        # Drop duplicate index entries — keep last (newer data wins)
        merged = merged[~merged.index.duplicated(keep="last")]

    merged = merged.sort_index()

    # Re-apply compact dtypes after concat (concat can upcast to float64)
    for col, dtype in PARQUET_DTYPES.items():
        if col in merged.columns:
            merged[col] = merged[col].astype(dtype)

    merged.to_parquet(
        existing_path,
        compression="snappy",   # fast read/write, good compression
        engine="pyarrow",
    )
    logger.debug(f"Wrote {len(merged)} rows to {existing_path}")


# ---------------------------------------------------------------------------
# Step 5: Load from cache (selective — only months/files needed)
# ---------------------------------------------------------------------------

def _load_from_cache(
    trading_symbol: str,
    storage_unit:   str,
    query_start:    date,
    query_end:      date,
) -> pd.DataFrame:
    """
    Load only the Parquet files that overlap with [query_start, query_end].

    For minute data: loads only the monthly files whose month overlaps with
    the query range — avoids loading years of history when user asks for
    a 3-month window.

    For day data: loads the single file (it's small, fine to load all).
    Then filters to the query range.
    """
    if storage_unit == "minutes":
        return _load_minute_cache(trading_symbol, query_start, query_end)
    elif storage_unit == "days":
        return _load_day_cache(trading_symbol, query_start, query_end)
    return pd.DataFrame()


def _load_minute_cache(
    trading_symbol: str,
    query_start:    date,
    query_end:      date,
) -> pd.DataFrame:
    """
    Load monthly Parquet files that overlap [query_start, query_end].
    Only loads the relevant months — not the full history.
    """
    symbol_dir = MINUTE_DIR / trading_symbol
    if not symbol_dir.exists():
        return pd.DataFrame()

    # Determine which year-month periods we need
    needed_periods = set()
    cursor = query_start.replace(day=1)
    while cursor <= query_end:
        needed_periods.add(f"{cursor.year}-{cursor.month:02d}")
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)

    # Load only the matching files
    frames = []
    for period_str in sorted(needed_periods):
        file_path = symbol_dir / f"{period_str}.parquet"
        if file_path.exists():
            try:
                frames.append(pd.read_parquet(file_path))
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}. Skipping.")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()

    # Filter precisely to query range
    tz_start = pd.Timestamp(query_start, tz=IST)
    tz_end   = pd.Timestamp(query_end,   tz=IST) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[tz_start:tz_end]


def _load_day_cache(
    trading_symbol: str,
    query_start:    date,
    query_end:      date,
) -> pd.DataFrame:
    """Load the single daily Parquet file and slice to [query_start, query_end]."""
    file_path = DAY_DIR / f"{trading_symbol}.parquet"
    if not file_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}.")
        return pd.DataFrame()

    tz_start = pd.Timestamp(query_start, tz=IST)
    tz_end   = pd.Timestamp(query_end,   tz=IST) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[tz_start:tz_end]


# ---------------------------------------------------------------------------
# Step 6: Resample
# ---------------------------------------------------------------------------

# Maps (unit, interval) → pandas resample frequency string
def _build_resample_freq(unit: str, interval: int) -> str:
    """
    Build the pandas resample frequency string for the given unit and interval.

    Examples:
      ("minutes", 5)  → "5min"
      ("hours",   1)  → "1h"
      ("days",    1)  → "1B"   (business days — skip weekends automatically)
      ("weeks",   1)  → "1W"
      ("months",  1)  → "1ME"  (month end)
    """
    unit = unit.lower().strip()
    freq_map = {
        "minutes": f"{interval}min",
        "hours":   f"{interval}h",
        "days":    f"{interval}B",    # B = business day (Mon-Fri)
        "weeks":   f"{interval}W",
        "months":  f"{interval}ME",   # ME = month end
    }
    if unit not in freq_map:
        raise ValueError(
            f"Invalid unit '{unit}' for resample. "
            "Valid: minutes, hours, days, weeks, months."
        )
    return freq_map[unit]


def _resample(df: pd.DataFrame, unit: str, interval: int) -> pd.DataFrame:
    """
    Resample base-resolution (1-min or 1-day) OHLCV data into the requested
    unit and interval using standard OHLCV aggregation rules.

    For minute base data: first filter to NSE trading hours (9:15 AM – 3:30 PM IST)
    so that overnight gaps and weekends don't pollute candles.

    Aggregation:
      open   = first value in period
      high   = max value in period
      low    = min value in period
      close  = last value in period
      volume = sum (cumulative volume in period)
      oi     = last value (OI is a snapshot, not cumulative)
    """
    if df.empty:
        return df

    # For minute-based storage, filter to market hours before resampling
    unit_lower = unit.lower().strip()
    if unit_lower in ("minutes", "hours"):
        df = _filter_market_hours(df)

    freq = _build_resample_freq(unit, interval)

    resampled = df.resample(freq).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
        "oi":     "last",
    })

    # Drop periods with no trades (all NaN from agg of empty groups)
    resampled = resampled.dropna(subset=["open", "close"])

    # Re-apply compact dtypes after resample (resample returns float64)
    for col, dtype in PARQUET_DTYPES.items():
        if col in resampled.columns:
            resampled[col] = resampled[col].astype(dtype)

    return resampled


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows within NSE market hours: 09:15 to 15:30 IST.
    Applied before resampling minute data to avoid polluting candles
    with off-market timestamps.
    """
    if df.empty:
        return df

    times = df.index.time
    market_open  = pd.Timestamp("09:15:00").time()
    market_close = pd.Timestamp("15:30:00").time()

    mask = (times >= market_open) & (times <= market_close)
    return df.loc[mask]
