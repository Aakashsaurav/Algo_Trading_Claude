"""
data/parquet_store.py
----------------------
Handles reading and writing OHLCV data to/from Parquet files on disk.

WHY PARQUET (recap):
- Columnar format — reading only 'close' prices doesn't load open/high/low/volume.
- ~10x compression vs CSV (47M rows/year → ~800 MB instead of ~8 GB).
- Pandas reads it directly with zero conversion overhead.
- Perfect for time-series backtesting access patterns.

FILE ORGANISATION ON DISK:
    data/ohlcv/
    ├── daily/
    │   ├── NSE_EQ_INFY.parquet          ← All daily candles for Infosys
    │   ├── NSE_EQ_RELIANCE.parquet
    │   └── ...
    ├── weekly/
    │   ├── NSE_EQ_INFY.parquet
    │   └── ...
    └── minute/
        ├── NSE_EQ_INFY/
        │   ├── 2024-01.parquet          ← Jan 2024 minute candles
        │   ├── 2024-02.parquet          ← Feb 2024 minute candles
        │   └── ...
        └── NSE_EQ_RELIANCE/
            └── ...

FILENAME CONVENTION:
    instrument_key "NSE_EQ|INE009A01021" is stored as "NSE_EQ_INFY"
    (we use the symbol name, not the ISIN, for human readability)

USAGE:
    from data.parquet_store import parquet_store
    # Save
    parquet_store.save_daily(df, "NSE_EQ", "INFY")
    # Load
    df = parquet_store.load_daily("NSE_EQ", "INFY", from_date="2022-01-01")
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pq = None

from config import config


def _check_pyarrow():
    """Raise a clear error if pyarrow is not installed."""
    if not PYARROW_AVAILABLE:
        raise ImportError(
            "pyarrow is required for Parquet storage but is not installed.\n"
            "Install it with: pip install pyarrow"
        )

logger = logging.getLogger(__name__)

# Parquet compression algorithm — 'snappy' is fastest read/write
# 'gzip' gives better compression but slower. Snappy is the right tradeoff here.
COMPRESSION = "snappy"


def _make_filename(exchange: str, symbol: str) -> str:
    """
    Build a safe filename from exchange and symbol.

    e.g. exchange="NSE_EQ", symbol="INFY" → "NSE_EQ_INFY"

    We replace the pipe character '|' in instrument_keys and sanitise the name
    so it's always a valid filename on any OS.
    """
    # Remove any pipe characters and replace spaces with underscores
    safe_exchange = exchange.replace("|", "_").replace(" ", "_").upper()
    safe_symbol = symbol.replace("|", "_").replace(" ", "_").upper()
    return f"{safe_exchange}_{safe_symbol}"


class ParquetStore:
    """
    Read and write OHLCV data to Parquet files.

    All methods accept and return Pandas DataFrames with:
        index: datetime (timezone-aware, 'Asia/Kolkata')
        columns: open, high, low, close, volume, oi
    """

    def __init__(self):
        self.daily_dir = config.DAILY_DIR
        self.weekly_dir = config.WEEKLY_DIR
        self.minute_dir = config.MINUTE_DIR

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _validate_df(self, df: pd.DataFrame, label: str):
        """
        Basic validation before saving — catch problems early.

        Checks:
        - DataFrame is not empty
        - Required OHLCV columns exist
        - Index is a DatetimeIndex
        """
        if df is None or df.empty:
            raise ValueError(f"[{label}] Cannot save empty DataFrame.")

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"[{label}] Missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"[{label}] DataFrame index must be a DatetimeIndex.")

    def _merge_with_existing(
        self,
        new_df: pd.DataFrame,
        file_path: Path,
    ) -> pd.DataFrame:
        """
        Merge new data with existing Parquet file to avoid duplicates
        and preserve historical data.

        Strategy:
        - Load existing data from file.
        - Combine with new data.
        - Keep latest values for duplicate timestamps (new data wins).
        - Sort chronologically.
        """
        if not file_path.exists():
            return new_df

        try:
            existing_df = pq.read_table(file_path).to_pandas()

            # Restore datetime index
            if "datetime" in existing_df.columns:
                existing_df["datetime"] = pd.to_datetime(existing_df["datetime"], utc=True)
                existing_df["datetime"] = existing_df["datetime"].dt.tz_convert("Asia/Kolkata")
                existing_df.set_index("datetime", inplace=True)

            # Combine: new data overwrites existing for duplicate timestamps
            combined = pd.concat([existing_df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)

            logger.debug(
                f"Merged: {len(existing_df)} existing + {len(new_df)} new "
                f"= {len(combined)} total rows"
            )
            return combined

        except Exception as e:
            logger.warning(
                f"Could not load existing file for merge ({file_path.name}): {e}. "
                f"Will overwrite with new data."
            )
            return new_df

    def _save_to_parquet(self, df: pd.DataFrame, file_path: Path):
        """
        Write a DataFrame to a Parquet file.

        We convert the DatetimeIndex to a column before saving because
        Parquet handles timezone-aware datetime columns better than index.
        The index is restored on load.
        """
        _check_pyarrow()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert index to column for Parquet storage
        df_to_save = df.copy()
        df_to_save.index.name = "datetime"
        df_to_save = df_to_save.reset_index()

        # Ensure datetime is stored as UTC (standard practice)
        if df_to_save["datetime"].dt.tz is not None:
            df_to_save["datetime"] = df_to_save["datetime"].dt.tz_convert("UTC")
        else:
            df_to_save["datetime"] = pd.to_datetime(df_to_save["datetime"], utc=True)

        table = pa.Table.from_pandas(df_to_save, preserve_index=False)
        pq.write_table(table, file_path, compression=COMPRESSION)

    def _load_from_parquet(
        self,
        file_path: Path,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a Parquet file into a DataFrame with optional date filtering.

        Args:
            file_path: Path to the .parquet file.
            from_date: Filter rows >= this date "YYYY-MM-DD".
            to_date: Filter rows <= this date "YYYY-MM-DD".

        Returns:
            pd.DataFrame with DatetimeIndex (IST timezone), or empty DataFrame.
        """
        _check_pyarrow()
        if not file_path.exists():
            logger.debug(f"File not found: {file_path}")
            return pd.DataFrame()

        try:
            df = pq.read_table(file_path).to_pandas()

            # Restore datetime index
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata")
                df.set_index("datetime", inplace=True)

            # Apply date filters
            if from_date:
                start = pd.Timestamp(from_date, tz="Asia/Kolkata")
                df = df[df.index >= start]

            if to_date:
                end = pd.Timestamp(to_date, tz="Asia/Kolkata") + pd.Timedelta(days=1)
                df = df[df.index < end]

            return df

        except Exception as e:
            logger.error(f"Failed to load Parquet file {file_path}: {e}")
            return pd.DataFrame()

    # ── Daily OHLCV ───────────────────────────────────────────────────────────

    def save_daily(self, df: pd.DataFrame, exchange: str, symbol: str):
        """
        Save (or update) daily OHLCV data for a symbol.

        New data is merged with existing data — existing history is preserved.
        If the same date exists in both, the new data wins.

        Args:
            df (pd.DataFrame): Daily OHLCV DataFrame with DatetimeIndex.
            exchange (str): Exchange segment. e.g. "NSE_EQ"
            symbol (str): Trading symbol. e.g. "INFY"
        """
        self._validate_df(df, f"save_daily({symbol})")

        filename = _make_filename(exchange, symbol) + ".parquet"
        file_path = self.daily_dir / filename

        merged_df = self._merge_with_existing(df, file_path)
        self._save_to_parquet(merged_df, file_path)

        logger.info(f"✅ Saved daily OHLCV: {filename} ({len(merged_df)} rows total)")

    def load_daily(
        self,
        exchange: str,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load daily OHLCV data for a symbol.

        Args:
            exchange (str): Exchange segment. e.g. "NSE_EQ"
            symbol (str): Trading symbol. e.g. "INFY"
            from_date (str, optional): Start date filter "YYYY-MM-DD".
            to_date (str, optional): End date filter "YYYY-MM-DD".

        Returns:
            pd.DataFrame: Daily OHLCV data. Empty if no data found.

        Example:
            df = parquet_store.load_daily("NSE_EQ", "INFY", from_date="2022-01-01")
        """
        filename = _make_filename(exchange, symbol) + ".parquet"
        file_path = self.daily_dir / filename

        df = self._load_from_parquet(file_path, from_date, to_date)

        if df.empty:
            logger.warning(f"No daily data found for {exchange}:{symbol}")
        else:
            logger.debug(f"Loaded {len(df)} daily rows for {symbol}")

        return df

    def daily_exists(self, exchange: str, symbol: str) -> bool:
        """Check if daily data file exists for a symbol."""
        filename = _make_filename(exchange, symbol) + ".parquet"
        return (self.daily_dir / filename).exists()

    def get_daily_date_range(
        self,
        exchange: str,
        symbol: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Get the date range of stored daily data for a symbol.

        Returns:
            Tuple of (earliest_date, latest_date) as strings, or (None, None) if no data.
        """
        df = self.load_daily(exchange, symbol)
        if df.empty:
            return None, None
        return df.index[0].date().isoformat(), df.index[-1].date().isoformat()

    # ── Weekly OHLCV ──────────────────────────────────────────────────────────

    def save_weekly(self, df: pd.DataFrame, exchange: str, symbol: str):
        """Save (or update) weekly OHLCV data for a symbol."""
        self._validate_df(df, f"save_weekly({symbol})")

        filename = _make_filename(exchange, symbol) + ".parquet"
        file_path = self.weekly_dir / filename

        merged_df = self._merge_with_existing(df, file_path)
        self._save_to_parquet(merged_df, file_path)

        logger.info(f"✅ Saved weekly OHLCV: {filename} ({len(merged_df)} rows total)")

    def load_weekly(
        self,
        exchange: str,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load weekly OHLCV data for a symbol."""
        filename = _make_filename(exchange, symbol) + ".parquet"
        file_path = self.weekly_dir / filename
        return self._load_from_parquet(file_path, from_date, to_date)

    # ── 1-Minute OHLCV ───────────────────────────────────────────────────────

    def save_minute(self, df: pd.DataFrame, exchange: str, symbol: str):
        """
        Save (or update) 1-minute OHLCV data for a symbol.

        Minute data is stored in monthly Parquet files to keep file sizes
        manageable and speed up partial loads.

        File structure:
            data/ohlcv/minute/NSE_EQ_INFY/2024-01.parquet
                                          2024-02.parquet
                                          ...

        Args:
            df (pd.DataFrame): 1-minute OHLCV DataFrame with DatetimeIndex.
            exchange (str): Exchange segment.
            symbol (str): Trading symbol.
        """
        self._validate_df(df, f"save_minute({symbol})")

        symbol_dir = self.minute_dir / _make_filename(exchange, symbol)
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Split data into monthly chunks and save each month separately
        df_copy = df.copy()
        df_copy["month"] = df_copy.index.to_period("M").astype(str)  # e.g. "2024-01"

        months_saved = []
        for month_str, month_df in df_copy.groupby("month"):
            month_df = month_df.drop(columns=["month"])
            file_path = symbol_dir / f"{month_str}.parquet"

            merged = self._merge_with_existing(month_df, file_path)
            self._save_to_parquet(merged, file_path)
            months_saved.append(month_str)

        logger.info(
            f"✅ Saved minute OHLCV: {symbol} | "
            f"Months: {months_saved} | Total rows in batch: {len(df)}"
        )

    def load_minute(
        self,
        exchange: str,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load 1-minute OHLCV data for a symbol across the date range.

        Only reads the monthly files that overlap with the requested range
        — avoids loading years of data when you only need one week.

        Args:
            exchange (str): Exchange segment.
            symbol (str): Trading symbol.
            from_date (str, optional): Start date filter "YYYY-MM-DD".
            to_date (str, optional): End date filter "YYYY-MM-DD".

        Returns:
            pd.DataFrame: 1-minute OHLCV data.
        """
        symbol_dir = self.minute_dir / _make_filename(exchange, symbol)

        if not symbol_dir.exists():
            logger.warning(f"No minute data directory found for {exchange}:{symbol}")
            return pd.DataFrame()

        # Find all monthly files in the directory
        all_files = sorted(symbol_dir.glob("*.parquet"))

        if not all_files:
            return pd.DataFrame()

        # Filter which monthly files we actually need to read
        # e.g., if from_date="2024-02-15", we only need 2024-02.parquet onwards
        files_to_read = []
        for f in all_files:
            month_str = f.stem  # e.g. "2024-01"
            if from_date and month_str < from_date[:7]:  # "2024-01" < "2024-02"
                continue
            if to_date and month_str > to_date[:7]:
                continue
            files_to_read.append(f)

        if not files_to_read:
            logger.warning(f"No minute data files found for {symbol} in requested range.")
            return pd.DataFrame()

        # Load and combine selected monthly files
        chunks = []
        for f in files_to_read:
            df = self._load_from_parquet(f, from_date, to_date)
            if not df.empty:
                chunks.append(df)

        if not chunks:
            return pd.DataFrame()

        combined = pd.concat(chunks)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        logger.debug(f"Loaded {len(combined)} minute rows for {symbol}")
        return combined

    # ── Utility Methods ───────────────────────────────────────────────────────

    def list_available_symbols(self, interval: str = "daily") -> list[str]:
        """
        List all symbols that have data stored for a given interval.

        Args:
            interval (str): "daily", "weekly", or "minute".

        Returns:
            list[str]: Sorted list of symbol names (e.g. ["NSE_EQ_INFY", ...])
        """
        if interval == "daily":
            directory = self.daily_dir
            files = list(directory.glob("*.parquet"))
            return sorted([f.stem for f in files])

        elif interval == "weekly":
            directory = self.weekly_dir
            files = list(directory.glob("*.parquet"))
            return sorted([f.stem for f in files])

        elif interval == "minute":
            if not self.minute_dir.exists():
                return []
            return sorted([d.name for d in self.minute_dir.iterdir() if d.is_dir()])

        else:
            logger.warning(f"Unknown interval: {interval}. Use 'daily', 'weekly', or 'minute'.")
            return []

    def get_storage_summary(self) -> dict:
        """
        Get a summary of all stored data — useful for the dashboard's data status page.

        Returns:
            dict: Summary with counts and file sizes.
        """
        def dir_size_mb(directory: Path) -> float:
            if not directory.exists():
                return 0.0
            total = sum(f.stat().st_size for f in directory.rglob("*.parquet"))
            return round(total / (1024 * 1024), 2)

        daily_symbols = self.list_available_symbols("daily")
        minute_symbols = self.list_available_symbols("minute")
        weekly_symbols = self.list_available_symbols("weekly")

        return {
            "daily": {
                "symbol_count": len(daily_symbols),
                "size_mb": dir_size_mb(self.daily_dir),
            },
            "weekly": {
                "symbol_count": len(weekly_symbols),
                "size_mb": dir_size_mb(self.weekly_dir),
            },
            "minute": {
                "symbol_count": len(minute_symbols),
                "size_mb": dir_size_mb(self.minute_dir),
            },
            "total_size_mb": (
                dir_size_mb(self.daily_dir) +
                dir_size_mb(self.weekly_dir) +
                dir_size_mb(self.minute_dir)
            ),
        }


# ── Module-level singleton ────────────────────────────────────────────────────
parquet_store = ParquetStore()
