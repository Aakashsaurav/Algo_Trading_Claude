"""
broker/upstox/instrument_manager.py
------------------------------
Instrument key lookup for all Upstox-traded securities.

SUPPORTED INSTRUMENT TYPES:
    EQUITY  - Equities        (NSE_EQ, BSE_EQ)
    INDEX   - Indices         (NSE_INDEX, BSE_INDEX)
    FUTSTK  - Stock Futures   (NSE_FO, BSE_FO)
    FUTIDX  - Index Futures   (NSE_FO, BSE_FO)
    FUTCOM  - Commodity Fut.  (MCX_FO, NSE_COM)
    FUTCUR  - Currency Fut.   (NCD_FO, BCD_FO)
    FUTIRT  - IR Futures      (BCD_FO)
    OPTSTK  - Stock Options   (NSE_FO, BSE_FO)
    OPTIDX  - Index Options   (NSE_FO, BSE_FO)
    OPTCOM  - Commodity Opts  (NSE_COM, MCX_FO)
    OPTCUR  - Currency Opts   (NCD_FO, BCD_FO)
    OPTIRD  - IR Options      (BCD_FO)

USAGE:
    from broker.instrument_manager import get_instrument_key

    # Equity
    key = get_instrument_key("EQUITY", "NSE", "INFY")

    # Stock Future
    key = get_instrument_key("FUTSTK", "NSE", "RELIANCE", expiry="30MAR26")

    # Index Call Option
    key = get_instrument_key(
        "OPTIDX", "NSE", "NIFTY",
        option_type="CE", expiry="30MAR26", strike=25500
    )
    #Stock Put Option
    key = get_instrument_key_improved(
            instrument_type="OPTSTK",
            exchange="NSE",
            trading_symbol="INFY",
            option_type="PE",
            expiry="24FEB26",
            strike=1200
    )

    #Commodity Option
    key = get_instrument_key_improved(
            instrument_type="OPTCOM",
            exchange="MCX",
            trading_symbol="GOLD",
            option_type="CE",
            expiry="27FEB26",
            strike=118300
    )
"""

import gzip
import json
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from config import config

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#complete_instru_list = (
#    'https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz'
#) --- IGNORE --- moved to config.py

complete_instru_list = config.INSTRUMENT_KEY_URL

# DATA_DIR uses config.DATA_DIR (project-level data/ folder).
# Do NOT use Path(__file__).parent / 'data' here — that would create
# broker/data/ instead of the intended project-level data/ directory.
DATA_DIR = config.DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

INSTRUMENT_DATA_FILE = config.INSTRUMENT_KEY_PATH
INSTRUMENT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------

def download_and_save_instrument_list(force_download: bool = False) -> list:
    """
    Download instrument list from Upstox URL and save locally.
    Check local cache first before downloading.
    Re-download if the file was last modified before today's 7 AM.

    Args:
        force_download (bool): If True, download even if local file exists.

    Returns:
        list: Parsed instrument data (list of dicts).
    """
    # Check if local file exists and is recent enough
    if INSTRUMENT_DATA_FILE.exists() and not force_download:
        # Get file modification time as timezone-aware datetime (IST)
        ist_tz = ZoneInfo("Asia/Kolkata")
        file_mtime = datetime.fromtimestamp(INSTRUMENT_DATA_FILE.stat().st_mtime, tz=ist_tz)
        today_7am_ist = datetime.now(tz=ist_tz).replace(hour=7, minute=0, second=0, microsecond=0)

        if file_mtime >= today_7am_ist:
            logger.info(f"Loading instrument data from local cache: {INSTRUMENT_DATA_FILE}")
            with open(INSTRUMENT_DATA_FILE, 'r') as f:
                return json.load(f)
        else:
            logger.info(
                f"File last modified at {file_mtime}, which is before today's 7 AM IST. "
                "Re-downloading..."
            )

    logger.info(f"Downloading instrument list from: {complete_instru_list}")
    try:
        response = requests.get(complete_instru_list, timeout=30)
        response.raise_for_status()

        decompressed_data = gzip.decompress(response.content)
        instrument_data = json.loads(decompressed_data.decode('utf-8'))

        with open(INSTRUMENT_DATA_FILE, 'w') as f:
            json.dump(instrument_data, f, indent=2)

        logger.info(
            f"Instrument list downloaded and saved to: {INSTRUMENT_DATA_FILE} "
            f"({len(instrument_data):,} instruments)"
        )
        return instrument_data

    except Exception as e:
        logger.error(f"Error downloading instrument list: {e}")
        if INSTRUMENT_DATA_FILE.exists():
            logger.warning("Download failed. Using local cache as fallback.")
            with open(INSTRUMENT_DATA_FILE, 'r') as f:
                return json.load(f)
        raise


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_instrument_key(
    instrument_type: str,
    exchange: str,
    trading_symbol: str,
    option_type: Optional[str] = None,
    expiry: Optional[str] = None,
    strike: Optional[float] = None
) -> str:
    """
    Get the Upstox instrument_key for any tradeable security.

    Logic is identical to instrument_key.py from the project.
    Only print() calls have been replaced with logger calls.

    Args:
        instrument_type (str): Type of instrument (case-insensitive).
                               e.g. "EQUITY", "FUTSTK", "OPTIDX"
        exchange (str):        Exchange - NSE, BSE, MCX (case-insensitive).
        trading_symbol (str):  Symbol or underlying (case-insensitive).
                               e.g. "INFY", "NIFTY", "RELIANCE"
        option_type (str):     For options only - "CE" or "PE".
        expiry (str):          For F&O - date in DDMONYY format.
                               e.g. "27MAR26", "24APR26"
        strike (float):        For options only - strike price.

    Returns:
        str: Upstox instrument_key. e.g. "NSE_EQ|INE009A01021"

    Raises:
        ValueError: If required parameters are missing or instrument not found.
    """

    # Segment search strategy mapping (unchanged from instrument_key.py)
    segment_search_strategy = {
        'EQUITY': [
            {'segments': ['NSE_EQ', 'BSE_EQ'], 'data_instr_type': None, 'asset_type': None}
        ],
        'INDEX': [
            {'segments': ['NSE_INDEX', 'BSE_INDEX'], 'data_instr_type': 'INDEX', 'asset_type': None}
        ],
        'FUTSTK': [
            {'segments': ['NSE_FO', 'BSE_FO'], 'data_instr_type': 'FUT', 'asset_type': 'EQUITY'}
        ],
        'FUTIDX': [
            {'segments': ['NSE_FO', 'BSE_FO'], 'data_instr_type': 'FUT', 'asset_type': 'INDEX'}
        ],
        'FUTCOM': [
            {'segments': ['MCX_FO', 'NSE_COM'], 'data_instr_type': 'FUT', 'asset_type': 'COM'}
        ],
        'FUTCUR': [
            {'segments': ['NCD_FO', 'BCD_FO'], 'data_instr_type': 'FUT', 'asset_type': 'CUR'}
        ],
        'FUTIRT': [
            {'segments': ['BCD_FO'], 'data_instr_type': 'FUT', 'asset_type': 'IRD'}
        ],
        'OPTSTK': [
            {'segments': ['NSE_FO', 'BSE_FO'], 'data_instr_type': ['CE', 'PE'], 'asset_type': 'EQUITY'}
        ],
        'OPTIDX': [
            {'segments': ['NSE_FO', 'BSE_FO'], 'data_instr_type': ['CE', 'PE'], 'asset_type': 'INDEX'}
        ],
        'OPTCOM': [
            {'segments': ['NSE_COM', 'MCX_FO'], 'data_instr_type': ['CE', 'PE'], 'asset_type': 'COM'}
        ],
        'OPTCUR': [
            {'segments': ['NCD_FO', 'BCD_FO'], 'data_instr_type': ['CE', 'PE'], 'asset_type': 'CUR'}
        ],
        'OPTIRD': [
            {'segments': ['BCD_FO'], 'data_instr_type': ['CE', 'PE'], 'asset_type': 'IRD'}
        ]
    }

    # Segment -> exchange mapping (handles edge cases like NCD_FO)
    segment_exchange_map = {
        'NSE_EQ': 'NSE', 'BSE_EQ': 'BSE',
        'NSE_INDEX': 'NSE', 'BSE_INDEX': 'BSE',
        'NSE_FO': 'NSE', 'BSE_FO': 'BSE',
        'MCX_FO': 'MCX',
        'NSE_COM': 'NSE',
        'NCD_FO': 'NSE',   # Special case: segment=NCD_FO but exchange=NSE
        'BCD_FO': 'BSE'    # Special case: segment=BCD_FO but exchange=BSE
    }

    # Download / load instrument data
    instrument_data = download_and_save_instrument_list()

    # Normalise inputs
    instrument_type = instrument_type.upper().strip()
    exchange        = exchange.upper().strip()
    trading_symbol  = trading_symbol.upper().strip()

    if option_type:
        option_type = option_type.upper().strip()
    if expiry:
        expiry = expiry.upper().strip()
    if strike is not None:
        strike = float(strike)

    # Validate instrument type
    if instrument_type not in segment_search_strategy:
        raise ValueError(
            f"Unknown instrument type: {instrument_type}. "
            f"Supported types: {', '.join(segment_search_strategy.keys())}"
        )

    # Validate required parameters for options
    if instrument_type in ['OPTSTK', 'OPTIDX', 'OPTCUR', 'OPTCOM', 'OPTIRT']:
        if not option_type or not expiry or strike is None:
            raise ValueError(
                f"For {instrument_type}, option_type, expiry, and strike are required. "
                f"Received: option_type={option_type}, expiry={expiry}, strike={strike}"
            )

    # Validate required parameters for futures
    if instrument_type in ['FUTSTK', 'FUTIDX', 'FUTCOM', 'FUTCUR', 'FUTIRT']:
        if not expiry:
            raise ValueError(f"Expiry is required for {instrument_type}")

    search_strategies = segment_search_strategy[instrument_type]

    # Helper: convert DDMONYY expiry string to millisecond timestamp range
    def get_expiry_timestamp_range(expiry_str: str):
        try:
            date_obj = datetime.strptime(expiry_str, '%d%b%y')
            timestamp_ms = int(date_obj.timestamp() * 1000)
            # 1 hour before to 24 hours after midnight of expiry date
            return timestamp_ms - 3600000, timestamp_ms + 86400000
        except ValueError:
            return None, None

    # Search loop (unchanged from instrument_key.py)
    for strategy in search_strategies:
        segments_to_search = strategy['segments']
        data_instr_types   = strategy['data_instr_type']
        asset_type_filter  = strategy['asset_type']

        # Normalise data_instr_types to list for consistent handling
        if data_instr_types is None:
            data_instr_types = [None]
        elif isinstance(data_instr_types, str):
            data_instr_types = [data_instr_types]
        elif not isinstance(data_instr_types, list):
            data_instr_types = [data_instr_types]

        for instrument in instrument_data:

            # Filter by segment
            if instrument.get('segment') not in segments_to_search:
                continue

            # Verify exchange matches expected exchange for this segment
            expected_exchange = segment_exchange_map.get(instrument.get('segment'))
            if expected_exchange != exchange:
                continue

            # Filter by instrument_type field (EQ, FUT, CE, PE, INDEX)
            instr_type = instrument.get('instrument_type', '').upper()
            if data_instr_types != [None]:
                if instr_type not in data_instr_types:
                    continue

            # Filter by asset_type (EQUITY, INDEX, COM, CUR, IRD)
            if asset_type_filter:
                if instrument.get('asset_type') != asset_type_filter:
                    continue

            # EQUITY / INDEX: match trading_symbol directly
            if instrument_type in ['EQUITY', 'INDEX']:
                if instrument.get('trading_symbol', '').upper() == trading_symbol:
                    key = instrument.get('instrument_key', '')
                    logger.info(
                        f"Found {instrument_type}: {trading_symbol} -> {key}"
                    )
                    return key

            # FUTURES: match asset_symbol + expiry timestamp
            elif instrument_type in ['FUTSTK', 'FUTIDX', 'FUTCOM', 'FUTCUR', 'FUTIRT']:
                asset_sym = instrument.get('asset_symbol', '').upper()

                if trading_symbol != asset_sym:
                    continue

                exp_min, exp_max = get_expiry_timestamp_range(expiry)
                if exp_min is None:
                    raise ValueError(
                        f"Invalid expiry format: {expiry}. "
                        "Use format: DDMONYY (e.g., 24FEB26)"
                    )

                instr_expiry = instrument.get('expiry', 0)
                if isinstance(instr_expiry, (int, float)):
                    if exp_min <= instr_expiry <= exp_max:
                        key = instrument.get('instrument_key', '')
                        logger.info(
                            f"Found {instrument_type}: "
                            f"{trading_symbol} expiry={expiry} -> {key}"
                        )
                        return key

            # OPTIONS: match asset_symbol + option_type + strike + expiry
            elif instrument_type in ['OPTSTK', 'OPTIDX', 'OPTCOM', 'OPTCUR', 'OPTIRD']:
                # option_type (CE/PE) stored in instrument_type field of the JSON data
                instr_option_type = instrument.get('instrument_type', '').upper()
                if instr_option_type != option_type:
                    continue

                asset_sym = instrument.get('asset_symbol', '').upper()
                if trading_symbol != asset_sym:
                    continue

                # Strike price with floating-point tolerance
                strike_price = instrument.get('strike_price', 0)
                if abs(float(strike_price) - float(strike)) > 0.01:
                    continue

                exp_min, exp_max = get_expiry_timestamp_range(expiry)
                if exp_min is None:
                    raise ValueError(
                        f"Invalid expiry format: {expiry}. "
                        "Use format: DDMONYY (e.g., 24FEB26)"
                    )

                instr_expiry = instrument.get('expiry', 0)
                if isinstance(instr_expiry, (int, float)):
                    if exp_min <= instr_expiry <= exp_max:
                        key = instrument.get('instrument_key', '')
                        logger.info(
                            f"Found {instrument_type}: {trading_symbol} "
                            f"{option_type} strike={strike} expiry={expiry} -> {key}"
                        )
                        return key

    # Nothing found — build a helpful error message
    searched_segments = []
    for strategy in search_strategies:
        searched_segments.extend(strategy['segments'])

    raise ValueError(
        f"No {instrument_type} found for {trading_symbol} on {exchange}. "
        f"Searched segments: {searched_segments}. "
        f"Additional filters: expiry={expiry}, strike={strike}, option_type={option_type}"
    )
