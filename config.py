"""
config.py
---------
Central configuration file for the Algo Trading System.

All settings are loaded from the .env file using python-dotenv.
Every other module imports from here — never hardcode values elsewhere.

Usage:
    from config import config
    print(config.API_KEY)
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

<<<<<<< HEAD
from backtester.models import _HERE

=======
>>>>>>> 8d072798ed841b92b7056b98b3d612023cbaf223
# ── Load the .env file from the project root ──────────────────────────────────
# This must happen before we read any os.getenv() values.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ── Helper: read a required env variable or crash with a clear message ────────
def _require(key: str) -> str:
    """
    Fetch a required environment variable.
    Raises a clear error if it's missing so the developer knows exactly what to fix.
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"[config.py] Required environment variable '{key}' is not set. "
            f"Please add it to your .env file."
        )
    return value


# ── AppConfig class — single object that holds all settings ───────────────────
class AppConfig:
    """
    Central config object. Import this anywhere in the project.

    Example:
        from config import config
        print(config.TOTAL_CAPITAL)
    """

    # --- Upstox API Credentials ---
    API_KEY: str = _require("UPSTOX_API_KEY")
    API_SECRET: str = _require("UPSTOX_API_SECRET")
    REDIRECT_URI: str = os.getenv("UPSTOX_REDIRECT_URI", "https://127.0.0.1:5000/")

    # Upstox API base URLs (v2)
    BASE_URL: str = "https://api.upstox.com"
    HFT_URL: str = "https://api-hft.upstox.com"   # Used for order placement
    AUTH_URL: str = "https://api.upstox.com/v2/login/authorization/token"

    # --- Token Storage ---
    TOKEN_FILE_PATH: Path = BASE_DIR / os.getenv("TOKEN_FILE_PATH", "broker/upstox/token.json")

    # --- Capital & Risk ---
    TOTAL_CAPITAL: float = float(os.getenv("TOTAL_CAPITAL", "500000"))
    MAX_PORTFOLIO_DRAWDOWN: float = float(os.getenv("MAX_PORTFOLIO_DRAWDOWN", "20.0"))
    DRAWDOWN_WARNING_LEVEL: float = 15.0    # Alert me at 15% before hard stop at 20%
    PER_TRADE_RISK_PERCENT: float = float(os.getenv("PER_TRADE_RISK_PERCENT", "1.5"))

    # --- Trading Mode ---
    # This is your safety gate. NEVER set to False unless fully tested with paper trading.
    PAPER_TRADE: bool = os.getenv("PAPER_TRADE", "True").lower() == "true"

    # --- Market Hours (IST) ---
    MARKET_OPEN_TIME: str = "09:15"
    MARKET_CLOSE_TIME: str = "15:30"
    INTRADAY_SQUAREOFF_TIME: str = "15:20"   # Hard close all MIS positions by this time

    # --- Data Storage Paths ---
    DATA_DIR: Path = BASE_DIR / "data"
    OHLCV_DIR: Path = DATA_DIR / "ohlcv"
    DAILY_DIR: Path = OHLCV_DIR / "daily"
    MINUTE_DIR: Path = OHLCV_DIR / "minute"
    WEEKLY_DIR: Path = OHLCV_DIR / "weekly"
    SQLITE_DIR: Path = DATA_DIR / "sqlite"
<<<<<<< HEAD
    BACKTEST_DIR: Path = BASE_DIR / "backtest_output"
    OUTPUT_TRADE = BACKTEST_DIR / "trade"
    OUTPUT_RAW   = BACKTEST_DIR / "raw_data"
    OUTPUT_CHART = BACKTEST_DIR / "chart"
=======
>>>>>>> 8d072798ed841b92b7056b98b3d612023cbaf223

    # --- Database File Paths ---
    METADATA_DB: Path = SQLITE_DIR / "metadata.db"
    TRADES_DB: Path = SQLITE_DIR / "trades.db"
    STRATEGIES_DB: Path = SQLITE_DIR / "strategies.db"

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE: Path = BASE_DIR / "logs" / "app.log"

    # --- Notifications ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # --- NSE / Instrument Settings ---
    # Upstox provides a daily instrument dump at this URL
    INSTRUMENT_KEY_URL: str = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
    INSTRUMENT_KEY_PATH: Path = BASE_DIR / "broker" / "upstox" / "complete_instru_list.json"

    def __init__(self):
        """Create all required directories on startup."""
        self._create_directories()

    def _create_directories(self):
        """
        Ensure all required folders exist.
        Called automatically when config is first loaded.
        """
        dirs = [
            self.DATA_DIR,
            self.OHLCV_DIR,
            self.DAILY_DIR,
            self.MINUTE_DIR,
            self.WEEKLY_DIR,
            self.SQLITE_DIR,
            BASE_DIR / "logs",
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def display_summary(self):
        """Print a non-sensitive config summary for startup verification."""
        print("=" * 55)
        print("  ALGO TRADING SYSTEM — CONFIG SUMMARY")
        print("=" * 55)
        print(f"  Paper Trade Mode : {'✅ YES (safe)' if self.PAPER_TRADE else '🔴 NO — LIVE TRADING'}")
        print(f"  Total Capital    : ₹{self.TOTAL_CAPITAL:,.0f}")
        print(f"  Per-Trade Risk   : {self.PER_TRADE_RISK_PERCENT}%")
        print(f"  Max Drawdown     : {self.MAX_PORTFOLIO_DRAWDOWN}%")
        print(f"  Data Directory   : {self.DATA_DIR}")
        print(f"  Log Level        : {self.LOG_LEVEL}")
        print(f"  Redirect URI     : {self.REDIRECT_URI}")
        print("=" * 55)


# ── Logging Setup ─────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    """
    Configure application-wide logging.

    Logs go to BOTH:
    - Console (so you see output while running)
    - logs/app.log (persistent file for debugging)

    Returns the root logger. All modules use:
        import logging
        logger = logging.getLogger(__name__)
    """
    # Create the logs directory if it doesn't exist
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    # Format: timestamp | level | module name | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid adding duplicate handlers if setup_logging() is called multiple times
    if not root_logger.handlers:

        # Console handler — prints to terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler — writes to logs/app.log
        file_handler = logging.FileHandler(BASE_DIR / "logs" / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


# ── Singleton: create one config instance for the whole app ───────────────────
# All modules import this single object:
#   from config import config
config = AppConfig()
logger = setup_logging()
