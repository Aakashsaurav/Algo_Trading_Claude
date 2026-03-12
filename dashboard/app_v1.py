"""
dashboard/app.py
-----------------
FastAPI web application — Algorithmic Trading Dashboard backend.

IMPORTANT DESIGN DECISIONS
===========================
1. Location: lives inside dashboard/ (not web/)
2. Data retrieval: uses broker/upstox/data_manager.py (NOT parquet_store)
3. Strategy auto-discovery: scans strategies/ folder at startup — new strategy
   files appear in the dashboard automatically (no code change needed here)
4. HTML templates served from dashboard/templates/ directory
5. Static files served from dashboard/static/ directory

FOLDER STRUCTURE (relative to algo_trading/ root)
==================================================
  algo_trading/
  ├── dashboard/
  │   ├── app.py              ← this file
  │   ├── static/
  │   │   ├── css/main.css
  │   │   └── js/
  │   │       ├── code_editor.js
  │   │       ├── chart.js
  │   │       ├── backtest_results.js
  │   │       ├── screener_table.js
  │   │       └── live_bot_panel.js
  │   └── templates/
  │       ├── base.html
  │       ├── strategy_builder.html
  │       ├── backtester.html
  │       ├── screener.html
  │       └── live_bot.html

HOW TO RUN
==========
  cd algo_trading
  uvicorn dashboard.app:app --reload --port 8080
  Open http://localhost:8080 in your browser.

DEPENDENCIES
============
  pip install fastapi uvicorn python-multipart aiofiles jinja2
"""

import base64
import importlib
import inspect
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# =============================================================================
# Path Setup
# =============================================================================
# dashboard/app.py -> parent is dashboard/ -> parent.parent is algo_trading/
_DASHBOARD_DIR = Path(__file__).resolve().parent    # algo_trading/dashboard/
_ROOT          = _DASHBOARD_DIR.parent              # algo_trading/

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard")

# =============================================================================
# Output Directories
# =============================================================================
OUTPUT_TRADE = _ROOT / "strategies" / "output" / "trade"
OUTPUT_RAW   = _ROOT / "strategies" / "output" / "raw_data"
OUTPUT_CHART = _ROOT / "strategies" / "output" / "chart"
SCREENER_OUT = _ROOT / "screener"   / "output"

for _d in (OUTPUT_TRADE, OUTPUT_RAW, OUTPUT_CHART, SCREENER_OUT):
    _d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Algo Trading Dashboard",
    description="Backtester · Screener · Live Trading Bot",
    version="2.0.0",
    docs_url="/api/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (CSS / JS)
_STATIC_DIR = _DASHBOARD_DIR / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Serve saved backtest chart PNGs
app.mount("/charts", StaticFiles(directory=str(OUTPUT_CHART)), name="charts")

# Jinja2 templates
_TEMPLATES_DIR = _DASHBOARD_DIR / "templates"
_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# =============================================================================
# Strategy Auto-Discovery
# =============================================================================

def _discover_strategies() -> Dict[str, Dict]:
    """
    Scan the strategies/ package recursively and collect every class that
    inherits from BaseStrategy (but is not BaseStrategy itself).

    WHY AUTO-DISCOVERY?
    -------------------
    The user should be able to drop a new .py file into strategies/momentum/
    and see it appear in the dashboard immediately — without editing app.py.

    HOW IT WORKS
    ------------
    1. Walk every .py file under strategies/ with rglob
    2. Convert file path → dotted module name (e.g. strategies.momentum.ma_crossover)
    3. importlib.import_module() loads it
    4. inspect.isclass() + issubclass(BaseStrategy) filters the classes
    5. Each strategy class can define class-level attributes:
         PARAM_SCHEMA  - list of param dicts (name, type, default, min, max, options)
         DESCRIPTION   - one-line description string
         CATEGORY      - string e.g. "Trend Following"

    RETURNS
    -------
    dict: {ClassName: {class, description, params, category}}
    """
    registry: Dict[str, Dict] = {}

    try:
        from strategies.base_strategy_github import BaseStrategy as _Base
    except ImportError:
        logger.warning("strategies.base_strategy.BaseStrategy not found — "
                       "strategy auto-discovery disabled")
        return registry

    strategies_pkg = _ROOT / "strategies"
    if not strategies_pkg.exists():
        logger.warning(f"strategies/ folder not found at {strategies_pkg}")
        return registry

    for py_file in sorted(strategies_pkg.rglob("*.py")):
        # Skip __init__.py, __pycache__, private files
        if py_file.name.startswith("_"):
            continue

        # e.g. strategies/momentum/ma_crossover.py -> strategies.momentum.ma_crossover
        rel         = py_file.relative_to(_ROOT)
        module_name = ".".join(rel.with_suffix("").parts)

        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            logger.debug(f"Could not import {module_name}: {exc}")
            continue

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, _Base)
                and obj is not _Base
                # Avoid double-registering: only register where the class is defined
                and getattr(obj, "__module__", "") == module_name
            ):
                params   = getattr(obj, "PARAM_SCHEMA", [])
                desc     = getattr(obj, "DESCRIPTION",
                                   (obj.__doc__ or f"{attr_name} strategy").strip().split("\n")[0])
                category = getattr(obj, "CATEGORY", "Custom")

                registry[attr_name] = {
                    "class":       obj,
                    "description": desc,
                    "params":      params,
                    "category":    category,
                }
                logger.info(f"Discovered: {attr_name} (module: {module_name})")

    return registry


# Build registry once at import time
_STRATEGY_REGISTRY: Dict[str, Dict] = _discover_strategies()


def _get_strategy(name: str, params: Dict[str, Any]):
    """
    Retrieve a strategy from the registry and instantiate it with params.

    Edge cases:
    - Unknown name    -> HTTP 400 with helpful list of available strategies
    - Wrong param key -> filtered out (only valid constructor params are passed)
    - TypeError       -> HTTP 422 with details
    """
    if name not in _STRATEGY_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Strategy '{name}' not found. "
                f"Available: {sorted(_STRATEGY_REGISTRY.keys())}"
            ),
        )
    cls = _STRATEGY_REGISTRY[name]["class"]

    # Only pass kwargs that the constructor actually accepts
    sig         = inspect.signature(cls.__init__)
    safe_params = {
        k: v for k, v in params.items()
        if k in sig.parameters and k != "self"
    }

    try:
        return cls(**safe_params)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=f"Strategy param error: {exc}")


# =============================================================================
# Data Loading — via DataManager (single source of truth for OHLCV)
# =============================================================================

def _load_ohlcv(
    symbol: str,
    timeframe: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """
    Load OHLCV data using broker/upstox/data_manager.py.

    DataManager is the ONLY authorised way to retrieve market data in this
    application.  It handles:
      - Reading from the local Parquet cache (fast, offline)
      - Downloading from Upstox API when cache is stale / missing
      - Column normalisation

    FALLBACK BEHAVIOUR
    ------------------
    If DataManager is not available (ImportError) or returns insufficient data,
    synthetic OHLCV is generated so the dashboard remains usable during
    development without a live Upstox token.

    Parameters
    ----------
    symbol    : NSE trading symbol, e.g. "RELIANCE"
    timeframe : "daily" or "minute"
    from_date : "YYYY-MM-DD"
    to_date   : "YYYY-MM-DD" or "" (empty = today)
    """
    to = to_date if to_date else pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        from broker.upstox.data_manager import DataManager

        dm   = DataManager()
        unit = "days" if timeframe == "daily" else "minutes"

        df = dm.get_ohlcv(
            symbol    = symbol,
            unit      = unit,
            from_date = from_date,
            to_date   = to,
        )

        if df is not None and len(df) >= 50:
            # Normalise column names
            df.columns = [c.lower() for c in df.columns]
            required = {"open", "high", "low", "close"}
            missing  = required - set(df.columns)
            if missing:
                raise ValueError(f"DataManager missing columns: {missing}")
            if "volume" not in df.columns:
                df["volume"] = 0
            return df.sort_index()

        logger.warning(
            f"DataManager returned {len(df) if df is not None else 0} bars for "
            f"{symbol} — need ≥50. Using synthetic fallback."
        )

    except ImportError:
        logger.warning(
            "broker.upstox.data_manager not importable — using synthetic data. "
            "Ensure DataManager exists at broker/upstox/data_manager.py"
        )
    except Exception as exc:
        logger.warning(f"DataManager error for {symbol}: {exc} — using synthetic data")

    # Synthetic fallback
    return _synthetic_ohlcv(symbol, from_date, to)


def _synthetic_ohlcv(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Generate deterministic synthetic OHLCV for demo / CI / development.

    The random seed is derived from the symbol name so results are
    reproducible across runs (same symbol → same data).
    """
    np.random.seed(abs(hash(symbol)) % 10_000)

    end   = pd.Timestamp(to_date) if to_date else pd.Timestamp.today()
    start = pd.Timestamp(from_date)
    dates = pd.bdate_range(start, end)
    n     = len(dates)

    # Guarantee enough bars for indicator warm-up (EMA 200 needs at least 200)
    if n < 300:
        dates = pd.bdate_range(end - pd.Timedelta(days=500), end)
        n     = len(dates)

    base_price = 200 + abs(hash(symbol)) % 3_000
    returns    = np.random.randn(n) * 0.012          # ~1.2% daily volatility
    close      = base_price * np.cumprod(1 + returns)
    close      = np.maximum(close, 5.0)              # price floor

    noise = np.abs(np.random.randn(n) * close * 0.005)
    open_ = close + np.random.randn(n) * noise * 0.3
    high  = np.maximum(open_, close) + noise * 0.6
    low   = np.minimum(open_, close) - noise * 0.6
    low   = np.maximum(low, 1.0)

    return pd.DataFrame(
        {
            "open":   open_.astype(float),
            "high":   high.astype(float),
            "low":    low.astype(float),
            "close":  close.astype(float),
            "volume": np.random.randint(300_000, 8_000_000, n).astype(int),
        },
        index=dates,
    )


def _list_available_symbols() -> List[str]:
    """
    Return symbols that have local OHLCV data via DataManager.
    Falls back to a hardcoded Nifty 50 sample if DataManager unavailable.
    """
    try:
        from broker.upstox.data_manager import DataManager
        dm      = DataManager()
        symbols = dm.list_available_symbols(unit="days")
        if symbols:
            return sorted(symbols)
    except Exception as exc:
        logger.debug(f"Could not list symbols from DataManager: {exc}")

    # Hardcoded fallback — representative Nifty 500 sample
    return [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "WIPRO", "BAJFINANCE", "TITAN", "ASIANPAINT",
        "AXISBANK", "KOTAKBANK", "LT", "ITC", "MARUTI",
        "SUNPHARMA", "ULTRACEMCO", "NESTLEIND", "ADANIENT", "POWERGRID",
    ]


# =============================================================================
# Pydantic Request Models
# =============================================================================

class BacktestRequest(BaseModel):
    """All parameters the UI sends when the user clicks 'Run Backtest'."""

    symbol:            str            = Field("INFY")
    strategy_name:     str            = Field("EMACrossover")
    strategy_params:   Dict[str, Any] = Field(default_factory=dict)
    timeframe:         str            = Field("daily", description="'daily' or 'minute'")
    from_date:         str            = Field("2020-01-01")
    to_date:           str            = Field("",      description="YYYY-MM-DD or blank for today")
    initial_capital:   float          = Field(500_000, ge=10_000)
    capital_risk_pct:  float          = Field(0.02,   ge=0.001, le=0.05)
    segment:           str            = Field("EQUITY_DELIVERY")
    allow_shorting:    bool           = Field(False)
    max_positions:     int            = Field(5,       ge=1, le=50)
    lot_size:          int            = Field(1,       ge=1)
    order_type:        str            = Field("MARKET")
    stop_loss_pct:     float          = Field(0.0,     ge=0.0)
    trailing_stop_pct: float          = Field(0.0,     ge=0.0)
    use_trailing_stop: bool           = Field(False)
    save_trade_log:    bool           = Field(False)
    save_raw_data:     bool           = Field(False)
    save_chart:        bool           = Field(True)
    run_label:         str            = Field("dashboard_run")


class OptimizeRequest(BaseModel):
    """Parameters for the parameter optimizer panel."""

    symbol:          str                  = Field("INFY")
    strategy_name:   str                  = Field("EMACrossover")
    param_grid:      Dict[str, List[Any]] = Field(
        default={"fast_period": [5, 9, 13], "slow_period": [21, 34, 50]},
    )
    metric:          str   = Field("Sharpe Ratio")
    method:          str   = Field("grid", description="'grid' or 'random'")
    n_random:        int   = Field(30,    ge=1, le=500)
    timeframe:       str   = Field("daily")
    from_date:       str   = Field("2020-01-01")
    to_date:         str   = Field("")
    initial_capital: float = Field(500_000, ge=10_000)
    segment:         str   = Field("EQUITY_DELIVERY")


class ScreenerRequest(BaseModel):
    """Parameters for the screener scan panel."""

    strategy_name:   str            = Field("RSIMeanReversion")
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    signal_type:     int            = Field(1, ge=-1, le=1)
    min_volume:      float          = Field(200_000, ge=0)
    min_price:       float          = Field(50.0,    ge=0)
    max_results:     int            = Field(30,      ge=1, le=500)
    rank_by:         str            = Field("close")
    timeframe:       str            = Field("daily")
    from_date:       str            = Field("2022-01-01")


# =============================================================================
# HTML Page Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse, summary="Strategy Builder")
async def strategy_builder_page(request: Request):
    """Main page: Monaco code editor + backtest settings + run buttons."""
    return templates.TemplateResponse(
        "strategy_builder.html",
        {"request": request, "title": "Strategy Builder"},
    )


@app.get("/backtester", response_class=HTMLResponse)
async def backtester_page(request: Request):
    """Backtester results: candlestick chart + metrics + trade log."""
    return templates.TemplateResponse(
        "backtester.html",
        {"request": request, "title": "Backtester"},
    )


@app.get("/screener", response_class=HTMLResponse)
async def screener_page(request: Request):
    """Screener: strategy signal scan across Nifty 500 / F&O universe."""
    return templates.TemplateResponse(
        "screener.html",
        {"request": request, "title": "Screener"},
    )


@app.get("/live-bot", response_class=HTMLResponse)
async def live_bot_page(request: Request):
    """Live Bot: real-time P&L, position monitor, start/stop controls."""
    return templates.TemplateResponse(
        "live_bot.html",
        {"request": request, "title": "Live Bot"},
    )


# =============================================================================
# API: Strategy Discovery
# =============================================================================

@app.get("/api/strategies", summary="List all auto-discovered strategies")
async def list_strategies():
    """
    Return all strategy names discovered from the strategies/ package.
    New strategy files added by the user appear here automatically.
    """
    # Refresh on every call so newly added strategies are visible
    _STRATEGY_REGISTRY.update(_discover_strategies())

    return {
        "strategies": {
            name: {
                "description": info["description"],
                "params":      info["params"],
                "category":    info["category"],
            }
            for name, info in _STRATEGY_REGISTRY.items()
        }
    }


@app.get("/api/strategies/reload", summary="Hot-reload strategies without restart")
async def reload_strategies():
    """Force a fresh discovery scan. Use after adding a new strategy file."""
    global _STRATEGY_REGISTRY
    _STRATEGY_REGISTRY = _discover_strategies()
    return {"reloaded": sorted(_STRATEGY_REGISTRY.keys())}


# =============================================================================
# API: Data
# =============================================================================

@app.get("/api/data/symbols", summary="List symbols with local OHLCV data")
async def list_symbols():
    symbols = _list_available_symbols()
    return {"symbols": symbols, "count": len(symbols)}


@app.get("/api/data/ohlcv/{symbol}", summary="OHLCV bars for chart rendering")
async def get_ohlcv(
    symbol:    str,
    timeframe: str = "daily",
    from_date: str = "2022-01-01",
    to_date:   str = "",
):
    """
    Return the last 500 OHLCV bars for a symbol (for candlestick chart in UI).

    Edge cases:
    - NaN values → forward-fill then drop remaining NaN rows
    - Empty to_date → today
    - Data < 50 bars → synthetic fallback (never a 404)
    """
    df = _load_ohlcv(symbol, timeframe, from_date, to_date)
    df = df.ffill().dropna(subset=["open", "high", "low", "close"])

    tail    = df.tail(500)
    records = []
    for idx, row in tail.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        records.append({
            "date":   date_str,
            "open":   round(float(row["open"]),  2),
            "high":   round(float(row["high"]),  2),
            "low":    round(float(row["low"]),   2),
            "close":  round(float(row["close"]), 2),
            "volume": int(row.get("volume", 0)),
        })

    return {"symbol": symbol, "timeframe": timeframe, "count": len(records), "bars": records}


# =============================================================================
# API: Backtest
# =============================================================================

@app.post("/api/backtest", summary="Run backtest and return results")
async def run_backtest(req: BacktestRequest):
    """
    Execute a full backtest and return:
      - metrics     : Sharpe, CAGR, Max DD, Win Rate, etc.
      - trades      : list of all trades (entry/exit/P&L/charges)
      - equity      : [{date, value}] series for equity curve chart
      - drawdown    : [{date, value}] series for drawdown chart
      - chart_b64   : base64-encoded PNG backtest chart (Streak-style)

    All errors are handled gracefully — chart failure is non-fatal.
    """
    from backtester.engine_v2  import BacktestEngineV2, BacktestConfigV2
    from backtester.commission import Segment
    from backtester.order_types import OrderType

    # 1. Load data
    try:
        df = _load_ohlcv(req.symbol, req.timeframe, req.from_date, req.to_date)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data load error: {exc}")

    if len(df) < 50:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(df)} bars available for {req.symbol} (need ≥ 50).",
        )

    # 2. Strategy
    strategy = _get_strategy(req.strategy_name, req.strategy_params)

    # 3. Parse segment
    try:
        segment = Segment[req.segment]
    except KeyError:
        valid = [s.name for s in Segment]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown segment '{req.segment}'. Valid values: {valid}",
        )

    # 4. Parse order type (default to MARKET on unknown value)
    try:
        order_type = OrderType[req.order_type]
    except KeyError:
        logger.warning(f"Unknown order_type '{req.order_type}', defaulting to MARKET")
        order_type = OrderType.MARKET

    # 5. Build config and run
    config = BacktestConfigV2(
        initial_capital    = req.initial_capital,
        capital_risk_pct   = req.capital_risk_pct,
        segment            = segment,
        allow_shorting     = req.allow_shorting,
        max_positions      = req.max_positions,
        lot_size           = req.lot_size,
        default_order_type = order_type,
        stop_loss_pct      = req.stop_loss_pct,
        use_trailing_stop  = req.use_trailing_stop,
        trailing_stop_pct  = req.trailing_stop_pct,
        save_trade_log     = req.save_trade_log,
        save_raw_data      = req.save_raw_data,
        save_chart         = req.save_chart,
        run_label          = req.run_label,
    )

    try:
        engine = BacktestEngineV2(config)
        result = engine.run(df, strategy, symbol=req.symbol)
    except Exception as exc:
        logger.exception("Backtest engine error")
        raise HTTPException(status_code=500, detail=f"Backtest engine error: {exc}")

    metrics = result._compute_metrics()

    # 6. Chart as base64 (non-fatal if it fails)
    chart_b64 = None
    if req.save_chart:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                from backtester.report import generate_report
                fpath = generate_report(result, req.symbol, output_dir=tmpdir, show=False)
                with open(fpath, "rb") as fh:
                    chart_b64 = base64.b64encode(fh.read()).decode("utf-8")
        except Exception as exc:
            logger.warning(f"Chart generation failed (non-fatal): {exc}")

    # 7. Equity and drawdown series (safe NaN handling)
    def _to_list(series: pd.Series) -> List[Dict]:
        out = []
        for idx, val in series.dropna().items():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            out.append({
                "date":  date_str,
                "value": round(0.0 if (isinstance(val, float) and np.isnan(val)) else float(val), 2),
            })
        return out

    # 8. Trades (safe serialisation — skip malformed entries)
    trades = []
    for t in result.trade_log:
        try:
            trades.append(t.to_dict())
        except Exception:
            pass

    return JSONResponse({
        "symbol":    req.symbol,
        "strategy":  req.strategy_name,
        "metrics":   metrics,
        "trades":    trades,
        "equity":    _to_list(result.equity_curve),
        "drawdown":  _to_list(result.drawdown),
        "chart_b64": chart_b64,
    })


# =============================================================================
# API: Optimizer
# =============================================================================

@app.post("/api/optimize", summary="Run parameter optimizer")
async def run_optimizer(req: OptimizeRequest):
    """
    Grid or random search over strategy parameter combinations.

    Returns top-N results sorted by the target metric.

    Edge cases:
    - Empty param_grid → 422
    - Unknown strategy → 400
    - Optimizer returns empty → 404
    - Metric column missing → fallback sort by first numeric column
    """
    from backtester.engine_v2  import BacktestEngineV2, BacktestConfigV2
    from backtester.commission import Segment

    if not req.param_grid:
        raise HTTPException(status_code=422, detail="param_grid cannot be empty")

    if req.strategy_name not in _STRATEGY_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy '{req.strategy_name}' not found. Available: {sorted(_STRATEGY_REGISTRY.keys())}",
        )

    df = _load_ohlcv(req.symbol, req.timeframe, req.from_date, req.to_date)
    if len(df) < 50:
        raise HTTPException(status_code=422,
                            detail=f"Insufficient data for {req.symbol}: {len(df)} bars")

    try:
        segment = Segment[req.segment]
    except KeyError:
        segment = Segment.EQUITY_DELIVERY

    config = BacktestConfigV2(initial_capital=req.initial_capital, segment=segment)
    engine = BacktestEngineV2(config)
    cls    = _STRATEGY_REGISTRY[req.strategy_name]["class"]

    try:
        result_df = engine.optimize(
            df, cls, req.param_grid,
            symbol  = req.symbol,
            metric  = req.metric,
            method  = req.method,
            n_random= req.n_random,
        )
    except Exception as exc:
        logger.exception("Optimizer error")
        raise HTTPException(status_code=500, detail=f"Optimizer failed: {exc}")

    if result_df is None or result_df.empty:
        raise HTTPException(status_code=404,
                            detail="Optimizer returned no results — check param_grid values")

    return {
        "symbol":   req.symbol,
        "strategy": req.strategy_name,
        "metric":   req.metric,
        "trials":   len(result_df),
        "results":  result_df.to_dict(orient="records"),
    }


# =============================================================================
# API: Screener
# =============================================================================

@app.post("/api/screener/scan", summary="Scan universe for strategy signals")
async def run_screener(req: ScreenerRequest):
    """
    Run a strategy signal scan across all available symbols.

    Edge cases:
    - No symbols / data → 404 with clear message
    - Individual symbol errors → logged, skipped (scan never crashes)
    - rank_by column absent → screener falls back to 'close'
    """
    from screener.screener_v2 import Screener, ScreenerConfig

    symbols = _list_available_symbols()[:200]   # cap to prevent timeout

    data_dict: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = _load_ohlcv(sym, req.timeframe, req.from_date, "")
            if df is not None and len(df) >= 100:
                data_dict[sym] = df
        except Exception as exc:
            logger.debug(f"Screener: skipping {sym}: {exc}")

    if not data_dict:
        raise HTTPException(
            status_code=404,
            detail="No OHLCV data found. Download data first using DataManager.",
        )

    strategy = _get_strategy(req.strategy_name, req.strategy_params)

    cfg = ScreenerConfig(
        min_volume   = req.min_volume,
        min_price    = req.min_price,
        signal_type  = req.signal_type,
        max_results  = req.max_results,
        rank_by      = req.rank_by,
        save_results = True,
        label        = f"api_{req.strategy_name}",
    )
    screener = Screener(cfg)

    try:
        hits = screener.scan(data_dict, strategy)
    except Exception as exc:
        logger.exception("Screener error")
        raise HTTPException(status_code=500, detail=f"Screener failed: {exc}")

    return {
        "strategy":    req.strategy_name,
        "signal_type": req.signal_type,
        "scanned":     len(data_dict),
        "hits":        len(hits),
        "results":     hits,
    }


# =============================================================================
# API: Results Browser
# =============================================================================

@app.get("/api/results/trade_logs", summary="List saved trade log CSV files")
async def list_trade_logs():
    """Return all saved trade log CSVs sorted by most recent first."""
    files = sorted(
        OUTPUT_TRADE.glob("*.csv"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return {
        "files": [
            {
                "name":     f.name,
                "size_kb":  round(f.stat().st_size / 1024, 1),
                "modified": pd.Timestamp(f.stat().st_mtime, unit="s").strftime("%Y-%m-%d"),
            }
            for f in files
        ]
    }


@app.get("/api/results/charts", summary="List saved backtest chart PNGs")
async def list_charts():
    """Return all saved chart PNGs with direct URL for download / display."""
    files = sorted(
        OUTPUT_CHART.glob("*.png"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return {
        "files": [
            {
                "name":    f.name,
                "url":     f"/charts/{f.name}",
                "size_kb": round(f.stat().st_size / 1024, 1),
            }
            for f in files
        ]
    }


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health", summary="Health check")
async def health():
    return {
        "status":             "ok",
        "version":            "2.0.0",
        "strategies_loaded":  len(_STRATEGY_REGISTRY),
        "strategy_names":     sorted(_STRATEGY_REGISTRY.keys()),
    }
