"""
dashboard/app.py
-----------------
FastAPI web application — AlgoDesk Trading Dashboard.

BUGS FIXED vs previous version:
  BUG-1 [CRITICAL]: Trade.to_dict() returns pandas Timestamps → JSONResponse crashes.
         Fix: _serialize_trade() converts all Timestamps to ISO strings.
  BUG-2 [CRITICAL]: direction field was string "LONG"/"SHORT" in to_dict(),
         but frontend JS does (t.direction === 1) comparison → always false.
         Fix: _serialize_trade() returns direction as int (1 or -1) AND
              direction_label as string "LONG"/"SHORT".
  BUG-3 [CRITICAL]: chart.js uses t.entry_date / t.exit_date (YYYY-MM-DD),
         but backend sent only entry_time / exit_time (full ISO strings).
         Fix: _serialize_trade() adds both entry_date and exit_date fields.
  BUG-4: numpy scalars (np.int64, np.float64) are not JSON serializable.
         Fix: _serialize_trade() casts all numpy types to native Python types.
  BUG-5: NaN / Inf values in float fields crash json.dumps().
         Fix: _serialize_trade() replaces NaN/Inf with 0.0.

TO RUN:
  cd algo_trading
  uvicorn dashboard.app:app --reload --port 8080
  Open: http://localhost:8080
"""

import base64
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = Path(__file__).resolve().parent
_ROOT          = _DASHBOARD_DIR.parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ── Output directories ────────────────────────────────────────────────────────
OUTPUT_TRADE = _ROOT / "strategies" / "output" / "trade"
OUTPUT_RAW   = _ROOT / "strategies" / "output" / "raw_data"
OUTPUT_CHART = _ROOT / "strategies" / "output" / "chart"
DAILY_DIR    = _ROOT / "data" / "ohlcv" / "daily"
MINUTE_DIR   = _ROOT / "data" / "ohlcv" / "minute"

for _d in (OUTPUT_TRADE, OUTPUT_RAW, OUTPUT_CHART):
    _d.mkdir(parents=True, exist_ok=True)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AlgoDesk — Trading System",
    description="Backtester · Screener · Optimizer · Live Bot",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve chart images
app.mount("/charts", StaticFiles(directory=str(OUTPUT_CHART)), name="charts")

# Serve static assets (CSS/JS)
_static_dir = _DASHBOARD_DIR / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# =============================================================================
# Trade Serialization Helper  ←── THE CORE FIX
# =============================================================================

def _safe_scalar(v: Any) -> Any:
    """
    Convert a value to a JSON-safe Python native type.

    Handles:
      - pandas Timestamps           → ISO string (YYYY-MM-DD HH:MM:SS)
      - numpy int64/int32 etc.      → int
      - numpy float64/float32 etc.  → float
      - float NaN / Inf             → 0.0  (JSON has no concept of these)
      - everything else             → unchanged
    """
    if isinstance(v, pd.Timestamp):
        return str(v)[:19]   # "2021-04-12 09:15:00"
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        fv = float(v)
        return 0.0 if (np.isnan(fv) or np.isinf(fv)) else fv
    if isinstance(v, float):
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v
    return v


def _serialize_trade(t: Any) -> Dict[str, Any]:
    """
    Convert a Trade dataclass to a fully JSON-serializable dict.

    This replaces the raw `t.to_dict()` call which returns pandas Timestamps
    that crash json.dumps().

    Fields added vs to_dict():
      entry_date:      YYYY-MM-DD string used by chart.js for trade markers
      exit_date:       YYYY-MM-DD string used by chart.js for trade markers
      direction:       int (1 = LONG, -1 = SHORT) for JS strict equality checks
      direction_label: str ("LONG" / "SHORT") for display badges

    This design satisfies both:
      • chart.js:            t.direction === 1, t.entry_date, t.exit_date
      • backtest_results.js: t.direction === 1, t.direction_label,
                             t.entry_time, t.exit_time (string), all float fields
    """
    entry_ts = t.entry_time
    exit_ts  = t.exit_time

    # Convert Timestamps → strings safely
    entry_str = str(entry_ts)[:19] if entry_ts is not None else ""
    exit_str  = str(exit_ts)[:19]  if exit_ts  is not None else ""

    # Date-only "YYYY-MM-DD" for chart.js trade markers
    if hasattr(entry_ts, "date"):
        entry_date = str(entry_ts.date())
    else:
        entry_date = entry_str[:10]

    if hasattr(exit_ts, "date"):
        exit_date = str(exit_ts.date())
    else:
        exit_date = exit_str[:10]

    return {
        # Identity
        "symbol":          str(t.symbol),

        # Time fields — strings so JS can .slice(0,10) them
        "entry_time":      entry_str,
        "exit_time":       exit_str,

        # Date-only fields — used by chart.js for marker placement
        "entry_date":      entry_date,
        "exit_date":       exit_date,

        # Direction as INT (1/-1) for `t.direction === 1` in JS
        # Plus string label for display badges
        "direction":       int(t.direction),
        "direction_label": str(t.direction_label),

        # Price / quantity — cast numpy types away
        "entry_price":     _safe_scalar(round(float(t.entry_price), 2)),
        "exit_price":      _safe_scalar(round(float(t.exit_price),  2)),
        "quantity":        _safe_scalar(t.quantity),

        # P&L
        "gross_pnl":       _safe_scalar(round(float(t.gross_pnl),    2)),
        "entry_charges":   _safe_scalar(round(float(t.entry_charges), 2)),
        "exit_charges":    _safe_scalar(round(float(t.exit_charges),  2)),
        "total_charges":   _safe_scalar(round(float(t.total_charges), 2)),
        "net_pnl":         _safe_scalar(round(float(t.net_pnl),       2)),
        "pnl_pct":         _safe_scalar(round(float(t.pnl_pct),       4)),

        # Signals / metadata
        "entry_signal":    str(t.entry_signal   or ""),
        "exit_signal":     str(t.exit_signal    or ""),
        "duration":        str(t.duration       or ""),
        "duration_bars":   _safe_scalar(t.duration_bars),

        # Excursion analytics
        "mae":             _safe_scalar(round(float(t.mae), 2)),
        "mfe":             _safe_scalar(round(float(t.mfe), 2)),

        # Running portfolio value at this trade's exit
        "portfolio_value": _safe_scalar(round(float(t.cumulative_portfolio), 2)),
    }


# =============================================================================
# Pydantic Request Models
# =============================================================================

class BacktestRequest(BaseModel):
    symbol:            str            = Field("INFY")
    strategy_name:     str            = Field("EMACrossover")
    strategy_params:   Dict[str, Any] = Field(default_factory=dict)
    timeframe:         str            = Field("daily",   description="daily | minute")
    from_date:         str            = Field("2020-01-01")
    to_date:           str            = Field("",        description="empty = today")
    initial_capital:   float          = Field(500_000)
    capital_risk_pct:  float          = Field(0.02)
    segment:           str            = Field("EQUITY_DELIVERY")
    allow_shorting:    bool           = Field(False)
    max_positions:     int            = Field(5)
    lot_size:          int            = Field(1)
    order_type:        str            = Field("MARKET")
    stop_loss_pct:     float          = Field(0.0)
    trailing_stop_pct: float          = Field(0.0)
    use_trailing_stop: bool           = Field(False)
    save_trade_log:    bool           = Field(False)
    save_raw_data:     bool           = Field(False)
    save_chart:        bool           = Field(True)
    run_label:         str            = Field("dashboard")


class OptimizeRequest(BaseModel):
    symbol:          str                   = Field("INFY")
    strategy_name:   str                   = Field("EMACrossover")
    param_grid:      Dict[str, List[Any]]  = Field(
        default={"fast_period": [5, 9, 13], "slow_period": [21, 34, 50]}
    )
    metric:          str                   = Field("Sharpe Ratio")
    method:          str                   = Field("grid")
    n_random:        int                   = Field(30)
    timeframe:       str                   = Field("daily")
    from_date:       str                   = Field("2020-01-01")
    initial_capital: float                 = Field(500_000)
    segment:         str                   = Field("EQUITY_DELIVERY")


class ScreenerRequest(BaseModel):
    strategy_name:   str            = Field("RSIMeanReversion")
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    signal_type:     int            = Field(1)
    min_volume:      float          = Field(200_000)
    min_price:       float          = Field(50.0)
    max_results:     int            = Field(30)
    rank_by:         str            = Field("close")
    timeframe:       str            = Field("daily")
    from_date:       str            = Field("2022-01-01")


# =============================================================================
# Timeframe / Unit mapping
# =============================================================================

def _timeframe_to_dm_args(timeframe: str) -> Dict[str, Any]:
    """
    Map dashboard timeframe string → data_manager.get_ohlcv() args.
    """
    mapping = {
        "daily":   {"unit": "days",    "interval": 1},
        "1d":      {"unit": "days",    "interval": 1},
        "minute":  {"unit": "minutes", "interval": 1},
        "1m":      {"unit": "minutes", "interval": 1},
        "3m":      {"unit": "minutes", "interval": 3},
        "5m":      {"unit": "minutes", "interval": 5},
        "15m":     {"unit": "minutes", "interval": 15},
        "30m":     {"unit": "minutes", "interval": 30},
        "60m":     {"unit": "minutes", "interval": 60},
        "weekly":  {"unit": "weeks",   "interval": 1},
        "monthly": {"unit": "months",  "interval": 1},
    }
    return mapping.get(timeframe.lower(), {"unit": "days", "interval": 1})


# =============================================================================
# Data Loading
# =============================================================================

def _load_ohlcv(
    symbol:    str,
    timeframe: str,
    from_date: str,
    to_date:   str,
) -> pd.DataFrame:
    """
    Load OHLCV data via data_manager.get_ohlcv().
    Falls back to synthetic data in demo/dev mode (no Upstox token).
    """
    dm_args = _timeframe_to_dm_args(timeframe)

    try:
        from broker.upstox.data_manager import get_ohlcv
        df = get_ohlcv(
            instrument_type = "EQUITY",
            exchange        = "NSE",
            trading_symbol  = symbol.upper().strip(),
            unit            = dm_args["unit"],
            interval        = dm_args["interval"],
            from_date       = from_date,
            to_date         = to_date if to_date else None,
        )
        if df is not None and len(df) >= 10:
            for col in ("open", "high", "low", "close"):
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df
    except Exception as e:
        logger.warning(
            f"data_manager.get_ohlcv failed for {symbol} ({timeframe}): {e}. "
            "Using synthetic fallback (demo mode)."
        )

    return _synthetic_ohlcv(symbol, from_date, to_date)


def _list_available_symbols() -> List[str]:
    """Scan parquet directories for available symbols."""
    symbols = set()
    if DAILY_DIR.exists():
        for f in DAILY_DIR.glob("*.parquet"):
            symbols.add(f.stem)
    if MINUTE_DIR.exists():
        for d in MINUTE_DIR.iterdir():
            if d.is_dir():
                symbols.add(d.name)
    return sorted(symbols)


def _synthetic_ohlcv(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV for demo/development mode."""
    np.random.seed(abs(hash(symbol)) % 10_000)
    try:
        end   = pd.Timestamp(to_date, tz="Asia/Kolkata") if to_date else pd.Timestamp.now(tz="Asia/Kolkata")
        start = pd.Timestamp(from_date, tz="Asia/Kolkata")
    except Exception:
        end   = pd.Timestamp.now(tz="Asia/Kolkata")
        start = end - pd.Timedelta(days=500)

    dates = pd.bdate_range(start, end, tz="Asia/Kolkata")
    if len(dates) < 100:
        dates = pd.bdate_range(end - pd.Timedelta(days=600), end, tz="Asia/Kolkata")

    n       = len(dates)
    base_px = 500 + abs(hash(symbol)) % 3000
    changes = np.random.randn(n) * (base_px * 0.015)
    close   = np.maximum(base_px + np.cumsum(changes), 10.0)
    noise   = np.abs(np.random.randn(n) * base_px * 0.008)

    df = pd.DataFrame({
        "open":   (close - noise * 0.4).clip(min=1.0),
        "close":  close,
        "volume": np.random.randint(200_000, 5_000_000, n).astype(int),
        "oi":     np.zeros(n, dtype=int),
    }, index=dates)
    df["high"] = (df[["open", "close"]].max(axis=1) + noise * 0.7).astype(float)
    df["low"]  = (df[["open", "close"]].min(axis=1) - noise * 0.7).clip(lower=1.0).astype(float)
    return df.astype({"open": float, "close": float, "high": float, "low": float})


# =============================================================================
# Page Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse, summary="Dashboard main page")
async def serve_dashboard():
    tmpl = _DASHBOARD_DIR / "templates" / "base.html"
    if tmpl.exists():
        return HTMLResponse(tmpl.read_text())
    return HTMLResponse("<h2>Dashboard templates not found.</h2>")


@app.get("/backtester", response_class=HTMLResponse)
async def serve_backtester():
    tmpl = _DASHBOARD_DIR / "templates" / "backtester.html"
    return HTMLResponse(tmpl.read_text() if tmpl.exists() else "<h2>backtester.html not found</h2>")


@app.get("/screener", response_class=HTMLResponse)
async def serve_screener():
    tmpl = _DASHBOARD_DIR / "templates" / "screener.html"
    return HTMLResponse(tmpl.read_text() if tmpl.exists() else "<h2>screener.html not found</h2>")


@app.get("/strategy-builder", response_class=HTMLResponse)
async def serve_strategy_builder():
    tmpl = _DASHBOARD_DIR / "templates" / "strategy_builder.html"
    return HTMLResponse(tmpl.read_text() if tmpl.exists() else "<h2>strategy_builder.html not found</h2>")


@app.get("/live-bot", response_class=HTMLResponse)
async def serve_live_bot():
    tmpl = _DASHBOARD_DIR / "templates" / "live_bot.html"
    return HTMLResponse(tmpl.read_text() if tmpl.exists() else "<h2>live_bot.html not found</h2>")


# =============================================================================
# Strategy API
# =============================================================================

@app.get("/api/strategies", summary="List all available strategies (auto-discovered)")
async def list_strategies(refresh: bool = False):
    """
    Return all strategies auto-discovered from the strategies/ directory.
    No hardcoding — adding a new strategy file updates this automatically.
    """
    from strategies.registry import get_strategy_registry
    registry = get_strategy_registry(force_refresh=refresh)

    public_registry = {
        name: {
            "class_name":   s["class_name"],
            "display_name": s["display_name"],
            "description":  s["description"][:200],
            "category":     s["category"],
            "params":       s["params"],
        }
        for name, s in registry.items()
    }
    return {
        "count":      len(public_registry),
        "strategies": public_registry,
    }


# =============================================================================
# Data API
# =============================================================================

@app.get("/api/data/symbols", summary="List symbols with local OHLCV data")
async def list_symbols():
    symbols = _list_available_symbols()
    return {
        "symbols": symbols,
        "count":   len(symbols),
        "note":    "Symbols with local Parquet cache" if symbols else "No local data — using demo mode",
    }


@app.get("/api/data/ohlcv/{symbol}", summary="Get OHLCV candles for a symbol")
async def get_ohlcv_api(
    symbol:    str,
    timeframe: str = "daily",
    from_date: str = "2022-01-01",
    to_date:   str = "",
):
    """Return OHLCV candles as JSON for candlestick chart rendering."""
    df = _load_ohlcv(symbol, timeframe, from_date, to_date)
    bars = [
        {
            "date":   str(idx.date()),
            "open":   round(float(r["open"]),   2),
            "high":   round(float(r["high"]),   2),
            "low":    round(float(r["low"]),    2),
            "close":  round(float(r["close"]),  2),
            "volume": int(r.get("volume", 0)),
            "oi":     int(r.get("oi", 0)),
        }
        for idx, r in df.tail(500).iterrows()
    ]
    return {"symbol": symbol, "timeframe": timeframe, "count": len(bars), "bars": bars}


# =============================================================================
# Backtest API  ←── KEY FIX: uses _serialize_trade() instead of t.to_dict()
# =============================================================================

@app.post("/api/backtest", summary="Run a backtest")
async def run_backtest(req: BacktestRequest):
    """
    Run a full backtest and return JSON-safe results.

    All trade dicts go through _serialize_trade() which:
      1. Converts Timestamps → strings (fixes JSON crash)
      2. Adds entry_date / exit_date (fixes chart markers)
      3. Sets direction as int 1/-1 (fixes JS === comparisons)
      4. Sanitises numpy scalars and NaN/Inf (fixes edge-case crashes)
    """
    from backtester.engine_v2 import BacktestEngineV2, BacktestConfigV2
    from backtester.commission import Segment
    from backtester.order_types import OrderType
    from strategies.registry import load_strategy

    # Load data
    try:
        df = _load_ohlcv(req.symbol, req.timeframe, req.from_date, req.to_date)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Data load failed: {e}")

    if df.empty or len(df) < 50:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data for {req.symbol} ({len(df)} bars). Need at least 50."
        )

    # Load strategy
    try:
        strategy = load_strategy(req.strategy_name, req.strategy_params)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate segment and order type
    try:
        segment = Segment[req.segment]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown segment: {req.segment}. Valid: {[s.name for s in Segment]}"
        )
    try:
        order_type = OrderType[req.order_type]
    except KeyError:
        order_type = OrderType.MARKET

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
    except Exception as e:
        logger.exception("Backtest engine error")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

    metrics = result._compute_metrics()

    # Generate chart PNG as base64
    chart_b64 = None
    if req.save_chart:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                from backtester.report import generate_report
                fpath     = generate_report(result, req.symbol, output_dir=tmpdir, show=False)
                chart_b64 = base64.b64encode(Path(fpath).read_bytes()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")

    # Equity curve and drawdown (JSON-safe)
    eq = result.equity_curve.dropna()
    dd = result.drawdown.dropna()

    # ── KEY FIX: use _serialize_trade() instead of t.to_dict() ───────────────
    serialised_trades = [_serialize_trade(t) for t in result.trade_log]

    return JSONResponse({
        "symbol":    req.symbol,
        "strategy":  req.strategy_name,
        "params":    req.strategy_params,
        "metrics":   metrics,
        "trades":    serialised_trades,                                  # ← fixed
        "equity":    [
            {"date": str(i.date()), "value": round(float(v), 2)}
            for i, v in zip(eq.index, eq.values)
        ],
        "drawdown":  [
            {"date": str(i.date()), "value": round(float(v), 2)}
            for i, v in zip(dd.index, dd.values)
        ],
        "chart_b64": chart_b64,
    })


# =============================================================================
# Optimizer API
# =============================================================================

@app.post("/api/optimize", summary="Run parameter optimizer")
async def run_optimizer(req: OptimizeRequest):
    """Run grid/random parameter optimizer. Returns top-N param sets."""
    from backtester.engine_v2 import BacktestEngineV2, BacktestConfigV2
    from backtester.commission import Segment
    from strategies.registry import get_strategy_registry
    import importlib

    registry = get_strategy_registry()
    if req.strategy_name not in registry:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {req.strategy_name}. "
                   f"Available: {sorted(registry.keys())}"
        )

    schema         = registry[req.strategy_name]
    module         = importlib.import_module(schema["module_path"])
    strategy_class = getattr(module, req.strategy_name)

    df = _load_ohlcv(req.symbol, req.timeframe, req.from_date, "")
    if df.empty or len(df) < 100:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data for {req.symbol} ({len(df)} bars)."
        )

    try:
        segment = Segment[req.segment]
    except KeyError:
        segment = Segment.EQUITY_DELIVERY

    config = BacktestConfigV2(initial_capital=req.initial_capital, segment=segment)
    engine = BacktestEngineV2(config)

    try:
        result_df = engine.optimize(
            df, strategy_class, req.param_grid,
            symbol   = req.symbol,
            metric   = req.metric,
            method   = req.method,
            n_random = req.n_random,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimizer failed: {e}")

    return {
        "symbol":   req.symbol,
        "strategy": req.strategy_name,
        "metric":   req.metric,
        "method":   req.method,
        "trials":   len(result_df),
        "results":  result_df.to_dict(orient="records") if not result_df.empty else [],
    }


# =============================================================================
# Screener API
# =============================================================================

@app.post("/api/screener/scan", summary="Scan universe for strategy signals")
async def run_screener(req: ScreenerRequest):
    """Scan available symbols for strategy signals."""
    from screener.screener import Screener, ScreenerConfig
    from strategies.registry import load_strategy

    symbols = _list_available_symbols()
    if not symbols:
        symbols = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "WIPRO", "BAJFINANCE", "TITAN", "ASIANPAINT",
        ]

    data_dict: Dict[str, pd.DataFrame] = {}
    for sym in symbols[:200]:
        try:
            df = _load_ohlcv(sym, req.timeframe, req.from_date, "")
            if df is not None and len(df) >= 100:
                data_dict[sym] = df
        except Exception:
            pass

    if not data_dict:
        raise HTTPException(
            status_code=404,
            detail="No OHLCV data available. Download data first or run in demo mode."
        )

    try:
        strategy = load_strategy(req.strategy_name, req.strategy_params)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    cfg = ScreenerConfig(
        min_volume   = req.min_volume,
        min_price    = req.min_price,
        signal_type  = req.signal_type,
        max_results  = req.max_results,
        rank_by      = req.rank_by,
        save_results = True,
        label        = f"api_{req.strategy_name.lower()}",
    )
    screener = Screener(cfg)
    hits     = screener.scan(data_dict, strategy)

    return {
        "strategy":    req.strategy_name,
        "signal_type": req.signal_type,
        "scanned":     len(data_dict),
        "hits":        len(hits),
        "results":     hits,
    }


# =============================================================================
# Results API
# =============================================================================

@app.get("/api/results/trade_logs", summary="List saved trade log CSV files")
async def list_trade_logs():
    files = sorted(
        OUTPUT_TRADE.glob("*.csv"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return {
        "count": len(files),
        "files": [
            {
                "name":     f.name,
                "size_kb":  round(f.stat().st_size / 1024, 1),
                "modified": str(
                    pd.Timestamp(f.stat().st_mtime, unit="s")
                    .tz_localize("UTC")
                    .tz_convert("Asia/Kolkata")
                    .date()
                ),
                "url":      f"/api/results/download/{f.name}",
            }
            for f in files
        ],
    }


@app.get("/api/results/charts", summary="List saved chart PNG files")
async def list_charts():
    files = sorted(
        OUTPUT_CHART.glob("*.png"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return {
        "count": len(files),
        "files": [
            {"name": f.name, "url": f"/charts/{f.name}", "size_kb": round(f.stat().st_size / 1024, 1)}
            for f in files
        ],
    }


@app.get("/api/results/download/{filename}", summary="Download a result file")
async def download_result(filename: str):
    from fastapi.responses import FileResponse
    fpath = OUTPUT_TRADE / filename
    if not fpath.exists():
        fpath = OUTPUT_RAW / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(str(fpath), filename=filename, media_type="text/csv")


# =============================================================================
# Health check
# =============================================================================

@app.get("/health", summary="Health check")
async def health():
    from strategies.registry import get_strategy_registry
    registry = get_strategy_registry()
    symbols  = _list_available_symbols()
    return {
        "status":            "ok",
        "version":           "2.0.0",
        "strategies_loaded": len(registry),
        "symbols_with_data": len(symbols),
        "data_mode":         "live" if symbols else "demo (synthetic)",
    }
