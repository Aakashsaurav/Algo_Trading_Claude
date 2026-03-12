"""
web/app.py
-----------
FastAPI web application — Phase 5 Trading Dashboard backend.

This file wires together all modules (backtester, screener, data)
into a clean REST API that serves the HTML dashboard.

ARCHITECTURE:
  Browser → FastAPI (this file) → Backtester/Screener/Data modules
                                 → OHLCV Parquet files (local)
                                 → Upstox API (live data, future)

ENDPOINTS:
  GET  /                              → Dashboard HTML page
  POST /api/backtest                  → Run backtest, return metrics + chart
  POST /api/optimize                  → Run parameter optimizer
  GET  /api/screener/scan             → Run screener, return hits
  GET  /api/strategies                → List available strategies
  GET  /api/data/symbols              → List available symbols with data
  GET  /api/data/ohlcv/{symbol}       → Get OHLCV data for a symbol
  GET  /api/results/trade_logs        → List saved trade log files
  GET  /api/results/charts            → List saved chart files
  GET  /static/{filename}             → Serve static files (charts)

HOW TO RUN:
  cd algo_trading
  uvicorn web.app:app --reload --port 8080
  uvicorn dashboard.app:app --reload --port 8080 #Modified by Aakash
  Then open http://localhost:8080 in your browser.

DEPENDENCIES:
  pip install fastapi uvicorn python-multipart aiofiles
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
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Path setup (run from algo_trading root) ───────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

logger = logging.getLogger(__name__)

# ── Output directories ────────────────────────────────────────────────────────
OUTPUT_TRADE = _HERE / "strategies" / "output" / "trade"
OUTPUT_RAW   = _HERE / "strategies" / "output" / "raw_data"
OUTPUT_CHART = _HERE / "strategies" / "output" / "chart"
for _d in (OUTPUT_TRADE, OUTPUT_RAW, OUTPUT_CHART):
    _d.mkdir(parents=True, exist_ok=True)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Algo Trading Dashboard",
    description="Backtester · Screener · Live Trading",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve chart images as static files
app.mount("/charts", StaticFiles(directory=str(OUTPUT_CHART)), name="charts")
app.mount(
    "/static",
    StaticFiles(directory=str(_HERE / "web" / "static")),
    name="static",
)


# =============================================================================
# Pydantic Models (request / response shapes)
# =============================================================================

class BacktestRequest(BaseModel):
    symbol:            str            = Field("INFY", description="NSE symbol")
    strategy_name:     str            = Field("EMACrossover", description="Strategy class name")
    strategy_params:   Dict[str, Any] = Field(default_factory=dict)
    timeframe:         str            = Field("daily", description="daily or minute")
    from_date:         str            = Field("2020-01-01")
    to_date:           str            = Field("")               # empty = today
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
    # Output options
    save_trade_log:    bool           = Field(False)
    save_raw_data:     bool           = Field(False)
    save_chart:        bool           = Field(True)
    run_label:         str            = Field("dashboard_run")


class OptimizeRequest(BaseModel):
    symbol:          str            = Field("INFY")
    strategy_name:   str            = Field("EMACrossover")
    param_grid:      Dict[str, List[Any]] = Field(
        default={"fast_period": [5, 9, 13], "slow_period": [21, 34, 50]}
    )
    metric:          str            = Field("Sharpe Ratio")
    method:          str            = Field("grid")
    n_random:        int            = Field(30)
    timeframe:       str            = Field("daily")
    from_date:       str            = Field("2020-01-01")
    initial_capital: float          = Field(500_000)
    segment:         str            = Field("EQUITY_DELIVERY")


class ScreenerRequest(BaseModel):
    strategy_name:   str            = Field("RSIMeanReversion")
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    signal_type:     int            = Field(1, description="+1 buy, -1 sell, 0 both")
    min_volume:      float          = Field(200_000)
    min_price:       float          = Field(50.0)
    max_results:     int            = Field(30)
    rank_by:         str            = Field("rsi")
    timeframe:       str            = Field("daily")


# =============================================================================
# Strategy Registry
# =============================================================================

def _load_strategy(name: str, params: dict):
    """Dynamically load a strategy class by name and instantiate it."""
    from strategies.base_strategy_github import (
        EMACrossover, RSIMeanReversion, BollingerBandStrategy,
        MACDStrategy, SupertrendStrategy,
    )
    registry = {
        "EMACrossover":         EMACrossover,
        "RSIMeanReversion":     RSIMeanReversion,
        "BollingerBandStrategy":BollingerBandStrategy,
        "MACDStrategy":         MACDStrategy,
        "SupertrendStrategy":   SupertrendStrategy,
    }
    if name not in registry:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{name}'. Available: {list(registry.keys())}"
        )
    try:
        return registry[name](**params)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid strategy params: {e}")


STRATEGY_REGISTRY = {
    "EMACrossover": {
        "description": "EMA Crossover — buy when fast EMA crosses above slow EMA",
        "params": [
            {"name": "fast_period", "type": "int",   "default": 9,  "min": 2,  "max": 50},
            {"name": "slow_period", "type": "int",   "default": 21, "min": 5,  "max": 200},
        ],
        "category": "Trend Following",
    },
    "RSIMeanReversion": {
        "description": "RSI Mean Reversion — buy oversold, sell overbought",
        "params": [
            {"name": "rsi_period",        "type": "int",   "default": 14, "min": 2,  "max": 50},
            {"name": "oversold_level",    "type": "float", "default": 30, "min": 5,  "max": 45},
            {"name": "overbought_level",  "type": "float", "default": 70, "min": 55, "max": 95},
            {"name": "sma_filter_period", "type": "int",   "default": 200,"min": 0,  "max": 500},
        ],
        "category": "Mean Reversion",
    },
    "BollingerBandStrategy": {
        "description": "Bollinger Band — reversion or breakout at band extremes",
        "params": [
            {"name": "period",   "type": "int",    "default": 20,  "min": 5,  "max": 100},
            {"name": "std_dev",  "type": "float",  "default": 2.0, "min": 1.0,"max": 4.0},
            {"name": "mode",     "type": "select", "default": "reversion",
             "options": ["reversion", "breakout"]},
        ],
        "category": "Mean Reversion",
    },
    "MACDStrategy": {
        "description": "MACD Histogram crossover with optional RSI filter",
        "params": [
            {"name": "fast_period",   "type": "int", "default": 12, "min": 3,  "max": 30},
            {"name": "slow_period",   "type": "int", "default": 26, "min": 10, "max": 60},
            {"name": "signal_period", "type": "int", "default": 9,  "min": 3,  "max": 20},
            {"name": "rsi_filter",    "type": "int", "default": 0,  "min": 0,  "max": 60},
        ],
        "category": "Trend Following",
    },
    "SupertrendStrategy": {
        "description": "Supertrend ATR-based trend direction signals",
        "params": [
            {"name": "period",     "type": "int",   "default": 10, "min": 5,  "max": 30},
            {"name": "multiplier", "type": "float", "default": 3.0,"min": 1.0,"max": 6.0},
        ],
        "category": "Trend Following",
    },
}


# =============================================================================
# Data Loading Helper
# =============================================================================

def _load_ohlcv(symbol: str, timeframe: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Load OHLCV data from the Parquet store.

    Falls back to synthetic data if real data isn't available locally
    (useful for development/demo without a live Upstox connection).
    """
    #from data.parquet_store import ParquetStore
    #store = ParquetStore()
    from broker.upstox.data_manager import get_ohlcv
    '''
    try:
        if timeframe == "daily":
            df = store.load_daily(symbol, from_date=from_date, to_date=to_date or None)
        else:
            df = store.load_minute(symbol, from_date=from_date, to_date=to_date or None)

        if df is not None and len(df) >= 50:
            return df
    '''
    try:
        df = get_ohlcv(instrument_type="EQUITY", exchange="NSE", trading_symbol=symbol,
      unit="minutes", interval=5, from_date="2022-01-01"
    )

if df.empty:
    raise RuntimeError("missing minute data for RELIANCE")

    except Exception as e:
        logger.debug(f"Data load failed for {symbol}: {e}")

    # ── Synthetic fallback (demo mode) ────────────────────────────────────────
    logger.warning(f"No local data for {symbol} — using synthetic demo data")
    return _synthetic_ohlcv(symbol, from_date, to_date)


def _synthetic_ohlcv(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for demo/testing."""
    np.random.seed(abs(hash(symbol)) % 10000)
    end   = pd.Timestamp(to_date) if to_date else pd.Timestamp.today()
    start = pd.Timestamp(from_date)
    dates = pd.bdate_range(start, end, tz="Asia/Kolkata")
    n     = len(dates)
    if n < 50:
        dates = pd.bdate_range(end - pd.Timedelta(days=400), end, tz="Asia/Kolkata")
        n     = len(dates)

    price = 1000 + abs(hash(symbol)) % 2000
    close = price + np.cumsum(np.random.randn(n) * 15)
    close = np.maximum(close, 10)
    noise = np.abs(np.random.randn(n) * 8)

    df = pd.DataFrame({
        "open":   (close - noise * 0.5).astype(float),
        "close":  close.astype(float),
        "volume": np.random.randint(500_000, 5_000_000, n).astype(int),
        "oi":     np.zeros(n, dtype=int),
    }, index=dates)
    df["high"] = (df[["open", "close"]].max(axis=1) + noise * 0.8).astype(float)
    df["low"]  = (df[["open", "close"]].min(axis=1) - noise * 0.8).clip(lower=1).astype(float)
    return df


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse, summary="Dashboard UI")
async def dashboard():
    """Serve the main trading dashboard HTML page."""
    html_path = _HERE / "web" / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Dashboard loading...</h1><p>Place dashboard.html in web/</p>")


@app.get("/api/strategies", summary="List available strategies")
async def list_strategies():
    """Return all available strategy names with their parameter schemas."""
    return {"strategies": STRATEGY_REGISTRY}


@app.get("/api/data/symbols", summary="List symbols with local data")
async def list_symbols():
    """Return all symbols that have local OHLCV data files."""
    try:
        from data.parquet_store import ParquetStore
        store   = ParquetStore()
        symbols = store.list_symbols("daily")
        return {"symbols": sorted(symbols), "count": len(symbols)}
    except Exception as e:
        # Return example symbols if data store not initialised
        demo = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                "SBIN", "WIPRO", "BAJFINANCE", "TITAN", "ASIANPAINT"]
        return {"symbols": demo, "count": len(demo), "note": "Demo symbols"}


@app.post("/api/backtest", summary="Run a backtest")
async def run_backtest(req: BacktestRequest):
    """
    Run a full backtest and return metrics, trade log, and chart as base64.

    Returns:
      metrics:    dict of performance metrics
      trades:     list of trade dicts (entry, exit, P&L, charges)
      chart_b64:  base64-encoded PNG chart (embed directly in <img> tag)
      equity:     list of {date, value} for the equity curve chart
    """
    from backtester.engine_v2 import BacktestEngineV2, BacktestConfigV2
    from backtester.commission import Segment
    from backtester.order_types import OrderType

    try:
        df = _load_ohlcv(req.symbol, req.timeframe, req.from_date, req.to_date)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Data load failed: {e}")

    strategy = _load_strategy(req.strategy_name, req.strategy_params)

    try:
        segment = Segment[req.segment]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown segment: {req.segment}")

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

    engine = BacktestEngineV2(config)
    result = engine.run(df, strategy, symbol=req.symbol)
    metrics = result._compute_metrics()

    # ── Generate chart as base64 ──────────────────────────────────────────────
    chart_b64 = None
    if req.save_chart:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                from backtester.report import generate_report
                fpath   = generate_report(result, req.symbol,
                                          output_dir=tmpdir, show=False)
                with open(fpath, "rb") as f:
                    chart_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")

    # ── Equity curve for JS chart ─────────────────────────────────────────────
    eq = result.equity_curve.dropna()
    equity_series = [
        {"date": str(idx.date()), "value": round(float(v), 2)}
        for idx, v in zip(eq.index, eq.values)
    ]

    # ── Drawdown series ───────────────────────────────────────────────────────
    dd = result.drawdown.dropna()
    dd_series = [
        {"date": str(idx.date()), "value": round(float(v), 2)}
        for idx, v in zip(dd.index, dd.values)
    ]

    return JSONResponse({
        "symbol":   req.symbol,
        "strategy": req.strategy_name,
        "metrics":  metrics,
        "trades":   [t.to_dict() for t in result.trade_log],
        "equity":   equity_series,
        "drawdown": dd_series,
        "chart_b64": chart_b64,
    }, headers={"Content-Type": "application/json"})


@app.post("/api/optimize", summary="Run parameter optimizer")
async def run_optimizer(req: OptimizeRequest):
    """
    Run grid/random parameter optimization and return top N parameter sets.

    Returns:
      results: DataFrame rows sorted by the target metric
    """
    from backtester.engine_v2 import BacktestEngineV2, BacktestConfigV2
    from backtester.commission import Segment
    from strategies.base_strategy_github import (
        EMACrossover, RSIMeanReversion, BollingerBandStrategy,
        MACDStrategy, SupertrendStrategy,
    )

    strategy_classes = {
        "EMACrossover":         EMACrossover,
        "RSIMeanReversion":     RSIMeanReversion,
        "BollingerBandStrategy":BollingerBandStrategy,
        "MACDStrategy":         MACDStrategy,
        "SupertrendStrategy":   SupertrendStrategy,
    }
    if req.strategy_name not in strategy_classes:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy_name}")

    df = _load_ohlcv(req.symbol, req.timeframe, req.from_date, "")

    try:
        segment = Segment[req.segment]
    except KeyError:
        segment = Segment.EQUITY_DELIVERY

    config = BacktestConfigV2(
        initial_capital = req.initial_capital,
        segment         = segment,
    )
    engine = BacktestEngineV2(config)

    result_df = engine.optimize(
        df, strategy_classes[req.strategy_name], req.param_grid,
        symbol=req.symbol, metric=req.metric,
        method=req.method, n_random=req.n_random,
    )

    return {"symbol": req.symbol, "metric": req.metric,
            "results": result_df.to_dict(orient="records")}


@app.post("/api/screener/scan", summary="Run screener scan")
async def run_screener(req: ScreenerRequest):
    """
    Scan available symbols for strategy signals.

    Returns all symbols where the strategy's last signal matches signal_type.
    """
    from screener.screener_v2 import Screener, ScreenerConfig

    # Load available symbols
    try:
        from data.parquet_store import ParquetStore
        store   = ParquetStore()
        symbols = store.list_symbols("daily")[:200]  # Cap for performance
    except Exception:
        symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                   "SBIN", "WIPRO", "BAJFINANCE", "TITAN", "ASIANPAINT"]

    # Load OHLCV for all symbols
    data_dict = {}
    for sym in symbols:
        try:
            df = _load_ohlcv(sym, req.timeframe, "2020-01-01", "")
            if df is not None and len(df) >= 100:
                data_dict[sym] = df
        except Exception:
            pass

    if not data_dict:
        raise HTTPException(status_code=404,
                            detail="No data available. Please download OHLCV data first.")

    strategy = _load_strategy(req.strategy_name, req.strategy_params)

    screener_cfg = ScreenerConfig(
        min_volume   = req.min_volume,
        min_price    = req.min_price,
        signal_type  = req.signal_type,
        max_results  = req.max_results,
        rank_by      = req.rank_by,
        save_results = True,
        label        = f"api_{req.strategy_name}",
    )
    screener = Screener(screener_cfg)
    hits     = screener.scan(data_dict, strategy)

    return {
        "strategy":       req.strategy_name,
        "signal_type":    req.signal_type,
        "scanned":        len(data_dict),
        "hits":           len(hits),
        "results":        hits,
    }


@app.get("/api/results/trade_logs", summary="List saved trade log files")
async def list_trade_logs():
    """Return list of saved trade log CSV files."""
    files = sorted(OUTPUT_TRADE.glob("*.csv"), key=os.path.getmtime, reverse=True)
    return {
        "files": [
            {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1),
             "modified": str(pd.Timestamp(f.stat().st_mtime, unit="s").date())}
            for f in files
        ]
    }


@app.get("/api/results/charts", summary="List saved chart files")
async def list_charts():
    """Return list of saved chart PNG files."""
    files = sorted(OUTPUT_CHART.glob("*.png"), key=os.path.getmtime, reverse=True)
    return {
        "files": [
            {"name": f.name, "url": f"/charts/{f.name}",
             "size_kb": round(f.stat().st_size / 1024, 1)}
            for f in files
        ]
    }


@app.get("/api/data/ohlcv/{symbol}", summary="Get OHLCV data for a symbol")
async def get_ohlcv(symbol: str, timeframe: str = "daily",
                    from_date: str = "2022-01-01", to_date: str = ""):
    """Return OHLCV data for a symbol as JSON (for candlestick chart in UI)."""
    df = _load_ohlcv(symbol, timeframe, from_date, to_date)
    records = [
        {
            "date":   str(idx.date()),
            "open":   round(float(r["open"]),   2),
            "high":   round(float(r["high"]),   2),
            "low":    round(float(r["low"]),    2),
            "close":  round(float(r["close"]),  2),
            "volume": int(r.get("volume", 0)),
        }
        for idx, r in df.tail(500).iterrows()
    ]
    return {"symbol": symbol, "timeframe": timeframe, "bars": records}


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "version": "2.0.0"}
