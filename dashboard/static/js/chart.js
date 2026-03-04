/**
 * dashboard/static/js/chart.js
 * ------------------------------
 * Candlestick chart using TradingView's Lightweight Charts library.
 * Renders OHLCV candles + volume + buy/sell markers from backtest results.
 *
 * BUGS FIXED vs previous version:
 *   BUG-4: updateChart(null, trades) was called from backtest_results.js when
 *           only overlaying markers on an existing chart (no new bars needed).
 *           bars.map() crashed with "Cannot read properties of null".
 *           Fix: guard at top of updateChart() — if bars is null/undefined,
 *           skip candle+volume rendering and only add markers.
 *
 * Exports (globals): initChart(containerId), updateChart(bars, trades),
 *                    fetchAndRenderChart(symbol, timeframe, fromDate, toDate)
 */

'use strict';

let _chart        = null;
let _candleSeries = null;
let _volSeries    = null;

/**
 * Initialise the Lightweight Charts instance inside `containerId`.
 * Must be called once before updateChart() or fetchAndRenderChart().
 *
 * @param {string} containerId - id of the <div> that will hold the chart
 */
function initChart(containerId = 'chart-container') {
  const container = document.getElementById(containerId);
  if (!container) {
    console.warn(`initChart: element #${containerId} not found`);
    return;
  }

  _chart = LightweightCharts.createChart(container, {
    width:  container.clientWidth,
    height: 420,
    layout: {
      background: { color: '#111419' },
      textColor:  '#94a3b8',
    },
    grid: {
      vertLines: { color: '#1e2330' },
      horzLines: { color: '#1e2330' },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
    },
    rightPriceScale: {
      borderColor: '#262b34',
    },
    timeScale: {
      borderColor:    '#262b34',
      timeVisible:    true,
      secondsVisible: false,
      fixLeftEdge:    true,
      fixRightEdge:   true,
    },
  });

  // ── Candlestick series ───────────────────────────────────────────────────
  _candleSeries = _chart.addCandlestickSeries({
    upColor:         '#26a65b',
    downColor:       '#e53e3e',
    borderUpColor:   '#26a65b',
    borderDownColor: '#e53e3e',
    wickUpColor:     '#26a65b',
    wickDownColor:   '#e53e3e',
  });

  // ── Volume histogram (scaled at bottom of chart) ──────────────────────────
  _volSeries = _chart.addHistogramSeries({
    priceFormat:  { type: 'volume' },
    priceScaleId: 'volume',
    color:        '#3b82f640',
  });
  _chart.priceScale('volume').applyOptions({
    scaleMargins: { top: 0.8, bottom: 0 },
  });

  // ── Responsive resize on container resize ─────────────────────────────────
  const ro = new ResizeObserver(() => {
    if (_chart) {
      _chart.resize(container.clientWidth, container.clientHeight || 420);
    }
  });
  ro.observe(container);
}


/**
 * Load candle data and/or trade markers onto the chart.
 *
 * BUG-4 FIX: bars can be null when caller only wants to add markers
 * over an already-loaded chart (e.g. after fetchAndRenderChart).
 * In that case we skip candle / volume rendering and only update markers.
 *
 * @param {Array|null} bars   — [{date:'YYYY-MM-DD', open, high, low, close, volume}, …]
 *                              Pass null to keep existing candles and only update markers.
 * @param {Array}      trades — trade objects from /api/backtest response.
 *                              Must have: entry_date, exit_date (YYYY-MM-DD strings),
 *                              direction (int: 1 = LONG, -1 = SHORT)
 */
function updateChart(bars, trades = []) {
  // Safety: chart must be initialised
  if (!_candleSeries) {
    console.warn('updateChart: chart not initialised — call initChart() first');
    return;
  }

  // ── Candles + Volume (only when bars provided) ───────────────────────────
  // BUG-4 FIX: bars can be null when only overlaying markers on existing data.
  // Skip rendering if bars is null/undefined/empty.
  if (bars && Array.isArray(bars) && bars.length > 0) {
    const candles = bars.map(b => ({
      time:  b.date,
      open:  b.open,
      high:  b.high,
      low:   b.low,
      close: b.close,
    }));
    _candleSeries.setData(candles);

    const volumes = bars.map(b => ({
      time:  b.date,
      value: b.volume || 0,
      color: b.close >= b.open ? '#26a65b30' : '#e53e3e30',
    }));
    _volSeries.setData(volumes);
  }

  // ── Trade markers (buy/sell/exit arrows on the chart) ────────────────────
  // trade.direction is int (1 = LONG, -1 = SHORT) — fixed in _serialize_trade()
  // trade.entry_date / trade.exit_date are "YYYY-MM-DD" strings — added in fix
  const markers = [];

  (trades || []).forEach(t => {
    // Entry marker: green ▲ for LONG, red ▼ for SHORT
    if (t.entry_date) {
      markers.push({
        time:     t.entry_date.slice(0, 10),
        position: t.direction === 1 ? 'belowBar' : 'aboveBar',
        color:    t.direction === 1 ? '#26a65b'  : '#e53e3e',
        shape:    t.direction === 1 ? 'arrowUp'  : 'arrowDown',
        text:     t.direction === 1 ? 'BUY'      : 'SELL',
        size:     1,
      });
    }

    // Exit marker: amber circle regardless of direction
    if (t.exit_date) {
      markers.push({
        time:     t.exit_date.slice(0, 10),
        position: t.direction === 1 ? 'aboveBar' : 'belowBar',
        color:    '#f0a500',
        shape:    'circle',
        text:     'EXIT',
        size:     1,
      });
    }
  });

  // Lightweight Charts requires markers sorted chronologically
  markers.sort((a, b) => a.time.localeCompare(b.time));
  _candleSeries.setMarkers(markers);

  // Fit visible range to show all loaded data
  _chart.timeScale().fitContent();
}


/**
 * Fetch OHLCV candles from /api/data/ohlcv/<symbol> and render them.
 * Called on page load or when symbol/timeframe changes.
 *
 * @param {string} symbol    - NSE symbol e.g. "INFY"
 * @param {string} timeframe - "daily" | "minute" | "5m" etc.
 * @param {string} fromDate  - "YYYY-MM-DD"
 * @param {string} toDate    - "YYYY-MM-DD" or "" for today
 */
async function fetchAndRenderChart(
  symbol,
  timeframe = 'daily',
  fromDate  = '2020-01-01',
  toDate    = '',
) {
  if (!_candleSeries) {
    initChart();
  }

  const url = `/api/data/ohlcv/${encodeURIComponent(symbol)}`
            + `?timeframe=${timeframe}&from_date=${fromDate}&to_date=${toDate}`;
  try {
    const res  = await fetch(url);
    if (!res.ok) {
      console.error(`fetchAndRenderChart: API error ${res.status}`);
      return;
    }
    const data = await res.json();
    if (data.bars && data.bars.length > 0) {
      updateChart(data.bars, []);   // bars provided, no markers yet
    } else {
      console.warn(`fetchAndRenderChart: no bars returned for ${symbol}`);
    }
  } catch (e) {
    console.error('fetchAndRenderChart error:', e);
  }
}
