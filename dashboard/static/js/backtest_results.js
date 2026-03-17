/**
 * dashboard/static/js/backtest_results.js
 * -----------------------------------------
 * Handles the Backtester page:
 *   - Loads strategy list and populates config dropdowns
 *   - Posts to /api/backtest and renders metrics, equity curve, drawdown, trade log
 *   - Posts to /api/optimize and renders optimizer results
 *   - Draws equity and drawdown mini-canvases
 *   - Renders buy/sell markers on the candlestick chart
 */

'use strict';

// Registry cache (loaded once on page init)
let _registry = null;

// ── Page init ────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  await loadStrategiesForBt();
  initChart('chart-container');

  // If navigated from strategy builder with a pre-run result, display it
  const stored = sessionStorage.getItem('bt_result');
  if (stored) {
    try {
      const data = JSON.parse(stored);
      sessionStorage.removeItem('bt_result');
      renderBacktestResults(data);
    } catch (e) { /* ignore */ }
  }
});


// ── Load strategies ───────────────────────────────────────────────────────────

async function loadStrategiesForBt() {
  try {
    const res  = await fetch('/api/strategies');
    _registry  = await res.json();
    const sel  = document.getElementById('cfg-strategy');
    sel.innerHTML = '';
    for (const [name, schema] of Object.entries(_registry.strategies || {})) {
      const opt = document.createElement('option');
      opt.value       = name;
      opt.textContent = `${schema.display_name}`;
      sel.appendChild(opt);
    }
    // Trigger param render for the first strategy
    onStrategyChange();
  } catch (e) {
    console.error('Failed to load strategies:', e);
  }
}


function onStrategyChange() {
  const sel  = document.getElementById('cfg-strategy');
  if (!sel || !_registry) return;
  const name   = sel.value;
  const schema = _registry.strategies[name];
  if (!schema) return;

  const container = document.getElementById('params-container');
  const card      = document.getElementById('params-card');
  container.innerHTML = '';

  if (schema.params && schema.params.length > 0) {
    schema.params.forEach(p => container.appendChild(_buildParamControl(p)));
    card.style.display = 'block';
  } else {
    card.style.display = 'none';
  }

  // Populate optimizer param grid with the strategy's params
  _populateOptParamGrid(schema.params);
}


// ── Build a single parameter form control ─────────────────────────────────────

function _buildParamControl(param) {
  const group = document.createElement('div');
  group.className = 'form-group';

  if (param.type === 'bool') {
    const row = document.createElement('div');
    row.className = 'checkbox-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.id = `param-${param.name}`; cb.checked = !!param.default;
    const lbl = document.createElement('label');
    lbl.setAttribute('for', cb.id); lbl.textContent = param.name;
    row.appendChild(cb); row.appendChild(lbl);
    group.appendChild(row);
    return group;
  }

  const label = document.createElement('label');
  label.textContent = param.name;
  group.appendChild(label);

  let input;
  if (param.type === 'select' && param.options) {
    input = document.createElement('select');
    param.options.forEach(opt => {
      const o = document.createElement('option');
      o.value = opt; o.textContent = opt;
      if (opt === param.default) o.selected = true;
      input.appendChild(o);
    });
  } else {
    input = document.createElement('input');
    input.type  = (param.type === 'int' || param.type === 'float') ? 'number' : 'text';
    input.value = param.default !== null ? param.default : '';
    if (param.min  != null) input.min  = param.min;
    if (param.max  != null) input.max  = param.max;
    if (param.step != null) input.step = param.step;
    if (param.type === 'int') input.step = '1';
  }
  input.id = `param-${param.name}`;
  group.appendChild(input);
  return group;
}


function _collectParams(strategyName) {
  if (!_registry) return {};
  const schema = _registry.strategies[strategyName];
  if (!schema) return {};
  const params = {};
  schema.params.forEach(p => {
    const el = document.getElementById(`param-${p.name}`);
    if (!el) return;
    if (p.type === 'bool')  params[p.name] = el.checked;
    else if (p.type === 'int')   params[p.name] = parseInt(el.value, 10) || p.default;
    else if (p.type === 'float') params[p.name] = parseFloat(el.value) || p.default;
    else params[p.name] = el.value;
  });
  return params;
}


// ── Run Backtest ──────────────────────────────────────────────────────────────

async function runBacktest() {
  const btn = document.getElementById('run-btn');
  const spn = document.getElementById('top-spinner');
  btn.disabled = true;
  if (spn) spn.className = 'spinner active';

  hideAllResults();
  showAlert('Running backtest… please wait.', 'info');

  const strategyName = document.getElementById('cfg-strategy').value;
  const payload = {
    symbol:            document.getElementById('cfg-symbol').value.trim().toUpperCase(),
    strategy_name:     strategyName,
    strategy_params:   _collectParams(strategyName),
    timeframe:         document.getElementById('cfg-timeframe').value,
    from_date:         document.getElementById('cfg-from').value,
    to_date:           document.getElementById('cfg-to').value || '',
    initial_capital:   parseFloat(document.getElementById('cfg-capital').value) || 500000,
    capital_risk_pct:  parseFloat(document.getElementById('cfg-risk').value) / 100 || 0.02,
    segment:           document.getElementById('cfg-segment').value,
    allow_shorting:    document.getElementById('cfg-shorting').checked,
    order_type:        document.getElementById('cfg-order-type').value,
    stop_loss_pct:     parseFloat(document.getElementById('cfg-sl').value) || 0,
    trailing_stop_pct: parseFloat(document.getElementById('cfg-trail').value) || 0,
    use_trailing_stop: parseFloat(document.getElementById('cfg-trail').value) > 0,
    save_chart:        document.getElementById('cfg-save-chart').checked,
    save_trade_log:    document.getElementById('cfg-save-trades').checked,
    save_raw_data:     document.getElementById('cfg-save-raw').checked,
    run_label:         document.getElementById('cfg-label').value || 'bt_run',
  };

  try {
    const res  = await fetch('/api/backtest', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      showAlert(`Error: ${data.detail || 'Backtest failed.'}`, 'error');
    } else {
      hideAllResults();
      renderBacktestResults(data, payload);
    }
  } catch (e) {
    showAlert(`Network error: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    if (spn) spn.className = 'spinner';
  }
}


// ── Render Backtest Results ───────────────────────────────────────────────────

function renderBacktestResults(data, payload) {
  const { metrics, trades, equity, drawdown, chart_b64 } = data;

  // Update topbar
  const meta = document.getElementById('topbar-meta');
  if (meta) meta.textContent = `${data.symbol} · ${data.strategy} · ${trades.length} trades`;

  // ── Metrics chips ─────────────────────────────────────────────────────────
  const metricsCard = document.getElementById('metrics-card');
  const metricsGrid = document.getElementById('metrics-grid');
  if (metrics && metricsGrid) {
    metricsGrid.innerHTML = '';
    const METRIC_DEFS = [
      { key: 'Total Return',       label: 'Total Return',       sign: true  },
      { key: 'CAGR',               label: 'CAGR',               sign: true  },
      { key: 'Sharpe Ratio',       label: 'Sharpe Ratio',       sign: true  },
      { key: 'Profit Factor',      label: 'Profit Factor',      sign: false },
      { key: 'Max Drawdown',       label: 'Max Drawdown',       sign: false, neg: true },
      { key: 'Win Rate',           label: 'Win Rate',           sign: false },
      { key: 'Total Trades',       label: 'Total Trades',       sign: false },
      { key: 'Total Net P&L',      label: 'Net P&L (₹)',        sign: true  },
      { key: 'Total Charges Paid', label: 'Total Charges',      sign: false },
      { key: 'Expectancy/Trade',   label: 'Expectancy / Trade', sign: true  },
    ];
    METRIC_DEFS.forEach(def => {
      const raw = metrics[def.key];
      if (raw === undefined) return;
      const chip = document.createElement('div');
      chip.className = 'metric-chip';
      const num = parseFloat(String(raw).replace(/[%₹,Rs]/g, '').trim());
      let cls = '';
      if (def.sign && !isNaN(num)) cls = num >= 0 ? 'pos' : 'neg';
      if (def.neg && !isNaN(num))  cls = num <= 0 ? 'neg' : 'pos';
      chip.innerHTML = `<div class="metric-label">${def.label}</div><div class="metric-value ${cls}">${raw}</div>`;
      metricsGrid.appendChild(chip);
    });
    metricsCard.style.display = 'block';
  }

  // ── Chart (Lightweight Charts — candles + markers) ─────────────────────────
  const chartCard = document.getElementById('chart-card');
  if (chartCard) {
    // If we have chart_b64 (static PNG), also wire up a download button
    if (chart_b64) {
      const dl = document.getElementById('chart-download');
      if (dl) {
        dl.href     = `data:image/png;base64,${chart_b64}`;
        dl.download = `${data.symbol}_chart.png`;
      }
    }
    // Fetch OHLCV and overlay buy/sell markers
    const fromDate = payload ? payload.from_date : '2020-01-01';
    const timeframe = payload ? payload.timeframe : 'daily';
    fetchAndRenderChart(data.symbol, timeframe, fromDate, '').then(() => {
      if (trades && trades.length > 0) {
        updateChart(null, trades);  // null bars = keep existing, just add markers
      }
    });
    chartCard.style.display = 'block';
  }

  // ── Mini equity + drawdown canvas charts ──────────────────────────────────
  const miniChartsEl = document.getElementById('mini-charts');
  if (miniChartsEl && equity && equity.length > 0) {
    drawLineCanvas('equity-canvas', equity.map(e => e.value), '#f0a500', `₹${_fmt(equity[equity.length-1].value)}`);
    if (drawdown && drawdown.length > 0) {
      drawLineCanvas('dd-canvas', drawdown.map(d => d.value), '#e53e3e', `${_fmt(Math.min(...drawdown.map(d=>d.value)))}%`);
    }
    miniChartsEl.style.display = 'grid';
  }

  // ── Trade log table ────────────────────────────────────────────────────────
  const tradesCard  = document.getElementById('trades-card');
  const tradesTbody = document.getElementById('trades-tbody');
  if (tradesTbody && trades && trades.length > 0) {
    tradesTbody.innerHTML = '';
    trades.forEach((t, i) => {
      const pnl = parseFloat(t.net_pnl) || 0;
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${i + 1}</td>
        <td><span class="badge ${t.direction === 1 ? 'badge-green' : 'badge-red'}">${t.direction_label || (t.direction === 1 ? 'LONG' : 'SHORT')}</span></td>
        <td>${_shortDate(t.entry_time)}</td>
        <td>₹${_fmt(t.entry_price)}</td>
        <td>${_shortDate(t.exit_time)}</td>
        <td>₹${_fmt(t.exit_price)}</td>
        <td>${t.quantity}</td>
        <td class="${t.gross_pnl >= 0 ? 'pos' : 'neg'}">₹${_fmt(t.gross_pnl)}</td>
        <td style="color:var(--muted);">₹${_fmt(t.total_charges)}</td>
        <td class="${pnl >= 0 ? 'pos' : 'neg'}"><strong>₹${_fmt(pnl)}</strong></td>
        <td class="${pnl >= 0 ? 'pos' : 'neg'}">${_fmt(t.pnl_pct)}%</td>
        <td style="color:var(--muted);">${t.duration || '—'}</td>
        <td style="color:var(--muted);font-size:11px;">${t.exit_signal || '—'}</td>
      `;
      tradesTbody.appendChild(row);
    });
    tradesCard.style.display = 'block';
  }

  hideAlert();
}


// ── Equity / Drawdown mini canvas chart ───────────────────────────────────────

function drawLineCanvas(canvasId, values, color, label) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  // Set actual pixel dimensions (avoid blurry HiDPI rendering)
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = Math.floor(rect.width) || 400;
  const H = 90;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const n     = values.length;
  const min   = Math.min(...values);
  const max   = Math.max(...values);
  const range = max - min || 1;

  // Background
  ctx.fillStyle = '#111419';
  ctx.fillRect(0, 0, W, H);

  // Zero line (for drawdown chart it's at max)
  const zeroY = H - ((0 - min) / range) * (H - 12) - 6;
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = '#262b34';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, Math.min(zeroY, H - 4)); ctx.lineTo(W, Math.min(zeroY, H - 4));
  ctx.stroke();
  ctx.setLineDash([]);

  // Area fill under line
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (n - 1)) * W;
    const y = H - ((v - min) / range) * (H - 12) - 6;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  const lastX = W;
  const lastY = H - ((values[n-1] - min) / range) * (H - 12) - 6;
  ctx.lineTo(lastX, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, color + '40');
  grad.addColorStop(1, color + '05');
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (n - 1)) * W;
    const y = H - ((v - min) / range) * (H - 12) - 6;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.stroke();

  // Label (last value)
  ctx.fillStyle = color;
  ctx.font      = '10px "IBM Plex Mono", monospace';
  ctx.fillText(label, 6, 14);
}


// ── Optimizer ─────────────────────────────────────────────────────────────────

function toggleOptimizer() {
  const body  = document.getElementById('optimizer-body');
  const label = document.getElementById('opt-toggle-label');
  if (!body) return;
  const open = body.style.display !== 'none';
  body.style.display  = open ? 'none' : 'block';
  label.textContent   = open ? '▼ expand' : '▲ collapse';
}

function _populateOptParamGrid(params) {
  const container = document.getElementById('opt-params');
  if (!container) return;
  container.innerHTML = '';
  (params || []).forEach(p => {
    if (p.type !== 'int' && p.type !== 'float') return;
    const row = document.createElement('div');
    row.className = 'param-row';
    row.innerHTML = `
      <div class="form-group" style="margin:0;">
        <label>${p.name}</label>
        <input type="text" id="opt-param-${p.name}"
               value="${p.default || ''}"
               placeholder="e.g. 5,9,13,21"
               style="font-size:12px;">
      </div>
      <div class="form-group" style="margin:0;">
        <label style="visibility:hidden;">.</label>
        <input type="text" readonly value="comma-separated values" style="font-size:11px;color:var(--muted);cursor:default;">
      </div>
    `;
    container.appendChild(row);
  });
}

function addOptParam() {
  const container = document.getElementById('opt-params');
  if (!container) return;
  const row = document.createElement('div');
  row.className = 'param-row';
  row.innerHTML = `
    <div class="form-group" style="margin:0;">
      <label>Param Name</label>
      <input type="text" class="opt-custom-name" placeholder="fast_period">
    </div>
    <div class="form-group" style="margin:0;">
      <label>Values</label>
      <input type="text" class="opt-custom-vals" placeholder="5,9,13,21">
    </div>
    <button class="btn btn-ghost" onclick="this.closest('.param-row').remove()"
            style="padding:6px 8px; align-self:flex-end; margin-bottom:12px;">✕</button>
  `;
  container.appendChild(row);
}

async function runOptimizer() {
  const strategyName = document.getElementById('cfg-strategy').value;
  const symbol       = document.getElementById('cfg-symbol').value.trim().toUpperCase();

  // Collect param grid from all opt-param inputs
  const paramGrid = {};

  // Named params from strategy schema
  if (_registry && _registry.strategies[strategyName]) {
    _registry.strategies[strategyName].params.forEach(p => {
      const el = document.getElementById(`opt-param-${p.name}`);
      if (!el || !el.value.trim()) return;
      const vals = el.value.split(',').map(v => {
        const n = parseFloat(v.trim());
        return isNaN(n) ? v.trim() : n;
      });
      if (vals.length > 0) paramGrid[p.name] = vals;
    });
  }

  // Custom params from addOptParam rows
  document.querySelectorAll('.param-row').forEach(row => {
    const nameEl = row.querySelector('.opt-custom-name');
    const valEl  = row.querySelector('.opt-custom-vals');
    if (!nameEl || !valEl) return;
    const name = nameEl.value.trim();
    const vals = valEl.value.split(',').map(v => {
      const n = parseFloat(v.trim()); return isNaN(n) ? v.trim() : n;
    }).filter(Boolean);
    if (name && vals.length > 0) paramGrid[name] = vals;
  });

  if (Object.keys(paramGrid).length === 0) {
    showAlert('Add at least one parameter range to optimise.', 'warn');
    return;
  }

  showAlert('Running optimizer… this may take a minute.', 'info');

  const payload = {
    symbol,
    strategy_name:   strategyName,
    param_grid:      paramGrid,
    metric:          document.getElementById('opt-metric').value,
    method:          document.getElementById('opt-method').value,
    n_random:        parseInt(document.getElementById('opt-n-random').value, 10) || 30,
    timeframe:       document.getElementById('cfg-timeframe').value,
    from_date:       document.getElementById('cfg-from').value,
    initial_capital: parseFloat(document.getElementById('cfg-capital').value) || 500000,
    segment:         document.getElementById('cfg-segment').value,
  };

  try {
    const res  = await fetch('/api/optimize', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      showAlert(`Optimizer error: ${data.detail || 'Failed.'}`, 'error');
      return;
    }
    renderOptimizerResults(data.results, data.metric);
    hideAlert();
  } catch (e) {
    showAlert(`Network error: ${e.message}`, 'error');
  }
}

function renderOptimizerResults(results, metric) {
  const card   = document.getElementById('opt-results-card');
  const thead  = document.getElementById('opt-results-thead');
  const tbody  = document.getElementById('opt-results-tbody');
  if (!card || !results || results.length === 0) {
    showAlert('Optimizer returned no results.', 'warn'); return;
  }

  const cols = Object.keys(results[0]);
  thead.innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;
  tbody.innerHTML = '';
  results.forEach((row, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = cols.map(c => {
      const v = row[c];
      const isMetric = c === metric;
      const cls = isMetric && i === 0 ? 'pos' : '';
      return `<td class="${cls}">${typeof v === 'number' ? _fmt(v) : v}</td>`;
    }).join('');
    tbody.appendChild(tr);
  });
  card.style.display = 'block';
}


// ── UI helpers ────────────────────────────────────────────────────────────────

function hideAllResults() {
  ['chart-card','metrics-card','mini-charts','trades-card','opt-results-card'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
}

function showAlert(msg, type = 'info') {
  const el = document.getElementById('result-alert');
  if (!el) return;
  el.className = `alert alert-${type}`;
  el.textContent = msg;
  el.style.display = 'block';
}

function hideAlert() {
  const el = document.getElementById('result-alert');
  if (el) el.style.display = 'none';
}

function _fmt(n) {
  if (n === undefined || n === null || n === '') return '—';
  const num = parseFloat(n);
  if (isNaN(num)) return String(n);
  if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(2) + 'M';
  if (Math.abs(num) >= 1e3) return num.toLocaleString('en-IN', { maximumFractionDigits: 2 });
  return num.toFixed(2);
}

function _shortDate(isoStr) {
  if (!isoStr) return '—';
  return String(isoStr).slice(0, 10);
}
