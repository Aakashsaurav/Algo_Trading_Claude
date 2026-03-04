/**
 * dashboard/static/js/screener_table.js
 * ---------------------------------------
 * Handles the Screener page:
 *   - Loads strategy list and populates config dropdown
 *   - Renders dynamic strategy parameter controls
 *   - Posts to /api/screener/scan and renders results table
 *   - Supports column sorting
 *   - CSV download via Blob URL
 *
 * This file was MISSING in previous build (screener.html referenced it
 * but it did not exist → screener page broken with 404).
 */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let _scRegistry    = null;    // strategy registry cache
let _scResults     = [];      // last scan results array
let _scSortCol     = '';      // currently sorted column
let _scSortAsc     = true;    // sort direction

// ── Page init ─────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  await _loadScStrategies();

  // If navigated from strategy builder with a pre-selected strategy
  const stored = sessionStorage.getItem('sc_strategy');
  if (stored) {
    const sel = document.getElementById('sc-strategy');
    if (sel) {
      // Try to set the stored value once options are loaded
      setTimeout(() => {
        sel.value = stored;
        sessionStorage.removeItem('sc_strategy');
        onScStrategyChange();
      }, 100);
    }
  }
});


// ── Load strategy list ────────────────────────────────────────────────────────

async function _loadScStrategies() {
  try {
    const res      = await fetch('/api/strategies');
    _scRegistry    = await res.json();
    const sel      = document.getElementById('sc-strategy');
    if (!sel) return;
    sel.innerHTML  = '';

    for (const [name, schema] of Object.entries(_scRegistry.strategies || {})) {
      const opt       = document.createElement('option');
      opt.value       = name;
      opt.textContent = `${schema.display_name} [${schema.category}]`;
      sel.appendChild(opt);
    }

    // Render params for default selection
    onScStrategyChange();
  } catch (e) {
    console.error('screener: failed to load strategies:', e);
    _scShowAlert('Failed to load strategies. Is the backend running?', 'error');
  }
}


/**
 * Called when strategy dropdown changes.
 * Renders dynamic parameter controls for the selected strategy.
 */
function onScStrategyChange() {
  const sel  = document.getElementById('sc-strategy');
  if (!sel || !_scRegistry) return;
  const name   = sel.value;
  const schema = (_scRegistry.strategies || {})[name];
  if (!schema) return;

  const container = document.getElementById('sc-params-container');
  const card      = document.getElementById('sc-params-card');
  if (!container || !card) return;
  container.innerHTML = '';

  const numericParams = (schema.params || []).filter(
    p => p.type === 'int' || p.type === 'float' || p.type === 'bool'
  );

  if (numericParams.length > 0) {
    numericParams.forEach(p => container.appendChild(_scBuildParamControl(p)));
    card.style.display = 'block';
  } else {
    card.style.display = 'none';
  }
}


/** Build a single parameter form control for the screener config panel. */
function _scBuildParamControl(param) {
  const group = document.createElement('div');
  group.className = 'form-group';

  if (param.type === 'bool') {
    const row = document.createElement('div');
    row.className = 'checkbox-row';
    const cb  = document.createElement('input');
    cb.type = 'checkbox'; cb.id = `sc-param-${param.name}`; cb.checked = !!param.default;
    const lbl = document.createElement('label');
    lbl.setAttribute('for', cb.id); lbl.textContent = param.name;
    row.appendChild(cb); row.appendChild(lbl);
    group.appendChild(row);
    return group;
  }

  const label = document.createElement('label');
  label.textContent = param.name;
  group.appendChild(label);

  const input = document.createElement('input');
  input.type  = (param.type === 'int' || param.type === 'float') ? 'number' : 'text';
  input.id    = `sc-param-${param.name}`;
  input.value = param.default !== null ? param.default : '';
  if (param.min  != null) input.min  = param.min;
  if (param.max  != null) input.max  = param.max;
  if (param.step != null) input.step = param.step;
  if (param.type === 'int') input.step = '1';
  group.appendChild(input);
  return group;
}


/** Collect param values from screener dynamic controls. */
function _scCollectParams(strategyName) {
  if (!_scRegistry) return {};
  const schema = (_scRegistry.strategies || {})[strategyName];
  if (!schema) return {};
  const params = {};
  (schema.params || []).forEach(p => {
    const el = document.getElementById(`sc-param-${p.name}`);
    if (!el) return;
    if (p.type === 'bool')       params[p.name] = el.checked;
    else if (p.type === 'int')   params[p.name] = parseInt(el.value, 10)   || p.default;
    else if (p.type === 'float') params[p.name] = parseFloat(el.value)     || p.default;
    else                         params[p.name] = el.value;
  });
  return params;
}


// ── Run Screener ──────────────────────────────────────────────────────────────

/**
 * Main screener function — called when user clicks "Run Scan".
 * Sends request to /api/screener/scan and renders results.
 */
async function runScreener() {
  const strategyName = document.getElementById('sc-strategy')?.value;
  if (!strategyName) {
    _scShowAlert('Select a strategy first.', 'warn'); return;
  }

  // Show spinner
  const spinner = document.getElementById('top-spinner');
  if (spinner) spinner.className = 'spinner active';

  // Hide previous results
  _scHideResults();
  _scShowAlert('Scanning universe… this may take a moment.', 'info');

  const payload = {
    strategy_name:   strategyName,
    strategy_params: _scCollectParams(strategyName),
    signal_type:     parseInt(document.getElementById('sc-signal-type')?.value ?? '1', 10),
    timeframe:       document.getElementById('sc-timeframe')?.value   || 'daily',
    from_date:       document.getElementById('sc-from')?.value        || '2022-01-01',
    min_volume:      parseFloat(document.getElementById('sc-min-vol')?.value)  || 200000,
    min_price:       parseFloat(document.getElementById('sc-min-price')?.value) || 50,
    max_results:     parseInt(document.getElementById('sc-max-results')?.value, 10) || 30,
    rank_by:         document.getElementById('sc-rank-by')?.value     || 'close',
  };

  try {
    const res  = await fetch('/api/screener/scan', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      _scShowAlert(`Error: ${data.detail || 'Scan failed.'}`, 'error');
      return;
    }

    _scResults = data.results || [];
    _scRenderResults(data);
    _scHideAlert();

  } catch (e) {
    _scShowAlert(`Network error: ${e.message}`, 'error');
  } finally {
    if (spinner) spinner.className = 'spinner';
  }
}


// ── Render Results ────────────────────────────────────────────────────────────

/**
 * Render scan results: summary bar + sortable table + CSV download.
 *
 * @param {Object} data - response from /api/screener/scan
 */
function _scRenderResults(data) {
  const results = data.results || [];

  // ── Summary bar ─────────────────────────────────────────────────────────
  const summary = document.getElementById('sc-summary');
  if (summary) {
    document.getElementById('sc-count-scanned').textContent = data.scanned  || 0;
    document.getElementById('sc-count-hits').textContent    = data.hits     || 0;
    document.getElementById('sc-strategy-label').textContent = data.strategy || '—';
    summary.style.display = 'block';
  }

  // ── Empty state ──────────────────────────────────────────────────────────
  const emptyEl = document.getElementById('sc-empty');
  if (results.length === 0) {
    if (emptyEl) {
      emptyEl.style.display = 'flex';
      emptyEl.innerHTML = `
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        <div>No signals found matching the criteria.</div>
        <div style="font-size:11px;color:var(--muted);margin-top:4px;">
          Try relaxing filters, changing strategy, or extending the date range.
        </div>`;
    }
    return;
  }
  if (emptyEl) emptyEl.style.display = 'none';

  // ── Table ────────────────────────────────────────────────────────────────
  const tableCard = document.getElementById('sc-table-card');
  const thead     = document.getElementById('sc-thead');
  const tbody     = document.getElementById('sc-tbody');
  if (!tableCard || !thead || !tbody) return;

  // Determine columns from first result object
  const cols = Object.keys(results[0]);

  // Build header with sort click handlers
  thead.innerHTML = `<tr>
    ${cols.map(c => `
      <th onclick="scSortBy('${c}')" style="cursor:pointer; user-select:none;">
        ${_scFmtColName(c)}
        <span id="sc-sort-${c}" style="font-size:9px;color:var(--muted);">⇅</span>
      </th>`
    ).join('')}
  </tr>`;

  _scSortCol = '';
  _scRenderTableRows(tbody, results, cols);

  tableCard.style.display = 'block';

  // ── CSV download ─────────────────────────────────────────────────────────
  const dlBtn = document.getElementById('sc-download');
  if (dlBtn) {
    const csvRows = [cols.join(',')];
    results.forEach(r => {
      csvRows.push(cols.map(c => {
        const v = r[c];
        // Escape commas/quotes in string values
        if (typeof v === 'string' && (v.includes(',') || v.includes('"'))) {
          return `"${v.replace(/"/g, '""')}"`;
        }
        return v === null || v === undefined ? '' : v;
      }).join(','));
    });
    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
    dlBtn.href     = URL.createObjectURL(blob);
    dlBtn.download = `screener_${data.strategy}_${new Date().toISOString().slice(0,10)}.csv`;
  }
}


/** Render table body rows from results array. */
function _scRenderTableRows(tbody, results, cols) {
  tbody.innerHTML = '';
  results.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = cols.map(c => {
      const v = row[c];
      return `<td>${_scFmtCell(c, v)}</td>`;
    }).join('');
    tbody.appendChild(tr);
  });
}


/**
 * Sort results table by a column when header is clicked.
 * Toggles asc/desc on repeated clicks of the same column.
 *
 * @param {string} col - column key to sort by
 */
function scSortBy(col) {
  if (_scSortCol === col) {
    _scSortAsc = !_scSortAsc;    // toggle direction on same column
  } else {
    _scSortCol = col;
    _scSortAsc = false;          // default: highest value first
  }

  // Update sort indicator icons
  document.querySelectorAll('[id^="sc-sort-"]').forEach(el => {
    el.textContent = '⇅';
    el.style.color = 'var(--muted)';
  });
  const indicator = document.getElementById(`sc-sort-${col}`);
  if (indicator) {
    indicator.textContent = _scSortAsc ? '▲' : '▼';
    indicator.style.color = 'var(--amber)';
  }

  const sorted = [..._scResults].sort((a, b) => {
    const va = a[col]; const vb = b[col];
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    const numA = parseFloat(va); const numB = parseFloat(vb);
    if (!isNaN(numA) && !isNaN(numB)) {
      return _scSortAsc ? numA - numB : numB - numA;
    }
    return _scSortAsc
      ? String(va).localeCompare(String(vb))
      : String(vb).localeCompare(String(va));
  });

  const tbody = document.getElementById('sc-tbody');
  const cols  = Object.keys(_scResults[0] || {});
  if (tbody && cols.length) _scRenderTableRows(tbody, sorted, cols);
}


// ── Formatting helpers ────────────────────────────────────────────────────────

/**
 * Format a column name for display in the table header.
 * e.g. "close_price" → "Close Price"
 */
function _scFmtColName(col) {
  return col.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}


/**
 * Format a single table cell value with appropriate colour / badge.
 *
 * @param {string} col - column key (used to detect signal/PnL columns)
 * @param {*}      val - raw value
 */
function _scFmtCell(col, val) {
  if (val === null || val === undefined) return '<span style="color:var(--muted);">—</span>';

  // Signal column: coloured badge
  if (col === 'signal' || col === 'direction') {
    const n = parseInt(val, 10);
    if (n === 1)  return '<span class="badge badge-green">BUY</span>';
    if (n === -1) return '<span class="badge badge-red">SELL</span>';
    return `<span class="badge">${val}</span>`;
  }

  // Numeric columns: colour positive/negative
  if (col.includes('pnl') || col.includes('return') || col.includes('change')) {
    const n = parseFloat(val);
    if (!isNaN(n)) {
      const cls = n >= 0 ? 'pos' : 'neg';
      return `<span class="${cls}">${n >= 0 ? '+' : ''}${n.toFixed(2)}</span>`;
    }
  }

  // Price columns: 2 decimal places
  if (col === 'close' || col === 'open' || col === 'high' || col === 'low' ||
      col.includes('price') || col.includes('ltp')) {
    const n = parseFloat(val);
    if (!isNaN(n)) return `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
  }

  // Volume: abbreviated
  if (col === 'volume') {
    const n = parseFloat(val);
    if (!isNaN(n)) {
      if (n >= 1e7) return `${(n/1e7).toFixed(2)}Cr`;
      if (n >= 1e5) return `${(n/1e5).toFixed(2)}L`;
      return n.toLocaleString('en-IN');
    }
  }

  // Date: short form
  if (col.includes('date') || col.includes('time')) {
    return `<span style="font-family:var(--mono);font-size:11px;">${String(val).slice(0,10)}</span>`;
  }

  // Symbol: monospaced + highlighted
  if (col === 'symbol') {
    return `<strong style="font-family:var(--mono);color:var(--text);">${val}</strong>`;
  }

  // Default: plain
  const n = parseFloat(val);
  if (!isNaN(n) && typeof val === 'number') return n.toFixed(2);
  return String(val);
}


// ── UI helpers ─────────────────────────────────────────────────────────────────

function _scShowAlert(msg, type = 'info') {
  const el = document.getElementById('sc-alert');
  if (!el) return;
  el.className    = `alert alert-${type}`;
  el.textContent  = msg;
  el.style.display = 'block';
}

function _scHideAlert() {
  const el = document.getElementById('sc-alert');
  if (el) el.style.display = 'none';
}

function _scHideResults() {
  ['sc-summary', 'sc-table-card'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  const emptyEl = document.getElementById('sc-empty');
  if (emptyEl) emptyEl.style.display = 'flex';
}
