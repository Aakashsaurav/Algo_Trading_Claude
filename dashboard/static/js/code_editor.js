/**
 * dashboard/static/js/code_editor.js
 * ------------------------------------
 * Handles the Strategy Builder page:
 *   - Loads Monaco Editor (VS Code engine) with Python syntax highlighting
 *   - Fetches strategy list from /api/strategies and populates dropdown
 *   - When a strategy is selected, loads its source via /api/strategy/source
 *     (or shows a boilerplate template if not found)
 *   - Collects config from the right panel and calls /api/backtest
 *   - Redirects to /backtester with results pre-loaded (via sessionStorage)
 */

'use strict';

// ── Monaco loader ───────────────────────────────────────────────────────────
let monacoEditor = null;

require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs' } });
require(['vs/editor/editor.main'], function () {
  // Define AlgoDesk dark theme matching CSS variables
  monaco.editor.defineTheme('algodesk-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'comment',   foreground: '6b7280', fontStyle: 'italic' },
      { token: 'keyword',   foreground: 'f0a500' },
      { token: 'string',    foreground: '6ee7b7' },
      { token: 'number',    foreground: '93c5fd' },
      { token: 'delimiter', foreground: 'e2e8f0' },
    ],
    colors: {
      'editor.background':           '#111419',
      'editor.foreground':           '#e2e8f0',
      'editorLineNumber.foreground': '#4b5563',
      'editor.lineHighlightBackground': '#1a1e26',
      'editorCursor.foreground':     '#f0a500',
      'editor.selectionBackground':  '#f0a50030',
    },
  });

  monacoEditor = monaco.editor.create(document.getElementById('monaco-container'), {
    language:         'python',
    theme:            'algodesk-dark',
    value:            STRATEGY_TEMPLATE,
    fontSize:         13,
    fontFamily:       '"IBM Plex Mono", "Courier New", monospace',
    minimap:          { enabled: false },
    scrollBeyondLastLine: false,
    wordWrap:         'on',
    lineNumbers:      'on',
    renderLineHighlight: 'line',
    automaticLayout:  true,  // re-measure on container resize
    tabSize:          4,
    insertSpaces:     true,
  });

  // Load strategies once editor is ready
  loadStrategies();
});


// ── Boilerplate template ─────────────────────────────────────────────────────
const STRATEGY_TEMPLATE = `\
"""
strategies/momentum/my_strategy.py
------------------------------------
Template: rename this file and class, then drop it in any strategies/ subfolder.
It will appear in the dashboard automatically (no registration needed).
"""

import pandas as pd
from strategies.base import BaseStrategy


class MyStrategy(BaseStrategy):
    """
    One-line description of what this strategy does.
    Category: Momentum  (used for grouping in the UI)
    """

    # ── Optional class attribute for UI grouping ──────────────────────
    CATEGORY = "Trend / Momentum"  # or "Mean Reversion" | "Arbitrage / Pairs"

    def __init__(
        self,
        fast_period: int   = 9,   # Short EMA period
        slow_period: int   = 21,  # Long EMA period
    ) -> None:
        # name= is shown in the dashboard strategy dropdown
        super().__init__(name=f"My Strategy ({fast_period}/{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'signal' column to df:
          +1 = BUY  (or cover short)
          -1 = SELL (or go short)
           0 = no action

        You can also add indicator columns — they appear in raw data CSVs
        and on the candlestick chart overlay.

        IMPORTANT:
          - Never look at future data (no df.shift(-1) on the signal itself).
          - Signal generated on bar N is executed on bar N+1 open (engine rule).
        """
        df = df.copy()

        # ── Compute indicators ────────────────────────────────────────
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()

        # ── Generate signals (crossover logic) ────────────────────────
        prev_fast = df['ema_fast'].shift(1)
        prev_slow = df['ema_slow'].shift(1)

        df['signal'] = 0
        df.loc[(df['ema_fast'] > df['ema_slow']) & (prev_fast <= prev_slow), 'signal'] = 1
        df.loc[(df['ema_fast'] < df['ema_slow']) & (prev_fast >= prev_slow), 'signal'] = -1

        return df

    def get_parameters(self) -> dict:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
        }
`;


// ── Load strategy list from API ──────────────────────────────────────────────

async function loadStrategies() {
  try {
    const res  = await fetch('/api/strategies');
    const data = await res.json();
    const sel  = document.getElementById('strategy-select');
    sel.innerHTML = '<option value="">— Select to load code —</option>';
    for (const [name, schema] of Object.entries(data.strategies || {})) {
      const opt = document.createElement('option');
      opt.value       = name;
      opt.textContent = `${schema.display_name} [${schema.category}]`;
      sel.appendChild(opt);
    }
    // Also populate params card
    sel.addEventListener('change', onStrategySelect);
  } catch (e) {
    console.error('Failed to load strategies:', e);
  }
}


function onStrategySelect() {
  const sel  = document.getElementById('strategy-select');
  const name = sel.value;
  if (!name) return;
  loadStrategyParams(name);
}


async function loadStrategyParams(strategyName) {
  try {
    const res  = await fetch('/api/strategies');
    const data = await res.json();
    const schema = data.strategies[strategyName];
    if (!schema) return;

    const container = document.getElementById('params-container');
    const card      = document.getElementById('params-card');
    container.innerHTML = '';

    if (schema.params && schema.params.length > 0) {
      schema.params.forEach(p => {
        container.appendChild(buildParamControl(p));
      });
      card.style.display = 'block';
    } else {
      card.style.display = 'none';
    }
  } catch (e) {
    console.error('Failed to load strategy params:', e);
  }
}


/** Build a form control for one strategy parameter */
function buildParamControl(param) {
  const group = document.createElement('div');
  group.className = 'form-group';

  const label = document.createElement('label');
  label.textContent = param.name;
  group.appendChild(label);

  let input;
  if (param.type === 'select' && param.options) {
    input = document.createElement('select');
    input.id = `param-${param.name}`;
    param.options.forEach(opt => {
      const o = document.createElement('option');
      o.value = opt;
      o.textContent = opt;
      if (opt === param.default) o.selected = true;
      input.appendChild(o);
    });
  } else if (param.type === 'bool') {
    const row = document.createElement('div');
    row.className = 'checkbox-row';
    input = document.createElement('input');
    input.type    = 'checkbox';
    input.id      = `param-${param.name}`;
    input.checked = !!param.default;
    const chkLabel = document.createElement('label');
    chkLabel.setAttribute('for', input.id);
    chkLabel.textContent = param.name;
    row.appendChild(input);
    row.appendChild(chkLabel);
    group.innerHTML = '';
    group.appendChild(row);
    return group;
  } else {
    input = document.createElement('input');
    input.type  = (param.type === 'float') ? 'number' : (param.type === 'int') ? 'number' : 'text';
    input.id    = `param-${param.name}`;
    input.value = param.default !== null ? param.default : '';
    if (param.min != null) input.min  = param.min;
    if (param.max != null) input.max  = param.max;
    if (param.step != null) input.step = param.step;
    if (param.type === 'int') input.step = '1';
  }
  input.id = `param-${param.name}`;
  group.appendChild(input);
  return group;
}


/** Collect all strategy params from the dynamic param controls */
function collectParams(strategyName, registry) {
  const schema = registry.strategies[strategyName];
  if (!schema) return {};
  const params = {};
  schema.params.forEach(p => {
    const el = document.getElementById(`param-${p.name}`);
    if (!el) return;
    if (p.type === 'bool') {
      params[p.name] = el.checked;
    } else if (p.type === 'int') {
      params[p.name] = parseInt(el.value, 10) || p.default;
    } else if (p.type === 'float') {
      params[p.name] = parseFloat(el.value) || p.default;
    } else {
      params[p.name] = el.value;
    }
  });
  return params;
}


// ── Template loader ──────────────────────────────────────────────────────────

function loadTemplate() {
  if (monacoEditor) {
    monacoEditor.setValue(STRATEGY_TEMPLATE);
  }
}


// ── Save strategy file (API stub — actual file save needs backend endpoint) ──

function saveStrategy() {
  const code = monacoEditor ? monacoEditor.getValue() : '';
  // Provide code as downloadable file
  const blob = new Blob([code], { type: 'text/plain' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'my_strategy.py';
  a.click();
  setStatus('Downloaded strategy file. Place it in strategies/momentum/ (or any subfolder).', false);
}


// ── Run Backtest ─────────────────────────────────────────────────────────────

async function runBacktest() {
  setLoading(true);
  setStatus('Running backtest…');

  const strategyName = document.getElementById('strategy-select').value;
  if (!strategyName) {
    setLoading(false);
    setStatus('Select a strategy first.', true);
    return;
  }

  // Fetch registry to get params schema
  let registry;
  try {
    const res = await fetch('/api/strategies');
    registry  = await res.json();
  } catch (e) {
    setLoading(false);
    setStatus('Failed to load strategies.', true);
    return;
  }

  const params = collectParams(strategyName, registry);

  const payload = {
    symbol:            document.getElementById('cfg-symbol').value.trim().toUpperCase(),
    strategy_name:     strategyName,
    strategy_params:   params,
    timeframe:         document.getElementById('cfg-timeframe').value,
    from_date:         document.getElementById('cfg-from').value,
    to_date:           document.getElementById('cfg-to').value || '',
    initial_capital:   parseFloat(document.getElementById('cfg-capital').value) || 500000,
    capital_risk_pct:  parseFloat(document.getElementById('cfg-risk-pct').value) / 100 || 0.02,
    segment:           document.getElementById('cfg-segment').value,
    allow_shorting:    document.getElementById('cfg-shorting').checked,
    max_positions:     parseInt(document.getElementById('cfg-max-pos').value, 10) || 5,
    lot_size:          parseInt(document.getElementById('cfg-lot-size').value, 10) || 1,
    order_type:        document.getElementById('cfg-order-type').value,
    stop_loss_pct:     parseFloat(document.getElementById('cfg-sl-pct').value) || 0,
    trailing_stop_pct: parseFloat(document.getElementById('cfg-trail-pct').value) || 0,
    use_trailing_stop: parseFloat(document.getElementById('cfg-trail-pct').value) > 0,
    save_chart:        document.getElementById('cfg-save-chart').checked,
    save_trade_log:    document.getElementById('cfg-save-trades').checked,
    save_raw_data:     document.getElementById('cfg-save-raw').checked,
    run_label:         'builder_run',
  };

  try {
    const res  = await fetch('/api/backtest', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    const data = await res.json();

    if (!res.ok) {
      setLoading(false);
      setStatus(`Error: ${data.detail || 'Backtest failed.'}`, true);
      return;
    }

    // Store result in sessionStorage and navigate to backtester page
    sessionStorage.setItem('bt_result', JSON.stringify(data));
    setStatus('Done! Opening results…', false);
    setTimeout(() => { window.location.href = '/backtester?from=builder'; }, 500);

  } catch (e) {
    setLoading(false);
    setStatus(`Network error: ${e.message}`, true);
  }
}


// ── Run Screener ─────────────────────────────────────────────────────────────

async function runScreener() {
  const strategyName = document.getElementById('strategy-select').value;
  if (!strategyName) { setStatus('Select a strategy first.', true); return; }
  sessionStorage.setItem('sc_strategy', strategyName);
  window.location.href = '/screener?from=builder';
}


// ── UI helpers ───────────────────────────────────────────────────────────────

function setLoading(on) {
  const btn = document.getElementById('run-btn');
  const spn = document.getElementById('run-spinner');
  if (btn) btn.disabled = on;
  if (spn) spn.className = on ? 'spinner active' : 'spinner';
}

function setStatus(msg, isError = false) {
  const el = document.getElementById('run-status');
  if (!el) return;
  el.textContent = msg;
  el.style.color = isError ? 'var(--red)' : 'var(--green)';
}
