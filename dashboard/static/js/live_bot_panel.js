/**
 * dashboard/static/js/live_bot_panel.js
 * ----------------------------------------
 * Handles the Live Bot page:
 *   - Loads strategy list and populates the bot config dropdown
 *   - Checks Upstox auth token status via /api/auth/status
 *   - Simulates paper trading (Phase 5 backend not yet wired)
 *   - Updates P&L display, positions table, completed trades table
 *   - Activity log with timestamped entries
 *   - Bot status badge (Stopped / Paper / Live)
 *
 * This file was MISSING in previous build (live_bot.html referenced it
 * but it did not exist → Live Bot page broken with 404).
 *
 * NOTE: Phase 5 (live order execution via Upstox API) will wire up the
 *       actual backend WebSocket in the next build. This file provides
 *       the complete UI layer — all elements are interactive and functional
 *       with demo simulation. Only the real broker order routing is pending.
 */

'use strict';

// ── Bot state ─────────────────────────────────────────────────────────────────
const _bot = {
  running:      false,              // is bot active?
  mode:         'paper',           // 'paper' | 'live'
  intervalId:   null,              // setInterval handle for tick simulation
  tickCount:    0,                 // how many ticks have fired
  realisedPnl:  0,                 // cumulative realised P&L for today
  unrealisedPnl:0,                 // running unrealised P&L
  tradesToday:  0,                 // completed trades count
  positions:    [],                // open positions [{symbol, dir, qty, entry, ltp, stop}]
  completed:    [],                // closed trades today
  dailyLimit:   0,                 // max daily loss (₹) — computed from capital × %
};

// ── Page init ─────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  await _botLoadStrategies();
  checkAuth();
  _botLog('info', 'Dashboard ready. Configure strategy and press Start.');
});


// ── Load strategies ───────────────────────────────────────────────────────────

async function _botLoadStrategies() {
  try {
    const res  = await fetch('/api/strategies');
    const data = await res.json();
    const sel  = document.getElementById('bot-strategy');
    if (!sel) return;
    sel.innerHTML = '';
    for (const [name, schema] of Object.entries(data.strategies || {})) {
      const opt       = document.createElement('option');
      opt.value       = name;
      opt.textContent = `${schema.display_name}`;
      sel.appendChild(opt);
    }
  } catch (e) {
    console.error('live_bot: failed to load strategies:', e);
    _botLog('warn', 'Could not load strategy list. Is backend running?');
  }
}


// ── Auth check ────────────────────────────────────────────────────────────────

/**
 * Check whether a valid Upstox token exists.
 * Calls /api/auth/status — gracefully handles 404 (endpoint not yet wired).
 */
async function checkAuth() {
  const el = document.getElementById('auth-status');
  if (!el) return;
  el.className   = 'alert alert-info';
  el.textContent = 'Checking Upstox token…';

  try {
    const res  = await fetch('/api/auth/status');
    if (res.ok) {
      const data = await res.json();
      if (data.authenticated) {
        el.className   = 'alert alert-ok';
        el.textContent = `✓ Upstox connected — ${data.user_name || data.client_id || 'account active'}`;
        _botLog('ok', 'Upstox token valid. Ready for paper/live trading.');
      } else {
        el.className   = 'alert alert-warn';
        el.textContent = '⚠ Upstox token expired or missing. Obtain a fresh token first.';
        _botLog('warn', 'Upstox token invalid. Live trading will not work.');
      }
    } else if (res.status === 404) {
      // Endpoint not yet implemented (Phase 5)
      el.className   = 'alert alert-warn';
      el.textContent = '⚠ Auth API not yet wired (Phase 5). Paper trading available.';
    } else {
      el.className   = 'alert alert-error';
      el.textContent = `Auth check failed (HTTP ${res.status}).`;
    }
  } catch (e) {
    el.className   = 'alert alert-warn';
    el.textContent = `Could not reach auth endpoint: ${e.message}`;
    _botLog('warn', 'Auth endpoint unreachable. Backend may be starting up.');
  }
}


// ── Bot controls ──────────────────────────────────────────────────────────────

/**
 * Start the bot. In paper mode, runs a simulated tick loop.
 * In live mode, shows a confirmation dialog first.
 */
async function startBot() {
  if (_bot.running) return;

  const mode    = document.getElementById('bot-mode')?.value    || 'paper';
  const capital = parseFloat(document.getElementById('bot-capital')?.value) || 500_000;
  const maxLoss = parseFloat(document.getElementById('bot-max-daily-loss')?.value) || 2;
  const strategy= document.getElementById('bot-strategy')?.value  || '—';
  const symbol  = document.getElementById('bot-symbol')?.value     || 'NIFTY';
  const timeframe = document.getElementById('bot-timeframe')?.value || 'minute';
  const risk    = parseFloat(document.getElementById('bot-risk')?.value) || 2;
  const allowShort = document.getElementById('bot-allow-short')?.checked || false;

  // Warn before switching to live mode
  if (mode === 'live') {
    const confirmed = window.confirm(
      `⚠ WARNING — LIVE TRADING MODE\n\n` +
      `This will place REAL ORDERS with REAL MONEY via Upstox.\n\n` +
      `Strategy : ${strategy}\n` +
      `Symbol   : ${symbol}\n` +
      `Capital  : ₹${capital.toLocaleString('en-IN')}\n\n` +
      `Are you absolutely sure?`
    );
    if (!confirmed) return;
  }

  _bot.running      = true;
  _bot.mode         = mode;
  _bot.tickCount    = 0;
  _bot.realisedPnl  = 0;
  _bot.unrealisedPnl= 0;
  _bot.tradesToday  = 0;
  _bot.dailyLimit   = -(capital * maxLoss / 100);
  _bot.positions    = [];
  _bot.completed    = [];

  // Update UI state
  _botSetStatus(mode);
  document.getElementById('start-btn').disabled = true;
  document.getElementById('stop-btn').disabled  = false;
  document.getElementById('start-btn').textContent =
    mode === 'live' ? '🔴 Running Live' : '▶ Running Paper';

  _botLog('ok', `Bot started — ${mode.toUpperCase()} mode on ${symbol} using ${strategy}`);
  _botLog('info', `Capital: ₹${capital.toLocaleString('en-IN')} | Max daily loss: ₹${Math.abs(_bot.dailyLimit).toLocaleString('en-IN')}`);

  if (mode === 'paper') {
    // Simulate ticks every 5 seconds in paper mode
    _bot.intervalId = setInterval(() => _botTick(symbol, capital), 5000);
    _botLog('info', 'Paper trading simulation active. Ticks every 5 seconds.');
  } else {
    // call backend to start live bot
    try {
      const payload = {
        strategy_name: strategy,
        strategy_params: {},
        symbol,
        timeframe,
        mode,
        capital,
        risk_pct: risk,
        max_daily_loss_pct: maxLoss,
        allow_short: allowShort,
      };
      const res = await fetch('/api/live/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'start failed');
      }
      _botLog('ok', 'Live bot start requested. Polling status...');
      _bot.pollInterval = setInterval(_botPollStatus, 1000);
    } catch (e) {
      _botLog('error', 'Failed to start live bot: ' + e.message);
      // revert UI state
      _bot.running = false;
      _botSetStatus('stopped');
      document.getElementById('stop-btn').disabled = true;
      document.getElementById('start-btn').disabled = false;
      document.getElementById('start-btn').textContent = '▶ Start Paper';
      return;
    }
  }
}


/**
 * Stop the bot — cancels the simulation tick and squares off open positions.
 */
async function stopBot() {
  if (!_bot.running) return;

  if (_bot.mode === 'paper') {
    if (_bot.intervalId) {
      clearInterval(_bot.intervalId);
      _bot.intervalId = null;
    }
  } else {
    if (_bot.pollInterval) {
      clearInterval(_bot.pollInterval);
      _bot.pollInterval = null;
    }
    try {
      await fetch('/api/live/stop', {method: 'POST'});
      _botLog('info', 'Sent stop request to server.');
    } catch (e) {
      _botLog('warn', 'Stop request failed: ' + e.message);
    }
  }

  // Close all open positions at last known price
  if (_bot.positions.length > 0) {
    _botLog('warn', `Stopping: squaring off ${_bot.positions.length} open position(s)…`);
    [..._bot.positions].forEach(pos => _botClosePosition(pos, pos.ltp, 'Bot stopped'));
  }

  _bot.running = false;
  _botSetStatus('stopped');
  document.getElementById('start-btn').disabled = false;
  document.getElementById('stop-btn').disabled  = true;
  document.getElementById('start-btn').textContent = '▶ Start Paper';

  _botLog('info', `Bot stopped. Today's P&L: ₹${_bot.realisedPnl.toFixed(2)}`);
}


/**
 * Square off all open positions immediately.
 */
function squareOff() {
  if (_bot.positions.length === 0) {
    _botLog('warn', 'No open positions to square off.');
    return;
  }
  _botLog('warn', `Squaring off ${_bot.positions.length} position(s)…`);
  [..._bot.positions].forEach(pos => _botClosePosition(pos, pos.ltp, 'Manual square-off'));
}


// ── Paper trading simulation ──────────────────────────────────────────────────

/**
 * Simulated tick function — runs every 5 seconds in paper mode.
 * Randomly generates price moves and occasionally triggers signals.
 *
 * @param {string} symbol  - the instrument being simulated
 * @param {number} capital - starting capital
 */
function _botTick(symbol, capital) {
  _bot.tickCount++;

  // Daily loss limit check
  if (_bot.realisedPnl <= _bot.dailyLimit) {
    _botLog('error', `DAILY LOSS LIMIT HIT (₹${_bot.realisedPnl.toFixed(2)}). Bot stopping.`);
    stopBot();
    return;
  }

  // Simulate LTP move for all open positions
  _bot.positions.forEach(pos => {
    const change  = (Math.random() - 0.48) * pos.entry * 0.005;  // ±0.5%
    pos.ltp       = Math.max(1, pos.ltp + change);
    pos.unrealised = (pos.ltp - pos.entry) * pos.dir * pos.qty;

    // Stop-loss check
    if (pos.dir === 1  && pos.ltp <= pos.stop) {
      _botLog('warn', `SL triggered for LONG ${pos.symbol} @ ₹${pos.ltp.toFixed(2)}`);
      _botClosePosition(pos, pos.ltp, 'Stop-loss');
      return;
    }
    if (pos.dir === -1 && pos.ltp >= pos.stop) {
      _botLog('warn', `SL triggered for SHORT ${pos.symbol} @ ₹${pos.ltp.toFixed(2)}`);
      _botClosePosition(pos, pos.ltp, 'Stop-loss');
    }
  });

  // Remove closed positions
  _bot.positions = _bot.positions.filter(p => !p.closed);

  // Occasionally generate a new signal (every ~6 ticks if no position open)
  if (_bot.positions.length === 0 && _bot.tickCount % 6 === 0) {
    const dir    = Math.random() > 0.5 ? 1 : -1;
    const price  = 1000 + Math.random() * 500;
    const qty    = Math.max(1, Math.floor(capital * 0.02 / price));
    const stop   = dir === 1
      ? price * (1 - 0.015)   // 1.5% SL for longs
      : price * (1 + 0.015);  // 1.5% SL for shorts

    const pos = {
      symbol,
      dir,
      qty,
      entry:     price,
      ltp:       price,
      stop,
      unrealised: 0,
      entryTime:  new Date().toLocaleTimeString('en-IN'),
      closed:    false,
    };
    _bot.positions.push(pos);

    _botLog('ok',
      `Signal ${dir === 1 ? 'BUY' : 'SELL'} ${symbol} — ` +
      `Qty: ${qty} @ ₹${price.toFixed(2)} | SL: ₹${stop.toFixed(2)}`
    );
  }

  // Occasionally close a position on target (every ~10 ticks)
  if (_bot.positions.length > 0 && _bot.tickCount % 10 === 0) {
    const pos = _bot.positions[0];
    _botClosePosition(pos, pos.ltp, 'Target reached');
  }

  // Recompute unrealised
  _bot.unrealisedPnl = _bot.positions.reduce((s, p) => s + p.unrealised, 0);

  // Refresh UI
  _botRefreshUI();
}


/**
 * Close an open position and record it in completed trades.
 *
 * @param {Object} pos       - position object
 * @param {number} exitPrice - price at which to close
 * @param {string} reason    - reason string for the log
 */
function _botClosePosition(pos, exitPrice, reason) {
  const pnl = (exitPrice - pos.entry) * pos.dir * pos.qty;
  pos.closed      = true;
  _bot.realisedPnl += pnl;
  _bot.tradesToday++;
  _bot.completed.push({
    n:       _bot.tradesToday,
    symbol:  pos.symbol,
    dir:     pos.dir === 1 ? 'LONG' : 'SHORT',
    entry:   pos.entry.toFixed(2),
    exit:    exitPrice.toFixed(2),
    pnl:     pnl.toFixed(2),
    reason,
  });

  _botLog(
    pnl >= 0 ? 'ok' : 'warn',
    `CLOSED ${pos.dir === 1 ? 'LONG' : 'SHORT'} ${pos.symbol} — ` +
    `P&L: ₹${pnl.toFixed(2)} (${reason})`
  );
}


// ── Server polling helpers ───────────────────────────────────────────────────

/**
 * Poll the backend /api/live/status endpoint and sync state to _bot.
 */
async function _botPollStatus() {
  try {
    const res = await fetch('/api/live/status');
    if (!res.ok) return;
    const data = await res.json();

    _bot.realisedPnl  = data.day_pnl || 0;
    _bot.unrealisedPnl= data.open_positions
      ? Object.values(data.open_positions).reduce((sum,p)=>sum+(p.unrealised_pnl||0),0)
      : 0;
    _bot.tradesToday  = data.closed_trades ? data.closed_trades.length : 0;

    // convert open_positions dict into _bot.positions array
    _bot.positions = [];
    if (data.open_positions) {
      for (const [sym, p] of Object.entries(data.open_positions)) {
        _bot.positions.push({
          symbol: sym,
          dir:    p.direction === 'LONG' ? 1 : -1,
          qty:    p.quantity,
          entry:  p.entry_price,
          ltp:    p.ltp,
          stop:   p.stop_loss || 0,
          unrealised: p.unrealised_pnl || 0,
          closed: false,
        });
      }
    }

    // closed trades
    _bot.completed = [];
    if (data.closed_trades) {
      let n = 0;
      for (const t of data.closed_trades) {
        n++;
        _bot.completed.push({
          n,
          symbol: t.symbol,
          dir:    t.direction === 'LONG' ? 1 : -1,
          entry:  t.entry_price,
          exit:   t.exit_price,
          pnl:    t.pnl,
          reason: t.exit_reason,
        });
      }
    }

    _botRefreshUI();
  } catch (e) {
    console.error('poll error', e);
  }
}

// ── UI Refresh ────────────────────────────────────────────────────────────────

/** Refresh all live UI panels with current bot state. */
function _botRefreshUI() {
  // ── P&L display ───────────────────────────────────────────────────────────
  const rPnl = document.getElementById('realised-pnl');
  const uPnl = document.getElementById('unrealised-pnl');
  const tTrd = document.getElementById('trades-today');

  if (rPnl) {
    rPnl.textContent  = `₹${_bot.realisedPnl.toFixed(2)}`;
    rPnl.style.color  = _bot.realisedPnl >= 0 ? 'var(--green)' : 'var(--red)';
  }
  if (uPnl) {
    uPnl.textContent  = `₹${_bot.unrealisedPnl.toFixed(2)}`;
    uPnl.style.color  = _bot.unrealisedPnl >= 0 ? 'var(--green)' : 'var(--red)';
  }
  if (tTrd) tTrd.textContent = _bot.tradesToday;

  // ── Open positions table ──────────────────────────────────────────────────
  const tbody = document.getElementById('positions-tbody');
  if (tbody) {
    if (_bot.positions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:20px;">
        No open positions</td></tr>`;
    } else {
      tbody.innerHTML = '';
      _bot.positions.filter(p => !p.closed).forEach(p => {
        const upnl  = p.unrealised || 0;
        const cls   = upnl >= 0 ? 'pos' : 'neg';
        const dirBadge = p.dir === 1
          ? '<span class="badge badge-green">LONG</span>'
          : '<span class="badge badge-red">SHORT</span>';
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong style="font-family:var(--mono);">${p.symbol}</strong></td>
          <td>${dirBadge}</td>
          <td>${p.qty}</td>
          <td>₹${p.entry.toFixed(2)}</td>
          <td style="font-family:var(--mono);">₹${p.ltp.toFixed(2)}</td>
          <td class="${cls}">₹${upnl.toFixed(2)}</td>
          <td style="color:var(--muted);">₹${p.stop.toFixed(2)}</td>
          <td>
            <button class="btn btn-ghost" style="padding:3px 8px;font-size:11px;"
                    onclick="_botClosePosition(
                      window._bot_pos_ref_${_bot.positions.indexOf(p)}, ${p.ltp}, 'Manual')">
              Exit
            </button>
          </td>`;
        tbody.appendChild(tr);

        // Expose position reference for inline button handler
        window[`_bot_pos_ref_${_bot.positions.indexOf(p)}`] = p;
      });
    }
  }

  // ── Completed trades table ────────────────────────────────────────────────
  const ctbody = document.getElementById('completed-tbody');
  if (ctbody) {
    if (_bot.completed.length === 0) {
      ctbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:20px;">
        No trades today</td></tr>`;
    } else {
      ctbody.innerHTML = '';
      // Show most recent first
      [..._bot.completed].reverse().forEach(t => {
        const pnl = parseFloat(t.pnl);
        const cls = pnl >= 0 ? 'pos' : 'neg';
        const tr  = document.createElement('tr');
        tr.innerHTML = `
          <td style="color:var(--muted);">${t.n}</td>
          <td><strong style="font-family:var(--mono);">${t.symbol}</strong></td>
          <td><span class="badge ${t.dir === 'LONG' ? 'badge-green' : 'badge-red'}">${t.dir}</span></td>
          <td>₹${t.entry}</td>
          <td>₹${t.exit}</td>
          <td class="${cls}"><strong>₹${t.pnl}</strong></td>
          <td style="font-size:11px;color:var(--muted);">${t.reason}</td>`;
        ctbody.appendChild(tr);
      });
    }
  }
}


// ── Bot status badge ──────────────────────────────────────────────────────────

/**
 * Update the status badge pill in the topbar.
 *
 * @param {'stopped'|'paper'|'live'} status
 */
function _botSetStatus(status) {
  const badge = document.getElementById('bot-status-badge');
  const label = document.getElementById('bot-status-label');
  if (!badge || !label) return;

  badge.className = `bot-status ${status}`;

  const labels = {
    stopped: 'Stopped',
    paper:   'Paper Trading',
    live:    '🔴 LIVE',
  };
  label.textContent = labels[status] || status;
}


// ── Activity log ──────────────────────────────────────────────────────────────

/**
 * Append a timestamped line to the activity log panel.
 *
 * @param {'info'|'ok'|'warn'|'error'} level
 * @param {string}                     msg
 */
function _botLog(level, msg) {
  const log = document.getElementById('bot-log');
  if (!log) return;

  const now    = new Date().toLocaleTimeString('en-IN', { hour12: false });
  const line   = document.createElement('div');
  line.className = `log-line ${level}`;
  line.textContent = `[${now}] ${msg}`;
  log.appendChild(line);

  // Auto-scroll to bottom
  log.scrollTop = log.scrollHeight;

  // Cap log at 200 lines to avoid memory growth
  while (log.children.length > 200) {
    log.removeChild(log.firstChild);
  }
}
