"""
Backtest: $200 account, 1 week, H1 primary + M15 confirm.

3-TP partial exit system (matches Pine Script ZP PRO exactly):
  - TP1 = 2.0x ATR  → close 1/3 of position
  - TP2 = 3.5x ATR  → close 1/3 of position, move SL to breakeven
  - TP3 = 5.0x ATR   → close final 1/3
  - SL = Smart Structure SL (swing low/high + min 1.5x ATR floor)
  - ZP flip exit on confirmed bar (closes remainder)
  - lastSignalDir filter (no duplicate direction signals)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.zeropoint_signal import (
    compute_zeropoint_state, TP1_MULT, TP2_MULT, TP3_MULT,
    SL_BUFFER_PCT, SL_ATR_MIN_MULT, SWING_LOOKBACK, ATR_MULTIPLIER,
)

mt5.initialize()

# ─── Config ──────────────────────────────────────────────────────────────────
STARTING_BALANCE = 200.0
FIXED_LOT = 0.04
SYMBOLS = ['AUDUSD', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDJPY']
FRESHNESS_BARS = 3
WEEK_H1_BARS = 24 * 5  # 120 bars = 1 trading week
WARMUP_BARS = 200
MAX_OPEN = 3             # max simultaneous positions

NUM_WEEKS = 1

# ─── Fetch data ──────────────────────────────────────────────────────────────
print("Fetching data...")
total_h1_needed = WARMUP_BARS + WEEK_H1_BARS * NUM_WEEKS + 50
total_m15_needed = total_h1_needed * 4 + 50

all_data = {}
for sym in SYMBOLS:
    info = mt5.symbol_info(sym)
    if info is None:
        continue
    mt5.symbol_select(sym, True)

    rates_h1 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, total_h1_needed)
    rates_m15 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, total_m15_needed)

    if rates_h1 is None or len(rates_h1) < WARMUP_BARS + WEEK_H1_BARS:
        continue

    df_h1 = pd.DataFrame(rates_h1)
    df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')

    df_m15 = None
    if rates_m15 is not None and len(rates_m15) >= 200:
        df_m15 = pd.DataFrame(rates_m15)
        df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')

    point = info.point
    pip_size = 0.01 if 'JPY' in sym else 0.0001
    contract_size = info.trade_contract_size

    all_data[sym] = {
        'df_h1': df_h1, 'df_m15': df_m15,
        'point': point, 'pip_size': pip_size, 'contract_size': contract_size,
    }
    print(f"  {sym}: {len(df_h1)} H1 bars, {len(df_m15) if df_m15 is not None else 0} M15 bars")

mt5.shutdown()

if not all_data:
    print("No data!")
    sys.exit(1)

# ─── Position class with 3-TP partial exits ──────────────────────────────────
class Position:
    def __init__(self, sym, direction, entry, sl, tp1, tp2, tp3, lot, entry_time, conf, m15, fresh, rr, pip_size, contract_size):
        self.sym = sym
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.original_sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.tp3 = tp3
        self.total_lot = lot
        self.remaining_lot = lot
        self.entry_time = entry_time
        self.conf = conf
        self.m15 = m15
        self.fresh = fresh
        self.rr = rr
        self.pip_size = pip_size
        self.contract_size = contract_size
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        self.sl_hit = False
        self.closed = False
        self.partials = []  # list of (exit_price, lot_closed, pnl_dollar, exit_type)
        self.max_profit_reached = 0.0

    def partial_lot(self):
        """1/3 of total lot, rounded to 0.01"""
        p = round(self.total_lot / 3, 2)
        return max(0.01, p)

    def pnl_for_price(self, price, lot):
        if self.direction == 'BUY':
            return (price - self.entry) * lot * self.contract_size
        else:
            return (self.entry - price) * lot * self.contract_size

    def pips_for_price(self, price):
        if self.direction == 'BUY':
            return (price - self.entry) / self.pip_size
        else:
            return (self.entry - price) / self.pip_size

    def check_bar(self, high, low, close, confirmed_pos):
        """Check TP/SL hits and ZP flip. Matches Pine Script logic EXACTLY.

        Pine Script order:
        1. Track max profit (high - entry for BUY, entry - low for SELL)
        2. Check TPs FIRST (TP1 → TP2 → TP3)
        3. Check SL ONLY if: not tp1Hit AND maxProfitReached <= 0
        4. On signal switch (ZP flip): close at market, count as WIN if tp1Hit or profit > 0
        """
        if self.closed:
            return []

        events = []

        # ─── Track max profit (Pine: maxProfitReached) ────────────────────
        if self.direction == 'BUY':
            cur_profit = high - self.entry
        else:
            cur_profit = self.entry - low
        if cur_profit > self.max_profit_reached:
            self.max_profit_reached = cur_profit

        # ─── Check TPs FIRST (Pine Script checks TPs before SL) ──────────

        # TP1
        if not self.tp1_hit:
            tp1_triggered = False
            if self.direction == 'BUY' and high >= self.tp1:
                tp1_triggered = True
            elif self.direction == 'SELL' and low <= self.tp1:
                tp1_triggered = True

            if tp1_triggered:
                self.tp1_hit = True
                partial = self.partial_lot()
                partial = min(partial, self.remaining_lot)
                pnl = self.pnl_for_price(self.tp1, partial)
                pips = self.pips_for_price(self.tp1)
                self.partials.append((self.tp1, partial, pnl, 'TP1'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP1', self.tp1, pnl, pips, partial))

                if self.remaining_lot <= 0:
                    self.closed = True
                    return events

        # TP2
        if self.tp1_hit and not self.tp2_hit:
            tp2_triggered = False
            if self.direction == 'BUY' and high >= self.tp2:
                tp2_triggered = True
            elif self.direction == 'SELL' and low <= self.tp2:
                tp2_triggered = True

            if tp2_triggered:
                self.tp2_hit = True
                # Move SL to breakeven (Pine Script exact)
                self.sl = self.entry
                partial = self.partial_lot()
                partial = min(partial, self.remaining_lot)
                pnl = self.pnl_for_price(self.tp2, partial)
                pips = self.pips_for_price(self.tp2)
                self.partials.append((self.tp2, partial, pnl, 'TP2'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP2', self.tp2, pnl, pips, partial))

                if self.remaining_lot <= 0:
                    self.closed = True
                    return events

        # TP3
        if self.tp2_hit and not self.tp3_hit:
            tp3_triggered = False
            if self.direction == 'BUY' and high >= self.tp3:
                tp3_triggered = True
            elif self.direction == 'SELL' and low <= self.tp3:
                tp3_triggered = True

            if tp3_triggered:
                self.tp3_hit = True
                pnl = self.pnl_for_price(self.tp3, self.remaining_lot)
                pips = self.pips_for_price(self.tp3)
                self.partials.append((self.tp3, self.remaining_lot, pnl, 'TP3'))
                events.append(('TP3', self.tp3, pnl, pips, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                return events

        # ─── Check SL (Pine: only if not tp1Hit AND maxProfitReached <= 0) ─
        # Pine Script exact:
        #   if low <= slLevel and not tp1Hit and maxProfitReached <= 0
        #       slHit := true, tradeClosed := true, losses += 1
        sl_touched = False
        if self.direction == 'BUY' and low <= self.sl:
            sl_touched = True
        elif self.direction == 'SELL' and high >= self.sl:
            sl_touched = True

        if sl_touched:
            if not self.tp1_hit and self.max_profit_reached <= 0:
                # Pure loss — no TP hit, no profit ever reached
                pnl = self.pnl_for_price(self.sl, self.remaining_lot)
                pips = self.pips_for_price(self.sl)
                self.partials.append((self.sl, self.remaining_lot, pnl, 'SL'))
                self.remaining_lot = 0
                self.closed = True
                self.sl_hit = True
                events.append(('SL', self.sl, pnl, pips, 0))
                return events
            elif self.tp2_hit:
                # SL at breakeven after TP2 — close remainder at entry (zero P/L)
                pnl = self.pnl_for_price(self.sl, self.remaining_lot)
                pips = self.pips_for_price(self.sl)
                self.partials.append((self.sl, self.remaining_lot, pnl, 'SL_BE'))
                self.remaining_lot = 0
                self.closed = True
                events.append(('SL_BE', self.sl, pnl, pips, 0))
                return events
            elif self.tp1_hit:
                # SL hit after TP1 but before TP2 — close remainder at SL
                # (This is still counted as a WIN in Pine Script because tp1Hit=true)
                pnl = self.pnl_for_price(self.sl, self.remaining_lot)
                pips = self.pips_for_price(self.sl)
                self.partials.append((self.sl, self.remaining_lot, pnl, 'SL_AFTER_TP1'))
                self.remaining_lot = 0
                self.closed = True
                events.append(('SL_AFTER_TP1', self.sl, pnl, pips, 0))
                return events

        # ─── ZP flip exit (confirmed bar — Pine: signal switch) ──────────
        if confirmed_pos != 0:
            trade_is_buy = self.direction == 'BUY'
            zp_is_buy = confirmed_pos == 1
            if trade_is_buy != zp_is_buy and self.remaining_lot > 0:
                pnl = self.pnl_for_price(close, self.remaining_lot)
                pips = self.pips_for_price(close)
                self.partials.append((close, self.remaining_lot, pnl, 'ZP_FLIP'))
                events.append(('ZP_FLIP', close, pnl, pips, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True

        return events

    def force_close(self, price, exit_type='WEEK_END'):
        """Close remaining position at given price."""
        if self.closed or self.remaining_lot <= 0:
            return 0
        pnl = self.pnl_for_price(price, self.remaining_lot)
        pips = self.pips_for_price(price)
        self.partials.append((price, self.remaining_lot, pnl, exit_type))
        self.remaining_lot = 0
        self.closed = True
        return pnl

    @property
    def total_pnl(self):
        return sum(p[2] for p in self.partials)

    @property
    def total_pips(self):
        return sum(self.pips_for_price(p[0]) * (p[1] / self.total_lot) for p in self.partials) if self.partials else 0


# ─── Run backtest ─────────────────────────────────────────────────────────────
print()
print("=" * 115)
print(f"BACKTEST: ${STARTING_BALANCE:.0f} | {NUM_WEEKS} week | H1+M15 | {FIXED_LOT} lots | 3-TP partials (TP1={TP1_MULT}x, TP2={TP2_MULT}x, TP3={TP3_MULT}x ATR)")
print(f"lastSignalDir filter + Smart Structure SL + ZP flip exit on confirmed bar")
print("=" * 115)
print()

balance = STARTING_BALANCE
all_positions = []  # all positions (open + closed)
open_positions = []

for sym, data in all_data.items():
    df_h1 = data['df_h1']
    df_m15 = data['df_m15']
    pip_size = data['pip_size']
    contract_size = data['contract_size']

    n = len(df_h1)
    test_start = n - WEEK_H1_BARS * NUM_WEEKS
    if test_start < WARMUP_BARS:
        test_start = WARMUP_BARS

    for bar_idx in range(test_start, n):
        window_h1 = df_h1.iloc[:bar_idx + 1].copy()
        zp_h1 = compute_zeropoint_state(window_h1)
        if zp_h1 is None or len(zp_h1) < 3:
            continue

        cur_bar = zp_h1.iloc[-1]
        confirmed_bar = zp_h1.iloc[-2]
        cur_time = df_h1.iloc[bar_idx]['time']
        cur_close = float(cur_bar['close'])
        cur_high = float(cur_bar['high'])
        cur_low = float(cur_bar['low'])
        confirmed_pos = int(confirmed_bar.get('pos', 0))

        # ─── Check exits ──────────────────────────────────────────────
        for pos in open_positions:
            if pos.sym != sym or pos.closed:
                continue
            events = pos.check_bar(cur_high, cur_low, cur_close, confirmed_pos)
            for ev in events:
                balance += ev[2]  # pnl_dollar

        # Remove closed
        open_positions = [p for p in open_positions if not p.closed]

        # ─── Check for new entries ────────────────────────────────────
        sym_in_pos = any(p.sym == sym for p in open_positions)
        if sym_in_pos:
            continue
        if len(open_positions) >= MAX_OPEN:
            continue

        pos_val = int(cur_bar.get('pos', 0))
        if pos_val == 0:
            continue

        direction = 'BUY' if pos_val == 1 else 'SELL'
        atr_val = float(cur_bar.get('atr', 0))
        if atr_val <= 0 or np.isnan(atr_val):
            continue

        trailing_stop = float(cur_bar.get('xATRTrailingStop', 0))
        if trailing_stop <= 0:
            continue

        bars_since = int(cur_bar.get('bars_since_flip', 999))
        is_fresh = bars_since <= FRESHNESS_BARS

        # Smart Structure SL
        sl = None
        if 'smart_sl' in zp_h1.columns:
            for sl_idx in range(len(zp_h1) - 1, -1, -1):
                sl_val = zp_h1.iloc[sl_idx].get('smart_sl', float('nan'))
                if not np.isnan(sl_val) and sl_val > 0:
                    sl = float(sl_val)
                    break
        if sl is None:
            sl = trailing_stop
            buffer = atr_val * SL_BUFFER_PCT
            sl = sl - buffer if direction == 'BUY' else sl + buffer

        if not is_fresh and trailing_stop > 0:
            buffer = atr_val * SL_BUFFER_PCT
            if direction == 'BUY':
                ts_sl = trailing_stop - buffer
                if ts_sl > sl:
                    sl = ts_sl
            else:
                ts_sl = trailing_stop + buffer
                if ts_sl < sl:
                    sl = ts_sl

        entry = cur_close

        # 3 TP levels (exact Pine Script)
        if direction == 'BUY':
            tp1 = entry + atr_val * TP1_MULT
            tp2 = entry + atr_val * TP2_MULT
            tp3 = entry + atr_val * TP3_MULT
        else:
            tp1 = entry - atr_val * TP1_MULT
            tp2 = entry - atr_val * TP2_MULT
            tp3 = entry - atr_val * TP3_MULT

        if direction == 'BUY' and entry >= tp1:
            continue
        if direction == 'SELL' and entry <= tp1:
            continue

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp1 - entry)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        # M15 confirmation
        m15_conf = False
        if data['df_m15'] is not None:
            m15_mask = data['df_m15']['time'] <= cur_time
            m15_window = data['df_m15'][m15_mask].tail(200)
            if len(m15_window) >= 20:
                zp_m15 = compute_zeropoint_state(m15_window)
                if zp_m15 is not None and len(zp_m15) > 0:
                    m15_pos = int(zp_m15.iloc[-1].get('pos', 0))
                    if direction == 'BUY' and m15_pos == 1:
                        m15_conf = True
                    elif direction == 'SELL' and m15_pos == -1:
                        m15_conf = True

        conf = 0.65
        if is_fresh: conf += 0.15
        if m15_conf: conf += 0.10
        if rr >= 2.0: conf += 0.05
        elif rr >= 1.5: conf += 0.03
        if not is_fresh:
            conf -= min(bars_since * 0.03, 0.20)
        conf = max(0.40, min(conf, 0.98))

        # Margin check
        margin_needed = FIXED_LOT * contract_size * entry / 100
        if margin_needed > balance * 0.5:
            continue

        new_pos = Position(
            sym=sym, direction=direction, entry=entry, sl=sl,
            tp1=tp1, tp2=tp2, tp3=tp3, lot=FIXED_LOT,
            entry_time=cur_time, conf=conf, m15=m15_conf,
            fresh=is_fresh, rr=rr, pip_size=pip_size, contract_size=contract_size,
        )
        open_positions.append(new_pos)
        all_positions.append(new_pos)

# Close remaining at week end
for pos in open_positions:
    if not pos.closed:
        last_close = float(all_data[pos.sym]['df_h1'].iloc[-1]['close'])
        pnl = pos.force_close(last_close, 'WEEK_END')
        balance += pnl

# ─── Results ──────────────────────────────────────────────────────────────────
print()
if not all_positions:
    print("No trades taken!")
    sys.exit(0)

# Collect all partial exits into trade log
print(f"{'Time':<16} {'Sym':<8} {'Dir':<5} {'Entry':>9} {'Exit':>9} {'Type':>7} {'Lot':>5} {'P/L $':>8} {'Pips':>7} {'M15':>4}")
print("-" * 105)

all_exits = []
for pos in sorted(all_positions, key=lambda p: p.entry_time):
    time_str = pos.entry_time.strftime('%m/%d %H:%M') if hasattr(pos.entry_time, 'strftime') else str(pos.entry_time)[:14]
    for exit_price, lot_closed, pnl, exit_type in pos.partials:
        pips = pos.pips_for_price(exit_price)
        m15_tag = "Y" if pos.m15 else ""
        print(f"{time_str:<16} {pos.sym:<8} {pos.direction:<5} {pos.entry:>9.5f} {exit_price:>9.5f} {exit_type:>7} {lot_closed:>5.2f} {pnl:>+8.2f} {pips:>+7.1f} {m15_tag:>4}")
        all_exits.append({
            'sym': pos.sym, 'dir': pos.direction, 'pnl': pnl,
            'pips': pips, 'type': exit_type, 'lot': lot_closed,
            'entry_time': pos.entry_time,
        })

print("-" * 105)
print()

# ─── Summary stats (Pine Script WIN/LOSS counting) ──────────────────────────
total_pnl = sum(e['pnl'] for e in all_exits)
n_trades = len(all_positions)
n_exits = len(all_exits)

# Pine Script win rate counting (EXACT):
#   WIN = tp1Hit OR maxProfitReached > 0 OR finalPnL > 0
#   LOSS = everything else (SL hit with no TP, no profit ever reached)
# This is how ZeroPoint PRO gets 80-90% — once TP1 hits at 2x ATR, it's a WIN
# even if the remainder of the position gets stopped out.
pine_wins = 0
pine_losses = 0
for pos in all_positions:
    if pos.tp1_hit or pos.max_profit_reached > 0 or pos.total_pnl > 0:
        pine_wins += 1
    else:
        pine_losses += 1

# Also track raw P/L-based wins for comparison
raw_pos_wins = sum(1 for p in all_positions if p.total_pnl > 0)
raw_pos_losses = n_trades - raw_pos_wins

# Per-exit wins (each partial close)
exit_wins = sum(1 for e in all_exits if e['pnl'] > 0)
exit_losses = sum(1 for e in all_exits if e['pnl'] <= 0)
exit_wr = exit_wins / len(all_exits) * 100 if all_exits else 0

pine_wr = pine_wins / n_trades * 100 if n_trades > 0 else 0
raw_wr = raw_pos_wins / n_trades * 100 if n_trades > 0 else 0
avg_win = np.mean([pos.total_pnl for pos in all_positions if pos.total_pnl > 0]) if raw_pos_wins > 0 else 0
avg_loss = np.mean([pos.total_pnl for pos in all_positions if pos.total_pnl <= 0]) if raw_pos_losses > 0 else 0
gross_win = sum(pos.total_pnl for pos in all_positions if pos.total_pnl > 0)
gross_loss = abs(sum(pos.total_pnl for pos in all_positions if pos.total_pnl <= 0))
pf = gross_win / gross_loss if gross_loss > 0 else 999

# Exit type counts
tp1_count = sum(1 for e in all_exits if e['type'] == 'TP1')
tp2_count = sum(1 for e in all_exits if e['type'] == 'TP2')
tp3_count = sum(1 for e in all_exits if e['type'] == 'TP3')
sl_count = sum(1 for e in all_exits if e['type'] == 'SL')  # pure SL (loss)
flip_count = sum(1 for e in all_exits if e['type'] == 'ZP_FLIP')
end_count = sum(1 for e in all_exits if e['type'] == 'WEEK_END')

# TP hit rates per position
tp1_positions = sum(1 for p in all_positions if p.tp1_hit)
tp2_positions = sum(1 for p in all_positions if p.tp2_hit)
tp3_positions = sum(1 for p in all_positions if p.tp3_hit)
sl_positions = sum(1 for p in all_positions if p.sl_hit)

print(f"{'=' * 50} ACCOUNT {'=' * 50}")
print(f"  Starting Balance:  ${STARTING_BALANCE:.2f}")
print(f"  Ending Balance:    ${balance:.2f}")
print(f"  Net P/L:           ${total_pnl:+.2f}  ({total_pnl/STARTING_BALANCE*100:+.1f}%)")
print()
print(f"  Positions Opened:  {n_trades}")
print(f"  Partial Exits:     {n_exits}")
print(f"  PINE WR (TV):      {pine_wr:.1f}%  ({pine_wins}W / {pine_losses}L)  <- TradingView method")
print(f"  Raw P/L WR:        {raw_wr:.1f}%  ({raw_pos_wins}W / {raw_pos_losses}L)  <- strict P/L")
print(f"  Per-Exit WR:       {exit_wr:.1f}%  ({exit_wins}W / {exit_losses}L)")
print(f"  Profit Factor:     {pf:.2f}")
print(f"  Avg Win (pos):     ${avg_win:+.2f}")
print(f"  Avg Loss (pos):    ${avg_loss:+.2f}")
print()
print(f"  TP Hit Rates:      TP1={tp1_positions}/{n_trades} ({tp1_positions/n_trades*100:.0f}%)  "
      f"TP2={tp2_positions}/{n_trades} ({tp2_positions/n_trades*100:.0f}%)  "
      f"TP3={tp3_positions}/{n_trades} ({tp3_positions/n_trades*100:.0f}%)")
print(f"  SL Hit:            {sl_positions}/{n_trades} ({sl_positions/n_trades*100:.0f}%)")

# Exit type breakdown
sl_be_count = sum(1 for e in all_exits if e['type'] == 'SL_BE')
sl_after_tp1_count = sum(1 for e in all_exits if e['type'] == 'SL_AFTER_TP1')
pure_sl_count = sum(1 for e in all_exits if e['type'] == 'SL')
print(f"  Exits:  TP1={tp1_count}  TP2={tp2_count}  TP3={tp3_count}  SL={pure_sl_count}  SL_BE={sl_be_count}  SL_AFT_TP1={sl_after_tp1_count}  ZP_FLIP={flip_count}  WEEK_END={end_count}")
print()

# Per-symbol breakdown
print(f"{'=' * 45} PER-SYMBOL {'=' * 45}")
print(f"{'Symbol':<10} {'Pos':>4} {'WR':>6} {'TP1':>4} {'TP2':>4} {'TP3':>4} {'SL':>4} {'P/L $':>10}")
print("-" * 60)

by_sym = {}
for pos in all_positions:
    s = pos.sym
    if s not in by_sym:
        by_sym[s] = {'positions': [], 'pnl': 0}
    by_sym[s]['positions'].append(pos)
    by_sym[s]['pnl'] += pos.total_pnl

for sym in sorted(by_sym.keys(), key=lambda s: by_sym[s]['pnl'], reverse=True):
    info = by_sym[sym]
    positions = info['positions']
    n_pos = len(positions)
    # Pine WR: WIN if tp1Hit or maxProfitReached > 0 or total_pnl > 0
    n_win = sum(1 for p in positions if p.tp1_hit or p.max_profit_reached > 0 or p.total_pnl > 0)
    swr = n_win / n_pos * 100 if n_pos > 0 else 0
    stp1 = sum(1 for p in positions if p.tp1_hit)
    stp2 = sum(1 for p in positions if p.tp2_hit)
    stp3 = sum(1 for p in positions if p.tp3_hit)
    ssl = sum(1 for p in positions if p.sl_hit)
    print(f"{sym:<10} {n_pos:>4} {swr:>5.0f}% {stp1:>4} {stp2:>4} {stp3:>4} {ssl:>4} {info['pnl']:>+10.2f}")

print("-" * 60)
print(f"{'TOTAL':<10} {n_trades:>4} {pine_wr:>5.1f}% {tp1_positions:>4} {tp2_positions:>4} {tp3_positions:>4} {sl_positions:>4} {total_pnl:>+10.2f}")
print()

# Equity curve
print(f"{'=' * 45} EQUITY CURVE {'=' * 45}")
running = STARTING_BALANCE
sorted_exits = sorted(all_exits, key=lambda x: x['entry_time'])
for i, e in enumerate(sorted_exits):
    running += e['pnl']
    pct = (running / STARTING_BALANCE - 1) * 100
    bar_len = max(1, int((running / STARTING_BALANCE) * 20))
    bar = "#" * bar_len
    print(f"  Exit {i+1:>2}: ${running:>8.2f} ({pct:>+5.1f}%)  {e['sym']:<8} {e['type']:<7} {bar}")
