"""
Council Predictor Trainer — Trains a learned model to predict ZP trade outcomes.

Pipeline:
  1. Fetch max MT5 history (5000+ H1 bars ≈ 208 days) for all symbols
  2. Walk through H1 bars detecting ZP signals (same logic as backtest_council.py)
  3. For each signal: extract features + simulate trade forward to get outcome label
  4. Train CouncilPredictor binary classifier: P(profitable)
  5. Evaluate with Bayesian inference on temporal validation split
  6. Save checkpoint to council_predictor_model.pth
"""

import sys
import os

# Add project root to sys.path so subpackage imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from collections import Counter

from app.zeropoint_signal import (
    compute_zeropoint_state, TP1_MULT, TP2_MULT, TP3_MULT,
    SL_BUFFER_PCT, SL_ATR_MIN_MULT, SWING_LOOKBACK, ATR_MULTIPLIER,
    BE_TRIGGER_MULT, BE_BUFFER_MULT,
    PROFIT_TRAIL_DISTANCE_MULT,
    TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,
    STALL_BARS, MICRO_TP_MULT, MICRO_TP_PCT,
)
from candle_intelligence.council_predictor import (
    CouncilPredictor, extract_features, FEATURE_DIM, DEFAULT_SYMBOLS,
    NUM_SYMBOLS, NUM_NUMERIC_FEATURES,
)

# ─── Config ──────────────────────────────────────────────────────────────────
SYMBOLS = ['AUDUSD', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDJPY']
FRESHNESS_BARS = 3
WARMUP_BARS = 200
CHECKPOINT_PATH = 'council_predictor_model.pth'

# Training hyperparams
MAX_EPOCHS = 200
PATIENCE = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.05
TRAIN_SPLIT = 0.80  # temporal: oldest 80% train, newest 20% val
N_BAYESIAN_SAMPLES = 20  # forward passes at eval


# ─── Position class (V4 PROTECT mode — matches live system) ──────────────────
class Position:
    """V4 Profit Capture: early BE, micro-partial, post-TP1 trail, stall exit."""

    def __init__(self, sym, direction, entry, sl, tp1, tp2, tp3, lot,
                 entry_bar_idx, pip_size, contract_size, atr_value=0.0):
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
        self.entry_bar_idx = entry_bar_idx
        self.pip_size = pip_size
        self.contract_size = contract_size
        self.atr_value = atr_value
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        self.sl_hit = False
        self.closed = False
        self.partials = []
        self.max_profit_reached = 0.0
        self.max_favorable_price = entry
        self.be_activated = False
        self.profit_lock_sl = None
        self.profit_lock_active = False
        self.micro_tp_hit = False
        self.stall_be_activated = False
        self.bars_in_trade = 0

    def partial_lot(self):
        return max(0.01, round(self.total_lot / 3, 2))

    def pnl_for_price(self, price, lot):
        if self.direction == 'BUY':
            return (price - self.entry) * lot * self.contract_size
        else:
            return (self.entry - price) * lot * self.contract_size

    def check_bar(self, high, low, close, confirmed_pos):
        if self.closed:
            return []
        events = []
        self.bars_in_trade += 1
        is_buy = self.direction == 'BUY'
        atr = self.atr_value if self.atr_value > 1e-12 else 1.0

        # Track max favorable excursion
        if is_buy:
            cur_profit = high - self.entry
            if high > self.max_favorable_price:
                self.max_favorable_price = high
        else:
            cur_profit = self.entry - low
            if low < self.max_favorable_price:
                self.max_favorable_price = low
        if cur_profit > self.max_profit_reached:
            self.max_profit_reached = cur_profit

        # V4: Post-TP1 trailing stop
        if self.tp1_hit:
            trail_dist = PROFIT_TRAIL_DISTANCE_MULT * atr
            if is_buy:
                new_lock = self.max_favorable_price - trail_dist
                if new_lock > self.entry and (self.profit_lock_sl is None or new_lock > self.profit_lock_sl):
                    self.profit_lock_sl = new_lock
                    self.profit_lock_active = True
            else:
                new_lock = self.max_favorable_price + trail_dist
                if new_lock < self.entry and (self.profit_lock_sl is None or new_lock < self.profit_lock_sl):
                    self.profit_lock_sl = new_lock
                    self.profit_lock_active = True

        # V4: Early breakeven at BE_TRIGGER_MULT * ATR
        if not self.be_activated:
            if self.max_profit_reached >= BE_TRIGGER_MULT * atr:
                be_buffer = BE_BUFFER_MULT * atr
                if is_buy:
                    new_sl = self.entry + be_buffer
                    if new_sl > self.sl:
                        self.sl = new_sl
                        self.be_activated = True
                else:
                    new_sl = self.entry - be_buffer
                    if new_sl < self.sl:
                        self.sl = new_sl
                        self.be_activated = True

        # V4: Time-based stall protection
        if not self.tp1_hit and not self.stall_be_activated:
            if self.bars_in_trade >= STALL_BARS:
                be_buffer = BE_BUFFER_MULT * atr
                if is_buy:
                    new_sl = self.entry + be_buffer
                    if new_sl > self.sl:
                        self.sl = new_sl
                        self.stall_be_activated = True
                        self.be_activated = True
                else:
                    new_sl = self.entry - be_buffer
                    if new_sl < self.sl:
                        self.sl = new_sl
                        self.stall_be_activated = True
                        self.be_activated = True

        # V4: Micro-partial at MICRO_TP_MULT * ATR
        if not self.micro_tp_hit and not self.tp1_hit:
            micro_price = (self.entry + MICRO_TP_MULT * atr) if is_buy else (self.entry - MICRO_TP_MULT * atr)
            micro_triggered = (is_buy and high >= micro_price) or (not is_buy and low <= micro_price)
            if micro_triggered:
                self.micro_tp_hit = True
                micro_lot = round(self.total_lot * MICRO_TP_PCT, 2)
                micro_lot = max(0.01, min(micro_lot, self.remaining_lot))
                pnl = self.pnl_for_price(micro_price, micro_lot)
                self.partials.append((micro_price, micro_lot, pnl, 'MICRO_TP'))
                self.remaining_lot = round(self.remaining_lot - micro_lot, 2)
                events.append(('MICRO_TP', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    return events

        # TP1
        if not self.tp1_hit:
            tp1_triggered = (is_buy and high >= self.tp1) or \
                            (not is_buy and low <= self.tp1)
            if tp1_triggered:
                self.tp1_hit = True
                partial = min(self.partial_lot(), self.remaining_lot)
                pnl = self.pnl_for_price(self.tp1, partial)
                self.partials.append((self.tp1, partial, pnl, 'TP1'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP1', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    return events

        # TP2
        if self.tp1_hit and not self.tp2_hit:
            tp2_triggered = (is_buy and high >= self.tp2) or \
                            (not is_buy and low <= self.tp2)
            if tp2_triggered:
                self.tp2_hit = True
                self.sl = self.entry  # breakeven (redundant with V4 BE but consistent)
                self.be_activated = True
                partial = min(self.partial_lot(), self.remaining_lot)
                pnl = self.pnl_for_price(self.tp2, partial)
                self.partials.append((self.tp2, partial, pnl, 'TP2'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP2', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    return events

        # TP3
        if self.tp2_hit and not self.tp3_hit:
            tp3_triggered = (is_buy and high >= self.tp3) or \
                            (not is_buy and low <= self.tp3)
            if tp3_triggered:
                self.tp3_hit = True
                pnl = self.pnl_for_price(self.tp3, self.remaining_lot)
                self.partials.append((self.tp3, self.remaining_lot, pnl, 'TP3'))
                events.append(('TP3', pnl))
                self.remaining_lot = 0
                self.closed = True
                return events

        # V4: Post-TP1 profit lock SL hit
        if self.profit_lock_active and self.profit_lock_sl is not None:
            lock_hit = (is_buy and low <= self.profit_lock_sl) or \
                       (not is_buy and high >= self.profit_lock_sl)
            if lock_hit:
                pnl = self.pnl_for_price(self.profit_lock_sl, self.remaining_lot)
                self.partials.append((self.profit_lock_sl, self.remaining_lot, pnl, 'PROFIT_LOCK'))
                self.remaining_lot = 0
                self.closed = True
                events.append(('PROFIT_LOCK', pnl))
                return events

        # SL
        sl_touched = (is_buy and low <= self.sl) or \
                     (not is_buy and high >= self.sl)
        if sl_touched:
            if self.stall_be_activated:
                exit_label = 'SL_STALL'
            elif self.be_activated:
                exit_label = 'SL_BE'
            elif self.tp1_hit:
                exit_label = 'SL_AFTER_TP'
            else:
                exit_label = 'SL'
                self.sl_hit = True
            pnl = self.pnl_for_price(self.sl, self.remaining_lot)
            self.partials.append((self.sl, self.remaining_lot, pnl, exit_label))
            self.remaining_lot = 0
            self.closed = True
            events.append((exit_label, pnl))
            return events

        # ZP flip exit (Pine Script: close on signal switch)
        if confirmed_pos != 0:
            trade_is_buy = is_buy
            zp_is_buy = confirmed_pos == 1
            if trade_is_buy != zp_is_buy:
                pnl = self.pnl_for_price(close, self.remaining_lot)
                self.partials.append((close, self.remaining_lot, pnl, 'ZP_FLIP'))
                self.remaining_lot = 0
                self.closed = True
                events.append(('ZP_FLIP', pnl))
                return events

        return events

    def force_close(self, price):
        if self.closed or self.remaining_lot <= 0:
            return 0
        pnl = self.pnl_for_price(price, self.remaining_lot)
        self.partials.append((price, self.remaining_lot, pnl, 'END'))
        self.remaining_lot = 0
        self.closed = True
        return pnl

    @property
    def total_pnl(self):
        return sum(p[2] for p in self.partials)


# ─── Data Generation ─────────────────────────────────────────────────────────
# Match live system: ZP flip closes existing trade before new signal opens.
# One position per symbol max (ZP flip exit ensures this).
MAX_OPEN = 5  # Match live system concurrent position limit (scaled from 3)

def generate_training_data():
    """Fetch MT5 data and generate labeled feature vectors from ZP signals.

    IMPORTANT: This now walks ALL symbols bar-by-bar in chronological order
    (not symbol-by-symbol) to match the live system's behavior:
      - Only enter if symbol not already in a position
      - Only enter if fewer than MAX_OPEN total positions
      - Track open positions and close them bar-by-bar
    This produces signals that match what the live system would actually trade.
    """

    mt5.initialize()

    symbol_to_index = {s: i for i, s in enumerate(DEFAULT_SYMBOLS)}

    # Fetch max history
    print("Fetching maximum history from MT5...")
    all_data = {}
    for sym in SYMBOLS:
        info = mt5.symbol_info(sym)
        if info is None:
            continue
        mt5.symbol_select(sym, True)

        rates_h1 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 5000)
        rates_m15 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 20000)
        rates_h4 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 2000)

        if rates_h1 is None or len(rates_h1) < WARMUP_BARS + 100:
            print(f"  {sym}: insufficient data, skipping")
            continue

        df_h1 = pd.DataFrame(rates_h1)
        df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')

        df_m15 = None
        if rates_m15 is not None and len(rates_m15) >= 200:
            df_m15 = pd.DataFrame(rates_m15)
            df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')

        df_h4 = None
        if rates_h4 is not None and len(rates_h4) >= 20:
            df_h4 = pd.DataFrame(rates_h4)
            df_h4['time'] = pd.to_datetime(df_h4['time'], unit='s')

        point = info.point
        pip_size = 0.01 if 'JPY' in sym else 0.0001
        contract_size = info.trade_contract_size

        all_data[sym] = {
            'df_h1': df_h1, 'df_m15': df_m15, 'df_h4': df_h4,
            'point': point, 'pip_size': pip_size,
            'contract_size': contract_size,
        }
        h4_count = len(df_h4) if df_h4 is not None else 0
        print(f"  {sym}: {len(df_h1)} H1 bars, "
              f"{len(df_m15) if df_m15 is not None else 0} M15, {h4_count} H4")

    mt5.shutdown()

    if not all_data:
        print("No data available!")
        sys.exit(1)

    # Build a unified timeline: find the common time range across all symbols
    # and walk bar-by-bar, checking all symbols at each bar (like the live system)
    print("\nGenerating training data from ZP signals (multi-symbol simulation)...")

    # Find the earliest start and latest end across all H1 data
    min_start = None
    max_end = None
    for sym, data in all_data.items():
        df = data['df_h1']
        t_start = df.iloc[WARMUP_BARS]['time']
        t_end = df.iloc[-21]['time']  # leave 20 bars for trade play-out
        if min_start is None or t_start > min_start:
            min_start = t_start  # use latest start so all syms have warmup
        if max_end is None or t_end < max_end:
            max_end = t_end  # use earliest end so all syms have play-out room

    print(f"  Simulation range: {min_start} to {max_end}")

    # Pre-compute ZP states for all symbols at all bar indices to avoid
    # recomputing compute_zeropoint_state() thousands of times.
    # We cache: sym -> { bar_idx -> zp_state_df }
    # But this is too memory-heavy. Instead, we walk each symbol's bar index
    # aligned to a master time axis.

    # Build bar-index lookup: sym -> time-sorted bar indices in simulation range
    sym_bar_lookup = {}
    for sym, data in all_data.items():
        df = data['df_h1']
        indices = []
        for i in range(len(df)):
            t = df.iloc[i]['time']
            if t >= min_start and t <= max_end:
                indices.append(i)
        sym_bar_lookup[sym] = indices

    # Build master timeline from the union of all H1 bar times
    all_times = set()
    for sym, data in all_data.items():
        df = data['df_h1']
        for i in sym_bar_lookup.get(sym, []):
            all_times.add(df.iloc[i]['time'])
    master_times = sorted(all_times)
    print(f"  Master timeline: {len(master_times)} unique H1 bar times")

    # Build time -> bar_idx mapping for each symbol
    sym_time_to_idx = {}
    for sym, data in all_data.items():
        df = data['df_h1']
        t2i = {}
        for i in range(len(df)):
            t2i[df.iloc[i]['time']] = i
        sym_time_to_idx[sym] = t2i

    # Simulation state
    open_positions = []  # list of Position objects
    all_features = []
    all_labels = []
    all_meta = []
    last_direction = {}  # sym -> last ZP direction (for dedup)
    sym_signals = Counter()
    sym_wins = Counter()

    for time_idx, cur_time in enumerate(master_times):
        # --- First: update all open positions at this bar ---
        still_open = []
        for pos in open_positions:
            if pos.closed:
                continue
            sym = pos.sym
            data = all_data[sym]
            t2i = sym_time_to_idx[sym]
            if cur_time not in t2i:
                still_open.append(pos)
                continue
            bar_idx = t2i[cur_time]
            df_h1 = data['df_h1']

            # Compute ZP for this bar to get confirmed_pos for ZP flip exit
            window_h1 = df_h1.iloc[:bar_idx + 1]
            zp_h1 = compute_zeropoint_state(window_h1)
            if zp_h1 is None or len(zp_h1) < 2:
                still_open.append(pos)
                continue

            fwd_bar = zp_h1.iloc[-1]
            fwd_confirmed = zp_h1.iloc[-2]
            fwd_high = float(fwd_bar['high'])
            fwd_low = float(fwd_bar['low'])
            fwd_close = float(fwd_bar['close'])
            fwd_conf_pos = int(fwd_confirmed.get('pos', 0))

            pos.check_bar(fwd_high, fwd_low, fwd_close, fwd_conf_pos)
            if not pos.closed:
                still_open.append(pos)
        open_positions = still_open

        # --- Second: check for new entries across all symbols ---
        for sym, data in all_data.items():
            t2i = sym_time_to_idx[sym]
            if cur_time not in t2i:
                continue
            bar_idx = t2i[cur_time]
            df_h1 = data['df_h1']
            df_m15 = data['df_m15']
            df_h4 = data['df_h4']
            pip_size = data['pip_size']
            contract_size = data['contract_size']
            n = len(df_h1)

            # Skip if already in a position for this symbol (live system: 1 per symbol)
            sym_has_pos = any(p.sym == sym and not p.closed for p in open_positions)
            if sym_has_pos:
                continue
            if len(open_positions) >= MAX_OPEN:
                continue

            window_h1 = df_h1.iloc[:bar_idx + 1].copy()
            zp_h1 = compute_zeropoint_state(window_h1)
            if zp_h1 is None or len(zp_h1) < 3:
                continue

            cur_bar = zp_h1.iloc[-1]
            confirmed_bar = zp_h1.iloc[-2]

            pos_val = int(cur_bar.get('pos', 0))
            if pos_val == 0:
                last_direction[sym] = None
                continue

            direction = 'BUY' if pos_val == 1 else 'SELL'

            # Only take signal on fresh flip (avoid re-signaling same direction)
            if direction == last_direction.get(sym):
                continue
            last_direction[sym] = direction

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

            entry = float(cur_bar['close'])

            # Use V4 PROTECT TPs (matching live system)
            if direction == 'BUY':
                tp1 = entry + atr_val * TP1_MULT_AGG
                tp2 = entry + atr_val * TP2_MULT_AGG
                tp3 = entry + atr_val * TP3_MULT_AGG
            else:
                tp1 = entry - atr_val * TP1_MULT_AGG
                tp2 = entry - atr_val * TP2_MULT_AGG
                tp3 = entry - atr_val * TP3_MULT_AGG

            if direction == 'BUY' and entry >= tp1:
                continue
            if direction == 'SELL' and entry <= tp1:
                continue

            # M15 confirmation
            m15_conf = False
            zp_m15_state = None
            if df_m15 is not None:
                m15_mask = df_m15['time'] <= cur_time
                m15_window = df_m15[m15_mask].tail(200)
                if len(m15_window) >= 20:
                    zp_m15_state = compute_zeropoint_state(m15_window)
                    if zp_m15_state is not None and len(zp_m15_state) > 0:
                        m15_pos = int(zp_m15_state.iloc[-1].get('pos', 0))
                        if direction == 'BUY' and m15_pos == 1:
                            m15_conf = True
                        elif direction == 'SELL' and m15_pos == -1:
                            m15_conf = True

            # H4 ZP state
            zp_h4_state = None
            h4_window = None
            if df_h4 is not None:
                h4_mask = df_h4['time'] <= cur_time
                h4_window = df_h4[h4_mask].tail(50)
                if len(h4_window) >= 10:
                    try:
                        zp_h4_state = compute_zeropoint_state(h4_window)
                    except Exception:
                        pass
                else:
                    h4_window = None

            # Confidence
            sl_dist = abs(entry - sl)
            tp_dist = abs(tp1 - entry)
            rr = tp_dist / sl_dist if sl_dist > 0 else 0

            conf = 0.65
            if is_fresh:
                conf += 0.15
            if m15_conf:
                conf += 0.10
            if rr >= 2.0:
                conf += 0.05
            elif rr >= 1.5:
                conf += 0.03
            if not is_fresh:
                conf -= min(bars_since * 0.03, 0.20)
            conf = max(0.40, min(conf, 0.98))

            # Extract features
            h1_window = window_h1.tail(60)
            m15_window_ctx = None
            if df_m15 is not None:
                m15_mask_ctx = df_m15['time'] <= cur_time
                m15_window_ctx = df_m15[m15_mask_ctx].tail(200)
                if len(m15_window_ctx) < 20:
                    m15_window_ctx = None

            feat = extract_features(
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                tp1=tp1, tp2=tp2, tp3=tp3,
                atr_value=atr_val,
                is_fresh=is_fresh,
                bars_since_flip=bars_since,
                m15_confirmed=m15_conf,
                confidence=conf,
                h1_bars=h1_window,
                m15_bars=m15_window_ctx,
                h4_bars=h4_window,
                zp_h1=zp_h1,
                zp_m15=zp_m15_state,
                zp_h4=zp_h4_state,
                symbol=sym,
                symbol_to_index=symbol_to_index,
            )

            # --- Create position and track it (matches live system) ---
            pos = Position(
                sym=sym, direction=direction, entry=entry, sl=sl,
                tp1=tp1, tp2=tp2, tp3=tp3, lot=0.04,
                entry_bar_idx=bar_idx, pip_size=pip_size,
                contract_size=contract_size, atr_value=atr_val,
            )
            open_positions.append(pos)

            # Store feature + position reference for later labeling
            all_features.append(feat)
            all_meta.append({
                'sym': sym, 'dir': direction, 'bar_idx': bar_idx,
                'time': cur_time, 'position': pos,
            })
            sym_signals[sym] += 1

        # Progress update every 500 bars
        if (time_idx + 1) % 500 == 0:
            n_open = len(open_positions)
            n_total = len(all_features)
            print(f"  Bar {time_idx+1}/{len(master_times)}: "
                  f"{n_total} signals, {n_open} open positions")

    # Force-close any remaining open positions
    for pos in open_positions:
        if not pos.closed:
            sym = pos.sym
            df_h1 = all_data[sym]['df_h1']
            last_close = float(df_h1.iloc[-1]['close'])
            pos.force_close(last_close)

    # Now label all samples from their completed positions
    all_labels = []
    exit_types = Counter()
    tp1_hit_count = 0
    tp2_hit_count = 0
    tp3_hit_count = 0
    for m in all_meta:
        pos = m['position']
        pnl = pos.total_pnl
        m['pnl'] = pnl
        m['tp1_hit'] = pos.tp1_hit
        m['tp2_hit'] = pos.tp2_hit
        m['tp3_hit'] = pos.tp3_hit
        if pos.tp1_hit:
            tp1_hit_count += 1
        if pos.tp2_hit:
            tp2_hit_count += 1
        if pos.tp3_hit:
            tp3_hit_count += 1
        # Track exit type
        if pos.partials:
            last_exit = pos.partials[-1][3]  # exit label
            exit_types[last_exit] += 1
        # Pine Script exact win counting:
        # if finalPnL > 0 or tp1Hit or maxProfitReached > 0 → WIN
        # This matches the indicator's WR display exactly.
        pine_win = (pos.total_pnl > 0 or pos.tp1_hit or pos.max_profit_reached > 0)
        m['pine_win'] = pine_win
        # Label: raw PnL > 0 (strict, real profitability)
        # Pine WR is tracked separately for diagnostics
        label = 1 if pos.total_pnl > 0 else 0
        all_labels.append(label)
        if label == 1:
            sym_wins[m['sym']] += 1

    # Remove position references from meta (not serializable)
    for m in all_meta:
        del m['position']

    # Print per-symbol stats
    sym_pine_wins = Counter()
    for m in all_meta:
        if m.get('pine_win'):
            sym_pine_wins[m['sym']] += 1

    for sym in SYMBOLS:
        if sym in sym_signals:
            n_sig = sym_signals[sym]
            n_win = sym_wins[sym]
            n_pine = sym_pine_wins.get(sym, 0)
            wr = n_win / n_sig * 100 if n_sig > 0 else 0
            pwr = n_pine / n_sig * 100 if n_sig > 0 else 0
            print(f"  {sym}: {n_sig} signals, raw WR={wr:.0f}%, Pine WR={pwr:.0f}%")

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)

    n_total = len(features)
    raw_wins = sum(1 for m in all_meta if m['pnl'] > 0)
    pine_wins = sum(1 for m in all_meta if m.get('pine_win'))
    print(f"\nTotal: {n_total} samples")
    print(f"  Raw PnL wins:  {raw_wins} ({raw_wins/n_total*100:.1f}%) <- model label")
    print(f"  Pine WR:       {pine_wins} ({pine_wins/n_total*100:.1f}%) <- indicator display")

    # Diagnostic: exit type distribution
    print(f"\nExit type distribution:")
    for exit_type, count in sorted(exit_types.items(), key=lambda x: -x[1]):
        print(f"  {exit_type:>15}: {count:>4} ({count/n_total*100:.1f}%)")
    print(f"\nTP rates:")
    print(f"  TP1 hit: {tp1_hit_count:>4} ({tp1_hit_count/n_total*100:.1f}%)")
    print(f"  TP2 hit: {tp2_hit_count:>4} ({tp2_hit_count/n_total*100:.1f}%)")
    print(f"  TP3 hit: {tp3_hit_count:>4} ({tp3_hit_count/n_total*100:.1f}%)")

    return features, labels, all_meta, symbol_to_index


# ─── Training ────────────────────────────────────────────────────────────────
def train_model(features, labels, symbol_to_index):
    """Train CouncilPredictor on labeled ZP signals."""

    n = len(features)
    split_idx = int(n * TRAIN_SPLIT)

    X_train = features[:split_idx]
    y_train = labels[:split_idx]
    X_val = features[split_idx:]
    y_val = labels[split_idx:]

    print(f"\nTrain: {len(X_train)} samples ({y_train.mean()*100:.1f}% wins)")
    print(f"Val:   {len(X_val)} samples ({y_val.mean()*100:.1f}% wins)")

    if len(X_train) < 20:
        print("ERROR: Too few training samples. Need at least 20.")
        return None, None, None

    # Normalize (fit on train only)
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    X_train_norm = (X_train - feature_mean) / feature_std
    X_val_norm = (X_val - feature_mean) / feature_std

    # Clean NaN/Inf
    X_train_norm = np.where(np.isfinite(X_train_norm), X_train_norm, 0.0)
    X_val_norm = np.where(np.isfinite(X_val_norm), X_val_norm, 0.0)

    # Convert to tensors
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Class weight for BCEWithLogitsLoss
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)])

    # Label smoothing
    if LABEL_SMOOTHING > 0:
        y_train_t = y_train_t * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

    # Model
    input_dim = features.shape[1]
    model = CouncilPredictor(input_dim=input_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    batch_size = min(32, max(8, len(X_train) // 4))
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    print(f"\nTraining: dim={input_dim}, batch={batch_size}, "
          f"pos_weight={pos_weight.item():.2f}")
    print(f"{'Epoch':>5} {'Train Loss':>10} {'Val Loss':>10} {'Val Acc':>8} {'LR':>10}")

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        perm = torch.randperm(len(X_train_t))
        train_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validate (Bayesian: multiple forward passes)
        model.train()  # keep dropout ON for Bayesian
        val_preds = np.zeros(len(X_val_t))
        with torch.no_grad():
            for s in range(N_BAYESIAN_SAMPLES):
                logits = model(X_val_t)
                probs = torch.sigmoid(logits).squeeze().numpy()
                val_preds += probs
        val_preds /= N_BAYESIAN_SAMPLES

        val_pred_t = torch.tensor(val_preds, dtype=torch.float32).unsqueeze(1)
        # Use original labels (no smoothing) for val loss
        y_val_orig = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        val_loss = nn.BCELoss()(val_pred_t, y_val_orig).item()

        val_acc = ((val_preds > 0.5) == y_val).mean() * 100
        cur_lr = optimizer.param_groups[0]['lr']

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:>5} {train_loss:>10.4f} {val_loss:>10.4f} "
                  f"{val_acc:>7.1f}% {cur_lr:>10.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest val loss: {best_val_loss:.4f}")

    return model, feature_mean, feature_std


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_model(model, features, labels, meta, feature_mean, feature_std):
    """Evaluate trained model with Bayesian inference."""

    n = len(features)
    split_idx = int(n * TRAIN_SPLIT)

    X_val = features[split_idx:]
    y_val = labels[split_idx:]
    meta_val = meta[split_idx:]

    # Normalize
    std = np.where(feature_std < 1e-8, 1.0, feature_std)
    X_val_norm = (X_val - feature_mean) / std
    X_val_norm = np.where(np.isfinite(X_val_norm), X_val_norm, 0.0)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32)

    # Bayesian prediction
    model.train()
    all_preds = np.zeros((N_BAYESIAN_SAMPLES, len(X_val_t)))
    with torch.no_grad():
        for s in range(N_BAYESIAN_SAMPLES):
            logits = model(X_val_t)
            all_preds[s] = torch.sigmoid(logits).squeeze().numpy()

    mean_preds = all_preds.mean(axis=0)
    std_preds = all_preds.std(axis=0)

    # Accuracy
    acc = ((mean_preds > 0.5) == y_val).mean() * 100
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS ({len(y_val)} trades)")
    print(f"{'='*80}")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"  Baseline WR: {y_val.mean()*100:.1f}%")
    print(f"  Mean prediction: {mean_preds.mean():.3f}")
    print(f"  Mean uncertainty: {std_preds.mean():.3f}")

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, mean_preds)
        print(f"  AUC-ROC: {auc:.3f}")
    except ImportError:
        auc = 0.5
        # Manual AUC approximation
        sorted_idx = np.argsort(mean_preds)
        sorted_labels = y_val[sorted_idx]
        n_pos = y_val.sum()
        n_neg = len(y_val) - n_pos
        if n_pos > 0 and n_neg > 0:
            cum_tp = np.cumsum(sorted_labels[::-1])
            cum_fp = np.cumsum(1 - sorted_labels[::-1])
            tpr = cum_tp / n_pos
            fpr = cum_fp / n_neg
            auc = np.trapz(tpr, fpr)
        print(f"  AUC-ROC: {auc:.3f} (manual)")

    # Threshold analysis
    print(f"\n  {'Threshold':>10} {'Trades':>7} {'Wins':>5} {'WR':>6} {'Avg Lot':>8} {'PnL':>10}")
    print(f"  {'-'*55}")

    # Simulate lot sizing at different thresholds
    for label, thresholds in [
        ("Full (1.0x)", (0.65, 1.0)),
        ("Mild (0.85x)", (0.50, 0.65)),
        ("Low (0.65x)", (0.35, 0.50)),
        ("Half (0.50x)", (0.0, 0.35)),
    ]:
        lo, hi = thresholds
        mask = (mean_preds >= lo) & (mean_preds < hi) if hi < 1.0 else (mean_preds >= lo)
        if mask.sum() == 0:
            continue
        bucket_labels = y_val[mask]
        bucket_pnl = sum(m['pnl'] for i, m in enumerate(meta_val) if mask[i])
        wr = bucket_labels.mean() * 100
        print(f"  {label:<15} {mask.sum():>5} {int(bucket_labels.sum()):>5} "
              f"{wr:>5.0f}% {'':>8} ${bucket_pnl:>+9.2f}")

    # Simulate learned lot sizing vs baseline
    print(f"\n  Simulated P/L comparison (validation period):")
    baseline_pnl = sum(m['pnl'] for m in meta_val)

    learned_pnl = 0.0
    for i, m in enumerate(meta_val):
        prob = mean_preds[i]
        uncertainty = std_preds[i]

        if uncertainty > 0.15:
            mult = 1.0
        elif prob >= 0.65:
            mult = 1.0
        elif prob >= 0.50:
            mult = 0.85
        elif prob >= 0.35:
            mult = 0.65
        else:
            mult = 0.50

        learned_pnl += m['pnl'] * mult

    print(f"    Baseline:  ${baseline_pnl:>+.2f}")
    print(f"    Learned:   ${learned_pnl:>+.2f}")
    improvement = learned_pnl - baseline_pnl
    print(f"    Delta:     ${improvement:>+.2f}")

    return auc


# ─── Feature Importance ───────────────────────────────────────────────────────
# Feature names corresponding to the 12 numeric indices in council_predictor.py V4
FEATURE_NAMES = [
    'ma_fan_score',
    'sma50_dist_aligned',
    'consecutive_dir_bars',
    'signal_bar_body_ratio',
    'engulfing',
    'pin_bar_ratio',
    'confidence',
    'freshness',
    'bars_since_flip',
    'm15_confirmed',
    'h4_zp_agreement',
    'sl_dist_in_atr',
]


def analyze_feature_importance(model, features, labels, feature_mean, feature_std,
                               n_permutations=10):
    """Permutation importance: shuffle each feature and measure AUC drop.

    For each of the 33 numeric features (not symbol one-hot), we:
      1. Shuffle that column in the normalized validation set
      2. Run Bayesian forward passes to get predictions
      3. Compute AUC on the shuffled predictions
      4. Importance = baseline_AUC - shuffled_AUC

    Features where shuffling causes a large AUC drop are important.
    Features where shuffling barely changes AUC are noise.
    """
    from sklearn.metrics import roc_auc_score

    n = len(features)
    split_idx = int(n * TRAIN_SPLIT)

    X_val = features[split_idx:]
    y_val = labels[split_idx:]

    if len(X_val) < 10 or y_val.sum() == 0 or y_val.sum() == len(y_val):
        print("\nSkipping feature importance: insufficient validation data or no class variance.")
        return

    # Normalize
    std = np.where(feature_std < 1e-8, 1.0, feature_std)
    X_val_norm = (X_val - feature_mean) / std
    X_val_norm = np.where(np.isfinite(X_val_norm), X_val_norm, 0.0)

    # --- Baseline AUC ---
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32)
    model.train()  # dropout ON for Bayesian
    baseline_preds = np.zeros(len(X_val_t))
    with torch.no_grad():
        for s in range(N_BAYESIAN_SAMPLES):
            logits = model(X_val_t)
            baseline_preds += torch.sigmoid(logits).squeeze().numpy()
    baseline_preds /= N_BAYESIAN_SAMPLES

    try:
        baseline_auc = roc_auc_score(y_val, baseline_preds)
    except ValueError:
        print("\nSkipping feature importance: AUC undefined (single class in val set).")
        return

    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE (Permutation Test)")
    print(f"{'='*80}")
    print(f"  Baseline AUC: {baseline_auc:.4f}")
    print(f"  Permutations per feature: {n_permutations}")
    print(f"  Evaluating {NUM_NUMERIC_FEATURES} numeric features (excluding symbol one-hot)...\n")

    num_features = NUM_NUMERIC_FEATURES  # only numeric, not one-hot
    importance_scores = np.zeros(num_features)

    rng = np.random.default_rng(42)

    for feat_idx in range(num_features):
        auc_drops = []

        for perm_run in range(n_permutations):
            # Copy and shuffle one column
            X_shuffled = X_val_norm.copy()
            X_shuffled[:, feat_idx] = rng.permutation(X_shuffled[:, feat_idx])

            X_shuf_t = torch.tensor(X_shuffled, dtype=torch.float32)

            # Bayesian prediction on shuffled data
            shuf_preds = np.zeros(len(X_shuf_t))
            with torch.no_grad():
                for s in range(N_BAYESIAN_SAMPLES):
                    logits = model(X_shuf_t)
                    shuf_preds += torch.sigmoid(logits).squeeze().numpy()
            shuf_preds /= N_BAYESIAN_SAMPLES

            try:
                shuf_auc = roc_auc_score(y_val, shuf_preds)
                auc_drops.append(baseline_auc - shuf_auc)
            except ValueError:
                auc_drops.append(0.0)

        importance_scores[feat_idx] = np.mean(auc_drops)

    # Sort by importance (descending)
    ranked_idx = np.argsort(importance_scores)[::-1]

    print(f"  {'Rank':>4}  {'Feature':<30}  {'AUC Drop':>10}  {'Verdict'}")
    print(f"  {'-'*70}")

    for rank, idx in enumerate(ranked_idx, 1):
        drop = importance_scores[idx]
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f'feature_{idx}'

        if drop >= 0.02:
            verdict = 'HIGH'
        elif drop >= 0.005:
            verdict = 'MEDIUM'
        elif drop >= 0.001:
            verdict = 'LOW'
        else:
            verdict = 'NOISE'

        print(f"  {rank:>4}  {name:<30}  {drop:>+10.4f}  {verdict}")

    # Summary by group
    group_ranges = {
        'Trend Context (0-2)': range(0, 3),
        'Signal Quality (3-6)': range(3, 7),
        'Timing (7-8)': range(7, 9),
        'Multi-TF (9-10)': range(9, 11),
        'Structure (11)': range(11, 12),
    }

    print(f"\n  {'Group':<30}  {'Mean AUC Drop':>13}  {'Max AUC Drop':>12}")
    print(f"  {'-'*60}")
    for group_name, idx_range in group_ranges.items():
        group_scores = [importance_scores[i] for i in idx_range]
        mean_drop = np.mean(group_scores)
        max_drop = np.max(group_scores)
        print(f"  {group_name:<30}  {mean_drop:>+13.4f}  {max_drop:>+12.4f}")

    # Identify noise features
    noise_features = [FEATURE_NAMES[i] for i in range(num_features)
                      if importance_scores[i] < 0.001]
    if noise_features:
        print(f"\n  Potential noise features ({len(noise_features)}):")
        for nf in noise_features:
            print(f"    - {nf}")

    print()


# ─── Save ────────────────────────────────────────────────────────────────────
def save_model(model, feature_mean, feature_std, symbol_to_index,
               n_samples, auc, path=CHECKPOINT_PATH):
    """Save model checkpoint."""

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_dim': FEATURE_DIM,
        'feature_mean': feature_mean.tolist(),
        'feature_std': feature_std.tolist(),
        'symbol_to_index': symbol_to_index,
        'metadata': {
            'model_type': 'council_predictor',
            'n_training_samples': n_samples,
            'val_auc': auc,
            'train_date': datetime.now().isoformat(),
            'feature_groups': {
                'A_zp_signal': 6,
                'B_price_structure': 7,
                'C_momentum_volume': 5,
                'D_multi_tf': 4,
                'E_account_state': 4,
                'F_symbol_onehot': NUM_SYMBOLS,
            },
        },
        'save_date': datetime.now().isoformat(),
    }

    torch.save(checkpoint, path)
    print(f"\nModel saved to {path}")
    print(f"  Feature dim: {FEATURE_DIM}")
    print(f"  Training samples: {n_samples}")
    print(f"  Val AUC: {auc:.3f}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("COUNCIL PREDICTOR TRAINER")
    print("Learns to predict ZP trade outcomes from historical data")
    print("=" * 80)
    print()

    # Phase 1: Generate data
    features, labels, meta, symbol_to_index = generate_training_data()

    if len(features) < 30:
        print(f"\nInsufficient data ({len(features)} samples). Need at least 30.")
        print("Try fetching more MT5 history.")
        sys.exit(1)

    # Phase 2: Train
    model, feature_mean, feature_std = train_model(features, labels, symbol_to_index)

    if model is None:
        print("Training failed.")
        sys.exit(1)

    # Phase 3: Evaluate
    auc = evaluate_model(model, features, labels, meta, feature_mean, feature_std)

    # Phase 3b: Feature importance analysis
    analyze_feature_importance(model, features, labels, feature_mean, feature_std)

    # Phase 4: Save
    save_model(model, feature_mean, feature_std, symbol_to_index,
               n_samples=len(features), auc=auc)

    print("\nDone! Run backtest_council.py to see 3-way comparison.")


if __name__ == '__main__':
    main()
