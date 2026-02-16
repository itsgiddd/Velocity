#!/usr/bin/env python3
"""
Multi-Timeframe Comparison Backtest
====================================
Tests V4 Profit Capture on H1, H2, H3, and H4 to find the optimal
timeframe for compounding frequency vs win rate.

All PnL properly converted to USD.
Compounding risk sizing matching live system.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import MetaTrader5 as mt5
from app.zeropoint_signal import (
    compute_zeropoint_state,
    ATR_PERIOD, ATR_MULTIPLIER,
    TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,
    SL_BUFFER_PCT, SL_ATR_MIN_MULT, SWING_LOOKBACK,
    BE_TRIGGER_MULT, BE_BUFFER_MULT,
    PROFIT_TRAIL_DISTANCE_MULT, STALL_BARS,
    MICRO_TP_MULT, MICRO_TP_PCT,
)

SYMBOLS = ["AUDUSD", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDJPY"]
FETCH_BARS = 10000  # More bars for lower TFs
WARMUP_BARS = 50
STARTING_BALANCE = 200.0

# Risk params
BASE_RISK_PCT = 0.30
WIN_STREAK_BONUS = 0.25
MAX_RISK_PCT = 0.40
LOSS_PENALTY = 0.375
HIGH_BALANCE_CAP_RISK = 0.20

LOT_CAP_TABLE = [
    (500, 0.10), (1000, 0.20), (3000, 0.50),
    (5000, 1.00), (10000, 2.00), (50000, 5.00),
    (float('inf'), 10.00),
]

TIMEFRAMES = {
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
}


def resolve_symbol(ticker):
    for c in [ticker, ticker + ".raw", ticker + "m", ticker + ".a", ticker + ".e"]:
        info = mt5.symbol_info(c)
        if info is not None:
            mt5.symbol_select(c, True)
            return c
    return None


def get_symbol_specs():
    specs = {}
    for sym in SYMBOLS:
        resolved = resolve_symbol(sym)
        if resolved:
            info = mt5.symbol_info(resolved)
            if info:
                tick_size = info.trade_tick_size or info.point
                tick_value = info.trade_tick_value
                if tick_value <= 0:
                    tick_value = info.trade_contract_size * tick_size
                specs[sym] = {
                    "tick_size": tick_size,
                    "tick_value": tick_value,
                    "contract_size": info.trade_contract_size,
                    "volume_min": info.volume_min,
                    "volume_max": info.volume_max,
                    "volume_step": info.volume_step,
                    "resolved": resolved,
                }
    return specs


def get_conversion_rates():
    rates = {}
    for pair in ["USDJPY", "USDCAD"]:
        resolved = resolve_symbol(pair)
        if resolved:
            tick = mt5.symbol_info_tick(resolved)
            if tick:
                rates[pair] = (tick.bid + tick.ask) / 2.0
    return rates


def pnl_to_usd(raw_pnl, symbol, conv_rates):
    sym = symbol.upper()
    if sym.endswith("JPY"):
        return raw_pnl / conv_rates.get("USDJPY", 152.0)
    elif sym.endswith("CAD"):
        return raw_pnl / conv_rates.get("USDCAD", 1.36)
    return raw_pnl


def calc_lot_size(balance, entry, sl, sym_specs, risk_pct):
    tick_size = sym_specs["tick_size"]
    tick_value = sym_specs["tick_value"]
    sl_distance = abs(entry - sl)
    sl_ticks = sl_distance / tick_size if tick_size > 0 else 0
    loss_per_lot = sl_ticks * tick_value
    if loss_per_lot <= 0:
        return sym_specs["volume_min"]
    risk_amount = balance * risk_pct
    lot = risk_amount / loss_per_lot
    vol_step = sym_specs["volume_step"]
    lot = round(lot / vol_step) * vol_step
    lot = max(sym_specs["volume_min"], min(lot, sym_specs["volume_max"]))
    for threshold, max_lot in LOT_CAP_TABLE:
        if balance <= threshold:
            lot = min(lot, max_lot)
            break
    return lot


def compute_smart_sl(df, bar_idx, direction, atr_val):
    lookback_start = max(0, bar_idx - SWING_LOOKBACK + 1)
    cur_close = float(df["close"].iloc[bar_idx])
    if direction == "BUY":
        recent_low = float(df["low"].iloc[lookback_start:bar_idx + 1].min())
        buffer = recent_low * SL_BUFFER_PCT
        structural_sl = recent_low - buffer
        atr_min_sl = cur_close - atr_val * SL_ATR_MIN_MULT
        return min(structural_sl, atr_min_sl)
    else:
        recent_high = float(df["high"].iloc[lookback_start:bar_idx + 1].max())
        buffer = recent_high * SL_BUFFER_PCT
        structural_sl = recent_high + buffer
        atr_max_sl = cur_close + atr_val * SL_ATR_MIN_MULT
        return max(structural_sl, atr_max_sl)


class V4Position:
    def __init__(self, sym, direction, entry, sl, atr_val, lot, contract_size):
        self.sym = sym
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.original_sl = sl
        self.atr_val = atr_val
        self.total_lot = lot
        self.remaining_lot = lot
        self.contract_size = contract_size
        sign = 1 if direction == "BUY" else -1
        self.tp1 = entry + sign * TP1_MULT_AGG * atr_val
        self.tp2 = entry + sign * TP2_MULT_AGG * atr_val
        self.tp3 = entry + sign * TP3_MULT_AGG * atr_val
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        self.closed = False
        self.partials = []
        self.bars_in_trade = 0
        self.exit_time = None
        self.entry_time = None
        self.max_profit_reached = 0.0
        self.max_favorable_price = entry
        self.sl_hit = False
        self.final_exit_type = None
        self.be_activated = False
        self.profit_lock_sl = None
        self.profit_lock_active = False
        self.micro_tp_hit = False
        self.stall_be_activated = False
        self._usd_pnl = 0.0

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
        atr = self.atr_val if self.atr_val > 1e-12 else 1.0

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
                    self.final_exit_type = 'MICRO_TP'
                    return events

        if not self.tp1_hit:
            if (is_buy and high >= self.tp1) or (not is_buy and low <= self.tp1):
                self.tp1_hit = True
                partial = min(self.partial_lot(), self.remaining_lot)
                pnl = self.pnl_for_price(self.tp1, partial)
                self.partials.append((self.tp1, partial, pnl, 'TP1'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP1', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'TP1'
                    return events

        if self.tp1_hit and not self.tp2_hit:
            if (is_buy and high >= self.tp2) or (not is_buy and low <= self.tp2):
                self.tp2_hit = True
                self.sl = self.entry
                self.be_activated = True
                partial = min(self.partial_lot(), self.remaining_lot)
                pnl = self.pnl_for_price(self.tp2, partial)
                self.partials.append((self.tp2, partial, pnl, 'TP2'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP2', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'TP2'
                    return events

        if self.tp2_hit and not self.tp3_hit:
            if (is_buy and high >= self.tp3) or (not is_buy and low <= self.tp3):
                self.tp3_hit = True
                pnl = self.pnl_for_price(self.tp3, self.remaining_lot)
                self.partials.append((self.tp3, self.remaining_lot, pnl, 'TP3'))
                events.append(('TP3', pnl))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'TP3'
                return events

        if self.profit_lock_active and self.profit_lock_sl is not None:
            lock_hit = (is_buy and low <= self.profit_lock_sl) or \
                       (not is_buy and high >= self.profit_lock_sl)
            if lock_hit:
                pnl = self.pnl_for_price(self.profit_lock_sl, self.remaining_lot)
                self.partials.append((self.profit_lock_sl, self.remaining_lot, pnl, 'PROFIT_LOCK'))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'PROFIT_LOCK'
                events.append(('PROFIT_LOCK', pnl))
                return events

        if (is_buy and low <= self.sl) or (not is_buy and high >= self.sl):
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
            self.final_exit_type = exit_label
            events.append((exit_label, pnl))
            return events

        if confirmed_pos != 0:
            if (is_buy and confirmed_pos == -1) or (not is_buy and confirmed_pos == 1):
                pnl = self.pnl_for_price(close, self.remaining_lot)
                self.partials.append((close, self.remaining_lot, pnl, 'ZP_FLIP'))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'ZP_FLIP'
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
        self.final_exit_type = 'END'
        return pnl

    @property
    def total_pnl(self):
        return sum(p[2] for p in self.partials)


def run_backtest_for_tf(tf_name, tf_mt5, sym_specs, conv_rates, start_date=None):
    """Run full compounding backtest for a given timeframe. Returns summary dict.

    If start_date is provided, only uses data from that date onward (for fair comparison).
    Extra warmup bars are fetched before start_date so ATR/ZP indicators are primed.
    """

    symbol_data = {}
    for sym in SYMBOLS:
        if sym not in sym_specs:
            continue
        resolved = sym_specs[sym]["resolved"]
        if start_date:
            # Fetch extra bars before start_date for indicator warmup
            # We need ~100 bars of warmup for ATR(10) + ZP to stabilize
            warmup_fetch = 200
            rates_warmup = mt5.copy_rates_from_pos(resolved, tf_mt5, 0, 50000)
            if rates_warmup is None or len(rates_warmup) < 100:
                continue
            df_all = pd.DataFrame(rates_warmup)
            df_all["time"] = pd.to_datetime(df_all["time"], unit="s")
            # Find index of start_date, include warmup bars before it
            start_mask = df_all["time"] >= pd.Timestamp(start_date)
            if not start_mask.any():
                continue
            first_idx = start_mask.idxmax()
            warmup_start = max(0, first_idx - warmup_fetch)
            df_all = df_all.iloc[warmup_start:].reset_index(drop=True)
            rates = df_all.to_records(index=False)
            df = df_all
        else:
            rates = mt5.copy_rates_from_pos(resolved, tf_mt5, 0, FETCH_BARS)
            if rates is None or len(rates) < 100:
                continue
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True, errors="ignore")
        df_zp = compute_zeropoint_state(df)
        if df_zp is None or len(df_zp) < WARMUP_BARS:
            continue
        # If start_date, trim to only trade from start_date onward
        # but keep warmup bars for indicator lookback
        if start_date:
            trade_start_mask = df_zp["time"] >= pd.Timestamp(start_date)
            if not trade_start_mask.any():
                continue
            trade_start_idx = trade_start_mask.idxmax()
            # Mark trade_start_idx so we only generate signals from here
            df_zp.attrs["trade_start_idx"] = trade_start_idx
        symbol_data[sym] = df_zp

    if not symbol_data:
        return None

    # Build unified bar stream
    bar_events = []
    for sym, df in symbol_data.items():
        n = len(df)
        # Only generate signals from trade_start_idx if set (fair comparison mode)
        signal_start = max(WARMUP_BARS, df.attrs.get("trade_start_idx", WARMUP_BARS))
        for i in range(signal_start, n):
            row = df.iloc[i]
            atr_val = float(row["atr"])
            if np.isnan(atr_val) or atr_val <= 0:
                continue
            bar_events.append((
                row["time"], sym,
                float(row["high"]), float(row["low"]), float(row["close"]),
                atr_val, int(row.get("pos", 0)),
                bool(row.get("buy_signal", False)), bool(row.get("sell_signal", False)),
                i,
            ))
    bar_events.sort(key=lambda x: x[0])

    # Simulate
    balance = STARTING_BALANCE
    risk_pct = BASE_RISK_PCT
    win_streak = 0
    all_trades = []
    open_positions = {}

    for time, sym, high, low, close, atr_val, pos, buy_sig, sell_sig, bar_idx in bar_events:
        if sym not in sym_specs:
            continue

        if sym in open_positions:
            p = open_positions[sym]
            if not p.closed:
                p.check_bar(high, low, close, pos)
                if p.closed:
                    p.exit_time = time
                    usd_pnl = pnl_to_usd(p.total_pnl, sym, conv_rates)
                    p._usd_pnl = usd_pnl
                    balance += usd_pnl
                    all_trades.append(p)
                    del open_positions[sym]
                    if usd_pnl > 0:
                        win_streak += 1
                        if win_streak >= 3:
                            risk_pct = min(BASE_RISK_PCT * (1 + WIN_STREAK_BONUS), MAX_RISK_PCT)
                    else:
                        win_streak = 0
                        risk_pct = max(BASE_RISK_PCT * (1 - LOSS_PENALTY), 0.05)
                    if balance > 50000:
                        risk_pct = min(risk_pct, HIGH_BALANCE_CAP_RISK)

        if buy_sig or sell_sig:
            direction = "BUY" if buy_sig else "SELL"
            entry = close
            smart_sl = compute_smart_sl(symbol_data[sym], bar_idx, direction, atr_val)

            if sym in open_positions and not open_positions[sym].closed:
                p = open_positions[sym]
                p.force_close(close)
                p.exit_time = time
                usd_pnl = pnl_to_usd(p.total_pnl, sym, conv_rates)
                p._usd_pnl = usd_pnl
                balance += usd_pnl
                all_trades.append(p)
                if usd_pnl > 0:
                    win_streak += 1
                    if win_streak >= 3:
                        risk_pct = min(BASE_RISK_PCT * (1 + WIN_STREAK_BONUS), MAX_RISK_PCT)
                else:
                    win_streak = 0
                    risk_pct = max(BASE_RISK_PCT * (1 - LOSS_PENALTY), 0.05)
                if balance > 50000:
                    risk_pct = min(risk_pct, HIGH_BALANCE_CAP_RISK)

            if balance < 10:
                continue

            lot = calc_lot_size(balance, entry, smart_sl, sym_specs[sym], risk_pct)
            p = V4Position(
                sym=sym, direction=direction, entry=entry, sl=smart_sl,
                atr_val=atr_val, lot=lot, contract_size=sym_specs[sym]["contract_size"],
            )
            p.entry_time = time
            open_positions[sym] = p

    # Force close remaining
    for sym, p in open_positions.items():
        if not p.closed and sym in symbol_data:
            df = symbol_data[sym]
            p.force_close(float(df.iloc[-1]["close"]))
            p.exit_time = df.iloc[-1]["time"]
            usd_pnl = pnl_to_usd(p.total_pnl, sym, conv_rates)
            p._usd_pnl = usd_pnl
            balance += usd_pnl
            all_trades.append(p)

    if not all_trades:
        return None

    all_trades.sort(key=lambda t: t.exit_time if t.exit_time else pd.Timestamp.min)

    usd_pnls = [t._usd_pnl for t in all_trades]
    wins = [p for p in usd_pnls if p > 0]
    losses = [p for p in usd_pnls if p <= 0]
    total_pnl = sum(usd_pnls)
    win_rate = len(wins) / len(usd_pnls) * 100
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    peak = STARTING_BALANCE
    max_dd = 0
    max_dd_pct = 0
    running = STARTING_BALANCE
    for p in usd_pnls:
        running += p
        if running > peak:
            peak = running
        dd = peak - running
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    first_time = min(t.exit_time for t in all_trades if t.exit_time is not None)
    last_time = max(t.exit_time for t in all_trades if t.exit_time is not None)
    days_tested = (last_time - first_time).days

    # Time to $100K
    running = STARTING_BALANCE
    time_to_100k = None
    trade_to_100k = None
    for i, t in enumerate(all_trades):
        running += t._usd_pnl
        if running >= 100_000 and time_to_100k is None:
            time_to_100k = (t.exit_time - first_time).days if t.exit_time else None
            trade_to_100k = i + 1

    # Per-symbol breakdown
    sym_stats = {}
    for sym in sorted(set(t.sym for t in all_trades)):
        st = [t for t in all_trades if t.sym == sym]
        sp = [t._usd_pnl for t in st]
        sw = [p for p in sp if p > 0]
        sl = [p for p in sp if p <= 0]
        wr = len(sw) / len(sp) * 100 if sp else 0
        sym_stats[sym] = {"trades": len(sp), "wr": wr, "pnl": sum(sp)}

    # Trades per week
    trades_per_week = len(all_trades) / (days_tested / 7) if days_tested > 0 else 0

    # Annual return estimate (from first year only for fair comparison)
    year1_end = first_time + pd.Timedelta(days=365)
    year1_balance = STARTING_BALANCE
    for t in all_trades:
        if t.exit_time and t.exit_time <= year1_end:
            year1_balance += t._usd_pnl
    year1_return = (year1_balance - STARTING_BALANCE) / STARTING_BALANCE * 100

    return {
        "tf": tf_name,
        "trades": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "final_balance": balance,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "days_tested": days_tested,
        "trades_per_week": trades_per_week,
        "time_to_100k_days": time_to_100k,
        "trade_to_100k": trade_to_100k,
        "year1_balance": year1_balance,
        "year1_return": year1_return,
        "sym_stats": sym_stats,
        "all_trades": all_trades,
    }


def main():
    print("=" * 120)
    print("  MULTI-TIMEFRAME COMPARISON — V4 Profit Capture + Compounding")
    print(f"  Testing: H1, H2, H3, H4 | Risk: {BASE_RISK_PCT*100:.0f}% | Start: ${STARTING_BALANCE:.0f}")
    print("=" * 120)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    acct = mt5.account_info()
    if acct:
        print(f"  MT5: Account {acct.login} | Leverage 1:{acct.leverage}")

    sym_specs = get_symbol_specs()
    conv_rates = get_conversion_rates()
    print(f"  Rates: USDJPY={conv_rates.get('USDJPY', 'N/A'):.4f}, USDCAD={conv_rates.get('USDCAD', 'N/A'):.4f}")

    # Find common start date: H1 with FETCH_BARS has the least history
    # All TFs must start from the same date for fair comparison
    print(f"\n  Finding common start date across all timeframes...")
    latest_start = None
    for tf_name, tf_mt5 in TIMEFRAMES.items():
        resolved = sym_specs.get("EURUSD", {}).get("resolved", "EURUSD")
        rates = mt5.copy_rates_from_pos(resolved, tf_mt5, 0, FETCH_BARS)
        if rates is not None and len(rates) > 0:
            df_tmp = pd.DataFrame(rates)
            df_tmp["time"] = pd.to_datetime(df_tmp["time"], unit="s")
            tf_start = df_tmp["time"].iloc[0]
            print(f"    {tf_name}: data starts at {tf_start.strftime('%Y-%m-%d')} ({len(rates)} bars)")
            if latest_start is None or tf_start > latest_start:
                latest_start = tf_start

    # Use the latest start + some buffer for warmup
    common_start = latest_start + pd.Timedelta(days=7)  # 1 week buffer for ATR warmup
    print(f"\n  Common start date: {common_start.strftime('%Y-%m-%d')} (all TFs will trade from this date)")

    results = {}
    for tf_name in ["H1", "H2", "H3", "H4"]:
        tf_mt5 = TIMEFRAMES[tf_name]
        print(f"\n{'-' * 120}")
        print(f"  Running {tf_name} backtest (from {common_start.strftime('%Y-%m-%d')})...")
        r = run_backtest_for_tf(tf_name, tf_mt5, sym_specs, conv_rates, start_date=common_start)
        if r:
            results[tf_name] = r
            print(f"  {tf_name}: {r['trades']} trades | {r['win_rate']:.1f}% WR | PF {r['profit_factor']:.2f} | "
                  f"${STARTING_BALANCE:.0f} -> ${r['final_balance']:,.0f} | "
                  f"DD {r['max_dd_pct']:.1f}% | {r['trades_per_week']:.1f} trades/wk")
        else:
            print(f"  {tf_name}: FAILED (no data/trades)")

    mt5.shutdown()

    if not results:
        print("No results!")
        return

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 120}")

    headers = ["Metric"] + list(results.keys())
    print(f"\n  {'Metric':<25}", end="")
    for tf in results:
        print(f" | {tf:>18}", end="")
    print()
    print(f"  " + "-" * (25 + 21 * len(results)))

    rows = [
        ("Total Trades", lambda r: f"{r['trades']:,}"),
        ("Trades/Week", lambda r: f"{r['trades_per_week']:.1f}"),
        ("Win Rate", lambda r: f"{r['win_rate']:.1f}%"),
        ("Losers", lambda r: f"{r['losses']}"),
        ("Profit Factor", lambda r: f"{r['profit_factor']:.2f}"),
        ("Final Balance", lambda r: f"${r['final_balance']:>12,.0f}"),
        ("Total PnL", lambda r: f"${r['total_pnl']:>12,.0f}"),
        ("Max Drawdown $", lambda r: f"${r['max_dd']:>10,.0f}"),
        ("Max Drawdown %", lambda r: f"{r['max_dd_pct']:.1f}%"),
        ("Days Tested", lambda r: f"{r['days_tested']}"),
        ("Year 1 Balance", lambda r: f"${r['year1_balance']:>10,.0f}"),
        ("Year 1 Return", lambda r: f"{r['year1_return']:+,.0f}%"),
        ("Time to $100K", lambda r: f"{r['time_to_100k_days']} days" if r['time_to_100k_days'] else "N/A"),
        ("Trade# to $100K", lambda r: f"#{r['trade_to_100k']}" if r['trade_to_100k'] else "N/A"),
    ]

    for label, fmt in rows:
        print(f"  {label:<25}", end="")
        for tf in results:
            print(f" | {fmt(results[tf]):>18}", end="")
        print()

    # Per-symbol win rates by TF
    print(f"\n  PER-SYMBOL WIN RATES:")
    print(f"  {'Symbol':<10}", end="")
    for tf in results:
        print(f" | {tf + ' WR':>10} | {tf + ' #':>6}", end="")
    print()
    print(f"  " + "-" * (10 + 21 * len(results)))
    for sym in SYMBOLS:
        print(f"  {sym:<10}", end="")
        for tf in results:
            ss = results[tf]["sym_stats"].get(sym)
            if ss:
                print(f" | {ss['wr']:>8.1f}% | {ss['trades']:>6}", end="")
            else:
                print(f" | {'N/A':>9} | {'N/A':>6}", end="")
        print()

    # ═══════════════════════════════════════════════════════════════
    # RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print(f"  ANALYSIS & RECOMMENDATION")
    print(f"{'=' * 120}")

    # Find best TF by different criteria
    best_balance = max(results.values(), key=lambda r: r['final_balance'])
    best_wr = max(results.values(), key=lambda r: r['win_rate'])
    best_pf = max(results.values(), key=lambda r: r['profit_factor'])
    fastest_100k = min((r for r in results.values() if r['time_to_100k_days']),
                       key=lambda r: r['time_to_100k_days'], default=None)
    best_yr1 = max(results.values(), key=lambda r: r['year1_return'])

    print(f"\n  Best final balance:  {best_balance['tf']} (${best_balance['final_balance']:,.0f})")
    print(f"  Best win rate:       {best_wr['tf']} ({best_wr['win_rate']:.1f}%)")
    print(f"  Best profit factor:  {best_pf['tf']} ({best_pf['profit_factor']:.2f})")
    if fastest_100k:
        print(f"  Fastest to $100K:    {fastest_100k['tf']} ({fastest_100k['time_to_100k_days']} days)")
    print(f"  Best Year 1:         {best_yr1['tf']} (${best_yr1['year1_balance']:,.0f})")

    # $100K/year target analysis
    print(f"\n  $100K/YEAR TARGET ANALYSIS:")
    for tf, r in results.items():
        annual_pnl = r['total_pnl'] / (r['days_tested'] / 365) if r['days_tested'] > 0 else 0
        print(f"    {tf}: ~${annual_pnl:,.0f}/year average ({r['trades_per_week']:.1f} trades/wk)")

    print(f"\n{'=' * 120}")


if __name__ == "__main__":
    main()
