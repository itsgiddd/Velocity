#!/usr/bin/env python3
"""
Repeatability Test — Does V4 work across different time periods?
================================================================
Splits history into non-overlapping windows and runs each independently.
If performance is consistent across windows, the edge is real.
If it only works in one window, it's curve-fitting.

Tests H3 and H4 (the main candidates) with 40% risk, 100 lot ECN cap.
Uses FLAT lot sizing (0.10) to isolate signal quality from compounding effects.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from collections import defaultdict

from app.zeropoint_signal import (
    compute_zeropoint_state,
    TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,
    SL_BUFFER_PCT, SL_ATR_MIN_MULT, SWING_LOOKBACK,
    BE_TRIGGER_MULT, BE_BUFFER_MULT,
    PROFIT_TRAIL_DISTANCE_MULT, STALL_BARS,
    MICRO_TP_MULT, MICRO_TP_PCT,
)

SYMBOLS = ["AUDUSD", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDJPY"]
LOT_SIZE = 0.10  # FLAT lot — isolates signal quality from compounding
WARMUP_BARS = 50


def resolve_symbol(ticker):
    for c in [ticker, ticker + ".raw", ticker + "m", ticker + ".a", ticker + ".e"]:
        info = mt5.symbol_info(c)
        if info is not None:
            mt5.symbol_select(c, True)
            return c
    return None


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


def contract_size(symbol):
    return 100_000.0


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
    def __init__(self, sym, direction, entry, sl, atr_val, lot, cont_sz):
        self.sym = sym
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.original_sl = sl
        self.atr_val = atr_val
        self.total_lot = lot
        self.remaining_lot = lot
        self.contract_size = cont_sz
        sign = 1 if direction == "BUY" else -1
        self.tp1 = entry + sign * TP1_MULT_AGG * atr_val
        self.tp2 = entry + sign * TP2_MULT_AGG * atr_val
        self.tp3 = entry + sign * TP3_MULT_AGG * atr_val
        self.tp1_hit = self.tp2_hit = self.tp3_hit = False
        self.closed = False
        self.partials = []
        self.bars_in_trade = 0
        self.exit_time = None
        self.max_profit_reached = 0.0
        self.max_favorable_price = entry
        self.sl_hit = False
        self.final_exit_type = None
        self.be_activated = False
        self.profit_lock_sl = None
        self.profit_lock_active = False
        self.micro_tp_hit = False
        self.stall_be_activated = False

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
            if (is_buy and high >= micro_price) or (not is_buy and low <= micro_price):
                self.micro_tp_hit = True
                micro_lot = max(0.01, round(self.total_lot * MICRO_TP_PCT, 2))
                micro_lot = min(micro_lot, self.remaining_lot)
                pnl = self.pnl_for_price(micro_price, micro_lot)
                self.partials.append((micro_price, micro_lot, pnl, 'MICRO_TP'))
                self.remaining_lot = round(self.remaining_lot - micro_lot, 2)
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
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'TP2'
                    return events

        if self.tp2_hit and not self.tp3_hit:
            if (is_buy and high >= self.tp3) or (not is_buy and low <= self.tp3):
                self.tp3_hit = True
                pnl = self.pnl_for_price(self.tp3, self.remaining_lot)
                self.partials.append((self.tp3, self.remaining_lot, pnl, 'TP3'))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'TP3'
                return events

        if self.profit_lock_active and self.profit_lock_sl is not None:
            lock_hit = (is_buy and low <= self.profit_lock_sl) or (not is_buy and high >= self.profit_lock_sl)
            if lock_hit:
                pnl = self.pnl_for_price(self.profit_lock_sl, self.remaining_lot)
                self.partials.append((self.profit_lock_sl, self.remaining_lot, pnl, 'PROFIT_LOCK'))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'PROFIT_LOCK'
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
            return events

        if confirmed_pos != 0:
            if (is_buy and confirmed_pos == -1) or (not is_buy and confirmed_pos == 1):
                pnl = self.pnl_for_price(close, self.remaining_lot)
                self.partials.append((close, self.remaining_lot, pnl, 'ZP_FLIP'))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'ZP_FLIP'
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


def run_window(tf_mt5, sym_resolved, conv_rates, start_dt, end_dt):
    """Run flat-lot V4 backtest on a specific date window. Returns stats dict."""
    symbol_data = {}
    for sym in SYMBOLS:
        resolved = sym_resolved.get(sym)
        if not resolved:
            continue
        # Fetch all available data, then slice
        rates = mt5.copy_rates_from_pos(resolved, tf_mt5, 0, 50000)
        if rates is None or len(rates) < 100:
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        # We need warmup bars BEFORE start_dt
        warmup_fetch = 200
        start_mask = df["time"] >= pd.Timestamp(start_dt)
        if not start_mask.any():
            continue
        first_idx = start_mask.idxmax()
        warmup_start = max(0, first_idx - warmup_fetch)

        end_mask = df["time"] <= pd.Timestamp(end_dt)
        if not end_mask.any():
            continue
        last_idx = end_mask[::-1].idxmax()

        df_slice = df.iloc[warmup_start:last_idx + 1].reset_index(drop=True)
        if len(df_slice) < WARMUP_BARS + 10:
            continue

        df_zp = compute_zeropoint_state(df_slice)
        if df_zp is None or len(df_zp) < WARMUP_BARS:
            continue

        # Mark where to start trading
        trade_mask = df_zp["time"] >= pd.Timestamp(start_dt)
        if not trade_mask.any():
            continue
        df_zp.attrs["trade_start_idx"] = trade_mask.idxmax()
        symbol_data[sym] = df_zp

    if not symbol_data:
        return None

    # Build bar stream
    bar_events = []
    for sym, df in symbol_data.items():
        signal_start = max(WARMUP_BARS, df.attrs.get("trade_start_idx", WARMUP_BARS))
        for i in range(signal_start, len(df)):
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

    trades = []
    open_positions = {}

    for time, sym, high, low, close, atr_val, pos, buy_sig, sell_sig, bar_idx in bar_events:
        if sym in open_positions:
            p = open_positions[sym]
            if not p.closed:
                p.check_bar(high, low, close, pos)
                if p.closed:
                    p.exit_time = time
                    p._usd_pnl = pnl_to_usd(p.total_pnl, sym, conv_rates)
                    trades.append(p)
                    del open_positions[sym]

        if buy_sig or sell_sig:
            direction = "BUY" if buy_sig else "SELL"
            entry = close
            smart_sl = compute_smart_sl(symbol_data[sym], bar_idx, direction, atr_val)

            if sym in open_positions and not open_positions[sym].closed:
                p = open_positions[sym]
                p.force_close(close)
                p.exit_time = time
                p._usd_pnl = pnl_to_usd(p.total_pnl, sym, conv_rates)
                trades.append(p)

            p = V4Position(sym, direction, entry, smart_sl, atr_val, LOT_SIZE, contract_size(sym))
            p.entry_time = time
            open_positions[sym] = p

    # Force close remaining
    for sym, p in open_positions.items():
        if not p.closed and sym in symbol_data:
            df = symbol_data[sym]
            p.force_close(float(df.iloc[-1]["close"]))
            p.exit_time = df.iloc[-1]["time"]
            p._usd_pnl = pnl_to_usd(p.total_pnl, sym, conv_rates)
            trades.append(p)

    if not trades:
        return None

    usd_pnls = [t._usd_pnl for t in trades]
    wins = [p for p in usd_pnls if p > 0]
    losses = [p for p in usd_pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "total_pnl": sum(usd_pnls),
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        "expectancy": sum(usd_pnls) / len(trades) if trades else 0,
    }


def main():
    print("=" * 120)
    print("  REPEATABILITY TEST -- V4 Profit Capture across different time periods")
    print("  Flat 0.10 lots (isolates signal quality from compounding)")
    print("  All PnL converted to USD")
    print("=" * 120)

    if not mt5.initialize():
        print("ERROR: Could not init MT5")
        return

    conv_rates = get_conversion_rates()
    print(f"  Rates: USDJPY={conv_rates.get('USDJPY', 'N/A'):.4f}, USDCAD={conv_rates.get('USDCAD', 'N/A'):.4f}")

    # Resolve symbols once
    sym_resolved = {}
    for sym in SYMBOLS:
        r = resolve_symbol(sym)
        if r:
            sym_resolved[sym] = r

    # Define non-overlapping 6-month windows
    windows = [
        ("2019-H2", "2019-07-01", "2019-12-31"),
        ("2020-H1", "2020-01-01", "2020-06-30"),
        ("2020-H2", "2020-07-01", "2020-12-31"),
        ("2021-H1", "2021-01-01", "2021-06-30"),
        ("2021-H2", "2021-07-01", "2021-12-31"),
        ("2022-H1", "2022-01-01", "2022-06-30"),
        ("2022-H2", "2022-07-01", "2022-12-31"),
        ("2023-H1", "2023-01-01", "2023-06-30"),
        ("2023-H2", "2023-07-01", "2023-12-31"),
        ("2024-H1", "2024-01-01", "2024-06-30"),
        ("2024-H2", "2024-07-01", "2024-12-31"),
        ("2025-H1", "2025-01-01", "2025-06-30"),
        ("2025-H2", "2025-07-01", "2025-12-31"),
        ("2026-H1", "2026-01-01", "2026-06-30"),
    ]

    for tf_name, tf_mt5 in [("H3", mt5.TIMEFRAME_H3), ("H4", mt5.TIMEFRAME_H4)]:
        print(f"\n{'=' * 120}")
        print(f"  {tf_name} TIMEFRAME")
        print(f"{'=' * 120}")
        print(f"  {'Window':>10} | {'Trades':>7} | {'Wins':>5} | {'Losses':>6} | {'Win%':>6} | {'PF':>6} | {'USD PnL':>12} | {'Avg Win':>9} | {'Avg Loss':>10} | {'Expect':>8}")
        print("  " + "-" * 108)

        all_wr = []
        all_pf = []
        all_trades = 0
        all_wins = 0
        all_losses = 0
        all_pnl = 0

        for label, start, end in windows:
            r = run_window(tf_mt5, sym_resolved, conv_rates, start, end)
            if r and r["trades"] > 0:
                print(f"  {label:>10} | {r['trades']:>7} | {r['wins']:>5} | {r['losses']:>6} | "
                      f"{r['win_rate']:>5.1f}% | {r['profit_factor']:>5.2f} | "
                      f"${r['total_pnl']:>+10.2f} | ${r['avg_win']:>7.2f} | "
                      f"${r['avg_loss']:>8.2f} | ${r['expectancy']:>6.2f}")
                all_wr.append(r["win_rate"])
                all_pf.append(r["profit_factor"])
                all_trades += r["trades"]
                all_wins += r["wins"]
                all_losses += r["losses"]
                all_pnl += r["total_pnl"]
            else:
                print(f"  {label:>10} | {'NO DATA':>7} |")

        if all_wr:
            print("  " + "-" * 108)
            total_wr = all_wins / all_trades * 100 if all_trades > 0 else 0
            print(f"  {'TOTAL':>10} | {all_trades:>7} | {all_wins:>5} | {all_losses:>6} | "
                  f"{total_wr:>5.1f}% |       | ${all_pnl:>+10.2f}")
            print(f"\n  Consistency metrics:")
            print(f"    Win rate range:  {min(all_wr):.1f}% - {max(all_wr):.1f}%")
            print(f"    Win rate mean:   {np.mean(all_wr):.1f}%")
            print(f"    Win rate std:    {np.std(all_wr):.1f}%")
            print(f"    PF range:        {min(all_pf):.2f} - {max(all_pf):.2f}")
            print(f"    Windows tested:  {len(all_wr)}")
            profitable_windows = sum(1 for p in all_pf if p > 1.0)
            print(f"    Profitable windows: {profitable_windows}/{len(all_pf)} ({profitable_windows/len(all_pf)*100:.0f}%)")

            # Is it repeatable?
            min_wr = min(all_wr)
            if min_wr >= 95:
                print(f"\n    VERDICT: HIGHLY REPEATABLE - worst window still {min_wr:.1f}% WR")
            elif min_wr >= 90:
                print(f"\n    VERDICT: REPEATABLE - worst window {min_wr:.1f}% WR (some variation)")
            elif min_wr >= 80:
                print(f"\n    VERDICT: SOMEWHAT REPEATABLE - worst window {min_wr:.1f}% WR (concerning)")
            else:
                print(f"\n    VERDICT: NOT REPEATABLE - worst window {min_wr:.1f}% WR (edge may not exist)")

    mt5.shutdown()
    print(f"\n{'=' * 120}")


if __name__ == "__main__":
    main()
