#!/usr/bin/env python3
"""
JPY / CAD PnL Currency Conversion Verification
===============================================
The main backtest calculates PnL as (price_diff * lot * contract_size), which
gives the result in the QUOTE currency (JPY for JPY pairs, CAD for USDCAD).
This script re-runs the V4 PROTECT backtest with proper conversion to USD.

Fix: divide JPY PnL by current USDJPY rate, CAD PnL by current USDCAD rate.
"""

import sys
import os
import numpy as np
import pandas as pd
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
FETCH_BARS = 5000
WARMUP_BARS = 50
LOT_SIZE = 0.10
STARTING_BALANCE = 200.0


# ═══════════════════════════════════════════════════════════════════
# Currency conversion
# ═══════════════════════════════════════════════════════════════════

def get_conversion_rates():
    """Get current USDJPY and USDCAD rates for PnL conversion."""
    rates = {}

    for pair in ["USDJPY", "USDCAD"]:
        resolved = resolve_symbol(pair)
        if resolved:
            tick = mt5.symbol_info_tick(resolved)
            if tick:
                rates[pair] = (tick.bid + tick.ask) / 2.0
                print(f"  Conversion rate {pair}: {rates[pair]:.4f}")

    return rates


def quote_currency(symbol):
    """Return the quote currency for a symbol."""
    sym = symbol.upper()
    if sym.endswith("JPY"):
        return "JPY"
    elif sym.endswith("CAD"):
        return "CAD"
    elif sym.endswith("USD"):
        return "USD"
    elif sym.endswith("GBP"):
        return "GBP"
    elif sym.endswith("CHF"):
        return "CHF"
    return "USD"


def pnl_to_usd(raw_pnl, symbol, conv_rates):
    """Convert raw PnL (in quote currency) to USD."""
    qc = quote_currency(symbol)
    if qc == "USD":
        return raw_pnl
    elif qc == "JPY":
        usdjpy = conv_rates.get("USDJPY", 152.0)
        return raw_pnl / usdjpy
    elif qc == "CAD":
        usdcad = conv_rates.get("USDCAD", 1.36)
        return raw_pnl / usdcad
    elif qc == "GBP":
        # EURGBP etc — would need GBPUSD rate
        # Not in our symbol list, but handle gracefully
        return raw_pnl * 1.27  # approximate
    elif qc == "CHF":
        return raw_pnl * 1.12  # approximate
    return raw_pnl


# ═══════════════════════════════════════════════════════════════════
# Helpers (same as main backtest)
# ═══════════════════════════════════════════════════════════════════

def pip_value(symbol):
    sym = symbol.upper()
    if "JPY" in sym:
        return 0.01
    return 0.0001


def contract_size(symbol):
    return 100_000.0


def resolve_symbol(ticker):
    for c in [ticker, ticker + ".raw", ticker + "m", ticker + ".a", ticker + ".e"]:
        info = mt5.symbol_info(c)
        if info is not None:
            mt5.symbol_select(c, True)
            return c
    return None


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


# ═══════════════════════════════════════════════════════════════════
# V4 Position (identical to main backtest)
# ═══════════════════════════════════════════════════════════════════

class V4Position:
    def __init__(self, sym, direction, entry, sl, atr_val, lot, pip_sz, cont_sz):
        self.sym = sym
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.original_sl = sl
        self.atr_val = atr_val
        self.total_lot = lot
        self.remaining_lot = lot
        self.pip_size = pip_sz
        self.contract_size = cont_sz
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

        # Post-TP1 trailing stop
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

        # Early breakeven
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

        # Stall exit
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

        # Micro-partial
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

        # TP1
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

        # TP2
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

        # TP3
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

        # Profit lock SL
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

        # SL
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

        # ZP flip
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

    def total_pnl_usd(self, conv_rates):
        """Total PnL converted to USD."""
        return pnl_to_usd(self.total_pnl, self.sym, conv_rates)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print("  JPY / CAD PnL CURRENCY CONVERSION VERIFICATION")
    print("  V4 PROTECT — Corrected USD PnL")
    print("=" * 100)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    acct = mt5.account_info()
    if acct:
        print(f"  MT5: Account {acct.login} | Leverage 1:{acct.leverage}")

    # Get conversion rates
    print(f"\n  Getting conversion rates...")
    conv_rates = get_conversion_rates()
    if "USDJPY" not in conv_rates:
        print("  WARNING: Could not get USDJPY rate, using 152.0")
        conv_rates["USDJPY"] = 152.0
    if "USDCAD" not in conv_rates:
        print("  WARNING: Could not get USDCAD rate, using 1.36")
        conv_rates["USDCAD"] = 1.36

    # Fetch data
    symbol_data = {}
    print(f"\n  Fetching up to {FETCH_BARS} H4 bars per symbol...")
    for sym in SYMBOLS:
        resolved = resolve_symbol(sym)
        if resolved is None:
            print(f"    {sym}: SKIP (cannot resolve)")
            continue
        rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H4, 0, FETCH_BARS)
        if rates is None or len(rates) < 100:
            print(f"    {sym}: SKIP ({len(rates) if rates is not None else 0} bars)")
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df_zp = compute_zeropoint_state(df)
        if df_zp is None or len(df_zp) < WARMUP_BARS:
            print(f"    {sym}: SKIP (ZP failed)")
            continue
        symbol_data[sym] = df_zp
        days = (df_zp["time"].iloc[-1] - df_zp["time"].iloc[WARMUP_BARS]).days
        print(f"    {sym}: {len(df_zp)} bars ({days} days)")

    if not symbol_data:
        print("ERROR: No data loaded")
        mt5.shutdown()
        return

    # Simulate trades
    print(f"\n  Simulating V4 PROTECT trades...")
    trades = []

    for sym, df in symbol_data.items():
        pip_sz = pip_value(sym)
        cont_sz = contract_size(sym)
        n = len(df)
        pos_obj = None

        for i in range(WARMUP_BARS, n):
            row = df.iloc[i]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            atr_val = float(row["atr"])
            pos = int(row.get("pos", 0))
            buy_sig = bool(row.get("buy_signal", False))
            sell_sig = bool(row.get("sell_signal", False))

            if np.isnan(atr_val) or atr_val <= 0:
                continue

            if pos_obj is not None and not pos_obj.closed:
                pos_obj.check_bar(high, low, close, pos)
                if pos_obj.closed:
                    pos_obj.exit_time = row["time"]
                    trades.append(pos_obj)
                    pos_obj = None

            if buy_sig or sell_sig:
                direction = "BUY" if buy_sig else "SELL"
                entry = close
                smart_sl = compute_smart_sl(df, i, direction, atr_val)

                if pos_obj is not None and not pos_obj.closed:
                    pos_obj.force_close(close)
                    pos_obj.exit_time = row["time"]
                    trades.append(pos_obj)

                pos_obj = V4Position(
                    sym=sym, direction=direction, entry=entry, sl=smart_sl,
                    atr_val=atr_val, lot=LOT_SIZE, pip_sz=pip_sz, cont_sz=cont_sz,
                )

        if pos_obj is not None and not pos_obj.closed:
            pos_obj.force_close(float(df.iloc[-1]["close"]))
            pos_obj.exit_time = df.iloc[-1]["time"]
            trades.append(pos_obj)

    mt5.shutdown()

    if not trades:
        print("No trades generated!")
        return

    # ═══════════════════════════════════════════════════════════════
    # Results: RAW (quote currency) vs CORRECTED (USD)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 120}")
    print(f"  RESULTS: RAW (quote currency) vs CORRECTED (USD)")
    print(f"{'=' * 120}")

    # Per-symbol comparison
    print(f"\n  {'Symbol':>8} | {'Quote':>5} | {'Trades':>7} | {'Win%':>6} | {'Raw PnL':>14} | {'USD PnL':>14} | {'Conversion':>12} | {'Divisor':>8}")
    print(f"  " + "-" * 100)

    total_raw = 0
    total_usd = 0
    all_usd_pnls = []

    for sym in sorted(set(t.sym for t in trades)):
        st = [t for t in trades if t.sym == sym]
        qc = quote_currency(sym)
        raw_pnl = sum(t.total_pnl for t in st)
        usd_pnl = sum(t.total_pnl_usd(conv_rates) for t in st)

        usd_pnls = [t.total_pnl_usd(conv_rates) for t in st]
        wins_usd = [p for p in usd_pnls if p > 0]
        wr = len(wins_usd) / len(usd_pnls) * 100

        if qc == "JPY":
            divisor = conv_rates.get("USDJPY", 152.0)
        elif qc == "CAD":
            divisor = conv_rates.get("USDCAD", 1.36)
        else:
            divisor = 1.0

        total_raw += raw_pnl
        total_usd += usd_pnl
        all_usd_pnls.extend(usd_pnls)

        # Show the raw currency label
        raw_label = f"¥{raw_pnl:+,.0f}" if qc == "JPY" else (f"C${raw_pnl:+,.2f}" if qc == "CAD" else f"${raw_pnl:+,.2f}")

        print(f"  {sym:>8} | {qc:>5} | {len(st):>7} | {wr:>5.1f}% | {raw_label:>14} | ${usd_pnl:>+12,.2f} | {'÷ ' + f'{divisor:.2f}' if divisor != 1.0 else 'none':>12} | {divisor:>8.2f}")

    # Totals
    wins_total = [p for p in all_usd_pnls if p > 0]
    losses_total = [p for p in all_usd_pnls if p <= 0]
    win_rate = len(wins_total) / len(all_usd_pnls) * 100
    avg_win = np.mean(wins_total) if wins_total else 0
    avg_loss = np.mean(losses_total) if losses_total else 0
    gross_profit = sum(wins_total)
    gross_loss = abs(sum(losses_total))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy = total_usd / len(all_usd_pnls)

    # Max drawdown (in USD)
    peak = STARTING_BALANCE
    max_dd = 0
    running = STARTING_BALANCE
    # Sort trades by exit time for proper DD calc
    trades_sorted = sorted(trades, key=lambda t: t.exit_time if t.exit_time else pd.Timestamp.min)
    for t in trades_sorted:
        running += t.total_pnl_usd(conv_rates)
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    final_balance = STARTING_BALANCE + total_usd

    print(f"\n{'=' * 120}")
    print(f"  CORRECTED SUMMARY (all values in USD)")
    print(f"{'=' * 120}")
    print(f"\n  Total trades:      {len(trades)}")
    print(f"  Winners:           {len(wins_total)} ({win_rate:.1f}%)")
    print(f"  Losers:            {len(losses_total)} ({100 - win_rate:.1f}%)")
    print(f"\n  Starting balance:  ${STARTING_BALANCE:.2f}")
    print(f"  Total PnL (USD):   ${total_usd:+,.2f}")
    print(f"  Final balance:     ${final_balance:+,.2f}")
    print(f"  Return:            {(total_usd / STARTING_BALANCE * 100):+,.1f}%")
    print(f"\n  Avg win (USD):     ${avg_win:+,.2f}")
    print(f"  Avg loss (USD):    ${avg_loss:+,.2f}")
    print(f"  Expectancy:        ${expectancy:+,.2f} per trade")
    print(f"  Profit factor:     {profit_factor:.2f}")
    print(f"  Max drawdown:      ${max_dd:,.2f}")

    # Impact of conversion
    print(f"\n{'=' * 120}")
    print(f"  CONVERSION IMPACT")
    print(f"{'=' * 120}")
    print(f"\n  Raw PnL (uncorrected):    ${total_raw:+,.2f}")
    print(f"  Corrected PnL (USD):      ${total_usd:+,.2f}")
    print(f"  Difference:               ${total_usd - total_raw:+,.2f}")
    print(f"  Raw was inflated by:      {total_raw / total_usd:.1f}x" if total_usd > 0 else "")

    # JPY-specific breakdown
    jpy_syms = ["EURJPY", "GBPJPY", "USDJPY"]
    jpy_trades = [t for t in trades if t.sym in jpy_syms]
    if jpy_trades:
        raw_jpy = sum(t.total_pnl for t in jpy_trades)
        usd_jpy = sum(t.total_pnl_usd(conv_rates) for t in jpy_trades)
        print(f"\n  JPY Pairs ({', '.join(jpy_syms)}):")
        print(f"    Raw PnL:       ¥{raw_jpy:+,.0f}")
        print(f"    USD PnL:       ${usd_jpy:+,.2f}")
        print(f"    Inflation:     {raw_jpy / usd_jpy:.1f}x" if usd_jpy != 0 else "")

    # CAD-specific breakdown
    cad_trades = [t for t in trades if t.sym == "USDCAD"]
    if cad_trades:
        raw_cad = sum(t.total_pnl for t in cad_trades)
        usd_cad = sum(t.total_pnl_usd(conv_rates) for t in cad_trades)
        print(f"\n  USDCAD:")
        print(f"    Raw PnL:       C${raw_cad:+,.2f}")
        print(f"    USD PnL:       ${usd_cad:+,.2f}")
        print(f"    Inflation:     {raw_cad / usd_cad:.1f}x" if usd_cad != 0 else "")

    # USD pairs (should be identical)
    usd_syms = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD"]
    usd_trades = [t for t in trades if t.sym in usd_syms]
    if usd_trades:
        raw_usd = sum(t.total_pnl for t in usd_trades)
        usd_usd = sum(t.total_pnl_usd(conv_rates) for t in usd_trades)
        print(f"\n  USD-quote Pairs ({', '.join(usd_syms)}):")
        print(f"    Raw PnL:       ${raw_usd:+,.2f}")
        print(f"    USD PnL:       ${usd_usd:+,.2f}")
        print(f"    Match:         {'YES' if abs(raw_usd - usd_usd) < 0.01 else 'NO'}")

    # Per-trade losers (corrected)
    losers_usd = [(t, t.total_pnl_usd(conv_rates)) for t in trades if t.total_pnl_usd(conv_rates) <= 0]
    if losers_usd:
        print(f"\n{'=' * 120}")
        print(f"  ALL {len(losers_usd)} LOSING TRADES (corrected USD)")
        print(f"{'=' * 120}")
        print(f"  {'#':>3} | {'Symbol':>8} | {'Dir':>4} | {'Exit':>12} | {'Raw PnL':>12} | {'USD PnL':>12} | {'Bars':>5} | {'MFE(ATR)':>9}")
        print(f"  " + "-" * 85)
        total_loss_usd = 0
        for idx, (t, usd_p) in enumerate(sorted(losers_usd, key=lambda x: x[1]), 1):
            mfe_atr = t.max_profit_reached / t.atr_val if t.atr_val > 0 else 0
            qc = quote_currency(t.sym)
            raw_str = f"¥{t.total_pnl:+,.0f}" if qc == "JPY" else f"${t.total_pnl:+,.2f}"
            total_loss_usd += usd_p
            print(f"  {idx:>3} | {t.sym:>8} | {t.direction:>4} | {t.final_exit_type:>12} | {raw_str:>12} | ${usd_p:>+10.2f} | {t.bars_in_trade:>5} | {mfe_atr:>8.3f}")
        print(f"\n  Total losses (USD): ${total_loss_usd:+,.2f}")
        print(f"  Total losses (raw): ${sum(t.total_pnl for t, _ in losers_usd):+,.2f}")

    print(f"\n{'=' * 120}")
    print(f"  VERDICT")
    print(f"{'=' * 120}")
    print(f"  Win rate is UNCHANGED: {win_rate:.1f}% (win/loss determination same regardless of currency)")
    print(f"  Dollar amounts were inflated for JPY pairs by ~{conv_rates.get('USDJPY', 152):.0f}x and USDCAD by ~{conv_rates.get('USDCAD', 1.36):.2f}x")
    print(f"  Corrected final balance: ${final_balance:,.2f} (from ${STARTING_BALANCE:.2f})")
    print(f"  Corrected return: {(total_usd / STARTING_BALANCE * 100):+,.1f}%")
    print(f"  Corrected profit factor: {profit_factor:.2f}")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
