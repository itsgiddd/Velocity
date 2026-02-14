#!/usr/bin/env python3
"""
REALISTIC 1-YEAR PROJECTION -- V4 ZeroPoint from $200
=====================================================
Uses actual backtest trade-by-trade R-multiples (PnL / risk_amount)
to project growth WITHOUT lot cap distortions.

Approach: For each trade in the last year, compute its R-multiple
(how many risk units it returned). Then compound from $200 using
30% risk * R-multiple for each trade.

This is the MOST ACCURATE projection because:
  - R-multiples are lot-size independent
  - Compounding at 30% risk is what you'll actually do
  - No lot cap artifacts
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import MetaTrader5 as mt5
from app.zeropoint_signal import (
    compute_zeropoint_state,
    BE_TRIGGER_MULT, BE_BUFFER_MULT, PROFIT_TRAIL_DISTANCE_MULT,
    STALL_BARS, MICRO_TP_MULT, MICRO_TP_PCT,
    TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,
    SL_BUFFER_PCT, SL_ATR_MIN_MULT, SWING_LOOKBACK,
)

SYMBOLS = ["AUDUSD", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDJPY"]
WARMUP = 50
LOT = 1.0  # Fixed lot for R-multiple calculation


def contract_size(sym):
    return 1.0 if "BTC" in sym.upper() else 100_000.0


def resolve_symbol(ticker):
    for c in [ticker, ticker + ".raw", ticker + "m", ticker + ".a", ticker + ".e"]:
        info = mt5.symbol_info(c)
        if info is not None:
            mt5.symbol_select(c, True)
            return c, info
    return None, None


def compute_smart_sl(df, bar_idx, direction, atr_val):
    lookback_start = max(0, bar_idx - SWING_LOOKBACK + 1)
    cur_close = float(df["close"].iloc[bar_idx])
    if direction == "BUY":
        recent = float(df["low"].iloc[lookback_start:bar_idx + 1].min())
        buf = recent * SL_BUFFER_PCT
        struct = recent - buf
        minsl = cur_close - atr_val * SL_ATR_MIN_MULT
        return minsl if struct > minsl else struct
    else:
        recent = float(df["high"].iloc[lookback_start:bar_idx + 1].max())
        buf = recent * SL_BUFFER_PCT
        struct = recent + buf
        minsl = cur_close + atr_val * SL_ATR_MIN_MULT
        return minsl if struct < minsl else struct


class V4Pos:
    def __init__(self, sym, d, entry, sl, atr, lot, cs, t=None):
        self.sym, self.direction, self.entry, self.sl = sym, d, entry, sl
        self.atr_val, self.total_lot, self.remaining_lot, self.cs = atr, lot, lot, cs
        self.entry_time, self.exit_time = t, None
        sign = 1 if d == "BUY" else -1
        self.tp1 = entry + sign * TP1_MULT_AGG * atr
        self.tp2 = entry + sign * TP2_MULT_AGG * atr
        self.tp3 = entry + sign * TP3_MULT_AGG * atr
        self.tp1_hit = self.tp2_hit = self.tp3_hit = False
        self.closed = False
        self.partials = []
        self.bars_in_trade = 0
        self.max_profit_reached = 0.0
        self.max_favorable_price = entry
        self.be_activated = False
        self.profit_lock_sl = None
        self.profit_lock_active = False
        self.micro_tp_hit = self.stall_be_activated = False
        self.exit_type = None
        # SL distance in dollars (for R-multiple calc)
        self.sl_distance_dollar = abs(entry - sl) * lot * cs

    def partial_lot(self):
        return max(0.01, round(self.total_lot / 3, 2))

    def pnl_for(self, price, lot):
        return ((price - self.entry) if self.direction == "BUY" else (self.entry - price)) * lot * self.cs

    def check_bar(self, h, l, c, pos):
        if self.closed: return
        self.bars_in_trade += 1
        buy = self.direction == "BUY"
        atr = self.atr_val
        if buy:
            if h > self.max_favorable_price: self.max_favorable_price = h
            cp = h - self.entry
        else:
            if l < self.max_favorable_price: self.max_favorable_price = l
            cp = self.entry - l
        if cp > self.max_profit_reached: self.max_profit_reached = cp

        if self.tp1_hit:
            td = PROFIT_TRAIL_DISTANCE_MULT * atr
            if buy:
                nl = self.max_favorable_price - td
                if nl > self.entry and (self.profit_lock_sl is None or nl > self.profit_lock_sl):
                    self.profit_lock_sl = nl; self.profit_lock_active = True
            else:
                nl = self.max_favorable_price + td
                if nl < self.entry and (self.profit_lock_sl is None or nl < self.profit_lock_sl):
                    self.profit_lock_sl = nl; self.profit_lock_active = True

        if not self.be_activated and self.max_profit_reached >= BE_TRIGGER_MULT * atr:
            bb = BE_BUFFER_MULT * atr
            ns = (self.entry + bb) if buy else (self.entry - bb)
            if (buy and ns > self.sl) or (not buy and ns < self.sl):
                self.sl = ns; self.be_activated = True

        if not self.tp1_hit and not self.stall_be_activated and self.bars_in_trade >= STALL_BARS:
            bb = BE_BUFFER_MULT * atr
            ns = (self.entry + bb) if buy else (self.entry - bb)
            if (buy and ns > self.sl) or (not buy and ns < self.sl):
                self.sl = ns; self.stall_be_activated = True; self.be_activated = True

        if not self.micro_tp_hit and not self.tp1_hit:
            mp = self.entry + MICRO_TP_MULT * atr if buy else self.entry - MICRO_TP_MULT * atr
            if (buy and h >= mp) or (not buy and l <= mp):
                self.micro_tp_hit = True
                ml = max(0.01, min(round(self.total_lot * MICRO_TP_PCT, 2), self.remaining_lot))
                self.partials.append(self.pnl_for(mp, ml))
                self.remaining_lot = round(self.remaining_lot - ml, 2)
                if self.remaining_lot <= 0: self.closed = True; self.exit_type = "MICRO_TP"; return

        if not self.tp1_hit:
            if (buy and h >= self.tp1) or (not buy and l <= self.tp1):
                self.tp1_hit = True
                p = min(self.partial_lot(), self.remaining_lot)
                self.partials.append(self.pnl_for(self.tp1, p))
                self.remaining_lot = round(self.remaining_lot - p, 2)
                if self.remaining_lot <= 0: self.closed = True; self.exit_type = "TP1"; return

        if self.tp1_hit and not self.tp2_hit:
            if (buy and h >= self.tp2) or (not buy and l <= self.tp2):
                self.tp2_hit = True; self.sl = self.entry; self.be_activated = True
                p = min(self.partial_lot(), self.remaining_lot)
                self.partials.append(self.pnl_for(self.tp2, p))
                self.remaining_lot = round(self.remaining_lot - p, 2)
                if self.remaining_lot <= 0: self.closed = True; self.exit_type = "TP2"; return

        if self.tp2_hit and not self.tp3_hit:
            if (buy and h >= self.tp3) or (not buy and l <= self.tp3):
                self.tp3_hit = True
                self.partials.append(self.pnl_for(self.tp3, self.remaining_lot))
                self.remaining_lot = 0; self.closed = True; self.exit_type = "TP3"; return

        if self.profit_lock_active and self.profit_lock_sl is not None:
            if (buy and l <= self.profit_lock_sl) or (not buy and h >= self.profit_lock_sl):
                self.partials.append(self.pnl_for(self.profit_lock_sl, self.remaining_lot))
                self.remaining_lot = 0; self.closed = True; self.exit_type = "PROFIT_LOCK"; return

        if (buy and l <= self.sl) or (not buy and h >= self.sl):
            self.partials.append(self.pnl_for(self.sl, self.remaining_lot))
            self.remaining_lot = 0; self.closed = True
            self.exit_type = "SL_STALL" if self.stall_be_activated else ("SL_BE" if self.be_activated else "SL")
            return

        if pos != 0 and ((buy and pos == -1) or (not buy and pos == 1)):
            self.partials.append(self.pnl_for(c, self.remaining_lot))
            self.remaining_lot = 0; self.closed = True; self.exit_type = "ZP_FLIP"; return

    def force_close(self, price):
        if self.closed or self.remaining_lot <= 0: return
        self.partials.append(self.pnl_for(price, self.remaining_lot))
        self.remaining_lot = 0; self.closed = True; self.exit_type = "END"

    @property
    def total_pnl(self):
        return sum(self.partials)

    @property
    def r_multiple(self):
        """PnL as multiple of SL risk. +1R = won exactly your risk amount."""
        if self.sl_distance_dollar > 0:
            return self.total_pnl / self.sl_distance_dollar
        return 0


def main():
    print("=" * 90)
    print("  REALISTIC 1-YEAR PROJECTION -- V4 ZeroPoint from $200")
    print("  Using R-multiples (lot-size independent) + compounding")
    print("=" * 90)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5"); return

    symbol_data = {}
    sym_info_map = {}
    print("\nLoading H4 data...")
    for sym in SYMBOLS:
        resolved, info = resolve_symbol(sym)
        if resolved is None: continue
        rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H4, 0, 5000)
        if rates is None or len(rates) < 100: continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df_zp = compute_zeropoint_state(df)
        if df_zp is not None and len(df_zp) >= WARMUP:
            symbol_data[sym] = df_zp
            sym_info_map[sym] = info
    mt5.shutdown()

    # Run fixed-lot simulation to get R-multiples
    print("\nRunning fixed-lot simulation to extract R-multiples...")
    events = []
    for sym, df in symbol_data.items():
        cs = contract_size(sym)
        for i in range(WARMUP, len(df)):
            row = df.iloc[i]
            events.append({
                "time": row["time"], "sym": sym, "idx": i,
                "high": float(row["high"]), "low": float(row["low"]),
                "close": float(row["close"]), "atr": float(row["atr"]),
                "pos": int(row.get("pos", 0)),
                "buy_signal": bool(row.get("buy_signal", False)),
                "sell_signal": bool(row.get("sell_signal", False)),
                "cs": cs,
            })
    events.sort(key=lambda e: (e["time"], e["sym"]))

    last_time = max(e["time"] for e in events)
    one_year_ago = last_time - pd.Timedelta(days=365)

    open_positions = {}
    all_trades = []

    for ev in events:
        sym = ev["sym"]
        h, l, c, atr, cp = ev["high"], ev["low"], ev["close"], ev["atr"], ev["pos"]
        bs, ss = ev["buy_signal"], ev["sell_signal"]
        if np.isnan(atr) or atr <= 0: continue

        if sym in open_positions and not open_positions[sym].closed:
            pos_obj = open_positions[sym]
            pos_obj.check_bar(h, l, c, cp)
            if pos_obj.closed:
                pos_obj.exit_time = ev["time"]
                all_trades.append(pos_obj)
                del open_positions[sym]

        if bs or ss:
            d = "BUY" if bs else "SELL"
            if sym in open_positions and not open_positions[sym].closed:
                pos_obj = open_positions[sym]
                pos_obj.force_close(c)
                pos_obj.exit_time = ev["time"]
                all_trades.append(pos_obj)

            df = symbol_data[sym]
            smart_sl = compute_smart_sl(df, ev["idx"], d, atr)
            pos_obj = V4Pos(sym=sym, d=d, entry=c, sl=smart_sl, atr=atr, lot=LOT, cs=ev["cs"], t=ev["time"])
            open_positions[sym] = pos_obj

    for sym, pos_obj in open_positions.items():
        if not pos_obj.closed:
            pos_obj.force_close(float(symbol_data[sym].iloc[-1]["close"]))
            all_trades.append(pos_obj)

    # Filter to last year
    year_trades = [t for t in all_trades if t.entry_time >= one_year_ago]
    year_trades.sort(key=lambda t: t.entry_time)

    print(f"\n  Last year: {one_year_ago.date()} to {last_time.date()}")
    print(f"  Trades in last year: {len(year_trades)}")

    # R-multiple analysis
    r_multiples = [t.r_multiple for t in year_trades]
    wins_r = [r for r in r_multiples if r > 0]
    losses_r = [r for r in r_multiples if r <= 0]

    print(f"\n  R-Multiple Distribution:")
    print(f"  Winners: {len(wins_r)}  avg R = +{np.mean(wins_r):.3f}")
    print(f"  Losers:  {len(losses_r)}  avg R = {np.mean(losses_r):.3f}" if losses_r else "  Losers:  0")
    print(f"  Overall: avg R = {np.mean(r_multiples):+.3f}")

    # Show R distribution
    print(f"\n  R-Multiple Buckets:")
    buckets = [(-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.02), (0.02, 0.05),
               (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1.0), (1.0, 999)]
    for lo, hi in buckets:
        count = sum(1 for r in r_multiples if lo <= r < hi)
        if count > 0:
            label = f"  {lo:+.2f}R to {hi:+.2f}R" if hi < 999 else f"  {lo:+.2f}R+"
            print(f"  {label:<22} {count:>4} trades ({count/len(r_multiples)*100:.1f}%)")

    # ── Compound projection at different risk levels ──
    print(f"\n{'='*90}")
    print(f"  COMPOUND GROWTH PROJECTION FROM $200")
    print(f"  Each trade: balance changes by (risk% * balance * R_multiple)")
    print(f"{'='*90}")

    for risk_pct in [0.10, 0.20, 0.30]:
        balance = 200.0
        peak = 200.0
        max_dd_pct = 0
        monthly = {}
        week_bals = []

        for i, t in enumerate(year_trades):
            r = t.r_multiple
            pnl = balance * risk_pct * r
            balance += pnl

            # Prevent going below $10 (account blown)
            if balance < 10:
                balance = 10  # effectively blown

            if balance > peak:
                peak = balance
            dd_pct = (peak - balance) / peak * 100 if peak > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

            # Monthly tracking
            month_key = (t.entry_time.year, t.entry_time.month)
            monthly[month_key] = balance

        print(f"\n  --- Risk: {risk_pct*100:.0f}% per trade ---")
        print(f"  Starting:     $200")
        print(f"  Final:        ${balance:>14,.2f}")
        print(f"  Profit:       ${balance - 200:>14,.2f}")
        print(f"  Return:       {(balance/200 - 1)*100:>14,.1f}%")
        print(f"  Max Drawdown: {max_dd_pct:>14.1f}%")

        print(f"\n  Monthly Growth:")
        prev_bal = 200.0
        for mk in sorted(monthly.keys()):
            yr, mo = mk
            bal = monthly[mk]
            growth = (bal / prev_bal - 1) * 100 if prev_bal > 0 else 0
            print(f"    {yr}-{mo:02d}  ${bal:>14,.2f}  ({growth:>+7.1f}%)")
            prev_bal = bal

    # ── With slippage/spread discount ──
    print(f"\n{'='*90}")
    print(f"  WITH REAL-WORLD FRICTION (slippage, spread, missed signals)")
    print(f"  Reduce each R-multiple by 0.02R (approx 2-pip slippage per trade)")
    print(f"{'='*90}")

    slippage_r = 0.02  # ~2 pips equivalent friction

    for risk_pct in [0.30]:
        balance = 200.0
        peak = 200.0
        max_dd_pct = 0
        monthly = {}

        for t in year_trades:
            r = t.r_multiple - slippage_r  # Subtract friction
            pnl = balance * risk_pct * r
            balance += pnl
            if balance < 10: balance = 10

            if balance > peak: peak = balance
            dd_pct = (peak - balance) / peak * 100 if peak > 0 else 0
            if dd_pct > max_dd_pct: max_dd_pct = dd_pct

            month_key = (t.entry_time.year, t.entry_time.month)
            monthly[month_key] = balance

        print(f"\n  --- Risk: {risk_pct*100:.0f}% with 0.02R friction ---")
        print(f"  Starting:     $200")
        print(f"  Final:        ${balance:>14,.2f}")
        print(f"  Profit:       ${balance - 200:>14,.2f}")
        print(f"  Return:       {(balance/200 - 1)*100:>14,.1f}%")
        print(f"  Max Drawdown: {max_dd_pct:>14.1f}%")

        print(f"\n  Monthly Growth:")
        prev_bal = 200.0
        for mk in sorted(monthly.keys()):
            yr, mo = mk
            bal = monthly[mk]
            growth = (bal / prev_bal - 1) * 100 if prev_bal > 0 else 0
            print(f"    {yr}-{mo:02d}  ${bal:>14,.2f}  ({growth:>+7.1f}%)")
            prev_bal = bal


if __name__ == "__main__":
    main()
