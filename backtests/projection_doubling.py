#!/usr/bin/env python3
"""
DOUBLING MILESTONE PROJECTION -- V4 ZeroPoint from $200
========================================================
Shows exactly when the account doubles, triples, etc.
Trade-by-trade compounding at 30% risk using actual R-multiples
from the last year of backtest data.

Also shows: how many trades to double, and the effective
doubling period in calendar days.
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
LOT = 1.0


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
        if self.sl_distance_dollar > 0:
            return self.total_pnl / self.sl_distance_dollar
        return 0


def main():
    print("=" * 90)
    print("  DOUBLING MILESTONES -- V4 ZeroPoint from $200")
    print("  Trade-by-trade compounding at 30% risk")
    print("=" * 90)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5"); return

    symbol_data = {}
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
    mt5.shutdown()

    # Build events
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

    # Get R-multiples for last year
    open_positions = {}
    year_trades = []

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
                if pos_obj.entry_time >= one_year_ago:
                    year_trades.append(pos_obj)
                del open_positions[sym]

        if bs or ss:
            d = "BUY" if bs else "SELL"
            if sym in open_positions and not open_positions[sym].closed:
                pos_obj = open_positions[sym]
                pos_obj.force_close(c)
                pos_obj.exit_time = ev["time"]
                if pos_obj.entry_time >= one_year_ago:
                    year_trades.append(pos_obj)

            df = symbol_data[sym]
            smart_sl = compute_smart_sl(df, ev["idx"], d, atr)
            pos_obj = V4Pos(sym=sym, d=d, entry=c, sl=smart_sl, atr=atr, lot=LOT, cs=ev["cs"], t=ev["time"])
            open_positions[sym] = pos_obj

    for sym, pos_obj in open_positions.items():
        if not pos_obj.closed:
            pos_obj.force_close(float(symbol_data[sym].iloc[-1]["close"]))
            if pos_obj.entry_time >= one_year_ago:
                year_trades.append(pos_obj)

    year_trades.sort(key=lambda t: t.entry_time)

    print(f"\n  Trades in last year: {len(year_trades)}")
    print(f"  Winners: {sum(1 for t in year_trades if t.total_pnl > 0)}")
    print(f"  Losers:  {sum(1 for t in year_trades if t.total_pnl <= 0)}")
    print(f"  Avg R-multiple (winners): +{np.mean([t.r_multiple for t in year_trades if t.total_pnl > 0]):.4f}")
    print(f"  Avg R-multiple (losers):  {np.mean([t.r_multiple for t in year_trades if t.total_pnl <= 0]):.4f}")

    # ── Compound with doubling milestones ──
    RISK_PCT = 0.30
    SLIPPAGE_R = 0.02  # realistic friction

    for scenario, slip in [("IDEAL (no slippage)", 0.0), ("REALISTIC (2-pip slippage)", SLIPPAGE_R)]:
        print(f"\n{'='*90}")
        print(f"  {scenario} -- 30% risk from $200")
        print(f"{'='*90}")

        balance = 200.0
        start_balance = 200.0
        peak = 200.0
        max_dd = 0
        max_dd_pct = 0

        next_double = 400.0  # first doubling target
        double_count = 0
        doubling_log = []

        trade_log = []  # every trade
        week_number = 0
        last_week_key = None

        for i, t in enumerate(year_trades):
            r = t.r_multiple - slip
            pnl_pct = RISK_PCT * r  # % of balance gained/lost
            pnl_dollar = balance * pnl_pct
            old_balance = balance
            balance += pnl_dollar

            if balance < 1:
                balance = 1  # blown

            # Peak / drawdown
            if balance > peak:
                peak = balance
            dd = peak - balance
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd = dd

            # Week tracking
            wk = (t.entry_time.isocalendar()[0], t.entry_time.isocalendar()[1])
            if wk != last_week_key:
                week_number += 1
                last_week_key = wk

            # Check doubling
            while balance >= next_double:
                double_count += 1
                days_elapsed = (t.exit_time - year_trades[0].entry_time).days if t.exit_time else 0
                doubling_log.append({
                    "double": double_count,
                    "target": next_double,
                    "balance": balance,
                    "trade_num": i + 1,
                    "date": t.exit_time or t.entry_time,
                    "days": days_elapsed,
                    "week": week_number,
                })
                next_double *= 2

            # Log trade
            wl = "W" if pnl_dollar > 0 else "L"
            trade_log.append({
                "num": i + 1,
                "sym": t.sym,
                "dir": t.direction,
                "r": r,
                "pnl_pct": pnl_pct * 100,
                "pnl_dollar": pnl_dollar,
                "balance": balance,
                "wl": wl,
                "date": t.entry_time,
                "exit_type": t.exit_type,
            })

        # ── Print doubling milestones ──
        print(f"\n  DOUBLING MILESTONES:")
        print(f"  {'#':>3} {'Target':>12} {'Balance':>14} {'Trade#':>7} {'Date':>12} {'Days':>5} {'Week':>5}")
        print(f"  {'-'*3} {'-'*12} {'-'*14} {'-'*7} {'-'*12} {'-'*5} {'-'*5}")

        for d in doubling_log:
            print(f"  {d['double']:>3} ${d['target']:>11,.0f} ${d['balance']:>13,.2f} "
                  f"{d['trade_num']:>7} {str(d['date'].date()):>12} {d['days']:>5} {d['week']:>5}")

        if doubling_log:
            # Average days per doubling
            doubling_days = []
            for j in range(1, len(doubling_log)):
                dd = (doubling_log[j]["date"] - doubling_log[j-1]["date"]).days
                doubling_days.append(dd)
            if doubling_days:
                avg_double_days = np.mean(doubling_days)
                print(f"\n  Avg days between doublings: {avg_double_days:.1f}")
                print(f"  Avg trades between doublings: {len(year_trades) / len(doubling_log):.1f}")

        # ── Summary ──
        print(f"\n  FINAL SUMMARY:")
        print(f"  Starting:         $200")
        print(f"  Final:            ${balance:>14,.2f}")
        print(f"  Total doublings:  {double_count}x")
        print(f"  Max drawdown:     {max_dd_pct:.1f}% (${max_dd:,.2f})")
        print(f"  Total trades:     {len(year_trades)}")

        # First 20 trades detail
        print(f"\n  First 20 trades:")
        print(f"  {'#':>3} {'Sym':<8} {'Dir':<4} {'R':>7} {'%Chg':>7} {'PnL$':>12} {'Balance':>14} {'Exit':>12}")
        print(f"  {'-'*3} {'-'*8} {'-'*4} {'-'*7} {'-'*7} {'-'*12} {'-'*14} {'-'*12}")
        for tl in trade_log[:20]:
            print(f"  {tl['num']:>3} {tl['sym']:<8} {tl['dir']:<4} {tl['r']:>+6.3f}R "
                  f"{tl['pnl_pct']:>+6.1f}% ${tl['pnl_dollar']:>11,.2f} ${tl['balance']:>13,.2f} {tl['exit_type']:>12}")

        # Weekly balance snapshots
        print(f"\n  Weekly balance (every 4 weeks):")
        week_bals = {}
        bal_track = 200.0
        for tl in trade_log:
            wk = (tl["date"].isocalendar()[0], tl["date"].isocalendar()[1])
            week_bals[wk] = tl["balance"]

        sorted_weeks = sorted(week_bals.keys())
        for j, wk in enumerate(sorted_weeks):
            if j % 4 == 0 or j == len(sorted_weeks) - 1:
                print(f"    Week {j+1:>3}: ${week_bals[wk]:>14,.2f}")


if __name__ == "__main__":
    main()
