#!/usr/bin/env python3
"""
1-YEAR PROJECTION -- V4 ZeroPoint from $200
============================================
Uses actual backtest trade statistics with risk-normalized compounding.
Not a fixed-lot backtest — simulates real account growth with:
  - 30% risk per trade
  - Proper lot sizing (risk_amount / sl_distance)
  - Lot cap table (same as live)
  - Adaptive risk: +25% after 3 wins, -37.5% after loss
  - 97.5% WR, actual win/loss distribution from backtest
"""

import sys, os
import numpy as np
import pandas as pd
from collections import defaultdict

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
STARTING_BALANCE = 200.0
BASE_RISK_PCT = 0.30


def resolve_symbol(ticker):
    for c in [ticker, ticker + ".raw", ticker + "m", ticker + ".a", ticker + ".e"]:
        info = mt5.symbol_info(c)
        if info is not None:
            mt5.symbol_select(c, True)
            return c, info
    return None, None


def contract_size(sym):
    return 1.0 if "BTC" in sym.upper() else 100_000.0


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


def calc_risk_lot(balance, risk_pct, entry, sl, tick_size, tick_value, cs,
                  vol_min, vol_max, vol_step):
    risk_amount = balance * risk_pct
    sl_distance = abs(entry - sl)
    sl_ticks = sl_distance / tick_size if tick_size > 0 else 0
    if tick_value <= 0:
        tick_value = cs * tick_size
    loss_per_lot = sl_ticks * tick_value
    if loss_per_lot <= 0:
        return vol_min
    lot = risk_amount / loss_per_lot
    lot = round(lot / vol_step) * vol_step
    lot = max(vol_min, min(lot, vol_max))
    cap_table = [
        (500, 0.10), (1000, 0.20), (3000, 0.50),
        (5000, 1.00), (10000, 2.00), (50000, 5.00),
        (float('inf'), 10.00),
    ]
    for threshold, max_lot in cap_table:
        if balance <= threshold:
            lot = min(lot, max_lot)
            break
    return lot


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
        self.profit_lock_sl = self.profit_lock_active = None
        self.profit_lock_active = False
        self.micro_tp_hit = self.stall_be_activated = False
        self.exit_type = None

    def partial_lot(self):
        return max(0.01, round(self.total_lot / 3, 2))

    def pnl_for(self, price, lot):
        return ((price - self.entry) if self.direction == "BUY" else (self.entry - price)) * lot * self.cs

    def check_bar(self, h, l, c, pos):
        if self.closed:
            return
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


def main():
    print("=" * 90)
    print("  1-YEAR PROJECTION -- V4 ZeroPoint from $200")
    print("  Risk-normalized compounding with adaptive risk")
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
            print(f"  {sym}: {len(df_zp)} bars")
    mt5.shutdown()

    if not symbol_data: print("No data!"); return

    # Build chronological event list
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

    # Find time range
    all_times = [e["time"] for e in events]
    first_time = min(all_times)
    last_time = max(all_times)
    total_days = (last_time - first_time).days
    print(f"\n  Data range: {first_time.date()} to {last_time.date()} ({total_days} days)")

    # Find the 1-year mark from the END (most recent year)
    one_year_ago = last_time - pd.Timedelta(days=365)
    print(f"  Last 1 year: {one_year_ago.date()} to {last_time.date()}")

    # Run simulation
    balance = STARTING_BALANCE
    open_positions = {}
    trades = []
    consecutive_wins = 0
    risk_pct = BASE_RISK_PCT

    # Track monthly/weekly snapshots
    weekly_snapshots = []
    monthly_snapshots = []
    last_week = None
    last_month = None
    trade_count_1yr = 0

    for ev in events:
        sym = ev["sym"]
        h, l, c = ev["high"], ev["low"], ev["close"]
        atr = ev["atr"]
        cp = ev["pos"]
        bs, ss = ev["buy_signal"], ev["sell_signal"]

        if np.isnan(atr) or atr <= 0:
            continue

        # Only count trades in the last year
        in_last_year = ev["time"] >= one_year_ago

        # Weekly/monthly snapshots
        week_num = ev["time"].isocalendar()[1]
        month_num = ev["time"].month
        year_num = ev["time"].year
        week_key = (year_num, week_num)
        month_key = (year_num, month_num)

        if in_last_year:
            if last_week is not None and week_key != last_week:
                weekly_snapshots.append({"week": last_week, "balance": balance})
            if last_month is not None and month_key != last_month:
                monthly_snapshots.append({"month": last_month, "balance": balance})
            last_week = week_key
            last_month = month_key

        # Check active position
        if sym in open_positions and not open_positions[sym].closed:
            pos_obj = open_positions[sym]
            pos_obj.check_bar(h, l, c, cp)
            if pos_obj.closed:
                pos_obj.exit_time = ev["time"]
                pnl = pos_obj.total_pnl
                balance += pnl

                if in_last_year:
                    trades.append(pos_obj)
                    trade_count_1yr += 1

                    # Adaptive risk
                    if pnl > 0:
                        consecutive_wins += 1
                        if consecutive_wins >= 3:
                            risk_pct = min(BASE_RISK_PCT * 1.25, 0.40)
                    else:
                        consecutive_wins = 0
                        risk_pct = BASE_RISK_PCT * 0.625  # -37.5%

                    # Cap risk above $50K
                    if balance > 50000:
                        risk_pct = min(risk_pct, 0.20)

                del open_positions[sym]

        # Open on signal
        if bs or ss:
            d = "BUY" if bs else "SELL"

            if sym in open_positions and not open_positions[sym].closed:
                pos_obj = open_positions[sym]
                pos_obj.force_close(c)
                pos_obj.exit_time = ev["time"]
                pnl = pos_obj.total_pnl
                balance += pnl
                if in_last_year:
                    trades.append(pos_obj)
                    trade_count_1yr += 1

            df = symbol_data[sym]
            smart_sl = compute_smart_sl(df, ev["idx"], d, atr)

            info = sym_info_map.get(sym)
            if info and balance > 0:
                lot = calc_risk_lot(
                    balance=max(balance, 1.0), risk_pct=risk_pct,
                    entry=c, sl=smart_sl,
                    tick_size=info.trade_tick_size or info.point,
                    tick_value=info.trade_tick_value if info.trade_tick_value > 0 else info.trade_contract_size * (info.trade_tick_size or info.point),
                    cs=ev["cs"], vol_min=info.volume_min, vol_max=info.volume_max, vol_step=info.volume_step,
                )
            else:
                lot = 0.01

            pos_obj = V4Pos(sym=sym, d=d, entry=c, sl=smart_sl, atr=atr, lot=lot, cs=ev["cs"], t=ev["time"])
            open_positions[sym] = pos_obj

    # Close remaining
    for sym, pos_obj in open_positions.items():
        if not pos_obj.closed:
            df = symbol_data[sym]
            pos_obj.force_close(float(df.iloc[-1]["close"]))
            pos_obj.exit_time = df.iloc[-1]["time"]
            balance += pos_obj.total_pnl
            trades.append(pos_obj)

    # Add final snapshots
    if last_week: weekly_snapshots.append({"week": last_week, "balance": balance})
    if last_month: monthly_snapshots.append({"month": last_month, "balance": balance})

    # ── Results ──
    wins_1yr = [t for t in trades if t.total_pnl > 0]
    losses_1yr = [t for t in trades if t.total_pnl <= 0]
    wr = len(wins_1yr) / len(trades) * 100 if trades else 0
    gross_win = sum(t.total_pnl for t in wins_1yr)
    gross_loss = abs(sum(t.total_pnl for t in losses_1yr))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print(f"\n{'='*90}")
    print(f"  1-YEAR RESULTS (last 365 days of data)")
    print(f"{'='*90}")
    print(f"  Starting balance:  ${STARTING_BALANCE:.0f}")
    print(f"  Final balance:     ${balance:,.2f}")
    print(f"  Net profit:        ${balance - STARTING_BALANCE:,.2f}")
    print(f"  Return:            {(balance / STARTING_BALANCE - 1) * 100:,.1f}%")
    print(f"  Trades (1yr):      {len(trades)}")
    print(f"  Winners:           {len(wins_1yr)} ({wr:.1f}%)")
    print(f"  Losers:            {len(losses_1yr)} ({100-wr:.1f}%)")
    print(f"  Gross Win:         ${gross_win:,.2f}")
    print(f"  Gross Loss:        ${gross_loss:,.2f}")
    print(f"  Profit Factor:     {pf:.2f}")

    # Trades per week
    trades_per_week = len(trades) / 52 if trades else 0
    print(f"  Trades/week:       {trades_per_week:.1f}")

    # Monthly growth
    if monthly_snapshots:
        print(f"\n  Monthly Balance Snapshots:")
        print(f"  {'Month':>10} {'Balance':>14} {'Growth':>10}")
        print(f"  {'-'*10} {'-'*14} {'-'*10}")
        prev_bal = STARTING_BALANCE
        for ms in monthly_snapshots:
            yr, mo = ms["month"]
            growth = (ms["balance"] / prev_bal - 1) * 100 if prev_bal > 0 else 0
            print(f"  {yr}-{mo:02d}     ${ms['balance']:>13,.2f} {growth:>+9.1f}%")
            prev_bal = ms["balance"]

    # Worst drawdown in the year
    peak = STARTING_BALANCE
    max_dd = 0
    max_dd_pct = 0
    running_bal = STARTING_BALANCE
    for t in trades:
        running_bal += t.total_pnl
        if running_bal > peak:
            peak = running_bal
        dd = peak - running_bal
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd = dd

    print(f"\n  Max drawdown:      ${max_dd:,.2f} ({max_dd_pct:.1f}%)")

    # Loser details
    if losses_1yr:
        print(f"\n  Losers in last year ({len(losses_1yr)}):")
        for t in sorted(losses_1yr, key=lambda x: x.total_pnl):
            mfe = t.max_profit_reached / t.atr_val if t.atr_val > 0 else 0
            print(f"    {t.sym:<8} {t.direction:<4} lot={t.total_lot:.2f} "
                  f"${t.total_pnl:>+12,.2f} MFE={mfe:.3f}x {t.exit_type} "
                  f"@ {t.entry_time}")

    # Per-symbol 1yr breakdown
    sym_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in trades:
        sym_stats[t.sym]["trades"] += 1
        sym_stats[t.sym]["pnl"] += t.total_pnl
        if t.total_pnl > 0:
            sym_stats[t.sym]["wins"] += 1

    print(f"\n  Per-Symbol (1yr):")
    print(f"  {'Symbol':<10} {'Trades':>6} {'WR':>6} {'Net PnL':>14}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*14}")
    for sym in sorted(sym_stats.keys()):
        s = sym_stats[sym]
        swr = s["wins"] / s["trades"] * 100 if s["trades"] else 0
        print(f"  {sym:<10} {s['trades']:>6} {swr:>5.1f}% ${s['pnl']:>13,.2f}")

    # ── CONSERVATIVE PROJECTION (what to actually expect) ──
    print(f"\n{'='*90}")
    print(f"  REALISTIC EXPECTATIONS")
    print(f"{'='*90}")

    # The backtest is idealized. Real trading has:
    # - Slippage (1-3 pips per trade)
    # - Spread widening during news
    # - Missed signals (MT5 disconnects, server issues)
    # - Emotional interference
    # Discount factor: ~60-70% of backtest performance

    for discount_label, discount in [("Backtest (ideal)", 1.0),
                                      ("Conservative (70%)", 0.70),
                                      ("Realistic (50%)", 0.50),
                                      ("Pessimistic (30%)", 0.30)]:
        adj_profit = (balance - STARTING_BALANCE) * discount
        adj_final = STARTING_BALANCE + adj_profit
        print(f"  {discount_label:<25} Final: ${adj_final:>12,.0f}  Profit: ${adj_profit:>12,.0f}")


if __name__ == "__main__":
    main()
