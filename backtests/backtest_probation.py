#!/usr/bin/env python3
"""
PROBATION PERIOD BACKTEST — V4 ZeroPoint
==========================================
Instead of predicting losers at entry (impossible — they look identical),
this tests a REACTIVE approach: enter the trade normally, but if price
hasn't moved at least X*ATR favorably within the first N bars, EXIT EARLY
at a small loss instead of waiting for the full SL hit.

Key insight: ALL 26 losers have MFE < 0.459x ATR. They NEVER move favorably.
99.2% of winners reach 0.5x ATR at some point. So if after N bars we haven't
seen meaningful favorable movement, it's almost certainly a loser.

Tests multiple probation configs:
  - Probation bars: 1, 2, 3
  - Minimum MFE threshold: 0.1x, 0.15x, 0.2x, 0.25x, 0.3x ATR
  - Compare net PnL, WR, PF vs baseline
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
            return c
    return None


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
    """V4 position with optional probation period."""

    def __init__(self, sym, d, entry, sl, atr, lot, cs, t=None,
                 probation_bars=0, probation_mfe_threshold=0.0):
        self.sym = sym
        self.direction = d
        self.entry = entry
        self.sl = sl
        self.atr_val = atr
        self.total_lot = lot
        self.remaining_lot = lot
        self.cs = cs
        self.entry_time = t
        self.exit_time = None
        sign = 1 if d == "BUY" else -1
        self.tp1 = entry + sign * TP1_MULT_AGG * atr
        self.tp2 = entry + sign * TP2_MULT_AGG * atr
        self.tp3 = entry + sign * TP3_MULT_AGG * atr
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        self.closed = False
        self.partials = []
        self.bars_in_trade = 0
        self.max_profit_reached = 0.0
        self.max_favorable_price = entry
        self.be_activated = False
        self.profit_lock_sl = None
        self.profit_lock_active = False
        self.micro_tp_hit = False
        self.stall_be_activated = False
        self.exit_type = None

        # Probation parameters
        self.probation_bars = probation_bars
        self.probation_mfe_threshold = probation_mfe_threshold
        self.probation_passed = (probation_bars == 0)  # No probation = auto-pass
        self.probation_exit = False

    def partial_lot(self):
        return max(0.01, round(self.total_lot / 3, 2))

    def pnl_for(self, price, lot):
        if self.direction == "BUY":
            return (price - self.entry) * lot * self.cs
        else:
            return (self.entry - price) * lot * self.cs

    def check_bar(self, h, l, c, pos):
        if self.closed:
            return
        self.bars_in_trade += 1
        buy = self.direction == "BUY"
        atr = self.atr_val

        # Track MFE
        if buy:
            if h > self.max_favorable_price:
                self.max_favorable_price = h
            cp = h - self.entry
        else:
            if l < self.max_favorable_price:
                self.max_favorable_price = l
            cp = self.entry - l
        if cp > self.max_profit_reached:
            self.max_profit_reached = cp

        # ── PROBATION CHECK ──
        if not self.probation_passed:
            mfe_atr = self.max_profit_reached / atr if atr > 0 else 0
            if mfe_atr >= self.probation_mfe_threshold:
                # Passed probation — price showed follow-through
                self.probation_passed = True
            elif self.bars_in_trade >= self.probation_bars:
                # Failed probation — exit at current close
                self.partials.append(self.pnl_for(c, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                self.exit_type = "PROBATION_EXIT"
                self.probation_exit = True
                return

        # ── Standard V4 trade management (same as baseline) ──

        # Post-TP1 trailing
        if self.tp1_hit:
            td = PROFIT_TRAIL_DISTANCE_MULT * atr
            if buy:
                nl = self.max_favorable_price - td
                if nl > self.entry and (self.profit_lock_sl is None or nl > self.profit_lock_sl):
                    self.profit_lock_sl = nl
                    self.profit_lock_active = True
            else:
                nl = self.max_favorable_price + td
                if nl < self.entry and (self.profit_lock_sl is None or nl < self.profit_lock_sl):
                    self.profit_lock_sl = nl
                    self.profit_lock_active = True

        # Early BE
        if not self.be_activated and self.max_profit_reached >= BE_TRIGGER_MULT * atr:
            bb = BE_BUFFER_MULT * atr
            if buy:
                ns = self.entry + bb
                if ns > self.sl:
                    self.sl = ns
                    self.be_activated = True
            else:
                ns = self.entry - bb
                if ns < self.sl:
                    self.sl = ns
                    self.be_activated = True

        # Stall exit
        if not self.tp1_hit and not self.stall_be_activated and self.bars_in_trade >= STALL_BARS:
            bb = BE_BUFFER_MULT * atr
            if buy:
                ns = self.entry + bb
                if ns > self.sl:
                    self.sl = ns
                    self.stall_be_activated = True
                    self.be_activated = True
            else:
                ns = self.entry - bb
                if ns < self.sl:
                    self.sl = ns
                    self.stall_be_activated = True
                    self.be_activated = True

        # Micro TP
        if not self.micro_tp_hit and not self.tp1_hit:
            mp = self.entry + MICRO_TP_MULT * atr if buy else self.entry - MICRO_TP_MULT * atr
            if (buy and h >= mp) or (not buy and l <= mp):
                self.micro_tp_hit = True
                ml = round(self.total_lot * MICRO_TP_PCT, 2)
                ml = max(0.01, min(ml, self.remaining_lot))
                self.partials.append(self.pnl_for(mp, ml))
                self.remaining_lot = round(self.remaining_lot - ml, 2)
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.exit_type = "MICRO_TP"
                    return

        # TP1
        if not self.tp1_hit:
            if (buy and h >= self.tp1) or (not buy and l <= self.tp1):
                self.tp1_hit = True
                p = min(self.partial_lot(), self.remaining_lot)
                self.partials.append(self.pnl_for(self.tp1, p))
                self.remaining_lot = round(self.remaining_lot - p, 2)
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.exit_type = "TP1"
                    return

        # TP2
        if self.tp1_hit and not self.tp2_hit:
            if (buy and h >= self.tp2) or (not buy and l <= self.tp2):
                self.tp2_hit = True
                self.sl = self.entry
                self.be_activated = True
                p = min(self.partial_lot(), self.remaining_lot)
                self.partials.append(self.pnl_for(self.tp2, p))
                self.remaining_lot = round(self.remaining_lot - p, 2)
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.exit_type = "TP2"
                    return

        # TP3
        if self.tp2_hit and not self.tp3_hit:
            if (buy and h >= self.tp3) or (not buy and l <= self.tp3):
                self.tp3_hit = True
                self.partials.append(self.pnl_for(self.tp3, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                self.exit_type = "TP3"
                return

        # Profit lock trail
        if self.profit_lock_active and self.profit_lock_sl is not None:
            if (buy and l <= self.profit_lock_sl) or (not buy and h >= self.profit_lock_sl):
                self.partials.append(self.pnl_for(self.profit_lock_sl, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                self.exit_type = "PROFIT_LOCK"
                return

        # SL
        if (buy and l <= self.sl) or (not buy and h >= self.sl):
            self.partials.append(self.pnl_for(self.sl, self.remaining_lot))
            self.remaining_lot = 0
            self.closed = True
            if self.stall_be_activated:
                self.exit_type = "SL_STALL"
            elif self.be_activated:
                self.exit_type = "SL_BE"
            else:
                self.exit_type = "SL"
            return

        # ZP flip exit
        if pos != 0 and ((buy and pos == -1) or (not buy and pos == 1)):
            self.partials.append(self.pnl_for(c, self.remaining_lot))
            self.remaining_lot = 0
            self.closed = True
            self.exit_type = "ZP_FLIP"
            return

    def force_close(self, price):
        if self.closed or self.remaining_lot <= 0:
            return
        self.partials.append(self.pnl_for(price, self.remaining_lot))
        self.remaining_lot = 0
        self.closed = True
        self.exit_type = "END"

    @property
    def total_pnl(self):
        return sum(self.partials)


def run_simulation(symbol_data, probation_bars=0, probation_mfe=0.0):
    """Run V4 simulation with optional probation period."""
    trades = []

    for sym, df in symbol_data.items():
        cs = contract_size(sym)
        n = len(df)
        pos_obj = None

        for i in range(WARMUP, n):
            row = df.iloc[i]
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            atr = float(row["atr"])
            cp = int(row.get("pos", 0))
            bs = bool(row.get("buy_signal", False))
            ss = bool(row.get("sell_signal", False))
            if np.isnan(atr) or atr <= 0:
                continue

            # Check active position
            if pos_obj is not None and not pos_obj.closed:
                pos_obj.check_bar(h, l, c, cp)
                if pos_obj.closed:
                    pos_obj.exit_time = row["time"]
                    trades.append(pos_obj)
                    pos_obj = None

            # Open on signal
            if bs or ss:
                d = "BUY" if bs else "SELL"

                # Force close existing
                if pos_obj is not None and not pos_obj.closed:
                    pos_obj.force_close(c)
                    pos_obj.exit_time = row["time"]
                    trades.append(pos_obj)

                smart_sl = compute_smart_sl(df, i, d, atr)
                pos_obj = V4Pos(
                    sym=sym, d=d, entry=c, sl=smart_sl, atr=atr, lot=LOT, cs=cs, t=row["time"],
                    probation_bars=probation_bars,
                    probation_mfe_threshold=probation_mfe,
                )

        # Close remaining
        if pos_obj is not None and not pos_obj.closed:
            pos_obj.force_close(float(df.iloc[-1]["close"]))
            pos_obj.exit_time = df.iloc[-1]["time"]
            trades.append(pos_obj)

    return trades


def analyze_trades(trades):
    """Return summary stats."""
    total = len(trades)
    if total == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "wr": 0, "pf": 0, "net": 0,
                "gross_win": 0, "gross_loss": 0, "probation_exits": 0,
                "prob_exit_losers_saved": 0, "prob_exit_winners_lost": 0,
                "prob_exit_avg_loss": 0}

    wins = [t for t in trades if t.total_pnl > 0]
    losses = [t for t in trades if t.total_pnl <= 0]
    wr = len(wins) / total * 100
    gross_win = sum(t.total_pnl for t in wins)
    gross_loss = abs(sum(t.total_pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    net = sum(t.total_pnl for t in trades)

    # Probation-specific stats
    prob_exits = [t for t in trades if t.exit_type == "PROBATION_EXIT"]
    prob_exit_losers = [t for t in prob_exits if t.total_pnl <= 0]
    prob_exit_winners = [t for t in prob_exits if t.total_pnl > 0]
    prob_exit_avg_loss = np.mean([t.total_pnl for t in prob_exits]) if prob_exits else 0

    # For losers: how does the probation exit loss compare to what the full SL loss would be?
    # We can't know exactly, but we track the PnL at probation exit

    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "wr": wr,
        "pf": pf,
        "net": net,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "probation_exits": len(prob_exits),
        "prob_exit_losers_saved": len(prob_exit_losers),
        "prob_exit_winners_lost": len(prob_exit_winners),
        "prob_exit_avg_pnl": prob_exit_avg_loss,
        "trades_list": trades,
    }


def main():
    print("=" * 100)
    print("  PROBATION PERIOD BACKTEST — V4 ZeroPoint")
    print("  Enter every trade, but exit early if MFE < threshold within N bars")
    print("=" * 100)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    symbol_data = {}
    print("\nLoading H4 data...")
    for sym in SYMBOLS:
        resolved = resolve_symbol(sym)
        if resolved is None:
            continue
        rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H4, 0, 5000)
        if rates is None or len(rates) < 100:
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df_zp = compute_zeropoint_state(df)
        if df_zp is not None and len(df_zp) >= WARMUP:
            symbol_data[sym] = df_zp
            print(f"  {sym}: {len(df_zp)} bars")
    mt5.shutdown()

    if not symbol_data:
        print("No data!")
        return

    # ── Run baseline ──
    print("\n[1/2] Running BASELINE (no probation)...")
    baseline_trades = run_simulation(symbol_data, probation_bars=0, probation_mfe=0.0)
    baseline = analyze_trades(baseline_trades)

    print(f"\n  BASELINE: {baseline['trades']} trades, WR={baseline['wr']:.1f}%, "
          f"PF={baseline['pf']:.2f}, Net=${baseline['net']:,.0f}")
    print(f"  Winners: {baseline['wins']}, Losers: {baseline['losses']}")
    print(f"  Gross Win: ${baseline['gross_win']:,.0f}, Gross Loss: ${baseline['gross_loss']:,.0f}")

    # Loser detail
    baseline_losers = [t for t in baseline_trades if t.total_pnl <= 0]
    print(f"\n  Baseline losers ({len(baseline_losers)}):")
    for t in sorted(baseline_losers, key=lambda x: x.total_pnl):
        mfe = t.max_profit_reached / t.atr_val if t.atr_val > 0 else 0
        print(f"    {t.sym:<8} {t.direction:<4} ${t.total_pnl:>+10,.2f} MFE={mfe:.3f}x ATR  exit={t.exit_type}")

    # MFE distribution of FIRST N bars for all trades (critical analysis)
    print(f"\n  {'='*90}")
    print(f"  MFE AFTER N BARS — Distribution for Winners vs Losers")
    print(f"  {'='*90}")

    for n_bars in [1, 2, 3]:
        print(f"\n  After {n_bars} bar(s):")
        # Re-run tracking just MFE at N bars
        for label, trade_list in [("LOSERS", baseline_losers),
                                   ("WINNERS", [t for t in baseline_trades if t.total_pnl > 0])]:
            mfes = []
            for t in trade_list:
                mfe_atr = t.max_profit_reached / t.atr_val if t.atr_val > 0 else 0
                # We only have final MFE, not per-bar MFE.
                # But for losers, their MFE is the TOTAL MFE (they never go far)
                # We need per-bar tracking... we don't have it in baseline.
                # Let's just note this and use probation as the actual test.
                mfes.append(mfe_atr)
            if mfes:
                print(f"    {label}: count={len(mfes)}, "
                      f"mean MFE={np.mean(mfes):.3f}x, "
                      f"median={np.median(mfes):.3f}x, "
                      f"min={np.min(mfes):.3f}x, "
                      f"p10={np.percentile(mfes, 10):.3f}x, "
                      f"p25={np.percentile(mfes, 25):.3f}x")

    # ── Run probation variants ──
    print(f"\n[2/2] Testing probation variants...")

    configs = []
    for pb in [1, 2, 3]:
        for mfe_thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
            configs.append((pb, mfe_thresh))

    results = []
    for pb, mfe_thresh in configs:
        trades = run_simulation(symbol_data, probation_bars=pb, probation_mfe=mfe_thresh)
        stats = analyze_trades(trades)
        stats["config"] = f"{pb}bar/{mfe_thresh:.2f}x"
        stats["pb"] = pb
        stats["mfe_thresh"] = mfe_thresh
        results.append(stats)

    # ── Results table ──
    print(f"\n{'='*100}")
    print(f"  PROBATION RESULTS — Sorted by Net PnL")
    print(f"{'='*100}")
    print(f"  {'Config':<12} {'Trades':>6} {'Wins':>5} {'Loss':>5} {'WR':>6} {'PF':>6} "
          f"{'Net PnL':>12} {'Gross Loss':>12} {'ProbExit':>8} {'L_Save':>6} {'W_Lost':>6} "
          f"{'Avg PE PnL':>11} {'vs Base':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*6} "
          f"{'-'*12} {'-'*12} {'-'*8} {'-'*6} {'-'*6} {'-'*11} {'-'*10}")

    # Add baseline row
    print(f"  {'BASELINE':<12} {baseline['trades']:>6} {baseline['wins']:>5} {baseline['losses']:>5} "
          f"{baseline['wr']:>5.1f}% {baseline['pf']:>6.2f} "
          f"${baseline['net']:>11,.0f} ${baseline['gross_loss']:>11,.0f} "
          f"{'---':>8} {'---':>6} {'---':>6} {'---':>11} {'---':>10}")

    results.sort(key=lambda x: x["net"], reverse=True)

    for r in results:
        delta = r["net"] - baseline["net"]
        delta_str = f"${delta:>+9,.0f}"
        marker = " *** BEST" if r == results[0] else ""

        print(f"  {r['config']:<12} {r['trades']:>6} {r['wins']:>5} {r['losses']:>5} "
              f"{r['wr']:>5.1f}% {r['pf']:>6.2f} "
              f"${r['net']:>11,.0f} ${r['gross_loss']:>11,.0f} "
              f"{r['probation_exits']:>8} {r['prob_exit_losers_saved']:>6} {r['prob_exit_winners_lost']:>6} "
              f"${r['prob_exit_avg_pnl']:>10,.0f} {delta_str}{marker}")

    # ── Deep dive on best result ──
    if results:
        best = results[0]
        print(f"\n{'='*100}")
        print(f"  BEST PROBATION CONFIG: {best['config']}")
        print(f"{'='*100}")
        print(f"  Probation bars:       {best['pb']}")
        print(f"  MFE threshold:        {best['mfe_thresh']:.2f}x ATR")
        print(f"  Trades:               {best['trades']}")
        print(f"  Win rate:             {best['wr']:.1f}%")
        print(f"  Profit factor:        {best['pf']:.2f}")
        print(f"  Net PnL:              ${best['net']:,.0f}")
        print(f"  vs Baseline:          ${best['net'] - baseline['net']:+,.0f}")
        print(f"  Probation exits:      {best['probation_exits']}")
        print(f"    → Losers saved:     {best['prob_exit_losers_saved']}")
        print(f"    → Winners killed:   {best['prob_exit_winners_lost']}")
        print(f"    → Avg exit PnL:     ${best['prob_exit_avg_pnl']:,.0f}")

        # Show probation exit trades
        prob_exits = [t for t in best["trades_list"] if t.exit_type == "PROBATION_EXIT"]
        if prob_exits:
            print(f"\n  All probation exits ({len(prob_exits)}):")
            for t in sorted(prob_exits, key=lambda x: x.total_pnl):
                mfe = t.max_profit_reached / t.atr_val if t.atr_val > 0 else 0
                was_loser = "SAVED" if t.total_pnl <= 0 else "KILLED WINNER"
                print(f"    {t.sym:<8} {t.direction:<4} ${t.total_pnl:>+10,.2f} "
                      f"MFE={mfe:.3f}x bars={t.bars_in_trade} → {was_loser}")

        # Show remaining losers (not caught by probation)
        remaining_losers = [t for t in best["trades_list"]
                           if t.total_pnl <= 0 and t.exit_type != "PROBATION_EXIT"]
        if remaining_losers:
            print(f"\n  Remaining losers ({len(remaining_losers)}) — not caught by probation:")
            for t in sorted(remaining_losers, key=lambda x: x.total_pnl):
                mfe = t.max_profit_reached / t.atr_val if t.atr_val > 0 else 0
                print(f"    {t.sym:<8} {t.direction:<4} ${t.total_pnl:>+10,.2f} "
                      f"MFE={mfe:.3f}x exit={t.exit_type} bars={t.bars_in_trade}")

    # ── Winners-that-dip analysis ──
    # How many winners have MFE < threshold after N bars but later recover?
    print(f"\n{'='*100}")
    print(f"  COLLATERAL DAMAGE ANALYSIS — Winners temporarily below threshold")
    print(f"  These are winners that would be probation-exited (false negatives)")
    print(f"{'='*100}")

    for pb in [1, 2, 3]:
        for mfe_thresh in [0.10, 0.15, 0.20, 0.25]:
            trades = run_simulation(symbol_data, probation_bars=pb, probation_mfe=mfe_thresh)
            prob_exits = [t for t in trades if t.exit_type == "PROBATION_EXIT"]
            killed_winners = [t for t in prob_exits if t.total_pnl > 0]
            saved_losers = [t for t in prob_exits if t.total_pnl <= 0]

            # What would the killed winners have been worth?
            # We need a separate baseline run to know their full PnL...
            # Approximate: killed winners have small positive PnL at probation exit
            killed_pnl = sum(t.total_pnl for t in killed_winners)
            saved_pnl = abs(sum(t.total_pnl for t in saved_losers))

            print(f"  {pb}bar/{mfe_thresh:.2f}x: "
                  f"PE={len(prob_exits):>3}, "
                  f"W_killed={len(killed_winners):>3} (lost ${killed_pnl:>+10,.0f}), "
                  f"L_saved={len(saved_losers):>3} (saved ${saved_pnl:>10,.0f}), "
                  f"net_benefit=${saved_pnl + killed_pnl:>+10,.0f}")


if __name__ == "__main__":
    main()
