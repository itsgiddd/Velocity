#!/usr/bin/env python3
"""
RISK-NORMALIZED BACKTEST -- V4 ZeroPoint
==========================================
The original backtest uses LOT=1.0 for every pair, which means JPY pairs
(with 3-5x wider ATR) produce 3-5x larger dollar losses. This is a
BACKTEST ARTIFACT -- live trading uses risk-based lot sizing.

This version:
  1. Computes risk-normalized lot size: risk_amount / sl_distance_in_dollars
  2. Ensures each trade risks the SAME dollar amount regardless of pair
  3. Shows whether JPY concentration disappears with proper sizing

Also tests: should we reduce JPY pair risk allocation given their
inherently higher volatility?
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
STARTING_BALANCE = 200.0
RISK_PCT = 0.30  # 30% risk per trade


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
    """
    Calculate lot size so that SL hit = risk_pct * balance in USD.
    Same logic as webhook_app._calc_lot_size.
    """
    risk_amount = balance * risk_pct
    sl_distance = abs(entry - sl)
    sl_ticks = sl_distance / tick_size if tick_size > 0 else 0

    if tick_value <= 0:
        tick_value = cs * tick_size

    loss_per_lot = sl_ticks * tick_value
    if loss_per_lot <= 0:
        return vol_min

    lot = risk_amount / loss_per_lot

    # Round to volume step
    lot = round(lot / vol_step) * vol_step
    lot = max(vol_min, min(lot, vol_max))

    # Cap table (same as live)
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

        if self.tp2_hit and not self.tp3_hit:
            if (buy and h >= self.tp3) or (not buy and l <= self.tp3):
                self.tp3_hit = True
                self.partials.append(self.pnl_for(self.tp3, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                self.exit_type = "TP3"
                return

        if self.profit_lock_active and self.profit_lock_sl is not None:
            if (buy and l <= self.profit_lock_sl) or (not buy and h >= self.profit_lock_sl):
                self.partials.append(self.pnl_for(self.profit_lock_sl, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                self.exit_type = "PROFIT_LOCK"
                return

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


def run_simulation(symbol_data, sym_info_map, risk_pct=RISK_PCT, jpy_risk_mult=1.0):
    """
    Run V4 sim with risk-normalized lot sizing.
    jpy_risk_mult: multiplier for JPY pair risk (1.0 = same, 0.5 = half risk)
    """
    trades = []
    balance = STARTING_BALANCE

    # We need to process trades in chronological order across all symbols
    # Build a list of all signal events with bar data
    events = []
    for sym, df in symbol_data.items():
        cs = contract_size(sym)
        for i in range(WARMUP, len(df)):
            row = df.iloc[i]
            events.append({
                "time": row["time"],
                "sym": sym,
                "idx": i,
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "atr": float(row["atr"]),
                "pos": int(row.get("pos", 0)),
                "buy_signal": bool(row.get("buy_signal", False)),
                "sell_signal": bool(row.get("sell_signal", False)),
                "cs": cs,
            })

    events.sort(key=lambda e: (e["time"], e["sym"]))

    # Track open position per symbol
    open_positions = {}  # sym -> V4Pos

    for ev in events:
        sym = ev["sym"]
        h, l, c = ev["high"], ev["low"], ev["close"]
        atr = ev["atr"]
        cp = ev["pos"]
        bs = ev["buy_signal"]
        ss = ev["sell_signal"]

        if np.isnan(atr) or atr <= 0:
            continue

        # Check active position for this symbol
        if sym in open_positions and not open_positions[sym].closed:
            pos_obj = open_positions[sym]
            pos_obj.check_bar(h, l, c, cp)
            if pos_obj.closed:
                pos_obj.exit_time = ev["time"]
                trades.append(pos_obj)
                balance += pos_obj.total_pnl
                del open_positions[sym]

        # Open on signal
        if bs or ss:
            d = "BUY" if bs else "SELL"
            df = symbol_data[sym]

            # Force close existing
            if sym in open_positions and not open_positions[sym].closed:
                pos_obj = open_positions[sym]
                pos_obj.force_close(c)
                pos_obj.exit_time = ev["time"]
                trades.append(pos_obj)
                balance += pos_obj.total_pnl

            smart_sl = compute_smart_sl(df, ev["idx"], d, atr)

            # Risk-normalized lot sizing
            info = sym_info_map.get(sym)
            if info is not None:
                effective_risk = risk_pct
                # Apply JPY risk multiplier
                is_jpy = "JPY" in sym.upper()
                if is_jpy:
                    effective_risk *= jpy_risk_mult

                lot = calc_risk_lot(
                    balance=max(balance, 1.0),  # floor at $1 to prevent negative lot
                    risk_pct=effective_risk,
                    entry=c,
                    sl=smart_sl,
                    tick_size=info.trade_tick_size or info.point,
                    tick_value=info.trade_tick_value if info.trade_tick_value > 0 else info.trade_contract_size * (info.trade_tick_size or info.point),
                    cs=ev["cs"],
                    vol_min=info.volume_min,
                    vol_max=info.volume_max,
                    vol_step=info.volume_step,
                )
            else:
                lot = 0.01  # fallback

            pos_obj = V4Pos(
                sym=sym, d=d, entry=c, sl=smart_sl, atr=atr,
                lot=lot, cs=ev["cs"], t=ev["time"],
            )
            open_positions[sym] = pos_obj

    # Close remaining
    for sym, pos_obj in open_positions.items():
        if not pos_obj.closed:
            df = symbol_data[sym]
            pos_obj.force_close(float(df.iloc[-1]["close"]))
            pos_obj.exit_time = df.iloc[-1]["time"]
            trades.append(pos_obj)
            balance += pos_obj.total_pnl

    return trades, balance


def print_results(label, trades, final_balance):
    total = len(trades)
    wins = [t for t in trades if t.total_pnl > 0]
    losses = [t for t in trades if t.total_pnl <= 0]
    wr = len(wins) / total * 100 if total else 0
    gross_win = sum(t.total_pnl for t in wins)
    gross_loss = abs(sum(t.total_pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    net = sum(t.total_pnl for t in trades)

    print(f"\n  {'='*80}")
    print(f"  {label}")
    print(f"  {'='*80}")
    print(f"  Starting balance: ${STARTING_BALANCE:.0f}")
    print(f"  Final balance:    ${final_balance:,.2f}")
    print(f"  Net PnL:          ${net:,.2f}")
    print(f"  Trades: {total}  |  Winners: {len(wins)} ({wr:.1f}%)  |  Losers: {len(losses)} ({100-wr:.1f}%)")
    print(f"  Gross Win: ${gross_win:,.2f}  |  Gross Loss: ${gross_loss:,.2f}  |  PF: {pf:.2f}")

    # Per-symbol breakdown
    sym_stats = {}
    for t in trades:
        if t.sym not in sym_stats:
            sym_stats[t.sym] = {"trades": 0, "wins": 0, "losses": 0,
                                "win_pnl": 0, "loss_pnl": 0, "loss_details": []}
        sym_stats[t.sym]["trades"] += 1
        if t.total_pnl > 0:
            sym_stats[t.sym]["wins"] += 1
            sym_stats[t.sym]["win_pnl"] += t.total_pnl
        else:
            sym_stats[t.sym]["losses"] += 1
            sym_stats[t.sym]["loss_pnl"] += t.total_pnl
            sym_stats[t.sym]["loss_details"].append(t)

    print(f"\n  Per-Symbol Breakdown:")
    print(f"  {'Symbol':<10} {'Trades':>6} {'Wins':>5} {'Loss':>5} {'WR':>6} "
          f"{'Win$':>12} {'Loss$':>12} {'Net$':>12} {'AvgLoss$':>10} {'%ofTotalLoss':>12}")
    print(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*5} {'-'*6} "
          f"{'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")

    for sym in sorted(sym_stats.keys()):
        s = sym_stats[sym]
        sym_wr = s["wins"] / s["trades"] * 100 if s["trades"] else 0
        sym_net = s["win_pnl"] + s["loss_pnl"]
        avg_loss = s["loss_pnl"] / s["losses"] if s["losses"] > 0 else 0
        pct_loss = abs(s["loss_pnl"]) / gross_loss * 100 if gross_loss > 0 else 0
        is_jpy = "JPY" in sym
        marker = " *JPY" if is_jpy else ""
        print(f"  {sym:<10} {s['trades']:>6} {s['wins']:>5} {s['losses']:>5} {sym_wr:>5.1f}% "
              f"${s['win_pnl']:>11,.2f} ${s['loss_pnl']:>11,.2f} ${sym_net:>11,.2f} "
              f"${avg_loss:>9,.2f} {pct_loss:>11.1f}%{marker}")

    # JPY vs non-JPY summary
    jpy_loss = sum(abs(s["loss_pnl"]) for sym, s in sym_stats.items() if "JPY" in sym)
    non_jpy_loss = sum(abs(s["loss_pnl"]) for sym, s in sym_stats.items() if "JPY" not in sym)
    jpy_pct = jpy_loss / (jpy_loss + non_jpy_loss) * 100 if (jpy_loss + non_jpy_loss) > 0 else 0

    print(f"\n  JPY loss concentration: {jpy_pct:.1f}% of total dollar losses")
    print(f"    JPY pairs:     ${jpy_loss:>12,.2f}")
    print(f"    Non-JPY pairs: ${non_jpy_loss:>12,.2f}")

    # Loser details
    if losses:
        print(f"\n  Loser Details:")
        for t in sorted(losses, key=lambda x: x.total_pnl):
            mfe = t.max_profit_reached / t.atr_val if t.atr_val > 0 else 0
            print(f"    {t.sym:<8} {t.direction:<4} lot={t.total_lot:.2f} "
                  f"${t.total_pnl:>+12,.2f} MFE={mfe:.3f}x ATR  exit={t.exit_type}")

    return {
        "trades": total, "wins": len(wins), "losses": len(losses),
        "wr": wr, "pf": pf, "net": net, "final_balance": final_balance,
        "gross_loss": gross_loss, "jpy_pct": jpy_pct,
    }


def main():
    print("=" * 90)
    print("  RISK-NORMALIZED BACKTEST -- V4 ZeroPoint")
    print("  Each trade risks RISK_PCT * balance (same $ risk regardless of pair)")
    print("  Tests JPY risk reduction to equalize loss concentration")
    print("=" * 90)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    # Load data + symbol info
    symbol_data = {}
    sym_info_map = {}
    print(f"\nLoading H4 data (starting balance: ${STARTING_BALANCE}, risk: {RISK_PCT*100:.0f}%)...")
    for sym in SYMBOLS:
        resolved, info = resolve_symbol(sym)
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
            sym_info_map[sym] = info
            tv = info.trade_tick_value if info.trade_tick_value > 0 else info.trade_contract_size * (info.trade_tick_size or info.point)
            print(f"  {sym}: {len(df_zp)} bars | tick_size={info.trade_tick_size} "
                  f"tick_value=${tv:.4f} vol_step={info.volume_step}")

    mt5.shutdown()

    if not symbol_data:
        print("No data!")
        return

    # Run baseline (equal risk)
    print("\n[1] BASELINE: Equal 30% risk for all pairs...")
    baseline_trades, baseline_bal = run_simulation(symbol_data, sym_info_map, RISK_PCT, jpy_risk_mult=1.0)
    b = print_results("BASELINE -- Equal Risk (30% all pairs)", baseline_trades, baseline_bal)

    # Run JPY risk reduction variants
    configs = [
        (0.75, "JPY at 75% risk (22.5% vs 30%)"),
        (0.50, "JPY at 50% risk (15% vs 30%)"),
        (0.33, "JPY at 33% risk (10% vs 30%)"),
    ]

    results = [b]
    for jpy_mult, label in configs:
        print(f"\n[{configs.index((jpy_mult, label)) + 2}] {label}...")
        trades, bal = run_simulation(symbol_data, sym_info_map, RISK_PCT, jpy_risk_mult=jpy_mult)
        r = print_results(f"{label}", trades, bal)
        r["jpy_mult"] = jpy_mult
        r["label"] = label
        results.append(r)

    # Comparison
    print(f"\n{'='*90}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"  {'Config':<40} {'Final$':>12} {'Net$':>12} {'WR':>6} {'PF':>6} {'Losses':>6} {'GrossLoss':>12} {'JPY%Loss':>10}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*12} {'-'*10}")

    for i, r in enumerate(results):
        lbl = "BASELINE (equal risk)" if i == 0 else results[i].get("label", "?")
        print(f"  {lbl:<40} ${r['final_balance']:>11,.0f} ${r['net']:>11,.0f} "
              f"{r['wr']:>5.1f}% {r['pf']:>6.2f} {r['losses']:>6} "
              f"${r['gross_loss']:>11,.0f} {r['jpy_pct']:>9.1f}%")


if __name__ == "__main__":
    main()
