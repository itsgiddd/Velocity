#!/usr/bin/env python3
"""Quick backtest: BASELINE vs FILTERED (5-flag risk scoring) on V4 ZeroPoint."""

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
        self.lot_mult = 1.0
        self.risk_flags = 0
        self.risk_reasons = []

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


def compute_risk_flags(df, idx, direction, atr_val, entry_price):
    """Compute 5 risk flags — same logic as trading_app.py."""
    flags = 0
    reasons = []
    row = df.iloc[idx]
    t = row["time"]

    # Flag 1: BUY side (3.3% vs 1.5% loss rate)
    if direction == "BUY":
        flags += 1
        reasons.append("BUY")

    # Flag 2: Friday (4.1% loss rate)
    if hasattr(t, "dayofweek") and t.dayofweek == 4:
        flags += 1
        reasons.append("FRI")

    # Flag 3: Price at 20-bar extreme
    if idx >= 21:
        h20 = float(df["high"].iloc[idx - 20:idx].max())
        l20 = float(df["low"].iloc[idx - 20:idx].min())
        rng = h20 - l20
        if rng > 0:
            pp = (entry_price - l20) / rng
            if (direction == "BUY" and pp > 0.85) or (direction == "SELL" and pp < 0.15):
                flags += 1
                reasons.append(f"EXTREME({pp:.2f})")

    # Flag 4: ATR contracting
    if idx >= 11:
        recent_trs = []
        older_trs = []
        for k in range(idx - 10, idx):
            tr = max(
                float(df["high"].iloc[k]) - float(df["low"].iloc[k]),
                abs(float(df["high"].iloc[k]) - float(df["close"].iloc[k - 1])),
                abs(float(df["low"].iloc[k]) - float(df["close"].iloc[k - 1])),
            )
            if k >= idx - 5:
                recent_trs.append(tr)
            else:
                older_trs.append(tr)
        if older_trs and recent_trs:
            r_avg = sum(recent_trs) / len(recent_trs)
            o_avg = sum(older_trs) / len(older_trs)
            if o_avg > 0 and r_avg / o_avg < 0.95:
                flags += 1
                reasons.append(f"ATR_DOWN({r_avg / o_avg:.2f})")

    # Flag 5: Low 3-bar momentum (< 1.2 ATR)
    if idx >= 3 and atr_val > 0:
        close_now = float(df["close"].iloc[idx])
        close_3ago = float(df["close"].iloc[idx - 3])
        mom = abs(close_now - close_3ago) / atr_val
        if mom < 1.2:
            flags += 1
            reasons.append(f"LOW_MOM({mom:.2f})")

    return flags, reasons


def run_simulation(symbol_data, use_filter):
    """Run full V4 simulation with or without risk filter."""
    trades = []
    skipped = 0

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

                # Risk scoring
                lot_mult = 1.0
                flags = 0
                reasons = []
                if use_filter:
                    flags, reasons = compute_risk_flags(df, i, d, atr, c)
                    if flags >= 4:
                        skipped += 1
                        pos_obj = None
                        continue
                    elif flags >= 3:
                        lot_mult = 0.50
                    elif flags >= 2:
                        lot_mult = 0.75

                lot = LOT * lot_mult
                pos_obj = V4Pos(sym=sym, d=d, entry=c, sl=smart_sl, atr=atr, lot=lot, cs=cs, t=row["time"])
                pos_obj.lot_mult = lot_mult
                pos_obj.risk_flags = flags
                pos_obj.risk_reasons = reasons

        # Close remaining
        if pos_obj is not None and not pos_obj.closed:
            pos_obj.force_close(float(df.iloc[-1]["close"]))
            pos_obj.exit_time = df.iloc[-1]["time"]
            trades.append(pos_obj)

    return trades, skipped


def print_results(label, trades, skipped=0):
    total = len(trades)
    wins = [t for t in trades if t.total_pnl > 0]
    losses = [t for t in trades if t.total_pnl <= 0]
    wr = len(wins) / total * 100 if total else 0
    gross_win = sum(t.total_pnl for t in wins)
    gross_loss = abs(sum(t.total_pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    net = sum(t.total_pnl for t in trades)

    print(f"\n  {'=' * 70}")
    print(f"  {label}")
    print(f"  {'=' * 70}")
    print(f"  Trades:       {total}" + (f"  (skipped {skipped})" if skipped else ""))
    print(f"  Winners:      {len(wins)} ({wr:.1f}%)")
    print(f"  Losers:       {len(losses)} ({100 - wr:.1f}%)")
    print(f"  Gross Win:    ${gross_win:>14,.2f}")
    print(f"  Gross Loss:   ${gross_loss:>14,.2f}")
    print(f"  Net PnL:      ${net:>14,.2f}")
    print(f"  Profit Factor: {pf:.2f}")

    # Breakdown by lot multiplier
    by_mult = {}
    for t in trades:
        m = t.lot_mult
        if m not in by_mult:
            by_mult[m] = {"trades": 0, "wins": 0, "losses": 0, "loss_pnl": 0.0, "win_pnl": 0.0}
        by_mult[m]["trades"] += 1
        if t.total_pnl > 0:
            by_mult[m]["wins"] += 1
            by_mult[m]["win_pnl"] += t.total_pnl
        else:
            by_mult[m]["losses"] += 1
            by_mult[m]["loss_pnl"] += t.total_pnl

    if len(by_mult) > 1:
        print(f"\n  Lot Sizing Breakdown:")
        for m in sorted(by_mult.keys(), reverse=True):
            d = by_mult[m]
            lr = d["losses"] / d["trades"] * 100 if d["trades"] else 0
            print(f"    x{m:.2f}: {d['trades']:>4} trades | {d['wins']} W / {d['losses']} L ({lr:.1f}% loss rate) | Loss $: ${d['loss_pnl']:>+12,.2f} | Win $: ${d['win_pnl']:>+12,.2f}")

    # Individual losers
    if losses:
        print(f"\n  Losers Detail:")
        for t in sorted(losses, key=lambda x: x.total_pnl):
            mfe = t.max_profit_reached / t.atr_val if t.atr_val > 0 else 0
            mult_str = f" x{t.lot_mult}" if t.lot_mult < 1.0 else ""
            flag_str = f" [{t.risk_flags}F: {','.join(t.risk_reasons)}]" if t.risk_flags > 0 else ""
            print(f"    {t.sym:<8} {t.direction:<4} ${t.total_pnl:>+12,.2f} MFE={mfe:.3f}ATR {t.exit_type}{mult_str}{flag_str}")

    return {"trades": total, "wins": len(wins), "losses": len(losses),
            "wr": wr, "pf": pf, "net": net, "gross_loss": gross_loss}


def main():
    print("=" * 80)
    print("  V4 ZEROPOINT: BASELINE vs FILTERED (5-Flag Risk Scoring)")
    print("=" * 80)

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

    # Run both modes
    baseline_trades, _ = run_simulation(symbol_data, use_filter=False)
    filtered_trades, skipped = run_simulation(symbol_data, use_filter=True)

    b = print_results("BASELINE (no filter)", baseline_trades)
    f = print_results("FILTERED (5-flag risk scoring)", filtered_trades, skipped)

    # Comparison
    print(f"\n  {'=' * 70}")
    print(f"  COMPARISON")
    print(f"  {'=' * 70}")
    print(f"  {'Metric':<20} | {'BASELINE':>14} | {'FILTERED':>14} | {'Delta':>14}")
    print(f"  {'-' * 65}")
    print(f"  {'Trades':<20} | {b['trades']:>14} | {f['trades']:>14} | {f['trades'] - b['trades']:>+14}")
    print(f"  {'Winners':<20} | {b['wins']:>14} | {f['wins']:>14} | {f['wins'] - b['wins']:>+14}")
    print(f"  {'Losers':<20} | {b['losses']:>14} | {f['losses']:>14} | {f['losses'] - b['losses']:>+14}")
    print(f"  {'Win Rate':<20} | {b['wr']:>13.1f}% | {f['wr']:>13.1f}% | {f['wr'] - b['wr']:>+13.1f}%")
    print(f"  {'Profit Factor':<20} | {b['pf']:>14.2f} | {f['pf']:>14.2f} | {f['pf'] - b['pf']:>+14.2f}")
    print(f"  {'Net PnL':<20} | ${b['net']:>13,.2f} | ${f['net']:>13,.2f} | ${f['net'] - b['net']:>+13,.2f}")
    print(f"  {'Gross Loss':<20} | ${b['gross_loss']:>13,.2f} | ${f['gross_loss']:>13,.2f} | ${f['gross_loss'] - b['gross_loss']:>+13,.2f}")
    loss_reduction = (1 - f["gross_loss"] / b["gross_loss"]) * 100 if b["gross_loss"] > 0 else 0
    print(f"\n  Loss reduction: {loss_reduction:.1f}%")
    print(f"  Trades skipped: {skipped}")


if __name__ == "__main__":
    main()
