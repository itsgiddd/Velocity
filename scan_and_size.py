#!/usr/bin/env python3
"""Scan all symbols, filter by confidence, calculate safe lot sizes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime
from app.zeropoint_signal import (
    ZeroPointEngine, ZEROPOINT_ENABLED_SYMBOLS, ZeroPointSignal,
    compute_zeropoint_state, SL_BUFFER_PCT, TP1_MULT
)

MIN_CONFIDENCE = 0.65
MAX_LOT_PER_TRADE = 0.15  # cap per trade to protect margin
MARGIN_BUDGET_PCT = 0.70  # use 70% of free margin total

if not mt5.initialize():
    print("MT5 init failed:", mt5.last_error())
    sys.exit(1)

acct = mt5.account_info()
balance = acct.balance
free_margin = acct.margin_free
print(f"Balance: ${balance:.2f} | Equity: ${acct.equity:.2f} | Free Margin: ${free_margin:.2f}")

open_positions = mt5.positions_get()
open_syms = set()
if open_positions:
    for p in open_positions:
        open_syms.add(p.symbol.upper().replace(".", "").replace("#", ""))
        d = "BUY" if p.type == 0 else "SELL"
        print(f"  Open: {p.symbol} {d} {p.volume} lots P/L=${p.profit:.2f}")

SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "NZDUSD", "GBPJPY", "USDJPY"]
zp_engine = ZeroPointEngine()
signals = []

print()

for symbol in SYMBOLS:
    norm = symbol.upper().replace(".", "").replace("#", "")
    if norm in open_syms:
        continue

    sym_info = mt5.symbol_info(symbol)
    sym_resolved = symbol
    if sym_info is None:
        for alt in [symbol + ".raw", symbol[:3]]:
            sym_info = mt5.symbol_info(alt)
            if sym_info is not None:
                sym_resolved = alt
                break
    if sym_info is None:
        continue
    mt5.symbol_select(sym_resolved, True)

    rates_h4 = mt5.copy_rates_from_pos(sym_resolved, mt5.TIMEFRAME_H4, 0, 200)
    rates_h1 = mt5.copy_rates_from_pos(sym_resolved, mt5.TIMEFRAME_H1, 0, 200)
    rates_h2 = mt5.copy_rates_from_pos(sym_resolved, mt5.TIMEFRAME_H2, 0, 200)

    df_h4, df_h1, df_h2 = None, None, None
    if rates_h4 is not None and len(rates_h4) >= 20:
        df_h4 = pd.DataFrame(rates_h4)
        df_h4["time"] = pd.to_datetime(df_h4["time"], unit="s")
    if rates_h1 is not None and len(rates_h1) >= 20:
        df_h1 = pd.DataFrame(rates_h1)
        df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
    if rates_h2 is not None and len(rates_h2) >= 20:
        df_h2 = pd.DataFrame(rates_h2)
        df_h2["time"] = pd.to_datetime(df_h2["time"], unit="s")
    if df_h4 is None:
        continue

    # H4 signal
    sig_h4 = zp_engine.generate_signal(sym_resolved, df_h4, df_h1)

    # H2 signal
    sig_h2 = None
    if df_h2 is not None:
        zp_h2 = compute_zeropoint_state(df_h2)
        if zp_h2 is not None and len(zp_h2) >= 2:
            last = zp_h2.iloc[-1]
            pos = int(last.get("pos", 0))
            if pos != 0:
                direction = "BUY" if pos == 1 else "SELL"
                entry = float(last["close"])
                atr_val = float(last["atr"])
                ts = float(last.get("xATRTrailingStop", 0))
                if atr_val > 0 and ts > 0:
                    sl = ts
                    buf = atr_val * SL_BUFFER_PCT
                    if direction == "BUY":
                        sl -= buf
                        tp1 = entry + atr_val * TP1_MULT
                    else:
                        sl += buf
                        tp1 = entry - atr_val * TP1_MULT

                    ok = True
                    if direction == "BUY" and entry >= tp1:
                        ok = False
                    if direction == "SELL" and entry <= tp1:
                        ok = False
                    if ok:
                        sl_dist = abs(entry - sl)
                        tp_dist = abs(tp1 - entry)
                        rr = tp_dist / sl_dist if sl_dist > 0 else 0

                        buy_s = bool(last.get("buy_signal", False))
                        sell_s = bool(last.get("sell_signal", False))
                        fresh = buy_s or sell_s
                        if not fresh:
                            prev = zp_h2.iloc[-2]
                            fresh = bool(prev.get("buy_signal", False)) or bool(prev.get("sell_signal", False))
                        bars = 1
                        for idx in range(len(zp_h2) - 2, -1, -1):
                            if int(zp_h2.iloc[idx].get("pos", 0)) == pos:
                                bars += 1
                            else:
                                break

                        h1c = False
                        if df_h1 is not None:
                            zh1 = compute_zeropoint_state(df_h1)
                            if zh1 is not None and len(zh1) > 0:
                                h1p = int(zh1.iloc[-1].get("pos", 0))
                                if direction == "BUY" and h1p == 1:
                                    h1c = True
                                elif direction == "SELL" and h1p == -1:
                                    h1c = True

                        if fresh:
                            conf = 0.70 + (0.15 if h1c else 0.0) + min(rr * 0.05, 0.10)
                        else:
                            ap = min(bars * 0.02, 0.15)
                            conf = 0.65 + (0.12 if h1c else 0.0) + min(rr * 0.05, 0.08) - ap
                        conf = max(0.40, min(conf, 0.98))

                        sig_h2 = ZeroPointSignal(
                            symbol=sym_resolved, direction=direction, entry_price=entry,
                            stop_loss=sl, tp1=tp1, tp2=tp1, tp3=tp1,
                            atr_value=atr_val, confidence=conf,
                            signal_time=datetime.now(), timeframe="H2",
                            tier="S" if h1c else "A", trailing_stop=ts, risk_reward=rr,
                        )

    # Pick better of H4/H2
    best = sig_h4
    tf = "H4"
    if sig_h2 is not None and (best is None or sig_h2.confidence > best.confidence):
        best = sig_h2
        tf = "H2"

    if best is None:
        continue
    if best.confidence < MIN_CONFIDENCE:
        print(f"  {symbol:8s}: {tf} {best.direction} conf={best.confidence:.0%} -- below {MIN_CONFIDENCE:.0%}, skip")
        continue

    signals.append((best, sym_resolved, tf))

signals.sort(key=lambda x: x[0].confidence, reverse=True)

print(f"\n{'='*65}")
print(f"Signals >= {MIN_CONFIDENCE:.0%} confidence (sorted by confidence)")
print(f"{'='*65}")

if not signals:
    print("  No signals pass the confidence filter.")
    mt5.shutdown()
    sys.exit(0)

# Calculate lot sizes
total_conf = sum(s[0].confidence for s in signals)
margin_budget = free_margin * MARGIN_BUDGET_PCT
margin_running = 0
trades = []

for sig, sym, tf in signals:
    sym_info = mt5.symbol_info(sym)
    if sym_info is None:
        continue

    # Proportional margin by confidence
    conf_weight = sig.confidence / total_conf
    margin_alloc = margin_budget * conf_weight

    price = sym_info.ask if sig.direction == "BUY" else sym_info.bid
    otype = mt5.ORDER_TYPE_BUY if sig.direction == "BUY" else mt5.ORDER_TYPE_SELL
    margin_per_lot = mt5.order_calc_margin(otype, sym, 1.0, price)
    if margin_per_lot is None or margin_per_lot <= 0:
        continue

    lot = margin_alloc / margin_per_lot
    vol_step = sym_info.volume_step
    lot = round(lot / vol_step) * vol_step
    lot = max(sym_info.volume_min, min(lot, MAX_LOT_PER_TRADE))

    actual_margin = mt5.order_calc_margin(otype, sym, lot, price)
    if actual_margin is None:
        actual_margin = 0
    margin_running += actual_margin

    sl_pips = abs(sig.entry_price - sig.stop_loss)
    tp_pips = abs(sig.tp1 - sig.entry_price)

    print(f"  {tf} {sym:8s} {sig.direction:4s} | conf={sig.confidence:.0%} R:R={sig.risk_reward:.2f} "
          f"| lot={lot:.2f} margin=${actual_margin:.2f}")
    print(f"       entry={sig.entry_price:.5f} SL={sig.stop_loss:.5f} TP={sig.tp1:.5f}")
    trades.append((sig, sym, tf, lot))

print(f"\nTotal margin: ${margin_running:.2f} / ${free_margin:.2f} free ({margin_running/free_margin*100:.0f}%)")
total_margin_after = acct.margin + margin_running
if total_margin_after > 0:
    margin_level = (acct.equity / total_margin_after) * 100
    print(f"Margin level after: {margin_level:.0f}%")
print(f"Trades: {len(trades)}")

mt5.shutdown()
