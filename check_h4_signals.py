"""Quick check: What H4 ZP signals are active right now?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import MetaTrader5 as mt5
import pandas as pd
from app.zeropoint_signal import compute_zeropoint_state, SL_BUFFER_PCT, TP1_MULT

SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'BTCUSD']

if not mt5.initialize():
    print("MT5 init failed")
    sys.exit(1)

print("=" * 90)
print(f"{'Symbol':<10} {'Dir':<5} {'R:R':>5} {'Fresh?':>7} {'Age(b)':>7} {'Entry':>12} {'SL':>12} {'TP':>12} {'ATR':>10}")
print("=" * 90)

for sym in SYMBOLS:
    info = mt5.symbol_info(sym)
    if info is None:
        continue
    mt5.symbol_select(sym, True)

    # H4 data
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 200)
    if rates is None or len(rates) < 20:
        continue
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    zp = compute_zeropoint_state(df)
    if zp is None or len(zp) < 2:
        continue

    last = zp.iloc[-1]
    pos_val = int(last.get("pos", 0))
    if pos_val == 0:
        print(f"{sym:<10} FLAT")
        continue

    direction = "BUY" if pos_val == 1 else "SELL"
    entry = float(last["close"])
    atr_val = float(last["atr"])
    trailing_stop = float(last.get("xATRTrailingStop", 0))
    bars_since = int(last.get("bars_since_flip", 999))
    is_fresh = bars_since <= 3

    sl = trailing_stop
    buffer = atr_val * SL_BUFFER_PCT
    if direction == "BUY":
        sl -= buffer
        tp1 = entry + atr_val * TP1_MULT
    else:
        sl += buffer
        tp1 = entry - atr_val * TP1_MULT

    sl_dist = abs(entry - sl)
    tp_dist = abs(tp1 - entry)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0

    # Also check H1 confirmation
    rates_h1 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 200)
    h1_conf = ""
    if rates_h1 is not None and len(rates_h1) >= 20:
        df_h1 = pd.DataFrame(rates_h1)
        df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
        zp_h1 = compute_zeropoint_state(df_h1)
        if zp_h1 is not None and len(zp_h1) > 0:
            h1_pos = int(zp_h1.iloc[-1].get("pos", 0))
            if (direction == "BUY" and h1_pos == 1) or (direction == "SELL" and h1_pos == -1):
                h1_conf = " [H1 OK]"

    fresh_tag = "YES" if is_fresh else "no"
    rr_tag = f"{rr:.2f}"
    skip = ""
    if rr < 1.5:
        skip = " << SKIP (R:R < 1.5)"

    print(f"{sym:<10} {direction:<5} {rr_tag:>5} {fresh_tag:>7} {bars_since:>7} {entry:>12.5f} {sl:>12.5f} {tp1:>12.5f} {atr_val:>10.5f}{h1_conf}{skip}")

print("=" * 90)
mt5.shutdown()
