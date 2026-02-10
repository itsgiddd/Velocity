"""Scan all symbols for ZeroPoint state — show current position + how recent the last flip was."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from app.zeropoint_signal import compute_zeropoint_state, ZEROPOINT_ENABLED_SYMBOLS

mt5.initialize()
info = mt5.account_info()
print(f"Balance: ${info.balance:.2f} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print()

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "EURJPY", "GBPJPY"]

print(f"{'Symbol':8s} | {'H4 Pos':7s} | {'Flip ago':9s} | {'In pos':7s} | {'Direction':9s} | {'Stop':>10s} | {'Price':>10s} | {'Dist':>8s} | {'H1 Pos':7s} | {'H1 Flip':8s}")
print("-" * 110)

for symbol in SYMBOLS:
    sym = symbol
    si = mt5.symbol_info(sym)
    if si is None:
        for alt in [symbol + ".raw", symbol[:3]]:
            si = mt5.symbol_info(alt)
            if si: sym = alt; break
    if si is None:
        print(f"  {symbol:8s}: not found"); continue

    # H4
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 200)
    if rates is None or len(rates) < 20:
        print(f"  {symbol:8s}: no H4 data"); continue
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    zp = compute_zeropoint_state(df)
    if zp is None:
        print(f"  {symbol:8s}: ZP failed"); continue

    last = zp.iloc[-1]
    pos = int(last["pos"])
    bars_flip = int(last["bars_since_flip"])
    bars_in = int(last["bars_in_position"])
    stop = float(last["xATRTrailingStop"])
    price = float(last["close"])
    atr = float(last["atr"])
    direction = "BULLISH" if pos == 1 else "BEARISH"
    dist_pips = abs(price - stop)

    # Check if flip happened on last or second-to-last bar
    buy_last = bool(zp.iloc[-1].get("buy_signal", False))
    sell_last = bool(zp.iloc[-1].get("sell_signal", False))
    buy_prev = bool(zp.iloc[-2].get("buy_signal", False))
    sell_prev = bool(zp.iloc[-2].get("sell_signal", False))
    flip_note = ""
    if buy_last or sell_last:
        flip_note = " *** FRESH FLIP ***"
    elif buy_prev or sell_prev:
        flip_note = " * recent flip (1 bar ago)"

    # H1
    rates_h1 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 200)
    h1_pos_str = "---"
    h1_flip_str = "---"
    if rates_h1 is not None and len(rates_h1) >= 20:
        df_h1 = pd.DataFrame(rates_h1)
        df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
        zp_h1 = compute_zeropoint_state(df_h1)
        if zp_h1 is not None:
            h1_last = zp_h1.iloc[-1]
            h1_pos = int(h1_last["pos"])
            h1_pos_str = "BULL" if h1_pos == 1 else "BEAR"
            h1_flip_str = f"{int(h1_last['bars_since_flip'])} bars"

    print(
        f"{symbol:8s} | {'BULL' if pos==1 else 'BEAR':7s} | {bars_flip:3d} bars  | {bars_in:3d} bars | {direction:9s} | {stop:10.5f} | {price:10.5f} | {dist_pips:.5f} | {h1_pos_str:7s} | {h1_flip_str:8s}{flip_note}"
    )

print()
print("*** FRESH FLIP = signal on current bar (tradeable)")
print("* recent flip  = signal 1 bar ago (still in early move, tradeable)")
print("No mark         = position ongoing, no new entry signal")

mt5.shutdown()
