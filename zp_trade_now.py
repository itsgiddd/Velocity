#!/usr/bin/env python3
"""
Pure ZeroPoint Trade — 40% risk per trade.
Scans all symbols for ZeroPoint ATR trailing stop flips only.
"""

import sys, os, time, logging
sys.path.insert(0, os.path.dirname(__file__))

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime

from app.mt5_connector import MT5Connector
from app.zeropoint_signal import (
    ZeroPointEngine, ZeroPointSignal, ZEROPOINT_ENABLED_SYMBOLS,
    compute_zeropoint_state, ATR_PERIOD, ATR_MULTIPLIER, TP1_MULT,
    SWING_LOOKBACK, SL_BUFFER_PCT, SL_ATR_MIN_MULT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

FIXED_LOT = 0.40  # Fixed lot size
RISK_PCT = 0.40  # fallback only
SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "BTCUSD"]
SKIP_SYMBOLS = {"USDJPY"}  # Excluded per user


def generate_zp_signal_raw(symbol, df_h4, df_h1=None):
    """Generate ZP signal for any symbol — enters on active positions, not just flips."""
    import numpy as np
    zp = compute_zeropoint_state(df_h4)
    if zp is None or len(zp) < 2:
        return None

    last = zp.iloc[-1]
    pos = int(last.get("pos", 0))
    if pos == 0:
        return None

    direction = "BUY" if pos == 1 else "SELL"

    # Check freshness
    buy_sig = bool(last.get("buy_signal", False))
    sell_sig = bool(last.get("sell_signal", False))
    is_fresh = buy_sig or sell_sig
    if not is_fresh and len(zp) >= 2:
        prev = zp.iloc[-2]
        is_fresh = bool(prev.get("buy_signal", False)) or bool(prev.get("sell_signal", False))

    bars_in_pos = 1
    for idx in range(len(zp) - 2, -1, -1):
        if int(zp.iloc[idx].get("pos", 0)) == pos:
            bars_in_pos += 1
        else:
            break

    entry = float(last["close"])
    atr_val = float(last["atr"])
    if atr_val <= 0 or np.isnan(atr_val):
        return None

    trailing_stop = float(last.get("xATRTrailingStop", 0))
    if trailing_stop <= 0:
        return None

    # SL = trailing stop with buffer
    sl = trailing_stop
    buffer = atr_val * SL_BUFFER_PCT
    if direction == "BUY":
        sl = sl - buffer
        tp1 = entry + atr_val * TP1_MULT
    else:
        sl = sl + buffer
        tp1 = entry - atr_val * TP1_MULT

    # No room to profit — skip
    if direction == "BUY" and entry >= tp1:
        return None
    if direction == "SELL" and entry <= tp1:
        return None

    sl_dist = abs(entry - sl)
    tp_dist = abs(tp1 - entry)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0

    # H1 confirmation
    h1_conf = False
    if df_h1 is not None:
        zp_h1 = compute_zeropoint_state(df_h1)
        if zp_h1 is not None and len(zp_h1) > 0:
            h1_pos = int(zp_h1.iloc[-1].get("pos", 0))
            if direction == "BUY" and h1_pos == 1:
                h1_conf = True
            elif direction == "SELL" and h1_pos == -1:
                h1_conf = True

    if is_fresh:
        conf = 0.70 + (0.15 if h1_conf else 0.0) + min(rr * 0.05, 0.10)
    else:
        age_penalty = min(bars_in_pos * 0.02, 0.15)
        conf = 0.65 + (0.12 if h1_conf else 0.0) + min(rr * 0.05, 0.08) - age_penalty
    conf = max(0.40, min(conf, 0.98))

    if is_fresh and h1_conf:
        tier = "S"
    elif is_fresh:
        tier = "A"
    elif h1_conf and bars_in_pos <= 5:
        tier = "B"
    else:
        tier = "C"

    return ZeroPointSignal(
        symbol=symbol, direction=direction, entry_price=entry,
        stop_loss=sl, tp1=tp1, tp2=tp1, tp3=tp1,
        atr_value=atr_val, confidence=conf,
        signal_time=datetime.now(), timeframe="H4",
        tier=tier, trailing_stop=trailing_stop,
        risk_reward=rr,
    )


def main():
    print("=" * 50)
    print(f"ZeroPoint Pure Trade | Risk: {RISK_PCT:.0%}")
    print("=" * 50)

    conn = MT5Connector()
    if not conn.connect():
        print("ERROR: Cannot connect to MT5")
        return False

    acct = conn.get_account_info()
    balance = float(acct.get("balance", 0))
    print(f"Balance: ${balance:.2f} | Risk amount: ${balance * RISK_PCT:.2f}")

    zp = ZeroPointEngine()
    best_signal = None

    # Skip symbols that already have open positions
    open_positions = mt5.positions_get()
    open_symbols = set()
    if open_positions:
        for pos in open_positions:
            open_symbols.add(pos.symbol)
        if open_symbols:
            print(f"Open positions on: {', '.join(sorted(open_symbols))}")

    for symbol in SYMBOLS:
        if symbol in SKIP_SYMBOLS:
            print(f"  {symbol:8s}: excluded by user")
            continue

        # Skip if already have a position on this symbol
        sym_resolved_check = symbol
        if symbol in open_symbols or any(symbol in s or s in symbol for s in open_symbols):
            print(f"  {symbol:8s}: already have open position, skip")
            continue

        # Check if ZP enabled — also allow EURUSD (ZP neural model's best symbol)
        norm = symbol.upper().replace(".", "").replace("#", "")
        if norm not in ZEROPOINT_ENABLED_SYMBOLS and norm != "EURUSD":
            print(f"  {symbol:8s}: not ZP-enabled, skip")
            continue

        # Fetch H4 + H1 data
        sym_resolved = symbol
        sym_info_mt5 = mt5.symbol_info(symbol)
        if sym_info_mt5 is None:
            # Try common alternatives
            for alt in [symbol, symbol + ".raw", symbol[:3]]:
                sym_info_mt5 = mt5.symbol_info(alt)
                if sym_info_mt5 is not None:
                    sym_resolved = alt
                    break
        if sym_info_mt5 is None:
            print(f"  {symbol:8s}: not found on broker")
            continue

        rates_h4 = mt5.copy_rates_from_pos(sym_resolved, mt5.TIMEFRAME_H4, 0, 200)
        rates_h1 = mt5.copy_rates_from_pos(sym_resolved, mt5.TIMEFRAME_H1, 0, 200)

        df_h4 = None
        if rates_h4 is not None and len(rates_h4) >= 20:
            df_h4 = pd.DataFrame(rates_h4)
            df_h4["time"] = pd.to_datetime(df_h4["time"], unit="s")

        df_h1 = None
        if rates_h1 is not None and len(rates_h1) >= 20:
            df_h1 = pd.DataFrame(rates_h1)
            df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")

        if df_h4 is None:
            print(f"  {symbol:8s}: insufficient H4 data")
            continue

        # Try standard ZP engine first, fallback to raw for non-enabled symbols (EURUSD)
        sig = zp.generate_signal(sym_resolved, df_h4, df_h1)
        if sig is None and norm not in ZEROPOINT_ENABLED_SYMBOLS:
            sig = generate_zp_signal_raw(sym_resolved, df_h4, df_h1)
        if sig is None:
            print(f"  {symbol:8s}: no ZP flip")
            continue

        print(
            f"  {symbol:8s}: ZP {sig.direction} | entry={sig.entry_price:.5f} "
            f"SL={sig.stop_loss:.5f} TP1={sig.tp1:.5f} R:R={sig.risk_reward:.2f} "
            f"conf={sig.confidence:.0%} tier={sig.tier}"
        )

        if best_signal is None or sig.confidence > best_signal.confidence:
            best_signal = sig

    if best_signal is None:
        print("\nNo ZeroPoint signal found. No H4 flip on latest bar.")
        return False

    sig = best_signal
    symbol = sig.symbol
    direction = sig.direction
    entry = sig.entry_price
    sl = sig.stop_loss
    tp = sig.tp1  # Use TP1

    print(f"\n>>> BEST: {symbol} {direction}")
    print(f"    Entry:  {entry:.5f}")
    print(f"    SL:     {sl:.5f}")
    print(f"    TP1:    {tp:.5f}")
    print(f"    Conf:   {sig.confidence:.0%}")

    # --- Calculate lot size at 40% risk ---
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        print("ERROR: Cannot get symbol info")
        return False

    point = sym_info.point
    tick_size = sym_info.trade_tick_size or point
    tick_value = sym_info.trade_tick_value
    if tick_value <= 0:
        tick_value = sym_info.trade_contract_size * tick_size

    sl_distance = abs(entry - sl)
    sl_ticks = sl_distance / tick_size
    loss_per_lot = sl_ticks * tick_value
    risk_amount = balance * RISK_PCT

    if loss_per_lot <= 0:
        print("ERROR: Cannot calculate lot size (loss_per_lot=0)")
        return False

    lot_size = risk_amount / loss_per_lot
    vol_min = sym_info.volume_min
    vol_max = sym_info.volume_max
    vol_step = sym_info.volume_step
    lot_size = round(lot_size / vol_step) * vol_step
    lot_size = max(vol_min, min(vol_max, lot_size))

    lot_size = FIXED_LOT
    risk_amount = sl_ticks * tick_value * lot_size
    print(f"    Lot size: {lot_size:.2f} (FIXED)")
    print(f"    SL dist: {sl_distance:.5f} ({sl_ticks:.0f} ticks)")
    print(f"    Risk $:  ${risk_amount:.2f} ({risk_amount/balance*100:.0f}% of ${balance:.2f})")

    # --- Place the trade ---
    if direction == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = sym_info.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = sym_info.bid

    print(f"\n>>> SENDING {direction} {symbol} @ {lot_size:.2f} lots, price={price:.5f}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 777777,
        "comment": f"ZP-{direction}-{sig.confidence:.0%}",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    # Try fill modes
    for fill_mode in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
        request["type_filling"] = fill_mode
        result = mt5.order_send(request)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n*** TRADE EXECUTED ***")
            print(f"    Ticket:  {result.order}")
            print(f"    Symbol:  {symbol}")
            print(f"    Type:    {direction}")
            print(f"    Volume:  {lot_size}")
            print(f"    Price:   {result.price}")
            print(f"    SL:      {sl:.5f}")
            print(f"    TP:      {tp:.5f}")
            return True
        elif result is not None and result.retcode == 10018:
            print(f"Market closed (retcode=10018)")
            return False
        elif result is not None and result.retcode != 10030:
            print(f"Order failed: retcode={result.retcode}, comment={result.comment}")
            return False

    print("All fill modes failed")
    if result:
        print(f"Last result: retcode={result.retcode}, comment={result.comment}")
    return False


if __name__ == "__main__":
    try:
        success = main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()
        success = False
    finally:
        mt5.shutdown()
    raise SystemExit(0 if success else 1)
