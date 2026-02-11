"""
Backtest: CASCADE strategy profitability (FIXED).
Strategy: When M15+H1 both flip AGAINST H4, enter in M15/H1 direction.
SL/TP calculated from the M15/H1 perspective using H1 ATR.

Simulates $300 account, 0.10 lots fixed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from app.zeropoint_signal import compute_zeropoint_state

SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY']

if not mt5.initialize():
    print("MT5 init failed")
    sys.exit(1)

STARTING_BALANCE = 300.0
LOT_SIZE = 0.10
SPREAD_PIPS = 2

print("=" * 100)
print("CASCADE STRATEGY BACKTEST (FIXED SL/TP)")
print(f"Account: ${STARTING_BALANCE} | Lots: {LOT_SIZE}")
print("Entry: M15+H1 flip against H4 -> trade M15/H1 direction")
print("SL: 1.5x H1 ATR behind entry | TP: 2x H1 ATR in trade direction")
print("=" * 100)

all_trades = []

for sym in SYMBOLS:
    info = mt5.symbol_info(sym)
    if info is None:
        continue
    mt5.symbol_select(sym, True)
    point = info.point

    # Pip value for USD account (approximate)
    if sym.endswith("USD"):
        pip_value_per_lot = 10.0
    elif sym.startswith("USD"):
        tick = mt5.symbol_info_tick(sym)
        p = tick.ask if tick else 1.0
        pip_value_per_lot = 10.0 / p if p > 0 else 10.0
    elif "JPY" in sym:
        tick = mt5.symbol_info_tick(sym)
        p = tick.ask if tick else 150.0
        pip_value_per_lot = 1000.0 / p if p > 0 else 6.67
    else:
        pip_value_per_lot = 10.0

    # Get data
    rates_h4 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 1100)
    rates_h1 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 4400)
    rates_m15 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 17600)

    if rates_h4 is None or rates_h1 is None or rates_m15 is None:
        continue
    if len(rates_h4) < 200 or len(rates_h1) < 800 or len(rates_m15) < 3200:
        continue

    df_h4 = pd.DataFrame(rates_h4)
    df_h4["time"] = pd.to_datetime(df_h4["time"], unit="s")
    df_h1 = pd.DataFrame(rates_h1)
    df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
    df_m15 = pd.DataFrame(rates_m15)
    df_m15["time"] = pd.to_datetime(df_m15["time"], unit="s")

    zp_h4 = compute_zeropoint_state(df_h4)
    zp_h1 = compute_zeropoint_state(df_h1)
    zp_m15 = compute_zeropoint_state(df_m15)

    if zp_h4 is None or zp_h1 is None or zp_m15 is None:
        continue

    h4_pos = zp_h4["pos"].values
    h4_times = zp_h4["time"].values

    h1_pos = zp_h1["pos"].values
    h1_close = zp_h1["close"].values
    h1_high = zp_h1["high"].values
    h1_low = zp_h1["low"].values
    h1_atr = zp_h1["atr"].values
    h1_times = zp_h1["time"].values
    h1_stop = zp_h1["xATRTrailingStop"].values

    m15_pos = zp_m15["pos"].values
    m15_times = zp_m15["time"].values

    sym_trades = []
    last_trade_time = pd.Timestamp("2000-01-01")

    for i in range(1, len(h1_pos)):
        # Check for H1 flip
        if h1_pos[i] == h1_pos[i-1] or h1_pos[i] == 0:
            continue

        h1_time = pd.Timestamp(h1_times[i])

        # Cooldown: don't take trades within 4 hours of last
        if (h1_time - last_trade_time).total_seconds() < 4 * 3600:
            continue

        h1_dir = h1_pos[i]  # +1 (BUY) or -1 (SELL)
        entry_atr = h1_atr[i]
        if np.isnan(entry_atr) or entry_atr <= 0:
            continue

        # Find current H4 position
        h4_idx = None
        for j in range(len(h4_times) - 1, -1, -1):
            if pd.Timestamp(h4_times[j]) <= h1_time:
                h4_idx = j
                break
        if h4_idx is None:
            continue

        h4_dir = h4_pos[h4_idx]

        # CASCADE: H1 must oppose H4
        if h1_dir == h4_dir:
            continue

        # Check M15 alignment
        m15_aligned = False
        for j in range(len(m15_times) - 1, -1, -1):
            if pd.Timestamp(m15_times[j]) <= h1_time:
                if m15_pos[j] == h1_dir:
                    m15_aligned = True
                break
        if not m15_aligned:
            continue

        # --- CASCADE SIGNAL: M15+H1 oppose H4 ---
        entry_dir = "BUY" if h1_dir == 1 else "SELL"
        entry_price = h1_close[i]

        # SL/TP from H1 perspective
        if entry_dir == "BUY":
            sl = entry_price - entry_atr * 1.5  # 1.5x ATR stop
            tp = entry_price + entry_atr * 2.0  # 2x ATR target
        else:
            sl = entry_price + entry_atr * 1.5
            tp = entry_price - entry_atr * 2.0

        sl_dist = abs(entry_price - sl)
        tp_dist = abs(tp - entry_price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        if rr < 1.0:
            continue

        # Simulate on H1 bars (check every H1 candle after entry)
        hit = None
        exit_price = entry_price
        bars_held = 0

        for j in range(i + 1, min(i + 100, len(h1_high))):
            bars_held = j - i
            if entry_dir == "BUY":
                if h1_high[j] >= tp:
                    hit = "TP"
                    exit_price = tp
                    break
                if h1_low[j] <= sl:
                    hit = "SL"
                    exit_price = sl
                    break
            else:
                if h1_low[j] <= tp:
                    hit = "TP"
                    exit_price = tp
                    break
                if h1_high[j] >= sl:
                    hit = "SL"
                    exit_price = sl
                    break

        if hit is None:
            exit_price = h1_close[min(i + 100, len(h1_close) - 1)]
            hit = "EXPIRE"

        # P/L
        if entry_dir == "BUY":
            pnl_pips = (exit_price - entry_price) / point - SPREAD_PIPS
        else:
            pnl_pips = (entry_price - exit_price) / point - SPREAD_PIPS

        pnl_dollars = pnl_pips * pip_value_per_lot * LOT_SIZE

        trade = {
            "symbol": sym,
            "direction": entry_dir,
            "entry": entry_price,
            "exit": exit_price,
            "sl_dist_pips": abs(entry_price - sl) / point,
            "tp_dist_pips": abs(tp - entry_price) / point,
            "rr": rr,
            "result": hit,
            "pnl_pips": pnl_pips,
            "pnl_usd": pnl_dollars,
            "bars_held": bars_held,
            "entry_time": h1_time,
        }
        sym_trades.append(trade)
        all_trades.append(trade)
        last_trade_time = h1_time

    if sym_trades:
        wins = [t for t in sym_trades if t["pnl_usd"] > 0]
        losses = [t for t in sym_trades if t["pnl_usd"] <= 0]
        total_pnl = sum(t["pnl_usd"] for t in sym_trades)
        win_rate = len(wins) / len(sym_trades) * 100
        avg_win = np.mean([t["pnl_usd"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_usd"] for t in losses]) if losses else 0
        gross_w = sum(t["pnl_usd"] for t in wins) if wins else 0
        gross_l = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0.01
        pf = gross_w / gross_l if gross_l > 0 else 999

        print(f"\n{sym}: {len(sym_trades)} trades | WR={win_rate:.0f}% | "
              f"PF={pf:.2f} | Total=${total_pnl:+.2f}")
        print(f"  Avg win=${avg_win:+.2f} | Avg loss=${avg_loss:+.2f} | "
              f"Avg hold={np.mean([t['bars_held'] for t in sym_trades]):.0f}h1 bars")

        tp_c = sum(1 for t in sym_trades if t["result"] == "TP")
        sl_c = sum(1 for t in sym_trades if t["result"] == "SL")
        ex_c = sum(1 for t in sym_trades if t["result"] == "EXPIRE")
        print(f"  TP={tp_c} SL={sl_c} EXPIRE={ex_c}")

print("\n" + "=" * 100)
print("OVERALL BACKTEST RESULTS")
print("=" * 100)

if all_trades:
    all_trades.sort(key=lambda t: t["entry_time"])
    total_trades = len(all_trades)
    wins = [t for t in all_trades if t["pnl_usd"] > 0]
    losses = [t for t in all_trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in all_trades)
    win_rate = len(wins) / total_trades * 100
    avg_win = np.mean([t["pnl_usd"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_usd"] for t in losses]) if losses else 0
    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0.01
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    print(f"\nTotal trades: {total_trades}")
    print(f"Win rate: {win_rate:.1f}% ({len(wins)}/{total_trades})")
    print(f"Profit factor: {pf:.2f}")
    print(f"Total P/L: ${total_pnl:+.2f}")
    print(f"Avg win: ${avg_win:+.2f} | Avg loss: ${avg_loss:+.2f}")

    balance = STARTING_BALANCE
    peak = balance
    max_dd = 0
    worst_dd_usd = 0

    for t in all_trades:
        balance += t["pnl_usd"]
        if balance > peak:
            peak = balance
        dd_usd = peak - balance
        dd_pct = dd_usd / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd:
            max_dd = dd_pct
            worst_dd_usd = dd_usd

    print(f"\nStarting: ${STARTING_BALANCE:.2f} -> Final: ${balance:.2f}")
    print(f"Return: {(balance - STARTING_BALANCE) / STARTING_BALANCE * 100:+.1f}%")
    print(f"Max drawdown: {max_dd:.1f}% (${worst_dd_usd:.2f})")

    # Monthly
    months = {}
    for t in all_trades:
        mk = t["entry_time"].strftime("%Y-%m")
        if mk not in months:
            months[mk] = {"trades": 0, "pnl": 0, "wins": 0}
        months[mk]["trades"] += 1
        months[mk]["pnl"] += t["pnl_usd"]
        if t["pnl_usd"] > 0:
            months[mk]["wins"] += 1

    print(f"\n{'Month':<10} {'Trades':>7} {'WR':>6} {'P/L':>10}")
    for m in sorted(months.keys()):
        d = months[m]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
        print(f"  {m:<10} {d['trades']:>7} {wr:>5.0f}% ${d['pnl']:>+8.2f}")

    pnls = [t["pnl_usd"] for t in all_trades]
    print(f"\nP/L range: ${min(pnls):+.2f} to ${max(pnls):+.2f} | Median: ${np.median(pnls):+.2f}")

    tp_c = sum(1 for t in all_trades if t["result"] == "TP")
    sl_c = sum(1 for t in all_trades if t["result"] == "SL")
    ex_c = sum(1 for t in all_trades if t["result"] == "EXPIRE")
    print(f"Exits: TP={tp_c} ({tp_c/total_trades*100:.0f}%) | "
          f"SL={sl_c} ({sl_c/total_trades*100:.0f}%) | "
          f"EXPIRE={ex_c} ({ex_c/total_trades*100:.0f}%)")

mt5.shutdown()
print("\nDone.")
