"""
Analyze H4 ZeroPoint flip history — how predictable are flips?
Checks: flip frequency, trend duration distribution, move magnitude after flips,
and whether there are learnable patterns in the bars leading up to a flip.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from app.zeropoint_signal import compute_zeropoint_state

SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'BTCUSD']

if not mt5.initialize():
    print("MT5 init failed")
    sys.exit(1)

print("=" * 100)
print("H4 ZEROPOINT FLIP ANALYSIS — Historical Accuracy Check")
print("=" * 100)

all_durations = []
all_moves_pips = []
all_rr_at_flip = []

for sym in SYMBOLS:
    info = mt5.symbol_info(sym)
    if info is None:
        continue
    mt5.symbol_select(sym, True)
    point = info.point

    # Get max H4 data (up to 5000 bars = ~3.3 years)
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 5000)
    if rates is None or len(rates) < 100:
        print(f"{sym}: Not enough data")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    zp = compute_zeropoint_state(df)
    if zp is None:
        continue

    # Find all flips
    flips = []
    pos = zp["pos"].values
    close = zp["close"].values
    high = zp["high"].values
    low = zp["low"].values
    atr = zp["atr"].values
    stop = zp["xATRTrailingStop"].values

    for i in range(1, len(pos)):
        if pos[i] != pos[i-1] and pos[i] != 0:
            flips.append(i)

    if len(flips) < 5:
        print(f"{sym}: Only {len(flips)} flips — skipping")
        continue

    # Analyze each flip
    durations = []  # bars between consecutive flips
    moves_pips = []  # move after flip (favorable direction, in pips)
    moves_pct = []  # move after flip as % of entry
    win_count = 0
    total_count = 0

    for idx, flip_bar in enumerate(flips):
        # Duration: bars from this flip to next flip
        if idx + 1 < len(flips):
            duration = flips[idx + 1] - flip_bar
            durations.append(duration)
            all_durations.append(duration)

        direction = "BUY" if pos[flip_bar] == 1 else "SELL"
        entry = close[flip_bar]
        flip_atr = atr[flip_bar] if not np.isnan(atr[flip_bar]) else 0
        flip_stop = stop[flip_bar]

        # Look ahead: max favorable move within 20 bars after flip
        look_ahead = min(20, len(close) - flip_bar - 1)
        if look_ahead < 1 or flip_atr <= 0:
            continue

        if direction == "BUY":
            best_price = np.max(high[flip_bar+1 : flip_bar+1+look_ahead])
            worst_price = np.min(low[flip_bar+1 : flip_bar+1+look_ahead])
            move_pips = (best_price - entry) / point
            drawdown_pips = (entry - worst_price) / point
        else:
            best_price = np.min(low[flip_bar+1 : flip_bar+1+look_ahead])
            worst_price = np.max(high[flip_bar+1 : flip_bar+1+look_ahead])
            move_pips = (entry - best_price) / point
            drawdown_pips = (worst_price - entry) / point

        moves_pips.append(move_pips)
        all_moves_pips.append(move_pips)

        # Check if TP1 (2x ATR) would have been hit before SL
        tp_dist = flip_atr * 2.0  # TP1 mult
        sl_dist = abs(entry - flip_stop) + flip_atr * 0.001  # SL with buffer

        if direction == "BUY":
            tp_price = entry + tp_dist
            sl_price = flip_stop - flip_atr * 0.001
            # Scan bars to see which hit first
            hit_tp = False
            hit_sl = False
            for j in range(flip_bar+1, min(flip_bar+1+look_ahead, len(high))):
                if high[j] >= tp_price and not hit_sl:
                    hit_tp = True
                    break
                if low[j] <= sl_price:
                    hit_sl = True
                    break
        else:
            tp_price = entry - tp_dist
            sl_price = flip_stop + flip_atr * 0.001
            hit_tp = False
            hit_sl = False
            for j in range(flip_bar+1, min(flip_bar+1+look_ahead, len(low))):
                if low[j] <= tp_price and not hit_sl:
                    hit_tp = True
                    break
                if high[j] >= sl_price:
                    hit_sl = True
                    break

        if hit_tp:
            win_count += 1
        total_count += 1

        if sl_dist > 0:
            rr_actual = move_pips * point / sl_dist
            all_rr_at_flip.append(rr_actual)

    # Print per-symbol stats
    win_rate = win_count / total_count * 100 if total_count > 0 else 0
    avg_dur = np.mean(durations) if durations else 0
    med_dur = np.median(durations) if durations else 0
    avg_move = np.mean(moves_pips) if moves_pips else 0
    med_move = np.median(moves_pips) if moves_pips else 0

    days_of_data = len(df) * 4 / 24  # H4 bars to days

    print(f"\n{sym} ({len(df)} H4 bars = {days_of_data:.0f} days)")
    print(f"  Total flips: {len(flips)} ({len(flips)/days_of_data*7:.1f} per week)")
    print(f"  TP1 Win rate: {win_rate:.1f}% ({win_count}/{total_count})")
    print(f"  Trend duration: avg={avg_dur:.1f} bars, median={med_dur:.0f} bars ({med_dur*4:.0f} hours)")
    print(f"  Move after flip: avg={avg_move:.0f} pips, median={med_move:.0f} pips")
    if durations:
        print(f"  Duration distribution: min={min(durations)}, p25={np.percentile(durations,25):.0f}, "
              f"p50={np.percentile(durations,50):.0f}, p75={np.percentile(durations,75):.0f}, max={max(durations)}")

print("\n" + "=" * 100)
print("AGGREGATE STATS (all symbols combined)")
print("=" * 100)
if all_durations:
    print(f"Total flips analyzed: {len(all_durations)}")
    print(f"Trend duration: avg={np.mean(all_durations):.1f} bars, median={np.median(all_durations):.0f} bars")
    print(f"  p10={np.percentile(all_durations,10):.0f}, p25={np.percentile(all_durations,25):.0f}, "
          f"p50={np.percentile(all_durations,50):.0f}, p75={np.percentile(all_durations,75):.0f}, "
          f"p90={np.percentile(all_durations,90):.0f}")
    print(f"Move after flip: avg={np.mean(all_moves_pips):.0f} pips, median={np.median(all_moves_pips):.0f} pips")

    # Key question: is duration predictable?
    # If std/mean < 0.5, it's fairly regular; if > 1.0, it's very noisy
    cv = np.std(all_durations) / np.mean(all_durations) if np.mean(all_durations) > 0 else 999
    print(f"\nPredictability check:")
    print(f"  Duration CV (std/mean): {cv:.2f}")
    if cv < 0.6:
        print(f"  >> GOOD: Trend durations are fairly regular — a model CAN learn timing patterns")
    elif cv < 1.0:
        print(f"  >> MODERATE: Some regularity, model will help but won't be precise")
    else:
        print(f"  >> NOISY: Trend durations are very irregular — timing prediction will be rough")

    # Check if short trends (1-3 bars) are common — these are whipsaws
    whipsaws = sum(1 for d in all_durations if d <= 2)
    whipsaw_pct = whipsaws / len(all_durations) * 100
    print(f"  Whipsaws (duration <= 2 bars): {whipsaws} ({whipsaw_pct:.1f}%)")

    # Check move magnitude distribution
    positive_moves = [m for m in all_moves_pips if m > 0]
    if positive_moves:
        pct_positive = len(positive_moves) / len(all_moves_pips) * 100
        print(f"  Moves in favorable direction: {pct_positive:.1f}%")
        print(f"  Avg favorable move: {np.mean(positive_moves):.0f} pips")

mt5.shutdown()
print("\nDone.")
