"""Backtest the full app logic: H4 ZP + H1 confirm + quality score + ZP flip exit."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from app.zeropoint_signal import compute_zeropoint_state, TP1_MULT, SL_BUFFER_PCT

mt5.initialize()

symbols = ['AUDUSD', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDJPY']
results = []

for sym in symbols:
    info = mt5.symbol_info(sym)
    if info is None:
        continue
    mt5.symbol_select(sym, True)
    point = info.point

    rates_h4 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 5000)
    rates_h1 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 20000)
    if rates_h4 is None or len(rates_h4) < 200:
        continue

    df_h4 = pd.DataFrame(rates_h4)
    df_h4['time'] = pd.to_datetime(df_h4['time'], unit='s')
    zp_h4 = compute_zeropoint_state(df_h4)
    if zp_h4 is None:
        continue

    zp_h1 = None
    h1_times = None
    if rates_h1 is not None and len(rates_h1) >= 200:
        df_h1 = pd.DataFrame(rates_h1)
        df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
        zp_h1 = compute_zeropoint_state(df_h1)
        if zp_h1 is not None:
            h1_times = df_h1['time'].values

    pos = zp_h4['pos'].values
    close_arr = zp_h4['close'].values
    high_arr = zp_h4['high'].values
    low_arr = zp_h4['low'].values
    atr_arr = zp_h4['atr'].values
    stop_arr = zp_h4['xATRTrailingStop'].values
    bars_flip_arr = zp_h4['bars_since_flip'].values
    h4_times = df_h4['time'].values

    trades = []
    in_trade = False
    entry_price = entry_sl = entry_tp = 0
    entry_dir = ''
    entry_idx = 0
    entry_h1 = False
    entry_score = 0

    for i in range(50, len(pos)):
        cur_pos = pos[i]
        if cur_pos == 0:
            continue

        direction = 'BUY' if cur_pos == 1 else 'SELL'
        atr_val = atr_arr[i]
        if atr_val <= 0 or np.isnan(atr_val):
            continue
        trailing_stop = stop_arr[i]
        if trailing_stop <= 0 or np.isnan(trailing_stop):
            continue
        bars_since = int(bars_flip_arr[i]) if not np.isnan(bars_flip_arr[i]) else 999
        is_fresh = bars_since <= 3
        price = close_arr[i]

        # SL/TP
        sl = trailing_stop
        buffer = atr_val * SL_BUFFER_PCT
        if direction == 'BUY':
            sl -= buffer
            tp1 = price + atr_val * TP1_MULT
        else:
            sl += buffer
            tp1 = price - atr_val * TP1_MULT

        sl_dist = abs(price - sl)
        tp_dist = abs(tp1 - price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        # H1 confirmation
        h1_conf = False
        if zp_h1 is not None and h1_times is not None:
            h4t = h4_times[i]
            mask = h1_times <= h4t
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                last_h1 = idxs[-1]
                h1_pos = int(zp_h1.iloc[last_h1]['pos'])
                if direction == 'BUY' and h1_pos == 1:
                    h1_conf = True
                elif direction == 'SELL' and h1_pos == -1:
                    h1_conf = True

        # Quality score
        score = 0.0
        score += min(rr, 5.0) * 20
        if is_fresh:
            score += 30
        if h1_conf:
            score += 20
        if not is_fresh:
            score -= min(bars_since * 1.0, 40)
        score = max(0, score)

        # Check ZP flip exit for existing trade
        if in_trade and entry_dir != direction:
            exit_price = price
            if entry_dir == 'BUY':
                pnl = (exit_price - entry_price) / point
            else:
                pnl = (entry_price - exit_price) / point
            trades.append({
                'dir': entry_dir, 'pnl_pips': pnl,
                'duration': i - entry_idx, 'h1_conf': entry_h1,
                'score': entry_score, 'exit': 'ZP_FLIP',
            })
            in_trade = False

        # Check SL/TP for existing trade
        if in_trade:
            if entry_dir == 'BUY':
                if low_arr[i] <= entry_sl:
                    pnl = (entry_sl - entry_price) / point
                    trades.append({'dir': entry_dir, 'pnl_pips': pnl, 'duration': i - entry_idx, 'h1_conf': entry_h1, 'score': entry_score, 'exit': 'SL'})
                    in_trade = False
                elif high_arr[i] >= entry_tp:
                    pnl = (entry_tp - entry_price) / point
                    trades.append({'dir': entry_dir, 'pnl_pips': pnl, 'duration': i - entry_idx, 'h1_conf': entry_h1, 'score': entry_score, 'exit': 'TP'})
                    in_trade = False
            else:
                if high_arr[i] >= entry_sl:
                    pnl = (entry_price - entry_sl) / point
                    trades.append({'dir': entry_dir, 'pnl_pips': pnl, 'duration': i - entry_idx, 'h1_conf': entry_h1, 'score': entry_score, 'exit': 'SL'})
                    in_trade = False
                elif low_arr[i] <= entry_tp:
                    pnl = (entry_price - entry_tp) / point
                    trades.append({'dir': entry_dir, 'pnl_pips': pnl, 'duration': i - entry_idx, 'h1_conf': entry_h1, 'score': entry_score, 'exit': 'TP'})
                    in_trade = False

        # Enter on fresh flip with score > 0
        if not in_trade and is_fresh and score > 0:
            in_trade = True
            entry_price = price
            entry_dir = direction
            entry_sl = sl
            entry_tp = tp1
            entry_idx = i
            entry_h1 = h1_conf
            entry_score = score

    if not trades:
        continue

    wins = [t for t in trades if t['pnl_pips'] > 0]
    losses = [t for t in trades if t['pnl_pips'] <= 0]
    total_pips = sum(t['pnl_pips'] for t in trades)
    avg_win = np.mean([t['pnl_pips'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pips'] for t in losses]) if losses else 0
    wr = len(wins) / len(trades) * 100
    gross_win = sum(t['pnl_pips'] for t in wins)
    gross_loss = abs(sum(t['pnl_pips'] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 999

    tp_exits = len([t for t in trades if t['exit'] == 'TP'])
    sl_exits = len([t for t in trades if t['exit'] == 'SL'])
    flip_exits = len([t for t in trades if t['exit'] == 'ZP_FLIP'])
    h1_trades = [t for t in trades if t['h1_conf']]
    h1_wins = len([t for t in h1_trades if t['pnl_pips'] > 0])
    h1_wr = h1_wins / len(h1_trades) * 100 if h1_trades else 0
    no_h1 = [t for t in trades if not t['h1_conf']]
    no_h1_wins = len([t for t in no_h1 if t['pnl_pips'] > 0])
    no_h1_wr = no_h1_wins / len(no_h1) * 100 if no_h1 else 0

    if 'JPY' in sym:
        pip_val = 0.04 * 100000 * 0.01 / close_arr[-1]
    else:
        pip_val = 0.04 * 100000 * 0.0001
    total_dollar = total_pips * pip_val

    results.append({
        'symbol': sym, 'trades': len(trades), 'wr': wr, 'pf': pf,
        'total_pips': total_pips, 'total_dollar': total_dollar,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'tp': tp_exits, 'sl': sl_exits, 'flip': flip_exits,
        'h1_count': len(h1_trades), 'h1_wr': h1_wr,
        'no_h1_count': len(no_h1), 'no_h1_wr': no_h1_wr,
    })

mt5.shutdown()

results.sort(key=lambda x: x['total_pips'], reverse=True)

print('=' * 105)
print('FULL APP LOGIC BACKTEST - H4 ZP + H1 Confirm + Quality Score + SL/TP + ZP Flip Exit')
print('Entry: fresh flip (<=3 bars), score > 0 | Exit: SL, TP, or ZP flip')
print('0.04 lots | 833 days of data')
print('=' * 105)
print()
print(f'{"Symbol":<10} {"Trades":>6} {"WR":>6} {"PF":>6} {"Pips":>10} {"~USD":>9} {"TP":>4} {"SL":>4} {"Flip":>5} {"H1 conf":>8} {"H1 WR":>6} {"noH1 WR":>7}')
print('-' * 105)
for r in results:
    h1_str = f'{r["h1_count"]}/{r["trades"]}'
    print(f'{r["symbol"]:<10} {r["trades"]:>6} {r["wr"]:>5.1f}% {r["pf"]:>6.2f} {r["total_pips"]:>+10.0f} {r["total_dollar"]:>+9.1f} {r["tp"]:>4} {r["sl"]:>4} {r["flip"]:>5} {h1_str:>8} {r["h1_wr"]:>5.0f}% {r["no_h1_wr"]:>6.0f}%')
print('-' * 105)
tp = sum(r['total_pips'] for r in results)
td = sum(r['total_dollar'] for r in results)
print(f'{"TOTAL":<10} {sum(r["trades"] for r in results):>6} {"":>6} {"":>6} {tp:>+10.0f} {td:>+9.1f}')
print()
print('TOP PERFORMERS:')
for i, r in enumerate(results[:3]):
    print(f'  #{i+1} {r["symbol"]}: {r["total_pips"]:+.0f} pips (~${r["total_dollar"]:+.0f}) | WR={r["wr"]:.0f}% PF={r["pf"]:.2f} | H1 conf WR={r["h1_wr"]:.0f}%')
