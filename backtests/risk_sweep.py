#!/usr/bin/env python3
"""Risk level sweep — test 30-50% risk across H1/H3/H4."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import MetaTrader5 as mt5
import backtests.backtest_multi_tf as btf
from backtests.backtest_multi_tf import (
    get_symbol_specs, get_conversion_rates, run_backtest_for_tf,
    FETCH_BARS, STARTING_BALANCE, TIMEFRAMES,
)

mt5.initialize()
sym_specs = get_symbol_specs()
conv_rates = get_conversion_rates()

# Find common start from H1 (least history)
resolved = sym_specs.get("EURUSD", {}).get("resolved", "EURUSD")
rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H1, 0, FETCH_BARS)
df_tmp = pd.DataFrame(rates)
df_tmp["time"] = pd.to_datetime(df_tmp["time"], unit="s")
common_start = df_tmp["time"].iloc[0] + pd.Timedelta(days=7)

print("=" * 130)
print(f"  RISK LEVEL SWEEP (all from {common_start.strftime('%Y-%m-%d')}, start=${STARTING_BALANCE:.0f})")
print("=" * 130)
print(f"  {'Risk':>5} | {'TF':>3} | {'Trades':>6} | {'WR':>6} | {'Losers':>6} | {'PF':>6} | {'Final Balance':>16} | {'Max DD %':>8} | {'Max DD $':>12} | {'$100K in':>10}")
print("  " + "-" * 118)

for risk_pct in [0.30, 0.35, 0.40, 0.45, 0.50]:
    btf.BASE_RISK_PCT = risk_pct
    btf.MAX_RISK_PCT = min(risk_pct + 0.10, 0.70)

    for tf_name in ["H1", "H3", "H4"]:
        tf_id = TIMEFRAMES[tf_name]
        r = run_backtest_for_tf(tf_name, tf_id, sym_specs, conv_rates, start_date=common_start)
        if r:
            t100k = f"{r['time_to_100k_days']}d" if r['time_to_100k_days'] else "N/A"
            print(f"  {risk_pct*100:>4.0f}% | {tf_name:>3} | {r['trades']:>6} | {r['win_rate']:>5.1f}% | "
                  f"{r['losses']:>6} | {r['profit_factor']:>5.2f} | ${r['final_balance']:>14,.0f} | "
                  f"{r['max_dd_pct']:>7.1f}% | ${r['max_dd']:>10,.0f} | {t100k:>10}")
    print()

mt5.shutdown()

print("=" * 130)
print("  NOTE: DD % = peak-to-trough as % of peak balance (not starting balance)")
print("  All PnL converted to USD. Lot caps: $500->0.10, $1K->0.20, $3K->0.50, $5K->1.0, $10K->2.0, $50K->5.0, >$50K->10.0")
print("=" * 130)
