#!/usr/bin/env python3
"""Risk sweep with raised lot caps to allow true compounding."""

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

# Find common start
resolved = sym_specs.get("EURUSD", {}).get("resolved", "EURUSD")
rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H1, 0, FETCH_BARS)
df_tmp = pd.DataFrame(rates)
df_tmp["time"] = pd.to_datetime(df_tmp["time"], unit="s")
common_start = df_tmp["time"].iloc[0] + pd.Timedelta(days=7)

# ECN broker max = 100 lots. Remove artificial caps, let broker limit be the ceiling.
AGGRESSIVE_LOT_CAP = [
    (float('inf'), 100.00),
]

print("=" * 140)
print(f"  RISK SWEEP — ECN 100 LOT MAX (no artificial caps)")
print(f"  Period: {common_start.strftime('%Y-%m-%d')} -> present | Start: ${STARTING_BALANCE:.0f}")
print(f"  Lot cap: 100.00 (broker ECN max)")
print("=" * 140)

print(f"\n  {'Risk':>5} | {'TF':>3} | {'Trades':>6} | {'WR':>6} | {'Losers':>6} | {'PF':>6} | {'Final Balance':>16} | {'Max DD %':>8} | {'Max DD $':>14} | {'$100K in':>9} | {'Yr1 Bal':>14}")
print("  " + "-" * 128)

for risk_pct in [0.30, 0.35, 0.40, 0.45, 0.50]:
    btf.BASE_RISK_PCT = risk_pct
    btf.MAX_RISK_PCT = min(risk_pct + 0.10, 0.70)
    btf.LOT_CAP_TABLE = AGGRESSIVE_LOT_CAP
    btf.HIGH_BALANCE_CAP_RISK = risk_pct  # Don't throttle above $50K

    for tf_name in ["H1", "H3", "H4"]:
        tf_id = TIMEFRAMES[tf_name]
        r = run_backtest_for_tf(tf_name, tf_id, sym_specs, conv_rates, start_date=common_start)
        if r:
            t100k = f"{r['time_to_100k_days']}d" if r['time_to_100k_days'] else "N/A"
            yr1 = f"${r['year1_balance']:>12,.0f}"
            print(f"  {risk_pct*100:>4.0f}% | {tf_name:>3} | {r['trades']:>6} | {r['win_rate']:>5.1f}% | "
                  f"{r['losses']:>6} | {r['profit_factor']:>5.2f} | ${r['final_balance']:>14,.0f} | "
                  f"{r['max_dd_pct']:>7.1f}% | ${r['max_dd']:>12,.0f} | {t100k:>9} | {yr1}")
    print()

mt5.shutdown()
print("=" * 140)
