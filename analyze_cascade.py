"""
Analyze the M15 -> H1 -> H4 flip cascade.
Key question: How early does M15/H1 flip before H4 follows?
This proves whether lower TF flips are a reliable LEADING indicator for H4.
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

print("=" * 100)
print("M15 -> H1 -> H4 FLIP CASCADE ANALYSIS")
print("How early do lower timeframes predict H4 flips?")
print("=" * 100)

all_h1_lead = []   # H1 minutes before H4 flip
all_m15_lead = []  # M15 minutes before H4 flip
all_h1_accuracy = []  # Did H1 flip in same direction before H4?
all_m15_accuracy = []

for sym in SYMBOLS:
    info = mt5.symbol_info(sym)
    if info is None:
        continue
    mt5.symbol_select(sym, True)

    # Get aligned data — 180 days
    rates_h4 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 1100)
    rates_h1 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 4400)
    rates_m15 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 17600)

    if rates_h4 is None or rates_h1 is None or rates_m15 is None:
        continue
    if len(rates_h4) < 100 or len(rates_h1) < 400 or len(rates_m15) < 1600:
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

    # Find H4 flips
    h4_pos = zp_h4["pos"].values
    h4_times = zp_h4["time"].values

    h4_flips = []
    for i in range(1, len(h4_pos)):
        if h4_pos[i] != h4_pos[i-1] and h4_pos[i] != 0:
            flip_dir = "BUY" if h4_pos[i] == 1 else "SELL"
            h4_flips.append((h4_times[i], flip_dir))

    # For each H4 flip, look back in H1 and M15 to find when they flipped to same direction
    h1_pos = zp_h1["pos"].values
    h1_times = zp_h1["time"].values
    m15_pos = zp_m15["pos"].values
    m15_times = zp_m15["time"].values

    h1_leads = []
    m15_leads = []
    h1_predicted = 0
    m15_predicted = 0
    total_flips = 0

    for h4_time, h4_dir in h4_flips:
        h4_ts = pd.Timestamp(h4_time)
        target_pos = 1 if h4_dir == "BUY" else -1
        total_flips += 1

        # --- Check H1: when did it last flip to same direction BEFORE H4? ---
        # Look back up to 24 H1 bars (24 hours) before the H4 flip
        h1_lead_minutes = None
        for j in range(len(h1_times) - 1, 0, -1):
            h1_ts = pd.Timestamp(h1_times[j])
            if h1_ts > h4_ts:
                continue
            if h1_ts < h4_ts - pd.Timedelta(hours=48):
                break
            # Check if this H1 bar is a flip to the same direction
            if h1_pos[j] == target_pos and h1_pos[j-1] != target_pos:
                h1_lead_minutes = (h4_ts - h1_ts).total_seconds() / 60
                break

        if h1_lead_minutes is not None and h1_lead_minutes > 0:
            h1_leads.append(h1_lead_minutes)
            all_h1_lead.append(h1_lead_minutes)
            h1_predicted += 1

        # --- Check M15: when did it last flip to same direction BEFORE H4? ---
        m15_lead_minutes = None
        for j in range(len(m15_times) - 1, 0, -1):
            m15_ts = pd.Timestamp(m15_times[j])
            if m15_ts > h4_ts:
                continue
            if m15_ts < h4_ts - pd.Timedelta(hours=48):
                break
            if m15_pos[j] == target_pos and m15_pos[j-1] != target_pos:
                m15_lead_minutes = (h4_ts - m15_ts).total_seconds() / 60
                break

        if m15_lead_minutes is not None and m15_lead_minutes > 0:
            m15_leads.append(m15_lead_minutes)
            all_m15_lead.append(m15_lead_minutes)
            m15_predicted += 1

    # Also check: at the moment of H4 flip, what % of time was H1/M15 already aligned?
    h1_aligned_at_flip = 0
    m15_aligned_at_flip = 0
    for h4_time, h4_dir in h4_flips:
        h4_ts = pd.Timestamp(h4_time)
        target_pos = 1 if h4_dir == "BUY" else -1

        # Find H1 position at H4 flip time
        for j in range(len(h1_times) - 1, -1, -1):
            if pd.Timestamp(h1_times[j]) <= h4_ts:
                if h1_pos[j] == target_pos:
                    h1_aligned_at_flip += 1
                break

        # Find M15 position at H4 flip time
        for j in range(len(m15_times) - 1, -1, -1):
            if pd.Timestamp(m15_times[j]) <= h4_ts:
                if m15_pos[j] == target_pos:
                    m15_aligned_at_flip += 1
                break

    h1_pct = h1_predicted / total_flips * 100 if total_flips > 0 else 0
    m15_pct = m15_predicted / total_flips * 100 if total_flips > 0 else 0
    h1_align_pct = h1_aligned_at_flip / total_flips * 100 if total_flips > 0 else 0
    m15_align_pct = m15_aligned_at_flip / total_flips * 100 if total_flips > 0 else 0

    print(f"\n{sym} ({total_flips} H4 flips)")
    print(f"  H1 flipped first: {h1_pct:.0f}% ({h1_predicted}/{total_flips})")
    if h1_leads:
        print(f"    Lead time: avg={np.mean(h1_leads):.0f}min, median={np.median(h1_leads):.0f}min "
              f"({np.median(h1_leads)/60:.1f}h)")
    print(f"  M15 flipped first: {m15_pct:.0f}% ({m15_predicted}/{total_flips})")
    if m15_leads:
        print(f"    Lead time: avg={np.mean(m15_leads):.0f}min, median={np.median(m15_leads):.0f}min "
              f"({np.median(m15_leads)/60:.1f}h)")
    print(f"  At H4 flip moment — H1 aligned: {h1_align_pct:.0f}%, M15 aligned: {m15_align_pct:.0f}%")

print("\n" + "=" * 100)
print("AGGREGATE CASCADE STATS")
print("=" * 100)

if all_h1_lead:
    print(f"\nH1 -> H4 cascade ({len(all_h1_lead)} events):")
    print(f"  H1 flips before H4: {len(all_h1_lead)} times")
    print(f"  Lead time: avg={np.mean(all_h1_lead):.0f}min ({np.mean(all_h1_lead)/60:.1f}h), "
          f"median={np.median(all_h1_lead):.0f}min ({np.median(all_h1_lead)/60:.1f}h)")
    print(f"  p25={np.percentile(all_h1_lead,25):.0f}min, "
          f"p50={np.percentile(all_h1_lead,50):.0f}min, "
          f"p75={np.percentile(all_h1_lead,75):.0f}min")

if all_m15_lead:
    print(f"\nM15 -> H4 cascade ({len(all_m15_lead)} events):")
    print(f"  M15 flips before H4: {len(all_m15_lead)} times")
    print(f"  Lead time: avg={np.mean(all_m15_lead):.0f}min ({np.mean(all_m15_lead)/60:.1f}h), "
          f"median={np.median(all_m15_lead):.0f}min ({np.median(all_m15_lead)/60:.1f}h)")
    print(f"  p25={np.percentile(all_m15_lead,25):.0f}min, "
          f"p50={np.percentile(all_m15_lead,50):.0f}min, "
          f"p75={np.percentile(all_m15_lead,75):.0f}min")

# Key insight: what % of H4 flips were preceded by BOTH H1 and M15?
if all_h1_lead and all_m15_lead:
    total_h4 = sum(len(h4f) for h4f in [h4_flips])  # approximate
    print(f"\nKEY INSIGHT:")
    print(f"  If M15+H1 both flip same direction -> H4 flip is likely incoming")
    print(f"  This is the cascade signal the FlipPredictor should learn")

mt5.shutdown()
print("\nDone.")
