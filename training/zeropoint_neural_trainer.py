#!/usr/bin/env python3
"""
ZeroPoint Neural Trainer
========================

Neural model that learns the ZeroPoint ATR trailing stop strategy.
Uses H4 ZeroPoint direction as the labeling authority so the model can
identify H4-quality signals while operating on M15 for higher trade frequency.

Key innovations:
- 62-dim feature vector: 38 base + 15 ZeroPoint (5 per TF x M15/H1/H4) + 9 one-hot
- H4 ZeroPoint + price confirmation labeling (not just future return)
- Anti-look-ahead: H4/H1 features use last COMPLETED bar only
- Reuses all evaluation/training functions from simple_neural_trainer.py

Target performance:
- M15 execution speed (~18 trades/week) with H4-level accuracy (70%+ win rate)
- PF > 2.0, expectancy > 0.002
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path so subpackage imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Reuse everything from existing trainer
from training.simple_neural_trainer import (
    LABEL_SELL,
    LABEL_HOLD,
    LABEL_BUY,
    LABEL_NAMES,
    M15_BARS_PER_HOUR,
    WEEK_DAYS_FOR_OBJECTIVE,
    DEFAULT_WEEKLY_SAMPLES_PER_SYMBOL,
    TARGET_TRADES_PER_HOUR,
    TARGET_WEEKLY_TRADES,
    set_reproducibility,
    normalize_symbol,
    estimate_symbol_point,
    setup_mt5,
    resolve_symbol,
    calculate_rsi,
    calculate_atr,
    build_symbol_index,
    DatasetBundle,
    fit_scaler,
    apply_scaler,
    compute_class_weights,
    SimpleNeuralNetwork,
    Metrics,
    evaluate_predictions,
    evaluate_directional_quality,
    evaluate_profitability_quality,
    compute_symbol_validation_stats,
    compute_symbol_profitability_stats,
    build_symbol_live_profile,
    model_predict_proba,
    train_neural_network,
    run_walk_forward_validation,
    calibrate_symbol_thresholds,
    save_model,
)
from models.push_structure_analyzer import (
    PUSH_FEATURE_COUNT,
    PushFeatureExtractor,
    SymbolPushProfile,
    compute_all_push_profiles,
    infer_direction_from_closes,
)
from models.pattern_recognition import (
    PatternRecognizer,
    FormingPattern,
    FORMING_PATTERN_FEATURE_COUNT,
)
from app.zeropoint_signal import (
    compute_zeropoint_state,
    extract_zeropoint_bar_features,
    ZEROPOINT_FEATURES_PER_TF,
    ZEROPOINT_TOTAL_FEATURES,
    ATR_PERIOD as ZP_ATR_PERIOD,
    MIN_BARS as ZP_MIN_BARS,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 18 base + 8 push + 12 forming pattern + 15 ZeroPoint = 53
BASE_FEATURE_DIM_ZP = 53
ZEROPOINT_MODEL_PATH = "zeropoint_neural_model.pth"
ZEROPOINT_REPORT_PATH = "zeropoint_training_report.json"

# Symbols to train on (all 9 model symbols)
TRAINING_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "BTCUSD"
]


# ---------------------------------------------------------------------------
# Multi-timeframe data collection
# ---------------------------------------------------------------------------

def collect_multi_timeframe_data(
    symbols: Optional[List[str]] = None,
    days: int = 180,
) -> Dict[str, Dict[str, object]]:
    """
    Collect M15, H1, and H4 historical data from MT5 for each symbol.
    Also computes ZeroPoint state on each timeframe.

    Returns dict keyed by normalized symbol with:
      - M15: DataFrame with indicators + ZeroPoint state
      - H1:  DataFrame with ZeroPoint state
      - H4:  DataFrame with ZeroPoint state
      - point, digits, resolved_symbol, etc.
    """
    if symbols is None:
        symbols = TRAINING_SYMBOLS

    print(f"Collecting {days} days of M15/H1/H4 data for {len(symbols)} symbols...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    all_data: Dict[str, Dict[str, object]] = {}
    for requested_symbol in symbols:
        resolved_symbol = resolve_symbol(requested_symbol)
        if not resolved_symbol:
            print(f"  {requested_symbol}: not available on this broker")
            continue

        # --- M15 ---
        rates_m15 = mt5.copy_rates_range(
            resolved_symbol, mt5.TIMEFRAME_M15, start_date, end_date
        )
        if rates_m15 is None or len(rates_m15) < 400:
            print(f"  {requested_symbol} -> {resolved_symbol}: insufficient M15 bars")
            continue
        df_m15 = pd.DataFrame(rates_m15)
        df_m15["time"] = pd.to_datetime(df_m15["time"], unit="s")
        df_m15.set_index("time", inplace=True)

        # M15 indicators (same as simple_neural_trainer)
        df_m15["sma_5"] = df_m15["close"].rolling(5).mean()
        df_m15["sma_20"] = df_m15["close"].rolling(20).mean()
        df_m15["sma_50"] = df_m15["close"].rolling(50).mean()
        df_m15["ema_12"] = df_m15["close"].ewm(span=12, adjust=False).mean()
        df_m15["ema_26"] = df_m15["close"].ewm(span=26, adjust=False).mean()
        df_m15["rsi"] = calculate_rsi(df_m15["close"])
        df_m15["returns"] = df_m15["close"].pct_change()
        df_m15["volatility"] = df_m15["returns"].rolling(20).std()
        df_m15["atr_14"] = calculate_atr(df_m15, period=14)
        bb_std = df_m15["close"].rolling(20).std()
        df_m15["bb_upper"] = df_m15["sma_20"] + 2.0 * bb_std
        df_m15["bb_lower"] = df_m15["sma_20"] - 2.0 * bb_std
        df_m15["volume_z"] = (
            (df_m15["tick_volume"] - df_m15["tick_volume"].rolling(20).mean())
            / (df_m15["tick_volume"].rolling(20).std() + 1e-8)
        )
        candle_range = (df_m15["high"] - df_m15["low"]).replace(0, np.nan)
        df_m15["body_ratio"] = (df_m15["close"] - df_m15["open"]) / (candle_range + 1e-8)

        # --- H1 ---
        rates_h1 = mt5.copy_rates_range(
            resolved_symbol, mt5.TIMEFRAME_H1, start_date, end_date
        )
        df_h1 = None
        if rates_h1 is not None and len(rates_h1) >= ZP_MIN_BARS:
            df_h1 = pd.DataFrame(rates_h1)
            df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
            df_h1.set_index("time", inplace=True)

        # --- H4 ---
        rates_h4 = mt5.copy_rates_range(
            resolved_symbol, mt5.TIMEFRAME_H4, start_date, end_date
        )
        df_h4 = None
        if rates_h4 is not None and len(rates_h4) >= ZP_MIN_BARS:
            df_h4 = pd.DataFrame(rates_h4)
            df_h4["time"] = pd.to_datetime(df_h4["time"], unit="s")
            df_h4.set_index("time", inplace=True)

        # --- Compute ZeroPoint state on all timeframes ---
        zp_m15 = compute_zeropoint_state(df_m15)
        zp_h1 = compute_zeropoint_state(df_h1) if df_h1 is not None else None
        zp_h4 = compute_zeropoint_state(df_h4) if df_h4 is not None else None

        if zp_m15 is None:
            print(f"  {requested_symbol}: ZeroPoint computation failed on M15")
            continue

        symbol_info = mt5.symbol_info(resolved_symbol)
        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        digits = int(getattr(symbol_info, "digits", 5) or 5)

        key = normalize_symbol(requested_symbol)
        all_data[key] = {
            "requested_symbol": requested_symbol,
            "resolved_symbol": resolved_symbol,
            "M15": df_m15,
            "H1": df_h1,
            "H4": df_h4,
            "ZP_M15": zp_m15,
            "ZP_H1": zp_h1,
            "ZP_H4": zp_h4,
            "point": point,
            "digits": digits,
        }
        h1_count = len(df_h1) if df_h1 is not None else 0
        h4_count = len(df_h4) if df_h4 is not None else 0
        print(
            f"  {requested_symbol} -> {resolved_symbol}: "
            f"M15={len(df_m15)}, H1={h1_count}, H4={h4_count}"
        )

    print(f"Collected multi-TF data for {len(all_data)} symbols")
    return all_data


# ---------------------------------------------------------------------------
# Higher-timeframe alignment utilities (anti-look-ahead)
# ---------------------------------------------------------------------------

def align_htf_index(m15_time, htf_df: pd.DataFrame, htf_bar_duration_hours: int) -> int:
    """
    Find the index of the last COMPLETED higher-timeframe bar at m15_time.

    A bar is 'completed' when its close time <= m15_time.
    Close time = bar_open_time + htf_bar_duration.

    Returns index into htf_df, or -1 if no completed bar exists yet.
    """
    if htf_df is None or len(htf_df) == 0:
        return -1

    htf_times = htf_df.index
    # Close time of each HTF bar
    htf_close_times = htf_times + pd.Timedelta(hours=htf_bar_duration_hours)

    # Convert m15_time to numpy datetime64 for consistent comparison
    m15_np = np.datetime64(pd.Timestamp(m15_time))

    # Find the rightmost HTF bar whose close time <= m15_time
    htf_close_np = htf_close_times.values.astype("datetime64[ns]")
    idx = np.searchsorted(htf_close_np, m15_np, side="right") - 1
    return int(idx)


# ---------------------------------------------------------------------------
# Feature + Label creation with ZeroPoint
# ---------------------------------------------------------------------------

def create_zeropoint_features_and_labels(
    data: Dict[str, Dict[str, object]],
    symbol_to_index: Dict[str, int],
    lookback: int = 20,
    horizon: int = 12,
    push_profiles: Optional[Dict[str, SymbolPushProfile]] = None,
) -> DatasetBundle:
    """
    Create training samples with ZeroPoint-enriched features and H4-confirmed labels.

    Feature vector: 62 dimensions
      [18 base M15] + [8 push] + [12 forming pattern] + [5 ZP M15] + [5 ZP H1] + [5 ZP H4] + [9 one-hot]

    Labels: H4 ZeroPoint direction + price confirmation
      BUY  = H4 ZP bullish AND future_return > cost_threshold
      SELL = H4 ZP bearish AND future_return < -cost_threshold
      HOLD = everything else
    """
    print("Creating ZeroPoint-enriched features and labels...")

    rows: List[Tuple[pd.Timestamp, str, np.ndarray, int, float, float]] = []
    symbol_count = len(symbol_to_index)

    # Pattern scan config (same as simple_neural_trainer)
    PATTERN_SCAN_INTERVAL = 4
    PATTERN_WINDOW = 80
    MTF_PAT_CONFIG = {
        "M15": {"weight": 0.85, "min_completion": 0.40},
        "H1":  {"weight": 1.0,  "min_completion": 0.55},
        "H4":  {"weight": 1.15, "min_completion": 0.75},
    }

    # Minimum start index to ensure all rolling indicators (SMA50 etc.) are valid.
    # SMA50 needs 50 bars, BB needs 20 bars, ATR needs 14, volatility needs 20.
    # Use max(lookback, 55) to safely skip the warm-up region.
    safe_start = max(lookback, 55)

    for symbol_key, payload in data.items():
        # dropna() to match simple_neural_trainer — drops rows where ANY indicator
        # is NaN (SMA50 warm-up region, etc.).  This prevents NaN features from
        # poisoning the scaler's mean/std computation.
        df_m15 = payload["M15"].dropna()
        zp_m15 = payload.get("ZP_M15")
        zp_h1 = payload.get("ZP_H1")
        zp_h4 = payload.get("ZP_H4")
        df_h1 = payload.get("H1")
        df_h4 = payload.get("H4")

        if len(df_m15) < safe_start + horizon + 20:
            continue

        symbol_point = float(payload.get("point", 0.0) or 0.0)
        if symbol_point <= 0:
            symbol_point = estimate_symbol_point(symbol_key, float(df_m15["close"].iloc[-1]))

        # Symbol-specific movement threshold for cost floor
        future_returns_series = (df_m15["close"].shift(-horizon) / (df_m15["close"] + 1e-12)) - 1.0
        abs_future_returns = future_returns_series.abs().dropna()
        if len(abs_future_returns) < 200:
            continue
        symbol_move_threshold = float(np.clip(np.quantile(abs_future_returns, 0.45), 0.0005, 0.0100))

        one_hot = np.zeros(symbol_count, dtype=np.float32)
        one_hot[symbol_to_index[symbol_key]] = 1.0

        # Push profile
        push_profile = (
            push_profiles.get(symbol_key, SymbolPushProfile.default(symbol_key))
            if push_profiles
            else SymbolPushProfile.default(symbol_key)
        )
        all_highs = df_m15["high"].astype(float).values
        all_lows = df_m15["low"].astype(float).values
        all_closes = df_m15["close"].astype(float).values
        push_lookback = max(lookback, 60)

        # Pre-compute forming pattern features (multi-TF, cached)
        cached_pattern_feats: Dict[int, np.ndarray] = {}
        zero_pattern_feats = np.zeros(FORMING_PATTERN_FEATURE_COUNT, dtype=np.float32)

        # Build resampled H1/H4 for pattern detection from M15
        _resampled_h1 = None
        _resampled_h4 = None
        try:
            _df_resample = df_m15.copy()
            if not isinstance(_df_resample.index, pd.DatetimeIndex):
                _df_resample = _df_resample.reset_index(drop=True)

            h1_chunks = len(_df_resample) // 4
            if h1_chunks >= 40:
                h1_ohlc = []
                for ci in range(h1_chunks):
                    chunk = _df_resample.iloc[ci * 4:(ci + 1) * 4]
                    h1_ohlc.append({
                        "open": float(chunk["open"].iloc[0]),
                        "high": float(chunk["high"].max()),
                        "low": float(chunk["low"].min()),
                        "close": float(chunk["close"].iloc[-1]),
                        "tick_volume": float(chunk["tick_volume"].sum()) if "tick_volume" in chunk.columns else 1.0,
                    })
                _resampled_h1 = pd.DataFrame(h1_ohlc)

            h4_chunks = len(_df_resample) // 16
            if h4_chunks >= 30:
                h4_ohlc = []
                for ci in range(h4_chunks):
                    chunk = _df_resample.iloc[ci * 16:(ci + 1) * 16]
                    h4_ohlc.append({
                        "open": float(chunk["open"].iloc[0]),
                        "high": float(chunk["high"].max()),
                        "low": float(chunk["low"].min()),
                        "close": float(chunk["close"].iloc[-1]),
                        "tick_volume": float(chunk["tick_volume"].sum()) if "tick_volume" in chunk.columns else 1.0,
                    })
                _resampled_h4 = pd.DataFrame(h4_ohlc)
        except Exception:
            pass

        # Pre-compute ZP feature arrays for M15 (already done in ZP_M15)
        # After dropna(), df_m15 indices may not start at 0.  We need to align
        # ZP data with df_m15 by using matching indices.
        zp_m15_valid = zp_m15 is not None and "pos" in zp_m15.columns
        if zp_m15_valid:
            # Restrict ZP data to only the indices present in df_m15
            zp_m15_aligned = zp_m15.loc[zp_m15.index.intersection(df_m15.index)]
        else:
            zp_m15_aligned = None

        # Count stats
        symbol_samples = 0
        symbol_buy = 0
        symbol_sell = 0

        for i in range(safe_start, len(df_m15) - horizon):
            window = df_m15.iloc[i - lookback:i]
            current = df_m15.iloc[i]
            prev = df_m15.iloc[i - 1]
            m15_time = df_m15.index[i]

            current_price = float(current["close"])
            prev_price = float(prev["close"])
            price_std = float(window["close"].std())
            returns_std = float(window["returns"].std()) if "returns" in window.columns else 1e-8

            if not np.isfinite(price_std) or price_std < 1e-12:
                price_std = 1e-8
            if not np.isfinite(returns_std) or returns_std < 1e-12:
                returns_std = 1e-8

            close_3 = float(df_m15.iloc[i - 3]["close"]) if i >= 3 else prev_price
            close_6 = float(df_m15.iloc[i - 6]["close"]) if i >= 6 else prev_price
            close_12 = float(df_m15.iloc[i - 12]["close"]) if i >= 12 else prev_price
            atr_14 = float(current["atr_14"]) if np.isfinite(current["atr_14"]) else 0.0
            bb_upper = float(current["bb_upper"]) if np.isfinite(current["bb_upper"]) else current_price
            bb_lower = float(current["bb_lower"]) if np.isfinite(current["bb_lower"]) else current_price
            bb_width = max(bb_upper - bb_lower, 1e-8)
            bb_pos = (current_price - bb_lower) / bb_width
            volume_z = float(current["volume_z"]) if np.isfinite(current["volume_z"]) else 0.0
            body_ratio = float(current["body_ratio"]) if np.isfinite(current["body_ratio"]) else 0.0
            spread_points = float(current.get("spread", 0.0) or 0.0)
            spread_cost = (spread_points * symbol_point) / (current_price + 1e-12)
            spread_cost = float(np.clip(spread_cost, 0.0, 0.02))
            atr_ratio = atr_14 / (current_price + 1e-12)
            spread_pressure = spread_cost / (atr_ratio + 1e-12)

            # === 18 base features (identical to simple_neural_trainer) ===
            base_features = np.array([
                (current_price - prev_price) / (prev_price + 1e-12),
                (current_price - float(window["close"].mean())) / (price_std + 1e-12),
                float(current["sma_5"]) / (current_price + 1e-12) - 1.0,
                float(current["sma_20"]) / (current_price + 1e-12) - 1.0,
                float(current["rsi"]) / 100.0,
                returns_std * 100.0,
                (current_price - close_3) / (close_3 + 1e-12),
                (current_price - close_6) / (close_6 + 1e-12),
                (current_price - close_12) / (close_12 + 1e-12),
                float(current["sma_50"]) / (current_price + 1e-12) - 1.0,
                float(current["ema_12"]) / (current_price + 1e-12) - 1.0,
                float(current["ema_26"]) / (current_price + 1e-12) - 1.0,
                atr_14 / (current_price + 1e-12),
                bb_pos,
                volume_z,
                body_ratio,
                spread_cost,
                spread_pressure,
            ], dtype=np.float32)

            # === 8 push structure features ===
            push_start = max(0, i - push_lookback)
            window_highs = all_highs[push_start:i + 1]
            window_lows = all_lows[push_start:i + 1]
            window_closes = all_closes[push_start:i + 1]
            direction = infer_direction_from_closes(window_closes, lookback=10)
            push_features = PushFeatureExtractor.extract_push_features(
                highs=window_highs,
                lows=window_lows,
                closes=window_closes,
                profile=push_profile,
                point=symbol_point,
                direction=direction,
            )

            # === 12 forming pattern features (multi-TF, cached) ===
            scan_key = (i // PATTERN_SCAN_INTERVAL) * PATTERN_SCAN_INTERVAL
            if scan_key not in cached_pattern_feats:
                try:
                    best_mtf_score = -1.0
                    best_mtf_feats = zero_pattern_feats

                    # M15 scan
                    m15_cfg = MTF_PAT_CONFIG["M15"]
                    pat_start = max(0, i - PATTERN_WINDOW)
                    pat_slice = df_m15.iloc[pat_start:i + 1].copy()
                    if len(pat_slice) >= 25:
                        rec = PatternRecognizer(pat_slice)
                        forming_m15 = rec.detect_forming_patterns()
                        if forming_m15:
                            filtered = [fp for fp in forming_m15 if fp.completion_pct >= m15_cfg["min_completion"]]
                            for fp in filtered:
                                time_bonus = (fp.completion_pct - m15_cfg["min_completion"]) / (1.0 - m15_cfg["min_completion"] + 1e-8)
                                sc = fp.confidence * fp.completion_pct * m15_cfg["weight"] * (1.0 + 0.3 * time_bonus)
                                if sc > best_mtf_score:
                                    best_mtf_score = sc
                                    best_mtf_feats = PatternRecognizer.forming_pattern_features(filtered)

                    # H1 scan (resampled from M15)
                    if _resampled_h1 is not None:
                        h1_cfg = MTF_PAT_CONFIG["H1"]
                        h1_idx = i // 4
                        h1_end = h1_idx + 1
                        h1_start = max(0, h1_end - PATTERN_WINDOW)
                        h1_slice = _resampled_h1.iloc[h1_start:h1_end].copy()
                        if len(h1_slice) >= 25:
                            rec_h1 = PatternRecognizer(h1_slice)
                            forming_h1 = rec_h1.detect_forming_patterns()
                            if forming_h1:
                                filtered = [fp for fp in forming_h1 if fp.completion_pct >= h1_cfg["min_completion"]]
                                for fp in filtered:
                                    time_bonus = (fp.completion_pct - h1_cfg["min_completion"]) / (1.0 - h1_cfg["min_completion"] + 1e-8)
                                    sc = fp.confidence * fp.completion_pct * h1_cfg["weight"] * (1.0 + 0.3 * time_bonus)
                                    if sc > best_mtf_score:
                                        best_mtf_score = sc
                                        best_mtf_feats = PatternRecognizer.forming_pattern_features(filtered)

                    # H4 scan (resampled from M15)
                    if _resampled_h4 is not None:
                        h4_cfg = MTF_PAT_CONFIG["H4"]
                        h4_idx = i // 16
                        h4_end = h4_idx + 1
                        h4_start = max(0, h4_end - PATTERN_WINDOW)
                        h4_slice = _resampled_h4.iloc[h4_start:h4_end].copy()
                        if len(h4_slice) >= 25:
                            rec_h4 = PatternRecognizer(h4_slice)
                            forming_h4 = rec_h4.detect_forming_patterns()
                            if forming_h4:
                                filtered = [fp for fp in forming_h4 if fp.completion_pct >= h4_cfg["min_completion"]]
                                for fp in filtered:
                                    time_bonus = (fp.completion_pct - h4_cfg["min_completion"]) / (1.0 - h4_cfg["min_completion"] + 1e-8)
                                    sc = fp.confidence * fp.completion_pct * h4_cfg["weight"] * (1.0 + 0.3 * time_bonus)
                                    if sc > best_mtf_score:
                                        best_mtf_score = sc
                                        best_mtf_feats = PatternRecognizer.forming_pattern_features(filtered)

                    cached_pattern_feats[scan_key] = best_mtf_feats
                except Exception:
                    cached_pattern_feats[scan_key] = zero_pattern_feats
            pattern_features = cached_pattern_feats[scan_key]

            # === 5 ZeroPoint M15 features ===
            zp_m15_feats = np.zeros(ZEROPOINT_FEATURES_PER_TF, dtype=np.float32)
            if zp_m15_aligned is not None and m15_time in zp_m15_aligned.index:
                zp_m15_feats = extract_zeropoint_bar_features(zp_m15_aligned.loc[m15_time])

            # === 5 ZeroPoint H1 features (anti-look-ahead) ===
            zp_h1_feats = np.zeros(ZEROPOINT_FEATURES_PER_TF, dtype=np.float32)
            if zp_h1 is not None and "pos" in zp_h1.columns:
                h1_idx = align_htf_index(m15_time, zp_h1, htf_bar_duration_hours=1)
                if h1_idx >= 0 and h1_idx < len(zp_h1):
                    zp_h1_feats = extract_zeropoint_bar_features(zp_h1.iloc[h1_idx])

            # === 5 ZeroPoint H4 features (anti-look-ahead) ===
            zp_h4_feats = np.zeros(ZEROPOINT_FEATURES_PER_TF, dtype=np.float32)
            h4_pos_for_label = 0  # Default: no H4 position info
            if zp_h4 is not None and "pos" in zp_h4.columns:
                h4_idx = align_htf_index(m15_time, zp_h4, htf_bar_duration_hours=4)
                if h4_idx >= 0 and h4_idx < len(zp_h4):
                    zp_h4_feats = extract_zeropoint_bar_features(zp_h4.iloc[h4_idx])
                    h4_pos_for_label = int(zp_h4.iloc[h4_idx]["pos"])

            # === Assemble full feature vector: 53 base + 9 one-hot = 62 ===
            features = np.concatenate([
                base_features,       # 18
                push_features,       # 8
                pattern_features,    # 12
                zp_m15_feats,        # 5
                zp_h1_feats,         # 5
                zp_h4_feats,         # 5
                one_hot,             # 9
            ]).astype(np.float32)

            # Skip rows with any NaN/inf in base features (should be rare after
            # dropna() + safe_start, but guard against edge cases).
            if not np.all(np.isfinite(features)):
                continue

            # === LABEL: H4 ZeroPoint direction + price confirmation ===
            future_prices = df_m15.iloc[i:i + horizon]["close"]
            if len(future_prices) < horizon:
                continue
            future_return = float(future_prices.iloc[-1] / (current_price + 1e-12) - 1.0)

            # Cost threshold (same hybrid logic as simple_neural_trainer)
            threshold = max(
                0.0004,
                min(0.0120, 0.35 * symbol_move_threshold + 0.85 * returns_std),
            )
            cost_floor = (spread_cost * 1.25) + 0.00005
            threshold = max(threshold, cost_floor)

            # H4 ZeroPoint confirmed labeling:
            # BUY only when H4 ZP is bullish AND price moves up
            # SELL only when H4 ZP is bearish AND price moves down
            # Everything else is HOLD
            if h4_pos_for_label == 1 and future_return > threshold:
                label = LABEL_BUY
                symbol_buy += 1
            elif h4_pos_for_label == -1 and future_return < -threshold:
                label = LABEL_SELL
                symbol_sell += 1
            else:
                label = LABEL_HOLD

            rows.append((m15_time, symbol_key, features, label, future_return, spread_cost))
            symbol_samples += 1

        if symbol_samples > 0:
            print(
                f"    {symbol_key}: {symbol_samples} samples "
                f"(BUY={symbol_buy}, SELL={symbol_sell}, "
                f"HOLD={symbol_samples - symbol_buy - symbol_sell})"
            )

    if not rows:
        return DatasetBundle(
            features=np.zeros((0, 0), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            symbols=np.array([], dtype=object),
            timestamps=np.array([], dtype="datetime64[ns]"),
            future_returns=np.zeros((0,), dtype=np.float32),
            spread_costs=np.zeros((0,), dtype=np.float32),
        )

    # Strict temporal ordering
    rows.sort(key=lambda row: row[0])

    features = np.vstack([row[2] for row in rows]).astype(np.float32)
    labels = np.array([row[3] for row in rows], dtype=np.int64)
    symbols = np.array([row[1] for row in rows], dtype=object)
    timestamps = np.array([row[0] for row in rows], dtype="datetime64[ns]")
    future_returns = np.array([row[4] for row in rows], dtype=np.float32)
    spread_costs = np.array([row[5] for row in rows], dtype=np.float32)

    # Verify no NaN/inf in final feature matrix
    nan_count = np.sum(~np.isfinite(features))
    print(f"\nCreated {len(features)} samples with feature_dim={features.shape[1]}")
    if nan_count > 0:
        nan_per_dim = np.sum(~np.isfinite(features), axis=0)
        print(f"WARNING: {nan_count} NaN/inf values found in features!")
        for d in range(features.shape[1]):
            if nan_per_dim[d] > 0:
                print(f"  dim {d}: NaN={nan_per_dim[d]}")
    else:
        print(f"Feature quality: CLEAN (no NaN/inf values)")
    label_counts = np.bincount(labels, minlength=3)
    print(
        f"Label distribution: "
        f"SELL={label_counts[LABEL_SELL]}, "
        f"HOLD={label_counts[LABEL_HOLD]}, "
        f"BUY={label_counts[LABEL_BUY]}"
    )
    buy_pct = label_counts[LABEL_BUY] / max(1, len(labels)) * 100
    sell_pct = label_counts[LABEL_SELL] / max(1, len(labels)) * 100
    hold_pct = label_counts[LABEL_HOLD] / max(1, len(labels)) * 100
    print(f"Label %: BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%, HOLD={hold_pct:.1f}%")

    return DatasetBundle(
        features=features,
        labels=labels,
        symbols=symbols,
        timestamps=timestamps,
        future_returns=future_returns,
        spread_costs=spread_costs,
    )


# ---------------------------------------------------------------------------
# Custom save_model with ZeroPoint metadata
# ---------------------------------------------------------------------------

def save_zeropoint_model(
    model: nn.Module,
    model_path: str,
    feature_dim: int,
    symbol_to_index: Dict[str, int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    metadata: Dict[str, object],
    push_profiles: Optional[Dict[str, SymbolPushProfile]] = None,
) -> None:
    """Save model with ZeroPoint-specific metadata flags."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_dim": int(feature_dim),
        "symbol_to_index": {normalize_symbol(k): int(v) for k, v in symbol_to_index.items()},
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "metadata": metadata,
        "save_date": datetime.now().isoformat(),
    }
    if push_profiles:
        checkpoint["push_profiles"] = {
            k: v.to_dict() for k, v in push_profiles.items()
        }
    torch.save(checkpoint, model_path)
    print(f"ZeroPoint model saved to {model_path}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> bool:
    print("=" * 70)
    print("ZeroPoint Neural Training System")
    print("H4 ZeroPoint signals as labels, M15 execution features")
    print("=" * 70)
    set_reproducibility(42)

    if not setup_mt5():
        print("Failed to setup MT5")
        return False

    # --- Step 1: Collect multi-timeframe data ---
    data = collect_multi_timeframe_data(symbols=TRAINING_SYMBOLS, days=180)
    if len(data) < 3:
        print("Insufficient symbol coverage for robust training")
        return False

    symbol_to_index = build_symbol_index(list(data.keys()))

    # --- Step 2: Compute push profiles ---
    print("\nComputing push profiles from MT5 data...")
    # Build a compatible dict for compute_all_push_profiles (needs "M15" key)
    push_data = {}
    for k, v in data.items():
        push_data[k] = {
            "M15": v["M15"],
            "point": v["point"],
            "digits": v["digits"],
            "requested_symbol": v["requested_symbol"],
            "resolved_symbol": v["resolved_symbol"],
        }
    push_profiles = compute_all_push_profiles(push_data)
    print(f"Push profiles computed for {len(push_profiles)} symbols")

    # --- Step 3: Candidate training loop ---
    candidate_configs = [
        {
            "name": "balanced",
            "epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 256,
            "class_weight_mode": "balanced",
            "label_smoothing": 0.01,
        },
        {
            "name": "sqrt_balanced",
            "epochs": 200,
            "learning_rate": 0.0008,
            "batch_size": 256,
            "class_weight_mode": "sqrt_balanced",
            "label_smoothing": 0.02,
        },
        {
            "name": "natural",
            "epochs": 200,
            "learning_rate": 0.0006,
            "batch_size": 192,
            "class_weight_mode": "none",
            "label_smoothing": 0.01,
        },
        {
            "name": "balanced_precise",
            "epochs": 250,
            "learning_rate": 0.0005,
            "batch_size": 512,
            "class_weight_mode": "balanced",
            "label_smoothing": 0.0,
        },
    ]

    selected = None
    horizon_candidates = [8, 12, 16]
    lookback = 20

    for horizon in horizon_candidates:
        print(f"\n{'='*50}")
        print(f"Evaluating horizon={horizon} ({horizon * 15} min = {horizon * 15 / 60:.1f} hours)")
        print(f"{'='*50}")

        dataset = create_zeropoint_features_and_labels(
            data,
            symbol_to_index=symbol_to_index,
            lookback=lookback,
            horizon=horizon,
            push_profiles=push_profiles,
        )
        if len(dataset.features) < 2000:
            print(f"Skipping horizon={horizon} due to insufficient samples ({len(dataset.features)})")
            continue

        split_index = int(len(dataset.features) * 0.8)
        train_features = dataset.features[:split_index]
        train_labels = dataset.labels[:split_index]
        val_features = dataset.features[split_index:]
        val_labels = dataset.labels[split_index:]
        val_symbols = dataset.symbols[split_index:]
        val_future_returns = dataset.future_returns[split_index:]
        val_spread_costs = dataset.spread_costs[split_index:]
        validation_symbol_count = max(1, len(set(val_symbols.tolist())))
        weekly_sample_count = float(DEFAULT_WEEKLY_SAMPLES_PER_SYMBOL * validation_symbol_count)

        feature_mean, feature_std = fit_scaler(train_features)
        train_scaled = apply_scaler(train_features, feature_mean, feature_std)
        val_scaled = apply_scaler(val_features, feature_mean, feature_std)
        all_scaled = apply_scaler(dataset.features, feature_mean, feature_std)

        for cfg in candidate_configs:
            print(f"\nTraining candidate: {cfg['name']} (horizon={horizon})")
            candidate_model, candidate_summary = train_neural_network(
                train_features=train_scaled,
                train_labels=train_labels,
                val_features=val_scaled,
                val_labels=val_labels,
                epochs=cfg["epochs"],
                learning_rate=cfg["learning_rate"],
                batch_size=cfg["batch_size"],
                class_weight_mode=cfg["class_weight_mode"],
                label_smoothing=cfg["label_smoothing"],
            )
            candidate_probs = model_predict_proba(candidate_model, val_scaled)
            candidate_quality = evaluate_directional_quality(
                probs=candidate_probs,
                labels=val_labels,
                min_trades=max(60, int(0.008 * len(val_labels))),
            )
            candidate_profit_quality = evaluate_profitability_quality(
                probs=candidate_probs,
                future_returns=val_future_returns,
                spread_costs=val_spread_costs,
                min_trades=max(60, int(0.008 * len(val_labels))),
                weekly_sample_count=weekly_sample_count,
                tail_focus_weight=0.55,
                target_weekly_trades=TARGET_WEEKLY_TRADES,
                target_trade_weight=0.60,
            )
            combined_score = (
                0.30 * float(candidate_quality["score"])
                + 0.70 * float(candidate_profit_quality["score"])
            )
            print(
                f"  Directional: "
                f"win={candidate_quality['win_rate']:.3f}, "
                f"trades={int(candidate_quality['trade_count'])}, "
                f"threshold={candidate_quality['threshold']:.2f}, "
                f"score={candidate_quality['score']:.3f}"
            )
            print(
                f"  Profitability: "
                f"exp={candidate_profit_quality['expectancy']:.5f}, "
                f"pf={candidate_profit_quality['profit_factor']:.3f}, "
                f"trades={int(candidate_profit_quality['trade_count'])}, "
                f"threshold={candidate_profit_quality['threshold']:.2f}, "
                f"score={candidate_profit_quality['score']:.3f}"
            )
            print(f"  Combined score: {combined_score:.4f}")

            if selected is None or combined_score > selected["combined_score"]:
                selected = {
                    "horizon": horizon,
                    "lookback": lookback,
                    "dataset": dataset,
                    "train_scaled": train_scaled,
                    "val_scaled": val_scaled,
                    "all_scaled": all_scaled,
                    "val_labels": val_labels,
                    "val_symbols": val_symbols,
                    "val_future_returns": val_future_returns,
                    "val_spread_costs": val_spread_costs,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "config": cfg,
                    "model": candidate_model,
                    "summary": candidate_summary,
                    "quality": candidate_quality,
                    "profit_quality": candidate_profit_quality,
                    "combined_score": float(combined_score),
                }

    if selected is None:
        print("\nERROR: No viable candidate found across all horizons.")
        return False

    # --- Step 4: Selected model analysis ---
    dataset = selected["dataset"]
    val_scaled = selected["val_scaled"]
    val_labels = selected["val_labels"]
    val_symbols = selected["val_symbols"]
    val_future_returns = selected["val_future_returns"]
    val_spread_costs = selected["val_spread_costs"]
    validation_symbol_count = max(1, len(set(val_symbols.tolist())))
    weekly_sample_count_global = float(DEFAULT_WEEKLY_SAMPLES_PER_SYMBOL * validation_symbol_count)
    all_scaled = selected["all_scaled"]
    feature_mean = selected["feature_mean"]
    feature_std = selected["feature_std"]
    model = selected["model"]
    val_summary = selected["summary"]
    directional_quality = selected["quality"]
    profitability_quality = selected["profit_quality"]
    selected_config = selected["config"]
    selected_horizon = int(selected["horizon"])
    selected_lookback = int(selected["lookback"])

    print(f"\n{'='*60}")
    print(f"SELECTED: {selected_config['name']} (horizon={selected_horizon})")
    print(f"{'='*60}")

    val_probs = model_predict_proba(model, val_scaled)
    symbol_validation = compute_symbol_validation_stats(
        probs=val_probs,
        labels=val_labels,
        symbols=val_symbols,
    )

    walk_forward = run_walk_forward_validation(
        features=all_scaled,
        labels=dataset.labels,
        folds=3,
        epochs_per_fold=25,
        batch_size=224,
        class_weight_mode=selected_config["class_weight_mode"],
    )

    symbol_thresholds, symbol_action_modes, threshold_diagnostics, global_threshold = calibrate_symbol_thresholds(
        model=model,
        val_features=val_scaled,
        val_labels=val_labels,
        val_symbols=val_symbols,
        val_future_returns=val_future_returns,
        val_spread_costs=val_spread_costs,
        weekly_sample_count=float(DEFAULT_WEEKLY_SAMPLES_PER_SYMBOL),
        tail_focus_weight=0.45,
        target_weekly_trades=float(TARGET_WEEKLY_TRADES / max(1, validation_symbol_count)),
        target_trade_weight=0.35,
    )
    symbol_profitability = compute_symbol_profitability_stats(
        probs=val_probs,
        symbols=val_symbols,
        future_returns=val_future_returns,
        spread_costs=val_spread_costs,
        thresholds=symbol_thresholds,
        action_modes=symbol_action_modes,
        weekly_sample_count=float(DEFAULT_WEEKLY_SAMPLES_PER_SYMBOL),
    )
    live_profile_config = {
        "min_samples": 20,
        "min_profit_factor": 1.05,
        "min_expectancy": 0.00001,
        "min_trade_rate": 0.010,
        "min_weekly_prob_positive": 0.55,
        "min_weekly_p10_return": -0.04,
    }
    symbol_live_profile = build_symbol_live_profile(
        symbol_profitability=symbol_profitability,
        min_samples=int(live_profile_config["min_samples"]),
        min_profit_factor=float(live_profile_config["min_profit_factor"]),
        min_expectancy=float(live_profile_config["min_expectancy"]),
        min_trade_rate=float(live_profile_config["min_trade_rate"]),
        min_weekly_prob_positive=float(live_profile_config["min_weekly_prob_positive"]),
        min_weekly_p10_return=float(live_profile_config["min_weekly_p10_return"]),
    )
    overall_profitability = evaluate_profitability_quality(
        probs=val_probs,
        future_returns=val_future_returns,
        spread_costs=val_spread_costs,
        threshold_grid=np.array([global_threshold], dtype=float),
        min_trades=1,
        weekly_sample_count=weekly_sample_count_global,
        tail_focus_weight=0.55,
        target_weekly_trades=TARGET_WEEKLY_TRADES,
        target_trade_weight=0.60,
    )

    # --- Step 5: Build metadata with ZeroPoint-specific fields ---
    metadata = {
        "trainer_type": "zeropoint",
        "training_date": datetime.now().isoformat(),
        "symbols_used": TRAINING_SYMBOLS,
        "resolved_symbols": [data[normalize_symbol(s)]["resolved_symbol"] for s in TRAINING_SYMBOLS if normalize_symbol(s) in data],
        "symbol_to_index": symbol_to_index,
        "samples": int(len(dataset.features)),
        "feature_dim": int(dataset.features.shape[1]),
        "lookback": selected_lookback,
        "horizon": selected_horizon,
        "base_feature_dim": BASE_FEATURE_DIM_ZP,
        "symbol_feature_dim": len(symbol_to_index),
        "zeropoint_feature_count": ZEROPOINT_TOTAL_FEATURES,
        "zeropoint_features_per_tf": ZEROPOINT_FEATURES_PER_TF,
        "labeling_strategy": "h4_zeropoint_confirmed",
        "selected_training_config": selected_config,
        "directional_quality": directional_quality,
        "profitability_quality": profitability_quality,
        "global_trade_threshold": float(global_threshold),
        "symbol_thresholds": symbol_thresholds,
        "symbol_action_modes": symbol_action_modes,
        "symbol_validation": symbol_validation,
        "symbol_profitability_validation": symbol_profitability,
        "symbol_live_profile": symbol_live_profile,
        "live_profile_config": live_profile_config,
        "training_objective": {
            "tail_focus_weight": 0.55,
            "target_weekly_trades": float(TARGET_WEEKLY_TRADES),
            "target_trade_weight": 0.60,
        },
        "validation_profitability": overall_profitability,
        "validation": val_summary,
        "walk_forward": walk_forward,
        "threshold_diagnostics": threshold_diagnostics,
        "note": (
            "ZeroPoint neural trainer: H4 ATR trailing stop direction used as labeling authority. "
            "15 ZeroPoint features (5 per TF x M15/H1/H4) added to feature vector. "
            "Model learns to identify H4-quality signals on M15 data."
        ),
    }

    # --- Step 6: Save ---
    save_zeropoint_model(
        model=model,
        model_path=ZEROPOINT_MODEL_PATH,
        feature_dim=dataset.features.shape[1],
        symbol_to_index=symbol_to_index,
        feature_mean=feature_mean,
        feature_std=feature_std,
        metadata=metadata,
        push_profiles=push_profiles,
    )

    with open(ZEROPOINT_REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Training report saved to {ZEROPOINT_REPORT_PATH}")

    # --- Step 7: Summary ---
    print("\n" + "=" * 70)
    print("ZEROPOINT NEURAL TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Trainer type: ZeroPoint (H4-confirmed labels)")
    print(f"  Samples: {len(dataset.features)}")
    print(f"  Feature dim: {dataset.features.shape[1]} (base={BASE_FEATURE_DIM_ZP} + {len(symbol_to_index)} one-hot)")
    print(f"  ZeroPoint features: {ZEROPOINT_TOTAL_FEATURES} ({ZEROPOINT_FEATURES_PER_TF} per TF x 3)")
    print(f"  Lookback/Horizon: {selected_lookback}/{selected_horizon}")
    print(f"  Validation accuracy: {val_summary['val_accuracy']:.3f}")
    print(f"  Validation win rate: {val_summary['val_win_rate']:.3f}")
    print(f"  Validation trade rate: {val_summary['val_trade_rate']:.3f}")
    print(
        f"  Directional quality: "
        f"win={directional_quality['win_rate']:.3f}, "
        f"trades={int(directional_quality['trade_count'])}, "
        f"threshold={directional_quality['threshold']:.2f}"
    )
    print(
        f"  Profitability quality: "
        f"exp={profitability_quality['expectancy']:.5f}, "
        f"pf={profitability_quality['profit_factor']:.3f}, "
        f"trades={int(profitability_quality['trade_count'])}, "
        f"threshold={profitability_quality['threshold']:.2f}"
    )
    if overall_profitability.get("expectancy", -1) > 0:
        print(
            f"  Overall profitability (at global threshold {global_threshold:.2f}): "
            f"exp={overall_profitability['expectancy']:.5f}, "
            f"pf={overall_profitability['profit_factor']:.3f}, "
            f"wr={overall_profitability['win_rate']:.3f}"
        )
    print(f"  Walk-forward avg accuracy: {walk_forward.get('avg_accuracy', 0.0):.3f}")
    print(f"  Walk-forward avg win rate: {walk_forward.get('avg_win_rate', 0.0):.3f}")
    print(f"  Global threshold: {global_threshold:.2f}")

    enabled_symbols = [
        symbol for symbol, cfg in symbol_live_profile.items()
        if bool(cfg.get("enabled", False))
    ]
    if enabled_symbols:
        print(f"  Live-enabled symbols ({len(enabled_symbols)}): " + ", ".join(enabled_symbols))
    else:
        print("  Live-enabled symbols: none")

    # Per-symbol breakdown
    print(f"\n  Per-Symbol Results:")
    for symbol in sorted(symbol_profitability.keys()):
        sp = symbol_profitability[symbol]
        lp = symbol_live_profile.get(symbol, {})
        enabled = "ON" if lp.get("enabled", False) else "OFF"
        pf = sp.get("profit_factor", 0.0)
        wr = sp.get("win_rate", 0.0)
        exp = sp.get("avg_trade_return", 0.0)
        trades = int(sp.get("trade_count", 0))
        rm = lp.get("risk_multiplier", 0.0)
        print(
            f"    {symbol:8s}: {enabled:3s} | PF={pf:.2f} WR={wr:.1%} Exp={exp:.5f} "
            f"Trades={trades:4d} RM={rm:.2f}"
        )

    quality_ok = (
        profitability_quality.get("expectancy", -1.0) > 0.0
        and profitability_quality.get("profit_factor", 0.0) > 1.0
        and walk_forward.get("avg_win_rate", 0.0) >= 0.45
    )
    print()
    if quality_ok:
        print("SUCCESS: ZeroPoint model quality is acceptable for controlled live testing.")
    else:
        print("WARNING: Model quality is weak; retraining with more data is recommended.")

    print(f"\nModel saved to: {ZEROPOINT_MODEL_PATH}")
    print(f"Report saved to: {ZEROPOINT_REPORT_PATH}")
    return True


if __name__ == "__main__":
    try:
        success = main()
    finally:
        mt5.shutdown()
    raise SystemExit(0 if success else 1)
