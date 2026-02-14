"""
FlipPredictor Feature Engineering — 45-dim vectors per H4 bar.

Feature groups:
  [0-17]  18 price/indicator features (same as zeropoint_neural_trainer)
  [18-27] 10 ZeroPoint state features (5 existing + 5 NEW predictive)
  [28-35]  8 push structure features
  [36-44]  9 symbol one-hot encoding

5 NEW predictive ZP features:
  - stop_distance_velocity:     rate of price convergence toward trailing stop
  - stop_distance_acceleration: is convergence speeding up? (imminent flip signal)
  - atr_momentum:               ATR expansion = volatility regime change
  - flip_frequency:             flips in last 20 bars / 20 (whipsaw detection)
  - trend_age_normalized:       bars in trend / symbol avg trend duration
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from app.zeropoint_signal import (
    compute_zeropoint_state,
    extract_zeropoint_bar_features,
    ZEROPOINT_FEATURES_PER_TF,
)
from push_structure_analyzer import PushFeatureExtractor, PushStatisticsCollector

# Constants
FEATURE_DIM_BASE = 18          # price/indicator features
FEATURE_DIM_ZP = 10            # 5 existing + 5 new predictive
FEATURE_DIM_PUSH = 8           # push structure features
NUM_SYMBOLS = 9                # one-hot encoding
FEATURE_DIM_TOTAL = FEATURE_DIM_BASE + FEATURE_DIM_ZP + FEATURE_DIM_PUSH + NUM_SYMBOLS  # 45

SYMBOLS = ['AUDUSD', 'BTCUSD', 'EURJPY', 'EURUSD', 'GBPJPY',
           'GBPUSD', 'NZDUSD', 'USDCAD', 'USDJPY']
SYMBOL_TO_INDEX = {s: i for i, s in enumerate(SYMBOLS)}

SEQ_LEN = 16                   # H4 bars per sequence (64 hours lookback)
LOOKBACK = 20                  # rolling window for indicators


def compute_base_features(df: pd.DataFrame, i: int) -> np.ndarray:
    """Extract 18 price/indicator features for bar i.

    Matches zeropoint_neural_trainer.py feature layout exactly.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["tick_volume"].values if "tick_volume" in df.columns else df.get("volume", pd.Series(np.zeros(len(df)))).values

    current_price = close[i]
    prev_price = close[i - 1] if i > 0 else current_price

    # Rolling window
    start = max(0, i - LOOKBACK + 1)
    window = close[start:i + 1]
    window_mean = np.mean(window)
    price_std = np.std(window) + 1e-12

    # SMAs
    def _sma(arr, period, idx):
        s = max(0, idx - period + 1)
        seg = arr[s:idx + 1]
        return np.mean(seg) if len(seg) >= period else np.mean(seg) if len(seg) > 0 else arr[idx]

    sma_5 = _sma(close, 5, i)
    sma_20 = _sma(close, 20, i)
    sma_50 = _sma(close, 50, i)

    # RSI(14)
    rsi_period = 14
    if i >= rsi_period:
        deltas = np.diff(close[max(0, i - rsi_period):i + 1])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) + 1e-12
        avg_loss = np.mean(losses) + 1e-12
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi = 50.0

    # Returns volatility
    if i >= 5:
        returns = np.diff(close[max(0, i - 20):i + 1]) / (close[max(0, i - 20):i] + 1e-12)
        returns_std = np.std(returns) if len(returns) > 1 else 0.0
    else:
        returns_std = 0.0

    # Lagged returns
    close_3 = close[max(0, i - 3)]
    close_6 = close[max(0, i - 6)]
    close_12 = close[max(0, i - 12)]

    # EMAs
    def _ema(arr, period, idx):
        alpha = 2.0 / (period + 1)
        s = max(0, idx - period * 3)
        val = arr[s]
        for k in range(s + 1, idx + 1):
            val = alpha * arr[k] + (1 - alpha) * val
        return val

    ema_12 = _ema(close, 12, i)
    ema_26 = _ema(close, 26, i)

    # ATR(14)
    if i >= 1:
        tr_vals = []
        for k in range(max(1, i - 13), i + 1):
            tr = max(high[k] - low[k],
                     abs(high[k] - close[k - 1]),
                     abs(low[k] - close[k - 1]))
            tr_vals.append(tr)
        atr_14 = np.mean(tr_vals) if tr_vals else (high[i] - low[i])
    else:
        atr_14 = high[i] - low[i]

    # Bollinger position
    bb_upper = window_mean + 2 * price_std
    bb_lower = window_mean - 2 * price_std
    bb_range = bb_upper - bb_lower
    bb_pos = (current_price - bb_lower) / bb_range if bb_range > 1e-12 else 0.5

    # Volume z-score
    vol_start = max(0, i - 20 + 1)
    vol_window = volume[vol_start:i + 1].astype(float)
    vol_mean = np.mean(vol_window) + 1e-12
    vol_std = np.std(vol_window) + 1e-12
    volume_z = (volume[i] - vol_mean) / vol_std

    # Candle body ratio
    candle_range = high[i] - low[i]
    body = abs(close[i] - df["open"].values[i])
    body_ratio = body / candle_range if candle_range > 1e-12 else 0.5

    # Spread features (approximated for H4 — use ATR fraction)
    spread_cost = atr_14 * 0.01 / (current_price + 1e-12)
    spread_pressure = 0.01  # minimal on H4

    features = np.array([
        (current_price - prev_price) / (prev_price + 1e-12),    # 0: price change
        (current_price - window_mean) / price_std,               # 1: z-score
        sma_5 / current_price - 1.0,                             # 2: SMA5 dev
        sma_20 / current_price - 1.0,                            # 3: SMA20 dev
        rsi / 100.0,                                              # 4: RSI norm
        returns_std * 100.0,                                      # 5: volatility
        (current_price - close_3) / (close_3 + 1e-12),          # 6: 3-bar return
        (current_price - close_6) / (close_6 + 1e-12),          # 7: 6-bar return
        (current_price - close_12) / (close_12 + 1e-12),        # 8: 12-bar return
        sma_50 / current_price - 1.0,                            # 9: SMA50 dev
        ema_12 / current_price - 1.0,                            # 10: EMA12 dev
        ema_26 / current_price - 1.0,                            # 11: EMA26 dev
        atr_14 / current_price,                                  # 12: ATR ratio
        np.clip(bb_pos, 0.0, 1.0),                               # 13: BB position
        np.clip(volume_z, -3.0, 3.0),                            # 14: volume z
        body_ratio,                                               # 15: body ratio
        spread_cost,                                              # 16: spread cost
        spread_pressure,                                          # 17: spread pressure
    ], dtype=np.float32)

    return np.where(np.isfinite(features), features, 0.0)


def compute_zp_features(zp_df: pd.DataFrame, i: int, avg_trend_duration: float = 26.0) -> np.ndarray:
    """Extract 10 ZeroPoint features: 5 existing + 5 new predictive.

    Args:
        zp_df: DataFrame from compute_zeropoint_state()
        i:     bar index
        avg_trend_duration: symbol average trend duration in bars
    """
    row = zp_df.iloc[i]

    # 5 existing features (from extract_zeropoint_bar_features)
    existing = extract_zeropoint_bar_features(row)  # shape (5,)

    # 5 NEW predictive features
    pos = zp_df["pos"].values
    close = zp_df["close"].values
    stop = zp_df["xATRTrailingStop"].values
    atr = zp_df["atr"].values

    # 1. stop_distance_velocity: rate of convergence toward trailing stop
    #    Negative = price approaching stop (flip imminent)
    if i >= 2 and atr[i] > 1e-12:
        dist_now = (close[i] - stop[i]) / atr[i]
        dist_prev = (close[i - 1] - stop[i - 1]) / (atr[i - 1] + 1e-12)
        stop_velocity = np.clip(dist_now - dist_prev, -3.0, 3.0)
    else:
        stop_velocity = 0.0

    # 2. stop_distance_acceleration: is convergence speeding up?
    if i >= 3 and atr[i] > 1e-12:
        dist_2 = (close[i - 2] - stop[i - 2]) / (atr[i - 2] + 1e-12)
        dist_1 = (close[i - 1] - stop[i - 1]) / (atr[i - 1] + 1e-12)
        dist_0 = (close[i] - stop[i]) / atr[i]
        vel_1 = dist_1 - dist_2
        vel_0 = dist_0 - dist_1
        stop_accel = np.clip(vel_0 - vel_1, -3.0, 3.0)
    else:
        stop_accel = 0.0

    # 3. atr_momentum: ATR expansion/contraction rate
    if i >= 5:
        atr_now = atr[i]
        atr_5ago = atr[max(0, i - 5)]
        atr_momentum = np.clip((atr_now - atr_5ago) / (atr_5ago + 1e-12), -2.0, 2.0)
    else:
        atr_momentum = 0.0

    # 4. flip_frequency: flips in last 20 bars / 20
    lookback_start = max(1, i - 19)
    flips_in_window = 0
    for j in range(lookback_start, i + 1):
        if pos[j] != pos[j - 1] and pos[j] != 0:
            flips_in_window += 1
    flip_freq = flips_in_window / 20.0

    # 5. trend_age_normalized: bars in current trend / avg trend duration
    bars_in_trend = float(row.get("bars_in_position", 0))
    trend_age_norm = np.clip(bars_in_trend / max(avg_trend_duration, 1.0), 0.0, 5.0)

    new_features = np.array([
        stop_velocity,
        stop_accel,
        atr_momentum,
        flip_freq,
        trend_age_norm,
    ], dtype=np.float32)

    return np.concatenate([existing, new_features])  # shape (10,)


def compute_push_features(df: pd.DataFrame, i: int, profile, point: float) -> np.ndarray:
    """Extract 8 push structure features for bar i.

    Uses last 60 bars of data for swing detection context.
    """
    lookback = min(60, i + 1)
    start = i - lookback + 1

    highs = df["high"].values[start:i + 1]
    lows = df["low"].values[start:i + 1]
    closes = df["close"].values[start:i + 1]

    if len(highs) < 10 or profile is None:
        return np.zeros(8, dtype=np.float32)

    # Determine direction from recent price action
    if closes[-1] > closes[0]:
        direction = "bullish"
    else:
        direction = "bearish"

    try:
        features = PushFeatureExtractor.extract_push_features(
            highs=highs, lows=lows, closes=closes,
            profile=profile, point=point,
            direction=direction, order=5,
        )
        return np.where(np.isfinite(features), features, 0.0).astype(np.float32)
    except Exception:
        return np.zeros(8, dtype=np.float32)


def extract_h4_bar_features(
    h4_df: pd.DataFrame,
    zp_h4: pd.DataFrame,
    bar_idx: int,
    symbol: str,
    push_profile=None,
    point: float = 0.00001,
    avg_trend_duration: float = 26.0,
) -> Optional[np.ndarray]:
    """Extract full 45-dim feature vector for a single H4 bar.

    Returns None if insufficient data.
    """
    if bar_idx < max(LOOKBACK, 55):
        return None

    # 18 base features
    base = compute_base_features(h4_df, bar_idx)

    # 10 ZP features (5 existing + 5 new)
    zp = compute_zp_features(zp_h4, bar_idx, avg_trend_duration)

    # 8 push features
    push = compute_push_features(h4_df, bar_idx, push_profile, point)

    # 9 symbol one-hot
    one_hot = np.zeros(NUM_SYMBOLS, dtype=np.float32)
    if symbol in SYMBOL_TO_INDEX:
        one_hot[SYMBOL_TO_INDEX[symbol]] = 1.0

    features = np.concatenate([base, zp, push, one_hot])
    assert features.shape[0] == FEATURE_DIM_TOTAL, f"Expected {FEATURE_DIM_TOTAL}, got {features.shape[0]}"

    return features


def extract_sequence(
    h4_df: pd.DataFrame,
    zp_h4: pd.DataFrame,
    end_idx: int,
    symbol: str,
    push_profile=None,
    point: float = 0.00001,
    avg_trend_duration: float = 26.0,
) -> Optional[np.ndarray]:
    """Extract a (SEQ_LEN, 45) feature sequence ending at end_idx.

    Returns None if insufficient data for full sequence.
    """
    start_idx = end_idx - SEQ_LEN + 1
    if start_idx < max(LOOKBACK, 55):
        return None

    seq = []
    for i in range(start_idx, end_idx + 1):
        feat = extract_h4_bar_features(
            h4_df, zp_h4, i, symbol,
            push_profile=push_profile,
            point=point,
            avg_trend_duration=avg_trend_duration,
        )
        if feat is None:
            return None
        seq.append(feat)

    return np.stack(seq, axis=0)  # (SEQ_LEN, 45)


def compute_avg_trend_durations(zp_h4: pd.DataFrame) -> float:
    """Compute average trend duration from H4 ZP data."""
    pos = zp_h4["pos"].values
    durations = []
    current_duration = 0
    for i in range(1, len(pos)):
        if pos[i] == pos[i - 1]:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 1
    if current_duration > 0:
        durations.append(current_duration)
    return float(np.median(durations)) if durations else 26.0
