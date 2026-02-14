#!/usr/bin/env python3
"""
Advanced neural trainer compatible with the live app runtime.

Key upgrades:
- Symbol-aware features (one-hot symbol embedding appended to base features)
- Feature scaling learned from training split only
- Walk-forward validation for temporal robustness checks
- Per-symbol confidence threshold calibration
- Runtime-compatible checkpoint format for app/model_manager.py
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

warnings.filterwarnings("ignore")


LABEL_SELL = 0
LABEL_HOLD = 1
LABEL_BUY = 2
LABEL_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}
BASE_FEATURE_DIM = 38  # 18 price/indicator + 8 push structure + 12 forming pattern features
M15_BARS_PER_HOUR = 4.0
WEEK_DAYS_FOR_OBJECTIVE = 5.0
DEFAULT_WEEKLY_SAMPLES_PER_SYMBOL = M15_BARS_PER_HOUR * 24.0 * WEEK_DAYS_FOR_OBJECTIVE
TARGET_TRADES_PER_HOUR = 9.0
TARGET_WEEKLY_TRADES = TARGET_TRADES_PER_HOUR * 24.0 * WEEK_DAYS_FOR_OBJECTIVE


def set_reproducibility(seed: int = 42) -> None:
    """Set reproducible seeds for stable training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(symbol).upper())


def estimate_symbol_point(symbol: str, fallback_price: float = 1.0) -> float:
    """
    Estimate tick size when broker metadata is unavailable.
    Used only as a fallback for spread-cost approximation.
    """
    normalized = normalize_symbol(symbol)
    if normalized.endswith("JPY"):
        return 0.01
    if normalized.startswith(("BTC", "ETH", "XRP", "LTC", "SOL")):
        return 0.01 if fallback_price >= 1000 else 0.0001
    return 0.0001


def setup_mt5() -> bool:
    """Setup MT5 connection."""
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False

    account_info = mt5.account_info()
    if account_info:
        print(f"Connected to account: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Balance: ${account_info.balance:.2f}")
        return True

    print("MT5 initialized but no account info is available.")
    return False


def resolve_symbol(requested_symbol: str) -> Optional[str]:
    """
    Resolve requested symbol to a broker-available symbol.
    Keeps full-symbol matches preferred over base aliases.
    """
    requested = normalize_symbol(requested_symbol)
    if not requested:
        return None

    symbols = mt5.symbols_get() or []
    names = [s.name for s in symbols if getattr(s, "name", None)]
    if not names:
        return None

    candidates: List[str] = []

    def add_candidate(name: Optional[str]) -> None:
        if name and name not in candidates:
            candidates.append(name)

    # Exact first.
    add_candidate(requested_symbol)

    # Exact case-insensitive match.
    exact = [name for name in names if normalize_symbol(name) == requested]
    exact.sort(key=len)
    for name in exact:
        add_candidate(name)

    # Full symbol containment (e.g., BTCUSDm).
    full_matches = [name for name in names if requested in normalize_symbol(name)]
    full_matches.sort(key=len)
    for name in full_matches:
        add_candidate(name)

    # Base alias for XXXUSD symbols.
    if requested.endswith("USD") and len(requested) > 3:
        base = requested[:-3]
        base_matches = [name for name in names if normalize_symbol(name).startswith(base)]
        base_matches.sort(key=len)
        for name in base_matches:
            add_candidate(name)

    for candidate in candidates:
        if not mt5.symbol_select(candidate, True):
            continue
        rates = mt5.copy_rates_from_pos(candidate, mt5.TIMEFRAME_M15, 0, 5)
        if rates is not None and len(rates) >= 5:
            return candidate

    return None


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR using OHLC columns from dataframe."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def collect_historical_data(
    symbols: Optional[List[str]] = None,
    days: int = 180
) -> Dict[str, Dict[str, object]]:
    """Collect historical MT5 M15 data and basic indicators per symbol."""
    if symbols is None:
        symbols = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
            "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "BTCUSD"
        ]

    print(f"Collecting {days} days of M15 data for {len(symbols)} requested symbols...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    all_data: Dict[str, Dict[str, object]] = {}
    for requested_symbol in symbols:
        resolved_symbol = resolve_symbol(requested_symbol)
        if not resolved_symbol:
            print(f"  {requested_symbol}: not available on this broker")
            continue

        rates = mt5.copy_rates_range(resolved_symbol, mt5.TIMEFRAME_M15, start_date, end_date)
        if rates is None or len(rates) < 400:
            print(f"  {requested_symbol} -> {resolved_symbol}: insufficient bars")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        # Indicators shared with runtime feature extractor.
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["rsi"] = calculate_rsi(df["close"])
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(20).std()
        df["atr_14"] = calculate_atr(df, period=14)
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["sma_20"] + 2.0 * bb_std
        df["bb_lower"] = df["sma_20"] - 2.0 * bb_std
        df["volume_z"] = (
            (df["tick_volume"] - df["tick_volume"].rolling(20).mean())
            / (df["tick_volume"].rolling(20).std() + 1e-8)
        )
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["body_ratio"] = (df["close"] - df["open"]) / (candle_range + 1e-8)

        symbol_info = mt5.symbol_info(resolved_symbol)
        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        digits = int(getattr(symbol_info, "digits", 5) or 5)

        key = normalize_symbol(requested_symbol)
        all_data[key] = {
            "requested_symbol": requested_symbol,
            "resolved_symbol": resolved_symbol,
            "M15": df,
            "point": point,
            "digits": digits,
        }
        print(f"  {requested_symbol} -> {resolved_symbol}: {len(df)} candles")

    print(f"Collected usable data for {len(all_data)} symbols")
    return all_data


def build_symbol_index(symbol_keys: List[str]) -> Dict[str, int]:
    """Build deterministic symbol index for one-hot encoding."""
    ordered = sorted(set(symbol_keys))
    return {symbol: idx for idx, symbol in enumerate(ordered)}


@dataclass
class DatasetBundle:
    features: np.ndarray
    labels: np.ndarray
    symbols: np.ndarray
    timestamps: np.ndarray
    future_returns: np.ndarray
    spread_costs: np.ndarray


def create_features_and_labels(
    data: Dict[str, Dict[str, object]],
    symbol_to_index: Dict[str, int],
    lookback: int = 20,
    horizon: int = 8,
    push_profiles: Optional[Dict[str, SymbolPushProfile]] = None,
) -> DatasetBundle:
    """
    Create training samples with base features + symbol one-hot features.
    Labels are direction over `horizon` candles with volatility-aware deadzone.
    """
    print("Creating symbol-aware features and labels...")

    rows: List[Tuple[pd.Timestamp, str, np.ndarray, int, float, float]] = []
    symbol_count = len(symbol_to_index)

    # Pre-compute forming pattern features for each symbol using a sliding window.
    # We run the pattern detector every N bars (not every bar) for efficiency.
    # Multi-TF: We resample M15 to H1 (4:1) and H4 (16:1) and scan ALL timeframes,
    # picking the strongest forming pattern — matching the live engine's approach.
    # TIME-AWARE: Only include patterns likely to resolve within ~24hr profit window.
    # Higher TFs need higher completion to qualify (same thresholds as live engine).
    PATTERN_SCAN_INTERVAL = 4  # Scan every 4 bars (~1 hour of M15 data)
    PATTERN_WINDOW = 80  # Look back 80 bars for pattern detection
    MTF_PAT_CONFIG = {
        "M15": {"weight": 0.85, "min_completion": 0.40},
        "H1":  {"weight": 1.0,  "min_completion": 0.55},
        "H4":  {"weight": 1.15, "min_completion": 0.75},
    }  # No D1 — not enough M15 data to resample; live engine handles D1 separately

    for symbol_key, payload in data.items():
        df = payload["M15"].dropna()
        if len(df) < lookback + horizon + 20:
            continue
        symbol_point = float(payload.get("point", 0.0) or 0.0)
        if symbol_point <= 0:
            symbol_point = estimate_symbol_point(symbol_key, float(df["close"].iloc[-1]))

        # Symbol-specific movement profile keeps labeling consistent across pairs.
        future_returns_series = (df["close"].shift(-horizon) / (df["close"] + 1e-12)) - 1.0
        abs_future_returns = future_returns_series.abs().dropna()
        if len(abs_future_returns) < 200:
            continue
        symbol_move_threshold = float(np.clip(np.quantile(abs_future_returns, 0.45), 0.0005, 0.0100))

        one_hot = np.zeros(symbol_count, dtype=np.float32)
        one_hot[symbol_to_index[symbol_key]] = 1.0

        # Push profile for this symbol (learned from historical swing structure).
        push_profile = (
            push_profiles.get(symbol_key, SymbolPushProfile.default(symbol_key))
            if push_profiles
            else SymbolPushProfile.default(symbol_key)
        )
        # Pre-extract arrays for push feature computation.
        all_highs = df["high"].astype(float).values
        all_lows = df["low"].astype(float).values
        all_closes = df["close"].astype(float).values
        # Use a wider window for swing detection (need enough bars for structure).
        push_lookback = max(lookback, 60)

        # Pre-compute forming pattern features for the whole symbol.
        # Cache pattern features and reuse for nearby bars.
        # Multi-TF: resample M15 to H1 (4:1) and H4 (16:1) for richer pattern context.
        cached_pattern_feats: Dict[int, np.ndarray] = {}
        zero_pattern_feats = np.zeros(FORMING_PATTERN_FEATURE_COUNT, dtype=np.float32)

        # Build resampled dataframes for H1 and H4 (once per symbol).
        # H1: group every 4 M15 candles; H4: group every 16 M15 candles.
        _resampled_h1 = None
        _resampled_h4 = None
        try:
            _df_resample = df.copy()
            # Need a datetime index for proper resampling
            if not isinstance(_df_resample.index, pd.DatetimeIndex):
                _df_resample = _df_resample.reset_index(drop=True)
            # Build H1 by grouping every 4 M15 bars
            h1_chunks = len(_df_resample) // 4
            if h1_chunks >= 40:
                h1_ohlc = []
                for ci in range(h1_chunks):
                    chunk = _df_resample.iloc[ci * 4:(ci + 1) * 4]
                    h1_ohlc.append({
                        'open': float(chunk['open'].iloc[0]),
                        'high': float(chunk['high'].max()),
                        'low': float(chunk['low'].min()),
                        'close': float(chunk['close'].iloc[-1]),
                        'tick_volume': float(chunk['tick_volume'].sum()) if 'tick_volume' in chunk.columns else 1.0,
                    })
                _resampled_h1 = pd.DataFrame(h1_ohlc)

            # Build H4 by grouping every 16 M15 bars
            h4_chunks = len(_df_resample) // 16
            if h4_chunks >= 30:
                h4_ohlc = []
                for ci in range(h4_chunks):
                    chunk = _df_resample.iloc[ci * 16:(ci + 1) * 16]
                    h4_ohlc.append({
                        'open': float(chunk['open'].iloc[0]),
                        'high': float(chunk['high'].max()),
                        'low': float(chunk['low'].min()),
                        'close': float(chunk['close'].iloc[-1]),
                        'tick_volume': float(chunk['tick_volume'].sum()) if 'tick_volume' in chunk.columns else 1.0,
                    })
                _resampled_h4 = pd.DataFrame(h4_ohlc)
        except Exception:
            pass  # Fall back to M15-only pattern scanning

        for i in range(lookback, len(df) - horizon):
            window = df.iloc[i - lookback:i]
            current = df.iloc[i]
            prev = df.iloc[i - 1]

            current_price = float(current["close"])
            prev_price = float(prev["close"])
            price_std = float(window["close"].std())
            returns_std = float(window["returns"].std())

            if not np.isfinite(price_std) or price_std < 1e-12:
                price_std = 1e-8
            if not np.isfinite(returns_std) or returns_std < 1e-12:
                returns_std = 1e-8

            close_3 = float(df.iloc[i - 3]["close"]) if i >= 3 else prev_price
            close_6 = float(df.iloc[i - 6]["close"]) if i >= 6 else prev_price
            close_12 = float(df.iloc[i - 12]["close"]) if i >= 12 else prev_price
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

            base_features = np.array(
                [
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
                ],
                dtype=np.float32,
            )

            # Push structure features (8 features from swing analysis).
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

            # Forming pattern features (12 features from puzzle-piece detection).
            # Multi-TF: scan M15, H1, H4 and pick the strongest forming pattern.
            # Cached every PATTERN_SCAN_INTERVAL bars for efficiency.
            scan_key = (i // PATTERN_SCAN_INTERVAL) * PATTERN_SCAN_INTERVAL
            if scan_key not in cached_pattern_feats:
                try:
                    best_mtf_score = -1.0
                    best_mtf_feats = zero_pattern_feats

                    # M15 scan (primary) — time-aware: only patterns >= 40% complete
                    m15_cfg = MTF_PAT_CONFIG["M15"]
                    pat_start = max(0, i - PATTERN_WINDOW)
                    pat_slice = df.iloc[pat_start:i + 1].copy()
                    if len(pat_slice) >= 25:
                        rec = PatternRecognizer(pat_slice)
                        forming_m15 = rec.detect_forming_patterns()
                        if forming_m15:
                            filtered_m15 = [fp for fp in forming_m15 if fp.completion_pct >= m15_cfg["min_completion"]]
                            for fp in filtered_m15:
                                time_bonus = (fp.completion_pct - m15_cfg["min_completion"]) / (1.0 - m15_cfg["min_completion"] + 1e-8)
                                sc = fp.confidence * fp.completion_pct * m15_cfg["weight"] * (1.0 + 0.3 * time_bonus)
                                if sc > best_mtf_score:
                                    best_mtf_score = sc
                                    best_mtf_feats = PatternRecognizer.forming_pattern_features(filtered_m15)

                    # H1 scan (4:1 resample of M15) — time-aware: only patterns >= 55% complete
                    if _resampled_h1 is not None:
                        h1_cfg = MTF_PAT_CONFIG["H1"]
                        h1_idx = i // 4  # Which H1 bar corresponds to M15 bar i
                        h1_end = h1_idx + 1
                        h1_start = max(0, h1_end - PATTERN_WINDOW)
                        h1_slice = _resampled_h1.iloc[h1_start:h1_end].copy()
                        if len(h1_slice) >= 25:
                            rec_h1 = PatternRecognizer(h1_slice)
                            forming_h1 = rec_h1.detect_forming_patterns()
                            if forming_h1:
                                filtered_h1 = [fp for fp in forming_h1 if fp.completion_pct >= h1_cfg["min_completion"]]
                                for fp in filtered_h1:
                                    time_bonus = (fp.completion_pct - h1_cfg["min_completion"]) / (1.0 - h1_cfg["min_completion"] + 1e-8)
                                    sc = fp.confidence * fp.completion_pct * h1_cfg["weight"] * (1.0 + 0.3 * time_bonus)
                                    if sc > best_mtf_score:
                                        best_mtf_score = sc
                                        best_mtf_feats = PatternRecognizer.forming_pattern_features(filtered_h1)

                    # H4 scan (16:1 resample of M15) — time-aware: only patterns >= 75% complete
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
                                filtered_h4 = [fp for fp in forming_h4 if fp.completion_pct >= h4_cfg["min_completion"]]
                                for fp in filtered_h4:
                                    time_bonus = (fp.completion_pct - h4_cfg["min_completion"]) / (1.0 - h4_cfg["min_completion"] + 1e-8)
                                    sc = fp.confidence * fp.completion_pct * h4_cfg["weight"] * (1.0 + 0.3 * time_bonus)
                                    if sc > best_mtf_score:
                                        best_mtf_score = sc
                                        best_mtf_feats = PatternRecognizer.forming_pattern_features(filtered_h4)

                    cached_pattern_feats[scan_key] = best_mtf_feats
                except Exception:
                    cached_pattern_feats[scan_key] = zero_pattern_feats
            pattern_features = cached_pattern_feats[scan_key]

            features = np.concatenate([base_features, push_features, pattern_features, one_hot]).astype(np.float32)

            future_prices = df.iloc[i:i + horizon]["close"]
            future_return = float(future_prices.iloc[-1] / (current_price + 1e-12) - 1.0)

            # Hybrid threshold balances symbol-specific move profile and local volatility.
            threshold = max(
                0.0004,
                min(
                    0.0120,
                    0.35 * symbol_move_threshold + 0.85 * returns_std,
                ),
            )
            # Ensure predicted movement exceeds transaction friction.
            cost_floor = (spread_cost * 1.25) + 0.00005
            threshold = max(threshold, cost_floor)
            if future_return > threshold:
                label = LABEL_BUY
            elif future_return < -threshold:
                label = LABEL_SELL
            else:
                label = LABEL_HOLD

            rows.append((df.index[i], symbol_key, features, label, future_return, spread_cost))

    if not rows:
        return DatasetBundle(
            features=np.zeros((0, 0), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            symbols=np.array([], dtype=object),
            timestamps=np.array([], dtype="datetime64[ns]"),
            future_returns=np.zeros((0,), dtype=np.float32),
            spread_costs=np.zeros((0,), dtype=np.float32),
        )

    # Keep strict temporal ordering for train/validation splits.
    rows.sort(key=lambda row: row[0])

    features = np.vstack([row[2] for row in rows]).astype(np.float32)
    labels = np.array([row[3] for row in rows], dtype=np.int64)
    symbols = np.array([row[1] for row in rows], dtype=object)
    timestamps = np.array([row[0] for row in rows], dtype="datetime64[ns]")
    future_returns = np.array([row[4] for row in rows], dtype=np.float32)
    spread_costs = np.array([row[5] for row in rows], dtype=np.float32)

    print(f"Created {len(features)} samples with feature_dim={features.shape[1]}")
    label_counts = np.bincount(labels, minlength=3)
    print(
        "Label distribution: "
        f"SELL={label_counts[LABEL_SELL]}, "
        f"HOLD={label_counts[LABEL_HOLD]}, "
        f"BUY={label_counts[LABEL_BUY]}"
    )
    return DatasetBundle(
        features=features,
        labels=labels,
        symbols=symbols,
        timestamps=timestamps,
        future_returns=future_returns,
        spread_costs=spread_costs,
    )


def fit_scaler(train_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(np.abs(std) < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / (std + 1e-8)).astype(np.float32)


def compute_class_weights(labels: np.ndarray, mode: str = "sqrt_balanced") -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    total = float(np.sum(counts))
    weights = np.ones(3, dtype=np.float32)
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        if mode == "balanced":
            weights[idx] = total / (3.0 * count)
        elif mode == "sqrt_balanced":
            weights[idx] = math.sqrt(total / (3.0 * count))
        elif mode == "none":
            weights[idx] = 1.0
        else:
            weights[idx] = total / (3.0 * count)
    return torch.tensor(weights, dtype=torch.float32)


class SimpleNeuralNetwork(nn.Module):
    """Runtime-compatible classifier architecture."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class Metrics:
    accuracy: float
    win_rate: float
    trade_rate: float
    trade_count: int


def evaluate_predictions(probs: np.ndarray, true_labels: np.ndarray) -> Metrics:
    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == true_labels)) if len(true_labels) else 0.0
    trade_mask = preds != LABEL_HOLD
    trade_count = int(np.sum(trade_mask))
    trade_rate = float(np.mean(trade_mask)) if len(true_labels) else 0.0
    if trade_count > 0:
        win_rate = float(np.mean(preds[trade_mask] == true_labels[trade_mask]))
    else:
        win_rate = 0.0
    return Metrics(
        accuracy=accuracy,
        win_rate=win_rate,
        trade_rate=trade_rate,
        trade_count=trade_count,
    )


def evaluate_directional_quality(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold_grid: Optional[np.ndarray] = None,
    min_trades: int = 200,
) -> Dict[str, float]:
    """
    Evaluate selective trade quality using directional probabilities only.
    Returns the best thresholded win-rate profile.
    """
    if threshold_grid is None:
        threshold_grid = np.arange(0.35, 0.96, 0.02)

    preds = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(probs)), preds]

    def quality_score(win_rate: float, trades: int, total_samples: int) -> float:
        trade_rate = trades / max(1, total_samples)
        sample_penalty = 0.0
        if trades < min_trades:
            sample_penalty = (min_trades - trades) / max(1.0, float(min_trades))
        return (win_rate - 0.5) * math.sqrt(max(1, trades)) + 0.35 * trade_rate - 0.80 * sample_penalty

    best = {
        "threshold": 0.65,
        "win_rate": 0.0,
        "trade_count": 0.0,
        "trade_rate": 0.0,
        "score": -1e9,
    }

    for threshold in threshold_grid:
        trade_mask = (preds != LABEL_HOLD) & (confidence >= threshold)
        trades = int(np.sum(trade_mask))
        if trades < min_trades:
            continue

        wins = int(np.sum(preds[trade_mask] == labels[trade_mask]))
        win_rate = wins / trades
        trade_rate = trades / max(1, len(labels))
        score = quality_score(win_rate, trades, len(labels))
        if score > best["score"]:
            best = {
                "threshold": float(threshold),
                "win_rate": float(win_rate),
                "trade_count": float(trades),
                "trade_rate": float(trade_rate),
                "score": float(score),
            }

    # Fallback when model confidence is generally low.
    if best["trade_count"] == 0:
        for threshold in threshold_grid:
            trade_mask = (preds != LABEL_HOLD) & (confidence >= threshold)
            trades = int(np.sum(trade_mask))
            if trades < 10:
                continue
            wins = int(np.sum(preds[trade_mask] == labels[trade_mask]))
            win_rate = wins / trades
            trade_rate = trades / max(1, len(labels))
            score = quality_score(win_rate, trades, len(labels))
            if score > best["score"]:
                best = {
                    "threshold": float(threshold),
                    "win_rate": float(win_rate),
                    "trade_count": float(trades),
                    "trade_rate": float(trade_rate),
                    "score": float(score),
                }

    return best


def _normal_cdf(x: float) -> float:
    """Numerically-stable Normal(0,1) CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def estimate_weekly_tail_metrics(
    net_returns: np.ndarray,
    trade_rate: float,
    weekly_sample_count: float,
) -> Dict[str, float]:
    """
    Approximate 5-day tail behavior from per-trade net returns.

    We model summed net-return over expected weekly trades using
    a normal approximation. This is lightweight and stable enough
    for threshold-search loops.
    """
    if len(net_returns) == 0 or weekly_sample_count <= 0:
        return {
            "weekly_expected_trades": 0.0,
            "weekly_expected_return": 0.0,
            "weekly_p10_return": 0.0,
            "weekly_p50_return": 0.0,
            "weekly_p90_return": 0.0,
            "weekly_prob_positive": 0.0,
        }

    expected_trades = max(1.0, float(trade_rate) * float(weekly_sample_count))
    mean_trade_return = float(np.mean(net_returns))
    std_trade_return = float(np.std(net_returns))
    std_trade_return = max(std_trade_return, 1e-8)

    weekly_mean = expected_trades * mean_trade_return
    weekly_std = math.sqrt(expected_trades) * std_trade_return
    z90 = 1.2815515655446004
    z10 = -z90
    z = weekly_mean / (weekly_std + 1e-12)

    return {
        "weekly_expected_trades": float(expected_trades),
        "weekly_expected_return": float(weekly_mean),
        "weekly_p10_return": float(weekly_mean + z10 * weekly_std),
        "weekly_p50_return": float(weekly_mean),
        "weekly_p90_return": float(weekly_mean + z90 * weekly_std),
        "weekly_prob_positive": float(np.clip(_normal_cdf(z), 0.0, 1.0)),
    }


def evaluate_profitability_quality(
    probs: np.ndarray,
    future_returns: np.ndarray,
    spread_costs: np.ndarray,
    threshold_grid: Optional[np.ndarray] = None,
    min_trades: int = 120,
    weekly_sample_count: float = 0.0,
    tail_focus_weight: float = 0.0,
    target_weekly_trades: float = 0.0,
    target_trade_weight: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate thresholded directional profitability.
    Uses net directional return (after spread approximation) instead of pure win rate.
    """
    if threshold_grid is None:
        threshold_grid = np.arange(0.35, 0.96, 0.02)

    preds = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(probs)), preds]

    best = {
        "threshold": 0.65,
        "trade_count": 0.0,
        "trade_rate": 0.0,
        "win_rate": 0.0,
        "expectancy": -1.0,
        "profit_factor": 0.0,
        "profitability_rate": -1.0,
        "weekly_expected_trades": 0.0,
        "weekly_expected_return": 0.0,
        "weekly_p10_return": 0.0,
        "weekly_p50_return": 0.0,
        "weekly_p90_return": 0.0,
        "weekly_prob_positive": 0.0,
        "score": -1e9,
    }

    for threshold in threshold_grid:
        trade_mask = (preds != LABEL_HOLD) & (confidence >= threshold)
        trades = int(np.sum(trade_mask))
        if trades < min_trades:
            continue

        directional = np.where(
            preds[trade_mask] == LABEL_BUY,
            future_returns[trade_mask],
            -future_returns[trade_mask],
        )
        net_returns = directional - (spread_costs[trade_mask] * 1.1)
        if len(net_returns) == 0:
            continue

        win_rate = float(np.mean(net_returns > 0))
        expectancy = float(np.mean(net_returns))
        trade_rate = trades / max(1, len(future_returns))

        gains = float(np.sum(net_returns[net_returns > 0]))
        losses = float(-np.sum(net_returns[net_returns < 0]))
        profit_factor = gains / losses if losses > 1e-12 else float("inf")
        profitability_rate = float(np.prod(1.0 + np.clip(net_returns, -0.95, None)) - 1.0)

        sample_penalty = max(0.0, (min_trades - trades) / max(1.0, float(min_trades)))
        pf_component = min(float(profit_factor), 3.0)
        tail_metrics = estimate_weekly_tail_metrics(
            net_returns=net_returns,
            trade_rate=trade_rate,
            weekly_sample_count=weekly_sample_count,
        )
        tail_score = (
            tail_metrics["weekly_p50_return"] * 22.0
            + tail_metrics["weekly_p90_return"] * 8.0
            + (tail_metrics["weekly_prob_positive"] - 0.5) * 1.2
        )
        cadence_alignment = 0.0
        if target_weekly_trades > 0:
            cadence_alignment = float(
                np.clip(
                    1.0 - abs(tail_metrics["weekly_expected_trades"] - target_weekly_trades) / target_weekly_trades,
                    0.0,
                    1.0,
                )
            )
        score = (
            expectancy * 12000.0
            + (pf_component - 1.0)
            + (win_rate - 0.5) * 0.50
            + trade_rate * 0.10
            + profitability_rate * 4.0
            + tail_focus_weight * tail_score
            + target_trade_weight * cadence_alignment
            - sample_penalty
        )

        if score > best["score"]:
            best = {
                "threshold": float(threshold),
                "trade_count": float(trades),
                "trade_rate": float(trade_rate),
                "win_rate": float(win_rate),
                "expectancy": float(expectancy),
                "profit_factor": float(profit_factor),
                "profitability_rate": float(profitability_rate),
                "weekly_expected_trades": float(tail_metrics["weekly_expected_trades"]),
                "weekly_expected_return": float(tail_metrics["weekly_expected_return"]),
                "weekly_p10_return": float(tail_metrics["weekly_p10_return"]),
                "weekly_p50_return": float(tail_metrics["weekly_p50_return"]),
                "weekly_p90_return": float(tail_metrics["weekly_p90_return"]),
                "weekly_prob_positive": float(tail_metrics["weekly_prob_positive"]),
                "score": float(score),
            }

    # Fallback for low-confidence models.
    if best["trade_count"] == 0:
        for threshold in np.arange(0.30, 0.96, 0.03):
            trade_mask = (preds != LABEL_HOLD) & (confidence >= threshold)
            trades = int(np.sum(trade_mask))
            if trades < 10:
                continue
            directional = np.where(
                preds[trade_mask] == LABEL_BUY,
                future_returns[trade_mask],
                -future_returns[trade_mask],
            )
            net_returns = directional - (spread_costs[trade_mask] * 1.1)
            if len(net_returns) == 0:
                continue
            expectancy = float(np.mean(net_returns))
            gains = float(np.sum(net_returns[net_returns > 0]))
            losses = float(-np.sum(net_returns[net_returns < 0]))
            profit_factor = gains / losses if losses > 1e-12 else float("inf")
            win_rate = float(np.mean(net_returns > 0))
            trade_rate = trades / max(1, len(future_returns))
            tail_metrics = estimate_weekly_tail_metrics(
                net_returns=net_returns,
                trade_rate=trade_rate,
                weekly_sample_count=weekly_sample_count,
            )
            tail_score = (
                tail_metrics["weekly_p50_return"] * 22.0
                + tail_metrics["weekly_p90_return"] * 8.0
                + (tail_metrics["weekly_prob_positive"] - 0.5) * 1.2
            )
            cadence_alignment = 0.0
            if target_weekly_trades > 0:
                cadence_alignment = float(
                    np.clip(
                        1.0 - abs(tail_metrics["weekly_expected_trades"] - target_weekly_trades) / target_weekly_trades,
                        0.0,
                        1.0,
                    )
                )
            score = (
                expectancy * 10000.0
                + min(float(profit_factor), 3.0)
                + (win_rate - 0.5) * 0.30
                + tail_focus_weight * tail_score
                + target_trade_weight * cadence_alignment
            )
            if score > best["score"]:
                best = {
                    "threshold": float(threshold),
                    "trade_count": float(trades),
                    "trade_rate": float(trade_rate),
                    "win_rate": float(win_rate),
                    "expectancy": float(expectancy),
                    "profit_factor": float(profit_factor),
                    "profitability_rate": float(np.prod(1.0 + np.clip(net_returns, -0.95, None)) - 1.0),
                    "weekly_expected_trades": float(tail_metrics["weekly_expected_trades"]),
                    "weekly_expected_return": float(tail_metrics["weekly_expected_return"]),
                    "weekly_p10_return": float(tail_metrics["weekly_p10_return"]),
                    "weekly_p50_return": float(tail_metrics["weekly_p50_return"]),
                    "weekly_p90_return": float(tail_metrics["weekly_p90_return"]),
                    "weekly_prob_positive": float(tail_metrics["weekly_prob_positive"]),
                    "score": float(score),
                }

    return best


def compute_symbol_validation_stats(
    probs: np.ndarray,
    labels: np.ndarray,
    symbols: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute per-symbol diagnostics for validation slice."""
    diagnostics: Dict[str, Dict[str, float]] = {}
    for symbol in sorted(set(symbols.tolist())):
        mask = symbols == symbol
        symbol_probs = probs[mask]
        symbol_labels = labels[mask]
        metrics = evaluate_predictions(symbol_probs, symbol_labels)
        directional = evaluate_directional_quality(
            probs=symbol_probs,
            labels=symbol_labels,
            min_trades=max(20, int(0.02 * len(symbol_labels))),
        )
        diagnostics[symbol] = {
            "samples": float(np.sum(mask)),
            "accuracy": round(metrics.accuracy, 4),
            "win_rate": round(metrics.win_rate, 4),
            "trade_rate": round(metrics.trade_rate, 4),
            "trade_count": float(metrics.trade_count),
            "directional_win_rate": round(float(directional["win_rate"]), 4),
            "directional_trade_count": float(directional["trade_count"]),
            "directional_threshold": round(float(directional["threshold"]), 3),
        }
    return diagnostics


def compute_symbol_profitability_stats(
    probs: np.ndarray,
    symbols: np.ndarray,
    future_returns: np.ndarray,
    spread_costs: np.ndarray,
    thresholds: Dict[str, float],
    action_modes: Optional[Dict[str, str]] = None,
    weekly_sample_count: float = DEFAULT_WEEKLY_SAMPLES_PER_SYMBOL,
) -> Dict[str, Dict[str, float]]:
    """Compute per-symbol net-return stats using calibrated thresholds."""
    preds = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(probs)), preds]
    diagnostics: Dict[str, Dict[str, float]] = {}

    for symbol in sorted(set(symbols.tolist())):
        mask = symbols == symbol
        sym_preds = preds[mask]
        sym_conf = confidence[mask]
        sym_future = future_returns[mask]
        sym_spread = spread_costs[mask]
        threshold = float(thresholds.get(symbol, np.median(list(thresholds.values())) if thresholds else 0.65))
        action_mode = str((action_modes or {}).get(symbol, "normal")).lower()

        trade_mask = (sym_preds != LABEL_HOLD) & (sym_conf >= threshold)
        trades = int(np.sum(trade_mask))
        if trades == 0:
            diagnostics[symbol] = {
                "samples": float(np.sum(mask)),
                "threshold": float(threshold),
                "trade_count": 0.0,
                "trade_rate": 0.0,
                "win_rate": 0.0,
                "avg_trade_return": 0.0,
                "profit_factor": 0.0,
                "profitability_rate": 0.0,
                "weekly_expected_trades": 0.0,
                "weekly_expected_return": 0.0,
                "weekly_p10_return": 0.0,
                "weekly_p50_return": 0.0,
                "weekly_p90_return": 0.0,
                "weekly_prob_positive": 0.0,
                "action_mode": action_mode,
            }
            continue

        directional_normal = np.where(
            sym_preds[trade_mask] == LABEL_BUY,
            sym_future[trade_mask],
            -sym_future[trade_mask],
        )
        directional = -directional_normal if action_mode == "invert" else directional_normal
        net_returns = directional - (sym_spread[trade_mask] * 1.1)

        gains = float(np.sum(net_returns[net_returns > 0]))
        losses = float(-np.sum(net_returns[net_returns < 0]))
        profit_factor = gains / losses if losses > 1e-12 else float("inf")
        trade_rate = float(trades / max(1, np.sum(mask)))
        tail_metrics = estimate_weekly_tail_metrics(
            net_returns=net_returns,
            trade_rate=trade_rate,
            weekly_sample_count=weekly_sample_count,
        )

        diagnostics[symbol] = {
            "samples": float(np.sum(mask)),
            "threshold": float(threshold),
            "trade_count": float(trades),
            "trade_rate": trade_rate,
            "win_rate": float(np.mean(net_returns > 0)),
            "avg_trade_return": float(np.mean(net_returns)),
            "profit_factor": float(profit_factor),
            "profitability_rate": float(np.prod(1.0 + np.clip(net_returns, -0.95, None)) - 1.0),
            "weekly_expected_trades": float(tail_metrics["weekly_expected_trades"]),
            "weekly_expected_return": float(tail_metrics["weekly_expected_return"]),
            "weekly_p10_return": float(tail_metrics["weekly_p10_return"]),
            "weekly_p50_return": float(tail_metrics["weekly_p50_return"]),
            "weekly_p90_return": float(tail_metrics["weekly_p90_return"]),
            "weekly_prob_positive": float(tail_metrics["weekly_prob_positive"]),
            "action_mode": action_mode,
        }

    return diagnostics


def build_symbol_live_profile(
    symbol_profitability: Dict[str, Dict[str, float]],
    min_samples: int = 20,
    min_profit_factor: float = 1.05,
    min_expectancy: float = 0.00001,
    min_trade_rate: float = 0.010,
    min_weekly_prob_positive: float = 0.55,
    min_weekly_p10_return: float = -0.04,
) -> Dict[str, Dict[str, float]]:
    """
    Build a profitability-first live trading profile per symbol.

    Each symbol receives:
    - `enabled`: whether live autotrading should run for this symbol
    - `risk_multiplier`: size scaling for live risk budget
    """
    profile: Dict[str, Dict[str, float]] = {}
    for symbol, stats in sorted(symbol_profitability.items()):
        if not isinstance(stats, dict):
            continue

        trade_count = int(round(float(stats.get("trade_count", stats.get("trades", 0.0)) or 0.0)))
        samples = int(round(float(stats.get("samples", 0.0) or 0.0)))
        expectancy = float(stats.get("avg_trade_return", 0.0) or 0.0)
        profit_factor = float(stats.get("profit_factor", 0.0) or 0.0)
        trade_rate = float(
            stats.get(
                "trade_rate",
                (trade_count / samples) if samples > 0 else 0.0,
            ) or 0.0
        )
        weekly_prob_positive = float(stats.get("weekly_prob_positive", 0.5) or 0.5)
        weekly_p10_return = float(stats.get("weekly_p10_return", -1.0) or -1.0)

        enabled = bool(
            trade_count >= min_samples
            and expectancy > min_expectancy
            and profit_factor >= min_profit_factor
            and trade_rate >= min_trade_rate
            and weekly_prob_positive >= min_weekly_prob_positive
            and weekly_p10_return >= min_weekly_p10_return
        )

        if enabled:
            pf_component = float(np.clip((profit_factor - min_profit_factor) / 0.35, 0.0, 1.0))
            exp_denominator = max(min_expectancy * 4.0, 1e-8)
            exp_component = float(np.clip((expectancy - min_expectancy) / exp_denominator, 0.0, 1.0))
            sample_component = float(np.clip(trade_count / 250.0, 0.0, 1.0))
            tail_prob_component = float(
                np.clip((weekly_prob_positive - min_weekly_prob_positive) / 0.25, 0.0, 1.0)
            )
            tail_p10_component = float(
                np.clip((weekly_p10_return - min_weekly_p10_return) / 0.06, 0.0, 1.0)
            )
            risk_multiplier = float(
                np.clip(
                    0.50
                    + 0.26 * pf_component
                    + 0.22 * exp_component
                    + 0.18 * sample_component
                    + 0.12 * tail_prob_component
                    + 0.12 * tail_p10_component,
                    0.45,
                    1.35,
                )
            )
        else:
            risk_multiplier = 0.0

        profile[symbol] = {
            "enabled": bool(enabled),
            "risk_multiplier": float(risk_multiplier),
            "expectancy": float(expectancy),
            "profit_factor": float(profit_factor),
            "trade_count": int(trade_count),
            "trade_rate": float(trade_rate),
            "weekly_prob_positive": float(weekly_prob_positive),
            "weekly_p10_return": float(weekly_p10_return),
        }

    return profile


def model_predict_proba(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def train_neural_network(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    epochs: int = 120,
    learning_rate: float = 0.001,
    batch_size: int = 256,
    class_weight_mode: str = "sqrt_balanced",
    label_smoothing: float = 0.03,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train model with class balancing and early stopping."""
    model = SimpleNeuralNetwork(train_features.shape[1])
    class_weights = compute_class_weights(train_labels, mode=class_weight_mode)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    x_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    x_val = torch.tensor(val_features, dtype=torch.float32)
    y_val = torch.tensor(val_labels, dtype=torch.long)

    best_state = None
    best_val_loss = float("inf")
    patience = 20
    epochs_without_improvement = 0

    print(
        f"Training neural network (epochs={epochs}, batch_size={batch_size}, "
        f"class_weights={class_weight_mode})..."
    )
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_train))
        running_loss = 0.0

        for start in range(0, len(x_train), batch_size):
            idx = permutation[start:start + batch_size]
            batch_x = x_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_x)

        train_loss = running_loss / max(1, len(x_train))

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(criterion(val_logits, y_val).item())
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
            val_metrics = evaluate_predictions(val_probs, val_labels)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_metrics.accuracy:.3f} | "
                f"val_win={val_metrics.win_rate:.3f} | "
                f"val_trade_rate={val_metrics.trade_rate:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_probs = model_predict_proba(model, val_features)
    final_metrics = evaluate_predictions(final_probs, val_labels)
    return model, {
        "val_accuracy": final_metrics.accuracy,
        "val_win_rate": final_metrics.win_rate,
        "val_trade_rate": final_metrics.trade_rate,
        "val_trade_count": float(final_metrics.trade_count),
    }


def run_walk_forward_validation(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int = 3,
    epochs_per_fold: int = 25,
    batch_size: int = 256,
    class_weight_mode: str = "sqrt_balanced",
) -> Dict[str, object]:
    """
    Expanding-window walk-forward validation.
    Trains lightweight fold models to estimate temporal robustness.
    """
    n_samples = len(features)
    if n_samples < 1200:
        return {
            "folds": [],
            "avg_accuracy": 0.0,
            "avg_win_rate": 0.0,
            "note": "insufficient samples for walk-forward",
        }

    segment = max(300, n_samples // (folds + 2))
    fold_results: List[Dict[str, float]] = []

    print("Running walk-forward validation...")
    for fold in range(folds):
        train_end = segment * (fold + 1)
        test_end = min(train_end + segment, n_samples)
        if test_end - train_end < 150:
            break

        x_train = features[:train_end]
        y_train = labels[:train_end]
        x_test = features[train_end:test_end]
        y_test = labels[train_end:test_end]

        fold_model = SimpleNeuralNetwork(features.shape[1])
        class_weights = compute_class_weights(y_train, mode=class_weight_mode)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(fold_model.parameters(), lr=0.001)

        tensor_x = torch.tensor(x_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.long)

        for _ in range(epochs_per_fold):
            fold_model.train()
            permutation = torch.randperm(len(tensor_x))
            for start in range(0, len(tensor_x), batch_size):
                idx = permutation[start:start + batch_size]
                batch_x = tensor_x[idx]
                batch_y = tensor_y[idx]
                optimizer.zero_grad()
                logits = fold_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        probs = model_predict_proba(fold_model, x_test)
        metrics = evaluate_predictions(probs, y_test)
        result = {
            "fold": float(fold + 1),
            "train_samples": float(len(x_train)),
            "test_samples": float(len(x_test)),
            "accuracy": metrics.accuracy,
            "win_rate": metrics.win_rate,
            "trade_rate": metrics.trade_rate,
            "trade_count": float(metrics.trade_count),
        }
        fold_results.append(result)
        print(
            f"  Fold {fold + 1}: "
            f"acc={metrics.accuracy:.3f}, win={metrics.win_rate:.3f}, "
            f"trade_rate={metrics.trade_rate:.3f}"
        )

    if not fold_results:
        return {
            "folds": [],
            "avg_accuracy": 0.0,
            "avg_win_rate": 0.0,
            "note": "walk-forward folds not available",
        }

    avg_accuracy = float(np.mean([f["accuracy"] for f in fold_results]))
    avg_win_rate = float(np.mean([f["win_rate"] for f in fold_results]))
    return {
        "folds": fold_results,
        "avg_accuracy": avg_accuracy,
        "avg_win_rate": avg_win_rate,
    }


def calibrate_symbol_thresholds(
    model: nn.Module,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_symbols: np.ndarray,
    val_future_returns: np.ndarray,
    val_spread_costs: np.ndarray,
    weekly_sample_count: float = DEFAULT_WEEKLY_SAMPLES_PER_SYMBOL,
    tail_focus_weight: float = 0.0,
    target_weekly_trades: float = 0.0,
    target_trade_weight: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, Dict[str, float]], float]:
    """
    Calibrate per-symbol confidence thresholds from validation split.
    Objective balances profitability expectancy, profit factor, and trade quality.
    Also selects symbol action mode (`normal` or `invert`) from validation data.
    """
    probs = model_predict_proba(model, val_features)
    preds = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(probs)), preds]

    thresholds: Dict[str, float] = {}
    action_modes: Dict[str, str] = {}
    diagnostics: Dict[str, Dict[str, float]] = {}

    threshold_grid = np.arange(0.35, 0.96, 0.02)
    unique_symbols = sorted(set(val_symbols.tolist()))

    for symbol in unique_symbols:
        mask = val_symbols == symbol
        sym_preds = preds[mask]
        sym_labels = val_labels[mask]
        sym_conf = confidence[mask]
        sym_future = val_future_returns[mask]
        sym_spread = val_spread_costs[mask]
        min_trades = max(20, int(0.015 * len(sym_labels)))

        best_score = -1e9
        best_threshold = 0.65
        best_win_rate = 0.0
        best_trade_count = 0
        best_expectancy = -1.0
        best_profit_factor = 0.0
        best_profitability_rate = -1.0
        best_mode = "normal"
        best_weekly_expected_trades = 0.0
        best_weekly_expected_return = 0.0
        best_weekly_p10 = 0.0
        best_weekly_p50 = 0.0
        best_weekly_p90 = 0.0
        best_weekly_prob_positive = 0.0

        for action_mode in ("normal", "invert"):
            for threshold in threshold_grid:
                trade_mask = (sym_preds != LABEL_HOLD) & (sym_conf >= threshold)
                trades = int(np.sum(trade_mask))
                if trades < min_trades:
                    continue

                directional_normal = np.where(
                    sym_preds[trade_mask] == LABEL_BUY,
                    sym_future[trade_mask],
                    -sym_future[trade_mask],
                )
                directional = -directional_normal if action_mode == "invert" else directional_normal
                net_returns = directional - (sym_spread[trade_mask] * 1.1)
                if len(net_returns) == 0:
                    continue

                win_rate = float(np.mean(net_returns > 0))
                expectancy = float(np.mean(net_returns))
                trade_rate = trades / max(1, len(sym_labels))
                gains = float(np.sum(net_returns[net_returns > 0]))
                losses = float(-np.sum(net_returns[net_returns < 0]))
                profit_factor = gains / losses if losses > 1e-12 else float("inf")
                profitability_rate = float(np.prod(1.0 + np.clip(net_returns, -0.95, None)) - 1.0)

                # Strongly prefer positive expectancy and PF with adequate trade count.
                sample_penalty = max(0.0, (min_trades - trades) / max(1.0, float(min_trades)))
                pf_component = min(float(profit_factor), 3.0)
                tail_metrics = estimate_weekly_tail_metrics(
                    net_returns=net_returns,
                    trade_rate=trade_rate,
                    weekly_sample_count=weekly_sample_count,
                )
                tail_score = (
                    tail_metrics["weekly_p50_return"] * 22.0
                    + tail_metrics["weekly_p90_return"] * 8.0
                    + (tail_metrics["weekly_prob_positive"] - 0.5) * 1.2
                )
                cadence_alignment = 0.0
                if target_weekly_trades > 0:
                    cadence_alignment = float(
                        np.clip(
                            1.0
                            - abs(tail_metrics["weekly_expected_trades"] - target_weekly_trades)
                            / target_weekly_trades,
                            0.0,
                            1.0,
                        )
                    )
                score = (
                    expectancy * 12000.0
                    + (pf_component - 1.0)
                    + (win_rate - 0.5) * 0.40
                    + trade_rate * 0.15
                    + profitability_rate * 4.0
                    + tail_focus_weight * tail_score
                    + target_trade_weight * cadence_alignment
                    - sample_penalty
                )
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)
                    best_win_rate = float(win_rate)
                    best_trade_count = trades
                    best_expectancy = float(expectancy)
                    best_profit_factor = float(profit_factor)
                    best_profitability_rate = float(profitability_rate)
                    best_mode = action_mode
                    best_weekly_expected_trades = float(tail_metrics["weekly_expected_trades"])
                    best_weekly_expected_return = float(tail_metrics["weekly_expected_return"])
                    best_weekly_p10 = float(tail_metrics["weekly_p10_return"])
                    best_weekly_p50 = float(tail_metrics["weekly_p50_return"])
                    best_weekly_p90 = float(tail_metrics["weekly_p90_return"])
                    best_weekly_prob_positive = float(tail_metrics["weekly_prob_positive"])

        # Fallback for symbols where minimum trade count is too strict.
        if best_trade_count == 0:
            for action_mode in ("normal", "invert"):
                for threshold in np.arange(0.30, 0.96, 0.03):
                    trade_mask = (sym_preds != LABEL_HOLD) & (sym_conf >= threshold)
                    trades = int(np.sum(trade_mask))
                    if trades < 10:
                        continue

                    directional_normal = np.where(
                        sym_preds[trade_mask] == LABEL_BUY,
                        sym_future[trade_mask],
                        -sym_future[trade_mask],
                    )
                    directional = -directional_normal if action_mode == "invert" else directional_normal
                    net_returns = directional - (sym_spread[trade_mask] * 1.1)
                    if len(net_returns) == 0:
                        continue
                    expectancy = float(np.mean(net_returns))
                    wins = float(np.mean(net_returns > 0))
                    gains = float(np.sum(net_returns[net_returns > 0]))
                    losses = float(-np.sum(net_returns[net_returns < 0]))
                    profit_factor = gains / losses if losses > 1e-12 else float("inf")
                    trade_rate = trades / max(1, len(sym_labels))
                    tail_metrics = estimate_weekly_tail_metrics(
                        net_returns=net_returns,
                        trade_rate=trade_rate,
                        weekly_sample_count=weekly_sample_count,
                    )
                    tail_score = (
                        tail_metrics["weekly_p50_return"] * 22.0
                        + tail_metrics["weekly_p90_return"] * 8.0
                        + (tail_metrics["weekly_prob_positive"] - 0.5) * 1.2
                    )
                    cadence_alignment = 0.0
                    if target_weekly_trades > 0:
                        cadence_alignment = float(
                            np.clip(
                                1.0
                                - abs(tail_metrics["weekly_expected_trades"] - target_weekly_trades)
                                / target_weekly_trades,
                                0.0,
                                1.0,
                            )
                        )
                    score = (
                        expectancy * 10000.0
                        + min(float(profit_factor), 3.0)
                        + (wins - 0.5) * 0.30
                        + tail_focus_weight * tail_score
                        + target_trade_weight * cadence_alignment
                    )
                    if score > best_score:
                        best_score = float(score)
                        best_threshold = float(threshold)
                        best_win_rate = float(wins)
                        best_trade_count = trades
                        best_expectancy = float(expectancy)
                        best_profit_factor = float(profit_factor)
                        best_profitability_rate = float(np.prod(1.0 + np.clip(net_returns, -0.95, None)) - 1.0)
                        best_mode = action_mode
                        best_weekly_expected_trades = float(tail_metrics["weekly_expected_trades"])
                        best_weekly_expected_return = float(tail_metrics["weekly_expected_return"])
                        best_weekly_p10 = float(tail_metrics["weekly_p10_return"])
                        best_weekly_p50 = float(tail_metrics["weekly_p50_return"])
                        best_weekly_p90 = float(tail_metrics["weekly_p90_return"])
                        best_weekly_prob_positive = float(tail_metrics["weekly_prob_positive"])

        # If a symbol remains negative expectancy, push threshold high to suppress weak trades.
        if best_trade_count > 0 and (best_expectancy <= 0.0 or best_profit_factor < 1.0):
            best_threshold = max(best_threshold, 0.90)

        thresholds[symbol] = round(best_threshold, 3)
        action_modes[symbol] = best_mode
        diagnostics[symbol] = {
            "threshold": round(best_threshold, 3),
            "action_mode": best_mode,
            "win_rate": round(best_win_rate, 4),
            "expectancy": round(best_expectancy, 6),
            "profit_factor": round(best_profit_factor, 4),
            "profitability_rate": round(best_profitability_rate, 6),
            "trade_count": float(best_trade_count),
            "samples": float(np.sum(mask)),
            "weekly_expected_trades": round(best_weekly_expected_trades, 2),
            "weekly_expected_return": round(best_weekly_expected_return, 6),
            "weekly_p10_return": round(best_weekly_p10, 6),
            "weekly_p50_return": round(best_weekly_p50, 6),
            "weekly_p90_return": round(best_weekly_p90, 6),
            "weekly_prob_positive": round(best_weekly_prob_positive, 6),
        }

    if thresholds:
        global_threshold = float(np.median(list(thresholds.values())))
    else:
        global_threshold = 0.65

    return thresholds, action_modes, diagnostics, round(global_threshold, 3)


def save_model(
    model: nn.Module,
    model_path: str,
    feature_dim: int,
    symbol_to_index: Dict[str, int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    metadata: Dict[str, object],
    push_profiles: Optional[Dict[str, SymbolPushProfile]] = None,
) -> None:
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
    print(f"Model saved to {model_path}")


def main() -> bool:
    print("Advanced Neural Training System")
    print("=" * 60)
    set_reproducibility(42)

    if not setup_mt5():
        print("Failed to setup MT5")
        return False

    requested_symbols = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "BTCUSD"
    ]
    data = collect_historical_data(symbols=requested_symbols, days=180)
    if len(data) < 3:
        print("Insufficient symbol coverage for robust training")
        return False

    symbol_to_index = build_symbol_index(list(data.keys()))

    # Learn per-symbol push profiles from historical swing structure.
    print("\nComputing push profiles from MT5 data...")
    push_profiles = compute_all_push_profiles(data)
    print(f"Push profiles computed for {len(push_profiles)} symbols")

    candidate_configs = [
        {
            "name": "balanced",
            "epochs": 140,
            "learning_rate": 0.001,
            "batch_size": 256,
            "class_weight_mode": "balanced",
            "label_smoothing": 0.02,
        },
        {
            "name": "sqrt_balanced",
            "epochs": 170,
            "learning_rate": 0.0009,
            "batch_size": 224,
            "class_weight_mode": "sqrt_balanced",
            "label_smoothing": 0.03,
        },
        {
            "name": "natural",
            "epochs": 190,
            "learning_rate": 0.0008,
            "batch_size": 192,
            "class_weight_mode": "none",
            "label_smoothing": 0.04,
        },
    ]

    selected = None
    horizon_candidates = [8, 16, 24]
    lookback = 20

    for horizon in horizon_candidates:
        print(f"\nEvaluating horizon={horizon}")
        dataset = create_features_and_labels(
            data,
            symbol_to_index=symbol_to_index,
            lookback=lookback,
            horizon=horizon,
            push_profiles=push_profiles,
        )
        if len(dataset.features) < 2000:
            print(f"Skipping horizon={horizon} due to insufficient samples")
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

        feature_mean = None
        feature_std = None
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
                f"  Candidate {cfg['name']} directional quality: "
                f"win={candidate_quality['win_rate']:.3f}, "
                f"trades={int(candidate_quality['trade_count'])}, "
                f"threshold={candidate_quality['threshold']:.2f}, "
                f"score={candidate_quality['score']:.3f}"
            )
            print(
                f"  Candidate {cfg['name']} profitability quality: "
                f"exp={candidate_profit_quality['expectancy']:.5f}, "
                f"pf={candidate_profit_quality['profit_factor']:.3f}, "
                f"trades={int(candidate_profit_quality['trade_count'])}, "
                f"threshold={candidate_profit_quality['threshold']:.2f}, "
                f"score={candidate_profit_quality['score']:.3f}"
            )

            if (
                selected is None
                or combined_score > selected["combined_score"]
            ):
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

    assert selected is not None
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
    print(
        f"\nSelected candidate: {selected_config['name']} "
        f"(horizon={selected_horizon}, lookback={selected_lookback})"
    )
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

    # Build metadata for runtime + diagnostics.
    metadata = {
        "training_date": datetime.now().isoformat(),
        "symbols_used": requested_symbols,
        "resolved_symbols": [data[s]["resolved_symbol"] for s in sorted(data.keys())],
        "symbol_to_index": symbol_to_index,
        "samples": int(len(dataset.features)),
        "feature_dim": int(dataset.features.shape[1]),
        "lookback": selected_lookback,
        "horizon": selected_horizon,
        "base_feature_dim": BASE_FEATURE_DIM,
        "symbol_feature_dim": len(symbol_to_index),
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
        "note": "symbol-aware training with walk-forward validation and 5-day tail-focused objective",
    }

    save_model(
        model=model,
        model_path="neural_model.pth",
        feature_dim=dataset.features.shape[1],
        symbol_to_index=symbol_to_index,
        feature_mean=feature_mean,
        feature_std=feature_std,
        metadata=metadata,
        push_profiles=push_profiles,
    )

    with open("training_report.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("Training report saved to training_report.json")

    print("\nTraining Summary")
    print(f"  Samples: {len(dataset.features)}")
    print(f"  Feature dim: {dataset.features.shape[1]}")
    print(f"  Lookback/Horizon: {selected_lookback}/{selected_horizon}")
    print(f"  Validation accuracy: {val_summary['val_accuracy']:.3f}")
    print(f"  Validation win rate: {val_summary['val_win_rate']:.3f}")
    print(f"  Validation trade rate: {val_summary['val_trade_rate']:.3f}")
    print(
        "  Directional quality: "
        f"win={directional_quality['win_rate']:.3f}, "
        f"trades={int(directional_quality['trade_count'])}, "
        f"threshold={directional_quality['threshold']:.2f}"
    )
    print(
        "  Profitability quality: "
        f"exp={profitability_quality['expectancy']:.5f}, "
        f"pf={profitability_quality['profit_factor']:.3f}, "
        f"trades={int(profitability_quality['trade_count'])}, "
        f"threshold={profitability_quality['threshold']:.2f}"
    )
    print(f"  Walk-forward avg accuracy: {walk_forward.get('avg_accuracy', 0.0):.3f}")
    print(f"  Walk-forward avg win rate: {walk_forward.get('avg_win_rate', 0.0):.3f}")
    print(f"  Global threshold: {global_threshold:.2f}")
    enabled_symbols = [
        symbol for symbol, cfg in symbol_live_profile.items()
        if bool(cfg.get("enabled", False))
    ]
    if enabled_symbols:
        print("  Live-enabled symbols: " + ", ".join(enabled_symbols))
    else:
        print("  Live-enabled symbols: none")

    quality_ok = (
        profitability_quality.get("expectancy", -1.0) > 0.0
        and profitability_quality.get("profit_factor", 0.0) > 1.0
        and walk_forward.get("avg_win_rate", 0.0) >= 0.45
    )
    if quality_ok:
        print("\nSUCCESS: Model quality is acceptable for controlled live testing.")
    else:
        print("\nWARNING: Model quality is weak; retraining with more data is recommended.")
    return True


if __name__ == "__main__":
    try:
        success = main()
    finally:
        mt5.shutdown()
    raise SystemExit(0 if success else 1)
