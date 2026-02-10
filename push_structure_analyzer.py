"""
Trading Anarchy Push Structure Analyzer
========================================

Data-driven push counting and structural exhaustion detection trained on MT5 data.

Replaces hardcoded push heuristics with learned per-symbol push profiles:
- Proper swing high/low detection (not just bar-by-bar diffs)
- Learned "thread pitch" (pips per push) per symbol
- Learned exhaustion point per symbol (replaces hardcoded 4-push rule)
- 8 neural network features derived from push structure

Analogy (aircraft mechanic):
  Bolt turns = Price pushes (structural impulse moves)
  Thread pitch = Pips per push (market unit of movement)
  Torque limit = Exhaustion push count (stop trading after Nth push)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import argrelextrema


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    """A detected swing high or swing low."""
    index: int
    price: float
    swing_type: str  # "high" or "low"


@dataclass
class PushSequence:
    """A sequence of impulse pushes in one direction."""
    push_count: int
    push_sizes_pips: List[float]
    pullback_sizes_pips: List[float]
    avg_push_pips: float
    avg_pullback_ratio: float
    is_exhausting: bool
    last_push_index: int
    direction: str  # "bullish" or "bearish"


@dataclass
class SymbolPushProfile:
    """Learned push characteristics for a single symbol."""
    symbol: str
    median_pips_per_push: float
    mean_pips_per_push: float
    std_pips_per_push: float
    exhaustion_push_count: int
    reversal_prob_by_push: Dict[int, float]
    min_swing_pips: float
    sample_count: int
    pullback_ratio_mean: float
    pullback_ratio_std: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "median_pips_per_push": self.median_pips_per_push,
            "mean_pips_per_push": self.mean_pips_per_push,
            "std_pips_per_push": self.std_pips_per_push,
            "exhaustion_push_count": self.exhaustion_push_count,
            "reversal_prob_by_push": {str(k): v for k, v in self.reversal_prob_by_push.items()},
            "min_swing_pips": self.min_swing_pips,
            "sample_count": self.sample_count,
            "pullback_ratio_mean": self.pullback_ratio_mean,
            "pullback_ratio_std": self.pullback_ratio_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SymbolPushProfile":
        rev_prob = d.get("reversal_prob_by_push", {})
        rev_prob_int = {int(k): float(v) for k, v in rev_prob.items()}
        return cls(
            symbol=str(d.get("symbol", "")),
            median_pips_per_push=float(d.get("median_pips_per_push", 0.0)),
            mean_pips_per_push=float(d.get("mean_pips_per_push", 0.0)),
            std_pips_per_push=float(d.get("std_pips_per_push", 1.0)),
            exhaustion_push_count=int(d.get("exhaustion_push_count", 4)),
            reversal_prob_by_push=rev_prob_int,
            min_swing_pips=float(d.get("min_swing_pips", 0.0)),
            sample_count=int(d.get("sample_count", 0)),
            pullback_ratio_mean=float(d.get("pullback_ratio_mean", 0.5)),
            pullback_ratio_std=float(d.get("pullback_ratio_std", 0.2)),
        )

    @classmethod
    def default(cls, symbol: str) -> "SymbolPushProfile":
        """Fallback profile when insufficient data is available."""
        return cls(
            symbol=symbol,
            median_pips_per_push=0.0,
            mean_pips_per_push=0.0,
            std_pips_per_push=1.0,
            exhaustion_push_count=4,
            reversal_prob_by_push={1: 0.10, 2: 0.20, 3: 0.35, 4: 0.55, 5: 0.70, 6: 0.80},
            min_swing_pips=0.0,
            sample_count=0,
            pullback_ratio_mean=0.5,
            pullback_ratio_std=0.2,
        )


PUSH_FEATURE_COUNT = 8


# ---------------------------------------------------------------------------
# SwingDetector
# ---------------------------------------------------------------------------

class SwingDetector:
    """
    Detects swing highs and swing lows from OHLC data using local extrema,
    then counts structural pushes (impulse legs) in a given direction.
    """

    @staticmethod
    def detect_swings(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        order: int = 3,
        min_swing_pips: float = 0.0,
    ) -> List[SwingPoint]:
        """
        Detect swing highs and swing lows with alternation enforcement.

        Parameters
        ----------
        highs : array of high prices
        lows : array of low prices
        closes : array of close prices (unused but kept for API consistency)
        order : minimum bars on each side for a swing to qualify
        min_swing_pips : minimum pip distance between consecutive swings
                         (swings closer than this are filtered as noise)

        Returns
        -------
        List of SwingPoint in chronological order, alternating high/low.
        """
        if len(highs) < 2 * order + 1:
            return []

        # Step 1: Find candidate swing highs and lows.
        high_indices = argrelextrema(highs, np.greater_equal, order=order)[0]
        low_indices = argrelextrema(lows, np.less_equal, order=order)[0]

        # Step 2: Merge into chronological list.
        raw_swings: List[SwingPoint] = []
        for idx in high_indices:
            raw_swings.append(SwingPoint(index=int(idx), price=float(highs[idx]), swing_type="high"))
        for idx in low_indices:
            raw_swings.append(SwingPoint(index=int(idx), price=float(lows[idx]), swing_type="low"))
        raw_swings.sort(key=lambda s: (s.index, 0 if s.swing_type == "low" else 1))

        # Step 3: Enforce alternation (no two consecutive highs or two consecutive lows).
        alternated = _enforce_alternation(raw_swings)

        # Step 4: Filter noise — remove swings where pip distance is too small.
        if min_swing_pips > 0 and len(alternated) > 1:
            alternated = _filter_noise_swings(alternated, min_swing_pips)

        return alternated

    @staticmethod
    def count_pushes(
        swings: List[SwingPoint],
        direction: str,
    ) -> PushSequence:
        """
        Count impulse pushes from a swing sequence.

        For bullish: pushes are swing-low-to-swing-high legs where each high
        exceeds the previous high (higher highs) and each low exceeds the
        previous low (higher lows).

        For bearish: pushes are swing-high-to-swing-low legs where each low
        is lower than the previous low and each high is lower than the
        previous high.

        Returns a PushSequence summarizing the current structural state.
        """
        if len(swings) < 2:
            return PushSequence(
                push_count=0, push_sizes_pips=[], pullback_sizes_pips=[],
                avg_push_pips=0.0, avg_pullback_ratio=0.0, is_exhausting=False,
                last_push_index=swings[-1].index if swings else 0,
                direction=direction,
            )

        if direction == "bullish":
            return _count_bullish_pushes(swings)
        else:
            return _count_bearish_pushes(swings)


# ---------------------------------------------------------------------------
# PushStatisticsCollector
# ---------------------------------------------------------------------------

class PushStatisticsCollector:
    """
    Analyzes historical MT5 data to learn per-symbol push behaviour.
    Runs during the training phase.
    """

    @staticmethod
    def compute_symbol_push_stats(
        df: "pd.DataFrame",
        symbol: str,
        point: float,
        order: int = 5,
    ) -> SymbolPushProfile:
        """
        Compute push statistics for a single symbol from M15 OHLC data.

        Parameters
        ----------
        df : DataFrame with at least 'high', 'low', 'close' columns
        symbol : symbol name (e.g. "EURUSD")
        point : MT5 symbol point (e.g. 0.0001 for EURUSD, 0.01 for USDJPY)
        order : swing detection order parameter

        Returns
        -------
        SymbolPushProfile with learned statistics.
        """
        import pandas as pd  # deferred to avoid circular imports

        if df is None or len(df) < 100:
            return SymbolPushProfile.default(symbol)

        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values

        pip_divisor = point if point > 0 else 0.0001

        # First pass: detect swings without noise filter to learn min_swing_pips.
        swings = SwingDetector.detect_swings(highs, lows, closes, order=order, min_swing_pips=0.0)
        if len(swings) < 6:
            return SymbolPushProfile.default(symbol)

        # Compute all inter-swing distances in RAW PRICE (not pips).
        # The swing filter operates on raw prices, so the threshold must be
        # in the same units.
        inter_swing_raw = []
        for i in range(1, len(swings)):
            dist = abs(swings[i].price - swings[i - 1].price)
            inter_swing_raw.append(dist)

        if not inter_swing_raw:
            return SymbolPushProfile.default(symbol)

        # min_swing_filter is in raw price units (for noise filtering).
        min_swing_filter = float(np.percentile(inter_swing_raw, 10))
        # min_swing_pips is stored in pips for the profile (human readable).
        min_swing_pips = min_swing_filter / pip_divisor

        # Second pass: detect swings with noise filter (using raw price threshold).
        swings = SwingDetector.detect_swings(highs, lows, closes, order=order, min_swing_pips=min_swing_filter)
        if len(swings) < 6:
            return SymbolPushProfile.default(symbol)

        # Identify all completed push sequences in both directions.
        all_sequences = _extract_all_push_sequences(swings, pip_divisor)

        if len(all_sequences) < 10:
            return SymbolPushProfile.default(symbol)

        # Aggregate statistics.
        all_push_sizes = []
        all_pullback_ratios = []
        sequence_lengths = []

        for seq in all_sequences:
            sequence_lengths.append(seq.push_count)
            all_push_sizes.extend(seq.push_sizes_pips)
            for i, pb in enumerate(seq.pullback_sizes_pips):
                if i < len(seq.push_sizes_pips) and seq.push_sizes_pips[i] > 0:
                    all_pullback_ratios.append(pb / seq.push_sizes_pips[i])

        if not all_push_sizes:
            return SymbolPushProfile.default(symbol)

        median_pips = float(np.median(all_push_sizes))
        mean_pips = float(np.mean(all_push_sizes))
        std_pips = float(np.std(all_push_sizes)) if len(all_push_sizes) > 1 else 1.0
        if std_pips < 1e-8:
            std_pips = 1.0

        pb_mean = float(np.mean(all_pullback_ratios)) if all_pullback_ratios else 0.5
        pb_std = float(np.std(all_pullback_ratios)) if len(all_pullback_ratios) > 1 else 0.2

        # Build reversal probability curve.
        # Only consider sequences with 2+ pushes for meaningful statistics.
        # A single push that reverses is noise, not a tradeable structure.
        meaningful_lengths = [l for l in sequence_lengths if l >= 2]
        reversal_prob = _compute_reversal_probabilities(
            meaningful_lengths if meaningful_lengths else sequence_lengths,
            max_push=8,
        )

        # Exhaustion push count: smallest N (>= 3) where P(reversal) >= 0.60.
        # Minimum of 3 ensures the system stays tradeable — pushes 1-2 are the
        # entry zone, not the exhaustion zone. Use 60% threshold (not 50%) to
        # avoid being overly conservative on M15 noise.
        exhaustion_count = 4  # default fallback (mentor's heuristic)
        for n in range(3, 9):
            if reversal_prob.get(n, 0.0) >= 0.60:
                exhaustion_count = n
                break

        return SymbolPushProfile(
            symbol=symbol,
            median_pips_per_push=median_pips,
            mean_pips_per_push=mean_pips,
            std_pips_per_push=std_pips,
            exhaustion_push_count=exhaustion_count,
            reversal_prob_by_push=reversal_prob,
            min_swing_pips=min_swing_pips,
            sample_count=len(all_sequences),
            pullback_ratio_mean=pb_mean,
            pullback_ratio_std=pb_std,
        )


# ---------------------------------------------------------------------------
# PushFeatureExtractor
# ---------------------------------------------------------------------------

class PushFeatureExtractor:
    """
    Extracts 8 push-derived features for neural network input.
    """

    @staticmethod
    def extract_push_features(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        profile: SymbolPushProfile,
        point: float,
        direction: str,
        order: int = 5,
    ) -> np.ndarray:
        """
        Extract push structure features for a single data point.

        Parameters
        ----------
        highs : recent high prices (lookback window)
        lows : recent low prices (lookback window)
        closes : recent close prices (lookback window)
        profile : learned SymbolPushProfile for this symbol
        point : MT5 symbol point for pip conversion
        direction : "bullish" or "bearish" (from market context)
        order : swing detection sensitivity

        Returns
        -------
        np.ndarray of shape (8,) with push features.
        """
        zeros = np.zeros(PUSH_FEATURE_COUNT, dtype=np.float32)

        if len(highs) < 2 * order + 2:
            return zeros

        pip_divisor = point if point > 0 else 0.0001

        # Convert min_swing_pips (in pip units) back to raw price for the filter.
        min_swing_raw = profile.min_swing_pips * pip_divisor

        swings = SwingDetector.detect_swings(
            highs, lows, closes,
            order=order,
            min_swing_pips=min_swing_raw,
        )

        if len(swings) < 2:
            return zeros

        seq = SwingDetector.count_pushes(swings, direction)

        if seq.push_count == 0:
            return zeros

        # Feature 0: push_count_normalized = current / exhaustion
        exh = max(profile.exhaustion_push_count, 1)
        push_count_normalized = float(seq.push_count) / exh

        # Feature 1: push_count_raw_scaled = raw / 8
        push_count_raw_scaled = float(min(seq.push_count, 8)) / 8.0

        # Feature 2: pips_per_push_ratio = latest push pips / median
        latest_push_pips = seq.push_sizes_pips[-1] if seq.push_sizes_pips else 0.0
        median = max(profile.median_pips_per_push, 1e-8)
        pips_per_push_ratio = latest_push_pips / median

        # Feature 3: exhaustion_proximity (same as 0 but kept for clarity; captures non-linearity)
        exhaustion_proximity = push_count_normalized

        # Feature 4: reversal_probability
        reversal_probability = profile.reversal_prob_by_push.get(seq.push_count, 0.0)

        # Feature 5: push_momentum_decay
        push_momentum_decay = 0.0
        if len(seq.push_sizes_pips) >= 2:
            prev_push = seq.push_sizes_pips[-2]
            curr_push = seq.push_sizes_pips[-1]
            if prev_push > 1e-8:
                push_momentum_decay = (curr_push - prev_push) / prev_push

        # Feature 6: pullback_deepening
        pullback_deepening = 0.0
        if seq.pullback_sizes_pips and profile.pullback_ratio_mean > 1e-8:
            latest_pullback = seq.pullback_sizes_pips[-1]
            latest_push_for_ratio = seq.push_sizes_pips[-1] if seq.push_sizes_pips else 1.0
            if latest_push_for_ratio > 1e-8:
                current_ratio = latest_pullback / latest_push_for_ratio
                pullback_deepening = current_ratio / max(profile.pullback_ratio_mean, 1e-8)

        # Feature 7: thread_pitch_deviation (z-score)
        thread_pitch_deviation = 0.0
        std = max(profile.std_pips_per_push, 1e-8)
        if latest_push_pips > 0:
            thread_pitch_deviation = (latest_push_pips - profile.median_pips_per_push) / std

        features = np.array(
            [
                push_count_normalized,
                push_count_raw_scaled,
                _clip_feature(pips_per_push_ratio, 0.0, 5.0),
                exhaustion_proximity,
                reversal_probability,
                _clip_feature(push_momentum_decay, -2.0, 2.0),
                _clip_feature(pullback_deepening, 0.0, 5.0),
                _clip_feature(thread_pitch_deviation, -4.0, 4.0),
            ],
            dtype=np.float32,
        )

        # Replace any NaN/Inf with 0.
        features = np.where(np.isfinite(features), features, 0.0).astype(np.float32)
        return features


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _clip_feature(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _enforce_alternation(swings: List[SwingPoint]) -> List[SwingPoint]:
    """
    Enforce that swings alternate between high and low.
    If two consecutive highs appear, keep the higher one.
    If two consecutive lows appear, keep the lower one.
    """
    if len(swings) < 2:
        return list(swings)

    result: List[SwingPoint] = [swings[0]]

    for i in range(1, len(swings)):
        current = swings[i]
        prev = result[-1]

        if current.swing_type == prev.swing_type:
            # Same type: keep the more extreme one.
            if current.swing_type == "high":
                if current.price >= prev.price:
                    result[-1] = current
            else:
                if current.price <= prev.price:
                    result[-1] = current
        else:
            result.append(current)

    return result


def _filter_noise_swings(swings: List[SwingPoint], min_pips: float) -> List[SwingPoint]:
    """
    Remove swing pairs where the distance between them is less than min_pips.
    Iteratively removes the smallest swing pair until all remaining pairs
    exceed the threshold.
    """
    if min_pips <= 0 or len(swings) < 3:
        return list(swings)

    filtered = list(swings)
    changed = True
    while changed and len(filtered) > 2:
        changed = False
        smallest_dist = float("inf")
        smallest_idx = -1

        for i in range(len(filtered) - 1):
            dist = abs(filtered[i + 1].price - filtered[i].price)
            if dist < smallest_dist:
                smallest_dist = dist
                smallest_idx = i

        if smallest_dist < min_pips and smallest_idx >= 0:
            # Remove the less significant point of the pair.
            # If both are removable, remove the inner one (i+1 unless it's the last).
            if smallest_idx + 1 < len(filtered) - 1:
                filtered.pop(smallest_idx + 1)
            elif smallest_idx > 0:
                filtered.pop(smallest_idx)
            else:
                filtered.pop(smallest_idx + 1)
            # Re-enforce alternation after removal.
            filtered = _enforce_alternation(filtered)
            changed = True

    return filtered


def _count_bullish_pushes(swings: List[SwingPoint]) -> PushSequence:
    """
    Count bullish impulse pushes: higher highs with higher lows.
    A push is a swing-low-to-swing-high leg in an uptrend structure.
    """
    push_sizes: List[float] = []
    pullback_sizes: List[float] = []
    last_push_idx = swings[-1].index if swings else 0

    # Find the most recent bullish sequence scanning backward.
    # We look for the latest run of higher-highs / higher-lows.

    # Separate highs and lows from the alternating sequence.
    highs = [(s.index, s.price) for s in swings if s.swing_type == "high"]
    lows = [(s.index, s.price) for s in swings if s.swing_type == "low"]

    if not highs or not lows:
        return PushSequence(
            push_count=0, push_sizes_pips=[], pullback_sizes_pips=[],
            avg_push_pips=0.0, avg_pullback_ratio=0.0, is_exhausting=False,
            last_push_index=last_push_idx, direction="bullish",
        )

    # Walk the alternating swing sequence to find push legs.
    # For bullish: each (low → high) pair where the high exceeds previous high.
    push_legs: List[Tuple[SwingPoint, SwingPoint]] = []  # (low, high) pairs
    pullback_legs: List[Tuple[SwingPoint, SwingPoint]] = []  # (high, low) pairs

    # Build pairs from the alternating sequence.
    prev_high_price = -float("inf")
    prev_low_price = -float("inf")
    sequence_started = False

    i = 0
    # Find the first low to start from.
    while i < len(swings) and swings[i].swing_type != "low":
        i += 1

    while i < len(swings) - 1:
        low = swings[i]
        high = swings[i + 1] if i + 1 < len(swings) else None

        if low.swing_type != "low" or high is None or high.swing_type != "high":
            i += 1
            continue

        if not sequence_started:
            # Start a new sequence.
            push_sizes.append(high.price - low.price)
            prev_high_price = high.price
            prev_low_price = low.price
            last_push_idx = high.index
            sequence_started = True
            i += 2
            continue

        # Check higher-high and higher-low conditions.
        if high.price > prev_high_price and low.price > prev_low_price:
            # Valid continuation push.
            push_sizes.append(high.price - low.price)
            # The pullback is from previous high to this low.
            pullback_sizes.append(prev_high_price - low.price)
            prev_high_price = high.price
            prev_low_price = low.price
            last_push_idx = high.index
            i += 2
        else:
            # Structure broken. Start fresh from this point.
            # Keep only the most recent sequence.
            push_sizes = [high.price - low.price]
            pullback_sizes = []
            prev_high_price = high.price
            prev_low_price = low.price
            last_push_idx = high.index
            i += 2

    return _build_push_sequence(push_sizes, pullback_sizes, last_push_idx, "bullish")


def _count_bearish_pushes(swings: List[SwingPoint]) -> PushSequence:
    """
    Count bearish impulse pushes: lower lows with lower highs.
    A push is a swing-high-to-swing-low leg in a downtrend structure.
    """
    push_sizes: List[float] = []
    pullback_sizes: List[float] = []
    last_push_idx = swings[-1].index if swings else 0

    highs = [(s.index, s.price) for s in swings if s.swing_type == "high"]
    lows = [(s.index, s.price) for s in swings if s.swing_type == "low"]

    if not highs or not lows:
        return PushSequence(
            push_count=0, push_sizes_pips=[], pullback_sizes_pips=[],
            avg_push_pips=0.0, avg_pullback_ratio=0.0, is_exhausting=False,
            last_push_index=last_push_idx, direction="bearish",
        )

    prev_high_price = float("inf")
    prev_low_price = float("inf")
    sequence_started = False

    i = 0
    # Find the first high to start from.
    while i < len(swings) and swings[i].swing_type != "high":
        i += 1

    while i < len(swings) - 1:
        high = swings[i]
        low = swings[i + 1] if i + 1 < len(swings) else None

        if high.swing_type != "high" or low is None or low.swing_type != "low":
            i += 1
            continue

        if not sequence_started:
            push_sizes.append(high.price - low.price)
            prev_high_price = high.price
            prev_low_price = low.price
            last_push_idx = low.index
            sequence_started = True
            i += 2
            continue

        # Check lower-low and lower-high conditions.
        if low.price < prev_low_price and high.price < prev_high_price:
            push_sizes.append(high.price - low.price)
            pullback_sizes.append(low.price - prev_low_price)  # negative means pullback up
            # Actually for bearish: pullback is from previous low back UP to this high.
            # Let's compute it as abs(this_high - prev_low).
            pullback_sizes[-1] = abs(high.price - prev_low_price)
            prev_high_price = high.price
            prev_low_price = low.price
            last_push_idx = low.index
            i += 2
        else:
            # Structure broken. Restart.
            push_sizes = [high.price - low.price]
            pullback_sizes = []
            prev_high_price = high.price
            prev_low_price = low.price
            last_push_idx = low.index
            i += 2

    return _build_push_sequence(push_sizes, pullback_sizes, last_push_idx, "bearish")


def _build_push_sequence(
    push_sizes: List[float],
    pullback_sizes: List[float],
    last_push_idx: int,
    direction: str,
) -> PushSequence:
    """Build a PushSequence from collected push and pullback sizes."""
    push_count = len(push_sizes)
    avg_push = float(np.mean(push_sizes)) if push_sizes else 0.0

    avg_pullback_ratio = 0.0
    if pullback_sizes and push_sizes:
        ratios = []
        for i, pb in enumerate(pullback_sizes):
            if i < len(push_sizes) and push_sizes[i] > 1e-8:
                ratios.append(abs(pb) / push_sizes[i])
        avg_pullback_ratio = float(np.mean(ratios)) if ratios else 0.0

    # Detect exhaustion: latest push smaller than previous and/or pullback deepening.
    is_exhausting = False
    if len(push_sizes) >= 2:
        if push_sizes[-1] < push_sizes[-2] * 0.8:
            is_exhausting = True
    if len(pullback_sizes) >= 2:
        if abs(pullback_sizes[-1]) > abs(pullback_sizes[-2]) * 1.2:
            is_exhausting = True

    return PushSequence(
        push_count=push_count,
        push_sizes_pips=push_sizes,
        pullback_sizes_pips=pullback_sizes,
        avg_push_pips=avg_push,
        avg_pullback_ratio=avg_pullback_ratio,
        is_exhausting=is_exhausting,
        last_push_index=last_push_idx,
        direction=direction,
    )


def _extract_all_push_sequences(
    swings: List[SwingPoint],
    pip_divisor: float,
) -> List[PushSequence]:
    """
    Extract all completed push sequences (both bullish and bearish) from
    a full chronological swing series. A sequence is "completed" when the
    structure breaks (lower high in an uptrend, higher low in a downtrend).
    """
    if len(swings) < 4:
        return []

    sequences: List[PushSequence] = []

    # Walk through swings and detect trend segments.
    i = 0
    while i < len(swings) - 3:
        # Try bullish sequence starting from a low.
        if swings[i].swing_type == "low" and i + 1 < len(swings) and swings[i + 1].swing_type == "high":
            seq, consumed = _extract_one_bullish_sequence(swings, i, pip_divisor)
            if seq and seq.push_count >= 1:
                sequences.append(seq)
            i += max(consumed, 2)
            continue

        # Try bearish sequence starting from a high.
        if swings[i].swing_type == "high" and i + 1 < len(swings) and swings[i + 1].swing_type == "low":
            seq, consumed = _extract_one_bearish_sequence(swings, i, pip_divisor)
            if seq and seq.push_count >= 1:
                sequences.append(seq)
            i += max(consumed, 2)
            continue

        i += 1

    return sequences


def _extract_one_bullish_sequence(
    swings: List[SwingPoint],
    start: int,
    pip_divisor: float,
) -> Tuple[Optional[PushSequence], int]:
    """Extract one bullish push sequence starting at index `start`."""
    push_sizes: List[float] = []
    pullback_sizes: List[float] = []
    consumed = 0

    prev_high = -float("inf")
    prev_low = -float("inf")

    i = start
    while i < len(swings) - 1:
        low = swings[i]
        high = swings[i + 1] if i + 1 < len(swings) else None

        if low.swing_type != "low" or high is None or high.swing_type != "high":
            break

        if not push_sizes:
            # First push.
            push_pips = (high.price - low.price) / pip_divisor
            push_sizes.append(push_pips)
            prev_high = high.price
            prev_low = low.price
            consumed = 2
            i += 2
            continue

        if high.price > prev_high and low.price > prev_low:
            push_pips = (high.price - low.price) / pip_divisor
            pullback_pips = (prev_high - low.price) / pip_divisor
            push_sizes.append(push_pips)
            pullback_sizes.append(abs(pullback_pips))
            prev_high = high.price
            prev_low = low.price
            consumed += 2
            i += 2
        else:
            # Structure broken — sequence is complete.
            break

    if not push_sizes:
        return None, max(consumed, 2)

    last_idx = swings[start + consumed - 1].index if start + consumed - 1 < len(swings) else swings[-1].index
    seq = _build_push_sequence(push_sizes, pullback_sizes, last_idx, "bullish")
    return seq, consumed


def _extract_one_bearish_sequence(
    swings: List[SwingPoint],
    start: int,
    pip_divisor: float,
) -> Tuple[Optional[PushSequence], int]:
    """Extract one bearish push sequence starting at index `start`."""
    push_sizes: List[float] = []
    pullback_sizes: List[float] = []
    consumed = 0

    prev_high = float("inf")
    prev_low = float("inf")

    i = start
    while i < len(swings) - 1:
        high = swings[i]
        low = swings[i + 1] if i + 1 < len(swings) else None

        if high.swing_type != "high" or low is None or low.swing_type != "low":
            break

        if not push_sizes:
            push_pips = (high.price - low.price) / pip_divisor
            push_sizes.append(push_pips)
            prev_high = high.price
            prev_low = low.price
            consumed = 2
            i += 2
            continue

        if low.price < prev_low and high.price < prev_high:
            push_pips = (high.price - low.price) / pip_divisor
            pullback_pips = abs(high.price - prev_low) / pip_divisor
            push_sizes.append(push_pips)
            pullback_sizes.append(pullback_pips)
            prev_high = high.price
            prev_low = low.price
            consumed += 2
            i += 2
        else:
            break

    if not push_sizes:
        return None, max(consumed, 2)

    last_idx = swings[start + consumed - 1].index if start + consumed - 1 < len(swings) else swings[-1].index
    seq = _build_push_sequence(push_sizes, pullback_sizes, last_idx, "bearish")
    return seq, consumed


def _compute_reversal_probabilities(
    sequence_lengths: List[int],
    max_push: int = 8,
) -> Dict[int, float]:
    """
    Compute P(reversal after push N) from observed sequence lengths.

    For each N: count sequences that ended at exactly N pushes, divided
    by count of sequences that reached at least N pushes.
    """
    if not sequence_lengths:
        return {n: 0.0 for n in range(1, max_push + 1)}

    lengths = np.array(sequence_lengths)
    prob: Dict[int, float] = {}

    for n in range(1, max_push + 1):
        reached_n = int(np.sum(lengths >= n))
        ended_at_n = int(np.sum(lengths == n))

        if reached_n > 0:
            prob[n] = float(ended_at_n) / float(reached_n)
        else:
            # No sequences reached this far; use 1.0 (certain reversal).
            prob[n] = 1.0 if n > 1 else 0.0

    return prob


# ---------------------------------------------------------------------------
# Convenience functions for integration
# ---------------------------------------------------------------------------

def compute_all_push_profiles(
    data: Dict[str, Dict[str, object]],
    order: int = 5,
) -> Dict[str, SymbolPushProfile]:
    """
    Compute push profiles for all symbols in a training data dict.

    Parameters
    ----------
    data : Dict from collect_historical_data(), keyed by symbol.
           Each value has 'M15' (DataFrame), 'point' (float).

    Returns
    -------
    Dict mapping symbol key to SymbolPushProfile.
    """
    profiles: Dict[str, SymbolPushProfile] = {}

    for symbol_key, payload in data.items():
        df = payload.get("M15")
        point = float(payload.get("point", 0.0) or 0.0)
        if point <= 0:
            # Estimate point from price level.
            if df is not None and len(df) > 0:
                price = float(df["close"].iloc[-1])
                point = 0.01 if price > 50 else 0.0001
            else:
                point = 0.0001

        profile = PushStatisticsCollector.compute_symbol_push_stats(
            df=df, symbol=symbol_key, point=point, order=order,
        )
        profiles[symbol_key] = profile

        print(
            f"  {symbol_key}: thread_pitch={profile.median_pips_per_push:.1f} pips, "
            f"exhaustion_at={profile.exhaustion_push_count} pushes, "
            f"samples={profile.sample_count}"
        )

    return profiles


def infer_direction_from_closes(closes: np.ndarray, lookback: int = 10) -> str:
    """Infer bullish/bearish direction from recent close prices."""
    if len(closes) < 2:
        return "bullish"
    recent = closes[-min(lookback, len(closes)):]
    if recent[-1] > recent[0]:
        return "bullish"
    return "bearish"
