"""Chart pattern recognition engine.

Detects classical price-action patterns with proper measured-move targets,
structural stop losses, and retest detection per Trading Anarchy rules.

Every pattern's ``details`` dict contains at minimum:
  - height        (float) — pattern height used for measured move
  - stop_loss     (float) — invalidation price
  - target_price  (float) — exact TP level via measured-move rule

Patterns detected:
  Reversals  — Double/Triple Top & Bottom, Head & Shoulders (Top/Bottom),
               Falling Wedge, Rising Wedge, Rounding Bottom/Top,
               Bullish/Bearish Diamond
  Continuations — Bull/Bear Flag, Bull/Bear Pennant,
                  Bullish/Bearish Rectangle, Ascending/Descending/
                  Symmetrical Triangle, Cup and Handle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

try:
    from push_structure_analyzer import SwingDetector as _SwingDetector
    _HAS_PUSH_ANALYZER = True
except ImportError:
    _HAS_PUSH_ANALYZER = False


@dataclass
class Pattern:
    name: str
    index_start: int
    index_end: int
    confidence: float
    direction: str          # bullish / bearish / neutral
    details: Dict
    push_count: int = 1
    volume_score: float = 0.5


@dataclass
class FormingPattern:
    """A pattern that is still building — the 'puzzle piece' before breakout.

    The neural model uses these as features to predict whether the
    breakout will happen and in which direction, allowing the system
    to trade *before* the breakout instead of waiting for confirmation.
    """
    name: str               # e.g. "Forming Double Bottom", "Triangle Near Apex"
    completion_pct: float   # 0.0–1.0 how close to breakout
    predicted_direction: str  # bullish / bearish
    breakout_level: float   # price level that would confirm the pattern
    stop_loss: float        # invalidation price if pattern fails
    target_price: float     # measured-move TP if pattern completes
    pattern_height: float   # height of the structure
    confidence: float       # base confidence of the forming pattern
    volume_trend: float     # volume behavior during formation (-1 to 1)
    index_start: int
    index_end: int
    details: Dict           # extra info (neckline, slopes, etc.)


class PatternRecognizer:
    """Detect classical chart patterns from OHLCV data."""

    def __init__(self, data: pd.DataFrame):
        required = {"open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.data = data.reset_index(drop=True)
        self.open = self.data["open"].astype(float).values
        self.highs = self.data["high"].astype(float).values
        self.lows = self.data["low"].astype(float).values
        self.close = self.data["close"].astype(float).values
        self.vol = (
            self.data["tick_volume"].astype(float).values
            if "tick_volume" in self.data.columns
            else np.ones(len(self.data), dtype=float)
        )

    # ------------------------------------------------------------------ #
    # Utility helpers                                                      #
    # ------------------------------------------------------------------ #

    def find_peaks(self, order: int = 5) -> np.ndarray:
        return argrelextrema(self.highs, np.greater, order=order)[0]

    def find_troughs(self, order: int = 5) -> np.ndarray:
        return argrelextrema(self.lows, np.less, order=order)[0]

    def _linear_fit(self, indices: np.ndarray, values: np.ndarray) -> Optional[Tuple[float, float]]:
        if len(indices) < 2:
            return None
        slope, intercept = np.polyfit(indices, values, 1)
        return float(slope), float(intercept)

    def _line_value(self, slope: float, intercept: float, idx: int) -> float:
        return slope * idx + intercept

    def _calc_volume_score(self, start_idx: int, end_idx: int) -> float:
        if len(self.vol) < 10:
            return 0.5
        start_idx = max(0, start_idx)
        end_idx = min(len(self.vol) - 1, end_idx)
        if end_idx <= start_idx:
            return 0.5
        window = self.vol[start_idx: end_idx + 1]
        baseline_start = max(0, start_idx - max(20, len(window)))
        baseline = self.vol[baseline_start:start_idx] if start_idx > baseline_start else self.vol[:start_idx]
        if len(baseline) == 0:
            baseline = self.vol[:max(5, len(window))]
        mean_window = float(np.mean(window))
        mean_base = float(np.mean(baseline)) if len(baseline) > 0 else mean_window
        ratio = mean_window / (mean_base + 1e-8)
        return float(max(0.0, min(1.0, 0.5 + (ratio - 1.0) * 0.5)))

    def _estimate_push_count(self, start_idx: int, end_idx: int, direction: str) -> int:
        if end_idx - start_idx < 4:
            return 1
        if _HAS_PUSH_ANALYZER:
            try:
                wh = self.highs[start_idx: end_idx + 1]
                wl = self.lows[start_idx: end_idx + 1]
                wc = self.close[start_idx: end_idx + 1]
                swings = _SwingDetector.detect_swings(wh, wl, wc, order=3)
                if len(swings) >= 2:
                    seq = _SwingDetector.count_pushes(swings, direction)
                    if seq.push_count > 0:
                        return int(max(1, min(seq.push_count, 5)))
            except Exception:
                pass
        closes = self.close[start_idx: end_idx + 1]
        diffs = np.diff(closes)
        sign = 1 if direction == "bullish" else -1
        aligned = (diffs * sign) > 0
        pushes, in_run = 0, False
        for v in aligned:
            if v and not in_run:
                pushes += 1
                in_run = True
            elif not v:
                in_run = False
        return int(max(1, min(pushes, 5)))

    # ------------------------------------------------------------------ #
    # Master detect                                                        #
    # ------------------------------------------------------------------ #

    def detect_all(self, max_age: int = 10) -> List[Pattern]:
        """Detect all patterns and return validated, deduplicated results.

        Parameters
        ----------
        max_age : int
            Only keep patterns whose ``index_end`` is within the last
            *max_age* candles of the dataset.  Set to 0 to disable the
            recency filter (returns everything).
        """
        raw: List[Pattern] = []
        raw.extend(self.detect_double_top_bottom())
        raw.extend(self.detect_triple_top_bottom())
        raw.extend(self.detect_head_and_shoulders())
        raw.extend(self.detect_triangles_and_wedges())
        raw.extend(self.detect_flags_pennants_rectangles())
        raw.extend(self.detect_rounding_and_cup())
        raw.extend(self.detect_diamonds())
        return self._validate_and_filter(raw, max_age=max_age)

    # ------------------------------------------------------------------ #
    # Post-processing: validate TP direction, recency, dedup              #
    # ------------------------------------------------------------------ #

    def _validate_and_filter(
        self, patterns: List[Pattern], max_age: int = 10,
    ) -> List[Pattern]:
        """Remove stale / invalid patterns and deduplicate overlapping ones."""
        last_idx = len(self.data) - 1
        current_price = float(self.close[last_idx])
        out: List[Pattern] = []

        for p in patterns:
            # --- Recency filter ---
            if max_age > 0 and p.index_end < last_idx - max_age:
                continue

            tp = p.details.get("target_price")
            sl = p.details.get("stop_loss")
            if tp is None or sl is None:
                continue

            # --- TP must be on the correct side of current price ---
            if p.direction == "bullish" and tp <= current_price:
                continue
            if p.direction == "bearish" and tp >= current_price:
                continue

            # --- SL must be on the correct side of current price ---
            if p.direction == "bullish" and sl >= current_price:
                continue
            if p.direction == "bearish" and sl <= current_price:
                continue

            # --- Minimum R:R sanity (at least 0.5:1 raw, the 2.0 filter
            #     is enforced in the trading engine) ---
            risk = abs(current_price - sl)
            reward = abs(tp - current_price)
            if risk <= 0 or reward / risk < 0.5:
                continue

            out.append(p)

        # --- Deduplicate overlapping patterns of the same type ---
        out = self._deduplicate(out)
        return out

    @staticmethod
    def _deduplicate(patterns: List[Pattern]) -> List[Pattern]:
        """Keep only the best (highest confidence, then highest volume_score)
        among overlapping patterns of the same type/direction."""
        if not patterns:
            return patterns

        # Sort by confidence descending, then volume_score descending
        patterns = sorted(
            patterns, key=lambda p: (-p.confidence, -p.volume_score),
        )
        kept: List[Pattern] = []
        for p in patterns:
            overlap = False
            for k in kept:
                if k.name != p.name or k.direction != p.direction:
                    continue
                # Check overlap: if >50 % of the shorter span overlaps the other
                overlap_start = max(p.index_start, k.index_start)
                overlap_end = min(p.index_end, k.index_end)
                if overlap_end >= overlap_start:
                    overlap_len = overlap_end - overlap_start
                    shorter = min(
                        p.index_end - p.index_start,
                        k.index_end - k.index_start,
                    )
                    if shorter > 0 and overlap_len / shorter > 0.50:
                        overlap = True
                        break
            if not overlap:
                kept.append(p)
        return kept

    # ------------------------------------------------------------------ #
    # Double Top / Bottom                                                  #
    # TP = neckline projected by pattern height                           #
    # ------------------------------------------------------------------ #

    def detect_double_top_bottom(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 25:
            return patterns
        peaks = self.find_peaks(order=4)
        troughs = self.find_troughs(order=4)
        last_idx = len(self.data) - 1

        # Double Top (bearish)
        for i in range(len(peaks) - 1):
            p1, p2 = int(peaks[i]), int(peaks[i + 1])
            if p2 - p1 < 3:
                continue
            if abs(self.highs[p1] - self.highs[p2]) / (abs(self.highs[p1]) + 1e-8) > 0.01:
                continue
            neck_troughs = [t for t in troughs if p1 < t < p2]
            if not neck_troughs:
                continue
            neckline = float(self.lows[int(neck_troughs[0])])
            # Find actual breakout candle (first close below neckline after p2)
            breakout_idx = None
            for bi in range(p2 + 1, last_idx + 1):
                if self.close[bi] < neckline:
                    breakout_idx = bi
                    break
            if breakout_idx is None:
                continue
            height = float(max(self.highs[p1], self.highs[p2]) - neckline)
            sl = float(max(self.highs[p1], self.highs[p2]) + 0.1 * height)
            tp = float(neckline - height)
            patterns.append(Pattern(
                "Double Top", p1, breakout_idx, 0.82, "bearish",
                {"neckline": neckline, "height": height, "stop_loss": sl, "target_price": tp},
                self._estimate_push_count(p1, breakout_idx, "bearish"),
                self._calc_volume_score(p1, breakout_idx),
            ))

        # Double Bottom (bullish)
        for i in range(len(troughs) - 1):
            t1, t2 = int(troughs[i]), int(troughs[i + 1])
            if t2 - t1 < 3:
                continue
            if abs(self.lows[t1] - self.lows[t2]) / (abs(self.lows[t1]) + 1e-8) > 0.01:
                continue
            neck_peaks = [p for p in peaks if t1 < p < t2]
            if not neck_peaks:
                continue
            neckline = float(self.highs[int(neck_peaks[0])])
            # Find actual breakout candle (first close above neckline after t2)
            breakout_idx = None
            for bi in range(t2 + 1, last_idx + 1):
                if self.close[bi] > neckline:
                    breakout_idx = bi
                    break
            if breakout_idx is None:
                continue
            height = float(neckline - min(self.lows[t1], self.lows[t2]))
            sl = float(min(self.lows[t1], self.lows[t2]) - 0.1 * height)
            tp = float(neckline + height)
            patterns.append(Pattern(
                "Double Bottom", t1, breakout_idx, 0.82, "bullish",
                {"neckline": neckline, "height": height, "stop_loss": sl, "target_price": tp},
                self._estimate_push_count(t1, breakout_idx, "bullish"),
                self._calc_volume_score(t1, breakout_idx),
            ))
        return patterns

    # ------------------------------------------------------------------ #
    # Triple Top / Bottom                                                  #
    # TP = neckline projected by pattern height (like double but 3 touch) #
    # ------------------------------------------------------------------ #

    def detect_triple_top_bottom(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 35:
            return patterns
        peaks = self.find_peaks(order=4)
        troughs = self.find_troughs(order=4)
        last_idx = len(self.data) - 1

        # Triple Top (bearish): 3 peaks at similar level
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                p1, p2, p3 = int(peaks[i]), int(peaks[i + 1]), int(peaks[i + 2])
                if p3 - p1 < 8:
                    continue
                levels = [self.highs[p1], self.highs[p2], self.highs[p3]]
                avg = np.mean(levels)
                if any(abs(l - avg) / (abs(avg) + 1e-8) > 0.015 for l in levels):
                    continue
                neck_t = [t for t in troughs if p1 < t < p3]
                if len(neck_t) < 1:
                    continue
                neckline = float(np.min(self.lows[neck_t]))
                # Find actual breakout candle
                breakout_idx = None
                for bi in range(p3 + 1, last_idx + 1):
                    if self.close[bi] < neckline:
                        breakout_idx = bi
                        break
                if breakout_idx is None:
                    continue
                height = float(max(levels) - neckline)
                sl = float(max(levels) + 0.1 * height)
                tp = float(neckline - height)
                patterns.append(Pattern(
                    "Triple Top", p1, breakout_idx, 0.85, "bearish",
                    {"neckline": neckline, "height": height, "stop_loss": sl, "target_price": tp},
                    self._estimate_push_count(p1, breakout_idx, "bearish"),
                    self._calc_volume_score(p1, breakout_idx),
                ))

        # Triple Bottom (bullish)
        if len(troughs) >= 3:
            for i in range(len(troughs) - 2):
                t1, t2, t3 = int(troughs[i]), int(troughs[i + 1]), int(troughs[i + 2])
                if t3 - t1 < 8:
                    continue
                levels = [self.lows[t1], self.lows[t2], self.lows[t3]]
                avg = np.mean(levels)
                if any(abs(l - avg) / (abs(avg) + 1e-8) > 0.015 for l in levels):
                    continue
                neck_p = [p for p in peaks if t1 < p < t3]
                if len(neck_p) < 1:
                    continue
                neckline = float(np.max(self.highs[neck_p]))
                # Find actual breakout candle
                breakout_idx = None
                for bi in range(t3 + 1, last_idx + 1):
                    if self.close[bi] > neckline:
                        breakout_idx = bi
                        break
                if breakout_idx is None:
                    continue
                height = float(neckline - min(levels))
                sl = float(min(levels) - 0.1 * height)
                tp = float(neckline + height)
                patterns.append(Pattern(
                    "Triple Bottom", t1, breakout_idx, 0.85, "bullish",
                    {"neckline": neckline, "height": height, "stop_loss": sl, "target_price": tp},
                    self._estimate_push_count(t1, breakout_idx, "bullish"),
                    self._calc_volume_score(t1, breakout_idx),
                ))
        return patterns

    # ------------------------------------------------------------------ #
    # Head and Shoulders (Top / Bottom)                                    #
    # TP = neckline projected by head→neckline distance                   #
    # ------------------------------------------------------------------ #

    def detect_head_and_shoulders(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 35:
            return patterns
        peaks = self.find_peaks(order=4)
        troughs = self.find_troughs(order=4)
        last_idx = len(self.data) - 1

        # H&S Top (bearish)
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                l_sh, head, r_sh = int(peaks[i]), int(peaks[i + 1]), int(peaks[i + 2])
                if not (self.highs[head] > self.highs[l_sh] and self.highs[head] > self.highs[r_sh]):
                    continue
                if abs(self.highs[l_sh] - self.highs[r_sh]) / (abs(self.highs[l_sh]) + 1e-8) > 0.05:
                    continue
                neck = [t for t in troughs if l_sh < t < r_sh]
                if not neck:
                    continue
                neckline = float(np.mean(self.lows[neck]))
                # Find actual breakout candle
                breakout_idx = None
                for bi in range(r_sh + 1, last_idx + 1):
                    if self.close[bi] < neckline:
                        breakout_idx = bi
                        break
                if breakout_idx is None:
                    continue
                height = float(self.highs[head] - neckline)
                sl = float(self.highs[r_sh] + 0.1 * height)
                tp = float(neckline - height)
                patterns.append(Pattern(
                    "Head and Shoulders (Top)", l_sh, breakout_idx, 0.80, "bearish",
                    {"neckline": neckline, "height": height, "stop_loss": sl, "target_price": tp},
                    self._estimate_push_count(l_sh, breakout_idx, "bearish"),
                    self._calc_volume_score(l_sh, breakout_idx),
                ))

        # Inverse H&S (bullish)
        if len(troughs) >= 3:
            for i in range(len(troughs) - 2):
                l_sh, head, r_sh = int(troughs[i]), int(troughs[i + 1]), int(troughs[i + 2])
                if not (self.lows[head] < self.lows[l_sh] and self.lows[head] < self.lows[r_sh]):
                    continue
                if abs(self.lows[l_sh] - self.lows[r_sh]) / (abs(self.lows[l_sh]) + 1e-8) > 0.05:
                    continue
                neck = [p for p in peaks if l_sh < p < r_sh]
                if not neck:
                    continue
                neckline = float(np.mean(self.highs[neck]))
                # Find actual breakout candle
                breakout_idx = None
                for bi in range(r_sh + 1, last_idx + 1):
                    if self.close[bi] > neckline:
                        breakout_idx = bi
                        break
                if breakout_idx is None:
                    continue
                height = float(neckline - self.lows[head])
                sl = float(self.lows[r_sh] - 0.1 * height)
                tp = float(neckline + height)
                patterns.append(Pattern(
                    "Head and Shoulders (Bottom)", l_sh, breakout_idx, 0.80, "bullish",
                    {"neckline": neckline, "height": height, "stop_loss": sl, "target_price": tp},
                    self._estimate_push_count(l_sh, breakout_idx, "bullish"),
                    self._calc_volume_score(l_sh, breakout_idx),
                ))
        return patterns

    # ------------------------------------------------------------------ #
    # Triangles and Wedges                                                 #
    # Triangles: TP = pole length (impulse before consolidation)          #
    # Wedges: TP = first structure zone (height of widest point)          #
    # ------------------------------------------------------------------ #

    def detect_triangles_and_wedges(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 45:
            return patterns

        peaks = self.find_peaks(order=3)
        troughs = self.find_troughs(order=3)
        if len(peaks) < 3 or len(troughs) < 3:
            return patterns

        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - 60)
        rp = np.array([p for p in peaks if p >= start_idx])
        rt = np.array([t for t in troughs if t >= start_idx])
        if len(rp) < 2 or len(rt) < 2:
            return patterns

        res_fit = self._linear_fit(rp, self.highs[rp])
        sup_fit = self._linear_fit(rt, self.lows[rt])
        if not res_fit or not sup_fit:
            return patterns

        res_slope, res_intercept = res_fit
        sup_slope, sup_intercept = sup_fit
        res_level = self._line_value(res_slope, res_intercept, end_idx)
        sup_level = self._line_value(sup_slope, sup_intercept, end_idx)
        pattern_height = float(np.max(self.highs[start_idx: end_idx + 1]) - np.min(self.lows[start_idx: end_idx + 1]))
        if pattern_height <= 0:
            return patterns

        vol_score = self._calc_volume_score(start_idx, end_idx)
        converging = (res_level - sup_level) < (pattern_height * 0.7)
        if not converging:
            return patterns

        # Measure the pole (impulse move leading into the triangle)
        pole_lookback = max(0, start_idx - 20)
        pole_high = float(np.max(self.highs[pole_lookback: start_idx + 1]))
        pole_low = float(np.min(self.lows[pole_lookback: start_idx + 1]))
        pole_length = pole_high - pole_low
        # Use pole length if meaningful, otherwise fall back to pattern height
        measured_move = pole_length if pole_length > pattern_height * 0.5 else pattern_height

        breakout_price = float(self.close[end_idx])

        # Ascending Triangle: flat resistance, rising support → bullish breakout
        if abs(res_slope) < 1e-5 and sup_slope > 1e-5 and self.close[end_idx] > res_level:
            sl = float(np.min(self.lows[rt[-2:]]))
            tp = breakout_price + measured_move
            patterns.append(Pattern(
                "Ascending Triangle", start_idx, end_idx, 0.78, "bullish",
                {"height": pattern_height, "stop_loss": sl, "target_price": tp,
                 "pole_length": measured_move},
                self._estimate_push_count(start_idx, end_idx, "bullish"), vol_score,
            ))

        # Descending Triangle: declining resistance, flat support → bearish breakout
        elif res_slope < -1e-5 and abs(sup_slope) < 1e-5 and self.close[end_idx] < sup_level:
            sl = float(np.max(self.highs[rp[-2:]]))
            tp = breakout_price - measured_move
            patterns.append(Pattern(
                "Descending Triangle", start_idx, end_idx, 0.78, "bearish",
                {"height": pattern_height, "stop_loss": sl, "target_price": tp,
                 "pole_length": measured_move},
                self._estimate_push_count(start_idx, end_idx, "bearish"), vol_score,
            ))

        # Symmetrical Triangle: both sides converging
        elif res_slope < 0 and sup_slope > 0:
            if self.close[end_idx] > res_level:
                sl = float(np.min(self.lows[rt[-2:]]))
                tp = breakout_price + measured_move
                patterns.append(Pattern(
                    "Symmetrical Triangle", start_idx, end_idx, 0.74, "bullish",
                    {"height": pattern_height, "stop_loss": sl, "target_price": tp,
                     "pole_length": measured_move},
                    self._estimate_push_count(start_idx, end_idx, "bullish"), vol_score,
                ))
            elif self.close[end_idx] < sup_level:
                sl = float(np.max(self.highs[rp[-2:]]))
                tp = breakout_price - measured_move
                patterns.append(Pattern(
                    "Symmetrical Triangle", start_idx, end_idx, 0.74, "bearish",
                    {"height": pattern_height, "stop_loss": sl, "target_price": tp,
                     "pole_length": measured_move},
                    self._estimate_push_count(start_idx, end_idx, "bearish"), vol_score,
                ))

        # Rising Wedge: both rising, support steeper → bearish reversal
        elif res_slope > 0 and sup_slope > 0 and sup_slope > res_slope and self.close[end_idx] < sup_level:
            sl = float(np.max(self.highs[rp[-2:]]))
            # Wedge TP = first structure zone (widest point of the wedge)
            tp = breakout_price - pattern_height
            patterns.append(Pattern(
                "Rising Wedge", start_idx, end_idx, 0.80, "bearish",
                {"height": pattern_height, "stop_loss": sl, "target_price": tp},
                self._estimate_push_count(start_idx, end_idx, "bearish"), vol_score,
            ))

        # Falling Wedge: both falling, resistance steeper → bullish reversal
        elif res_slope < 0 and sup_slope < 0 and res_slope < sup_slope and self.close[end_idx] > res_level:
            sl = float(np.min(self.lows[rt[-2:]]))
            tp = breakout_price + pattern_height
            patterns.append(Pattern(
                "Falling Wedge", start_idx, end_idx, 0.80, "bullish",
                {"height": pattern_height, "stop_loss": sl, "target_price": tp},
                self._estimate_push_count(start_idx, end_idx, "bullish"), vol_score,
            ))

        return patterns

    # ------------------------------------------------------------------ #
    # Flags, Pennants, and Rectangles                                      #
    # TP = pole length (measured move) from breakout                      #
    # ------------------------------------------------------------------ #

    def detect_flags_pennants_rectangles(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 40:
            return patterns

        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - 30)
        mid_idx = max(start_idx + 5, end_idx - 15)

        pole_move = self.close[mid_idx] - self.close[start_idx]
        trend_dir = "bullish" if pole_move >= 0 else "bearish"
        cons_highs = self.highs[mid_idx: end_idx + 1]
        cons_lows = self.lows[mid_idx: end_idx + 1]
        cons_high = float(np.max(cons_highs))
        cons_low = float(np.min(cons_lows))
        cons_range = cons_high - cons_low
        pole_range = float(np.max(self.highs[start_idx: mid_idx + 1]) - np.min(self.lows[start_idx: mid_idx + 1]))
        if pole_range <= 0 or cons_range > 0.6 * pole_range:
            return patterns

        vol_score = self._calc_volume_score(start_idx, end_idx)
        breakout_price = float(self.close[end_idx])

        # Classify consolidation shape: flag vs pennant vs rectangle
        cons_indices = np.arange(mid_idx, end_idx + 1)
        if len(cons_indices) >= 4:
            p_in_cons = np.array([p for p in self.find_peaks(order=2) if mid_idx <= p <= end_idx])
            t_in_cons = np.array([t for t in self.find_troughs(order=2) if mid_idx <= t <= end_idx])

            res_fit = self._linear_fit(p_in_cons, self.highs[p_in_cons]) if len(p_in_cons) >= 2 else None
            sup_fit = self._linear_fit(t_in_cons, self.lows[t_in_cons]) if len(t_in_cons) >= 2 else None

            if res_fit and sup_fit:
                res_slope = res_fit[0]
                sup_slope = sup_fit[0]
                # Pennant: both lines converging
                is_pennant = (res_slope < 0 and sup_slope > 0) or abs(res_slope + sup_slope) < abs(res_slope - sup_slope) * 0.5
                # Rectangle: both roughly flat
                flat_threshold = cons_range * 0.003
                is_rectangle = abs(res_slope) < flat_threshold and abs(sup_slope) < flat_threshold
            else:
                is_pennant = False
                is_rectangle = False
        else:
            is_pennant = False
            is_rectangle = False

        if trend_dir == "bullish" and self.close[end_idx] > cons_high:
            sl = cons_low
            tp = breakout_price + pole_range
            if is_pennant:
                name = "Bull Pennant"
                conf = 0.74
            elif is_rectangle:
                name = "Bullish Rectangle"
                conf = 0.72
            else:
                name = "Bull Flag"
                conf = 0.72
            patterns.append(Pattern(
                name, start_idx, end_idx, conf, "bullish",
                {"height": pole_range, "stop_loss": sl, "target_price": tp,
                 "pole_range": pole_range, "cons_high": cons_high, "cons_low": cons_low},
                2, vol_score,
            ))
        elif trend_dir == "bearish" and self.close[end_idx] < cons_low:
            sl = cons_high
            tp = breakout_price - pole_range
            if is_pennant:
                name = "Bear Pennant"
                conf = 0.74
            elif is_rectangle:
                name = "Bearish Rectangle"
                conf = 0.72
            else:
                name = "Bear Flag"
                conf = 0.72
            patterns.append(Pattern(
                name, start_idx, end_idx, conf, "bearish",
                {"height": pole_range, "stop_loss": sl, "target_price": tp,
                 "pole_range": pole_range, "cons_high": cons_high, "cons_low": cons_low},
                2, vol_score,
            ))
        return patterns

    # ------------------------------------------------------------------ #
    # Rounding Bottom/Top + Cup and Handle                                 #
    # TP = neckline + pattern height                                      #
    # ------------------------------------------------------------------ #

    def detect_rounding_and_cup(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 60:
            return patterns

        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - 50)
        idx = np.arange(start_idx, end_idx + 1)
        prices = self.close[start_idx: end_idx + 1]
        if len(prices) < 10:
            return patterns

        curvature = float(np.polyfit(idx, prices, 2)[0])
        pattern_high = float(np.max(self.highs[start_idx: end_idx + 1]))
        pattern_low = float(np.min(self.lows[start_idx: end_idx + 1]))
        height = pattern_high - pattern_low
        vol_score = self._calc_volume_score(start_idx, end_idx)
        if height <= 0:
            return patterns

        if curvature > 0:
            neckline = pattern_high
            handle_start = end_idx - max(3, (end_idx - start_idx) // 5)
            handle_low = float(np.min(self.lows[handle_start: end_idx + 1]))
            handle_depth = neckline - handle_low
            has_handle = 0 < handle_depth < 0.5 * height and self.close[end_idx] > neckline

            # Find actual breakout candle — first close above neckline
            breakout_idx = None
            # Search from the midpoint of the pattern forward
            search_start = start_idx + (end_idx - start_idx) // 2
            for bi in range(search_start, end_idx + 1):
                if self.close[bi] > neckline:
                    breakout_idx = bi
                    break
            if breakout_idx is None:
                pass  # no breakout yet
            else:
                name = "Cup and Handle" if has_handle else "Rounding Bottom"
                conf = 0.75 if has_handle else 0.68
                sl = handle_low if has_handle else pattern_low
                tp = float(neckline + height)
                patterns.append(Pattern(
                    name, start_idx, breakout_idx, conf, "bullish",
                    {"neckline": neckline, "height": height, "stop_loss": sl, "target_price": tp,
                     "has_handle": has_handle},
                    3, vol_score,
                ))

        elif curvature < 0:
            neckline = pattern_low
            # Find actual breakout candle
            breakout_idx = None
            search_start = start_idx + (end_idx - start_idx) // 2
            for bi in range(search_start, end_idx + 1):
                if self.close[bi] < neckline:
                    breakout_idx = bi
                    break
            if breakout_idx is not None:
                sl = pattern_high
                tp = float(neckline - height)
                patterns.append(Pattern(
                    "Rounding Top", start_idx, breakout_idx, 0.68, "bearish",
                    {"neckline": neckline, "height": height, "stop_loss": sl, "target_price": tp},
                    3, vol_score,
                ))
        return patterns

    # ------------------------------------------------------------------ #
    # Diamonds                                                             #
    # TP = pattern height projected from breakout                         #
    # ------------------------------------------------------------------ #

    def detect_diamonds(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 60:
            return patterns

        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - 40)
        mid = start_idx + ((end_idx - start_idx) // 2)

        first_range = float(np.max(self.highs[start_idx: mid + 1]) - np.min(self.lows[start_idx: mid + 1]))
        second_range = float(np.max(self.highs[mid: end_idx + 1]) - np.min(self.lows[mid: end_idx + 1]))
        if first_range <= 0 or second_range <= 0 or first_range < second_range:
            return patterns

        upper = float(max(np.max(self.highs[start_idx: mid + 1]), np.max(self.highs[mid: end_idx + 1])))
        lower = float(min(np.min(self.lows[start_idx: mid + 1]), np.min(self.lows[mid: end_idx + 1])))
        height = upper - lower
        if height <= 0:
            return patterns

        vol_score = self._calc_volume_score(start_idx, end_idx)

        # Find actual breakout candle
        for bi in range(mid, end_idx + 1):
            if self.close[bi] > upper:
                tp = float(self.close[bi]) + height
                patterns.append(Pattern(
                    "Bullish Diamond", start_idx, bi, 0.65, "bullish",
                    {"height": height, "stop_loss": lower, "target_price": tp}, 3, vol_score,
                ))
                break
            elif self.close[bi] < lower:
                tp = float(self.close[bi]) - height
                patterns.append(Pattern(
                    "Bearish Diamond", start_idx, bi, 0.65, "bearish",
                    {"height": height, "stop_loss": upper, "target_price": tp}, 3, vol_score,
                ))
                break
        return patterns

    # ------------------------------------------------------------------ #
    # Retest detection                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_retest(pattern: Pattern, current_price: float) -> bool:
        """Check if price has pulled back to the broken level (retest).

        Returns True if price is near the breakout zone (within 0.15% tolerance).
        """
        neckline = pattern.details.get("neckline")
        cons_high = pattern.details.get("cons_high")
        cons_low = pattern.details.get("cons_low")

        # Determine the key level to retest
        if neckline is not None:
            level = neckline
        elif pattern.direction == "bullish" and cons_high is not None:
            level = cons_high
        elif pattern.direction == "bearish" and cons_low is not None:
            level = cons_low
        else:
            return False

        tolerance = abs(level) * 0.0015  # 0.15% tolerance
        return abs(current_price - level) <= tolerance

    # ================================================================== #
    # FORMING PATTERN DETECTION — "Puzzle Piece" Pre-Breakout Detection   #
    # ================================================================== #
    #
    # These methods detect INCOMPLETE patterns that are still building.
    # Instead of waiting for breakout confirmation, they identify the
    # structural setup and predict the probable breakout direction.
    # The neural model uses these as features to learn which forming
    # patterns actually complete and which fail.

    def detect_forming_patterns(self) -> List[FormingPattern]:
        """Detect all currently forming (incomplete) patterns.

        Returns a list of ``FormingPattern`` objects that represent
        structures building up but not yet broken out.  These are the
        'puzzle pieces' that predict what happens next.
        """
        forming: List[FormingPattern] = []
        forming.extend(self._forming_double_top_bottom())
        forming.extend(self._forming_head_and_shoulders())
        forming.extend(self._forming_triangle_wedge())
        forming.extend(self._forming_flag_pennant())
        forming.extend(self._forming_support_resistance_approach())
        return forming

    # -------------------------------------------------------------- #
    # Forming Double Top / Bottom                                      #
    # -------------------------------------------------------------- #

    def _forming_double_top_bottom(self) -> List[FormingPattern]:
        """Detect double top/bottom where the second touch is in but
        neckline has NOT been broken yet."""
        forming: List[FormingPattern] = []
        if len(self.data) < 25:
            return forming

        peaks = self.find_peaks(order=4)
        troughs = self.find_troughs(order=4)
        last_idx = len(self.data) - 1
        current_price = float(self.close[last_idx])

        # Forming Double Top: two peaks at similar level, price still above neckline
        for i in range(len(peaks) - 1):
            p1, p2 = int(peaks[i]), int(peaks[i + 1])
            if p2 - p1 < 3:
                continue
            # Second peak must be recent (within last 15 candles)
            if last_idx - p2 > 15:
                continue
            # Peaks at similar level
            if abs(self.highs[p1] - self.highs[p2]) / (abs(self.highs[p1]) + 1e-8) > 0.012:
                continue
            # Find neckline (lowest trough between peaks)
            neck_troughs = [t for t in troughs if p1 < t < p2]
            if not neck_troughs:
                continue
            neckline = float(self.lows[int(neck_troughs[0])])
            top_level = float(max(self.highs[p1], self.highs[p2]))
            height = top_level - neckline
            if height <= 0:
                continue

            # Key check: neckline NOT broken — pattern is still forming
            if current_price < neckline:
                continue  # Already broken out — let the completed detector handle it

            # How close to breakout? Measure proximity to neckline
            distance_to_neckline = current_price - neckline
            pattern_range = top_level - neckline
            completion = 1.0 - min(1.0, distance_to_neckline / (pattern_range + 1e-8))

            # Volume trend: declining volume on second peak is classic
            vol_p1 = float(np.mean(self.vol[max(0, p1 - 2): p1 + 3]))
            vol_p2 = float(np.mean(self.vol[max(0, p2 - 2): p2 + 3]))
            vol_trend = (vol_p2 - vol_p1) / (vol_p1 + 1e-8)
            vol_trend_score = float(np.clip(-vol_trend, -1.0, 1.0))  # declining vol = positive

            sl = float(top_level + 0.1 * height)
            tp = float(neckline - height)

            forming.append(FormingPattern(
                name="Forming Double Top",
                completion_pct=float(np.clip(completion, 0.0, 1.0)),
                predicted_direction="bearish",
                breakout_level=neckline,
                stop_loss=sl,
                target_price=tp,
                pattern_height=height,
                confidence=0.65 + 0.15 * completion,
                volume_trend=vol_trend_score,
                index_start=p1,
                index_end=last_idx,
                details={"neckline": neckline, "top_level": top_level,
                         "peak1": p1, "peak2": p2},
            ))

        # Forming Double Bottom: two troughs at similar level, price still below neckline
        for i in range(len(troughs) - 1):
            t1, t2 = int(troughs[i]), int(troughs[i + 1])
            if t2 - t1 < 3:
                continue
            if last_idx - t2 > 15:
                continue
            if abs(self.lows[t1] - self.lows[t2]) / (abs(self.lows[t1]) + 1e-8) > 0.012:
                continue
            neck_peaks = [p for p in peaks if t1 < p < t2]
            if not neck_peaks:
                continue
            neckline = float(self.highs[int(neck_peaks[0])])
            bottom_level = float(min(self.lows[t1], self.lows[t2]))
            height = neckline - bottom_level
            if height <= 0:
                continue

            # Key check: neckline NOT broken
            if current_price > neckline:
                continue

            distance_to_neckline = neckline - current_price
            pattern_range = neckline - bottom_level
            completion = 1.0 - min(1.0, distance_to_neckline / (pattern_range + 1e-8))

            vol_t1 = float(np.mean(self.vol[max(0, t1 - 2): t1 + 3]))
            vol_t2 = float(np.mean(self.vol[max(0, t2 - 2): t2 + 3]))
            vol_trend = (vol_t2 - vol_t1) / (vol_t1 + 1e-8)
            vol_trend_score = float(np.clip(-vol_trend, -1.0, 1.0))

            sl = float(bottom_level - 0.1 * height)
            tp = float(neckline + height)

            forming.append(FormingPattern(
                name="Forming Double Bottom",
                completion_pct=float(np.clip(completion, 0.0, 1.0)),
                predicted_direction="bullish",
                breakout_level=neckline,
                stop_loss=sl,
                target_price=tp,
                pattern_height=height,
                confidence=0.65 + 0.15 * completion,
                volume_trend=vol_trend_score,
                index_start=t1,
                index_end=last_idx,
                details={"neckline": neckline, "bottom_level": bottom_level,
                         "trough1": t1, "trough2": t2},
            ))

        return forming

    # -------------------------------------------------------------- #
    # Forming Head & Shoulders                                         #
    # -------------------------------------------------------------- #

    def _forming_head_and_shoulders(self) -> List[FormingPattern]:
        """Detect H&S where right shoulder is forming or complete but
        neckline hasn't been broken."""
        forming: List[FormingPattern] = []
        if len(self.data) < 35:
            return forming

        peaks = self.find_peaks(order=4)
        troughs = self.find_troughs(order=4)
        last_idx = len(self.data) - 1
        current_price = float(self.close[last_idx])

        # Forming H&S Top: left shoulder, head, right shoulder forming/complete
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                l_sh, head, r_sh = int(peaks[i]), int(peaks[i + 1]), int(peaks[i + 2])
                if last_idx - r_sh > 15:
                    continue
                # Head must be highest
                if not (self.highs[head] > self.highs[l_sh] and self.highs[head] > self.highs[r_sh]):
                    continue
                # Shoulders roughly equal
                if abs(self.highs[l_sh] - self.highs[r_sh]) / (abs(self.highs[l_sh]) + 1e-8) > 0.05:
                    continue
                neck = [t for t in troughs if l_sh < t < r_sh]
                if not neck:
                    continue
                neckline = float(np.mean(self.lows[neck]))
                height = float(self.highs[head] - neckline)
                if height <= 0:
                    continue

                # NOT yet broken neckline
                if current_price < neckline:
                    continue

                distance = current_price - neckline
                completion = 1.0 - min(1.0, distance / (height + 1e-8))

                sl = float(self.highs[r_sh] + 0.1 * height)
                tp = float(neckline - height)

                forming.append(FormingPattern(
                    name="Forming Head & Shoulders Top",
                    completion_pct=float(np.clip(completion, 0.0, 1.0)),
                    predicted_direction="bearish",
                    breakout_level=neckline,
                    stop_loss=sl,
                    target_price=tp,
                    pattern_height=height,
                    confidence=0.60 + 0.20 * completion,
                    volume_trend=self._vol_trend_score(l_sh, last_idx),
                    index_start=l_sh,
                    index_end=last_idx,
                    details={"neckline": neckline, "head_idx": head,
                             "left_shoulder": l_sh, "right_shoulder": r_sh},
                ))

        # Forming Inverse H&S
        if len(troughs) >= 3:
            for i in range(len(troughs) - 2):
                l_sh, head, r_sh = int(troughs[i]), int(troughs[i + 1]), int(troughs[i + 2])
                if last_idx - r_sh > 15:
                    continue
                if not (self.lows[head] < self.lows[l_sh] and self.lows[head] < self.lows[r_sh]):
                    continue
                if abs(self.lows[l_sh] - self.lows[r_sh]) / (abs(self.lows[l_sh]) + 1e-8) > 0.05:
                    continue
                neck = [p for p in peaks if l_sh < p < r_sh]
                if not neck:
                    continue
                neckline = float(np.mean(self.highs[neck]))
                height = float(neckline - self.lows[head])
                if height <= 0:
                    continue

                if current_price > neckline:
                    continue

                distance = neckline - current_price
                completion = 1.0 - min(1.0, distance / (height + 1e-8))

                sl = float(self.lows[r_sh] - 0.1 * height)
                tp = float(neckline + height)

                forming.append(FormingPattern(
                    name="Forming Inverse H&S",
                    completion_pct=float(np.clip(completion, 0.0, 1.0)),
                    predicted_direction="bullish",
                    breakout_level=neckline,
                    stop_loss=sl,
                    target_price=tp,
                    pattern_height=height,
                    confidence=0.60 + 0.20 * completion,
                    volume_trend=self._vol_trend_score(l_sh, last_idx),
                    index_start=l_sh,
                    index_end=last_idx,
                    details={"neckline": neckline, "head_idx": head,
                             "left_shoulder": l_sh, "right_shoulder": r_sh},
                ))

        return forming

    # -------------------------------------------------------------- #
    # Forming Triangle / Wedge (near apex)                             #
    # -------------------------------------------------------------- #

    def _forming_triangle_wedge(self) -> List[FormingPattern]:
        """Detect triangles/wedges that are compressing near apex but
        haven't broken out yet.  The closer to apex = higher completion."""
        forming: List[FormingPattern] = []
        if len(self.data) < 45:
            return forming

        peaks = self.find_peaks(order=3)
        troughs = self.find_troughs(order=3)
        end_idx = len(self.data) - 1
        current_price = float(self.close[end_idx])

        start_idx = max(0, end_idx - 60)
        rp = np.array([p for p in peaks if p >= start_idx])
        rt = np.array([t for t in troughs if t >= start_idx])
        if len(rp) < 2 or len(rt) < 2:
            return forming

        res_fit = self._linear_fit(rp, self.highs[rp])
        sup_fit = self._linear_fit(rt, self.lows[rt])
        if not res_fit or not sup_fit:
            return forming

        res_slope, res_intercept = res_fit
        sup_slope, sup_intercept = sup_fit
        res_level = self._line_value(res_slope, res_intercept, end_idx)
        sup_level = self._line_value(sup_slope, sup_intercept, end_idx)

        pattern_height = float(np.max(self.highs[start_idx: end_idx + 1]) - np.min(self.lows[start_idx: end_idx + 1]))
        current_width = res_level - sup_level
        if pattern_height <= 0 or current_width <= 0:
            return forming

        # Must be converging
        if current_width >= pattern_height * 0.8:
            return forming

        # NOT yet broken out (price inside the triangle)
        if current_price > res_level or current_price < sup_level:
            return forming

        # Completion = how squeezed the price is relative to the original range
        compression = 1.0 - (current_width / pattern_height)
        completion = float(np.clip(compression, 0.0, 1.0))

        # Pole measurement for TP
        pole_lookback = max(0, start_idx - 20)
        pole_high = float(np.max(self.highs[pole_lookback: start_idx + 1]))
        pole_low = float(np.min(self.lows[pole_lookback: start_idx + 1]))
        pole_length = pole_high - pole_low
        measured_move = pole_length if pole_length > pattern_height * 0.5 else pattern_height

        vol_score = self._vol_trend_score(start_idx, end_idx)

        # Determine triangle type and predicted direction
        # Ascending: flat resistance, rising support → bullish
        if abs(res_slope) < 1e-5 and sup_slope > 1e-5:
            name = "Forming Ascending Triangle"
            direction = "bullish"
            breakout_level = res_level
            sl = float(np.min(self.lows[rt[-2:]]))
            tp = breakout_level + measured_move
        # Descending: declining resistance, flat support → bearish
        elif res_slope < -1e-5 and abs(sup_slope) < 1e-5:
            name = "Forming Descending Triangle"
            direction = "bearish"
            breakout_level = sup_level
            sl = float(np.max(self.highs[rp[-2:]]))
            tp = breakout_level - measured_move
        # Symmetrical: both converging → predict based on prior trend
        elif res_slope < 0 and sup_slope > 0:
            prior_trend = float(self.close[start_idx] - self.close[max(0, start_idx - 20)])
            if prior_trend > 0:
                name = "Forming Symmetrical Triangle (Bullish)"
                direction = "bullish"
                breakout_level = res_level
                sl = float(np.min(self.lows[rt[-2:]]))
                tp = breakout_level + measured_move
            else:
                name = "Forming Symmetrical Triangle (Bearish)"
                direction = "bearish"
                breakout_level = sup_level
                sl = float(np.max(self.highs[rp[-2:]]))
                tp = breakout_level - measured_move
        # Rising Wedge: both rising, converging → bearish
        elif res_slope > 0 and sup_slope > 0 and sup_slope > res_slope:
            name = "Forming Rising Wedge"
            direction = "bearish"
            breakout_level = sup_level
            sl = float(np.max(self.highs[rp[-2:]]))
            tp = breakout_level - pattern_height
        # Falling Wedge: both falling, converging → bullish
        elif res_slope < 0 and sup_slope < 0 and res_slope < sup_slope:
            name = "Forming Falling Wedge"
            direction = "bullish"
            breakout_level = res_level
            sl = float(np.min(self.lows[rt[-2:]]))
            tp = breakout_level + pattern_height
        else:
            return forming

        forming.append(FormingPattern(
            name=name,
            completion_pct=completion,
            predicted_direction=direction,
            breakout_level=breakout_level,
            stop_loss=sl,
            target_price=tp,
            pattern_height=pattern_height,
            confidence=0.55 + 0.25 * completion,
            volume_trend=vol_score,
            index_start=start_idx,
            index_end=end_idx,
            details={"res_slope": res_slope, "sup_slope": sup_slope,
                     "res_level": res_level, "sup_level": sup_level,
                     "compression": compression, "measured_move": measured_move},
        ))

        return forming

    # -------------------------------------------------------------- #
    # Forming Flag / Pennant (consolidation after impulse)             #
    # -------------------------------------------------------------- #

    def _forming_flag_pennant(self) -> List[FormingPattern]:
        """Detect flag/pennant consolidation still in progress (no breakout yet)."""
        forming: List[FormingPattern] = []
        if len(self.data) < 40:
            return forming

        end_idx = len(self.data) - 1
        current_price = float(self.close[end_idx])
        start_idx = max(0, end_idx - 30)
        mid_idx = max(start_idx + 5, end_idx - 15)

        pole_move = self.close[mid_idx] - self.close[start_idx]
        if abs(pole_move) < 1e-8:
            return forming
        trend_dir = "bullish" if pole_move > 0 else "bearish"

        cons_highs = self.highs[mid_idx: end_idx + 1]
        cons_lows = self.lows[mid_idx: end_idx + 1]
        cons_high = float(np.max(cons_highs))
        cons_low = float(np.min(cons_lows))
        cons_range = cons_high - cons_low
        pole_range = float(np.max(self.highs[start_idx: mid_idx + 1]) - np.min(self.lows[start_idx: mid_idx + 1]))
        if pole_range <= 0 or cons_range > 0.6 * pole_range:
            return forming

        # NOT yet broken out
        if trend_dir == "bullish" and current_price > cons_high:
            return forming
        if trend_dir == "bearish" and current_price < cons_low:
            return forming

        # Consolidation duration relative to pole
        cons_bars = end_idx - mid_idx
        completion = float(np.clip(cons_bars / 12.0, 0.0, 1.0))

        vol_score = self._vol_trend_score(mid_idx, end_idx)

        if trend_dir == "bullish":
            breakout_level = cons_high
            sl = cons_low
            tp = breakout_level + pole_range
            forming.append(FormingPattern(
                name="Forming Bull Flag",
                completion_pct=completion,
                predicted_direction="bullish",
                breakout_level=breakout_level,
                stop_loss=sl,
                target_price=tp,
                pattern_height=pole_range,
                confidence=0.55 + 0.20 * completion,
                volume_trend=vol_score,
                index_start=start_idx,
                index_end=end_idx,
                details={"pole_range": pole_range, "cons_high": cons_high,
                         "cons_low": cons_low, "cons_bars": cons_bars},
            ))
        else:
            breakout_level = cons_low
            sl = cons_high
            tp = breakout_level - pole_range
            forming.append(FormingPattern(
                name="Forming Bear Flag",
                completion_pct=completion,
                predicted_direction="bearish",
                breakout_level=breakout_level,
                stop_loss=sl,
                target_price=tp,
                pattern_height=pole_range,
                confidence=0.55 + 0.20 * completion,
                volume_trend=vol_score,
                index_start=start_idx,
                index_end=end_idx,
                details={"pole_range": pole_range, "cons_high": cons_high,
                         "cons_low": cons_low, "cons_bars": cons_bars},
            ))

        return forming

    # -------------------------------------------------------------- #
    # Support / Resistance approach detector                           #
    # -------------------------------------------------------------- #

    def _forming_support_resistance_approach(self) -> List[FormingPattern]:
        """Detect price approaching a key S/R level built from recent
        peaks/troughs.  This is the simplest 'puzzle piece' — price
        approaching a level where it bounced before."""
        forming: List[FormingPattern] = []
        if len(self.data) < 30:
            return forming

        last_idx = len(self.data) - 1
        current_price = float(self.close[last_idx])
        peaks = self.find_peaks(order=5)
        troughs = self.find_troughs(order=5)

        # Build resistance levels from peaks
        for pk in peaks:
            pk = int(pk)
            if last_idx - pk < 5:
                continue  # too recent to be a level
            level = float(self.highs[pk])
            distance_pct = (level - current_price) / (current_price + 1e-8)
            # Price approaching resistance from below (within 0.3%)
            if 0.0 < distance_pct < 0.003:
                proximity = 1.0 - (distance_pct / 0.003)
                # Count touches at this level
                touches = sum(1 for p in peaks if abs(self.highs[int(p)] - level) / (level + 1e-8) < 0.002)
                if touches < 2:
                    continue
                height = level - float(np.min(self.lows[max(0, pk - 20): pk + 1]))
                if height <= 0:
                    continue

                forming.append(FormingPattern(
                    name="Approaching Resistance",
                    completion_pct=float(np.clip(proximity, 0.0, 1.0)),
                    predicted_direction="bearish",
                    breakout_level=level,
                    stop_loss=float(level + 0.15 * height),
                    target_price=float(current_price - height),
                    pattern_height=height,
                    confidence=0.50 + 0.10 * touches + 0.10 * proximity,
                    volume_trend=self._vol_trend_score(max(0, last_idx - 10), last_idx),
                    index_start=pk,
                    index_end=last_idx,
                    details={"level": level, "touches": touches, "type": "resistance"},
                ))

        # Build support levels from troughs
        for tr in troughs:
            tr = int(tr)
            if last_idx - tr < 5:
                continue
            level = float(self.lows[tr])
            distance_pct = (current_price - level) / (current_price + 1e-8)
            # Price approaching support from above (within 0.3%)
            if 0.0 < distance_pct < 0.003:
                proximity = 1.0 - (distance_pct / 0.003)
                touches = sum(1 for t in troughs if abs(self.lows[int(t)] - level) / (level + 1e-8) < 0.002)
                if touches < 2:
                    continue
                height = float(np.max(self.highs[max(0, tr - 20): tr + 1])) - level
                if height <= 0:
                    continue

                forming.append(FormingPattern(
                    name="Approaching Support",
                    completion_pct=float(np.clip(proximity, 0.0, 1.0)),
                    predicted_direction="bullish",
                    breakout_level=level,
                    stop_loss=float(level - 0.15 * height),
                    target_price=float(current_price + height),
                    pattern_height=height,
                    confidence=0.50 + 0.10 * touches + 0.10 * proximity,
                    volume_trend=self._vol_trend_score(max(0, last_idx - 10), last_idx),
                    index_start=tr,
                    index_end=last_idx,
                    details={"level": level, "touches": touches, "type": "support"},
                ))

        return forming

    # -------------------------------------------------------------- #
    # Volume trend helper                                              #
    # -------------------------------------------------------------- #

    def _vol_trend_score(self, start_idx: int, end_idx: int) -> float:
        """Score volume trend over a window.  Returns -1.0 (declining) to +1.0 (rising)."""
        if end_idx <= start_idx + 3:
            return 0.0
        mid = (start_idx + end_idx) // 2
        first_half = float(np.mean(self.vol[start_idx: mid + 1]))
        second_half = float(np.mean(self.vol[mid + 1: end_idx + 1]))
        if first_half < 1e-8:
            return 0.0
        ratio = (second_half - first_half) / (first_half + 1e-8)
        return float(np.clip(ratio, -1.0, 1.0))

    # -------------------------------------------------------------- #
    # Feature extraction for neural model                              #
    # -------------------------------------------------------------- #

    @staticmethod
    def forming_pattern_features(forming: List[FormingPattern]) -> np.ndarray:
        """Convert forming patterns into a fixed-size feature vector for the neural model.

        Returns 12 features encoding the strongest forming pattern state:
          [0] has_forming_pattern (0/1)
          [1] best_completion_pct (0-1)
          [2] best_confidence (0-1)
          [3] predicted_direction_sign (-1 bearish, 0 none, +1 bullish)
          [4] distance_to_breakout_pct (normalized)
          [5] pattern_rr_ratio (reward/risk, capped at 5)
          [6] volume_trend (-1 to +1)
          [7] is_reversal_pattern (0/1)
          [8] is_continuation_pattern (0/1)
          [9] is_sr_approach (0/1)
          [10] num_forming_patterns (0-5 normalized)
          [11] avg_completion_pct (0-1)
        """
        FEATURE_COUNT = 12
        feats = np.zeros(FEATURE_COUNT, dtype=np.float32)
        if not forming:
            return feats

        # Find best forming pattern by confidence * completion
        best = max(forming, key=lambda f: f.confidence * f.completion_pct)

        feats[0] = 1.0  # has forming pattern
        feats[1] = best.completion_pct
        feats[2] = best.confidence
        feats[3] = 1.0 if best.predicted_direction == "bullish" else -1.0
        # Distance to breakout as pct of pattern height
        if best.pattern_height > 0:
            feats[4] = float(np.clip(
                abs(best.breakout_level - (best.stop_loss + best.pattern_height / 2)) / best.pattern_height,
                0.0, 2.0
            )) / 2.0
        risk = abs(best.stop_loss - best.breakout_level) if best.stop_loss != best.breakout_level else 1e-8
        reward = abs(best.target_price - best.breakout_level)
        feats[5] = float(np.clip(reward / (risk + 1e-8), 0.0, 5.0)) / 5.0
        feats[6] = best.volume_trend

        name_lower = best.name.lower()
        reversal_names = {"double", "triple", "head", "shoulder", "wedge", "resistance", "support"}
        continuation_names = {"flag", "pennant", "triangle", "rectangle"}
        sr_names = {"approaching", "support", "resistance"}
        feats[7] = 1.0 if any(r in name_lower for r in reversal_names) else 0.0
        feats[8] = 1.0 if any(c in name_lower for c in continuation_names) else 0.0
        feats[9] = 1.0 if any(s in name_lower for s in sr_names) else 0.0
        feats[10] = float(np.clip(len(forming), 0, 5)) / 5.0
        feats[11] = float(np.mean([f.completion_pct for f in forming]))

        return feats


# Module-level constant for feature counting
FORMING_PATTERN_FEATURE_COUNT = 12
