from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


@dataclass
class Pattern:
    name: str
    index_start: int
    index_end: int
    confidence: float
    direction: str  # bullish / bearish / neutral
    details: Dict
    push_count: int = 1
    volume_score: float = 0.5


class PatternRecognizer:
    """
    Pattern detector used by rule-based decision flow.
    """

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

    def find_peaks(self, order: int = 5) -> np.ndarray:
        return argrelextrema(self.highs, np.greater, order=order)[0]

    def find_troughs(self, order: int = 5) -> np.ndarray:
        return argrelextrema(self.lows, np.less, order=order)[0]

    def get_slope(self, idx1: int, idx2: int, val1: float, val2: float) -> float:
        if idx2 == idx1:
            return 0.0
        return (val2 - val1) / (idx2 - idx1)

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

        window = self.vol[start_idx : end_idx + 1]
        baseline_start = max(0, start_idx - max(20, len(window)))
        baseline = self.vol[baseline_start:start_idx] if start_idx > baseline_start else self.vol[:start_idx]
        if len(baseline) == 0:
            baseline = self.vol[: max(5, len(window))]

        mean_window = float(np.mean(window))
        mean_base = float(np.mean(baseline)) if len(baseline) > 0 else mean_window
        ratio = mean_window / (mean_base + 1e-8)
        # Normalize around 0.5 baseline.
        return float(max(0.0, min(1.0, 0.5 + (ratio - 1.0) * 0.5)))

    def _estimate_push_count(self, start_idx: int, end_idx: int, direction: str) -> int:
        if end_idx - start_idx < 4:
            return 1
        closes = self.close[start_idx : end_idx + 1]
        diffs = np.diff(closes)
        sign = 1 if direction == "bullish" else -1
        aligned = (diffs * sign) > 0
        # Count directional runs as impulse pushes.
        pushes = 0
        in_run = False
        for v in aligned:
            if v and not in_run:
                pushes += 1
                in_run = True
            elif not v:
                in_run = False
        return int(max(1, min(pushes, 5)))

    def detect_all(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        patterns.extend(self.detect_double_top_bottom())
        patterns.extend(self.detect_head_and_shoulders())
        patterns.extend(self.detect_triangles_and_wedges())
        patterns.extend(self.detect_flags_pennants_rectangles())
        patterns.extend(self.detect_rounding_and_cup())
        patterns.extend(self.detect_diamonds())
        return patterns

    def detect_double_top_bottom(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 25:
            return patterns

        peaks = self.find_peaks(order=4)
        troughs = self.find_troughs(order=4)
        last_idx = len(self.data) - 1

        for i in range(len(peaks) - 1):
            p1, p2 = int(peaks[i]), int(peaks[i + 1])
            if p2 - p1 < 3:
                continue
            price_diff = abs(self.highs[p1] - self.highs[p2]) / (abs(self.highs[p1]) + 1e-8)
            if price_diff > 0.01:
                continue
            neck_troughs = [t for t in troughs if p1 < t < p2]
            if not neck_troughs:
                continue
            neckline = float(self.lows[int(neck_troughs[0])])
            if self.close[last_idx] >= neckline:
                continue
            height = max(self.highs[p1], self.highs[p2]) - neckline
            vol_score = self._calc_volume_score(p1, last_idx)
            patterns.append(
                Pattern(
                    name="Double Top",
                    index_start=p1,
                    index_end=p2,
                    confidence=0.82,
                    direction="bearish",
                    details={
                        "neckline": neckline,
                        "height": float(height),
                        "stop_loss": float(max(self.highs[p1], self.highs[p2]) + 0.1 * height),
                    },
                    push_count=self._estimate_push_count(p1, last_idx, "bearish"),
                    volume_score=vol_score,
                )
            )

        for i in range(len(troughs) - 1):
            t1, t2 = int(troughs[i]), int(troughs[i + 1])
            if t2 - t1 < 3:
                continue
            price_diff = abs(self.lows[t1] - self.lows[t2]) / (abs(self.lows[t1]) + 1e-8)
            if price_diff > 0.01:
                continue
            neck_peaks = [p for p in peaks if t1 < p < t2]
            if not neck_peaks:
                continue
            neckline = float(self.highs[int(neck_peaks[0])])
            if self.close[last_idx] <= neckline:
                continue
            height = neckline - min(self.lows[t1], self.lows[t2])
            vol_score = self._calc_volume_score(t1, last_idx)
            patterns.append(
                Pattern(
                    name="Double Bottom",
                    index_start=t1,
                    index_end=t2,
                    confidence=0.82,
                    direction="bullish",
                    details={
                        "neckline": neckline,
                        "height": float(height),
                        "stop_loss": float(min(self.lows[t1], self.lows[t2]) - 0.1 * height),
                    },
                    push_count=self._estimate_push_count(t1, last_idx, "bullish"),
                    volume_score=vol_score,
                )
            )
        return patterns

    def detect_head_and_shoulders(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 35:
            return patterns
        peaks = self.find_peaks(order=4)
        troughs = self.find_troughs(order=4)
        last_idx = len(self.data) - 1

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
                if self.close[last_idx] >= neckline:
                    continue
                height = self.highs[head] - neckline
                patterns.append(
                    Pattern(
                        "Head and Shoulders (Top)",
                        l_sh,
                        r_sh,
                        0.80,
                        "bearish",
                        {"neckline": neckline, "height": float(height), "stop_loss": float(self.highs[r_sh] + 0.1 * height)},
                        self._estimate_push_count(l_sh, last_idx, "bearish"),
                        self._calc_volume_score(l_sh, last_idx),
                    )
                )

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
                if self.close[last_idx] <= neckline:
                    continue
                height = neckline - self.lows[head]
                patterns.append(
                    Pattern(
                        "Head and Shoulders (Bottom)",
                        l_sh,
                        r_sh,
                        0.80,
                        "bullish",
                        {"neckline": neckline, "height": float(height), "stop_loss": float(self.lows[r_sh] - 0.1 * height)},
                        self._estimate_push_count(l_sh, last_idx, "bullish"),
                        self._calc_volume_score(l_sh, last_idx),
                    )
                )
        return patterns

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
        height = float(np.max(self.highs[start_idx : end_idx + 1]) - np.min(self.lows[start_idx : end_idx + 1]))
        if height <= 0:
            return patterns

        vol_score = self._calc_volume_score(start_idx, end_idx)
        converging = (res_level - sup_level) < (height * 0.7)
        if not converging:
            return patterns

        if abs(res_slope) < 1e-5 and sup_slope > 1e-5 and self.close[end_idx] > res_level:
            patterns.append(
                Pattern(
                    "Ascending Triangle",
                    start_idx,
                    end_idx,
                    0.78,
                    "bullish",
                    {"height": height, "stop_loss": float(np.min(self.lows[rt[-2:]]))},
                    self._estimate_push_count(start_idx, end_idx, "bullish"),
                    vol_score,
                )
            )
        elif res_slope < -1e-5 and abs(sup_slope) < 1e-5 and self.close[end_idx] < sup_level:
            patterns.append(
                Pattern(
                    "Descending Triangle",
                    start_idx,
                    end_idx,
                    0.78,
                    "bearish",
                    {"height": height, "stop_loss": float(np.max(self.highs[rp[-2:]]))},
                    self._estimate_push_count(start_idx, end_idx, "bearish"),
                    vol_score,
                )
            )
        elif res_slope < 0 and sup_slope > 0:
            if self.close[end_idx] > res_level:
                patterns.append(
                    Pattern(
                        "Symmetrical Triangle",
                        start_idx,
                        end_idx,
                        0.74,
                        "bullish",
                        {"height": height, "stop_loss": float(np.min(self.lows[rt[-2:]]))},
                        self._estimate_push_count(start_idx, end_idx, "bullish"),
                        vol_score,
                    )
                )
            elif self.close[end_idx] < sup_level:
                patterns.append(
                    Pattern(
                        "Symmetrical Triangle",
                        start_idx,
                        end_idx,
                        0.74,
                        "bearish",
                        {"height": height, "stop_loss": float(np.max(self.highs[rp[-2:]]))},
                        self._estimate_push_count(start_idx, end_idx, "bearish"),
                        vol_score,
                    )
                )
        elif res_slope > 0 and sup_slope > 0 and sup_slope > res_slope and self.close[end_idx] < sup_level:
            patterns.append(
                Pattern(
                    "Rising Wedge",
                    start_idx,
                    end_idx,
                    0.80,
                    "bearish",
                    {"height": height, "stop_loss": float(np.max(self.highs[rp[-2:]]))},
                    self._estimate_push_count(start_idx, end_idx, "bearish"),
                    vol_score,
                )
            )
        elif res_slope < 0 and sup_slope < 0 and res_slope < sup_slope and self.close[end_idx] > res_level:
            patterns.append(
                Pattern(
                    "Falling Wedge",
                    start_idx,
                    end_idx,
                    0.80,
                    "bullish",
                    {"height": height, "stop_loss": float(np.min(self.lows[rt[-2:]]))},
                    self._estimate_push_count(start_idx, end_idx, "bullish"),
                    vol_score,
                )
            )

        return patterns

    def detect_flags_pennants_rectangles(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 40:
            return patterns

        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - 30)
        mid_idx = max(start_idx + 5, end_idx - 15)

        pole_move = self.close[mid_idx] - self.close[start_idx]
        trend_dir = "bullish" if pole_move >= 0 else "bearish"
        cons_high = float(np.max(self.highs[mid_idx:end_idx + 1]))
        cons_low = float(np.min(self.lows[mid_idx:end_idx + 1]))
        cons_range = cons_high - cons_low
        pole_range = float(np.max(self.highs[start_idx:mid_idx + 1]) - np.min(self.lows[start_idx:mid_idx + 1]))
        if pole_range <= 0 or cons_range > 0.6 * pole_range:
            return patterns

        vol_score = self._calc_volume_score(start_idx, end_idx)
        details = {"height": pole_range}
        if trend_dir == "bullish" and self.close[end_idx] > cons_high:
            details["stop_loss"] = cons_low
            patterns.append(Pattern("Bull Flag", start_idx, end_idx, 0.72, "bullish", details, 2, vol_score))
        elif trend_dir == "bearish" and self.close[end_idx] < cons_low:
            details["stop_loss"] = cons_high
            patterns.append(Pattern("Bear Flag", start_idx, end_idx, 0.72, "bearish", details, 2, vol_score))
        return patterns

    def detect_rounding_and_cup(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 60:
            return patterns

        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - 50)
        idx = np.arange(start_idx, end_idx + 1)
        prices = self.close[start_idx : end_idx + 1]
        if len(prices) < 10:
            return patterns

        curvature = float(np.polyfit(idx, prices, 2)[0])
        height = float(np.max(self.highs[start_idx : end_idx + 1]) - np.min(self.lows[start_idx : end_idx + 1]))
        vol_score = self._calc_volume_score(start_idx, end_idx)
        if height <= 0:
            return patterns

        if curvature > 0:
            neckline = float(np.max(self.highs[start_idx : end_idx + 1]))
            if self.close[end_idx] > neckline:
                patterns.append(
                    Pattern(
                        "Rounding Bottom",
                        start_idx,
                        end_idx,
                        0.68,
                        "bullish",
                        {"height": height, "stop_loss": float(np.min(self.lows[start_idx : end_idx + 1]))},
                        3,
                        vol_score,
                    )
                )
        elif curvature < 0:
            neckline = float(np.min(self.lows[start_idx : end_idx + 1]))
            if self.close[end_idx] < neckline:
                patterns.append(
                    Pattern(
                        "Rounding Top",
                        start_idx,
                        end_idx,
                        0.68,
                        "bearish",
                        {"height": height, "stop_loss": float(np.max(self.highs[start_idx : end_idx + 1]))},
                        3,
                        vol_score,
                    )
                )
        return patterns

    def detect_diamonds(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        if len(self.data) < 60:
            return patterns

        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - 40)
        mid = start_idx + ((end_idx - start_idx) // 2)

        first_range = float(np.max(self.highs[start_idx:mid + 1]) - np.min(self.lows[start_idx:mid + 1]))
        second_range = float(np.max(self.highs[mid:end_idx + 1]) - np.min(self.lows[mid:end_idx + 1]))
        if first_range <= 0 or second_range <= 0:
            return patterns
        if first_range < second_range:
            return patterns

        upper = float(max(np.max(self.highs[start_idx:mid + 1]), np.max(self.highs[mid:end_idx + 1])))
        lower = float(min(np.min(self.lows[start_idx:mid + 1]), np.min(self.lows[mid:end_idx + 1])))
        height = upper - lower
        if height <= 0:
            return patterns

        vol_score = self._calc_volume_score(start_idx, end_idx)
        if self.close[end_idx] > upper:
            patterns.append(Pattern("Bullish Diamond", start_idx, end_idx, 0.65, "bullish", {"height": height, "stop_loss": lower}, 3, vol_score))
        elif self.close[end_idx] < lower:
            patterns.append(Pattern("Bearish Diamond", start_idx, end_idx, 0.65, "bearish", {"height": height, "stop_loss": upper}, 3, vol_score))
        return patterns
