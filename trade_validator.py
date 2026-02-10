from dataclasses import dataclass, field
from typing import List, Optional

try:
    from push_structure_analyzer import SymbolPushProfile
    _HAS_PUSH_PROFILE = True
except ImportError:
    _HAS_PUSH_PROFILE = False


@dataclass
class TradeDecision:
    should_trade: bool
    rejection_reason: str = ""
    confluence_score: int = 0
    confidence: float = 0.0
    rationale: List[str] = field(default_factory=list)


class TradeValidator:
    """
    Performs confluence checks before execution.
    Uses learned per-symbol push profiles when available, falls back to
    hardcoded rules otherwise.
    """

    def _normalize_trend(self, raw_trend) -> int:
        if isinstance(raw_trend, str):
            t = raw_trend.lower()
            if t in {"bullish", "up", "long"}:
                return 1
            if t in {"bearish", "down", "short"}:
                return -1
            return 0
        if isinstance(raw_trend, (int, float)):
            if raw_trend > 0:
                return 1
            if raw_trend < 0:
                return -1
        return 0

    def validate(self, pattern, market_state: dict, features: dict) -> TradeDecision:
        reasons: List[str] = []
        score = 0

        direction = getattr(pattern, "direction", "neutral")
        ai_conf = float(getattr(pattern, "confidence", 0.0))

        global_trend = self._normalize_trend(market_state.get("global_trend", 0))
        if global_trend == 1 and direction == "bearish":
            return TradeDecision(False, "Against higher timeframe uptrend", rationale=["Rejected: trend conflict"])
        if global_trend == -1 and direction == "bullish":
            return TradeDecision(False, "Against higher timeframe downtrend", rationale=["Rejected: trend conflict"])

        if global_trend == 1 and direction == "bullish":
            score += 2
            reasons.append("Aligned with higher timeframe uptrend (+2)")
        elif global_trend == -1 and direction == "bearish":
            score += 2
            reasons.append("Aligned with higher timeframe downtrend (+2)")
        else:
            reasons.append("No higher timeframe trend boost (+0)")

        # Push structure validation — use learned profile when available.
        push_count = int(max(getattr(pattern, "push_count", 1), 0))
        push_profile = features.get("push_profile") if _HAS_PUSH_PROFILE else None

        if push_count < 1:
            return TradeDecision(False, "Push count below 1", rationale=["Rejected: no impulse structure"])

        if push_profile is not None and isinstance(push_profile, SymbolPushProfile):
            # Data-learned exhaustion thresholds.
            exhaustion_threshold = push_profile.exhaustion_push_count
            reversal_prob = push_profile.reversal_prob_by_push.get(push_count, 0.0)

            if reversal_prob > 0.65:
                return TradeDecision(
                    False,
                    f"Push exhaustion (P(reversal)={reversal_prob:.0%} at push {push_count})",
                    rationale=[f"Rejected: learned reversal prob {reversal_prob:.0%} at push {push_count}"],
                )
            if push_count >= exhaustion_threshold:
                return TradeDecision(
                    False,
                    f"Push exhaustion (>={exhaustion_threshold} for {push_profile.symbol})",
                    rationale=[f"Rejected: learned exhaustion at {exhaustion_threshold} pushes"],
                )

            # Scoring: pushes in [2, exhaustion-1] are the trading sweet spot.
            if push_count >= 2:
                score += 1
            if push_count == exhaustion_threshold - 1:
                # Right before exhaustion = peak momentum.
                score += 2
            elif push_count >= 3:
                score += 1
            reasons.append(
                f"Push count {push_count} (exhaust@{exhaustion_threshold}, "
                f"P(rev)={reversal_prob:.0%})"
            )
        else:
            # Fallback: hardcoded rules (backward compatible).
            if push_count >= 4:
                return TradeDecision(False, "Push exhaustion (>=4)", rationale=["Rejected: exhausted structure"])
            if push_count >= 2:
                score += 1
            if push_count >= 3:
                score += 1
            reasons.append(f"Push count {push_count} (+{1 if push_count >= 2 else 0}{'+1' if push_count >= 3 else ''})")

        vol_anomaly = float(features.get("vol_anomaly", 1.0))
        volume_score = float(getattr(pattern, "volume_score", 0.5))
        if vol_anomaly > 1.5 or volume_score > 0.6:
            score += 1
            reasons.append("Volume confirms move (+1)")
        else:
            reasons.append("Weak volume confirmation (+0)")

        if ai_conf > 0.6:
            score += 1
            reasons.append("Pattern confidence > 0.6 (+1)")
        else:
            reasons.append("Pattern confidence <= 0.6 (+0)")

        session_score = float(market_state.get("session", 0.0))
        if session_score > 0.6:
            score += 1
            reasons.append("Session quality high (+1)")
        else:
            reasons.append("Session quality low (+0)")

        strength_score = float(market_state.get("strength", 0.0))
        if strength_score > 1.0:
            strength_score = strength_score / 100.0
        if strength_score > 0.4:
            score += 1
            reasons.append("Market strength supportive (+1)")
        else:
            reasons.append("Market strength weak (+0)")

        if direction == "neutral":
            reasons.append("Rejected: neutral direction")
            return TradeDecision(False, "Neutral pattern", score, ai_conf, reasons)

        if score >= 5:
            reasons.append("Approved: confluence threshold met")
            return TradeDecision(True, "High confluence", score, ai_conf, reasons)

        reasons.append("Rejected: confluence below threshold")
        return TradeDecision(False, f"Low confluence ({score}/8)", score, ai_conf, reasons)
