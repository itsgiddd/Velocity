"""
FlipPredictor Integration — Bridges AI predictions into trading decisions.

Translates FlipPredictor outputs into actionable trade signals that
trading_app.py can use alongside the existing ZeroPoint scanner.

Decision matrix:
  1. flip_imminent + big move expected   -> PREDICTIVE ENTRY (before flip)
  2. ZP flip confirmed + predictor agrees -> FULL CONFIDENCE entry
  3. ZP flip confirmed + predictor says exhausted -> SKIP or reduce size
  4. trend_exhausted -> EXIT existing positions early
"""
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass
from flip_predictor_inference import FlipPrediction


@dataclass
class TradingDecision:
    """Actionable trading decision from FlipPredictor."""
    symbol: str
    action: str              # "PREDICTIVE_ENTRY", "FULL_CONFIDENCE", "REDUCED_SIZE", "SKIP", "EXIT_EARLY", "NO_ACTION"
    direction: str           # "BUY" or "SELL" or "" if no action
    confidence: float        # 0.0 - 1.0
    size_multiplier: float   # 1.0 = full size, 0.5 = half, etc.
    reason: str
    prediction: Optional[FlipPrediction] = None

    def summary(self) -> str:
        return (f"{self.symbol}: {self.action} {self.direction} "
                f"(conf={self.confidence:.0%}, size={self.size_multiplier:.0%}) "
                f"-- {self.reason}")


class FlipPredictorDecisionMaker:
    """Translates FlipPredictor outputs into trade decisions."""

    # Thresholds (tunable)
    IMMINENT_PROB_THRESHOLD = 0.45    # P(imminent) must exceed this for predictive entry
    APPROACHING_PROB_THRESHOLD = 0.55 # P(imminent+approaching) for awareness
    TREND_EXHAUSTED_THRESHOLD = 0.30  # below this = exit early
    MIN_MOVE_PIPS = 50                # minimum expected move for predictive entry
    STRONG_MOVE_PIPS = 150            # big move = aggressive entry

    def evaluate_signal(
        self,
        prediction: FlipPrediction,
        current_zp_direction: int,     # +1 (BUY) or -1 (SELL)
        has_zp_flip: bool = False,      # did ZP just flip this bar?
        zp_flip_direction: int = 0,     # +1 or -1 if flipped
    ) -> TradingDecision:
        """Evaluate a FlipPredictor output and produce a trading decision.

        Args:
            prediction:          FlipPrediction from inference engine
            current_zp_direction: current H4 ZP position (+1 or -1)
            has_zp_flip:         did H4 ZP just flip on this bar?
            zp_flip_direction:   direction of the flip (+1 or -1)
        """
        sym = prediction.symbol
        p_imminent = prediction.imminence_probs[0]
        p_approaching = prediction.imminence_probs[0] + prediction.imminence_probs[1]
        move_pips = abs(prediction.move_magnitude_pips)
        trend_cont = prediction.trend_continues_prob
        bars_to_flip = prediction.bars_to_flip

        # ─── Case 1: ZP flip just happened ────────────────────────────────────
        if has_zp_flip:
            flip_dir = "BUY" if zp_flip_direction == 1 else "SELL"

            # Predictor says trend will continue (agrees with flip)
            if trend_cont > 0.60 and move_pips > self.MIN_MOVE_PIPS:
                return TradingDecision(
                    symbol=sym,
                    action="FULL_CONFIDENCE",
                    direction=flip_dir,
                    confidence=min(0.95, 0.6 + trend_cont * 0.3),
                    size_multiplier=1.0,
                    reason=f"ZP flip + predictor agrees (cont={trend_cont:.0%}, move={move_pips:.0f}p)",
                    prediction=prediction,
                )

            # Predictor says another flip is imminent (disagrees)
            if p_imminent > self.IMMINENT_PROB_THRESHOLD:
                return TradingDecision(
                    symbol=sym,
                    action="SKIP",
                    direction="",
                    confidence=0.0,
                    size_multiplier=0.0,
                    reason=f"ZP flip but predictor sees reversal imminent (P={p_imminent:.0%})",
                    prediction=prediction,
                )

            # Weak move expected
            if move_pips < self.MIN_MOVE_PIPS:
                return TradingDecision(
                    symbol=sym,
                    action="REDUCED_SIZE",
                    direction=flip_dir,
                    confidence=0.4,
                    size_multiplier=0.5,
                    reason=f"ZP flip but small expected move ({move_pips:.0f}p < {self.MIN_MOVE_PIPS}p)",
                    prediction=prediction,
                )

            # Default: normal confidence
            return TradingDecision(
                symbol=sym,
                action="FULL_CONFIDENCE",
                direction=flip_dir,
                confidence=0.7,
                size_multiplier=0.8,
                reason=f"ZP flip confirmed, moderate conviction",
                prediction=prediction,
            )

        # ─── Case 2: No ZP flip — check for predictive entry ─────────────────
        # Flip is imminent + big move expected -> enter early
        if (p_imminent > self.IMMINENT_PROB_THRESHOLD and
                move_pips > self.MIN_MOVE_PIPS and
                bars_to_flip <= 3):
            # Direction: opposite of current ZP (we're predicting a FLIP)
            pred_dir = "BUY" if current_zp_direction == -1 else "SELL"

            size_mult = 0.6 if move_pips > self.STRONG_MOVE_PIPS else 0.4
            conf = min(0.8, p_imminent * 0.8 + (move_pips / 500) * 0.2)

            return TradingDecision(
                symbol=sym,
                action="PREDICTIVE_ENTRY",
                direction=pred_dir,
                confidence=conf,
                size_multiplier=size_mult,
                reason=f"Flip imminent in ~{bars_to_flip:.0f}b (P={p_imminent:.0%}, move={move_pips:.0f}p)",
                prediction=prediction,
            )

        # ─── Case 3: Trend exhausted -> exit early ────────────────────────────
        if trend_cont < self.TREND_EXHAUSTED_THRESHOLD:
            return TradingDecision(
                symbol=sym,
                action="EXIT_EARLY",
                direction="",
                confidence=1.0 - trend_cont,
                size_multiplier=0.0,
                reason=f"Trend exhausted (cont={trend_cont:.0%}), consider closing positions",
                prediction=prediction,
            )

        # ─── Case 4: Nothing actionable ───────────────────────────────────────
        return TradingDecision(
            symbol=sym,
            action="NO_ACTION",
            direction="",
            confidence=0.0,
            size_multiplier=0.0,
            reason=f"No signal (flip in ~{bars_to_flip:.0f}b, cont={trend_cont:.0%})",
            prediction=prediction,
        )

    def evaluate_all(
        self,
        predictions: Dict[str, FlipPrediction],
        zp_states: Dict[str, dict],    # {symbol: {"direction": +/-1, "has_flip": bool}}
    ) -> Dict[str, TradingDecision]:
        """Evaluate all predictions and return actionable decisions."""
        decisions = {}
        for sym, pred in predictions.items():
            zp = zp_states.get(sym, {})
            decision = self.evaluate_signal(
                prediction=pred,
                current_zp_direction=zp.get("direction", 0),
                has_zp_flip=zp.get("has_flip", False),
                zp_flip_direction=zp.get("flip_direction", 0),
            )
            decisions[sym] = decision
        return decisions

    def rank_decisions(self, decisions: Dict[str, TradingDecision], max_trades: int = 2) -> list:
        """Rank actionable decisions by quality, return top N."""
        actionable = [d for d in decisions.values()
                      if d.action in ("FULL_CONFIDENCE", "PREDICTIVE_ENTRY", "REDUCED_SIZE")]

        # Score: confidence * size_multiplier * (1 + bonus for FULL_CONFIDENCE)
        def score(d):
            base = d.confidence * d.size_multiplier
            if d.action == "FULL_CONFIDENCE":
                base *= 1.5
            elif d.action == "PREDICTIVE_ENTRY":
                base *= 1.2
            return base

        actionable.sort(key=score, reverse=True)
        return actionable[:max_trades]
