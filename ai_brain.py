from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from adaptive_risk import AdaptiveRiskManager
from daily_planner import DailyPlanner
from market_context import MarketContextAnalyzer
from pattern_recognition import Pattern, PatternRecognizer
from trade_validator import TradeDecision, TradeValidator
from trading_memory import TradingMemory

try:
    from push_structure_analyzer import SymbolPushProfile
    _HAS_PUSH_PROFILE = True
except ImportError:
    _HAS_PUSH_PROFILE = False


class AIBrain:
    """
    Rule-driven trading brain with daily-bias filtering and confluence validation.
    """

    def __init__(self, push_profiles: Optional[Dict[str, "SymbolPushProfile"]] = None):
        self.market_analyzer = MarketContextAnalyzer()
        self.trade_validator = TradeValidator()
        self.risk_manager = AdaptiveRiskManager()
        self.memory = TradingMemory()
        self.planner = DailyPlanner()
        self.reasoning_engine = ReasoningEngine()
        self.required_columns = {"open", "high", "low", "close"}
        self.push_profiles: Dict = push_profiles or {}

    def set_daily_plan(self, plan: dict):
        self.planner.set_plan(plan)

    def _sanitize_data(self, data: pd.DataFrame, name: str) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError(f"{name} data is empty")
        missing = self.required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} data missing columns: {', '.join(sorted(missing))}")
        cleaned = data.dropna(subset=list(self.required_columns)).copy()
        if cleaned.empty:
            raise ValueError(f"{name} data has no usable rows after cleanup")
        if "time" in cleaned.columns:
            cleaned = cleaned.sort_values("time")
        return cleaned

    def _validate_symbol_info(self, symbol_info) -> None:
        required_attrs = ("point", "volume_step", "volume_min", "volume_max", "trade_tick_value")
        missing = [attr for attr in required_attrs if not hasattr(symbol_info, attr)]
        if missing:
            raise ValueError(f"symbol_info missing fields: {', '.join(missing)}")

    def _fallback_stop_loss(self, pattern: Pattern, data_h1: pd.DataFrame) -> float:
        lookback = data_h1.tail(20)
        if pattern.direction == "bullish":
            return float(lookback["low"].min())
        return float(lookback["high"].max())

    def _fallback_target_distance(self, data_h1: pd.DataFrame, risk: float) -> float:
        lookback = data_h1.tail(20)
        recent_range = float(lookback["high"].max() - lookback["low"].min())
        return max(recent_range, risk * 2.0)

    def _collect_patterns(
        self,
        data_h1: pd.DataFrame,
        data_h4: pd.DataFrame,
        data_d1: pd.DataFrame,
    ) -> List[Tuple[str, Pattern]]:
        all_patterns: List[Tuple[str, Pattern]] = []
        for tf_name, df in [("H1", data_h1), ("H4", data_h4), ("D1", data_d1)]:
            recognizer = PatternRecognizer(df)
            patterns = recognizer.detect_all()
            fresh = [p for p in patterns if p.index_end >= len(df) - 3]
            all_patterns.extend((tf_name, p) for p in fresh)
        return all_patterns

    def _pick_best_pattern(
        self,
        patterns: List[Tuple[str, Pattern]],
        market_state: Dict,
        symbol: str = "",
    ) -> Tuple[Optional[str], Optional[Pattern], Optional[TradeDecision]]:
        best_tf = None
        best_pattern = None
        best_decision = None

        # Look up learned push profile for this symbol.
        push_profile = self.push_profiles.get(symbol) if self.push_profiles else None

        for tf, pattern in patterns:
            features = {"vol_anomaly": 1.0 + float(getattr(pattern, "volume_score", 0.5))}
            if push_profile is not None:
                features["push_profile"] = push_profile
            decision = self.trade_validator.validate(pattern, market_state, features)
            if not decision.should_trade:
                continue

            if best_decision is None:
                best_tf, best_pattern, best_decision = tf, pattern, decision
                continue

            cur_key = (decision.confluence_score, decision.confidence)
            best_key = (best_decision.confluence_score, best_decision.confidence)
            if cur_key > best_key:
                best_tf, best_pattern, best_decision = tf, pattern, decision

        return best_tf, best_pattern, best_decision

    def think(
        self,
        symbol: str,
        data_h1: pd.DataFrame,
        data_h4: pd.DataFrame,
        data_d1: pd.DataFrame,
        account_info,
        symbol_info,
    ) -> dict:
        try:
            data_h1 = self._sanitize_data(data_h1, "H1")
            data_h4 = self._sanitize_data(data_h4, "H4")
            data_d1 = self._sanitize_data(data_d1, "D1")
            self._validate_symbol_info(symbol_info)
        except ValueError as exc:
            return {"decision": "REJECT", "reason": str(exc)}

        daily_bias = self.planner.get_bias(symbol)
        if daily_bias == "NEUTRAL":
            return {"decision": "REJECT", "reason": "Daily Bias: NEUTRAL - No trading"}

        if not self.memory.can_trade(symbol):
            return {"decision": "REJECT", "reason": "Memory Block (Loss Streak or Cooldown)"}

        if len(data_h1) < 60 or len(data_h4) < 50 or len(data_d1) < 30:
            return {"decision": "WAIT", "reason": "Insufficient market history"}

        market_state = self.market_analyzer.get_market_state(symbol, data_h1, data_h4, data_d1)
        patterns = self._collect_patterns(data_h1, data_h4, data_d1)
        if not patterns:
            return {"decision": "WAIT", "reason": "No fresh patterns"}

        filtered: List[Tuple[str, Pattern]] = []
        for tf, pattern in patterns:
            if daily_bias == "LONG" and pattern.direction == "bullish":
                filtered.append((tf, pattern))
            if daily_bias == "SHORT" and pattern.direction == "bearish":
                filtered.append((tf, pattern))

        if not filtered:
            return {"decision": "REJECT", "reason": f"Patterns don't match {daily_bias} bias"}

        best_tf, best_pattern, best_decision = self._pick_best_pattern(filtered, market_state, symbol)
        if not best_pattern or not best_decision:
            return {"decision": "REJECT", "reason": "All candidate patterns failed validation"}

        price = float(data_h1["close"].iloc[-1])
        sl = float(best_pattern.details.get("stop_loss", self._fallback_stop_loss(best_pattern, data_h1)))
        risk = abs(price - sl)
        if risk <= 0:
            return {"decision": "REJECT", "reason": "Zero risk distance"}

        target_distance = float(best_pattern.details.get("height", 0.0))
        if target_distance <= 0:
            target_distance = self._fallback_target_distance(data_h1, risk)
        if target_distance <= 0:
            return {"decision": "REJECT", "reason": "No valid target distance"}

        if best_pattern.direction == "bullish" and sl >= price:
            return {"decision": "REJECT", "reason": "Invalid SL for bullish setup"}
        if best_pattern.direction == "bearish" and sl <= price:
            return {"decision": "REJECT", "reason": "Invalid SL for bearish setup"}

        tp = price + target_distance if best_pattern.direction == "bullish" else price - target_distance
        reward = abs(tp - price)
        rr = reward / risk if risk > 0 else 0.0
        if rr < 2.0:
            return {"decision": "REJECT", "reason": f"RR below 1:2 ({rr:.2f})"}

        lot = self.risk_manager.calculate_lot_size(
            symbol,
            price,
            sl,
            best_decision.confidence if best_decision.confidence > 0 else best_decision.confluence_score,
            account_info,
            symbol_info,
        )
        if lot <= 0:
            return {"decision": "REJECT", "reason": "Risk Calc = 0 Lot"}

        reasoning_notes = self.reasoning_engine.build_reasoning(
            best_pattern,
            market_state,
            rr,
            best_decision.rationale,
            price,
            sl,
            tp,
        )

        return {
            "decision": "TRADE",
            "pattern": best_pattern,
            "lot": lot,
            "sl": sl,
            "tp": tp,
            "reason": f"[{best_tf}] Score {best_decision.confluence_score} | RR {rr:.2f} | {daily_bias}",
            "reasoning": reasoning_notes,
            "confidence": best_decision.confidence,
            "market_state": market_state,
        }

    def log_result(self, symbol, result, profit):
        self.memory.close_trade(symbol, profit)


class ReasoningEngine:
    def build_reasoning(
        self,
        pattern: Pattern,
        market_state: dict,
        rr: float,
        validator_notes: list,
        price: float,
        sl: float,
        tp: float,
    ) -> list:
        notes = list(validator_notes or [])
        notes.append(f"Pattern: {pattern.name} ({pattern.direction})")
        notes.append(f"Pattern height: {pattern.details.get('height', 0):.5f}")
        notes.append(f"SL distance: {abs(price - sl):.5f}")
        notes.append(f"TP distance: {abs(tp - price):.5f}")
        notes.append(f"RR check: {rr:.2f} >= 2.0")
        notes.append(f"Session score: {market_state.get('session', 0):.2f}")
        notes.append(f"Strength score: {market_state.get('strength', 0):.2f}")
        return notes
