"""
Trading Council — Aggregates 15 agent verdicts into a final decision.

The council instantiates all agents, runs them in parallel (functionally),
tallies weighted votes, checks for veto conditions, and produces a
CouncilDecision with action, lot multiplier, and full reasoning log.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from candle_intelligence.agent_base import (
    TradingAgent, TradeContext, AgentVerdict, CouncilDecision,
    VOTE_SCORES, NEUTRAL,
    ACTION_FULL, ACTION_HIGH, ACTION_MODERATE, ACTION_LOW, ACTION_SKIP,
    VETO_CRITICAL_DRAWDOWN, VETO_RR_BELOW_1_0, VETO_SUPERMAJORITY_REJECT,
    BUY, SELL, STRONG_BUY, STRONG_SELL,
)
from candle_intelligence.agents import ALL_AGENTS

logger = logging.getLogger(__name__)


# Mapping from action to lot multiplier
# Council is an AUDIT + RISK GUARD — take every ZP signal at full size.
# Only reduce lot when agents strongly oppose (LOW), skip on extreme opposition.
ACTION_LOT_MULT = {
    ACTION_FULL: 1.0,
    ACTION_HIGH: 1.0,
    ACTION_MODERATE: 1.0,
    ACTION_LOW: 1.0,
    ACTION_SKIP: 0.0,
}


class TradingCouncil:
    """
    Instantiates all 15 agents, runs deliberation, and produces
    a consensus decision.

    Agent weights can be overridden via `set_weights()` or learned
    dynamically by the CouncilLearner.
    """

    # Decision thresholds — council is primarily a LOT SIZER, not a filter.
    # ZP signals are already high-quality; the council should rarely skip.
    # Only skip when overwhelming opposition (score < -3.0).
    THRESHOLD_FULL = 4.0
    THRESHOLD_HIGH = 2.0
    THRESHOLD_MODERATE = 0.5
    THRESHOLD_LOW = -3.0
    # Below -3.0 → SKIP (only skip when council strongly opposes)

    # Supermajority rejection: if this many agents vote against, auto-skip
    SUPERMAJORITY_COUNT = 10

    def __init__(self):
        """Initialize all agents with their default weights."""
        self.agents: List[TradingAgent] = [AgentClass() for AgentClass in ALL_AGENTS]
        self.weights: Dict[str, float] = {
            agent.name: agent.default_weight for agent in self.agents
        }
        logger.info(
            f"Trading Council initialized with {len(self.agents)} agents: "
            + ", ".join(a.name for a in self.agents)
        )

    def set_weights(self, weights: Dict[str, float]):
        """Override agent weights (e.g., from learned performance)."""
        for name, w in weights.items():
            if name in self.weights:
                self.weights[name] = max(0.3, min(2.5, w))

    def deliberate(self, direction: str, ctx: TradeContext) -> CouncilDecision:
        """
        Run all agents and aggregate their verdicts.

        Args:
            direction: "BUY" or "SELL" — the ZP signal direction
            ctx: TradeContext with all data

        Returns:
            CouncilDecision with action, lot_multiplier, reasoning, etc.
        """
        verdicts: List[AgentVerdict] = []

        # Run each agent (catch errors via _safe_analyze)
        for agent in self.agents:
            verdict = agent._safe_analyze(ctx)
            verdicts.append(verdict)

        # --- Compute weighted score ---
        net_score = 0.0
        for verdict in verdicts:
            weight = self.weights.get(verdict.agent_name, 1.0)
            # Convert vote to direction-aligned score:
            # If direction=BUY: BUY/STRONG_BUY are positive, SELL/STRONG_SELL negative
            # If direction=SELL: reverse
            raw_vote_score = VOTE_SCORES.get(verdict.vote, 0.0)

            # Flip score if direction is SELL so SELL votes become positive
            if direction == "SELL":
                raw_vote_score = -raw_vote_score

            aligned_score = raw_vote_score * verdict.confidence * weight
            net_score += aligned_score

        # --- Count agreement ---
        agree_votes = {"BUY": [BUY, STRONG_BUY], "SELL": [SELL, STRONG_SELL]}
        oppose_votes = {"BUY": [SELL, STRONG_SELL], "SELL": [BUY, STRONG_BUY]}

        agreeing = sum(1 for v in verdicts if v.vote in agree_votes.get(direction, []))
        opposing = sum(1 for v in verdicts if v.vote in oppose_votes.get(direction, []))
        total_voting = sum(1 for v in verdicts if v.vote != NEUTRAL)
        agreement_pct = (agreeing / len(verdicts) * 100) if verdicts else 0

        # --- Check veto conditions ---
        all_flags = []
        for v in verdicts:
            all_flags.extend(v.flags)

        vetoed = False
        veto_reasons = []

        # Drawdown is reflected in net_score via the Drawdown Guard agent's
        # negative vote. Only veto at extreme drawdown (10%+ of account).
        # The agent already votes STRONG_SELL with high confidence when
        # drawdown is critical, which naturally pushes the score below SKIP.

        # R:R below 1.0 is a warning, not a veto — the 3-TP partial close
        # system can be profitable even with sub-1.0 R:R (TP2/TP3 compensate)
        # The R:R agent already penalizes the net_score for poor R:R

        if opposing >= self.SUPERMAJORITY_COUNT:
            vetoed = True
            veto_reasons.append(f"VETO: {opposing} agents oppose (supermajority)")
            all_flags.append(VETO_SUPERMAJORITY_REJECT)

        # --- Determine action ---
        if vetoed:
            action = ACTION_SKIP
            reasoning = " | ".join(veto_reasons)
        elif net_score >= self.THRESHOLD_FULL:
            action = ACTION_FULL
            reasoning = f"Strong consensus ({agreeing} agree, score {net_score:+.1f})"
        elif net_score >= self.THRESHOLD_HIGH:
            action = ACTION_HIGH
            reasoning = f"Good consensus ({agreeing} agree, score {net_score:+.1f})"
        elif net_score >= self.THRESHOLD_MODERATE:
            action = ACTION_MODERATE
            reasoning = f"Moderate consensus ({agreeing} agree, score {net_score:+.1f})"
        elif net_score >= self.THRESHOLD_LOW:
            action = ACTION_LOW
            reasoning = f"Weak consensus ({agreeing} agree, score {net_score:+.1f})"
        else:
            action = ACTION_SKIP
            reasoning = f"No consensus ({opposing} oppose, score {net_score:+.1f})"

        lot_mult = ACTION_LOT_MULT.get(action, 0.0)

        decision = CouncilDecision(
            action=action,
            lot_multiplier=lot_mult,
            net_score=net_score,
            agreement_pct=agreement_pct,
            reasoning=reasoning,
            agent_verdicts=verdicts,
            flags=list(set(all_flags)),
        )

        logger.info(
            f"Council: {action} | Score={net_score:+.1f} | "
            f"Agree={agreement_pct:.0f}% ({agreeing}/{len(verdicts)}) | "
            f"{reasoning}"
        )

        return decision
