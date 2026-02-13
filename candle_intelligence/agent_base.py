"""
Base classes for the Agentic Trading Council.

TradingAgent  — abstract base for all 15 agents
TradeContext  — all data an agent needs to analyze a trade
AgentVerdict  — an agent's vote + reasoning
CouncilDecision — the final aggregated decision
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vote enum-like constants
# ---------------------------------------------------------------------------

STRONG_BUY = "STRONG_BUY"
BUY = "BUY"
NEUTRAL = "NEUTRAL"
SELL = "SELL"
STRONG_SELL = "STRONG_SELL"

VOTE_SCORES = {
    STRONG_BUY: +2.0,
    BUY: +1.0,
    NEUTRAL: 0.0,
    SELL: -1.0,
    STRONG_SELL: -2.0,
}

# Decision action constants
ACTION_FULL = "FULL"
ACTION_HIGH = "HIGH"
ACTION_MODERATE = "MODERATE"
ACTION_LOW = "LOW"
ACTION_SKIP = "SKIP"

# Veto flag constants
VETO_CRITICAL_DRAWDOWN = "CRITICAL_DRAWDOWN"
VETO_RR_BELOW_1_0 = "RR_BELOW_1_0"
VETO_SUPERMAJORITY_REJECT = "SUPERMAJORITY_REJECT"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeContext:
    """All data an agent needs — passed to every agent."""

    # Trade setup
    symbol: str
    direction: str              # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    atr_value: float

    # Raw candle data (DataFrames with open/high/low/close/tick_volume/spread)
    m15_bars: Optional[pd.DataFrame] = None
    h1_bars: Optional[pd.DataFrame] = None
    h4_bars: Optional[pd.DataFrame] = None

    # Pre-computed ZeroPoint state DataFrames
    zp_h1: Optional[pd.DataFrame] = None
    zp_m15: Optional[pd.DataFrame] = None
    zp_h4: Optional[pd.DataFrame] = None

    # Account state
    balance: float = 0.0
    equity: float = 0.0
    open_positions: list = field(default_factory=list)
    recent_trades: list = field(default_factory=list)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0

    # Extras (push profiles, etc.)
    push_profiles: Optional[dict] = None


@dataclass
class AgentVerdict:
    """A single agent's vote on the trade."""

    agent_name: str
    specialty: str
    vote: str                   # STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
    confidence: float           # 0.0 to 1.0
    reasoning: str              # Brief text for logging
    flags: List[str] = field(default_factory=list)

    @property
    def numeric_score(self) -> float:
        """Convert vote to numeric score × confidence."""
        return VOTE_SCORES.get(self.vote, 0.0) * self.confidence


@dataclass
class CouncilDecision:
    """The final aggregated decision from the Trading Council."""

    action: str                 # FULL / HIGH / MODERATE / LOW / SKIP
    lot_multiplier: float       # 0.0 to 1.0
    net_score: float            # weighted aggregate score
    agreement_pct: float        # % of agents agreeing with trade direction
    reasoning: str              # council summary
    agent_verdicts: List[AgentVerdict] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base agent
# ---------------------------------------------------------------------------

class TradingAgent(ABC):
    """Base class for all Trading Council agents."""

    name: str = "BaseAgent"
    specialty: str = "general"
    default_weight: float = 1.0

    @abstractmethod
    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        """Analyze the trade setup and return a verdict."""
        ...

    def _safe_analyze(self, ctx: TradeContext) -> AgentVerdict:
        """Wrapper that catches exceptions and returns NEUTRAL on error."""
        try:
            return self.analyze(ctx)
        except Exception as e:
            logger.warning(f"{self.name} error: {e}")
            return AgentVerdict(
                agent_name=self.name,
                specialty=self.specialty,
                vote=NEUTRAL,
                confidence=0.0,
                reasoning=f"Error: {e}",
            )

    def _verdict(self, vote: str, confidence: float, reasoning: str,
                 flags: Optional[List[str]] = None) -> AgentVerdict:
        """Helper to create a verdict."""
        return AgentVerdict(
            agent_name=self.name,
            specialty=self.specialty,
            vote=vote,
            confidence=min(1.0, max(0.0, confidence)),
            reasoning=reasoning,
            flags=flags or [],
        )

    def _direction_vote(self, score: float, direction: str) -> str:
        """Convert a -1..+1 score into a vote aligned with trade direction.

        score > 0 means the analysis SUPPORTS the trade direction.
        score < 0 means the analysis OPPOSES the trade direction.

        Wide NEUTRAL band (-0.4 to +0.4) so agents only vote when they
        have real conviction — prevents weak signals from dragging the
        council score negative and causing excessive skips.
        """
        if score > 0.7:
            return STRONG_BUY if direction == "BUY" else STRONG_SELL
        elif score > 0.4:
            return BUY if direction == "BUY" else SELL
        elif score < -0.7:
            return STRONG_SELL if direction == "BUY" else STRONG_BUY
        elif score < -0.4:
            return SELL if direction == "BUY" else BUY
        return NEUTRAL
