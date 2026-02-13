"""
Agentic Trading Council — Multi-Agent Reasoning System
=======================================================

A team of 15 specialized AI agents that each analyze a trade from
a different angle (candle patterns, momentum, volume, risk, etc.),
cast votes, and reach a weighted consensus before any trade is placed.
"""

from candle_intelligence.agent_base import (
    TradingAgent,
    TradeContext,
    AgentVerdict,
    CouncilDecision,
)
from candle_intelligence.trading_council import TradingCouncil
from candle_intelligence.council_predictor import (
    CouncilPredictor,
    CouncilPredictorEngine,
    extract_features,
    extract_features_from_context,
)

__all__ = [
    "TradingAgent",
    "TradeContext",
    "AgentVerdict",
    "CouncilDecision",
    "TradingCouncil",
    "CouncilPredictor",
    "CouncilPredictorEngine",
    "extract_features",
    "extract_features_from_context",
]
