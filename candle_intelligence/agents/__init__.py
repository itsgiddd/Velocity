"""Agent registry — imports all 15 agents for the Trading Council."""

from candle_intelligence.agents.candle_agents import (
    SingleCandleAgent,
    MultiBarFormationAgent,
    CandleMomentumAgent,
    CandleAnatomyAgent,
)
from candle_intelligence.agents.trend_agents import (
    MultiTFTrendAgent,
    MovingAverageAgent,
    SupportResistanceAgent,
    PushStructureAgent,
)
from candle_intelligence.agents.volume_agents import (
    VolumeProfileAgent,
    VolumeMomentumAgent,
    SpreadLiquidityAgent,
)
from candle_intelligence.agents.risk_agents import (
    RiskRewardAgent,
    SessionTimingAgent,
    DrawdownGuardAgent,
    CorrelationAgent,
)

ALL_AGENTS = [
    SingleCandleAgent,
    MultiBarFormationAgent,
    CandleMomentumAgent,
    CandleAnatomyAgent,
    MultiTFTrendAgent,
    MovingAverageAgent,
    SupportResistanceAgent,
    PushStructureAgent,
    VolumeProfileAgent,
    VolumeMomentumAgent,
    SpreadLiquidityAgent,
    RiskRewardAgent,
    SessionTimingAgent,
    DrawdownGuardAgent,
    CorrelationAgent,
]
