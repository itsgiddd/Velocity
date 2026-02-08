from __future__ import annotations

from typing import Dict


class DailyPlanner:
    """
    Stores operator-defined daily bias by symbol.
    """

    def __init__(self):
        self.plan: Dict[str, str] = {}

    def set_plan(self, plan: Dict[str, str]) -> None:
        self.plan = {k.upper(): str(v).upper() for k, v in (plan or {}).items()}

    def get_bias(self, symbol: str) -> str:
        # Expected output: LONG / SHORT / NEUTRAL
        value = self.plan.get(symbol.upper(), "NEUTRAL")
        if value in {"BUY", "BULL", "BULLISH"}:
            return "LONG"
        if value in {"SELL", "BEAR", "BEARISH"}:
            return "SHORT"
        if value in {"LONG", "SHORT", "NEUTRAL"}:
            return value
        return "NEUTRAL"
