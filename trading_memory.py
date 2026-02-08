from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict


class TradingMemory:
    """
    Minimal guardrails against revenge trading and rapid repeat entries.
    """

    def __init__(self, max_loss_streak: int = 3, cooldown_minutes: int = 30):
        self.max_loss_streak = max_loss_streak
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.state: Dict[str, Dict] = {}

    def _symbol_state(self, symbol: str) -> Dict:
        key = symbol.upper()
        if key not in self.state:
            self.state[key] = {
                "loss_streak": 0,
                "cooldown_until": None,
                "last_direction": None,
            }
        return self.state[key]

    def can_trade(self, symbol: str) -> bool:
        s = self._symbol_state(symbol)
        now = datetime.now(timezone.utc)
        cooldown_until = s.get("cooldown_until")
        if cooldown_until and now < cooldown_until:
            return False
        return True

    def close_trade(self, symbol: str, profit: float):
        s = self._symbol_state(symbol)
        now = datetime.now(timezone.utc)

        if profit < 0:
            s["loss_streak"] += 1
        else:
            s["loss_streak"] = 0
            s["cooldown_until"] = None

        if s["loss_streak"] >= self.max_loss_streak:
            s["cooldown_until"] = now + self.cooldown
            s["loss_streak"] = 0

    def log_trade(self, symbol: str, direction: str):
        s = self._symbol_state(symbol)
        s["last_direction"] = str(direction).upper()
