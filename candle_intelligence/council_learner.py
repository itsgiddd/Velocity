"""
Council Learner — Agentic self-learning for the Trading Council.

Tracks per-agent accuracy from trade outcomes and dynamically adjusts
agent weights so the council gets smarter over time.

After each trade closes:
  1. Record which agents voted correctly vs incorrectly
  2. Maintain per-agent accuracy (rolling 100-trade window)
  3. Adjust weights: higher accuracy → more weight (min 0.3, max 2.5)
  4. Persist performance in SQLite for continuity across restarts
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from candle_intelligence.agent_base import (
    AgentVerdict, CouncilDecision, BUY, SELL, STRONG_BUY, STRONG_SELL, NEUTRAL,
)

logger = logging.getLogger(__name__)

DB_FILE = "council_performance.db"
ROLLING_WINDOW = 100       # Track last N trades for accuracy
MIN_TRADES_FOR_ADJUST = 20 # Need at least this many to start adjusting
WEIGHT_MIN = 0.3
WEIGHT_MAX = 2.5


class CouncilLearner:
    """Learns agent weights from trade outcomes."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_FILE
        self._init_db()
        self._agent_history: Dict[str, List[dict]] = defaultdict(list)
        self._load_history()

    def _init_db(self):
        """Create SQLite tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS council_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    outcome TEXT NOT NULL,         -- 'WIN' or 'LOSS'
                    pnl REAL NOT NULL,
                    council_action TEXT NOT NULL,
                    council_score REAL NOT NULL,
                    agent_verdicts TEXT NOT NULL    -- JSON array of verdicts
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_weights (
                    agent_name TEXT PRIMARY KEY,
                    weight REAL NOT NULL,
                    accuracy REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    correct_trades INTEGER NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Council DB init error: {e}")

    def _load_history(self):
        """Load recent agent performance from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT agent_verdicts, outcome FROM council_trades "
                "ORDER BY id DESC LIMIT ?",
                (ROLLING_WINDOW,)
            ).fetchall()
            conn.close()

            for verdicts_json, outcome in reversed(rows):
                verdicts = json.loads(verdicts_json)
                for v in verdicts:
                    name = v.get("agent_name", "")
                    if name:
                        self._agent_history[name].append({
                            "vote": v.get("vote", NEUTRAL),
                            "confidence": v.get("confidence", 0),
                            "outcome": outcome,
                        })
                        # Keep only last ROLLING_WINDOW entries
                        if len(self._agent_history[name]) > ROLLING_WINDOW:
                            self._agent_history[name] = self._agent_history[name][-ROLLING_WINDOW:]

        except Exception as e:
            logger.warning(f"Could not load council history: {e}")

    def record_trade_outcome(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        decision: CouncilDecision,
    ):
        """Record a trade outcome and update agent accuracy."""
        outcome = "WIN" if pnl > 0 else "LOSS"

        # Serialize verdicts
        verdicts_data = []
        for v in decision.agent_verdicts:
            verdicts_data.append({
                "agent_name": v.agent_name,
                "vote": v.vote,
                "confidence": v.confidence,
                "reasoning": v.reasoning,
            })

        # Store in DB
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO council_trades "
                "(timestamp, symbol, direction, outcome, pnl, council_action, "
                "council_score, agent_verdicts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    symbol, direction, outcome, pnl,
                    decision.action, decision.net_score,
                    json.dumps(verdicts_data),
                )
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to record council trade: {e}")

        # Update per-agent history
        for v in decision.agent_verdicts:
            self._agent_history[v.agent_name].append({
                "vote": v.vote,
                "confidence": v.confidence,
                "outcome": outcome,
            })
            # Trim to rolling window
            if len(self._agent_history[v.agent_name]) > ROLLING_WINDOW:
                self._agent_history[v.agent_name] = \
                    self._agent_history[v.agent_name][-ROLLING_WINDOW:]

    def compute_agent_weights(self) -> Dict[str, float]:
        """Compute dynamic weights based on per-agent accuracy.

        An agent is "correct" if:
          - Trade WON and agent voted with the direction (BUY/STRONG_BUY for BUY trade)
          - Trade LOST and agent voted against or NEUTRAL

        Weight formula:
          accuracy = correct / total
          weight = default_weight × (0.5 + accuracy)
          Clamped to [WEIGHT_MIN, WEIGHT_MAX]
        """
        weights = {}

        for agent_name, history in self._agent_history.items():
            if len(history) < MIN_TRADES_FOR_ADJUST:
                continue

            correct = 0
            total = len(history)

            for entry in history:
                vote = entry["vote"]
                outcome = entry["outcome"]

                # "Correct" = voted with direction AND won, OR voted against AND lost
                voted_bullish = vote in (BUY, STRONG_BUY)
                voted_bearish = vote in (SELL, STRONG_SELL)
                voted_neutral = vote == NEUTRAL

                if outcome == "WIN":
                    if voted_bullish or voted_neutral:
                        correct += 1
                elif outcome == "LOSS":
                    if voted_bearish or voted_neutral:
                        correct += 1

            accuracy = correct / total if total > 0 else 0.5
            # Scale weight: 50% accuracy = 1.0x, 70% = 1.2x, 30% = 0.8x
            new_weight = 0.5 + accuracy  # Range: 0.5 to 1.5
            new_weight = max(WEIGHT_MIN, min(WEIGHT_MAX, new_weight))
            weights[agent_name] = new_weight

            # Persist
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute(
                    "INSERT OR REPLACE INTO agent_weights "
                    "(agent_name, weight, accuracy, total_trades, correct_trades, last_updated) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (agent_name, new_weight, accuracy, total, correct,
                     datetime.now().isoformat()),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to persist weight for {agent_name}: {e}")

        return weights

    def get_agent_stats(self) -> Dict[str, dict]:
        """Get per-agent accuracy stats for dashboard display."""
        stats = {}
        for agent_name, history in self._agent_history.items():
            total = len(history)
            if total == 0:
                continue

            correct = 0
            for entry in history:
                vote = entry["vote"]
                outcome = entry["outcome"]
                voted_bullish = vote in (BUY, STRONG_BUY)
                voted_bearish = vote in (SELL, STRONG_SELL)
                voted_neutral = vote == NEUTRAL
                if outcome == "WIN" and (voted_bullish or voted_neutral):
                    correct += 1
                elif outcome == "LOSS" and (voted_bearish or voted_neutral):
                    correct += 1

            accuracy = correct / total
            vote_dist = defaultdict(int)
            for entry in history:
                vote_dist[entry["vote"]] += 1

            stats[agent_name] = {
                "accuracy": accuracy,
                "total_trades": total,
                "correct_trades": correct,
                "vote_distribution": dict(vote_dist),
            }

        return stats

    def load_saved_weights(self) -> Dict[str, float]:
        """Load previously saved weights from database."""
        weights = {}
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT agent_name, weight FROM agent_weights"
            ).fetchall()
            conn.close()
            for name, w in rows:
                weights[name] = max(WEIGHT_MIN, min(WEIGHT_MAX, w))
        except Exception as e:
            logger.warning(f"Could not load saved weights: {e}")
        return weights
