"""Detects model degradation by comparing live performance to training baseline.

The monitor reads from the persistent TradeJournal and compares rolling
win rate, profit factor, and expectancy against the model's validation
metrics stored at training time.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Add project root to sys.path so cross-folder imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agentic.trade_journal import TradeJournal

logger = logging.getLogger(__name__)


class AdaptivePerformanceMonitor:
    """Compares live trading performance to model training baseline."""

    def __init__(
        self,
        trade_journal: TradeJournal,
        baseline_win_rate: float = 0.50,
        baseline_profit_factor: float = 1.05,
        baseline_expectancy: float = 0.0001,
        rolling_window: int = 50,
        degradation_threshold_pct: float = 0.15,
        min_trades_for_evaluation: int = 30,
    ) -> None:
        self.trade_journal = trade_journal
        self.baseline_win_rate = baseline_win_rate
        self.baseline_profit_factor = baseline_profit_factor
        self.baseline_expectancy = baseline_expectancy
        self.rolling_window = rolling_window
        self.degradation_threshold = degradation_threshold_pct
        self.min_trades = min_trades_for_evaluation

        self._last_retrain_time: Optional[datetime] = None
        self._trades_at_last_retrain: int = 0
        self._retrain_interval_days: int = 7
        self._retrain_trade_threshold: int = 100

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def update_baseline_from_model_metadata(self, metadata: Dict) -> None:
        """Extract baseline stats from model checkpoint metadata."""
        if not isinstance(metadata, dict):
            return

        prof_quality = metadata.get("profitability_quality") or {}
        if isinstance(prof_quality, dict):
            self.baseline_win_rate = float(prof_quality.get("win_rate", self.baseline_win_rate))
            self.baseline_profit_factor = float(
                prof_quality.get("profit_factor", self.baseline_profit_factor)
            )
            self.baseline_expectancy = float(
                prof_quality.get("expectancy", self.baseline_expectancy)
            )

        training_date = metadata.get("training_date")
        if training_date and self._last_retrain_time is None:
            try:
                self._last_retrain_time = datetime.fromisoformat(str(training_date))
            except (ValueError, TypeError):
                pass

        logger.info(
            f"Performance baseline updated: win_rate={self.baseline_win_rate:.2%}, "
            f"PF={self.baseline_profit_factor:.2f}, exp={self.baseline_expectancy:.6f}"
        )

    # ------------------------------------------------------------------
    # Degradation check
    # ------------------------------------------------------------------

    def check_degradation(self) -> Tuple[bool, Dict[str, float]]:
        """Compare live rolling stats to training baseline.

        Returns (is_degraded, stats_dict).
        """
        stats = self.trade_journal.get_rolling_stats(self.rolling_window)
        trade_count = int(stats.get("trade_count", 0))

        result: Dict[str, float] = {
            "live_win_rate": stats.get("win_rate", 0.0),
            "live_profit_factor": stats.get("profit_factor", 0.0),
            "live_avg_pnl_pct": stats.get("avg_pnl_pct", 0.0),
            "live_trade_count": float(trade_count),
            "baseline_win_rate": self.baseline_win_rate,
            "baseline_profit_factor": self.baseline_profit_factor,
            "baseline_expectancy": self.baseline_expectancy,
        }

        if trade_count < self.min_trades:
            return False, result

        live_wr = stats["win_rate"]
        live_pf = stats["profit_factor"]

        # Relative drop vs baseline.
        wr_drop = (self.baseline_win_rate - live_wr) / max(self.baseline_win_rate, 1e-8)
        pf_drop = (self.baseline_profit_factor - live_pf) / max(self.baseline_profit_factor, 1e-8)

        is_degraded = wr_drop > self.degradation_threshold or pf_drop > self.degradation_threshold

        return is_degraded, result

    # ------------------------------------------------------------------
    # Retrain triggers
    # ------------------------------------------------------------------

    def check_retrain_triggers(self) -> Dict[str, bool]:
        """Return a dict of retrain trigger flags."""
        is_degraded, _ = self.check_degradation()

        total_trades = self.trade_journal.get_trade_count()
        trades_since = total_trades - self._trades_at_last_retrain

        days_since = 0
        if self._last_retrain_time:
            days_since = (datetime.now() - self._last_retrain_time).days

        trade_trigger = trades_since >= self._retrain_trade_threshold
        time_trigger = days_since >= self._retrain_interval_days if self._last_retrain_time else False

        return {
            "degradation": is_degraded,
            "trade_count": trade_trigger,
            "time_elapsed": time_trigger,
            "should_retrain": is_degraded or trade_trigger or time_trigger,
            "trades_since": trades_since,
            "days_since": float(days_since),
        }

    def record_retrain_event(self) -> None:
        """Reset counters after a retrain completes."""
        self._last_retrain_time = datetime.now()
        self._trades_at_last_retrain = self.trade_journal.get_trade_count()
        logger.info("Retrain event recorded — counters reset")
