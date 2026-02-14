"""Agentic orchestrator: autonomous model lifecycle management.

Wires together:
  - TradeJournal          (persistent trade store)
  - AdaptivePerformanceMonitor (degradation detection)
  - WarmStartRetrainer    (retraining pipeline)
  - ModelHotSwap          (atomic deploy + probation + rollback)

Runs a background daemon that checks retrain triggers periodically
and executes the full retrain -> validate -> deploy -> probation cycle.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

# Add project root to sys.path so cross-folder imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agentic.adaptive_performance_monitor import AdaptivePerformanceMonitor
from agentic.model_hot_swap import ModelHotSwap
from agentic.trade_journal import TradeJournal, TradeRecord
from agentic.warm_start_retrainer import WarmStartRetrainer

logger = logging.getLogger(__name__)


class AgenticOrchestrator:
    """Autonomous model lifecycle: monitor -> retrain -> validate -> deploy."""

    def __init__(
        self,
        model_manager,
        trading_engine=None,
        model_path: str = "neural_model.pth",
        backup_dir: str = "model_backups",
        journal_path: str = "trade_journal.db",
        check_interval_minutes: int = 15,
        retrain_after_n_trades: int = 100,
        retrain_after_n_days: int = 7,
        probation_trades: int = 20,
        probation_min_win_rate: float = 0.35,
        symbols: list | None = None,
    ) -> None:
        self.model_manager = model_manager
        self.trading_engine = trading_engine
        self.model_path = model_path
        self.check_interval = check_interval_minutes * 60  # seconds

        # Sub-components
        self.journal = TradeJournal(journal_path)

        self.monitor = AdaptivePerformanceMonitor(
            trade_journal=self.journal,
            baseline_win_rate=0.50,
            baseline_profit_factor=1.0,
            rolling_window=50,
            degradation_threshold_pct=0.15,
            min_trades_for_evaluation=30,
        )
        self.monitor._retrain_trade_threshold = retrain_after_n_trades
        self.monitor._retrain_interval_days = retrain_after_n_days

        self.retrainer = WarmStartRetrainer(
            model_path=model_path,
            backup_dir=backup_dir,
            symbols=symbols or [
                "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
                "NZDUSD", "EURJPY", "GBPJPY", "BTCUSD",
            ],
        )

        self.hot_swap = ModelHotSwap(
            model_manager=model_manager,
            live_model_path=model_path,
            backup_dir=backup_dir,
            probation_trades=probation_trades,
            probation_min_win_rate=probation_min_win_rate,
        )

        # Load baseline from current model checkpoint
        self._init_baseline()

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._retrain_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Orchestrator already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._orchestration_loop, daemon=True, name="AgenticOrchestrator",
        )
        self._thread.start()
        logger.info(
            f"Agentic orchestrator started (check every {self.check_interval // 60} min)"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        logger.info("Agentic orchestrator stopped")

    # ------------------------------------------------------------------
    # Trade event hook (called by trading engine)
    # ------------------------------------------------------------------

    def notify_trade_closed(self, record: TradeRecord) -> None:
        """Called by the trading engine when a position closes."""
        try:
            self.journal.record_trade(record)
        except Exception as e:
            logger.error(f"Journal write failed: {e}")

        # Track probation trades
        self.hot_swap.record_probation_trade(record.pnl)
        probation_result = self.hot_swap.check_probation()
        if probation_result == "rollback":
            logger.warning("Probation FAILED — rolling back to previous model")
            self.hot_swap.rollback()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _orchestration_loop(self) -> None:
        logger.info("Orchestration loop started")
        while not self._stop_event.is_set():
            try:
                self._check_and_act()
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}", exc_info=True)

            # Sleep in small increments so we can stop promptly
            for _ in range(self.check_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _check_and_act(self) -> None:
        triggers = self.monitor.check_retrain_triggers()
        if not triggers.get("should_retrain", False):
            return

        reason_parts = []
        if triggers.get("degradation"):
            reason_parts.append("performance degradation")
        if triggers.get("trade_count"):
            reason_parts.append(f"trade count ({triggers.get('trades_since', '?')})")
        if triggers.get("time_elapsed"):
            reason_parts.append(f"time elapsed ({triggers.get('days_since', '?'):.1f}d)")

        logger.info(f"Retrain triggered: {', '.join(reason_parts)}")
        self._execute_retrain_cycle()

    def _execute_retrain_cycle(self) -> None:
        if not self._retrain_lock.acquire(blocking=False):
            logger.info("Retrain already in progress, skipping")
            return
        try:
            self._run_retrain()
        finally:
            self._retrain_lock.release()

    def _run_retrain(self) -> None:
        logger.info("=" * 60)
        logger.info("AGENTIC RETRAIN CYCLE STARTING")
        logger.info("=" * 60)

        # 1. Get current model metrics for comparison
        current_metrics = self.retrainer.get_current_model_metrics()
        if not current_metrics:
            logger.warning("No current model metrics available — using defaults")
            current_metrics = {
                "expectancy": 0, "profit_factor": 1.0, "win_rate": 0.5,
            }

        # 2. Retrain
        retrain_result = self.retrainer.retrain(warm_start=True)
        if not retrain_result.get("success"):
            logger.error(f"Retrain failed: {retrain_result.get('error', 'unknown')}")
            self.monitor.record_retrain_event()  # reset timers even on failure
            return

        candidate_metrics = retrain_result.get("candidate_metrics", {})
        warm_used = retrain_result.get("warm_start_used", False)
        duration = retrain_result.get("duration_seconds", 0)

        logger.info(
            f"Retrain complete in {duration:.0f}s (warm_start={warm_used}). "
            f"Candidate: exp={candidate_metrics.get('expectancy', 0):.6f}, "
            f"PF={candidate_metrics.get('profit_factor', 0):.3f}, "
            f"WR={candidate_metrics.get('win_rate', 0):.3f}"
        )
        logger.info(
            f"Current:   exp={current_metrics.get('expectancy', 0):.6f}, "
            f"PF={current_metrics.get('profit_factor', 0):.3f}, "
            f"WR={current_metrics.get('win_rate', 0):.3f}"
        )

        # 3. Validate candidate vs current
        if not self.hot_swap.validate_candidate(candidate_metrics, current_metrics):
            logger.info("Candidate did NOT pass validation gate — keeping current model")
            self.monitor.record_retrain_event()
            return

        # 4. Deploy
        candidate_path = retrain_result["candidate_path"]
        deployed = self.hot_swap.deploy_candidate(candidate_path)
        if deployed:
            logger.info(
                f"Candidate DEPLOYED. Probation started "
                f"({self.hot_swap.probation_trades} trades)."
            )
            # Update monitor baseline from new model
            self._init_baseline()
        else:
            logger.error("Candidate deployment FAILED")

        self.monitor.record_retrain_event()

        logger.info("=" * 60)
        logger.info("AGENTIC RETRAIN CYCLE COMPLETE")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_baseline(self) -> None:
        """Load baseline stats from the current model checkpoint metadata."""
        try:
            import torch
            from pathlib import Path

            p = Path(self.model_path).resolve()
            if not p.exists():
                return
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            meta = ckpt.get("metadata", {})
            self.monitor.update_baseline_from_model_metadata(meta)
        except Exception as e:
            logger.warning(f"Could not init baseline from checkpoint: {e}")

    def get_status(self) -> dict:
        """Return orchestrator status for dashboards."""
        is_running = self._thread is not None and self._thread.is_alive()
        trade_count = self.journal.get_trade_count()
        rolling = self.journal.get_rolling_stats(window=50)
        triggers = self.monitor.check_retrain_triggers()
        probation = self.hot_swap._probation_active

        return {
            "running": is_running,
            "total_journal_trades": trade_count,
            "rolling_win_rate": rolling.get("win_rate"),
            "rolling_profit_factor": rolling.get("profit_factor"),
            "rolling_avg_pnl_pct": rolling.get("avg_pnl_pct"),
            "retrain_triggers": triggers,
            "probation_active": probation,
            "probation_trades": self.hot_swap._probation_trade_count,
            "probation_wins": self.hot_swap._probation_wins,
        }
