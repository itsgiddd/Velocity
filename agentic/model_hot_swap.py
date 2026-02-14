"""Atomic model swap with validation gate, probation, and auto-rollback.

Flow:
  1. Candidate model trained and saved to temp path.
  2. Validate candidate metrics against current model.
  3. If better: backup current, swap, start probation.
  4. During probation: monitor first N trades; rollback if bad.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ModelHotSwap:
    """Safe model deployment with probation and rollback."""

    def __init__(
        self,
        model_manager,
        live_model_path: str = "neural_model.pth",
        backup_dir: str = "model_backups",
        probation_trades: int = 20,
        probation_min_win_rate: float = 0.35,
    ) -> None:
        self.model_manager = model_manager
        self.live_model_path = Path(live_model_path).resolve()
        self.backup_dir = Path(backup_dir).resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.probation_trades = probation_trades
        self.probation_min_win_rate = probation_min_win_rate

        # Probation state
        self._probation_active = False
        self._probation_trade_count = 0
        self._probation_wins = 0
        self._rollback_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Validation gate
    # ------------------------------------------------------------------

    def validate_candidate(
        self,
        candidate_metrics: Dict,
        current_metrics: Dict,
    ) -> bool:
        """Candidate must be strictly better on expectancy or PF, not worse on win_rate."""
        c_exp = float(candidate_metrics.get("expectancy", 0.0))
        c_pf = float(candidate_metrics.get("profit_factor", 0.0))
        c_wr = float(candidate_metrics.get("win_rate", 0.0))

        cur_exp = float(current_metrics.get("expectancy", 0.0))
        cur_pf = float(current_metrics.get("profit_factor", 0.0))
        cur_wr = float(current_metrics.get("win_rate", 0.0))

        better_exp = c_exp > cur_exp * 1.0  # at least as good
        better_pf = c_pf > cur_pf * 0.95  # within 5%
        not_worse_wr = c_wr >= cur_wr * 0.90  # within 10%
        strictly_better = c_exp > cur_exp or c_pf > cur_pf

        result = strictly_better and not_worse_wr
        logger.info(
            f"Candidate validation: exp={c_exp:.6f} vs {cur_exp:.6f}, "
            f"PF={c_pf:.3f} vs {cur_pf:.3f}, "
            f"WR={c_wr:.3f} vs {cur_wr:.3f} -> {'PASS' if result else 'FAIL'}"
        )
        return result

    # ------------------------------------------------------------------
    # Deploy / rollback
    # ------------------------------------------------------------------

    def deploy_candidate(self, candidate_path: str) -> bool:
        """Backup current model, copy candidate to live path, reload model."""
        candidate = Path(candidate_path).resolve()
        if not candidate.exists():
            logger.error(f"Candidate model not found: {candidate}")
            return False

        try:
            # Backup current model.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{timestamp}.pth"
            if self.live_model_path.exists():
                shutil.copy2(str(self.live_model_path), str(backup_path))
                self._rollback_path = backup_path
                logger.info(f"Current model backed up to {backup_path}")

            # Copy candidate to live path.
            shutil.copy2(str(candidate), str(self.live_model_path))

            # Hot-reload via model_manager (thread-safe via its internal lock).
            success = self.model_manager.load_model(str(self.live_model_path))
            if not success:
                logger.error("Model manager failed to load candidate — rolling back")
                self.rollback()
                return False

            # Start probation.
            self._probation_active = True
            self._probation_trade_count = 0
            self._probation_wins = 0
            logger.info(
                f"Candidate deployed. Probation started ({self.probation_trades} trades)."
            )
            return True

        except Exception as e:
            logger.error(f"Deploy failed: {e}")
            self.rollback()
            return False

    def record_probation_trade(self, pnl: float) -> None:
        if not self._probation_active:
            return
        self._probation_trade_count += 1
        if pnl > 0:
            self._probation_wins += 1

    def check_probation(self) -> Optional[str]:
        """Returns None if ongoing/passed, 'rollback' if model should be reverted."""
        if not self._probation_active:
            return None

        if self._probation_trade_count < self.probation_trades:
            return None  # still ongoing

        # Probation period complete — evaluate.
        win_rate = self._probation_wins / max(self._probation_trade_count, 1)
        self._probation_active = False

        if win_rate < self.probation_min_win_rate:
            logger.warning(
                f"Probation FAILED: win_rate={win_rate:.2%} < {self.probation_min_win_rate:.2%} "
                f"({self._probation_wins}/{self._probation_trade_count})"
            )
            return "rollback"

        logger.info(
            f"Probation PASSED: win_rate={win_rate:.2%} "
            f"({self._probation_wins}/{self._probation_trade_count})"
        )
        return None

    def rollback(self) -> bool:
        """Restore previous model from backup."""
        if self._rollback_path is None or not self._rollback_path.exists():
            logger.error("No rollback model available")
            return False

        try:
            shutil.copy2(str(self._rollback_path), str(self.live_model_path))
            success = self.model_manager.load_model(str(self.live_model_path))
            self._probation_active = False
            if success:
                logger.info(f"Rolled back to {self._rollback_path}")
            return success
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
