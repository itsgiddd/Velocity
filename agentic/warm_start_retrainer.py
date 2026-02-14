"""Warm-start retraining wrapper around simple_neural_trainer.

Imports the existing training pipeline and adds:
  - Warm-start: load previous model weights, fine-tune with lower LR.
  - Feature-dim guard: fall back to full retrain if architecture changed.
  - Saves candidate to a temp path for hot-swap validation.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to sys.path so cross-folder imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim

from training.simple_neural_trainer import (
    SimpleNeuralNetwork,
    setup_mt5,
    collect_historical_data,
    build_symbol_index,
    compute_all_push_profiles,
    create_features_and_labels,
    fit_scaler,
    apply_scaler,
    train_neural_network,
    compute_class_weights,
    evaluate_predictions,
    calibrate_symbol_thresholds,
    compute_symbol_profitability_stats,
    build_symbol_live_profile,
    evaluate_profitability_quality,
    save_model,
    model_predict_proba,
    DEFAULT_WEEKLY_SAMPLES_PER_SYMBOL,
    TARGET_WEEKLY_TRADES,
)

logger = logging.getLogger(__name__)


class WarmStartRetrainer:
    """Retrain the neural model with optional warm-start from previous weights."""

    def __init__(
        self,
        model_path: str = "neural_model.pth",
        candidate_path: str = "neural_model_candidate.pth",
        backup_dir: str = "model_backups",
        symbols: Optional[List[str]] = None,
        days: int = 180,
        warm_start_epochs: int = 60,
        full_train_epochs: int = 140,
        warm_start_lr: float = 0.0003,
        full_train_lr: float = 0.001,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.candidate_path = Path(candidate_path).resolve()
        self.backup_dir = Path(backup_dir).resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.symbols = symbols or [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "NZDUSD", "EURGBP", "GBPJPY", "BTCUSD",
        ]
        self.days = days
        self.warm_start_epochs = warm_start_epochs
        self.full_train_epochs = full_train_epochs
        self.warm_start_lr = warm_start_lr
        self.full_train_lr = full_train_lr

    # ------------------------------------------------------------------
    # Previous checkpoint helpers
    # ------------------------------------------------------------------

    def _load_previous_checkpoint(self) -> Optional[Dict]:
        if not self.model_path.exists():
            return None
        try:
            ckpt = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
            return ckpt
        except Exception as e:
            logger.warning(f"Could not load previous checkpoint: {e}")
            return None

    def _can_warm_start(self, prev_ckpt: Optional[Dict], new_feature_dim: int) -> bool:
        if prev_ckpt is None:
            return False
        prev_dim = prev_ckpt.get("feature_dim")
        if prev_dim is None or prev_dim != new_feature_dim:
            logger.info(
                f"Feature dim mismatch (prev={prev_dim}, new={new_feature_dim}). "
                "Falling back to full retrain."
            )
            return False
        if "model_state_dict" not in prev_ckpt:
            return False
        return True

    # ------------------------------------------------------------------
    # Core retrain
    # ------------------------------------------------------------------

    def retrain(self, warm_start: bool = True) -> Dict:
        """Run full retrain pipeline.  Returns result dict with metrics."""
        result: Dict = {"success": False, "warm_start_used": False}
        start_time = datetime.now()

        # 1. MT5 setup
        if not setup_mt5():
            result["error"] = "MT5 initialization failed"
            logger.error(result["error"])
            return result

        # 2. Collect data
        logger.info(f"Collecting {self.days} days of data for {len(self.symbols)} symbols...")
        data = collect_historical_data(symbols=self.symbols, days=self.days)
        if not data:
            result["error"] = "No data collected"
            logger.error(result["error"])
            return result
        logger.info(f"Data collected for {len(data)} symbols")

        # 3. Symbol index + push profiles
        symbol_keys = sorted(data.keys())
        symbol_to_index = build_symbol_index(symbol_keys)
        push_profiles = compute_all_push_profiles(data)

        # 4. Features + labels
        dataset = create_features_and_labels(
            data, symbol_to_index, lookback=20, horizon=8, push_profiles=push_profiles,
        )
        n_total = len(dataset.features)
        if n_total < 500:
            result["error"] = f"Insufficient samples: {n_total}"
            logger.error(result["error"])
            return result

        # 5. Train/val split (80/20)
        split = int(n_total * 0.8)
        train_features, val_features = dataset.features[:split], dataset.features[split:]
        train_labels, val_labels = dataset.labels[:split], dataset.labels[split:]
        val_symbols = dataset.symbols[split:]
        val_future_returns = dataset.future_returns[split:]
        val_spread_costs = dataset.spread_costs[split:]

        # 6. Scaler
        mean, std = fit_scaler(train_features)
        train_scaled = apply_scaler(train_features, mean, std)
        val_scaled = apply_scaler(val_features, mean, std)

        feature_dim = train_scaled.shape[1]

        # 7. Decide warm-start vs full
        prev_ckpt = self._load_previous_checkpoint() if warm_start else None
        use_warm_start = self._can_warm_start(prev_ckpt, feature_dim)

        if use_warm_start:
            epochs = self.warm_start_epochs
            lr = self.warm_start_lr
            logger.info(f"Warm-start retrain: {epochs} epochs, LR={lr}")
            model, summary = self._warm_start_train(
                prev_ckpt, feature_dim,
                train_scaled, train_labels, val_scaled, val_labels,
                epochs=epochs, lr=lr,
            )
            result["warm_start_used"] = True
        else:
            epochs = self.full_train_epochs
            lr = self.full_train_lr
            logger.info(f"Full retrain: {epochs} epochs, LR={lr}")
            model, summary = train_neural_network(
                train_scaled, train_labels, val_scaled, val_labels,
                epochs=epochs, learning_rate=lr, batch_size=256,
                class_weight_mode="sqrt_balanced", label_smoothing=0.03,
            )

        # 9. Calibrate thresholds
        weekly_samples = DEFAULT_WEEKLY_SAMPLES_PER_SYMBOL
        thresholds, action_modes, diagnostics, global_threshold = calibrate_symbol_thresholds(
            model, val_scaled, val_labels, val_symbols,
            val_future_returns, val_spread_costs,
            weekly_sample_count=weekly_samples,
            tail_focus_weight=0.3,
            target_weekly_trades=TARGET_WEEKLY_TRADES,
            target_trade_weight=0.15,
        )

        # 10. Profitability stats + live profile
        symbol_profitability = compute_symbol_profitability_stats(
            model_predict_proba(model, val_scaled),
            val_symbols, val_future_returns, val_spread_costs,
            thresholds, action_modes, weekly_samples,
        )
        live_profile = build_symbol_live_profile(symbol_profitability)

        # 11. Overall profitability
        val_probs = model_predict_proba(model, val_scaled)
        overall_quality = evaluate_profitability_quality(
            val_probs, val_future_returns, val_spread_costs,
            min_trades=50, weekly_sample_count=weekly_samples * len(symbol_keys),
            tail_focus_weight=0.3,
            target_weekly_trades=TARGET_WEEKLY_TRADES,
            target_trade_weight=0.15,
        )

        # 12. Build metadata
        metadata = {
            "symbols": symbol_keys,
            "days": self.days,
            "total_samples": n_total,
            "train_samples": split,
            "val_samples": n_total - split,
            "warm_start": use_warm_start,
            "epochs": epochs,
            "learning_rate": lr,
            "val_accuracy": summary.get("val_accuracy", 0),
            "val_win_rate": summary.get("val_win_rate", 0),
            "val_trade_rate": summary.get("val_trade_rate", 0),
            "symbol_thresholds": thresholds,
            "action_modes": action_modes,
            "symbol_diagnostics": diagnostics,
            "global_threshold": global_threshold,
            "symbol_profitability": symbol_profitability,
            "live_profile": {k: v for k, v in live_profile.items()},
            "profitability_quality": overall_quality,
            "retrain_timestamp": datetime.now().isoformat(),
        }

        # 13. Save candidate
        save_model(
            model, str(self.candidate_path), feature_dim,
            symbol_to_index, mean, std, metadata, push_profiles,
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Retrain complete in {duration:.0f}s. Candidate saved to {self.candidate_path}")

        result.update({
            "success": True,
            "candidate_path": str(self.candidate_path),
            "duration_seconds": duration,
            "feature_dim": feature_dim,
            "candidate_metrics": {
                "expectancy": overall_quality.get("expectancy", 0),
                "profit_factor": overall_quality.get("profit_factor", 0),
                "win_rate": overall_quality.get("win_rate", 0),
                "trade_rate": overall_quality.get("trade_rate", 0),
                "weekly_expected_return": overall_quality.get("weekly_expected_return", 0),
            },
            "symbols_enabled": sum(
                1 for v in live_profile.values() if v.get("enabled", False)
            ),
            "symbols_total": len(live_profile),
        })
        return result

    # ------------------------------------------------------------------
    # Warm-start training (loads previous weights, fine-tunes)
    # ------------------------------------------------------------------

    def _warm_start_train(
        self,
        prev_ckpt: Dict,
        feature_dim: int,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        epochs: int = 60,
        lr: float = 0.0003,
        batch_size: int = 256,
    ):
        """Load previous weights into a fresh model, then fine-tune."""
        model = SimpleNeuralNetwork(feature_dim)
        model.load_state_dict(prev_ckpt["model_state_dict"])
        logger.info("Loaded previous model weights for warm-start")

        class_weights = compute_class_weights(train_labels, mode="sqrt_balanced")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        x_train = torch.tensor(train_features, dtype=torch.float32)
        y_train = torch.tensor(train_labels, dtype=torch.long)
        x_val = torch.tensor(val_features, dtype=torch.float32)
        y_val = torch.tensor(val_labels, dtype=torch.long)

        best_state = None
        best_val_loss = float("inf")
        patience = 15
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(x_train))
            running_loss = 0.0

            for start in range(0, len(x_train), batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                logits = model(x_train[idx])
                loss = criterion(logits, y_train[idx])
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(idx)

            model.eval()
            with torch.no_grad():
                val_logits = model(x_val)
                val_loss = float(criterion(val_logits, y_val).item())
                val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
                val_metrics = evaluate_predictions(val_probs, val_labels)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"  WarmStart Epoch {epoch:3d} | val_loss={val_loss:.4f} "
                    f"acc={val_metrics.accuracy:.3f} wr={val_metrics.win_rate:.3f}"
                )

            if epochs_no_improve >= patience:
                logger.info(f"  Early stop at epoch {epoch} (no improvement for {patience})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final eval summary
        model.eval()
        with torch.no_grad():
            final_probs = torch.softmax(model(x_val), dim=1).cpu().numpy()
            final_metrics = evaluate_predictions(final_probs, val_labels)

        summary = {
            "val_accuracy": final_metrics.accuracy,
            "val_win_rate": final_metrics.win_rate,
            "val_trade_rate": final_metrics.trade_rate,
            "val_trade_count": final_metrics.trade_count,
        }
        return model, summary

    # ------------------------------------------------------------------
    # Get current model metrics (for comparison)
    # ------------------------------------------------------------------

    def get_current_model_metrics(self) -> Dict:
        """Extract metrics from the currently deployed model checkpoint."""
        ckpt = self._load_previous_checkpoint()
        if ckpt is None:
            return {}
        meta = ckpt.get("metadata", {})
        quality = meta.get("profitability_quality", {})
        return {
            "expectancy": quality.get("expectancy", 0),
            "profit_factor": quality.get("profit_factor", 0),
            "win_rate": quality.get("win_rate", 0),
            "trade_rate": quality.get("trade_rate", 0),
            "weekly_expected_return": quality.get("weekly_expected_return", 0),
        }
