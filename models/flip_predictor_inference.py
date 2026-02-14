"""
FlipPredictor Inference Engine — Live predictions on H4 bar close.

Loads trained model, maintains ring buffer per symbol, outputs predictions
on each new H4 bar.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional
from dataclasses import dataclass

from flip_predictor_model import FlipPredictorGRU
from flip_predictor_features import (
    extract_sequence, compute_avg_trend_durations,
    FEATURE_DIM_TOTAL, SEQ_LEN, SYMBOLS, SYMBOL_TO_INDEX,
)
from app.zeropoint_signal import compute_zeropoint_state
from push_structure_analyzer import SymbolPushProfile

MODEL_PATH = os.path.join(os.path.dirname(__file__), "flip_predictor_model.pth")


@dataclass
class FlipPrediction:
    """Prediction output for a single symbol."""
    symbol: str
    bars_to_flip: float           # estimated bars until next flip
    imminence_class: int          # 0=imminent, 1=approaching, 2=not_soon
    imminence_probs: list         # [P(imminent), P(approaching), P(not_soon)]
    move_magnitude_pips: float    # expected pips of next move
    trend_continues_prob: float   # P(trend continues 3+ bars)

    @property
    def flip_imminent(self) -> bool:
        return self.imminence_class == 0 and self.imminence_probs[0] > 0.40

    @property
    def flip_approaching(self) -> bool:
        return self.imminence_class <= 1 and (self.imminence_probs[0] + self.imminence_probs[1]) > 0.55

    @property
    def trend_exhausted(self) -> bool:
        return self.trend_continues_prob < 0.30

    @property
    def strong_move_expected(self) -> bool:
        return abs(self.move_magnitude_pips) > 100

    def summary(self) -> str:
        imm_names = ["IMMINENT", "APPROACHING", "NOT_SOON"]
        return (f"{self.symbol}: flip in ~{self.bars_to_flip:.0f} bars "
                f"({imm_names[self.imminence_class]} "
                f"P={self.imminence_probs[self.imminence_class]:.0%}) | "
                f"move={self.move_magnitude_pips:+.0f} pips | "
                f"trend_continues={self.trend_continues_prob:.0%}")


class FlipPredictorEngine:
    """Live inference engine for FlipPredictor."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = None
        self.checkpoint = None
        self.feat_mean = None
        self.feat_std = None
        self.push_profiles: Dict[str, SymbolPushProfile] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

        if os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, path: str):
        """Load trained FlipPredictor model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.checkpoint = checkpoint

        input_dim = checkpoint.get("input_dim", FEATURE_DIM_TOTAL)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        num_layers = checkpoint.get("num_layers", 2)

        self.model = FlipPredictorGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,  # no dropout at inference
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Feature normalization
        num_norm = input_dim - len(SYMBOLS)
        self.feat_mean = np.array(checkpoint["feature_mean"][:num_norm], dtype=np.float32)
        self.feat_std = np.array(checkpoint["feature_std"][:num_norm], dtype=np.float32)

        # Rebuild push profiles
        for sym, prof_dict in checkpoint.get("push_profiles", {}).items():
            rev_prob = {int(k): v for k, v in prof_dict.get("reversal_prob_by_push", {}).items()}
            self.push_profiles[sym] = SymbolPushProfile(
                symbol=prof_dict["symbol"],
                median_pips_per_push=prof_dict["median_pips_per_push"],
                mean_pips_per_push=prof_dict["mean_pips_per_push"],
                std_pips_per_push=prof_dict["std_pips_per_push"],
                exhaustion_push_count=prof_dict["exhaustion_push_count"],
                reversal_prob_by_push=rev_prob,
                min_swing_pips=prof_dict["min_swing_pips"],
                sample_count=prof_dict["sample_count"],
                pullback_ratio_mean=prof_dict["pullback_ratio_mean"],
                pullback_ratio_std=prof_dict["pullback_ratio_std"],
            )

        meta = checkpoint.get("metadata", {})
        self._loaded = True
        print(f"FlipPredictor loaded: {input_dim}-dim, {hidden_dim} hidden, "
              f"val_loss={meta.get('best_val_loss', '?'):.4f}, "
              f"timing_MAE={meta.get('timing_mae', '?')}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, h4_df: pd.DataFrame, symbol: str, point: float) -> Optional[FlipPrediction]:
        """Generate prediction for a symbol given its H4 OHLCV data.

        Args:
            h4_df:  DataFrame with OHLCV columns (200+ bars recommended)
            symbol: e.g. "EURUSD"
            point:  MT5 point value

        Returns:
            FlipPrediction or None if insufficient data
        """
        if not self._loaded or self.model is None:
            return None

        if symbol not in SYMBOL_TO_INDEX:
            return None

        # Compute ZeroPoint state
        zp = compute_zeropoint_state(h4_df)
        if zp is None or len(zp) < SEQ_LEN + 60:
            return None

        # Compute avg trend duration
        avg_trend = compute_avg_trend_durations(zp)

        # Get push profile
        profile = self.push_profiles.get(symbol, None)

        # Extract sequence from last bar
        end_idx = len(h4_df) - 1
        seq = extract_sequence(
            h4_df, zp, end_idx, symbol,
            push_profile=profile,
            point=point,
            avg_trend_duration=avg_trend,
        )
        if seq is None:
            return None

        # Normalize (non-one-hot features only)
        num_norm = FEATURE_DIM_TOTAL - len(SYMBOLS)
        seq_norm = seq.copy()
        seq_norm[:, :num_norm] = (seq_norm[:, :num_norm] - self.feat_mean) / self.feat_std

        # Inference
        with torch.no_grad():
            x = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
            preds = self.model(x)

            bars_to_flip = preds["timing"].item()
            imm_logits = preds["imminence"][0]
            imm_probs = torch.softmax(imm_logits, dim=0).cpu().numpy().tolist()
            imm_class = int(torch.argmax(imm_logits).item())
            magnitude = preds["magnitude"].item()
            cont_prob = torch.sigmoid(preds["continuation"]).item()

        return FlipPrediction(
            symbol=symbol,
            bars_to_flip=max(0, bars_to_flip),
            imminence_class=imm_class,
            imminence_probs=imm_probs,
            move_magnitude_pips=magnitude,
            trend_continues_prob=cont_prob,
        )

    def predict_all(self, h4_data: Dict[str, dict]) -> Dict[str, FlipPrediction]:
        """Predict for all symbols at once.

        Args:
            h4_data: {symbol: {"df": DataFrame, "point": float}}

        Returns:
            {symbol: FlipPrediction}
        """
        results = {}
        for sym, data in h4_data.items():
            pred = self.predict(data["df"], sym, data["point"])
            if pred is not None:
                results[sym] = pred
        return results
