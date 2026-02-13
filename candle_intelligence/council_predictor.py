"""
Learned Council Predictor — Binary classifier for ZP trade outcomes.

Replaces hand-coded heuristic agents with a trained model that learns
which ZP signals are likely to produce profitable trades.

Architecture:
  35-dim feature vector → MLP (32→16→1) → P(profitable)
  Bayesian dropout: 20 forward passes at inference for uncertainty

Feature groups:
  A. ZP Signal characteristics (6)
  B. Price structure from H1 bars (7)
  C. Momentum & volume (5)
  D. Multi-TF ZP alignment (4)
  E. Account state (4)
  F. Symbol one-hot (9)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from candle_intelligence.agent_base import TradeContext

logger = logging.getLogger(__name__)

# Default symbol order (deterministic, matches existing trainers)
DEFAULT_SYMBOLS = [
    'AUDUSD', 'BTCUSD', 'EURJPY', 'EURUSD',
    'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDJPY',
]
NUM_SYMBOLS = len(DEFAULT_SYMBOLS)

# Feature dimensions
# V4: 12 curated numeric features + symbol one-hot
# Predicts P(TP1 hit) — maps directly to trade profitability
# since 3-TP partial system banks profit at TP1
NUM_NUMERIC_FEATURES = 12
FEATURE_DIM = NUM_NUMERIC_FEATURES + NUM_SYMBOLS  # 21


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CouncilPredictor(nn.Module):
    """Binary classifier: P(ZP trade hits TP1).

    V4: Moderate capacity, 0.35 dropout for Bayesian uncertainty.
    Predicts TP1-hit probability (57% base rate) which maps well
    to profitability since 3-TP partials bank profit at TP1.
    """

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit (apply sigmoid externally)."""
        return self.network(x)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    direction: str,
    entry_price: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    tp3: float,
    atr_value: float,
    is_fresh: bool,
    bars_since_flip: int,
    m15_confirmed: bool,
    confidence: float,
    h1_bars,           # DataFrame or None
    m15_bars=None,     # DataFrame or None
    h4_bars=None,      # DataFrame or None
    zp_h1=None,        # DataFrame or None
    zp_m15=None,       # DataFrame or None
    zp_h4=None,        # DataFrame or None
    daily_pnl: float = 0.0,
    weekly_pnl: float = 0.0,
    balance: float = 200.0,
    consecutive_losses: int = 0,
    open_position_count: int = 0,
    symbol: str = '',
    symbol_to_index: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Extract 21-dim feature vector for council predictor V4.

    V4: 12 curated numeric features + 9 symbol one-hot.
    Predicts P(TP1 hit) — the best proxy for trade profitability
    since the 3-TP partial system banks 1/3 at TP1.

    Features selected from V1/V2 permutation importance (top performers
    across both runs):
      - Signal quality: body ratio, engulfing, pin bar, confidence
      - Trend context: MA fan, SMA50 distance, consecutive bars
      - Timing: freshness, bars_since_flip
      - Multi-TF: M15 confirmed, H4 ZP agreement
      - Structure: SL distance in ATR

    Returns np.float32 array of shape (21,).
    """
    features = np.zeros(NUM_NUMERIC_FEATURES, dtype=np.float32)

    is_buy = direction == 'BUY'
    atr = atr_value if atr_value > 1e-12 else 1.0
    sl_dist = abs(entry_price - stop_loss)

    # --- Trend Context ---
    # [0] MA fan score: SMA5>SMA20>SMA50 alignment for trade direction
    if h1_bars is not None and len(h1_bars) >= 50:
        close = h1_bars['close'].values.astype(float)
        sma5 = np.mean(close[-5:])
        sma20 = np.mean(close[-20:])
        sma50 = np.mean(close[-50:])
        if is_buy:
            fan = float(sma5 > sma20) + float(sma20 > sma50)
        else:
            fan = float(sma5 < sma20) + float(sma20 < sma50)
        features[0] = fan / 2.0
    else:
        features[0] = 0.5

    # [1] Distance from SMA50 in ATR (trend overshoot/undershoot)
    if h1_bars is not None and len(h1_bars) >= 50:
        close = h1_bars['close'].values.astype(float)
        sma50 = np.mean(close[-50:])
        dist = (close[-1] - sma50) / atr
        # Align to direction: positive = price is in our favor
        if not is_buy:
            dist = -dist
        features[1] = np.clip(dist, -5, 5)
    else:
        features[1] = 0.0

    # [2] Consecutive bars in trade direction (momentum)
    if h1_bars is not None and len(h1_bars) >= 10:
        close = h1_bars['close'].values.astype(float)
        opn = h1_bars['open'].values.astype(float)
        bodies = close - opn
        consecutive = 0
        for i in range(len(bodies) - 1, -1, -1):
            if (is_buy and bodies[i] > 0) or (not is_buy and bodies[i] < 0):
                consecutive += 1
            else:
                break
        features[2] = np.clip(consecutive / 10.0, 0, 2)

    # --- Signal Quality ---
    # [3] Signal bar body ratio (direction-aligned)
    if h1_bars is not None and len(h1_bars) >= 1:
        close = h1_bars['close'].values.astype(float)
        opn = h1_bars['open'].values.astype(float)
        high = h1_bars['high'].values.astype(float)
        low = h1_bars['low'].values.astype(float)
        bar_rng = high[-1] - low[-1]
        body_ratio = (close[-1] - opn[-1]) / bar_rng if bar_rng > 1e-12 else 0.0
        if not is_buy:
            body_ratio = -body_ratio
        features[3] = np.clip(body_ratio, -1, 1)

    # [4] Engulfing pattern (direction-aligned)
    if h1_bars is not None and len(h1_bars) >= 3:
        close = h1_bars['close'].values.astype(float)
        opn = h1_bars['open'].values.astype(float)
        body = close[-1] - opn[-1]
        prev_body_signed = close[-2] - opn[-2]
        curr_body = abs(body)
        prev_body = abs(prev_body_signed)
        engulfing = 0.0
        if curr_body > prev_body * 1.2:
            if (is_buy and body > 0 and prev_body_signed < 0):
                engulfing = 1.0
            elif (not is_buy and body < 0 and prev_body_signed > 0):
                engulfing = 1.0
        features[4] = engulfing

    # [5] Pin bar: long wick on rejection side
    if h1_bars is not None and len(h1_bars) >= 1:
        close = h1_bars['close'].values.astype(float)
        opn = h1_bars['open'].values.astype(float)
        high = h1_bars['high'].values.astype(float)
        low = h1_bars['low'].values.astype(float)
        bar_rng = high[-1] - low[-1]
        upper_wick = high[-1] - max(close[-1], opn[-1])
        lower_wick = min(close[-1], opn[-1]) - low[-1]
        if is_buy:
            features[5] = np.clip(lower_wick / bar_rng if bar_rng > 1e-12 else 0.0, 0, 1)
        else:
            features[5] = np.clip(upper_wick / bar_rng if bar_rng > 1e-12 else 0.0, 0, 1)

    # [6] Confidence score (composite from ZP logic)
    features[6] = np.clip(confidence, 0.0, 1.0)

    # --- Timing ---
    # [7] Freshness (binary)
    features[7] = 1.0 if is_fresh else 0.0

    # [8] Bars since flip (normalized)
    features[8] = np.clip(bars_since_flip / 20.0, 0, 5)

    # --- Multi-TF Confirmation ---
    # [9] M15 ZP confirmation (binary)
    features[9] = 1.0 if m15_confirmed else 0.0

    # [10] H4 ZP agreement
    if zp_h4 is not None and len(zp_h4) > 0:
        h4_pos = int(zp_h4.iloc[-1].get('pos', 0))
        if h4_pos == 0:
            features[10] = 0.0
        elif (is_buy and h4_pos == 1) or (not is_buy and h4_pos == -1):
            features[10] = 1.0
        else:
            features[10] = -1.0
    else:
        features[10] = 0.0

    # --- Structure ---
    # [11] SL distance in ATR (wider SL = more room, but also more risk)
    features[11] = np.clip(sl_dist / atr, 0, 10)

    # --- Symbol One-Hot (9 features) ---
    sym_clean = symbol.upper().replace('.', '').replace('#', '')
    if symbol_to_index is None:
        symbol_to_index = {s: i for i, s in enumerate(DEFAULT_SYMBOLS)}
    one_hot = np.zeros(NUM_SYMBOLS, dtype=np.float32)
    idx = symbol_to_index.get(sym_clean, -1)
    if idx >= 0 and idx < NUM_SYMBOLS:
        one_hot[idx] = 1.0

    return np.concatenate([features, one_hot])


def extract_features_from_context(
    ctx: TradeContext,
    is_fresh: bool = True,
    bars_since_flip: int = 1,
    m15_confirmed: bool = False,
    confidence: float = 0.65,
    symbol_to_index: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Convenience wrapper: extract features from a TradeContext object."""
    # Count consecutive losses from recent_trades
    consecutive_losses = 0
    if ctx.recent_trades:
        for trade in reversed(ctx.recent_trades):
            pnl = trade.get('pnl', 0) if isinstance(trade, dict) else getattr(trade, 'pnl', 0)
            if pnl < 0:
                consecutive_losses += 1
            else:
                break

    return extract_features(
        direction=ctx.direction,
        entry_price=ctx.entry_price,
        stop_loss=ctx.stop_loss,
        tp1=ctx.tp1, tp2=ctx.tp2, tp3=ctx.tp3,
        atr_value=ctx.atr_value,
        is_fresh=is_fresh,
        bars_since_flip=bars_since_flip,
        m15_confirmed=m15_confirmed,
        confidence=confidence,
        h1_bars=ctx.h1_bars,
        m15_bars=ctx.m15_bars,
        h4_bars=ctx.h4_bars,
        zp_h1=ctx.zp_h1,
        zp_m15=ctx.zp_m15,
        zp_h4=ctx.zp_h4,
        daily_pnl=ctx.daily_pnl,
        weekly_pnl=ctx.weekly_pnl,
        balance=ctx.balance,
        consecutive_losses=consecutive_losses,
        open_position_count=len(ctx.open_positions),
        symbol=ctx.symbol,
        symbol_to_index=symbol_to_index,
    )


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class CouncilPredictorEngine:
    """Loads trained council predictor and provides P(win) predictions.

    Uses Bayesian dropout (dropout ON at inference) with multiple
    forward passes to estimate uncertainty.
    """

    def __init__(self, model_path: str = 'council_predictor_model.pth'):
        self.model: Optional[CouncilPredictor] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.symbol_to_index: Dict[str, int] = {
            s: i for i, s in enumerate(DEFAULT_SYMBOLS)
        }
        self.loaded = False
        self.n_training_samples = 0

        if os.path.exists(model_path):
            self._load(model_path)

    def _load(self, path: str):
        """Load model checkpoint."""
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            feat_dim = ckpt.get('feature_dim', FEATURE_DIM)

            self.model = CouncilPredictor(input_dim=feat_dim)
            self.model.load_state_dict(ckpt['model_state_dict'])
            # Keep model in train mode for Bayesian dropout
            self.model.train()

            if 'feature_mean' in ckpt:
                self.feature_mean = np.array(ckpt['feature_mean'], dtype=np.float32)
            if 'feature_std' in ckpt:
                self.feature_std = np.array(ckpt['feature_std'], dtype=np.float32)
            if 'symbol_to_index' in ckpt:
                self.symbol_to_index = ckpt['symbol_to_index']

            meta = ckpt.get('metadata', {})
            self.n_training_samples = meta.get('n_training_samples', 0)

            self.loaded = True
            logger.info(
                f"Council predictor loaded: dim={feat_dim}, "
                f"samples={self.n_training_samples}"
            )
        except Exception as e:
            logger.warning(f"Failed to load council predictor: {e}")
            self.loaded = False

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalization using training statistics."""
        if self.feature_mean is not None and self.feature_std is not None:
            std = np.where(self.feature_std > 1e-8, self.feature_std, 1.0)
            return (features - self.feature_mean) / std
        return features

    def predict(self, ctx: TradeContext,
                is_fresh: bool = True,
                bars_since_flip: int = 1,
                m15_confirmed: bool = False,
                confidence: float = 0.65,
                n_samples: int = 20) -> Tuple[float, float]:
        """Bayesian prediction: returns (mean_probability, uncertainty_std).

        Runs n_samples forward passes with dropout ON, averages predictions.
        High uncertainty → default to baseline (take the trade).
        """
        if not self.loaded or self.model is None:
            return 0.5, 1.0  # unknown → neutral with max uncertainty

        features = extract_features_from_context(
            ctx, is_fresh=is_fresh, bars_since_flip=bars_since_flip,
            m15_confirmed=m15_confirmed, confidence=confidence,
            symbol_to_index=self.symbol_to_index,
        )
        features = self._normalize(features)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Bayesian: multiple forward passes with dropout active
        self.model.train()  # ensure dropout is ON
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                logit = self.model(x)
                prob = torch.sigmoid(logit).item()
                preds.append(prob)

        mean_prob = float(np.mean(preds))
        std_prob = float(np.std(preds))
        return mean_prob, std_prob

    def get_lot_multiplier(self, ctx: TradeContext,
                           is_fresh: bool = True,
                           bars_since_flip: int = 1,
                           m15_confirmed: bool = False,
                           confidence: float = 0.65) -> Tuple[float, float, float]:
        """Convert prediction to lot multiplier.

        Returns: (lot_multiplier, mean_probability, uncertainty)

        Conservative: defaults to 1.0x unless model is confidently negative.
        Never skips — minimum lot is 0.50x.
        """
        if not self.loaded:
            return 1.0, 0.5, 1.0

        prob, uncertainty = self.predict(
            ctx, is_fresh=is_fresh, bars_since_flip=bars_since_flip,
            m15_confirmed=m15_confirmed, confidence=confidence,
        )

        # High uncertainty → take trade at full size (defer to baseline)
        if uncertainty > 0.15:
            return 1.0, prob, uncertainty

        # Probability-based lot sizing
        if prob >= 0.65:
            lot_mult = 1.0
        elif prob >= 0.50:
            lot_mult = 0.85
        elif prob >= 0.35:
            lot_mult = 0.65
        else:
            lot_mult = 0.50

        return lot_mult, prob, uncertainty
