"""
FlipPredictor — Multi-Head GRU for predicting ZeroPoint flips.

4 prediction heads:
  1. Flip Timing:       Bars until next flip (regression, 0-50)
  2. Flip Imminence:    3-class (imminent 1-2, approaching 3-6, not soon 6+)
  3. Move Magnitude:    Expected pips after flip (signed regression)
  4. Trend Continuation: P(trend lasts 3+ more bars) (binary)

Architecture: 2-layer GRU → shared projection → 4 heads (~200K params)
"""
import torch
import torch.nn as nn


class FlipPredictorGRU(nn.Module):
    """Multi-head GRU for H4 ZeroPoint flip prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection (stabilizes GRU training)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # GRU backbone
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Shared projection from GRU output
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head 1: Flip Timing (regression — bars until next flip)
        self.head_timing = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),      # output: raw bars (0-50)
        )

        # Head 2: Flip Imminence (3-class classification)
        self.head_imminence = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),      # output: logits for [imminent, approaching, not_soon]
        )

        # Head 3: Move Magnitude (signed pips regression)
        self.head_magnitude = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),      # output: signed pip movement
        )

        # Head 4: Trend Continuation (binary classification)
        self.head_continuation = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),      # output: logit for P(trend continues 3+ bars)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, input_dim) — sequence of H4 bar features

        Returns:
            dict with keys:
                timing:       (batch, 1) — bars until next flip
                imminence:    (batch, 3) — logits [imminent, approaching, not_soon]
                magnitude:    (batch, 1) — signed pips of next move
                continuation: (batch, 1) — logit for trend continuation
        """
        # Input projection
        proj = self.input_proj(x)        # (batch, seq, hidden)

        # GRU
        gru_out, _ = self.gru(proj)      # (batch, seq, hidden)

        # Take last timestep
        last = gru_out[:, -1, :]         # (batch, hidden)

        # Shared projection
        shared = self.shared(last)       # (batch, hidden)

        return {
            "timing":       self.head_timing(shared),        # (batch, 1)
            "imminence":    self.head_imminence(shared),      # (batch, 3)
            "magnitude":    self.head_magnitude(shared),      # (batch, 1)
            "continuation": self.head_continuation(shared),   # (batch, 1)
        }


class FlipPredictorLoss(nn.Module):
    """Combined multi-task loss for FlipPredictor."""

    def __init__(self, imminence_weights=None):
        super().__init__()
        self.timing_loss = nn.HuberLoss(delta=5.0)
        self.imminence_loss = nn.CrossEntropyLoss(
            weight=imminence_weights if imminence_weights is not None else None
        )
        self.magnitude_loss = nn.HuberLoss(delta=2.0)  # ATR multiples (0-10 range)
        self.continuation_loss = nn.BCEWithLogitsLoss()

        # Task weights (tunable)
        self.w_timing = 1.0
        self.w_imminence = 2.0       # prioritize this — most actionable
        self.w_magnitude = 0.5
        self.w_continuation = 1.0

    def forward(self, preds: dict, targets: dict):
        """
        Args:
            preds:   output of FlipPredictorGRU.forward()
            targets: dict with keys:
                bars_to_flip:  (batch,) float — bars until next flip
                imminence:     (batch,) long  — class index (0, 1, 2)
                magnitude:     (batch,) float — signed pips
                continuation:  (batch,) float — 0.0 or 1.0
        """
        l_timing = self.timing_loss(
            preds["timing"].squeeze(-1),
            targets["bars_to_flip"]
        )
        l_imminence = self.imminence_loss(
            preds["imminence"],
            targets["imminence"]
        )
        l_magnitude = self.magnitude_loss(
            preds["magnitude"].squeeze(-1),
            targets["magnitude"]
        )
        l_continuation = self.continuation_loss(
            preds["continuation"].squeeze(-1),
            targets["continuation"]
        )

        total = (self.w_timing * l_timing +
                 self.w_imminence * l_imminence +
                 self.w_magnitude * l_magnitude +
                 self.w_continuation * l_continuation)

        return {
            "total": total,
            "timing": l_timing,
            "imminence": l_imminence,
            "magnitude": l_magnitude,
            "continuation": l_continuation,
        }
