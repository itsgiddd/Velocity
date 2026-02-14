"""
FlipPredictor Trainer — Collects H4 data, creates labels, trains GRU, saves model.

Usage:
    python flip_predictor_trainer.py

Labels (per H4 bar, computed from forward-looking ZP flip data):
  - bars_to_flip:    distance to next ZP flip (0-50, capped)
  - imminence:       0=imminent(1-2 bars), 1=approaching(3-6), 2=not_soon(6+)
  - move_magnitude:  signed pips of move after next flip
  - trend_continues: does current direction hold for 3+ more bars? (0/1)
"""
import sys
import os

# Add project root to sys.path so subpackage imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import MetaTrader5 as mt5
from datetime import datetime

from models.flip_predictor_model import FlipPredictorGRU, FlipPredictorLoss
from models.flip_predictor_features import (
    extract_sequence, compute_avg_trend_durations,
    FEATURE_DIM_TOTAL, SEQ_LEN, SYMBOLS, SYMBOL_TO_INDEX,
)
from app.zeropoint_signal import compute_zeropoint_state
from models.push_structure_analyzer import PushStatisticsCollector

# ─── Config ───────────────────────────────────────────────────────────────────
H4_BARS = 5000           # ~833 days of H4 data
MAX_BARS_TO_FLIP = 50    # cap for timing regression
TRAIN_SPLIT = 0.80
EPOCHS = 150
BATCH_SIZE = 64
LR = 0.001
PATIENCE = 20            # early stopping
MODEL_PATH = "flip_predictor_model.pth"


# ─── Dataset ──────────────────────────────────────────────────────────────────
class FlipDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.bars_to_flip = torch.tensor(labels["bars_to_flip"], dtype=torch.float32)
        self.imminence = torch.tensor(labels["imminence"], dtype=torch.long)
        self.magnitude = torch.tensor(labels["magnitude"], dtype=torch.float32)
        self.continuation = torch.tensor(labels["continuation"], dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            {
                "bars_to_flip": self.bars_to_flip[idx],
                "imminence": self.imminence[idx],
                "magnitude": self.magnitude[idx],
                "continuation": self.continuation[idx],
            },
        )


# ─── Label Creation ──────────────────────────────────────────────────────────
def create_labels(zp_h4: pd.DataFrame, point: float):
    """Create forward-looking labels for every H4 bar.

    Returns dict of arrays aligned to zp_h4 index.
    Magnitude is normalized to ATR multiples (not raw pips) for stable training.
    """
    pos = zp_h4["pos"].values
    close = zp_h4["close"].values
    high = zp_h4["high"].values
    low = zp_h4["low"].values
    atr = zp_h4["atr"].values
    n = len(pos)

    bars_to_flip = np.full(n, MAX_BARS_TO_FLIP, dtype=np.float32)
    imminence = np.full(n, 2, dtype=np.int64)      # default: not_soon
    magnitude = np.zeros(n, dtype=np.float32)       # in ATR multiples
    continuation = np.ones(n, dtype=np.float32)     # default: trend continues

    # Find all flip indices
    flip_indices = []
    for i in range(1, n):
        if pos[i] != pos[i - 1] and pos[i] != 0:
            flip_indices.append(i)

    # For each bar, find distance to NEXT flip
    flip_ptr = 0
    for i in range(n):
        # Advance pointer past current bar
        while flip_ptr < len(flip_indices) and flip_indices[flip_ptr] <= i:
            flip_ptr += 1

        if flip_ptr < len(flip_indices):
            next_flip = flip_indices[flip_ptr]
            dist = next_flip - i
            bars_to_flip[i] = min(dist, MAX_BARS_TO_FLIP)

            # Imminence classes
            if dist <= 2:
                imminence[i] = 0   # imminent
            elif dist <= 6:
                imminence[i] = 1   # approaching
            else:
                imminence[i] = 2   # not soon

            # Move magnitude after the flip (in ATR multiples for normalization)
            flip_dir = pos[next_flip]  # direction AFTER flip
            flip_atr = atr[next_flip] if not np.isnan(atr[next_flip]) and atr[next_flip] > 0 else atr[i]
            look_end = min(next_flip + 20, n)
            if look_end > next_flip + 1 and flip_atr > 1e-12:
                if flip_dir == 1:  # BUY flip
                    best = np.max(high[next_flip + 1:look_end])
                    move_raw = best - close[next_flip]
                else:              # SELL flip
                    best = np.min(low[next_flip + 1:look_end])
                    move_raw = close[next_flip] - best
                # Normalize to ATR multiples (typical range: 0-5x ATR)
                magnitude[i] = np.clip(move_raw / flip_atr, -10.0, 10.0)

            # Trend continuation: does current direction hold for 3+ more bars?
            if dist <= 3:
                continuation[i] = 0.0   # trend breaks within 3 bars
            else:
                continuation[i] = 1.0   # trend continues

    return {
        "bars_to_flip": bars_to_flip,
        "imminence": imminence,
        "magnitude": magnitude,
        "continuation": continuation,
    }


# ─── Data Collection ─────────────────────────────────────────────────────────
def collect_training_data():
    """Collect H4 data from MT5 and build sequences + labels."""
    print("Connecting to MT5...")
    if not mt5.initialize():
        print("MT5 init failed!")
        sys.exit(1)

    all_sequences = []
    all_labels = {"bars_to_flip": [], "imminence": [], "magnitude": [], "continuation": []}
    push_profiles = {}

    for sym in SYMBOLS:
        info = mt5.symbol_info(sym)
        if info is None:
            print(f"  {sym}: not found, skipping")
            continue
        mt5.symbol_select(sym, True)
        point = info.point

        print(f"\n  {sym}: fetching {H4_BARS} H4 bars...")
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, H4_BARS)
        if rates is None or len(rates) < 200:
            print(f"  {sym}: insufficient data ({0 if rates is None else len(rates)} bars)")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        # Compute ZeroPoint state
        zp = compute_zeropoint_state(df)
        if zp is None or len(zp) < 100:
            print(f"  {sym}: ZP computation failed")
            continue

        # Compute push profile
        try:
            profile = PushStatisticsCollector.compute_symbol_push_stats(df, sym, point, order=5)
            push_profiles[sym] = profile
        except Exception as e:
            print(f"  {sym}: push profile failed ({e})")
            profile = None

        # Compute avg trend duration for this symbol
        avg_trend = compute_avg_trend_durations(zp)
        print(f"  {sym}: {len(df)} bars, avg trend={avg_trend:.0f} bars, "
              f"push profile={'OK' if profile else 'NONE'}")

        # Create labels
        labels = create_labels(zp, point)

        # Extract sequences
        sym_count = 0
        # Step by 1 bar for max training data (overlapping sequences)
        for end_idx in range(SEQ_LEN + 55, len(df) - 1):  # -1 to ensure labels exist
            seq = extract_sequence(
                df, zp, end_idx, sym,
                push_profile=profile,
                point=point,
                avg_trend_duration=avg_trend,
            )
            if seq is None:
                continue

            all_sequences.append(seq)
            all_labels["bars_to_flip"].append(labels["bars_to_flip"][end_idx])
            all_labels["imminence"].append(labels["imminence"][end_idx])
            all_labels["magnitude"].append(labels["magnitude"][end_idx])
            all_labels["continuation"].append(labels["continuation"][end_idx])
            sym_count += 1

        print(f"  {sym}: {sym_count} sequences extracted")

    mt5.shutdown()

    # Convert to arrays
    all_sequences = np.array(all_sequences, dtype=np.float32)
    for k in all_labels:
        all_labels[k] = np.array(all_labels[k])

    print(f"\nTotal sequences: {len(all_sequences)}")
    print(f"  Shape: {all_sequences.shape}")  # (N, SEQ_LEN, 45)

    # Label distribution
    imm = all_labels["imminence"]
    print(f"  Imminence: imminent={np.sum(imm==0)} ({np.mean(imm==0)*100:.1f}%), "
          f"approaching={np.sum(imm==1)} ({np.mean(imm==1)*100:.1f}%), "
          f"not_soon={np.sum(imm==2)} ({np.mean(imm==2)*100:.1f}%)")
    cont = all_labels["continuation"]
    print(f"  Continuation: continues={np.sum(cont==1.0)} ({np.mean(cont)*100:.1f}%), "
          f"breaks={np.sum(cont==0.0)} ({(1-np.mean(cont))*100:.1f}%)")

    return all_sequences, all_labels, push_profiles


# ─── Feature Normalization ───────────────────────────────────────────────────
def normalize_features(train_seq, val_seq):
    """Compute mean/std on training data and normalize both sets.

    Only normalizes non-one-hot features (first 36 dims).
    """
    num_norm = FEATURE_DIM_TOTAL - len(SYMBOLS)  # 36

    # Flatten train sequences for stats
    flat = train_seq[:, :, :num_norm].reshape(-1, num_norm)
    feat_mean = np.mean(flat, axis=0)
    feat_std = np.std(flat, axis=0) + 1e-8

    # Normalize
    train_norm = train_seq.copy()
    train_norm[:, :, :num_norm] = (train_norm[:, :, :num_norm] - feat_mean) / feat_std

    val_norm = val_seq.copy()
    val_norm[:, :, :num_norm] = (val_norm[:, :, :num_norm] - feat_mean) / feat_std

    return train_norm, val_norm, feat_mean, feat_std


# ─── Training Loop ───────────────────────────────────────────────────────────
def train_model(sequences, labels, push_profiles):
    """Train the FlipPredictor GRU model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Train/val split (chronological — last 20% is validation)
    n = len(sequences)
    split = int(n * TRAIN_SPLIT)
    train_seq, val_seq = sequences[:split], sequences[split:]
    train_labels = {k: v[:split] for k, v in labels.items()}
    val_labels = {k: v[split:] for k, v in labels.items()}

    print(f"Train: {len(train_seq)} | Val: {len(val_seq)}")

    # Normalize features
    train_seq, val_seq, feat_mean, feat_std = normalize_features(train_seq, val_seq)

    # Datasets
    train_ds = FlipDataset(train_seq, train_labels)
    val_ds = FlipDataset(val_seq, val_labels)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Imminence class weights (handle imbalance)
    imm_counts = np.bincount(train_labels["imminence"].astype(int), minlength=3).astype(float)
    imm_weights = 1.0 / (imm_counts + 1.0)
    imm_weights = imm_weights / imm_weights.sum() * 3.0
    imm_weights_tensor = torch.tensor(imm_weights, dtype=torch.float32).to(device)
    print(f"Imminence weights: {imm_weights}")

    # Model
    model = FlipPredictorGRU(
        input_dim=FEATURE_DIM_TOTAL,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    criterion = FlipPredictorLoss(imminence_weights=imm_weights_tensor).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    print(f"\n{'Epoch':>5} {'Train':>10} {'Val':>10} {'Val-Timing':>12} {'Val-Imm':>10} "
          f"{'Val-Mag':>10} {'Val-Cont':>10} {'LR':>10}")
    print("-" * 90)

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_total = 0.0
        train_count = 0
        for batch_seq, batch_labels in train_dl:
            batch_seq = batch_seq.to(device)
            targets = {k: v.to(device) for k, v in batch_labels.items()}

            preds = model(batch_seq)
            losses = criterion(preds, targets)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_total += losses["total"].item() * batch_seq.size(0)
            train_count += batch_seq.size(0)

        train_loss = train_total / max(train_count, 1)

        # --- Validate ---
        model.eval()
        val_total = 0.0
        val_timing = 0.0
        val_imm = 0.0
        val_mag = 0.0
        val_cont = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_seq, batch_labels in val_dl:
                batch_seq = batch_seq.to(device)
                targets = {k: v.to(device) for k, v in batch_labels.items()}

                preds = model(batch_seq)
                losses = criterion(preds, targets)

                bs = batch_seq.size(0)
                val_total += losses["total"].item() * bs
                val_timing += losses["timing"].item() * bs
                val_imm += losses["imminence"].item() * bs
                val_mag += losses["magnitude"].item() * bs
                val_cont += losses["continuation"].item() * bs
                val_count += bs

        val_loss = val_total / max(val_count, 1)
        val_t = val_timing / max(val_count, 1)
        val_i = val_imm / max(val_count, 1)
        val_m = val_mag / max(val_count, 1)
        val_c = val_cont / max(val_count, 1)

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        print(f"{epoch:>5} {train_loss:>10.4f} {val_loss:>10.4f} {val_t:>12.4f} "
              f"{val_i:>10.4f} {val_m:>10.4f} {val_c:>10.4f} {lr:>10.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # Load best model for evaluation
    model.load_state_dict(best_state)
    model.eval()

    # ─── Final Evaluation ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("FINAL VALIDATION METRICS")
    print("=" * 90)

    all_preds_timing = []
    all_true_timing = []
    all_preds_imm = []
    all_true_imm = []
    all_preds_cont = []
    all_true_cont = []

    with torch.no_grad():
        for batch_seq, batch_labels in val_dl:
            batch_seq = batch_seq.to(device)
            preds = model(batch_seq)

            all_preds_timing.extend(preds["timing"].squeeze(-1).cpu().numpy())
            all_true_timing.extend(batch_labels["bars_to_flip"].numpy())

            pred_imm = torch.argmax(preds["imminence"], dim=1)
            all_preds_imm.extend(pred_imm.cpu().numpy())
            all_true_imm.extend(batch_labels["imminence"].numpy())

            pred_cont = (torch.sigmoid(preds["continuation"].squeeze(-1)) > 0.5).float()
            all_preds_cont.extend(pred_cont.cpu().numpy())
            all_true_cont.extend(batch_labels["continuation"].numpy())

    all_preds_timing = np.array(all_preds_timing)
    all_true_timing = np.array(all_true_timing)
    all_preds_imm = np.array(all_preds_imm)
    all_true_imm = np.array(all_true_imm)
    all_preds_cont = np.array(all_preds_cont)
    all_true_cont = np.array(all_true_cont)

    # Timing MAE
    timing_mae = np.mean(np.abs(all_preds_timing - all_true_timing))
    print(f"  Timing MAE: {timing_mae:.2f} bars")

    # Imminence accuracy
    imm_acc = np.mean(all_preds_imm == all_true_imm) * 100
    print(f"  Imminence accuracy: {imm_acc:.1f}%")

    # Per-class imminence
    for cls, name in [(0, "imminent"), (1, "approaching"), (2, "not_soon")]:
        mask = all_true_imm == cls
        if mask.sum() > 0:
            cls_acc = np.mean(all_preds_imm[mask] == cls) * 100
            print(f"    {name}: {cls_acc:.1f}% ({mask.sum()} samples)")

    # Continuation accuracy
    cont_acc = np.mean(all_preds_cont == all_true_cont) * 100
    print(f"  Continuation accuracy: {cont_acc:.1f}%")

    # Imminence precision for "imminent" class (most actionable)
    pred_imminent = all_preds_imm == 0
    true_imminent = all_true_imm == 0
    if pred_imminent.sum() > 0:
        precision = np.sum(pred_imminent & true_imminent) / pred_imminent.sum() * 100
        print(f"  Imminent precision: {precision:.1f}% ({pred_imminent.sum()} predicted)")

    # ─── Save Checkpoint ──────────────────────────────────────────────────────
    checkpoint = {
        "model_state_dict": best_state,
        "input_dim": FEATURE_DIM_TOTAL,
        "hidden_dim": 128,
        "num_layers": 2,
        "seq_len": SEQ_LEN,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "symbol_to_index": SYMBOL_TO_INDEX,
        "metadata": {
            "trainer_type": "flip_predictor",
            "feature_dim_total": FEATURE_DIM_TOTAL,
            "seq_len": SEQ_LEN,
            "max_bars_to_flip": MAX_BARS_TO_FLIP,
            "epochs_trained": epoch,
            "best_val_loss": float(best_val_loss),
            "timing_mae": float(timing_mae),
            "imminence_accuracy": float(imm_acc),
            "continuation_accuracy": float(cont_acc),
            "train_samples": split,
            "val_samples": n - split,
            "trained_at": datetime.now().isoformat(),
        },
        "push_profiles": {},
    }

    # Save push profiles
    for sym, prof in push_profiles.items():
        checkpoint["push_profiles"][sym] = {
            "symbol": prof.symbol,
            "median_pips_per_push": prof.median_pips_per_push,
            "mean_pips_per_push": prof.mean_pips_per_push,
            "std_pips_per_push": prof.std_pips_per_push,
            "exhaustion_push_count": prof.exhaustion_push_count,
            "reversal_prob_by_push": {str(k): v for k, v in prof.reversal_prob_by_push.items()},
            "min_swing_pips": prof.min_swing_pips,
            "sample_count": prof.sample_count,
            "pullback_ratio_mean": prof.pullback_ratio_mean,
            "pullback_ratio_std": prof.pullback_ratio_std,
        }

    torch.save(checkpoint, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"  Feature dim: {FEATURE_DIM_TOTAL}")
    print(f"  Seq len: {SEQ_LEN}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    return model, checkpoint


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 90)
    print("FLIPPREDICTOR TRAINER")
    print("Predictive AI for H4 ZeroPoint Flip Forecasting")
    print("=" * 90)

    sequences, labels, push_profiles = collect_training_data()

    if len(sequences) < 100:
        print("Not enough training data!")
        sys.exit(1)

    model, checkpoint = train_model(sequences, labels, push_profiles)

    print("\nDone!")
