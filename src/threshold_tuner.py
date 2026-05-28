"""
Threshold Tuner -- finds optimal per-label classification thresholds.

For heavily imbalanced labels (e.g. threat at ~1:430 ratio), the
default 0.5 decision boundary causes the model to predict all-negative.
This module sweeps thresholds from 0.05 to 0.95 per label and selects
the value that maximises F1 on the validation set.

Usage (standalone)::

    python -m src.threshold_tuner

This will load the trained model, run inference on the validation set,
find optimal thresholds, and save them to models/thresholds.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

# Ensure project root is on sys.path so ``src.*`` imports work in Colab.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import Config, get_default_config
from src.data_preprocessing import get_data_loaders
from src.model import ToxicityClassifier


# ===================================================================
# Core threshold search
# ===================================================================

def find_optimal_thresholds(
    y_true,
    y_proba,
    label_columns,
    search_min=0.05,
    search_max=0.95,
    step=0.05,
    min_precision=0.40,
):
    """Find the threshold that maximises F1 score for each label.

    A precision floor is enforced: candidate thresholds are only
    accepted if they achieve precision >= min_precision.  This prevents
    the tuner from selecting very low thresholds that inflate recall at
    the cost of unusable precision.

    Takes ground-truth binary labels (N, C), predicted probabilities
    (N, C), a list of label names, search boundaries, step size, and
    the minimum acceptable precision.

    Returns a dict mapping each label name to its optimal threshold.
    """
    candidates = np.arange(search_min, search_max + step / 2, step)
    thresholds = {}

    for i, col in enumerate(label_columns):
        best_f1 = -1.0
        best_t = 0.5  # fallback
        best_p = 0.0
        best_r = 0.0

        for t in candidates:
            preds = (y_proba[:, i] >= t).astype(int)
            p = precision_score(y_true[:, i], preds, zero_division=0)
            r = recall_score(y_true[:, i], preds, zero_division=0)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)

            # Only accept if precision meets the floor
            if p >= min_precision and f1 > best_f1:
                best_f1 = f1
                best_t = float(round(t, 2))
                best_p = p
                best_r = r

        thresholds[col] = best_t
        pos = int(y_true[:, i].sum())
        print(
            f"  {col:20s} → t={best_t:.2f}  "
            f"P={best_p:.3f}  R={best_r:.3f}  F1={best_f1:.4f}  "
            f"(pos={pos})"
        )

    return thresholds


# ===================================================================
# I/O helpers
# ===================================================================

def save_thresholds(thresholds, path):
    """Save thresholds dictionary to a JSON file.

    Takes a label-to-threshold mapping and a destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(thresholds, fh, indent=2)
    print(f"\n[threshold] Thresholds saved to {path}")


def load_thresholds(path):
    """Load thresholds from a JSON file.

    Takes a path to thresholds.json.  Returns a label-to-threshold
    mapping, or None if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===================================================================
# High-level API
# ===================================================================

def tune_thresholds_from_data(model, dataloader, device, label_columns, save_path):
    """Run model on a data loader, find optimal thresholds, and save.

    Takes a trained model (outputs raw logits), a validation DataLoader,
    the inference device, a list of label names, and a destination path
    for the resulting thresholds.json.

    Returns a dict mapping each label to its optimal threshold.
    """
    model.eval()
    all_proba = []
    all_true = []

    with torch.no_grad():
        for batch in dataloader:
            texts, labels = batch
            texts = texts.to(device)
            logits = model(texts)
            proba = torch.sigmoid(logits).cpu().numpy()
            all_proba.append(proba)
            all_true.append(labels.numpy())

    y_proba = np.concatenate(all_proba, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    print("\n" + "=" * 60)
    print("  Finding optimal thresholds (maximising F1 per label)")
    print("=" * 60 + "\n")

    thresholds = find_optimal_thresholds(y_true, y_proba, label_columns)
    save_thresholds(thresholds, save_path)

    return thresholds


# ===================================================================
# CLI entry-point
# ===================================================================

if __name__ == "__main__":
    config = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[threshold_tuner] Device: {device}")

    # Load data (we only need the validation set)
    print("[threshold_tuner] Loading data…")
    _, val_loader, test_loader, vocab = get_data_loaders(config)

    # Load trained model
    model_path = config.model_path / "toxicity_model.pth"
    actual_vocab_size = len(vocab) + 2
    model = ToxicityClassifier(
        vocab_size=actual_vocab_size,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        num_labels=config.num_labels,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    print(f"[threshold_tuner] Model loaded from {model_path}")

    # Tune on validation set
    thresholds_path = config.model_path / "thresholds.json"
    thresholds = tune_thresholds_from_data(
        model, val_loader, device, config.LABEL_COLUMNS, thresholds_path
    )

    # Show what the thresholds look like vs default 0.5
    print("\n" + "=" * 60)
    print("  Summary: Default (0.5) vs Optimised Thresholds")
    print("=" * 60)
    for col in config.LABEL_COLUMNS:
        t = thresholds[col]
        marker = " ←" if abs(t - 0.5) > 0.1 else ""
        print(f"  {col:20s} : 0.50 → {t:.2f}{marker}")
    print("=" * 60 + "\n")
