"""
Threshold Tuner - finds optimal per-label classification thresholds.

For heavily imbalanced labels (e.g. threat at 1:333 ratio), the
default 0.5 decision boundary causes massive false positives or false
negatives. This module sweeps thresholds per label and selects the
value that maximises F1 while enforcing a minimum precision floor.

Usage (standalone):
    python -m src.threshold_tuner
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so src.* imports work in Colab.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import Config, get_default_config
from src.data_preprocessing import get_data_loaders
from src.model import create_model


# ===================================================================
# Core threshold search
# ===================================================================

def find_optimal_thresholds(
    y_true,
    y_proba,
    label_columns,
    search_min=0.10,
    search_max=0.90,
    step=0.01,
    min_precision=0.40,
):
    """
    Find threshold that maximises F1 per label with a precision floor.
    Candidates below min_precision are rejected to prevent false-positive floods.
    """
    candidates = np.arange(search_min, search_max + step / 2, step)
    thresholds = {}

    for i, col in enumerate(label_columns):
        best_f1 = -1.0
        best_t = 0.5
        best_p = 0.0
        best_r = 0.0

        for t in candidates:
            preds = (y_proba[:, i] >= t).astype(int)
            p = precision_score(y_true[:, i], preds, zero_division=0)
            r = recall_score(y_true[:, i], preds, zero_division=0)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)

            if p >= min_precision and f1 > best_f1:
                best_f1 = f1
                best_t = float(round(t, 2))
                best_p = p
                best_r = r

        thresholds[col] = best_t
        pos = int(y_true[:, i].sum())
        logger.info(
            "  %-20s -> t=%.2f  P=%.3f  R=%.3f  F1=%.4f  (pos=%d)",
            col, best_t, best_p, best_r, best_f1, pos,
        )

    return thresholds


def find_precision_optimized_thresholds(
    y_true,
    y_proba,
    label_columns,
    min_recall=0.70,
    search_min=0.10,
    search_max=0.90,
    step=0.01,
):
    """Find threshold that maximises precision while maintaining recall."""
    candidates = np.arange(search_min, search_max + step / 2, step)
    thresholds = {}

    for i, col in enumerate(label_columns):
        best_p = -1.0
        best_t = 0.5
        best_r = 0.0
        best_f1 = 0.0

        for t in candidates:
            preds = (y_proba[:, i] >= t).astype(int)
            p = precision_score(y_true[:, i], preds, zero_division=0)
            r = recall_score(y_true[:, i], preds, zero_division=0)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)

            if r >= min_recall and p > best_p:
                best_p = p
                best_t = float(round(t, 2))
                best_r = r
                best_f1 = f1

        thresholds[col] = best_t
        pos = int(y_true[:, i].sum())
        logger.info(
            "  %-20s -> t=%.2f  P=%.3f  R=%.3f  F1=%.4f  (pos=%d)",
            col, best_t, best_p, best_r, best_f1, pos,
        )

    return thresholds


# ===================================================================
# I/O helpers
# ===================================================================

def save_thresholds(thresholds, path):
    """Save thresholds dictionary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(thresholds, fh, indent=2)
    logger.info("[threshold] Thresholds saved to %s", path)


def load_thresholds(path):
    """Load thresholds from a JSON file."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===================================================================
# High-level API
# ===================================================================

def tune_thresholds_from_data(
    model,
    dataloader,
    device,
    label_columns,
    save_path,
    step=0.01,
    min_precision=0.40,
):
    """Run model on DataLoader, find optimal thresholds, and save."""
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

    logger.info("")
    logger.info("=" * 60)
    logger.info("  F1-optimised thresholds (precision >= %.2f)", min_precision)
    logger.info("=" * 60)

    thresholds = find_optimal_thresholds(
        y_true, y_proba, label_columns,
        step=step, min_precision=min_precision,
    )
    save_thresholds(thresholds, save_path)

    # Also compute precision-optimised thresholds (informational)
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Precision-optimised thresholds (recall >= 0.70)")
    logger.info("=" * 60)

    prec_thresholds = find_precision_optimized_thresholds(
        y_true, y_proba, label_columns, min_recall=0.70, step=step,
    )
    prec_path = Path(save_path).parent / "thresholds_precision.json"
    save_thresholds(prec_thresholds, prec_path)

    return thresholds


# ===================================================================
# CLI entry-point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("[threshold_tuner] Device: %s", device)

    # Load data (we only need the validation set)
    logger.info("[threshold_tuner] Loading data...")
    _, val_loader, test_loader, vocab = get_data_loaders(config)

    # Load trained model
    model_path = config.model_path / "toxicity_model.pth"
    actual_vocab_size = len(vocab) + 2
    model = create_model(config, vocab_size=actual_vocab_size)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True),
    )
    model.to(device)
    logger.info("[threshold_tuner] Model loaded from %s", model_path)

    # Tune on validation set
    thresholds_path = config.model_path / "thresholds.json"
    thresholds = tune_thresholds_from_data(
        model, val_loader, device, config.LABEL_COLUMNS, thresholds_path,
        step=config.THRESHOLD_STEP,
        min_precision=config.MIN_PRECISION_FLOOR,
    )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Summary: Default (0.5) vs Optimised Thresholds")
    logger.info("=" * 60)
    for col in config.LABEL_COLUMNS:
        t = thresholds[col]
        marker = " <-" if abs(t - 0.5) > 0.1 else ""
        logger.info("  %-20s : 0.50 -> %.2f%s", col, t, marker)
    logger.info("=" * 60)
