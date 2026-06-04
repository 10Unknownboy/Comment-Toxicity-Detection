"""
Threshold Tuner — finds optimal per-label classification thresholds.

For heavily imbalanced labels (e.g. ``threat`` at 1:333 ratio), the
default 0.5 decision boundary causes massive false positives or false
negatives.  This module sweeps thresholds per label and selects the
value that maximises F1 while enforcing a minimum precision floor.

Usage (standalone)::

    python -m src.threshold_tuner
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so ``src.*`` imports work in Colab.
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
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_columns: list[str],
    search_min: float = 0.10,
    search_max: float = 0.90,
    step: float = 0.01,
    min_precision: float = 0.40,
) -> dict[str, float]:
    """Find threshold that maximises F1 per label with a precision floor.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels ``(N, C)``.
    y_proba : np.ndarray
        Predicted probabilities ``(N, C)``.
    label_columns : list[str]
        Label names (length C).
    search_min, search_max : float
        Boundaries of the threshold sweep.
    step : float
        Increment between candidate thresholds.
    min_precision : float
        Minimum acceptable precision.  Candidates below this are
        rejected to prevent false-positive floods.

    Returns
    -------
    dict[str, float]
        Label name → optimal threshold.
    """
    candidates = np.arange(search_min, search_max + step / 2, step)
    thresholds: dict[str, float] = {}

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
            "  %-20s → t=%.2f  P=%.3f  R=%.3f  F1=%.4f  (pos=%d)",
            col, best_t, best_p, best_r, best_f1, pos,
        )

    return thresholds


def find_precision_optimized_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_columns: list[str],
    min_recall: float = 0.70,
    search_min: float = 0.10,
    search_max: float = 0.90,
    step: float = 0.01,
) -> dict[str, float]:
    """Find threshold that maximises precision while maintaining recall.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels ``(N, C)``.
    y_proba : np.ndarray
        Predicted probabilities ``(N, C)``.
    label_columns : list[str]
        Label names.
    min_recall : float
        Minimum acceptable recall.
    search_min, search_max, step : float
        Sweep boundaries and step size.

    Returns
    -------
    dict[str, float]
        Label name → precision-optimised threshold.
    """
    candidates = np.arange(search_min, search_max + step / 2, step)
    thresholds: dict[str, float] = {}

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
            "  %-20s → t=%.2f  P=%.3f  R=%.3f  F1=%.4f  (pos=%d)",
            col, best_t, best_p, best_r, best_f1, pos,
        )

    return thresholds


# ===================================================================
# I/O helpers
# ===================================================================

def save_thresholds(thresholds: dict[str, float], path: str | Path) -> None:
    """Save thresholds dictionary to a JSON file.

    Parameters
    ----------
    thresholds : dict[str, float]
        Label → threshold mapping.
    path : str or Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(thresholds, fh, indent=2)
    logger.info("[threshold] Thresholds saved to %s", path)


def load_thresholds(path: str | Path) -> dict[str, float] | None:
    """Load thresholds from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to ``thresholds.json``.

    Returns
    -------
    dict[str, float] or None
        Label → threshold mapping, or ``None`` if file missing.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===================================================================
# High-level API
# ===================================================================

def tune_thresholds_from_data(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label_columns: list[str],
    save_path: str | Path,
    step: float = 0.01,
    min_precision: float = 0.40,
) -> dict[str, float]:
    """Run model on DataLoader, find optimal thresholds, and save.

    Parameters
    ----------
    model : nn.Module
        Trained model (outputs raw logits).
    dataloader : DataLoader
        Validation data loader.
    device : torch.device
        Inference device.
    label_columns : list[str]
        Label names.
    save_path : str or Path
        Where to save ``thresholds.json``.
    step : float
        Threshold sweep granularity.
    min_precision : float
        Minimum precision floor.

    Returns
    -------
    dict[str, float]
        Optimal thresholds per label.
    """
    model.eval()
    all_proba: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

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
    logger.info("  F1-optimised thresholds (precision ≥ %.2f)", min_precision)
    logger.info("=" * 60)

    thresholds = find_optimal_thresholds(
        y_true, y_proba, label_columns,
        step=step, min_precision=min_precision,
    )
    save_thresholds(thresholds, save_path)

    # Also compute precision-optimised thresholds (informational)
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Precision-optimised thresholds (recall ≥ 0.70)")
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
    logger.info("[threshold_tuner] Loading data…")
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
        marker = " ←" if abs(t - 0.5) > 0.1 else ""
        logger.info("  %-20s : 0.50 → %.2f%s", col, t, marker)
    logger.info("=" * 60)
