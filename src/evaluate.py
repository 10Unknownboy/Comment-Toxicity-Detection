"""
Evaluation and visualisation utilities for Comment Toxicity Detection.

Provides functions for:
* Computing classification metrics (ROC-AUC, precision, recall, F1).
* Plotting confusion matrices, ROC curves, and training history.
* Generating formatted text reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for Colab & servers)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ===================================================================
# Model evaluation
# ===================================================================

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label_columns: list[str],
) -> dict[str, Any]:
    """Run inference on a dataloader and compute classification metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained toxicity classifier (already on ``device``).
    dataloader : DataLoader
        Evaluation data loader yielding ``(texts, labels)`` batches.
    device : torch.device
        Device on which to perform inference.
    label_columns : list[str]
        Names of the label columns (length must match model output).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - ``y_true``  – ground-truth labels ``(N, C)``
        - ``y_pred``  – binary predictions ``(N, C)`` (threshold 0.5)
        - ``y_proba`` – raw probabilities ``(N, C)``
        - ``metrics``  – nested dict with per-label and macro scores
    """
    model.eval()
    all_proba: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            texts, labels = batch
            texts = texts.to(device)
            proba = model(texts).cpu().numpy()
            all_proba.append(proba)
            all_true.append(labels.numpy())

    y_proba = np.concatenate(all_proba, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    y_pred = (y_proba >= 0.5).astype(int)

    # --- Per-label metrics ---
    metrics: dict[str, Any] = {"per_label": {}}
    for i, col in enumerate(label_columns):
        try:
            label_auc = roc_auc_score(y_true[:, i], y_proba[:, i])
        except ValueError:
            label_auc = float("nan")

        metrics["per_label"][col] = {
            "roc_auc": label_auc,
            "accuracy": accuracy_score(y_true[:, i], y_pred[:, i]),
            "precision": precision_score(
                y_true[:, i], y_pred[:, i], zero_division=0
            ),
            "recall": recall_score(
                y_true[:, i], y_pred[:, i], zero_division=0
            ),
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
        }

    # --- Macro averages ---
    try:
        macro_auc = roc_auc_score(y_true, y_proba, average="macro")
    except ValueError:
        macro_auc = float("nan")

    metrics["macro"] = {
        "roc_auc": macro_auc,
        "accuracy": accuracy_score(y_true.ravel(), y_pred.ravel()),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }


# ===================================================================
# Confusion matrices
# ===================================================================

def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_columns: list[str],
    save_path: str | Path,
) -> None:
    """Plot a 2×3 grid of per-label confusion matrices and save as PNG.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels ``(N, C)``.
    y_pred : np.ndarray
        Predicted binary labels ``(N, C)``.
    label_columns : list[str]
        Label names.
    save_path : str or Path
        Destination file path for the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (col, ax) in enumerate(zip(label_columns, axes)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(col.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])

        # Annotate cells
        for row in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                ax.text(
                    col_idx,
                    row,
                    f"{cm[row, col_idx]:,}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if cm[row, col_idx] > cm.max() / 2 else "black",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for j in range(len(label_columns), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices (per label)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Confusion matrices saved to {save_path}")


# ===================================================================
# ROC curves
# ===================================================================

def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_columns: list[str],
    save_path: str | Path,
) -> None:
    """Plot per-label ROC curves with AUC scores and save as PNG.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels ``(N, C)``.
    y_proba : np.ndarray
        Predicted probabilities ``(N, C)``.
    label_columns : list[str]
        Label names.
    save_path : str or Path
        Destination file path for the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 8))

    colours = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    for i, (col, colour) in enumerate(zip(label_columns, colours)):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            label_name = col.replace("_", " ").title()
            ax.plot(fpr, tpr, color=colour, lw=2, label=f"{label_name} (AUC = {roc_auc_val:.4f})")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5, label="Random Baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – Per Label", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] ROC curves saved to {save_path}")


# ===================================================================
# Training history
# ===================================================================

def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | Path,
) -> None:
    """Plot training / validation loss and ROC-AUC over epochs.

    Parameters
    ----------
    history : dict[str, list[float]]
        Must contain keys ``train_loss``, ``val_loss``, and ``val_roc_auc``.
    save_path : str or Path
        Destination file path for the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss ---
    ax1.plot(epochs, history["train_loss"], "o-", color="#FF6B6B", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "o-", color="#4ECDC4", label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("BCE Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # --- ROC-AUC ---
    ax2.plot(epochs, history["val_roc_auc"], "o-", color="#FFEAA7", label="Val ROC-AUC")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("ROC-AUC", fontsize=12)
    ax2.set_title("Validation ROC-AUC", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Training history plot saved to {save_path}")


# ===================================================================
# Text report
# ===================================================================

def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_columns: list[str],
) -> str:
    """Return a formatted multi-label classification report.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels ``(N, C)``.
    y_pred : np.ndarray
        Predicted binary labels ``(N, C)``.
    label_columns : list[str]
        Label names.

    Returns
    -------
    str
        Human-readable classification report.
    """
    header = "=" * 60 + "\n  Classification Report – Comment Toxicity Detection\n" + "=" * 60 + "\n"
    per_label_reports: list[str] = []

    for i, col in enumerate(label_columns):
        report = classification_report(
            y_true[:, i],
            y_pred[:, i],
            target_names=["Non-toxic", col.replace("_", " ").title()],
            zero_division=0,
        )
        per_label_reports.append(f"\n--- {col.replace('_', ' ').title()} ---\n{report}")

    return header + "\n".join(per_label_reports)
