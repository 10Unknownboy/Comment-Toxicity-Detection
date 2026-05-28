"""
Training script for the Comment Toxicity Detection model.

Designed to run on Google Colab with a GPU.  The script:
  1. Auto-detects CUDA and uses the best available device.
  2. Prepares data loaders (with optional sub-sampling).
  3. Trains a Bidirectional LSTM classifier with early stopping.
  4. Saves the best checkpoint, vocabulary, training history, and
     evaluation plots so they can be downloaded from Colab.

Usage (Colab cell)::

    !python -m src.train --epochs 5 --batch-size 256

Or simply::

    !python -m src.train
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Ensure project root is on sys.path so ``src.*`` imports work in Colab.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import Config, get_default_config
from src.data_preprocessing import get_data_loaders
from src.evaluate import (
    evaluate_model,
    generate_classification_report,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_training_history,
)
from src.model import ToxicityClassifier, FocalLoss
from src.threshold_tuner import tune_thresholds_from_data


# ===================================================================
# Reproducibility
# ===================================================================

def _set_seeds(seed):
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Takes a seed value to use across all random-number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ===================================================================
# Core training loop
# ===================================================================

def train_model(config):
    """Train the toxicity classifier end-to-end.

    Takes a Config object containing hyper-parameters and paths.
    Returns a training history dict with per-epoch train loss,
    validation loss, and validation ROC-AUC.
    """
    # ---- Reproducibility ----
    _set_seeds(config.RANDOM_SEED)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 60}\n")

    # ---- Data ----
    train_loader, val_loader, test_loader, vocab = get_data_loaders(config)

    # ---- Model ----
    actual_vocab_size = len(vocab) + 2  # +2 for PAD and UNK
    model = ToxicityClassifier(
        vocab_size=actual_vocab_size,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        num_labels=config.num_labels,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] Total parameters     : {total_params:,}")
    print(f"[model] Trainable parameters : {trainable_params:,}\n")

    # ---- Calculate class weights for imbalance ----
    print("[train] Calculating class weights from training data\u2026")
    train_labels_list = []
    for batch in train_loader:
        _, labels = batch
        train_labels_list.append(labels.numpy())
    train_labels_all = np.concatenate(train_labels_list, axis=0)

    pos_counts = train_labels_all.sum(axis=0)
    neg_counts = train_labels_all.shape[0] - pos_counts
    raw_weights = neg_counts / np.maximum(pos_counts, 1.0)
    pos_weights = np.sqrt(raw_weights)  # sqrt dampening to avoid overcorrection
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    print("[train] Positive weights per label (sqrt-dampened):")
    for col_name, rw, sw in zip(config.LABEL_COLUMNS, raw_weights, pos_weights):
        print(f"  {col_name:20s} : {rw:7.2f} → sqrt → {sw:.2f}")
    print()

    # ---- Loss & optimiser ----
    criterion = FocalLoss(alpha=pos_weights_tensor, gamma=2.0)
    lr = config.LEARNING_RATE * 0.5  # reduced LR for stability
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[train] Learning rate: {lr} (0.5× default)")

    # ---- Training history ----
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_roc_auc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    model_save_path = config.model_path / "toxicity_model.pth"

    print(f"{'=' * 60}")
    print("  Starting training …")
    print(f"{'=' * 60}\n")

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        # ---- Train phase ----
        model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.EPOCHS} [train]",
            leave=True,
            file=sys.stdout,
        )
        for batch in pbar:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(num_batches, 1)

        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_val_proba = []
        all_val_true = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{config.EPOCHS} [val]",
                leave=True,
                file=sys.stdout,
            ):
                texts, labels = batch
                texts, labels = texts.to(device), labels.to(device)

                outputs = model(texts)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_batches += 1

                proba = torch.sigmoid(outputs)  # logits -> probabilities
                all_val_proba.append(proba.cpu().numpy())
                all_val_true.append(labels.cpu().numpy())

        avg_val_loss = val_loss / max(val_batches, 1)

        # ROC-AUC
        y_val_true = np.concatenate(all_val_true, axis=0)
        y_val_proba = np.concatenate(all_val_proba, axis=0)
        try:
            val_roc_auc = roc_auc_score(y_val_true, y_val_proba, average="macro")
        except ValueError:
            val_roc_auc = 0.0

        elapsed = time.time() - epoch_start
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_roc_auc"].append(val_roc_auc)

        print(
            f"\n  Epoch {epoch}/{config.EPOCHS}  │  "
            f"Train Loss: {avg_train_loss:.4f}  │  "
            f"Val Loss: {avg_val_loss:.4f}  │  "
            f"Val AUC: {val_roc_auc:.4f}  │  "
            f"Time: {elapsed:.1f}s\n"
        )

        # ---- Early stopping & checkpoint ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  [OK] Best model saved to {model_save_path}\n")
        else:
            patience_counter += 1
            print(
                f"  [--] No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})\n"
            )
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("  [STOP] Early stopping triggered.\n")
                break

    # ---- Save training history ----
    history_path = config.model_path / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"[train] Training history saved to {history_path}")

    # ---- Training history plot ----
    history_plot_path = config.model_path / "training_history.png"
    plot_training_history(history, history_plot_path)

    # ---- Load best model & evaluate on test set ----
    print("\n" + "=" * 60)
    print("  Evaluating best model on test set …")
    print("=" * 60 + "\n")

    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    results = evaluate_model(model, test_loader, device, config.LABEL_COLUMNS)

    # Print metrics
    report = generate_classification_report(
        results["y_true"], results["y_pred"], config.LABEL_COLUMNS
    )
    print(report)

    # Print macro metrics summary
    macro = results["metrics"]["macro"]
    print(f"\n{'=' * 60}")
    print("  Macro Averages")
    print(f"{'=' * 60}")
    print(f"  ROC-AUC   : {macro['roc_auc']:.4f}")
    print(f"  Accuracy  : {macro['accuracy']:.4f}")
    print(f"  Precision : {macro['precision']:.4f}")
    print(f"  Recall    : {macro['recall']:.4f}")
    print(f"  F1 Score  : {macro['f1']:.4f}")
    print(f"{'=' * 60}\n")

    # ---- Evaluation plots ----
    cm_path = config.model_path / "confusion_matrices.png"
    roc_path = config.model_path / "roc_curves.png"
    plot_confusion_matrices(
        results["y_true"], results["y_pred"], config.LABEL_COLUMNS, cm_path
    )
    plot_roc_curves(
        results["y_true"], results["y_proba"], config.LABEL_COLUMNS, roc_path
    )

    # ---- Save evaluation metrics as JSON ----
    eval_metrics_path = config.model_path / "evaluation_results.json"
    serialisable_metrics = results["metrics"].copy()
    with open(eval_metrics_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable_metrics, fh, indent=2)
    print(f"[eval] Evaluation metrics saved to {eval_metrics_path}")

    # ---- Threshold tuning on validation set ----
    print("\n" + "=" * 60)
    print("  Tuning per-label decision thresholds on validation set …")
    print("=" * 60)

    thresholds_path = config.model_path / "thresholds.json"
    thresholds = tune_thresholds_from_data(
        model, val_loader, device, config.LABEL_COLUMNS, thresholds_path
    )

    # ---- Re-evaluate test set with optimised thresholds ----
    thresholds_list = [thresholds[col] for col in config.LABEL_COLUMNS]
    results_opt = evaluate_model(
        model, test_loader, device, config.LABEL_COLUMNS, thresholds=thresholds_list
    )
    macro_opt = results_opt["metrics"]["macro"]
    print(f"\n{'=' * 60}")
    print("  Test Metrics WITH optimised thresholds")
    print(f"{'=' * 60}")
    print(f"  ROC-AUC   : {macro_opt['roc_auc']:.4f}")
    print(f"  Precision : {macro_opt['precision']:.4f}")
    print(f"  Recall    : {macro_opt['recall']:.4f}")
    print(f"  F1 Score  : {macro_opt['f1']:.4f}")
    print(f"{'=' * 60}\n")

    # ---- Final instructions ----
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print("\n  Download these files from Colab:\n")
    print(f"    1. {model_save_path}")
    print(f"    2. {config.model_path / 'vocab.json'}")
    print(f"    3. {history_path}")
    print(f"    4. {history_plot_path}")
    print(f"    5. {cm_path}")
    print(f"    6. {roc_path}")
    print(f"    7. {eval_metrics_path}")
    print(f"    8. {thresholds_path}")
    print("\n  Place them in the 'models/' directory of your local project.")
    print("=" * 60 + "\n")

    return history


# ===================================================================
# CLI entry-point
# ===================================================================

def _parse_args():
    """Parse command-line arguments for training overrides.

    Returns parsed arguments with optional --epochs, --sample-size,
    and --batch-size overrides.
    """
    parser = argparse.ArgumentParser(
        description="Train the Toxicity Classifier (designed for Google Colab w/ GPU)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override TRAIN_SAMPLE_SIZE (use 0 for full dataset).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override mini-batch size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = get_default_config()

    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.sample_size is not None:
        cfg.TRAIN_SAMPLE_SIZE = args.sample_size if args.sample_size > 0 else None
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size

    train_model(cfg)
