"""
Training script for the Comment Toxicity Detection model.

Designed to run on Google Colab with a GPU. The script:
  1. Auto-detects CUDA and uses the best available device.
  2. Prepares data loaders (with augmentation for rare classes).
  3. Trains a BiLSTM+Attention classifier with:
     - Weighted BCE / Focal loss for class imbalance
     - Label smoothing
     - Gradient clipping
     - LR scheduler (ReduceLROnPlateau)
     - Early stopping with min-delta
     - Mixed precision (auto-detected)
  4. Tunes per-label decision thresholds on the validation set.
  5. Saves all artefacts for download from Colab.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure project root is on sys.path so src.* imports work in Colab.
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
from src.loss import (
    apply_label_smoothing,
    compute_pos_weights,
    get_loss_function,
)
from src.model import create_model
from src.threshold_tuner import tune_thresholds_from_data

logger = logging.getLogger(__name__)


# ===================================================================
# Reproducibility
# ===================================================================

def _set_seeds(seed):
    """Set random seeds for Python, NumPy, and PyTorch."""
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
    """Train the toxicity classifier end-to-end."""
    _set_seeds(config.RANDOM_SEED)

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("  Device : %s", device)
    if device.type == "cuda":
        logger.info("  GPU    : %s", torch.cuda.get_device_name(0))
    logger.info("  AMP    : %s", "enabled" if config.USE_AMP else "disabled")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, vocab = get_data_loaders(config)

    # ── Model ─────────────────────────────────────────────────────
    actual_vocab_size = len(vocab) + 2  # +2 for PAD and UNK
    model = create_model(config, vocab_size=actual_vocab_size)
    model.to(device)

    # ── Class weights & loss ──────────────────────────────────────
    logger.info("[train] Computing class weights from training data ...")
    train_labels_list = []
    for batch in train_loader:
        _, labels_batch = batch
        train_labels_list.append(labels_batch.numpy())
    train_labels_all = np.concatenate(train_labels_list, axis=0)

    pos_weights = compute_pos_weights(
        train_labels_all, config.LABEL_COLUMNS, dampening="sqrt",
    )
    criterion = get_loss_function(config, pos_weights, device)

    # ── Optimiser ─────────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ── LR scheduler ──────────────────────────────────────────────
    scheduler = CosineAnnealingLR(
        optimiser,
        T_max=config.EPOCHS,
        eta_min=config.MIN_LR,
    )

    # ── Mixed precision ───────────────────────────────────────────
    scaler = GradScaler(enabled=config.USE_AMP)
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"

    # ── History & early stopping ──────────────────────────────────
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_roc_auc": [],
    }
    best_val_auc = 0.0
    patience_counter = 0
    model_save_path = config.model_path / "toxicity_model.pth"

    logger.info("=" * 60)
    logger.info("  Starting training (%d epochs) ...", config.EPOCHS)
    logger.info("=" * 60)

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        # ── Train phase ───────────────────────────────────────────
        model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.EPOCHS} [train]",
            leave=True,
        )
        for batch in pbar:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)

            # Label smoothing
            if config.LABEL_SMOOTHING > 0:
                labels = apply_label_smoothing(labels, config.LABEL_SMOOTHING)

            optimiser.zero_grad()

            with autocast(device_type=amp_device_type, enabled=config.USE_AMP):
                outputs = model(texts)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient clipping (unscale first for correct norm)
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.GRAD_CLIP_NORM,
            )

            scaler.step(optimiser)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(num_batches, 1)

        # ── Validation phase ──────────────────────────────────────
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
            ):
                texts, labels = batch
                texts, labels = texts.to(device), labels.to(device)

                with autocast(device_type=amp_device_type, enabled=config.USE_AMP):
                    outputs = model(texts)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_batches += 1

                proba = torch.sigmoid(outputs).cpu().numpy()
                all_val_proba.append(proba)
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

        current_lr = optimiser.param_groups[0]["lr"]
        logger.info(
            "  Epoch %d/%d  |  Train: %.4f  |  Val: %.4f  |  "
            "AUC: %.4f  |  LR: %.2e  |  %.1fs",
            epoch, config.EPOCHS,
            avg_train_loss, avg_val_loss, val_roc_auc,
            current_lr, elapsed,
        )

        # ── LR scheduler step ────────────────────────────────────
        scheduler.step()

        # ── Early stopping & checkpoint ───────────────────────────
        if val_roc_auc > best_val_auc + config.EARLY_STOPPING_MIN_DELTA:
            best_val_auc = val_roc_auc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info("  v Best model saved (val_roc_auc=%.4f)", val_roc_auc)
        else:
            patience_counter += 1
            logger.info(
                "  x No improvement (%d/%d)",
                patience_counter, config.EARLY_STOPPING_PATIENCE,
            )
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info("  ! Early stopping triggered!")
                break

    # ── Save training history ─────────────────────────────────────
    history_path = config.model_path / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    logger.info("[train] History saved to %s", history_path)

    history_plot_path = config.model_path / "training_history.png"
    plot_training_history(history, history_plot_path)

    # ── Load best model & evaluate on test set ────────────────────
    logger.info("=" * 60)
    logger.info("  Evaluating best model on test set ...")
    logger.info("=" * 60)

    model.load_state_dict(
        torch.load(model_save_path, map_location=device, weights_only=True),
    )
    results = evaluate_model(model, test_loader, device, config.LABEL_COLUMNS)

    report = generate_classification_report(
        results["y_true"], results["y_pred"], config.LABEL_COLUMNS,
    )
    logger.info("\n%s", report)

    macro = results["metrics"]["macro"]
    logger.info("=" * 60)
    logger.info("  Macro Averages (default threshold = 0.5)")
    logger.info("=" * 60)
    logger.info("  ROC-AUC   : %.4f", macro["roc_auc"])
    logger.info("  Precision : %.4f", macro["precision"])
    logger.info("  Recall    : %.4f", macro["recall"])
    logger.info("  F1 Score  : %.4f", macro["f1"])
    logger.info("=" * 60)

    # ── Evaluation plots ──────────────────────────────────────────
    cm_path = config.model_path / "confusion_matrices.png"
    roc_path = config.model_path / "roc_curves.png"
    plot_confusion_matrices(
        results["y_true"], results["y_pred"], config.LABEL_COLUMNS, cm_path,
    )
    plot_roc_curves(
        results["y_true"], results["y_proba"], config.LABEL_COLUMNS, roc_path,
    )

    # ── Save evaluation metrics ───────────────────────────────────
    eval_metrics_path = config.model_path / "evaluation_results.json"
    with open(eval_metrics_path, "w", encoding="utf-8") as fh:
        json.dump(results["metrics"], fh, indent=2)
    logger.info("[eval] Metrics saved to %s", eval_metrics_path)

    # ── Threshold tuning on validation set ────────────────────────
    logger.info("=" * 60)
    logger.info("  Tuning per-label decision thresholds ...")
    logger.info("=" * 60)

    thresholds_path = config.model_path / "thresholds.json"
    thresholds = tune_thresholds_from_data(
        model, val_loader, device, config.LABEL_COLUMNS, thresholds_path,
        step=config.THRESHOLD_STEP,
        min_precision=config.MIN_PRECISION_FLOOR,
    )

    # ── Re-evaluate test set with optimised thresholds ────────────
    thresholds_list = [thresholds[col] for col in config.LABEL_COLUMNS]
    results_opt = evaluate_model(
        model, test_loader, device, config.LABEL_COLUMNS,
        thresholds=thresholds_list,
    )
    macro_opt = results_opt["metrics"]["macro"]
    logger.info("=" * 60)
    logger.info("  Test Metrics WITH optimised thresholds")
    logger.info("=" * 60)
    logger.info("  ROC-AUC   : %.4f", macro_opt["roc_auc"])
    logger.info("  Precision : %.4f", macro_opt["precision"])
    logger.info("  Recall    : %.4f", macro_opt["recall"])
    logger.info("  F1 Score  : %.4f", macro_opt["f1"])
    logger.info("=" * 60)

    # ── Final download instructions ───────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  [DONE] TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  Download these files from Colab:")
    logger.info("    1. %s", model_save_path)
    logger.info("    2. %s", config.model_path / "vocab.json")
    logger.info("    3. %s", history_path)
    logger.info("    4. %s", history_plot_path)
    logger.info("    5. %s", cm_path)
    logger.info("    6. %s", roc_path)
    logger.info("    7. %s", eval_metrics_path)
    logger.info("    8. %s", thresholds_path)
    logger.info("")
    logger.info("  Place them in the 'models/' directory of your local project.")
    logger.info("=" * 60)

    return history


# ===================================================================
# CLI entry-point
# ===================================================================

def _parse_args():
    """Parse command-line arguments for training overrides."""
    parser = argparse.ArgumentParser(
        description="Train the Toxicity Classifier (designed for Colab w/ GPU).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Override TRAIN_SAMPLE_SIZE (0 = full dataset).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override mini-batch size.",
    )
    parser.add_argument(
        "--focal-loss", action="store_true",
        help="Use Focal Loss instead of weighted BCE.",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable rare-class augmentation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # ── Logging setup ─────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    cfg = get_default_config()

    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.sample_size is not None:
        cfg.TRAIN_SAMPLE_SIZE = args.sample_size if args.sample_size > 0 else None
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.focal_loss:
        cfg.USE_FOCAL_LOSS = True
    if args.no_augment:
        cfg.AUGMENT_RARE_CLASSES = False

    train_model(cfg)
