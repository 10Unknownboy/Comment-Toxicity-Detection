"""
Loss functions for the Comment Toxicity Detection model.

Provides:
  - ``FocalLoss``            — focuses training on hard / rare examples.
  - ``compute_pos_weights``  — calculates sqrt-dampened class weights.
  - ``apply_label_smoothing``— prevents overconfident predictions.
  - ``get_loss_function``    — factory that returns the configured loss.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config

logger = logging.getLogger(__name__)


# ===================================================================
# Focal Loss
# ===================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-label binary classification.

    Dynamically scales the standard BCE loss based on prediction
    confidence, so that *easy* (well-classified) examples contribute
    less to the total loss while *hard* (mis-classified) examples
    receive a much higher weight.

    Parameters
    ----------
    alpha : float or torch.Tensor or None
        Weighting factor for the positive class. Standard is 0.25.
        If ``None``, no alpha weighting is applied.
    gamma : float
        Focusing parameter.  ``gamma=0`` recovers standard BCE;
        higher values increasingly down-weight easy examples.
    reduction : str
        ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(
        self,
        alpha: float | torch.Tensor | None = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from raw logits.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model output of shape ``(batch, num_labels)``.
        targets : torch.Tensor
            Binary ground-truth labels of shape ``(batch, num_labels)``.

        Returns
        -------
        torch.Tensor
            Scalar loss (if reduction is ``"mean"`` or ``"sum"``).
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
        pt = torch.exp(-bce)  # probability of the correct class
        focal = ((1.0 - pt) ** self.gamma) * bce

        if self.alpha is not None:
            # Standard focal loss alpha weighting
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# ===================================================================
# Class-weight computation
# ===================================================================

def compute_pos_weights(
    labels: np.ndarray,
    label_columns: list[str],
    dampening: str = "sqrt",
) -> torch.Tensor:
    """Compute per-label positive-class weights from training labels.

    Parameters
    ----------
    labels : np.ndarray
        Binary label matrix of shape ``(N, num_labels)``.
    label_columns : list[str]
        Human-readable label names (used for logging only).
    dampening : str
        ``"sqrt"`` applies ``sqrt(neg/pos)`` to avoid overcorrection
        on extremely rare classes.  ``"none"`` uses the raw ratio.

    Returns
    -------
    torch.Tensor
        Weight tensor of shape ``(num_labels,)`` suitable for
        ``BCEWithLogitsLoss(pos_weight=...)`` or ``FocalLoss(alpha=...)``.
    """
    pos_counts = labels.sum(axis=0)
    neg_counts = labels.shape[0] - pos_counts
    raw_weights = neg_counts / np.maximum(pos_counts, 1.0)

    if dampening == "sqrt":
        weights = np.sqrt(raw_weights)
    else:
        weights = raw_weights

    logger.info("Positive weights per label (%s-dampened):", dampening)
    for name, rw, sw in zip(label_columns, raw_weights, weights):
        logger.info("  %-20s : %7.2f → %s → %.2f", name, rw, dampening, sw)

    return torch.tensor(weights, dtype=torch.float32)


# ===================================================================
# Label smoothing
# ===================================================================

def apply_label_smoothing(
    labels: torch.Tensor,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Apply label smoothing to binary targets.

    Hard labels ``{0, 1}`` are softened to ``{ε/2, 1 − ε/2}`` so the
    model does not become overconfident on noisy annotations.

    Parameters
    ----------
    labels : torch.Tensor
        Binary labels of shape ``(batch, num_labels)``.
    epsilon : float
        Smoothing strength.  ``0`` = no smoothing.

    Returns
    -------
    torch.Tensor
        Smoothed labels with the same shape as input.
    """
    return labels * (1.0 - epsilon) + epsilon / 2.0


# ===================================================================
# Factory
# ===================================================================

def get_loss_function(
    config: Config,
    pos_weights: torch.Tensor,
    device: torch.device,
) -> nn.Module:
    """Return the loss function specified in *config*.

    Parameters
    ----------
    config : Config
        Project configuration.
    pos_weights : torch.Tensor
        Per-label weights from :func:`compute_pos_weights`.
    device : torch.device
        Target device for the loss weights.

    Returns
    -------
    nn.Module
        Ready-to-use loss criterion.
    """
    pw = pos_weights.to(device)

    if config.USE_FOCAL_LOSS:
        logger.info(
            "Using FocalLoss (gamma=%.1f, alpha=%.2f)",
            config.FOCAL_GAMMA, config.FOCAL_ALPHA
        )
        return FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

    logger.info("Using BCEWithLogitsLoss with sqrt-dampened pos_weight")
    return nn.BCEWithLogitsLoss(pos_weight=pw)
