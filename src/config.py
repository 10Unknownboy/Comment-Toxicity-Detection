"""
Configuration module for the Comment Toxicity Detection project.

Provides a centralised ``Config`` class with all hyperparameters, paths,
and project-wide constants.  Helper utilities locate the project root
directory and return a ready-to-use default configuration instance.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Walks upward from the directory containing this file until it finds
    a directory that contains a ``src`` sub-directory (the package
    directory).  If no such directory is found it falls back to the
    parent of ``src/``.

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    current = Path(__file__).resolve().parent  # …/src
    for _ in range(5):
        if (current / "src").is_dir() or (current / "app.py").is_file():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


class Config:
    """Central configuration for the Comment Toxicity Detection pipeline.

    All hyperparameters, file paths, and project-wide constants are
    defined here with sensible defaults.  Property helpers provide
    resolved absolute paths derived from the project root.
    """

    def __init__(
        self,
        # ── Paths ──────────────────────────────────────────────────
        DATA_DIR: str = "data",
        MODEL_DIR: str = "models",
        TRAIN_FILE: str = "train.csv",
        # ── Model architecture ─────────────────────────────────────
        MODEL_TYPE: str = "lstm_attention",  # "lstm_attention" | "distilbert"
        VOCAB_SIZE: int = 50_000,
        EMBED_DIM: int = 128,
        HIDDEN_DIM: int = 128,
        NUM_LAYERS: int = 2,
        DROPOUT: float = 0.3,
        # ── Sequence ───────────────────────────────────────────────
        MAX_SEQ_LEN: int = 300,  # 95th percentile ≈ 270 tokens
        # ── Training ───────────────────────────────────────────────
        BATCH_SIZE: int = 256,
        EPOCHS: int = 8,
        LEARNING_RATE: float = 5e-4,
        TRAIN_SAMPLE_SIZE: int | None = None,  # None = full dataset
        VAL_SPLIT: float = 0.10,
        TEST_SPLIT: float = 0.10,
        RANDOM_SEED: int = 42,
        # ── Loss function ──────────────────────────────────────────
        USE_FOCAL_LOSS: bool = False,
        FOCAL_GAMMA: float = 2.0,
        FOCAL_ALPHA: float = 0.25,
        LABEL_SMOOTHING: float = 0.05,
        # ── Regularisation / stability ─────────────────────────────
        GRAD_CLIP_NORM: float = 1.0,
        # ── LR scheduler (ReduceLROnPlateau) ───────────────────────
        LR_SCHEDULER_FACTOR: float = 0.5,
        LR_SCHEDULER_PATIENCE: int = 2,
        MIN_LR: float = 1e-6,
        # ── Early stopping ─────────────────────────────────────────
        EARLY_STOPPING_PATIENCE: int = 3,
        EARLY_STOPPING_MIN_DELTA: float = 0.001,
        # ── Augmentation ───────────────────────────────────────────
        AUGMENT_RARE_CLASSES: bool = True,
        AUGMENT_TARGET_THREAT: int = 2000,
        AUGMENT_TARGET_IDENTITY_HATE: int = 4000,
        # ── Threshold tuner ────────────────────────────────────────
        THRESHOLD_STEP: float = 0.01,
        MIN_PRECISION_FLOOR: float = 0.40,
        # ── Mixed precision ────────────────────────────────────────
        USE_AMP: bool | None = None,  # None = auto-detect
        # ── Labels ─────────────────────────────────────────────────
        LABEL_COLUMNS: list[str] | None = None,
    ):
        # Paths
        self.DATA_DIR = DATA_DIR
        self.MODEL_DIR = MODEL_DIR
        self.TRAIN_FILE = TRAIN_FILE

        # Model architecture
        self.MODEL_TYPE = MODEL_TYPE
        self.VOCAB_SIZE = VOCAB_SIZE
        self.EMBED_DIM = EMBED_DIM
        self.HIDDEN_DIM = HIDDEN_DIM
        self.NUM_LAYERS = NUM_LAYERS
        self.DROPOUT = DROPOUT

        # Sequence
        self.MAX_SEQ_LEN = MAX_SEQ_LEN

        # Training
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.TRAIN_SAMPLE_SIZE = TRAIN_SAMPLE_SIZE
        self.VAL_SPLIT = VAL_SPLIT
        self.TEST_SPLIT = TEST_SPLIT
        self.RANDOM_SEED = RANDOM_SEED

        # Loss
        self.USE_FOCAL_LOSS = USE_FOCAL_LOSS
        self.FOCAL_GAMMA = FOCAL_GAMMA
        self.FOCAL_ALPHA = FOCAL_ALPHA
        self.LABEL_SMOOTHING = LABEL_SMOOTHING

        # Regularisation
        self.GRAD_CLIP_NORM = GRAD_CLIP_NORM

        # LR scheduler
        self.LR_SCHEDULER_FACTOR = LR_SCHEDULER_FACTOR
        self.LR_SCHEDULER_PATIENCE = LR_SCHEDULER_PATIENCE
        self.MIN_LR = MIN_LR

        # Early stopping
        self.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
        self.EARLY_STOPPING_MIN_DELTA = EARLY_STOPPING_MIN_DELTA

        # Augmentation
        self.AUGMENT_RARE_CLASSES = AUGMENT_RARE_CLASSES
        self.AUGMENT_TARGET_THREAT = AUGMENT_TARGET_THREAT
        self.AUGMENT_TARGET_IDENTITY_HATE = AUGMENT_TARGET_IDENTITY_HATE

        # Threshold tuner
        self.THRESHOLD_STEP = THRESHOLD_STEP
        self.MIN_PRECISION_FLOOR = MIN_PRECISION_FLOOR

        # Mixed precision (auto-detect if None)
        if USE_AMP is None:
            self.USE_AMP = torch.cuda.is_available()
        else:
            self.USE_AMP = USE_AMP

        # Labels
        if LABEL_COLUMNS is None:
            self.LABEL_COLUMNS = [
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ]
        else:
            self.LABEL_COLUMNS = LABEL_COLUMNS

    # ── Derived properties ─────────────────────────────────────────

    @property
    def data_path(self) -> Path:
        """Resolved absolute path to the data directory."""
        return get_project_root() / self.DATA_DIR

    @property
    def model_path(self) -> Path:
        """Resolved absolute path to the model directory."""
        return get_project_root() / self.MODEL_DIR

    @property
    def train_csv_path(self) -> Path:
        """Resolved absolute path to the training CSV."""
        return self.data_path / self.TRAIN_FILE

    @property
    def num_labels(self) -> int:
        """Number of target labels."""
        return len(self.LABEL_COLUMNS)


def get_default_config() -> Config:
    """Create and return a Config instance with default values.

    Also ensures that the ``models/`` directory exists so that
    downstream code can write checkpoints without additional checks.

    Returns
    -------
    Config
        A ready-to-use configuration object.
    """
    config = Config()
    os.makedirs(config.model_path, exist_ok=True)
    return config
