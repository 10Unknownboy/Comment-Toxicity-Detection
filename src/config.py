"""
Configuration module for the Comment Toxicity Detection project.

Provides a centralised Config dataclass with all hyperparameters, paths,
and project-wide constants.  Helper utilities locate the project root
directory and return a ready-to-use default configuration instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    The function walks upward from the directory containing *this* file
    until it finds a directory that contains a ``src`` sub-directory (the
    package directory).  If no such directory is found it falls back to
    the parent of ``src/``.

    Returns
    -------
    Path
        Absolute ``pathlib.Path`` to the project root.
    """
    current = Path(__file__).resolve().parent  # …/src
    # Walk up at most 5 levels looking for a recognisable root marker.
    for _ in range(5):
        if (current / "src").is_dir() or (current / "app.py").is_file():
            return current
        current = current.parent
    # Fallback: assume src/ is one level below root.
    return Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Central configuration for the Comment Toxicity Detection pipeline.

    Attributes
    ----------
    DATA_DIR : str
        Relative path to the data directory (resolved against project root).
    MODEL_DIR : str
        Relative path to the model / artifact directory.
    TRAIN_FILE : str
        Filename of the training CSV inside ``DATA_DIR``.
    VOCAB_SIZE : int
        Maximum vocabulary size (most-frequent tokens kept).
    EMBED_DIM : int
        Dimensionality of the token-embedding layer.
    HIDDEN_DIM : int
        Number of hidden units per direction in the BiLSTM.
    NUM_LAYERS : int
        Number of stacked LSTM layers.
    DROPOUT : float
        Dropout probability applied after the LSTM and between FC layers.
    MAX_SEQ_LEN : int
        Maximum token-sequence length (pad / truncate to this).
    BATCH_SIZE : int
        Mini-batch size used during training and evaluation.
    EPOCHS : int
        Maximum number of training epochs.
    LEARNING_RATE : float
        Initial learning rate for the Adam optimiser.
    TRAIN_SAMPLE_SIZE : Optional[int]
        If set, randomly sample this many rows from the training CSV.
        Use ``None`` to train on the full dataset.
    VAL_SPLIT : float
        Fraction of training data reserved for validation.
    TEST_SPLIT : float
        Fraction of training data reserved for testing.
    EARLY_STOPPING_PATIENCE : int
        Number of epochs without validation-loss improvement before
        stopping early.
    LABEL_COLUMNS : list[str]
        Target column names in the CSV.
    RANDOM_SEED : int
        Global random seed for reproducibility.
    """

    # ---- Paths ----
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    TRAIN_FILE: str = "train.csv"

    # ---- Model hyperparameters ----
    VOCAB_SIZE: int = 50_000
    EMBED_DIM: int = 128
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.3

    # ---- Training ----
    MAX_SEQ_LEN: int = 200
    BATCH_SIZE: int = 256
    EPOCHS: int = 5
    LEARNING_RATE: float = 1e-3
    TRAIN_SAMPLE_SIZE: Optional[int] = 160_000
    VAL_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.1
    EARLY_STOPPING_PATIENCE: int = 2

    # ---- Labels ----
    LABEL_COLUMNS: list[str] = field(
        default_factory=lambda: [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
    )

    # ---- Seed ----
    RANDOM_SEED: int = 42

    # ---- Derived helpers (not part of __init__) ----

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
    """Create and return a ``Config`` instance with default values.

    The function also ensures that the ``models/`` directory exists so
    that downstream code can write checkpoints without additional checks.

    Returns
    -------
    Config
        A freshly-created default configuration object.
    """
    config = Config()
    os.makedirs(config.model_path, exist_ok=True)
    return config
