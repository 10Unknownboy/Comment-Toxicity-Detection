"""
Configuration module for the Comment Toxicity Detection project.

Provides a centralised Config class with all hyperparameters, paths,
and project-wide constants.  Helper utilities locate the project root
directory and return a ready-to-use default configuration instance.
"""

import os
from pathlib import Path


def get_project_root():
    """Return the absolute path to the project root directory.

    Walks upward from the directory containing this file until it finds
    a directory that contains a ``src`` sub-directory (the package
    directory).  If no such directory is found it falls back to the
    parent of ``src/``.
    """
    current = Path(__file__).resolve().parent  # …/src
    # Walk up at most 5 levels looking for a recognisable root marker.
    for _ in range(5):
        if (current / "src").is_dir() or (current / "app.py").is_file():
            return current
        current = current.parent
    # Fallback: assume src/ is one level below root.
    return Path(__file__).resolve().parent.parent


class Config:
    """Central configuration for the Comment Toxicity Detection pipeline.

    All hyperparameters, file paths, and project-wide constants are
    defined here with sensible defaults.  Property helpers provide
    resolved absolute paths derived from the project root.
    """

    def __init__(
        self,
        DATA_DIR="data",
        MODEL_DIR="models",
        TRAIN_FILE="train.csv",
        VOCAB_SIZE=50_000,
        EMBED_DIM=128,
        HIDDEN_DIM=128,
        NUM_LAYERS=2,
        DROPOUT=0.3,
        MAX_SEQ_LEN=200,
        BATCH_SIZE=256,
        EPOCHS=5,
        LEARNING_RATE=1e-3,
        TRAIN_SAMPLE_SIZE=160_000,
        VAL_SPLIT=0.1,
        TEST_SPLIT=0.1,
        EARLY_STOPPING_PATIENCE=2,
        LABEL_COLUMNS=None,
        RANDOM_SEED=42,
    ):
        # ---- Paths ----
        self.DATA_DIR = DATA_DIR
        self.MODEL_DIR = MODEL_DIR
        self.TRAIN_FILE = TRAIN_FILE

        # ---- Model hyperparameters ----
        self.VOCAB_SIZE = VOCAB_SIZE
        self.EMBED_DIM = EMBED_DIM
        self.HIDDEN_DIM = HIDDEN_DIM
        self.NUM_LAYERS = NUM_LAYERS
        self.DROPOUT = DROPOUT

        # ---- Training ----
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.TRAIN_SAMPLE_SIZE = TRAIN_SAMPLE_SIZE
        self.VAL_SPLIT = VAL_SPLIT
        self.TEST_SPLIT = TEST_SPLIT
        self.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE

        # ---- Labels ----
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

        # ---- Seed ----
        self.RANDOM_SEED = RANDOM_SEED

    # ---- Derived helpers ----

    @property
    def data_path(self):
        """Resolved absolute path to the data directory."""
        return get_project_root() / self.DATA_DIR

    @property
    def model_path(self):
        """Resolved absolute path to the model directory."""
        return get_project_root() / self.MODEL_DIR

    @property
    def train_csv_path(self):
        """Resolved absolute path to the training CSV."""
        return self.data_path / self.TRAIN_FILE

    @property
    def num_labels(self):
        """Number of target labels."""
        return len(self.LABEL_COLUMNS)


def get_default_config():
    """Create and return a Config instance with default values.

    Also ensures that the models/ directory exists so that downstream
    code can write checkpoints without additional checks.
    """
    config = Config()
    os.makedirs(config.model_path, exist_ok=True)
    return config
