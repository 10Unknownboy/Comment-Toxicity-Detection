"""
Configuration module for the Comment Toxicity Detection project.

Provides a centralised Config class with all hyperparameters, paths,
and project-wide constants. Helper utilities locate the project root
directory and return a ready-to-use default configuration instance.
"""

import os
from pathlib import Path
import torch

def get_project_root():
    """
    Return the absolute path to the project root directory.
    Walks upward from the directory containing this file until it finds
    a directory that contains a src sub-directory. If no such directory
    is found it falls back to the parent of src/.
    """
    current = Path(__file__).resolve().parent  # …/src
    for _ in range(5):
        if (current / "src").is_dir() or (current / "app.py").is_file():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent

class Config:
    """
    Central configuration for the Comment Toxicity Detection pipeline.
    All hyperparameters, file paths, and project-wide constants are
    defined here with sensible defaults.
    """

    def __init__(
        self,
        # ── Paths ──────────────────────────────────────────────────
        DATA_DIR="data",
        MODEL_DIR="models",
        TRAIN_FILE="train.csv",
        
        # ── Model architecture ─────────────────────────────────────
        MODEL_TYPE="lstm_attention",
        VOCAB_SIZE=50000,
        EMBED_DIM=128,
        HIDDEN_DIM=128,
        NUM_LAYERS=2,
        DROPOUT=0.3,
        
        # ── Sequence ───────────────────────────────────────────────
        MAX_SEQ_LEN=300,  # Limits comment length (95th percentile is ≈ 270 tokens)
        
        # ── Training ───────────────────────────────────────────────
        BATCH_SIZE=256,
        EPOCHS=8,
        LEARNING_RATE=1e-4,
        TRAIN_SAMPLE_SIZE=None,  # Set to an integer to subset data for faster runs
        VAL_SPLIT=0.10,
        TEST_SPLIT=0.10,
        RANDOM_SEED=42,
        
        # ── Loss function ──────────────────────────────────────────
        USE_FOCAL_LOSS=True,     # Dynamically handles extreme class imbalance
        FOCAL_GAMMA=2.0,         # Downweights easy negatives
        FOCAL_ALPHA=0.25,        # Weighting for the positive class
        LABEL_SMOOTHING=0.05,    # Prevents model overconfidence on noisy labels
        
        # ── Regularisation / stability ─────────────────────────────
        GRAD_CLIP_NORM=1.0,      # Clips gradients to prevent exploding gradients
        
        # ── LR scheduler ───────────────────────────────────────────
        LR_SCHEDULER_FACTOR=0.5,
        LR_SCHEDULER_PATIENCE=2,
        MIN_LR=1e-6,
        
        # ── Early stopping ─────────────────────────────────────────
        EARLY_STOPPING_PATIENCE=5,       # Halts training if val_roc_auc stops improving
        EARLY_STOPPING_MIN_DELTA=0.001,
        
        # ── Augmentation ───────────────────────────────────────────
        AUGMENT_RARE_CLASSES=True,       # Generates synthetic data for rare classes
        AUGMENT_TARGET_THREAT=2000,
        AUGMENT_TARGET_IDENTITY_HATE=2500,
        
        # ── Threshold tuner ────────────────────────────────────────
        THRESHOLD_STEP=0.01,             # Granularity of threshold search
        MIN_PRECISION_FLOOR=0.10,        # Prevents threshold tuner from choosing overly permissive cutoffs
        
        # ── Mixed precision ────────────────────────────────────────
        USE_AMP=None,  # Automatically detects CUDA if None
        
        # ── Labels ─────────────────────────────────────────────────
        LABEL_COLUMNS=None,
    ):
        # Initialise paths
        self.DATA_DIR = DATA_DIR
        self.MODEL_DIR = MODEL_DIR
        self.TRAIN_FILE = TRAIN_FILE

        # Initialise model architecture parameters
        self.MODEL_TYPE = MODEL_TYPE
        self.VOCAB_SIZE = VOCAB_SIZE
        self.EMBED_DIM = EMBED_DIM
        self.HIDDEN_DIM = HIDDEN_DIM
        self.NUM_LAYERS = NUM_LAYERS
        self.DROPOUT = DROPOUT

        # Initialise sequence constraints
        self.MAX_SEQ_LEN = MAX_SEQ_LEN

        # Initialise training loop hyperparams
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.TRAIN_SAMPLE_SIZE = TRAIN_SAMPLE_SIZE
        self.VAL_SPLIT = VAL_SPLIT
        self.TEST_SPLIT = TEST_SPLIT
        self.RANDOM_SEED = RANDOM_SEED

        # Initialise loss and regularisation
        self.USE_FOCAL_LOSS = USE_FOCAL_LOSS
        self.FOCAL_GAMMA = FOCAL_GAMMA
        self.FOCAL_ALPHA = FOCAL_ALPHA
        self.LABEL_SMOOTHING = LABEL_SMOOTHING
        self.GRAD_CLIP_NORM = GRAD_CLIP_NORM

        # Initialise schedulers and early stopping
        self.LR_SCHEDULER_FACTOR = LR_SCHEDULER_FACTOR
        self.LR_SCHEDULER_PATIENCE = LR_SCHEDULER_PATIENCE
        self.MIN_LR = MIN_LR
        self.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
        self.EARLY_STOPPING_MIN_DELTA = EARLY_STOPPING_MIN_DELTA

        # Initialise data augmentation rules
        self.AUGMENT_RARE_CLASSES = AUGMENT_RARE_CLASSES
        self.AUGMENT_TARGET_THREAT = AUGMENT_TARGET_THREAT
        self.AUGMENT_TARGET_IDENTITY_HATE = AUGMENT_TARGET_IDENTITY_HATE

        # Initialise threshold sweeping bounds
        self.THRESHOLD_STEP = THRESHOLD_STEP
        self.MIN_PRECISION_FLOOR = MIN_PRECISION_FLOOR

        # Mixed precision setup (fallback to cpu if no cuda)
        if USE_AMP is None:
            self.USE_AMP = torch.cuda.is_available()
        else:
            self.USE_AMP = USE_AMP

        # Define the target classification columns
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
    """
    Create and return a Config instance with default values.
    Also ensures that the models/ directory exists.
    """
    config = Config()
    os.makedirs(config.model_path, exist_ok=True)
    return config




