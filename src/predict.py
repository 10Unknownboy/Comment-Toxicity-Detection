"""
Inference utilities for the Comment Toxicity Detection Streamlit app.

All inference runs on CPU so no GPU is required at serving time.
The module provides:

  - load_model         - load model, vocabulary, config, thresholds.
  - predict_single     - predict toxicity scores for one comment.
  - predict_batch      - predict on a list of comments -> DataFrame.
  - get_toxicity_label - human-readable summary label.
  - get_severity_color - hex colour on a green->red gradient.
"""

import logging
from pathlib import Path

import pandas as pd
import torch

from src.config import Config, get_default_config
from src.data_preprocessing import clean_text, encode_text, load_vocabulary
from src.model import create_model
from src.threshold_tuner import load_thresholds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache to avoid reloading on every call
# ---------------------------------------------------------------------------
_cached_model = None
_cached_vocab = None
_cached_config = None
_cached_thresholds = None


# ===================================================================
# Model loading
# ===================================================================

def load_model(model_dir="models"):
    """
    Load the trained model, vocabulary, configuration, and thresholds.
    The model is forced onto CPU so the Streamlit app works without
    a GPU. Results are cached at module level.
    """
    global _cached_model, _cached_vocab, _cached_config, _cached_thresholds

    if _cached_model is not None and _cached_vocab is not None:
        assert _cached_config is not None
        assert _cached_thresholds is not None
        return _cached_model, _cached_vocab, _cached_config, _cached_thresholds

    config = get_default_config()
    model_path = Path(model_dir)

    # Resolve relative paths against the project root
    if not model_path.is_absolute():
        model_path = config.model_path

    weights_file = model_path / "toxicity_model.pth"
    vocab_file = model_path / "vocab.json"
    thresholds_file = model_path / "thresholds.json"

    if not weights_file.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_file}. "
            "Train the model first or download from Colab."
        )
    if not vocab_file.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found at {vocab_file}. "
            "Train the model first or download from Colab."
        )

    # Load vocab
    vocab = load_vocabulary(vocab_file)

    # Load thresholds (fall back to 0.5 for all labels)
    thresholds = load_thresholds(thresholds_file)
    if thresholds is None:
        thresholds = {col: 0.5 for col in config.LABEL_COLUMNS}

    # Instantiate model via factory
    actual_vocab_size = len(vocab) + 2  # +2 for PAD & UNK
    model = create_model(config, vocab_size=actual_vocab_size)

    # Load weights (CPU)
    device = torch.device("cpu")
    state_dict = torch.load(weights_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Cache
    _cached_model = model
    _cached_vocab = vocab
    _cached_config = config
    _cached_thresholds = thresholds

    return model, vocab, config, thresholds


# ===================================================================
# Single prediction
# ===================================================================

def predict_single(text, model, vocab, config):
    """Predict toxicity scores for a single comment."""
    if not text or not text.strip():
        return {col: 0.0 for col in config.LABEL_COLUMNS}

    cleaned = clean_text(text)
    encoded = encode_text(cleaned, vocab, config.MAX_SEQ_LEN)
    tensor = torch.tensor([encoded], dtype=torch.long)  # (1, seq_len)

    with torch.no_grad():
        logits = model(tensor)
        proba = torch.sigmoid(logits).squeeze(0).numpy()

    return {col: float(proba[i]) for i, col in enumerate(config.LABEL_COLUMNS)}


# ===================================================================
# Batch prediction
# ===================================================================

def predict_batch(texts, model, vocab, config, thresholds=None):
    """Predict toxicity scores for multiple comments."""
    if not texts:
        return pd.DataFrame(
            columns=["text"] + config.LABEL_COLUMNS
            + ["overall_toxicity", "label"],
        )

    results = []
    for t in texts:
        preds = predict_single(t, model, vocab, config)
        row = {
            **preds,
            "text": t,
            "overall_toxicity": max(
                preds.get(c, 0.0) for c in config.LABEL_COLUMNS
            ),
            "label": get_toxicity_label(preds, thresholds=thresholds),
        }
        results.append(row)

    df = pd.DataFrame(results)
    cols = ["text"] + config.LABEL_COLUMNS + ["overall_toxicity", "label"]
    return df[cols]


# ===================================================================
# Human-readable helpers
# ===================================================================

def get_toxicity_label(predictions, thresholds=None):
    """
    Return a human-readable toxicity summary label.
    Uses per-label thresholds from thresholds.json.
    """
    if not predictions:
        return "Clean"

    if thresholds is None:
        thresholds = {k: 0.5 for k in predictions}

    if predictions.get("severe_toxic", 0.0) >= thresholds.get("severe_toxic", 0.5):
        return "Highly Toxic"
    if predictions.get("threat", 0.0) >= thresholds.get("threat", 0.5):
        return "Threat Detected"

    for label, score in predictions.items():
        if label in ("text", "overall_toxicity", "label"):
            continue
        t = thresholds.get(label, 0.5)
        if score >= t:
            return "Toxic"

    return "Clean"


def get_severity_color(score):
    """Return a hex colour on a green -> yellow -> red gradient."""
    score = max(0.0, min(1.0, score))

    if score <= 0.5:
        t = score / 0.5
        r = int(0x2E + (0xF1 - 0x2E) * t)
        g = int(0xCC + (0xC4 - 0xCC) * t)
        b = int(0x71 + (0x0F - 0x71) * t)
    else:
        t = (score - 0.5) / 0.5
        r = int(0xF1 + (0xE7 - 0xF1) * t)
        g = int(0xC4 + (0x4C - 0xC4) * t)
        b = int(0x0F + (0x3C - 0x0F) * t)

    return f"#{r:02X}{g:02X}{b:02X}"





