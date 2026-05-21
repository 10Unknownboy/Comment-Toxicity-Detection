"""
Inference utilities for the Comment Toxicity Detection Streamlit app.

All inference runs on **CPU** so no GPU is required at serving time.
The module provides:

* ``load_model`` – load model weights, vocabulary, and config (cached).
* ``predict_single`` – predict toxicity scores for one comment.
* ``predict_batch`` – predict on a list of comments → DataFrame.
* ``get_toxicity_label`` – human-readable summary label.
* ``get_severity_color`` – hex colour on a green→yellow→red gradient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from src.config import Config, get_default_config
from src.data_preprocessing import clean_text, encode_text, load_vocabulary
from src.model import ToxicityClassifier
from src.threshold_tuner import load_thresholds

# ---------------------------------------------------------------------------
# Module-level cache to avoid reloading on every call
# ---------------------------------------------------------------------------
_cached_model: Optional[ToxicityClassifier] = None
_cached_vocab: Optional[dict[str, int]] = None
_cached_config: Optional[Config] = None
_cached_thresholds: Optional[dict[str, float]] = None


# ===================================================================
# Model loading
# ===================================================================

def load_model(
    model_dir: str = "models",
) -> tuple[ToxicityClassifier, dict[str, int], Config, dict[str, float]]:
    """Load the trained model, vocabulary, configuration, and thresholds.

    The model is forced onto **CPU** so the Streamlit app works without
    a GPU.  Results are cached at module level – subsequent calls return
    the same objects without reloading from disk.

    Parameters
    ----------
    model_dir : str, optional
        Path (relative to project root) that contains
        ``toxicity_model.pth``, ``vocab.json``, and optionally
        ``thresholds.json``.  Defaults to ``"models"``.

    Returns
    -------
    tuple[ToxicityClassifier, dict[str, int], Config, dict[str, float]]
        ``(model, vocab, config, thresholds)`` – the model is in eval
        mode.  If ``thresholds.json`` is missing, default 0.5 thresholds
        are returned.

    Raises
    ------
    FileNotFoundError
        If the model checkpoint or vocabulary file is missing.
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

    # Instantiate model
    actual_vocab_size = len(vocab) + 2  # +2 for PAD & UNK
    model = ToxicityClassifier(
        vocab_size=actual_vocab_size,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        num_labels=config.num_labels,
    )

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

def predict_single(
    text: str,
    model: ToxicityClassifier,
    vocab: dict[str, int],
    config: Config,
) -> dict[str, float]:
    """Predict toxicity scores for a single comment.

    Parameters
    ----------
    text : str
        Raw comment text.
    model : ToxicityClassifier
        Loaded model in eval mode.
    vocab : dict[str, int]
        Word-to-index mapping.
    config : Config
        Project configuration (needed for ``MAX_SEQ_LEN`` and
        ``LABEL_COLUMNS``).

    Returns
    -------
    dict[str, float]
        Mapping from label name to predicted probability, e.g.
        ``{'toxic': 0.95, 'severe_toxic': 0.02, …}``.
    """
    # Handle NaN/None/non-string types gracefully
    if not isinstance(text, str) or not text.strip():
        return {col: 0.0 for col in config.LABEL_COLUMNS}

    cleaned = clean_text(text)
    encoded = encode_text(cleaned, vocab, config.MAX_SEQ_LEN)
    tensor = torch.tensor([encoded], dtype=torch.long)  # (1, seq_len)

    with torch.no_grad():
        logits = model(tensor)
        proba = torch.sigmoid(logits).squeeze(0).numpy()  # (num_labels,)

    return {col: float(proba[i]) for i, col in enumerate(config.LABEL_COLUMNS)}


# ===================================================================
# Batch prediction
# ===================================================================

def predict_batch(
    texts: list[str],
    model: ToxicityClassifier,
    vocab: dict[str, int],
    config: Config,
) -> pd.DataFrame:
    """Predict toxicity scores for multiple comments.

    Parameters
    ----------
    texts : list[str]
        Raw comment texts.
    model : ToxicityClassifier
        Loaded model in eval mode.
    vocab : dict[str, int]
        Word-to-index mapping.
    config : Config
        Project configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["text"] + config.LABEL_COLUMNS``.
    """
    if not texts:
        return pd.DataFrame(columns=["text"] + config.LABEL_COLUMNS)

    # Clean input: filter out NaN/None and convert to strings
    cleaned_texts = [
        str(t).strip() if isinstance(t, str) or (t is not None and str(t) != 'nan') else ""
        for t in texts
    ]
    cleaned_texts = [t for t in cleaned_texts if t]  # Filter empty strings

    results: list[dict[str, float]] = []
    for t in cleaned_texts:
        try:
            preds = predict_single(t, model, vocab, config)
        except Exception as e:
            # If a single row fails, return zeros for that row instead of crashing
            preds = {col: 0.0 for col in config.LABEL_COLUMNS}
        preds["text"] = t
        preds["overall_toxicity"] = max(
            preds.get(c, 0.0) for c in config.LABEL_COLUMNS
        )
        preds["label"] = get_toxicity_label(preds)
        results.append(preds)

    df = pd.DataFrame(results)
    # Reorder columns so "text" comes first
    cols = ["text"] + config.LABEL_COLUMNS + ["overall_toxicity", "label"]
    return df[cols]


# ===================================================================
# Human-readable helpers
# ===================================================================

def get_toxicity_label(
    predictions: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> str:
    """Return a human-readable toxicity summary label.

    The label reflects the most severe positive prediction using
    per-label thresholds (from ``thresholds.json``).

    Parameters
    ----------
    predictions : dict[str, float]
        Label-to-probability mapping from :func:`predict_single`.
    thresholds : dict[str, float], optional
        Per-label decision thresholds.  Falls back to 0.5 for all.

    Returns
    -------
    str
        One of ``"Clean ✅"``, ``"Toxic ⚠️"``, ``"Highly Toxic 🚨"``,
        ``"Threat Detected ⚔️"``.
    """
    if not predictions:
        return "Clean ✅"

    if thresholds is None:
        thresholds = {k: 0.5 for k in predictions}

    if predictions.get("severe_toxic", 0.0) >= thresholds.get("severe_toxic", 0.5):
        return "Highly Toxic 🚨"
    if predictions.get("threat", 0.0) >= thresholds.get("threat", 0.5):
        return "Threat Detected ⚔️"

    # Check remaining labels
    for label, score in predictions.items():
        if label in ("text", "overall_toxicity", "label"):
            continue
        t = thresholds.get(label, 0.5)
        if score >= t:
            return "Toxic ⚠️"

    return "Clean ✅"


def get_severity_color(score: float) -> str:
    """Return a hex colour on a green → yellow → red gradient.

    Parameters
    ----------
    score : float
        Toxicity score in ``[0, 1]``.

    Returns
    -------
    str
        Hex colour string, e.g. ``"#2ECC71"`` (green) or
        ``"#E74C3C"`` (red).

    Examples
    --------
    >>> get_severity_color(0.0)
    '#2ECC71'
    >>> get_severity_color(1.0)
    '#E74C3C'
    """
    score = max(0.0, min(1.0, score))

    if score <= 0.5:
        # Green (#2ECC71) → Yellow (#F1C40F)
        t = score / 0.5
        r = int(0x2E + (0xF1 - 0x2E) * t)
        g = int(0xCC + (0xC4 - 0xCC) * t)
        b = int(0x71 + (0x0F - 0x71) * t)
    else:
        # Yellow (#F1C40F) → Red (#E74C3C)
        t = (score - 0.5) / 0.5
        r = int(0xF1 + (0xE7 - 0xF1) * t)
        g = int(0xC4 + (0x4C - 0xC4) * t)
        b = int(0x0F + (0x3C - 0x0F) * t)

    return f"#{r:02X}{g:02X}{b:02X}"
