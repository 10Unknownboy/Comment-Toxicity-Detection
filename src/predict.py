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

# ---------------------------------------------------------------------------
# Module-level cache to avoid reloading on every call
# ---------------------------------------------------------------------------
_cached_model: Optional[ToxicityClassifier] = None
_cached_vocab: Optional[dict[str, int]] = None
_cached_config: Optional[Config] = None


# ===================================================================
# Model loading
# ===================================================================

def load_model(
    model_dir: str = "models",
) -> tuple[ToxicityClassifier, dict[str, int], Config]:
    """Load the trained model, vocabulary, and configuration.

    The model is forced onto **CPU** so the Streamlit app works without
    a GPU.  Results are cached at module level – subsequent calls return
    the same objects without reloading from disk.

    Parameters
    ----------
    model_dir : str, optional
        Path (relative to project root) that contains
        ``toxicity_model.pth`` and ``vocab.json``.  Defaults to
        ``"models"``.

    Returns
    -------
    tuple[ToxicityClassifier, dict[str, int], Config]
        ``(model, vocab, config)`` – the model is already in eval mode.

    Raises
    ------
    FileNotFoundError
        If the model checkpoint or vocabulary file is missing.
    """
    global _cached_model, _cached_vocab, _cached_config  # noqa: PLW0603

    if _cached_model is not None and _cached_vocab is not None:
        assert _cached_config is not None
        return _cached_model, _cached_vocab, _cached_config

    config = get_default_config()
    model_path = Path(model_dir)

    # Resolve relative paths against the project root
    if not model_path.is_absolute():
        model_path = config.model_path

    weights_file = model_path / "toxicity_model.pth"
    vocab_file = model_path / "vocab.json"

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

    return model, vocab, config


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
    if not text or not text.strip():
        return {col: 0.0 for col in config.LABEL_COLUMNS}

    cleaned = clean_text(text)
    encoded = encode_text(cleaned, vocab, config.MAX_SEQ_LEN)
    tensor = torch.tensor([encoded], dtype=torch.long)  # (1, seq_len)

    with torch.no_grad():
        proba = model(tensor).squeeze(0).numpy()  # (num_labels,)

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

    results: list[dict[str, float]] = []
    for t in texts:
        preds = predict_single(t, model, vocab, config)
        preds["text"] = t
        results.append(preds)

    df = pd.DataFrame(results)
    # Reorder columns so "text" comes first
    cols = ["text"] + config.LABEL_COLUMNS
    return df[cols]


# ===================================================================
# Human-readable helpers
# ===================================================================

def get_toxicity_label(
    predictions: dict[str, float],
    threshold: float = 0.5,
) -> str:
    """Return a human-readable toxicity summary label.

    The label reflects the most severe positive prediction:

    * ``"Highly Toxic 🚨"`` – if ``severe_toxic ≥ threshold``
    * ``"Threat Detected ⚔️"`` – if ``threat ≥ threshold``
    * ``"Toxic ⚠️"`` – if any other label ≥ threshold
    * ``"Clean ✅"`` – if no label exceeds the threshold

    Parameters
    ----------
    predictions : dict[str, float]
        Label-to-probability mapping from :func:`predict_single`.
    threshold : float, optional
        Decision threshold (default ``0.5``).

    Returns
    -------
    str
        One of ``"Clean ✅"``, ``"Toxic ⚠️"``, ``"Highly Toxic 🚨"``,
        ``"Threat Detected ⚔️"``.
    """
    if not predictions:
        return "Clean ✅"

    if predictions.get("severe_toxic", 0.0) >= threshold:
        return "Highly Toxic 🚨"
    if predictions.get("threat", 0.0) >= threshold:
        return "Threat Detected ⚔️"

    # Check remaining labels
    for label, score in predictions.items():
        if score >= threshold:
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
