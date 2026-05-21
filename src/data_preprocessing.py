"""
Data preprocessing pipeline for Comment Toxicity Detection.

Responsibilities
----------------
* Text cleaning (HTML, URLs, special characters)
* Vocabulary construction, serialisation, and loading
* Sequence encoding (token → index with padding / truncation)
* PyTorch ``Dataset`` and ``DataLoader`` creation
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import Config

# ---------------------------------------------------------------------------
# Special token indices
# ---------------------------------------------------------------------------
PAD_IDX: int = 0
UNK_IDX: int = 1


# ===================================================================
# Text cleaning
# ===================================================================

def clean_text(text: str) -> str:
    """Normalise and clean a raw comment string.

    Processing steps (in order):
    1. Convert to lowercase.
    2. Strip HTML tags (``<…>``).
    3. Remove URLs (``http…`` / ``www.…``).
    4. Remove all characters that are not alphanumeric or whitespace.
    5. Collapse multiple whitespace characters into a single space.
    6. Strip leading / trailing whitespace.

    Parameters
    ----------
    text : str
        Raw comment text.  ``None`` or non-string values are coerced to
        an empty string.

    Returns
    -------
    str
        Cleaned text ready for tokenisation.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)                # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # non-alphanum
    text = re.sub(r"\s+", " ", text)                     # collapse ws
    return text.strip()


# ===================================================================
# Vocabulary utilities
# ===================================================================

def build_vocabulary(texts: list[str], max_vocab: int) -> dict[str, int]:
    """Build a word-to-index mapping from a list of *cleaned* texts.

    Tokens are split on whitespace.  The ``max_vocab`` most-frequent
    tokens are kept; indices start at **2** to reserve:

    * 0 → ``<PAD>``
    * 1 → ``<UNK>``

    Parameters
    ----------
    texts : list[str]
        Pre-cleaned text strings.
    max_vocab : int
        Maximum number of vocabulary entries (excluding PAD / UNK).

    Returns
    -------
    dict[str, int]
        Mapping from token string to integer index.
    """
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(t.split())

    most_common = counter.most_common(max_vocab)
    word2idx: dict[str, int] = {
        word: idx + 2 for idx, (word, _) in enumerate(most_common)
    }
    return word2idx


def save_vocabulary(vocab: dict[str, int], path: str | Path) -> None:
    """Persist a vocabulary dictionary to a JSON file.

    Parameters
    ----------
    vocab : dict[str, int]
        Word-to-index mapping produced by :func:`build_vocabulary`.
    path : str or Path
        Destination file path (will be created / overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)


def load_vocabulary(path: str | Path) -> dict[str, int]:
    """Load a vocabulary dictionary from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON vocabulary file.

    Returns
    -------
    dict[str, int]
        Word-to-index mapping.

    Raises
    ------
    FileNotFoundError
        If the vocabulary file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===================================================================
# Encoding
# ===================================================================

def encode_text(
    text: str,
    vocab: dict[str, int],
    max_len: int,
) -> list[int]:
    """Convert a cleaned text string into a fixed-length list of token indices.

    Unknown tokens are mapped to ``UNK_IDX`` (1).  Sequences shorter
    than ``max_len`` are right-padded with ``PAD_IDX`` (0); longer
    sequences are truncated.

    Parameters
    ----------
    text : str
        A single cleaned text string.
    vocab : dict[str, int]
        Word-to-index mapping.
    max_len : int
        Desired output length.

    Returns
    -------
    list[int]
        Integer-encoded, padded / truncated sequence of length ``max_len``.
    """
    tokens = text.split()
    indices = [vocab.get(tok, UNK_IDX) for tok in tokens]

    # Truncate
    if len(indices) > max_len:
        indices = indices[:max_len]
    # Pad
    elif len(indices) < max_len:
        indices += [PAD_IDX] * (max_len - len(indices))

    return indices


# ===================================================================
# PyTorch Dataset
# ===================================================================

class ToxicCommentsDataset(Dataset):
    """PyTorch ``Dataset`` for encoded toxic-comment sequences.

    Parameters
    ----------
    encoded_texts : list[list[int]]
        Token-index sequences, each of length ``max_len``.
    labels : np.ndarray or None
        Label matrix of shape ``(N, num_labels)``.  Pass ``None`` for
        unlabelled / test-only data.
    """

    def __init__(
        self,
        encoded_texts: list[list[int]],
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self.encoded_texts = encoded_texts
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.encoded_texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Return a single sample.

        Returns
        -------
        tuple[torch.Tensor, ...]
            ``(text_tensor,)`` when labels are absent, otherwise
            ``(text_tensor, label_tensor)``.
        """
        text_tensor = torch.tensor(self.encoded_texts[idx], dtype=torch.long)

        if self.labels is not None:
            label_tensor = torch.tensor(
                self.labels[idx], dtype=torch.float32
            )
            return text_tensor, label_tensor

        return (text_tensor,)


# ===================================================================
# DataLoader factory
# ===================================================================

def get_data_loaders(
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Prepare train / validation / test ``DataLoader`` objects.

    Workflow
    --------
    1. Read the training CSV.
    2. Optionally sub-sample rows (``config.TRAIN_SAMPLE_SIZE``).
    3. Clean all comment texts.
    4. Split into train / val / test sets (stratified where possible).
    5. Build vocabulary from the **training** split only.
    6. Encode all splits.
    7. Wrap in ``ToxicCommentsDataset`` → ``DataLoader``.
    8. Save the vocabulary to ``models/vocab.json``.

    Parameters
    ----------
    config : Config
        Project configuration object.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]
        ``(train_loader, val_loader, test_loader, vocab)``
    """
    # ---- 1. Load CSV ----
    csv_path = config.train_csv_path
    print(f"[data] Loading training data from {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"[data] Loaded {len(df):,} rows.")

    # ---- 2. Optional sampling ----
    if config.TRAIN_SAMPLE_SIZE is not None and config.TRAIN_SAMPLE_SIZE < len(df):
        df = df.sample(n=config.TRAIN_SAMPLE_SIZE, random_state=config.RANDOM_SEED)
        df = df.reset_index(drop=True)
        print(f"[data] Sampled down to {len(df):,} rows.")

    # ---- 3. Clean texts ----
    print("[data] Cleaning texts …")
    df["clean_text"] = df["comment_text"].apply(clean_text)

    texts = df["clean_text"].tolist()
    labels = df[config.LABEL_COLUMNS].values.astype(np.float32)

    # ---- 4. Train / val / test split ----
    #   First split off the test set, then split remainder into train/val.
    val_test_frac = config.VAL_SPLIT + config.TEST_SPLIT
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=val_test_frac,
        random_state=config.RANDOM_SEED,
    )

    relative_val = config.VAL_SPLIT / val_test_frac
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=(1 - relative_val),
        random_state=config.RANDOM_SEED,
    )

    print(
        f"[data] Splits → train: {len(train_texts):,}  "
        f"val: {len(val_texts):,}  test: {len(test_texts):,}"
    )

    # ---- 5. Build vocabulary (training set only) ----
    print("[data] Building vocabulary …")
    vocab = build_vocabulary(train_texts, config.VOCAB_SIZE)
    print(f"[data] Vocabulary size: {len(vocab):,} tokens (+ PAD, UNK).")

    vocab_path = config.model_path / "vocab.json"
    save_vocabulary(vocab, vocab_path)
    print(f"[data] Vocabulary saved to {vocab_path}")

    # ---- 6. Encode ----
    print("[data] Encoding sequences …")
    train_encoded = [encode_text(t, vocab, config.MAX_SEQ_LEN) for t in train_texts]
    val_encoded = [encode_text(t, vocab, config.MAX_SEQ_LEN) for t in val_texts]
    test_encoded = [encode_text(t, vocab, config.MAX_SEQ_LEN) for t in test_texts]

    # ---- 7. Datasets & DataLoaders ----
    train_ds = ToxicCommentsDataset(train_encoded, train_labels)
    val_ds = ToxicCommentsDataset(val_encoded, val_labels)
    test_ds = ToxicCommentsDataset(test_encoded, test_labels)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False,
    )

    return train_loader, val_loader, test_loader, vocab
