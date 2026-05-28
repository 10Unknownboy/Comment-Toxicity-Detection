"""
Data preprocessing pipeline for Comment Toxicity Detection.

Responsibilities:
  - Text cleaning (HTML, URLs, special characters)
  - Vocabulary construction, serialisation, and loading
  - Sequence encoding (token to index with padding / truncation)
  - PyTorch Dataset and DataLoader creation
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import Config

# ---------------------------------------------------------------------------
# Special token indices
# ---------------------------------------------------------------------------
PAD_IDX = 0
UNK_IDX = 1


# ===================================================================
# Text cleaning
# ===================================================================

def clean_text(text):
    """Normalise and clean a raw comment string.

    Processing steps (in order):
      1. Convert to lowercase.
      2. Strip HTML tags.
      3. Remove URLs.
      4. Remove all characters that are not alphanumeric or whitespace.
      5. Collapse multiple whitespace characters into a single space.
      6. Strip leading / trailing whitespace.

    Accepts a raw comment string.  None or non-string values are coerced
    to an empty string.  Returns the cleaned text ready for tokenisation.
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

def build_vocabulary(texts, max_vocab):
    """Build a word-to-index mapping from a list of cleaned texts.

    Tokens are split on whitespace.  The most-frequent tokens (up to
    max_vocab) are kept; indices start at 2 to reserve 0 for PAD and
    1 for UNK.

    Takes a list of pre-cleaned text strings and the maximum vocabulary
    size.  Returns a dict mapping token strings to integer indices.
    """
    counter = Counter()
    for t in texts:
        counter.update(t.split())

    most_common = counter.most_common(max_vocab)
    word2idx = {
        word: idx + 2 for idx, (word, _) in enumerate(most_common)
    }
    return word2idx


def save_vocabulary(vocab, path):
    """Persist a vocabulary dictionary to a JSON file.

    Takes a word-to-index mapping and a destination file path
    (will be created / overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)


def load_vocabulary(path):
    """Load a vocabulary dictionary from a JSON file.

    Takes a path to the JSON vocabulary file.  Returns a word-to-index
    mapping.  Raises FileNotFoundError if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===================================================================
# Encoding
# ===================================================================

def encode_text(text, vocab, max_len):
    """Convert a cleaned text string into a fixed-length list of token indices.

    Unknown tokens are mapped to UNK_IDX (1).  Sequences shorter than
    max_len are right-padded with PAD_IDX (0); longer sequences are
    truncated.

    Takes a single cleaned text string, a word-to-index mapping, and
    the desired output length.  Returns the integer-encoded, padded /
    truncated sequence.
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
    """PyTorch Dataset for encoded toxic-comment sequences.

    Takes a list of token-index sequences (each of length max_len) and
    an optional label matrix of shape (N, num_labels).  Pass None for
    unlabelled / test-only data.
    """

    def __init__(self, encoded_texts, labels=None):
        self.encoded_texts = encoded_texts
        self.labels = labels

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        """Return a single sample as (text_tensor,) or (text_tensor, label_tensor)."""
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

def get_data_loaders(config):
    """Prepare train / validation / test DataLoader objects.

    Workflow:
      1. Read the training CSV.
      2. Optionally sub-sample rows (config.TRAIN_SAMPLE_SIZE).
      3. Clean all comment texts.
      4. Split into train / val / test sets (stratified where possible).
      5. Build vocabulary from the training split only.
      6. Encode all splits.
      7. Wrap in ToxicCommentsDataset then DataLoader.
      8. Save the vocabulary to models/vocab.json.

    Takes a Config object.  Returns a tuple of
    (train_loader, val_loader, test_loader, vocab).
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
