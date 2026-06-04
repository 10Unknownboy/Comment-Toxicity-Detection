"""
Data preprocessing and augmentation pipeline for Comment Toxicity Detection.

Responsibilities:
  - Text cleaning (HTML, URLs, unicode, repetition, slang)
  - Vocabulary construction, serialisation, and loading
  - Sequence encoding (token → index with padding / truncation)
  - Text augmentation for rare classes (deletion, swap, synonym)
  - PyTorch Dataset and DataLoader creation
"""

from __future__ import annotations

import json
import logging
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special token indices
# ---------------------------------------------------------------------------
PAD_IDX = 0
UNK_IDX = 1


# ===================================================================
# Text cleaning
# ===================================================================

# Common internet slang → expansion (applied after lowercasing)
_SLANG_MAP = {
    " u ": " you ",
    " r ": " are ",
    " ur ": " your ",
    " pls ": " please ",
    " plz ": " please ",
    " thx ": " thanks ",
    " thnx ": " thanks ",
    " bcz ": " because ",
    " cuz ": " because ",
    " idk ": " i do not know ",
    " imo ": " in my opinion ",
    " tbh ": " to be honest ",
    " smh ": " shaking my head ",
    " stfu ": " shut the fuck up ",
    " gtfo ": " get the fuck out ",
    " lmao ": " laughing my ass off ",
    " af ": " as fuck ",
}

# Regex: collapse 3+ consecutive identical words → single word
_RE_REPEAT_WORD = re.compile(r"\b(\w+)(\s+\1){2,}\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalise and clean a raw comment string.

    Processing steps (in order):
      1. Unicode normalisation (NFKD).
      2. Convert to lowercase.
      3. Strip HTML tags.
      4. Remove URLs.
      5. Expand common internet slang.
      6. Keep alphanumeric + sentiment punctuation ``[!?.,]``.
      7. Collapse repeating words (e.g. "die die die die" → "die").
      8. Collapse whitespace and strip.

    Parameters
    ----------
    text : str
        Raw comment text.  ``None`` or non-string values return ``""``.

    Returns
    -------
    str
        Cleaned text ready for tokenisation.
    """
    if not isinstance(text, str):
        return ""

    # Unicode normalisation (é → e, etc.)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs

    # Expand slang (pad with spaces for word-boundary matching)
    text = f" {text} "
    for slang, expansion in _SLANG_MAP.items():
        text = text.replace(slang, expansion)

    # Keep alphanumeric + useful punctuation
    text = re.sub(r"[^a-z0-9!?., \n]", " ", text)

    # Collapse repeating words: "fuck fuck fuck fuck" → "fuck"
    text = _RE_REPEAT_WORD.sub(r"\1", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===================================================================
# Vocabulary utilities
# ===================================================================

def build_vocabulary(texts: list[str], max_vocab: int) -> dict[str, int]:
    """Build a word-to-index mapping from a list of cleaned texts.

    Tokens are split on whitespace.  The most-frequent tokens (up to
    *max_vocab*) are kept; indices start at 2 to reserve 0 for PAD and
    1 for UNK.

    Parameters
    ----------
    texts : list[str]
        Pre-cleaned text strings.
    max_vocab : int
        Maximum vocabulary size.

    Returns
    -------
    dict[str, int]
        Word → index mapping.
    """
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(t.split())

    most_common = counter.most_common(max_vocab)
    return {word: idx + 2 for idx, (word, _) in enumerate(most_common)}


def save_vocabulary(vocab: dict[str, int], path: str | Path) -> None:
    """Persist a vocabulary dictionary to a JSON file.

    Parameters
    ----------
    vocab : dict[str, int]
        Word → index mapping.
    path : str or Path
        Destination file path.
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
        Word → index mapping.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===================================================================
# Encoding
# ===================================================================

def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    """Convert a cleaned text string to a fixed-length token-index list.

    Parameters
    ----------
    text : str
        Cleaned text.
    vocab : dict[str, int]
        Word → index mapping.
    max_len : int
        Desired output length (truncate / pad as needed).

    Returns
    -------
    list[int]
        Integer-encoded, padded/truncated sequence.
    """
    tokens = text.split()
    indices = [vocab.get(tok, UNK_IDX) for tok in tokens]

    if len(indices) > max_len:
        indices = indices[:max_len]
    elif len(indices) < max_len:
        indices += [PAD_IDX] * (max_len - len(indices))

    return indices


# ===================================================================
# Text augmentation for rare classes
# ===================================================================

def _ensure_nltk_wordnet() -> bool:
    """Try to load NLTK WordNet, downloading it if necessary.

    Returns
    -------
    bool
        ``True`` if WordNet is available, ``False`` otherwise.
    """
    try:
        import nltk
        from nltk.corpus import wordnet  # noqa: F401

        # Quick check — will raise LookupError if not downloaded
        wordnet.synsets("test")
        return True
    except LookupError:
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            return True
        except Exception:
            logger.warning("Could not download NLTK WordNet — synonym augmentation disabled.")
            return False
    except ImportError:
        logger.warning("NLTK not installed — synonym augmentation disabled.")
        return False


def augment_random_deletion(text: str, delete_pct: float = 0.12) -> str:
    """Randomly delete a fraction of words from a text.

    Preserves at least 70 % of the original tokens.

    Parameters
    ----------
    text : str
        Cleaned input text.
    delete_pct : float
        Target fraction of tokens to delete (0.10–0.15).

    Returns
    -------
    str
        Augmented text.
    """
    words = text.split()
    if len(words) <= 3:
        return text

    n_delete = max(1, int(len(words) * delete_pct))
    # Ensure we keep at least 70 %
    n_delete = min(n_delete, int(len(words) * 0.30))

    indices_to_delete = set(random.sample(range(len(words)), n_delete))
    return " ".join(w for i, w in enumerate(words) if i not in indices_to_delete)


def augment_word_swap(text: str) -> str:
    """Randomly swap two adjacent words.

    Parameters
    ----------
    text : str
        Cleaned input text.

    Returns
    -------
    str
        Augmented text with one pair of adjacent words swapped.
    """
    words = text.split()
    if len(words) < 2:
        return text

    idx = random.randint(0, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return " ".join(words)


def augment_synonym_replace(text: str, n: int = 2) -> str:
    """Replace up to *n* non-stopword tokens with WordNet synonyms.

    Gracefully falls back to the original text if WordNet is unavailable.

    Parameters
    ----------
    text : str
        Cleaned input text.
    n : int
        Maximum number of replacements.

    Returns
    -------
    str
        Augmented text.
    """
    try:
        from nltk.corpus import wordnet
    except (ImportError, LookupError):
        return text

    _stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "and", "but", "or", "nor", "for", "yet", "so", "to", "of",
        "in", "on", "at", "by", "with", "from", "up", "out", "if",
        "than", "too", "very", "just", "not", "no", "i", "you", "he",
        "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "this", "that",
    }

    words = text.split()
    candidates = [
        (i, w) for i, w in enumerate(words) if w not in _stopwords and len(w) > 2
    ]

    if not candidates:
        return text

    random.shuffle(candidates)
    replaced = 0

    for idx, word in candidates:
        if replaced >= n:
            break
        synsets = wordnet.synsets(word)
        if not synsets:
            continue
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name != word:
                    synonyms.add(name)
        if synonyms:
            words[idx] = random.choice(list(synonyms))
            replaced += 1

    return " ".join(words)


def augment_rare_classes(
    df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Augment training data to up-sample rare classes.

    Only augments rows where ``threat == 1`` or ``identity_hate == 1``.
    Three augmentation techniques are applied in rotation: random
    deletion, word swap, and synonym replacement.

    Parameters
    ----------
    df : pd.DataFrame
        Training split with ``clean_text`` and label columns.
    config : Config
        Project configuration with augmentation targets.

    Returns
    -------
    pd.DataFrame
        Original + augmented rows concatenated.
    """
    targets = {
        "threat": config.AUGMENT_TARGET_THREAT,
        "identity_hate": config.AUGMENT_TARGET_IDENTITY_HATE,
    }

    has_wordnet = _ensure_nltk_wordnet()
    aug_funcs = [augment_random_deletion, augment_word_swap]
    if has_wordnet:
        aug_funcs.append(augment_synonym_replace)

    new_rows: list[dict] = []

    for label, target_count in targets.items():
        subset = df[df[label] == 1]
        current = len(subset)
        needed = max(0, target_count - current)

        if needed == 0:
            logger.info("  %s: already has %d samples (target %d), skipping.", label, current, target_count)
            continue

        logger.info("  %s: %d → %d (augmenting %d samples)", label, current, target_count, needed)

        idx = 0
        for _ in range(needed):
            row = subset.iloc[idx % current].to_dict()
            aug_fn = aug_funcs[idx % len(aug_funcs)]
            row["clean_text"] = aug_fn(row["clean_text"])
            new_rows.append(row)
            idx += 1

    if new_rows:
        aug_df = pd.DataFrame(new_rows)
        logger.info("  Total augmented rows: %d", len(aug_df))
        return pd.concat([df, aug_df], ignore_index=True)

    return df


# ===================================================================
# PyTorch Dataset
# ===================================================================

class ToxicCommentsDataset(Dataset):
    """PyTorch Dataset for encoded toxic-comment sequences.

    Parameters
    ----------
    encoded_texts : list[list[int]]
        Token-index sequences, each of length ``max_len``.
    labels : np.ndarray or None
        Label matrix of shape ``(N, num_labels)``.  ``None`` for
        unlabelled / inference data.
    """

    def __init__(
        self,
        encoded_texts: list[list[int]],
        labels: np.ndarray | None = None,
    ):
        self.encoded_texts = encoded_texts
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.encoded_texts)

    def __getitem__(self, idx: int):
        """Return ``(text_tensor,)`` or ``(text_tensor, label_tensor)``."""
        text_tensor = torch.tensor(self.encoded_texts[idx], dtype=torch.long)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return text_tensor, label_tensor

        return (text_tensor,)


# ===================================================================
# DataLoader factory
# ===================================================================

def get_data_loaders(
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Prepare train / validation / test DataLoader objects.

    Workflow:
      1. Read the training CSV.
      2. Optionally sub-sample rows.
      3. Clean all comment texts.
      4. Split into train / val / test sets.
      5. Augment rare classes in the training split.
      6. Build vocabulary from the training split only.
      7. Encode all splits.
      8. Wrap in DataLoaders.
      9. Save vocabulary to ``models/vocab.json``.

    Parameters
    ----------
    config : Config
        Project configuration.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]
        ``(train_loader, val_loader, test_loader, vocab)``.
    """
    # ---- 1. Load CSV ----
    csv_path = config.train_csv_path
    logger.info("[data] Loading training data from %s …", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("[data] Loaded %s rows.", f"{len(df):,}")

    # ---- 2. Optional sampling ----
    if config.TRAIN_SAMPLE_SIZE is not None and config.TRAIN_SAMPLE_SIZE < len(df):
        df = df.sample(n=config.TRAIN_SAMPLE_SIZE, random_state=config.RANDOM_SEED)
        df = df.reset_index(drop=True)
        logger.info("[data] Sampled down to %s rows.", f"{len(df):,}")

    # ---- 3. Clean texts ----
    logger.info("[data] Cleaning texts …")
    df["clean_text"] = df["comment_text"].apply(clean_text)

    texts = df["clean_text"].tolist()
    labels = df[config.LABEL_COLUMNS].values.astype(np.float32)

    # ---- 4. Train / val / test split ----
    val_test_frac = config.VAL_SPLIT + config.TEST_SPLIT
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        test_size=val_test_frac,
        random_state=config.RANDOM_SEED,
    )
    relative_val = config.VAL_SPLIT / val_test_frac
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=(1 - relative_val),
        random_state=config.RANDOM_SEED,
    )

    logger.info(
        "[data] Splits → train: %s  val: %s  test: %s",
        f"{len(train_texts):,}", f"{len(val_texts):,}", f"{len(test_texts):,}",
    )

    # ---- 5. Augment rare classes (training split only) ----
    if config.AUGMENT_RARE_CLASSES:
        logger.info("[data] Augmenting rare classes …")
        train_df = pd.DataFrame({"clean_text": train_texts})
        for i, col in enumerate(config.LABEL_COLUMNS):
            train_df[col] = train_labels[:, i]

        train_df = augment_rare_classes(train_df, config)

        train_texts = train_df["clean_text"].tolist()
        train_labels = train_df[config.LABEL_COLUMNS].values.astype(np.float32)
        logger.info("[data] After augmentation: %s training samples.", f"{len(train_texts):,}")

    # ---- 6. Build vocabulary (training set only) ----
    logger.info("[data] Building vocabulary …")
    vocab = build_vocabulary(train_texts, config.VOCAB_SIZE)
    logger.info("[data] Vocabulary size: %s tokens (+ PAD, UNK).", f"{len(vocab):,}")

    vocab_path = config.model_path / "vocab.json"
    save_vocabulary(vocab, vocab_path)
    logger.info("[data] Vocabulary saved to %s", vocab_path)

    # ---- 7. Encode ----
    logger.info("[data] Encoding sequences …")
    train_encoded = [encode_text(t, vocab, config.MAX_SEQ_LEN) for t in train_texts]
    val_encoded = [encode_text(t, vocab, config.MAX_SEQ_LEN) for t in val_texts]
    test_encoded = [encode_text(t, vocab, config.MAX_SEQ_LEN) for t in test_texts]

    # ---- 8. Datasets & DataLoaders ----
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
