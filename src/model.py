"""
Model architectures for multi-label toxicity classification.

Provides:
  - ``AttentionLayer``               — self-attention over LSTM outputs.
  - ``BiLSTMAttentionClassifier``    — BiLSTM + attention (recommended).
  - ``DistilBERTClassifier``         — stub for future DistilBERT fine-tuning.
  - ``create_model``                 — factory function.
  - ``ToxicityClassifier``           — backwards-compatible alias.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ===================================================================
# Self-Attention Layer
# ===================================================================

class AttentionLayer(nn.Module):
    """Additive (Bahdanau-style) self-attention over a sequence.

    Given an LSTM output of shape ``(batch, seq_len, hidden*2)``, it
    learns a per-timestep importance score, normalises with softmax,
    and returns a weighted-sum context vector of shape
    ``(batch, hidden*2)``.

    Parameters
    ----------
    hidden_dim : int
        Size of *each direction* of the BiLSTM.  The input is expected
        to have size ``hidden_dim * 2``.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        input_dim = hidden_dim * 2
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1, bias=False),
        )

    def forward(
        self,
        lstm_output: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention-weighted context vector.

        Parameters
        ----------
        lstm_output : torch.Tensor
            BiLSTM output of shape ``(batch, seq_len, hidden*2)``.
        mask : torch.Tensor or None
            Boolean mask of shape ``(batch, seq_len)`` where ``True``
            indicates a *padded* position.  If provided, padded
            positions receive ``-inf`` before softmax.

        Returns
        -------
        torch.Tensor
            Context vector of shape ``(batch, hidden*2)``.
        """
        # scores: (batch, seq_len, 1)
        scores = self.score(lstm_output)

        if mask is not None:
            # If a sequence is completely padded, unmask it to prevent softmax returning NaN
            all_masked = mask.all(dim=-1, keepdim=True)
            mask = mask & ~all_masked
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))

        # weights: (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)

        # context: (batch, hidden*2)
        context = (weights * lstm_output).sum(dim=1)
        return context


# ===================================================================
# BiLSTM + Attention Classifier
# ===================================================================

class BiLSTMAttentionClassifier(nn.Module):
    """BiLSTM with self-attention for multi-label toxicity classification.

    Architecture::

        Embedding → BiLSTM (stacked) → Attention → Dropout → FC → ReLU
        → Dropout → FC → (raw logits)

    The attention layer replaces the naive "last hidden state"
    concatenation of the plain BiLSTM, allowing the model to focus on
    the most toxicity-relevant tokens regardless of position.

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size (including PAD and UNK).
    embed_dim : int
        Token-embedding dimensionality.
    hidden_dim : int
        Number of hidden units *per direction* in the LSTM.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability.
    num_labels : int
        Number of independent binary labels (default 6).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_labels: int = 6,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        # --- Embedding (PAD index = 0) ---
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # --- Bidirectional LSTM ---
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Self-attention ---
        self.attention = AttentionLayer(hidden_dim)

        # --- Classifier head ---
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Integer-encoded input of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(batch, num_labels)``.
            Apply ``torch.sigmoid()`` externally for probabilities.
        """
        # Build a padding mask (True where PAD)
        pad_mask = (x == 0)  # (batch, seq_len)

        embedded = self.embedding(x)             # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)         # (batch, seq_len, hidden*2)
        context = self.attention(lstm_out, pad_mask)  # (batch, hidden*2)

        out = self.dropout(context)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch, num_labels) — raw logits
        return out


# ===================================================================
# DistilBERT Classifier (STUB)
# ===================================================================

class DistilBERTClassifier(nn.Module):
    """DistilBERT-based multi-label toxicity classifier.

    .. note::
        **This is a stub.**  The full implementation (tokenizer
        pipeline, differential learning rates, warmup scheduler) will
        be added in a future iteration.  For now, ``create_model``
        will raise ``NotImplementedError`` if this model type is
        requested.

    Parameters
    ----------
    num_labels : int
        Number of independent binary labels.
    dropout : float
        Dropout applied to the [CLS] embedding.
    """

    def __init__(self, num_labels: int = 6, dropout: float = 0.3):
        super().__init__()
        self.num_labels = num_labels
        # Will be:
        #   self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        #   self.dropout = nn.Dropout(dropout)
        #   self.classifier = nn.Linear(768, num_labels)
        raise NotImplementedError(
            "DistilBERT support is stubbed out.  "
            "Set MODEL_TYPE = 'lstm_attention' in config.py."
        )

    def forward(self, input_ids, attention_mask):
        """Forward pass (not yet implemented)."""
        raise NotImplementedError


# ===================================================================
# Backwards-compatible alias
# ===================================================================

# Old code imported ``ToxicityClassifier`` — keep that working.
ToxicityClassifier = BiLSTMAttentionClassifier


# ===================================================================
# Model factory
# ===================================================================

def create_model(
    config,
    vocab_size: int | None = None,
) -> nn.Module:
    """Instantiate the model specified in ``config.MODEL_TYPE``.

    Parameters
    ----------
    config : Config
        Project configuration.
    vocab_size : int or None
        Required for LSTM-based models.  For DistilBERT it is ignored.

    Returns
    -------
    nn.Module
        Uninitialised model (not yet moved to a device).

    Raises
    ------
    NotImplementedError
        If ``config.MODEL_TYPE == "distilbert"`` (stub).
    ValueError
        If ``config.MODEL_TYPE`` is unrecognised.
    """
    if config.MODEL_TYPE == "lstm_attention":
        if vocab_size is None:
            raise ValueError("vocab_size is required for LSTM models.")
        model = BiLSTMAttentionClassifier(
            vocab_size=vocab_size,
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            num_labels=config.num_labels,
        )
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "[model] BiLSTM+Attention  |  params: %s total, %s trainable",
            f"{total:,}", f"{trainable:,}",
        )
        return model

    if config.MODEL_TYPE == "distilbert":
        return DistilBERTClassifier(
            num_labels=config.num_labels,
            dropout=config.DROPOUT,
        )

    raise ValueError(f"Unknown MODEL_TYPE: {config.MODEL_TYPE!r}")
