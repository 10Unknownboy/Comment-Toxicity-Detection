"""
Model architectures for multi-label toxicity classification.

Provides:
  - AttentionLayer               - self-attention over LSTM outputs.
  - BiLSTMAttentionClassifier    - BiLSTM + attention (recommended).
  - create_model                 - factory function.
  - ToxicityClassifier           - backwards-compatible alias.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ===================================================================
# Self-Attention Layer
# ===================================================================

class AttentionLayer(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over a sequence.

    Given an LSTM output of shape (batch, seq_len, hidden*2), it
    learns a per-timestep importance score, normalises with softmax,
    and returns a weighted-sum context vector of shape
    (batch, hidden*2).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        input_dim = hidden_dim * 2
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1, bias=False),
        )

    def forward(self, lstm_output, mask=None):
        """Compute attention-weighted context vector."""
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
    """
    BiLSTM with self-attention for multi-label toxicity classification.

    Architecture:
        Embedding -> BiLSTM (stacked) -> Attention -> Dropout -> FC -> ReLU
        -> Dropout -> FC -> (raw logits)

    The attention layer replaces the naive "last hidden state"
    concatenation of the plain BiLSTM, allowing the model to focus on
    the most toxicity-relevant tokens regardless of position.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
        num_labels=6,
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

    def forward(self, x):
        """
        Forward pass.
        Returns raw logits. Apply torch.sigmoid() externally for probabilities.
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
        out = self.fc2(out)  # (batch, num_labels) - raw logits
        return out




# ===================================================================
# Backwards-compatible alias
# ===================================================================

# Old code imported ToxicityClassifier - keep that working.
ToxicityClassifier = BiLSTMAttentionClassifier


# ===================================================================
# Model factory
# ===================================================================

def create_model(config, vocab_size=None):
    """Instantiate the model specified in config.MODEL_TYPE."""
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

    raise ValueError(f"Unknown MODEL_TYPE: {config.MODEL_TYPE!r}")

